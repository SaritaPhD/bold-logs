"""Unified experiment runner.

Usage:
  python3 scripts/run_full.py --config bgl_global --split chrono [--seeds 0]
  python3 scripts/run_full.py --config hdfs --split chrono --seeds 0 1 2 3 4
  python3 scripts/run_full.py --config bgl_nodetime --split chrono --correlator
  python3 scripts/run_full.py --config hdfs --split chrono --correlator

Results append into results/full_<config>_<split>.json (per-model per-seed),
including config + timings — this is the manuscript's run log.
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import joblib

from logad.features import CountVectors, seq_stats
from logad.models import PCASPE, IF, OCSVM, MLPAE, MarkovNext
from logad.correlate import CrossComponentCorrelator
from logad.eval import split, threshold_from_benign, metrics, pr_auc

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

FILES = {"hdfs": "hdfs_sessions.joblib", "bgl_global": "bgl_global.joblib",
         "thunderbird": "thunderbird_global.joblib",
         "bgl_nodetime": "bgl_nodetime.joblib"}
ALPHA = 0.005
OCSVM_SUB = 10_000

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, choices=FILES)
ap.add_argument("--split", default="chrono", choices=["chrono", "random"])
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--correlator", action="store_true")
ap.add_argument("--models", nargs="+",
                default=["PCA-SPE", "IForest", "MLP-AE", "OCSVM", "MarkovNext"])
args = ap.parse_args()

df = joblib.load(os.path.join(DATA, FILES[args.config]))["df"]
train, cal, test = split(df, mode=args.split, seed=0)
fx = CountVectors().fit(train.seq.tolist())
Xtr = np.hstack([fx.transform(train.seq.tolist()), seq_stats(train.seq.tolist())])
Xca = np.hstack([fx.transform(cal.seq.tolist()), seq_stats(cal.seq.tolist())])
Xte = np.hstack([fx.transform(test.seq.tolist()), seq_stats(test.seq.tolist())])
y = test.label.to_numpy()
print(f"[{args.config}/{args.split}] train={len(train)} cal={len(cal)} "
      f"test={len(test)} anomalies={int(y.sum())} dim={Xtr.shape[1]}", flush=True)

path = os.path.join(OUT, f"full_{args.config}_{args.split}.json")
store = json.load(open(path)) if os.path.exists(path) else {
    "config": {"alpha": ALPHA, "ocsvm_subsample": OCSVM_SUB,
               "split": "60/10/30 benign-only train+cal",
               "features": "frozen-vocab tfidf counts + seq stats + OOV bin"},
    "shape": {"train": len(train), "cal": len(cal), "test": len(test),
              "test_anomalies": int(y.sum()), "dim": int(Xtr.shape[1])},
    "models": {},
}


def evaluate(sc_cal, sc_te, secs):
    thr = threshold_from_benign(sc_cal, alpha=ALPHA)
    m = metrics(y, sc_te, thr)
    m["pr_auc"] = pr_auc(y, sc_te)
    m["seconds"] = round(secs, 1)
    return m


def record(name, seed, m):
    store["models"].setdefault(name, {})[str(seed)] = m
    print(f"{name:<16} seed={seed} P={m['precision']:.3f} R={m['recall']:.3f} "
          f"F1={m['f1']:.3f} FPR={m['fpr']:.4f} PRAUC={m['pr_auc']:.3f} "
          f"({m['seconds']}s)", flush=True)


def base_scores(name, seed):
    t0 = time.time()
    if name == "PCA-SPE":
        mdl = PCASPE().fit(Xtr)
    elif name == "IForest":
        mdl = IF(seed=seed).fit(Xtr)
    elif name == "MLP-AE":
        mdl = MLPAE(seed=seed, max_iter=60).fit(Xtr)
    elif name == "OCSVM":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xtr), size=min(OCSVM_SUB, len(Xtr)), replace=False)
        mdl = OCSVM().fit(Xtr[idx])
    elif name == "MarkovNext":
        mk = MarkovNext(k=2).fit_seqs(train.seq.tolist())
        return (mk.score_seqs(train.seq.tolist()), mk.score_seqs(cal.seq.tolist()),
                mk.score_seqs(test.seq.tolist()), time.time() - t0)
    else:
        raise ValueError(name)
    return mdl.score(Xtr), mdl.score(Xca), mdl.score(Xte), time.time() - t0


DETERMINISTIC = {"PCA-SPE", "MarkovNext"}
for name in args.models:
    seeds = [0] if name in DETERMINISTIC else args.seeds
    for seed in seeds:
        if str(seed) in store["models"].get(name, {}):
            continue  # resume support: skip already-recorded runs
        sc_tr, sc_ca, sc_te, secs = base_scores(name, seed)
        record(name, seed, evaluate(sc_ca, sc_te, secs))
        if args.correlator:
            t0 = time.time()
            corr = CrossComponentCorrelator(bucket=5000).fit(train, sc_tr)
            m = evaluate(corr.score(cal, sc_ca), corr.score(test, sc_te),
                         secs + (time.time() - t0))
            record(name + "+Corr", seed, m)
        json.dump(store, open(path, "w"), indent=2)

json.dump(store, open(path, "w"), indent=2)
print(f"saved {path}", flush=True)
