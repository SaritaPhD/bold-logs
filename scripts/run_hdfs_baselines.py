"""First real baseline run on HDFS sessions (chronological + random splits).

Writes results/hdfs_baselines.json with full config for the paper's run log.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import joblib

from logad.features import CountVectors, seq_stats
from logad.models import PCASPE, IF, OCSVM, MLPAE, MarkovNext
from logad.eval import split, threshold_from_benign, metrics, pr_auc

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.005
SEEDS = [0]
OCSVM_SUB = 10_000

bundle = joblib.load(os.path.join(DATA, "hdfs_sessions.joblib"))
df = bundle["df"]

all_results = {}
import os as _os
for mode in [_os.environ.get("SPLIT", "chrono")]:
    train, cal, test = split(df, mode=mode, seed=0)
    fx = CountVectors().fit(train.seq.tolist())
    Xtr = np.hstack([fx.transform(train.seq.tolist()), seq_stats(train.seq.tolist())])
    Xca = np.hstack([fx.transform(cal.seq.tolist()), seq_stats(cal.seq.tolist())])
    Xte = np.hstack([fx.transform(test.seq.tolist()), seq_stats(test.seq.tolist())])
    y = test.label.to_numpy()
    print(f"\n=== {mode} split: train={len(train)} cal={len(cal)} "
          f"test={len(test)} (anomalies={int(y.sum())}) dim={Xtr.shape[1]} ===")

    res = {}

    def run(name, fit_score, seeds=(0,)):
        per_seed = []
        for s in seeds:
            t0 = time.time()
            sc_cal, sc_te = fit_score(s)
            thr = threshold_from_benign(sc_cal, alpha=ALPHA)
            m = metrics(y, sc_te, thr)
            m["pr_auc"] = pr_auc(y, sc_te)
            m["fit_score_seconds"] = round(time.time() - t0, 1)
            per_seed.append(m)
        agg = {k: float(np.mean([m[k] for m in per_seed]))
               for k in ["precision", "recall", "f1", "fpr", "pr_auc"]}
        agg["seconds"] = float(np.mean([m["fit_score_seconds"] for m in per_seed]))
        agg["per_seed"] = per_seed
        res[name] = agg
        print(f"{name:<12} P={agg['precision']:.3f} R={agg['recall']:.3f} "
              f"F1={agg['f1']:.3f} FPR={agg['fpr']:.4f} PRAUC={agg['pr_auc']:.3f} "
              f"({agg['seconds']:.0f}s)")

    run("PCA-SPE", lambda s: (lambda m: (m.score(Xca), m.score(Xte)))(PCASPE().fit(Xtr)))
    run("IForest", lambda s: (lambda m: (m.score(Xca), m.score(Xte)))(IF(seed=s).fit(Xtr)), SEEDS)
    run("MLP-AE", lambda s: (lambda m: (m.score(Xca), m.score(Xte)))(MLPAE(seed=s, max_iter=60).fit(Xtr)), SEEDS)

    def ocsvm(s):
        rng = np.random.default_rng(s)
        idx = rng.choice(len(Xtr), size=min(OCSVM_SUB, len(Xtr)), replace=False)
        m = OCSVM().fit(Xtr[idx])
        return m.score(Xca), m.score(Xte)
    run("OCSVM-20k", ocsvm, SEEDS)

    def markov(s):
        m = MarkovNext(k=2).fit_seqs(train.seq.tolist())
        return m.score_seqs(cal.seq.tolist()), m.score_seqs(test.seq.tolist())
    run("MarkovNext", markov)

    all_results[mode] = {
        "n_train": len(train), "n_cal": len(cal), "n_test": len(test),
        "test_anomalies": int(y.sum()), "models": res,
    }

payload = {
    "dataset": "HDFS_v1 (LogHub, Zenodo 8196385)",
    "config": {"alpha": ALPHA, "seeds": SEEDS, "ocsvm_subsample": OCSVM_SUB,
               "split": "60/10/30 benign-only train+cal", "features":
               "frozen-vocab tfidf counts + seq stats"},
    "results": all_results,
}
with open(os.path.join(OUT, f"hdfs_baselines_{list(all_results)[0]}.json"), "w") as fh:
    json.dump(payload, fh, indent=2)
print("saved results/hdfs_baselines_" + list(all_results)[0] + ".json")
