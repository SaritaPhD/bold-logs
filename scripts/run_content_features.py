"""Content-hashed features vs template-ID features under drift.

The decisive experiment motivated by the drift diagnosis: do content-based
features (which generalize to never-seen templates) rescue the chronological
splits that identity features cannot survive?

Feature sets compared, identical models and protocol:
  ids       — frozen-vocab tf-idf counts + seq stats (the original)
  content   — hashed template-content vectors (drift-robust by construction)
  both      — concatenation
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import joblib

from logad.features import CountVectors, TemplateContentVectors, seq_stats
from logad.models import PCASPE, MLPAE
from logad.eval import split, threshold_from_benign, metrics, pr_auc

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 0.005

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="bgl_global",
                choices=["bgl_global", "hdfs", "bgl_nodetime"])
ap.add_argument("--split", default="chrono", choices=["chrono", "random"])
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
args = ap.parse_args()

FILES = {"hdfs": "hdfs_sessions.joblib", "bgl_global": "bgl_global.joblib",
         "bgl_nodetime": "bgl_nodetime.joblib"}
bundle = joblib.load(os.path.join(DATA, FILES[args.config]))
df, templates = bundle["df"], bundle["templates"]

train, cal, test = split(df, mode=args.split, seed=0)
y = test.label.to_numpy()

fx_ids = CountVectors().fit(train.seq.tolist())
fx_ct = TemplateContentVectors(templates, dim=128)

def build(which, sub):
    seqs = sub.seq.tolist()
    if which == "ids":
        return np.hstack([fx_ids.transform(seqs), seq_stats(seqs)])
    if which == "content":
        return np.hstack([fx_ct.transform(seqs), seq_stats(seqs)])
    return np.hstack([fx_ids.transform(seqs), fx_ct.transform(seqs),
                      seq_stats(seqs)])

print(f"[{args.config}/{args.split}] train={len(train)} cal={len(cal)} "
      f"test={len(test)} anomalies={int(y.sum())}", flush=True)

results = {}
for which in ["ids", "content", "both"]:
    Xtr, Xca, Xte = build(which, train), build(which, cal), build(which, test)
    for mname, make in [("PCA-SPE", lambda s: PCASPE()),
                        ("MLP-AE", lambda s: MLPAE(seed=s, max_iter=60))]:
        per_seed = []
        for s in (args.seeds if mname == "MLP-AE" else [0]):
            t0 = time.time()
            m = make(s).fit(Xtr)
            thr = threshold_from_benign(m.score(Xca), alpha=ALPHA)
            r = metrics(y, m.score(Xte), thr)
            r["pr_auc"] = pr_auc(y, m.score(Xte))
            r["seconds"] = round(time.time() - t0, 1)
            per_seed.append(r)
        agg = {k: float(np.mean([x[k] for x in per_seed]))
               for k in ["precision", "recall", "f1", "fpr", "pr_auc"]}
        results[f"{which}/{mname}"] = {"agg": agg, "per_seed": per_seed}
        print(f"{which:<8} {mname:<8} P={agg['precision']:.3f} "
              f"R={agg['recall']:.3f} F1={agg['f1']:.3f} "
              f"FPR={agg['fpr']:.4f} PRAUC={agg['pr_auc']:.3f}", flush=True)

path = os.path.join(OUT, f"content_{args.config}_{args.split}.json")
json.dump({"config": {"alpha": ALPHA, "dim": 128, "seeds": args.seeds},
           "results": results}, open(path, "w"), indent=2)
print("saved", path)
