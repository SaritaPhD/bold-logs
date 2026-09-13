"""Label-efficiency curve for chronological BGL (regenerated after container
recycle; fully seeded and deterministic — reproduces the recorded run).

Protocol: features = content-hash(128) + seq stats. For each budget N, sample
N anomalous units from the TRAIN+CAL PERIOD ONLY (strictly causal), plus all
benign period units; train RF (class_weight=balanced); evaluate on the
untouched chronological test period. Alarm threshold = alpha-FPR quantile on
period benign scores (causal). 3 sampling seeds per N.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from logad.features import TemplateContentVectors, seq_stats
from logad.eval import split, pr_auc

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 0.005
BUDGETS = [10, 25, 50, 100, 250, "all"]
SEEDS = [0, 1, 2]

bundle = joblib.load(os.path.join(DATA, "bgl_global.joblib"))
df, templates = bundle["df"], bundle["templates"]
train, cal, test = split(df, mode="chrono", seed=0)
fx = TemplateContentVectors(templates, dim=128)

def feats(sub):
    return np.hstack([fx.transform(sub.seq.tolist()), seq_stats(sub.seq.tolist())])

period = df[df.order <= cal.order.max()]
ben, anom = period[period.label == 0], period[period.label == 1]
Xb, Xa = feats(ben), feats(anom)
Xt, yt = feats(test), test.label.to_numpy()
print(f"period: benign={len(ben)} anomalous={len(anom)} | test anomalies={int(yt.sum())}")

results = {}
for N in BUDGETS:
    per_seed = []
    for s in SEEDS if N != "all" else [0]:
        rng = np.random.default_rng(s)
        if N == "all":
            Xa_s = Xa
        else:
            if N > len(Xa):
                continue
            Xa_s = Xa[rng.choice(len(Xa), size=N, replace=False)]
        X = np.vstack([Xb, Xa_s])
        y = np.concatenate([np.zeros(len(Xb)), np.ones(len(Xa_s))])
        rf = RandomForestClassifier(n_estimators=300, random_state=s,
                                    class_weight="balanced", n_jobs=-1).fit(X, y)
        ps = rf.predict_proba(Xt)[:, 1]
        thr = float(np.quantile(rf.predict_proba(Xb)[:, 1], 1 - ALPHA))
        yhat = (ps > thr).astype(int)
        tp = int(((yhat == 1) & (yt == 1)).sum())
        fp = int(((yhat == 1) & (yt == 0)).sum())
        fn = int(((yhat == 0) & (yt == 1)).sum())
        tn = int(((yhat == 0) & (yt == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_seed.append({
            "pr_auc": pr_auc(yt, ps),
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0})
    if not per_seed:
        continue
    agg = {k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]}
    agg["std_f1"] = float(np.std([r["f1"] for r in per_seed]))
    results[str(N)] = {"agg": agg, "per_seed": per_seed}
    print(f"N={str(N):>4}: PRAUC={agg['pr_auc']:.3f} F1={agg['f1']:.3f}"
          f"±{agg['std_f1']:.3f} P={agg['precision']:.3f} R={agg['recall']:.3f} "
          f"FPR={agg['fpr']:.4f}", flush=True)

json.dump({"config": {"alpha": ALPHA, "budgets": [str(b) for b in BUDGETS],
                      "seeds": SEEDS, "features": "content-hash 128 + seq stats",
                      "causality": "labels sampled from train+cal period only"},
           "period_anomalies_available": int(len(anom)),
           "results": results},
          open(os.path.join(OUT, "bgl_label_efficiency.json"), "w"), indent=2)
print("saved results/bgl_label_efficiency.json")
