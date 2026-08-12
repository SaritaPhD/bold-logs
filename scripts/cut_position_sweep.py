"""Cut-position sweep on chronological BGL (reviewer-requested robustness).

Question: is the chronological-BGL collapse a regime, or an artifact of where
the 60/10/30 knife falls (Fig. drift shows vocabulary steps near windows 30k
and 42k)? We repeat the identity-feature protocol at three cut positions and
report detection metrics plus the drift statistics at each cut.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import joblib

from logad.features import CountVectors, seq_stats
from logad.models import PCASPE, MLPAE
from logad.eval import threshold_from_benign, metrics, pr_auc

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = 0.005
CUTS = [(0.5, 0.1), (0.6, 0.1), (0.7, 0.1)]

bundle = joblib.load(os.path.join(DATA, "bgl_global.joblib"))
df = bundle["df"].sort_values("order").reset_index(drop=True)
n = len(df)

results = {}
for tr_frac, ca_frac in CUTS:
    cut1, cut2 = int(n * tr_frac), int(n * (tr_frac + ca_frac))
    train = df.iloc[:cut1]; cal = df.iloc[cut1:cut2]; test = df.iloc[cut2:]
    train_b, cal_b = train[train.label == 0], cal[cal.label == 0]
    y = test.label.to_numpy()

    # drift statistics at this cut
    first_seen = {}
    for r in df.itertuples():
        for t in r.seq:
            if t not in first_seen or r.order < first_seen[t]:
                first_seen[t] = r.order
    cut_order = train.order.max()
    post = sum(1 for o in first_seen.values() if o > cut_order)
    train_vocab = {t for s in train_b.seq for t in s}
    def unseen_share(sub):
        evs = [t for s in sub.seq for t in s]
        return float(np.mean([t not in train_vocab for t in evs])) if evs else 0.0
    ben_u = unseen_share(test[test.label == 0])
    an_u = unseen_share(test[test.label == 1])

    fx = CountVectors().fit(train_b.seq.tolist())
    F = lambda sub: np.hstack([fx.transform(sub.seq.tolist()),
                               seq_stats(sub.seq.tolist())])
    Xtr, Xca, Xte = F(train_b), F(cal_b), F(test)

    row = {"train_frac": tr_frac,
           "test_units": len(test), "test_anomalies": int(y.sum()),
           "templates_post_cutoff": post, "templates_total": len(first_seen),
           "benign_test_unseen_share": round(ben_u, 3),
           "anom_test_unseen_share": round(an_u, 3), "models": {}}
    for name, m in [("PCA-SPE", PCASPE()), ("MLP-AE", MLPAE(seed=0, max_iter=60))]:
        m.fit(Xtr)
        thr = threshold_from_benign(m.score(Xca), alpha=ALPHA)
        r = metrics(y, m.score(Xte), thr)
        r["pr_auc"] = pr_auc(y, m.score(Xte))
        row["models"][name] = {k: round(v, 4) for k, v in r.items()}
        print(f"cut {int(tr_frac*100)}/10/{int((1-tr_frac-ca_frac)*100)} "
              f"{name:<8} F1={r['f1']:.3f} PRAUC={r['pr_auc']:.3f} "
              f"FPR={r['fpr']:.4f} | post-cutoff templates {post}/{len(first_seen)} "
              f"unseen: benign {ben_u:.2f} anom {an_u:.2f}", flush=True)
    results[f"{int(tr_frac*100)}/10/{int((1-tr_frac-ca_frac)*100)}"] = row

json.dump({"config": {"alpha": ALPHA, "features": "identity (frozen vocab)"},
           "results": results},
          open(os.path.join(OUT, "bgl_cut_sweep.json"), "w"), indent=2)
print("saved results/bgl_cut_sweep.json")
