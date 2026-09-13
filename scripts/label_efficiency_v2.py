"""Label-efficiency v2 (chronological BGL, content features). Same causal
protocol as v1, but the operating point is chosen properly for a supervised
model: the labeled period is split by time into fit (first 85%) and
validation (last 15%); the threshold maximising F1 on the causal validation
slice is applied to the untouched test period. Also reports the oracle
best-F1 on test as an upper bound, and the v1 benign-quantile point."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from logad.features import TemplateContentVectors, seq_stats
from logad.eval import split, pr_auc, metrics
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
BUDGETS = [10, 25, 50, 100, 250, "all"]; SEEDS = [0, 1, 2]
bundle = joblib.load(os.path.join(DATA, "bgl_global.joblib")); df, templates = bundle["df"], bundle["templates"]
train, cal, test = split(df, mode="chrono", seed=0)
fx = TemplateContentVectors(templates, dim=128)
feats = lambda sub: np.hstack([fx.transform(sub.seq.tolist()), seq_stats(sub.seq.tolist())])
period = df[df.order <= cal.order.max()].sort_values("order")
fit = period  # full labeled period, as in v1 (models identical to v1)
Xt, yt = feats(test), test.label.to_numpy()
fb, fa = fit[fit.label == 0], fit[fit.label == 1]
Xfb, Xfa = feats(fb), feats(fa)
print(f"fit: benign={len(fb)} anom={len(fa)} | test anom={int(yt.sum())}")
def best_f1(y, s):
    p, r, t = precision_recall_curve(y, s); f = 2*p*r/np.maximum(p+r, 1e-12); i = int(np.nanargmax(f[:-1])); return float(f[i]), float(t[i])
results = {}
for N in BUDGETS:
    per = []
    for s in (SEEDS if N != "all" else [0]):
        rng = np.random.default_rng(s)
        Xa_s = Xfa if N == "all" else Xfa[rng.choice(len(Xfa), size=min(N, len(Xfa)), replace=False)]
        X = np.vstack([Xfb, Xa_s]); y = np.concatenate([np.zeros(len(Xfb)), np.ones(len(Xa_s))])
        rf = RandomForestClassifier(n_estimators=300, random_state=s, class_weight="balanced", n_jobs=2, oob_score=True).fit(X, y)
        pt = rf.predict_proba(Xt)[:, 1]
        oob = rf.oob_decision_function_[:, 1]; ok = ~np.isnan(oob)
        _, thr_val = best_f1(y[ok], oob[ok])  # causal: out-of-bag on the labeled period only
        m_val = metrics(yt, pt, thr_val)
        bf, _ = best_f1(yt, pt)
        thr_v1 = float(np.quantile(rf.predict_proba(Xfb)[:, 1], 0.995)); m_v1 = metrics(yt, pt, thr_v1)
        per.append({"pr_auc": pr_auc(yt, pt), "f1_valthr": m_val["f1"], "recall_valthr": m_val["recall"], "fpr_valthr": m_val["fpr"],
                    "f1_oracle": bf, "f1_v1_benignq": m_v1["f1"], "fpr_v1_benignq": m_v1["fpr"], "n_labels_used": int(len(Xa_s))})
    agg = {k: float(np.mean([r[k] for r in per])) for k in per[0]}; agg["std_f1_valthr"] = float(np.std([r["f1_valthr"] for r in per]))
    results[str(N)] = {"agg": agg, "per_seed": per}
    print(f"N={str(N):>4}: PRAUC={agg['pr_auc']:.3f} F1@val={agg['f1_valthr']:.3f}±{agg['std_f1_valthr']:.3f} (R={agg['recall_valthr']:.3f} FPR={agg['fpr_valthr']:.4f}) oracleF1={agg['f1_oracle']:.3f} v1F1={agg['f1_v1_benignq']:.3f}@FPR{agg['fpr_v1_benignq']:.3f}", flush=True)
json.dump({"protocol": "fit=full train+cal labeled period (as v1); threshold=argmax F1 on out-of-bag period predictions (causal); oracle=best F1 on test",
           "fit_anomalies_available": int(len(fa)), "results": results},
          open(os.path.join(OUT, "bgl_label_efficiency_v2.json"), "w"), indent=2)
