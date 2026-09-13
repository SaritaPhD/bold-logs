import sys, os, json, numpy as np
sys.path.insert(0, "src")
from logad.eval import metrics, pr_auc, threshold_from_benign
from sklearn.metrics import precision_recall_curve
S = "results/scores"
def bestf1(y, s):
    p, r, t = precision_recall_curve(y, s); f = 2*p*r/np.maximum(p+r,1e-12); i=int(np.nanargmax(f[:-1])); return float(f[i]), float(t[i])
out = {}
for f in sorted(os.listdir(S)):
    d = np.load(os.path.join(S, f)); cond = f[:-4]; y = d["y"]; out[cond] = {}
    print("==", cond, "prev=%.4f" % y.mean())
    for det in ["PCA", "AE", "OCSVM", "IF", "Markov", "AEnoOOV"]:
        sc, st = d[f"{det}_cal"], d[f"{det}_test"]
        row = {"pr_auc": pr_auc(y, st), "best_f1": bestf1(y, st)[0]}
        for a in [0.001, 0.005, 0.01, 0.02, 0.05]:
            m = metrics(y, st, threshold_from_benign(sc, a)); row[f"a{a}"] = {"f1": m["f1"], "recall": m["recall"], "fpr": m["fpr"]}
        # oracle recalibration on the FIRST 2000 benign test units (chronological order within test)
        o = np.argsort(d["order_test"]); first = o[:2000]; fb = first[y[first] == 0]
        thr_o = threshold_from_benign(st[fb], 0.005); rest = o[2000:]
        m = metrics(y[rest], st[rest], thr_o); row["recal_first2000"] = {"f1": m["f1"], "recall": m["recall"], "fpr": m["fpr"], "n_benign_used": int(len(fb))}
        # oracle: threshold from ALL benign test units (upper bound for calibration alone)
        m = metrics(y, st, threshold_from_benign(st[y == 0], 0.005)); row["recal_alltestbenign"] = {"f1": m["f1"], "recall": m["recall"], "fpr": m["fpr"]}
        out[cond][det] = row
        print(f"  {det:<8} PRAUC={row['pr_auc']:.3f} bestF1={row['best_f1']:.3f} | " + " ".join(f"a{a}:F1={row[f'a{a}']['f1']:.3f}/FPR={row[f'a{a}']['fpr']:.4f}" for a in [0.001,0.005,0.01,0.02,0.05]) + f" | recal2000 F1={row['recal_first2000']['f1']:.3f} FPR={row['recal_first2000']['fpr']:.4f} | recalAll F1={row['recal_alltestbenign']['f1']:.3f}")
json.dump(out, open("results/score_analysis.json", "w"), indent=2)
