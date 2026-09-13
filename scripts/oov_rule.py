"""Trivial leak-free OOV-rule baseline: vocabulary frozen on benign TRAIN units
(same as CountVectors), unit flagged iff it contains >= k unseen templates.
Also reports PR-AUC of the OOV-share score. Deterministic; no threshold needed."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, joblib
from logad.eval import split, metrics, pr_auc
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FILES = {"hdfs": "hdfs_sessions.joblib", "bgl_global": "bgl_global.joblib", "thunderbird": "thunderbird_global.joblib"}
out = {}
for cfg, f in FILES.items():
    df = joblib.load(os.path.join(DATA, f))["df"]
    for mode in ["chrono", "random"]:
        train, cal, test = split(df, mode=mode, seed=0)
        vocab = {t for s in train.seq for t in s}
        y = test.label.to_numpy()
        cnt = np.array([sum(t not in vocab for t in s) for s in test.seq], dtype=float)
        share = cnt / np.array([len(s) for s in test.seq], dtype=float)
        res = {"vocab": len(vocab), "pr_auc_count": pr_auc(y, cnt), "pr_auc_share": pr_auc(y, share)}
        for k in [1, 2, 5]:
            m = metrics(y, cnt, k - 0.5)  # flag iff cnt >= k
            res[f"k{k}"] = m
        # benign-calibrated variant of the share score (rule 3), for comparison
        cal_share = np.array([sum(t not in vocab for t in s) / len(s) for s in cal.seq])
        thr = float(np.quantile(cal_share, 0.995))
        res["share_calibrated_thr"] = thr
        res["share_calibrated"] = metrics(y, share, thr)
        out[f"{cfg}_{mode}"] = res
        m = res["k1"]
        print(f"{cfg:<11} {mode:<6} V={len(vocab):4d} k>=1: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} FPR={m['fpr']:.4f} | PRAUC(count)={res['pr_auc_count']:.3f} share={res['pr_auc_share']:.3f} | cal-thr={thr:.3f} F1={res['share_calibrated']['f1']:.3f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "oov_rule.json"), "w"), indent=2)
