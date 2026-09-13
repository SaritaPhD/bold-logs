"""THE candidate headline experiment: content-hashed features + rolling
benign-only recalibration on chronological BGL.

Rationale chain (all previously measured, see results/):
  * identity features: no transferable signal under drift (probe PRAUC 0.24)
  * content features: strong transferable signal (probe PRAUC 0.85)
  * static benign-only models on content: fail (calibration, not signal)
  * -> rolling model+threshold on content features should convert the
    signal into detections; affordable ONLY because refits cost seconds.

Protocol (strictly causal, labels used for evaluation only):
  initial fit on first 20%; advance in chunks; score each chunk with the
  current model; refresh model on trailing self-cleaned history (units not
  flagged by the model itself); threshold from the most recent 20% of the
  history (held out from the fit) at the benign quantile.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import joblib

from logad.features import TemplateContentVectors, seq_stats
from logad.models import PCASPE, MLPAE
from logad.eval import threshold_from_benign, pr_auc

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

INIT_FRAC, CHUNK, HISTORY, ALPHA = 0.2, 2000, 15000, 0.005

bundle = joblib.load(os.path.join(DATA, "bgl_global.joblib"))
df, templates = bundle["df"], bundle["templates"]
df = df.sort_values("order").reset_index(drop=True)
fx = TemplateContentVectors(templates, dim=128)   # static content map


def featurize(seqs):
    return np.hstack([fx.transform(seqs), seq_stats(seqs)])


def run(model_name, make, seed=0):
    n = len(df)
    init = int(n * INIT_FRAC)
    hist = df.seq.iloc[:init].tolist()

    def fit(seqs):
        k = max(1, int(len(seqs) * 0.8))
        fit_seqs, cal_seqs = seqs[:k], seqs[k:] or seqs[-max(1, len(seqs)//5):]
        m = make(seed).fit(featurize(fit_seqs))
        thr = threshold_from_benign(m.score(featurize(cal_seqs)), alpha=ALPHA)
        return m, thr

    t0 = time.time()
    m, thr = fit(hist)
    static = (m, thr)
    refresh = [time.time() - t0]

    ys, roll_s, roll_f, stat_s, stat_f = [], [], [], [], []
    for start in range(init, n, CHUNK):
        chunk = df.iloc[start:start + CHUNK]
        X = featurize(chunk.seq.tolist())
        sc = m.score(X)
        flags = sc > thr
        ys.append(chunk.label.to_numpy())
        roll_s.append(sc); roll_f.append(flags)
        sm, sthr = static
        ssc = sm.score(X)
        stat_s.append(ssc); stat_f.append(ssc > sthr)
        hist = (hist + [s for s, f in zip(chunk.seq.tolist(), flags) if not f])[-HISTORY:]
        t0 = time.time()
        m, thr = fit(hist)
        refresh.append(time.time() - t0)

    y = np.concatenate(ys)
    out = {}
    for tag, ss, ff in [("rolling", roll_s, roll_f), ("static", stat_s, stat_f)]:
        s = np.concatenate(ss); f = np.concatenate(ff)
        tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
        fn = int((~f & (y == 1)).sum()); tn = int((~f & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[tag] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
                    "fpr": round(fp / (fp + tn), 4) if fp + tn else 0.0,
                    "pr_auc": round(pr_auc(y, s), 4)}
    out["mean_refresh_seconds"] = round(float(np.mean(refresh)), 2)
    out["evaluated_units"] = int(len(y)); out["anomalies"] = int(y.sum())
    return out


results = {}
for name, make in [("PCA-SPE", lambda s: PCASPE()),
                   ("MLP-AE", lambda s: MLPAE(seed=s, max_iter=60))]:
    r = run(name, make)
    results[name] = r
    print(f"{name}: rolling F1={r['rolling']['f1']:.3f} "
          f"(P={r['rolling']['precision']:.3f} R={r['rolling']['recall']:.3f} "
          f"FPR={r['rolling']['fpr']:.4f} PRAUC={r['rolling']['pr_auc']:.3f}) | "
          f"static F1={r['static']['f1']:.3f} (PRAUC={r['static']['pr_auc']:.3f}) | "
          f"refresh={r['mean_refresh_seconds']}s", flush=True)

json.dump({"config": {"init_frac": INIT_FRAC, "chunk": CHUNK, "history": HISTORY,
                      "alpha": ALPHA, "features": "content-hash dim=128 + seq stats",
                      "self_cleaning": "model's own flags; labels never used for training"},
           "results": results},
          open(os.path.join(OUT, "bgl_prequential_content.json"), "w"), indent=2)
print("saved results/bgl_prequential_content.json")
