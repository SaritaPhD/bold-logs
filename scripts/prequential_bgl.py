"""Prequential (rolling) evaluation on chronological BGL global windows.

Protocol: initial fit on the first INIT windows (benign-only, as before).
Then advance in chunks: score the incoming chunk with the CURRENT model
(these predictions are what we evaluate — strictly causal), then refresh
the model on the trailing HISTORY of units NOT flagged by the model itself
(self-cleaning; ground-truth labels are NEVER used for training/refresh).
A static model (fit once, never refreshed) runs on the identical stream for
the comparison. Ground-truth labels are used for final evaluation only.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import joblib

from logad.features import CountVectors, seq_stats
from logad.models import PCASPE, MLPAE
from logad.eval import threshold_from_benign, metrics, pr_auc

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

INIT_FRAC = 0.2
CHUNK = 2000
HISTORY = 15000
ALPHA = 0.005

MODELS = {"PCA-SPE": lambda s: PCASPE(),
          "MLP-AE": lambda s: MLPAE(seed=s, max_iter=60)}


def featurize(fx, seqs):
    return np.hstack([fx.transform(seqs), seq_stats(seqs)])


def run(df, model_name, seed=0):
    n = len(df)
    init = int(n * INIT_FRAC)
    hist_seqs = df.seq.iloc[:init].tolist()          # pseudo-benign history
    make = MODELS[model_name]

    def fit(seqs):
        fx = CountVectors().fit(seqs)
        X = featurize(fx, seqs)
        m = make(seed).fit(X)
        thr = threshold_from_benign(m.score(X), alpha=ALPHA)
        return fx, m, thr

    t0 = time.time()
    fx, m, thr = fit(hist_seqs)
    static = (fx, m, thr)
    refresh_times = [time.time() - t0]

    rows = []
    for start in range(init, n, CHUNK):
        chunk = df.iloc[start:start + CHUNK]
        seqs = chunk.seq.tolist()
        # 1) causal scoring with current rolling model
        sc = m.score(featurize(fx, seqs))
        flags = sc > thr
        # 2) causal scoring with the frozen static model
        sfx, sm, sthr = static
        ssc = sm.score(featurize(sfx, seqs))
        rows.append({"y": chunk.label.to_numpy(), "roll_s": sc, "roll_f": flags,
                     "stat_s": ssc, "stat_f": ssc > sthr})
        # 3) self-cleaned refresh (no ground truth): keep unflagged units
        hist_seqs = (hist_seqs + [s for s, f in zip(seqs, flags) if not f])[-HISTORY:]
        t0 = time.time()
        fx, m, thr = fit(hist_seqs)
        refresh_times.append(time.time() - t0)

    y = np.concatenate([r["y"] for r in rows])
    out = {}
    for tag in ["roll", "stat"]:
        s = np.concatenate([r[f"{tag}_s"] for r in rows])
        f = np.concatenate([r[f"{tag}_f"] for r in rows])
        mm = metrics(y, s, 0.0)  # placeholder; use flags directly:
        tp = int(((f) & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
        fn = int(((~f) & (y == 1)).sum()); tn = int(((~f) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[tag] = {"precision": prec, "recall": rec,
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                    "fpr": fp / (fp + tn) if fp + tn else 0.0,
                    "pr_auc": pr_auc(y, s)}
    out["mean_refresh_seconds"] = float(np.mean(refresh_times))
    out["n_refreshes"] = len(refresh_times) - 1
    out["evaluated_units"] = int(len(y))
    out["anomalies"] = int(y.sum())
    return out


if __name__ == "__main__":
    df = joblib.load(os.path.join(DATA, "bgl_global.joblib"))["df"]
    df = df.sort_values("order").reset_index(drop=True)
    results = {}
    for name in ["PCA-SPE", "MLP-AE"]:
        r = run(df, name)
        results[name] = r
        print(f"{name}: rolling F1={r['roll']['f1']:.3f} "
              f"(P={r['roll']['precision']:.3f} R={r['roll']['recall']:.3f} "
              f"FPR={r['roll']['fpr']:.4f} PRAUC={r['roll']['pr_auc']:.3f}) | "
              f"static F1={r['stat']['f1']:.3f} "
              f"(PRAUC={r['stat']['pr_auc']:.3f}) | "
              f"refresh={r['mean_refresh_seconds']:.1f}s x{r['n_refreshes']}",
              flush=True)
    with open(os.path.join(OUT, "bgl_prequential.json"), "w") as fh:
        json.dump({"config": {"init_frac": INIT_FRAC, "chunk": CHUNK,
                              "history": HISTORY, "alpha": ALPHA,
                              "self_cleaning": "model's own flags; labels never used"},
                   "results": results}, fh, indent=2)
    print("saved results/bgl_prequential.json")
