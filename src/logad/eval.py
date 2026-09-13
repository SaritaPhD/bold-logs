"""Leak-free evaluation protocol.

Design follows the pitfalls catalogued by Le & Zhang, "Log-based Anomaly
Detection with Deep Learning: How Far Are We?" (ICSE 2022):
  * chronological split option (no future templates in training),
  * benign-only training with vocabulary frozen on the training split,
  * thresholds set by benign-calibration quantile on a held-out benign slice,
  * multi-seed runs with bootstrap confidence intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split(df: pd.DataFrame, mode: str = "chrono", train_frac: float = 0.6,
          cal_frac: float = 0.1, seed: int = 0):
    """Return (train_benign, calib_benign, test) DataFrames.
    train/calib contain ONLY benign units; test contains the remainder."""
    if mode == "chrono":
        d = df.sort_values("order").reset_index(drop=True)
    elif mode == "random":
        d = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        raise ValueError(mode)
    n = len(d)
    cut1, cut2 = int(n * train_frac), int(n * (train_frac + cal_frac))
    head, mid, test = d.iloc[:cut1], d.iloc[cut1:cut2], d.iloc[cut2:]
    return head[head.label == 0], mid[mid.label == 0], test


def threshold_from_benign(cal_scores: np.ndarray, alpha: float = 0.005):
    """Alarm threshold = (1-alpha) quantile of benign calibration scores."""
    return float(np.quantile(cal_scores, 1.0 - alpha))


def metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    yhat = (s > thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def bootstrap_ci(values, n_boot: int = 10_000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI over per-seed metric values."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(v.mean()), float(lo), float(hi)


def cliffs_delta(a, b) -> float:
    """Effect size between two samples of per-seed scores."""
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))
