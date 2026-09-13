"""Cross-component correlation layer (the novel contribution, v0).

Idea carried over from the CoTemp-Guard design: a distributed fault or attack
can keep every component's own anomaly score sub-threshold while the JOINT
elevation across components in the same time bucket is large. We aggregate
per-unit score elevations (deviation above each component's own benign
baseline) within time buckets and emit a collective score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class CrossComponentCorrelator:
    def __init__(self, bucket: int = 5000, kappa: float = 0.05,
                 gate: float = 1.0):
        self.bucket = bucket   # bucket width in `order` units (line rank)
        self.kappa = kappa
        self.gate = gate       # min own elevation (sigmas) to join a coalition

    def fit(self, df: pd.DataFrame, scores: np.ndarray):
        """Learn per-component benign baselines (mean, std) of unit scores."""
        d = df.assign(_s=scores)
        g = d.groupby("component")["_s"]
        self.mu = g.mean().to_dict()
        self.sd = (g.std().fillna(0.0) + 1e-9).to_dict()
        self.global_mu = float(np.mean(scores))
        self.global_sd = float(np.std(scores) + 1e-9)
        return self

    def _elev(self, comp, s):
        mu = self.mu.get(comp, self.global_mu)
        sd = self.sd.get(comp, self.global_sd)
        return max(0.0, (s - mu) / sd)

    def score(self, df: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
        """Collective score per unit: sum of positive elevations of DISTINCT
        components sharing the unit's time bucket, scaled by coalition size."""
        d = df.assign(_s=scores)
        d["_b"] = (d["order"] // self.bucket).astype(int)
        d["_e"] = [self._elev(c, s) for c, s in zip(d["component"], d["_s"])]
        hot = d[d["_e"] >= self.gate]  # noqa: E501  (v0 amplifier, kept for ablation)
        agg = hot.groupby("_b").agg(E=("_e", "sum"), n=("component", "nunique"))
        agg = agg[agg["n"] >= 2]  # a coalition needs >= 2 distinct components
        coll = (self.kappa * agg["E"] * agg["n"]).to_dict()
        own = d["_e"].to_numpy()
        bucket_score = d["_b"].map(coll).fillna(0.0).to_numpy()
        # bucket evidence only amplifies units that are themselves elevated
        gated = np.where(own >= self.gate, np.maximum(own, bucket_score), own)
        return gated


class DriftAwareNormalizer:
    """Inverted use of cross-component structure (v1, designed after v0's
    amplifier failed): per time bucket, the AMBIENT elevation (median across
    units in the bucket) is treated as environment drift and subtracted.
    A unit is anomalous for exceeding its bucket's ambient level, not for
    being elevated in absolute terms. Where drift is absent, ambient ~ 0 and
    the score reduces to the base elevation (harmless by construction)."""

    def __init__(self, bucket: int = 5000):
        self.bucket = bucket

    def fit(self, df, scores):
        import numpy as np
        d = df.assign(_s=scores)
        g = d.groupby("component")["_s"]
        self.mu = g.mean().to_dict()
        self.sd = (g.std().fillna(0.0) + 1e-9).to_dict()
        self.global_mu = float(np.mean(scores))
        self.global_sd = float(np.std(scores) + 1e-9)
        return self

    def _elev(self, comp, s):
        mu = self.mu.get(comp, self.global_mu)
        sd = self.sd.get(comp, self.global_sd)
        return (s - mu) / sd

    def score(self, df, scores):
        import numpy as np
        d = df.assign(_s=scores)
        d["_b"] = (d["order"] // self.bucket).astype(int)
        d["_e"] = [self._elev(c, s) for c, s in zip(d["component"], d["_s"])]
        ambient = d.groupby("_b")["_e"].transform("median")
        return (d["_e"] - ambient).to_numpy()
