"""Benign-only (unsupervised) detectors with a common fit/score API.

All models: fit(X_benign) then score(X) where HIGHER = more anomalous.
Classical baselines run in-repo; deep baselines (DeepLog, LogBERT, ...) are
compared via their published numbers on identical benchmarks and, optionally,
re-run on user hardware.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class PCASPE:
    """PCA with squared prediction error (He et al., classical log AD)."""

    def __init__(self, var: float = 0.95):
        self.var = var

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.pca = PCA(n_components=self.var, svd_solver="full").fit(Z)
        return self

    def score(self, X):
        Z = self.scaler.transform(X)
        R = Z - self.pca.inverse_transform(self.pca.transform(Z))
        return (R ** 2).sum(axis=1)


class IF:
    def __init__(self, trees: int = 200, seed: int = 0):
        self.m = IsolationForest(n_estimators=trees, random_state=seed)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class OCSVM:
    def __init__(self, nu: float = 0.05):
        self.m = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        self.m.fit(self.scaler.transform(X))
        return self

    def score(self, X):
        return -self.m.decision_function(self.scaler.transform(X))


class LOF:
    def __init__(self, k: int = 20):
        self.m = LocalOutlierFactor(n_neighbors=k, novelty=True)

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        self.m.fit(self.scaler.transform(X))
        return self

    def score(self, X):
        return -self.m.decision_function(self.scaler.transform(X))


class MLPAE:
    """Bottleneck autoencoder via sklearn MLPRegressor (X -> X)."""

    def __init__(self, hidden=(64, 16, 64), seed: int = 0, max_iter: int = 200):
        self.m = MLPRegressor(hidden_layer_sizes=hidden, random_state=seed,
                              max_iter=max_iter, early_stopping=True)

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.m.fit(Z, Z)
        return self

    def score(self, X):
        Z = self.scaler.transform(X)
        P = self.m.predict(Z)
        return ((Z - P) ** 2).mean(axis=1)


class MarkovNext:
    """Order-k template-transition model: unit score = mean surprisal of its
    transitions under benign statistics. Our own classical sequence baseline
    (NOT a reimplementation of DeepLog)."""

    def __init__(self, k: int = 2, alpha: float = 0.1):
        self.k, self.alpha = k, alpha

    def fit_seqs(self, seqs):
        self.counts = defaultdict(lambda: defaultdict(float))
        self.totals = defaultdict(float)
        self.vocab = set()
        for s in seqs:
            for i in range(len(s) - self.k):
                ctx, nxt = tuple(s[i:i + self.k]), s[i + self.k]
                self.counts[ctx][nxt] += 1
                self.totals[ctx] += 1
                self.vocab.add(nxt)
        self.V = max(len(self.vocab), 1)
        return self

    def score_seqs(self, seqs):
        out = []
        for s in seqs:
            sur = [0.0]
            for i in range(len(s) - self.k):
                ctx, nxt = tuple(s[i:i + self.k]), s[i + self.k]
                p = (self.counts[ctx][nxt] + self.alpha) / (
                    self.totals[ctx] + self.alpha * self.V)
                sur.append(-np.log(p))
            out.append(float(np.mean(sur)) if len(sur) > 1 else 0.0)
        return np.asarray(out)
