"""Feature builders: template-count vectors (+ tf-idf style weighting) and
sequence-derived statistics per unit."""
from __future__ import annotations

import numpy as np


class CountVectors:
    """Unit -> V-dim template count vector. Vocabulary is FROZEN on the
    training (benign) split; unseen templates at test time map to an OOV bin —
    this avoids the vocabulary-leakage pitfall flagged by Le & Zhang (ICSE'22).
    """

    def __init__(self, tfidf: bool = True):
        self.tfidf = tfidf
        self.vocab: dict[int, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, seqs) -> "CountVectors":
        for s in seqs:
            for t in s:
                self.vocab.setdefault(t, len(self.vocab))
        X = self._counts(seqs)
        if self.tfidf:
            df = (X > 0).sum(axis=0) + 1
            self.idf = np.log((len(seqs) + 1) / df)
        return self

    def transform(self, seqs) -> np.ndarray:
        X = self._counts(seqs)
        if self.tfidf and self.idf is not None:
            X = X * self.idf
        return X

    def _counts(self, seqs) -> np.ndarray:
        V = len(self.vocab) + 1  # last column = OOV
        X = np.zeros((len(seqs), V))
        for i, s in enumerate(seqs):
            for t in s:
                X[i, self.vocab.get(t, V - 1)] += 1
        return X


class TemplateContentVectors:
    """Content-based unit features that GENERALIZE ACROSS TEMPLATE DRIFT.

    Each template string is hashed into a fixed D-dim bag-of-words vector
    (stable md5 token hashing with sign trick; mask tokens like <*>/<NUM>
    skipped). A unit's feature vector is the sum of its events' template
    vectors. Nothing is fitted on data — the mapping is pure content — so a
    template first seen at test time still lands near templates that share
    its words ("parity error corrected" ≈ other parity-error messages).
    Motivated by the BGL drift diagnosis (results/bgl_drift_diagnosis.json):
    42% of templates appear only after the train cutoff, killing identity
    features; published semantic methods (LogRobust, NeuralLog) survive
    drift for exactly this reason, at embedding cost. This is the
    lightweight equivalent.
    """

    MASKS = {"<*>", "<NUM>", "<IP>", "<HEX>", "<BLK>", "<EMPTY>"}

    def __init__(self, templates: dict[int, str], dim: int = 128,
                 log_scale: bool = True):
        import hashlib
        import re as _re
        self.dim = dim
        self.log_scale = log_scale
        n = max(templates) + 1
        self.T = np.zeros((n, dim))
        word = _re.compile(r"[a-z0-9]+")
        for tid, tpl in templates.items():
            for tok in tpl.split():
                if tok in self.MASKS:
                    continue
                for w in word.findall(tok.lower()):
                    h = hashlib.md5(w.encode()).digest()
                    idx = int.from_bytes(h[:4], "little") % dim
                    sign = 1.0 if h[4] % 2 else -1.0
                    self.T[tid, idx] += sign

    def transform(self, seqs) -> np.ndarray:
        n = self.T.shape[0]
        out = np.zeros((len(seqs), self.dim))
        for i, s in enumerate(seqs):
            c = np.bincount([t for t in s if t < n], minlength=n)
            out[i] = c @ self.T
        if self.log_scale:
            out = np.sign(out) * np.log1p(np.abs(out))
        return out


def seq_stats(seqs) -> np.ndarray:
    """Cheap per-unit sequence statistics: length, distinct templates,
    max run length, transition novelty placeholder columns."""
    rows = []
    for s in seqs:
        arr = np.asarray(s)
        run, best = 1, 1
        for a, b in zip(arr[:-1], arr[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        rows.append([len(arr), len(set(s)), best])
    return np.asarray(rows, dtype=float)
