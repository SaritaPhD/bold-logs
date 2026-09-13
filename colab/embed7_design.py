#!/usr/bin/env python3
"""The seventh benign-only design: FROZEN PRETRAINED EMBEDDINGS.

This is the direct test of the paper's central claim. The six failed designs
all used features fitted on (or hashed from) the observed data; this one
replaces them with frozen sentence-transformer embeddings of template text
-- an external semantic prior. If it also fails on chronological BGL, the
boundary claim strengthens; if it succeeds, the paper becomes constructive.
Either outcome goes in the manuscript exactly as measured.

Self-contained: needs numpy, scikit-learn, sentence-transformers.
Usage:  python embed7_design.py --data data/bgl_chrono --templates templates_bgl.json
Writes: <data>/embed7_results.json
"""
import argparse, gzip, json, os, time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score


def read_seqs(path):
    with gzip.open(path, "rt") as fh:
        return [[int(t) for t in ln.split()] for ln in fh if ln.strip()]


def seq_stats(seqs):
    rows = []
    for s in seqs:
        run = best = 1
        for a, b in zip(s[:-1], s[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        rows.append([len(s), len(set(s)), best])
    return np.asarray(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    templates = {int(k): v for k, v in json.load(open(args.templates)).items()}
    n_t = max(templates) + 1

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(args.model)
    texts = [templates.get(i, "") for i in range(n_t)]
    E = st.encode(texts, batch_size=64, show_progress_bar=False,
                  normalize_embeddings=True)          # (n_t, 384), frozen
    print(f"embedded {n_t} templates -> {E.shape}", flush=True)

    def unit_vecs(seqs):
        out = np.zeros((len(seqs), E.shape[1]))
        for i, s in enumerate(seqs):
            c = np.bincount([t for t in s if t < n_t], minlength=n_t).astype(float)
            tot = c.sum()
            if tot > 0:
                out[i] = c @ E / tot                   # mean event embedding
        return np.hstack([out, seq_stats(seqs)])

    benign = read_seqs(os.path.join(args.data, "train.txt.gz"))
    test_n = read_seqs(os.path.join(args.data, "test_normal.txt.gz"))
    test_a = read_seqs(os.path.join(args.data, "test_abnormal.txt.gz"))
    k = (len(benign) * 6) // 7
    Xfit, Xcal = unit_vecs(benign[:k]), unit_vecs(benign[k:])
    Xn, Xa = unit_vecs(test_n), unit_vecs(test_a)
    y = np.concatenate([np.zeros(len(Xn)), np.ones(len(Xa))])
    Xt = np.vstack([Xn, Xa])
    print(f"fit={len(Xfit)} cal={len(Xcal)} test={len(Xt)} "
          f"(anomalies={int(y.sum())})", flush=True)

    def evaluate(s_cal, s_te, secs):
        thr = float(np.quantile(s_cal, 1 - args.alpha))
        pred = (s_te > thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return {"precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(2*prec*rec/(prec+rec), 4) if prec + rec else 0.0,
                "fpr": round(fp/(fp+tn), 4) if fp + tn else 0.0,
                "pr_auc": round(float(average_precision_score(y, s_te)), 4),
                "seconds": round(secs, 1)}

    results = {}
    # PCA reconstruction (deterministic)
    t0 = time.time()
    sc = StandardScaler().fit(Xfit)
    p = PCA(n_components=0.95, svd_solver="full").fit(sc.transform(Xfit))
    def pca_score(X):
        Z = sc.transform(X)
        R = Z - p.inverse_transform(p.transform(Z))
        return (R ** 2).sum(axis=1)
    results["PCA-SPE+frozen-emb"] = evaluate(pca_score(Xcal), pca_score(Xt),
                                             time.time() - t0)
    print("PCA-SPE ", results["PCA-SPE+frozen-emb"], flush=True)

    # MLP autoencoder over embeddings (multi-seed)
    per_seed = []
    for s in args.seeds:
        t0 = time.time()
        scl = StandardScaler().fit(Xfit)
        Z = scl.transform(Xfit)
        m = MLPRegressor(hidden_layer_sizes=(128, 16, 128), random_state=s,
                         max_iter=60, early_stopping=True).fit(Z, Z)
        def ae_score(X):
            Zx = scl.transform(X)
            return ((Zx - m.predict(Zx)) ** 2).mean(axis=1)
        per_seed.append(evaluate(ae_score(Xcal), ae_score(Xt), time.time() - t0))
        print(f"MLP-AE seed={s} {per_seed[-1]}", flush=True)
    agg = {kk: round(float(np.mean([r[kk] for r in per_seed])), 4)
           for kk in ["precision", "recall", "f1", "fpr", "pr_auc"]}
    results["MLP-AE+frozen-emb"] = {"agg": agg, "per_seed": per_seed}

    out = os.path.join(args.data, "embed7_results.json")
    json.dump({"design": "7th benign-only design: frozen pretrained "
                          "sentence embeddings (external semantic prior)",
               "embedding_model": args.model,
               "config_dir": os.path.basename(os.path.normpath(args.data)),
               "alpha": args.alpha, "results": results},
              open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
