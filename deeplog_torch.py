#!/usr/bin/env python3
"""DeepLog re-run (Du et al., CCS 2017) on the exported splits.

Faithful re-implementation of the log-key anomaly model: 2-layer LSTM
predicting the next template id; a window is anomalous when the true next
key is not in the top-g predictions; a unit (session/window) is anomalous
when any of its windows is. Trains ONLY on benign units.

Usage (per config):
  python deeplog_torch.py --data data/hdfs_chrono
  python deeplog_torch.py --data data/hdfs_random
  python deeplog_torch.py --data data/bgl_chrono
  python deeplog_torch.py --data data/bgl_random

Requires: pip install torch numpy   (GPU used automatically if available)
Writes: <data>/deeplog_results.json  — send this file back / drop it in the
connected folder. If anything errors, send the full traceback instead;
please never edit the JSON by hand.
"""
import argparse, gzip, json, os, random, time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def read_seqs(path):
    with gzip.open(path, "rt") as fh:
        return [[int(t) for t in ln.split()] for ln in fh if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--h", type=int, default=10, help="window size (paper: 10)")
    ap.add_argument("--g", type=int, default=9, help="top-g candidates (paper: 9)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--max_train_windows", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_seqs = read_seqs(os.path.join(args.data, "train.txt.gz"))
    test_n = read_seqs(os.path.join(args.data, "test_normal.txt.gz"))
    test_a = read_seqs(os.path.join(args.data, "test_abnormal.txt.gz"))

    # vocabulary from benign training only; unseen test keys -> OOV id V
    vocab = sorted({t for s in train_seqs for t in s})
    remap = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)  # OOV id = V; classes = V + 1
    print(f"train units={len(train_seqs)} vocab={V} device={dev}", flush=True)

    def encode(s):
        return [remap.get(t, V) for t in s]

    X, Y = [], []
    for s in train_seqs:
        e = encode(s)
        for i in range(len(e) - args.h):
            X.append(e[i:i + args.h]); Y.append(e[i + args.h])
    if len(X) > args.max_train_windows:
        idx = np.random.default_rng(args.seed).choice(
            len(X), size=args.max_train_windows, replace=False)
        X = [X[i] for i in idx]; Y = [Y[i] for i in idx]
    print(f"train windows={len(X)}", flush=True)

    Xt = torch.tensor(X, dtype=torch.long)
    Yt = torch.tensor(Y, dtype=torch.long)
    dl = DataLoader(TensorDataset(Xt, Yt), batch_size=args.batch, shuffle=True)

    class DeepLog(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V + 1, args.hidden)
            self.lstm = nn.LSTM(args.hidden, args.hidden, args.layers,
                                batch_first=True)
            self.fc = nn.Linear(args.hidden, V + 1)

        def forward(self, x):
            out, _ = self.lstm(self.emb(x))
            return self.fc(out[:, -1, :])

    model = DeepLog().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += float(loss) * len(xb)
        print(f"epoch {ep + 1}/{args.epochs} loss={tot / len(Xt):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    train_secs = time.time() - t0

    @torch.no_grad()
    def unit_flags(seqs):
        model.eval()
        flags = []
        for s in seqs:
            e = encode(s)
            if len(e) <= args.h:
                flags.append(int(any(t == V for t in e)))  # short unit: OOV key only
                continue
            wins = torch.tensor([e[i:i + args.h] for i in range(len(e) - args.h)],
                                dtype=torch.long)
            tgts = torch.tensor([e[i + args.h] for i in range(len(e) - args.h)],
                                dtype=torch.long)
            flag = False
            for b in range(0, len(wins), args.batch):
                logits = model(wins[b:b + args.batch].to(dev))
                topg = logits.topk(args.g, dim=1).indices.cpu()
                tg = tgts[b:b + args.batch]
                miss = (topg != tg.unsqueeze(1)).all(dim=1)
                oov_next = tg == V
                if bool((miss | oov_next).any()):
                    flag = True
                    break
            flags.append(int(flag))
        return np.array(flags)

    t0 = time.time()
    fn_ = unit_flags(test_n)
    fa_ = unit_flags(test_a)
    test_secs = time.time() - t0

    tp = int(fa_.sum()); fn = int(len(fa_) - tp)
    fp = int(fn_.sum()); tn = int(len(fn_) - fp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    result = {
        "method": "DeepLog (re-implementation, Du et al. CCS 2017)",
        "config_dir": os.path.basename(os.path.normpath(args.data)),
        "params": vars(args),
        "vocab_size": V,
        "metrics": {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "fpr": round(fpr, 4),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "train_seconds": round(train_secs, 1),
        "test_seconds": round(test_secs, 1),
        "device": str(dev), "torch": torch.__version__,
    }
    out = os.path.join(args.data, "deeplog_results.json")
    json.dump(result, open(out, "w"), indent=2)
    print(json.dumps(result["metrics"], indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
