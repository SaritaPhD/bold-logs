#!/usr/bin/env python3
"""DeepLog re-run — v1.1.

CHANGELOG vs deeplog_torch.py (v1.0), agreed with Claude before use:
  * Device selection now tries CUDA, then Apple MPS, then CPU; --device
    overrides. (v1.0 fell back to CPU on Apple silicon — metrics unaffected,
    only timing fields.)
  * --g now accepts one or more values (e.g. --g 3 5 7 9), all evaluated in
    a single pass over one trained model — a free sensitivity sweep.
  * With a single --g value the output file and metrics are identical in
    meaning to v1.0 (deeplog_results.json). With several values it writes
    deeplog_results_multi_g.json holding per-g metrics.
Defaults are unchanged: h=10, g=9, epochs=5, hidden=64x2, seed=0.

Usage examples:
  python deeplog_torch_v1_1.py --data data/bgl_random --g 3 5 7 9
  python deeplog_torch_v1_1.py --data data/hdfs_chrono --g 3 5 7 9 --epochs 10
"""
import argparse, gzip, json, os, random, time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def read_seqs(path):
    with gzip.open(path, "rt") as fh:
        return [[int(t) for t in ln.split()] for ln in fh if ln.strip()]


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--h", type=int, default=10)
    ap.add_argument("--g", type=int, nargs="+", default=[9])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--max_train_windows", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | mps")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = pick_device(args.device)
    gs = sorted(set(args.g))
    gmax = max(gs)

    train_seqs = read_seqs(os.path.join(args.data, "train.txt.gz"))
    test_n = read_seqs(os.path.join(args.data, "test_normal.txt.gz"))
    test_a = read_seqs(os.path.join(args.data, "test_abnormal.txt.gz"))

    vocab = sorted({t for s in train_seqs for t in s})
    remap = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    print(f"train units={len(train_seqs)} vocab={V} device={dev} g={gs}", flush=True)

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
        """Return {g: np.array(flags)} for every g in one pass."""
        model.eval()
        flags = {g: [] for g in gs}
        for s in seqs:
            e = encode(s)
            if len(e) <= args.h:
                f = int(any(t == V for t in e))
                for g in gs:
                    flags[g].append(f)
                continue
            wins = torch.tensor([e[i:i + args.h] for i in range(len(e) - args.h)],
                                dtype=torch.long)
            tgts = torch.tensor([e[i + args.h] for i in range(len(e) - args.h)],
                                dtype=torch.long)
            unit_flag = {g: False for g in gs}
            for b in range(0, len(wins), args.batch):
                logits = model(wins[b:b + args.batch].to(dev))
                topk = logits.topk(gmax, dim=1).indices.cpu()
                tg = tgts[b:b + args.batch]
                oov_next = tg == V
                for g in gs:
                    if unit_flag[g]:
                        continue
                    miss = (topk[:, :g] != tg.unsqueeze(1)).all(dim=1)
                    if bool((miss | oov_next).any()):
                        unit_flag[g] = True
                if all(unit_flag.values()):
                    break
            for g in gs:
                flags[g].append(int(unit_flag[g]))
        return {g: np.array(v) for g, v in flags.items()}

    t0 = time.time()
    fn_ = unit_flags(test_n)
    fa_ = unit_flags(test_a)
    test_secs = time.time() - t0

    per_g = {}
    for g in gs:
        tp = int(fa_[g].sum()); fn = int(len(fa_[g]) - tp)
        fp = int(fn_[g].sum()); tn = int(len(fn_[g]) - fp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_g[str(g)] = {"precision": round(prec, 4), "recall": round(rec, 4),
                         "f1": round(f1, 4),
                         "fpr": round(fp / (fp + tn) if fp + tn else 0.0, 4),
                         "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        print(f"g={g}: {per_g[str(g)]}", flush=True)

    result = {
        "method": "DeepLog (re-implementation v1.1, Du et al. CCS 2017)",
        "config_dir": os.path.basename(os.path.normpath(args.data)),
        "params": {**vars(args), "device_resolved": str(dev)},
        "vocab_size": V,
        "metrics_per_g": per_g,
        "train_seconds": round(train_secs, 1),
        "test_seconds": round(test_secs, 1),
        "torch": torch.__version__,
    }
    if len(gs) == 1:
        result["metrics"] = per_g[str(gs[0])]
        out = os.path.join(args.data, "deeplog_results.json")
    else:
        out = os.path.join(args.data, "deeplog_results_multi_g.json")
    json.dump(result, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
