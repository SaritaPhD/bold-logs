#!/usr/bin/env python3
"""LogBERT re-run (protocol-matched re-implementation of Guo, Yuan & Wu,
IJCNN 2021) on the exported BOLD splits. GPU (CUDA), Apple MPS, or CPU.

Faithful core: a Transformer encoder trained with masked-key prediction (MLM)
on benign key sequences; a unit is scored by how badly masked positions are
predicted. Documented deviations, chosen to match the paper's leak-free
protocol (state these in the manuscript):
  * the center-loss auxiliary term is omitted (MLM-only objective);
  * detection uses score = miss-ratio (share of masked positions whose true
    key is outside the top-g predictions, OOV counting as a miss), with the
    alarm threshold set at the (1-alpha) benign-calibration quantile instead
    of LogBERT's validation-tuned (g, r) rule -- consistent with rule 3.
The benign file (train.txt.gz = benign train+cal in unit order) is split
6:1 by order into fit and calibration portions, mirroring the 60/10 design.

Usage:  python logbert_torch.py --data data/bgl_chrono
Writes: <data>/logbert_results.json  (never edit by hand; send tracebacks
on error instead of patching).
"""
import argparse, gzip, json, os, random, time

import numpy as np
import torch
import torch.nn as nn


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
    ap.add_argument("--g", type=int, default=9)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--mask_ratio", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max_train_units", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = pick_device(args.device)

    benign = read_seqs(os.path.join(args.data, "train.txt.gz"))
    test_n = read_seqs(os.path.join(args.data, "test_normal.txt.gz"))
    test_a = read_seqs(os.path.join(args.data, "test_abnormal.txt.gz"))
    k = (len(benign) * 6) // 7
    fit_seqs, cal_seqs = benign[:k], benign[k:]

    vocab = sorted({t for s in fit_seqs for t in s})
    remap = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)          # ids 0..V-1
    OOV, MASK, PAD = V, V + 1, V + 2
    n_tokens = V + 3
    print(f"fit={len(fit_seqs)} cal={len(cal_seqs)} vocab={V} device={dev}", flush=True)

    def encode(s):
        e = [remap.get(t, OOV) for t in s][: args.maxlen]
        return e

    class LogBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_tokens, args.d_model, padding_idx=PAD)
            self.pos = nn.Embedding(args.maxlen, args.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=args.d_model, nhead=args.nhead,
                dim_feedforward=args.d_model * 2, dropout=0.1,
                batch_first=True)
            self.enc = nn.TransformerEncoder(layer, args.layers)
            self.out = nn.Linear(args.d_model, n_tokens)

        def forward(self, x, pad_mask):
            pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            h = self.emb(x) + self.pos(pos)
            h = self.enc(h, src_key_padding_mask=pad_mask)
            return self.out(h)

    model = LogBert().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    def batchify(seqs, bs):
        for i in range(0, len(seqs), bs):
            chunk = [encode(s) for s in seqs[i:i + bs]]
            L = max(len(e) for e in chunk)
            x = torch.full((len(chunk), L), PAD, dtype=torch.long)
            for j, e in enumerate(chunk):
                x[j, :len(e)] = torch.tensor(e)
            yield x

    if len(fit_seqs) > args.max_train_units:
        idx = np.random.default_rng(args.seed).choice(
            len(fit_seqs), size=args.max_train_units, replace=False)
        fit_seqs = [fit_seqs[i] for i in idx]

    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); random.shuffle(fit_seqs); tot = cnt = 0
        for x in batchify(fit_seqs, args.batch):
            x = x.to(dev)
            pad_mask = x == PAD
            target = torch.full_like(x, -100)
            can = ~pad_mask
            rnd = torch.rand_like(x, dtype=torch.float)
            sel = (rnd < args.mask_ratio) & can
            # guarantee at least one masked position per sequence
            none = sel.sum(1) == 0
            if bool(none.any()):
                first = can.float().argmax(dim=1)
                sel[torch.arange(x.size(0), device=dev)[none], first[none]] = True
            target[sel] = x[sel]
            xin = x.clone(); xin[sel] = MASK
            logits = model(xin, pad_mask)
            loss = lossf(logits.reshape(-1, n_tokens), target.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); cnt += 1
        print(f"epoch {ep+1}/{args.epochs} loss={tot/max(cnt,1):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    train_secs = time.time() - t0

    @torch.no_grad()
    def scores(seqs):
        model.eval(); out = []
        for i in range(0, len(seqs), args.batch):
            chunk = [encode(s) for s in seqs[i:i + args.batch]]
            L = max(len(e) for e in chunk)
            x = torch.full((len(chunk), L), PAD, dtype=torch.long)
            for j, e in enumerate(chunk):
                x[j, :len(e)] = torch.tensor(e)
            x = x.to(dev)
            pad_mask = x == PAD
            miss = torch.zeros(x.size(0), device=dev)
            npos = (~pad_mask).sum(1).clamp(min=1).float()
            for phase in (0, 1):   # two complementary strided passes
                pos_idx = torch.arange(x.size(1), device=dev).unsqueeze(0)
                sel = (~pad_mask) & (pos_idx % 2 == phase)
                if not bool(sel.any()):
                    continue
                xin = x.clone(); xin[sel] = MASK
                logits = model(xin, pad_mask)
                topg = logits.topk(args.g, dim=2).indices
                true = x.unsqueeze(2)
                hit = (topg == true).any(dim=2)
                is_oov = x == OOV
                m = sel & (~hit | is_oov)
                miss += m.sum(1).float()
            out.append((miss / npos).cpu().numpy())
        return np.concatenate(out)

    t0 = time.time()
    s_cal = scores(cal_seqs)
    thr = float(np.quantile(s_cal, 1 - args.alpha))
    s_n, s_a = scores(test_n), scores(test_a)
    test_secs = time.time() - t0

    fp = int((s_n > thr).sum()); tn = len(s_n) - fp
    tp = int((s_a > thr).sum()); fn = len(s_a) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    ys = np.concatenate([np.zeros(len(s_n)), np.ones(len(s_a))])
    ss = np.concatenate([s_n, s_a])
    order = np.argsort(-ss)
    tps = np.cumsum(ys[order]); precs = tps / (np.arange(len(ys)) + 1)
    pr_auc = float(np.sum(precs * ys[order]) / max(ys.sum(), 1))

    result = {
        "method": "LogBERT (protocol-matched re-implementation; MLM-only, "
                  "benign-quantile threshold -- see header for deviations)",
        "config_dir": os.path.basename(os.path.normpath(args.data)),
        "params": {**vars(args), "device_resolved": str(dev)},
        "vocab_size": V,
        "metrics": {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(2*prec*rec/(prec+rec), 4) if prec+rec else 0.0,
                    "fpr": round(fp/(fp+tn), 4) if fp+tn else 0.0,
                    "pr_auc": round(pr_auc, 4),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "threshold": thr,
        "train_seconds": round(train_secs, 1),
        "test_seconds": round(test_secs, 1),
        "torch": torch.__version__,
    }
    out = os.path.join(args.data, "logbert_results.json")
    json.dump(result, open(out, "w"), indent=2)
    print(json.dumps(result["metrics"], indent=2)); print("saved", out)


if __name__ == "__main__":
    main()
