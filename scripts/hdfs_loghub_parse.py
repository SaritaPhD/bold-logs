"""Parser-sensitivity check: rebuild HDFS sessions from LogHub's reference parse
(Event_traces.csv, 29 event templates) with the SAME unit order (first-line rank)
as our Drain parse, then run the lightweight stack + OOV rule under the identical
protocol, and export DeepLog-format splits."""
import sys, os, json, gzip, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd, joblib
from logad.features import CountVectors, seq_stats
from logad.models import PCASPE, IF, OCSVM, MLPAE, MarkovNext
from logad.eval import split, threshold_from_benign, metrics, pr_auc
src = "/mnt/user-data/uploads/Log Anomaly Detection Paper/HDFS_v1/preprocessed/Event_traces.csv"
tr = pd.read_csv(src, usecols=["BlockId", "Label", "Features"])
ours = joblib.load("data/hdfs_sessions.joblib")["df"][["unit_id", "order", "label"]]
tr["seq"] = tr.Features.str.strip("[]").str.split(",").map(lambda l: [int(e[1:]) for e in l])
tr["label"] = (tr.Label == "Fail").astype(int)
df = tr.merge(ours, left_on="BlockId", right_on="unit_id", suffixes=("", "_ours"))
print("blocks", len(tr), "matched", len(df), "label agreement", float((df.label == df.label_ours).mean()))
df = df[["unit_id", "seq", "label", "order"]].sort_values("order").reset_index(drop=True)
templates = sorted({t for s in df.seq for t in s}); print("event types", len(templates))
joblib.dump({"df": df}, "data/hdfs_loghub.joblib", compress=3)
out = {}
for mode in ["chrono", "random"]:
    train, cal, test = split(df, mode=mode, seed=0)
    fx = CountVectors().fit(train.seq.tolist()); V = len(fx.vocab)
    F = lambda d: np.hstack([fx.transform(d.seq.tolist()), seq_stats(d.seq.tolist())])
    Xtr, Xca, Xte = F(train), F(cal), F(test); y = test.label.to_numpy()
    res = {"vocab_train": V, "test": len(test), "test_anom": int(y.sum())}
    def ev(name, sc, st):
        m = metrics(y, st, threshold_from_benign(sc, 0.005)); m["pr_auc"] = pr_auc(y, st); res[name] = m
        print(f"loghub/{mode} {name:<8} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} FPR={m['fpr']:.4f} PRAUC={m['pr_auc']:.3f}", flush=True)
    m = PCASPE().fit(Xtr); ev("PCA", m.score(Xca), m.score(Xte))
    for s in [0, 1, 2]:
        m = MLPAE(seed=s, max_iter=60).fit(Xtr); ev(f"AE_s{s}", m.score(Xca), m.score(Xte))
    idx = np.random.default_rng(0).choice(len(Xtr), 10000, replace=False); m = OCSVM().fit(Xtr[idx]); ev("OCSVM", m.score(Xca), m.score(Xte))
    m = IF(seed=0).fit(Xtr); ev("IF", m.score(Xca), m.score(Xte))
    mk = MarkovNext(k=2).fit_seqs(train.seq.tolist()); ev("Markov", mk.score_seqs(cal.seq.tolist()), mk.score_seqs(test.seq.tolist()))
    vocab = set(fx.vocab); cnt = np.array([sum(t not in vocab for t in s) for s in test.seq], float)
    m = metrics(y, cnt, 0.5); m["pr_auc"] = pr_auc(y, cnt); res["OOVrule"] = m; print(f"loghub/{mode} OOVrule  F1={m['f1']:.3f} R={m['recall']:.3f} FPR={m['fpr']:.4f}")
    d = f"splits/hdfs_loghub_{mode}"; os.makedirs(d, exist_ok=True)
    for name, seqs in [("train", pd.concat([train, cal]).seq), ("test_normal", test[test.label == 0].seq), ("test_abnormal", test[test.label == 1].seq)]:
        with gzip.open(f"{d}/{name}.txt.gz", "wt") as fh:
            for s in seqs: fh.write(" ".join(map(str, s)) + "\n")
    out[mode] = res
json.dump(out, open("results/hdfs_loghub_parse.json", "w"), indent=2)
