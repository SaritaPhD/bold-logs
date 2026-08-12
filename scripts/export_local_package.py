"""Export the EXACT experimental splits for deep-baseline re-runs on the
user's machine. Output: local_runs/data/<config>/{train.txt.gz,
test_normal.txt.gz, test_abnormal.txt.gz, meta.json}

Format: one unit per line, space-separated integer template ids.
train.txt.gz contains BENIGN train+cal units (what benign-only methods see).
"""
import sys, os, gzip, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import pandas as pd

from logad.eval import split

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
ROOT = os.path.join(os.path.dirname(__file__), "..", "local_runs", "data")

CONFIGS = {
    "hdfs_chrono": ("hdfs_sessions.joblib", "chrono"),
    "hdfs_random": ("hdfs_sessions.joblib", "random"),
    "bgl_chrono": ("bgl_global.joblib", "chrono"),
    "bgl_random": ("bgl_global.joblib", "random"),
}


def dump(path, seqs):
    with gzip.open(path, "wt") as fh:
        for s in seqs:
            fh.write(" ".join(map(str, s)) + "\n")


for name, (fname, mode) in CONFIGS.items():
    df = joblib.load(os.path.join(DATA, fname))["df"]
    train, cal, test = split(df, mode=mode, seed=0)
    out = os.path.join(ROOT, name)
    os.makedirs(out, exist_ok=True)
    benign_fit = pd.concat([train, cal])
    dump(os.path.join(out, "train.txt.gz"), benign_fit.seq.tolist())
    dump(os.path.join(out, "test_normal.txt.gz"),
         test[test.label == 0].seq.tolist())
    dump(os.path.join(out, "test_abnormal.txt.gz"),
         test[test.label == 1].seq.tolist())
    meta = {
        "config": name, "split_mode": mode, "seed": 0,
        "split_def": "60/10/30 by unit order; train+cal benign-only",
        "n_train_benign": len(benign_fit), "n_test_normal": int((test.label == 0).sum()),
        "n_test_abnormal": int((test.label == 1).sum()),
        "source": fname,
    }
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(name, meta["n_train_benign"], meta["n_test_normal"], meta["n_test_abnormal"])
print("exported to", ROOT)
