"""Parse HDFS.log with Drain and persist session DataFrame for experiments."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
from logad.datasets import load_hdfs

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

t0 = time.time()
df, drain = load_hdfs(os.path.join(DATA, "HDFS.log"),
                      os.path.join(DATA, "anomaly_label.csv"))
dt = time.time() - t0

templates = {cid: drain.template_of(cid) for cid in drain.clusters}
joblib.dump({"df": df, "templates": templates},
            os.path.join(DATA, "hdfs_sessions.joblib"), compress=3)

stats = {
    "parse_seconds": round(dt, 1),
    "sessions": int(len(df)),
    "anomalous_sessions": int(df.label.sum()),
    "anomaly_rate": round(float(df.label.mean()), 5),
    "templates": len(templates),
    "total_events": int(df.seq.map(len).sum()),
}
print(json.dumps(stats, indent=2))
with open(os.path.join(DATA, "hdfs_prepare_stats.json"), "w") as fh:
    json.dump(stats, fh, indent=2)
