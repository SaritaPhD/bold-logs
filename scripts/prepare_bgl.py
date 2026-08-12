"""Parse BGL.log with Drain and persist per-node windowed DataFrame."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
from logad.datasets import load_bgl

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

all_stats = {}
for mode in ["global", "node-time"]:
    t0 = time.time()
    df, drain = load_bgl(os.path.join(DATA, "BGL.log"), mode=mode, bucket_hours=3, min_unit=10)
    dt = time.time() - t0
    templates = {cid: drain.template_of(cid) for cid in drain.clusters}
    out = "bgl_global.joblib" if mode == "global" else "bgl_nodetime.joblib"
    joblib.dump({"df": df, "templates": templates},
                os.path.join(DATA, out), compress=3)
    all_stats[mode] = {
        "parse_seconds": round(dt, 1),
        "units": int(len(df)),
        "anomalous_units": int(df.label.sum()),
        "anomaly_rate": round(float(df.label.mean()), 5),
        "templates": len(templates),
        "components": int(df.component.nunique()),
        "total_events": int(df.seq.map(len).sum()),
    }
    print(mode, json.dumps(all_stats[mode], indent=2))

with open(os.path.join(DATA, "bgl_prepare_stats.json"), "w") as fh:
    json.dump(all_stats, fh, indent=2)
