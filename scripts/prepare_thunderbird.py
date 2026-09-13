"""Parse Thunderbird_10M.log with Drain and persist windowed DataFrame (README recipe)."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import joblib
from logad.datasets import load_thunderbird
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
t0 = time.time()
df, dr = load_thunderbird(os.path.join(DATA, "Thunderbird_10M.log"))
dt = time.time() - t0
templates = {c: dr.template_of(c) for c in dr.clusters}
joblib.dump({"df": df, "templates": templates}, os.path.join(DATA, "thunderbird_global.joblib"), compress=3)
stats = {"parse_seconds": round(dt, 1), "units": int(len(df)), "anomalous_units": int(df.label.sum()),
         "anomaly_rate": round(float(df.label.mean()), 5), "templates": len(templates)}
print(json.dumps(stats, indent=2))
json.dump(stats, open(os.path.join(DATA, "thunderbird_prepare_stats.json"), "w"), indent=2)
