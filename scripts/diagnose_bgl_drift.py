"""Diagnose WHY chronological BGL defeats benign-only detection.

Questions:
  1. Template drift: how many templates first appear after the training
     cutoff, and how much of the test stream do they cover — in benign vs
     anomalous units separately? (If benign test units are as OOV-heavy as
     anomalous ones, the OOV signal is confounded — the suspected mechanism.)
  2. Alert-template overlap: do alert-bearing templates ever occur in the
     benign training period?
  3. Signal existence: a SUPERVISED probe (labels used, diagnostic only,
     never a claim) on the same features/split. If the probe succeeds, the
     failure is calibration/adaptation, not representation. If it fails,
     count-vector features are insufficient under drift.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

from logad.features import CountVectors, seq_stats
from logad.eval import split, pr_auc

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

bundle = joblib.load(os.path.join(DATA, "bgl_global.joblib"))
df = bundle["df"].sort_values("order").reset_index(drop=True)

train, cal, test = split(df, mode="chrono", seed=0)
cut = train.order.max()

# --- 1. template first-appearance vs the cutoff ---
first_seen = {}
for r in df.itertuples():
    for t in r.seq:
        if t not in first_seen or r.order < first_seen[t]:
            first_seen[t] = r.order
post = {t for t, o in first_seen.items() if o > cut}

def oov_stats(sub, vocab_post):
    rates = []
    for s in sub.seq:
        n = len(s)
        rates.append(sum(1 for t in s if t in vocab_post) / n if n else 0.0)
    return float(np.mean(rates)), float(np.median(rates))

ben, anom = test[test.label == 0], test[test.label == 1]
ben_mean, ben_med = oov_stats(ben, post)
an_mean, an_med = oov_stats(anom, post)

# also: OOV relative to the *benign-train vocabulary* (what the model sees)
train_vocab = {t for s in train.seq for t in s}
def unseen_stats(sub):
    rates = []
    for s in sub.seq:
        n = len(s)
        rates.append(sum(1 for t in s if t not in train_vocab) / n if n else 0.0)
    return float(np.mean(rates)), float(np.median(rates))
ben_u_mean, ben_u_med = unseen_stats(ben)
an_u_mean, an_u_med = unseen_stats(anom)

# --- 2. alert-template overlap with benign training period ---
alert_units = df[df.label == 1]
alert_templates = {t for s in alert_units.seq for t in s}
alert_in_train_vocab = len(alert_templates & train_vocab)

# --- 3. supervised probe (diagnostic only) ---
fx = CountVectors().fit(train.seq.tolist())
def feats(sub):
    return np.hstack([fx.transform(sub.seq.tolist()), seq_stats(sub.seq.tolist())])
import pandas as pd
trc = pd.concat([train, cal])
# supervised probe needs labels in its training window: use ALL units (both
# classes) from the train+cal PERIOD — chronologically causal, labels cheat.
period = df[df.order <= cal.order.max()]
Xp, yp = feats(period), period.label.to_numpy()
Xt, yt = feats(test), test.label.to_numpy()
probe = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
probe.fit(Xp, yp)
ps = probe.predict_proba(Xt)[:, 1]
probe_auc = pr_auc(yt, ps)
yhat = (ps > 0.5).astype(int)
tp = int(((yhat == 1) & (yt == 1)).sum()); fp = int(((yhat == 1) & (yt == 0)).sum())
fn = int(((yhat == 0) & (yt == 1)).sum())
prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0
probe_f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

report = {
    "templates_total": len(first_seen),
    "templates_first_seen_after_train_cutoff": len(post),
    "test_event_share_from_post_cutoff_templates": {
        "benign_mean": round(ben_mean, 4), "benign_median": round(ben_med, 4),
        "anomalous_mean": round(an_mean, 4), "anomalous_median": round(an_med, 4)},
    "test_event_share_unseen_vs_benign_train_vocab": {
        "benign_mean": round(ben_u_mean, 4), "benign_median": round(ben_u_med, 4),
        "anomalous_mean": round(an_u_mean, 4), "anomalous_median": round(an_u_med, 4)},
    "alert_templates": len(alert_templates),
    "alert_templates_also_in_benign_train_vocab": alert_in_train_vocab,
    "supervised_probe": {"pr_auc": round(probe_auc, 4), "f1_at_0.5": round(probe_f1, 4),
                         "precision": round(prec, 4), "recall": round(rec, 4),
                         "note": "labels used causally for diagnosis only"},
}
print(json.dumps(report, indent=2))
with open(os.path.join(OUT, "bgl_drift_diagnosis.json"), "w") as fh:
    json.dump(report, fh, indent=2)
