"""Generate the manuscript's five figures from result files / bundles.
Palette: validated categorical slots (blue #2a78d6, orange #eb6834,
aqua #1baf7a) + neutral gray for the external baseline; white surface.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86"
INK, MUTED = "#0b0b0b", "#52514e"
ROOT = os.path.join(os.path.dirname(__file__), "..")
FIGS = "/mnt/user-data/working/claude-docs/figs"
os.makedirs(FIGS, exist_ok=True)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 8.5,
    "axes.edgecolor": "#d8d7d2", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#eceae5", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def save(fig, name):
    for p in (os.path.join(FIGS, name), os.path.join(ROOT, "figures", name)):
        fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)

R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))

# ---------------- F1: four-condition comparison ----------------
def mean_f1(d, m):
    return float(np.mean([x["f1"] for x in d["models"][m].values()])) if m in d["models"] else None
full = {c: R(f"full_{c}.json") for c in
        ["hdfs_chrono", "hdfs_random", "bgl_global_chrono", "bgl_global_random"]}
old = {"chrono": R("hdfs_baselines_chrono.json"), "random": R("hdfs_baselines_random.json")}
dl = {j["config_dir"]: j["metrics"]["f1"]
      for j in (json.load(open(os.path.join(ROOT, "results", "user_runs", f"deeplog_{c}.json")))
                for c in ["hdfs_chrono", "hdfs_random", "bgl_chrono", "bgl_random"])}
conds = ["HDFS\nchrono", "HDFS\nrandom", "BGL\nchrono", "BGL\nrandom"]
series = {
    "MLP-AE (ours)": [mean_f1(full["hdfs_chrono"], "MLP-AE"), mean_f1(full["hdfs_random"], "MLP-AE"),
                      mean_f1(full["bgl_global_chrono"], "MLP-AE"), mean_f1(full["bgl_global_random"], "MLP-AE")],
    "OC-SVM (ours)": [mean_f1(full["hdfs_chrono"], "OCSVM"), mean_f1(full["hdfs_random"], "OCSVM"),
                      mean_f1(full["bgl_global_chrono"], "OCSVM"), mean_f1(full["bgl_global_random"], "OCSVM")],
    "PCA (ours)": [old["chrono"]["results"]["chrono"]["models"]["PCA-SPE"]["f1"],
                   old["random"]["results"]["random"]["models"]["PCA-SPE"]["f1"],
                   mean_f1(full["bgl_global_chrono"], "PCA-SPE"), mean_f1(full["bgl_global_random"], "PCA-SPE")],
    "DeepLog (re-run)": [dl["hdfs_chrono"], dl["hdfs_random"], dl["bgl_chrono"], dl["bgl_random"]],
}
fig, ax = plt.subplots(figsize=(6.4, 2.9))
x = np.arange(4); w = 0.19
for i, (name, vals) in enumerate(series.items()):
    color = [BLUE, ORANGE, AQUA, GRAY][i]
    b = ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=color, label=name, zorder=3)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}".lstrip("0") or "0",
                ha="center", va="bottom", fontsize=6.6, color=MUTED)
ax.set_xticks(x, conds); ax.set_ylim(0, 1.09)
ax.set_ylabel("F1 (session/window level)")
ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=7.4)
style(ax)
save(fig, "f1_conditions.png")

# ---------------- F2: label-efficiency ----------------
le = R("bgl_label_efficiency.json")["results"]
Ns, pr, f1v = [], [], []
for k in ["10", "25", "50", "100", "250", "all"]:
    if k in le:
        Ns.append(3296 if k == "all" else int(k))
        pr.append(le[k]["agg"]["pr_auc"]); f1v.append(le[k]["agg"]["f1"])
fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.plot(Ns, pr, "-o", color=BLUE, ms=4.5, lw=1.8, label="PR-AUC", zorder=3)
ax.plot(Ns, f1v, "-s", color=ORANGE, ms=4.5, lw=1.8, label="F1 at benign-quantile threshold", zorder=3)
ax.axhline(0.846, color=AQUA, lw=1.2, ls="--", zorder=2)
ax.text(Ns[0], 0.858, "content-probe ceiling (0.846)", fontsize=7, color=AQUA)
ax.axhline(0.108, color=GRAY, lw=1, ls=":", zorder=2)
ax.text(Ns[0], 0.06, "chance (prevalence 0.108)", fontsize=7, color=GRAY)
ax.set_xscale("log"); ax.set_xticks(Ns, [str(n) for n in Ns[:-1]] + ["3296\n(all)"])
ax.set_xlabel("labeled anomalous windows (period-causal)"); ax.set_ylabel("score on test period")
ax.set_ylim(0, 1); ax.legend(fontsize=7.4, loc="center right")
style(ax)
save(fig, "f2_label_efficiency.png")

# ---------------- F3: drift timeline ----------------
bundle = joblib.load(os.path.join(ROOT, "data", "bgl_global.joblib"))
df = bundle["df"].sort_values("order").reset_index(drop=True)
n = len(df)
cut_tr, cut_ca = int(n * 0.6), int(n * 0.7)
train_vocab = {t for s in df.seq.iloc[:cut_tr] for t in s}
seen, cum = set(), []
for s in df.seq:
    seen.update(s); cum.append(len(seen))
CH = 1000
xs, ben_sh, an_sh = [], [], []
for st in range(0, n, CH):
    ch = df.iloc[st:st + CH]
    xs.append(st + len(ch) / 2)
    for lab, out in [(0, ben_sh), (1, an_sh)]:
        sub = ch[ch.label == lab]
        ev = [t for s in sub.seq for t in s]
        out.append(np.mean([t not in train_vocab for t in ev]) if ev else np.nan)
fig, axes = plt.subplots(2, 1, figsize=(6.2, 3.8), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
a = axes[0]
a.plot(range(n), cum, color=BLUE, lw=1.8, zorder=3)
a.set_ylabel("distinct templates\n(cumulative)")
b = axes[1]
b.plot(xs, ben_sh, color=BLUE, lw=1.6, label="benign windows", zorder=3)
b.plot(xs, an_sh, color=ORANGE, lw=1.6, label="anomalous windows", zorder=3)
b.set_ylabel("event share unseen\nvs. training vocab")
b.set_xlabel("window index (chronological)")
b.set_ylim(0, 1.02); b.legend(fontsize=7.4, loc="upper left")
for ax_ in axes:
    style(ax_)
    ax_.axvline(cut_tr, color=GRAY, lw=1, ls="--", zorder=2)
    ax_.axvline(cut_ca, color=GRAY, lw=1, ls=":", zorder=2)
a.text(cut_tr, a.get_ylim()[1] * 0.94, " train end", fontsize=7, color=MUTED)
a.text(cut_ca, a.get_ylim()[1] * 0.78, " cal end", fontsize=7, color=MUTED)
save(fig, "f3_drift_timeline.png")

# ---------------- F4: score distributions ----------------
from logad.features import CountVectors, seq_stats
from logad.models import MLPAE
from logad.eval import split, threshold_from_benign
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=False)
for ax_, mode in zip(axes, ["random", "chrono"]):
    tr, ca, te = split(df, mode=mode, seed=0)
    fx = CountVectors().fit(tr.seq.tolist())
    F = lambda sub: np.hstack([fx.transform(sub.seq.tolist()), seq_stats(sub.seq.tolist())])
    m = MLPAE(seed=0, max_iter=60).fit(F(tr))
    thr = threshold_from_benign(m.score(F(ca)), alpha=0.005)
    s = m.score(F(te)); y = te.label.to_numpy()
    bins = np.logspace(np.log10(max(s.min(), 1e-4)), np.log10(s.max() + 1e-9), 45)
    ax_.hist(s[y == 0], bins=bins, color=GRAY, alpha=0.75, label="benign", zorder=3)
    ax_.hist(s[y == 1], bins=bins, color=ORANGE, alpha=0.75, label="anomalous", zorder=3)
    ax_.axvline(thr, color=BLUE, lw=1.4, ls="--", zorder=4)
    ax_.set_xscale("log"); ax_.set_yscale("log")
    ax_.set_title(f"BGL {mode} split", fontsize=8.5)
    ax_.set_xlabel("MLP-AE reconstruction score (log)")
    style(ax_)
axes[0].set_ylabel("windows (log)")
axes[0].legend(fontsize=7.4)
axes[1].text(0.03, 0.9, "threshold exceeds\nnearly all mass", transform=axes[1].transAxes,
             fontsize=7, color=BLUE)
save(fig, "f4_score_distributions.png")

# ---------------- F5: probe importance words ----------------
import hashlib, re
from sklearn.ensemble import RandomForestClassifier
from logad.features import TemplateContentVectors
from logad.eval import split as _split
templates = bundle["templates"]
tr, ca, te = _split(df, mode="chrono", seed=0)
fx_ct = TemplateContentVectors(templates, dim=128)
period = df[df.order <= ca.order.max()]
F = lambda sub: np.hstack([fx_ct.transform(sub.seq.tolist()), seq_stats(sub.seq.tolist())])
probe = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
probe.fit(F(period), period.label.to_numpy())
imp = probe.feature_importances_
word_re = re.compile(r"[a-z0-9]+")
dim_words = {d: [] for d in range(128)}
seen_w = set()
for tpl in templates.values():
    for tok in tpl.split():
        if tok in fx_ct.MASKS: continue
        for w in word_re.findall(tok.lower()):
            if w in seen_w: continue
            seen_w.add(w)
            h = hashlib.md5(w.encode()).digest()
            dim_words[int.from_bytes(h[:4], "little") % 128].append(w)
top = np.argsort(imp)[::-1][:10]
labels, vals = [], []
for i in top:
    if i < 128:
        ws = [w for w in sorted(dim_words[i], key=len) if not w.isdigit()][:3]
        labels.append(", ".join(ws))
    else:
        labels.append(["length", "distinct", "max-run"][i - 128] + " (seq stat)")
    vals.append(imp[i])
fig, ax = plt.subplots(figsize=(4.8, 2.9))
ax.barh(range(len(vals))[::-1], vals, color=BLUE, height=0.62, zorder=3)
ax.set_yticks(range(len(vals))[::-1], labels, fontsize=7.6)
ax.set_xlabel("supervised-probe feature importance")
style(ax)
save(fig, "f5_importance_words.png")
print("all figures done")
