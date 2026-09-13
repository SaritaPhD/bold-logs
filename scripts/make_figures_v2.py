"""Revision figures: f1 (six conditions, primary AE + OOV rule + deep), f2 (label efficiency v2),
f6 (Thunderbird calibration-era mismatch)."""
import sys, os, json, numpy as np
sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, GRAY, DARK, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86", "#3d3d3a", "#8e5bd6"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 8.5, "axes.edgecolor": "#d8d7d2", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#eceae5", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
FIGS = "figures"; os.makedirs(FIGS, exist_ok=True)
def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", name)
R = lambda f: json.load(open(os.path.join("results", f)))
def mean_f1(d, m): return float(np.mean([x["f1"] for x in d["models"][m].values()]))
conds = [("hdfs_chrono", "HDFS\nchrono"), ("hdfs_random", "HDFS\nrandom"), ("bgl_global_chrono", "BGL\nchrono"),
         ("bgl_global_random", "BGL\nrandom"), ("thunderbird_chrono", "Thunderbird\nchrono"), ("thunderbird_random", "Thunderbird\nrandom")]
full = {c: R(f"full_{c}.json") for c, _ in conds}
oov = R("oov_rule.json"); okey = {"hdfs_chrono": "hdfs_chrono", "hdfs_random": "hdfs_random", "bgl_global_chrono": "bgl_global_chrono",
        "bgl_global_random": "bgl_global_random", "thunderbird_chrono": "thunderbird_chrono", "thunderbird_random": "thunderbird_random"}
dl9 = {j["config_dir"]: j["metrics"]["f1"] for j in (json.load(open(f"results/user_runs/deeplog_{c}.json")) for c in ["hdfs_chrono", "hdfs_random", "bgl_chrono", "bgl_random"])}
dl3 = {}
for c in ["hdfs_chrono", "hdfs_random", "bgl_chrono", "bgl_random"]:
    p = f"splits/{c}/deeplog_results_multi_g.json"
    if os.path.exists(p):
        pg = json.load(open(p))["metrics_per_g"]; bg = max(pg, key=lambda g: pg[g]["f1"])
        if bg != "9": dl3[c] = pg[bg]["f1"]
lb = R("colab_runs_merged.json") if os.path.exists("results/colab_runs_merged.json") else {}
def lbf1(c):
    try: return lb[f"{c}/logbert"]["metrics"]["f1"]
    except Exception: return None
dmap = {"hdfs_chrono": "hdfs_chrono", "hdfs_random": "hdfs_random", "bgl_global_chrono": "bgl_chrono", "bgl_global_random": "bgl_random"}
series = {
    "MLP-AE (primary)": [mean_f1(full[c], "MLP-AE") for c, _ in conds],
    "PCA": [mean_f1(full[c], "PCA-SPE") if "PCA-SPE" in full[c]["models"] else R(f"hdfs_baselines_{c.split('_')[1]}.json")["results"][c.split('_')[1]]["models"]["PCA-SPE"]["f1"] for c, _ in conds],
    "OOV rule (k$\\geq$1)": [oov[okey[c]]["k1"]["f1"] for c, _ in conds],
    "DeepLog g=9 / best g": [dl9.get(dmap.get(c)) for c, _ in conds],
    "LogBERT-style": [lbf1(dmap.get(c)) for c, _ in conds],
}
fig, ax = plt.subplots(figsize=(7.0, 3.0)); x = np.arange(len(conds)); w = 0.16
for i, (name, vals) in enumerate(series.items()):
    color = [BLUE, AQUA, PURPLE, GRAY, DARK][i]
    v = [0 if a is None else a for a in vals]
    b = ax.bar(x + (i - 2) * w, v, w * 0.92, color=color, label=name, zorder=3)
    for j, (rect, a) in enumerate(zip(b, vals)):
        if a is None: continue
        lab = f"{a:.2f}".lstrip("0") or "0"
        if name.startswith("DeepLog") and dmap.get(conds[j][0]) in dl3: lab += f"/{dl3[dmap[conds[j][0]]]:.2f}".replace("/0.", "/.")
        ax.text(rect.get_x() + rect.get_width() / 2, a + 0.015, lab, ha="center", va="bottom", fontsize=5.6, color=MUTED, rotation=90 if name.startswith("DeepLog") else 0)
ax.set_xticks(x, [n for _, n in conds]); ax.set_ylim(0, 1.12); ax.set_ylabel("F1 at the benign-calibrated threshold")
ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=6.8); style(ax); save(fig, "f1_conditions.png")

# f2: label efficiency v2
le = R("bgl_label_efficiency_v2.json")["results"]; ks = ["10", "25", "50", "100", "250", "all"]
Ns = [10, 25, 50, 100, 250, 3296]
pr = [le[k]["agg"]["pr_auc"] for k in ks]; fv = [le[k]["agg"]["f1_valthr"] for k in ks]; fo = [le[k]["agg"]["f1_oracle"] for k in ks]; f1v1 = [le[k]["agg"]["f1_v1_benignq"] for k in ks]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.plot(Ns, pr, "-o", color=BLUE, ms=3.5, label="PR-AUC (ranking)")
ax.plot(Ns, fo, "-s", color=AQUA, ms=3.5, label="F1, oracle threshold (upper bound)")
ax.plot(Ns, fv, "-^", color=ORANGE, ms=3.5, label="F1, causal threshold (out-of-bag)")
ax.plot(Ns, f1v1, ":", color=GRAY, label="F1, benign-quantile threshold (v1)")
ax.axhline(0.846, color=MUTED, lw=0.8, ls="--"); ax.text(11, 0.86, "content-probe ceiling 0.846", fontsize=6.5, color=MUTED)
ax.set_xscale("log"); ax.set_xticks(Ns, ["10", "25", "50", "100", "250", "all\n(3,296)"]); ax.set_ylim(0, 1.0)
ax.set_xlabel("labeled anomalies from the train+calibration period"); ax.set_ylabel("score on the untouched test period")
ax.legend(fontsize=6.5, loc="lower right"); style(ax); save(fig, "f2_label_efficiency.png")

# f6: Thunderbird calibration-era mismatch: PCA score distributions, calibration vs test benign vs test anomalous
d = np.load("results/scores/thunderbird_chrono.npz"); y = d["y"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ax, det, title in [(axes[0], "PCA", "PCA (recovers on recalibration)"), (axes[1], "AE", "MLP-AE (does not recover)")]:
    sc, st = d[f"{det}_cal"], d[f"{det}_test"]; allv = np.concatenate([sc, st]); lo, hi = np.log10(max(allv.min(), 1e-6)), np.log10(allv.max())
    bins = np.logspace(lo, hi, 60)
    ax.hist(sc, bins=bins, color=GRAY, alpha=0.9, label="calibration benign (storm era)")
    ax.hist(st[y == 0], bins=bins, color=BLUE, alpha=0.55, label="test benign")
    ax.hist(st[y == 1], bins=bins, color=ORANGE, alpha=0.75, label="test anomalous")
    thr = float(np.quantile(sc, 0.995)); ax.axvline(thr, color=DARK, ls="--", lw=1.0)
    o = np.argsort(d["order_test"]); first = o[:2000]; fb = first[y[first] == 0]; thr2 = float(np.quantile(st[fb], 0.995)); ax.axvline(thr2, color=AQUA, ls="-.", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(title, fontsize=8); ax.set_xlabel(f"{det} score (log)"); style(ax)
axes[0].set_ylabel("windows (log)")
axes[0].legend(fontsize=6.2, loc="upper left")
axes[1].text(0.02, 0.97, "dashed: calibration-era threshold\ndash-dot: recalibrated on first 2,000 test units", transform=axes[1].transAxes, fontsize=6.2, va="top", color=MUTED)
save(fig, "f6_thunderbird_calibration.png")
