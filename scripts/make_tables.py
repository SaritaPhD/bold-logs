import sys, os, json, numpy as np
sys.path.insert(0, "src")
from logad.eval import bootstrap_ci
R = lambda f: json.load(open(os.path.join("results", f)))
def agg(runs):
    keys = ["precision", "recall", "f1", "fpr", "pr_auc"]
    vals = {k: [r[k] for r in runs.values()] for k in keys}
    out = {k: float(np.mean(v)) for k, v in vals.items()}; out["n"] = len(runs)
    if len(runs) >= 2: m, lo, hi = bootstrap_ci(vals["f1"]); out["ci"] = (lo, hi)
    return out
def f1cell(a, bold=False):
    s = f"{a['f1']:.3f}"
    if "ci" in a: s += f" [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]"
    return f"\\textbf{{{s}}}" if bold else s
NAMES = {"MLP-AE": "MLP-AE (primary)", "OCSVM": "OC-SVM", "PCA-SPE": "PCA", "MarkovNext": "Markov ($k{=}2$)", "IForest": "Isolation Forest"}
ORDER = ["MLP-AE", "PCA-SPE", "OCSVM", "MarkovNext", "IForest"]
oov = R("oov_rule.json"); lb = R("colab_runs_merged.json")
dl9 = {c: json.load(open(f"results/user_runs/deeplog_{c}.json"))["metrics"] for c in ["hdfs_chrono", "hdfs_random", "bgl_chrono", "bgl_random"]}
dl3 = {}
for c in ["hdfs_chrono", "hdfs_random", "bgl_chrono", "bgl_random"]:
    p = f"splits/{c}/deeplog_results_multi_g.json"
    if os.path.exists(p): dl3[c] = json.load(open(p))["metrics_per_g"]["3"]
def rows(cfg, mode, dlkey=None):
    d = R(f"full_{cfg}_{mode}.json")["models"]
    if cfg == "hdfs":  # PCA/Markov live in the baselines file
        b = R(f"hdfs_baselines_{mode}.json")["results"][mode]["models"]
        for k in ["PCA-SPE", "MarkovNext"]:
            if k not in d and k in b: d[k] = {"0": b[k]}
    out = []
    for m in ORDER:
        if m not in d: continue
        a = agg(d[m]); out.append(f"{NAMES[m]} & {mode} & {a['precision']:.3f} & {a['recall']:.3f} & {f1cell(a, m=='MLP-AE')} & {a['fpr']:.4f} & {a['pr_auc']:.3f}\\\\")
    o = oov[f"{cfg if cfg!='hdfs' else 'hdfs'}_{mode}" if cfg != "bgl_global" else f"bgl_global_{mode}"]
    k1 = o["k1"]; out.append(f"OOV rule ($k{{\\geq}}1$) & {mode} & {k1['precision']:.3f} & {k1['recall']:.3f} & {k1['f1']:.3f} & {k1['fpr']:.4f} & {o['pr_auc_count']:.3f}\\\\")
    if dlkey:
        g9 = dl9[dlkey]; out.append(f"DeepLog re-run ($g{{=}}9$) & {mode} & {g9['precision']:.3f} & {g9['recall']:.3f} & {g9['f1']:.3f} & {g9['fpr']:.4f} & ---\\\\")
        p = f"splits/{dlkey}/deeplog_results_multi_g.json"
        if os.path.exists(p):
            pg = json.load(open(p))["metrics_per_g"]; bg = max(pg, key=lambda g: pg[g]["f1"])
            if bg != "9":
                b = pg[bg]; out.append(f"DeepLog re-run (best sweep point, $g{{=}}{bg}$) & {mode} & {b['precision']:.3f} & {b['recall']:.3f} & {b['f1']:.3f} & {b['fpr']:.4f} & ---\\\\")
        l = lb[f"{dlkey}/logbert"]["metrics"]; out.append(f"LogBERT-style & {mode} & {l['precision']:.3f} & {l['recall']:.3f} & {l['f1']:.3f} & {l['fpr']:.4f} & {l['pr_auc']:.3f}\\\\")
    return out
print("%% HDFS"); print("\n".join(rows("hdfs", "chrono", "hdfs_chrono"))); print("\\midrule"); print("\n".join(rows("hdfs", "random", "hdfs_random")))
print("%% BGL"); print("\n".join(rows("bgl_global", "random", "bgl_random"))); print("\\midrule"); print("\n".join(rows("bgl_global", "chrono", "bgl_chrono")))
print("%% TBIRD"); print("\n".join(rows("thunderbird", "random"))); print("\\midrule"); print("\n".join(rows("thunderbird", "chrono")))
