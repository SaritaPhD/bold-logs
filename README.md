# BOLD — Benign-Only Lightweight Detection for Log Anomaly Detection

Artifact for the paper **"How Far Can Lightweight, Benign-Only Log Anomaly
Detection Go?"** (Sarita Nandal, Indian Institute of Technology Roorkee).

Every number in the paper is produced by code in this repository. Per-seed
result JSONs — including the author-hardware DeepLog runs and the Colab-T4
LogBERT and frozen-embedding runs — are archived unmodified under `results/`.

## Contents

| Path | What it is |
|---|---|
| `src/logad/` | Drain parser, dataset loaders, features, detectors, correlator, leak-free evaluation protocol |
| `scripts/` | Every experiment in the paper, one script per study (see mapping below) |
| `splits/` | The exact exported unit splits used for the DeepLog/LogBERT re-runs |
| `colab/` | GPU experiment scripts + notebook (LogBERT re-implementation, frozen-embedding seventh design) |
| `results/` | Per-seed result JSONs for every table and figure, plus archived re-run outputs |
| `figures/` | Generated paper figures |
| `deeplog_torch.py`, `deeplog_torch_v1_1.py` | Protocol-matched DeepLog re-implementation (v1.1 adds MPS + multi-g sweep) |
| `convert_logbert.py`, `README_LOCAL.md` | Split-format converter and instructions for the author-hardware re-runs |
| `DATASET_CHECKSUMS.md5` | MD5 checksums of the exact dataset files used |

## Datasets (not redistributed here)

All three are the standard public LogHub distributions
(https://github.com/logpai/loghub, Zenodo record 8196385):

- **HDFS_v1** (`HDFS.log` + `preprocessed/anomaly_label.csv`)
- **BGL** (`BGL.log`)
- **Thunderbird** — the standard first-10,000,000-line subset:
  `tar -xzOf Thunderbird.tar.gz Thunderbird.log | head -n 10000000 > Thunderbird_10M.log`

Verify your copies against `DATASET_CHECKSUMS.md5` (checksums are over the
gzip-compressed files as listed).

## Reproduction

```bash
pip install -r requirements.txt          # numpy, pandas, scikit-learn, scipy, matplotlib
# place decompressed logs under data/ as HDFS.log, anomaly_label.csv, BGL.log, Thunderbird_10M.log

python scripts/prepare_hdfs.py           # parse + persist sessions
python scripts/prepare_bgl.py            # parse + persist BGL views
python - <<'P'                            # parse Thunderbird
import sys; sys.path.insert(0,'src'); import joblib
from logad.datasets import load_thunderbird
df, dr = load_thunderbird('data/Thunderbird_10M.log')
joblib.dump({'df': df, 'templates': {c: dr.template_of(c) for c in dr.clusters}},
            'data/thunderbird_global.joblib', compress=3)
P

# Tables 3, 4, and the Thunderbird table (10 seeds; resume-aware):
python scripts/run_full.py --config hdfs        --split chrono --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/run_full.py --config hdfs        --split random --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/run_full.py --config bgl_global  --split chrono --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/run_full.py --config bgl_global  --split random --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/run_full.py --config thunderbird --split chrono --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/run_full.py --config thunderbird --split random --seeds 0 1 2 3 4 5 6 7 8 9

python scripts/run_full.py --config bgl_nodetime --split chrono --correlator   # coalition ablation
python scripts/run_full.py --config bgl_nodetime --split random --correlator
python scripts/diagnose_bgl_drift.py            # drift diagnosis table
python scripts/run_content_features.py --config bgl_global --split chrono      # designs 1-2
python scripts/prequential_bgl.py               # designs 5 (identity, rolling)
python scripts/prequential_content.py           # design 6 (content, rolling)
python scripts/label_efficiency_bgl.py          # label-efficiency table
python scripts/cut_position_sweep.py            # cut-position sweep table
python scripts/make_figures.py                  # regenerate all figures

# Deep re-runs (author hardware / Colab; scripts + exact splits provided):
python deeplog_torch.py --data splits/hdfs_chrono          # etc. for all four configs
# colab/BOLD_colab.ipynb runs the LogBERT re-implementation and the
# frozen-embedding seventh design end to end on a free T4.
```

Environment used in the paper: Python 3.11.15, scikit-learn 1.8.0 (CPU
container); PyTorch 2.8.0 (Apple-silicon CPU) and 2.11.0+cu128 (Colab T4)
for the deep re-runs, as recorded inside each result JSON.

## Integrity notes

- Result JSONs are written only by the scripts; none were edited by hand.
- `results/user_runs/` and `results/colab_runs/` are archived exactly as
  produced on the author's hardware, with device and library versions inside.

## License

MIT (see LICENSE).

## Revision materials (manuscript v4)

Added in v3 of this artifact, together with the archived official-LogBERT runs below.
Everything was produced on a 2-vCPU container, Python 3.12, scikit-learn 1.8.0,
PyTorch 2.13 (CPU), from the LogHub files whose MD5s match `DATASET_CHECKSUMS.md5`;
run logs are under `logs/`.

| Script | Produces | Used in the paper for |
|---|---|---|
| `scripts/prepare_thunderbird.py` | `data/thunderbird_global.joblib` | Thunderbird parse as a script |
| `scripts/oov_rule.py` | `results/oov_rule.json` | unseen-template rule rows in Tables 3, 4, 11; Fig. 1 |
| `scripts/dump_scores.py` | `results/scores/<config>_<split>.npz` (seed-0 calibration/test scores of every detector, plus the MLP-AE-without-OOV ablation) | inputs for the two scripts below |
| `scripts/analyze_scores.py` | `results/score_analysis.json` | alpha sweep (Table 13), RQ2 ablation, Thunderbird recalibration counterfactual (Table 12) |
| `scripts/label_efficiency_v2.py` | `results/bgl_label_efficiency_v2.json` | Table 9 / Fig. 2 (causal out-of-bag threshold, oracle F1, benign-quantile reference) |
| `scripts/hdfs_loghub_parse.py` | `results/hdfs_loghub_parse.json`, `splits/hdfs_loghub_{chrono,random}/` | Table 14 (parser sensitivity), DeepLog-on-reference-parse row of Table 5 |
| `deeplog_torch_v1_1.py --g 1 3 5 7 9` | `splits/<config>/deeplog_results_multi_g.json` | DeepLog sweep in Tables 3--5 |
| `scripts/run_full.py --config bgl_global --split chrono --seeds 1..9` | `results/full_bgl_global_chrono.json` (10 seeds) | Table 4 |
| `scripts/make_tables.py` | LaTeX rows for Tables 3, 4, 11 | --- |
| `scripts/make_figures_v2.py` | `figures/f1_conditions.png`, `f2_label_efficiency.png`, `f6_thunderbird_calibration.png` | Figs. 1, 2, 6 |

`hdfs_loghub_parse.py` expects LogHub's `Event_traces.csv` (HDFS_v1/preprocessed);
set the path at the top of the script. The per-unit score dumps in `results/scores/`
are what the alpha sweep and the recalibration counterfactual read, so those analyses
train nothing new; result JSONs are written by the scripts only.

## Official LogBERT runs

`results/logbert_official/` archives runs of the official LogBERT codebase
(HelenGuohx/logbert, unmodified, default hyperparameters, its own test-tuned
threshold sweep) on the exported BGL splits, with full console logs.
`colab_logbert_official/` contains the Colab notebook and inputs to reproduce them
(see its README); the notebook also patches the official code's ragged-array
construction for NumPy >= 1.24, which the variable-length HDFS sessions require.
