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
