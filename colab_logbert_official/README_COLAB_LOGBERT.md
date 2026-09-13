# Official LogBERT on the BOLD splits

Runs the official LogBERT codebase (HelenGuohx/logbert), unmodified except for two
compatibility patches applied in the notebook's first cell (`torch.load` under
PyTorch >= 2.6, and ragged-array construction under NumPy >= 1.24, which the
variable-length HDFS sessions require), on the exact unit splits exported from this
repository's pipeline.

## Usage
1. Create a Google Drive folder `My Drive/colab_logbert_official/` and upload
   `logbert_inputs.zip` into it.
2. Open `LogBERT_official_on_BOLD_splits_v2_2026-09-12.ipynb` in Google Colab and
   select a GPU runtime (T4 is sufficient).
3. Run all cells and approve the Drive mount. Results (logs + JSONs) are written back
   into the Drive folder as they finish, so a disconnect loses nothing; on a re-run,
   configurations that already finished are skipped. BGL configurations take roughly
   80--95 minutes each on a T4; HDFS 1--2 hours each.
4. The last cell zips everything to `logbert_official_results.zip` in the same Drive
   folder. The archived outputs in `results/logbert_official/` were produced this way.

## Protocol notes (recorded automatically in each JSON)
- The official `predict` step chooses the anomaly-ratio threshold by sweeping it on the
  TEST set and keeping the best F1. That is LogBERT's published protocol; it is
  test-tuned (optimistic), and the paper labels these numbers as such.
- Sessions with fewer than 10 keys are dropped (`min_len = 10`). BGL windows are 100
  lines, so nothing is dropped there; on HDFS a number of short sessions are dropped
  and the JSON records how many.
- HDFS training uses a seeded 100,000-session subsample of the benign training
  sessions (the official code trains up to 200 epochs at batch size 32). Change
  `MAX_TRAIN_HDFS` in the notebook to use more.
