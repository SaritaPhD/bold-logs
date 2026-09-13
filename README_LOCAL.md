# Deep-baseline re-runs on your machine

These runs are the strongest form of validation for the paper: DeepLog and
LogBERT executed on **exactly the same splits** as our lightweight models.
The cloud workspace has no deep-learning stack, so these run on your machine.

## One-time setup

```bash
cd local_runs
python3 -m venv venv && source venv/bin/activate
pip install torch numpy
```

## DeepLog (do this first — simple and fast)

Run all four configs; each writes `deeplog_results.json` into its data folder:

```bash
python deeplog_torch.py --data data/hdfs_chrono
python deeplog_torch.py --data data/hdfs_random
python deeplog_torch.py --data data/bgl_chrono
python deeplog_torch.py --data data/bgl_random
```

Expected runtime per config: roughly 10–40 min on a laptop CPU, a few
minutes on GPU. HDFS is the big one; BGL configs are quick.

## LogBERT (optional second pass, heavier)

```bash
git clone https://github.com/HelenGuohx/logbert
python convert_logbert.py --data data/hdfs_chrono --out logbert_input/hdfs_chrono
# ...same for the other three configs...
```

Then follow the LogBERT repo's README for training/prediction, pointing its
data options at the converted folders. Budget: ~30–60 min per config on GPU,
hours on CPU. If it fights you, skip it — DeepLog alone already anchors the
comparison, and the published LogBERT numbers cover the rest.

## Sending results back

Keep the `deeplog_results.json` files inside the `data/<config>/` folders and
put this whole `local_runs` folder inside your connected "Log Anomaly
Detection Paper" folder — I'll pick the JSONs up from there and merge them
into the paper's results tables with everything logged.

Two rules that keep us safe with reviewers:
1. If a run errors, send the traceback — don't tweak the script to make it
   pass unless we discuss it.
2. Never edit the result JSONs by hand; numbers only enter the paper from
   files a script wrote.
