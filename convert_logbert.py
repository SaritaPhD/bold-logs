#!/usr/bin/env python3
"""Convert an exported config dir into the file layout the official LogBERT
repository expects (https://github.com/HelenGuohx/logbert).

Usage:
  python convert_logbert.py --data data/hdfs_chrono --out logbert_input/hdfs_chrono

Then follow the LogBERT repo's HDFS instructions, pointing its options at
the produced directory ('train', 'test_normal', 'test_abnormal' files of
space-separated key sequences, one unit per line).
"""
import argparse, gzip, os

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
for src, dst in [("train.txt.gz", "train"),
                 ("test_normal.txt.gz", "test_normal"),
                 ("test_abnormal.txt.gz", "test_abnormal")]:
    with gzip.open(os.path.join(args.data, src), "rt") as fi, \
         open(os.path.join(args.out, dst), "w") as fo:
        for line in fi:
            fo.write(line)
    print("wrote", os.path.join(args.out, dst))
