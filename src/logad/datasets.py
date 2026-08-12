"""Dataset loaders: HDFS_v1 (session/block grouped) and BGL (windowed).

Both loaders emit a pandas DataFrame of units (sessions or windows) with:
    unit_id, seq (list of template ids), label (0 benign / 1 anomaly),
    order (chronological rank of the unit), component (node/block key).
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from .drain import Drain

_BLK = re.compile(r"(blk_-?\d+)")


def load_hdfs(log_path: str, label_csv: str, drain: Drain | None = None,
              max_lines: int | None = None):
    """Parse raw HDFS.log and group template-id sequences per block id."""
    drain = drain or Drain()
    seqs: dict[str, list[int]] = defaultdict(list)
    first_seen: dict[str, int] = {}
    with open(log_path, "r", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if max_lines is not None and i >= max_lines:
                break
            # HDFS_v1 line: <date> <time> <pid> <level> <component>: <content>
            parts = line.rstrip("\n").split(" ", 5)
            content = parts[5] if len(parts) == 6 else line.rstrip("\n")
            tid = drain.parse_line(content)
            for blk in set(_BLK.findall(line)):
                seqs[blk].append(tid)
                first_seen.setdefault(blk, i)
    labels = pd.read_csv(label_csv)
    lab = {r.BlockId: int(r.Label == "Anomaly") for r in labels.itertuples()}
    rows = [
        {"unit_id": blk, "seq": s, "label": lab.get(blk, 0),
         "order": first_seen[blk], "component": blk}
        for blk, s in seqs.items()
    ]
    df = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
    return df, drain


def _bgl_lines(log_path, drain, max_lines):
    """Yield (idx, epoch_ts, node, is_alert, template_id) per BGL line."""
    with open(log_path, "r", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if max_lines is not None and i >= max_lines:
                break
            parts = line.rstrip("\n").split(" ", 9)
            if len(parts) < 10:
                continue
            tag, ts, node, content = parts[0], parts[1], parts[3], parts[9]
            try:
                ts = int(ts)
            except ValueError:
                continue
            yield i, ts, node, int(tag != "-"), drain.parse_line(content)


def load_bgl(log_path: str, drain: Drain | None = None, mode: str = "global",
             window: int = 100, stride: int = 100, bucket_hours: int = 6,
             min_unit: int = 20, max_lines: int | None = None):
    """Parse BGL.log. Line is anomalous when the leading alert tag != '-';
    a unit is anomalous if any of its lines is.

    mode="global":    fixed `window`-line windows over the chronological log —
                      the convention used by published BGL results.
    mode="node-time": units are (node, wall-clock bucket of `bucket_hours`);
                      preserves cross-component structure for the correlator.
                      Units with fewer than `min_unit` lines are dropped
                      (reported by the caller, never silently in the paper).
    """
    drain = drain or Drain()
    rows = []
    if mode == "global":
        buf: list[tuple[int, int, int, str]] = []
        for i, ts, node, alert, tid in _bgl_lines(log_path, drain, max_lines):
            buf.append((i, alert, tid, node))
            if len(buf) == window:
                nodes = {n for _, _, _, n in buf}
                rows.append({
                    "unit_id": f"w{buf[0][0]}",
                    "seq": [t for _, _, t, _ in buf],
                    "label": int(any(a for _, a, _, _ in buf)),
                    "order": buf[0][0],
                    "component": f"{len(nodes)}nodes",
                })
                buf = buf[stride:] if stride < window else []
    elif mode == "node-time":
        units: dict[tuple[str, int], list[tuple[int, int, int]]] = defaultdict(list)
        for i, ts, node, alert, tid in _bgl_lines(log_path, drain, max_lines):
            units[(node, ts // (bucket_hours * 3600))].append((i, alert, tid))
        for (node, b), items in units.items():
            if len(items) < min_unit:
                continue
            rows.append({
                "unit_id": f"{node}@{b}",
                "seq": [t for _, _, t in items],
                "label": int(any(a for _, a, _ in items)),
                "order": items[0][0],
                "component": node,
                "bucket": b,
            })
    else:
        raise ValueError(mode)
    df = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
    return df, drain


def load_thunderbird(log_path: str, drain: Drain | None = None,
                     window: int = 100, stride: int = 100,
                     max_lines: int | None = None):
    """Parse Thunderbird (10M-line standard subset); fixed 100-line windows
    over the chronological stream, same convention as BGL. Line format:
    `tag epoch date node month day time location component: content`;
    a line is anomalous when the leading alert tag != '-'."""
    drain = drain or Drain()
    rows, buf = [], []
    with open(log_path, "r", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if max_lines is not None and i >= max_lines:
                break
            parts = line.rstrip("\n").split(" ", 9)
            if len(parts) < 10:
                continue
            tag, node, content = parts[0], parts[3], parts[9]
            tid = drain.parse_line(content)
            buf.append((i, int(tag != "-"), tid, node))
            if len(buf) == window:
                nodes = {n for _, _, _, n in buf}
                rows.append({
                    "unit_id": f"w{buf[0][0]}",
                    "seq": [t for _, _, t, _ in buf],
                    "label": int(any(a for _, a, _, _ in buf)),
                    "order": buf[0][0],
                    "component": f"{len(nodes)}nodes",
                })
                buf = buf[stride:] if stride < window else []
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
    return df, drain
