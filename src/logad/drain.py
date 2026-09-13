"""Drain log parser — fixed-depth prefix-tree template miner.

Own implementation of: He et al., "Drain: An Online Log Parsing Approach with
Fixed Depth Tree", ICWS 2017. No external dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class LogCluster:
    cluster_id: int
    template: list[str]
    size: int = 0


@dataclass
class _Node:
    children: dict = field(default_factory=dict)
    clusters: list = field(default_factory=list)


_DIGIT = re.compile(r"\d")

DEFAULT_MASKS = [
    (re.compile(r"blk_-?\d+"), "<BLK>"),
    (re.compile(r"(\d{1,3}\.){3}\d{1,3}(:\d+)?"), "<IP>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),
    (re.compile(r"(?<=[^A-Za-z0-9])(-?\+?\d+)(?=[^A-Za-z0-9]|$)"), "<NUM>"),
]


class Drain:
    def __init__(self, depth: int = 4, sim_th: float = 0.4,
                 max_children: int = 100, masks=None):
        # depth counts the token-prefix levels between the length node and leaf
        self.depth = max(1, depth - 2)
        self.sim_th = sim_th
        self.max_children = max_children
        self.masks = DEFAULT_MASKS if masks is None else masks
        self.root: dict[int, _Node] = {}
        self.clusters: dict[int, LogCluster] = {}
        self._next_id = 0

    # ---------------- public API ----------------
    def parse_line(self, content: str) -> int:
        """Return the cluster (template) id for one log message content."""
        for pat, repl in self.masks:
            content = pat.sub(repl, content)
        tokens = content.strip().split()
        if not tokens:
            tokens = ["<EMPTY>"]
        leaf = self._descend(tokens)
        best = self._match(leaf.clusters, tokens)
        if best is None:
            cl = LogCluster(self._next_id, list(tokens), size=1)
            self._next_id += 1
            self.clusters[cl.cluster_id] = cl
            leaf.clusters.append(cl)
            return cl.cluster_id
        self._merge(best, tokens)
        best.size += 1
        return best.cluster_id

    def template_of(self, cluster_id: int) -> str:
        return " ".join(self.clusters[cluster_id].template)

    # ---------------- internals ----------------
    def _descend(self, tokens: list[str]) -> _Node:
        length_key = len(tokens)
        node = self.root.setdefault(length_key, _Node())
        for i in range(min(self.depth, len(tokens))):
            tok = tokens[i]
            if _DIGIT.search(tok):
                tok = "<*>"
            if tok not in node.children:
                if "<*>" in node.children and len(node.children) >= self.max_children:
                    tok = "<*>"
                elif len(node.children) >= self.max_children:
                    tok = "<*>"
                    node.children.setdefault(tok, _Node())
                else:
                    node.children[tok] = _Node()
            node = node.children.setdefault(tok, _Node())
        return node

    def _match(self, clusters: list[LogCluster], tokens: list[str]):
        best, best_sim, best_params = None, -1.0, -1
        for cl in clusters:
            if len(cl.template) != len(tokens):
                continue
            same = params = 0
            for a, b in zip(cl.template, tokens):
                if a == "<*>":
                    params += 1
                elif a == b:
                    same += 1
            sim = (same + params) / len(tokens)
            if sim > best_sim or (sim == best_sim and params > best_params):
                best, best_sim, best_params = cl, sim, params
        if best is not None and best_sim >= self.sim_th:
            return best
        return None

    @staticmethod
    def _merge(cl: LogCluster, tokens: list[str]) -> None:
        cl.template = [a if a == b or a == "<*>" else "<*>"
                       for a, b in zip(cl.template, tokens)]
