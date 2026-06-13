"""Isolate WHAT about the match task breaks learning (vs equality, which works).

All variants: 2 nodes, label = (discriminant symbols equal), L=2, K=1, one-hot
encoding (zero collisions), raw cair TM (no RGTM wrapper). Variants peel apart
the marker symbols and the dual edge types.
"""
import random

import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from rgtm.graphs_builder import GraphSpec, build_graphs

SYMS = {
    "equality": ["s0", "s1"],
    "match_min": ["col0", "col1"],
    "match_mark": ["QUERY", "KEY", "col0", "col1"],
    "match_full": ["QUERY", "KEY", "col0", "col1"],
}


def make(task, n, seed):
    rng = random.Random(seed)
    specs, Y = [], np.empty(n, dtype=np.uint32)
    for i in range(n):
        cq, ck = rng.randint(0, 1), rng.randint(0, 1)
        if task == "equality":
            specs.append(GraphSpec(2, [(0, 1, "e"), (1, 0, "e")], [[f"s{cq}"], [f"s{ck}"]]))
        elif task == "match_min":
            specs.append(GraphSpec(2, [(0, 1, "e"), (1, 0, "e")], [[f"col{cq}"], [f"col{ck}"]]))
        elif task == "match_mark":
            specs.append(GraphSpec(2, [(0, 1, "e"), (1, 0, "e")], [["QUERY", f"col{cq}"], ["KEY", f"col{ck}"]]))
        else:  # match_full
            specs.append(GraphSpec(2, [(0, 1, "next"), (1, 0, "prev")], [["QUERY", f"col{cq}"], ["KEY", f"col{ck}"]]))
        Y[i] = 1 if cq == ck else 0
    return specs, Y


def run(task, seed, epochs=40):
    syms = SYMS[task]
    sp_tr, Ytr = make(task, 800, seed)
    sp_te, Yte = make(task, 300, seed + 1000)
    g_tr = build_graphs(sp_tr, syms, None, None, True, True, len(syms), 1)
    g_te = build_graphs(sp_te, syms, g_tr, None, True, True, len(syms), 1)
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=100, T=100, s=5.0, depth=2, message_size=256,
        message_bits=2, max_included_literals=4, one_hot_encoding=True)
    for _ in range(epochs):
        tm.fit(g_tr, Ytr, epochs=1, incremental=True)
    return float((tm.predict(g_te) == Yte).mean())


if __name__ == "__main__":
    for task in ("equality", "match_min", "match_mark", "match_full"):
        accs = [run(task, sd) for sd in (42, 43)]
        print(f"{task:12s}: {[f'{a:.3f}' for a in accs]} mean={np.mean(accs):.3f}",
              flush=True)
