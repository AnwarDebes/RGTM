"""Correctness check for RecurrentGraphTM on a stable task (equality).

Equality (label = two nodes carry the same symbol) REQUIRES message passing to
compare the nodes. So rounds=0 (node-literal only) must be ~chance, and rounds>=1
should learn on good seeds. This validates the custom CUDA loop and that the
message rounds + deep supervision actually do work.
"""
import numpy as np

from rgtm.recurrent_tm import RecurrentGraphTM
from rgtm.graphs_builder import GraphSpec, build_graphs

SYMS = ["s0", "s1"]


def make_eq(n, seed):
    rng = np.random.default_rng(seed)
    specs, Y = [], np.empty(n, dtype=np.uint32)
    for i in range(n):
        a, b = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        specs.append(GraphSpec(2, [(0, 1, "adj"), (1, 0, "adj")],
                               [[f"s{a}"], [f"s{b}"]]))
        Y[i] = 1 if a == b else 0
    return specs, Y


def run(rounds, seed, epochs=30):
    sp_tr, Ytr = make_eq(800, seed)
    sp_te, Yte = make_eq(300, seed + 1000)
    gtr = build_graphs(sp_tr, SYMS, None, None, True, True, 2, 1)
    gte = build_graphs(sp_te, SYMS, gtr, None, True, True, 2, 1)
    m = RecurrentGraphTM(number_of_clauses=200, T=200, s=5.0, rounds=rounds,
                         max_included_literals=6, one_hot_encoding=True)
    m.fit(gtr, Ytr, epochs=epochs)
    return float((m.predict(gte) == Yte).mean())


if __name__ == "__main__":
    for rounds in (0, 1, 2):
        accs = [run(rounds, sd) for sd in (41, 42, 43)]
        print(f"equality rounds={rounds}: {[f'{a:.3f}' for a in accs]} "
              f"mean={np.mean(accs):.3f} max={np.max(accs):.3f}", flush=True)
