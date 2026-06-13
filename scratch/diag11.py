"""Does deep-supervised weight-tied recurrent message passing solve L=4 match?

Plain cair depth failed here (diag10: chance at depth up to 6). RecurrentGraphTM
adds (a) one shared message-TA reused across rounds and (b) true-label feedback
every round. One-hot 5-symbol vocab -> no poison. rounds=1 == cair depth-2.
"""
import numpy as np

from rgtm.recurrent_tm import RecurrentGraphTM
from rgtm.graphs_builder import build_graphs
from rgtm.datasets import long_range_match


def build(L, n, syms, codebook, seed):
    specs, Y, _ = long_range_match(n, L, seed=seed)
    g = build_graphs(specs, syms, codebook, None, True, True, len(syms), 1)
    return g, Y


def run(L, rounds, seed, supervise="all", clauses=200, T=200, s=5.0, mil=6,
        epochs=30, ntr=400, nte=200):
    _, _, syms = long_range_match(1, L, seed=0)
    gtr, Ytr = build(L, ntr, syms, None, seed)
    gte, Yte = build(L, nte, syms, gtr, seed + 1000)
    m = RecurrentGraphTM(
        number_of_clauses=clauses, T=T, s=s, rounds=rounds, supervise=supervise,
        max_included_literals=mil, one_hot_encoding=True)
    m.fit(gtr, Ytr, epochs=epochs)
    return float((m.predict(gte) == Yte).mean())


if __name__ == "__main__":
    plan = [
        ("L2 r1   all ", 2, 1, "all"),
        ("L4 r1   all ", 4, 1, "all"),
        ("L4 r4   all ", 4, 4, "all"),
        ("L4 r4   last", 4, 4, "last"),
    ]
    for (label, L, r, sup) in plan:
        accs = [run(L, r, sd, sup) for sd in (41, 42)]
        print(f"{label}: {[f'{a:.3f}' for a in accs]} "
              f"mean={np.mean(accs):.3f} max={np.max(accs):.3f}", flush=True)
