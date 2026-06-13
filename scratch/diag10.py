"""DECISIVE: does cair message-passing depth provide reach, under clean one-hot?

One-hot over the 5-symbol match vocab (every symbol appears in the data) has no
always-0 literals -> no Type II poison. Raw cair GraphTM, sweep depth. If
depth=D solves match at distance L-1 (needs D>=L), reach-via-depth works and the
right RGTM is weight-tied message passing (share message-TAs across rounds).
"""
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from rgtm.graphs_builder import build_graphs
from rgtm.datasets import long_range_match


def build(L, n, syms, codebook, seed):
    specs, Y, _ = long_range_match(n, L, seed=seed)
    g = build_graphs(specs, syms, codebook, None, True, True, len(syms), 1)
    return g, Y


def run(L, depth, seed, clauses=200, T=200, s=5.0, mil=6, epochs=50,
        ntr=800, nte=300):
    _, _, syms = long_range_match(1, L, seed=0)
    gtr, Ytr = build(L, ntr, syms, None, seed)
    gte, Yte = build(L, nte, syms, gtr, seed + 1000)
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=clauses, T=T, s=s, depth=depth, message_size=256,
        message_bits=2, max_included_literals=mil, one_hot_encoding=True)
    for _ in range(epochs):
        tm.fit(gtr, Ytr, epochs=1, incremental=True)
    return float((tm.predict(gte) == Yte).mean())


if __name__ == "__main__":
    plan = [(2, 2), (4, 2), (4, 4), (4, 6), (6, 2), (6, 6), (6, 8)]
    for (L, depth) in plan:
        accs = [run(L, depth, sd) for sd in (41, 42, 43)]
        print(f"L={L} depth={depth}: {[f'{a:.3f}' for a in accs]} "
              f"mean={np.mean(accs):.3f} max={np.max(accs):.3f}", flush=True)
