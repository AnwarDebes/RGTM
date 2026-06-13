"""Premise test: does cair's native message-passing depth provide reach?

Raw cair GraphTM (no RGTM wrapper, no clause-symbol inflation -> no poison) on
the match task, sweeping depth. cair depth=D does D-1 message rounds = reaches
D-1 hops, so match(L) at distance L-1 should need D >= L. If depth solves it,
the right recurrence is weight-tied message passing, and RGTM's contribution is
parameter-efficient / adaptive depth rather than 'extends reach'.
"""
import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from rgtm.graphs_builder import build_graphs
from rgtm.datasets import long_range_match


def build(L, n, syms, codebook, seed, hv=64):
    specs, Y, _ = long_range_match(n, L, seed=seed)
    g = build_graphs(specs, syms, codebook, None, True, False, hv, 2)
    return g, Y


def run(L, depth, seed, clauses=200, T=200, s=5.0, mil=8, epochs=50,
        ntr=800, nte=300):
    _, _, syms = long_range_match(1, L, seed=0)  # just to get symbol list
    gtr, Ytr = build(L, ntr, syms, None, seed)
    gte, Yte = build(L, nte, syms, gtr, seed + 1000)
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=clauses, T=T, s=s, depth=depth,
        message_size=256, message_bits=2, max_included_literals=mil)
    for _ in range(epochs):
        tm.fit(gtr, Ytr, epochs=1, incremental=True)
    return float((tm.predict(gte) == Yte).mean())


if __name__ == "__main__":
    print("base: match L=2 depth=2")
    accs = [run(2, 2, sd) for sd in (41, 42, 43)]
    print(f"  L2 d2: {[f'{a:.3f}' for a in accs]} max={np.max(accs):.3f}", flush=True)
    for depth in (2, 4, 6, 8):
        accs = [run(4, depth, sd) for sd in (41, 42, 43)]
        print(f"  L4 d{depth}: {[f'{a:.3f}' for a in accs]} "
              f"mean={np.mean(accs):.3f} max={np.max(accs):.3f}", flush=True)
