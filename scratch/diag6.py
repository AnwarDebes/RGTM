"""THE core contrast: does recurrence extend reach?

Single symmetric edge type (the fix). match(L=4) puts the KEY 3 hops from the
QUERY node. Single-pass depth-2 (K=1) reaches 1 hop -> must be ~chance.
Recurrence with K>=3 should reach and solve. Report 3 seeds (mean + max,
because TM training is high-variance at these sizes).
"""
import numpy as np

from rgtm.model import RecurrentGraphTsetlinMachine
from rgtm.datasets import long_range_match


def run(L, K, seed, clauses=300, T=300, s=5.0, mil=8, epochs=40,
        ntr=500, nte=300):
    tr, trY, syms = long_range_match(ntr, L, seed=seed)
    te, teY, _ = long_range_match(nte, L, seed=seed + 1000)
    m = RecurrentGraphTsetlinMachine(
        number_of_clauses=clauses, T=T, s=s, depth=2, iterations=K,
        one_hot=True, max_included_literals=mil)
    m.fit(tr, trY, syms, epochs=epochs)
    return float((m.predict(te) == teY).mean())


if __name__ == "__main__":
    for (L, K) in [(2, 1), (4, 1), (4, 4)]:
        accs = [run(L, K, sd) for sd in (42, 43)]
        print(f"L={L} K={K}: {[f'{a:.3f}' for a in accs]} "
              f"mean={np.mean(accs):.3f} max={np.max(accs):.3f}", flush=True)
