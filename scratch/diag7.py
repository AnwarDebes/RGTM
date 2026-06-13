"""Does the RGTM wrapper learn the base case with diag5's winning config?

diag5 (raw TM, 4-symbol vocab, single edge, one_hot, clauses=100/T=100/s=5/mil=4)
hit 1.000 on seed 43. diag6 (wrapper, 305-symbol vocab from 300 clause symbols,
clauses=300/T=300/mil=8) failed on all seeds. Reproduce diag5's config through
the wrapper across 5 seeds to separate vocab-inflation from config/variance.
"""
import numpy as np

from rgtm.model import RecurrentGraphTsetlinMachine
from rgtm.datasets import long_range_match


def run(L, K, seed, clauses, T, s, mil, epochs=50, ntr=800, nte=300):
    tr, trY, syms = long_range_match(ntr, L, seed=seed)
    te, teY, _ = long_range_match(nte, L, seed=seed + 1000)
    m = RecurrentGraphTsetlinMachine(
        number_of_clauses=clauses, T=T, s=s, depth=2, iterations=K,
        one_hot=True, max_included_literals=mil)
    m.fit(tr, trY, syms, epochs=epochs)
    return float((m.predict(te) == teY).mean())


if __name__ == "__main__":
    seeds = (41, 42, 43, 44, 45)
    for (clauses, T, s, mil) in [(100, 100, 5.0, 4), (100, 100, 5.0, 8)]:
        accs = [run(2, 1, sd, clauses, T, s, mil) for sd in seeds]
        print(f"L2K1 clauses={clauses} T={T} s={s} mil={mil}: "
              f"{[f'{a:.3f}' for a in accs]} "
              f"mean={np.mean(accs):.3f} #solved(>0.9)={sum(a>0.9 for a in accs)}",
              flush=True)
