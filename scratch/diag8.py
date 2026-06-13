"""Test sparse encoding (cair-native) as the fix for the one-hot poison-literal death.

With sparse hypervectors, symbols share literal positions, so there are no
dedicated always-0 positive literals for Type II feedback to exploit. Test the
base case (match L=2 K=1) across a few hypervector sizes, 3 seeds each.
"""
import numpy as np

from rgtm.model import RecurrentGraphTsetlinMachine
from rgtm.datasets import long_range_match


def run(L, K, seed, hv, clauses=100, T=100, s=5.0, mil=4, epochs=50,
        ntr=800, nte=300):
    tr, trY, syms = long_range_match(ntr, L, seed=seed)
    te, teY, _ = long_range_match(nte, L, seed=seed + 1000)
    m = RecurrentGraphTsetlinMachine(
        number_of_clauses=clauses, T=T, s=s, depth=2, iterations=K,
        one_hot=False, hypervector_size=hv, hypervector_bits=2,
        max_included_literals=mil, message_size=256, message_bits=2)
    m.fit(tr, trY, syms, epochs=epochs)
    return float((m.predict(te) == teY).mean())


if __name__ == "__main__":
    for hv in (64, 128, 256):
        accs = [run(2, 1, sd, hv) for sd in (41, 42, 43)]
        print(f"sparse hv={hv}: L2K1 {[f'{a:.3f}' for a in accs]} "
              f"mean={np.mean(accs):.3f} #solved={sum(a > 0.9 for a in accs)}",
              flush=True)
