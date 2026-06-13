"""Find a config that reliably learns the BASE match case (L=2, K=1).

match(L=2) is single-pass learnable (distance 1, like equality but the
discriminant is the colour beside a constant QUERY/KEY marker). Sweep clause
count / T / s / max_included_literals over two seeds (sparse encoding). Once a
config clears ~0.9 here, recurrence on longer paths can be tested.
"""
import numpy as np

from rgtm.model import RecurrentGraphTsetlinMachine
from rgtm.datasets import long_range_match


def run(L, K, clauses, T, s, mil, hv=64, epochs=40, seed=42, ntr=800, nte=300):
    tr, trY, syms = long_range_match(ntr, L, seed=seed)
    te, teY, _ = long_range_match(nte, L, seed=seed + 1000)
    m = RecurrentGraphTsetlinMachine(
        number_of_clauses=clauses, T=T, s=s, depth=2, iterations=K,
        one_hot=False, max_included_literals=mil, hypervector_size=hv,
        message_size=256, message_bits=2)
    m.fit(tr, trY, syms, epochs=epochs)
    return float((m.predict(te) == teY).mean())


if __name__ == "__main__":
    configs = [
        dict(clauses=100, T=200, s=5.0, mil=4),
        dict(clauses=200, T=400, s=5.0, mil=4),
        dict(clauses=400, T=400, s=8.0, mil=8),
        dict(clauses=200, T=200, s=3.0, mil=6),
    ]
    for cfg in configs:
        accs = [run(2, 1, seed=sd, **cfg) for sd in (42, 43)]
        print(f"clauses={cfg['clauses']:4d} T={cfg['T']:4d} s={cfg['s']:.1f} "
              f"mil={cfg['mil']}: L2K1 accs={[f'{a:.3f}' for a in accs]} "
              f"mean={np.mean(accs):.3f}", flush=True)
