"""Isolate the no-firing failure on the easy base case (L=2, K=1).

Factorial probe over the two suspected differences from the known-good cair
NoisyXOR config: one_hot encoding and max_included_literals. Reports final
train/test accuracy and iteration-1 firing density (the smoking gun: if density
is 0 the clauses are unsatisfiable; if ~1 they are empty/vacuous).
"""
import numpy as np

from rgtm.model import RecurrentGraphTsetlinMachine
from rgtm.datasets import long_range_match


def probe(one_hot, mil, epochs=20, clauses=200, T=100, s=5.0,
          hv=256, ntr=800, nte=300, seed=42):
    tr, trY, syms = long_range_match(ntr, 2, seed=seed)
    te, teY, _ = long_range_match(nte, 2, seed=seed + 1000)
    m = RecurrentGraphTsetlinMachine(
        number_of_clauses=clauses, T=T, s=s, depth=2, iterations=1,
        one_hot=one_hot, max_included_literals=mil, hypervector_size=hv,
        message_size=256, message_bits=2)
    # density at init (before any fit) and after fit
    m._ensure_tm(); m._prepare_vocab(syms)
    m.fit(tr, trY, syms, epochs=1)
    G1 = m._build(te, None)
    A_init, _ = m._tm.transform_nodewise(G1)
    m.fit(tr, trY, syms, epochs=epochs - 1)
    G1 = m._build(te, None)
    A_fin, _ = m._tm.transform_nodewise(G1)
    tr_acc = float((m.predict(tr) == trY).mean())
    te_acc = float((m.predict(te) == teY).mean())
    print(f"one_hot={str(one_hot):5s} max_inc_lit={str(mil):5s}  "
          f"train={tr_acc:.3f} test={te_acc:.3f}  "
          f"dens@1ep={A_init.mean():.3f} dens@fin={A_fin.mean():.3f}", flush=True)


if __name__ == "__main__":
    for oh in (True, False):
        for mil in (4, None):
            probe(oh, mil)
