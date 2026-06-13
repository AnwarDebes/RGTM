"""Find a config that learns NCI1 (one-hot, longer epochs)."""
import time

import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from rgtm.datasets.tudataset import (atom_symbols, build_tu_graphs, labels,
                                     load_tu, split_indices)

ds = load_tu("NCI1")
syms = atom_symbols(ds)
n = len(ds)
tr, te = split_indices(n, 42)
g_tr = build_tu_graphs(ds, tr, syms, one_hot=True)
g_te = build_tu_graphs(ds, te, syms, codebook=g_tr, one_hot=True)
Ytr, Yte = labels(ds, tr), labels(ds, te)
print(f"majority {max(np.mean(Yte), 1-np.mean(Yte)):.3f}", flush=True)

configs = [
    dict(number_of_clauses=2000, T=2000, s=5.0, depth=2, epochs=100),
    dict(number_of_clauses=2000, T=2000, s=10.0, depth=2, epochs=100),
    dict(number_of_clauses=4000, T=4000, s=8.0, depth=2, epochs=100),
    dict(number_of_clauses=2000, T=1000, s=3.0, depth=2, epochs=150),
]
for c in configs:
    ep = c.pop("epochs")
    m = MultiClassGraphTsetlinMachine(message_size=64, message_bits=2,
                                      max_included_literals=32,
                                      one_hot_encoding=True, **c)
    t0 = time.time()
    m.fit(g_tr, Ytr, epochs=ep)
    acc = float((m.predict(g_te) == Yte).mean())
    print(f"clauses={c['number_of_clauses']} T={c['T']} s={c['s']} ep={ep}: "
          f"acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
