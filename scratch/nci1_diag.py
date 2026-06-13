"""Phase-1 NCI1 baseline search: does cair fit the TRAIN set?

New levers vs the failed sweep: low s (1.5-2.5), and decoupled encoding
(one-hot node features -> no always-0 poison; sparse compact messages -> message
space stays small). Reports TRAIN and TEST so we can tell "not learning at all"
(train at majority) from "overfitting" (train high, test low).
"""
import time

import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from rgtm.datasets.tudataset import (atom_symbols, build_tu_graphs, labels,
                                     load_tu)

ds = load_tu("NCI1")
syms = atom_symbols(ds)
n = len(ds)
rng = np.random.default_rng(42)
perm = rng.permutation(n)
tr, te = perm[:1200].tolist(), perm[1200:1700].tolist()
g_tr = build_tu_graphs(ds, tr, syms, one_hot=True)
g_te = build_tu_graphs(ds, te, syms, codebook=g_tr, one_hot=True)
Ytr, Yte = labels(ds, tr), labels(ds, te)
print(f"train_maj {max(np.mean(Ytr), 1-np.mean(Ytr)):.3f}  "
      f"test_maj {max(np.mean(Yte), 1-np.mean(Yte)):.3f}", flush=True)

configs = [
    dict(number_of_clauses=4000, T=4000, s=1.5, depth=2, epochs=40),
    dict(number_of_clauses=4000, T=4000, s=2.5, depth=2, epochs=40),
    dict(number_of_clauses=8000, T=8000, s=2.0, depth=2, epochs=40),
    dict(number_of_clauses=8000, T=8000, s=1.5, depth=3, epochs=40),
]
for c in configs:
    ep = c.pop("epochs")
    m = MultiClassGraphTsetlinMachine(message_size=512, message_bits=2,
                                      max_included_literals=32,
                                      one_hot_encoding=False, **c)
    t0 = time.time()
    m.fit(g_tr, Ytr, epochs=ep)
    tra = float((m.predict(g_tr) == Ytr).mean())
    tea = float((m.predict(g_te) == Yte).mean())
    print(f"cl={c['number_of_clauses']} T={c['T']} s={c['s']} d={c['depth']} "
          f"ep={ep}: TRAIN={tra:.3f} TEST={tea:.3f} ({time.time()-t0:.0f}s)",
          flush=True)
