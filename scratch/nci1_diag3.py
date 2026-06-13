"""Phase-3 NCI1: richer node features (atom + degree). Last representational lever."""
import time

import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from rgtm.datasets.tudataset import (atom_symbols, build_tu_graphs, labels,
                                     load_tu)

ds = load_tu("NCI1")
n = len(ds)
syms = atom_symbols(ds) + [f"deg_{i}" for i in range(7)]
rng = np.random.default_rng(42)
perm = rng.permutation(n)
tr, te = perm[:1200].tolist(), perm[1200:1700].tolist()
g_tr = build_tu_graphs(ds, tr, syms, one_hot=True, add_degree=True)
g_te = build_tu_graphs(ds, te, syms, codebook=g_tr, one_hot=True, add_degree=True)
Ytr, Yte = labels(ds, tr), labels(ds, te)
print(f"train_maj {max(np.mean(Ytr), 1-np.mean(Ytr)):.3f} "
      f"test_maj {max(np.mean(Yte), 1-np.mean(Yte)):.3f}", flush=True)

configs = [
    ("d1 atom+deg", dict(number_of_clauses=2000, T=2000, s=1.5, depth=1, message_size=256, epochs=100)),
    ("d2 atom+deg s1.5", dict(number_of_clauses=2000, T=2000, s=1.5, depth=2, message_size=2048, epochs=100)),
    ("d2 atom+deg s1.2 ep200", dict(number_of_clauses=2000, T=2000, s=1.2, depth=2, message_size=2048, epochs=200)),
]
for name, c in configs:
    ep = c.pop("epochs")
    m = MultiClassGraphTsetlinMachine(message_bits=2, max_included_literals=32,
                                      one_hot_encoding=False, **c)
    t0 = time.time()
    m.fit(g_tr, Ytr, epochs=ep)
    tra = float((m.predict(g_tr) == Ytr).mean())
    tea = float((m.predict(g_te) == Yte).mean())
    print(f"{name}: TRAIN={tra:.3f} TEST={tea:.3f} ({time.time()-t0:.0f}s)", flush=True)
