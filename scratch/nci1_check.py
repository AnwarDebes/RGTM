"""Quick learnability check on NCI1 before the full comparison."""
import time

import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from rgtm.datasets.tudataset import (atom_symbols, build_tu_graphs, labels,
                                     load_tu, split_indices)

ds = load_tu("NCI1")
syms = atom_symbols(ds)
n = len(ds)
tr, te = split_indices(n, 42)
t0 = time.time()
g_tr = build_tu_graphs(ds, tr, syms)
g_te = build_tu_graphs(ds, te, syms, codebook=g_tr)
print(f"build {time.time()-t0:.0f}s  train={len(tr)} test={len(te)}", flush=True)
Ytr, Yte = labels(ds, tr), labels(ds, te)
maj = max(float(np.mean(Yte)), 1 - float(np.mean(Yte)))
print(f"majority-class test baseline: {maj:.3f}", flush=True)

common = dict(number_of_clauses=1000, T=1500, s=10.0, message_size=64,
              message_bits=2, max_included_literals=32)
for name, depth in [("single-pass(d1)", 1), ("cair-depth2", 2)]:
    m = MultiClassGraphTsetlinMachine(depth=depth, **common)
    t0 = time.time()
    m.fit(g_tr, Ytr, epochs=15)
    acc = float((m.predict(g_te) == Yte).mean())
    print(f"{name}: acc={acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
