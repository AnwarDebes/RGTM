"""Evidence-gathering ladder for the RGTM long-range-match failure.

Rungs (simplest -> hardest):
  1. L=2 K=1  : single-pass-learnable (equality at distance 1). MUST learn,
                else the wrapper/vocab/hyperparameters are broken.
  2. L=2 K=2  : recurrence on an easy task. Should not hurt.
  3. L=4 K=1  : single-pass baseline. EXPECTED to be at chance.
  4. L=4 K=3  : the target. K>=L-1 should solve if recurrence works.
Plus instrumentation: firing density per iteration (is the re-feed non-trivial,
or are clauses all-on / all-off so iteration-2 graph == iteration-1 graph?).
"""
import numpy as np

from rgtm.model import RecurrentGraphTsetlinMachine
from rgtm.datasets import long_range_match


def train_curve(L, K, epochs=30, chunk=5, clauses=200, T=200, s=10.0,
                ntr=800, nte=300, seed=42):
    tr, trY, syms = long_range_match(ntr, L, seed=seed)
    te, teY, _ = long_range_match(nte, L, seed=seed + 1000)
    m = RecurrentGraphTsetlinMachine(
        number_of_clauses=clauses, T=T, s=s, depth=2, iterations=K)
    done = 0
    while done < epochs:
        step = min(chunk, epochs - done)
        m.fit(tr, trY, syms, epochs=step)
        done += step
        tr_acc = float((m.predict(tr) == trY).mean())
        te_acc = float((m.predict(te) == teY).mean())
        print(f"   L={L} K={K} ep={done:2d}  train={tr_acc:.3f} test={te_acc:.3f}",
              flush=True)
    # instrumentation on the recurrence
    if K >= 2:
        _, traj = m._forward_graphs(te, return_trajectory=True)
        for i, A in enumerate(traj):
            dens = float(A.mean())
            per_node = float(A.sum(axis=1).mean())
            print(f"      iter{i + 2} firing density={dens:.3f} "
                  f"mean_fired_clauses/node={per_node:.2f}", flush=True)
    return m


if __name__ == "__main__":
    print("=== RUNG 1: L=2 K=1 (must learn) ===")
    train_curve(2, 1)
    print("=== RUNG 2: L=2 K=2 (recurrence, easy) ===")
    train_curve(2, 2)
    print("=== RUNG 3: L=4 K=1 (expected chance) ===")
    train_curve(4, 1)
    print("=== RUNG 4: L=4 K=3 (target) ===")
    train_curve(4, 3)
