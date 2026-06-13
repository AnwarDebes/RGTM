"""Smoke test: confirm a plain cair GraphTM trains on this GPU.

Task: 2-node graph, each node carries symbol A or B; label = (A==B).
This is the cair NoisyXOR-equality task, shrunk. If this learns to
~100%, the full stack (pycuda + kernels + V100) is healthy.
"""
import random
import time

import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine


def build(n, symbols, init_with=None, seed=0):
    rng = random.Random(seed)
    g = Graphs(n, symbols=symbols, hypervector_size=16, hypervector_bits=2,
               init_with=init_with)
    for i in range(n):
        g.set_number_of_graph_nodes(i, 2)
    g.prepare_node_configuration()
    for i in range(n):
        g.add_graph_node(i, "n0", 1)
        g.add_graph_node(i, "n1", 1)
    g.prepare_edge_configuration()
    for i in range(n):
        g.add_graph_node_edge(i, "n0", "n1", "e")
        g.add_graph_node_edge(i, "n1", "n0", "e")
    Y = np.empty(n, dtype=np.uint32)
    for i in range(n):
        a = rng.choice(symbols)
        b = rng.choice(symbols)
        g.add_graph_node_property(i, "n0", a)
        g.add_graph_node_property(i, "n1", b)
        Y[i] = 0 if a == b else 1
    g.encode()
    return g, Y


def main():
    tr, Ytr = build(1000, ["A", "B"], seed=1)
    te, Yte = build(500, ["A", "B"], init_with=tr, seed=2)
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=20, T=40, s=3.0, depth=2,
        message_size=32, message_bits=2,
    )
    t0 = time.time()
    for e in range(10):
        tm.fit(tr, Ytr, epochs=1, incremental=True)
        acc = 100.0 * (tm.predict(te) == Yte).mean()
        print(f"epoch {e:2d}  test_acc {acc:6.2f}")
    print(f"total {time.time()-t0:.1f}s")
    final = 100.0 * (tm.predict(te) == Yte).mean()
    print("SMOKE_OK" if final >= 95.0 else f"SMOKE_WEAK final={final:.1f}")


if __name__ == "__main__":
    main()
