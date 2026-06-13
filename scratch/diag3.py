"""Isolate build_graphs vs the cair reference build on the SAME equality task.

If the demo-style build learns but my GraphSpec/build_graphs build does not, the
bug is in build_graphs. Also prints my graph #0 structure to verify nodes,
edges, and properties are constructed correctly.
"""
import random

import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from rgtm.graphs_builder import GraphSpec, build_graphs


def demo_build(n, syms, init_with=None, seed=0):
    rng = random.Random(seed)
    g = Graphs(n, symbols=syms, hypervector_size=32, hypervector_bits=2,
               init_with=init_with)
    for i in range(n):
        g.set_number_of_graph_nodes(i, 2)
    g.prepare_node_configuration()
    for i in range(n):
        g.add_graph_node(i, "Node 1", 1)
        g.add_graph_node(i, "Node 2", 1)
    g.prepare_edge_configuration()
    for i in range(n):
        g.add_graph_node_edge(i, "Node 1", "Node 2", "Plain")
        g.add_graph_node_edge(i, "Node 2", "Node 1", "Plain")
    Y = np.empty(n, dtype=np.uint32)
    for i in range(n):
        a = rng.choice(syms)
        b = rng.choice(syms)
        g.add_graph_node_property(i, "Node 1", a)
        g.add_graph_node_property(i, "Node 2", b)
        Y[i] = 0 if a == b else 1
    g.encode()
    return g, Y


def my_specs(n, syms, seed=0):
    rng = random.Random(seed)
    specs = []
    Y = np.empty(n, dtype=np.uint32)
    for i in range(n):
        a = rng.choice(syms)
        b = rng.choice(syms)
        specs.append(GraphSpec(2, [(0, 1, "Plain"), (1, 0, "Plain")], [[a], [b]]))
        Y[i] = 0 if a == b else 1
    return specs, Y


def train_eval(gtr, Ytr, gte, Yte, label):
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=100, T=100, s=5.0, depth=2,
        message_size=256, message_bits=2, max_included_literals=4)
    for _ in range(15):
        tm.fit(gtr, Ytr, epochs=1, incremental=True)
    acc = float((tm.predict(gte) == Yte).mean())
    A, _ = tm.transform_nodewise(gte)
    print(f"{label}: test={acc:.3f} density={A.mean():.3f}", flush=True)


if __name__ == "__main__":
    syms = ["A", "B"]

    # 1) cair reference build
    gtr, Ytr = demo_build(800, syms, seed=1)
    gte, Yte = demo_build(300, syms, init_with=gtr, seed=2)
    train_eval(gtr, Ytr, gte, Yte, "DEMO-build  ")

    # 2) my build_graphs on identical task
    sp_tr, Ytr2 = my_specs(800, syms, seed=1)
    sp_te, Yte2 = my_specs(300, syms, seed=2)
    g_tr = build_graphs(sp_tr, syms, None, None, True, False, 32, 2)
    g_te = build_graphs(sp_te, syms, g_tr, None, True, False, 32, 2)
    print("--- my build_graphs, graph #0 structure ---")
    g_tr.print_graph(0)
    train_eval(g_tr, Ytr2, g_te, Yte2, "MY-build    ")
