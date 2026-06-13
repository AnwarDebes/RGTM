"""Load a TUDataset (e.g. MUTAG, NCI1, PROTEINS) into cair Graphs objects.

Node symbol = the atom's one-hot label (atom_i). Edge type = the bond's one-hot
label (bond_j), or a single "bond" type if `single_edge_type` is set. Returns
cair Graphs that any cair-family TM (single-pass, depth-D, or RGTM) consumes.
"""
from __future__ import annotations

import numpy as np
from GraphTsetlinMachine.graphs import Graphs


def load_tu(name="MUTAG", root=None):
    from torch_geometric.datasets import TUDataset
    return TUDataset(root=root or f"/tmp/{name}", name=name)


def atom_symbols(dataset):
    return [f"atom_{i}" for i in range(int(dataset[0].x.shape[1]))]


def labels(dataset, indices):
    return np.array([int(dataset[i].y.item()) for i in indices], dtype=np.uint32)


def build_tu_graphs(dataset, indices, symbols, codebook=None,
                    hypervector_size=128, hypervector_bits=2, one_hot=False,
                    single_edge_type=False, add_degree=False, degree_cap=6):
    n = len(indices)
    if codebook is None:
        g = Graphs(n, symbols=symbols, hypervector_size=hypervector_size,
                   hypervector_bits=hypervector_bits, one_hot_encoding=one_hot)
    else:
        g = Graphs(n, init_with=codebook)

    for gi, idx in enumerate(indices):
        g.set_number_of_graph_nodes(gi, int(dataset[idx].num_nodes))
    g.prepare_node_configuration()

    for gi, idx in enumerate(indices):
        d = dataset[idx]
        ei = d.edge_index.numpy() if d.edge_index is not None else np.zeros((2, 0), int)
        outdeg = np.zeros(int(d.num_nodes), dtype=int)
        for s in ei[0]:
            outdeg[int(s)] += 1
        for ni in range(int(d.num_nodes)):
            g.add_graph_node(gi, ni, int(outdeg[ni]), node_type_name="atom")
    g.prepare_edge_configuration()

    for gi, idx in enumerate(indices):
        d = dataset[idx]
        ei = d.edge_index.numpy() if d.edge_index is not None else np.zeros((2, 0), int)
        ea = d.edge_attr.numpy() if getattr(d, "edge_attr", None) is not None else None
        for k in range(ei.shape[1]):
            s, t = int(ei[0, k]), int(ei[1, k])
            et = "bond" if (single_edge_type or ea is None) else f"bond_{int(ea[k].argmax())}"
            g.add_graph_node_edge(gi, s, t, et)

    for gi, idx in enumerate(indices):
        d = dataset[idx]
        x = d.x.numpy()
        deg = None
        if add_degree:
            ei = d.edge_index.numpy() if d.edge_index is not None else np.zeros((2, 0), int)
            deg = np.zeros(int(d.num_nodes), dtype=int)
            for s in ei[0]:
                deg[int(s)] += 1
        for ni in range(int(d.num_nodes)):
            g.add_graph_node_property(gi, ni, f"atom_{int(x[ni].argmax())}")
            if add_degree:
                g.add_graph_node_property(gi, ni, f"deg_{min(int(deg[ni]), degree_cap)}")
    g.encode()
    return g


def split_indices(n, seed, train_frac=0.8):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(train_frac * n)
    return perm[:k], perm[k:]
