"""Load LRGB Peptides-func into cair Graphs for a MultiClass-TM probe.

Peptides-func is multilabel (10 tasks) over large molecular graphs (~150 nodes,
long-range by design). To probe with a MultiClass TM we (a) binarize to the most
class-balanced of the 10 tasks and (b) encode each node's 9 categorical features
as symbols `f{i}_{value}`. Edges use a single "bond" type. Large graphs make this
expensive; intended for a scoped probe / future full evaluation.
"""
from __future__ import annotations

import numpy as np
from GraphTsetlinMachine.graphs import Graphs


def load_peptides_func(root="/tmp/LRGB"):
    from torch_geometric.datasets import LRGBDataset
    return LRGBDataset(root=root, name="Peptides-func")


def pick_balanced_task(dataset, indices):
    Y = np.array([dataset[i].y.numpy().ravel() for i in indices])
    rates = Y.mean(axis=0)
    return int(np.argmin(np.abs(rates - 0.5))), rates


def labels_binary(dataset, indices, task):
    return np.array([int(dataset[i].y.numpy().ravel()[task]) for i in indices],
                    dtype=np.uint32)


def feature_symbols(dataset, indices, scan=3000):
    syms = set()
    for i in indices[:scan]:
        for row in dataset[i].x.numpy():
            for fi, v in enumerate(row):
                syms.add(f"f{fi}_{int(v)}")
    return sorted(syms)


def build_peptides_graphs(dataset, indices, symbols, codebook=None, hv=256,
                          hv_bits=2, one_hot=False):
    n = len(indices)
    symset = set(symbols)
    if codebook is None:
        g = Graphs(n, symbols=symbols, hypervector_size=hv,
                   hypervector_bits=hv_bits, one_hot_encoding=one_hot)
    else:
        g = Graphs(n, init_with=codebook)
    for gi, idx in enumerate(indices):
        g.set_number_of_graph_nodes(gi, int(dataset[idx].num_nodes))
    g.prepare_node_configuration()
    for gi, idx in enumerate(indices):
        d = dataset[idx]
        ei = d.edge_index.numpy()
        outdeg = np.zeros(int(d.num_nodes), dtype=int)
        for s in ei[0]:
            outdeg[int(s)] += 1
        for ni in range(int(d.num_nodes)):
            g.add_graph_node(gi, ni, int(outdeg[ni]), node_type_name="atom")
    g.prepare_edge_configuration()
    for gi, idx in enumerate(indices):
        ei = dataset[idx].edge_index.numpy()
        for k in range(ei.shape[1]):
            g.add_graph_node_edge(gi, int(ei[0, k]), int(ei[1, k]), "bond")
    for gi, idx in enumerate(indices):
        x = dataset[idx].x.numpy()
        for ni in range(x.shape[0]):
            for fi, v in enumerate(x[ni]):
                s = f"f{fi}_{int(v)}"
                if s in symset:
                    g.add_graph_node_property(gi, ni, s)
    g.encode()
    return g
