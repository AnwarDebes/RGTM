"""Host-side construction of cair ``Graphs`` objects for one RGTM iteration.

The unified symbol vocabulary is ``original_symbols + [__c0 .. __c{K-1}]`` so a
single weight-tied model can read both the raw node symbols and the previous
iteration's clause firings in one fixed literal space. With one-hot encoding the
re-encoding of clause firings into literals is bit-exact (no hypervector
collisions), mirroring HGTM's ``encode_layer_output_as_literals`` invariant but
pointed back at the *same* automata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from GraphTsetlinMachine.graphs import Graphs

CLAUSE_PREFIX = "__c"


@dataclass
class GraphSpec:
    """A single graph: topology + the original (iteration-1) node symbols.

    ``edges`` are directed ``(src, dst, edge_type)``; for an undirected graph add
    both directions. ``node_symbols[i]`` is the list of original symbols present
    at node ``i``. ``node_types`` is optional (defaults to a single shared type
    so every clause applies to every node).
    """

    n_nodes: int
    edges: List[Tuple[int, int, str]]
    node_symbols: List[List[str]]
    node_types: Optional[List[str]] = None


def out_degrees(spec: GraphSpec) -> List[int]:
    deg = [0] * spec.n_nodes
    for (src, _dst, _t) in spec.edges:
        deg[src] += 1
    return deg


def build_graphs(
    specs: Sequence[GraphSpec],
    all_symbols: Sequence[str],
    codebook: Optional[Graphs],
    firings,  # None for iteration 1, else firings[g][node] -> iterable of clause ids
    input_injection: bool,
    one_hot: bool,
    hypervector_size: int,
    hypervector_bits: int,
) -> Graphs:
    n = len(specs)
    if codebook is None:
        graphs = Graphs(
            n,
            symbols=list(all_symbols),
            hypervector_size=hypervector_size,
            hypervector_bits=hypervector_bits,
            one_hot_encoding=one_hot,
        )
    else:
        graphs = Graphs(n, init_with=codebook)

    for gi, sp in enumerate(specs):
        graphs.set_number_of_graph_nodes(gi, sp.n_nodes)
    graphs.prepare_node_configuration()

    for gi, sp in enumerate(specs):
        deg = out_degrees(sp)
        for ni in range(sp.n_nodes):
            nt = sp.node_types[ni] if sp.node_types is not None else "n"
            graphs.add_graph_node(gi, ni, deg[ni], node_type_name=nt)
    graphs.prepare_edge_configuration()

    for gi, sp in enumerate(specs):
        for (src, dst, et) in sp.edges:
            graphs.add_graph_node_edge(gi, src, dst, et)

    for gi, sp in enumerate(specs):
        for ni in range(sp.n_nodes):
            if firings is None or input_injection:
                for sym in sp.node_symbols[ni]:
                    graphs.add_graph_node_property(gi, ni, sym)
            if firings is not None:
                for j in firings[gi][ni]:
                    graphs.add_graph_node_property(gi, ni, f"{CLAUSE_PREFIX}{int(j)}")

    graphs.encode()
    return graphs
