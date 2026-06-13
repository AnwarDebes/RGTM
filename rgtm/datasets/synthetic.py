"""Synthetic path-graph tasks that separate single-pass reach from recurrence.

long_range_match
    A path of ``L`` nodes. Node 0 (the readout end) carries QUERY + a colour;
    node L-1 carries KEY + a colour; interior nodes are BLANK. Label = (the two
    colours match). The decisive signal must travel L-1 hops, so a single-pass
    GraphTM with message-passing depth D < L-1 is at chance, while a recurrence
    of K iterations (each ~1 hop at depth=2) solves it once K >= L-1. This is
    the cleanest demonstration that recurrence extends the receptive field.

parity_on_path
    HGTM's load-bearing benchmark. Interior nodes of a length-L path carry a
    {0,1} bit; label = parity of the bit count. Requires both long reach and
    iterative accumulation -- the task HGTM's stacked L>=2 path collapsed on.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..graphs_builder import GraphSpec


def _chain_edges(L: int) -> List[Tuple[int, int, str]]:
    # A single symmetric edge type for both directions. cair's GraphTM learns
    # message passing far more reliably with one undirected edge type than with
    # distinct directed types (next/prev), which empirically fail to train on
    # these path tasks.
    edges: List[Tuple[int, int, str]] = []
    for i in range(L - 1):
        edges.append((i, i + 1, "adj"))
        edges.append((i + 1, i, "adj"))
    return edges


def long_range_match(n_graphs: int, L: int, n_colors: int = 2, seed: int = 0):
    """Return (specs, Y, original_symbols) for the key/query match task."""
    assert L >= 2
    rng = np.random.default_rng(seed)
    colors = [f"col{c}" for c in range(n_colors)]
    original_symbols = ["QUERY", "KEY", "BLANK"] + colors

    specs: List[GraphSpec] = []
    Y = np.empty(n_graphs, dtype=np.uint32)
    edges = _chain_edges(L)
    for g in range(n_graphs):
        q = int(rng.integers(0, n_colors))
        k = int(rng.integers(0, n_colors))
        node_symbols: List[List[str]] = []
        for ni in range(L):
            if ni == 0:
                node_symbols.append(["QUERY", colors[q]])
            elif ni == L - 1:
                node_symbols.append(["KEY", colors[k]])
            else:
                node_symbols.append(["BLANK"])
        specs.append(GraphSpec(n_nodes=L, edges=list(edges), node_symbols=node_symbols))
        Y[g] = 1 if q == k else 0
    return specs, Y, original_symbols


def adjacent_pair(n_graphs: int, n_nodes: int = 12, avg_deg: int = 3, seed: int = 0):
    """Return (specs, Y, symbols) for the A-adjacent-to-B task.

    Each graph is a random connected undirected graph with exactly one A node and
    one B node (all others blank). Label = 1 iff A and B are adjacent. Presence
    of A and B is identical in both classes, so a presence/count classifier
    (single pass, no message round) is at chance; one message round suffices to
    detect the adjacency relation. Stable and learnable with many samples.
    """
    rng = np.random.default_rng(seed)
    symbols = ["A", "B", "blank"]
    specs: List[GraphSpec] = []
    Y = np.empty(n_graphs, dtype=np.uint32)
    for g in range(n_graphs):
        edges = set()
        perm = rng.permutation(n_nodes)
        for i in range(1, n_nodes):  # random spanning tree (connectivity)
            j = int(perm[int(rng.integers(0, i))])
            u, v = int(perm[i]), j
            edges.add((min(u, v), max(u, v)))
        n_extra = max(0, int(avg_deg * n_nodes / 2) - (n_nodes - 1))
        for _ in range(n_extra):
            u, v = int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes))
            if u != v:
                edges.add((min(u, v), max(u, v)))
        adj = {i: set() for i in range(n_nodes)}
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        cand = [i for i in range(n_nodes) if 0 < len(adj[i]) < n_nodes - 1]
        a = int(rng.choice(cand))
        label = int(rng.integers(0, 2))
        if label == 1:
            b = int(rng.choice(list(adj[a])))
        else:
            non = [i for i in range(n_nodes) if i != a and i not in adj[a]]
            b = int(rng.choice(non))
        node_symbols = [["blank"] for _ in range(n_nodes)]
        node_symbols[a] = ["A"]
        node_symbols[b] = ["B"]
        edge_list: List[Tuple[int, int, str]] = []
        for (u, v) in edges:
            edge_list.append((u, v, "adj"))
            edge_list.append((v, u, "adj"))
        specs.append(GraphSpec(n_nodes, edge_list, node_symbols))
        Y[g] = label
    return specs, Y, symbols


def within_distance(n_graphs: int, n_nodes: int = 14, k: int = 2,
                    avg_deg: int = 2, seed: int = 0):
    """Return (specs, Y, symbols) for the 'is B within k hops of A?' task.

    One A node, one B node (presence balanced), random connected graph. Label = 1
    iff the shortest-path distance from A to B is <= k, else (distance > k). A
    classifier needs at least k message rounds to separate the classes: with
    fewer rounds A's signal cannot reach a B sitting at distance k. This makes
    depth genuinely necessary, so RGTM(R=k) with one shared bank can be compared
    against cair depth=k+1 with k separate banks.
    """
    from collections import deque

    rng = np.random.default_rng(seed)
    symbols = ["A", "B", "blank"]
    specs: List[GraphSpec] = []
    Y = np.empty(n_graphs, dtype=np.uint32)
    g = 0
    attempts = 0
    while g < n_graphs:
        attempts += 1
        edges = set()
        perm = rng.permutation(n_nodes)
        for i in range(1, n_nodes):
            j = int(perm[int(rng.integers(0, i))])
            u, v = int(perm[i]), j
            edges.add((min(u, v), max(u, v)))
        for _ in range(max(0, int(avg_deg * n_nodes / 2) - (n_nodes - 1))):
            u, v = int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes))
            if u != v:
                edges.add((min(u, v), max(u, v)))
        adj = {i: set() for i in range(n_nodes)}
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        a = int(rng.integers(0, n_nodes))
        dist = {a: 0}
        dq = deque([a])
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    dq.append(w)
        label = int(rng.integers(0, 2))
        if label == 1:
            cand = [v for v in dist if v != a and dist[v] <= k]
        else:
            cand = [v for v in dist if dist[v] > k]
        if not cand:
            if attempts > 50 * n_graphs:
                break
            continue
        b = int(rng.choice(cand))
        node_symbols = [["blank"] for _ in range(n_nodes)]
        node_symbols[a] = ["A"]
        node_symbols[b] = ["B"]
        edge_list: List[Tuple[int, int, str]] = []
        for (u, v) in edges:
            edge_list.append((u, v, "adj"))
            edge_list.append((v, u, "adj"))
        specs.append(GraphSpec(n_nodes, edge_list, node_symbols))
        Y[g] = label
        g += 1
    return specs, Y[:g], symbols


def parity_on_path(n_graphs: int, L: int, seed: int = 0):
    """Return (specs, Y, original_symbols) for depth-N parity-on-path.

    Path of L nodes; endpoints carry no bit; interior nodes 1..L-2 carry a
    random {0,1} bit; label = parity of the bit count.
    """
    assert L >= 3
    rng = np.random.default_rng(seed)
    original_symbols = ["b0", "b1"]
    n_interior = L - 2

    specs: List[GraphSpec] = []
    Y = np.empty(n_graphs, dtype=np.uint32)
    edges = _chain_edges(L)
    for g in range(n_graphs):
        bits = rng.integers(0, 2, size=n_interior)
        node_symbols: List[List[str]] = []
        for ni in range(L):
            if ni == 0 or ni == L - 1:
                node_symbols.append([])  # endpoints: no feature
            else:
                node_symbols.append([f"b{int(bits[ni - 1])}"])
        specs.append(GraphSpec(n_nodes=L, edges=list(edges), node_symbols=node_symbols))
        Y[g] = int(bits.sum() % 2)
    return specs, Y, original_symbols
