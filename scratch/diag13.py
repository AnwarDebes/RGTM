"""Clean positive-signal test: A-adjacent-to-B needs >=1 message round.

rounds=0 (single pass) must be ~chance (presence balanced); rounds>=1 should
solve and be stable across seeds (many samples, local relational pattern).
"""
import numpy as np

from rgtm.recurrent_tm import RecurrentGraphTM
from rgtm.graphs_builder import build_graphs
from rgtm.datasets import adjacent_pair


def build(n, syms, codebook, seed):
    specs, Y, _ = adjacent_pair(n, n_nodes=12, seed=seed)
    g = build_graphs(specs, syms, codebook, None, True, True, len(syms), 1)
    return g, Y


def run(rounds, seed, clauses=300, T=300, s=5.0, mil=16, epochs=25,
        ntr=800, nte=400):
    _, _, syms = adjacent_pair(1, seed=0)
    gtr, Ytr = build(ntr, syms, None, seed)
    gte, Yte = build(nte, syms, gtr, seed + 1000)
    m = RecurrentGraphTM(number_of_clauses=clauses, T=T, s=s, rounds=rounds,
                         message_size=128, max_included_literals=mil,
                         one_hot_encoding=True)
    m.fit(gtr, Ytr, epochs=epochs)
    return float((m.predict(gte) == Yte).mean())


if __name__ == "__main__":
    for rounds in (0, 1, 2):
        accs = [run(rounds, sd) for sd in (41, 42)]
        print(f"adjacent_pair rounds={rounds}: {[f'{a:.3f}' for a in accs]} "
              f"mean={np.mean(accs):.3f}", flush=True)
