"""RGTM on the A-adjacent-to-B task: does a message round add capability, stably?

Single pass (rounds=0) is at chance (presence is balanced); >=1 message round can
detect the adjacency relation. Reports test accuracy over multiple seeds for each
round count, plus the message-TA bank count (always 1 for RGTM regardless of
rounds -> constant parameters).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from rgtm.recurrent_tm import RecurrentGraphTM
from rgtm.graphs_builder import build_graphs
from rgtm.datasets import adjacent_pair


def build(n, syms, codebook, n_nodes, seed):
    specs, Y, _ = adjacent_pair(n, n_nodes=n_nodes, seed=seed)
    g = build_graphs(specs, syms, codebook, None, True, True, len(syms), 1)
    return g, Y


def run(rounds, seed, clauses, T, s, mil, msg, epochs, ntr, nte, n_nodes):
    _, _, syms = adjacent_pair(1, seed=0)
    gtr, Ytr = build(ntr, syms, None, n_nodes, seed)
    gte, Yte = build(nte, syms, gtr, n_nodes, seed + 1000)
    m = RecurrentGraphTM(number_of_clauses=clauses, T=T, s=s, rounds=rounds,
                         message_size=msg, max_included_literals=mil,
                         one_hot_encoding=True)
    t0 = time.time()
    m.fit(gtr, Ytr, epochs=epochs)
    acc = float((m.predict(gte) == Yte).mean())
    return acc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clauses", type=int, default=1000)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--mil", type=int, default=16)
    ap.add_argument("--msg", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    ap.add_argument("--n-nodes", type=int, default=12)
    ap.add_argument("--ntr", type=int, default=800)
    ap.add_argument("--nte", type=int, default=400)
    ap.add_argument("--out", default="results/adjacency.jsonl")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = {}
    for rounds in a.rounds:
        for seed in a.seeds:
            acc, dt = run(rounds, seed, a.clauses, a.T, a.s, a.mil, a.msg,
                          a.epochs, a.ntr, a.nte, a.n_nodes)
            res.setdefault(rounds, []).append(acc)
            rec = dict(task="adjacent_pair", rounds=rounds, seed=seed, acc=acc,
                       clauses=a.clauses, T=a.T, s=a.s, epochs=a.epochs,
                       msg_ta_banks=1 if rounds > 0 else 0, sec=round(dt, 1))
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  rounds={rounds} seed={seed} acc={acc:.3f} ({dt:.0f}s)", flush=True)

    print(f"\n=== adjacent_pair summary, clauses={a.clauses} ===")
    print(f"{'rounds':>6} {'mean':>7} {'std':>6} {'min':>6} {'max':>6}")
    for rounds in a.rounds:
        accs = res[rounds]
        print(f"{rounds:>6} {np.mean(accs):7.3f} {np.std(accs):6.3f} "
              f"{np.min(accs):6.3f} {np.max(accs):6.3f}", flush=True)


if __name__ == "__main__":
    main()
