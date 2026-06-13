"""Parameter efficiency: RGTM(R, ONE shared bank) vs cair depth=R+1 (R banks).

Task: within_distance(k) -- B is within k hops of A. A model must do >=k message
rounds to separate the classes. We compare, at matched clauses and 5 seeds:
  - single-pass / depth-2     (reach <= 1)  -> should fail for k>=2
  - cair depth=k+1 (k banks)  (reach k)     -> should solve
  - RGTM rounds=k (1 bank)    (reach k)     -> should solve at 1/k the banks
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from rgtm.recurrent_tm import RecurrentGraphTM
from rgtm.graphs_builder import build_graphs
from rgtm.datasets import within_distance


def build(n, syms, codebook, n_nodes, k, seed):
    specs, Y, _ = within_distance(n, n_nodes=n_nodes, k=k, seed=seed)
    g = build_graphs(specs, syms, codebook, None, True, True, len(syms), 1)
    return g, Y


def configs(clauses, T, s, msg):
    common = dict(number_of_clauses=clauses, T=T, s=s, message_size=msg,
                  message_bits=2, max_included_literals=16, one_hot_encoding=True)
    return [
        ("single-pass(d1)", lambda: MultiClassGraphTsetlinMachine(depth=1, **common), 0, 1),
        ("cair-depth2",     lambda: MultiClassGraphTsetlinMachine(depth=2, **common), 1, 1),
        ("cair-depth3",     lambda: MultiClassGraphTsetlinMachine(depth=3, **common), 2, 2),
        ("RGTM-r1",         lambda: RecurrentGraphTM(rounds=1, **common), 1, 1),
        ("RGTM-r2",         lambda: RecurrentGraphTM(rounds=2, **common), 1, 2),
        ("RGTM-r3",         lambda: RecurrentGraphTM(rounds=3, **common), 1, 3),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-nodes", type=int, default=14)
    ap.add_argument("--clauses", type=int, default=1000)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--msg", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    ap.add_argument("--ntr", type=int, default=800)
    ap.add_argument("--nte", type=int, default=400)
    ap.add_argument("--out", default="results/depth_efficiency.jsonl")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    _, _, syms = within_distance(1, n_nodes=a.n_nodes, k=a.k, seed=0)
    res = {}
    for seed in a.seeds:
        g_tr, Ytr = build(a.ntr, syms, None, a.n_nodes, a.k, seed)
        g_te, Yte = build(a.nte, syms, g_tr, a.n_nodes, a.k, seed + 1000)
        for (name, factory, banks, reach) in configs(a.clauses, a.T, a.s, a.msg):
            t0 = time.time()
            m = factory()
            m.fit(g_tr, Ytr, epochs=a.epochs)
            acc = float((m.predict(g_te) == Yte).mean())
            res.setdefault(name, []).append(acc)
            with open(a.out, "a") as f:
                f.write(json.dumps(dict(task=f"within_distance_k{a.k}", model=name,
                        seed=int(seed), acc=acc, msg_ta_banks=banks, reach=reach,
                        clauses=a.clauses, sec=round(time.time() - t0, 1))) + "\n")
            print(f"  seed={seed} {name:16s} acc={acc:.3f} banks={banks} reach={reach}",
                  flush=True)

    print(f"\n=== within_distance(k={a.k}) summary, {len(a.seeds)} seeds, clauses={a.clauses} ===")
    print(f"{'model':16s} {'mean':>7} {'std':>6} {'msgTAbanks':>11} {'reach':>6}")
    for (name, _f, banks, reach) in configs(a.clauses, a.T, a.s, a.msg):
        accs = res[name]
        print(f"{name:16s} {np.mean(accs):7.3f} {np.std(accs):6.3f} {banks:>11d} {reach:>6d}",
              flush=True)


if __name__ == "__main__":
    main()
