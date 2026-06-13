"""RGTM vs single-pass vs param-heavy cair depth on a TUDataset.

Key comparison: RGTM ties one message-TA across R rounds (constant params),
while cair depth=D needs D-1 separate message-TA banks. We report test accuracy
(5 seeds, random 80/20) and the message-TA bank count, to show whether tying
matches growing-depth accuracy at a fraction of the parameters.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from rgtm.recurrent_tm import RecurrentGraphTM
from rgtm.datasets.tudataset import (atom_symbols, build_tu_graphs, labels,
                                     load_tu, split_indices)


def make_configs(clauses, T, s, msg):
    common = dict(number_of_clauses=clauses, T=T, s=s, message_size=msg,
                  message_bits=2, max_included_literals=32)
    return [
        ("single-pass(d1)", lambda: MultiClassGraphTsetlinMachine(depth=1, **common), 0),
        ("cair-depth2",     lambda: MultiClassGraphTsetlinMachine(depth=2, **common), 1),
        ("cair-depth3",     lambda: MultiClassGraphTsetlinMachine(depth=3, **common), 2),
        ("cair-depth5",     lambda: MultiClassGraphTsetlinMachine(depth=5, **common), 4),
        ("RGTM-r1",         lambda: RecurrentGraphTM(rounds=1, **common), 1),
        ("RGTM-r2",         lambda: RecurrentGraphTM(rounds=2, **common), 1),
        ("RGTM-r4",         lambda: RecurrentGraphTM(rounds=4, **common), 1),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUTAG")
    ap.add_argument("--clauses", type=int, default=1000)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--s", type=float, default=10.0)
    ap.add_argument("--msg", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--out", default="results/tudataset.jsonl")
    a = ap.parse_args()

    ds = load_tu(a.dataset)
    syms = atom_symbols(ds)
    n = len(ds)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    results = {}
    for seed in a.seeds:
        tr_idx, te_idx = split_indices(n, seed)
        g_tr = build_tu_graphs(ds, tr_idx, syms)
        g_te = build_tu_graphs(ds, te_idx, syms, codebook=g_tr)
        Ytr, Yte = labels(ds, tr_idx), labels(ds, te_idx)
        for (name, factory, msg_banks) in make_configs(a.clauses, a.T, a.s, a.msg):
            t0 = time.time()
            m = factory()
            m.fit(g_tr, Ytr, epochs=a.epochs)
            acc = float((m.predict(g_te) == Yte).mean())
            dt = time.time() - t0
            results.setdefault(name, []).append(acc)
            rec = dict(dataset=a.dataset, model=name, seed=int(seed), acc=acc,
                       msg_ta_banks=msg_banks, clauses=a.clauses, T=a.T, s=a.s,
                       epochs=a.epochs, sec=round(dt, 1))
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  seed={seed} {name:16s} acc={acc:.4f} banks={msg_banks} ({dt:.0f}s)",
                  flush=True)

    print(f"\n=== {a.dataset} summary ({len(a.seeds)} seeds), clauses={a.clauses} ===")
    print(f"{'model':16s} {'mean':>7} {'std':>6} {'msgTAbanks':>11}")
    for (name, _f, banks) in make_configs(a.clauses, a.T, a.s, a.msg):
        accs = results[name]
        print(f"{name:16s} {np.mean(accs):7.4f} {np.std(accs):6.4f} {banks:>11d}",
              flush=True)


if __name__ == "__main__":
    main()
