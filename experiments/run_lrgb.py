"""Scoped probe on LRGB Peptides-func (binarized): does message passing help on
a real long-range molecular dataset? Large graphs (~150 nodes) make this
expensive, so we subset for a probe and report accuracy vs the majority baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from rgtm.recurrent_tm import RecurrentGraphTM
from rgtm.datasets.lrgb import (build_peptides_graphs, feature_symbols,
                                labels_binary, load_peptides_func,
                                pick_balanced_task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntr", type=int, default=2500)
    ap.add_argument("--nte", type=int, default=1000)
    ap.add_argument("--clauses", type=int, default=2000)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--rounds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    ap.add_argument("--out", default="results/lrgb_peptides.jsonl")
    a = ap.parse_args()

    from torch_geometric.datasets import LRGBDataset
    tr_ds = LRGBDataset(root="/tmp/LRGB", name="Peptides-func", split="train")
    te_ds = LRGBDataset(root="/tmp/LRGB", name="Peptides-func", split="test")
    task, rates = pick_balanced_task(tr_ds, list(range(min(3000, len(tr_ds)))))
    print(f"chosen binary task={task} (train pos-rate {rates[task]:.3f})", flush=True)
    syms = feature_symbols(tr_ds, list(range(len(tr_ds))))
    print(f"feature symbols: {len(syms)}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = {}
    for seed in a.seeds:
        rng = np.random.default_rng(seed)
        tr_idx = rng.permutation(len(tr_ds))[:a.ntr].tolist()
        te_idx = rng.permutation(len(te_ds))[:a.nte].tolist()
        t0 = time.time()
        g_tr = build_peptides_graphs(tr_ds, tr_idx, syms)
        g_te = build_peptides_graphs(te_ds, te_idx, syms, codebook=g_tr)
        Ytr = labels_binary(tr_ds, tr_idx, task)
        Yte = labels_binary(te_ds, te_idx, task)
        maj = max(float(np.mean(Yte)), 1 - float(np.mean(Yte)))
        print(f"seed={seed} build {time.time()-t0:.0f}s  majority={maj:.3f}", flush=True)
        for rounds in a.rounds:
            t0 = time.time()
            if rounds == 0:
                m = MultiClassGraphTsetlinMachine(
                    number_of_clauses=a.clauses, T=a.T, s=a.s, depth=1,
                    message_size=128, message_bits=2, max_included_literals=32)
            else:
                m = RecurrentGraphTM(
                    number_of_clauses=a.clauses, T=a.T, s=a.s, rounds=rounds,
                    message_size=128, message_bits=2, max_included_literals=32)
            m.fit(g_tr, Ytr, epochs=a.epochs)
            acc = float((m.predict(g_te) == Yte).mean())
            res.setdefault(rounds, []).append(acc)
            with open(a.out, "a") as f:
                f.write(json.dumps(dict(task="peptides_func_bin", rounds=rounds,
                        seed=int(seed), acc=acc, majority=maj, clauses=a.clauses,
                        sec=round(time.time() - t0, 1))) + "\n")
            print(f"  rounds={rounds} acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== Peptides-func (binary task {task}) ===")
    for rounds in a.rounds:
        accs = res[rounds]
        print(f"rounds={rounds}: mean={np.mean(accs):.4f} std={np.std(accs):.4f}", flush=True)


if __name__ == "__main__":
    main()
