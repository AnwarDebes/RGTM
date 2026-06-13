"""Proof-of-mechanism: recurrence extends the receptive field.

Sweeps path length L and iteration count K on the long-range key/query match
task. Expectation: accuracy is at chance while K < L-1 (the KEY colour never
reaches the QUERY node) and jumps to ~1.0 once K >= L-1. K=1 is the single-pass
GraphTM baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from rgtm.datasets import long_range_match
from rgtm.model import RecurrentGraphTsetlinMachine


def run_one(L, K, clauses, T, s, depth, epochs, n_train, n_test, seed):
    tr_specs, tr_Y, syms = long_range_match(n_train, L, seed=seed)
    te_specs, te_Y, _ = long_range_match(n_test, L, seed=seed + 1000)
    model = RecurrentGraphTsetlinMachine(
        number_of_clauses=clauses, T=T, s=s, depth=depth, iterations=K,
        message_bits=2,
    )
    t0 = time.time()
    model.fit(tr_specs, tr_Y, syms, epochs=epochs)
    acc = float((model.predict(te_specs) == te_Y).mean())
    return acc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--K", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--clauses", type=int, default=100)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/match_proof.jsonl")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    print(f"{'L':>3} {'K':>3} {'acc':>7} {'sec':>7}", flush=True)
    with open(a.out, "a") as f:
        for L in a.L:
            for K in a.K:
                acc, sec = run_one(
                    L, K, a.clauses, a.T, a.s, a.depth, a.epochs,
                    a.n_train, a.n_test, a.seed,
                )
                rec = dict(
                    task="long_range_match", L=L, K=K, clauses=a.clauses, T=a.T,
                    s=a.s, depth=a.depth, epochs=a.epochs, n_train=a.n_train,
                    n_test=a.n_test, seed=a.seed, acc=acc, sec=round(sec, 1),
                )
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"{L:>3} {K:>3} {acc:>7.3f} {sec:>7.1f}", flush=True)


if __name__ == "__main__":
    main()
