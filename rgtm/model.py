"""NEGATIVE-RESULT REFERENCE (PAPER.md section 3) -- node-symbol re-encoding recurrence.

This is the FAILED approach, kept only for reproducibility of the negatives: it
re-encodes per-node clause firings as node symbols each iteration, which floods
the input with mostly-absent symbols and lets Type II feedback poison the clauses
to death (section 3.1); the per-round deep supervision below also corrupts the
shared bank (section 3.2). The WORKING model is
``rgtm.recurrent_tm.RecurrentGraphTM`` (weight-tied message passing).

Original design notes (deep supervision, weight-tied):

    for each epoch:
        G <- build iteration-1 graph (original symbols only)
        fit(G, Y)                      # supervise iteration 1 on the true label
        for t in 2..K:
            A  <- transform_nodewise(G)   # per-node clause firings
            G  <- build iteration-t graph (firings + original symbols)
            fit(G, Y)                  # SAME automata, supervise on true label

Because there is a single automata bank trained against the true label at every
iteration, credit assignment is trivial -- there is no projected feedback to go
vacuous (the failure mode that collapses HGTM's joint L>=2 path). K = 1 reduces
to a single-pass GraphTM and is the correctness oracle.

Inference repeats the forward recurrence; ``predict`` returns the argmax of the
final-iteration class sums. Optional halting stops early per graph when the
per-node firing pattern stabilises (fixed point) -- adaptive test-time depth.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

from .graphs_builder import GraphSpec, build_graphs


class RecurrentGraphTsetlinMachine:
    def __init__(
        self,
        number_of_clauses: int,
        T: int,
        s: float,
        depth: int = 2,
        iterations: int = 3,
        message_size: int = 256,
        message_bits: int = 2,
        number_of_state_bits: int = 8,
        max_included_literals: Optional[int] = None,
        boost_true_positive_feedback: int = 1,
        q: float = 1.0,
        input_injection: bool = True,
        max_refeed: Optional[int] = None,
        one_hot: bool = True,
        hypervector_size: Optional[int] = None,
        hypervector_bits: int = 2,
        grid=(16 * 13 * 4, 1, 1),
        block=(128, 1, 1),
    ):
        self.number_of_clauses = int(number_of_clauses)
        self.T = int(T)
        self.s = s
        self.depth = int(depth)
        self.iterations = int(iterations)
        self.message_size = message_size
        self.message_bits = message_bits
        self.number_of_state_bits = number_of_state_bits
        self.max_included_literals = max_included_literals
        self.boost_true_positive_feedback = boost_true_positive_feedback
        self.q = q
        self.input_injection = input_injection
        self.max_refeed = max_refeed
        self.one_hot = one_hot
        self.hypervector_bits = hypervector_bits
        self.grid = grid
        self.block = block

        self._tm: Optional[MultiClassGraphTsetlinMachine] = None
        self._codebook = None
        self._all_symbols: Optional[list] = None
        self._hv_size = hypervector_size

    # ------------------------------------------------------------------ setup
    def _ensure_tm(self):
        if self._tm is None:
            self._tm = MultiClassGraphTsetlinMachine(
                number_of_clauses=self.number_of_clauses,
                T=self.T,
                s=self.s,
                depth=self.depth,
                message_size=self.message_size,
                message_bits=self.message_bits,
                number_of_state_bits=self.number_of_state_bits,
                max_included_literals=self.max_included_literals,
                boost_true_positive_feedback=self.boost_true_positive_feedback,
                q=self.q,
                one_hot_encoding=self.one_hot,
                grid=self.grid,
                block=self.block,
            )

    def _prepare_vocab(self, original_symbols):
        if self._all_symbols is None:
            clause_syms = [f"__c{j}" for j in range(self.number_of_clauses)]
            self._all_symbols = list(original_symbols) + clause_syms
            if self._hv_size is None:
                self._hv_size = max(32, 2 * len(self._all_symbols))

    def _build(self, specs, firings):
        graphs = build_graphs(
            specs,
            self._all_symbols,
            self._codebook,
            firings,
            self.input_injection,
            self.one_hot,
            self._hv_size,
            self.hypervector_bits,
        )
        if self._codebook is None:
            self._codebook = graphs
        return graphs

    def _firings(self, A, specs):
        out = []
        for gi, sp in enumerate(specs):
            nodes = []
            for ni in range(sp.n_nodes):
                fired = np.nonzero(A[gi, :, ni])[0]
                if self.max_refeed and len(fired) > self.max_refeed:
                    fired = fired[: self.max_refeed]
                nodes.append(fired)
            out.append(nodes)
        return out

    # --------------------------------------------------------------- training
    def fit(
        self,
        specs: Sequence[GraphSpec],
        Y,
        original_symbols,
        epochs: int = 50,
        eval_specs=None,
        eval_Y=None,
        verbose: bool = False,
    ):
        self._ensure_tm()
        self._prepare_vocab(original_symbols)
        Y = np.asarray(Y).astype(np.uint32)
        for ep in range(epochs):
            graphs = self._build(specs, None)
            self._tm.fit(graphs, Y, epochs=1, incremental=True)
            for _t in range(2, self.iterations + 1):
                A, _ = self._tm.transform_nodewise(graphs)
                graphs = self._build(specs, self._firings(A, specs))
                self._tm.fit(graphs, Y, epochs=1, incremental=True)
            if verbose and eval_specs is not None:
                acc = float((self.predict(eval_specs) == np.asarray(eval_Y)).mean())
                print(f"  epoch {ep:3d}  test_acc {acc:.4f}", flush=True)
        return self

    # -------------------------------------------------------------- inference
    def _forward_graphs(self, specs, return_trajectory=False):
        graphs = self._build(specs, None)
        traj = []
        for _t in range(2, self.iterations + 1):
            A, _ = self._tm.transform_nodewise(graphs)
            if return_trajectory:
                traj.append(A)
            graphs = self._build(specs, self._firings(A, specs))
        if return_trajectory:
            return graphs, traj
        return graphs

    def score(self, specs):
        graphs = self._forward_graphs(specs)
        return self._tm.score(graphs)

    def predict(self, specs):
        return np.argmax(self.score(specs), axis=1)
