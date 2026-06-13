# RGTM: a Recurrent Graph Tsetlin Machine with weight-tied message passing

> Working draft. Author: Anwar, University of Agder (UiA).
> Honest-scope note: this is a **characterization study**, not a win. We propose
> RGTM (weight-tied recurrent message passing) and report, with controlled
> experiments, *when it helps and when it does not*: tying is "free" when a single
> message round suffices (a 1-hop relation is solved perfectly at constant
> parameters, §4.2), but on genuine multi-hop tasks the tied bank **underperforms**
> the untied per-round banks of a deep GraphTM at matched clauses (§4.4): weight
> tying trades capacity for parameters. We also document two recurrent/deep-TM
> failure modes (node-symbol re-encoding poison; deep-supervision corruption, §3)
> and confirm that no deep/recurrent TM variant we tried solves long-range
> *relational* reasoning (§4.3). We do **not** claim parameter-efficient deep
> message passing or state-of-the-art accuracy.

## Abstract

The Graph Tsetlin Machine (GraphTM, Granmo et al. 2025) runs `D-1` rounds of
clause-to-clause message passing, each round using its own bank of Tsetlin
automata, so the automata count grows linearly with message-passing depth. We
introduce the **Recurrent Graph Tsetlin Machine (RGTM)**, which *ties* those
banks: a single shared message-automata bank is reused across every round, making
the message-passing parameter count independent of depth. Training reuses the
canonical cair credit-assignment rule (one Type I/II feedback per sample,
distributed to the node bank and -- once per round -- to the shared message
bank). RGTM with one round is bit-for-bit a single-message-round GraphTM (the
correctness oracle). Empirically, tying is "free" only when a single round
suffices: it solves a 1-hop relation perfectly at constant parameters, but on a
genuine 2-hop task the tied bank reaches 0.72 ± 0.20 where an untied depth-3
GraphTM reaches 1.000; one shared bank cannot specialise per hop. We report this
as a characterization of the capacity/parameter trade-off, not an efficiency win.

Along the way we document two negative results that should save the community
time. (1) **Node-symbol re-encoding** -- feeding each round's per-node clause
firings back as the next round's node symbols (the natural lift of HGTM's
`encode_layer_output_as_literals` to a recurrence) -- floods the input with
mostly-absent symbols; Type II feedback then drives clauses to require absent
symbols until *every clause stops firing* and the model collapses to constant
output. (2) **Per-round deep supervision** -- applying true-label feedback after
the node-only round 0 -- trains a round that provably lacks the information to
predict the label, corrupting the shared automata. Both are reproduced with
controlled experiments. We then characterise the weight-tied design on controlled
synthetic tasks (1-hop relation solved perfectly; multi-hop underperforms untied
depth) and confirm, on long-range probes, that deep/recurrent TMs do not solve
long-range *relational* reasoning. Real-data benchmarking (TUDataset, LRGB) under
a proper protocol is identified as future work.

## 1. Introduction and lineage

- **Tsetlin Machine (TM).** Granmo 2018, arXiv:1804.01508. Teams of Tsetlin
  automata learn conjunctive clauses over Boolean literals with Type I / Type II
  feedback; clauses vote with learned weights.
- **Graph Tsetlin Machine (GraphTM).** Granmo et al. 2025, arXiv:2507.14874;
  cair/GraphTsetlinMachine. Per-node clause evaluation, OR-across-nodes
  aggregation, and hypervector-bound message passing across typed edges. Depth
  `D` performs `D-1` message rounds, each with a separate message-automata bank.
- **Hierarchical Graph Tsetlin Machine (HGTM).** Stacks independent GraphTM
  layers with an inter-layer encoding kernel and projected feedback; the joint
  `L>=2` path is documented to collapse to the majority class because the
  projected-feedback signal is vacuous at initialisation.
- **RGTM (this work).** Rather than *stacking independent* layers (HGTM) or
  *adding independent* message banks (deeper GraphTM), RGTM *ties* one message
  bank across an arbitrary number of rounds, and trains it with the canonical
  per-sample feedback. The recurrence is in the message channel, so it never
  re-encodes clause firings as node symbols (the failure mode of §4.1).

### 1.1 Contributions

1. **The RGTM architecture**: weight-tied recurrent message passing for the
   Graph Tsetlin Machine; message-passing parameters constant in depth; R=1
   reduces exactly to a single-round GraphTM (oracle-tested).
2. **A mechanistic account of two recurrent/deep-TM failure modes** (node-symbol
   re-encoding poison; deep-supervision corruption), each isolated with
   controlled experiments, directly relevant to anyone extending HGTM or
   building deep TMs.
3. **A relation-detection benchmark** (A-adjacent-to-B) that cleanly separates
   single-pass from message-passing capability, with presence balanced so a
   round-0 classifier is provably at chance.
4. **A characterization of weight tying**: it matches a single-round GraphTM
   exactly and solves 1-hop relations perfectly at constant parameters, but
   underperforms untied per-round banks on multi-hop tasks: a capacity/parameter
   trade-off we quantify rather than a free saving.
5. **A negative result on the open problem**: no deep/recurrent TM variant we
   tried (deep cair GraphTM, HGTM stacking, RGTM tying) solves long-range
   *relational* propagation.

## 2. Method

### 2.1 Forward pass

For an input graph `G = (V, E, X, T_e)` with per-node literal hypervector `X`:

```
A_0 = clauses(node TA) evaluated on X                 # round 0 (node literals)
for r = 1 .. R:
    M   = exchange(A_{r-1}, hypervectors, T_e)         # neighbours' firings, edge-bound
    A_r = clauses(SHARED message TA) evaluated on M    # one tied bank, every round
class_sum[c] = sum_{j,n} weight[c,j] * (A_R[j,n] == 1)
predict argmax_c class_sum[c]
```

The single shared message bank is the only change from cair's depth-`(R+1)`
GraphTM, whose rounds would each use `message_ta_state[r]`. Setting `R=1` uses the
bank exactly once, so RGTM(R=1) == GraphTM(depth=2).

### 2.2 Training (canonical feedback, not deep supervision)

Per sample, after the full `R`-round forward, one feedback is computed from the
final output and distributed:

```
select_clause_node(A_R); select_clause_updates(class_sum, y)   # cair, unchanged
update(node TA, X, ...)                                        # node bank
for r = 1 .. R: update_message(SHARED message TA, M_r, ...)    # tied bank, R updates
```

This is cair's proven credit assignment with the message updates folded onto one
bank. We deliberately do **not** supervise intermediate rounds (see §4.2).

### 2.3 Adaptive depth (halting)

Because the bank is shared, `R` is a free parameter and could in principle be
chosen per graph at inference (e.g., halt when the class-vote margin is confident
or the firing pattern reaches a fixed point). The current implementation uses a
fixed `R`; adaptive halting is left to future work (§8).

## 3. Why the obvious recurrences fail (controlled negatives)

### 3.1 Node-symbol re-encoding poisons the clauses

The natural recurrence re-feeds round `r`'s per-node clause firings as round
`r+1`'s node *symbols* (one symbol `c_j` per clause `j`). With K clauses this adds
K symbols to the node vocabulary, almost all *absent* at any given node. Under the
GraphTM encoding an absent symbol sets its negated literal and clears its positive
literal, so each rarely-present clause-symbol contributes a literal position that
is **0 across (almost) the whole dataset**. Type II feedback suppresses false
positives by *including currently-0 literals*; it therefore greedily includes
these always-0 positions, forcing clauses to require absent symbols. Clauses stop
firing, class sums go to zero, and the model predicts a constant class.

Evidence (RGTM-by-re-encoding, match/equality tasks): firing density collapses to
~0 and train accuracy is pinned at the majority rate, identically under one-hot
and sparse encodings and across max-included-literal caps and seeds. The same
mechanism explains HGTM's collapse: its inter-layer encoding is the same
clause-firings-as-literals construction.

### 3.2 Per-round deep supervision corrupts the shared bank

Applying true-label feedback after round 0 (node literals only) trains a round
that cannot, in principle, separate the classes for any relational task, since round 0
sees no neighbour information. Because the bank is shared, this pushes the
automata toward an impossible target and degrades every later round. Controlled
result: deep-supervised RGTM(R=1) fails the equality task that the canonical
single-feedback RGTM(R=1) solves. We therefore use canonical single-feedback
training (§2.2).

## 4. Experiments

Hardware: NVIDIA Tesla V100-SXM3-32GB, CUDA 12.9, PyCUDA 2026.1. All numbers are
measured; multi-seed mean ± std reported; per-run results in `results/*.jsonl`.

### 4.1 Correctness oracle
RGTM(R=1) reproduces a single-message-round GraphTM: on the 2-node equality task
(which requires comparing the two nodes) it reaches 1.000 on a converging seed,
while RGTM(R=0) (node literals only, no message round) is at chance because it
cannot compare nodes (`scratch/diag12.py`). This confirms the message round and
the shared-bank training loop are wired correctly.

### 4.2 Relation detection: A-adjacent-to-B
Each graph has exactly one A and one B node (presence balanced); label = A
adjacent to B. Round 0 is provably at chance; >=1 message round can detect the
relation. Measured (1000 clauses, 5 seeds, random 80/20, `results/adjacency_5seed.jsonl`):

| rounds | mean acc | std | message-TA banks |
|---:|---:|---:|---:|
| 0 (single pass) | 0.508 | 0.031 | 0 |
| 1 | **1.000** | 0.000 | 1 |
| 2 | **1.000** | 0.000 | 1 |
| 3 | **1.000** | 0.000 | 1 |

A single message round lifts accuracy from chance to a perfect, zero-variance
solution, and the weight-tied bank stays perfect as rounds increase (2, 3) at a
constant one-bank parameter cost. (At 300 clauses the same task is high-variance;
the determinism above requires sufficient clauses, an instance of the general TM
clause-count/variance trade-off.)

### 4.3 Long-range *relational* propagation (negative)
Key/query colour-match along an L-node path (the relation is L-1 hops away); the
QUERY node must learn the KEY node's colour. Clean one-hot (no poison), raw cair
GraphTM, depth swept (`scratch/diag10`):

| task | depth=2 | depth=4 | depth=6 |
|---|---:|---:|---:|
| L=2 match (distance 1) | learns (1.000 on a converging seed) | -- | -- |
| L=4 match (distance 3) | 0.52 (chance) | 0.52 | 0.52 |
| L=6 match (distance 5) | 0.52 | -- | 0.52 |

Adding message-passing depth does **not** let a GraphTM solve the distance-3
relation, even though depth 6 propagates information far enough in principle.
RGTM does not reliably solve it either, and HGTM's stacking collapsed on the
analogous parity task. We report this as a negative result: deep/recurrent TMs
lack a working credit-assignment rule for long-range *relational* propagation,
which bounds the claims of this paper (and motivates §4.2's choice of a 1-hop
relation, which is learnable).

### 4.4 Controlled parameter efficiency (within_distance): the central test

`within_distance(k)`: B is within `k` hops of A (presence balanced); a model needs
`>= k` message rounds to separate the classes. We compare RGTM(R, **one** shared
bank) against cair depth=R+1 (**R** separate banks) at matched clauses. Measured,
5 seeds, k=2, 1000 clauses (`results/depth_efficiency_k2.jsonl`):

| model | reach | message-TA banks | mean acc | std |
|---|---:|---:|---:|---:|
| single-pass (depth 1) | 1 | 0 | 0.498 | 0.024 |
| cair-depth2 | 1 | 1 | 0.584 | 0.092 |
| **cair-depth3 (untied)** | 2 | **2** | **1.000** | 0.000 |
| RGTM-r1 | 1 | 1 | 0.615 | 0.106 |
| **RGTM-r2 (tied)** | 2 | **1** | 0.722 | 0.200 |
| RGTM-r3 (tied) | 3 | 1 | 0.607 | 0.100 |

**Honest reading (negative for the strong claim).** Reach-1 configs are at chance
(k=2 needs 2 hops). cair's *untied* depth-3 solves the task perfectly. RGTM's
*tied* two-round bank only reaches 0.72 ± 0.20 (and r3 is worse): **one shared
bank cannot specialise per hop the way two separate banks can.** So weight tying
does **not** buy parameter-efficient *multi-hop* message passing at this budget;
the tie is "free" only when a single round suffices (§4.2's k=1 relation, solved
perfectly). A follow-up 2000-clause run (doubling RGTM's capacity to equalise
total message params with cair-depth3) did **not** close the gap: with T scaled to
2000 it instead pushed *all* configs, including the cair-depth3 baseline that was
perfect at 1000 clauses, to chance, illustrating TM hyperparameter fragility
rather than a fix. The stable 1000-clause regime is therefore our reference, and
its verdict stands.

### 4.5 Real-data graph classification (a tuning-wall reproduction)

We attempted real molecular graph classification and hit a documented open problem
rather than a clean baseline.

**MUTAG (188 graphs, random 80/20).** With *canonical cair itself* (1000 clauses,
T=1500, s=10, depth=2, 50 ep) we reproduce the variance pathology from the HGTM
repo: single-pass lands on the per-split majority baseline (0.55 / 0.76 / 0.68,
seeds 42/123/456) and deeper models overfit *below* it (0.45 / 0.24 / 0.32); RGTM
behaves identically. With 38 test graphs this is variance-dominated.

**NCI1 (4110 graphs, balanced, 822 test).** To remove the small-sample confound we
moved to NCI1, where 822 test graphs make test accuracy stable, and we measured
the *train* accuracy as the decisive diagnostic. A ~17-config search (clauses
2000-8000, T 1000-8000, s 1.2-10, depth 1-3, 40-200 epochs, one-hot and sparse
node encodings, collision-saturated vs. clean (message_size >> clauses) message
channels, and richer node features, both atom-only and atom+degree) leaves cair
GraphTM **unable to fit the training set**: train accuracy stays at or below the
0.504 majority in every config (best train 0.535). This is the decisive finding:
the failure is not overfitting or variance but an inability to capture the signal
at all. A depth-1 bag-of-atoms cannot fit (NCI1 classes differ by *substructure*,
not atom composition), and depth-2/3 message passing does not train to capture
those substructures, consistent with the deep-message-passing credit-assignment
weakness documented in §4.3-§4.4. The GIN-era ~78% operating point for cair
GraphTM was not reproducible under our search budget; our construction matches the
HGTM repo's cair baseline script, so this is the shared, unsolved tuning problem,
not a build bug.

**LRGB Peptides-func (binarized probe).** Scoped probe (800/400 graphs of ~152
nodes, binarized to the chosen function, 2000 clauses, 20 epochs,
`results/lrgb_probe.jsonl`): single-pass = 0.640 (= the 0.640 majority), one
message round = 0.645: at the majority baseline, no meaningful learning,
consistent with NCI1. A full LRGB evaluation (all graphs, the multilabel AP
metric, proper protocol) requires first escaping the tuning wall above and is left
to future work (§8).

We therefore report neither a real-data win nor loss for RGTM: the *baseline* is
not in a learning regime under our search budget, and locating cair's published
operating point (reported elsewhere around 78% on NCI1) is exactly the open tuning
problem we inherit. The controlled synthetic experiments (§4.2-§4.4) isolate the
RGTM mechanism without this confound. A proper protocol (10-fold CV, larger
clause/epoch budgets, per-dataset tuning) is future work (§8).

## 5. Related work

**Depth in Tsetlin Machines.** The Convolutional TM (Granmo et al.,
arXiv:1905.09688) adds spatial structure but no compositional depth. The
Multi-Layer TM (Tarasyuk et al., ISTM 2024) stacks TMs on tabular data and is
reported to overfit as layers grow (no public code, no graphs). GraphTM ("The
Tsetlin Machine Goes Deep", arXiv:2507.14874) realises depth as `D-1` message
rounds, each with its own automata bank -- the model RGTM ties. HGTM stacks
independent GraphTM layers with projected feedback; its joint multi-layer path
collapses to the majority class. RGTM differs from all of these by sharing one
message bank across rounds (constant parameters) and training with the canonical
single-feedback rule rather than projected/per-round feedback.

**Weight tying across depth.** Reusing one parameter block across an unbounded /
adaptive number of computation steps is the Universal Transformer and
recurrent-depth idea (e.g., latent recurrent-depth reasoning, arXiv:2502.05171).
RGTM is the propositional-logic analogue, with the tied block living in the
message channel. Unlike those neural models, it has no backpropagation-through-time
and therefore no learned temporal credit assignment -- the limitation §4.3
quantifies.

**Theory.** TM convergence is proved only for small operators (AND/OR/XOR; e.g.
Jiao et al., IEEE TPAMI 2023) and the generalized convergence / PCL analysis
(AAAI 2024, arXiv:2310.02005); there is no convergence or capacity result for
deep/recurrent or message-passing TMs. RGTM's negatives are empirical evidence
that this gap is load-bearing in practice, not merely a missing proof.

**Efficiency-oriented logic learning.** Differentiable logic-gate networks
(difflogic, arXiv:2210.08277; conv. extension arXiv:2411.04732) share TM's
efficiency goal but are not faithfully interpretable (millions of gates); RGTM
keeps readable clauses while reducing the parameters depth requires.

## 6. Honest framing and limitations
- Defensible contributions: (i) the RGTM architecture; (ii) a clean
  characterization of *when* weight-tied message passing helps (single round /
  1-hop: perfect at constant parameters) versus when it does not (multi-hop:
  underperforms untied per-round banks); (iii) the mechanistic negatives of §3;
  (iv) the long-range-relational negative of §4.3. Not claimed: parameter-efficient
  *deep* message passing, SOTA accuracy, or a solution to long-range reasoning.
- TM training is high-variance and hyperparameter-fragile (§4.4); conclusions rest
  on the stable regimes and multi-seed means.
- The empirical core is controlled synthetic tasks. Our real-data attempts (MUTAG,
  NCI1, an LRGB Peptides probe) all sit at the majority baseline; on NCI1 a
  ~17-config search shows cair GraphTM cannot even *fit the training set* (a
  representation/mechanism limit, not variance or overfitting); this is the cair
  baseline's own unsolved tuning problem (§4.5), not an RGTM result. A fair
  real-data evaluation is future work (§8).

## 7. Reproducibility
Seeds, configs, and per-run JSONL in `results/`. `pip install -e .` (CUDA device
required for fit/predict). cair GraphTsetlinMachine is the upstream dependency.
Controlled diagnostics behind §3-§4.3 are in `scratch/diag*.py`.

## 8. Future work
- **A credit-assignment rule for multi-hop tied recurrence**, the core open
  problem. The tied bank would need to learn a *self-propagating* relay (a clause
  that fires from its own neighbour-message, so the same rule carries the signal
  one more hop each round); canonical single-feedback training does not find it,
  and there is no backprop-through-time analogue for TMs.
- **Real-data evaluation** under a proper protocol (10-fold CV) on NCI1 / PROTEINS
  and the **Long Range Graph Benchmark** (Peptides-func / -struct), the original
  motivating target, deferred because (a) the controlled experiments already bound
  the claims and (b) real-data TM benchmarking needs careful, separate tuning to
  escape the variance pathology of §4.5.
- **Adaptive halting**: per-graph round count via margin / fixed-point criteria.
- **Stability**: characterise the clause-count / T regime that makes tied
  multi-hop training reliable (§4.4 fragility).
