<div align="center">

<img src="assets/banner.svg" alt="RGTM: Recurrent Graph Tsetlin Machine" width="100%"/>

# RGTM: Recurrent Graph Tsetlin Machine

[![Release](https://img.shields.io/github/v/release/AnwarDebes/RGTM)](https://github.com/AnwarDebes/RGTM/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-6%20passing-brightgreen)](tests/test_rgtm.py)
[![Paper](https://img.shields.io/badge/paper-working%20draft-orange)](PAPER.md)

</div>

**Weight-tied recurrent message passing for the Graph Tsetlin Machine.** One
shared message-automata bank is reused across `R` message-passing rounds, so the
message-passing parameter count is independent of depth. Built on the canonical
`cair/GraphTsetlinMachine`. University of Agder (UiA).

## Honest status (alpha)

```
+-------------------------------------------------------------------+
| Implementation: complete; RGTM(R=1) == cair GraphTM(depth=2)      |
|                 (correctness oracle: learns 2-node equality;      |
|                  R=0 is at chance, as it cannot compare nodes).    |
| Training:       canonical cair single-feedback per sample, with   |
|                 the message updates folded onto ONE shared bank.   |
| Negative results (reproduced, see PAPER.md §3 and scratch/diag*): |
|   * node-symbol re-encoding recurrence -> Type II "poison"        |
|     literals -> clauses die -> constant output. (Same mechanism   |
|     behind HGTM's encode-as-literals collapse.)                   |
|   * per-round deep supervision -> corrupts the shared bank.       |
| Relation-detection (A-adjacent-to-B): message round adds real     |
|   capability over single-pass. Multi-seed numbers in results/.    |
| Long-range *relational* propagation: NOT solved by RGTM, and not  |
|   by deep cair GraphTM either (reported as a negative result).    |
| Hardware: validated on Tesla V100 (CUDA 12.9, PyCUDA 2026.1).     |
+-------------------------------------------------------------------+
```

## What RGTM changes vs cair GraphTM

```
cair GraphTM, depth D:   round 0 (node TA) + (D-1) rounds, each its OWN message TA
                         => message-TA parameters grow linearly with depth

RGTM, rounds R:          round 0 (node TA) + R rounds, ALL sharing ONE message TA
                         => message-TA parameters constant in R; R is free/adaptive
```

`R = 1` uses the shared bank exactly once and is therefore identical to a
single-message-round GraphTM, which serves as the correctness oracle.

## Install

```bash
pip install -e .                 # needs the cair GraphTsetlinMachine package
# CUDA-capable GPU required for fit/predict; torch_geometric only for TUDataset/LRGB loaders
```

## API

```python
from rgtm.recurrent_tm import RecurrentGraphTM

model = RecurrentGraphTM(
    number_of_clauses=1000, T=1500, s=10.0,
    rounds=4,                 # R message rounds, ONE shared message-TA bank
    message_size=64, one_hot_encoding=True,
)
model.fit(graphs, Y, epochs=50)   # graphs: a cair Graphs object
preds = model.predict(graphs)
```

Datasets: `rgtm.datasets.adjacent_pair` (relation detection),
`rgtm.datasets.long_range_match` / `parity_on_path` (hard propagation probes),
`rgtm.datasets.tudataset` (MUTAG/NCI1 → cair Graphs).

## Experiments

```bash
python experiments/run_adjacency.py    --clauses 1000 --rounds 0 1 2 3 --seeds 41 42 43 44 45
python experiments/run_tudataset.py    --dataset MUTAG --seeds 42 123 456 789 1337
```

## Layout

```
rgtm/
  recurrent_tm.py    RecurrentGraphTM (the model)
  graphs_builder.py  GraphSpec + cair Graphs construction
  datasets/          synthetic probes + TUDataset loader
experiments/         runnable studies, write results/*.jsonl
scratch/             controlled diagnostics behind the §3 negative results
PAPER.md             working paper draft
```

## License
MIT. Builds on cair/GraphTsetlinMachine (MIT).
