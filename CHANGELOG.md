# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-13

### Added
- `RecurrentGraphTM`: weight-tied recurrent message passing on top of the
  canonical cair GraphTsetlinMachine, with one shared message-automata bank
  reused across `R` rounds (message-passing parameter count constant in depth).
- Correctness oracle: RGTM(R=1) reproduces a single-message-round GraphTM
  (depth=2), verified on the 2-node equality task.
- `GraphSpec` builder for constructing cair `Graphs` objects.
- Synthetic dataset probes: `adjacent_pair` (relation detection),
  `long_range_match`, `parity_on_path`, `within_distance`, plus a TUDataset
  loader (MUTAG/NCI1).
- Experiment scripts (`experiments/`) with per-run JSONL output in `results/`:
  adjacency, depth efficiency, long-range match, TUDataset, LRGB probe.
- Controlled diagnostics (`scratch/diag*.py`) reproducing the two recurrent-TM
  failure modes documented in PAPER.md: node-symbol re-encoding poison and
  per-round deep-supervision corruption.
- Working paper draft (PAPER.md) with the full characterization: 1-hop
  relation solved perfectly at constant parameters; multi-hop underperforms
  untied per-round banks; long-range relational propagation unsolved by any
  deep/recurrent TM variant tried.
- Test suite (6 tests).
