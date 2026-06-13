"""Recurrent Graph Tsetlin Machine (RGTM).

Weight-tied recurrent message passing for the Graph Tsetlin Machine: one shared
message-automata bank is reused across R message-passing rounds (parameters
independent of depth), trained with cair's canonical single-feedback rule.
R = 1 reproduces a single-message-round GraphTM exactly.

The working model is ``RecurrentGraphTM`` (rgtm.recurrent_tm). The earlier
node-symbol re-encoding variant in ``rgtm.model`` is kept only as the
negative-result reference for PAPER.md section 3 (it poisons clauses via Type II
feedback and is not the model used in the experiments).
"""
from .recurrent_tm import RecurrentGraphTM
from .graphs_builder import GraphSpec

__all__ = ["RecurrentGraphTM", "GraphSpec"]
__version__ = "0.1.0"
