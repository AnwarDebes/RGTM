"""Recurrent Graph Tsetlin Machine = weight-tied deep message passing.

A cair GraphTM runs `depth-1` rounds of message passing, each with its OWN
message-TA bank, so parameters grow linearly with depth. RGTM ties those banks:
a SINGLE shared message-TA is reused across all R = depth-1 rounds, so the
message-TA parameter count is constant regardless of R. The recurrence can then
be made arbitrarily / adaptively deep at fixed parameter cost.

Training reuses cair's exact, proven credit assignment (one feedback per sample
computed from the final output, distributed to the node TA via `update` and to
the shared message TA via `update_message`, once per round). This is the key
correction over an earlier deep-supervision attempt, which trained the node-only
round 0 toward the final label -- an impossible target that corrupts the shared
automata (see scratch/diag12).

R = 1 is identical to cair depth=2 (a single message round; the tie is a no-op),
which is the correctness oracle. R >= 2 exercises the weight tying.
"""
from __future__ import annotations

import numpy as np
import pycuda.driver as cuda

import GraphTsetlinMachine.tm as cairtm
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine


class RecurrentGraphTM(MultiClassGraphTsetlinMachine):
    def __init__(self, number_of_clauses, T, s, rounds=3, **kw):
        kw["depth"] = int(rounds) + 1  # round 0 (node) + `rounds` message rounds
        super().__init__(number_of_clauses, T, s, **kw)
        self.rounds = int(rounds)

    # cair's _evaluate, with the message-TA bank tied to slot 0 across rounds.
    def _evaluate(self, graphs, number_of_graph_nodes, node_index, edge_index,
                  current_clause_node_output, next_clause_node_output, node_type,
                  number_of_graph_node_edges, edge, clause_X_int, clause_X,
                  encoded_X):
        class_sum = np.zeros(self.number_of_outputs).astype(np.int32)
        cuda.memcpy_htod(self.class_sum_gpu, class_sum)

        self.calculate_messages.prepared_call(
            self.grid, self.block, self.ta_state_gpu, node_type,
            np.int32(graphs.number_of_node_types()),
            np.int32(number_of_graph_nodes), np.int32(node_index),
            current_clause_node_output, self.number_of_include_actions, encoded_X)
        cuda.Context.synchronize()

        for depth in range(self.depth - 1):
            self.prepare_messages.prepared_call(
                self.grid, self.block, number_of_graph_nodes, clause_X_int)
            cuda.Context.synchronize()
            self.exchange_messages.prepared_call(
                self.grid, self.block, number_of_graph_nodes,
                self.hypervectors_gpu, current_clause_node_output,
                np.int32(node_index), np.int32(edge_index),
                number_of_graph_node_edges, edge, clause_X_int)
            cuda.Context.synchronize()
            self.encode_messages.prepared_call(
                self.grid, self.block, number_of_graph_nodes, clause_X_int,
                clause_X[depth])
            cuda.Context.synchronize()
            self.calculate_messages_conditional.prepared_call(
                self.grid, self.block, self.message_ta_state_gpu[0],  # TIED
                node_type, np.int32(graphs.number_of_node_types()),
                np.int32(number_of_graph_nodes), np.int32(node_index),
                current_clause_node_output, next_clause_node_output,
                self.number_of_include_actions, clause_X[depth])
            cuda.Context.synchronize()
            current_clause_node_output, next_clause_node_output = (
                next_clause_node_output, current_clause_node_output)

        self.evaluate.prepared_call(
            self.grid, self.block, current_clause_node_output,
            self.clause_weights_gpu, np.int32(number_of_graph_nodes),
            self.class_sum_gpu)
        cuda.Context.synchronize()
        return current_clause_node_output

    # cair's _fit, with every message round updating the single shared TA (slot 0).
    def _fit(self, graphs, encoded_Y, epochs=100, incremental=False):
        self._init_fit(graphs, encoded_Y, incremental)
        for _epoch in range(epochs):
            for e in range(graphs.number_of_graphs):
                current_clause_node_output = self._evaluate(
                    graphs, np.int32(graphs.number_of_graph_nodes[e]),
                    np.int32(graphs.node_index[e]),
                    np.int32(graphs.edge_index[graphs.node_index[e]]),
                    self.current_clause_node_output_train_gpu,
                    self.next_clause_node_output_train_gpu,
                    self.node_type_train_gpu,
                    self.number_of_graph_node_edges_train_gpu,
                    self.edge_train_gpu, self.clause_X_int_train_gpu,
                    self.clause_X_train_gpu, self.encoded_X_train_gpu)

                self.select_clause_node.prepared_call(
                    self.grid, self.block, cairtm.g.state,
                    current_clause_node_output,
                    int(graphs.number_of_graph_nodes[e]), self.clause_node_gpu)
                cuda.Context.synchronize()

                self.select_clause_updates.prepared_call(
                    self.grid, self.block, cairtm.g.state, self.clause_weights_gpu,
                    self.class_sum_gpu, self.encoded_Y_gpu, np.int32(e),
                    self.clause_node_gpu, self.class_clause_update_gpu)
                cuda.Context.synchronize()

                self.update.prepared_call(
                    self.grid, self.block, cairtm.g.state, self.s[0],
                    self.ta_state_gpu, np.int32(graphs.number_of_graph_nodes[e]),
                    np.int32(graphs.node_index[e]), self.clause_node_gpu,
                    self.number_of_include_actions, self.encoded_X_train_gpu,
                    self.class_clause_update_gpu)
                cuda.Context.synchronize()

                for depth in range(self.depth - 1):
                    self.update_message.prepared_call(
                        self.grid, self.block, cairtm.g.state, self.s[1],
                        self.message_ta_state_gpu[0],  # TIED
                        np.int32(graphs.number_of_graph_nodes[e]),
                        self.clause_node_gpu, self.number_of_include_actions,
                        self.clause_X_train_gpu[depth],
                        self.class_clause_update_gpu)
                    cuda.Context.synchronize()
        self.ta_state = np.array([])
        self.clause_weights = np.array([])
