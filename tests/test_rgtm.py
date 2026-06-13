"""Non-flaky correctness/structure tests for RGTM.

Accuracy is high-variance for TMs, so these assert structure and that the model
runs end-to-end and produces valid predictions -- not specific accuracies. GPU
tests skip gracefully on a CPU-only box.
"""
import numpy as np
import pytest


def _cuda_available():
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as drv
        return drv.Device.count() > 0
    except Exception:
        return False


def test_imports():
    from rgtm.recurrent_tm import RecurrentGraphTM  # noqa: F401
    from rgtm.datasets import adjacent_pair, long_range_match, parity_on_path  # noqa: F401
    from rgtm.graphs_builder import GraphSpec, build_graphs  # noqa: F401


def test_depth_is_rounds_plus_one():
    from rgtm.recurrent_tm import RecurrentGraphTM
    for r in (0, 1, 4):
        m = RecurrentGraphTM(number_of_clauses=10, T=10, s=3.0, rounds=r)
        assert m.depth == r + 1
        assert m.rounds == r


def test_adjacent_pair_balanced_and_shaped():
    from rgtm.datasets import adjacent_pair
    specs, Y, syms = adjacent_pair(400, n_nodes=10, seed=0)
    assert len(specs) == 400 and len(Y) == 400
    assert set(np.unique(Y)).issubset({0, 1})
    assert 0.3 < Y.mean() < 0.7  # roughly balanced by construction
    for sp in specs[:20]:
        n_A = sum("A" in s for s in sp.node_symbols)
        n_B = sum("B" in s for s in sp.node_symbols)
        assert n_A == 1 and n_B == 1  # exactly one A and one B (presence balanced)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA device required")
@pytest.mark.parametrize("rounds", [0, 1, 2])
def test_fit_predict_runs_and_valid_labels(rounds):
    from rgtm.recurrent_tm import RecurrentGraphTM
    from rgtm.graphs_builder import build_graphs
    from rgtm.datasets import adjacent_pair

    _, _, syms = adjacent_pair(1, seed=0)
    sp_tr, Ytr, _ = adjacent_pair(120, n_nodes=8, seed=1)
    sp_te, Yte, _ = adjacent_pair(60, n_nodes=8, seed=2)
    g_tr = build_graphs(sp_tr, syms, None, None, True, True, len(syms), 1)
    g_te = build_graphs(sp_te, syms, g_tr, None, True, True, len(syms), 1)

    m = RecurrentGraphTM(number_of_clauses=50, T=50, s=5.0, rounds=rounds,
                         message_size=64, one_hot_encoding=True)
    m.fit(g_tr, Ytr, epochs=3)
    preds = m.predict(g_te)
    assert preds.shape == (60,)
    assert set(np.unique(preds)).issubset({0, 1})
