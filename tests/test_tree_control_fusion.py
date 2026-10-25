"""Fused GPU control preserves state transitions and exact materialization."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, inputs, profile, snapshot
from test_gpu_tree_deep import document_tree, fit_frontier


def train(enabled, case="regression", *, params=None, rounds=2):
    X, y, metadata, case_params = inputs(case)
    options = dict(max_depth=6, max_bin=32, base_score=0, seed=41,
                   metal_gpu_tree=1, metal_tree_control_fusion=enabled,
                   metal_coalesce_encoders=0)
    options.update(case_params)
    options.update(params or {})
    data = xgb.DMatrix(X, label=y, **metadata)
    model = xgb.Booster(options, [data])
    stats = []
    for iteration in range(rounds):
        custom = (lambda prediction, matrix:
                  ((prediction - matrix.get_label()).astype(np.float32),
                   np.ones_like(prediction, dtype=np.float32))) if case == "custom" else None
        model.update(data, iteration, fobj=custom)
        stats.append(profile(model))
    query_metadata = {key: value for key, value in metadata.items()
                      if key in ("missing", "feature_types", "base_margin")}
    return snapshot(model, [data, xgb.DMatrix(X, **query_metadata)]), stats


def compare(case="regression", *, params=None, expected_active=1, rounds=2):
    expected, control = train(0, case, params=params, rounds=rounds)
    actual, fused = train(1, case, params=params, rounds=rounds)
    assert_identical(expected, actual)
    depth = (params or {}).get("max_depth", 6)
    coalesce = (params or {}).get("metal_coalesce_encoders", 0)
    for old, new in zip(control, fused):
        assert old["tuning"]["tree_control_fusion"] == 0
        assert new["tuning"]["tree_control_fusion"] == expected_active
        assert old["command_buffers"] == new["command_buffers"]
        if expected_active and not coalesce:
            assert old["tuning"]["tree_encoders"] - new["tuning"]["tree_encoders"] == depth
        else:
            assert old["tuning"]["tree_encoders"] == new["tuning"]["tree_encoders"]
    return fused


@pytest.mark.parametrize("depth", [2, 6, 9, 10])
@pytest.mark.parametrize("case", ["regression", "missing", "constant", "empty"])
@pytest.mark.parametrize("parallel", [0, 1])
def test_depth_serial_simd_and_empty_frontiers(depth, case, parallel):
    stats = compare(case, params={"max_depth": depth, "metal_parallel_splits": parallel})
    assert all(item["tuning"]["gpu_tree"] == 1 for item in stats)


@pytest.mark.parametrize("quantile,pruned", list(itertools.product([False, True], repeat=2)))
def test_full_terminal_frontier_and_pruned_parent_compaction(quantile, pruned):
    results, stats = [], []
    for enabled in [0, 1]:
        model, _, result, info = fit_frontier(1, 10, quantile,
            params={"metal_tree_control_fusion": enabled, "metal_coalesce_encoders": 0}, pruned=pruned)
        assert info["tuning"]["tree_control_fusion"] == enabled
        assert info["tuning"]["gpu_tree_levels"] == 10
        tree = document_tree(model)
        left = np.asarray(tree["left_children"])
        np.testing.assert_array_equal(left[:511], 2 * np.arange(511) + 1)
        if pruned:
            assert 256 < np.count_nonzero(left[511:1023] >= 0) < 512
        else:
            assert len(left) == 2047
        results.append(result)
        stats.append(info)
    assert_identical(*results)
    assert stats[0]["tuning"]["tree_encoders"] - stats[1]["tuning"]["tree_encoders"] == 10


@pytest.mark.parametrize("coalesce,gradient_fusion", list(itertools.product([0, 1], repeat=2)))
def test_encoder_and_borrowed_gradient_commands(coalesce, gradient_fusion):
    compare("missing", params={"metal_coalesce_encoders": coalesce,
                              "metal_gradient_root_fusion": gradient_fusion})


@pytest.mark.parametrize("case", ["custom", "weighted", "multiclass", "parallel", "bytree",
                                  "ties", "subnormal_features", "positive_infinity_missing"])
def test_gradient_groups_sampling_rng_and_numeric_edges(case):
    compare(case)


@pytest.mark.parametrize("workspace,persistent", list(itertools.product([0, 1], repeat=2)))
def test_workspace_reset_reuse_and_toggle(workspace, persistent):
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=6, max_bin=32, base_score=0, seed=19,
                                 colsample_bytree=0.7, metal_gpu_tree=1,
                                 metal_tree_workspace=workspace,
                                 metal_tree_persistent_workspace=persistent))
        trace = []
        for iteration, rows in enumerate([513, 65, 1027, 0, 513, 129]):
            X, y, _, _ = inputs(rows=rows)
            data = xgb.DMatrix(X, label=y)
            active = int(enabled and iteration % 3 != 1)
            model.set_param("metal_tree_control_fusion", active)
            model.set_param("max_depth", 10 if iteration == 2 else 6)
            if iteration == 3:
                model.reset()
            if iteration == 4:
                model.load_config(model.save_config())
            model.update(data, iteration)
            assert profile(model)["tuning"]["tree_control_fusion"] == active
            trace.append(snapshot(model, [data]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


@pytest.mark.parametrize("gradient_fusion", [0, 1])
def test_reachable_constructor_error_then_retry(gradient_fusion):
    # Internal node/capacity errors are not reachable through valid public
    # inputs. This exercises the public failure path while a borrowed gradient
    # command may exist, then retries with the same model and RNG state.
    results = []
    for enabled in [0, 1]:
        X, y, _, _ = inputs()
        data = xgb.DMatrix(X, label=y)
        model = xgb.Booster(dict(max_depth=6, max_bin=32, base_score=0, seed=89,
                                 colsample_bytree=0.6, metal_gpu_tree=1,
                                 metal_tree_control_fusion=enabled,
                                 metal_gradient_root_fusion=gradient_fusion), [data])
        for iteration in range(3):
            if iteration == 1:
                model.set_param("metal_histogram_rows", 1)
                with pytest.raises(xgb.core.XGBoostError, match="metal_histogram_rows"):
                    model.update(data, iteration)
                model.set_param("metal_histogram_rows", 2048)
            model.update(data, iteration)
            assert profile(model)["tuning"]["tree_control_fusion"] == enabled
        results.append(snapshot(model, [data]))
    assert_identical(*results)


@pytest.mark.parametrize("case,params", [
    ("regression", {"metal_gpu_tree": 0}),
    ("regression", {"max_depth": 1}),
    ("regression", {"max_depth": 11}),
    ("regression", {"max_leaves": 7}),
    ("regression", {"grow_policy": "lossguide"}),
    ("regression", {"subsample": 0.7}),
    ("regression", {"max_cached_hist_node": 1}),
    ("regression", {"max_bin": 512}),
    ("regression", {"monotone_constraints": "(1,0,-1,0,0,0,0)"}),
    ("vector", {}),
    ("categorical", {}),
])
def test_original_driver_fallback_ignores_control_fusion(case, params):
    compare(case, params=params, expected_active=0)
