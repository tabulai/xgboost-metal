"""Depth-nine/ten GPU growth preserves the CPU driver's batch boundary."""
import itertools
import json

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, assert_path, profile, snapshot


def full_frontier_data(pruned=False):
    # Every binary combination is present. The ordered target makes all ten
    # bits useful in descending significance, producing a balanced full tree.
    values = np.arange(1024, dtype=np.uint32)
    X = ((values[:, None] >> np.arange(10, dtype=np.uint32)) & 1).astype(np.float32)
    y = values.astype(np.float32)
    if pruned:
        # Leave holes in the final accepted-parent sequence while keeping all
        # 512 final candidates. Both sides of the CPU's 256-parent boundary
        # contain accepted and rejected nodes.
        constant_pairs = (values // 2) % 3 == 0
        y[constant_pairs] = (values[constant_pairs] // 2 * 2).astype(np.float32)
    return X, y


def document_tree(model, index=0):
    return json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"][index]


def fit_frontier(enabled, depth, quantile=False, params=None, pruned=False, target_scale=1.0):
    X, y = full_frontier_data(pruned)
    y *= np.float32(target_scale)
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = constructor(X, label=y, **({"max_bin": 2} if quantile else {}))
    options = dict(max_depth=depth, max_bin=2, min_child_weight=1, reg_lambda=0,
                   base_score=0, learning_rate=0.3, seed=71, metal_gpu_tree=enabled)
    options.update(params or {})
    model = xgb.Booster(options, [data])
    model.update(data, 0)
    stats = profile(model)
    return model, data, snapshot(model, [data, xgb.DMatrix(X)]), stats


@pytest.mark.parametrize("depth,quantile,parallel_splits",
                         list(itertools.product([9, 10], [False, True], [0, 1])))
def test_full_frontiers_cross_the_terminal_cpu_batch_boundary(depth, quantile, parallel_splits):
    results, commands = [], []
    for enabled in [0, 1]:
        model, _, result, stats = fit_frontier(
            enabled, depth, quantile, {"metal_parallel_splits": parallel_splits})
        assert_path([stats], enabled)
        tree = document_tree(model)
        left = np.asarray(tree["left_children"])
        right = np.asarray(tree["right_children"])
        internal = 2**depth - 1
        assert len(left) == 2 ** (depth + 1) - 1
        np.testing.assert_array_equal(left[:internal], 2 * np.arange(internal) + 1)
        np.testing.assert_array_equal(right[:internal], 2 * np.arange(internal) + 2)
        assert np.all(left[internal:] == -1) and np.all(right[internal:] == -1)
        if enabled:
            assert stats["tuning"]["gpu_tree_levels"] == depth
        results.append(result)
        commands.append(stats["command_buffers"])
    assert_identical(*results)
    assert commands[1] < commands[0]


@pytest.mark.parametrize("quantile", [False, True])
def test_pruned_final_frontier_preserves_parent_order_across_batches(quantile):
    results = []
    for enabled in [0, 1]:
        model, _, result, stats = fit_frontier(enabled, 10, quantile, pruned=True)
        assert_path([stats], enabled)
        tree = document_tree(model)
        left = np.asarray(tree["left_children"])
        # The first nine levels still create a complete 512-node frontier.
        np.testing.assert_array_equal(left[:511], 2 * np.arange(511) + 1)
        expanded = left[511:1023] >= 0
        assert 256 < np.count_nonzero(expanded) < 512
        assert np.any(expanded[:256]) and np.any(~expanded[:256])
        assert np.any(expanded[256:]) and np.any(~expanded[256:])
        assert len(left) == 1023 + 2 * np.count_nonzero(expanded)
        results.append(result)
    assert_identical(*results)


@pytest.mark.parametrize("depth", [9, 10])
def test_deep_leaf_materialization_retains_subnormal_scaling(depth):
    results = []
    for enabled in [0, 1]:
        model, _, result, stats = fit_frontier(
            enabled, depth, params={"learning_rate": float(np.finfo(np.float32).tiny)},
            target_scale=1 / 128)
        assert_path([stats], enabled)
        tree = document_tree(model)
        leaves = np.asarray(tree["split_conditions"], np.float32)[np.asarray(tree["left_children"]) < 0]
        assert np.any((np.abs(leaves) > 0) & (np.abs(leaves) < np.finfo(np.float32).tiny))
        results.append(result)
    assert_identical(*results)


@pytest.mark.parametrize("depth,cache_limit,expected_path", [(9, 510, 0), (9, 511, 1),
                                                           (10, 1022, 0), (10, 1023, 1)])
def test_deep_cache_limit_fallback_keeps_original_histogram_semantics(depth, cache_limit, expected_path):
    # A constant target makes the actual tree tiny while testing the complete
    # speculative-cache bound, without rebuilding many evicted histograms.
    X, _ = full_frontier_data()
    results = []
    for enabled in [0, 1]:
        data = xgb.DMatrix(X, label=np.zeros(len(X), np.float32))
        model = xgb.Booster(dict(max_depth=depth, max_bin=2, base_score=0,
                                 max_cached_hist_node=cache_limit, metal_gpu_tree=enabled), [data])
        model.update(data, 0)
        assert_path([profile(model)], expected_path if enabled else 0)
        assert len(document_tree(model)["left_children"]) == 1
        results.append(snapshot(model, [data]))
    assert_identical(*results)


@pytest.mark.parametrize("depth,features", [(9, 128), (10, 64)])
def test_deep_speculative_memory_still_respects_the_budget(depth, features):
    # Histogram banks alone exceed 256 MiB at these shapes with 256 bins.
    X = np.broadcast_to(np.arange(256, dtype=np.float32)[:, None], (256, features)).copy()
    results = []
    for enabled in [0, 1]:
        data = xgb.DMatrix(X, label=np.zeros(len(X), np.float32))
        model = xgb.Booster(dict(max_depth=depth, max_bin=256, base_score=0,
                                 metal_gpu_tree=enabled), [data])
        model.update(data, 0)
        assert_path([profile(model)], 0)
        assert len(document_tree(model)["left_children"]) == 1
        results.append(snapshot(model, [data]))
    assert_identical(*results)


@pytest.mark.parametrize("workspace,persistent", list(itertools.product([0, 1], repeat=2)))
def test_deep_workspace_growth_reset_and_driver_switches(workspace, persistent):
    traces = []
    X, y = full_frontier_data()
    for use_gpu in [False, True]:
        model = xgb.Booster(dict(max_bin=2, min_child_weight=1, reg_lambda=0, base_score=0,
                                 seed=17, metal_tree_workspace=workspace,
                                 metal_tree_persistent_workspace=persistent))
        trace = []
        for iteration, (depth, rows, reset) in enumerate([(9, 1024, False), (10, 1024, False),
                                                         (10, 0, True), (9, 256, False),
                                                         (10, 1024, False)]):
            enabled = int(use_gpu and iteration != 3)
            model.set_param({"max_depth": depth, "metal_gpu_tree": enabled})
            if reset:
                model.reset()
            data = xgb.DMatrix(X[:rows], label=y[:rows])
            model.update(data, iteration)
            # GPU timings accumulate across updates, including before a driver
            # switch. The tuning value records the path for this update.
            assert profile(model)["tuning"]["gpu_tree"] == enabled
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)
