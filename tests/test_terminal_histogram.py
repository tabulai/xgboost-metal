"""Terminal histogram stores can be omitted without changing split arithmetic."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, profile, snapshot
from test_gpu_tree_deep import fit_frontier, full_frontier_data
from test_histogram_missing import inputs
from test_split_subtraction_fusion import fit


@pytest.mark.parametrize("compact", [0, 1])
def test_default_matches_explicit_terminal_controls(compact):
    X, y = inputs(rows=1025, density=0.65)
    results = []
    for enabled in [0, 1, None]:
        options = {"max_depth": 6, "metal_compact_splits": compact}
        if enabled is not None:
            options["metal_terminal_histogram_elision"] = enabled
        result, traces = fit(X, y, 1, bins=256, rounds=2, params=options)
        for trace in traces:
            tuning = trace["tuning"]
            assert tuning["gpu_tree"] == 1
            assert tuning["gpu_tree_levels"] == 6
            assert tuning["compact_splits"] == compact
            assert tuning["terminal_histogram_elision"] == int(enabled != 0)
        results.append(result)
    assert_identical(results[0], results[1])
    assert_identical(results[0], results[2])


@pytest.mark.parametrize("depth,compact,missing,fast_stats", list(itertools.product(
    [6, 8, 10], [0, 1], [False, True], [0, 1])))
def test_terminal_parallel_and_compact_paths_are_exact(depth, compact, missing, fast_stats):
    X, y = inputs(rows=1025, density=0.65)
    if not missing:
        X = np.nan_to_num(X)
    results = []
    for enabled in [0, 1]:
        result, traces = fit(X, y, 1, bins=256, rounds=2, params={
            "max_depth": depth, "metal_compact_splits": compact,
            "metal_fast_stats": fast_stats, "metal_terminal_histogram_elision": enabled})
        assert all(t["tuning"]["gpu_tree"] == 1 for t in traces)
        assert all(t["tuning"]["terminal_histogram_elision"] == enabled for t in traces)
        results.append(result)
    assert_identical(*results)


@pytest.mark.parametrize("pruned,compact", list(itertools.product([False, True], [0, 1])))
def test_full_and_pruned_final_frontiers_preserve_node_order(pruned, compact):
    results = []
    for enabled in [0, 1]:
        _, _, result, stats = fit_frontier(1, 10, pruned=pruned, params={
            "metal_compact_splits": compact, "metal_terminal_histogram_elision": enabled})
        assert stats["tuning"]["gpu_tree_levels"] == 10
        assert stats["tuning"]["terminal_histogram_elision"] == enabled
        results.append(result)
    assert_identical(*results)


@pytest.mark.parametrize("params", [
    {"metal_gpu_tree": 0}, {"metal_fused_subtraction": 0},
    {"metal_parallel_splits": 0}, {"metal_fast_weight": 1}, {"metal_fast_gain": 0},
    {"metal_compact_splits": 0, "metal_split_threads": 32},
])
def test_unsupported_paths_do_not_elide_histograms(params):
    X, y = inputs(rows=513, density=0.7)
    results = []
    for enabled in [0, 1]:
        result, traces = fit(X, y, 1, bins=128, rounds=2, params={
            **params, "metal_terminal_histogram_elision": enabled})
        assert all(t["tuning"]["terminal_histogram_elision"] == 0 for t in traces)
        results.append(result)
    assert_identical(*results)


@pytest.mark.parametrize("params", [
    pytest.param({"metal_skip_empty_bins": 0, "metal_fast_stats": 0}, id="parallel-plain"),
    pytest.param({"metal_skip_empty_bins": 0, "metal_fast_stats": 1}, id="parallel-stats"),
    pytest.param({"metal_compact_splits": 1, "metal_fused_score": 0}, id="compact-stats-unfused-score"),
    pytest.param({"metal_histogram_present": 1}, id="present-only-parallel"),
    pytest.param({"metal_histogram_present": 1, "metal_compact_splits": 1}, id="present-only-compact"),
    pytest.param({"metal_histogram_clear_built": 0, "metal_compact_splits": 1}, id="full-bank-clear"),
    pytest.param({"metal_tree_control_fusion": 0}, id="separate-control"),
    pytest.param({"metal_coalesce_encoders": 1, "metal_skip_empty_bins": 0,
                  "metal_histogram_clear_built": 0}, id="coalesced-full-clear"),
])
def test_terminal_kernel_and_histogram_variants(params):
    X, y = inputs(rows=1025, density=0.65)
    options = {"max_depth": 6, "metal_compact_splits": 0, "metal_fast_stats": 1, **params}
    results = []
    for enabled in [0, 1]:
        result, traces = fit(X, y, 1, bins=256, rounds=2, params={
            **options, "metal_terminal_histogram_elision": enabled})
        for trace in traces:
            tuning = trace["tuning"]
            assert tuning["gpu_tree"] == 1
            # A selected option with an empty final frontier would not test
            # these specialized kernels. Require actual terminal-depth work.
            assert tuning["gpu_tree_levels"] == options["max_depth"]
            assert tuning["terminal_histogram_elision"] == enabled
            assert tuning["compact_splits"] == options["metal_compact_splits"]
            assert tuning["fast_stats"] == options["metal_fast_stats"]
            assert tuning["histogram_present"] == options.get("metal_histogram_present", 0)
            assert tuning["histogram_clear_built"] == options.get("metal_histogram_clear_built", 1)
            assert tuning["tree_control_fusion"] == options.get("metal_tree_control_fusion", 1)
            assert tuning["fused_score"] == int(bool(options["metal_compact_splits"] and
                options["metal_fast_stats"] and options.get("metal_fused_score", 1)))
        results.append(result)
    assert_identical(*results)


@pytest.mark.parametrize("workspace,persistent", list(itertools.product([0, 1], repeat=2)))
def test_bank_reuse_reset_and_runtime_switches_preserve_outputs(workspace, persistent):
    X, y = full_frontier_data()
    runs = []
    for experiment in [0, 1]:
        model = xgb.Booster({"max_bin": 2, "reg_lambda": 0, "base_score": 0, "seed": 23,
                             "metal_tree_workspace": workspace, "metal_tree_persistent_workspace": persistent})
        snapshots = []
        for iteration, (depth, rows, reset) in enumerate([
                (10, 1024, False), (6, 1024, False), (8, 0, True),
                (2, 1024, False), (10, 1024, False)]):
            enabled = experiment if iteration != 1 else 0
            model.set_param({"max_depth": depth, "metal_terminal_histogram_elision": enabled})
            if reset:
                model.reset()
            data = xgb.DMatrix(X[:rows], label=y[:rows])
            model.update(data, iteration)
            assert profile(model)["tuning"]["terminal_histogram_elision"] == enabled
            snapshots.append(snapshot(model, [data, xgb.DMatrix(X)]))
        runs.append(snapshots)
    for expected, actual in zip(*runs):
        assert_identical(expected, actual)
