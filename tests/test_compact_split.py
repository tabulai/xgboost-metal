"""Packing nonzero bins preserves original split prefixes and candidate order."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from test_fast_scalar_weight import tree
from test_gpu_tree import assert_identical, profile, snapshot
from test_gpu_tree_deep import full_frontier_data
from test_histogram_missing import inputs, layout
from test_split_subtraction_fusion import fit


def compare(X, y, *, params=None, expected_active=1, **kwargs):
    params = {"metal_fast_stats": 1, "metal_split_threads": 32, **(params or {})}
    expected, controls = fit(X, y, 1, params={**params, "metal_compact_splits": 0}, **kwargs)
    actual, stats = fit(X, y, 1, params={**params, "metal_compact_splits": 1}, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["compact_splits"] == 0 for item in controls)
    assert all(item["tuning"]["compact_splits"] == expected_active for item in stats)
    if expected_active:
        assert all(item["tuning"]["split_threads"] == 256 for item in stats)
    return actual, stats


@pytest.mark.parametrize("bins,missing",
                         list(itertools.product([2, 31, 32, 33, 64, 65, 128, 256], [False, True])))
def test_bin_boundaries_dense_and_sparse(bins, missing):
    X, y = inputs(rows=1025, density=0.3)
    X = layout(X, "csr") if missing else np.nan_to_num(X)
    compare(X, y, bins=bins, rounds=3)


@pytest.mark.parametrize("stats,threads", list(itertools.product([0, 1], [32, 256])))
def test_statistics_variants_and_requested_thread_width(stats, threads):
    X, y = inputs(rows=1025, density=0.1)
    compare(X, y, bins=256, rounds=3,
            params={"metal_fast_stats": stats, "metal_split_threads": threads})


@pytest.mark.parametrize("default_left,stats", list(itertools.product([0, 1], [0, 1])))
def test_nonroot_missing_direction_ties_keep_original_threshold(default_left, stats):
    reference_values = np.column_stack([np.arange(128) % 2, np.arange(128)]).astype(np.float32)
    reference = xgb.QuantileDMatrix(reference_values, max_bin=128)
    values = np.repeat(np.asarray([16, 100, np.nan], np.float32), 64)
    X = np.column_stack([np.repeat([0, 1], len(values)), np.tile(values, 2)]).astype(np.float32)
    target = np.repeat(np.asarray([-1, 1, -1 if default_left else 1], np.float32), 64)
    y = np.tile(target, 2) + np.where(X[:, 0] == 0, -10, 10).astype(np.float32)
    actual, _ = compare(X, y, bins=128, quantile=True, metadata={"ref": reference}, rounds=1,
                        params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0,
                                "metal_fast_stats": stats})
    document = tree(actual)
    assert len(document["base_weights"]) == 7
    assert document["split_indices"][:3] == [0, 1, 1]
    assert document["default_left"][1:3] == [default_left, default_left]


@pytest.mark.parametrize("boundary", [31, 32, 33])
def test_compacted_lane_boundary_retains_gap_threshold(boundary):
    reference_values = np.column_stack([np.arange(256) % 2, np.arange(256)]).astype(np.float32)
    reference = xgb.QuantileDMatrix(reference_values, max_bin=256)
    values = np.arange(64, dtype=np.float32) * 4
    X = np.column_stack([np.repeat([0, 1], len(values)), np.tile(values, 2)]).astype(np.float32)
    target = np.where(np.arange(64) < boundary, -1, 1).astype(np.float32)
    y = np.tile(target, 2) + np.where(X[:, 0] == 0, -10, 10).astype(np.float32)
    actual, _ = compare(X, y, bins=256, quantile=True, metadata={"ref": reference}, rounds=1,
                        params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0})
    document = tree(actual)
    assert len(document["base_weights"]) == 7
    assert document["split_indices"][:3] == [0, 1, 1]
    assert all(values[boundary - 1] < threshold < values[boundary]
               for threshold in document["split_conditions"][1:3])


@pytest.mark.parametrize("zero_hessian", [False, True])
def test_signed_cancellation_keeps_nonzero_hessian_candidates(zero_hessian):
    X, y = inputs(rows=1025, density=0.3)
    X[:, 1] = X[:, 0]
    gradient = np.resize(np.asarray([2**40, -2**40, 0.25, -0.25, 0], np.float32), len(X))
    hessian = np.ones_like(gradient)
    if zero_hessian:
        hessian[np.arange(len(X)) % 5 < 2] = 0
    compare(X, y, bins=256, rounds=3, params={"min_child_weight": 0},
            custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("exponent", [-75, 62])
def test_nonroot_scalar_rounding_boundaries(exponent):
    outer = 15 if exponent == -75 else 3
    hessian_exponent = 125 if exponent == -75 else -20
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)
    gradient = np.asarray([-1, -outer, 1, outer], np.float32) * np.float32(2.0**(exponent + hessian_exponent))
    hessian = np.full(4, 2.0**hessian_exponent, np.float32)
    actual, stats = compare(X, np.zeros(4, np.float32), bins=4, quantile=True, rounds=1,
                            params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0, "eta": 1},
                            custom=lambda prediction, matrix: (gradient, hessian))
    assert len(tree(actual)["base_weights"]) == 7
    assert stats[-1]["tuning"]["gpu_tree_levels"] >= 2


@pytest.mark.parametrize("case", ["empty", "all_missing", "constant", "zero_derivatives", "zero_weight"])
def test_empty_and_invalid_compacted_frontiers_are_initialized(case):
    X, y = inputs(density=0.1)
    custom, metadata = None, {}
    if case == "empty":
        X, y = X[:0], y[:0]
    elif case == "all_missing":
        X.fill(np.nan)
    elif case == "constant":
        y.fill(0)
    elif case == "zero_derivatives":
        zeros = np.zeros_like(y)
        custom = lambda prediction, matrix: (zeros, zeros)
    else:
        metadata["weight"] = np.zeros_like(y)
    actual, _ = compare(X, y, bins=256, rounds=2, custom=custom, metadata=metadata)
    assert len(tree(actual)["base_weights"]) == 1


@pytest.mark.parametrize("case", ["cpu_driver", "serial", "subtraction_control", "original_gain", "fast_weight",
                                  "skip_control", "depth_one", "large_bins", "categorical", "monotone"])
def test_ineligible_paths_keep_compaction_disabled(case):
    X, y = inputs(density=0.8)
    params, metadata, bins = {}, {}, 128
    if case == "cpu_driver":
        params["metal_gpu_tree"] = 0
    elif case == "serial":
        params["metal_parallel_splits"] = 0
    elif case == "subtraction_control":
        params["metal_fused_subtraction"] = 0
    elif case == "original_gain":
        params["metal_fast_gain"] = 0
    elif case == "fast_weight":
        params["metal_fast_weight"] = 1
    elif case == "skip_control":
        params["metal_skip_empty_bins"] = 0
    elif case == "depth_one":
        params["max_depth"] = 1
    elif case == "large_bins":
        bins = 257
    elif case == "categorical":
        present = ~np.isnan(X[:, 0])
        X[present, 0] = np.arange(np.count_nonzero(present)) % 5
        metadata["feature_types"] = ["c"] + ["q"] * 6
    else:
        params["monotone_constraints"] = "(1,0,0,0,0,0,0)"
    compare(X, y, bins=bins, params=params, metadata=metadata, expected_active=0, rounds=2)


@pytest.mark.parametrize("pruned", [False, True])
def test_full_and_compacted_depth_ten_frontiers(pruned):
    X, y = full_frontier_data(pruned)
    actual, stats = compare(X, y, bins=2, quantile=True, rounds=1,
                            params={"max_depth": 10, "reg_lambda": 0, "min_child_weight": 1})
    assert stats[-1]["tuning"]["gpu_tree_levels"] == 10
    assert len(tree(actual)["base_weights"]) > 1023


def test_masked_features_keep_initialized_candidate_statistics():
    X, y = inputs(density=0.5)
    compare(X, y, bins=256, params={"colsample_bytree": 0.125, "seed_per_iteration": True},
            metadata={"feature_weights": np.asarray([1, 2, 3, 4, 5, 6, 7], np.float32)}, rounds=3)


def test_workspace_growth_reset_and_toggle():
    X, y = inputs(rows=1025, density=0.5)
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=5, max_bin=256, base_score=0, seed=72, metal_fast_stats=1))
        trace = []
        for iteration, rows in enumerate([257, 0, 1025, 513]):
            selected = enabled if iteration != 2 else 0
            model.set_param("metal_compact_splits", selected)
            if iteration == 1:
                model.reset()
            data = xgb.DMatrix(X[:rows], label=y[:rows])
            model.update(data, iteration)
            assert profile(model)["tuning"]["compact_splits"] == selected
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


@pytest.mark.parametrize("missing,depth,disabled", list(itertools.product([False, True], [7, 8], [False, True])))
def test_adaptive_defaults_and_explicit_compaction_control(missing, depth, disabled):
    X, y = inputs(rows=257, density=0.5)
    if not missing:
        X = np.nan_to_num(X)
    params = {"max_depth": depth}
    if disabled:
        params["metal_compact_splits"] = 0
    compact = int(not disabled and (missing or depth >= 8))
    threads = 256
    group = 2 if missing else 1
    expected_tuning = {"fast_stats": 1, "compact_splits": compact,
                       "split_threads": 256 if compact else threads, "parallel_root": 1,
                       "histogram_present": 0, "features_per_group": group}
    explicit = {**params, "metal_fast_stats": 1, "metal_compact_splits": compact,
                "metal_split_threads": threads, "metal_parallel_root": 1,
                "metal_histogram_present": 0, "metal_features_per_group": group}
    expected, _ = fit(X, y, 1, bins=256, rounds=2, params=explicit)
    actual, profiles = fit(X, y, 1, bins=256, rounds=2, params=params)
    assert_identical(expected, actual)
    for item in profiles:
        assert {key: item["tuning"][key] for key in expected_tuning} == expected_tuning


def test_long_sequence_preserves_models_predictions_and_rng():
    X, y = inputs(rows=513, density=0.8)
    y += np.random.default_rng(627).normal(scale=0.25, size=len(y)).astype(np.float32)
    compare(X, y, bins=128, rounds=100, params={"max_depth": 6, "learning_rate": 0.01})
