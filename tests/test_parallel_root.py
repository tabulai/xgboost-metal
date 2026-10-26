"""Parallel root scoring preserves stored missing totals and ordered root statistics."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from test_fast_scalar_weight import tree
from test_gpu_tree import assert_identical, profile, snapshot
from test_histogram_missing import inputs, layout
from test_split_subtraction_fusion import fit


def compare(X, y, *, params=None, expected_active=1, **kwargs):
    options = {"metal_fast_stats": 1, **(params or {})}
    expected, controls = fit(X, y, 1, params={**options, "metal_parallel_root": 0}, **kwargs)
    actual, stats = fit(X, y, 1, params={**options, "metal_parallel_root": 1}, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["parallel_root"] == 0 for item in controls)
    assert all(item["tuning"]["parallel_root"] == expected_active for item in stats)
    return actual, stats


@pytest.mark.parametrize("bins,missing",
                         list(itertools.product([2, 31, 32, 33, 64, 65, 128, 256], [False, True])))
def test_root_bin_and_simd_boundaries(bins, missing):
    X, y = inputs(rows=1025, density=0.5)
    X = layout(X, "csr") if missing else np.nan_to_num(X)
    compare(X, y, bins=bins, rounds=3)


@pytest.mark.parametrize("fast_stats,threads", list(itertools.product([0, 1], [32, 256])))
def test_statistics_and_nonroot_thread_modes(fast_stats, threads):
    X, y = inputs(rows=1025, density=0.3)
    compare(X, y, bins=256, rounds=2,
            params={"metal_fast_stats": fast_stats, "metal_split_threads": threads,
                    "metal_compact_splits": 0})


@pytest.mark.parametrize("quantile,missing_feature", list(itertools.product([False, True], [None, 0, 1])))
def test_ordered_root_statistics_retain_cancellation(quantile, missing_feature):
    X = np.asarray([[0, 0], [1, 0], [0, 0]], np.float32)
    if missing_feature is not None:
        X[0, missing_feature] = np.nan
    gradient = np.asarray([2**40, -2**40, 3 * 2**-14], np.float32)
    hessian = np.asarray([1, 2, 1], np.float32)
    compare(X, np.zeros(3, np.float32), bins=4, quantile=quantile, rounds=1,
            params={"max_depth": 3, "eta": 1, "reg_lambda": 1, "min_child_weight": 1},
            custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("default_left", [0, 1])
def test_root_missing_direction_ties_retain_original_order(default_left):
    reference = xgb.QuantileDMatrix(np.arange(128, dtype=np.float32).reshape(-1, 1), max_bin=128)
    X = np.repeat(np.asarray([16, 100, np.nan], np.float32), 64).reshape(-1, 1)
    y = np.repeat(np.asarray([-1, 1, -1 if default_left else 1], np.float32), 64)
    actual, _ = compare(X, y, bins=128, quantile=True, metadata={"ref": reference}, rounds=1,
                        params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0})
    document = tree(actual)
    assert len(document["base_weights"]) == 3
    assert document["default_left"][0] == default_left


@pytest.mark.parametrize("group", [1, 4])
def test_signed_cancellation_zero_hessians_and_duplicate_features(group):
    X, y = inputs(rows=1025, density=0.4)
    X[:, 1] = X[:, 0]
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.resize(np.asarray([1, 2, 1, 0, 2**-20], np.float32), len(X))
    compare(X, y, bins=256, rounds=2,
            params={"metal_features_per_group": group, "min_child_weight": 0},
            custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("case", ["cpu_driver", "serial", "subtraction_control", "original_gain", "fast_weight",
                                  "depth_one", "large_bins", "tiled", "categorical", "sampling"])
def test_ineligible_paths_keep_original_root(case):
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
    elif case == "depth_one":
        params["max_depth"] = 1
    elif case == "large_bins":
        bins = 257
    elif case == "tiled":
        params["metal_scalar_split_bin_tile"] = 32
    elif case == "categorical":
        present = ~np.isnan(X[:, 0])
        X[present, 0] = np.arange(np.count_nonzero(present)) % 5
        metadata["feature_types"] = ["c"] + ["q"] * 6
    else:
        params["subsample"] = 0.7
    # This test isolates GPU-driver eligibility; fallback roots have their own control.
    params["metal_fallback_parallel_root"] = 0
    compare(X, y, bins=bins, params=params, metadata=metadata, expected_active=0, rounds=2)


@pytest.mark.parametrize("case", ["empty", "all_missing", "constant", "zero_weight"])
def test_empty_and_invalid_roots_have_initialized_group_winners(case):
    X, y = inputs(rows=513)
    metadata = {}
    if case == "empty":
        X, y = X[:0], y[:0]
    elif case == "all_missing":
        X.fill(np.nan)
    elif case == "constant":
        y.fill(0)
    else:
        metadata["weight"] = np.zeros_like(y)
    actual, _ = compare(X, y, bins=256, metadata=metadata, rounds=2)
    assert len(tree(actual)["base_weights"]) == 1


def test_masked_features_retain_initialized_root_candidates():
    X, y = inputs(density=0.5)
    compare(X, y, bins=256, params={"colsample_bytree": 0.125, "seed_per_iteration": True},
            metadata={"feature_weights": np.asarray([1, 2, 3, 4, 5, 6, 7], np.float32)}, rounds=3)


def test_workspace_resizing_reset_and_root_switches():
    X, y = inputs(rows=1025, density=0.5)
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=5, max_bin=256, base_score=0, seed=72, metal_fast_stats=1))
        trace = []
        for iteration, rows in enumerate([257, 1025, 0, 513, 1025]):
            selected = 0 if iteration == 3 else enabled
            model.set_param("metal_parallel_root", selected)
            if iteration == 2:
                model.reset()
            values = np.nan_to_num(X[:rows]) if iteration % 2 else X[:rows]
            data = xgb.DMatrix(values, label=y[:rows])
            model.update(data, iteration)
            assert profile(model)["tuning"]["parallel_root"] == selected
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


def test_long_sequence_preserves_models_predictions_and_rng():
    X, y = inputs(rows=513, density=0.8)
    y += np.random.default_rng(627).normal(scale=0.25, size=len(y)).astype(np.float32)
    compare(X, y, bins=128, rounds=100, params={"max_depth": 6, "learning_rate": 0.01})
