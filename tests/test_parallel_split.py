"""Eight-SIMD split scoring keeps the original prefix sums and candidate order."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from test_fast_scalar_weight import tree
from test_gpu_tree import assert_identical, profile, snapshot
from test_gpu_tree_deep import full_frontier_data
from test_histogram_missing import inputs, layout
from test_split_subtraction_fusion import fit


def compare(X, y, *, params=None, expected_threads=256, **kwargs):
    params = {"metal_compact_splits": 0, **(params or {})}
    expected, controls = fit(X, y, 1, params={**params, "metal_split_threads": 32}, **kwargs)
    actual, stats = fit(X, y, 1, params={**params, "metal_split_threads": 256}, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["split_threads"] == 32 for item in controls)
    assert all(item["tuning"]["split_threads"] == expected_threads for item in stats)
    return actual, stats


@pytest.mark.parametrize("bins,missing",
                         list(itertools.product([2, 31, 32, 33, 64, 65, 128, 256], [False, True])))
def test_bin_and_simd_group_boundaries(bins, missing):
    X, y = inputs(rows=1025, density=0.5)
    X = layout(X, "csr") if missing else np.nan_to_num(X)
    compare(X, y, bins=bins, rounds=3)


@pytest.mark.parametrize("default_left", [0, 1])
def test_nonroot_forward_backward_ties_keep_first_threshold(default_left):
    reference_values = np.column_stack([np.arange(128) % 2, np.arange(128)]).astype(np.float32)
    reference = xgb.QuantileDMatrix(reference_values, max_bin=128)
    values = np.repeat(np.asarray([16, 100, np.nan], np.float32), 64)
    X = np.column_stack([np.repeat([0, 1], len(values)), np.tile(values, 2)]).astype(np.float32)
    labels = np.repeat(np.asarray([-1, 1, -1 if default_left else 1], np.float32), 64)
    y = np.tile(labels, 2) + np.where(X[:, 0] == 0, -10, 10).astype(np.float32)
    actual, stats = compare(X, y, bins=128, quantile=True, metadata={"ref": reference}, rounds=1,
                            params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0})
    document = tree(actual)
    assert len(document["base_weights"]) == 7
    assert document["split_indices"][:3] == [0, 1, 1]
    assert document["default_left"][1:3] == [default_left, default_left]
    assert stats[-1]["tuning"]["gpu_tree_levels"] >= 2


@pytest.mark.parametrize("skip,group", list(itertools.product([0, 1], [1, 4])))
def test_signed_cancellation_zero_hessians_and_duplicate_features(skip, group):
    X, y = inputs(rows=1025, density=0.4)
    X[:, 1] = X[:, 0]
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.resize(np.asarray([1, 2, 1, 0, 2**-20], np.float32), len(X))
    compare(X, y, bins=256, rounds=3,
            params={"metal_skip_empty_bins": skip, "metal_features_per_group": group, "min_child_weight": 0},
            custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("exponent", [-75, -63, 62])
def test_nonroot_scalar_rounding_boundaries(exponent):
    outer = 15 if exponent == -75 else 3
    hessian_exponent = 125 if exponent == -75 else (120 if exponent < 0 else -20)
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)
    gradient = np.asarray([-1, -outer, 1, outer], np.float32) * np.float32(2.0**(exponent + hessian_exponent))
    hessian = np.full(4, 2.0**hessian_exponent, np.float32)
    actual, stats = compare(X, np.zeros(4, np.float32), bins=4, quantile=True, rounds=1,
                            params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0, "eta": 1},
                            custom=lambda prediction, matrix: (gradient, hessian))
    assert len(tree(actual)["base_weights"]) == 7
    assert stats[-1]["tuning"]["gpu_tree_levels"] >= 2


@pytest.mark.parametrize("case", ["cpu_driver", "serial", "subtraction_control", "original_gain", "fast_weight",
                                  "depth_one", "large_bins", "tiled", "categorical", "sampling"])
def test_unsupported_paths_keep_32_thread_selection(case):
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
    compare(X, y, bins=bins, params=params, metadata=metadata, expected_threads=32, rounds=2)


@pytest.mark.parametrize("fraction", [0.125, 0.75])
def test_masked_features_retain_initialized_candidates(fraction):
    X, y = inputs(density=0.5)
    compare(X, y, bins=256, params={"colsample_bytree": fraction, "seed_per_iteration": True},
            metadata={"feature_weights": np.asarray([1, 2, 3, 4, 5, 6, 7], np.float32)}, rounds=3)


@pytest.mark.parametrize("control,present", list(itertools.product([0, 1], repeat=2)))
def test_control_fusion_and_present_histogram_combinations(control, present):
    X, y = inputs(density=0.05)
    compare(X, y, bins=256,
            params={"metal_tree_control_fusion": control, "metal_histogram_present": present}, rounds=3)


@pytest.mark.parametrize("pruned", [False, True])
def test_full_and_compacted_depth_ten_frontiers(pruned):
    X, y = full_frontier_data(pruned)
    actual, stats = compare(X, y, bins=2, quantile=True, rounds=1,
                            params={"max_depth": 10, "reg_lambda": 0, "min_child_weight": 1})
    assert stats[-1]["tuning"]["gpu_tree_levels"] == 10
    assert len(tree(actual)["base_weights"]) > 1023


def test_empty_frontier_has_no_stale_group_winner():
    X, y = inputs(rows=0)
    actual, _ = compare(X, y, bins=256, rounds=2)
    assert len(tree(actual)["base_weights"]) == 1


def test_workspace_resizing_reset_and_thread_switches():
    X, y = inputs(rows=1025, density=0.5)
    traces = []
    for requested in [32, 256]:
        model = xgb.Booster(dict(max_depth=5, max_bin=256, base_score=0, seed=72,
                                metal_compact_splits=0))
        trace = []
        for iteration, rows in enumerate([257, 1025, 0, 513, 1025]):
            selected = 32 if iteration == 3 else requested
            model.set_param("metal_split_threads", selected)
            if iteration == 2:
                model.reset()
            data = xgb.DMatrix(X[:rows], label=y[:rows])
            model.update(data, iteration)
            assert profile(model)["tuning"]["split_threads"] == selected
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


def test_long_sequence_preserves_models_predictions_and_rng():
    X, y = inputs(rows=513, density=0.8)
    y += np.random.default_rng(627).normal(scale=0.25, size=len(y)).astype(np.float32)
    compare(X, y, bins=128, rounds=100, params={"max_depth": 6, "learning_rate": 0.01})


def test_invalid_thread_count_is_rejected():
    X, y = inputs(rows=64)
    data = xgb.DMatrix(X, label=y)
    model = xgb.Booster(dict(max_depth=3, max_bin=32, metal_split_threads=128,
                            metal_compact_splits=0), [data])
    with pytest.raises(xgb.core.XGBoostError, match="metal_split_threads must be 32 or 256"):
        model.update(data, 0)
