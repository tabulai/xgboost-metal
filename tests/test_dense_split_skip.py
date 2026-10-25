"""Dense nonroot histograms can skip duplicate split candidates exactly."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from test_fast_scalar_weight import tree
from test_gpu_tree import assert_identical, profile, snapshot
from test_gpu_tree_deep import full_frontier_data
from test_histogram_missing import inputs
from test_split_subtraction_fusion import fit


def dense_data(rows=1025):
    X, y = inputs(rows=rows, density=0.9)
    return np.nan_to_num(X), y


def compare(X, y, *, params=None, expected_active=1, **kwargs):
    params = params or {}
    expected, controls = fit(X, y, 1, params={**params, "metal_skip_empty_bins": 0}, **kwargs)
    actual, stats = fit(X, y, 1, params={**params, "metal_skip_empty_bins": 1}, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["split_skip_empty_bins"] == 0 for item in controls)
    assert all(item["tuning"]["split_skip_empty_bins"] == expected_active for item in stats)
    return actual, stats


@pytest.mark.parametrize("bins,quantile", list(itertools.product([2, 31, 32, 33, 128, 256], [False, True])))
def test_dense_bin_boundaries_and_quantile_matrices(bins, quantile):
    X, y = dense_data()
    actual, stats = compare(X, y, bins=bins, quantile=quantile,
                            params={"max_depth": 6}, rounds=3)
    assert stats[-1]["tuning"]["gpu_tree"] == 1
    assert stats[-1]["tuning"]["histogram_missing"] == 0
    assert len(tree(actual)["base_weights"]) > 3


@pytest.mark.parametrize("pruned,group", list(itertools.product([False, True], [1, 4])))
def test_dense_depth_ten_full_and_compacted_frontiers(pruned, group):
    X, y = full_frontier_data(pruned)
    actual, stats = compare(X, y, bins=2, quantile=True, rounds=1,
                            params={"max_depth": 10, "reg_lambda": 0, "min_child_weight": 1,
                                    "metal_features_per_group": group})
    assert stats[-1]["tuning"]["gpu_tree_levels"] == 10
    assert len(tree(actual)["base_weights"]) > 1023


@pytest.mark.parametrize("group", [1, 4])
def test_dense_reference_gaps_and_cancelled_nonempty_bins_keep_first_threshold(group):
    reference_values = np.column_stack([np.arange(128) % 2, np.arange(128)]).astype(np.float32)
    reference = xgb.QuantileDMatrix(reference_values, max_bin=128)
    values = np.repeat(np.asarray([16, 31, 32, 63, 64, 100], np.float32), 64)
    X = np.column_stack([np.repeat([0, 1], len(values)), np.tile(values, 2)]).astype(np.float32)
    gradient = np.tile(np.asarray([2**40, -2**40], np.float32), len(X) // 2)
    hessian = np.zeros(len(X), np.float32)
    active = (X[:, 1] == 16) | (X[:, 1] == 100)
    sign = np.where(X[:, 0] == 0, -1, 1).astype(np.float32)
    gradient[active] = sign[active] * np.where(X[active, 1] == 16, 1, 3)
    hessian[active] = 1
    actual, _ = compare(X, np.zeros(len(X), np.float32), bins=128, quantile=True,
                        metadata={"ref": reference}, rounds=1,
                        params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0,
                                "metal_features_per_group": group},
                        custom=lambda prediction, matrix: (gradient, hessian))
    document = tree(actual)
    assert len(document["base_weights"]) == 7
    assert document["split_indices"][:3] == [0, 1, 1]
    assert all(16 < threshold < 31 for threshold in document["split_conditions"][1:3])


@pytest.mark.parametrize("params", [{"reg_lambda": 0}, {"reg_alpha": 0.5},
                                     {"max_delta_step": 0.125}, {"min_child_weight": -0.0}])
def test_dense_regularization_retains_candidate_scores(params):
    X, y = dense_data()
    compare(X, y, params=params, rounds=3)


@pytest.mark.parametrize("case", ["cpu_driver", "serial", "depth_one", "large_bins", "tiled",
                                  "subtraction_control", "monotone", "sampling"])
def test_dense_fallbacks_retain_original_kernel(case):
    X, y = dense_data()
    params, bins = {}, 128
    if case == "cpu_driver":
        params["metal_gpu_tree"] = 0
    elif case == "serial":
        params["metal_parallel_splits"] = 0
    elif case == "depth_one":
        params["max_depth"] = 1
    elif case == "large_bins":
        bins = 257
    elif case == "tiled":
        params["metal_scalar_split_bin_tile"] = 32
    elif case == "subtraction_control":
        params["metal_fused_subtraction"] = 0
    elif case == "monotone":
        params["monotone_constraints"] = "(1,0,0,0,0,0,0)"
    else:
        params["subsample"] = 0.7
    compare(X, y, bins=bins, params=params, expected_active=0, rounds=2)


@pytest.mark.parametrize("case", ["empty", "constant"])
def test_empty_and_pruned_dense_frontiers(case):
    X, y = dense_data()
    if case == "empty":
        X, y = X[:0], y[:0]
    else:
        y.fill(0)
    actual, _ = compare(X, y, rounds=2)
    assert len(tree(actual)["base_weights"]) == 1


@pytest.mark.parametrize("weight,gain", list(itertools.product([0, 1], repeat=2)))
def test_all_fused_arithmetic_variants(weight, gain):
    X, y = dense_data()
    compare(X, y, params={"metal_fast_weight": weight, "metal_fast_gain": gain}, rounds=3)


@pytest.mark.parametrize("depth", [6, 10])
def test_dense_gpu_matches_original_cpu_tree_driver(depth):
    X, y = dense_data()
    expected, controls = fit(X, y, 1, bins=128, rounds=3,
                             params={"max_depth": depth, "metal_gpu_tree": 0, "metal_skip_empty_bins": 0})
    actual, stats = compare(X, y, bins=128, params={"max_depth": depth}, rounds=3)
    assert all(item["tuning"]["gpu_tree"] == 0 for item in controls)
    assert all(item["tuning"]["gpu_tree"] == 1 for item in stats)
    assert_identical(expected, actual)


def test_dense_missing_transitions_reset_and_toggle():
    missing, y = inputs(density=0.5)
    dense = np.nan_to_num(missing)
    traces = []
    for enabled in [0, 1]:
        matrices = [xgb.DMatrix(dense, label=y), xgb.DMatrix(missing, label=y)]
        model = xgb.Booster(dict(max_depth=5, max_bin=128, seed=37, base_score=0), matrices)
        trace = []
        for iteration, index in enumerate([0, 1, 0, 0, 1]):
            selected = enabled if iteration != 2 else 0
            if not enabled or iteration:
                model.set_param("metal_skip_empty_bins", selected)
            if iteration == 2:
                model.reset()
            model.update(matrices[index], iteration)
            tuning = profile(model)["tuning"]
            assert tuning["split_skip_empty_bins"] == selected
            assert tuning["histogram_missing"] == index
            trace.append(snapshot(model, matrices))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)
