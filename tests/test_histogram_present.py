"""Inherited nonroot totals make stored missing-bin statistics unnecessary."""
import itertools
import json

import numpy as np
import pytest
import xgboost as xgb
from scipy import sparse

from test_gpu_tree import assert_identical, profile, snapshot
from test_histogram_missing import inputs, layout
from test_split_subtraction_fusion import fit


def compare(X, y, *, params=None, expected_active=1, **kwargs):
    params = params or {}
    expected, old_stats = fit(X, y, 1, params={**params, "metal_histogram_present": 0}, **kwargs)
    actual, stats = fit(X, y, 1, params={**params, "metal_histogram_present": 1}, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["histogram_present"] == 0 for item in old_stats)
    assert all(item["tuning"]["histogram_present"] == expected_active for item in stats)
    return actual, stats


@pytest.mark.parametrize("kind,bins,group", list(itertools.product(
    ["dense", "csr", "csc"], [2, 32, 256], [1, 2, 3, 4])))
def test_sparse_layouts_bins_and_partial_feature_groups(kind, bins, group):
    X, y = inputs(rows=769)
    compare(layout(X, kind), y, bins=bins, rounds=2, params={"metal_features_per_group": group})


@pytest.mark.parametrize("rows", [0, 1, 31, 32, 33, 255, 256, 257, 2049])
def test_partial_simd_tiles_and_empty_input(rows):
    X, y = inputs(rows)
    if rows == 1:
        X.fill(np.nan)
    compare(X, y, expected_active=int(rows != 0), rounds=1, params={"metal_histogram_rows": 256})


@pytest.mark.parametrize("group", [1, 3, 4])
def test_all_missing_root_and_pruned_frontier(group):
    X = np.full((529, 7), np.nan, np.float32)
    y = np.linspace(-1, 1, len(X), dtype=np.float32)
    actual, stats = compare(X, y, rounds=2, params={"metal_features_per_group": group})
    assert stats[-1]["tuning"]["gpu_tree_levels"] == 1
    assert actual["prediction"][2].shape == (len(X), 2)


@pytest.mark.parametrize("quantile,group", list(itertools.product([False, True], [1, 4])))
@pytest.mark.parametrize("hessian_case", ["positive", "zero", "signed", "negative", "tiny"])
def test_signed_carries_and_hessian_cancellation(quantile, group, hessian_case):
    X, y = inputs(rows=2053, density=0.3)
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    patterns = {"positive": [1, 2, 1, 0, 2**-20], "zero": [0],
                "signed": [-3, 4, -1, 2, 0], "negative": [-1],
                "tiny": [2**-40, 2**-39, 0]}
    hessian = np.resize(np.asarray(patterns[hessian_case], np.float32), len(X))
    compare(X, y, quantile=quantile, rounds=2,
            params={"metal_features_per_group": group, "min_child_weight": 0},
            custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("group,subtraction", list(itertools.product([1, 4], [0, 1])))
def test_nonroot_signed_totals_and_missing_placeholders(group, subtraction):
    # Each bucket has a positive total Hessian despite negative individual
    # Hessians, and its exact gradient survives cancellation across 40 bits.
    # The entirely missing third feature creates nonzero root missing sums,
    # whose derived-child placeholders must never replace inherited totals.
    X = np.repeat(np.asarray([[0, 0, np.nan], [0, 1, np.nan],
                              [1, 0, np.nan], [1, 1, np.nan]], np.float32), 3, axis=0)
    gradient = np.asarray([[2**40, -2**40, factor] for factor in [-1, -3, 1, 3]], np.float32).ravel()
    hessian = np.tile(np.asarray([-1, 1, 1], np.float32), 4)
    actual, stats = compare(X, np.zeros(len(X), np.float32), bins=4, rounds=1,
                            params={"max_depth": 3, "eta": 1, "reg_lambda": 0, "min_child_weight": 0,
                                    "metal_features_per_group": group, "metal_fused_subtraction": subtraction},
                            custom=lambda prediction, matrix: (gradient, hessian))
    assert stats[-1]["tuning"]["gpu_tree_levels"] >= 2
    model = xgb.Booster(model_file=bytearray(actual["model"]))
    tree = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"][0]
    assert len(tree["base_weights"]) == 7
    np.testing.assert_array_equal(np.asarray(tree["base_weights"][3:], np.float32), [1, 3, -1, -3])


@pytest.mark.parametrize("fused,parallel,missing_histogram", list(itertools.product([0, 1], repeat=3)))
def test_original_and_fused_evaluator_combinations(fused, parallel, missing_histogram):
    X, y = inputs(density=0.5)
    compare(X, y, rounds=2, expected_active=missing_histogram,
            params={"metal_fused_subtraction": fused, "metal_parallel_splits": parallel,
                                  "metal_histogram_missing": missing_histogram})


@pytest.mark.parametrize("coalesce,fill", list(itertools.product([0, 1], repeat=2)))
def test_serial_command_order_with_encoder_and_clear_settings(coalesce, fill):
    X, y = inputs(density=0.25)
    compare(X, y, rounds=2, params={"metal_coalesce_encoders": coalesce, "metal_compute_fill": fill})


@pytest.mark.parametrize("case", ["dense", "cpu_driver", "depth_one", "large_bins", "vector",
                                  "categorical", "sampling", "no_subtraction", "cache", "lossguide"])
def test_ineligible_paths_keep_complete_histograms(case):
    X, y = inputs(density=0.7)
    params, metadata, bins = {}, {}, 32
    if case == "dense":
        X = np.nan_to_num(X)
    elif case == "cpu_driver":
        params["metal_gpu_tree"] = 0
    elif case == "depth_one":
        params["max_depth"] = 1
    elif case == "large_bins":
        bins = 513
    elif case == "vector":
        y = np.column_stack([y, -y]).astype(np.float32)
        params["multi_strategy"] = "multi_output_tree"
    elif case == "categorical":
        present = ~np.isnan(X[:, 0])
        X[present, 0] = np.arange(np.count_nonzero(present)) % 5
        metadata["feature_types"] = ["c"] + ["q"] * 6
    elif case == "sampling":
        params["subsample"] = 0.7
    elif case == "no_subtraction":
        params["metal_histogram_subtraction"] = 0
    elif case == "cache":
        params["max_cached_hist_node"] = 1
    elif case == "lossguide":
        params.update(grow_policy="lossguide", max_depth=0, max_leaves=7)
    compare(X, y, bins=bins, rounds=2, params=params, metadata=metadata, expected_active=0)


@pytest.mark.parametrize("kind", ["multiclass", "parallel", "weighted", "column_sample"])
def test_multiple_trees_weights_and_rng(kind):
    X, y = inputs(density=0.3)
    params, metadata = {}, {}
    if kind == "multiclass":
        y = (np.arange(len(X)) % 3).astype(np.float32)
        params.update(objective="multi:softprob", num_class=3)
    elif kind == "parallel":
        params["num_parallel_tree"] = 3
    elif kind == "weighted":
        metadata["weight"] = np.linspace(0, 3, len(X), dtype=np.float32)
    else:
        params.update(colsample_bytree=0.6, seed_per_iteration=True)
        metadata["feature_weights"] = np.asarray([0.1, 0.2, 1, 2, 4, 8, 16], np.float32)
    compare(X, y, rounds=3, params=params, metadata=metadata)


def test_raw_sparse_duplicates_and_explicit_zero_values():
    rows = 257
    indices = np.tile(np.asarray([4, 1, 4, 0], np.int32), rows)
    values = np.zeros((rows, 4), np.float32)
    values[:, 0] = np.arange(rows) % 11
    values[:, 2] = -values[:, 0]
    values[::3, 2] = np.nan
    values[:, 3] = np.arange(rows) % 3
    X = sparse.csr_matrix((values.ravel(), indices, np.arange(rows + 1, dtype=np.int32) * 4),
                          shape=(rows, 7))
    compare(X, (np.arange(rows) % 5).astype(np.float32), rounds=2)


@pytest.mark.parametrize("workspace", [0, 1])
def test_reused_model_shape_changes_and_toggle(workspace):
    X, y = inputs(density=0.4)
    variants = [(X, y), (np.nan_to_num(X[:513]), y[:513]), (X[:0], y[:0]), (X[:257], y[:257]), (X, y)]
    results = []
    for enabled in [0, 1]:
        matrices = [xgb.DMatrix(values, label=labels) for values, labels in variants]
        model = xgb.Booster(dict(max_depth=4, max_bin=32, seed=271, base_score=0,
                                 metal_tree_workspace=workspace), matrices)
        trace = []
        for iteration, data in enumerate(matrices):
            selected = enabled if iteration != 3 else 0
            model.set_param("metal_histogram_present", selected)
            if iteration == 2:
                model.reset()
            model.update(data, iteration)
            active = selected and iteration not in (1, 2)
            assert profile(model)["tuning"]["histogram_present"] == int(active)
            trace.append(snapshot(model, matrices))
        results.append(trace)
    for expected, actual in zip(*results):
        assert_identical(expected, actual)
