"""Fused sibling subtraction preserves complete histograms and split state."""
import itertools
import json

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, profile, snapshot
from test_gpu_tree_deep import full_frontier_data
from test_histogram_missing import inputs, layout


def fit(X, y, enabled, *, bins=32, params=None, metadata=None, rounds=3,
        quantile=False, custom=None):
    options = dict(max_depth=5, max_bin=bins, base_score=0, seed=271,
                   metal_fused_subtraction=enabled)
    options.update(params or {})
    metadata = metadata or {}
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = constructor(X, label=y, **metadata,
                       **({"max_bin": bins} if quantile else {}))
    model = xgb.Booster(options, [data])
    stats = []
    for iteration in range(rounds):
        model.update(data, iteration, fobj=custom)
        stats.append(profile(model))
    query_metadata = {key: value for key, value in metadata.items()
                      if key in ("missing", "feature_types", "base_margin")}
    return snapshot(model, [data, xgb.DMatrix(X, **query_metadata)]), stats


def compare(X, y, *, expected_active=1, **kwargs):
    expected, control_stats = fit(X, y, 0, **kwargs)
    actual, stats = fit(X, y, 1, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["fused_subtraction"] == 0 for item in control_stats)
    assert all(item["tuning"]["fused_subtraction"] == expected_active for item in stats)
    return actual, stats


@pytest.mark.parametrize("bins", [2, 16, 31, 32, 33, 128, 256])
@pytest.mark.parametrize("missing", [False, True])
def test_bin_and_simd_boundaries_preserve_derived_histograms(bins, missing):
    X, y = inputs(density=0.7)
    if not missing:
        X = np.nan_to_num(X)
    compare(X, y, bins=bins, params={"metal_features_per_group": 4})


@pytest.mark.parametrize("group,skip,missing_histogram",
                         list(itertools.product([1, 2, 4], [0, 1], [0, 1])))
def test_csr_grouping_and_sparse_kernel_combinations(group, skip, missing_histogram):
    X, y = inputs(rows=1025)
    compare(layout(X, "csr"), y, bins=128, rounds=2,
            params={"metal_features_per_group": group, "metal_skip_empty_bins": skip,
                    "metal_histogram_missing": missing_histogram})


@pytest.mark.parametrize("coalesce,compute_fill", list(itertools.product([0, 1], repeat=2)))
def test_device_barrier_with_encoder_and_clear_variants(coalesce, compute_fill):
    X, y = inputs(density=0.4)
    compare(X, y, rounds=2,
            params={"metal_coalesce_encoders": coalesce, "metal_compute_fill": compute_fill})


@pytest.mark.parametrize("left_rows", [37, 732])
def test_either_sibling_can_be_the_smaller_built_histogram(left_rows):
    X = np.zeros((769, 3), np.float32)
    X[left_rows:, 0] = 1
    X[:, 2] = np.nan
    y = X[:, 0].copy()
    actual, _ = compare(X, y, bins=2, rounds=1,
                        params={"reg_lambda": 0, "min_child_weight": 0})
    tree = json.loads(xgb.Booster(model_file=bytearray(actual["model"])).save_raw(raw_format="json"))
    tree = tree["learner"]["gradient_booster"]["model"]["trees"][0]
    assert tree["split_indices"][0] == 0
    leaf_counts = np.unique(actual["prediction"][2], return_counts=True)[1]
    np.testing.assert_array_equal(np.sort(leaf_counts), np.sort([left_rows, len(X) - left_rows]))


@pytest.mark.parametrize("group", [1, 4])
def test_signed_gradient_carries_and_zero_hessian_cancellation(group):
    X, y = inputs(rows=2053, density=0.3)
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.resize(np.asarray([1, 2, 1, 0, 2**-20], np.float32), len(X))
    compare(X, y, rounds=2, params={"metal_features_per_group": group},
            custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("case", ["empty", "all_missing", "constant_target", "zero_weight", "pruned"])
def test_empty_and_pruned_frontiers_retire_without_stale_subtractions(case):
    X, y = inputs(density=0.3)
    metadata, params = {}, {}
    if case == "empty":
        X, y = X[:0], y[:0]
    elif case == "all_missing":
        X.fill(np.nan)
    elif case == "constant_target":
        y.fill(0)
    elif case == "zero_weight":
        metadata["weight"] = np.zeros_like(y)
    else:
        params["gamma"] = 1e20
    compare(X, y, metadata=metadata, params=params, rounds=2)


@pytest.mark.parametrize("case", ["serial", "cpu_driver", "depth_one", "large_bins", "tiled",
                                  "categorical", "vector", "monotone", "sampling", "no_subtraction"])
def test_unsupported_paths_keep_the_original_subtraction(case):
    X, y = inputs(density=0.95)
    metadata, params, bins = {}, {"metal_fallback_parallel_splits": 0}, 256
    if case == "serial":
        params["metal_parallel_splits"] = 0
    elif case == "cpu_driver":
        params["metal_gpu_tree"] = 0
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
    elif case == "vector":
        y = np.column_stack([y, -y]).astype(np.float32)
        params["multi_strategy"] = "multi_output_tree"
    elif case == "monotone":
        params["monotone_constraints"] = "(1,0,0,0,0,0,0)"
    elif case == "sampling":
        params["subsample"] = 0.7
    else:
        params["metal_histogram_subtraction"] = 0
    compare(X, y, bins=bins, params=params, metadata=metadata,
            expected_active=int(case in ("monotone", "sampling")), rounds=2)


@pytest.mark.parametrize("depth,pruned", list(itertools.product([8, 9, 10], [False, True])))
def test_full_and_compacted_deep_frontiers_reuse_materialized_parents(depth, pruned):
    X, y = full_frontier_data(pruned)
    compare(X, y, bins=2, quantile=True, rounds=1,
            params={"max_depth": depth, "reg_lambda": 0, "min_child_weight": 1})


@pytest.mark.parametrize("fraction", [0.35, 0.75])
def test_bytree_column_sampling_preserves_unused_parent_feature_slices(fraction):
    X, y = inputs(density=0.5)
    compare(X, y, params={"colsample_bytree": fraction, "seed_per_iteration": True},
            metadata={"feature_weights": np.asarray([0.1, 0.2, 1, 2, 4, 8, 16], np.float32)})


@pytest.mark.parametrize("workspace", [0, 1])
def test_workspace_growth_reset_and_fusion_switches(workspace):
    X, y = inputs(rows=1025, density=0.4)
    traces = []
    for fused in [0, 1]:
        model = xgb.Booster(dict(max_bin=128, max_depth=4, base_score=0, seed=19,
                                 metal_tree_workspace=workspace))
        trace = []
        for iteration, (rows, depth, reset) in enumerate([(257, 4, False), (1025, 6, False),
                                                         (0, 6, True), (513, 5, False),
                                                         (1025, 6, False)]):
            selected = fused if iteration != 3 else 0
            model.set_param({"max_depth": depth, "metal_fused_subtraction": selected})
            if reset:
                model.reset()
            data = xgb.DMatrix(X[:rows], label=y[:rows])
            model.update(data, iteration)
            assert profile(model)["tuning"]["fused_subtraction"] == selected
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)
