"""Histogram scratch sizing and built-sibling clearing retain exact tree state."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from test_fast_scalar_weight import tree
from test_gpu_tree import assert_identical, profile, snapshot
from test_gpu_tree_deep import full_frontier_data
from test_histogram_missing import inputs, layout
from test_split_subtraction_fusion import fit


FLAGS = {"histogram_compact_scratch": "metal_histogram_compact_scratch",
         "histogram_clear_built": "metal_histogram_clear_built"}


def compare(X, y, *, scratch=1, clear=1, params=None, expected_active=None, **kwargs):
    params = params or {}
    selected = {"histogram_compact_scratch": scratch, "histogram_clear_built": clear}
    expected, controls = fit(X, y, 1, params={**params, **{flag: 0 for flag in FLAGS.values()}}, **kwargs)
    actual, profiles = fit(X, y, 1,
                           params={**params, **{FLAGS[key]: value for key, value in selected.items()}}, **kwargs)
    assert_identical(expected, actual)
    wanted = selected if expected_active is None else expected_active
    for item in controls:
        assert {key: item["tuning"][key] for key in FLAGS} == dict.fromkeys(FLAGS, 0)
    for item in profiles:
        assert {key: item["tuning"][key] for key in FLAGS} == wanted
    return actual, profiles


@pytest.mark.parametrize("bins,missing,group", list(itertools.product([2, 32, 33, 128, 256], [False, True], [1, 2])))
@pytest.mark.parametrize("scratch,clear", [(1, 0), (0, 1), (1, 1)])
def test_bins_features_and_independent_flags(bins, missing, group, scratch, clear):
    X, y = inputs(rows=1025, density=0.4)
    X = layout(X, "csr") if missing else np.nan_to_num(X)
    compare(X, y, bins=bins, scratch=scratch, clear=clear, rounds=2,
            params={"metal_features_per_group": group})


@pytest.mark.parametrize("case", ["empty", "all_missing", "pruned"])
@pytest.mark.parametrize("scratch,clear", [(1, 0), (0, 1), (1, 1)])
def test_empty_and_pruned_frontiers(case, scratch, clear):
    X, y = inputs(rows=513)
    params = {}
    if case == "empty":
        X, y = X[:0], y[:0]
    elif case == "all_missing":
        X.fill(np.nan)
    else:
        params["gamma"] = 1e20
    actual, _ = compare(X, y, bins=256, scratch=scratch, clear=clear, params=params, rounds=2)
    assert len(tree(actual)["base_weights"]) == 1


@pytest.mark.parametrize("rows", [1, 255, 256, 257, 2047, 2048, 2049])
def test_partial_groups_and_histogram_tile_boundaries(rows):
    X, y = inputs(rows=rows, density=0.5)
    compare(X, y, bins=256, rounds=2,
            params={"metal_features_per_group": 2, "metal_histogram_rows": 2048})


@pytest.mark.parametrize("group,missing_kernel", list(itertools.product([1, 2], [0, 1])))
def test_signed_carries_and_missing_kernel_controls(group, missing_kernel):
    X, y = inputs(rows=2053, density=0.3)
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.resize(np.asarray([1, 2, 1, 0, 2**-20], np.float32), len(X))
    compare(X, y, bins=256, rounds=3,
            params={"metal_features_per_group": group, "metal_histogram_missing": missing_kernel,
                    "min_child_weight": 0},
            custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("left_rows", [37, 384, 731])
def test_clear_chooses_either_smaller_sibling_and_left_on_ties(left_rows):
    X = np.zeros((768, 3), np.float32)
    X[left_rows:, 0] = 1
    X[:, 1] = np.arange(len(X)) % 2
    X[:, 2] = np.nan
    y = X[:, 0] * 10 + X[:, 1]
    actual, _ = compare(X, y, bins=2, rounds=1,
                        params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0})
    document = tree(actual)
    assert document["split_indices"][0] == 0
    assert len(document["base_weights"]) == 7


@pytest.mark.parametrize("case", ["cpu_driver", "depth_one", "depth_eleven", "large_bins", "vector",
                                  "categorical", "sampling", "serial", "subtraction_control",
                                  "group_three", "group_four", "present_only"])
def test_unsupported_paths_retain_original_allocations_and_clear(case):
    X, y = inputs(rows=513, density=0.5)
    params, metadata, bins = {}, {}, 256
    expected = dict.fromkeys(FLAGS, 0)
    if case == "cpu_driver":
        params["metal_gpu_tree"] = 0
    elif case == "depth_one":
        params["max_depth"] = 1
    elif case == "depth_eleven":
        params["max_depth"] = 11
    elif case == "large_bins":
        bins = 257
    elif case == "vector":
        y = np.column_stack([y, -y]).astype(np.float32)
        params["multi_strategy"] = "multi_output_tree"
    elif case == "categorical":
        present = ~np.isnan(X[:, 0])
        X[present, 0] = np.arange(np.count_nonzero(present)) % 5
        metadata["feature_types"] = ["c"] + ["q"] * 6
    elif case == "sampling":
        params["subsample"] = 0.7
    elif case == "serial":
        params["metal_parallel_splits"] = 0
        expected["histogram_compact_scratch"] = 1
    elif case == "subtraction_control":
        params["metal_fused_subtraction"] = 0
        expected["histogram_compact_scratch"] = 1
    elif case in ("group_three", "group_four"):
        params["metal_features_per_group"] = 3 if case == "group_three" else 4
        expected["histogram_clear_built"] = 1
    else:
        params["metal_histogram_present"] = 1
        expected["histogram_clear_built"] = 1
    compare(X, y, bins=bins, params=params, metadata=metadata, expected_active=expected, rounds=2)


@pytest.mark.parametrize("pruned", [False, True])
def test_depth_ten_full_and_pruned_frontiers(pruned):
    X, y = full_frontier_data(pruned)
    actual, profiles = compare(X, y, bins=2, quantile=True, rounds=1,
                               params={"max_depth": 10, "reg_lambda": 0, "min_child_weight": 1})
    assert profiles[-1]["tuning"]["gpu_tree_levels"] == 10
    assert len(tree(actual)["base_weights"]) > 1023


@pytest.mark.parametrize("fusion", [0, 1])
def test_control_fusion_and_coalesced_encoders(fusion):
    X, y = inputs(rows=1025, density=0.5)
    compare(X, y, bins=256, rounds=3,
            params={"metal_tree_control_fusion": fusion, "metal_coalesce_encoders": 1,
                    "metal_compute_fill": 1})


def test_workspace_growth_reset_and_independent_toggles():
    X, y = inputs(rows=2053, density=0.5)
    traces = []
    modes = [(1, 1), (0, 1), (1, 0), (0, 0), (1, 1)]
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=6, max_bin=256, base_score=0, seed=72))
        trace = []
        for iteration, (rows, mode) in enumerate(zip([257, 2053, 0, 513, 2053], modes)):
            selected = dict(zip(FLAGS, mode if enabled else (0, 0)))
            model.set_param({FLAGS[key]: value for key, value in selected.items()})
            if iteration == 2:
                model.reset()
            values = np.nan_to_num(X[:rows]) if iteration % 2 else X[:rows]
            data = xgb.DMatrix(values, label=y[:rows])
            model.update(data, iteration)
            assert {key: profile(model)["tuning"][key] for key in FLAGS} == selected
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


def test_explicit_controls_disable_all_new_paths():
    X, y = inputs(rows=257, density=0.5)
    controls = {**{flag: 0 for flag in FLAGS.values()}, "metal_fused_score": 0}
    expected, profiles = fit(X, y, 1, rounds=2, params=controls)
    actual, _ = fit(X, y, 1, rounds=2)
    assert_identical(expected, actual)
    assert all(item["tuning"][key] == 0 for item in profiles for key in (*FLAGS, "fused_score"))


@pytest.mark.parametrize("missing,depth,group", list(itertools.product([False, True], [6, 10], [1, 2, 4])))
def test_adaptive_defaults_match_explicit_original_paths(missing, depth, group):
    X, y = inputs(rows=513, density=0.5)
    if not missing:
        X = np.nan_to_num(X)
    params = {"max_depth": depth, "metal_features_per_group": group}
    controls = {**params, **{flag: 0 for flag in FLAGS.values()}, "metal_fused_score": 0}
    expected, old_profiles = fit(X, y, 1, bins=256, rounds=2, params=controls)
    actual, profiles = fit(X, y, 1, bins=256, rounds=2, params=params)
    assert_identical(expected, actual)
    for item in old_profiles:
        assert all(item["tuning"][key] == 0 for key in (*FLAGS, "fused_score"))
    compact = int(missing or depth >= 8)
    wanted = {"histogram_compact_scratch": int(missing and group <= 2),
              "histogram_clear_built": 1, "fused_score": compact,
              "compact_splits": compact, "fast_stats": 1, "parallel_root": 1,
              "histogram_present": 0, "features_per_group": group,
              "split_threads": 256}
    for item in profiles:
        assert {key: item["tuning"][key] for key in wanted} == wanted


def test_long_sequence_preserves_models_predictions_and_rng():
    X, y = inputs(rows=513, density=0.8)
    y += np.random.default_rng(627).normal(scale=0.25, size=len(y)).astype(np.float32)
    compare(X, y, bins=128, rounds=100, params={"max_depth": 6, "learning_rate": 0.01})
