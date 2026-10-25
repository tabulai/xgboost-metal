"""Skipping duplicate numeric split candidates preserves exact tie order."""
import json

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, profile, snapshot
from test_histogram_missing import fit, inputs, layout


def compare(X, y, *, params=None, expected_active=1, histogram_missing=1, **kwargs):
    params = params or {}
    expected, control_stats = fit(X, y, histogram_missing,
                                 params={**params, "metal_skip_empty_bins": 0}, **kwargs)
    actual, stats = fit(X, y, histogram_missing,
                       params={**params, "metal_skip_empty_bins": 1}, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["split_skip_empty_bins"] == 0 for item in control_stats)
    assert all(item["tuning"]["split_skip_empty_bins"] == expected_active for item in stats)
    return actual, stats


@pytest.mark.parametrize("gpu_tree", [0, 1])
@pytest.mark.parametrize("bins", [16, 128, 256])
@pytest.mark.parametrize("group", [1, 4])
@pytest.mark.parametrize("histogram_missing", [0, 1])
def test_sparse_cpu_and_gpu_drivers_with_both_histogram_kernels(gpu_tree, bins, group, histogram_missing):
    X, y = inputs(rows=1025)
    _, stats = compare(layout(X, "csr"), y, bins=bins, group=group,
                       params={"metal_gpu_tree": gpu_tree}, histogram_missing=histogram_missing)
    assert stats[-1]["tuning"]["gpu_tree"] == gpu_tree


@pytest.mark.parametrize("default_left", [0, 1])
@pytest.mark.parametrize("histogram_missing", [0, 1])
def test_reference_cut_gaps_keep_first_forward_and_backward_threshold(default_left, histogram_missing):
    # A dense reference supplies cuts in long empty stretches, including across
    # SIMD lane 31/32 and scan chunk boundaries. The missing labels determine
    # which direction wins; duplicate gains must keep that direction's first
    # threshold, not the threshold at the other end of an empty stretch.
    reference = xgb.QuantileDMatrix(np.arange(128, dtype=np.float32)[:, None], max_bin=128)
    X = np.repeat(np.asarray([16, 100, np.nan], np.float32), 64)[:, None]
    y = np.repeat(np.asarray([-1, 1, -1 if default_left else 1], np.float32), 64)
    actual, _ = compare(X, y, bins=128, quantile=True, metadata={"ref": reference},
                        histogram_missing=histogram_missing,
                        params={"max_depth": 1, "reg_lambda": 0, "min_child_weight": 0}, rounds=1)
    model = xgb.Booster(model_file=bytearray(actual["model"]))
    tree = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"][0]
    assert tree["default_left"][0] == default_left


@pytest.mark.parametrize("monotone", [-1, 0, 1])
@pytest.mark.parametrize("group", [1, 4])
def test_nonempty_bins_with_zero_integer_pair_and_signed_cancellation(monotone, group):
    reference_values = np.tile(np.arange(128, dtype=np.float32)[:, None], (1, 5))
    reference = xgb.QuantileDMatrix(reference_values, max_bin=128)
    values = np.repeat(np.asarray([16, 31, 32, 63, 64, 100, np.nan], np.float32), 64)
    X = np.tile(values[:, None], (1, 5))
    gradient = np.tile(np.asarray([2**40, -2**40], np.float32), len(X) // 2)
    hessian = np.zeros(len(X), np.float32)
    active = np.isnan(values) | (values == 16) | (values == 100)
    gradient[active] = np.where(values[active] == 100, -1, 1)
    hessian[active] = 1
    compare(X, np.zeros(len(X), np.float32), bins=128, group=group, quantile=True,
            metadata={"ref": reference}, rounds=2,
            params={"min_child_weight": 0, "reg_lambda": 0,
                    "monotone_constraints": "(" + ",".join([str(monotone)] * 5) + ")"},
            custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("case", ["dense", "serial", "tiled", "vector", "empty"])
def test_execution_paths_preserve_evaluation(case):
    X, y = inputs(density=0.95)
    params = {}
    if case == "dense":
        X = np.nan_to_num(X)
    elif case == "serial":
        params["metal_parallel_splits"] = 0
    elif case == "tiled":
        params["metal_scalar_split_bin_tile"] = 32
    elif case == "vector":
        y = np.column_stack([y, -y]).astype(np.float32)
        params["multi_strategy"] = "multi_output_tree"
    else:
        X, y = X[:0], y[:0]
    compare(X, y, bins=256, params=params, expected_active=int(case in ("dense", "empty")), rounds=2)


@pytest.mark.parametrize("case", ["all_missing", "all_zero_derivatives", "multiclass",
                                  "weighted", "sampling", "categorical", "constraints"])
def test_degenerate_multigroup_and_constrained_candidates(case):
    X, y = inputs(density=0.3)
    params, metadata, custom = {}, {}, None
    if case == "all_missing":
        X.fill(np.nan)
    elif case == "all_zero_derivatives":
        zeros = np.zeros_like(y)
        custom = lambda prediction, matrix: (zeros, zeros)
    elif case == "multiclass":
        y = (np.arange(len(X)) % 3).astype(np.float32)
        params = {"objective": "multi:softprob", "num_class": 3}
    elif case == "weighted":
        weights = np.ones_like(y)
        weights[::2] = 0
        metadata["weight"] = weights
    elif case == "sampling":
        params = {"subsample": 0.7, "colsample_bytree": 0.8, "colsample_bynode": 0.7,
                  "sampling_method": "gradient_based", "seed_per_iteration": True}
    elif case == "categorical":
        present = ~np.isnan(X[:, 0])
        X[present, 0] = np.arange(np.count_nonzero(present)) % 5
        metadata["feature_types"] = ["c"] + ["q"] * 6
        params["max_cat_to_onehot"] = 2
    else:
        params = {"interaction_constraints": "[[0,1,2],[3,4,5,6]]", "reg_alpha": 0.25,
                  "max_delta_step": 0.5, "min_child_weight": 0}
    compare(X, y, params=params, metadata=metadata, custom=custom, rounds=2)


def test_default_enablement_reset_and_toggle():
    X, y = inputs(density=0.3)
    results = []
    for enabled in [0, 1]:
        matrices = [xgb.DMatrix(X, label=y), xgb.DMatrix(np.nan_to_num(X), label=y)]
        model = xgb.Booster(dict(max_depth=3, max_bin=16, base_score=0, seed=82), matrices)
        for iteration in range(4):
            if not enabled or iteration:
                model.set_param("metal_skip_empty_bins", enabled if iteration != 2 else 0)
            if iteration == 2:
                model.reset()
            matrix = matrices[int(iteration == 1)]
            model.update(matrix, iteration)
            active = enabled and iteration != 2
            assert profile(model)["tuning"]["split_skip_empty_bins"] == int(active)
        results.append(snapshot(model, matrices))
    assert_identical(*results)
