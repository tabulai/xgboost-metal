"""Reusing rounded curvature preserves scalar weights, gains and tree bytes."""
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
    params = {"metal_compact_splits": 1, "metal_fast_stats": 1, "metal_fast_gain": 1,
              "metal_fast_weight": 0, **(params or {})}
    expected, controls = fit(X, y, 1, params={**params, "metal_fused_score": 0}, **kwargs)
    actual, stats = fit(X, y, 1, params={**params, "metal_fused_score": 1}, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["fused_score"] == 0 for item in controls)
    assert all(item["tuning"]["fused_score"] == expected_active for item in stats)
    return actual, stats


@pytest.mark.parametrize("kind,bins", list(itertools.product(["dense", "csr"], [2, 33, 256])))
@pytest.mark.parametrize("root", [0, 1])
def test_sparse_dense_bins_and_root_controls(kind, bins, root):
    X, y = inputs(rows=1025, density=0.4)
    X = np.nan_to_num(X) if kind == "dense" else layout(X, kind)
    compare(X, y, bins=bins, rounds=2, params={"metal_parallel_root": root})


@pytest.mark.parametrize("depth,quantile", [(2, False), (6, True), (9, False), (10, True)])
def test_full_frontiers_and_quantile_materialization(depth, quantile):
    X, y = full_frontier_data(False)
    compare(X, y, bins=2, rounds=1, quantile=quantile,
            params={"max_depth": depth, "reg_lambda": 0, "min_child_weight": 1})


@pytest.mark.parametrize("params", [
    # Subnormal parameter bits are covered by the direct helper probe; the
    # public string parser rejects their textual representation as out of range.
    {"reg_alpha": 0.0}, {"reg_alpha": -0.0}, {"reg_alpha": 2**-120},
    {"reg_alpha": 0.125}, {"reg_alpha": 1000}, {"reg_lambda": 0},
    {"reg_lambda": 2**-80}, {"reg_lambda": 2**30},
    {"max_delta_step": -0.0}, {"max_delta_step": 0.125},
    {"reg_alpha": -0.0, "reg_lambda": -0.0, "max_delta_step": -0.0},
])
def test_regularization_signed_zero_and_clamps(params):
    X, y = inputs(rows=513, density=0.6)
    compare(X, y, params=params, rounds=3)


@pytest.mark.parametrize("exponent,sign", list(itertools.product([-75, -64, -63, -62, 0, 62], [-1, 1])))
def test_nonroot_square_boundaries_and_signed_weights(exponent, sign):
    outer = 15 if exponent == -75 else 3
    hessian_exponent = 125 if exponent == -75 else (120 if exponent < -1 else (-20 if exponent > 1 else 0))
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)
    gradient = np.asarray([-1, -outer, 1, outer], np.float32) * np.float32(sign * 2.0**(exponent + hessian_exponent))
    hessian = np.full(4, 2.0**hessian_exponent, np.float32)
    actual, stats = compare(X, np.zeros(4, np.float32), bins=4, quantile=True, rounds=1,
                            params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0, "eta": 1},
                            custom=lambda prediction, matrix: (gradient, hessian))
    document = tree(actual)
    assert len(document["base_weights"]) == 7 and stats[-1]["tuning"]["gpu_tree_levels"] >= 2
    expected = np.asarray([sign, sign * outer, -sign, -sign * outer], np.float32) * np.float32(2.0**exponent)
    np.testing.assert_array_equal(np.asarray(document["base_weights"][3:], np.float32).view(np.uint32),
                                  expected.view(np.uint32))


@pytest.mark.parametrize("case", ["cancellation", "signed_zero", "zero_hessian", "negative_hessian", "tiny_hessian", "alpha_boundary"])
def test_custom_derivative_cancellation_and_zero_routes(case):
    X, y = inputs(rows=257, density=0.5)
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.ones_like(gradient)
    params = {"min_child_weight": 0, "reg_lambda": 0}
    if case == "signed_zero":
        gradient = np.resize(np.asarray([0.0, -0.0], np.float32), len(X))
    elif case == "zero_hessian":
        hessian[::3] = 0
    elif case == "negative_hessian":
        hessian = np.resize(np.asarray([-1, 2, 1, 0], np.float32), len(X))
    elif case == "tiny_hessian":
        gradient = np.resize(np.asarray([-3, -1, 1, 3], np.float32), len(X)) * np.float32(2**-20)
        hessian.fill(2**-40)
    elif case == "alpha_boundary":
        gradient = np.resize(np.asarray([-0.5, 0.5, -0.25, 0.25, 0], np.float32), len(X))
        params["reg_alpha"] = 0.5
    compare(X, y, params=params, rounds=2, custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("case", ["all_missing", "constant_target", "empty", "pruned"])
def test_empty_frontiers_and_unsplit_root_weights(case):
    X, y = inputs(rows=257, density=0.5)
    params = {}
    if case == "all_missing":
        X.fill(np.nan)
    elif case == "constant_target":
        y.fill(0)
    elif case == "empty":
        X, y = X[:0], y[:0]
    else:
        params["gamma"] = 1e20
    compare(X, y, params=params, rounds=2)


@pytest.mark.parametrize("params", [
    {"metal_compact_splits": 0}, {"metal_fast_stats": 0}, {"metal_fast_gain": 0},
    {"metal_fast_weight": 1}, {"metal_gpu_tree": 0}, {"metal_parallel_splits": 0},
    {"metal_fused_subtraction": 0}, {"max_depth": 1}, {"max_bin": 513},
    {"monotone_constraints": "(1,0,0,0,0,0,0)"}, {"subsample": 0.7},
])
def test_unsupported_paths_keep_standalone_helpers(params):
    X, y = inputs(rows=513, density=0.6)
    params = {**params, "metal_fallback_parallel_splits": 0}
    bins = params.pop("max_bin", 32)
    compare(X, y, bins=bins, params=params, rounds=2, expected_active=0)


@pytest.mark.parametrize("case", ["multiclass", "parallel", "weighted", "column_sample"])
def test_multiple_trees_weights_and_rng(case):
    X, y = inputs(rows=513, density=0.4)
    params, metadata = {}, {}
    if case == "multiclass":
        y = (np.arange(len(X)) % 3).astype(np.float32)
        params.update(objective="multi:softprob", num_class=3)
    elif case == "parallel":
        params["num_parallel_tree"] = 3
    elif case == "weighted":
        metadata["weight"] = np.linspace(0, 3, len(X), dtype=np.float32)
    else:
        params.update(colsample_bytree=0.6, seed_per_iteration=True)
        metadata["feature_weights"] = np.asarray([0.1, 0.2, 1, 2, 4, 8, 16], np.float32)
    compare(X, y, params=params, metadata=metadata, rounds=3)


@pytest.mark.parametrize("workspace", [0, 1])
def test_reuse_reset_shape_changes_and_toggle(workspace):
    X, y = inputs(rows=1025, density=0.5)
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_bin=32, base_score=0, seed=19, metal_tree_workspace=workspace,
                                 metal_compact_splits=1, metal_fast_stats=1))
        trace = []
        for iteration, (rows, depth, reset) in enumerate([(257, 4, False), (1025, 6, False),
                                                         (0, 6, True), (513, 5, False), (1025, 6, False)]):
            selected = enabled if iteration != 3 else 0
            model.set_param({"max_depth": depth, "metal_fused_score": selected})
            if reset:
                model.reset()
            data = xgb.DMatrix(X[:rows], label=y[:rows])
            model.update(data, iteration)
            assert profile(model)["tuning"]["fused_score"] == selected
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)
