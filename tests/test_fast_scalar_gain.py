"""Fast scalar gain arithmetic preserves nonroot scoring and exact tree state."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from test_fast_scalar_weight import tree
from test_gpu_tree import assert_identical, profile, snapshot
from test_histogram_missing import inputs, layout
from test_split_subtraction_fusion import fit


def compare(X, y, *, params=None, expected_active=1, **kwargs):
    params = {"metal_fast_weight": 0, **(params or {})}
    expected, controls = fit(X, y, 1, params={**params, "metal_fast_gain": 0}, **kwargs)
    actual, stats = fit(X, y, 1, params={**params, "metal_fast_gain": 1}, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["fast_gain"] == 0 for item in controls)
    assert all(item["tuning"]["fast_gain"] == expected_active for item in stats)
    return actual, stats


@pytest.mark.parametrize("bins,kind,skip", [(2, "dense", 0), (33, "csr", 1), (256, "dense", 1), (256, "csr", 0)])
@pytest.mark.parametrize("weight", [0, 1])
def test_bins_sparse_candidates_and_weight_variants(bins, kind, skip, weight):
    X, y = inputs(rows=1025, density=0.4)
    X = np.nan_to_num(X) if kind == "dense" else layout(X, kind)
    compare(X, y, bins=bins, rounds=3,
            params={"metal_skip_empty_bins": skip, "metal_fast_weight": weight})


@pytest.mark.parametrize("depth,quantile", [(2, False), (6, True), (10, True)])
def test_depth_and_quantile_parent_gain(depth, quantile):
    X, y = inputs(density=0.8)
    compare(X, y, bins=32, quantile=quantile, params={"max_depth": depth}, rounds=3)


REGULARIZATION = [
    {"reg_lambda": 2**-80}, {"reg_lambda": 2**30, "reg_alpha": 0.125},
    {"reg_alpha": 1000}, {"max_delta_step": 0.125},
]


@pytest.mark.parametrize("params", REGULARIZATION)
def test_regularization_clamps_and_zero_signs(params):
    X, y = inputs(density=0.6)
    compare(X, y, params=params, rounds=3)


@pytest.mark.parametrize("exponent,sign",
                         list(itertools.product([-75, -64, -63, -62, 0, 62], [-1, 1])))
def test_nonroot_squared_weight_boundaries(exponent, sign):
    # Feature zero separates the signs at the root. Both depth-one nodes then
    # split on feature one. Large Hessians keep gains normal even when the
    # float32 square is subnormal or rounds to zero, so these cases exercise
    # the nonroot gain implementation instead of stopping at an unsplit root.
    # The first case needs a wider gap to clear the 1e-6 split threshold.
    # Keep small Hessians above the quantizer's minimum nonzero unit, 2^-62.
    outer = 15 if exponent == -75 else 3
    hessian_exponent = 125 if exponent == -75 else (120 if exponent < -1 else (-20 if exponent > 1 else 0))
    unit = 2.0**exponent
    X = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)
    gradient = np.asarray([-1, -outer, 1, outer], np.float32) * np.float32(sign * 2.0**(exponent + hessian_exponent))
    hessian = np.full(4, 2.0**hessian_exponent, np.float32)
    actual, stats = compare(X, np.zeros(4, np.float32), bins=4, quantile=True, rounds=1,
                            params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0, "eta": 1},
                            custom=lambda prediction, matrix: (gradient, hessian))
    document = tree(actual)
    assert len(document["base_weights"]) == 7
    assert document["split_indices"][:3] == [0, 1, 1]
    assert stats[-1]["tuning"]["gpu_tree_levels"] >= 2
    expected = np.asarray([sign * unit, sign * outer * unit, -sign * unit, -sign * outer * unit], np.float32)
    np.testing.assert_array_equal(np.asarray(document["base_weights"][3:], np.float32).view(np.uint32),
                                  expected.view(np.uint32))


@pytest.mark.parametrize("exponent", [64, 100])
def test_nonroot_overflowed_candidate_squares(exponent):
    # An ordinary root split isolates two cancelling large derivatives. Its
    # nonroot candidates have finite weights whose squares overflow, while
    # the accepted root and the resulting model remain finite.
    X = np.asarray([[0, 0], [0, 0], [1, 0], [1, 1]], np.float32)
    small, large = 2.0**(exponent - 40), 2.0**exponent
    gradient = np.asarray([-small, -small, -large, large], np.float32)
    hessian = np.ones(4, np.float32)
    actual, stats = compare(X, np.zeros(4, np.float32), bins=4, rounds=1,
                            params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0, "eta": 1},
                            custom=lambda prediction, matrix: (gradient, hessian))
    document = tree(actual)
    assert len(document["base_weights"]) == 3
    assert document["split_indices"][0] == 0
    assert stats[-1]["tuning"]["gpu_tree_levels"] >= 2


@pytest.mark.parametrize("case", ["cancellation", "zero_hessian", "tiny_hessian", "alpha_boundary"])
def test_custom_derivatives_and_cancellation(case):
    X, y = inputs(rows=257, density=0.5)
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.ones_like(gradient)
    params = {"min_child_weight": 0, "reg_lambda": 0}
    if case == "zero_hessian":
        hessian[::3] = 0
    elif case == "tiny_hessian":
        gradient = np.resize(np.asarray([-3, -1, 1, 3], np.float32), len(X)) * np.float32(2**-20)
        hessian.fill(2**-40)
    elif case == "alpha_boundary":
        gradient = np.resize(np.asarray([-0.5, 0.5, -0.25, 0.25, 0], np.float32), len(X))
        params["reg_alpha"] = 0.5
    compare(X, y, params=params, rounds=2, custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("case", ["subtraction_control", "cpu_driver", "serial", "categorical", "monotone",
                                  "depth_one", "vector", "sampling", "large_bins", "tiled"])
def test_ineligible_paths_keep_original_gain(case):
    X, y = inputs(density=0.95)
    params, metadata, bins = {"metal_fallback_parallel_splits": 0}, {}, 32
    if case == "subtraction_control":
        params["metal_fused_subtraction"] = 0
    elif case == "cpu_driver":
        params["metal_gpu_tree"] = 0
    elif case == "serial":
        params["metal_parallel_splits"] = 0
    elif case == "categorical":
        present = ~np.isnan(X[:, 0])
        X[present, 0] = np.arange(np.count_nonzero(present)) % 5
        metadata["feature_types"] = ["c"] + ["q"] * 6
    elif case == "monotone":
        params["monotone_constraints"] = "(1,0,0,0,0,0,0)"
    elif case == "depth_one":
        params["max_depth"] = 1
    elif case == "vector":
        y = np.column_stack([y, -y]).astype(np.float32)
        params["multi_strategy"] = "multi_output_tree"
    elif case == "sampling":
        params["subsample"] = 0.7
    elif case == "tiled":
        params["metal_scalar_split_bin_tile"] = 32
        bins = 256
    else:
        bins = 257
    compare(X, y, bins=bins, params=params, metadata=metadata, expected_active=0, rounds=2)


@pytest.mark.parametrize("control,gradient", [(0, 0), (1, 1)])
def test_control_and_gradient_fusion_combinations(control, gradient):
    X, y = inputs(density=0.4)
    compare(X, y, rounds=3,
            params={"metal_tree_control_fusion": control, "metal_gradient_root_fusion": gradient})


@pytest.mark.parametrize("workspace", [0, 1])
def test_default_enablement_reset_and_toggle(workspace):
    X, y = inputs(density=0.4)
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=4, max_bin=32, base_score=0, seed=37,
                                 metal_tree_workspace=workspace))
        trace = []
        for iteration, rows in enumerate([257, 769, 0, 513, 769]):
            selected = enabled if iteration != 3 else 0
            if not enabled or iteration:
                model.set_param("metal_fast_gain", selected)
            if iteration == 2:
                model.reset()
            data = xgb.DMatrix(X[:rows], label=y[:rows])
            model.update(data, iteration)
            tuning = profile(model)["tuning"]
            assert tuning["fast_gain"] == selected
            assert tuning["fast_weight"] == 0
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


def test_depth_six_thousand_rounds_preserve_predictions_models_and_rng():
    X, y = inputs(rows=513, density=0.8)
    y += np.random.default_rng(627).normal(scale=0.25, size=len(y)).astype(np.float32)
    compare(X, y, bins=32, rounds=1000, params={"max_depth": 6, "learning_rate": 0.01})
