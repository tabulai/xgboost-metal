"""Certified quotient estimates retain the restoring divider's exact outputs."""
import itertools
import json

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, profile, snapshot
from test_histogram_missing import inputs, layout
from test_scalar_weight import float32_integer_parts
from test_split_subtraction_fusion import fit


def compare(X, y, *, params=None, expected_active=1, **kwargs):
    params = params or {}
    expected, control_stats = fit(X, y, 1, params={**params, "metal_fast_weight": 0}, **kwargs)
    actual, stats = fit(X, y, 1, params={**params, "metal_fast_weight": 1}, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["fast_weight"] == 0 for item in control_stats)
    assert all(item["tuning"]["fast_weight"] == expected_active for item in stats)
    return actual, stats


def tree(result):
    model = xgb.Booster(model_file=bytearray(result["model"]))
    return json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"][0]


@pytest.mark.parametrize("bins,kind,skip", list(itertools.product([2, 32, 256], ["dense", "csr"], [0, 1])))
def test_bin_layout_and_sparse_candidate_modes(bins, kind, skip):
    X, y = inputs(rows=1025, density=0.4)
    X = np.nan_to_num(X) if kind == "dense" else layout(X, kind)
    compare(X, y, bins=bins, params={"metal_skip_empty_bins": skip}, rounds=3)


@pytest.mark.parametrize("depth,quantile", list(itertools.product([2, 6, 10], [False, True])))
def test_depth_and_quantile_parent_statistics(depth, quantile):
    X, y = inputs(density=0.8)
    compare(X, y, bins=32, quantile=quantile, params={"max_depth": depth}, rounds=3)


REGULARIZATION = [
    {"reg_lambda": 0}, {"reg_lambda": 0.125}, {"reg_lambda": 2**30},
    {"reg_lambda": 2**-80}, {"reg_alpha": 0.125}, {"reg_alpha": 1000},
    {"max_delta_step": 0.125}, {"max_delta_step": 2.0**120},
    {"min_child_weight": -0.0},
    {"reg_lambda": -0.0, "reg_alpha": -0.0, "max_delta_step": -0.0},
]


@pytest.mark.parametrize("params", REGULARIZATION)
def test_regularization_clamps_and_zero_signs(params):
    X, y = inputs(density=0.6)
    compare(X, y, params=params, rounds=3)


@pytest.mark.parametrize("sign,parity,inside_cell,missing",
                         list(itertools.product([-1, 1], [0, 1], [False, True], [False, True])))
def test_nonroot_binary64_midpoint_cells(sign, parity, inside_cell, missing):
    # Preserve the original double-rounding fixture, but place its quotient in
    # a grandchild: root search stays on the original divider, while the two
    # depth-one splits exercise the new candidate-weight implementation.
    q = 2**23 + parity
    delta = (1 if parity == 0 else -1) * (2**23 + (-1 if inside_cell else 1))
    denominator = 2**52 + (-delta * pow(2 * q + 1, -1, 2**24)) % 2**24
    numerator, remainder = divmod((2 * q + 1) * denominator + delta, 2**24)
    assert remainder == 0
    rows, gradient, hessian = [], [], []
    for bucket, factor in enumerate([-1, -3, 1, 3]):
        g = float32_integer_parts(abs(factor) * numerator)
        h = float32_integer_parts(denominator)
        count = max(len(g), len(h))
        rows.extend([[bucket // 2, bucket % 2]] * count)
        gradient.extend([sign * np.sign(factor) * value for value in g] + [0] * (count - len(g)))
        hessian.extend(h + [0] * (count - len(h)))
    X = np.asarray(rows, np.float32)
    if missing:
        X = np.column_stack([X, np.full(len(X), np.nan, np.float32)])
    gradient, hessian = np.asarray(gradient, np.float32), np.asarray(hessian, np.float32)
    actual, stats = compare(X, np.zeros(len(X), np.float32), bins=4, quantile=True, rounds=1,
                            params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0, "eta": 1},
                            custom=lambda prediction, matrix: (gradient, hessian))
    document = tree(actual)
    assert len(document["base_weights"]) == 7
    assert stats[-1]["tuning"]["gpu_tree_levels"] >= 2
    wanted = np.float32(sign * (numerator / denominator))
    assert np.float32(document["base_weights"][3]).view(np.uint32) == wanted.view(np.uint32)


@pytest.mark.parametrize("exponent,sign", list(itertools.product([-151, -150, -149, -140, -130, -126], [-1, 1])))
def test_nonroot_subnormal_rounding_and_signed_underflow(exponent, sign):
    # The gradient and Hessian spans stay within 55 bits so quantization keeps
    # the small numerator. Feature zero wins the root tie; its left child's
    # feature-one split stores a tiny weight using the nonroot divider.
    small_gradient = 2.0 ** (exponent + 125)
    large_gradient = small_gradient * 2**55
    X = np.asarray([[0, 0], [0, 1], [1, 0]], np.float32)
    gradient = np.asarray([-small_gradient, -large_gradient, large_gradient], np.float32) * sign
    hessian = np.asarray([2**125, 2**70, 2**70], np.float32)
    actual, _ = compare(X, np.zeros(3, np.float32), bins=4, rounds=1,
                        params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0, "eta": 1},
                        custom=lambda prediction, matrix: (gradient, hessian))
    document = tree(actual)
    assert len(document["base_weights"]) == 5
    wanted = np.float32(sign * 2.0**exponent)
    assert np.float32(document["base_weights"][3]).view(np.uint32) == wanted.view(np.uint32)


@pytest.mark.parametrize("case", ["cancelled", "overflow_clamp", "tiny_hessian", "zero", "zero_hessian", "extreme_gain"])
def test_extreme_custom_gradient_and_hessian_cases(case):
    X, y = inputs(rows=257, density=0.5)
    params = {"min_child_weight": 0, "reg_lambda": 0}
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14], np.float32), len(X))
    hessian = np.ones_like(gradient)
    if case == "overflow_clamp":
        gradient = np.where(np.arange(len(X)) % 2, 2.0**100, -(2.0**100)).astype(np.float32)
        hessian.fill(2**-40)
        params["max_delta_step"] = 0.5
    elif case == "tiny_hessian":
        gradient = np.where(np.arange(len(X)) % 2, 2**-20, -(2**-20)).astype(np.float32)
        hessian.fill(2**-40)
    elif case == "zero":
        gradient.fill(0)
    elif case == "zero_hessian":
        hessian[::3] = 0
    elif case == "extreme_gain":
        gradient = np.where(np.arange(len(X)) % 2, 2.0**120, -(2.0**120)).astype(np.float32)
        hessian.fill(2**100)
    compare(X, y, params=params, rounds=2,
            custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("case", ["subtraction_control", "cpu_driver", "serial", "categorical", "monotone",
                                  "depth_one", "vector", "sampling", "large_bins"])
def test_ineligible_paths_keep_original_division(case):
    X, y = inputs(density=0.95)
    params, metadata, bins = {}, {}, 32
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
    else:
        bins = 257
    compare(X, y, bins=bins, params=params, metadata=metadata,
            expected_active=int(case in ("monotone", "sampling")), rounds=2)


@pytest.mark.parametrize("control_fusion,histogram_missing,gradient_fusion",
                         list(itertools.product([0, 1], repeat=3)))
def test_other_fusion_flags_keep_the_same_weight_bits(control_fusion, histogram_missing, gradient_fusion):
    X, y = inputs(density=0.4)
    compare(X, y, params={"metal_tree_control_fusion": control_fusion,
                         "metal_histogram_missing": histogram_missing,
                         "metal_gradient_root_fusion": gradient_fusion}, rounds=2)


@pytest.mark.parametrize("workspace", [0, 1])
def test_explicit_enablement_reset_and_toggle(workspace):
    X, y = inputs(density=0.4)
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=4, max_bin=32, base_score=0, seed=37,
                                 metal_tree_workspace=workspace))
        trace = []
        for iteration, rows in enumerate([257, 769, 0, 513, 769]):
            selected = enabled if iteration != 3 else 0
            model.set_param("metal_fast_weight", selected)
            if iteration == 2:
                model.reset()
            data = xgb.DMatrix(X[:rows], label=y[:rows])
            model.update(data, iteration)
            assert profile(model)["tuning"]["fast_weight"] == selected
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


def test_depth_six_thousand_rounds_preserve_predictions_models_and_rng():
    X, y = inputs(rows=513, density=0.8)
    rng = np.random.default_rng(627)
    y += rng.normal(scale=0.25, size=len(y)).astype(np.float32)
    compare(X, y, bins=32, rounds=1000, params={"max_depth": 6, "learning_rate": 0.01})
