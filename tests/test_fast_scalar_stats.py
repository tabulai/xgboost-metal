"""Specialized integer conversion preserves the scalar scoring boundary."""
import itertools

import numpy as np
import pytest

from test_gpu_tree import assert_identical
from test_histogram_missing import inputs
from test_split_subtraction_fusion import fit


@pytest.mark.parametrize("threads,bins,missing", list(itertools.product(
    [32, 256], [2, 31, 33, 128, 255, 256], [False, True])))
def test_integer_conversion_preserves_models(threads, bins, missing):
    X, y = inputs(rows=1025, density=0.3)
    if not missing:
        X = np.nan_to_num(X)
    params = {"metal_split_threads": threads, "metal_compact_splits": 0, "max_depth": 10}
    expected, _ = fit(X, y, 1, bins=bins, rounds=3, params={**params, "metal_fast_stats": 0})
    actual, profiles = fit(X, y, 1, bins=bins, rounds=3, params={**params, "metal_fast_stats": 1})
    assert_identical(expected, actual)
    assert all(p["tuning"]["fast_stats"] == 1 for p in profiles)


@pytest.mark.parametrize("threads,exponent", list(itertools.product([32, 256], [-70, -20, 0, 40, 70])))
def test_signed_statistics_and_scale_rounding(threads, exponent):
    X, y = inputs(rows=2051, density=0.3)
    gradient = np.resize(np.asarray([1, -1, 1.0000001, -0.125, 0], np.float32), len(X))
    gradient *= np.float32(2.0**exponent)
    hessian = np.resize(np.asarray([1, 2, 0, 2**-20], np.float32), len(X))
    hessian *= np.float32(2.0**max(-50, min(80, exponent)))
    params = {"metal_split_threads": threads, "metal_compact_splits": 0, "max_depth": 10, "min_child_weight": 0,
              "reg_lambda": 0, "reg_alpha": 0, "max_delta_step": 0}
    custom = lambda prediction, matrix: (gradient, hessian)
    expected, _ = fit(X, y, 1, rounds=3, params={**params, "metal_fast_stats": 0}, custom=custom)
    actual, _ = fit(X, y, 1, rounds=3, params={**params, "metal_fast_stats": 1}, custom=custom)
    assert_identical(expected, actual)


@pytest.mark.parametrize("control", [
    {"metal_gpu_tree": 0}, {"metal_fused_subtraction": 0}, {"metal_fast_gain": 0},
    {"metal_fast_weight": 1}, {"metal_parallel_splits": 0}, {"max_depth": 1},
])
def test_unsupported_scoring_keeps_original_conversion(control):
    X, y = inputs(rows=513, density=0.5)
    control = {**control, "metal_fallback_parallel_splits": 0}
    expected, _ = fit(X, y, 1, rounds=2, params={**control, "metal_fast_stats": 0})
    actual, profiles = fit(X, y, 1, rounds=2, params={**control, "metal_fast_stats": 1})
    assert_identical(expected, actual)
    assert all(p["tuning"]["fast_stats"] == 0 for p in profiles)
