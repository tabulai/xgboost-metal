"""Scalar weights retain the CPU's binary64-to-float32 double rounding."""
from fractions import Fraction
import json

import numpy as np
import pytest
import xgboost as xgb

from test_parent_statistics import compare


def float32_integer_parts(value):
    """Express an exact positive integer as nonnegative float32 row values."""
    parts = []
    while value:
        part = np.float32(value)
        if int(part) > value:
            part = np.nextafter(part, np.float32(0))
        parts.append(part)
        value -= int(part)
    return parts


@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('parity', [0, 1])
@pytest.mark.parametrize('inside_cell', [False, True])
def test_scalar_weight_double_rounding(reference, quantile, split, sign, parity, inside_cell):
    # Put the exact quotient on the side of a float32 midpoint opposite its
    # tie-to-even result. The inner pair rounds onto that midpoint in binary64;
    # the outer pair stays outside its binary64 rounding cell.
    q = 2**23 + parity
    delta = (1 if parity == 0 else -1) * (2**23 + (-1 if inside_cell else 1))
    modulus = 2**24
    denominator = 2**52 + (-delta * pow(2 * q + 1, -1, modulus)) % modulus
    numerator, remainder = divmod((2 * q + 1) * denominator + delta, modulus)
    assert remainder == 0
    exact = Fraction(numerator, denominator)
    endpoints = [Fraction(q, 2**23), Fraction(q + 1, 2**23)]
    nearest = min(endpoints, key=lambda endpoint: abs(exact - endpoint))
    direct = np.float32(float(nearest))
    cpu_weight = np.float32(numerator / denominator)
    assert bool(cpu_weight.view(np.uint32) != direct.view(np.uint32)) == inside_cell

    gradient = float32_integer_parts(numerator)
    hessian = float32_integer_parts(denominator)
    rows = max(len(gradient), len(hessian))
    gradient = np.asarray(gradient + [0] * (rows - len(gradient)), np.float32) * -sign
    hessian = np.asarray(hessian + [0] * (rows - len(hessian)), np.float32)
    assert sum(map(float, gradient)) == -sign * numerator
    assert sum(map(float, hessian)) == denominator
    X = np.zeros((rows + int(split), 1), np.float32)
    if split:
        X[-1, 0] = 1
        gradient = np.append(gradient, np.float32(0))
        hessian = np.append(hessian, np.float32(2**52))
    y = np.zeros_like(gradient)
    params = dict(tree_method='hist', max_depth=1, max_bin=4, base_score=0,
                  eta=1, reg_lambda=0, min_child_weight=0, seed=9)
    expected, files = reference(X, y, params, rounds=1, quantile=quantile,
                                cuts=True, grad=gradient, hess=hessian)
    tree = json.loads((files / 'model.json').read_text())['learner']['gradient_booster']['model']['trees'][0]
    assert len(tree['base_weights']) == (3 if split else 1)
    wanted = np.float32(sign * cpu_weight)
    actual = np.float32(tree['base_weights'][1 if split else 0])
    assert actual.view(np.uint32) == wanted.view(np.uint32)
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = constructor(X, label=y, **({'max_bin': 4} if quantile else {}))
    model = xgb.train(params, data, 1, obj=lambda prediction, matrix: (gradient, hessian))
    compare(model, xgb.DMatrix(X), data, expected, files)
