"""Gradient quantization keeps small residuals and finite magnitude bounds."""
import json

import numpy as np
import pytest
import xgboost as xgb

from test_parent_statistics import compare


@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('strategy', ['one_output_per_tree', 'multi_output_tree'])
@pytest.mark.parametrize('case', ['cancelled-residual', 'finite-overflow'])
def test_gradient_magnitude_precision(reference, quantile, strategy, case):
    if case == 'cancelled-residual':
        # A float32 residual between two adjacent old fixed-point units.
        # Put it after the cancelling terms so the CPU row/bin sum is exact.
        first = np.asarray([2**16, -2**16, 6.118193027759844e-7], np.float32)
    else:
        # Every gradient is finite, while a float32 magnitude reduction of a
        # full tile overflows. The regularized root weight remains finite.
        first = np.full(257, np.finfo(np.float32).max, np.float32)
    gradient = np.column_stack([first, -first])
    hessian = np.ones_like(gradient)
    X = np.zeros((len(first), 1), np.float32)
    y = np.zeros_like(gradient)
    params = dict(tree_method='hist', multi_strategy=strategy, max_depth=1,
                  max_leaves=1, max_bin=4, base_score=0, eta=1, reg_lambda=1,
                  min_child_weight=1, seed=9)
    expected, files = reference(X, y, params, rounds=1, quantile=quantile,
                                cuts=True, grad=gradient, hess=hessian)
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = constructor(X, label=y, **({'max_bin': 4} if quantile else {}))
    model = xgb.train(params, data, 1, obj=lambda prediction, matrix: (gradient, hessian))
    compare(model, xgb.DMatrix(X), data, expected, files)


@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('strategy', ['one_output_per_tree', 'multi_output_tree'])
def test_weighted_output_root_gradient_precision(reference, quantile, strategy):
    # Reproduce the four-output benchmark, retaining its first four features
    # and one split. The second target exposed the missing quantization bit.
    rng = np.random.default_rng(43)
    original = rng.normal(size=(100000, 32)).astype(np.float32)
    outputs = []
    for target in range(4):
        response = 2 * original[:, target] + original[:, target + 1] ** 2
        outputs.append((response + rng.normal(0, 0.1, len(original))).astype(np.float32))
    y = np.column_stack(outputs)
    X = original[:, :4].copy()
    weight = np.random.default_rng(144).uniform(0.1, 2, len(X)).astype(np.float32)
    params = dict(tree_method='hist', multi_strategy=strategy,
                  max_depth=1, max_bin=4096, seed=9)
    expected, files = reference(X, y, params, rounds=1, quantile=quantile, cuts=True, weight=weight)
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = constructor(X, label=y, weight=weight, **({'max_bin': 4096} if quantile else {}))
    model = xgb.train(params, data, 1)
    trees = json.loads((files / 'model.json').read_text())['learner']['gradient_booster']['model']['trees']
    value = trees[0]['base_weights'][1] if strategy == 'multi_output_tree' else trees[1]['base_weights'][0]
    wanted = 793275639 if strategy == 'multi_output_tree' else 793275633
    assert int(np.asarray(value, np.float32).view(np.uint32)) == wanted
    compare(model, xgb.DMatrix(X), data, expected, files)
