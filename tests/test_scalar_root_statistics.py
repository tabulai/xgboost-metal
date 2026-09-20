"""Dense scalar roots retain the CPU histogram's ordered binary64 statistics."""
import json

import numpy as np
import pytest
import xgboost as xgb

from test_parent_statistics import compare


@pytest.mark.parametrize('matrix', ['raw', 'quantile', 'ref-dense', 'ref-missing'])
@pytest.mark.parametrize('missing_feature', [None, 0, 1])
@pytest.mark.parametrize('split', [False, True])
def test_scalar_root_density_and_bin_order(reference, matrix, missing_feature, split):
    X = np.asarray([[0, 0], [1, 0], [0, 0]], np.float32)
    if missing_feature is not None:
        X[0, missing_feature] = np.nan
    y = np.zeros(len(X), np.float32)
    # The first bin rounds (large + residual) before cancellation with the
    # second bin. Sparse roots instead add the input gradients in row order.
    gradient = np.asarray([2**40, -2**40, 3 * 2**-14], np.float32)
    # Unequal Hessians isolate root accumulation from the separately retained
    # extreme-gradient split-gain and missing-direction tie diagnostics.
    hessian = np.asarray([1, 2, 1], np.float32)
    params = dict(tree_method='hist', max_depth=1, max_bin=4, base_score=0,
                  eta=1, reg_lambda=1, min_child_weight=1, seed=9)
    if not split:
        params['max_leaves'] = 1
    expected, files = reference(X, y, params, rounds=1, quantile=matrix != 'raw',
                                cuts=True, grad=gradient, hess=hessian)
    if matrix == 'raw':
        data = xgb.DMatrix(X, label=y)
    else:
        options = {}
        if matrix.startswith('ref-'):
            # These references have the same unique values and cuts. Their
            # density must not replace the current matrix's density.
            ref_values = np.asarray([[0, 0], [1, 0], [0, 0]], np.float32)
            if matrix == 'ref-missing':
                ref_values[2, 1] = np.nan
            options['ref'] = xgb.QuantileDMatrix(ref_values, max_bin=4)
        data = xgb.QuantileDMatrix(X, label=y, max_bin=4, **options)
    model = xgb.train(params, data, 1, obj=lambda prediction, matrix: (gradient, hessian))
    compare(model, xgb.DMatrix(X), data, expected, files)


@pytest.mark.parametrize('quantile', [False, True])
def test_categorical_stopped_root_bin_order(reference, quantile):
    X = np.asarray([0, 0, 1, 1], np.float32).reshape(-1, 1)
    y = np.zeros(len(X), np.float32)
    gradient = np.asarray([2**40, 2**-13, -2**40, 3 * 2**-13], np.float32)
    hessian = np.ones_like(gradient)
    params = dict(tree_method='hist', max_depth=1, max_leaves=1, max_bin=4,
                  base_score=0, eta=1, reg_lambda=1, seed=9)
    expected, files = reference(X, y, params, rounds=1, quantile=quantile,
                                feature_types=['c'], cuts=True, grad=gradient, hess=hessian)
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = constructor(X, label=y, feature_types=['c'], **({'max_bin': 4} if quantile else {}))
    model = xgb.train(params, data, 1, obj=lambda prediction, matrix: (gradient, hessian))
    tree = json.loads((files / 'model.json').read_text())['learner']['gradient_booster']['model']['trees'][0]
    assert int(np.asarray(tree['base_weights'][0], np.float32).view(np.uint32)) == 3097074074
    compare(model, xgb.DMatrix(X, feature_types=['c']), data, expected, files)


@pytest.mark.parametrize('quantile', [False, True])
def test_weighted_million_root_bin_order(reference, quantile):
    rng = np.random.default_rng(43)
    original = rng.normal(size=(1000000, 64)).astype(np.float32)
    noise = rng.normal(0, 0.1, len(original))
    y = (2 * original[:, 0] + original[:, 1] ** 2 + noise).astype(np.float32)
    X = original[:, :4].copy()
    del original
    weight = np.random.default_rng(144).uniform(0.1, 2, len(X)).astype(np.float32)
    params = dict(tree_method='hist', max_depth=1, max_bin=256, seed=9)
    expected, files = reference(X, y, params, rounds=1, quantile=quantile, cuts=True, weight=weight)
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = constructor(X, label=y, weight=weight, **({'max_bin': 256} if quantile else {}))
    model = xgb.train(params, data, 1)
    tree = json.loads((files / 'model.json').read_text())['learner']['gradient_booster']['model']['trees'][0]
    assert int(np.asarray(tree['base_weights'][0], np.float32).view(np.uint32)) == 2915099116
    compare(model, xgb.DMatrix(X), data, expected, files)
