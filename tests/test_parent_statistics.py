"""Scalar split application preserves the reference's rounded child statistics."""
import json

import numpy as np
import pytest
import xgboost as xgb

from test_large_bins import check_model


def compare(model, prediction_data, training_data, expected, files):
    own = json.loads(model.save_raw(raw_format='json'))
    wanted = json.loads((files / 'model.json').read_text())
    check_model(own, wanted)
    for name, options in [('prediction', {}), ('margin', {'output_margin': True}),
                          ('leaf', {'pred_leaf': True})]:
        actual = model.predict(prediction_data, **options)
        np.testing.assert_array_equal(actual.view(np.uint32), expected[name].view(np.uint32))
    with np.load(files / 'cuts.npz') as cuts:
        pointer, values = training_data.get_quantile_cut()
        np.testing.assert_array_equal(pointer, cuts['indptr'])
        np.testing.assert_array_equal(values.view(np.uint32), cuts['values'].view(np.uint32))


@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('categorical', [False, True])
@pytest.mark.parametrize('split', [False, True])
def test_applied_scalar_parent_rounds_children(reference, quantile, categorical, split):
    # Summing these three bins in order retains the small final term exactly.
    # Converting the selected large children to binary64 before adding rounds
    # it differently. Only applying a split changes the stored parent weight.
    X = np.arange(3, dtype=np.float32).reshape(-1, 1)
    y = np.zeros(3, np.float32)
    gradient = np.asarray([2**40, -2**40, 3 * 2**-14], np.float32)
    # Unequal Hessians give a unique categorical winner. The equal-Hessian
    # extreme-gradient gain tie is retained as a separate numerical diagnostic.
    hessian = np.asarray([1, 2, 1], np.float32)
    types = ['c'] if categorical else None
    params = dict(tree_method='hist', max_depth=1, max_bin=4, base_score=0,
                  eta=1, reg_lambda=1, min_child_weight=1, seed=9)
    if not split:
        params['max_leaves'] = 1
    expected, files = reference(X, y, params, rounds=1, feature_types=types,
                                quantile=quantile, cuts=True, grad=gradient, hess=hessian)
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = constructor(X, label=y, feature_types=types, **({'max_bin': 4} if quantile else {}))
    model = xgb.train(params, data, 1, obj=lambda prediction, matrix: (gradient, hessian))
    compare(model, xgb.DMatrix(X, feature_types=types), data, expected, files)


@pytest.mark.parametrize('quantile', [False, True])
def test_weighted_benchmark_root_statistic(reference, quantile):
    # Keep the installed benchmark's generator, then retain its first four
    # features. The same selected root exposes a one-ULP stored-weight gap.
    rng = np.random.default_rng(43)
    original = rng.normal(size=(100000, 32)).astype(np.float32)
    noise = rng.normal(0, 0.1, len(original))
    y = (2 * original[:, 0] + original[:, 1] ** 2 + noise).astype(np.float32)
    X = original[:, :4].copy()
    weight = np.random.default_rng(144).uniform(0.1, 2.0, len(X)).astype(np.float32)
    params = dict(tree_method='hist', max_depth=1, max_bin=4096, seed=9)
    expected, files = reference(X, y, params, rounds=1, quantile=quantile, cuts=True, weight=weight)
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = constructor(X, label=y, weight=weight, **({'max_bin': 4096} if quantile else {}))
    model = xgb.train(params, data, 1)
    wanted = json.loads((files / 'model.json').read_text())['learner']['gradient_booster']['model']['trees'][0]
    assert int(np.asarray(wanted['base_weights'][0], np.float32).view(np.uint32)) == 2990026907
    compare(model, xgb.DMatrix(X), data, expected, files)
