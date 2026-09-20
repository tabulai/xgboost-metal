"""Scalar split scoring preserves the CPU's mixed binary64/float32 arithmetic."""
import numpy as np
import pytest
import xgboost as xgb

from test_parent_statistics import compare


def train_and_compare(reference, X, gradient, hessian, params, quantile=False,
                      feature_types=None, native_options=None, matrix_reference=None):
    y = np.zeros_like(gradient)
    expected, files = reference(X, y, params, rounds=1, quantile=quantile,
                                feature_types=feature_types, cuts=True, grad=gradient, hess=hessian)
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    options = dict(max_bin=params['max_bin']) if quantile else {}
    if matrix_reference is not None:
        options['ref'] = xgb.QuantileDMatrix(matrix_reference, max_bin=params['max_bin'])
    data = constructor(X, label=y, feature_types=feature_types, **options)
    model = xgb.train({**params, **(native_options or {})}, data, 1,
                      obj=lambda prediction, matrix: (gradient, hessian))
    compare(model, xgb.DMatrix(X, feature_types=feature_types), data, expected, files)


def parameters(bins=4):
    return dict(tree_method='hist', max_depth=1, max_bin=bins, base_score=0,
                eta=1, reg_lambda=1, min_child_weight=1, seed=9)


@pytest.mark.parametrize('matrix', ['raw', 'quantile', 'ref-dense', 'ref-missing'])
@pytest.mark.parametrize('missing_feature', [None, 0, 1])
def test_equal_hessian_scalar_gain(reference, matrix, missing_feature):
    X = np.asarray([[0, 0], [1, 0], [0, 0]], np.float32)
    if missing_feature is not None:
        X[0, missing_feature] = np.nan
    gradient = np.asarray([2**40, -2**40, 3 * 2**-14], np.float32)
    ref = None
    if matrix.startswith('ref-'):
        ref = np.asarray([[0, 0], [1, 0], [0, 0]], np.float32)
        if matrix == 'ref-missing':
            ref[2, 1] = np.nan
    train_and_compare(reference, X, gradient, np.ones_like(gradient), parameters(),
                      quantile=matrix != 'raw', matrix_reference=ref)


@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('missing', [False, True])
@pytest.mark.parametrize('execution', ['serial', 'tiled'])
def test_equal_hessian_scalar_scan_modes(reference, quantile, missing, execution):
    X = np.asarray([[0, 0], [1, 0], [0, 0]], np.float32)
    if missing:
        X[0, 0] = np.nan
    gradient = np.asarray([2**40, -2**40, 3 * 2**-14], np.float32)
    hessian = np.ones_like(gradient)
    options = dict(metal_parallel_splits=0)
    if execution == 'tiled':
        # Zero derivatives add distinct cuts without changing the difficult
        # candidate. More than 32 bins exercise carries between scoring tiles.
        X = np.concatenate([X, np.column_stack([np.arange(2, 66), np.zeros(64)]).astype(np.float32)])
        gradient = np.concatenate([gradient, np.zeros(64, np.float32)])
        hessian = np.concatenate([hessian, np.zeros(64, np.float32)])
        options = dict(metal_scalar_split_bin_tile=32)
    train_and_compare(reference, X, gradient, hessian, parameters(128),
                      quantile=quantile, native_options=options)


@pytest.mark.parametrize('quantile', [False, True])
def test_equal_hessian_categorical_gain(reference, quantile):
    X = np.arange(3, dtype=np.float32).reshape(-1, 1)
    gradient = np.asarray([2**40, -2**40, 3 * 2**-14], np.float32)
    train_and_compare(reference, X, gradient, np.ones_like(gradient), parameters(),
                      quantile=quantile, feature_types=['c'])


@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('strategy', ['one_output_per_tree', 'multi_output_tree'])
@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('monotone', [-1, 1])
@pytest.mark.parametrize('step_limit', [0, 1])
def test_finite_monotone_root_bounds(reference, quantile, strategy, sign, monotone, step_limit):
    # The unconstrained Newton step is 2^140 and overflows float32. A monotonic
    # evaluator starts with finite FLT_MAX bounds; max_delta_step acts first.
    X = np.zeros((2, 1), np.float32)
    first = np.full(2, sign * 2**100, np.float32)
    gradient = np.column_stack([first, -first])
    hessian = np.full_like(gradient, 2**-40)
    params = dict(parameters(), multi_strategy=strategy, max_leaves=1,
                  reg_lambda=0, min_child_weight=0,
                  monotone_constraints=f'({monotone})', max_delta_step=step_limit)
    train_and_compare(reference, X, gradient, hessian, params, quantile=quantile)


@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('parameter', ['min_child_weight', 'reg_lambda', 'reg_alpha', 'max_delta_step'])
def test_negative_zero_scalar_parameters(reference, quantile, parameter):
    X = np.arange(3, dtype=np.float32).reshape(-1, 1)
    gradient = np.asarray([2, -2, 0.25], np.float32)
    params = parameters()
    params[parameter] = -0.0
    train_and_compare(reference, X, gradient, np.ones_like(gradient), params, quantile=quantile)
