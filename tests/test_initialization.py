import numpy as np
import pickle
import pytest
import xgboost as xgb


@pytest.mark.parametrize('objective', ['reg:squarederror', 'binary:logistic', 'count:poisson',
                                      'reg:gamma', 'reg:tweedie', 'multi:softprob'])
def test_zero_round_intercept(reference, objective):
    X = np.arange(93, dtype=np.float32).reshape(31, 3)
    y = np.arange(len(X), dtype=np.float32) + 1
    params = {'objective': objective}
    if objective == 'binary:logistic':
        y = (y > 19).astype(np.float32)
    if objective.startswith('multi:'):
        params['num_class'] = 3
        y %= 3
    expected, _ = reference(X, y, params, rounds=0)
    data = xgb.DMatrix(X, label=y)
    actual = xgb.train(params, data, 0)
    np.testing.assert_allclose(actual.predict(data), expected['prediction'], atol=2e-6)
    np.testing.assert_allclose(actual.predict(data, output_margin=True), expected['margin'], atol=2e-6)


def test_unfitted_prediction_requires_training_initialization():
    d = xgb.DMatrix(np.ones((7, 2)), label=np.arange(7))
    b = xgb.Booster(cache=[d])
    with pytest.raises(xgb.core.XGBoostError, match='not fitted'):
        b.predict(d)
    np.testing.assert_array_equal(b.predict(d, training=True), np.full(7, 0.5))


def test_custom_objective_uses_default_intercept(reference):
    rng = np.random.default_rng(41)
    X = rng.normal(size=(71, 4)).astype(np.float32)
    y = (20 + X[:, 0] - X[:, 1]).astype(np.float32)
    params = {'max_depth': 3}
    expected, _ = reference(X, y, params, custom_square=True)

    def objective(prediction, data):
        return prediction - y, np.ones_like(prediction)

    for labels in [y, None]:
        d = xgb.DMatrix(X, label=labels)
        model = xgb.train(params, d, 5, obj=objective)
        np.testing.assert_allclose(model.predict(d), expected['prediction'], atol=2e-5)


def test_boost_from_average_overrides_explicit_score(reference):
    X = np.arange(35, dtype=np.float32).reshape(7, 5)
    y = np.arange(7, dtype=np.float32) + 4
    params = {'base_score': 0.1, 'boost_from_average': 1, 'max_depth': 2}
    expected, _ = reference(X, y, params, rounds=2)
    d = xgb.DMatrix(X, label=y)
    b = xgb.train(params, d, 2)
    np.testing.assert_allclose(b.predict(d), expected['prediction'], atol=2e-6)


def test_custom_objective_sampling_matches_builtin():
    rng = np.random.default_rng(511)
    X = rng.normal(size=(311, 4)).astype(np.float32)
    y = X[:, 0] - X[:, 1]
    data = xgb.DMatrix(X, label=y)
    params = {'base_score': 0.5, 'max_depth': 3, 'subsample': 0.65, 'seed': 72}
    ordinary = xgb.train(params, data, 5)
    custom = xgb.train(params, data, 5, obj=lambda prediction, d: (prediction - y, np.ones_like(prediction)))
    np.testing.assert_array_equal(ordinary.predict(data), custom.predict(data))


@pytest.mark.parametrize('objective', ['binary:logistic', 'count:poisson', 'reg:gamma', 'reg:tweedie'])
def test_linked_intercept_roundtrip_is_exact(objective):
    rng = np.random.default_rng(91)
    X = rng.normal(size=(191, 5)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] > 0).astype(np.float32)
    if objective != 'binary:logistic':
        y = np.exp(X[:, 0] / 4)
    d = xgb.DMatrix(X, label=y)
    b = xgb.train({'objective': objective, 'max_depth': 3}, d, 8)
    predicted = b.predict(d)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(b)).predict(d), predicted)
    for kind in ['json', 'ubj']:
        restored = xgb.Booster(model_file=b.save_raw(raw_format=kind))
        np.testing.assert_array_equal(restored.predict(d), predicted)


def test_zero_poisson_score_roundtrip(reference):
    X = np.arange(35, dtype=np.float32).reshape(7, 5)
    y = np.zeros(7, dtype=np.float32)
    params = {'objective': 'count:poisson'}
    expected, _ = reference(X, y, params, rounds=2)
    d = xgb.DMatrix(X, label=y)
    b = xgb.train(params, d, 2)
    np.testing.assert_array_equal(b.predict(d), expected['prediction'])
    restored = xgb.Booster(model_file=b.save_raw())
    np.testing.assert_array_equal(restored.predict(d), expected['prediction'])


def test_label_validation_cache_is_invalidated():
    d = xgb.DMatrix(np.arange(30, dtype=np.float32).reshape(10, 3), label=np.arange(10) % 2)
    b = xgb.Booster({'objective': 'binary:logistic'}, cache=[d])
    b.update(d, 0)
    d.set_label(np.full(10, 2))
    with pytest.raises(xgb.core.XGBoostError, match='label must be in'):
        b.update(d, 1)


@pytest.mark.parametrize('rounds', [0, 2])
@pytest.mark.parametrize('changes', [{'base_score': 0.7}, {'objective': 'binary:logistic'},
                                    {'num_target': 2}, {'base_score': '[0.1,0.9]', 'num_class': 2}])
def test_reconfigure_intercept_objective_and_output_shape(reference, rounds, changes):
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    params = dict(base_score=0.25, max_depth=1)
    expected, _ = reference(X, y, params, rounds=rounds, post_params=changes)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, rounds)
    # Refreshes must invalidate native prediction/model buffers, including after a
    # prior prediction. Use a fresh oracle DMatrix to avoid its stale-cache behavior.
    model.predict(data)
    model.set_param(changes)
    np.testing.assert_allclose(model.predict(data, output_margin=True), expected['margin'], atol=3e-6)
    np.testing.assert_allclose(model.inplace_predict(X), expected['prediction'], atol=3e-6)
    restored = xgb.Booster(model_file=model.save_raw())
    np.testing.assert_array_equal(restored.predict(data), model.predict(data))


def test_num_class_selects_default_multiclass_objective(reference):
    rng = np.random.default_rng(492)
    X = rng.normal(size=(127, 4)).astype(np.float32)
    y = np.argmax(X[:, :3], axis=1).astype(np.float32)
    params = dict(num_class=3, max_depth=3)
    expected, _ = reference(X, y, params, rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 3)
    np.testing.assert_array_equal(model.predict(data), expected['prediction'])
    np.testing.assert_allclose(model.predict(data, output_margin=True), expected['margin'], atol=3e-6)
