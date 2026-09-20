import json
import pickle

import numpy as np
import pytest
import xgboost as xgb

from test_vector_training import compare_vector_models, training_data
from test_vector_reduced import ProjectedSquare


@pytest.mark.parametrize("objective", ["binary:logistic", "binary:hinge", "multi:softprob",
                                      "reg:squaredlogerror", "count:poisson", "reg:gamma",
                                      "reg:tweedie", "reg:pseudohubererror", "reg:absoluteerror",
                                      "reg:quantileerror", "reg:expectileerror", "reg:normal"])
def test_vector_objectives(reference, objective):
    X, y, weight, _ = training_data(3)
    params = dict(multi_strategy="multi_output_tree", objective=objective, max_depth=3, max_bin=16)
    if objective in ["binary:logistic", "binary:hinge"]:
        y = (y > np.median(y, axis=0)).astype(np.float32)
    elif objective == "multi:softprob":
        y = (np.arange(len(X)) % 3).astype(np.float32)
        params["num_class"] = 3
    elif objective in ["reg:squaredlogerror", "count:poisson", "reg:gamma", "reg:tweedie"]:
        y = np.abs(y) / np.float32(10) + np.float32(0.1)
    elif objective in ["reg:quantileerror", "reg:expectileerror", "reg:normal"]:
        y = y[:, 0]
        if objective != "reg:normal":
            params["quantile_alpha" if objective == "reg:quantileerror" else "expectile_alpha"] = [0.2, 0.5, 0.8]
    expected, files = reference(X, y, params, rounds=3, weight=weight)
    data = xgb.DMatrix(X, label=y, weight=weight)
    model = xgb.train(params, data, 3)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(model.predict(data, output_margin=True), expected["margin"], rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    for name, options in [("contributions", {"pred_contribs": True}), ("interactions", {"pred_interactions": True})]:
        np.testing.assert_allclose(model.predict(data, **options), expected[name], rtol=4e-5, atol=4e-5)


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("normalize", ["tree", "forest"])
def test_vector_dropout(reference, reduced, normalize):
    X, y, _, types = training_data(3, True)
    projection = np.asarray([[1, 0.5], [-0.5, 1], [0.5, -0.5]], np.float32)
    params = dict(multi_strategy="multi_output_tree", max_depth=3, max_bin=16, max_cat_to_onehot=2,
                  seed=79, rate_drop=0.45, one_drop=True, num_parallel_tree=2, sample_type="weighted",
                  normalize_type=normalize, subsample=0.7, sampling_method="gradient_based")
    meta = {"projection": projection} if reduced else {}
    expected, files = reference(X, y, params, rounds=6, feature_types=types, **meta)
    data = xgb.DMatrix(X, label=y, feature_types=types, enable_categorical=True)
    model = xgb.train(params, data, 6, obj=ProjectedSquare(projection) if reduced else None)
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    compare_vector_models(model, files / "model.json")
    own = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["weight_drop"]
    target = json.loads((files / "model.json").read_text())["learner"]["gradient_booster"]["model"]["weight_drop"]
    np.testing.assert_array_equal(np.asarray(own, np.float32), np.asarray(target, np.float32))
    np.testing.assert_allclose(model.predict(data, pred_contribs=True), expected["contributions"], rtol=4e-5, atol=4e-5)


@pytest.mark.parametrize("strategy", [None, "multi_output_tree"])
@pytest.mark.parametrize("kind", ["json", "ubj"])
def test_vector_file_continuation(reference, tmp_path, strategy, kind):
    X, y, weight, _ = training_data(3)
    params = dict(multi_strategy="multi_output_tree", max_depth=3, max_bin=16, seed=132,
                  subsample=0.7, colsample_bytree=0.8)
    _, files = reference(X, y, params, rounds=3, weight=weight)
    initial = files / f"model.{kind}"
    continuation = {k: v for k, v in params.items() if k != "multi_strategy"}
    if strategy is not None:
        continuation["multi_strategy"] = strategy
    expected, result = reference(X, y, continuation, model=initial, continue_rounds=3, weight=weight)
    data = xgb.DMatrix(X, label=y, weight=weight)
    model = xgb.train(continuation, data, 3, xgb_model=initial)
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    compare_vector_models(model, result / "model.json")
    destination = tmp_path / f"mixed.{kind}"
    model.save_model(destination)
    returned, _ = reference(X, y, {}, model=destination)
    np.testing.assert_array_equal(returned["prediction"], expected["prediction"])


@pytest.mark.parametrize("strategy", ["multi_output_tree", "one_output_per_tree"])
def test_vector_pickle_continuation(reference, strategy):
    X, y, _, _ = training_data(3)
    params = dict(multi_strategy="multi_output_tree", max_depth=3, max_bin=16, seed=183,
                  subsample=0.7, colsample_bytree=0.8)
    expected, files = reference(X, y, params, rounds=6, param_updates={3: {"multi_strategy": strategy}})
    data = xgb.DMatrix(X, label=y)
    first = xgb.train(params, data, 3)
    model = xgb.train({**params, "multi_strategy": strategy}, data, 3, xgb_model=pickle.loads(pickle.dumps(first)))
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    compare_vector_models(model, files / "model.json")


def test_vector_sklearn_and_quantile(reference):
    X, y, weight, _ = training_data(3)
    params = dict(multi_strategy="multi_output_tree", max_depth=3, max_bin=16, seed=14)
    expected, _ = reference(X, y, params, rounds=5, weight=weight)
    estimator = xgb.XGBRegressor(n_estimators=5, **params).fit(X, y, sample_weight=weight)
    np.testing.assert_array_equal(estimator.predict(X), expected["prediction"])
    quantile = xgb.QuantileDMatrix(X, label=y, weight=weight, max_bin=16)
    model = xgb.train(params, quantile, 5)
    np.testing.assert_array_equal(model.predict(quantile), expected["prediction"])
    np.testing.assert_array_equal(model.inplace_predict(X), expected["prediction"])


def test_multi_strategy_with_one_output(reference):
    X, y, _, _ = training_data(1)
    params = dict(multi_strategy="multi_output_tree", max_depth=3, max_bin=16)
    expected, files = reference(X, y, params, rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 3)
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    compare_vector_models(model, files / "model.json")
    np.testing.assert_allclose(model.predict(data, pred_contribs=True, approx_contribs=True), expected["approximate"], rtol=1e-6, atol=1e-6)
    assert set(json.loads((files / "prediction_errors.json").read_text())) == {"approximate_interactions"}
    with pytest.raises(xgb.core.XGBoostError, match="Approximate interaction"):
        model.predict(data, pred_interactions=True, approx_contribs=True)


@pytest.mark.parametrize("updater", ["refresh", "prune"])
def test_vector_existing_tree_updaters(updater):
    X, y, _, _ = training_data(3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train({"multi_strategy": "multi_output_tree", "max_depth": 2}, data, 3)
    for source in [model, model.save_raw()]:
        with pytest.raises(xgb.core.XGBoostError, match="multi-target"):
            xgb.train({"process_type": "update", "updater": updater}, data, 1, xgb_model=source)
