import json
import pickle

import numpy as np
import pytest
import xgboost as xgb


def dropout_data():
    rng = np.random.default_rng(594)
    X = rng.integers(-8, 9, (377, 5)).astype(np.float32) / np.float32(4)
    y = (2 * X[:, 0] - X[:, 1] + np.sin(X[:, 2]) + rng.normal(0, 0.1, len(X))).astype(np.float32)
    X[::17, 1] = np.nan
    return X, y


def tree_weights(model):
    return np.asarray(json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"].get("weight_drop", []), dtype=np.float32)


def compare(model, data, expected, files):
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(model.predict(data, output_margin=True), expected["margin"], rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    config = json.loads(model.save_config())["learner"]
    source = json.loads((files / "config.json").read_text())["learner"]
    assert config["generic_param"]["rng_state"] == source["generic_param"]["rng_state"]
    assert config["gradient_booster"]["dart_train_param"] == source["gradient_booster"]["dart_train_param"]
    reference_weights = json.loads((files / "model.json").read_text())["learner"]["gradient_booster"]["model"].get("weight_drop", [])
    np.testing.assert_array_equal(tree_weights(model), np.asarray(reference_weights, dtype=np.float32))


@pytest.mark.parametrize("sample", ["uniform", "weighted"])
@pytest.mark.parametrize("normalize", ["tree", "forest"])
@pytest.mark.parametrize("dropout", [
    {"rate_drop": 0.4}, {"one_drop": True},
    {"rate_drop": 0.7, "skip_drop": 0.5, "one_drop": True}, {"rate_drop": 1},
])
def test_dropout_training(reference, sample, normalize, dropout):
    X, y = dropout_data()
    params = dict(max_depth=3, max_bin=32, eta=0.15, seed=37,
                  sample_type=sample, normalize_type=normalize, **dropout)
    expected, files = reference(X, y, params, rounds=9)
    data = xgb.DMatrix(X, label=y)
    history = {}
    model = xgb.train(params, data, 9, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    compare(model, data, expected, files)
    oracle_history = json.loads((files / "history.json").read_text())
    np.testing.assert_allclose(history["train"]["rmse"], oracle_history["train"]["rmse"], rtol=3e-5, atol=3e-5)
    for name, options in [
        ("contributions", {"pred_contribs": True}), ("interactions", {"pred_interactions": True}),
        ("approximate", {"pred_contribs": True, "approx_contribs": True}),
        ("approximate_interactions", {"pred_interactions": True, "approx_contribs": True}),
    ]:
        np.testing.assert_allclose(model.predict(data, **options), expected[name], rtol=4e-5, atol=4e-5)


@pytest.mark.parametrize("per_iteration", [False, True])
@pytest.mark.parametrize("custom", [False, True])
def test_dropout_custom_objectives_and_column_sampling(reference, per_iteration, custom):
    X, y = dropout_data()
    params = dict(max_depth=3, max_bin=32, seed=91, seed_per_iteration=per_iteration,
                  rate_drop=0.4, one_drop=True, sample_type="weighted", normalize_type="forest",
                  colsample_bytree=0.8, colsample_bylevel=0.8, colsample_bynode=0.8)
    expected, files = reference(X, y, params, rounds=8, custom_square=custom)
    data = xgb.DMatrix(X, label=y)
    objective = (lambda pred, data: (pred - data.get_label(), np.ones_like(pred))) if custom else None
    model = xgb.train(params, data, 8, obj=objective)
    compare(model, data, expected, files)


@pytest.mark.parametrize("objective", ["multi:softprob", "reg:squarederror", "reg:quantileerror", "reg:expectileerror", "reg:normal", "rank:ndcg"])
def test_dropout_outputs_weights_and_parallel_trees(reference, objective):
    X, raw_y = dropout_data()
    y = raw_y
    params = dict(objective=objective, max_depth=2, max_bin=32, seed=83, eta=0.2,
                  num_parallel_tree=3, rate_drop=0.3, one_drop=True)
    meta = {"weight": np.linspace(0.1, 1.1, len(y), dtype=np.float32)}
    if objective == "multi:softprob":
        y = (np.arange(len(y)) % 3).astype(np.float32)
        params["num_class"] = 3
    elif objective == "reg:squarederror":
        y = np.column_stack([raw_y, np.sin(raw_y)])
    elif objective == "reg:quantileerror":
        params["quantile_alpha"] = [0.2, 0.8]
    elif objective == "reg:expectileerror":
        params["expectile_alpha"] = [0.2, 0.8]
    elif objective == "rank:ndcg":
        y = np.digitize(y, [-2, 0, 2]).astype(np.float32)
        meta = {"qid": np.arange(len(y), dtype=np.uint32) // 16,
                "weight": np.linspace(0.1, 1.1, (len(y) + 15) // 16, dtype=np.float32)}
        params.update(lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=3)
    expected, files = reference(X, y, params, rounds=6, **meta)
    data = xgb.DMatrix(X, label=y, **meta)
    model = xgb.train(params, data, 6)
    compare(model, data, expected, files)
    np.testing.assert_allclose(model.predict(data, pred_contribs=True).sum(axis=-1),
                               model.predict(data, output_margin=True), rtol=4e-5, atol=4e-5)


@pytest.mark.parametrize("sample", ["uniform", "weighted"])
def test_dropout_training_prediction_and_rng(reference, sample):
    X, y = dropout_data()
    params = dict(max_depth=3, max_bin=32, seed=152, rate_drop=0.4, one_drop=True, sample_type=sample)
    requests = [
        {"training": True, "output_margin": True},
        {"training": True, "pred_leaf": True},
        {"training": True, "pred_contribs": True},
        {"training": True, "output_margin": True, "iteration_range": [1, 5]},
        {"training": False, "output_margin": True},
        {"training": True, "output_margin": True},
    ]
    expected, files = reference(X, y, params, rounds=7, prediction_requests=requests)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 7)
    ordinary = model.predict(data)
    for i, request in enumerate(requests):
        np.testing.assert_allclose(model.predict(data, **request), expected[f"request_{i}"], rtol=4e-5, atol=4e-5)
    np.testing.assert_array_equal(model.predict(data), ordinary)
    compare(model, data, expected, files)


@pytest.mark.parametrize("format", ["json", "ubj"])
def test_dropout_model_exchange_and_continuation(reference, tmp_path, format):
    X, y = dropout_data()
    params = dict(max_depth=3, max_bin=32, seed=23, rate_drop=0.45, one_drop=True,
                  sample_type="weighted", normalize_type="forest")
    expected, files = reference(X, y, params, rounds=6)
    data = xgb.DMatrix(X, label=y)
    model = xgb.Booster(model_file=files / f"model.{format}")
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    model.set_param("metal_predict_shared", 1)
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    np.testing.assert_allclose(model.predict(data, pred_contribs=True), expected["contributions"], rtol=4e-5, atol=4e-5)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(data), model.predict(data))
    destination = tmp_path / f"metal.{format}"
    model.save_model(destination)
    exported, _ = reference(X, y, params, model=destination)
    np.testing.assert_array_equal(exported["prediction"], model.predict(data))
    continued, continued_files = reference(X, y, params, model=destination, continue_rounds=4)
    own = xgb.train(params, data, 4, xgb_model=destination)
    compare(own, data, continued, continued_files)


@pytest.mark.parametrize("normalize", ["tree", "forest"])
def test_dropout_pickle_and_booster_continuation(reference, normalize):
    X, y = dropout_data()
    params = dict(max_depth=3, max_bin=32, seed=932, rate_drop=0.3, one_drop=True,
                  num_parallel_tree=2, sample_type="weighted", normalize_type=normalize)
    expected, files = reference(X, y, params, rounds=10)
    data = xgb.DMatrix(X, label=y)
    first = xgb.train(params, data, 4)
    resumed = xgb.train(params, data, 6, xgb_model=pickle.loads(pickle.dumps(first)))
    compare(resumed, data, expected, files)
    uninterrupted = xgb.train(params, data, 10)
    np.testing.assert_array_equal(resumed.predict(data), uninterrupted.predict(data))
    sliced = resumed[1:9:2]
    chosen = np.concatenate([tree_weights(resumed)[i * 2:(i + 1) * 2] for i in [1, 3, 5, 7]])
    np.testing.assert_array_equal(tree_weights(sliced), chosen)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(sliced)).predict(data), sliced.predict(data))
    sliced_data = sliced.predict(data, pred_contribs=True).sum(axis=-1)
    np.testing.assert_allclose(sliced_data, sliced.predict(data, output_margin=True), rtol=4e-5, atol=4e-5)


@pytest.mark.parametrize("legacy", [False, True])
def test_legacy_dropout_model_and_config(reference, legacy):
    X, y = dropout_data()
    params = dict(max_depth=2, max_bin=32, seed=713, rate_drop=0.4, one_drop=True)
    expected, files = reference(X, y, params, rounds=5)
    document = json.loads((files / "model.json").read_text())
    booster = document["learner"]["gradient_booster"]
    weights = booster["model"].pop("weight_drop")
    document["learner"]["gradient_booster"] = {"name": "dart", "gbtree": booster, "weight_drop": weights} if legacy else {**booster, "weight_drop": weights}
    model = xgb.Booster(model_file=bytearray(json.dumps(document).encode()))
    data = xgb.DMatrix(X, label=y)
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    config = json.loads((files / "config.json").read_text())
    if legacy:
        inner = config["learner"]["gradient_booster"]
        dropout = inner.pop("dart_train_param")
        config["learner"]["gradient_booster"] = {"name": "dart", "gbtree": inner, "dart_train_param": dropout}
        config["learner"]["learner_train_param"]["booster"] = "dart"
    model.load_config(json.dumps(config))
    assert json.loads(model.save_config())["learner"]["learner_train_param"]["booster"] == "gbtree"
    compare(model, data, expected, files)


@pytest.mark.parametrize("params", [
    {"rate_drop": -0.1}, {"rate_drop": 1.1}, {"rate_drop": float("nan")},
    {"skip_drop": -1}, {"skip_drop": 1.1}, {"one_drop": "bad"},
    {"sample_type": "bad"}, {"normalize_type": "bad"},
])
def test_invalid_dropout_parameters(params):
    X, y = dropout_data()
    with pytest.raises(xgb.core.XGBoostError):
        xgb.train(params, xgb.DMatrix(X, label=y), 1)


def test_dropout_disabled_and_skip_all(reference):
    X, y = dropout_data()
    params = dict(max_depth=3, max_bin=32, seed=3)
    data = xgb.DMatrix(X, label=y)
    plain = xgb.train(params, data, 5)
    disabled = xgb.train({**params, "sample_type": "weighted", "normalize_type": "forest"}, data, 5)
    assert plain.save_raw() == disabled.save_raw()
    skipped_params = dict(params, booster="dart", rate_drop=1, skip_drop=1, one_drop=True)
    expected, files = reference(X, y, skipped_params, rounds=5)
    skipped = xgb.train(skipped_params, data, 5)
    compare(skipped, data, expected, files)
    np.testing.assert_array_equal(skipped.predict(data), plain.predict(data))
    np.testing.assert_array_equal(tree_weights(skipped), np.ones(5, dtype=np.float32))


def test_dropout_enabled_and_disabled_during_training(reference):
    X, y = dropout_data()
    params = dict(max_depth=3, max_bin=32, seed=17, colsample_bytree=0.8)
    updates = {3: {"rate_drop": 0.6, "one_drop": True}, 8: {"rate_drop": 0, "one_drop": False}}
    expected, files = reference(X, y, params, rounds=11, param_updates=updates)
    class Schedule(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            if epoch in updates:
                model.set_param(updates[epoch])
            return False
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 11, callbacks=[Schedule()])
    compare(model, data, expected, files)
    assert len(tree_weights(model)) == 11
    assert np.any(tree_weights(model) < 1)
    np.testing.assert_array_equal(tree_weights(model)[-3:], np.ones(3, dtype=np.float32))


def test_dropout_categorical_margins_and_iteration_ranges(reference):
    import pandas as pd
    rng = np.random.default_rng(9)
    values = rng.integers(0, 7, (191, 3)).astype(np.float32)
    y = (values[:, 0] % 3 + values[:, 1] - values[:, 2] % 2).astype(np.float32)
    names = np.asarray(["a", "b", "c", "d", "e", "f", "g"])
    frame = pd.DataFrame({"a": pd.Categorical(names[values[:, 0].astype(int)]),
                          "b": values[:, 1], "c": pd.Categorical(names[values[:, 2].astype(int)])})
    frame.loc[::13, "a"] = None
    test = frame.copy()
    test["a"] = test["a"].cat.reorder_categories(names[::-1])
    test["c"] = test["c"].cat.reorder_categories(names[::-1])
    margin = np.linspace(-1, 1, len(frame), dtype=np.float32)
    params = dict(max_depth=3, max_bin=16, max_cat_to_onehot=2, rate_drop=0.4, skip_drop=0.2,
                  one_drop=True, normalize_type="forest", seed=25)
    requests = [{"output_margin": True, "iteration_range": [2, 6]},
                {"pred_leaf": True, "strict_shape": True, "iteration_range": [0, 4]}]
    expected, files = reference(frame, y, params, rounds=7, test=test,
                                base_margin=margin, test_margin=margin, prediction_requests=requests)
    data = xgb.DMatrix(frame, label=y, base_margin=margin, enable_categorical=True)
    test_data = xgb.DMatrix(test, base_margin=margin, enable_categorical=True)
    model = xgb.train(params, data, 7)
    compare(model, test_data, expected, files)
    for i, request in enumerate(requests):
        np.testing.assert_allclose(model.predict(test_data, **request), expected[f"request_{i}"], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(model.predict(test_data, pred_interactions=True), expected["interactions"], rtol=4e-5, atol=4e-5)
    model.set_param("metal_predict_shared", 1)
    np.testing.assert_allclose(model.predict(test_data), expected["prediction"], rtol=3e-5, atol=3e-5)


def test_dropout_sklearn_early_stopping_and_cv():
    X, y = dropout_data()
    estimator = xgb.XGBRegressor(n_estimators=30, early_stopping_rounds=3, max_depth=2, max_bin=32,
                                rate_drop=0.4, one_drop=True, random_state=6)
    estimator.fit(X[:277], y[:277], eval_set=[(X[277:], -y[277:])], verbose=False)
    booster = estimator.get_booster()
    assert booster.num_boosted_rounds() < 30
    np.testing.assert_array_equal(estimator.predict(X), booster.predict(xgb.DMatrix(X), iteration_range=(0, estimator.best_iteration + 1)))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(estimator)).predict(X), estimator.predict(X))
    best = booster[:estimator.best_iteration + 1]
    np.testing.assert_array_equal(best.predict(xgb.DMatrix(X)), estimator.predict(X))
    result = xgb.cv(dict(max_depth=2, max_bin=32, rate_drop=0.4, one_drop=True),
                    xgb.DMatrix(X, label=y), num_boost_round=4, nfold=3)
    assert len(result) == 4 and np.isfinite(result.to_numpy()).all()
