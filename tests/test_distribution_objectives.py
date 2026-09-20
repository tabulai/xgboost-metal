import json
import pickle

import numpy as np
import pytest
import xgboost as xgb


def scores(model):
    return np.asarray(json.loads(json.loads(model.save_config())["learner"]["learner_model_param"]["base_score"]), dtype=np.float32)


def reference_scores(files):
    config = json.loads((files / "config.json").read_text())
    return np.asarray(json.loads(config["learner"]["learner_model_param"]["base_score"]), dtype=np.float32)


@pytest.mark.parametrize("objective,alpha", [
    ("reg:expectileerror", 0.5), ("reg:expectileerror", [0.05, 0.5, 0.95]),
    ("reg:expectileerror", [0, 1]), ("reg:expectileerror", [0.2, 0.2, 0.8]),
    ("reg:normal", None),
])
@pytest.mark.parametrize("weighted", [False, True])
def test_distribution_objectives_against_reference(reference, objective, alpha, weighted):
    rng = np.random.default_rng(192)
    X = rng.integers(0, 17, size=(197, 5)).astype(np.float32)
    noise = rng.normal(size=len(X)).astype(np.float32) * (0.1 + X[:, 2] / 30)
    y = (2 * np.sin(X[:, 0] / 4) - X[:, 1] / 5 + noise).astype(np.float32)
    X[::13, 1] = np.nan
    metadata = {}
    if weighted:
        weight = rng.uniform(0.05, 3, len(X)).astype(np.float32)
        weight[::17] = 0
        metadata["weight"] = weight
    params = dict(objective=objective, max_depth=3, max_bin=32, eta=0.2, reg_alpha=0.05)
    if alpha is not None:
        params["expectile_alpha"] = alpha
    expected, files = reference(X, y, params, rounds=6, **metadata)
    data = xgb.DMatrix(X, label=y, **metadata)
    history = {}
    model = xgb.train(params, data, 6, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    np.testing.assert_allclose(scores(model), reference_scores(files), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(model.predict(data, output_margin=True), expected["margin"], rtol=4e-5, atol=4e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    np.testing.assert_allclose(model.predict(data, pred_contribs=True), expected["contributions"], rtol=6e-5, atol=6e-5)
    metric = "expectile" if alpha is not None else "normal-nloglik"
    oracle_history = json.loads((files / "history.json").read_text())["train"][metric]
    np.testing.assert_allclose(history["train"][metric], oracle_history, rtol=4e-6, atol=4e-6)
    if isinstance(alpha, list):
        assert np.all(np.diff(model.predict(data), axis=1) >= 0)


@pytest.mark.parametrize("objective", ["reg:expectileerror", "reg:normal"])
def test_distribution_model_exchange_and_continuation(reference, tmp_path, objective):
    rng = np.random.default_rng(163)
    X = rng.integers(0, 13, size=(113, 4)).astype(np.float32)
    y = (X[:, 0] / 3 - X[:, 1] / 5 + rng.normal(0, 0.2, len(X))).astype(np.float32)
    params = dict(objective=objective, max_depth=3, max_bin=32, num_parallel_tree=2)
    if objective == "reg:expectileerror":
        params["expectile_alpha"] = [0.1, 0.5, 0.9]
    data = xgb.DMatrix(X, label=y)
    full = xgb.train(params, data, 6)
    first = xgb.train(params, data, 3)
    continued = xgb.train(params, data, 3, xgb_model=first)
    np.testing.assert_array_equal(continued.predict(data), full.predict(data))
    np.testing.assert_array_equal(full[:3].predict(data), first.predict(data))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(full)).predict(data), full.predict(data))
    for kind in ["json", "ubj"]:
        path = tmp_path / f"distribution.{kind}"
        full.save_model(path)
        expected, _ = reference(X, y, params, model=path)
        for field, options in [("prediction", {}), ("margin", dict(output_margin=True)),
                               ("leaf", dict(pred_leaf=True)), ("contributions", dict(pred_contribs=True)),
                               ("interactions", dict(pred_interactions=True))]:
            np.testing.assert_allclose(full.predict(data, **options), expected[field], rtol=6e-5, atol=6e-5)
        restored = xgb.Booster(model_file=path)
        np.testing.assert_array_equal(restored.predict(data), full.predict(data))
    expected, files = reference(X, y, params, rounds=3)
    imported = xgb.Booster(model_file=files / "model.ubj")
    np.testing.assert_allclose(imported.predict(data), expected["prediction"], rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("base", [0.5, [-3, 0.2, 4], [3, -2, 1]])
def test_expectile_inverse_link_and_base_margins(reference, base):
    X = np.zeros((17, 2), dtype=np.float32)
    y = np.arange(17, dtype=np.float32)
    params = dict(objective="reg:expectileerror", expectile_alpha=[0.1, 0.5, 0.9], base_score=base)
    expected, _ = reference(X, y, params, rounds=0)
    model = xgb.train(params, xgb.DMatrix(X, label=y), 0)
    data = xgb.DMatrix(X)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(model.predict(data, output_margin=True), expected["margin"], atol=3e-6, rtol=3e-6)
    margins = np.column_stack([y, y * 0 - 100, y * 0 + 100])
    data.set_base_margin(margins)
    expected, _ = reference(X, y, params, rounds=0, test_margin=margins)
    np.testing.assert_array_equal(model.predict(data, output_margin=True), margins)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], atol=3e-6, rtol=3e-6)
    expected_empty, _ = reference(X, y, params, rounds=0, test=X[:0])
    empty = xgb.DMatrix(X[:0])
    np.testing.assert_array_equal(model.predict(empty), expected_empty["prediction"])
    np.testing.assert_array_equal(model.predict(empty, output_margin=True), expected_empty["margin"])


@pytest.mark.parametrize("magnitude", [0, 1e-30, 1e20, 1e38])
@pytest.mark.parametrize("weighted", [False, True])
def test_normal_initial_variance_handles_full_label_range(reference, magnitude, weighted):
    X = np.zeros((67, 2), dtype=np.float32)
    y = np.resize(np.asarray([-1, -0.25, 0, 0.5, 1], dtype=np.float32), len(X)) * np.float32(magnitude)
    metadata = {"weight": np.resize(np.asarray([0, 0.0625, 0.5, 1, 4], dtype=np.float32), len(X))} if weighted else {}
    params = dict(objective="reg:normal")
    expected, files = reference(X, y, params, rounds=0, **metadata)
    data = xgb.DMatrix(X, label=y, **metadata)
    model = xgb.train(params, data, 0)
    assert np.isfinite(scores(model)).all()
    np.testing.assert_allclose(scores(model), reference_scores(files), rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(model.predict(data), model.predict(data, output_margin=True))


@pytest.mark.parametrize("log_variance", [-120, 120, -3e38, 3e38])
def test_normal_bounded_variance_updates(reference, log_variance):
    X = np.zeros((31, 2), dtype=np.float32)
    y = np.full(31, 3, dtype=np.float32)
    params = dict(objective="reg:normal", base_score=[3, log_variance], eta=1,
                  max_depth=1, reg_lambda=0, min_child_weight=0)
    expected, _ = reference(X, y, params, rounds=4)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 4)
    prediction = model.predict(data)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction, expected["prediction"], rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(prediction[:, 0], y)


def test_normal_saturated_pairs_preserve_splits(reference):
    X = np.resize(np.asarray([-1, 1], dtype=np.float32), 128).reshape(-1, 1)
    y = X[:, 0].copy()
    params = dict(objective="reg:normal", base_score=[0, -120], max_depth=2, max_bin=8)
    expected, _ = reference(X, y, params, rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 3)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    assert np.std(model.predict(data)[:, 0]) > 0.5


@pytest.mark.parametrize("alpha", [None, [], [-0.1, 0.5], [0.5, 1.1], [0.9, 0.1]])
def test_invalid_expectile_levels(alpha):
    params = dict(objective="reg:expectileerror")
    if alpha is not None:
        params["expectile_alpha"] = alpha
    with pytest.raises(xgb.core.XGBoostError, match="expectile"):
        xgb.train(params, xgb.DMatrix(np.ones((3, 2)), label=[1, 2, 3]), 1)


@pytest.mark.parametrize("objective", ["reg:expectileerror", "reg:normal"])
def test_distribution_requires_single_response(objective):
    data = xgb.DMatrix(np.ones((3, 2)), label=np.ones((3, 2)))
    params = dict(objective=objective, expectile_alpha=[0.1, 0.9])
    with pytest.raises(xgb.core.XGBoostError, match="Multi-target|single response"):
        xgb.train(params, data, 1)


@pytest.mark.parametrize("objective", ["reg:expectileerror", "reg:normal"])
def test_distribution_requires_positive_weight_for_initialization(objective):
    data = xgb.DMatrix(np.ones((3, 2)), label=[1, 2, 3], weight=np.zeros(3))
    with pytest.raises(xgb.core.XGBoostError, match="non-zero"):
        xgb.train(dict(objective=objective, expectile_alpha=[0.1, 0.9]), data, 0)


@pytest.mark.parametrize("weight", [1e38, np.nextafter(np.float32(0), np.float32(1))])
@pytest.mark.parametrize("negative_loss", [False, True])
def test_normal_metric_preserves_signed_weighted_sums(weight, negative_loss):
    rows = 8193
    y = np.full(rows, 0 if negative_loss else 1, dtype=np.float32)
    weights = np.full(rows, weight, dtype=np.float32)
    log_variance = np.float32(-4 if negative_loss else 0)
    data = xgb.DMatrix(np.zeros((rows, 1), dtype=np.float32), label=y, weight=weights)
    model = xgb.train(dict(objective="reg:normal", base_score=[0, float(log_variance)]), data, 0)
    loss = np.float32(0.5) * (np.float32(1.8378770664093453) + log_variance + y)
    expected = np.sum(loss * weights, dtype=np.float64) / np.sum(weights, dtype=np.float64)
    actual = float(model.eval(data).split(":")[-1])
    assert actual == pytest.approx(expected, rel=2e-7, abs=1e-12)


def test_expectile_metric_config_round_trip_with_squared_error(reference):
    X = np.arange(57, dtype=np.float32).reshape(19, 3) % 7
    y = (X[:, 0] - 0.3 * X[:, 1]).astype(np.float32)
    params = dict(objective="reg:squarederror", eval_metric="expectile", expectile_alpha=0.8,
                  max_depth=2, max_bin=16)
    data = xgb.DMatrix(X, label=y)
    history = {}
    model = xgb.train(params, data, 3, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    _, files = reference(X, y, params, rounds=3)
    expected = json.loads((files / "history.json").read_text())["train"]["expectile"]
    np.testing.assert_allclose(history["train"]["expectile"], expected, atol=1e-6)
    evaluation = model.eval(data)
    assert pickle.loads(pickle.dumps(model)).eval(data) == evaluation
    restored = xgb.Booster(model_file=model.save_raw())
    restored.load_config(model.save_config())
    assert restored.eval(data) == evaluation
