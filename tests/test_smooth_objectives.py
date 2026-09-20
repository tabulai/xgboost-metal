from fractions import Fraction
import json
import pickle

import numpy as np
import pytest
import xgboost as xgb


def intercept(model):
    config = json.loads(model.save_config())["learner"]["learner_model_param"]
    return np.asarray(json.loads(config["base_score"]), dtype=np.float32)


@pytest.mark.parametrize("objective,alpha,targets", [
    ("reg:absoluteerror", None, 1), ("reg:absoluteerror", None, 2),
    ("reg:quantileerror", 0.5, 1), ("reg:quantileerror", [0.05, 0.5, 0.95], 1),
    ("reg:quantileerror", [0.0, 1.0], 1), ("reg:quantileerror", [0.2, 0.2, 0.8], 1),
])
@pytest.mark.parametrize("weighted", [False, True])
def test_smoothed_objectives_against_reference(reference, objective, alpha, targets, weighted):
    rng = np.random.default_rng(743)
    X = rng.integers(0, 19, size=(193, 5)).astype(np.float32)
    y = (np.sin(X[:, 0] / 4) * 3 - X[:, 1] / 6 + rng.normal(0, 0.2, len(X))).astype(np.float32)
    y[::23] += 12
    if targets == 2:
        y = np.column_stack([y, -0.7 * y + X[:, 2] / 5])
    X[::11, 1] = np.nan
    metadata = {}
    if weighted:
        weights = rng.uniform(0.1, 3, len(X)).astype(np.float32)
        weights[::17] = 0
        metadata["weight"] = weights
    params = dict(objective=objective, max_depth=3, max_bin=64, eta=0.2, reg_alpha=0.1)
    if alpha is not None:
        params["quantile_alpha"] = alpha
    expected, files = reference(X, y, params, rounds=5, **metadata)
    data = xgb.DMatrix(X, label=y, **metadata)
    history = {}
    model = xgb.train(params, data, 5, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(model.predict(data, output_margin=True), expected["margin"], atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    metric = "mae" if objective == "reg:absoluteerror" else "quantile"
    oracle_history = json.loads((files / "history.json").read_text())["train"][metric]
    np.testing.assert_allclose(history["train"][metric], oracle_history, atol=3e-6, rtol=3e-6)
    oracle_base = np.asarray(json.loads(json.loads((files / "config.json").read_text())["learner"]["learner_model_param"]["base_score"]), dtype=np.float32)
    np.testing.assert_array_equal(intercept(model), oracle_base)
    np.testing.assert_allclose(model.predict(data, pred_contribs=True), expected["contributions"], atol=5e-5, rtol=5e-5)


def exact_weighted_quantiles(values, weights, alpha):
    bits = values.view(np.uint32)
    keys = bits ^ np.where(bits & 0x80000000, np.uint32(0xffffffff), np.uint32(0x80000000))
    order = np.argsort(keys, kind="stable")
    positive = [i for i in order if weights[i] > 0]
    if not positive:
        return np.zeros(len(alpha), dtype=np.float32)
    masses = [Fraction(float(w)) for w in weights]
    total = sum(masses)
    result = []
    for level in np.asarray(alpha, dtype=np.float32):
        if level == 1:
            result.append(values[positive[-1]])
            continue
        target, cumulative = Fraction(float(level)) * total, Fraction()
        for row in positive:
            cumulative += masses[row]
            if cumulative >= target:
                result.append(values[row])
                break
    return np.asarray(result, dtype=np.float32)


@pytest.mark.parametrize("kind", ["uniform", "fractional", "zero_endpoints", "all_zero", "tiny", "integer_carries"])
def test_radix_intercepts_use_exact_positive_weighted_ranks(reference, kind):
    rng = np.random.default_rng(641)
    rows = 8193 if kind == "integer_carries" else 67
    X = np.zeros((rows, 2), dtype=np.float32)
    y = rng.integers(-300, 300, rows).astype(np.float32) / 8
    y[:4] = [-300, -0.0, 0.0, 300]
    weights = np.ones(rows, dtype=np.float32)
    if kind == "fractional":
        weights = np.ldexp(weights, rng.integers(-10, 21, rows))
    elif kind == "zero_endpoints":
        weights[[0, 3]] = 0
    elif kind == "all_zero":
        weights.fill(0)
    elif kind == "tiny":
        weights.fill(np.nextafter(np.float32(0), np.float32(1)))
    elif kind == "integer_carries":
        weights = rng.integers(0, 4096, rows).astype(np.float32)
    levels = [0.0, float(np.nextafter(np.float32(0.5), np.float32(0))), 0.5, 0.9, 1.0]
    params = dict(objective="reg:quantileerror", quantile_alpha=levels, eta=0,
                  max_depth=1, max_bin=8, disable_default_eval_metric=1)
    expected, files = reference(X, y, params, rounds=1, weight=weights)
    data = xgb.DMatrix(X, label=y, weight=weights)
    model = xgb.train(params, data, 1)
    actual = intercept(model)
    np.testing.assert_array_equal(actual.view(np.uint32), exact_weighted_quantiles(y, weights, levels).view(np.uint32))
    oracle_base = np.asarray(json.loads(json.loads((files / "config.json").read_text())["learner"]["learner_model_param"]["base_score"]), dtype=np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), oracle_base.view(np.uint32))
    np.testing.assert_allclose(model.predict(data), expected["prediction"], atol=1e-6)


@pytest.mark.parametrize("objective", ["reg:absoluteerror", "reg:quantileerror"])
def test_smoothed_model_exchange_continuation_and_interpretation(reference, tmp_path, objective):
    rng = np.random.default_rng(391)
    X = rng.integers(0, 12, size=(129, 4)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] * 0.3 + rng.normal(0, 0.2, len(X))).astype(np.float32)
    params = dict(objective=objective, max_depth=3, max_bin=32, num_parallel_tree=2)
    if objective == "reg:quantileerror":
        params["quantile_alpha"] = [0.1, 0.5, 0.9]
    data = xgb.DMatrix(X, label=y)
    full = xgb.train(params, data, 6)
    first = xgb.train(params, data, 3)
    continued = xgb.train(params, data, 3, xgb_model=first)
    np.testing.assert_array_equal(continued.predict(data), full.predict(data))
    np.testing.assert_array_equal(full[:3].predict(data), first.predict(data))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(full)).predict(data), full.predict(data))
    for kind in ["json", "ubj"]:
        path = tmp_path / f"smooth.{kind}"
        full.save_model(path)
        expected, files = reference(X, y, params, model=path)
        for field, options in [("prediction", {}), ("margin", dict(output_margin=True)),
                               ("leaf", dict(pred_leaf=True)), ("contributions", dict(pred_contribs=True)),
                               ("interactions", dict(pred_interactions=True))]:
            np.testing.assert_allclose(full.predict(data, **options), expected[field], atol=5e-5, rtol=5e-5)
        restored = xgb.Booster(model_file=path)
        np.testing.assert_array_equal(restored.predict(data), full.predict(data))
    oracle, files = reference(X, y, params, rounds=3)
    imported = xgb.Booster(model_file=files / "model.ubj")
    np.testing.assert_allclose(imported.predict(data), oracle["prediction"], atol=3e-6)


def test_quantile_transform_preserves_raw_margins(reference):
    X = np.zeros((17, 2), dtype=np.float32)
    y = np.arange(17, dtype=np.float32)
    params = dict(objective="reg:quantileerror", quantile_alpha=[0.1, 0.5, 0.9], base_score=[3, -2, 1])
    expected, _ = reference(X, y, params, rounds=0)
    model = xgb.train(params, xgb.DMatrix(X, label=y), 0)
    data = xgb.DMatrix(X)
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    np.testing.assert_array_equal(model.predict(data, output_margin=True), expected["margin"])
    margin = np.column_stack([y + 3, y - 2, y + 1])
    data.set_base_margin(margin)
    np.testing.assert_array_equal(model.predict(data, output_margin=True), margin)
    np.testing.assert_array_equal(model.predict(data), np.sort(margin, axis=1))


@pytest.mark.parametrize("alpha", [None, [], [-0.1, 0.5], [0.5, 1.1], [0.9, 0.1]])
def test_invalid_quantile_levels(alpha):
    params = dict(objective="reg:quantileerror")
    if alpha is not None:
        params["quantile_alpha"] = alpha
    with pytest.raises(xgb.core.XGBoostError, match="quantile"):
        xgb.train(params, xgb.DMatrix(np.ones((3, 2)), label=[1, 2, 3]), 1)


def test_quantile_rejects_multiple_labels_like_reference():
    data = xgb.DMatrix(np.ones((3, 2)), label=np.ones((3, 2)))
    with pytest.raises(xgb.core.XGBoostError, match="Multi-target"):
        xgb.train(dict(objective="reg:quantileerror", quantile_alpha=[0.1, 0.9]), data, 1)


def test_absolute_error_small_positive_weight_matches_cuda_source(reference):
    # The pinned CUDA implementation divides by any positive total weight.
    # Its CPU counterpart treats totals below 1e-6 as zero. With no regularizer,
    # identical residuals must still take a full Newton step on the GPU.
    X = np.zeros((8, 2), dtype=np.float32)
    y = np.full(8, 3, dtype=np.float32)
    weight = np.full(8, 1e-10, dtype=np.float32)
    params = dict(objective="reg:absoluteerror", base_score=0, max_depth=1,
                  eta=1, reg_lambda=0, min_child_weight=0)
    data = xgb.DMatrix(X, label=y, weight=weight)
    model = xgb.train(params, data, 1)
    np.testing.assert_allclose(model.predict(data), y, atol=1e-6)
    expected, _ = reference(X, y, params, rounds=1, weight=weight)
    np.testing.assert_array_equal(expected["prediction"], np.zeros_like(y))


@pytest.mark.parametrize("objective", ["reg:absoluteerror", "reg:quantileerror"])
@pytest.mark.parametrize("zero_weights", [False, True])
def test_smoothed_zero_scale_and_zero_weight(objective, zero_weights):
    X = np.arange(38, dtype=np.float32).reshape(19, 2)
    params = dict(objective=objective, max_depth=2)
    if objective == "reg:quantileerror":
        params["quantile_alpha"] = [0.1, 0.5, 0.9]
    data = xgb.DMatrix(X, label=np.full(19, 3, dtype=np.float32),
                       weight=np.full(19, int(not zero_weights), dtype=np.float32))
    model = xgb.train(params, data, 3)
    prediction = model.predict(data)
    np.testing.assert_array_equal(prediction, np.full_like(prediction, 0 if zero_weights else 3))
    if zero_weights and objective == "reg:quantileerror":
        with pytest.raises(xgb.core.XGBoostError, match="positive total weight"):
            model.eval(data)


def test_quantile_metric_config_round_trip_with_squared_error(reference):
    X = np.arange(57, dtype=np.float32).reshape(19, 3) % 7
    y = (X[:, 0] - 0.3 * X[:, 1]).astype(np.float32)
    params = dict(objective="reg:squarederror", eval_metric="quantile", quantile_alpha=0.8,
                  max_depth=2, max_bin=16)
    data = xgb.DMatrix(X, label=y)
    history = {}
    model = xgb.train(params, data, 3, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    _, files = reference(X, y, params, rounds=3)
    expected = json.loads((files / "history.json").read_text())["train"]["quantile"]
    np.testing.assert_allclose(history["train"]["quantile"], expected, atol=1e-6)
    evaluation = model.eval(data)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(data), model.predict(data))
    assert pickle.loads(pickle.dumps(model)).eval(data) == evaluation
    restored = xgb.Booster(model_file=model.save_raw())
    restored.load_config(model.save_config())
    assert restored.eval(data) == evaluation


@pytest.mark.parametrize("weight", [1.0, 1e38, 1e-30, np.nextafter(np.float32(0), np.float32(1))])
def test_quantile_metric_accumulates_finite_weights_without_overflow(weight):
    y = np.resize(np.asarray([-1, 1], dtype=np.float32), 8193)
    X = np.zeros((len(y), 1), dtype=np.float32)
    weights = np.full(len(y), weight, dtype=np.float32)
    levels = np.asarray([0.1, 0.5, 0.9], dtype=np.float32)
    data = xgb.DMatrix(X, label=y, weight=weights)
    model = xgb.train(dict(objective="reg:quantileerror", quantile_alpha=levels.tolist(), base_score=0), data, 0)
    # Form the reference's float32 per-element losses, then accumulate in f64.
    residual = y[:, None]
    losses = np.where(residual >= 0, levels * residual, (levels - 1) * residual)
    expected = np.sum(losses * weights[:, None], dtype=np.float64) / (np.sum(weights, dtype=np.float64) * len(levels))
    actual = float(model.eval(data).split(":")[-1])
    assert actual == pytest.approx(expected, rel=2e-7, abs=1e-12)


@pytest.mark.parametrize("subnormal_weights", [False, True])
def test_quantile_metric_preserves_underflowing_weight_products(subnormal_weights):
    rng = np.random.default_rng(193)
    rows = 1031
    mantissa = rng.integers(1, 0x800000, rows, dtype=np.uint32)
    exponent = np.zeros(rows, dtype=np.uint32) if subnormal_weights else rng.integers(1, 8, rows, dtype=np.uint32)
    weights = ((exponent << 23) | mantissa).view(np.float32)
    magnitudes = ((rng.integers(118, 140, rows, dtype=np.uint32) << 23) |
                  rng.integers(0, 0x800000, rows, dtype=np.uint32)).view(np.float32)
    data = xgb.DMatrix(np.zeros((rows, 1), dtype=np.float32), label=-magnitudes, weight=weights)
    model = xgb.train(dict(objective="reg:quantileerror", quantile_alpha=0.0, base_score=0), data, 0)
    expected = np.sum(magnitudes * weights, dtype=np.float64) / np.sum(weights, dtype=np.float64)
    actual = float(model.eval(data).split(":")[-1])
    assert actual == pytest.approx(expected, rel=2e-7, abs=0)
