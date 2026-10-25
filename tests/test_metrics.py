import json
import math
import numpy as np
import pytest
import xgboost as xgb

@pytest.mark.parametrize("objective,metrics", [
    ("reg:squarederror", ["rmse", "mae", "mape"]),
    ("reg:squaredlogerror", ["rmsle"]),
    ("binary:logistic", ["logloss", "error", "error@0.3", "auc", "aucpr"]),
    ("multi:softprob", ["mlogloss", "merror", "auc", "aucpr"]),
    ("count:poisson", ["poisson-nloglik"]),
    ("reg:gamma", ["gamma-deviance"]),
    ("reg:tweedie", ["tweedie-nloglik@1.5"]),
    ("reg:pseudohubererror", ["mphe"]),
])
def test_metrics_against_reference(reference, objective, metrics):
    rng = np.random.default_rng(86)
    X = rng.normal(size=(527, 5)).astype(np.float32)
    y = np.exp(0.4 * X[:, 0] - 0.2 * X[:, 1]).astype(np.float32)
    if objective.startswith("binary"):
        y = (y > np.median(y)).astype(np.float32)
    if objective.startswith("multi"):
        y = np.digitize(y, np.quantile(y, [1/3, 2/3])).astype(np.float32)
    weight = rng.uniform(0.1, 2, len(X)).astype(np.float32)
    params = {"objective": objective, "max_depth": 3, "eval_metric": metrics}
    if objective.startswith("multi"):
        params["num_class"] = 3
    _, files = reference(X, y, params, rounds=3, weight=weight)
    history = json.loads((files / "history.json").read_text())["train"]
    model = xgb.Booster(model_file=files / "model.json")
    model.set_param([("eval_metric", name) for name in metrics])
    actual = model.eval(xgb.DMatrix(X, label=y, weight=weight), name="train")
    parsed = {field.split(":")[0][6:]: float(field.split(":")[1])
              for field in actual.split("\t")[1:]}
    for name in metrics:
        assert parsed[name] == pytest.approx(history[name][-1], rel=5e-6, abs=5e-6), (name, actual, history)

def test_constant_class_auc():
    X = np.arange(96, dtype=np.float32).reshape(-1, 3)
    d = xgb.DMatrix(X, label=np.ones(len(X)))
    model = xgb.train({"objective": "binary:logistic", "eval_metric": "auc"}, d, 1)
    assert "nan" in model.eval(d)


def reference_binary_auc(score, label, weight, pr):
    """The pinned BinaryROCAUC and BinaryPRAUC scans, accumulated in double."""
    order = np.argsort(-score.astype(np.float64), kind="stable")
    score, label, weight = score[order], label[order], weight[order]
    # The reference forms each increment in float, then adds it in double.
    positive = (label * weight).astype(np.float64)
    negative = ((np.float32(1) - label) * weight).astype(np.float64)
    total_pos = positive.sum()

    def area(fp_prev, fp, tp_prev, tp):
        if not pr:
            return abs(fp_prev - fp) * (tp_prev + tp) * 0.5
        a, b = 1.0, 0.0
        if tp != tp_prev:
            h = (fp - fp_prev) / (tp - tp_prev)
            a, b = h + 1, (fp_prev - h * tp_prev) / total_pos
        now, before = tp / total_pos, tp_prev / total_pos
        if b == 0:
            return (now - before) / a
        return (now - before - b / a * (math.log(a * now + b) - math.log(a * before + b))) / a

    tp = fp = tp_prev = fp_prev = auc = 0.0
    for i in range(len(score)):
        if i and score[i] != score[i - 1]:
            auc += area(fp_prev, fp, tp_prev, tp)
            tp_prev, fp_prev = tp, fp
        tp += positive[i]
        fp += negative[i]
    auc += area(fp_prev, fp, tp_prev, tp)
    return auc if pr else auc / (tp * fp)


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
def test_auc_class_sums_exceed_float_precision(metric):
    # A 2^25 negative ranked first puts every later sum where float spacing is
    # 4, so float accumulation turned each increment of 3 into 4. Tens of
    # millions of unit-weight rows reach the same regime.
    rng = np.random.default_rng(29)
    margin = np.sort(rng.uniform(-3, 3, 2000))[::-1].astype(np.float32)
    label = (rng.random(len(margin)) < 1 / (1 + np.exp(-2 * margin))).astype(np.float32)
    weight = np.full(len(margin), 3, np.float32)
    label[0], weight[0] = 0, 2**25
    X = rng.normal(size=(len(margin), 2)).astype(np.float32)
    d = xgb.DMatrix(X, label=label, weight=weight, base_margin=margin)
    # A zero learning rate leaves the prediction order to the base margins.
    model = xgb.train({"objective": "binary:logistic", "learning_rate": 0, "max_depth": 1,
                       "eval_metric": metric}, d, 1)
    actual = float(model.eval(d).split(":")[1])
    expected = reference_binary_auc(model.predict(d), label, weight, metric == "aucpr")
    assert actual == pytest.approx(expected, rel=5e-6)


@pytest.mark.parametrize("metric,threshold", [("error", 0.5), ("error@0.3", 0.3)])
def test_error_counts_fractional_labels(metric, threshold):
    # The reference EvalError scores pred > t ? 1 - label : label.
    rng = np.random.default_rng(31)
    X = rng.normal(size=(400, 3)).astype(np.float32)
    label = rng.uniform(0, 1, len(X)).astype(np.float32)
    weight = rng.uniform(0.2, 2, len(X)).astype(np.float32)
    d = xgb.DMatrix(X, label=label, weight=weight)
    model = xgb.train({"objective": "binary:logistic", "max_depth": 2, "eval_metric": metric}, d, 3)
    prediction = model.predict(d)
    expected = np.sum(weight * np.where(prediction > threshold, 1 - label, label), dtype=np.float64)
    actual = float(model.eval(d).split(":")[1])
    assert actual == pytest.approx(expected / np.sum(weight, dtype=np.float64), rel=5e-6)


@pytest.mark.parametrize("margin,label,weight", [
    ([1, 0], [0, 1], [1e-37, 1]),
    ([1, 0], [0, 1], [1, 1e37]),
    ([1, 0], [0, 1], [1e-20, 1e20]),
    ([2, 1, 1], [0, 1, 0], [1e-5, 2e38, 2e38]),
    ([2, 1, 1], [1, 1, 0], [1e-5, 1e5, 1e10]),
    ([1, 0], [0, 1], [1e4, 1]),
    ([1, 0], [0, 1], [4, 1]),
    ([1, 0], [0, 1], [3.999, 1]),
    ([1, 0], [0, 1], [3.2134451, 1]),
], ids=["tiny-leading-weight", "large-trailing-weight", "ratio-overflow",
        "step-sum-overflow", "tied-low-precision", "small-increment",
        "series-boundary", "above-series-boundary", "moderate-increment"])
def test_aucpr_extreme_weight_ratios(margin, label, weight):
    # Finite class totals can still overflow an unbounded step/cumulative
    # ratio or its logarithm's intermediates. Zero rounds isolate evaluation
    # from training; tied predictions also exercise a mixed-class PR segment.
    margin, label, weight = [np.asarray(value, dtype=np.float32)
                             for value in (margin, label, weight)]
    data = xgb.DMatrix(np.arange(len(label), dtype=np.float32)[:, None],
                       label=label, weight=weight, base_margin=margin)
    model = xgb.train({"objective": "reg:squarederror", "eval_metric": "aucpr",
                       "base_score": 0.5}, data, num_boost_round=0)
    prediction = model.predict(data)
    np.testing.assert_array_equal(prediction, margin)
    actual = float(model.eval(data).split(":")[1])
    expected = reference_binary_auc(prediction, label, weight, pr=True)
    assert math.isfinite(actual) and 0 <= actual <= 1
    assert actual == pytest.approx(expected, rel=5e-6, abs=1e-12)
