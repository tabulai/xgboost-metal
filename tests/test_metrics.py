import json
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
