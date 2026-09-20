import json

import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("rows,outputs", [(257, 1), (8193, 3), (100003, 1)])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("booster", ["gbtree", "gblinear"])
def test_automatic_mean_retains_reduction_precision(reference, rows, outputs, weighted, booster):
    rng = np.random.default_rng(439)
    X = np.zeros((rows, 1), dtype=np.float32)
    y = (rng.normal(size=(rows, outputs)) + np.arange(outputs) + 1.4).astype(np.float32)
    if outputs == 1:
        y = y[:, 0]
    meta = {"weight": rng.uniform(0.01, 2, rows).astype(np.float32)} if weighted else {}
    params = dict(booster=booster, eta=0)
    if booster == "gbtree":
        params.update(max_depth=1, max_bin=2)
    expected, files = reference(X, y, params, rounds=1, test=X[:64], **meta)
    model = xgb.train(params, xgb.DMatrix(X, label=y, **meta), 1)
    actual = json.loads(model.save_raw(raw_format="json"))["learner"]["learner_model_param"]
    source = json.loads((files / "model.json").read_text())["learner"]["learner_model_param"]
    np.testing.assert_array_equal(np.asarray(json.loads(actual["base_score"]), np.float32),
                                  np.asarray(json.loads(source["base_score"]), np.float32))
    np.testing.assert_array_equal(model.predict(xgb.DMatrix(X[:64])), expected["prediction"])
