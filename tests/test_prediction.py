import ctypes
import json

import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("kind,columns", [
    ("numeric", 7), ("numeric", 64), ("numeric", 129),
    ("categorical", 7), ("multioutput", 32),
    ("softprob", 32), ("softmax", 32),
])
@pytest.mark.parametrize("threads", [32, 64, 128])
def test_cached_prediction_matches_direct_traversal(kind, columns, threads):
    rng = np.random.default_rng(184)
    X = rng.normal(size=(257, columns)).astype(np.float32)
    metadata = {}
    if kind == "categorical":
        X = rng.integers(0, 9, size=X.shape).astype(np.float32)
        metadata = dict(feature_types=["c"] * columns, enable_categorical=True)
    y = X[:, 0] * (X[:, 1] > 0) - X[:, 2]
    params = dict(max_depth=4, max_bin=32, num_parallel_tree=2)
    groups = 1
    if kind == "multioutput":
        y = np.column_stack([y, -y * 0.7])
        groups = 2
    elif kind in ("softprob", "softmax"):
        y = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3])).astype(np.float32)
        params.update(objective=f"multi:{kind}", num_class=3)
        groups = 3
    X[::7, 0] = np.nan
    X[::11, -1] = np.nan
    model = xgb.train(params, xgb.DMatrix(X, label=y, **metadata), 7)
    model_bytes = model.save_raw()
    for rows in [0, 1, 31, 65, 257]:
        margin = rng.normal(size=(rows, groups)).astype(np.float32)
        data = xgb.DMatrix(X[:rows], base_margin=margin, **metadata)
        for options in [dict(), dict(output_margin=True), dict(pred_leaf=True),
                        dict(iteration_range=(1, 5), strict_shape=True),
                        dict(iteration_range=(0, 5), pred_leaf=True, strict_shape=True),
                        dict(iteration_range=(7, 7), output_margin=True)]:
            model.set_param("metal_predict_shared", 0)
            expected = model.predict(data, **options)
            model.set_param({"metal_predict_shared": 1, "metal_predict_threads": threads})
            actual = model.predict(data, **options)
            np.testing.assert_array_equal(actual, expected)
    out = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(out)))
    profile = json.loads(out.value)
    fits = (columns | 1) * threads * 4 <= 32 * 1024
    assert profile["tuning"]["prediction_shared"] == int(fits)
    assert model.save_raw() == model_bytes


def test_cached_prediction_without_trees():
    data = xgb.DMatrix(np.zeros((65, 3), dtype=np.float32), label=np.zeros(65))
    model = xgb.train(dict(base_score=0.75, metal_predict_shared=1), data, 0)
    np.testing.assert_array_equal(model.predict(data), np.full(65, 0.75, dtype=np.float32))
    assert model.predict(data, pred_leaf=True).shape == (65, 0)


@pytest.mark.parametrize("kind", ["dense", "csr", "dataframe"])
@pytest.mark.parametrize("outputs", [1, 3])
def test_inplace_prediction_base_margin_proxy(reference, kind, outputs):
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(409)
    X = rng.integers(1, 9, (137, 3)).astype(np.float32)
    label = rng.normal(size=(len(X), outputs)).astype(np.float32)
    margin = rng.normal(size=label.shape).astype(np.float32)
    if outputs == 1:
        label, margin = label[:, 0], margin[:, 0]
    params = dict(max_depth=2, max_bin=16)
    expected, files = reference(X, label, params, rounds=3, test_margin=margin)
    model = xgb.Booster(model_file=files / "model.json")
    values = sp.csr_matrix(X) if kind == "csr" else pd.DataFrame(X) if kind == "dataframe" else X
    for predict_type, field in [("value", "prediction"), ("margin", "margin")]:
        actual = model.inplace_predict(values, base_margin=margin, predict_type=predict_type)
        np.testing.assert_array_equal(actual, expected[field])
    for invalid in [margin[:-1], np.zeros((len(X), outputs + 1), np.float32)]:
        with pytest.raises(xgb.core.XGBoostError, match="base_margin"):
            model.inplace_predict(values, base_margin=invalid)
