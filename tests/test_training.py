from pathlib import Path
import ctypes
import json
import pickle
import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression
from scipy import sparse

@pytest.fixture
def regression():
    rng = np.random.default_rng(192)
    X = rng.normal(size=(127, 7)).astype(np.float32)
    y = 2 * X[:, 0] - X[:, 2] + 0.1 * rng.normal(size=len(X))
    return X, y.astype(np.float32)

def test_real_metal_backend():
    info = xgb.build_info()
    assert info["USE_METAL"] is True
    assert info["USE_CUDA"] is False
    assert "Apple" in info["METAL_DEVICE"]
    expected_library = Path(__file__).resolve().parents[1] / "lib" / "libxgboost.dylib"
    assert Path(info["libxgboost"]).resolve() == expected_library.resolve()

@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "reg:logistic",
    "binary:logitraw", "multi:softprob", "multi:softmax", "count:poisson", "reg:gamma",
    "reg:tweedie", "reg:squaredlogerror", "reg:pseudohubererror", "binary:hinge"])
def test_objectives_against_reference(regression, reference, objective):
    X, y = regression
    params = dict(objective=objective, max_depth=3, max_bin=256, eta=0.2)
    if objective.startswith("binary:") or objective == "reg:logistic":
        y = (y > np.median(y)).astype(np.float32)
    if objective.startswith("multi:"):
        y = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3])).astype(np.float32)
        params["num_class"] = 3
    if objective in ["count:poisson", "reg:gamma", "reg:tweedie", "reg:squaredlogerror"]:
        y = np.exp(y / 4).astype(np.float32)
    if objective == "reg:pseudohubererror":
        params["base_score"] = 0.0
    expected, _ = reference(X, y, params)
    booster = xgb.train(params, xgb.DMatrix(X, label=y), 5)
    query = xgb.DMatrix(X)
    actual = booster.predict(query)
    np.testing.assert_allclose(actual, expected["prediction"], rtol=2e-5, atol=3e-5)
    np.testing.assert_array_equal(booster.predict(query).view(np.uint32), actual.view(np.uint32))
    margin = booster.predict(query, output_margin=True)
    np.testing.assert_allclose(margin, expected["margin"], rtol=2e-5, atol=3e-5)
    np.testing.assert_array_equal(booster.predict(query, output_margin=True).view(np.uint32), margin.view(np.uint32))
    np.testing.assert_array_equal(booster.predict(query).view(np.uint32), actual.view(np.uint32))

@pytest.mark.parametrize("field", ["weight", "missing", "base_margin", "regularization", "multioutput"])
def test_training_semantics(regression, reference, field):
    X, y = regression
    params = {"max_depth": 3, "max_bin": 256}
    meta = {}
    if field == "weight":
        meta["weight"] = np.linspace(0.1, 2.0, len(y)).astype(np.float32)
    if field == "missing":
        X = X.copy()
        X[::3, 0] = np.nan
        X[::5, 2] = np.nan
    if field == "base_margin":
        meta["base_margin"] = np.linspace(-0.5, 0.5, len(y)).astype(np.float32)
    if field == "regularization":
        params.update(reg_alpha=0.3, reg_lambda=2, gamma=0.2, max_delta_step=0.7)
    if field == "multioutput":
        y = np.column_stack([y, -0.5 * y])
    expected, _ = reference(X, y, params, **meta)
    booster = xgb.train(params, xgb.DMatrix(X, label=y, **meta), 5)
    np.testing.assert_allclose(booster.predict(xgb.DMatrix(X)), expected["prediction"],
                               rtol=2e-5, atol=3e-5)

def test_deterministic_and_subsampling(regression):
    X, y = regression
    d = xgb.DMatrix(X, label=y)
    params = dict(max_depth=4, subsample=0.7, colsample_bytree=0.7, seed=21)
    a, b = [xgb.train(params, d, 6) for _ in range(2)]
    np.testing.assert_array_equal(a.predict(d), b.predict(d))
    assert a.save_raw(raw_format="json") == b.save_raw(raw_format="json")

@pytest.mark.parametrize("kind", ["json", "ubj"])
def test_model_exchange(regression, reference, tmp_path, kind):
    X, y = regression
    params = {"max_depth": 3, "base_score": 0.0}
    expected, reference_dir = reference(X, y, params)
    imported = xgb.Booster(model_file=reference_dir / f"model.{kind}")
    np.testing.assert_allclose(imported.inplace_predict(X), expected["prediction"],
                               rtol=2e-6, atol=2e-6)
    mine = xgb.train(params, xgb.DMatrix(X, label=y), 5)
    path = tmp_path / f"metal.{kind}"
    mine.save_model(path)
    external, _ = reference(X, y, params, model=path)
    np.testing.assert_allclose(mine.inplace_predict(X), external["prediction"],
                               rtol=2e-6, atol=2e-6)

def test_sklearn_pickle_and_cv():
    X, y = make_regression(n_samples=128, n_features=5, random_state=9)
    reg = xgb.XGBRegressor(n_estimators=5, max_depth=3, device="metal").fit(X, y)
    assert reg.score(X, y) > 0.7
    np.testing.assert_array_equal(reg.predict(X), pickle.loads(pickle.dumps(reg)).predict(X))
    assert reg.apply(X).shape == (len(X), 5)
    assert reg.feature_importances_.sum() == pytest.approx(1.0)
    assert xgb.cv({"max_depth": 2}, xgb.DMatrix(X, label=y),
                  num_boost_round=3, nfold=3).shape == (3, 4)
    X, y = make_classification(n_samples=128, n_features=5, random_state=3)
    clf = xgb.XGBClassifier(n_estimators=5, max_depth=2).fit(X, y)
    assert clf.score(X, y) > 0.85
    np.testing.assert_allclose(clf.predict_proba(X).sum(axis=1), 1)

def test_callbacks_early_stopping_and_continuation(regression):
    X, y = regression
    d = xgb.DMatrix(X, label=y)
    history = {}
    b = xgb.train({"eta": 0, "eval_metric": ["rmse", "mae"]}, d, 12, [(d, "train")],
                  early_stopping_rounds=2, evals_result=history, verbose_eval=False)
    assert b.best_iteration == 0
    assert b.num_boosted_rounds() == 3
    assert len(history["train"]["mae"]) == 3
    params = {"eta": 0.2, "max_depth": 3}
    first = xgb.train(params, d, 3)
    continued = xgb.train(params, d, 2, xgb_model=first)
    full = xgb.train(params, d, 5)
    np.testing.assert_array_equal(continued.predict(d), full.predict(d))
    np.testing.assert_array_equal(full[:3].predict(d), first.predict(d))

def test_sparse_strides_and_missing(regression):
    X, y = regression
    X = X.copy()
    X[::4, 1] = 0
    csr = sparse.csr_matrix(X)
    dense = X.copy()
    dense[dense == 0] = np.nan
    a = xgb.train({"max_depth": 3}, xgb.DMatrix(csr, label=y), 4)
    b = xgb.train({"max_depth": 3}, xgb.DMatrix(dense, label=y), 4)
    np.testing.assert_array_equal(a.inplace_predict(csr), b.inplace_predict(dense))
    for view in [X[::-1], np.asfortranarray(X), X[:, ::-1]]:
        d = xgb.DMatrix(view, label=y)
        np.testing.assert_equal(d.get_data().toarray(), np.ascontiguousarray(view))

def test_monotone_constraint(regression):
    X, y = regression
    d = xgb.DMatrix(X, label=y)
    b = xgb.train({"max_depth": 4, "monotone_constraints": "(1,0,-1,0,0,0,0)"}, d, 6)
    grid = np.zeros((301, X.shape[1]), dtype=np.float32)
    grid[:, 0] = np.linspace(-3, 3, len(grid))
    assert (np.diff(b.inplace_predict(grid)) >= -1e-6).all()
    grid[:, 0] = 0
    grid[:, 2] = np.linspace(-3, 3, len(grid))
    assert (np.diff(b.inplace_predict(grid)) <= 1e-6).all()

def test_custom_objective(regression):
    X, y = regression
    d = xgb.DMatrix(X, label=y)
    def squared(pred, data):
        return pred - data.get_label(), np.ones_like(pred)
    params = {"base_score": 0.0, "max_depth": 3}
    a = xgb.train(params, d, 4, obj=squared)
    b = xgb.train(params, d, 4)
    np.testing.assert_array_equal(a.predict(d), b.predict(d))

def test_quantile_reference_and_metadata(regression):
    X, y = regression
    d = xgb.QuantileDMatrix(X, label=y, max_bin=64)
    test = xgb.QuantileDMatrix(X + 0.1, label=y, max_bin=64, ref=d)
    a, b = d.get_quantile_cut(), test.get_quantile_cut()
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    model = xgb.train({"max_bin": 64}, d, 2)
    assert model.predict(test).shape == (len(X),)
    with pytest.raises(xgb.core.XGBoostError, match="max_bin"):
        xgb.train({"max_bin": 32}, d, 1)

def test_gpu_profile_reports_execution(regression):
    X, y = regression
    b = xgb.train({"max_depth": 2}, xgb.DMatrix(X, label=y), 3)
    out = ctypes.c_char_p()
    assert xgb.core._LIB.XGBoosterMetalProfile(b.handle, ctypes.byref(out)) == 0
    profile = json.loads(out.value)
    assert profile["command_buffers"] > 0
    assert profile["tuning"]["fused_partition"]
    assert profile["gpu_seconds"]["tree_level"] > 0

@pytest.mark.parametrize("params,match", [({"device": "cpu"}, "only"),
    ({"device": "cuda"}, "only"), ({"max_depth": -1}, "nonnegative"),
    ({"subsample": -0.1}, "Sampling"), ({"reg_lambda": -1}, "nonnegative")])
def test_invalid_parameters(regression, params, match):
    X, y = regression
    with pytest.raises(xgb.core.XGBoostError, match=match):
        xgb.train(params, xgb.DMatrix(X, label=y), 1)
