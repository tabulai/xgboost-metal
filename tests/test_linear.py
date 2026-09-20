import ctypes
import json
import pickle

import numpy as np
import pytest
from scipy import sparse
import xgboost as xgb


def weights(model):
    return np.asarray(json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["weights"], dtype=np.float32)


def compare(model, data, expected, directory, atol=8e-7):
    source = json.loads((directory / "model.json").read_text())
    np.testing.assert_allclose(weights(model), source["learner"]["gradient_booster"]["model"]["weights"], rtol=5e-6, atol=atol)
    for name, options in [("prediction", {}), ("margin", {"output_margin": True}),
                          ("contributions", {"pred_contribs": True}),
                          ("approximate", {"pred_contribs": True, "approx_contribs": True})]:
        np.testing.assert_allclose(model.predict(data, **options), expected[name], rtol=5e-6, atol=atol * 4)
    actual = json.loads(model.save_config())["learner"]
    oracle = json.loads((directory / "config.json").read_text())["learner"]
    assert actual["gradient_booster"] == oracle["gradient_booster"]
    assert actual["generic_param"]["rng_state"] == oracle["generic_param"]["rng_state"]
    assert model.num_boosted_rounds() == source["learner"]["gradient_booster"]["model"]["boosted_rounds"]


@pytest.mark.parametrize("updater,selector", [("shotgun", "cyclic"), ("shotgun", "shuffle"),
    *[("coord_descent", s) for s in ["cyclic", "shuffle", "random", "greedy", "thrifty"]]])
def test_linear_coordinate_selectors(reference, updater, selector):
    rng = np.random.default_rng(416)
    X = rng.normal(size=(8201, 7)).astype(np.float32)
    y = (1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0, 0.1, len(X))).astype(np.float32)
    X[rng.random(X.shape) < 0.15] = np.nan
    X[:, -1] = np.nan
    weight = rng.uniform(0, 3, len(X)).astype(np.float32)
    weight[::11] = 0
    params = dict(booster="gblinear", updater=updater, feature_selector=selector, seed=79,
                  learning_rate=0.37, reg_alpha=0.013, reg_lambda=0.17)
    d = xgb.DMatrix(X, label=y, weight=weight)
    model = xgb.train(params, d, 6)
    expected, out = reference(X, y, params, rounds=6, weight=weight)
    compare(model, d, expected, out)
    profile = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(profile)))
    phases = json.loads(profile.value)["gpu_seconds"]
    assert phases["linear"] > 0 and "quantiles" not in phases and "tree" not in phases
    assert model.get_score()["f6"] == 0


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "binary:logitraw", "binary:hinge",
    "multi:softprob", "multi:softmax", "count:poisson", "reg:squaredlogerror", "reg:gamma", "reg:tweedie",
    "reg:pseudohubererror", "reg:absoluteerror", "reg:quantileerror", "reg:expectileerror", "reg:normal",
    "survival:aft", "survival:cox", "rank:pairwise", "rank:ndcg", "rank:map"])
def test_linear_objectives(reference, objective):
    rng = np.random.default_rng(55)
    X = rng.normal(size=(257, 5)).astype(np.float32)
    y = (1 + np.exp(0.2 * X[:, 0] - 0.1 * X[:, 1])).astype(np.float32)
    meta, params = {}, dict(booster="gblinear", updater="coord_descent", objective=objective, eta=0.1, reg_lambda=0.2)
    if objective.startswith("binary") or objective == "rank:map": y = (X[:, 0] > 0).astype(np.float32)
    if objective.startswith("multi"):
        y = rng.integers(0, 3, len(X)).astype(np.float32)
        params["num_class"] = 3
    if objective == "reg:quantileerror": params["quantile_alpha"] = [0.2, 0.8]
    if objective == "reg:expectileerror": params["expectile_alpha"] = [0.2, 0.8]
    if objective == "survival:aft":
        meta = dict(label_lower_bound=y, label_upper_bound=np.where(np.arange(len(X)) % 3, y, np.inf))
        y = None
    if objective == "survival:cox": y[::3] *= -1
    if objective.startswith("rank"):
        meta["group"] = np.array([64, 64, 64, 65], dtype=np.uint32)
        if objective != "rank:map": y = rng.integers(0, 4, len(X)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    d = xgb.DMatrix(X, label=y, **meta)
    history = {}
    model = xgb.train(params, d, 4, evals=[(d, "train")], evals_result=history, verbose_eval=False)
    expected, out = reference(X, y, params, rounds=4, **meta)
    compare(model, d, expected, out, atol=3e-6)
    source_history = json.loads((out / "history.json").read_text())["train"]
    for metric, values in history["train"].items():
        np.testing.assert_allclose(values, source_history[metric], rtol=5e-6, atol=5e-6)


@pytest.mark.parametrize("selector", ["cyclic", "shuffle", "random", "greedy", "thrifty"])
@pytest.mark.parametrize("features,top_k", [(1, 0), (17, 1), (17, 0)])
def test_linear_selector_ties_and_top_k(reference, selector, features, top_k):
    rng = np.random.default_rng(96)
    X = np.tile(rng.normal(size=(67, 1)), (1, features)).astype(np.float32)
    X[:, ::3] *= -1
    y = np.column_stack([X[:, 0] + 2, -X[:, 0] - 1]).astype(np.float32)
    params = dict(booster="gblinear", updater="coord_descent", feature_selector=selector, top_k=top_k,
                  seed=983, seed_per_iteration=True, base_score=[0.2, 1.3])
    d = xgb.DMatrix(X, label=y)
    model = xgb.train(params, d, 4)
    expected, out = reference(X, y, params, rounds=4)
    compare(model, d, expected, out)
    if features == 1 and selector in ["greedy", "thrifty"]:
        np.testing.assert_array_equal(weights(model)[:2], [0, 0])
    assert model.predict(d, pred_contribs=True, strict_shape=True).shape == (len(X), 2, features + 1)


@pytest.mark.parametrize("selector", ["cyclic", "shuffle", "random", "greedy", "thrifty"])
def test_linear_model_exchange_and_continuation(reference, tmp_path, selector):
    rng = np.random.default_rng(320)
    X = rng.normal(size=(93, 6)).astype(np.float32)
    y = np.column_stack([X[:, 0] + 0.3, -X[:, 1]]).astype(np.float32)
    margin = rng.normal(0, 0.1, y.shape).astype(np.float32)
    params = dict(booster="gblinear", updater="coord_descent", feature_selector=selector,
                  learning_rate=0.35, reg_alpha=0.1, reg_lambda=0.3, seed=54)
    d = xgb.DMatrix(X, label=y, base_margin=margin)
    model = xgb.train(params, d, 3)
    expected, out = reference(X, y, params, rounds=3, base_margin=margin, test_margin=margin)
    compare(model, d, expected, out)
    for ext in ["json", "ubj"]:
        imported = xgb.Booster(model_file=out / ("model." + ext))
        np.testing.assert_array_equal(imported.predict(d), expected["prediction"])
        path = tmp_path / ("metal." + ext)
        model.save_model(path)
        received, _ = reference(X, y, params, model=path, base_margin=margin, test_margin=margin)
        np.testing.assert_allclose(received["prediction"], model.predict(d), atol=8e-7, rtol=3e-6)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.save_raw() == model.save_raw()
    resumed = xgb.train(params, d, 4, xgb_model=restored)
    wanted, directory = reference(X, y, params, model=out / "model.ubj", continue_rounds=4, base_margin=margin, test_margin=margin)
    # A model file does not contain RNG state. Match that continuation path.
    from_file = xgb.train(params, d, 4, xgb_model=out / "model.ubj")
    compare(from_file, d, wanted, directory)
    assert resumed.num_boosted_rounds() == 7
    legacy = json.loads(model.save_raw(raw_format="json"))
    del legacy["learner"]["gradient_booster"]["model"]["boosted_rounds"]
    old = xgb.Booster(model_file=bytearray(json.dumps(legacy).encode()))
    assert old.num_boosted_rounds() == 0
    np.testing.assert_array_equal(old.predict(d), model.predict(d))


@pytest.mark.parametrize("tolerance", [0, 1e-4, 100])
def test_linear_convergence_and_reconfiguration(reference, tolerance):
    rng = np.random.default_rng(13)
    X = rng.normal(size=(71, 5)).astype(np.float32)
    y = (X[:, 0] - X[:, 2]).astype(np.float32)
    params = dict(booster="gblinear", updater="coord_descent", feature_selector="shuffle", tolerance=tolerance, seed=413)
    updates = {"3": {"eta": 0.7}, "6": {"feature_selector": "random"}, "9": {"tolerance": 0}, "11": {"tolerance": tolerance}}
    class Updates(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            if str(epoch) in updates: model.set_param(updates[str(epoch)])
            return False
    d = xgb.DMatrix(X, label=y)
    model = xgb.train(params, d, 14, callbacks=[Updates()])
    expected, out = reference(X, y, params, rounds=14, param_updates=updates)
    compare(model, d, expected, out)


def test_linear_sparse_missing_and_sklearn(reference):
    rng = np.random.default_rng(13)
    X = rng.normal(size=(137, 9)).astype(np.float32)
    X[rng.random(X.shape) < 0.6] = 0
    y = (X[:, 0] - 0.6 * X[:, 2] + 0.4).astype(np.float32)
    csr = sparse.csr_matrix(X)
    dense_missing = np.where(X == 0, np.nan, X)
    params = dict(booster="gblinear", updater="coord_descent", reg_lambda=0.15)
    expected, out = reference(dense_missing, y, params, rounds=8)
    estimator = xgb.XGBRegressor(n_estimators=8, **params).fit(csr, y)
    compare(estimator.get_booster(), xgb.DMatrix(csr), expected, out)
    dump = json.loads(json.loads((out / "dump.json").read_text())[0])
    np.testing.assert_allclose(estimator.coef_, dump["weight"], atol=1e-6)
    np.testing.assert_allclose(estimator.intercept_, dump["bias"], atol=1e-6)
    score = np.array(list(estimator.get_booster().get_score().values()))
    np.testing.assert_allclose(estimator.feature_importances_, score / score.sum(), atol=1e-6)
    np.testing.assert_allclose(estimator.predict(csr), expected["prediction"], atol=1e-6)
    classifier = xgb.XGBClassifier(n_estimators=6, **params).fit(csr, (y > 0.4).astype(int))
    assert classifier.coef_.shape == (X.shape[1],)
    np.testing.assert_allclose(classifier.predict_proba(csr).sum(axis=1), 1, atol=1e-7)


def test_linear_prediction_errors_and_ranges(reference):
    X = np.arange(51, dtype=np.float32).reshape(17, 3) / 20
    y = X[:, 0]
    d = xgb.DMatrix(X, label=y)
    model = xgb.train({"booster": "gblinear"}, d, 2)
    expected, out = reference(X, y, {"booster": "gblinear"}, rounds=2)
    compare(model, d, expected, out)
    assert set(json.loads((out / "prediction_errors.json").read_text())) == {"leaf", "interactions", "approximate_interactions"}
    for options in [dict(pred_leaf=True), dict(pred_interactions=True), dict(pred_interactions=True, approx_contribs=True), dict(iteration_range=(1, 2))]:
        with pytest.raises(xgb.core.XGBoostError): model.predict(d, **options)
    np.testing.assert_array_equal(model.predict(d, iteration_range=(0, 100)), model.predict(d))
    with pytest.raises(xgb.core.XGBoostError, match="Inplace"): model.inplace_predict(X)
    with pytest.raises(xgb.core.XGBoostError, match="Slice"): model[:1]
    with pytest.raises(xgb.core.XGBoostError, match="weight"): model.get_score(importance_type="gain")
    with pytest.raises(xgb.core.XGBoostError, match="Dump format"): model.get_dump(dump_format="dot")
    padded = X.copy(); padded[:, -1] = np.nan
    np.testing.assert_array_equal(model.predict(xgb.DMatrix(X[:, :2])), model.predict(xgb.DMatrix(padded)))
    with pytest.raises(xgb.core.XGBoostError, match="shape"):
        model.predict(xgb.DMatrix(X[:, :2]), pred_contribs=True)
    np.testing.assert_array_equal(model.predict(xgb.DMatrix(X[:, :2]), pred_interactions=True), np.zeros((len(X), 3, 3)))
    assert model.predict(xgb.DMatrix(X[:0]), pred_interactions=True).shape == (0, 4, 4)
    with pytest.raises(xgb.core.XGBoostError, match="QuantileDMatrix"):
        xgb.train({"booster": "gblinear"}, xgb.QuantileDMatrix(X, label=y), 1)
    with pytest.raises(xgb.core.XGBoostError, match="Categorical"):
        xgb.train({"booster": "gblinear"}, xgb.DMatrix(X, label=y, feature_types=["c", "q", "q"]), 1)


@pytest.mark.parametrize("params", [dict(updater="gpu_coord_descent"), dict(updater="missing"),
    dict(feature_selector="random"), dict(feature_selector="greedy"), dict(feature_selector="thrifty"),
    dict(feature_selector="invalid"), dict(tolerance=-1), dict(reg_lambda=-1), dict(reg_alpha=-1),
    dict(learning_rate=-1), dict(updater="coord_descent", top_k=-1)])
def test_linear_parameter_validation(params):
    d = xgb.DMatrix(np.ones((3, 2), dtype=np.float32), label=[1, 2, 3])
    with pytest.raises(xgb.core.XGBoostError): xgb.train({"booster": "gblinear", **params}, d, 1)


def test_linear_instance_weight_mass_is_frozen(reference):
    rng = np.random.default_rng(221)
    X = rng.normal(size=(49, 4)).astype(np.float32)
    y = (X[:, 0] + 0.4 * X[:, 1]).astype(np.float32)
    weight = rng.uniform(0.2, 2, len(X)).astype(np.float32)
    updates = {"2": {"weight": (weight * 17).tolist()}, "4": {"weight": (weight / 9).tolist(), "label": (y + 0.7).tolist()}}
    d = xgb.DMatrix(X, label=y, weight=weight)
    class Updates(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            for field, values in updates.get(str(epoch), {}).items(): d.set_float_info(field, values)
            return False
    params = dict(booster="gblinear", updater="coord_descent", reg_alpha=0.03, reg_lambda=0.6)
    model = xgb.train(params, d, 7, callbacks=[Updates()])
    expected, out = reference(X, y, params, rounds=7, weight=weight, matrix_updates=updates)
    compare(model, d, expected, out)


@pytest.mark.parametrize("case", ["masked", "wide", "cancellation", "zero"])
def test_linear_custom_gradient_ranges(reference, case):
    rng = np.random.default_rng(27)
    X = rng.normal(size=(1025, 4)).astype(np.float32)
    g = rng.normal(size=len(X)).astype(np.float32)
    h = rng.uniform(0.1, 2, len(X)).astype(np.float32)
    y = np.zeros(len(X), dtype=np.float32)
    params = dict(booster="gblinear", updater="coord_descent", base_score=0, eta=0.2)
    meta = {}
    if case == "masked":
        h[::7] *= -1; h[1::11] = 0
        X[:, -1] = 1e-5
        X[rng.random(X.shape) < 0.1] = np.nan
    elif case == "wide":
        X = (np.abs(X) + 0.1) * np.float32(1e8)
        g = (np.abs(g) + 0.1) * np.float32(1e30)
        h[:] = 1
        params.update(eta=1e-20, reg_lambda=1e-19)
        meta["weight"] = np.full(len(X), 3e36, dtype=np.float32)
    elif case == "cancellation":
        X[:] = 1
        g[:] = np.where(np.arange(len(X)) % 2, -1e14, 1e14)
        g[-1] = 1; h[:] = 1
    else:
        h[:] = 0
        meta["weight"] = np.zeros(len(X), dtype=np.float32)
    original_g, original_h = g.copy(), h.copy()
    d = xgb.DMatrix(X, label=y, **meta)
    model = xgb.train(params, d, 3, obj=lambda prediction, data: (g, h))
    expected, out = reference(X, y, params, rounds=3, grad=g, hess=h, **meta)
    compare(model, d, expected, out)
    np.testing.assert_array_equal(g, original_g)
    np.testing.assert_array_equal(h, original_h)


def test_linear_ignores_tree_parameters(reference):
    X = np.arange(30, dtype=np.float32).reshape(10, 3) / 30
    y = np.sin(X[:, 0])
    params = dict(booster="gblinear", tree_method="exact", multi_strategy="multi_output_tree",
                  subsample=0.2, colsample_bytree=0.1, max_depth=0, max_leaves=0,
                  num_parallel_tree=0, process_type="update", rate_drop=0.8, one_drop=True)
    d = xgb.DMatrix(X, label=y)
    model = xgb.train(params, d, 4)
    expected, out = reference(X, y, params, rounds=4)
    compare(model, d, expected, out)


def test_linear_named_multioutput_scores_and_cv(reference):
    import pandas as pd
    rng = np.random.default_rng(387)
    X = pd.DataFrame(rng.normal(size=(83, 4)).astype(np.float32), columns=["height", "width", "depth", "offset"])
    y = np.column_stack([X.height - X.width, X.depth + 0.1]).astype(np.float32)
    params = dict(booster="gblinear", updater="coord_descent")
    d = xgb.DMatrix(X, label=y)
    model = xgb.train(params, d, 5)
    expected, out = reference(X, y, params, rounds=5)
    compare(model, d, expected, out)
    assert list(model.get_score()) == list(X.columns)
    np.testing.assert_array_equal(np.array(list(model.get_score().values())), weights(model)[:-2].reshape(4, 2))
    cv = xgb.cv(params, xgb.DMatrix(X, label=y[:, 0]), 5, nfold=3, seed=42)
    assert len(cv) == 5 and np.isfinite(cv.to_numpy()).all()
    assert cv["test-rmse-mean"].iloc[-1] < cv["test-rmse-mean"].iloc[0]


@pytest.mark.parametrize("initial", ["gbtree", "gblinear"])
def test_linear_switching_boosters_replaces_coefficients(reference, initial):
    rng = np.random.default_rng(548)
    X = rng.normal(size=(67, 5)).astype(np.float32)
    y = (X[:, 0] - X[:, 1]).astype(np.float32)
    params = dict(booster=initial, max_depth=2, tree_method="hist", eta=0.15)
    updates = {"2": {"booster": "gbtree"}, "4": {"booster": "gblinear", "updater": "coord_descent", "eta": 0.7}}
    class Updates(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            if str(epoch) in updates: model.set_param(updates[str(epoch)])
            return False
    d = xgb.DMatrix(X, label=y)
    model = xgb.train(params, d, 6, callbacks=[Updates()])
    expected, out = reference(X, y, params, rounds=6, param_updates=updates)
    compare(model, d, expected, out)
    assert model.num_boosted_rounds() == 2


@pytest.mark.parametrize("explicit", [False, True])
def test_linear_booster_settings_are_recreated(reference, explicit):
    rng = np.random.default_rng(922)
    X = rng.normal(size=(49, 4)).astype(np.float32)
    y = (X[:, 0] - X[:, 1]).astype(np.float32)
    params = dict(booster="gblinear", updater="coord_descent", feature_selector="greedy", tolerance=10, eta=0.12)
    # The booster key is deliberately last: clearing previous state must happen
    # before applying any of the new batch, not while visiting that key.
    final = dict(eta=0.6, updater="coord_descent", tolerance=0.01, booster="gblinear") if explicit else dict(booster="gblinear")
    updates = {"1": {"booster": "gbtree", "max_depth": 2}, "3": final}
    class Updates(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            if str(epoch) in updates: model.set_param(updates[str(epoch)])
            return False
    d = xgb.DMatrix(X, label=y)
    model = xgb.train(params, d, 5, callbacks=[Updates()])
    expected, out = reference(X, y, params, rounds=5, param_updates=updates)
    compare(model, d, expected, out)


def test_linear_empty_parameter_batch_preserves_updater(reference):
    rng = np.random.default_rng(563)
    X = rng.normal(size=(71, 4)).astype(np.float32)
    y = (X[:, 0] - X[:, 2]).astype(np.float32)
    params = dict(booster="gblinear", updater="coord_descent", feature_selector="random", seed=76, reg_lambda=0.3, eta=0.2)
    class Updates(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            if epoch == 2: model.set_param({})
            return False
    d = xgb.DMatrix(X, label=y)
    model = xgb.train(params, d, 5, callbacks=[Updates()])
    expected, out = reference(X, y, params, rounds=5, param_updates={"2": {}})
    compare(model, d, expected, out)


def test_linear_round_count_is_configured_before_continuation(reference):
    rng = np.random.default_rng(547)
    X = rng.normal(size=(53, 4)).astype(np.float32)
    y = (X[:, 0] - X[:, 1]).astype(np.float32)
    d = xgb.DMatrix(X, label=y)
    tree_params = dict(tree_method="hist", max_depth=2)
    tree = xgb.train(tree_params, d, 3)
    _, out = reference(X, y, tree_params, rounds=3)
    tree.set_param("booster", "gblinear")
    assert tree.num_boosted_rounds() == 0
    assert tree.num_features() == 4
    params = dict(booster="gblinear", updater="coord_descent", feature_selector="shuffle", seed=19, seed_per_iteration=True)
    model = xgb.train(params, d, 4, xgb_model=out / "model.ubj")
    expected, directory = reference(X, y, params, model=out / "model.ubj", continue_rounds=4)
    compare(model, d, expected, directory)
    assert model.num_boosted_rounds() == 4
