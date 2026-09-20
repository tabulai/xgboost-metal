import ctypes
import json
import pickle

import numpy as np
import pytest
import xgboost as xgb


def dataset(rows=521):
    rng = np.random.default_rng(281)
    X = rng.integers(-8, 9, (rows, 5)).astype(np.float32) / 4
    y = (1.7 * X[:, 0] + X[:, 1] * X[:, 2] - 0.3 * X[:, 3] ** 2
         + rng.normal(0, 0.03, rows)).astype(np.float32)
    X[::13, 1] = np.nan
    X[::17, 3] = np.nan
    return X, y


def document(model):
    return json.loads(model.save_raw(raw_format="json"))


def compare(model, data, expected, files, interpretation=False):
    own = document(model)["learner"]["gradient_booster"]["model"]
    source = json.loads((files / "model.json").read_text())["learner"]["gradient_booster"]["model"]
    assert own["iteration_indptr"] == source["iteration_indptr"]
    assert own["tree_info"] == source["tree_info"]
    np.testing.assert_array_equal(np.asarray(own.get("weight_drop", []), dtype=np.float32),
                                  np.asarray(source.get("weight_drop", []), dtype=np.float32))
    for a, b in zip(own["trees"], source["trees"]):
        assert a["tree_param"] == b["tree_param"]
        for key in ["left_children", "right_children", "parents", "split_indices", "default_left", "split_type",
                    "categories", "categories_nodes", "categories_segments", "categories_sizes"]:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        for key in ["split_conditions", "base_weights", "sum_hessian", "loss_changes"]:
            np.testing.assert_allclose(a[key], b[key], rtol=3e-5,
                                       atol=5e-4 if key == "loss_changes" else 3e-5, err_msg=key)
    actual_config = json.loads(model.save_config())["learner"]
    source_config = json.loads((files / "config.json").read_text())["learner"]
    assert actual_config["generic_param"]["rng_state"] == source_config["generic_param"]["rng_state"]
    for name, options in [("prediction", {}), ("margin", {"output_margin": True})]:
        np.testing.assert_allclose(model.predict(data, **options), expected[name], rtol=4e-5, atol=4e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    if interpretation:
        for name, options in [("contributions", {"pred_contribs": True}),
                              ("interactions", {"pred_interactions": True})]:
            np.testing.assert_allclose(model.predict(data, **options), expected[name], rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("rows,fraction", [(1, 0.5), (17, 0.7), (257, 0.25), (4097, 0.8),
                                          (8193, 0.001), (127, 0), (257, 1)])
def test_sampling_budget_and_boundaries(reference, method, rows, fraction):
    X, y = dataset(rows)
    params = dict(sampling_method=method, subsample=fraction, max_depth=3, max_bin=32,
                  seed=73, base_score=0.25, eta=0.2)
    expected, files = reference(X, y, params, rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 3)
    compare(model, data, expected, files)


@pytest.mark.parametrize("method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("parallel", [1, 3])
@pytest.mark.parametrize("outputs", [1, 3])
@pytest.mark.parametrize("per_iteration", [False, True])
def test_sampling_parallel_groups_and_rng(reference, method, parallel, outputs, per_iteration):
    X, y = dataset()
    if outputs > 1:
        y = np.column_stack([y, np.cos(y), -0.7 * y])
    params = dict(sampling_method=method, subsample=0.63, num_parallel_tree=parallel,
                  max_depth=3, max_bin=32, seed=239, seed_per_iteration=per_iteration,
                  colsample_bytree=0.8, colsample_bylevel=0.75, colsample_bynode=0.7,
                  base_score=0.25, eta=0.2)
    meta = dict(weight=np.linspace(0.1, 1.7, len(X), dtype=np.float32),
                feature_weights=np.linspace(0.2, 1, X.shape[1], dtype=np.float32))
    expected, files = reference(X, y, params, rounds=4, **meta)
    data = xgb.DMatrix(X, label=y, **meta)
    model = xgb.train(params, data, 4)
    compare(model, data, expected, files)


@pytest.mark.parametrize("method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("objective,extra", [
    ("binary:logistic", {}), ("multi:softprob", {"num_class": 3}),
    ("reg:absoluteerror", {}), ("reg:quantileerror", {"quantile_alpha": [0.2, 0.8]}),
    ("reg:expectileerror", {"expectile_alpha": [0.2, 0.8]}), ("reg:normal", {}),
    ("survival:aft", {"aft_loss_distribution": "logistic"}), ("survival:cox", {"max_delta_step": 1}),
    ("rank:ndcg", {"lambdarank_pair_method": "mean", "lambdarank_num_pair_per_sample": 3}),
])
def test_sampling_objective_gradients(reference, method, objective, extra):
    X, y = dataset(259)
    meta = {}
    if objective == "binary:logistic":
        y = (y > 0).astype(np.float32)
    elif objective == "multi:softprob":
        y = np.arange(len(X), dtype=np.float32) % 3
    elif objective.startswith("survival:"):
        y = np.exp(y * 0.05).astype(np.float32)
        if objective == "survival:aft":
            meta = dict(label_lower_bound=y, label_upper_bound=np.where(np.arange(len(y)) % 3, y, np.inf))
        else:
            y[::3] *= -1
    elif objective.startswith("rank:"):
        y = (np.arange(len(y)) % 5).astype(np.float32)
        meta = dict(group=[37] * 7, weight=np.linspace(0.3, 1.3, 7, dtype=np.float32))
    params = dict(objective=objective, sampling_method=method, subsample=0.61, seed=129,
                  base_score=0.5, max_depth=3, max_bin=32, eta=0.1, **extra)
    expected, files = reference(X, y, params, rounds=4, **meta)
    data = xgb.DMatrix(X, label=y, **meta)
    model = xgb.train(params, data, 4)
    compare(model, data, expected, files)


@pytest.mark.parametrize("method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("fraction", [0.63, 1])
@pytest.mark.parametrize("kind", ["negative_hessian", "zero", "constant", "tiny", "overflow"])
def test_sampling_custom_gradients(reference, method, fraction, kind):
    X, y = dataset(1025)
    grad = np.sin(np.arange(len(X), dtype=np.float32) * 0.37)
    hess = np.full_like(grad, 1.25)
    if kind == "negative_hessian":
        hess[::3] = -0.5
        grad[::3] += 0.15
    elif kind == "zero":
        grad[:] = hess[:] = 0
    elif kind == "constant":
        grad[:] = 0.7
    elif kind == "tiny":
        grad *= np.float32(1e-12)
        hess *= np.float32(1e-12)
    elif kind == "overflow":
        grad[:] = np.float32(1e20)
    params = dict(sampling_method=method, subsample=fraction, seed=31, base_score=0,
                  eta=0.2, max_depth=3, max_bin=32, min_child_weight=0)
    expected, files = reference(X, y, params, rounds=2, grad=grad, hess=hess)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 2, obj=lambda p, d: (grad, hess))
    compare(model, data, expected, files)


@pytest.mark.parametrize("method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("parallel", [1, 3])
@pytest.mark.parametrize("refresh_leaf", [False, True])
def test_sampling_grow_then_refresh(reference, method, parallel, refresh_leaf):
    X, y = dataset(513)
    y = np.column_stack([y, -0.6 * y])
    params = dict(updater="grow_quantile_histmaker,refresh,prune", sampling_method=method,
                  subsample=0.67, num_parallel_tree=parallel, refresh_leaf=refresh_leaf,
                  max_depth=3, max_bin=32, gamma=0.2, seed=817, base_score=0.25)
    expected, files = reference(X, y, params, rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 3)
    compare(model, data, expected, files, interpretation=True)


@pytest.mark.parametrize("method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("coalesce,fill", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_sampling_encoders_and_histogram_paths(method, coalesce, fill):
    X, y = dataset(769)
    X[:, 0] = np.arange(len(X), dtype=np.float32) % 37
    data = xgb.DMatrix(X, label=y, feature_types=["c", "q", "q", "q", "q"], enable_categorical=True)
    params = dict(sampling_method=method, subsample=0.59, num_parallel_tree=2, seed=193,
                  grow_policy="lossguide", max_depth=0, max_leaves=15, max_bin=64)
    expected = xgb.train(params, data, 3)
    actual = xgb.train({**params, "metal_coalesce_encoders": coalesce, "metal_compute_fill": fill,
                        "metal_histogram_subtraction": 0, "metal_partition_scan": 0,
                        "metal_partition_aggregation": 0, "metal_tree_workspace": 0,
                        "metal_parallel_splits": 0}, data, 3)
    assert expected.save_raw(raw_format="json") == actual.save_raw(raw_format="json")
    np.testing.assert_array_equal(expected.predict(data), actual.predict(data))
    profile = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(actual.handle, ctypes.byref(profile)))
    recorded = json.loads(profile.value)
    assert recorded["gpu_seconds"]["sampling"] > 0
    assert recorded["tuning"]["sampling_method"] == (2 if method == "gradient_based" else 1)


@pytest.mark.parametrize("field", ["colsample_bytree", "colsample_bylevel", "colsample_bynode", "all"])
def test_zero_column_sampling_retains_one_feature(reference, field):
    X, y = dataset()
    columns = [field] if field != "all" else ["colsample_bytree", "colsample_bylevel", "colsample_bynode"]
    params = dict(sampling_method="gradient_based", subsample=0.7, max_depth=3, max_bin=32,
                  seed=19, base_score=0, **{name: 0 for name in columns})
    expected, files = reference(X, y, params, rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 3)
    compare(model, data, expected, files)


@pytest.mark.parametrize("method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("format", ["json", "ubj"])
def test_sampling_model_exchange_and_continuation(reference, tmp_path, method, format):
    X, y = dataset(777)
    X[:, 0] = np.arange(len(X), dtype=np.float32) % 29
    types = ["c", "q", "q", "q", "q"]
    params = dict(sampling_method=method, subsample=0.67, seed=45, base_score=0.25,
                  max_depth=3, max_bin=32, rate_drop=0.4, one_drop=True,
                  colsample_bytree=0.8, colsample_bynode=0.7)
    expected, files = reference(X, y, params, rounds=3, feature_types=types)
    data = xgb.DMatrix(X, label=y, feature_types=types, enable_categorical=True)
    model = xgb.train(params, data, 3)
    compare(model, data, expected, files, interpretation=True)
    path = tmp_path / f"model.{format}"
    model.save_model(path)
    np.testing.assert_array_equal(xgb.Booster(model_file=path).predict(data), model.predict(data))
    restored = pickle.loads(pickle.dumps(model))
    complete = xgb.train(params, data, 6)
    continued = xgb.train(params, data, 3, xgb_model=restored)
    assert continued.save_raw(raw_format="json") == complete.save_raw(raw_format="json")
    expected_continued, continued_files = reference(X, y, params, model=path, continue_rounds=2, feature_types=types)
    continued = xgb.train(params, data, 2, xgb_model=path)
    compare(continued, data, expected_continued, continued_files)


@pytest.mark.parametrize("feature_type", ["q", "c"])
@pytest.mark.parametrize("minimum", [0.25, 1, 4])
@pytest.mark.parametrize("direction", ["below", "equal", "above"])
def test_minimum_child_weight_uses_unrounded_sum(reference, feature_type, minimum, direction):
    X = np.asarray([0, 0, 1, 1], dtype=np.float32)[:, None]
    y = np.zeros(len(X), dtype=np.float32)
    hess = np.asarray([minimum / 2, minimum / 2, minimum, minimum], dtype=np.float32)
    if direction != "equal":
        hess[0] = np.nextafter(hess[0], np.float32(0 if direction == "below" else np.inf))
    grad = np.asarray([-1, -1, 1, 1], dtype=np.float32)
    params = dict(min_child_weight=minimum, base_score=0, eta=1, max_depth=1)
    expected, files = reference(X, y, params, rounds=1, feature_types=[feature_type], grad=grad, hess=hess)
    data = xgb.DMatrix(X, label=y, feature_types=[feature_type], enable_categorical=True)
    model = xgb.train(params, data, 1, obj=lambda p, d: (grad, hess))
    compare(model, data, expected, files)
    trees = document(model)["learner"]["gradient_booster"]["model"]["trees"]
    assert len(trees[0]["left_children"]) == (1 if direction == "below" else 3)


@pytest.mark.parametrize("method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("parallel", [1, 3])
def test_sampling_after_large_automatic_intercept(reference, method, parallel):
    rng = np.random.default_rng(43)
    X = (rng.integers(0, 33, size=(100000, 32)).astype(np.float32) - 16) / 8
    y = (2 * X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 0.1, len(X))).astype(np.float32)
    params = dict(max_depth=6, max_bin=64, seed=9, subsample=0.6,
                  sampling_method=method, num_parallel_tree=parallel)
    expected, files = reference(X, y, params, rounds=30, test=X[:64])
    model = xgb.train(params, xgb.DMatrix(X, label=y), 30)
    own = document(model)["learner"]["learner_model_param"]
    source = json.loads((files / "model.json").read_text())["learner"]["learner_model_param"]
    np.testing.assert_array_equal(np.asarray(json.loads(own["base_score"]), np.float32),
                                  np.asarray(json.loads(source["base_score"]), np.float32))
    compare(model, xgb.DMatrix(X[:64]), expected, files)
