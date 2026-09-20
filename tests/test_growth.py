import ctypes
import itertools
import json
import pickle

import numpy as np
import pytest
import xgboost as xgb


def growth_data():
    rng = np.random.default_rng(819)
    X = rng.integers(-12, 13, (1297, 9)).astype(np.float32) / np.float32(4)
    y = (2 * X[:, 0] - X[:, 1] + X[:, 2] * (X[:, 3] > 0)
         + np.sin(X[:, 4]) + rng.normal(0, 0.1, len(X))).astype(np.float32)
    X[::13, 0] = np.nan
    X[::19, 3] = np.nan
    return X, y


def forest(document):
    return document["learner"]["gradient_booster"]["model"]["trees"]


def compare(model, data, expected, files):
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=4e-5, atol=4e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    own_config = json.loads(model.save_config())["learner"]
    source_config = json.loads((files / "config.json").read_text())["learner"]
    assert own_config["generic_param"]["rng_state"] == source_config["generic_param"]["rng_state"]
    own = forest(json.loads(model.save_raw(raw_format="json")))
    source = forest(json.loads((files / "model.json").read_text()))
    assert len(own) == len(source)
    for a, b in zip(own, source):
        for field in ("left_children", "right_children", "split_indices", "default_left", "split_type",
                      "categories", "categories_nodes", "categories_segments", "categories_sizes"):
            np.testing.assert_array_equal(a[field], b[field], err_msg=field)
        np.testing.assert_allclose(a["split_conditions"], b["split_conditions"], rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(a["base_weights"], b["base_weights"], rtol=4e-5, atol=4e-5)


def profile(model):
    result = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(result)))
    return json.loads(result.value)


def depths(tree):
    result = np.zeros(len(tree["left_children"]), dtype=np.int32)
    for i, (left, right) in enumerate(zip(tree["left_children"], tree["right_children"])):
        if left >= 0:
            result[left] = result[i] + 1
            result[right] = result[i] + 1
    return result


@pytest.mark.parametrize("policy", ["depthwise", "lossguide"])
@pytest.mark.parametrize("depth,leaves", [(0, 1), (0, 2), (0, 3), (0, 7), (0, 31),
                                          (1, 17), (3, 17), (5, 7), (31, 17), (6, 0)])
def test_growth_limits(reference, policy, depth, leaves):
    X, y = growth_data()
    params = dict(grow_policy=policy, max_depth=depth, max_leaves=leaves, max_bin=32,
                  base_score=0, eta=0.2, reg_alpha=0.01)
    expected, files = reference(X, y, params, rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 3)
    compare(model, data, expected, files)
    for tree in forest(json.loads(model.save_raw(raw_format="json"))):
        if leaves:
            assert np.count_nonzero(np.array(tree["left_children"]) < 0) <= leaves
        if depth:
            assert depths(tree).max() <= depth


@pytest.mark.parametrize("tie", [False, True])
def test_gain_priority_and_node_id_ties(reference, tie):
    X = np.repeat(np.array(list(itertools.product([0, 1], repeat=3)), dtype=np.float32), 17, axis=0)
    y = 8 * X[:, 0] + (1 if tie else (1 + 3 * X[:, 0])) * X[:, 1] + 0.125 * X[:, 2]
    data = xgb.DMatrix(X, label=y)
    for policy in ["depthwise", "lossguide"]:
        params = dict(grow_policy=policy, max_leaves=3, max_depth=0, reg_lambda=0, eta=1, base_score=0)
        expected, files = reference(X, y, params, rounds=1)
        model = xgb.train(params, data, 1)
        compare(model, data, expected, files)
        tree = forest(json.loads(model.save_raw(raw_format="json")))[0]
        selected = 1 if tie or policy == "depthwise" else 2
        assert tree["left_children"][selected] == 3
        assert tree["left_children"][3 - selected] == -1


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("per_iteration", [False, True])
@pytest.mark.parametrize("custom", [False, True])
def test_lossguide_column_sampling_by_actual_depth(reference, weighted, per_iteration, custom):
    X, y = growth_data()
    params = dict(grow_policy="lossguide", max_depth=0, max_leaves=23, max_bin=32,
                  colsample_bytree=0.8, colsample_bylevel=0.7, colsample_bynode=0.6,
                  seed=173, seed_per_iteration=per_iteration, base_score=0)
    meta = {"feature_weights": np.linspace(0.2, 1, X.shape[1], dtype=np.float32)} if weighted else {}
    expected, files = reference(X, y, params, rounds=5, custom_square=custom, **meta)
    data = xgb.DMatrix(X, label=y, **meta)
    objective = (lambda prediction, data: (prediction - data.get_label(), np.ones_like(prediction))) if custom else None
    model = xgb.train(params, data, 5, obj=objective)
    compare(model, data, expected, files)


def test_depthwise_batches_preserve_rng_and_node_order(reference):
    X = np.array(list(itertools.product([0, 1], repeat=11)), dtype=np.float32)
    y = X @ np.arange(1, 12, dtype=np.float32)
    params = dict(grow_policy="depthwise", max_depth=11, max_leaves=800, max_bin=8,
                  colsample_bynode=0.9, base_score=0, reg_lambda=0, eta=1, seed=733)
    expected, files = reference(X, y, params, rounds=1, test=X[:17])
    model = xgb.train(params, xgb.DMatrix(X, label=y), 1)
    compare(model, xgb.DMatrix(X[:17]), expected, files)
    tree = forest(json.loads(model.save_raw(raw_format="json")))[0]
    assert np.bincount(depths(tree)).max() > 256
    assert len(tree["left_children"]) == 1599


@pytest.mark.parametrize("policy", ["depthwise", "lossguide"])
def test_trees_deeper_than_thirty(reference, policy):
    X = np.arange(41, dtype=np.float32)[:, None]
    y = np.where(np.arange(41) % 2, 1, -1).astype(np.float32)
    weights = np.exp2(np.arange(41, dtype=np.float32))
    params = dict(grow_policy=policy, max_depth=0, max_leaves=41, max_bin=64,
                  base_score=0, reg_lambda=0, eta=1)
    expected, files = reference(X, y, params, rounds=1, weight=weights)
    data = xgb.DMatrix(X, label=y, weight=weights)
    model = xgb.train(params, data, 1)
    compare(model, data, expected, files)
    tree = forest(json.loads(model.save_raw(raw_format="json")))[0]
    assert depths(tree).max() > 30
    np.testing.assert_allclose(model.predict(data, pred_contribs=True), expected["contributions"], rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("policy", ["depthwise", "lossguide"])
@pytest.mark.parametrize("bins", [32, 513])
@pytest.mark.parametrize("cache", [1, 3, 16])
def test_eviction_subtraction_and_partition_paths_are_exact(policy, bins, cache):
    rng = np.random.default_rng(113)
    X = rng.normal(size=(1031, 5)).astype(np.float32)
    X[:, 0] = rng.integers(0, 53, len(X))
    y = (X[:, 1] * X[:, 2] + X[:, 3] ** 2 + (X[:, 0] % 5) * 0.3).astype(np.float32)
    X[rng.random(X.shape) < 0.11] = np.nan
    data = xgb.DMatrix(X, label=y, feature_types=["c", "q", "q", "q", "q"],
                       weight=rng.uniform(0.1, 2, len(X)).astype(np.float32))
    params = dict(grow_policy=policy, max_depth=0, max_leaves=19, max_bin=bins,
                  subsample=0.8, colsample_bynode=0.8, seed=193)
    ordinary = xgb.train({**params, "metal_histogram_subtraction": 0}, data, 3)
    for tuning in [
        {}, {"metal_fused_partition": 0},
        {"metal_partition_scan": 0, "metal_partition_aggregation": 0},
        {"metal_partition_scan": 0, "metal_partition_aggregation": 1, "metal_features_per_group": 4},
    ]:
        model = xgb.train({**params, **tuning, "max_cached_hist_node": cache}, data, 3)
        assert model.save_raw(raw_format="json") == ordinary.save_raw(raw_format="json")
        np.testing.assert_array_equal(model.predict(data), ordinary.predict(data))
        if cache == 1:
            assert profile(model)["tuning"]["histogram_rebuilds"] > 0


@pytest.mark.parametrize("constraints", [
    {"monotone_constraints": "(1,-1,0,0,0,0,0,0,0)"},
    {"interaction_constraints": "[[0,1],[2,3],[4,5,6,7,8]]"},
    {"monotone_constraints": "(1,-1,0,0,0,0,0,0,0)", "interaction_constraints": "[[0,1],[2,3,4,5,6,7,8]]"},
])
def test_lossguide_constraints_weights_and_regularization(reference, constraints):
    X, y = growth_data()
    weights = np.linspace(0.2, 1, len(y), dtype=np.float32)
    params = dict(grow_policy="lossguide", max_depth=0, max_leaves=29, max_bin=32,
                  base_score=0, reg_alpha=0.2, reg_lambda=2, gamma=0.15,
                  min_child_weight=3, max_delta_step=0.5, **constraints)
    expected, files = reference(X, y, params, rounds=4, weight=weights)
    data = xgb.DMatrix(X, label=y, weight=weights)
    model = xgb.train(params, data, 4)
    compare(model, data, expected, files)


@pytest.mark.parametrize("objective", ["multi:softprob", "reg:squarederror", "reg:quantileerror",
                                       "reg:expectileerror", "reg:normal", "rank:ndcg"])
def test_lossguide_objectives_and_dropout(reference, objective):
    X, raw_y = growth_data()
    y = raw_y
    params = dict(grow_policy="lossguide", max_depth=0, max_leaves=9, max_bin=32,
                  objective=objective, num_parallel_tree=2, rate_drop=0.3, one_drop=True, seed=73)
    meta = {}
    if objective == "multi:softprob":
        y = np.digitize(y, [-2, 2]).astype(np.float32)
        params["num_class"] = 3
    elif objective == "reg:squarederror":
        y = np.column_stack([raw_y, np.sin(raw_y)])
    elif objective == "reg:quantileerror":
        params["quantile_alpha"] = [0.2, 0.8]
    elif objective == "reg:expectileerror":
        params["expectile_alpha"] = [0.2, 0.8]
    elif objective == "rank:ndcg":
        y = np.digitize(y, [-2, 0, 2]).astype(np.float32)
        meta = {"qid": np.arange(len(y), dtype=np.uint32) // 16}
        params.update(lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=3)
    expected, files = reference(X, y, params, rounds=4, **meta)
    data = xgb.DMatrix(X, label=y, **meta)
    model = xgb.train(params, data, 4)
    compare(model, data, expected, files)
    np.testing.assert_allclose(model.predict(data, pred_contribs=True).sum(axis=-1),
                               model.predict(data, output_margin=True), rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("format", ["json", "ubj"])
@pytest.mark.parametrize("base_score", [None, 0.0])
def test_lossguide_model_exchange_config_and_continuation(reference, tmp_path, format, base_score):
    X, y = growth_data()
    params = dict(grow_policy="lossguide", max_depth=0, max_leaves=17, max_bin=32,
                  max_cached_hist_node=1, seed=37, colsample_bylevel=0.8, colsample_bynode=0.7)
    if base_score is not None:
        params["base_score"] = base_score
    expected, files = reference(X, y, params, rounds=3)
    data = xgb.DMatrix(X, label=y)
    imported = xgb.Booster(model_file=files / f"model.{format}")
    np.testing.assert_array_equal(imported.predict(data), expected["prediction"])
    imported_trees = forest(json.loads(imported.save_raw(raw_format="json")))
    reference_trees = forest(json.loads((files / "model.json").read_text()))
    for a, b in zip(imported_trees, reference_trees):
        np.testing.assert_array_equal(np.asarray(a["base_weights"], dtype=np.float32),
                                      np.asarray(b["base_weights"], dtype=np.float32))
    imported.load_config((files / "config.json").read_text())
    hist = json.loads(imported.save_config())["learner"]["gradient_booster"]["updater"][0]["hist_train_param"]
    assert hist["max_cached_hist_node"] == "1"
    model = xgb.train(params, data, 3)
    compare(model, data, expected, files)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.save_raw(raw_format="json") == model.save_raw(raw_format="json")
    assert json.loads(restored.save_config())["learner"]["gradient_booster"]["updater"][0]["hist_train_param"] == hist
    uninterrupted = xgb.train(params, data, 6)
    continued = xgb.train(params, data, 3, xgb_model=restored)
    assert continued.save_raw(raw_format="json") == uninterrupted.save_raw(raw_format="json")
    path = tmp_path / f"lossguide.{format}"
    model.save_model(path)
    assert xgb.Booster(model_file=path).save_raw(raw_format="json") == model.save_raw(raw_format="json")
    exported, _ = reference(X, y, params, model=path)
    np.testing.assert_array_equal(model.predict(data), exported["prediction"])
    external, external_files = reference(X, y, params, model=path, continue_rounds=2)
    resumed = xgb.train(params, data, 2, xgb_model=path)
    compare(resumed, data, external, external_files)
    np.testing.assert_array_equal(continued[:3].predict(data), model.predict(data))


@pytest.mark.parametrize("stop", [{"gamma": 1e9}, {"min_child_weight": 1e9}])
def test_lossguide_unsplit_root_statistics(reference, stop):
    X, y = growth_data()
    params = dict(grow_policy="lossguide", max_depth=0, max_leaves=31, max_bin=32,
                  base_score=0, eta=0.15, **stop)
    expected, files = reference(X, y, params, rounds=1)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 1)
    compare(model, data, expected, files)
    tree = forest(json.loads(model.save_raw(raw_format="json")))[0]
    assert len(tree["left_children"]) == 1
    assert tree["base_weights"][0] != 0
    assert tree["split_conditions"][0] == pytest.approx(0.15 * tree["base_weights"][0])


def test_single_tree_leaf_shapes_and_strict_ranges(reference):
    X, y = growth_data()
    params = dict(grow_policy="lossguide", max_depth=0, max_leaves=7, max_bin=32)
    requests = [dict(pred_leaf=True, strict_shape=True),
                dict(pred_leaf=True, iteration_range=[0, 1]),
                dict(pred_leaf=True, strict_shape=True, iteration_range=[0, 1])]
    expected, _ = reference(X, y, params, rounds=3, prediction_requests=requests)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 3)
    for shared in [0, 1]:
        model.set_param("metal_predict_shared", shared)
        for i, request in enumerate(requests):
            np.testing.assert_array_equal(model.predict(data, **request), expected[f"request_{i}"])


def test_lossguide_sklearn_early_stopping_and_cv():
    X, y = growth_data()
    params = dict(grow_policy="lossguide", max_depth=0, max_leaves=11, max_bin=32,
                  max_cached_hist_node=2)
    model = xgb.XGBRegressor(**params, n_estimators=8, learning_rate=0, early_stopping_rounds=2)
    model.fit(X, y, eval_set=[(X, y)], verbose=False)
    assert model.best_iteration == 0
    assert model.get_booster().num_boosted_rounds() == 3
    assert model.apply(X).shape == (len(X),)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(X), model.predict(X))
    result = xgb.cv(params, xgb.DMatrix(X, label=y), num_boost_round=4, nfold=3)
    assert result.shape == (4, 4)
    assert np.isfinite(result.to_numpy()).all()


@pytest.mark.parametrize("params", [
    {"grow_policy": "invalid"}, {"grow_policy": "lossguide", "max_depth": 0, "max_leaves": 0},
    {"max_depth": 31, "max_leaves": 0}, {"max_cached_hist_node": 0},
    {"max_cached_hist_node": -1}, {"max_cached_hist_node": 1.5},
])
def test_growth_parameter_validation(params):
    with pytest.raises(xgb.core.XGBoostError):
        xgb.train(params, xgb.DMatrix(np.arange(9, dtype=np.float32)[:, None], label=np.arange(9)), 1)
