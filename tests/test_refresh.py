import copy
import ctypes
import json

import numpy as np
import pytest
import xgboost as xgb


def dataset(rows=513, seed=772):
    rng = np.random.default_rng(seed)
    X = rng.integers(-12, 13, (rows, 5)).astype(np.float32) / 4
    y = (X[:, 0] * X[:, 1] + 0.7 * X[:, 2] - X[:, 3] ** 2
         + rng.normal(0, 0.1, rows)).astype(np.float32)
    X[::11, 1] = np.nan
    X[::17, 3] = np.nan
    return X, y


def document(model):
    return json.loads(model.save_raw(raw_format="json"))


def forest(model):
    return document(model)["learner"]["gradient_booster"]["model"]["trees"]


def compare(model, data, expected, files, *, interpretation=True, inherited_gain=False):
    source = json.loads((files / "model.json").read_text())["learner"]["gradient_booster"]["model"]
    own = document(model)["learner"]["gradient_booster"]["model"]
    assert own["iteration_indptr"] == source["iteration_indptr"]
    assert own["tree_info"] == source["tree_info"]
    np.testing.assert_array_equal(np.asarray(own.get("weight_drop", []), dtype=np.float32),
                                  np.asarray(source.get("weight_drop", []), dtype=np.float32))
    assert len(own["trees"]) == len(source["trees"])
    for a, b in zip(own["trees"], source["trees"]):
        assert a["tree_param"] == b["tree_param"]
        for field in ("left_children", "right_children", "split_indices", "parents", "default_left",
                      "split_type", "categories", "categories_nodes", "categories_segments", "categories_sizes"):
            np.testing.assert_array_equal(a[field], b[field], err_msg=field)
        for field in ("split_conditions", "base_weights", "sum_hessian", "loss_changes"):
            # Freshly grown hist trees retain their existing float32 split-gain
            # rounding; refresh recomputes these values with wider arithmetic.
            inherited = inherited_gain and field == "loss_changes"
            np.testing.assert_allclose(a[field], b[field], rtol=3e-5 if inherited else 8e-6,
                                       atol=1e-3 if inherited else 2e-5, err_msg=field)
    config = json.loads(model.save_config())["learner"]
    source_config = json.loads((files / "config.json").read_text())["learner"]
    assert config["generic_param"]["rng_state"] == source_config["generic_param"]["rng_state"]
    requests = {"prediction": {}, "margin": {"output_margin": True}, "leaf": {"pred_leaf": True}}
    if interpretation:
        requests.update(contributions={"pred_contribs": True},
                        approximate={"pred_contribs": True, "approx_contribs": True},
                        interactions={"pred_interactions": True},
                        approximate_interactions={"pred_interactions": True, "approx_contribs": True})
    for name, options in requests.items():
        if name == "leaf":
            np.testing.assert_array_equal(model.predict(data, **options), expected[name])
        else:
            np.testing.assert_allclose(model.predict(data, **options), expected[name], rtol=2e-5, atol=3e-5,
                                       err_msg=name)


@pytest.mark.parametrize("updater", ["refresh", "prune", "refresh,prune", "prune,refresh", "refresh,prune,refresh"])
@pytest.mark.parametrize("refresh_leaf", [False, True])
def test_refresh_and_prune_sequences(reference, updater, refresh_leaf):
    X, y = dataset()
    _, original = reference(X, y, dict(max_depth=4, base_score=0.25, eta=0.2), rounds=4)
    X, y = dataset(387, 891)
    weights = np.linspace(0, 2, len(X), dtype=np.float32)
    weights[::7] = 0
    params = dict(process_type="update", updater=updater, refresh_leaf=refresh_leaf,
                  eta=0.41, reg_alpha=0.13, reg_lambda=1.7, max_delta_step=0.8,
                  min_child_weight=1e6, max_depth=2, gamma=0.1)
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=3, weight=weights)
    data = xgb.DMatrix(X, label=y, weight=weights)
    model = xgb.train(params, data, 3, xgb_model=original / "model.json")
    compare(model, data, expected, files)
    assert model.num_boosted_rounds() == 3
    profile = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(profile)))
    phases = json.loads(profile.value)["gpu_seconds"]
    assert "quantiles" not in phases and "tree" not in phases
    for name in updater.split(","):
        assert phases[name] > 0
    # Deleted leaves remain serialized; their original IDs are not compacted.
    old_trees = json.loads((original / "model.json").read_text())["learner"]["gradient_booster"]["model"]["trees"]
    for old, new in zip(old_trees, forest(model)):
        assert old["tree_param"]["num_nodes"] == new["tree_param"]["num_nodes"]
        if updater == "refresh" and not refresh_leaf:
            np.testing.assert_array_equal(np.asarray(old["split_conditions"], dtype=np.float32),
                                          np.asarray(new["split_conditions"], dtype=np.float32))


@pytest.mark.parametrize("outputs,parallel", [(1, 1), (1, 3), (3, 1), (3, 2)])
@pytest.mark.parametrize("refresh_leaf", [False, True])
def test_refresh_groups_parallel_trees_and_base_margins(reference, outputs, parallel, refresh_leaf):
    X, target = dataset(281)
    y = target if outputs == 1 else np.column_stack([target, 0.3 * target + 1, -target])
    _, original = reference(X, y, dict(max_depth=3, base_score=0.25, num_parallel_tree=parallel), rounds=3)
    margins = np.linspace(-0.3, 0.6, len(X) * outputs, dtype=np.float32).reshape(len(X), outputs)
    data = xgb.DMatrix(X, label=y, base_margin=margins)
    params = dict(process_type="update", updater="refresh,prune", eta=0.7, gamma=0.05,
                  refresh_leaf=refresh_leaf, max_depth=2, num_parallel_tree=parallel)
    expected, files = reference(X, y, params, model=original / "model.ubj", continue_rounds=3,
                                base_margin=margins, test_margin=margins)
    model = xgb.train(params, data, 3, xgb_model=original / "model.ubj")
    compare(model, data, expected, files)
    assert model.predict(data, pred_leaf=True, strict_shape=True).shape == (len(X), 3, outputs, parallel)


@pytest.mark.parametrize("objective,extra", [
    ("binary:logistic", {}), ("multi:softprob", {"num_class": 3}),
    ("reg:absoluteerror", {}), ("reg:quantileerror", {"quantile_alpha": [0.2, 0.7]}),
    ("reg:expectileerror", {"expectile_alpha": [0.2, 0.7]}), ("reg:normal", {}),
    ("survival:aft", {"aft_loss_distribution": "logistic"}), ("survival:cox", {}),
    ("rank:pairwise", {}),
])
def test_refresh_objective_gradients(reference, objective, extra):
    X, y = dataset(259)
    meta = {}
    if objective == "binary:logistic":
        y = (y > np.median(y)).astype(np.float32)
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
        meta = dict(group=[37] * 7)
    seed = dict(objective=objective, base_score=0.4, max_depth=3, eta=0.1, **extra)
    _, original = reference(X, y, seed, rounds=3, **meta)
    params = dict(process_type="update", updater="refresh", eta=0.17, subsample=0,
                  colsample_bytree=0, colsample_bylevel=0, colsample_bynode=0,
                  sampling_method="gradient_based", max_depth=0, max_leaves=0, **extra)
    data = xgb.DMatrix(X, label=y, **meta)
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=3, **meta)
    model = xgb.train(params, data, 3, xgb_model=original / "model.json")
    compare(model, data, expected, files)


@pytest.mark.parametrize("kind", ["negative_hessian", "zero_hessian", "wide_gradient", "regularized"])
def test_refresh_custom_gradients(reference, kind):
    rows = 8193
    X = (np.arange(rows, dtype=np.float32) % 17)[:, None]
    y = np.zeros(rows, dtype=np.float32)
    _, original = reference(X, y, dict(max_depth=0, max_leaves=1, base_score=0), rounds=1)
    grad = np.sin(np.arange(rows, dtype=np.float32))
    hess = np.ones(rows, dtype=np.float32)
    params = dict(process_type="update", updater="refresh", eta=0.73, min_child_weight=1e9)
    if kind == "negative_hessian":
        hess[::3] = -0.75
        grad[::3] += 100
    elif kind == "zero_hessian":
        hess[:] = 0
    elif kind == "wide_gradient":
        grad[:] = np.float32(1e38)
        grad[::3] *= -1
        hess[:] = np.float32(1e30)
    else:
        params.update(reg_alpha=0.17, reg_lambda=0.83, max_delta_step=0.01)
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=1,
                                grad=grad, hess=hess)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 1, xgb_model=original / "model.json", obj=lambda p, d: (grad, hess))
    compare(model, data, expected, files)


@pytest.mark.parametrize("updater", ["prune", "refresh,prune", "prune,refresh"])
def test_pruned_categorical_model_exchange(reference, tmp_path, updater):
    X = np.column_stack([np.arange(257) % 5, np.zeros(257)]).astype(np.float32)
    y = (X[:, 0] ** 2).copy()
    X[::19, 0] = np.nan
    types = ["c", "q"]
    _, original = reference(X, y, dict(max_depth=3, max_cat_to_onehot=2, base_score=0), rounds=3,
                            feature_types=types)
    params = dict(process_type="update", updater=updater, gamma=1e6, max_depth=0)
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=3,
                                feature_types=types)
    data = xgb.DMatrix(X, label=y, feature_types=types, enable_categorical=True)
    model = xgb.train(params, data, 3, xgb_model=original / "model.json")
    compare(model, data, expected, files)
    assert model.get_score() == {}
    for tree in forest(model):
        assert int(tree["tree_param"]["num_deleted"]) > 0
        assert tree["split_type"][0] == 1 and tree["left_children"][0] == -1
        assert tree["categories_nodes"]
    for extension in ["json", "ubj"]:
        path = tmp_path / f"pruned.{extension}"
        model.save_model(path)
        loaded = xgb.Booster(model_file=path)
        np.testing.assert_array_equal(loaded.predict(data), model.predict(data))
        imported, _ = reference(X, y, {}, model=path, feature_types=types)
        np.testing.assert_array_equal(imported["leaf"], model.predict(data, pred_leaf=True))
        np.testing.assert_allclose(imported["prediction"], model.predict(data), atol=1e-6)


@pytest.mark.parametrize("depth,gamma", [(0, 0), (0, 1e6), (1, 0), (2, 1.5), (100, 0)])
def test_prune_limits_and_feature_scores(reference, depth, gamma):
    X, y = dataset()
    _, original = reference(X, y, dict(max_depth=5, base_score=0), rounds=2)
    params = dict(process_type="update", updater="prune", max_depth=depth, max_leaves=0, gamma=gamma)
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=2)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 2, xgb_model=original / "model.json")
    compare(model, data, expected, files)
    loaded = xgb.Booster(model_file=files / "model.json")
    for kind in ["weight", "gain", "cover", "total_gain", "total_cover"]:
        assert model.get_score(importance_type=kind) == loaded.get_score(importance_type=kind)


def test_update_queue_reset_copy_and_overrun(reference):
    X, y = dataset(129)
    _, original = reference(X, y, dict(max_depth=3, base_score=0.25), rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.Booster(model_file=original / "model.json")
    model.set_param({"process_type": "update", "updater": "refresh"})
    assert model.num_boosted_rounds() == 0
    np.testing.assert_array_equal(model.predict(data), np.full(len(X), 0.25, dtype=np.float32))
    model.update(data, 0)
    assert model.num_boosted_rounds() == 1
    config = json.loads(model.save_config())["learner"]["gradient_booster"]
    assert config["gbtree_train_param"]["process_type"] == "default"
    assert config["gbtree_train_param"]["updater"] == "refresh"
    assert config["specified_updater"] is True
    with pytest.raises(xgb.core.XGBoostError, match="Not all trees are updated"):
        model[:]
    model.set_param({})
    model.update(data, 1)
    model.update(data, 2)
    with pytest.raises(xgb.core.XGBoostError, match="No more tree"):
        model.update(data, 3)
    assert model[:].num_boosted_rounds() == 3
    model.set_param({"process_type": "update", "updater": "refresh"})
    with pytest.raises(xgb.core.XGBoostError, match="No more tree"):
        model.update(data, 0)
    for restored in [copy.deepcopy(model), model.reset()]:
        assert restored.num_boosted_rounds() == 3
        restored.update(data, 3)
        assert restored.num_boosted_rounds() == 4
    # Reset/save/load drops pending trees and retains only completed rounds.
    partial = xgb.Booster(model_file=original / "model.json")
    partial.set_param({"process_type": "update", "updater": "prune"})
    partial.update(data, 0)
    partial.reset()
    assert partial.num_boosted_rounds() == 1
    partial.update(data, 1)
    assert partial.num_boosted_rounds() == 2


def test_update_parameter_batches_and_eager_transition(reference):
    X, y = dataset(129)
    _, original = reference(X, y, dict(max_depth=3), rounds=2)
    data = xgb.DMatrix(X, label=y)
    model = xgb.Booster(model_file=original / "model.json")
    model.set_param({"process_type": "update", "updater": "refresh"})
    model.set_param({"gamma": 1})
    with pytest.raises(xgb.core.XGBoostError, match="cannot modify existing trees"):
        model.update(data, 0)
    model.set_param({"updater": "refresh"})
    model.update(data, 0)
    assert model.num_boosted_rounds() == 1
    fresh = xgb.Booster(model_file=original / "model.json")
    fresh.set_param({"process_type": "update", "updater": "refresh"})
    fresh.set_param({"process_type": "default"})
    assert fresh.num_boosted_rounds() == 0
    fresh.update(data, 0)
    assert fresh.num_boosted_rounds() == 1


def test_live_update_then_resume_growth(reference):
    X, y = dataset(257)
    params = dict(max_depth=3, base_score=0.2, eta=0.31)
    updates = {"2": {"process_type": "update", "updater": "refresh"},
               "3": {"updater": "refresh,prune", "max_depth": 2},
               "4": {"process_type": "default"}}
    expected, files = reference(X, y, params, rounds=5, param_updates=updates)
    class Updates(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            if str(epoch) in updates:
                model.set_param(updates[str(epoch)])
            return False
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 5, callbacks=[Updates()])
    compare(model, data, expected, files)
    assert model.num_boosted_rounds() == 3


@pytest.mark.parametrize("updater", ["grow_quantile_histmaker,refresh", "grow_quantile_histmaker,prune"])
def test_grow_then_modify_sequence(reference, updater):
    X, y = dataset(257)
    params = dict(updater=updater, max_depth=3, eta=0.2, reg_alpha=0.1, base_score=0)
    expected, files = reference(X, y, params, rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 3)
    compare(model, data, expected, files, inherited_gain=updater.endswith("prune"))
    if updater.endswith("prune"):
        # The grower already applied these depth/gain limits. Adding prune must
        # preserve its complete model, including the inherited gain values.
        baseline = xgb.train({key: value for key, value in params.items() if key != "updater"}, data, 3)
        assert model.save_raw() == baseline.save_raw()


@pytest.mark.parametrize("params,message", [
    ({"process_type": "update", "updater": "refresh"}, "No more tree"),
    ({"updater": "refresh"}, "cannot create new trees"),
    ({"updater": "prune"}, "cannot create new trees"),
    ({"process_type": "update"}, "cannot modify existing trees"),
    ({"updater": "nonsense"}, "updater"),
    ({"process_type": "unknown"}, "process_type"),
])
def test_update_validation(params, message):
    X, y = dataset(17)
    with pytest.raises(xgb.core.XGBoostError, match=message):
        xgb.train(params, xgb.DMatrix(X, label=y), 1)


def test_refresh_monotone_rejection_and_zero_constraints(reference):
    X, y = dataset(129)
    _, original = reference(X, y, dict(max_depth=3), rounds=2)
    data = xgb.DMatrix(X, label=y)
    params = dict(process_type="update", updater="refresh", monotone_constraints="(1,0,0,0,0)")
    with pytest.raises(xgb.core.XGBoostError, match="Monotonic constraint"):
        xgb.train(params, data, 1, xgb_model=original / "model.json")
    params["monotone_constraints"] = "(0,0,0,0,0)"
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=2)
    model = xgb.train(params, data, 2, xgb_model=original / "model.json")
    compare(model, data, expected, files)


@pytest.mark.parametrize("coalesce,fill", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("columns", [False, True])
def test_refresh_encoder_modes(reference, coalesce, fill, columns):
    X, y = dataset(1025)
    _, original = reference(X, y, dict(grow_policy="lossguide", max_leaves=17, max_depth=0, base_score=0), rounds=2)
    params = dict(process_type="update", updater="refresh,prune", max_depth=3, gamma=0.1)
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=2)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train({**params, "metal_coalesce_encoders": coalesce, "metal_compute_fill": fill,
                      "metal_refresh_columns": columns},
                      data, 2, xgb_model=original / "model.json")
    compare(model, data, expected, files)
    other = xgb.train({**params, "metal_refresh_columns": not columns}, data, 2, xgb_model=original / "model.json")
    assert other.save_raw() == model.save_raw()


@pytest.mark.parametrize("updater", ["refresh", "prune"])
def test_update_quantile_matrix_rejection(reference, updater):
    X, y = dataset(65)
    _, original = reference(X, y, dict(max_depth=3), rounds=1)
    data = xgb.QuantileDMatrix(X, label=y)
    with pytest.raises(xgb.core.XGBoostError, match="QuantileDMatrix"):
        xgb.train({"process_type": "update", "updater": updater}, data, 1, xgb_model=original / "model.json")


def test_update_rejects_reference_dropout_weight_failure():
    X, y = dataset(65)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train({"max_depth": 2, "rate_drop": 0.2}, data, 3)
    with pytest.raises(xgb.core.XGBoostError, match="stored dropout weights"):
        xgb.train({"process_type": "update", "updater": "refresh"}, data, 3, xgb_model=model)


@pytest.mark.parametrize("normalization", ["tree", "forest"])
def test_refresh_enables_dropout_on_unweighted_model(reference, normalization):
    X, y = dataset(257)
    _, original = reference(X, y, dict(max_depth=3, base_score=0), rounds=4)
    params = dict(process_type="update", updater="refresh", rate_drop=0.3, one_drop=True,
                  sample_type="weighted", normalize_type=normalization, seed=13)
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=4)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 4, xgb_model=original / "model.json")
    compare(model, data, expected, files)


@pytest.mark.parametrize("storage", ["sparse", "pandas"])
@pytest.mark.parametrize("updater", ["refresh", "prune", "refresh,prune"])
def test_refresh_missing_and_recoded_input(reference, storage, updater):
    X, y = dataset(193)
    types = None
    if storage == "pandas":
        import pandas as pd
        values = (np.arange(len(X)) % 4).astype(np.int32)
        X = pd.DataFrame({"category": pd.Categorical.from_codes(values, ["a", "b", "c", "d"]),
                          "value": X[:, 1], "other": X[:, 2]})
        y = (values ** 2 + np.nan_to_num(X["value"].to_numpy())).astype(np.float32)
    else:
        X[::3, 0] = np.nan
        X[X == 0] = np.nan
    _, original = reference(X, y, dict(max_depth=4, base_score=0), rounds=3, feature_types=types)
    if storage == "pandas":
        X["category"] = X["category"].cat.reorder_categories(["d", "b", "a", "c"])
        converted = X
    else:
        from scipy.sparse import csr_matrix
        converted = csr_matrix(np.nan_to_num(X, nan=0))
    data = xgb.DMatrix(converted, label=y, enable_categorical=True)
    params = dict(process_type="update", updater=updater, max_depth=2, eta=0.4,
                  seed=87, seed_per_iteration=True)
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=3,
                                feature_types=types)
    model = xgb.train(params, data, 3, xgb_model=original / "model.json")
    compare(model, data, expected, files)


@pytest.mark.parametrize("previous_recoding", [False, True])
def test_refresh_named_categories_custom_objective_cache(reference, previous_recoding):
    import pandas as pd
    values = np.arange(257) % 4
    frame = pd.DataFrame({"category": pd.Categorical.from_codes(values, ["a", "b", "c", "d"]),
                          "value": np.sin(np.arange(257)).astype(np.float32)})
    y = (values ** 2 + frame["value"].to_numpy()).astype(np.float32)
    seed = dict(max_depth=3, base_score=0)
    _, original = reference(frame, y, seed, rounds=3)
    frame["category"] = frame["category"].cat.reorder_categories(["c", "b", "d", "a"])
    data = xgb.DMatrix(frame, label=y, enable_categorical=True)
    if previous_recoding:
        # A prior hist continuation can cache a recoded prediction view.
        # Refresh must still see this DMatrix's stored SparsePage-equivalent codes.
        xgb.train(seed, data, 1, xgb_model=original / "model.json")
    weights = np.linspace(0.2, 1.4, len(y), dtype=np.float32)
    params = dict(process_type="update", updater="refresh", eta=0.3)
    updates = {"1": {"label": (0.4 * y).tolist(), "weight": weights.tolist()}}
    expected, files = reference(frame, y, params, model=original / "model.json", continue_rounds=3,
                                custom_square=True, matrix_updates=updates)
    class Metadata(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            if epoch == 1:
                data.set_label(0.4 * y)
                data.set_weight(weights)
            return False
    model = xgb.train(params, data, 3, xgb_model=original / "model.json", callbacks=[Metadata()],
                      obj=lambda prediction, d: (prediction - d.get_label(), np.ones_like(prediction)))
    compare(model, data, expected, files)


def test_refresh_empty_leaf_and_all_zero_weights(reference):
    X, y = dataset(193)
    _, original = reference(X, y, dict(max_depth=4, base_score=0), rounds=2)
    X[:] = np.nan
    weights = np.zeros(len(X), dtype=np.float32)
    params = dict(process_type="update", updater="refresh", eta=0.4)
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=2, weight=weights)
    data = xgb.DMatrix(X, label=y, weight=weights)
    model = xgb.train(params, data, 2, xgb_model=original / "model.json")
    compare(model, data, expected, files, interpretation=False)
    for tree in forest(model):
        assert not np.any(tree["base_weights"])
        assert not np.any(tree["sum_hessian"])


def test_refresh_reuses_workspace_after_matrix_metadata_change(reference):
    X, y = dataset(513)
    _, original = reference(X, y, dict(max_depth=4, base_score=0), rounds=3)
    weights = np.linspace(0.1, 1, len(X), dtype=np.float32)
    updates = {"1": {"weight": weights.tolist(), "label": (0.3 * y).tolist()}}
    params = dict(process_type="update", updater="refresh", eta=0.2)
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=3,
                                matrix_updates=updates)
    data = xgb.DMatrix(X, label=y)
    class Metadata(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            if epoch == 1:
                data.set_weight(weights)
                data.set_label(0.3 * y)
            return False
    model = xgb.train(params, data, 3, xgb_model=original / "model.json", callbacks=[Metadata()])
    compare(model, data, expected, files)


def test_refresh_configuration_load_resumes_growing(reference):
    X, y = dataset(129)
    _, original = reference(X, y, dict(max_depth=3), rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.Booster(model_file=original / "model.json")
    model.set_param({"process_type": "update", "updater": "refresh"})
    model.update(data, 0)
    model.load_config(model.save_config())
    assert not json.loads(model.save_config())["learner"]["gradient_booster"]["specified_updater"]
    model.update(data, 1)
    assert model.num_boosted_rounds() == 2


@pytest.mark.parametrize("updater", ["refresh", "prune"])
def test_update_empty_data(reference, updater):
    X, y = dataset(129)
    _, original = reference(X, y, dict(max_depth=3, base_score=0.25), rounds=2)
    X, y = X[:0], y[:0]
    params = dict(process_type="update", updater=updater, gamma=1)
    expected, files = reference(X, y, params, model=original / "model.json", continue_rounds=2)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 2, xgb_model=original / "model.json")
    compare(model, data, expected, files, interpretation=False)


@pytest.mark.parametrize("booster", ["gbtree", "gblinear"])
@pytest.mark.parametrize("objective", ["reg:squarederror", "multi:softprob", "multi:softmax"])
def test_empty_prediction_shapes_against_reference(reference, booster, objective):
    X, y = dataset(33)
    params = dict(booster=booster, objective=objective, max_depth=2)
    if objective.startswith("multi:"):
        params["num_class"] = 3
        y = np.arange(len(y), dtype=np.float32) % 3
    if booster == "gbtree":
        params["num_parallel_tree"] = 2
    requests = [{"strict_shape": True}, {"output_margin": True, "strict_shape": True},
                {"pred_contribs": True, "strict_shape": True},
                {"pred_interactions": True, "strict_shape": True}]
    if booster == "gbtree":
        requests += [{"pred_leaf": True, "strict_shape": True}]
    expected, _ = reference(X, y, params, rounds=2, test=X[:0], prediction_requests=requests)
    model = xgb.train(params, xgb.DMatrix(X, label=y), 2)
    empty = xgb.DMatrix(X[:0])
    for name, options in [("prediction", {}), ("margin", {"output_margin": True}),
                          ("contributions", {"pred_contribs": True}), ("interactions", {"pred_interactions": True})]:
        np.testing.assert_array_equal(model.predict(empty, **options), expected[name])
    for i, options in enumerate(requests):
        np.testing.assert_array_equal(model.predict(empty, **options), expected[f"request_{i}"])
