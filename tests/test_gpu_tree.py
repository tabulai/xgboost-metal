"""GPU depthwise control preserves the existing CPU driver's exact tree state."""
import ctypes
import itertools
import json

import numpy as np
import pytest
import xgboost as xgb


def profile(model):
    output = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(output)))
    return json.loads(output.value)


def snapshot(model, matrices):
    return {
        "model": bytes(model.save_raw(raw_format="ubj")),
        "rng": json.loads(model.save_config())["learner"]["generic_param"]["rng_state"],
        "prediction": [model.predict(data, **options) for data in matrices
                       for options in ({}, {"output_margin": True}, {"pred_leaf": True})],
    }


def assert_identical(expected, actual):
    assert actual["model"] == expected["model"]
    assert actual["rng"] == expected["rng"]
    assert len(actual["prediction"]) == len(expected["prediction"])
    for own, wanted in zip(actual["prediction"], expected["prediction"]):
        assert own.shape == wanted.shape and own.dtype == wanted.dtype
        assert own.tobytes() == wanted.tobytes()


def inputs(case="regression", rows=513):
    rng = np.random.default_rng(82171)
    X = rng.normal(size=(rows, 7)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] * X[:, 2] + X[:, 3] ** 2).astype(np.float32)
    metadata, params = {}, {}
    if case == "missing":
        X[::7, 0] = np.nan
        X[::11, 3] = np.nan
        X[:, 6] = np.nan
    elif case == "ties":
        X[:, 0] = np.arange(rows) % 4
        X[:, 1:] = X[:, :1]
        y = (X[:, 0] >= 2).astype(np.float32)
    elif case == "constant":
        y.fill(0)
    elif case == "empty":
        X, y = X[:0], y[:0]
    elif case == "weighted":
        metadata["weight"] = rng.uniform(0.01, 3, rows).astype(np.float32)
        metadata["weight"][::13] = 0
    elif case == "multiclass":
        y = np.argmax(X[:, :3], axis=1).astype(np.float32)
        params.update(objective="multi:softprob", num_class=3)
    elif case == "parallel":
        params["num_parallel_tree"] = 3
    elif case == "bytree":
        params.update(colsample_bytree=0.6, seed_per_iteration=True)
        metadata["feature_weights"] = np.asarray([0.1, 0.2, 1, 2, 4, 8, 16], np.float32)
    elif case == "dropout":
        params.update(rate_drop=0.3, one_drop=True, skip_drop=0.1)
    elif case == "base_margin":
        metadata["base_margin"] = np.linspace(-0.5, 0.4, rows, dtype=np.float32)
    elif case == "subnormal_features":
        tiny = np.nextafter(np.float32(0), np.float32(1))
        X[:, 0] = np.resize(np.asarray([-4 * tiny, -tiny, -0.0, 0.0, tiny, 4 * tiny], np.float32), rows)
        y = (np.arange(rows) % 6 >= 3).astype(np.float32)
    elif case in ("positive_infinity_missing", "negative_infinity_missing"):
        infinity = np.inf if case == "positive_infinity_missing" else -np.inf
        X[::9, 2] = infinity
        metadata["missing"] = infinity
    elif case == "categorical":
        X[:, 0] = np.arange(rows) % 5
        metadata["feature_types"] = ["c"] + ["q"] * 6
    elif case == "vector":
        y = np.column_stack([y, y + X[:, 4]]).astype(np.float32)
        params["multi_strategy"] = "multi_output_tree"
    return X, y, metadata, params


def train(enabled, case="regression", params=None, rounds=3, quantile=False, fusion=1):
    X, y, metadata, case_params = inputs(case)
    options = dict(max_depth=4, max_bin=32, base_score=0, seed=41,
                   metal_gpu_tree=int(enabled), metal_gradient_root_fusion=fusion)
    options.update(case_params)
    options.update(params or {})
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    extra = {"max_bin": options["max_bin"]} if quantile else {}
    data = constructor(X, label=y, **metadata, **extra)
    model = xgb.Booster(options, [data])

    def custom(prediction, matrix):
        return (prediction - matrix.get_label()).astype(np.float32), np.ones_like(prediction, dtype=np.float32)

    stats = []
    for iteration in range(rounds):
        model.update(data, iteration, fobj=custom if case == "custom" else None)
        stats.append(profile(model))
    query_meta = {k: v for k, v in metadata.items() if k in ("missing", "feature_types", "base_margin")}
    fresh = xgb.DMatrix(X, **query_meta)
    return snapshot(model, [data, fresh]), stats


def assert_path(stats, enabled):
    assert all(item["tuning"]["gpu_tree"] == enabled for item in stats)
    timing = stats[-1]["gpu_seconds"]
    assert (timing.get("gpu_tree", 0) + timing.get("gradient_tree", 0) > 0) == bool(enabled)


@pytest.mark.parametrize("case", ["regression", "missing", "ties", "constant", "empty", "weighted",
                                  "custom", "multiclass", "parallel", "bytree", "dropout", "base_margin",
                                  "subnormal_features", "positive_infinity_missing", "negative_infinity_missing"])
@pytest.mark.parametrize("fusion", [0, 1])
def test_gpu_driver_preserves_models_predictions_leaves_and_rng(case, fusion):
    expected, control_stats = train(0, case, fusion=fusion)
    actual, stats = train(1, case, fusion=fusion)
    assert_identical(expected, actual)
    assert_path(control_stats, 0)
    assert_path(stats, 1)
    assert stats[-1]["command_buffers"] <= control_stats[-1]["command_buffers"]
    if case in ("regression", "missing", "weighted", "custom", "multiclass", "parallel", "bytree"):
        assert stats[-1]["command_buffers"] < control_stats[-1]["command_buffers"]


@pytest.mark.parametrize("depth", [2, 3, 6, 8])
@pytest.mark.parametrize("quantile", [False, True])
def test_depth_limits_and_dense_quantile_roots(depth, quantile):
    expected, control_stats = train(0, params={"max_depth": depth}, quantile=quantile, rounds=2)
    actual, stats = train(1, params={"max_depth": depth}, quantile=quantile, rounds=2)
    assert_identical(expected, actual)
    assert_path(stats, 1)
    assert stats[-1]["command_buffers"] < control_stats[-1]["command_buffers"]


@pytest.mark.parametrize("coalesce,fill,parallel_splits", list(itertools.product([0, 1], repeat=3)))
def test_encoder_fill_and_split_modes(coalesce, fill, parallel_splits):
    tuning = dict(metal_coalesce_encoders=coalesce, metal_compute_fill=fill,
                  metal_parallel_splits=parallel_splits)
    expected, _ = train(0, "missing", tuning, rounds=2)
    actual, stats = train(1, "missing", tuning, rounds=2)
    assert_identical(expected, actual)
    assert_path(stats, 1)


FALLBACKS = [
    ("shallow", "regression", {"max_depth": 1}),
    ("deep", "regression", {"max_depth": 11}),
    ("unlimited", "regression", {"max_depth": 0, "max_leaves": 9}),
    ("lossguide", "regression", {"grow_policy": "lossguide"}),
    ("max_leaves", "regression", {"max_leaves": 7}),
    ("monotone", "regression", {"monotone_constraints": "(1,0,-1,0,0,0,0)"}),
    ("interaction", "regression", {"interaction_constraints": "[[0,1,2],[3,4,5,6]]"}),
    ("level_sampling", "regression", {"colsample_bylevel": 0.8}),
    ("node_sampling", "regression", {"colsample_bynode": 0.8}),
    ("uniform_sampling", "regression", {"subsample": 0.7}),
    ("gradient_sampling", "regression", {"subsample": 0.7, "sampling_method": "gradient_based"}),
    ("cache_eviction", "regression", {"max_cached_hist_node": 1}),
    ("atomic_partition", "regression", {"metal_partition_scan": 0}),
    ("separate_partition", "regression", {"metal_fused_partition": 0}),
    ("no_subtraction", "regression", {"metal_histogram_subtraction": 0}),
    ("large_bins", "regression", {"max_bin": 512}),
    ("categorical", "categorical", {}),
    ("vector", "vector", {}),
]


@pytest.mark.parametrize("_name,case,params", FALLBACKS, ids=[row[0] for row in FALLBACKS])
def test_unsupported_configurations_use_the_original_driver(_name, case, params):
    expected, control_stats = train(0, case, params, rounds=2)
    actual, stats = train(1, case, params, rounds=2)
    assert_identical(expected, actual)
    assert_path(stats, 0)
    assert stats[-1]["command_buffers"] == control_stats[-1]["command_buffers"]


@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_extreme_custom_gradients_preserve_dense_root_and_parent_statistics(quantile, missing):
    X = np.asarray([[0, 0], [1, 0], [0, 0]], np.float32)
    if missing:
        X[0, 0] = np.nan
    gradient = np.asarray([2**40, -2**40, 3 * 2**-14], np.float32)
    hessian = np.asarray([1, 2, 1], np.float32)
    results = []
    for enabled in [0, 1]:
        cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
        data = cls(X, label=np.zeros(3, np.float32), **({"max_bin": 4} if quantile else {}))
        model = xgb.Booster(dict(max_depth=3, max_bin=4, base_score=0, eta=1,
                                 reg_lambda=1, min_child_weight=1, metal_gpu_tree=enabled), [data])
        model.update(data, 0, fobj=lambda prediction, matrix: (gradient, hessian))
        assert_path([profile(model)], enabled)
        results.append(snapshot(model, [data, xgb.DMatrix(X)]))
    assert_identical(*results)


def test_leaf_scaling_preserves_subnormal_results():
    X = np.arange(64, dtype=np.float32)[:, None]
    y = np.where(X[:, 0] < 32, -0.25, 0.25).astype(np.float32)
    results = []
    for enabled in [0, 1]:
        data = xgb.DMatrix(X, label=y)
        # A normal learning rate produces subnormal leaves when multiplied by
        # these ordinary weights. Materialization must retain CPU arithmetic.
        model = xgb.Booster(dict(max_depth=3, max_bin=16, base_score=0, reg_lambda=0,
                                 learning_rate=float(np.finfo(np.float32).tiny), metal_gpu_tree=enabled), [data])
        model.update(data, 0)
        assert_path([profile(model)], enabled)
        document = json.loads(model.save_raw(raw_format="json"))
        tree = document["learner"]["gradient_booster"]["model"]["trees"][0]
        leaves = np.asarray(tree["split_conditions"], np.float32)[np.asarray(tree["left_children"]) < 0]
        assert np.any((np.abs(leaves) > 0) & (np.abs(leaves) < np.finfo(np.float32).tiny))
        results.append(snapshot(model, [data]))
    assert_identical(*results)


@pytest.mark.parametrize("workspace,persistent", list(itertools.product([0, 1], repeat=2)))
def test_workspace_reuse_reset_shape_changes_and_toggle(workspace, persistent):
    traces = []
    for use_gpu in [False, True]:
        model = xgb.Booster(dict(max_depth=4, max_bin=32, base_score=0, seed=19,
                                 colsample_bytree=0.7, metal_tree_workspace=workspace,
                                 metal_tree_persistent_workspace=persistent))
        trace = []
        for iteration, rows in enumerate([513, 65, 1027, 0, 513, 129]):
            X, y, _, _ = inputs(rows=rows)
            data = xgb.DMatrix(X, label=y)
            enabled = int(use_gpu and iteration % 3 != 1)
            model.set_param("metal_gpu_tree", enabled)
            if iteration == 3:
                model.reset()
            if iteration == 4:
                model.load_config(model.save_config())
            model.update(data, iteration)
            assert profile(model)["tuning"]["gpu_tree"] == enabled
            trace.append(snapshot(model, [data]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


@pytest.mark.parametrize("fusion", [0, 1])
def test_constructor_failure_retry_preserves_column_sampling_rng(fusion):
    results = []
    for enabled in [0, 1]:
        X, y, _, _ = inputs()
        data = xgb.DMatrix(X, label=y)
        model = xgb.Booster(dict(max_depth=4, max_bin=32, base_score=0, seed=89,
                                 colsample_bytree=0.6, metal_gpu_tree=enabled,
                                 metal_gradient_root_fusion=fusion), [data])
        for iteration in range(3):
            if iteration == 1:
                model.set_param("metal_histogram_rows", 1)
                with pytest.raises(xgb.core.XGBoostError, match="metal_histogram_rows"):
                    model.update(data, iteration)
                model.set_param("metal_histogram_rows", 2048)
            model.update(data, iteration)
            assert profile(model)["tuning"]["gpu_tree"] == enabled
        results.append(snapshot(model, [data]))
    assert_identical(*results)


def test_speculative_scratch_budget_falls_back_for_a_small_actual_tree():
    # At depth eight, two 128-node histogram banks for 4096 features and
    # sixteen bins alone exceed 256 MiB. The actual tree has just its root.
    X = np.broadcast_to(np.arange(16, dtype=np.float32)[:, None], (16, 4096)).copy()
    results = []
    commands = []
    for enabled in [0, 1]:
        data = xgb.DMatrix(X, label=np.zeros(len(X), np.float32))
        model = xgb.Booster(dict(max_depth=8, max_bin=16, base_score=0,
                                 metal_gpu_tree=enabled), [data])
        model.update(data, 0)
        stats = profile(model)
        assert_path([stats], 0)
        commands.append(stats["command_buffers"])
        document = json.loads(model.save_raw(raw_format="json"))
        tree = document["learner"]["gradient_booster"]["model"]["trees"][0]
        assert len(tree["left_children"]) == 1
        results.append(snapshot(model, [data]))
    assert commands[0] == commands[1]
    assert_identical(*results)


def test_minimum_gain_accepts_equality_and_rejects_the_next_float():
    X = np.asarray([0, 0, 1, 1], np.float32)[:, None]
    gradient = np.asarray([-1, -1, 1, 1], np.float32)
    hessian = np.ones(4, np.float32)
    # With lambda zero the child gains are exactly two each, while the root
    # gain is zero. Driver::Pop accepts gain == min_split_loss.
    for minimum, expected_nodes in [(np.float32(4), 3),
                                     (np.nextafter(np.float32(4), np.float32(np.inf)), 1)]:
        results = []
        for enabled in [0, 1]:
            data = xgb.DMatrix(X, label=np.zeros(4, np.float32))
            model = xgb.Booster(dict(max_depth=3, max_bin=4, base_score=0, eta=1,
                                     reg_lambda=0, min_child_weight=1,
                                     min_split_loss=float(minimum), metal_gpu_tree=enabled), [data])
            model.update(data, 0, fobj=lambda prediction, matrix: (gradient, hessian))
            assert_path([profile(model)], enabled)
            document = json.loads(model.save_raw(raw_format="json"))
            tree = document["learner"]["gradient_booster"]["model"]["trees"][0]
            assert len(tree["left_children"]) == expected_nodes
            if expected_nodes == 3:
                assert tree["loss_changes"][0] == 4
            results.append(snapshot(model, [data]))
        assert_identical(*results)
