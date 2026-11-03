"""Host-side round streamlining keeps models, predictions and RNG identical.

Each optimization has its own switch: persistent round buffers, GPU row-ID
initialization, write-once constant tree inputs, GPU-tree replay without
interaction paths, early prologue commits and the parameter snapshot.
"""
import ctypes
import itertools
import json

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, inputs, profile, snapshot

SWITCHES = ["metal_round_buffers", "metal_gpu_row_init", "metal_tree_constant_inputs",
            "metal_replay_path_elision", "metal_commit_early", "metal_parameter_snapshot"]
OFF = {name: 0 for name in SWITCHES}


def train(case="regression", params=None, *, rounds=4, quantile=False, action=None):
    X, y, metadata, options = inputs(case)
    options = dict(max_depth=4, max_bin=32, base_score=0, seed=41, **options)
    options.update(params or {})
    if options.get("objective") in ("binary:logistic", "binary:logitraw", "binary:hinge"):
        y = (y > 0).astype(np.float32)
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = constructor(X, label=y, **metadata, **({"max_bin": options["max_bin"]} if quantile else {}))
    model = xgb.Booster(options, [data])

    def custom(prediction, matrix):
        return (prediction - matrix.get_label()).astype(np.float32), np.ones_like(prediction, dtype=np.float32)

    stats = []
    for iteration in range(rounds):
        model.update(data, iteration, fobj=custom if case == "custom" else None)
        stats.append(profile(model))
        if action:
            action(model, data, iteration)
    fresh = xgb.DMatrix(X, **{k: v for k, v in metadata.items() if k in ("missing", "feature_types", "base_margin")})
    return snapshot(model, [data, fresh]), stats


CASES = ["regression", "missing", "ties", "constant", "empty", "weighted", "custom", "multiclass",
         "parallel", "bytree", "dropout", "base_margin", "subnormal_features", "categorical", "vector"]


@pytest.mark.parametrize("case", CASES)
def test_all_switches_preserve_models_predictions_and_rng(case):
    expected, control = train(case, OFF)
    actual, stats = train(case)
    assert_identical(expected, actual)
    assert all(row["tuning"]["round_buffers"] == 0 for row in control)
    assert all(row["tuning"]["round_buffers"] == 1 for row in stats)
    assert [row["command_buffers"] for row in stats] == [row["command_buffers"] for row in control]


@pytest.mark.parametrize("switch", SWITCHES)
@pytest.mark.parametrize("case,params", [("regression", {"max_depth": 2}), ("missing", {"max_depth": 6}),
                                         ("regression", {"max_depth": 10}), ("bytree", {}),
                                         ("regression", {"metal_gpu_tree": 0}),
                                         ("regression", {"monotone_constraints": "(1,0,0,0,0,0,-1)"}),
                                         ("regression", {"subsample": 0.6}),
                                         ("regression", {"metal_tree_fixed_workspace": 0}),
                                         ("regression", {"metal_tree_persistent_workspace": 0}),
                                         ("regression", {"metal_tree_workspace": 0})])
def test_each_switch_is_an_exact_same_binary_control(switch, case, params):
    expected, _ = train(case, {**params, switch: 0})
    actual, stats = train(case, {**params, switch: 1})
    assert_identical(expected, actual)
    gpu = stats[-1]["tuning"]["gpu_tree"]
    if switch == "metal_gpu_row_init":
        assert all(row["tuning"]["gpu_row_init"] == row["tuning"]["gpu_tree"] for row in stats)
    if switch == "metal_replay_path_elision":
        assert stats[-1]["tuning"]["replay_path_elision"] == gpu
    if switch == "metal_commit_early":
        # Row sampling keeps its separate gradient command.
        assert all(row["tuning"]["commit_early"] == int("subsample" not in params) for row in stats)


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "count:poisson",
                                       "reg:pseudohubererror", "reg:absoluteerror",
                                       "reg:quantileerror", "survival:cox", "rank:ndcg"])
def test_objectives_with_and_without_reusable_gradients(objective):
    params = {"objective": objective}
    if objective == "reg:quantileerror":
        params["quantile_alpha"] = 0.3
    if objective == "rank:ndcg":
        params["lambdarank_pair_method"] = "topk"

    def run(enabled):
        X, y, _, _ = inputs()
        if objective in ("binary:logistic", "rank:ndcg"):
            y = (y > 0).astype(np.float32)
        if objective == "count:poisson":
            y = np.abs(y)
        metadata = {"qid": np.repeat(np.arange(len(y) // 64 + 1), 64)[:len(y)]} if objective == "rank:ndcg" else {}
        data = xgb.DMatrix(X, label=y, **metadata)
        model = xgb.Booster(dict(max_depth=4, max_bin=32, seed=3, metal_round_buffers=enabled, **params), [data])
        for iteration in range(4):
            model.update(data, iteration)
        return snapshot(model, [data])

    assert_identical(run(0), run(1))


def raw_margin(model, data):
    """Returns the native result pointer, which stays valid until this thread's next prediction."""
    shape = ctypes.POINTER(ctypes.c_uint64)()
    dims = ctypes.c_uint64()
    values = ctypes.POINTER(ctypes.c_float)()
    config = json.dumps({"type": 1, "training": False, "iteration_begin": 0, "iteration_end": 0,
                         "strict_shape": False}).encode()
    xgb.core._check_call(xgb.core._LIB.XGBoosterPredictFromDMatrix(
        model.handle, data.handle, config, ctypes.byref(shape), ctypes.byref(dims), ctypes.byref(values)))
    return values


@pytest.mark.parametrize("async_rounds", [0, 2])
@pytest.mark.parametrize("round_buffers", [0, 1])
def test_returned_margins_are_never_rewritten_by_later_rounds(async_rounds, round_buffers):
    X, y, _, _ = inputs()
    data = xgb.DMatrix(X, label=y)
    model = xgb.Booster(dict(max_depth=4, max_bin=32, base_score=0, seed=5,
                             metal_round_buffers=round_buffers, metal_async_rounds=async_rounds), [data])
    for iteration in range(2):
        model.update(data, iteration)
    # The complete cached training margin is returned without a copy.
    pointer = raw_margin(model, data)
    before = np.ctypeslib.as_array(pointer, (len(y),)).copy()
    for iteration in range(2, 7):
        model.update(data, iteration)
    np.testing.assert_array_equal(np.ctypeslib.as_array(pointer, (len(y),)).view(np.uint32),
                                  before.view(np.uint32))
    # The earlier result still equals a fresh prediction of the first two rounds.
    expected = model.predict(data, output_margin=True, iteration_range=(0, 2))
    np.testing.assert_array_equal(expected.view(np.uint32), before.view(np.uint32))


@pytest.mark.parametrize("constant_inputs", [0, 1])
def test_constant_input_slots_survive_driver_shape_and_mask_changes(constant_inputs):
    def run(enabled):
        model = xgb.Booster(dict(max_depth=4, max_bin=32, base_score=0, seed=23, seed_per_iteration=True,
                                 metal_tree_constant_inputs=enabled))
        trace = []
        settings = [{"max_depth": 4}, {"max_depth": 12}, {"max_depth": 4},
                    {"colsample_bytree": 0.6}, {"colsample_bytree": 1, "max_depth": 6},
                    {"monotone_constraints": "(1,0,0,0,0,0,0)"}, {"monotone_constraints": "()"},
                    {"max_depth": 6}, {"max_depth": 2}]
        for iteration, (setting, rows) in enumerate(zip(settings, [513, 513, 129, 513, 1025, 513, 513, 0, 257])):
            X, y, _, _ = inputs(rows=rows)
            data = xgb.DMatrix(X, label=y)
            model.set_param(setting)
            if iteration == 6:
                model.reset()
            model.update(data, iteration)
            trace.append(snapshot(model, [data]))
            assert profile(model)["tuning"]["tree_constant_inputs"] == enabled
        return trace

    for expected, actual in zip(run(0), run(constant_inputs)):
        assert_identical(expected, actual)


def test_gpu_row_identities_cover_empty_single_and_large_matrices():
    for rows in [0, 1, 255, 256, 257, 70001]:
        results = []
        for enabled in [0, 1]:
            rng = np.random.default_rng(rows)
            X = rng.normal(size=(rows, 3)).astype(np.float32)
            y = (X[:, 0] - X[:, 1]).astype(np.float32)
            data = xgb.DMatrix(X, label=y)
            model = xgb.Booster(dict(max_depth=3, max_bin=16, base_score=0, metal_gpu_row_init=enabled), [data])
            for iteration in range(2):
                model.update(data, iteration)
            assert profile(model)["tuning"]["gpu_row_init"] == enabled
            results.append(snapshot(model, [data]))
        assert_identical(*results)


@pytest.mark.parametrize("coalesce", [None, 0, 1])
def test_early_commit_keeps_encoders_command_counts_and_failure_retry(coalesce):
    traces = []
    for enabled in [0, 1]:
        X, y, _, _ = inputs()
        data = xgb.DMatrix(X, label=y)
        params = dict(max_depth=6, max_bin=32, base_score=0, seed=71, metal_commit_early=enabled)
        if coalesce is not None:
            params["metal_coalesce_encoders"] = coalesce
        model = xgb.Booster(params, [data])
        stats = []
        for iteration in range(4):
            if iteration == 2:
                # The prologue may already be committed when tree setup fails.
                model.set_param("metal_split_threads", 17)
                with pytest.raises(xgb.core.XGBoostError, match="metal_split_threads"):
                    model.update(data, iteration)
                model.set_param("metal_split_threads", 256)
            model.update(data, iteration)
            stats.append(profile(model))
        assert all(row["tuning"]["commit_early"] == int(enabled and coalesce != 1) for row in stats)
        traces.append((snapshot(model, [data]), stats))
    assert_identical(traces[0][0], traces[1][0])
    for key in ["tree_encoders"]:
        assert [row["tuning"][key] for row in traces[0][1]] == [row["tuning"][key] for row in traces[1][1]]
    assert [row["command_buffers"] for row in traces[0][1]] == [row["command_buffers"] for row in traces[1][1]]


@pytest.mark.parametrize("invalid,message,restore", [
    ({"reg_lambda": -1}, "nonnegative", {"reg_lambda": 1}),
    ({"subsample": 2}, "Sampling fractions", {"subsample": 1}),
    ({"grow_policy": "x"}, "grow_policy", {"grow_policy": "depthwise"}),
    ({"max_depth": 31}, "max_depth", {"max_depth": 3}),
    ({"objective": "reg:tweedie", "tweedie_variance_power": 2.5}, "tweedie_variance_power",
     {"objective": "reg:squarederror", "tweedie_variance_power": 1.5}),
    ({"objective": "reg:pseudohubererror", "huber_slope": 0}, "huber_slope",
     {"objective": "reg:squarederror", "huber_slope": 1}),
])
def test_parameter_snapshot_still_rejects_changed_invalid_values(invalid, message, restore):
    results = []
    for enabled in [0, 1]:
        X, y, _, _ = inputs()
        data = xgb.DMatrix(X, label=np.abs(y))
        model = xgb.Booster(dict(max_depth=3, max_bin=16, base_score=0.5, metal_parameter_snapshot=enabled), [data])
        model.update(data, 0)
        model.set_param(invalid)
        # A rejected parameter set is never cached as validated.
        for _ in range(2):
            with pytest.raises(xgb.core.XGBoostError, match=message):
                model.update(data, 1)
        model.set_param(restore)
        for iteration in range(1, 3):
            model.update(data, iteration)
        results.append(snapshot(model, [data]))
    assert_identical(*results)


@pytest.mark.parametrize("depth", [6, 10])
def test_replay_path_elision_at_full_frontiers(depth):
    from test_gpu_tree_deep import full_frontier_data
    X, y = full_frontier_data(False)
    results = []
    for enabled in [0, 1]:
        data = xgb.QuantileDMatrix(X, label=y, max_bin=2)
        model = xgb.Booster(dict(max_depth=depth, max_bin=2, reg_lambda=0, min_child_weight=1,
                                 metal_replay_path_elision=enabled), [data])
        model.update(data, 0)
        stats = profile(model)
        assert stats["tuning"]["gpu_tree_levels"] == depth
        assert stats["tuning"]["replay_path_elision"] == enabled
        results.append(snapshot(model, [data]))
    assert_identical(*results)


@pytest.mark.parametrize("workspace,persistent", list(itertools.product([0, 1], repeat=2)))
def test_round_buffers_follow_shape_changes_and_reset(workspace, persistent):
    def run(enabled):
        model = xgb.Booster(dict(max_depth=3, max_bin=32, base_score=0, seed=9, metal_round_buffers=enabled,
                                 metal_tree_workspace=workspace, metal_tree_persistent_workspace=persistent))
        trace = []
        for iteration, (rows, objective) in enumerate([(513, "reg:squarederror"), (65, "reg:squarederror"),
                                                       (1025, "binary:logistic"), (0, "reg:squarederror"),
                                                       (513, "reg:squarederror"), (129, "multi:softprob")]):
            X, y, _, _ = inputs(rows=rows)
            params = {"objective": objective}
            if objective == "binary:logistic":
                y = (y > 0).astype(np.float32)
            if objective == "multi:softprob":
                y = np.argmax(X[:, :3], axis=1).astype(np.float32)
                params["num_class"] = 3
            else:
                params["num_class"] = 0
            data = xgb.DMatrix(X, label=y)
            model.set_param(params)
            if iteration == 4:
                model.reset()
            model.update(data, iteration)
            trace.append(snapshot(model, [data]))
        return trace

    for expected, actual in zip(run(0), run(1)):
        assert_identical(expected, actual)
