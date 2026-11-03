"""Asynchronous boosting rounds (metal_async_rounds) produce identical models.

An eligible GPU-driven round returns after committing its command. The next
round's margin update reads a GPU-emitted Node table; the host replays every
tree from the same history and compares the bytes before publishing it.
Every observation of model state first retires the committed rounds.
"""
import ctypes
import gc
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, inputs, profile, snapshot
from test_sampling import compare as compare_reference
from test_sampling import dataset as reference_dataset

OBJECTIVES = ["reg:squarederror", "binary:logistic", "binary:logitraw", "count:poisson",
              "reg:squaredlogerror", "reg:gamma", "reg:tweedie", "reg:pseudohubererror", "binary:hinge"]


def setup(depth, case="regression", params=None, *, rows=513):
    X, y, metadata, options = inputs(case, rows)
    options = dict(max_depth=4, max_bin=32, base_score=0.4, seed=41, metal_async_rounds=depth, **options)
    options.update(params or {})
    if options.get("objective") in ("binary:logistic", "binary:logitraw", "binary:hinge"):
        y = (y > 0).astype(np.float32)
    elif options.get("objective") in ("count:poisson", "reg:squaredlogerror", "reg:gamma", "reg:tweedie"):
        y = (np.abs(y) + 0.05).astype(np.float32)
    data = xgb.DMatrix(X, label=y, **metadata)
    fresh = xgb.DMatrix(X, **{k: v for k, v in metadata.items() if k in ("base_margin", "feature_types", "missing")})
    return xgb.Booster(options, [data]), data, fresh


def train(depth, case="regression", params=None, *, rounds=6, action=None):
    model, data, fresh = setup(depth, case, params)
    for iteration in range(rounds):
        model.update(data, iteration)
        if action:
            action(model, data, iteration)
    stats = profile(model)
    return snapshot(model, [data, fresh]), stats


def compare(case="regression", params=None, depths=(1, 2), **kwargs):
    expected, control = train(0, case, params, **kwargs)
    assert control["tuning"]["async_rounds"] == 0
    results = []
    for depth in depths:
        actual, stats = train(depth, case, params, **kwargs)
        assert_identical(expected, actual)
        results.append(stats)
    return control, results


@pytest.mark.parametrize("case", ["regression", "missing", "ties", "weighted", "constant", "empty",
                                  "subnormal_features", "positive_infinity_missing", "bytree"])
@pytest.mark.parametrize("max_depth", [2, 6, 10])
def test_models_predictions_and_rng_match_synchronous_rounds(case, max_depth):
    control, results = compare(case, {"max_depth": max_depth}, depths=(1, 2, 4))
    for stats in results:
        assert stats["tuning"]["async_rounds"] > 0
        assert stats["tuning"]["gpu_tree"] == 1
        for field in ["gpu_tree_levels", "tree_control_fusion", "fused_subtraction", "split_threads"]:
            assert stats["tuning"][field] == control["tuning"][field], field
        # Each round still uses one command buffer; only its wait moves.
        assert stats["command_buffers"] == control["command_buffers"]


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_scalar_objectives_preserve_rounding(objective):
    compare(params={"objective": objective, "reg_alpha": 0.01, "reg_lambda": 0.3, "max_delta_step": 0.5})


def test_subnormal_leaves_are_scaled_exactly_on_the_gpu():
    X = np.arange(64, dtype=np.float32)[:, None]
    y = np.where(X[:, 0] < 32, -0.25, 0.25).astype(np.float32)
    results = []
    for depth in [0, 2]:
        data = xgb.DMatrix(X, label=y)
        # Ordinary weights times the smallest normal learning rate are subnormal.
        model = xgb.Booster(dict(max_depth=3, max_bin=16, base_score=0, reg_lambda=0,
                                 learning_rate=float(np.finfo(np.float32).tiny),
                                 metal_async_rounds=depth), [data])
        for iteration in range(4):
            model.update(data, iteration)
        document = json.loads(model.save_raw(raw_format="json"))
        tree = document["learner"]["gradient_booster"]["model"]["trees"][0]
        leaves = np.asarray(tree["split_conditions"], np.float32)[np.asarray(tree["left_children"]) < 0]
        assert np.any((np.abs(leaves) > 0) & (np.abs(leaves) < np.finfo(np.float32).tiny))
        results.append(snapshot(model, [data]))
    assert_identical(*results)


@pytest.mark.parametrize("eta", [0.3, 0.1, 1.0, 0.0, 1e-30, float(np.finfo(np.float32).tiny),
                                 float(np.finfo(np.float32).max), -0.5])
def test_gpu_leaf_scaling_matches_host_multiply_on_boundary_patterns(eta):
    mismatches, first = ctypes.c_uint64(), ctypes.c_uint32()
    # All positive and negative subnormals and small normals, the overflow
    # boundary, infinities and every NaN class near them.
    for begin, count in [(0, 1 << 24), (0x80000000, 1 << 24), (0x7f000000, 0x01000000),
                         (0xff000000, 0x01000000), (0x3f000000, 1 << 20)]:
        xgb.core._check_call(xgb.core._LIB.XGBMetalValidateLeafScaling(
            ctypes.c_float(eta), ctypes.c_uint64(begin), ctypes.c_uint64(count),
            ctypes.byref(mismatches), ctypes.byref(first)))
        assert mismatches.value == 0, hex(first.value)


@pytest.mark.parametrize("begin,count", [(2**64 - 1, 2), (2**32, 1), (2**32 - 1, 2), (0, 2**32 + 1)])
def test_leaf_scaling_validation_rejects_out_of_range_sweeps(begin, count):
    mismatches, first = ctypes.c_uint64(), ctypes.c_uint32()
    status = xgb.core._LIB.XGBMetalValidateLeafScaling(
        ctypes.c_float(0.3), ctypes.c_uint64(begin), ctypes.c_uint64(count),
        ctypes.byref(mismatches), ctypes.byref(first))
    assert status != 0


@pytest.mark.parametrize("consumer", ["predict", "margin", "leaf", "shap", "evaluate", "save", "config",
                                      "dump", "rounds", "score", "slice", "copy", "profile", "attributes",
                                      "empty_evaluate", "feature_names"])
def test_consumers_observe_every_committed_round(consumer):
    observations = {}

    def action(model, data, iteration):
        if consumer in ("predict", "margin", "leaf", "shap"):
            options = {"margin": {"output_margin": True}, "leaf": {"pred_leaf": True},
                       "shap": {"pred_contribs": True}}.get(consumer, {})
            result = model.predict(data, **options)
            value = (result.shape, result.tobytes())
        elif consumer == "evaluate":
            value = model.eval_set([(data, "train")], iteration)
        elif consumer == "empty_evaluate":
            value = model.eval_set([], iteration)
        elif consumer == "save":
            value = bytes(model.save_raw(raw_format="ubj"))
        elif consumer == "config":
            value = json.loads(model.save_config())["learner"]["gradient_booster"]["gbtree_model_param"]
        elif consumer == "dump":
            value = model.get_dump(with_stats=True)
        elif consumer == "rounds":
            value = model.num_boosted_rounds()
        elif consumer == "score":
            value = model.get_score(importance_type="total_gain")
        elif consumer == "slice":
            value = bytes(model[: iteration + 1].save_raw(raw_format="ubj"))
        elif consumer == "copy":
            value = bytes(model.copy().save_raw(raw_format="ubj"))
        elif consumer == "profile":
            tuning = profile(model)["tuning"]
            value = (tuning["gpu_tree_levels"], tuning["gpu_tree"])
        elif consumer == "attributes":
            model.set_attr(round=str(iteration))
            value = model.attr("round")
        else:
            # Feature metadata does not depend on committed rounds.
            value = (model.feature_names, model.feature_types, model.num_features())
        observations.setdefault(iteration, []).append(value)

    compare(action=action, depths=(2,))
    assert all(a == b for a, b in observations.values())


@pytest.mark.parametrize("change", ["eta", "depth", "host_depth", "objective", "disable", "async_depth",
                                    "custom", "parallel", "base_margin", "sampling", "labels", "monotone",
                                    "categorical", "reset", "load_model", "load_config", "process_type"])
def test_mid_training_changes_retire_rounds_in_order(change):
    def run(depth):
        model, data, fresh = setup(depth)
        for iteration in range(3):
            model.update(data, iteration)
        custom = None
        if change == "eta":
            model.set_param("eta", 0.07)
        elif change == "depth":
            model.set_param("max_depth", 6)
        elif change == "host_depth":
            # Beyond the GPU depth limit the host driver retires rounds first.
            model.set_param("max_depth", 12)
        elif change == "objective":
            model.set_param("objective", "reg:pseudohubererror")
        elif change == "disable":
            model.set_param("metal_async_rounds", 0)
        elif change == "async_depth":
            model.set_param("metal_async_rounds", 1 if depth else 0)
        elif change == "custom":
            custom = lambda prediction, matrix: (prediction - matrix.get_label(), np.ones_like(prediction))
        elif change == "parallel":
            model.set_param("num_parallel_tree", 2)
        elif change == "base_margin":
            data.set_base_margin(np.linspace(-0.2, 0.3, data.num_row(), dtype=np.float32))
        elif change == "sampling":
            model.set_param("subsample", 0.7)
        elif change == "labels":
            data.set_label(-data.get_label())
            data.set_weight(np.linspace(0.1, 1.7, data.num_row(), dtype=np.float32))
        elif change == "monotone":
            model.set_param("monotone_constraints", "(1,0,0,0,0,0,0)")
        elif change == "categorical":
            data.feature_types = ["c"] + ["q"] * 6
        elif change == "reset":
            model.reset()
        elif change == "load_model":
            raw = model.save_raw(raw_format="ubj")
            model.load_model(raw)
            model.set_param("metal_async_rounds", depth)
        elif change == "load_config":
            model.load_config(model.save_config())
        else:
            model.set_param({"process_type": "update", "updater": "refresh"})
        for iteration in range(3, 6):
            model.update(data, iteration, fobj=custom)
        if change == "process_type":
            assert model.num_boosted_rounds() == 3
        return snapshot(model, [data, fresh])

    assert_identical(run(0), run(2))


def test_xgb_train_callbacks_evaluation_and_early_stopping():
    X, y, _, _ = inputs(rows=1025)
    results = []
    for depth in [0, 1, 3]:
        data = xgb.DMatrix(X[:769], label=y[:769])
        valid = xgb.DMatrix(X[769:], label=y[769:])
        params = dict(max_depth=5, max_bin=32, base_score=0, seed=3, eta=0.4, metal_async_rounds=depth)

        class Inspect(xgb.callback.TrainingCallback):
            def __init__(self):
                super().__init__()
                self.rounds = []

            def after_iteration(self, model, epoch, evals_log):
                if epoch % 3 == 2:
                    self.rounds.append(model.num_boosted_rounds())
                return False

        inspect = Inspect()
        plain = xgb.train(params, data, num_boost_round=12, callbacks=[inspect])
        history = {}
        stopped = xgb.train(params, data, num_boost_round=200, evals=[(data, "train"), (valid, "valid")],
                            evals_result=history, early_stopping_rounds=3, verbose_eval=False)
        results.append((snapshot(plain, [data, valid]), snapshot(stopped, [data, valid]), inspect.rounds,
                        history, stopped.best_iteration, stopped.num_boosted_rounds()))
    for other in results[1:]:
        assert_identical(results[0][0], other[0])
        assert_identical(results[0][1], other[1])
        assert results[0][2:] == other[2:]
    assert results[0][2] == [3, 6, 9, 12]


def test_continued_training_and_independent_reference(reference):
    X, y = reference_dataset(rows=257)
    options = {"max_depth": 3, "max_bin": 32, "seed": 239, "base_score": 0.25, "eta": 0.2}
    expected, files = reference(X, y, options, rounds=6)
    data = xgb.DMatrix(X, label=y)
    first = xgb.train({**options, "metal_async_rounds": 2}, data, num_boost_round=3)
    # Continue from the drained model through the ordinary training API.
    model = xgb.train({**options, "metal_async_rounds": 2}, data, num_boost_round=3, xgb_model=first)
    assert model.num_boosted_rounds() == 6
    compare_reference(model, data, expected, files)


@pytest.mark.parametrize("switch", ["matrix", "thread", "learner", "deleted_matrix", "deleted_booster"])
def test_matrix_thread_and_lifetime_changes(switch):
    def run(depth):
        model, first, fresh = setup(depth)
        _, second, _ = setup(depth, "missing")
        other, other_data, _ = setup(depth, "ties")
        X, y, _, _ = inputs()
        with ThreadPoolExecutor(max_workers=1) as pool:
            for iteration in range(6):
                data = second if switch == "matrix" and iteration % 2 else first
                if switch == "thread" and iteration % 2:
                    pool.submit(model.update, data, iteration).result()
                else:
                    model.update(data, iteration)
                if switch == "learner":
                    other.update(other_data, iteration)
                if switch == "deleted_matrix" and iteration == 2:
                    # Committed rounds retain their GPU inputs.
                    temporary = xgb.DMatrix(X, label=y)
                    model.update(temporary, iteration)
                    del temporary
                    gc.collect()
        if switch == "deleted_booster":
            doomed, doomed_data, _ = setup(depth)
            for iteration in range(4):
                doomed.update(doomed_data, iteration)
            del doomed
            gc.collect()
        return snapshot(model, [first, second, fresh]), snapshot(other, [other_data]) if switch == "learner" else None

    expected, actual = run(0), run(2)
    assert_identical(expected[0], actual[0])
    if expected[1]:
        assert_identical(expected[1], actual[1])


@pytest.mark.parametrize("fault,message", [(1, "GPU-emitted tree nodes differ"),
                                           (2, "node count differs")])
@pytest.mark.parametrize("observer", ["rounds", "update", "host_tree"])
def test_failed_round_rolls_back_and_training_continues(fault, message, observer):
    model, data, fresh = setup(1 if observer == "update" else 3)
    model.set_param({"metal_async_fault": fault, "metal_async_fault_round": 3})
    # The failure surfaces at the first retirement of round three: a draining
    # call, back-pressure in a later update, or this update if it completed.
    error = None
    for iteration in range(5 if observer == "update" else 4):
        try:
            model.update(data, iteration)
        except xgb.core.XGBoostError as raised:
            error = raised
            break
    if error is None:
        with pytest.raises(xgb.core.XGBoostError, match=message):
            if observer == "host_tree":
                # A host-driven tree retires rounds after consuming the
                # pending margin recipe of the failed round.
                model.set_param("max_depth", 12)
                model.update(data, 4)
            else:
                model.num_boosted_rounds()
    else:
        assert message in str(error) and iteration >= 3
    model.set_param("max_depth", 4)
    assert model.num_boosted_rounds() == 3
    model.set_param("metal_async_fault", 0)
    for iteration in range(3, 8):
        model.update(data, iteration)
    expected_model, expected_data, expected_fresh = setup(0)
    for iteration in range(8):
        expected_model.update(expected_data, iteration)
    assert_identical(snapshot(expected_model, [expected_data, expected_fresh]), snapshot(model, [data, fresh]))


def test_failure_reaches_xgb_train_and_leaves_a_consistent_model():
    X, y, _, _ = inputs()
    data = xgb.DMatrix(X, label=y)
    params = dict(max_depth=4, max_bin=32, base_score=0, seed=1, metal_async_rounds=2,
                  metal_async_fault=1, metal_async_fault_round=5)
    with pytest.raises(xgb.core.XGBoostError, match="GPU-emitted tree nodes differ"):
        xgb.train(params, data, num_boost_round=10)


@pytest.mark.parametrize("invalid,value", [("metal_async_rounds", 5), ("metal_histogram_rows", 1)])
def test_invalid_parameters_fail_before_commit(invalid, value):
    def run(depth):
        model, data, fresh = setup(depth)
        for iteration in range(2):
            model.update(data, iteration)
        before = bytes(model.save_raw())
        model.set_param(invalid, value)
        with pytest.raises(xgb.core.XGBoostError, match=invalid):
            model.update(data, 2)
        assert bytes(model.save_raw()) == before
        model.set_param({"metal_async_rounds": depth, "metal_histogram_rows": 2048})
        for iteration in range(2, 6):
            model.update(data, iteration)
        return snapshot(model, [data, fresh])

    assert_identical(run(0), run(2))


def test_long_pipelines_match_after_many_rounds():
    expected, control = train(0, "missing", {"max_depth": 6}, rounds=300)
    for depth in [1, 4]:
        actual, stats = train(depth, "missing", {"max_depth": 6}, rounds=300)
        assert_identical(expected, actual)
        assert stats["command_buffers"] == control["command_buffers"]
