"""Proposal tests: copy into tests/ only after coordinated source integration.

No asynchronous work crosses an API boundary. Deferred prediction is selected
with metal_defer_prediction=1; controls explicitly choose zero.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, inputs, profile, snapshot


def setup(enabled, case="regression", params=None, *, quantile=False, rows=513):
    X, y, metadata, options = inputs(case, rows)
    options = dict(max_depth=4, max_bin=32, base_score=0.4, seed=41,
                   metal_gradient_root_fusion=1, metal_gpu_tree=1,
                   metal_defer_prediction=enabled, **options)
    options.update(params or {})
    if options.get("objective") in ("binary:logistic", "binary:logitraw", "binary:hinge"):
        y = (y > 0).astype(np.float32)
    elif options.get("objective") in ("count:poisson", "reg:squaredlogerror", "reg:gamma", "reg:tweedie"):
        y = (np.abs(y) + 0.05).astype(np.float32)
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    matrix = constructor(X, label=y, **metadata, **({"max_bin": options["max_bin"]} if quantile else {}))
    model = xgb.Booster(options, [matrix])
    fresh = xgb.DMatrix(X, **{k: v for k, v in metadata.items() if k in ("base_margin", "feature_types", "missing")})
    return model, matrix, fresh


def train(enabled, case="regression", params=None, *, rounds=5, quantile=False, action=None):
    model, matrix, fresh = setup(enabled, case, params, quantile=quantile)
    history, checkpoints = [], []
    for iteration in range(rounds):
        model.update(matrix, iteration)
        history.append(profile(model))
        # Serialization alone must neither consume nor invalidate the recipe.
        checkpoints.append(bytes(model.save_raw(raw_format="ubj")))
        if action:
            action(model, matrix, iteration, enabled)
    result = snapshot(model, [matrix, fresh])
    return result, history, checkpoints, profile(model)


def compare(case="regression", params=None, **kwargs):
    expected = train(0, case, params, **kwargs)
    actual = train(1, case, params, **kwargs)
    assert_identical(expected[0], actual[0])
    assert actual[2] == expected[2]
    assert all(row["tuning"]["prediction_cache_deferred"] == 0 for row in expected[1])
    return expected, actual


@pytest.mark.parametrize("case", ["regression", "missing", "ties", "weighted", "constant", "empty", "multiclass"])
@pytest.mark.parametrize("depth", [2, 6, 10])
def test_exact_outputs_and_one_saved_wait_per_unobserved_round(case, depth):
    expected, actual = compare(case, {"max_depth": depth})
    assert all(row["tuning"]["prediction_cache_deferred"] == 1 for row in actual[1])
    assert [row["tuning"]["prediction_cache_consumed"] for row in actual[1]] == [0, 1, 1, 1, 1]
    assert expected[1][-1]["command_buffers"] - actual[1][-1]["command_buffers"] == 5
    # The final public prediction flushes the last deferred cache once.
    assert expected[3]["command_buffers"] - actual[3]["command_buffers"] == 4


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "binary:logitraw",
                                     "count:poisson", "reg:squaredlogerror", "reg:gamma",
                                     "reg:tweedie", "reg:pseudohubererror", "binary:hinge"])
def test_ordinary_scalar_objectives_preserve_rounding(objective):
    _, actual = compare(params={"objective": objective, "reg_alpha": 0.01, "reg_lambda": 0.3,
                                "max_delta_step": 0.5})
    assert all(row["tuning"]["prediction_cache_deferred"] == 1 for row in actual[1])


@pytest.mark.parametrize("case,params,quantile", [
    ("regression", {"metal_gpu_tree": 0}, False),
    ("regression", {"metal_gradient_root_fusion": 0}, False),
    ("regression", {"max_depth": 1}, False),
    ("regression", {"max_bin": 512}, False),
    ("regression", {"subsample": 0.7}, False),
    ("regression", {}, True),
    ("categorical", {}, False),
    ("parallel", {}, False),
    ("vector", {}, False),
    ("dropout", {}, False),
    ("base_margin", {}, False),
    ("regression", {"objective": "reg:absoluteerror"}, False),
    ("regression", {"updater": "grow_metal_hist,refresh"}, False),
])
def test_initial_fallbacks_preserve_eager_boundary(case, params, quantile):
    expected, actual = compare(case, params, quantile=quantile, rounds=3)
    assert all(row["tuning"]["prediction_cache_deferred"] == 0 for row in actual[1])
    assert all(row["tuning"]["prediction_cache_consumed"] == 0 for row in actual[1])
    assert expected[1][-1]["command_buffers"] == actual[1][-1]["command_buffers"]


@pytest.mark.parametrize("consumer", ["predict", "margin", "leaf", "shap", "evaluate"])
def test_public_consumers_flush_before_reading(consumer):
    observations = {}

    def action(model, matrix, iteration, enabled):
        if consumer == "evaluate":
            value = model.eval_set([(matrix, "train")], iteration)
        else:
            options = {"margin": {"output_margin": True}, "leaf": {"pred_leaf": True},
                       "shap": {"pred_contribs": True}}.get(consumer, {})
            result = model.predict(matrix, **options)
            value = (result.shape, result.dtype.str, result.tobytes())
        observations.setdefault(iteration, []).append(value)

    _, actual = compare(rounds=4, action=action)
    assert all(a == b for a, b in observations.values())
    assert all(row["tuning"]["prediction_cache_consumed"] == 0 for row in actual[1])


@pytest.mark.parametrize("operation", ["copy", "reset", "load_model", "load_config", "slice"])
def test_model_lifecycle_discards_only_unused_cache(operation):
    observations = []

    def action(model, matrix, iteration, enabled):
        if iteration != 1:
            return
        raw = model.save_raw(raw_format="ubj")
        if operation == "copy":
            other = model.copy()
            observations.append(snapshot(other, [matrix]))
        elif operation == "slice":
            other = model[:]
            observations.append(snapshot(other, [matrix]))
        elif operation == "reset":
            model.reset()
        elif operation == "load_model":
            model.load_model(raw)
            # Model files reset runtime tuning; keep this an explicit off/on
            # comparison after load instead of letting both use the default.
            model.set_param("metal_defer_prediction", enabled)
        else:
            model.load_config(model.save_config())

    compare(action=action)
    if observations:
        assert_identical(*observations)


@pytest.mark.parametrize("change", ["disable", "objective", "depth", "base_margin", "labels_weights", "custom"])
def test_parameter_and_metadata_switches_flush_correct_prefix(change):
    def run(enabled):
        model, data, fresh = setup(enabled)
        model.update(data, 0)
        if change == "disable":
            model.set_param("metal_defer_prediction", 0)
        elif change == "objective":
            model.set_param("objective", "reg:pseudohubererror")
        elif change == "depth":
            model.set_param("max_depth", 1)
        elif change == "base_margin":
            data.set_base_margin(np.linspace(-0.2, 0.3, data.num_row(), dtype=np.float32))
        elif change == "labels_weights":
            data.set_label(-data.get_label())
            data.set_weight(np.linspace(0.1, 1.7, data.num_row(), dtype=np.float32))

        def custom(prediction, matrix):
            return prediction - matrix.get_label(), np.ones_like(prediction)

        model.update(data, 1, fobj=custom if change == "custom" else None)
        model.set_param({"metal_defer_prediction": enabled, "max_depth": 4})
        if change == "base_margin":
            data.set_base_margin(np.empty(0, dtype=np.float32))
        model.update(data, 2)
        return snapshot(model, [data, fresh])

    assert_identical(run(0), run(1))


@pytest.mark.parametrize("invalid,value,valid", [("metal_histogram_rows", 1, 2048),
                                                ("metal_split_threads", 17, 32)])
def test_failed_encoding_never_publishes_partial_cache(invalid, value, valid):
    def run(enabled):
        model, data, fresh = setup(enabled)
        model.update(data, 0)
        before = bytes(model.save_raw())
        model.set_param(invalid, value)
        with pytest.raises(xgb.core.XGBoostError, match=invalid):
            model.update(data, 1)
        assert bytes(model.save_raw()) == before
        model.set_param(invalid, valid)
        for iteration in range(1, 5):
            model.update(data, iteration)
        return snapshot(model, [data, fresh])

    assert_identical(run(0), run(1))


@pytest.mark.parametrize("transition", ["max_bin", "feature_types", "quantile_feature_types"])
def test_requantization_and_feature_type_changes_preserve_captured_inputs(transition):
    def run(enabled):
        X, y, _, _ = inputs("regression")
        # Both numeric and categorical interpretations are valid; the original
        # numeric model has already captured its previous prediction input.
        X[:, 0] = np.arange(len(X)) % 5
        y = (X[:, 0] + X[:, 1] * X[:, 2]).astype(np.float32)
        quantile = transition == "quantile_feature_types"
        options = dict(max_depth=4, max_bin=32, base_score=0.4, seed=41,
                       metal_defer_prediction=enabled)
        constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
        data = constructor(X, label=y, **({"max_bin": 32} if quantile else {}))
        model = xgb.Booster(options, [data])
        model.update(data, 0)
        first = profile(model)
        assert first["tuning"]["prediction_cache_deferred"] == int(enabled and not quantile)
        if transition == "max_bin":
            model.set_param("max_bin", 64)
        else:
            data.feature_types = ["c"] + ["q"] * 6
        for iteration in range(1, 4):
            model.update(data, iteration)
        if quantile:
            assert profile(model)["tuning"]["prediction_cache_deferred"] == 0
        fresh = xgb.DMatrix(X, feature_types=data.feature_types)
        return snapshot(model, [data, fresh])

    assert_identical(run(0), run(1))


@pytest.mark.parametrize("switch", ["matrix", "thread", "learner", "cache_eviction"])
def test_cache_ownership_remains_per_matrix_thread_and_learner(switch):
    def run(enabled):
        model, first, fresh = setup(enabled)
        _, second, _ = setup(enabled, "missing")
        other, other_data, _ = setup(enabled, "ties")
        with ThreadPoolExecutor(max_workers=1) as pool:
            for iteration in range(5):
                data = second if switch == "matrix" and iteration % 2 else first
                if switch == "thread" and iteration % 2:
                    pool.submit(model.update, data, iteration).result()
                else:
                    model.update(data, iteration)
                if switch == "learner":
                    other.update(other_data, iteration)
                if switch == "cache_eviction" and iteration == 1:
                    # Keep matrices live to exercise the actual FIFO limit.
                    matrices = [xgb.DMatrix(np.full((2, 7), index / 100, np.float32)) for index in range(66)]
                    for matrix in matrices:
                        model.predict(matrix)
        return snapshot(model, [first, second, fresh]), snapshot(other, [other_data]) if switch == "learner" else None

    expected, actual = run(0), run(1)
    assert_identical(expected[0], actual[0])
    if expected[1]:
        assert_identical(expected[1], actual[1])


@pytest.mark.parametrize("case", ["regression", "missing"])
def test_depth_six_one_thousand_rounds(case):
    expected, actual = compare(case, {"max_depth": 6}, rounds=1000)
    assert expected[1][-1]["command_buffers"] - actual[1][-1]["command_buffers"] == 1000
