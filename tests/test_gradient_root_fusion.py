"""Gradient/root command fusion preserves model bytes and fallback behavior."""
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


CASES = [
    ("regression", {}, True),
    ("binary", {"objective": "binary:logistic"}, True),
    ("multiclass", {"objective": "multi:softprob", "num_class": 3, "num_parallel_tree": 2}, True),
    ("parallel", {"num_parallel_tree": 3}, True),
    ("vector", {"multi_strategy": "multi_output_tree", "num_parallel_tree": 2}, True),
    ("categorical", {"max_cat_to_onehot": 2}, True),
    ("constraints", {"monotone_constraints": "(1,0,-1,0,0,0,0)",
                     "interaction_constraints": "[[0,1,2],[3,4,5,6]]",
                     "colsample_bytree": 0.9, "colsample_bylevel": 0.8, "colsample_bynode": 0.7}, True),
    ("dropout", {"rate_drop": 0.3, "one_drop": True}, True),
    ("base_margin", {}, True),
    ("absolute", {"objective": "reg:absoluteerror"}, True),
    ("quantile", {"objective": "reg:quantileerror", "quantile_alpha": [0.2, 0.7]}, True),
    ("expectile", {"objective": "reg:expectileerror", "expectile_alpha": [0.2, 0.7]}, True),
    ("normal", {"objective": "reg:normal"}, True),
    ("aft", {"objective": "survival:aft"}, True),
    ("cox", {"objective": "survival:cox"}, False),
    ("sampled", {"subsample": 0.73}, False),
    ("gradient_sampled", {"subsample": 0.73, "sampling_method": "gradient_based"}, False),
    ("custom", {}, False),
    ("ranking", {"objective": "rank:ndcg", "lambdarank_unbiased": True}, False),
    ("linear", {"booster": "gblinear"}, False),
    ("refresh", {"updater": "grow_metal_hist,refresh"}, False),
    ("update", {}, False),
]


def data_for(case):
    rng = np.random.default_rng(7194)
    X = rng.normal(size=(259, 7)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] * X[:, 2] + 0.5 * X[:, 3] ** 2).astype(np.float32)
    meta = {"weight": rng.uniform(0.2, 1.8, len(X)).astype(np.float32)}
    if case == "binary":
        y = (y > 0).astype(np.float32)
    elif case == "multiclass":
        y = np.argmax(X[:, :3], axis=1).astype(np.float32)
    elif case == "vector":
        y = np.column_stack([y, y * 0.7 + X[:, 4], -y + X[:, 5]]).astype(np.float32)
    elif case == "categorical":
        X[:, 0] = np.arange(len(X)) % 7
        meta["feature_types"] = ["c"] + ["q"] * 6
    elif case == "base_margin":
        meta["base_margin"] = np.linspace(-0.3, 0.7, len(X), dtype=np.float32)
    elif case in ("aft", "cox"):
        y = np.exp(y * 0.05).astype(np.float32)
        if case == "aft":
            meta.update(label_lower_bound=y,
                        label_upper_bound=np.where(np.arange(len(y)) % 3, y, np.inf))
        else:
            y[::3] *= -1
    elif case == "ranking":
        y = (y > 0).astype(np.float32)
        meta = {"group": [37] * 7}
    X[::13, 4] = np.nan
    return X, y, meta


def train(case, params, enabled, tuning=None, retry=False, retry_iteration=1):
    X, y, meta = data_for(case)
    data = xgb.DMatrix(X, label=y, **meta)
    options = dict(max_depth=3, max_bin=32, base_score=0.4, seed=13,
                   metal_defer_prediction=0, metal_gradient_root_fusion=int(enabled), **params, **(tuning or {}))
    if case == "update":
        seed = xgb.train({**options, "metal_gradient_root_fusion": 0}, data, 4)
        model = xgb.Booster(options, [data], model_file=seed.save_raw())
        model.set_param({"process_type": "update", "updater": "refresh"})
    else:
        model = xgb.Booster(options, [data])

    def custom(prediction, matrix):
        return (prediction - matrix.get_label()).astype(np.float32), np.ones_like(prediction, dtype=np.float32)

    for iteration in range(4):
        if retry and iteration == retry_iteration:
            model.set_param("metal_histogram_rows", 1)
            with pytest.raises(xgb.core.XGBoostError, match="metal_histogram_rows"):
                model.update(data, iteration)
            model.set_param("metal_histogram_rows", 2048)
        model.update(data, iteration, fobj=custom if case == "custom" else None)
    stats = profile(model)
    fresh = xgb.DMatrix(X, **{k: v for k, v in meta.items() if k in ("base_margin", "feature_types")})
    arrays = [model.predict(data, output_margin=True), model.predict(fresh, output_margin=True)]
    if case != "linear":
        arrays.extend([model.predict(data, pred_leaf=True), model.predict(fresh, pred_leaf=True)])
    state = json.loads(model.save_config())["learner"]["generic_param"]["rng_state"]
    return bytes(model.save_raw()), arrays, state, stats


def compare(expected, actual):
    assert actual[0] == expected[0]
    assert actual[2] == expected[2]
    for wanted, own in zip(expected[1], actual[1]):
        assert wanted.dtype == own.dtype and wanted.shape == own.shape
        assert wanted.tobytes() == own.tobytes()


@pytest.mark.parametrize("case,params,fused", CASES, ids=[row[0] for row in CASES])
def test_fusion_preserves_models_and_expected_fallbacks(case, params, fused):
    expected = train(case, params, False)
    actual = train(case, params, True)
    compare(expected, actual)
    assert expected[3]["command_buffers"] - actual[3]["command_buffers"] == (4 if fused else 0)
    timings = actual[3]["gpu_seconds"]
    assert (timings.get("gradient_root", 0) + timings.get("gradient_tree", 0) > 0) == fused


@pytest.mark.parametrize("coalesce,fill", list(itertools.product([0, 1], repeat=2)))
@pytest.mark.parametrize("fused_partition", [0, 1])
def test_fusion_preserves_encoder_and_partition_modes(coalesce, fill, fused_partition):
    tuning = dict(metal_coalesce_encoders=coalesce, metal_compute_fill=fill,
                  metal_fused_partition=fused_partition)
    expected = train("categorical", {"max_cat_to_onehot": 2}, False, tuning)
    actual = train("categorical", {"max_cat_to_onehot": 2}, True, tuning)
    compare(expected, actual)
    assert expected[3]["command_buffers"] - actual[3]["command_buffers"] == 4


def test_pending_gradients_are_discarded_safely_after_tree_constructor_error():
    params = {"colsample_bytree": 0.8, "colsample_bynode": 0.7}
    compare(train("regression", params, False, retry=True),
            train("regression", params, True, retry=True))


def test_cox_order_cache_is_initialized_before_failed_first_tree_and_retry():
    params = {"objective": "survival:cox"}
    compare(train("cox", params, False, retry=True, retry_iteration=0),
            train("cox", params, True, retry=True, retry_iteration=0))
