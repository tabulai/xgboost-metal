"""Exact controls for retained per-tree row/monotone workspace buffers."""
import itertools
import json

import numpy as np
import pytest
import xgboost as xgb

from test_gradient_root_fusion import CASES, compare, profile, train


@pytest.mark.parametrize("case,params,_fused", CASES, ids=[row[0] for row in CASES])
def test_fixed_buffers_preserve_model_predictions_and_rng(case, params, _fused):
    expected = train(case, params, True, tuning={"metal_tree_fixed_workspace": 0})
    actual = train(case, params, True, tuning={"metal_tree_fixed_workspace": 1})
    compare(expected, actual)
    assert expected[3]["command_buffers"] == actual[3]["command_buffers"]


@pytest.mark.parametrize("workspace,persistent", list(itertools.product([0, 1], repeat=2)))
def test_fixed_allocations_reuse_only_when_workspace_is_persistent(workspace, persistent):
    data = xgb.DMatrix(np.zeros((129, 3), np.float32), label=np.zeros(129, np.float32))
    model = xgb.Booster(dict(max_depth=1, base_score=0, metal_tree_fixed_workspace=1,
                             metal_tree_workspace=workspace, metal_tree_persistent_workspace=persistent), [data])
    allocations = []
    for iteration in range(3):
        if iteration == 2:
            model.reset()
        model.update(data, iteration)
        tuning = profile(model)["tuning"]
        assert tuning["tree_fixed_workspace"] == 1
        allocations.append(tuning["tree_fixed_buffer_allocations"])
    assert allocations == [4, 0 if workspace and persistent else 4, 4]


def row_trace(fixed):
    rng = np.random.default_rng(3442)
    model = xgb.Booster(dict(max_depth=3, base_score=0, metal_tree_fixed_workspace=fixed, seed=31))
    output = []
    for iteration, rows in enumerate([64, 1024, 8, 0, 1024]):
        X = rng.normal(size=(rows, 3)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] ** 2).astype(np.float32)
        X[::7, 2] = np.nan
        data = xgb.DMatrix(X, label=y)
        model.set_param("monotone_constraints", "(1,0,-1)" if iteration % 2 else "(-1,0,1)")
        model.update(data, iteration)
        output.append((bytes(model.save_raw()), model.predict(data),
                       profile(model)["tuning"]["tree_fixed_buffer_allocations"]))
    return output


def test_row_growth_shrink_zero_and_monotone_reinitialization():
    expected, actual = row_trace(0), row_trace(1)
    for control, candidate in zip(expected, actual):
        assert control[0] == candidate[0]
        assert control[1].tobytes() == candidate[1].tobytes()
    assert [item[2] for item in expected] == [4] * 5
    assert [item[2] for item in actual] == [4, 3, 0, 0, 0]


@pytest.mark.parametrize("policy", ["depthwise", "lossguide"])
def test_toggle_between_rounds_respects_prefix_and_row_buffer_swaps(policy):
    params = dict(grow_policy=policy, max_leaves=19)
    expected = train("regression", params, True, tuning={"metal_tree_fixed_workspace": 0})
    # Reuse the exact fixture generator; this test changes only the toggle sequence.
    from test_gradient_root_fusion import data_for
    X, y, meta = data_for("regression")
    data = xgb.DMatrix(X, label=y, **meta)
    model = xgb.Booster(dict(max_depth=3, max_bin=32, base_score=0.4, seed=13,
                             metal_gradient_root_fusion=1, **params), [data])
    for iteration, enabled in enumerate([1, 0, 1, 0]):
        model.set_param("metal_tree_fixed_workspace", enabled)
        model.update(data, iteration)
    assert bytes(model.save_raw()) == expected[0]
    assert model.predict(data, output_margin=True).tobytes() == expected[1][0].tobytes()
    assert json.loads(model.save_config())["learner"]["generic_param"]["rng_state"] == expected[2]


def test_feature_count_change_after_load_and_failed_shape_validation():
    rng = np.random.default_rng(85)
    matrices = []
    for columns in [7, 5]:
        X = rng.normal(size=(263, columns)).astype(np.float32)
        matrices.append(xgb.DMatrix(X, label=X[:, 0] - X[:, 1]))
    replacement = xgb.train(dict(max_depth=2, base_score=0, metal_tree_fixed_workspace=0), matrices[1], 2).save_raw()
    results = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=2, base_score=0, metal_tree_fixed_workspace=enabled), [matrices[0]])
        model.update(matrices[0], 0)
        with pytest.raises(xgb.core.XGBoostError, match="Number of columns"):
            model.update(matrices[1], 1)
        model.update(matrices[0], 1)
        model.load_model(replacement)
        model.set_param(dict(max_depth=2, metal_tree_fixed_workspace=enabled))
        model.update(matrices[1], 2)
        results.append((bytes(model.save_raw()), model.predict(matrices[1]).tobytes()))
    assert results[0] == results[1]


@pytest.mark.parametrize("case,params", [("regression", {"colsample_bytree": 0.8}),
                                          ("cox", {"objective": "survival:cox"})])
def test_failed_constructor_then_retry_reinitializes_retained_slots(case, params):
    compare(train(case, params, True, tuning={"metal_tree_fixed_workspace": 0}, retry=True, retry_iteration=0),
            train(case, params, True, tuning={"metal_tree_fixed_workspace": 1}, retry=True, retry_iteration=0))
