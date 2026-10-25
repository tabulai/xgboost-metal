"""Default split width and late GPU-tree encoder selection retain exact results."""
import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, inputs, profile, snapshot, train


@pytest.mark.parametrize("case", ["regression", "missing"])
@pytest.mark.parametrize("depth", [2, 6, 7, 10])
def test_default_split_width_matches_both_explicit_exact_paths(case, depth):
    outputs = []
    for threads in [32, 256, None]:
        params = {"max_depth": depth, "metal_compact_splits": 0}
        if threads is not None:
            params["metal_split_threads"] = threads
        result, stats = train(1, case, params, rounds=2)
        assert all(row["tuning"]["split_threads"] == (threads or 256) for row in stats)
        outputs.append(result)
    assert_identical(outputs[0], outputs[1])
    assert_identical(outputs[0], outputs[2])


@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize("fusion", [0, 1])
def test_automatic_tree_encoder_and_explicit_controls(quantile, fusion):
    outputs, profiles = [], []
    for coalesce in [0, 1, None]:
        params = {"max_depth": 6}
        if coalesce is not None:
            params["metal_coalesce_encoders"] = coalesce
        result, stats = train(1, params=params, quantile=quantile, fusion=fusion)
        outputs.append(result); profiles.append(stats)
    assert_identical(outputs[0], outputs[1])
    assert_identical(outputs[0], outputs[2])
    assert all(row["tuning"]["coalesced_encoders"] == 0 for row in profiles[0])
    assert all(row["tuning"]["coalesced_encoders"] == 1 for row in profiles[1])
    assert all(row["tuning"]["coalesced_encoders"] == 1 for row in profiles[2])
    assert [row["tuning"]["tree_encoders"] for row in profiles[1]] == [1, 1, 1]
    expected = [5, 5, 5] if quantile and fusion else [5, 6, 6] if fusion else [1, 1, 1]
    assert [row["tuning"]["tree_encoders"] for row in profiles[2]] == expected
    assert [row["command_buffers"] for row in profiles[0]] == [row["command_buffers"] for row in profiles[2]]


@pytest.mark.parametrize("case,params,gpu", [
    ("regression", {"max_depth": 1}, 0),
    ("regression", {"max_depth": 7}, 1),
    ("regression", {"max_depth": 10}, 1),
    ("regression", {"metal_gpu_tree": 0}, 0),
    ("regression", {"max_cached_hist_node": 1}, 0),
    ("regression", {"subsample": 0.7}, 0),
    ("regression", {"monotone_constraints": "(1,0,0,0,0,0,0)"}, 0),
    ("missing", {}, 1),
    ("categorical", {}, 0),
    ("vector", {}, 0),
])
def test_automatic_encoder_selection_leaves_other_paths_unchanged(case, params, gpu):
    result, stats = train(1, case, params={"max_depth": 6, **params}, rounds=2)
    control, old_stats = train(1, case, params={"max_depth": 6, **params, "metal_coalesce_encoders": 0}, rounds=2)
    assert_identical(control, result)
    assert all(row["tuning"]["coalesced_encoders"] == 0 for row in stats)
    assert all(row["tuning"]["gpu_tree"] == gpu for row in stats)
    assert [row["tuning"]["tree_encoders"] for row in stats] == [row["tuning"]["tree_encoders"] for row in old_stats]


@pytest.mark.parametrize("case", ["custom", "parallel", "multiclass", "empty", "constant"])
def test_owned_and_borrowed_tree_commands_preserve_results(case):
    expected, _ = train(1, case, {"max_depth": 6, "metal_coalesce_encoders": 0}, rounds=2)
    actual, _ = train(1, case, {"max_depth": 6}, rounds=2)
    assert_identical(expected, actual)


@pytest.mark.parametrize("invalid,value,valid", [("metal_histogram_rows", 1, 2048),
                                                 ("metal_split_threads", 17, 256)])
def test_failed_unsubmitted_command_then_automatic_retry(invalid, value, valid):
    results = []
    for explicit in [True, False]:
        X, y, _, _ = inputs()
        data = xgb.DMatrix(X, label=y)
        params = {"max_depth": 6, "max_bin": 32, "base_score": 0, "seed": 71}
        if explicit:
            params["metal_coalesce_encoders"] = 0
        model = xgb.Booster(params, [data])
        model.update(data, 0)
        before = bytes(model.save_raw())
        model.set_param(invalid, value)
        with pytest.raises(xgb.core.XGBoostError, match=invalid):
            model.update(data, 1)
        assert bytes(model.save_raw()) == before
        model.set_param(invalid, valid)
        for iteration in range(1, 4):
            model.update(data, iteration)
        results.append(snapshot(model, [data, xgb.DMatrix(X)]))
    assert_identical(results[0], results[1])


def test_automatic_selection_rechecks_depth_and_matrix_without_stale_mode():
    results = []
    for explicit in [True, False]:
        X, y, _, _ = inputs()
        missing = X.copy(); missing[::5, 0] = np.nan
        matrices = [xgb.DMatrix(X, label=y), xgb.DMatrix(missing, label=y)]
        params = {"max_depth": 6, "max_bin": 32, "base_score": 0, "seed": 19}
        if explicit:
            params["metal_coalesce_encoders"] = 0
        model = xgb.Booster(params, matrices)
        selected = []
        for iteration, (depth, matrix) in enumerate([(6, 0), (7, 0), (6, 1), (2, 0), (1, 0), (6, 0)]):
            model.set_param("max_depth", depth)
            model.update(matrices[matrix], iteration)
            selected.append(profile(model)["tuning"]["coalesced_encoders"])
        assert selected == ([0] * 6 if explicit else [1, 0, 0, 1, 0, 1])
        results.append(snapshot(model, matrices))
    assert_identical(results[0], results[1])
