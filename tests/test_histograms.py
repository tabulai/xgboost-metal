import numpy as np
import pytest
import xgboost as xgb
import ctypes
import json


@pytest.mark.parametrize("bins", [16, 256, 513])
@pytest.mark.parametrize("feature_group", [1, 2, 4])
def test_histogram_subtraction_is_exact(bins, feature_group):
    rng = np.random.default_rng(502)
    X = rng.normal(size=(4097, 11)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, 2] ** 2).astype(np.float32)
    X[rng.random(X.shape) < 0.13] = np.nan
    weights = rng.uniform(0.01, 2, len(X)).astype(np.float32)
    d = xgb.DMatrix(X, label=y, weight=weights)
    params = dict(max_depth=5, max_bin=bins, subsample=0.8, colsample_bynode=0.7,
                  reg_alpha=0.1, seed=92)
    ordinary = xgb.train({**params, "metal_histogram_subtraction": 0, "metal_partition_aggregation": 0, "metal_partition_scan": 0}, d, 6)
    subtracted = xgb.train({**params, "metal_histogram_subtraction": 1, "metal_partition_aggregation": 1,
                            "metal_features_per_group": feature_group}, d, 6)
    assert ordinary.save_raw(raw_format="json") == subtracted.save_raw(raw_format="json")
    np.testing.assert_array_equal(ordinary.predict(d), subtracted.predict(d))


def test_update_cache_tracks_matrix_metadata():
    rng = np.random.default_rng(70)
    X = rng.normal(size=(257, 5)).astype(np.float32)
    y = X[:, 0] - X[:, 1]
    d = xgb.DMatrix(X, label=y)
    params = dict(max_depth=3, base_score=0)
    learner = xgb.Booster(params, cache=[d])
    learner.update(d, 0)
    fresh = xgb.Booster(params, model_file=learner.save_raw())
    margin = rng.normal(size=len(X)).astype(np.float32)
    d.set_base_margin(margin)
    learner.update(d, 1)
    fresh.update(xgb.DMatrix(X, label=y, base_margin=margin), 1)
    np.testing.assert_array_equal(learner.predict(d), fresh.predict(d))


def test_quantile_reference_out_of_range_values():
    X = np.arange(65, dtype=np.float32).reshape(-1, 1)
    reference = xgb.QuantileDMatrix(X, max_bin=16)
    # Upper outliers must not index the next column's histogram or the missing bin.
    Y = np.linspace(-1e5, 1e5, 71, dtype=np.float32).reshape(-1, 1)
    d = xgb.QuantileDMatrix(Y, label=(Y[:, 0] > 0).astype(np.float32),
                           max_bin=16, ref=reference)
    model = xgb.train(dict(max_depth=2, max_bin=16), d, 3)
    assert np.isfinite(model.predict(d)).all()


def test_feature_sampling_respects_weights():
    rng = np.random.default_rng(81)
    X = rng.normal(size=(301, 4)).astype(np.float32)
    y = X[:, 0] + X[:, 1]
    d = xgb.DMatrix(X, label=y, feature_weights=[0, 1, 0, 0])
    b = xgb.train(dict(max_depth=3, colsample_bytree=0.25, seed=2), d, 5)
    assert set(b.get_score()) == {'f1'}


@pytest.mark.parametrize("leaves", [0, 5, 19])
def test_gpu_work_planning_preserves_pruned_trees_and_removes_waits(leaves):
    rng = np.random.default_rng(308)
    X = rng.integers(0, 23, (1031, 5)).astype(np.float32)
    y = (X[:, 0] * (X[:, 1] > 8) - X[:, 2]).astype(np.float32)
    X[::13, 0] = np.nan
    data = xgb.DMatrix(X, label=y)
    params = dict(max_depth=5, max_bin=32, max_leaves=leaves, min_child_weight=10,
                  min_split_loss=0.3, subsample=0.8, colsample_bynode=0.7, seed=81)
    separate = xgb.train({**params, "metal_fused_partition": 0}, data, 4)
    fused = xgb.train({**params, "metal_fused_partition": 1}, data, 4)
    assert separate.save_raw(raw_format="json") == fused.save_raw(raw_format="json")
    np.testing.assert_array_equal(separate.predict(data), fused.predict(data))
    def profile(model):
        result = ctypes.c_char_p()
        xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(result)))
        return json.loads(result.value)
    control, current = profile(separate), profile(fused)
    assert current['tuning']['fused_partition'] == 1
    assert current['command_buffers'] < control['command_buffers']
