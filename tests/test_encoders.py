import ctypes
import itertools
import json

import numpy as np
import pytest
import xgboost as xgb


def profile(model):
    result = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(result)))
    return json.loads(result.value)


@pytest.mark.parametrize("policy", ["depthwise", "lossguide"])
@pytest.mark.parametrize("categorical", [False, True], ids=["grouped-numeric", "large-categorical"])
@pytest.mark.parametrize("partition", [
    pytest.param({}, id="indirect-scan"),
    pytest.param({"metal_fused_partition": 0}, id="separate-scan"),
    pytest.param({"metal_partition_scan": 0, "metal_partition_aggregation": 0}, id="atomic"),
    pytest.param({"metal_partition_scan": 0, "metal_partition_aggregation": 1}, id="aggregated-atomic"),
])
def test_encoder_fill_and_workspace_modes_preserve_models(policy, categorical, partition):
    """Exercise indirect arguments, buffer reuse, histogram eviction and 8-byte clears."""
    rng = np.random.default_rng(852)
    X = rng.normal(size=(1031, 7)).astype(np.float32)
    if categorical:
        X[:, 0] = rng.integers(0, 97, len(X))
    y = (np.sin(X[:, 0]) + X[:, 1] * (X[:, 2] > 0) + X[:, 3] ** 2).astype(np.float32)
    X[rng.random(X.shape) < 0.09] = np.nan
    weight = rng.uniform(0.1, 2, len(X)).astype(np.float32)
    feature_types = ["c" if categorical else "q", *(["q"] * 6)]
    params = dict(grow_policy=policy, max_depth=0, max_leaves=13,
                  max_bin=513 if categorical else 32, max_cached_hist_node=1,
                  metal_features_per_group=4, subsample=0.8,
                  colsample_bylevel=0.9, colsample_bynode=0.8, seed=25, **partition)
    baseline = None
    for coalesce, compute_fill, workspace in itertools.product([0, 1], repeat=3):
        # Each model gets fresh quantization so command-buffer counts are comparable.
        data = xgb.DMatrix(X, label=y, weight=weight, feature_types=feature_types)
        model = xgb.train({**params, "metal_coalesce_encoders": coalesce,
                          "metal_compute_fill": compute_fill, "metal_tree_workspace": workspace}, data, 2)
        observed = dict(model=model.save_raw(raw_format="ubj"), prediction=model.predict(data),
                        leaf=model.predict(data, pred_leaf=True), profile=profile(model),
                        rng=json.loads(model.save_config())["learner"]["generic_param"]["rng_state"])
        if baseline is None:
            baseline = observed
            continue
        assert observed["model"] == baseline["model"]
        np.testing.assert_array_equal(observed["prediction"], baseline["prediction"])
        np.testing.assert_array_equal(observed["leaf"], baseline["leaf"])
        assert observed["rng"] == baseline["rng"]
        assert observed["profile"]["command_buffers"] == baseline["profile"]["command_buffers"]
        assert observed["profile"]["tuning"]["histogram_rebuilds"] > 0
        if coalesce:
            assert observed["profile"]["tuning"]["tree_encoders"] < baseline["profile"]["tuning"]["tree_encoders"]
        if workspace:
            tuning = observed["profile"]["tuning"]
            assert tuning["tree_buffer_allocations"] < tuning["tree_buffer_requests"]
