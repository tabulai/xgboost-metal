import json

import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("bins", [2, 31, 32, 33, 127, 256, 257, 513])
@pytest.mark.parametrize("policy", ["depthwise", "lossguide"])
@pytest.mark.parametrize("custom", [False, True])
def test_parallel_split_scan_matches_serial(bins, policy, custom):
    rng = np.random.default_rng(833)
    X = rng.integers(-200, 201, (1025, 7)).astype(np.float32) / 64
    X[:, 1] = X[:, 0]
    y = (1.2 * X[:, 0] - X[:, 3] ** 2 + X[:, 5]).astype(np.float32)
    X[::11, 2] = np.nan
    X[::13, 4] = np.nan
    X[::17, 6] = np.nan
    data = xgb.DMatrix(X, label=y)
    params = dict(max_bin=bins, grow_policy=policy, max_depth=4 if policy == "depthwise" else 0,
                  max_leaves=15, seed=379, base_score=0, eta=0.2, reg_alpha=0.17,
                  reg_lambda=0.4, max_delta_step=1, min_child_weight=0.5,
                  colsample_bylevel=0.75, colsample_bynode=0.8,
                  monotone_constraints="(1,1,0,-1,0,0,0)",
                  sampling_method="gradient_based", subsample=0.71)
    objective = None
    if custom:
        row = np.arange(len(X))
        grad = ((row % 7 - 3) / 4).astype(np.float32)
        hess = np.where(row % 5, 1.25, -0.5).astype(np.float32)
        objective = lambda prediction, matrix: (grad, hess)
    serial = xgb.train({**params, "metal_parallel_splits": 0}, data, 3, obj=objective)
    parallel = xgb.train({**params, "metal_parallel_splits": 1}, data, 3, obj=objective)
    assert parallel.save_raw(raw_format="json") == serial.save_raw(raw_format="json")
    for options in ({}, {"pred_leaf": True}, {"pred_contribs": True}):
        np.testing.assert_array_equal(parallel.predict(data, **options), serial.predict(data, **options))
    assert (json.loads(parallel.save_config())["learner"]["generic_param"]["rng_state"]
            == json.loads(serial.save_config())["learner"]["generic_param"]["rng_state"])
