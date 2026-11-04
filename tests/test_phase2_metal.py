"""Phase 2 Metal tests: Universal GPU Tree Driver (M3) & Certified int32 / w8 Histograms (M4)."""
import itertools
import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, inputs, profile, snapshot, train


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("deferred_stats", [0, 1])
@pytest.mark.parametrize("depth", [2, 6, 9])
def test_gpu_tree_monotone_constraints_exact_parity_and_ordering(missing, deferred_stats, depth):
    case = "missing" if missing else "regression"
    constraints = "(1,0,-1,1,-1,0,1)"
    params = {
        "max_depth": depth,
        "monotone_constraints": constraints,
        "metal_deferred_parent_statistics": deferred_stats,
        "reg_alpha": 0.05,
        "reg_lambda": 0.7,
        "max_delta_step": 0.8,
    }
    expected, ctrl = train(0, case, params, rounds=4)
    actual, stats = train(1, case, params, rounds=4)
    assert_identical(expected, actual)
    assert all(s["tuning"]["gpu_tree"] == 0 for s in ctrl)
    assert all(s["tuning"]["gpu_tree"] == 1 for s in stats)

    # Verify monotonicity directly on probed inputs.
    rng = np.random.default_rng(123)
    X_base = rng.normal(size=(128, 7)).astype(np.float32)
    X, y, meta, _ = inputs(case)
    dtrain = xgb.DMatrix(X, label=y, **meta)
    bst = xgb.train({"max_depth": depth, "max_bin": 32, "monotone_constraints": constraints}, dtrain, num_boost_round=4)
    p0 = bst.predict(xgb.DMatrix(X_base))
    X_inc = X_base.copy()
    X_inc[:, 0] += 1.5
    assert np.all(bst.predict(xgb.DMatrix(X_inc)) >= p0 - 0.0)
    X_dec = X_base.copy()
    X_dec[:, 2] += 1.5
    assert np.all(bst.predict(xgb.DMatrix(X_dec)) <= p0 + 0.0)


def test_gpu_tree_monotone_async_rounds():
    params = {"max_depth": 5, "monotone_constraints": "(1,-1,0,1,0,-1,0)"}
    expected, _ = train(1, "regression", {**params, "metal_async_rounds": 0}, rounds=5)
    actual, stats = train(1, "regression", {**params, "metal_async_rounds": 2}, rounds=5)
    assert_identical(expected, actual)
    assert all(s["tuning"]["async_rounds"] == 2 for s in stats)


@pytest.mark.parametrize("method", ["uniform", "gradient_based"])
@pytest.mark.parametrize("subsample", [0.05, 0.5, 0.85])
@pytest.mark.parametrize("case", ["regression", "missing"])
def test_gpu_tree_subsample_exact_parity(method, subsample, case):
    params = {"max_depth": 6, "subsample": subsample, "sampling_method": method}
    expected, ctrl = train(0, case, params, rounds=4)
    actual, stats = train(1, case, params, rounds=4)
    assert_identical(expected, actual)
    assert all(s["tuning"]["gpu_tree"] == 0 for s in ctrl)
    assert all(s["tuning"]["gpu_tree"] == 1 for s in stats)


@pytest.mark.parametrize("cols", [7, 65, 700])
@pytest.mark.parametrize("bylevel,bynode,bytree", [
    (0.6, 1.0, 1.0),
    (1.0, 0.6, 1.0),
    (0.75, 0.7, 0.85),
])
def test_gpu_tree_colsample_bylevel_bynode_exact_mt19937_parity(cols, bylevel, bynode, bytree):
    rng = np.random.default_rng(9001 + cols)
    rows = 257 if cols == 700 else 513
    X = rng.normal(size=(rows, cols)).astype(np.float32)
    X[::11, 0] = np.nan
    y = (X[:, 0] * 0.5 - X[:, 1] + X[:, min(2, cols - 1)] ** 2).astype(np.float32)
    y = np.nan_to_num(y)
    base_params = {
        "max_depth": 5 if cols == 700 else 6,
        "max_bin": 32,
        "base_score": 0.2,
        "seed": 77,
        "colsample_bytree": bytree,
        "colsample_bylevel": bylevel,
        "colsample_bynode": bynode,
    }
    results = []
    profiles = []
    for enabled in [0, 1]:
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.Booster({**base_params, "metal_gpu_tree": enabled}, [dtrain])
        run_stats = []
        for it in range(3):
            model.update(dtrain, it)
            run_stats.append(profile(model))
        results.append(snapshot(model, [dtrain, xgb.DMatrix(X)]))
        profiles.append(run_stats)
    assert_identical(results[0], results[1])
    assert all(s["tuning"]["gpu_tree"] == 0 for s in profiles[0])
    assert all(s["tuning"]["gpu_tree"] == 1 for s in profiles[1])


@pytest.mark.parametrize("num_class,parallel,fusion", [
    (4, 1, 1),
    (4, 1, 0),
    (1, 4, 1),
    (3, 3, 1),
])
def test_multi_tree_batching_and_multiclass_deferred_prediction(num_class, parallel, fusion):
    rng = np.random.default_rng(4242)
    X = rng.normal(size=(600, 12)).astype(np.float32)
    if num_class > 1:
        y = rng.integers(0, num_class, size=600).astype(np.float32)
        obj_params = {"objective": "multi:softprob", "num_class": num_class}
    else:
        y = (X[:, 0] - X[:, 1]).astype(np.float32)
        obj_params = {"objective": "reg:squarederror"}

    base = {
        **obj_params,
        "num_parallel_tree": parallel,
        "max_depth": 5,
        "max_bin": 32,
        "seed": 17,
        "metal_gradient_root_fusion": fusion,
    }
    outputs = []
    stats_per_mode = []
    for batch in [0, 1]:
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.Booster({**base, "metal_batch_trees": batch}, [dtrain])
        run_stats = []
        for it in range(4):
            model.update(dtrain, it)
            run_stats.append(profile(model))
        outputs.append(snapshot(model, [dtrain, xgb.DMatrix(X)]))
        stats_per_mode.append(run_stats)

    assert_identical(outputs[0], outputs[1])
    assert all(s["tuning"]["batch_trees"] == 0 for s in stats_per_mode[0])
    assert all(s["tuning"]["batch_trees"] == 1 for s in stats_per_mode[1])
    trees_per_round = num_class * parallel
    # Batching collapses K per-tree waits into 1 shared wait per round.
    saved = stats_per_mode[0][-1]["command_buffers"] - stats_per_mode[1][-1]["command_buffers"]
    assert saved == 4 * (trees_per_round - 1)
    if num_class > 1 and parallel == 1 and fusion == 1:
        # Unobserved rounds 1, 2, 3 execute in exactly 1 command buffer per round!
        deltas = [stats_per_mode[1][i]["command_buffers"] - stats_per_mode[1][i - 1]["command_buffers"]
                  for i in range(1, 4)]
        assert deltas == [1, 1, 1]


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "multi:softprob"])
@pytest.mark.parametrize("missing,present", [(False, 0), (True, 0), (True, 1)])
def test_certified_i32_histograms_exact_parity(objective, missing, present):
    rng = np.random.default_rng(7711)
    X = rng.normal(size=(1500, 16)).astype(np.float32)
    if missing:
        X[rng.random(X.shape) < 0.2] = np.nan
    if objective == "multi:softprob":
        y = rng.integers(0, 3, size=1500).astype(np.float32)
        extra = {"num_class": 3}
    elif objective == "binary:logistic":
        y = (X[:, 0] > 0).astype(np.float32)
        extra = {}
    else:
        y = (X[:, 0] * 1.3 - 0.7).astype(np.float32)
        y = np.nan_to_num(y)
        extra = {}

    outputs = []
    for i32 in [0, 1]:
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": objective,
            **extra,
            "max_depth": 6,
            "max_bin": 128,
            "seed": 31,
            "metal_histogram_rows": 512,
            "metal_histogram_present": present,
            "metal_histogram_i32": i32,
        }
        model = xgb.Booster(params, [dtrain])
        for it in range(3):
            model.update(dtrain, it)
            assert profile(model)["tuning"]["histogram_i32"] == i32
        outputs.append(snapshot(model, [dtrain]))
    assert_identical(outputs[0], outputs[1])


@pytest.mark.parametrize("bins,packed,missing,present", [
    (64, 16, False, 0),
    (128, 16, False, 0),
    (256, 16, False, 0),
    (256, 4, False, 0),
    (256, 0, False, 0),
    (255, 16, True, 0),
    (255, 16, True, 1),
    (256, 0, True, 1),
])
def test_w8_feature_grouping_and_extreme_gradient_subpass_fallback(bins, packed, missing, present):
    rng = np.random.default_rng(8808)
    X = rng.normal(size=(1029, 37)).astype(np.float32)
    if missing:
        X[rng.random(X.shape) < 0.25] = np.nan
    y = np.nan_to_num(X[:, 0] - 0.5 * X[:, 8] + X[:, 19]).astype(np.float32)

    outputs = []
    for fpg in [4, 8]:
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "max_depth": 6,
            "max_bin": bins,
            "seed": 53,
            "metal_features_per_group": fpg,
            "metal_histogram_packed_bins": packed,
            "metal_histogram_present": present,
            "metal_histogram_rows": 512,
        }
        model = xgb.Booster(params, [dtrain])
        for it in range(3):
            model.update(dtrain, it)
            assert profile(model)["tuning"]["features_per_group"] == fpg
        outputs.append(snapshot(model, [dtrain]))
    assert_identical(outputs[0], outputs[1])

    # Also test custom gradients with wide dynamic range (shift=0, >31-bit sum)
    # at 256 bins to exercise the 2-subpass 4+4 fallback inside w8!
    if bins == 256 and not missing:
        g_wide = np.ones(len(y), dtype=np.float32)
        g_wide[0] = 2.0 ** 40
        g_wide[1] = -(2.0 ** 40)
        g_wide[2] = 2.0 ** -10
        h_wide = np.linspace(0.5, 2.5, len(y), dtype=np.float32)
        wide_outputs = []
        for fpg in [4, 8]:
            dtrain = xgb.DMatrix(X, label=y)
            model = xgb.Booster({
                "max_depth": 4,
                "max_bin": 256,
                "seed": 53,
                "metal_features_per_group": fpg,
                "metal_histogram_packed_bins": packed,
                "metal_histogram_rows": 1024,
            }, [dtrain])
            model.update(dtrain, 0, fobj=lambda pred, mat: (g_wide, h_wide))
            assert profile(model)["tuning"]["features_per_group"] == fpg
            wide_outputs.append(snapshot(model, [dtrain]))
        assert_identical(wide_outputs[0], wide_outputs[1])


def test_w8_selected_by_default_on_medium_wide_dense_matrices():
    rng = np.random.default_rng(1999)
    X = rng.normal(size=(16384, 32)).astype(np.float32)
    y = (X[:, 0] + X[:, 1]).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.Booster({"max_depth": 4, "max_bin": 128}, [dtrain])
    model.update(dtrain, 0)
    assert profile(model)["tuning"]["features_per_group"] == 8
