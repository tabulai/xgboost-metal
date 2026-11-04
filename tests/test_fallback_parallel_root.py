"""Default fallback root scoring preserves masks, constraints, and exact outputs."""
import numpy as np
import pytest
import xgboost as xgb

from test_fast_scalar_weight import tree
from test_gpu_tree import assert_identical, profile, snapshot
from test_histogram_missing import inputs, layout
from test_sampling import compare as compare_reference
from test_sampling import dataset as reference_dataset
from test_split_subtraction_fusion import fit


def compare(X, y, *, params=None, expected_active=1, **kwargs):
    options = {"metal_gpu_tree": 0, **(params or {})}
    expected, controls = fit(X, y, 1, params={**options, "metal_fallback_parallel_root": 0}, **kwargs)
    actual, stats = fit(X, y, 1, params={**options, "metal_fallback_parallel_root": 1}, **kwargs)
    default, default_stats = fit(X, y, 1, params=options, **kwargs)
    assert_identical(expected, actual)
    assert_identical(expected, default)
    for trace, active in ((controls, 0), (stats, expected_active), (default_stats, expected_active)):
        assert trace
        assert all(item["tuning"]["gpu_tree"] == 0 for item in trace)
        assert all(item["tuning"]["fallback_parallel_root"] == active for item in trace)
        assert all(item["tuning"]["parallel_root"] == active for item in trace)
    return actual, stats


FALLBACK_MODES = [
    pytest.param({}, id="explicit-host"),
    pytest.param({"metal_gpu_tree": 0, "monotone_constraints": "(1,0,-1,0,0,0,0)"}, id="monotone"),
    pytest.param({"metal_gpu_tree": 1, "interaction_constraints": "[[0,1,2],[3,4,5,6]]"}, id="interaction"),
    pytest.param({"metal_gpu_tree": 0, "colsample_bytree": 0.7, "colsample_bylevel": 0.6,
                  "seed_per_iteration": True}, id="tree-level-masks"),
    pytest.param({"metal_gpu_tree": 0, "colsample_bynode": 0.6}, id="node-masks"),
    pytest.param({"metal_gpu_tree": 1, "grow_policy": "lossguide", "max_depth": 0,
                  "max_leaves": 9}, id="lossguide"),
    pytest.param({"metal_gpu_tree": 0, "subsample": 0.63}, id="uniform-sampling"),
    pytest.param({"metal_gpu_tree": 0, "subsample": 0.63,
                  "sampling_method": "gradient_based"}, id="gradient-sampling"),
    pytest.param({"metal_gpu_tree": 1, "max_depth": 1}, id="stump"),
]


@pytest.mark.parametrize("params", FALLBACK_MODES)
def test_fallback_constraints_sampling_and_growth(params):
    X, y = inputs(rows=769, density=0.65)
    compare(X, y, bins=256, params=params, rounds=3,
            metadata={"weight": np.linspace(0.1, 1.7, len(y), dtype=np.float32),
                      "feature_weights": np.arange(1, 8, dtype=np.float32)})


def test_sampled_multiclass_parallel_trees_keep_compact_group_index():
    X, _ = inputs(rows=513, density=0.65)
    y = (np.arange(len(X)) % 3).astype(np.float32)
    compare(X, y, bins=128, rounds=2,
            params={"metal_gpu_tree": 0, "objective": "multi:softprob", "num_class": 3,
                    "num_parallel_tree": 2, "subsample": 0.63, "seed_per_iteration": True})


@pytest.mark.parametrize("bins,kind,quantile", [
    (2, "dense", False), (31, "missing", True), (32, "csr", False),
    (33, "missing", False), (128, "csc", False), (256, "missing", True),
])
def test_bin_layout_and_quantile_boundaries(bins, kind, quantile):
    X, y = inputs(rows=513, density=0.4)
    X = np.nan_to_num(X) if kind == "dense" else X if kind == "missing" else layout(X, kind)
    compare(X, y, bins=bins, quantile=quantile, rounds=2)


@pytest.mark.parametrize("case", ["large_bins", "tiled", "categorical", "vector", "serial",
                                  "root_disabled", "original_gain", "fast_weight"])
def test_unsupported_paths_and_explicit_controls(case):
    X, y = inputs(rows=513, density=0.8)
    params, metadata, bins = {}, {}, 128
    if case == "large_bins":
        bins = 257
    elif case == "tiled":
        params["metal_scalar_split_bin_tile"] = 32
    elif case == "categorical":
        present = ~np.isnan(X[:, 0])
        X[present, 0] = np.arange(np.count_nonzero(present)) % 5
        metadata["feature_types"] = ["c"] + ["q"] * 6
    elif case == "vector":
        y = np.column_stack([y, -y])
        params["multi_strategy"] = "multi_output_tree"
    elif case == "serial":
        params["metal_parallel_splits"] = 0
    elif case == "root_disabled":
        params["metal_parallel_root"] = 0
    elif case == "original_gain":
        params["metal_fast_gain"] = 0
    else:
        params["metal_fast_weight"] = 1
    compare(X, y, bins=bins, params=params, metadata=metadata, expected_active=0, rounds=2)


@pytest.mark.parametrize("default_left", [0, 1])
def test_missing_direction_and_feature_ties(default_left):
    # Two identical features also require the original lower-feature tie winner.
    ref = xgb.QuantileDMatrix(np.tile(np.arange(128, dtype=np.float32)[:, None], (1, 2)), max_bin=128)
    X = np.tile(np.repeat(np.asarray([16, 100, np.nan], np.float32), 64)[:, None], (1, 2))
    y = np.repeat(np.asarray([-1, 1, -1 if default_left else 1], np.float32), 64)
    actual, _ = compare(X, y, bins=128, quantile=True, metadata={"ref": ref}, rounds=1,
                        params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0})
    root = tree(actual)
    assert len(root["base_weights"]) == 3
    assert root["default_left"][0] == default_left
    assert root["split_indices"][0] == 0


def test_cancellation_zero_hessians_and_duplicate_features():
    X, y = inputs(rows=513, density=0.4)
    X[:, 1] = X[:, 0]
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.resize(np.asarray([1, 2, 1, 0, 2**-20], np.float32), len(X))
    compare(X, y, bins=256, rounds=2, params={"min_child_weight": 0, "metal_features_per_group": 4},
            custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("case", ["empty", "all_missing", "zero_weight"])
def test_invalid_roots_keep_initialized_winners(case):
    X, y = inputs(rows=257)
    metadata = {}
    if case == "empty":
        X, y = X[:0], y[:0]
    elif case == "all_missing":
        X.fill(np.nan)
    else:
        metadata["weight"] = np.zeros_like(y)
    actual, _ = compare(X, y, bins=256, metadata=metadata, rounds=2)
    assert len(tree(actual)["base_weights"]) == 1


@pytest.mark.parametrize("params", [
    pytest.param({"monotone_constraints": "(1,0,0,0,0)"}, id="monotone"),
    pytest.param({"interaction_constraints": "[[0,1],[2,3,4]]"}, id="interaction"),
    pytest.param({"colsample_bytree": 0.8, "colsample_bylevel": 0.75, "colsample_bynode": 0.7,
                  "seed_per_iteration": True}, id="column-masks"),
    pytest.param({"grow_policy": "lossguide", "max_depth": 0, "max_leaves": 7}, id="lossguide"),
    pytest.param({"subsample": 0.63, "sampling_method": "gradient_based"}, id="gradient-sampling"),
])
def test_independent_cpu_reference(reference, params):
    X, y = reference_dataset(rows=257)
    options = {"max_depth": 3, "max_bin": 32, "seed": 239, "base_score": 0.25, "eta": 0.2, **params}
    metadata = {"weight": np.linspace(0.1, 1.7, len(y), dtype=np.float32),
                "feature_weights": np.linspace(0.2, 1, X.shape[1], dtype=np.float32)}
    expected, files = reference(X, y, options, rounds=3, **metadata)
    data = xgb.DMatrix(X, label=y, **metadata)
    model = xgb.Booster({**options, "metal_gpu_tree": 0, "metal_fallback_parallel_root": 1}, [data])
    for iteration in range(3):
        model.update(data, iteration)
        assert profile(model)["tuning"]["fallback_parallel_root"] == 1
    # Preserve the repository's CPU structural/RNG checks and established float tolerances.
    compare_reference(model, data, expected, files)


def test_workspace_reset_and_runtime_switches():
    X, y = inputs(rows=769, density=0.5)
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=4, max_bin=256, base_score=0, seed=72, metal_gpu_tree=0))
        trace = []
        for iteration, rows in enumerate([257, 769, 0, 513, 769]):
            selected = 0 if iteration == 3 else enabled
            model.set_param("metal_fallback_parallel_root", selected)
            if iteration == 2:
                model.reset()
            values = np.nan_to_num(X[:rows]) if iteration % 2 else X[:rows]
            data = xgb.DMatrix(values, label=y[:rows])
            model.update(data, iteration)
            assert profile(model)["tuning"]["fallback_parallel_root"] == selected
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


def test_gpu_controller_keeps_its_existing_root_path():
    X, y = inputs(rows=513, density=0.7)
    traces = [fit(X, y, 1, bins=128, rounds=2,
                  params={} if enabled is None else {"metal_fallback_parallel_root": enabled})
              for enabled in [None, 0, 1]]
    for actual, _ in traces[1:]:
        assert_identical(traces[0][0], actual)
    for _, stats in traces:
        assert all(item["tuning"]["gpu_tree"] == 1 for item in stats)
        assert all(item["tuning"]["parallel_root"] == 1 for item in stats)
        assert all(item["tuning"]["fallback_parallel_root"] == 0 for item in stats)
