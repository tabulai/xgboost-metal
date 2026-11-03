"""Two-ended single-pass partitioning preserves child row sets and exact trees."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, inputs, profile, snapshot
from test_gpu_tree_deep import document_tree, fit_frontier
from test_histogram_missing import inputs as sparse_inputs
from test_histogram_missing import layout
from test_sampling import compare as compare_reference
from test_sampling import dataset as reference_dataset


def train(X, y, enabled, *, params=None, metadata=None, rounds=2, quantile=False, bins=32,
          custom=None):
    options = dict(max_depth=6, max_bin=bins, base_score=0, seed=59, metal_gpu_tree=1,
                   metal_partition_single_pass=enabled)
    options.update(params or {})
    metadata = metadata or {}
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = constructor(X, label=y, **metadata, **({"max_bin": bins} if quantile else {}))
    model = xgb.Booster(options, [data])
    stats = []
    for iteration in range(rounds):
        model.update(data, iteration, fobj=custom)
        stats.append(profile(model))
    query_metadata = {key: value for key, value in metadata.items()
                      if key in ("missing", "feature_types", "base_margin")}
    return snapshot(model, [data, xgb.DMatrix(X, **query_metadata)]), stats


def compare(X, y, *, expected_active=1, **kwargs):
    expected, controls = train(X, y, 0, **kwargs)
    actual, stats = train(X, y, 1, **kwargs)
    assert_identical(expected, actual)
    depth = (kwargs.get("params") or {}).get("max_depth", 6)
    coalesce = (kwargs.get("params") or {}).get("metal_coalesce_encoders", 0)
    for old, new in zip(controls, stats):
        assert old["tuning"]["partition_single_pass"] == 0
        assert new["tuning"]["partition_single_pass"] == expected_active
        assert old["command_buffers"] == new["command_buffers"]
        # Flags, scan and scatter become one scatter plus one offset pass on
        # every nonroot level of the pre-encoded loop, including empty ones.
        if expected_active and not coalesce and new["tuning"]["gpu_tree"]:
            assert old["tuning"]["tree_encoders"] - new["tuning"]["tree_encoders"] == depth - 1
        else:
            assert old["tuning"]["tree_encoders"] == new["tuning"]["tree_encoders"]
    return actual, stats


@pytest.mark.parametrize("case", ["regression", "missing", "ties", "constant", "empty",
                                  "weighted", "subnormal_features", "positive_infinity_missing",
                                  "negative_infinity_missing"])
@pytest.mark.parametrize("depth", [2, 6, 10])
def test_cases_and_depths(case, depth):
    X, y, metadata, params = inputs(case)
    compare(X, y, params={**params, "max_depth": depth, "metal_coalesce_encoders": 0},
            metadata=metadata)


@pytest.mark.parametrize("rows", [1, 2, 255, 256, 257, 511, 513, 4099])
@pytest.mark.parametrize("plan", [0, 1])
def test_tile_boundaries_with_both_planners(rows, plan):
    # Segment and tile edges: partial final tiles, single-tile and many-tile
    # parents, with each planner zeroing the atomic child cursors.
    X, y, metadata, _ = inputs("missing", rows=rows)
    compare(X, y, params={"max_depth": 7, "metal_parallel_partition_plan": plan,
                          "metal_coalesce_encoders": 0}, metadata=metadata)


@pytest.mark.parametrize("side", ["left", "right"])
def test_one_sided_tiles_and_empty_children(side):
    # Missing rows follow the default direction; whole tiles then contribute
    # to only one cursor, and a child can receive no rows at all.
    rng = np.random.default_rng(19)
    X = rng.normal(size=(3000, 4)).astype(np.float32)
    X[:1800, 0] = np.nan
    y = np.where(np.isnan(X[:, 0]), 5.0 if side == "left" else -5.0, X[:, 0]).astype(np.float32)
    y += (0.1 * X[:, 1]).astype(np.float32)
    compare(X, y, params={"max_depth": 6, "min_child_weight": 0, "metal_coalesce_encoders": 0})


@pytest.mark.parametrize("kind", ["dense", "csr", "csc"])
@pytest.mark.parametrize("group,present", list(itertools.product([1, 2, 4], [0, 1])))
def test_sparse_layouts_and_histogram_variants(kind, group, present):
    X, y = sparse_inputs(rows=2051, density=0.4)
    compare(layout(X, kind), y, bins=64,
            params={"max_depth": 6, "metal_features_per_group": group,
                    "metal_histogram_present": present, "metal_coalesce_encoders": 0})


@pytest.mark.parametrize("histogram_rows", [256, 2048])
@pytest.mark.parametrize("compact", [0, 1])
def test_histogram_tiles_span_reordered_runs(histogram_rows, compact):
    # Small histogram tiles cover several partition runs in arbitrary order.
    X, y, metadata, _ = inputs("missing", rows=6151)
    compare(X, y, metadata=metadata,
            params={"max_depth": 8, "metal_histogram_rows": histogram_rows,
                    "metal_compact_splits": compact, "metal_coalesce_encoders": 0})


def test_repeated_fits_are_identical_despite_row_order():
    # Tile arrival order can differ between runs; models and predictions must not.
    rng = np.random.default_rng(907)
    X = rng.normal(size=(40000, 12)).astype(np.float32)
    y = (X[:, 0] * 2 - X[:, 1] ** 2 + np.sin(X[:, 2])).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    expected, _ = train(X, y, 0, params={"max_depth": 8}, bins=128, rounds=4)
    for _ in range(3):
        actual, stats = train(X, y, 1, params={"max_depth": 8}, bins=128, rounds=4)
        assert all(row["tuning"]["partition_single_pass"] == 1 for row in stats)
        assert_identical(expected, actual)


@pytest.mark.parametrize("quantile,pruned", list(itertools.product([False, True], repeat=2)))
def test_full_and_pruned_deep_frontiers(quantile, pruned):
    results = []
    for enabled in [0, 1]:
        model, _, result, info = fit_frontier(
            1, 10, quantile, pruned=pruned,
            params={"metal_partition_single_pass": enabled, "metal_coalesce_encoders": 0})
        assert info["tuning"]["partition_single_pass"] == enabled
        left = np.asarray(document_tree(model)["left_children"])
        np.testing.assert_array_equal(left[:511], 2 * np.arange(511) + 1)
        results.append(result)
    assert_identical(*results)


@pytest.mark.parametrize("coalesce,gradient_fusion,control_fusion",
                         list(itertools.product([0, 1], repeat=3)))
def test_encoder_gradient_and_control_variants(coalesce, gradient_fusion, control_fusion):
    X, y, metadata, _ = inputs("missing", rows=1543)
    compare(X, y, metadata=metadata,
            params={"metal_coalesce_encoders": coalesce, "metal_gradient_root_fusion": gradient_fusion,
                    "metal_tree_control_fusion": control_fusion})


@pytest.mark.parametrize("case", ["custom", "multiclass", "parallel", "bytree"])
def test_gradient_groups_and_sampling(case):
    X, y, metadata, params = inputs(case, rows=1029)
    custom = None
    if case == "custom":
        custom = lambda prediction, matrix: ((prediction - matrix.get_label()).astype(np.float32),
                                             np.ones_like(prediction, dtype=np.float32))
    compare(X, y, params={**params, "metal_coalesce_encoders": 0}, metadata=metadata, custom=custom)


@pytest.mark.parametrize("quantile", [False, True])
def test_quantile_matrices(quantile):
    X, y, metadata, _ = inputs("missing", rows=2311)
    compare(X, y, metadata=metadata, quantile=quantile, bins=256,
            params={"max_depth": 6, "metal_coalesce_encoders": 0})


@pytest.mark.parametrize("params", [{"metal_gpu_tree": 0}, {"max_depth": 1}, {"max_depth": 11},
                                    {"grow_policy": "lossguide"}, {"subsample": 0.7}])
def test_original_driver_keeps_the_stable_partition(params):
    X, y, metadata, _ = inputs("missing")
    compare(X, y, metadata=metadata, params=params, expected_active=0)


def test_workspace_reuse_and_runtime_toggles():
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=6, max_bin=64, base_score=0, seed=29, metal_gpu_tree=1))
        trace = []
        for iteration, (rows, depth) in enumerate([(4099, 8), (513, 3), (0, 6), (2049, 10), (257, 2)]):
            X, y, _, _ = inputs("missing", rows=rows)
            data = xgb.DMatrix(X, label=y)
            active = int(enabled and iteration != 1)
            model.set_param({"max_depth": depth, "metal_partition_single_pass": active})
            model.update(data, iteration)
            assert profile(model)["tuning"]["partition_single_pass"] == active
            trace.append(snapshot(model, [data]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


@pytest.mark.parametrize("depth", [4, 8])
def test_independent_cpu_reference(reference, depth):
    X, y = reference_dataset(rows=2053)
    options = {"max_depth": depth, "max_bin": 64, "seed": 241, "base_score": 0.5, "eta": 0.3}
    expected, files = reference(X, y, options, rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.Booster({**options, "metal_gpu_tree": 1, "metal_partition_single_pass": 1}, [data])
    for iteration in range(3):
        model.update(data, iteration)
        assert profile(model)["tuning"]["partition_single_pass"] == 1
    compare_reference(model, data, expected, files)


def compare_bins(X, y, *, params=None, expected_active=1, **kwargs):
    params = params or {}
    expected, controls = train(X, y, 1, params={**params, "metal_partition_bin_predicate": 0}, **kwargs)
    actual, stats = train(X, y, 1, params={**params, "metal_partition_bin_predicate": 1}, **kwargs)
    assert_identical(expected, actual)
    for old, new in zip(controls, stats):
        assert old["tuning"]["partition_bin_predicate"] == 0
        assert new["tuning"]["partition_bin_predicate"] == expected_active
        assert old["tuning"]["tree_encoders"] == new["tuning"]["tree_encoders"]
    return actual, stats


@pytest.mark.parametrize("case", ["regression", "missing", "ties", "subnormal_features",
                                  "positive_infinity_missing", "negative_infinity_missing", "weighted"])
@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize("bins", [2, 16, 256])
def test_bin_predicate_matches_float_thresholds(case, quantile, bins):
    if quantile and "infinity" in case:
        pytest.skip("QuantileDMatrix rejects infinite feature values.")
    X, y, metadata, params = inputs(case, rows=1029)
    compare_bins(X, y, params={**params, "max_depth": 6}, metadata=metadata, quantile=quantile, bins=bins)


@pytest.mark.parametrize("quantile", [False, True])
def test_bin_predicate_cut_boundaries_signed_zero_and_subnormals(quantile):
    # Few distinct values put thresholds on the first, interior and last cuts
    # and at feature minima; values equal to cuts land on both sides.
    tiny = np.nextafter(np.float32(0), np.float32(1))
    columns = [
        np.asarray([-0.0, 0.0, tiny, -tiny, 4 * tiny, 1.0], np.float32),
        np.asarray([-3e38, -1.0, 0.0, 1.0, 3e38, np.nan], np.float32),
        np.asarray([3.0, 3.0, 3.0, 7.0], np.float32),
        np.asarray([1.0, 2.0], np.float32),
    ]
    rows = 2311
    X = np.column_stack([np.resize(column, rows) for column in columns]).astype(np.float32)
    rng = np.random.default_rng(5)
    for column in range(X.shape[1]):
        X[:, column] = X[rng.permutation(rows), column]
    y = (X[:, 0] > 0).astype(np.float32) + np.sign(np.nan_to_num(X[:, 1], nan=0.5)) \
        + (X[:, 2] > 3) - 0.5 * X[:, 3]
    for bins in [2, 3, 4, 256]:
        compare_bins(X, y.astype(np.float32), quantile=quantile, bins=bins,
                     params={"max_depth": 8, "min_child_weight": 0, "reg_lambda": 0})


def test_bin_predicate_requires_single_pass_and_parallel_planning():
    X, y, metadata, _ = inputs("missing")
    for params in [{"metal_partition_single_pass": 0}, {"metal_parallel_partition_plan": 0}]:
        expected, controls = train(X, y, 1, params={**params, "metal_partition_bin_predicate": 0},
                                   metadata=metadata)
        actual, stats = train(X, y, 1, params={**params, "metal_partition_bin_predicate": 1},
                              metadata=metadata)
        assert_identical(expected, actual)
        assert all(row["tuning"]["partition_bin_predicate"] == 0 for row in controls + stats)


@pytest.mark.parametrize("quantile", [False, True])
def test_feature_type_change_after_quantization(quantile):
    # After a type change, an ordinary matrix partitions a recoded copy of its
    # data, so bins are not used. A QuantileDMatrix re-decodes its own data
    # from the same bins, which re-quantize to those bins, so bins remain exact.
    rows = 1031
    rng = np.random.default_rng(31)
    X = rng.normal(size=(rows, 4)).astype(np.float32)
    X[:, 0] = np.arange(rows) % 6
    y = (X[:, 0] * 0.5 + X[:, 1] - X[:, 2] * X[:, 3]).astype(np.float32)
    results = []
    for enabled in [0, 1]:
        constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
        data = constructor(X, label=y, feature_types=["c", "q", "q", "q"], enable_categorical=True,
                           **({"max_bin": 32} if quantile else {}))
        model = xgb.Booster(dict(max_depth=5, max_bin=32, base_score=0, seed=7, metal_gpu_tree=1,
                                 metal_partition_bin_predicate=enabled), [data])
        model.update(data, 0)
        assert profile(model)["tuning"]["gpu_tree"] == 0
        data.feature_types = ["q", "q", "q", "q"]
        trace = []
        for iteration in range(1, 3):
            model.update(data, iteration)
            tuning = profile(model)["tuning"]
            assert tuning["gpu_tree"] == 1
            assert tuning["partition_bin_predicate"] == int(enabled and quantile)
            trace.append(snapshot(model, [data]))
        results.append(trace)
    for expected, actual in zip(*results):
        assert_identical(expected, actual)
