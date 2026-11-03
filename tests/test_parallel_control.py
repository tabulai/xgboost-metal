"""Parallel GPU-driver selection, expansion and planning retain exact trees."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, inputs, profile, snapshot
from test_gpu_tree_deep import document_tree, fit_frontier
from test_sampling import compare as compare_reference
from test_sampling import dataset as reference_dataset

SWITCHES = ["metal_parallel_select", "metal_parallel_tree_control", "metal_parallel_partition_plan"]
ALL = SWITCHES + ["metal_deferred_parent_statistics", "metal_partition_single_pass",
                  "metal_partition_bin_predicate"]


def field(switch):
    return switch[len("metal_"):]


def train(X, y, params, *, metadata=None, rounds=2, quantile=False, bins=32, custom=None):
    options = dict(max_depth=6, max_bin=bins, base_score=0, seed=43, metal_gpu_tree=1)
    options.update(params)
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


def compare(X, y, switch, *, params=None, expected_active=1, **kwargs):
    params = dict(params or {})
    if switch == "metal_parallel_select":
        # Deferral needs parallel selection; compare the selection kernels
        # alone here, with both arms computing statistics per level.
        params.setdefault("metal_deferred_parent_statistics", 0)
    expected, controls = train(X, y, {**params, switch: 0}, **kwargs)
    actual, stats = train(X, y, {**params, switch: 1}, **kwargs)
    assert_identical(expected, actual)
    for old, new in zip(controls, stats):
        assert old["tuning"][field(switch)] == 0
        assert new["tuning"][field(switch)] == expected_active
        # These kernels replace serial loops without adding or removing work.
        assert old["command_buffers"] == new["command_buffers"]
        assert old["tuning"]["tree_encoders"] == new["tuning"]["tree_encoders"]
    return actual, stats


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("depth", [2, 6, 10])
def test_defaults_match_explicit_controls(missing, depth):
    X, y, metadata, _ = inputs("missing" if missing else "regression", rows=1031)
    results = []
    for value in [None, 1, 0]:
        params = {"max_depth": depth}
        if value is not None:
            params.update({switch: value for switch in ALL})
        result, stats = train(X, y, params, metadata=metadata)
        for row in stats:
            assert row["tuning"]["gpu_tree"] == 1
            for switch in ALL:
                expected = 0 if value == 0 else 1
                if switch == "metal_deferred_parent_statistics" and value is None:
                    # Deferral starts at two per-level selections; a dense
                    # root uses select_dense_root_split instead.
                    expected = int(depth - (not missing) >= 2)
                assert row["tuning"][field(switch)] == expected
        results.append(result)
    assert_identical(results[0], results[1])
    assert_identical(results[0], results[2])


@pytest.mark.parametrize("switch", SWITCHES)
@pytest.mark.parametrize("depth", [2, 6, 9, 10])
@pytest.mark.parametrize("case", ["regression", "missing", "ties", "constant", "empty"])
def test_depths_and_frontier_shapes(switch, depth, case):
    X, y, metadata, params = inputs(case)
    compare(X, y, switch, params={**params, "max_depth": depth}, metadata=metadata)


@pytest.mark.parametrize("switch", SWITCHES)
@pytest.mark.parametrize("case", ["custom", "weighted", "multiclass", "parallel", "bytree",
                                  "subnormal_features", "positive_infinity_missing",
                                  "negative_infinity_missing"])
def test_objectives_sampling_and_numeric_edges(switch, case):
    X, y, metadata, params = inputs(case)
    custom = None
    if case == "custom":
        custom = lambda prediction, matrix: ((prediction - matrix.get_label()).astype(np.float32),
                                             np.ones_like(prediction, dtype=np.float32))
    compare(X, y, switch, params=params, metadata=metadata, custom=custom)


def duplicated_features(columns, copies, rows=1200):
    # Exact copies give identical candidates, so every node has gain ties
    # across lanes (features f and f + 32) and within one lane's stride.
    rng = np.random.default_rng(3301)
    base = rng.normal(size=(rows, 3)).astype(np.float32)
    X = rng.normal(size=(rows, columns)).astype(np.float32) * np.float32(1e-3)
    for source, positions in enumerate(copies):
        for position in positions:
            if position < columns:
                X[:, position] = base[:, source]
    y = (2 * base[:, 0] - base[:, 1] * base[:, 2] + np.sin(base[:, 1])).astype(np.float32)
    return X, y


@pytest.mark.parametrize("columns", [31, 32, 33, 64, 65, 97])
@pytest.mark.parametrize("missing", [False, True])
def test_selection_ties_choose_the_earliest_feature_across_lanes(columns, missing):
    copies = [[c for c in (30, 62, 94, 7, 39, 71) if c < columns],
              [c for c in (1, 33, 65, 96) if c < columns],
              [c for c in (12, 44, 76) if c < columns]]
    X, y = duplicated_features(columns, copies)
    if missing:
        X[::5, copies[0][0]] = np.nan
        X[::5, copies[0][1:]] = np.nan
    actual, _ = compare(X, y, "metal_parallel_select", params={"max_depth": 6}, bins=64, rounds=1)
    tree = document_tree(xgb.Booster(model_file=bytearray(actual["model"])))
    used = set(np.asarray(tree["split_indices"])[np.asarray(tree["left_children"]) >= 0].tolist())
    # Only the earliest copy of any duplicated input can ever be selected.
    for positions in copies:
        assert not (used & set(sorted(positions)[1:]))
    assert min(copies[0]) in used


def compare_deferred(X, y, *, params=None, expected_active=1, **kwargs):
    params = params or {}
    expected, controls = train(X, y, {**params, "metal_deferred_parent_statistics": 0}, **kwargs)
    actual, stats = train(X, y, {**params, "metal_deferred_parent_statistics": 1}, **kwargs)
    assert_identical(expected, actual)
    coalesce = params.get("metal_coalesce_encoders", 0)
    for old, new in zip(controls, stats):
        assert old["tuning"]["deferred_parent_statistics"] == 0
        assert new["tuning"]["deferred_parent_statistics"] == expected_active
        assert old["command_buffers"] == new["command_buffers"]
        # One pass per tree computes every expanded node's applied statistics.
        added = int(bool(expected_active) and not coalesce and bool(new["tuning"]["gpu_tree"]))
        assert new["tuning"]["tree_encoders"] - old["tuning"]["tree_encoders"] == added
    return actual, stats


@pytest.mark.parametrize("case", ["regression", "missing", "ties", "constant", "empty", "weighted",
                                  "multiclass", "subnormal_features"])
@pytest.mark.parametrize("depth", [2, 6, 10])
@pytest.mark.parametrize("coalesce", [0, 1])
def test_deferred_parent_statistics_dense_and_missing_roots(case, depth, coalesce):
    # Dense roots keep select_dense_root_split's ordered-sum statistics;
    # missing-valued roots are deferred like every descendant.
    X, y, metadata, params = inputs(case, rows=1031)
    compare_deferred(X, y, metadata=metadata,
                     params={**params, "max_depth": depth, "metal_coalesce_encoders": coalesce})


@pytest.mark.parametrize("params", [
    {"reg_alpha": 0.75}, {"reg_alpha": 1e-30}, {"max_delta_step": 0.05}, {"reg_lambda": 0},
    {"reg_lambda": 0, "min_child_weight": 0}, {"reg_lambda": 3e-38, "reg_alpha": 2.5, "max_delta_step": 0.3},
])
@pytest.mark.parametrize("missing", [False, True])
def test_deferred_parent_statistics_regularization_paths(params, missing):
    X, y, metadata, _ = inputs("missing" if missing else "regression", rows=1031)
    compare_deferred(X, y, metadata=metadata, params={**params, "max_depth": 7})


@pytest.mark.parametrize("group", [1, 4])
def test_deferred_parent_statistics_extreme_gradients(group):
    # Large, tiny and cancelling gradients exercise binary64 rounding and the
    # zero/negative Hessian guard in the recomputed parent weight.
    X, y, metadata, _ = inputs("missing", rows=2053)
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.resize(np.asarray([1, 2, 1, 0, 2**-20], np.float32), len(X))
    compare_deferred(X, y, metadata=metadata, rounds=2,
                     params={"max_depth": 6, "min_child_weight": 0, "metal_features_per_group": group},
                     custom=lambda prediction, matrix: (gradient, hessian))


def test_deferred_parent_statistics_requires_parallel_selection():
    X, y, metadata, _ = inputs("missing")
    compare_deferred(X, y, metadata=metadata, expected_active=0, params={"metal_parallel_select": 0})


@pytest.mark.parametrize("quantile,pruned", list(itertools.product([False, True], repeat=2)))
@pytest.mark.parametrize("depth", [9, 10])
def test_full_and_pruned_512_slot_frontiers(quantile, pruned, depth):
    results = []
    for enabled in [0, 1]:
        model, _, result, info = fit_frontier(
            1, depth, quantile, pruned=pruned,
            params={switch: enabled for switch in ALL} | {"metal_coalesce_encoders": 0})
        for switch in ALL:
            assert info["tuning"][field(switch)] == enabled
        assert info["tuning"]["gpu_tree_levels"] == depth
        left = np.asarray(document_tree(model)["left_children"])
        internal = 2 ** (depth - 1) - 1
        np.testing.assert_array_equal(left[:internal], 2 * np.arange(internal) + 1)
        if depth == 10 and pruned:
            assert 256 < np.count_nonzero(left[511:1023] >= 0) < 512
        results.append(result)
    assert_identical(*results)


@pytest.mark.parametrize("control_fusion", [0, 1])
@pytest.mark.parametrize("coalesce,gradient_fusion", list(itertools.product([0, 1], repeat=2)))
def test_encoder_gradient_and_control_fusion_variants(control_fusion, coalesce, gradient_fusion):
    X, y, metadata, _ = inputs("missing", rows=777)
    params = {"metal_tree_control_fusion": control_fusion, "metal_coalesce_encoders": coalesce,
              "metal_gradient_root_fusion": gradient_fusion}
    for switch in SWITCHES:
        active = int(switch != "metal_parallel_tree_control" or control_fusion)
        compare(X, y, switch, params=params, metadata=metadata, expected_active=active)


@pytest.mark.parametrize("parallel_splits,compact", [(0, 0), (1, 0), (1, 1)])
def test_serial_simd_and_compact_split_kernels(parallel_splits, compact):
    X, y, metadata, _ = inputs("missing", rows=901)
    params = {"metal_parallel_splits": parallel_splits, "metal_compact_splits": compact, "max_depth": 7}
    for switch in SWITCHES:
        compare(X, y, switch, params=params, metadata=metadata)


@pytest.mark.parametrize("workspace,persistent", list(itertools.product([0, 1], repeat=2)))
def test_stale_flags_workspace_reuse_and_runtime_toggles(workspace, persistent):
    # Lazy flag clearing must not observe flags left by a deeper earlier tree,
    # including across workspace reuse, resets and switch changes.
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=6, max_bin=32, base_score=0, seed=23, metal_gpu_tree=1,
                                 colsample_bytree=0.8, metal_tree_workspace=workspace,
                                 metal_tree_persistent_workspace=persistent))
        trace = []
        plan = [(1025, 10, False), (513, 3, False), (65, 6, False), (0, 6, True),
                (1025, 9, False), (257, 2, False), (1025, 10, False)]
        for iteration, (rows, depth, reset) in enumerate(plan):
            X, y, _, _ = inputs("missing" if iteration % 2 else "regression", rows=rows)
            data = xgb.DMatrix(X, label=y)
            active = int(enabled and iteration != 4)
            model.set_param({"max_depth": depth, **{switch: active for switch in ALL}})
            if reset:
                model.reset()
            model.update(data, iteration)
            tuning = profile(model)["tuning"]
            assert all(tuning[field(switch)] == active for switch in ALL)
            trace.append(snapshot(model, [data]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


@pytest.mark.parametrize("case,params", [
    ("regression", {"metal_gpu_tree": 0}),
    ("regression", {"max_depth": 1}),
    ("regression", {"max_depth": 11}),
    ("regression", {"max_leaves": 7}),
    ("regression", {"grow_policy": "lossguide"}),
    ("regression", {"subsample": 0.7}),
    ("regression", {"monotone_constraints": "(1,0,-1,0,0,0,0)"}),
    ("vector", {}),
    ("categorical", {}),
])
def test_original_driver_fallback_reports_no_parallel_control(case, params):
    X, y, metadata, case_params = inputs(case)
    for switch in ALL:
        expected, controls = train(X, y, {**case_params, **params, switch: 0}, metadata=metadata)
        actual, stats = train(X, y, {**case_params, **params, switch: 1}, metadata=metadata)
        assert_identical(expected, actual)
        assert all(row["tuning"]["gpu_tree"] == 0 for row in controls + stats)
        assert all(row["tuning"][field(switch)] == 0 for row in controls + stats)


@pytest.mark.parametrize("depth", [3, 6, 10])
def test_independent_cpu_reference(reference, depth):
    X, y = reference_dataset(rows=1031)
    options = {"max_depth": depth, "max_bin": 32, "seed": 239, "base_score": 0.25, "eta": 0.2}
    metadata = {"weight": np.linspace(0.1, 1.7, len(y), dtype=np.float32)}
    expected, files = reference(X, y, options, rounds=3, **metadata)
    data = xgb.DMatrix(X, label=y, **metadata)
    model = xgb.Booster({**options, "metal_gpu_tree": 1}, [data])
    for iteration in range(3):
        model.update(data, iteration)
        tuning = profile(model)["tuning"]
        assert tuning["gpu_tree"] == 1
        assert all(tuning[field(switch)] == 1 for switch in ALL)
    compare_reference(model, data, expected, files)
