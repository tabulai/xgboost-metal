"""Default host descendant scoring retains exact models, masks, and parent sums."""
import itertools
import json

import numpy as np
import pytest
import xgboost as xgb

from test_fast_scalar_weight import tree
from test_gpu_tree import assert_identical, profile, snapshot
from test_histogram_missing import inputs, layout
from test_sampling import compare as compare_reference
from test_sampling import dataset as reference_dataset
from test_split_subtraction_fusion import fit


def compare(X, y, *, params=None, expected_active=1, allow_root_only=False, **kwargs):
    options = {"metal_gpu_tree": 0, **(params or {})}
    expected, controls = fit(X, y, 1, params={**options, "metal_fallback_parallel_splits": 0}, **kwargs)
    actual, stats = fit(X, y, 1, params={**options, "metal_fallback_parallel_splits": 1}, **kwargs)
    assert_identical(expected, actual)
    assert len(controls) == len(stats) > 0
    active_by_round = [expected_active] * len(stats)
    if allow_root_only:
        # Column masks can leave a round with no valid root split. Prove this
        # from the control model, rather than accepting a missing dispatch on
        # a tree that actually evaluated descendants.
        assert expected_active == 1
        control_model = xgb.Booster(model_file=bytearray(expected["model"]))
        document = json.loads(control_model.save_raw(raw_format="json"))
        trees = document["learner"]["gradient_booster"]["model"]["trees"]
        assert len(trees) == len(stats)  # This exception is scalar, one tree per round.
        active_by_round = [int(len(item["left_children"]) > 1) for item in trees]
        assert any(active_by_round), "The fixture must exercise descendant scoring"
        for item, active in zip(trees, active_by_round):
            if not active:
                assert item["left_children"] == item["right_children"] == [-1]
    for control, candidate, active in zip(controls, stats, active_by_round):
        assert control["tuning"]["fallback_parallel_splits"] == 0
        assert candidate["tuning"]["fallback_parallel_splits"] == active
        assert control["tuning"]["gpu_tree"] == candidate["tuning"]["gpu_tree"] == 0
        assert control["tuning"]["fallback_parallel_root"] == candidate["tuning"]["fallback_parallel_root"]
        assert control["tuning"]["compact_splits"] == control["tuning"]["fused_score"] == 0
        assert candidate["tuning"]["terminal_histogram_elision"] == 0
        assert control["tuning"]["histogram_rebuilds"] == candidate["tuning"]["histogram_rebuilds"]
        assert control["command_buffers"] == candidate["command_buffers"]
        if active:
            assert candidate["tuning"]["split_threads"] == 256
            assert candidate["tuning"]["split_skip_empty_bins"] == int(bool(options.get("metal_skip_empty_bins", 1)))
            assert candidate["tuning"]["fast_gain"] == 1
            assert candidate["tuning"]["fast_stats"] == int(bool(options.get("metal_fast_stats", 1)))
            # The original host path skips empty bins only for missing input;
            # this also detects sparse layouts without densifying the matrix.
            missing = bool(control["tuning"]["split_skip_empty_bins"])
            compact = bool(options.get("metal_skip_empty_bins", 1)) and bool(
                options.get("metal_compact_splits", missing))
            score = compact and bool(options.get("metal_fast_stats", 1)) and bool(options.get("metal_fused_score", 1))
            assert candidate["tuning"]["compact_splits"] == int(compact)
            assert candidate["tuning"]["fused_score"] == int(score)
            if not options.get("metal_coalesce_encoders", 0):
                assert candidate["tuning"]["tree_encoders"] < control["tuning"]["tree_encoders"]
        else:
            assert candidate["tuning"]["compact_splits"] == candidate["tuning"]["fused_score"] == 0
    if options.get("max_cached_hist_node") == 1:
        assert any(item["tuning"]["histogram_rebuilds"] > 0 for item in controls)
    return actual, stats


@pytest.mark.parametrize("missing", [False, True])
def test_default_matches_explicit_descendant_controls(missing):
    X, y = inputs(rows=513, density=0.6)
    if not missing:
        # Distinct fill values keep about 256 cuts per feature, where the
        # default selects this path (zeros would leave too few cuts).
        X = np.where(np.isnan(X), np.random.default_rng(31).normal(size=X.shape), X).astype(np.float32)
    results = []
    for enabled in [0, 1, None]:
        options = {"metal_gpu_tree": 0, "max_depth": 8}
        if enabled is not None:
            options["metal_fallback_parallel_splits"] = enabled
        result, traces = fit(X, y, 1, bins=256, params=options, rounds=2)
        active = int(enabled != 0)
        for trace in traces:
            tuning = trace["tuning"]
            assert tuning["gpu_tree"] == tuning["terminal_histogram_elision"] == 0
            assert tuning["fallback_parallel_splits"] == active
            assert tuning["split_threads"] == (256 if active else 32)
            assert tuning["compact_splits"] == tuning["fused_score"] == int(active and missing)
        results.append(result)
    assert_identical(results[0], results[1])
    assert_identical(results[0], results[2])


@pytest.mark.parametrize("bins,kind,fast_stats", [
    (2, "dense", 0), (31, "csr", 1), (32, "missing", 0), (33, "missing", 1),
    (128, "csc", 0), (256, "dense", 1), (256, "missing", 1),
])
def test_bin_layout_and_statistics_variants(bins, kind, fast_stats):
    X, y = inputs(rows=769, density=0.45)
    X = np.nan_to_num(X) if kind == "dense" else X if kind == "missing" else layout(X, kind)
    compare(X, y, bins=bins, rounds=2, params={"metal_fast_stats": fast_stats})


@pytest.mark.parametrize("screening", [0, 1])
@pytest.mark.parametrize("compact,fast_stats,fused_score", list(itertools.product([0, 1], repeat=3)))
def test_compact_statistics_and_score_controls_match_serial(compact, fast_stats, fused_score, screening):
    X, y = inputs(rows=513, density=0.65)
    X = np.nan_to_num(X)
    options = {"metal_gpu_tree": 0, "metal_compact_splits": compact,
               "metal_fast_stats": fast_stats, "metal_fused_score": fused_score,
               "metal_terminal_histogram_elision": 1, "metal_split_screening": screening}
    actual, stats = compare(X, y, bins=128, rounds=2, params=options)
    assert all(t["tuning"]["split_screening"] == screening for t in stats)
    serial, traces = fit(X, y, 1, bins=128, rounds=2,
                         params={**options, "metal_parallel_splits": 0,
                                 "metal_fallback_parallel_splits": 1})
    assert_identical(serial, actual)
    assert all(t["tuning"]["fallback_parallel_splits"] == 0 for t in traces)
    assert all(t["tuning"]["compact_splits"] == t["tuning"]["fused_score"] == 0 for t in traces)


@pytest.mark.parametrize("missing,depth", list(itertools.product([False, True], [7, 8])))
def test_host_compact_default_follows_missing_input_at_both_depths(missing, depth):
    X, y = inputs(rows=513, density=0.65)
    if not missing:
        X = np.nan_to_num(X)
    _, stats = compare(X, y, bins=128, rounds=2, params={"max_depth": depth})
    expected = int(missing)
    assert all(t["tuning"]["compact_splits"] == t["tuning"]["fused_score"] == expected for t in stats)


@pytest.mark.parametrize("params", [
    pytest.param({"metal_gpu_tree": 1, "max_depth": 11}, id="deep"),
    pytest.param({"metal_gpu_tree": 0, "monotone_constraints": "(1,0,-1,0,0,0,0)"}, id="monotone"),
    pytest.param({"metal_gpu_tree": 1, "interaction_constraints": "[[0,1,2],[3,4,5,6]]"}, id="interaction"),
    pytest.param({"metal_gpu_tree": 0, "colsample_bytree": 0.8, "colsample_bylevel": 0.7,
                  "colsample_bynode": 0.6, "seed_per_iteration": True}, id="column-masks"),
    pytest.param({"metal_gpu_tree": 0, "subsample": 0.63}, id="uniform-sampling"),
    pytest.param({"metal_gpu_tree": 0, "subsample": 0.63, "sampling_method": "gradient_based"}, id="gradient-sampling"),
    pytest.param({"metal_gpu_tree": 1, "grow_policy": "lossguide", "max_depth": 0,
                  "max_leaves": 17, "max_cached_hist_node": 1}, id="lossguide-eviction"),
    pytest.param({"metal_gpu_tree": 1, "max_cached_hist_node": 1}, id="depthwise-eviction"),
    pytest.param({"metal_gpu_tree": 1, "metal_partition_scan": 0}, id="atomic-partition"),
    pytest.param({"metal_gpu_tree": 1, "metal_fused_partition": 0}, id="separate-partition"),
])
def test_host_constraints_sampling_and_cache_rebuilds(params):
    X, y = inputs(rows=769, density=0.65)
    compare(X, y, bins=256, rounds=3, params=params,
            allow_root_only="colsample_bynode" in params,
            metadata={"weight": np.linspace(0.1, 1.7, len(y), dtype=np.float32),
                      "feature_weights": np.arange(1, 8, dtype=np.float32)})


@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize("root_enabled,fallback_root", list(itertools.product([0, 1], repeat=2)))
def test_root_controls_are_independent(quantile, root_enabled, fallback_root):
    X, y = inputs(rows=513, density=0.5)
    compare(X, y, bins=128, quantile=quantile, rounds=2,
            params={"metal_parallel_root": root_enabled, "metal_fallback_parallel_root": fallback_root})


@pytest.mark.parametrize("case", ["large_bins", "tiled", "categorical", "vector", "serial",
                                  "subtraction_disabled", "fusion_disabled", "threads32",
                                  "original_gain", "fast_weight", "stump", "pruned"])
def test_ineligible_paths_keep_original_descendants(case):
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
    else:
        params.update({
            "serial": {"metal_parallel_splits": 0},
            "subtraction_disabled": {"metal_histogram_subtraction": 0},
            "fusion_disabled": {"metal_fused_subtraction": 0},
            "threads32": {"metal_split_threads": 32},
            "original_gain": {"metal_fast_gain": 0},
            "fast_weight": {"metal_fast_weight": 1},
            "stump": {"max_depth": 1},
            "pruned": {"gamma": 1e20},
        }[case])
    compare(X, y, bins=bins, params=params, metadata=metadata, expected_active=0, rounds=2)


@pytest.mark.parametrize("pruned", [False, True])
def test_nonterminal_frontiers_cross_host_parent_batch_boundary(pruned):
    # Eleven useful bits require evaluating descendants after the 512-parent
    # frontier is divided into two batches. Deferred rows and cached parents
    # must stay valid while the first batch is scored and assembled.
    values = np.arange(2048, dtype=np.uint32)
    X = ((values[:, None] >> np.arange(11, dtype=np.uint32)) & 1).astype(np.float32)
    y = values.astype(np.float32)
    if pruned:
        fixed = (values // 2) % 3 == 0
        y[fixed] = (values[fixed] // 2 * 2).astype(np.float32)
    actual, _ = compare(X, y, bins=2, rounds=1,
                        params={"max_depth": 11, "reg_lambda": 0, "min_child_weight": 1,
                                "metal_compact_splits": 1})
    count = len(tree(actual)["left_children"])
    assert 2047 < count < 4095 if pruned else count == 4095


@pytest.mark.parametrize("sampling", ["uniform", "gradient_based"])
def test_sampled_multiclass_parallel_trees_keep_compact_group_index(sampling):
    X, _ = inputs(rows=769, density=0.6)
    y = (np.arange(len(X)) % 3).astype(np.float32)
    compare(X, y, bins=128, rounds=2,
            params={"metal_gpu_tree": 0, "objective": "multi:softprob", "num_class": 3,
                    "num_parallel_tree": 2, "subsample": 0.63,
                    "sampling_method": sampling, "seed_per_iteration": True})


@pytest.mark.parametrize("group,coalesce,fill", [(1, 0, 0), (2, 1, 0), (4, 1, 1)])
def test_grouped_histograms_and_encoder_modes(group, coalesce, fill):
    X, y = inputs(rows=769, density=0.4)
    compare(X, y, bins=256, rounds=2,
            params={"metal_features_per_group": group, "metal_coalesce_encoders": coalesce,
                    "metal_compute_fill": fill, "metal_skip_empty_bins": 0})


@pytest.mark.parametrize("fast_stats", [0, 1])
def test_custom_cancellation_and_zero_hessians(fast_stats):
    X, y = inputs(rows=769, density=0.4)
    X[:, 1] = X[:, 0]
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.resize(np.asarray([1, 2, 1, 0, 2**-20], np.float32), len(X))
    compare(X, y, bins=256, rounds=2,
            params={"min_child_weight": 0, "metal_fast_stats": fast_stats},
            custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("params", [
    {"monotone_constraints": "(1,0,0,0,0)"},
    {"subsample": 0.63, "sampling_method": "gradient_based"},
    {"grow_policy": "lossguide", "max_depth": 0, "max_leaves": 9},
])
def test_independent_cpu_reference(reference, params):
    X, y = reference_dataset(rows=257)
    options = {"max_depth": 4, "max_bin": 32, "seed": 239, "base_score": 0.25, "eta": 0.2, **params}
    metadata = {"weight": np.linspace(0.1, 1.7, len(y), dtype=np.float32),
                "feature_weights": np.linspace(0.2, 1, X.shape[1], dtype=np.float32)}
    expected, files = reference(X, y, options, rounds=3, **metadata)
    data = xgb.DMatrix(X, label=y, **metadata)
    model = xgb.Booster({**options, "metal_gpu_tree": 0, "metal_fallback_parallel_splits": 1}, [data])
    for iteration in range(3):
        model.update(data, iteration)
        assert profile(model)["tuning"]["fallback_parallel_splits"] == 1
    compare_reference(model, data, expected, files)


def test_workspace_reset_and_runtime_switches():
    X, y = inputs(rows=769, density=0.6)
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=5, max_bin=256, base_score=0, seed=72, metal_gpu_tree=0))
        trace = []
        for iteration, rows in enumerate([257, 769, 0, 513, 769]):
            selected = 0 if iteration == 3 else enabled
            model.set_param("metal_fallback_parallel_splits", selected)
            if iteration == 2:
                model.reset()
            values = np.nan_to_num(X[:rows]) if iteration % 2 else X[:rows]
            data = xgb.DMatrix(values, label=y[:rows])
            model.update(data, iteration)
            assert profile(model)["tuning"]["fallback_parallel_splits"] == (selected if rows else 0)
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


def test_gpu_controller_keeps_its_existing_descendant_path():
    X, y = inputs(rows=513, density=0.7)
    traces = [fit(X, y, 1, bins=128, rounds=2,
                  params={"metal_fallback_parallel_splits": enabled}) for enabled in [0, 1]]
    assert_identical(traces[0][0], traces[1][0])
    for _, stats in traces:
        assert all(item["tuning"]["gpu_tree"] == 1 for item in stats)
        assert all(item["tuning"]["fallback_parallel_splits"] == 0 for item in stats)


def compare_dense_skip(X, y, *, params=None, **kwargs):
    options = {"metal_gpu_tree": 0, "metal_fallback_parallel_splits": 1, **(params or {})}
    results, exports, rebuilds = [], [], []
    active_by_round = None
    for enabled in [0, 1]:
        result, traces = fit(X, y, 1, params={**options, "metal_skip_empty_bins": enabled}, **kwargs)
        model = xgb.Booster(model_file=bytearray(result["model"]))
        raw_json = bytes(model.save_raw(raw_format="json"))
        if active_by_round is None:
            trees = json.loads(raw_json)["learner"]["gradient_booster"]["model"]["trees"]
            assert len(trees) == len(traces) > 0
            active_by_round = [int(len(item["left_children"]) > 1) for item in trees]
            assert any(active_by_round), "The fixture must exercise dense descendant scoring"
        for trace, active in zip(traces, active_by_round):
            tuning = trace["tuning"]
            assert tuning["gpu_tree"] == 0
            assert tuning["fallback_parallel_splits"] == active
            assert tuning["histogram_missing"] == 0
            assert tuning["split_skip_empty_bins"] == (enabled if active else 0)
            compact = bool(active and enabled and options.get("metal_compact_splits", 0))
            score = compact and bool(options.get("metal_fast_stats", 1)) and bool(options.get("metal_fused_score", 1))
            assert tuning["compact_splits"] == int(compact)
            assert tuning["fused_score"] == int(score)
            assert tuning["terminal_histogram_elision"] == 0
        rebuilds.append([t["tuning"]["histogram_rebuilds"] for t in traces])
        results.append(result)
        exports.append(raw_json)
    assert_identical(*results)
    assert exports[0] == exports[1]
    assert rebuilds[0] == rebuilds[1]
    if options.get("max_cached_hist_node") == 1:
        assert any(count > 0 for count in rebuilds[0])
    return results[1]


@pytest.mark.parametrize("group,fast_stats,cache", list(itertools.product([1, 4], [0, 1], [1, 4096])))
def test_dense_descendant_gaps_and_cancelled_bins_keep_first_threshold(group, fast_stats, cache):
    # Feature zero separates two populations; feature one's descendant scans
    # contain long empty stretches and occupied bins with exactly cancelling
    # gradients and zero Hessians, including across SIMD boundaries 31/32/63/64.
    reference_values = np.column_stack([np.arange(128) % 2, np.arange(128)]).astype(np.float32)
    reference = xgb.QuantileDMatrix(reference_values, max_bin=128)
    values = np.repeat(np.asarray([16, 31, 32, 63, 64, 100], np.float32), 64)
    X = np.column_stack([np.repeat([0, 1], len(values)), np.tile(values, 2)]).astype(np.float32)
    gradient = np.tile(np.asarray([2**40, -2**40], np.float32), len(X) // 2)
    hessian = np.zeros(len(X), np.float32)
    active = (X[:, 1] == 16) | (X[:, 1] == 100)
    sign = np.where(X[:, 0] == 0, -1, 1).astype(np.float32)
    gradient[active] = sign[active] * np.where(X[active, 1] == 16, 1, 3)
    hessian[active] = 1
    actual = compare_dense_skip(X, np.zeros(len(X), np.float32), bins=128, quantile=True,
                                metadata={"ref": reference}, rounds=1,
                                params={"max_depth": 3, "reg_lambda": 0, "min_child_weight": 0,
                                        "metal_features_per_group": group, "metal_fast_stats": fast_stats,
                                        "max_cached_hist_node": cache, "metal_compact_splits": 1},
                                custom=lambda prediction, matrix: (gradient, hessian))
    document = tree(actual)
    assert len(document["base_weights"]) == 7
    assert document["split_indices"][:3] == [0, 1, 1]
    assert all(16 < threshold < 31 for threshold in document["split_conditions"][1:3])


@pytest.mark.parametrize("params", [
    {"monotone_constraints": "(1,0,-1,0,0,0,0)"},
    {"colsample_bytree": 0.8, "colsample_bylevel": 0.7, "colsample_bynode": 0.6,
     "seed_per_iteration": True},
    {"grow_policy": "lossguide", "max_depth": 0, "max_leaves": 17, "max_cached_hist_node": 1},
])
@pytest.mark.parametrize("fast_stats", [0, 1])
def test_dense_skip_keeps_constraints_masks_and_rebuilt_parents_exact(params, fast_stats):
    X, y = inputs(rows=769, density=0.8)
    compare_dense_skip(np.nan_to_num(X), y, bins=256, rounds=2,
                       params={**params, "metal_fast_stats": fast_stats, "metal_compact_splits": 1})


@pytest.mark.parametrize("bins", [2, 31, 32, 33, 256])
def test_dense_skip_explicit_opt_out_at_bin_boundaries(bins):
    X, y = inputs(rows=513, density=0.8)
    compare_dense_skip(np.nan_to_num(X), y, bins=bins, rounds=2, params={"metal_compact_splits": 1})


@pytest.mark.parametrize("params", [
    {"metal_fallback_parallel_splits": 0},
    {"max_depth": 1},
    {"metal_fused_subtraction": 0},
    {"metal_split_threads": 32},
])
def test_dense_skip_does_not_enable_other_host_paths(params):
    X, y = inputs(rows=513, density=0.8)
    results = []
    for enabled in [0, 1]:
        result, traces = fit(np.nan_to_num(X), y, 1, bins=128, rounds=2,
                             params={"metal_gpu_tree": 0, "metal_fallback_parallel_splits": 1,
                                     "metal_compact_splits": 1, "metal_fused_score": 1,
                                     **params, "metal_skip_empty_bins": enabled})
        assert all(t["tuning"]["fallback_parallel_splits"] == 0 for t in traces)
        assert all(t["tuning"]["split_skip_empty_bins"] == 0 for t in traces)
        assert all(t["tuning"]["compact_splits"] == t["tuning"]["fused_score"] == 0 for t in traces)
        results.append(result)
    assert_identical(*results)


@pytest.mark.parametrize("missing,bins,levels,expected", [
    (False, 64, None, 0), (False, 192, None, 0), (False, 256, None, 1),
    (True, 128, None, 0), (True, 160, None, 0), (True, 192, None, 1), (False, 256, 16, 0)])
def test_default_follows_mean_numeric_cut_count(missing, bins, levels, expected):
    # One lane per bin: few cuts leave most of the 256 threads idle, where the
    # original 32-lane evaluator is faster. Both selections stay exact.
    rng = np.random.default_rng(1729)
    if levels is None:
        X = rng.normal(size=(4096, 6)).astype(np.float32)
    else:
        X = rng.integers(0, levels, size=(4096, 6)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2] * X[:, 3]).astype(np.float32)
    if missing:
        X[rng.random(X.shape) < 0.2] = np.nan
    options = {"metal_gpu_tree": 0, "max_depth": 6}
    default, traces = fit(X, y, 1, bins=bins, rounds=2, params=options)
    assert all(t["tuning"]["gpu_tree"] == 0 for t in traces)
    assert all(t["tuning"]["fallback_parallel_splits"] == expected for t in traces)
    explicit, controls = fit(X, y, 1, bins=bins, rounds=2,
                             params={**options, "metal_fallback_parallel_splits": 1 - expected})
    assert all(t["tuning"]["fallback_parallel_splits"] == 1 - expected for t in controls)
    assert_identical(default, explicit)
