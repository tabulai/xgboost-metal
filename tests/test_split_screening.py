"""Certified float32 split screening preserves exact split choices and model bytes."""
import itertools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from test_gpu_tree import assert_identical, profile, snapshot
from test_gpu_tree_deep import full_frontier_data
from test_histogram_missing import inputs, layout
from test_sampling import compare as compare_reference
from test_sampling import dataset as reference_dataset
from test_split_subtraction_fusion import fit

ROOT = Path(__file__).resolve().parents[1]


def compare(X, y, *, params=None, expected_active=1, expected_cutoff=None, **kwargs):
    params = params or {}
    expected, controls = fit(X, y, 1, params={**params, "metal_split_screening": 0}, **kwargs)
    actual, stats = fit(X, y, 1, params={**params, "metal_split_screening": 1}, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["split_screening"] == 0 for item in controls)
    assert all(item["tuning"]["root_screening"] == 0 for item in controls)
    assert all(item["tuning"]["split_node_cutoff"] == 0 for item in controls)
    assert all(item["tuning"]["split_screening"] == expected_active for item in stats)
    if expected_cutoff is not None:
        assert all(item["tuning"]["split_node_cutoff"] == expected_cutoff for item in stats)
    return actual, stats


def test_native_enclosures_and_kernel_records(tmp_path_factory):
    """Fuzz the float32 enclosures against every exact scorer family and diff kernel records."""
    cmake = shutil.which("cmake")
    if cmake is None or shutil.which("xcrun") is None:
        pytest.skip("The native harness needs cmake and the Metal toolchain.")
    build = tmp_path_factory.mktemp("split-screening-native")
    configure = [cmake, "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DMETALGB_BUILD_TESTS=ON"]
    if shutil.which("ninja"):
        configure += ["-G", "Ninja"]
    for command in (configure, [cmake, "--build", str(build), "--target", "split_screening_bounds_test"]):
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
        assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    completed = subprocess.run([str(build / "split_screening_bounds_test"), "250000"], capture_output=True,
                               text=True, timeout=900)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip().endswith("OK")


@pytest.mark.parametrize("kind,bins", list(itertools.product(["dense", "missing", "csr"], [2, 33, 256])))
@pytest.mark.parametrize("depth", [3, 8])
def test_layouts_bins_and_kernel_families(kind, bins, depth):
    X, y = inputs(rows=1025, density=0.4)
    X = np.nan_to_num(X) if kind == "dense" else layout(X, "csr") if kind == "csr" else X
    compare(X, y, bins=bins, rounds=2, params={"max_depth": depth}, expected_cutoff=1)


@pytest.mark.parametrize("kind,root", [("dense", None), ("dense", 1), ("missing", None), ("missing", 0)])
@pytest.mark.parametrize("driver", [1, 0])
def test_root_screening_defaults_to_missing_valued_input(kind, root, driver):
    X, y = inputs(rows=1025, density=0.6)
    if kind == "dense":
        X = np.nan_to_num(X)
    params = {"metal_gpu_tree": driver, "max_depth": 4}
    if root is not None:
        params["metal_split_screening_root"] = root
    _, stats = compare(X, y, bins=64, rounds=2, params=params)
    expected = int(kind == "missing") if root is None else root
    assert all(item["tuning"]["root_screening"] == expected for item in stats)


@pytest.mark.parametrize("params", [
    {"metal_compact_splits": 0}, {"metal_compact_splits": 1}, {"metal_fused_score": 0},
    {"metal_fast_stats": 0}, {"metal_parallel_root": 0}, {"metal_split_screening_root": 1},
    {"metal_terminal_histogram_elision": 0},
    {"metal_split_node_cutoff": 0}, {"metal_split_inherit_parent": 0}, {"metal_features_per_group": 1},
    {"metal_histogram_present": 1}, {"metal_split_node_cutoff": 0, "metal_split_inherit_parent": 0},
])
def test_scorer_variants_and_controls(params):
    X, y = inputs(rows=1025, density=0.5)
    cutoff = 0 if params.get("metal_split_node_cutoff") == 0 else 1
    _, stats = compare(X, y, bins=64, rounds=3, params={"max_depth": 6, **params}, expected_cutoff=cutoff)
    inherit = 0 if params.get("metal_split_inherit_parent") == 0 else 1
    assert all(item["tuning"]["split_inherit_parent"] == inherit for item in stats)


@pytest.mark.parametrize("depth,quantile,pruned", [(2, False, False), (6, True, False),
                                                   (9, False, True), (10, True, False), (10, False, True)])
def test_full_frontiers_terminal_levels_and_pruning(depth, quantile, pruned):
    X, y = full_frontier_data(pruned)
    compare(X, y, bins=2, rounds=1, quantile=quantile,
            params={"max_depth": depth, "reg_lambda": 0, "min_child_weight": 1})


@pytest.mark.parametrize("params", [
    {"reg_lambda": 0}, {"reg_lambda": 2**-80}, {"reg_lambda": 2**30}, {"reg_lambda": 0.31},
    {"min_child_weight": 0}, {"min_child_weight": 5}, {"min_child_weight": 37.5},
    {"gamma": 0.5}, {"reg_alpha": -0.0}, {"max_delta_step": -0.0},
])
def test_regularization_and_child_weight_boundaries(params):
    X, y = inputs(rows=769, density=0.6)
    compare(X, y, rounds=3, params=params)


@pytest.mark.parametrize("params", [
    {"reg_alpha": 0.125}, {"reg_alpha": 2**-120}, {"max_delta_step": 0.5},
    {"monotone_constraints": "(1,0,0,0,0,0,0)"}, {"metal_skip_empty_bins": 0, "metal_compact_splits": 0,
                                                  "metal_parallel_root": 0},
    {"metal_split_threads": 32, "metal_compact_splits": 0, "metal_parallel_root": 0},
])
def test_ineligible_settings_keep_the_unscreened_kernels(params):
    X, y = inputs(rows=513, density=0.6)
    compare(X, y, rounds=2, params={"metal_fallback_parallel_splits": 0, **params}, expected_active=0)


@pytest.mark.parametrize("case", ["cancellation", "signed_zero", "zero_hessian", "negative_hessian",
                                  "tiny_hessian", "huge_gradient", "near_child_weight"])
def test_custom_derivatives_rounding_and_validity_edges(case):
    X, y = inputs(rows=513, density=0.5)
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.ones_like(gradient)
    params = {"min_child_weight": 0, "reg_lambda": 0, "max_depth": 5}
    if case == "signed_zero":
        gradient = np.resize(np.asarray([0.0, -0.0], np.float32), len(X))
    elif case == "zero_hessian":
        hessian[::3] = 0
    elif case == "negative_hessian":
        hessian = np.resize(np.asarray([-1, 2, 1, 0], np.float32), len(X))
    elif case == "tiny_hessian":
        gradient = np.resize(np.asarray([-3, -1, 1, 3], np.float32), len(X)) * np.float32(2**-20)
        hessian.fill(2**-40)
    elif case == "huge_gradient":
        gradient = np.resize(np.asarray([-3, 1, 2, -1e30, 1e30], np.float32), len(X))
    elif case == "near_child_weight":
        # Unit Hessians make child sums integers; min_child_weight sits on
        # and between them, so validity is decided exactly at the boundary.
        gradient = np.resize(np.asarray([-1, 0.5, 0.25, 1, -0.75], np.float32), len(X))
        params.update(min_child_weight=7, reg_lambda=1)
    compare(X, y, params=params, rounds=2, custom=lambda prediction, matrix: (gradient, hessian))


@pytest.mark.parametrize("case", ["duplicate_features", "symmetric", "constant_target", "flat_logistic"])
def test_near_and_exact_ties_keep_first_feature_and_threshold(case):
    X, y = inputs(rows=1025, density=0.7)
    params = {"max_depth": 6}
    if case == "duplicate_features":
        X[:, 1] = X[:, 0]
        X[:, 3] = X[:, 2]
    elif case == "symmetric":
        X[:, 0] = np.resize(np.asarray([-2, -1, 1, 2], np.float32), len(X))
        y = np.abs(X[:, 0]).astype(np.float32)
    elif case == "constant_target":
        y.fill(1)
    else:
        # Separable labels drive logistic Hessians towards zero, leaving flat
        # nodes whose candidates tie or differ only by rounding noise.
        y = (np.nan_to_num(X[:, 0]) > 0).astype(np.float32)
        params.update(objective="binary:logistic", base_score=0.5, min_child_weight=0)
    compare(X, y, bins=128, params=params, rounds=4)


@pytest.mark.parametrize("case", ["multiclass", "parallel", "weighted", "column_sample"])
def test_multiple_trees_weights_and_rng(case):
    X, y = inputs(rows=513, density=0.4)
    params, metadata = {}, {}
    if case == "multiclass":
        y = (np.arange(len(X)) % 3).astype(np.float32)
        params.update(objective="multi:softprob", num_class=3)
    elif case == "parallel":
        params["num_parallel_tree"] = 3
    elif case == "weighted":
        metadata["weight"] = np.linspace(0, 3, len(X), dtype=np.float32)
    else:
        params.update(colsample_bytree=0.6, seed_per_iteration=True)
        metadata["feature_weights"] = np.asarray([0.1, 0.2, 1, 2, 4, 8, 16], np.float32)
    compare(X, y, params=params, metadata=metadata, rounds=3)


@pytest.mark.parametrize("case", ["all_missing", "constant_target", "empty", "pruned"])
def test_empty_frontiers_and_unsplit_roots(case):
    X, y = inputs(rows=257, density=0.5)
    params = {}
    if case == "all_missing":
        X.fill(np.nan)
    elif case == "constant_target":
        y.fill(0)
    elif case == "empty":
        X, y = X[:0], y[:0]
    else:
        params["gamma"] = 1e20
    compare(X, y, params=params, rounds=2)


@pytest.mark.parametrize("params", [{}, {"max_depth": 12}, {"grow_policy": "lossguide", "max_leaves": 9},
                                    {"metal_fallback_parallel_root": 0}])
def test_host_driver_screens_roots_and_descendants(params):
    X, y = inputs(rows=769, density=0.6)
    actual, stats = compare(X, y, bins=64, rounds=2, params={"metal_gpu_tree": 0, **params},
                            expected_cutoff=0)
    assert all(item["tuning"]["gpu_tree"] == 0 for item in stats)


def test_counters_report_exactly_scored_candidates():
    X, y = inputs(rows=2049, density=0.8)
    data = xgb.DMatrix(X, label=y)
    model = xgb.Booster(dict(max_depth=8, max_bin=256, seed=3, base_score=0, metal_split_screening_stats=1),
                        [data])
    for iteration in range(3):
        model.update(data, iteration)
    counters = profile(model)["split_screening"]
    assert counters["candidates"] > counters["certified"] > 0
    assert 0 < counters["exact"] < counters["candidates"] // 10
    assert 0 < counters["exact_groups"] <= counters["groups"]
    disabled = xgb.Booster(dict(max_depth=8, max_bin=256, seed=3, base_score=0), [data])
    disabled.update(data, 0)
    assert all(value == 0 for value in profile(disabled)["split_screening"].values())


@pytest.mark.parametrize("params", [{}, {"metal_gpu_tree": 0}, {"objective": "binary:logistic"}])
def test_independent_cpu_reference(reference, params):
    X, y = reference_dataset(rows=257)
    if params.get("objective") == "binary:logistic":
        y = (y > np.median(y)).astype(np.float32)
    options = {"max_depth": 5, "max_bin": 32, "seed": 239, "base_score": 0.25, "eta": 0.2, **params}
    metadata = {"weight": np.linspace(0.1, 1.7, len(y), dtype=np.float32)}
    expected, files = reference(X, y, options, rounds=3, **metadata)
    data = xgb.DMatrix(X, label=y, **metadata)
    model = xgb.Booster({**options, "metal_split_screening": 1}, [data])
    for iteration in range(3):
        model.update(data, iteration)
        assert profile(model)["tuning"]["split_screening"] == 1
    compare_reference(model, data, expected, files)


def test_runtime_toggle_and_reset_preserve_bytes():
    X, y = inputs(rows=1025, density=0.5)
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_bin=64, base_score=0, seed=19, max_depth=7))
        trace = []
        for iteration, (rows, reset) in enumerate([(513, False), (1025, False), (0, True), (769, False)]):
            selected = enabled if iteration != 1 else 1 - enabled
            model.set_param({"metal_split_screening": selected})
            if reset:
                model.reset()
            data = xgb.DMatrix(X[:rows], label=y[:rows])
            model.update(data, iteration)
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)
