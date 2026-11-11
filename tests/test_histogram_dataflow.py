"""Packed bins, row-tile records, single-owner stores and unit Hessians keep exact trees."""
import itertools

import numpy as np
import pytest
import xgboost as xgb

from test_fast_scalar_weight import tree
from test_gpu_tree import assert_identical, profile, snapshot
from test_gpu_tree_deep import full_frontier_data
from test_histogram_missing import inputs, layout
from test_sampling import compare as compare_reference
from test_sampling import dataset as reference_dataset
from test_split_subtraction_fusion import fit


OFF = {"metal_histogram_packed_bins": 0, "metal_histogram_schedule": 0,
       "metal_histogram_single_owner": 0, "metal_histogram_unit_hessian": 0}
FIELDS = {"histogram_packed_bins": "metal_histogram_packed_bins",
          "histogram_schedule": "metal_histogram_schedule",
          "histogram_single_owner": "metal_histogram_single_owner",
          "histogram_unit_hessian": "metal_histogram_unit_hessian"}


def selection(tuning):
    return {key: tuning[key] for key in FIELDS}


def compare(X, y, *, options=None, expected=None, params=None, **kwargs):
    """Compare the original histogram kernels with the selected dataflow options."""
    params = params or {}
    control, controls = fit(X, y, 1, params={**params, **OFF}, **kwargs)
    actual, stats = fit(X, y, 1, params={**params, **(options or {})}, **kwargs)
    assert_identical(control, actual)
    assert len(controls) == len(stats) > 0
    for item in controls:
        assert selection(item["tuning"]) == dict.fromkeys(FIELDS, 0)
    if expected is not None:
        for item in stats:
            assert selection(item["tuning"]) == expected
    return actual, stats


def wanted(packed=16, schedule=None, single=1, unit=1):
    if schedule is None:
        schedule = 2 if packed == 16 else 1
    return {"histogram_packed_bins": packed, "histogram_schedule": schedule,
            "histogram_single_owner": single, "histogram_unit_hessian": unit}


@pytest.mark.parametrize("bins,missing,group,width", list(itertools.product(
    [2, 33, 255, 256], [False, True], [1, 2, 3, 4], [4, 16])))
def test_bin_codes_feature_groups_and_word_widths(bins, missing, group, width):
    # Seven features leave a partial word group and a partial 16-feature chunk.
    # With 256 bins, dense code 255 is an ordinary bin; missing values are then
    # flagged in a separate plane. Three features per group would straddle
    # packed words.
    X, y = inputs(rows=1025, density=0.6)
    if not missing:
        X = np.nan_to_num(X)
    packable = group != 3
    _, stats = compare(X, y, bins=bins, rounds=2, expected=wanted(width if packable else 0),
                       options={"metal_histogram_packed_bins": width},
                       params={"metal_features_per_group": group, "max_depth": 6})
    assert all(item["tuning"]["features_per_group"] == group for item in stats)
    masked = int(packable and missing and bins == 256)
    assert all(item["tuning"]["histogram_packed_mask"] == masked for item in stats)


def masked_inputs(rows, cols, seed=4242):
    """At least 256 cuts per ordinary feature, with per-feature missing rates."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, cols)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, -1] + 0.25 * rng.normal(size=rows)).astype(np.float32)
    rates = [0.2, 0.0, 0.5, 0.02, 1.0]
    for column in range(cols):
        X[rng.random(rows) < rates[column % len(rates)], column] = np.nan
    if cols > 5:
        # A few-bin feature beside full ones: its byte codes stay far below 255.
        X[:, 5] = np.where(np.isnan(X[:, 5]), np.nan, np.round(X[:, 5]))
    return X, y


@pytest.mark.parametrize("cols", [1, 4, 5, 8, 9, 12, 13, 16, 17, 33])
@pytest.mark.parametrize("group", [1, 2, 4, 8])
@pytest.mark.parametrize("width", [4, 16])
def test_missing_flag_plane_with_256_bins(cols, group, width):
    # Word groups, chunks and eight-feature tasks end at every partial width.
    # Width 4 gives each word group its own flags, so an eight-feature task
    # reads two of them.
    X, y = masked_inputs(1537, cols)
    params = {"metal_features_per_group": group, "metal_histogram_rows": 256, "max_depth": 6}
    actual, stats = compare(X, y, bins=256, rounds=2, expected=wanted(width), params=params,
                            options={"metal_histogram_packed_bins": width})
    assert all(item["tuning"]["histogram_packed_mask"] == 1 for item in stats)
    # Same packed request without the plane: the uint16 bins, as before.
    control, controls = fit(X, y, 1, bins=256, rounds=2,
                            params={**params, "metal_histogram_packed_bins": width,
                                    "metal_histogram_packed_mask": 0})
    assert_identical(control, actual)
    for item in controls:
        assert item["tuning"]["histogram_packed_bins"] == 0
        assert item["tuning"]["histogram_packed_mask"] == 0


@pytest.mark.parametrize("case", ["present", "logistic", "weighted", "multiclass", "bytree", "sampling",
                                  "quantile", "no_single_owner", "group_major"])
def test_missing_flag_plane_kernel_variants(case):
    X, y = masked_inputs(2053, 11)
    params, metadata, options = {"metal_histogram_rows": 256, "max_depth": 6}, {}, {}
    if case == "present":
        params["metal_histogram_present"] = 1
    elif case == "logistic":
        y = (y > 0).astype(np.float32)
        params["objective"] = "binary:logistic"
    elif case == "weighted":
        metadata["weight"] = np.linspace(0.5, 2, len(y), dtype=np.float32)
    elif case == "multiclass":
        y = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3])).astype(np.float32)
        params.update(objective="multi:softprob", num_class=3)
    elif case == "bytree":
        params.update(colsample_bytree=0.6, seed_per_iteration=True)
    elif case == "sampling":
        params["subsample"] = 0.7
    elif case == "no_single_owner":
        options["metal_histogram_single_owner"] = 0
    elif case == "group_major":
        options["metal_histogram_schedule"] = 1
    unit = int(case in ("present", "bytree", "quantile", "no_single_owner", "group_major"))
    expected = wanted(unit=unit, single=int(case != "no_single_owner"),
                      schedule=1 if case == "group_major" else None)
    _, stats = compare(X, y, bins=256, rounds=2, params=params, metadata=metadata, options=options,
                       quantile=case == "quantile", expected=expected)
    for item in stats:
        assert item["tuning"]["gpu_tree"] == 1
        assert item["tuning"]["histogram_packed_mask"] == 1
        assert item["tuning"]["histogram_present"] == int(case == "present")


@pytest.mark.parametrize("width", [4, 16])
@pytest.mark.parametrize("cols", [8, 12, 21])
def test_missing_flag_plane_with_certified_eight_feature_sums(width, cols):
    # Dyadic gradients and unit Hessians certify 32-bit sums, so eight features
    # with 256 bins fit one task's scratch instead of two four-feature halves.
    X, y = masked_inputs(2053, cols)
    gradient = np.resize(np.asarray([1, -1, 2, -0.5, 0, 4], np.float32), len(X))
    hessian = np.ones(len(X), np.float32)
    _, stats = compare(X, y, bins=256, rounds=3, options={"metal_histogram_packed_bins": width},
                       expected=wanted(width),
                       params={"metal_features_per_group": 8, "metal_histogram_rows": 256,
                               "min_child_weight": 0, "max_depth": 6},
                       custom=lambda prediction, matrix: (gradient, hessian))
    for item in stats:
        assert item["tuning"]["histogram_packed_mask"] == 1
        assert item["tuning"]["histogram_i32"] == 1


def test_missing_flag_plane_needs_256_bins_and_missing_values():
    X, y = masked_inputs(1537, 9)
    for values, bins in [(np.nan_to_num(X), 256), (X, 255), (X, 64)]:
        _, stats = compare(values, y, bins=bins, rounds=1, expected=wanted())
        assert all(item["tuning"]["histogram_packed_mask"] == 0 for item in stats)


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("width,schedule,single,unit", [
    item for item in itertools.product([0, 4, 16], [0, 1, 2], [0, 1], [0, 1]) if any(item)])
def test_independent_layout_schedule_owner_and_hessian_controls(missing, width, schedule, single, unit):
    # 256-row tiles mix multi-tile and single-tile built nodes at every depth.
    X, y = inputs(rows=2053, density=0.7)
    if not missing:
        X = np.nan_to_num(X)
    options = {"metal_histogram_packed_bins": width, "metal_histogram_schedule": schedule,
               "metal_histogram_single_owner": single, "metal_histogram_unit_hessian": unit}
    compare(X, y, bins=128, rounds=2, options=options, expected=wanted(width, schedule, single, unit),
            params={"metal_histogram_rows": 256, "metal_features_per_group": 4, "max_depth": 7})


@pytest.mark.parametrize("rows", [0, 1, 255, 256, 257, 511, 513, 2049])
@pytest.mark.parametrize("missing", [False, True])
def test_empty_partial_and_single_tile_nodes(rows, missing):
    X, y = inputs(rows=max(rows, 1), density=0.5)
    X, y = X[:rows], y[:rows]
    if not missing:
        X = np.nan_to_num(X)
    compare(X, y, bins=64, rounds=2, expected=wanted() if rows else None,
            params={"metal_histogram_rows": 256, "metal_features_per_group": 2, "max_depth": 6})


@pytest.mark.parametrize("case", ["present", "present_grouped", "compact", "compact_pair",
                                  "full_clear", "separate_subtraction", "separate_control",
                                  "coalesced", "serial_splits", "no_terminal_elision"])
@pytest.mark.parametrize("missing", [False, True])
def test_histogram_kernel_and_driver_variants(case, missing):
    X, y = inputs(rows=1025, density=0.65)
    if not missing:
        X = np.nan_to_num(X)
    params = {"max_depth": 6, "metal_histogram_rows": 256, "metal_features_per_group": 1}
    if case.startswith("present"):
        params["metal_histogram_present"] = 1
        if case == "present_grouped":
            params["metal_features_per_group"] = 4
    elif case.startswith("compact"):
        params["metal_histogram_compact_scratch"] = 1
        if case == "compact_pair":
            params["metal_features_per_group"] = 2
    elif case == "full_clear":
        params["metal_histogram_clear_built"] = 0
    elif case == "separate_subtraction":
        params["metal_fused_subtraction"] = 0
    elif case == "separate_control":
        params["metal_tree_control_fusion"] = 0
    elif case == "coalesced":
        params.update(metal_coalesce_encoders=1, metal_compute_fill=1)
    elif case == "serial_splits":
        params["metal_parallel_splits"] = 0
    else:
        params["metal_terminal_histogram_elision"] = 0
    _, stats = compare(X, y, bins=255, rounds=2, expected=wanted(), params=params)
    for item in stats:
        assert item["tuning"]["gpu_tree"] == 1
        assert item["tuning"]["histogram_present"] == int(case.startswith("present") and missing)


@pytest.mark.parametrize("group", [1, 2, 4])
@pytest.mark.parametrize("width", [0, 16])
@pytest.mark.parametrize("hessians", [[1, 2, 1, 0, 2**-20], [1], [2**-3]])
def test_signed_carries_and_caller_hessians(group, width, hessians):
    # Single owners store totals that cancel to zero or carry across words.
    # Caller Hessians are verified on the GPU: only constant values use counts.
    X, y = inputs(rows=2053, density=0.4)
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.resize(np.asarray(hessians, np.float32), len(X))
    _, stats = compare(X, y, bins=255, rounds=3, options={"metal_histogram_packed_bins": width},
                       params={"metal_features_per_group": group, "metal_histogram_rows": 256,
                               "min_child_weight": 0},
                       custom=lambda prediction, matrix: (gradient, hessian))
    for item in stats:
        assert item["tuning"]["histogram_unit_hessian"] == int(len(hessians) == 1)


@pytest.mark.parametrize("objective,weighted,unit", [
    ("reg:squarederror", False, 1), ("reg:squarederror", True, 0), ("reg:linear", False, 1),
    ("reg:pseudohubererror", False, 0), ("binary:logistic", False, 0), ("binary:hinge", False, 0),
    ("count:poisson", False, 0)])
@pytest.mark.parametrize("missing", [False, True])
def test_unit_hessian_objectives_and_weights(objective, weighted, unit, missing):
    X, y = inputs(rows=1025, density=0.6)
    if not missing:
        X = np.nan_to_num(X)
    if objective in ("binary:logistic", "binary:hinge"):
        y = (y > np.median(y)).astype(np.float32)
    elif objective == "count:poisson":
        y = np.abs(y)
    metadata = {"weight": np.linspace(0.5, 2, len(y), dtype=np.float32)} if weighted else {}
    _, stats = compare(X, y, bins=255, rounds=2, metadata=metadata, expected=wanted(unit=unit),
                       options={"metal_histogram_unit_hessian": 1},
                       params={"objective": objective, "metal_histogram_rows": 256})
    assert all(item["tuning"]["gpu_tree"] == 1 for item in stats)


@pytest.mark.parametrize("case", ["multiclass", "parallel", "weighted", "base_margin", "bytree", "sampling"])
def test_group_offsets_metadata_and_column_sampling(case):
    X, y = inputs(rows=1025, density=0.8)
    X = np.nan_to_num(X)
    metadata, params = {}, {"metal_histogram_rows": 256}
    if case == "multiclass":
        y = np.argmax(X[:, :3], axis=1).astype(np.float32)
        params.update(objective="multi:softprob", num_class=3)
    elif case == "parallel":
        params["num_parallel_tree"] = 3
    elif case == "weighted":
        metadata["weight"] = np.linspace(0, 2, len(y), dtype=np.float32)
    elif case == "base_margin":
        metadata["base_margin"] = np.linspace(-0.5, 0.4, len(y), dtype=np.float32)
    elif case == "sampling":
        params["subsample"] = 0.7
    else:
        params.update(colsample_bytree=0.6, seed_per_iteration=True)
    compare(X, y, bins=256, rounds=2, params=params, metadata=metadata,
            expected=wanted(unit=int(case not in ("multiclass", "weighted", "sampling"))))


@pytest.mark.parametrize("kind", ["csr", "csc", "quantile"])
def test_sparse_and_quantile_inputs(kind):
    X, y = inputs(rows=1025, density=0.3)
    data = X if kind == "quantile" else layout(X, kind)
    compare(data, y, bins=200, rounds=2, quantile=kind == "quantile", expected=wanted(),
            params={"metal_histogram_rows": 512})


@pytest.mark.parametrize("pruned", [False, True])
def test_depth_ten_full_frontiers_store_single_owner_slices(pruned):
    X, y = full_frontier_data(pruned)
    actual, stats = compare(X, y, bins=2, quantile=True, rounds=1, expected=wanted(),
                            params={"max_depth": 10, "reg_lambda": 0, "min_child_weight": 1})
    assert stats[-1]["tuning"]["gpu_tree_levels"] == 10
    assert len(tree(actual)["base_weights"]) > 1023


@pytest.mark.parametrize("case", ["cpu_driver", "depth_eleven", "large_bins", "vector", "categorical",
                                  "missing_kernel_control"])
def test_unsupported_paths_keep_original_kernels(case):
    X, y = inputs(rows=513, density=0.5)
    params, metadata, bins = {}, {}, 128
    if case == "cpu_driver":
        params["metal_gpu_tree"] = 0
    elif case == "depth_eleven":
        params["max_depth"] = 11
    elif case == "large_bins":
        bins = 257
    elif case == "vector":
        y = np.column_stack([y, -y]).astype(np.float32)
        params["multi_strategy"] = "multi_output_tree"
    elif case == "categorical":
        present = ~np.isnan(X[:, 0])
        X[present, 0] = np.arange(np.count_nonzero(present)) % 5
        metadata["feature_types"] = ["c"] + ["q"] * 6
    else:
        params["metal_histogram_missing"] = 0
    compare(X, y, bins=bins, rounds=2, params=params, metadata=metadata,
            expected=dict.fromkeys(FIELDS, 0))


def test_dense_missing_kernel_control_still_uses_dataflow_variants():
    # Dense input never selects the missing kernels, so their control is moot.
    X, y = inputs(rows=513, density=0.5)
    compare(np.nan_to_num(X), y, bins=128, rounds=2, expected=wanted(),
            params={"metal_histogram_missing": 0})


@pytest.mark.parametrize("rows,missing,packed,policy,expected_group", [
    (16383, False, 16, "depthwise", 1), (16384, False, 16, "depthwise", 4), (16384, False, 0, "depthwise", 1),
    (16384, True, 16, "depthwise", 2), (16384, False, 16, "lossguide", 1)])
def test_default_feature_grouping_follows_packed_dense_input(rows, missing, packed, policy, expected_group):
    rng = np.random.default_rng(5150)
    X = rng.normal(size=(rows, 9)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] * X[:, 2]).astype(np.float32)
    if missing:
        X[rng.random(X.shape) < 0.2] = np.nan
    results = []
    for group in [None, expected_group]:
        params = {"metal_histogram_packed_bins": packed, "max_depth": 6, "grow_policy": policy}
        if group is not None:
            params["metal_features_per_group"] = group
        result, stats = fit(X, y, 1, bins=256, rounds=2, params=params)
        assert all(item["tuning"]["features_per_group"] == expected_group for item in stats)
        results.append(result)
    assert_identical(*results)
    control, _ = fit(X, y, 1, bins=256, rounds=2, params={"max_depth": 6, "grow_policy": policy, **OFF})
    assert_identical(control, results[0])


def test_requantization_and_width_changes_rebuild_packed_codes():
    X, y = inputs(rows=1025, density=0.7)
    data = xgb.DMatrix(X, label=y)
    sequence = [(32, 16), (64, 4), (255, 16), (32, 4), (256, 16)]
    traces = []
    for enabled in [0, 1]:
        trace = []
        for iteration, (bins, width) in enumerate(sequence):
            options = {"max_bin": bins, "max_depth": 5, "base_score": 0, "seed": 31,
                       "metal_histogram_packed_bins": width if enabled else 0}
            model = xgb.Booster(options, [data])
            model.update(data, 0)
            tuning = profile(model)["tuning"]
            assert tuning["histogram_packed_bins"] == (width if enabled else 0)
            assert tuning["histogram_packed_mask"] == int(enabled and bins == 256)
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


def test_workspace_reset_and_runtime_switches():
    X, y = inputs(rows=2053, density=0.6)
    modes = [(16, 2, 1, 1), (4, 1, 0, 0), (0, 0, 1, 1), (16, 0, 0, 1), (0, 2, 1, 0)]
    traces = []
    for enabled in [0, 1]:
        model = xgb.Booster(dict(max_depth=6, max_bin=255, base_score=0, seed=72,
                                 metal_histogram_rows=512))
        trace = []
        for iteration, (rows, mode) in enumerate(zip([257, 2053, 0, 513, 2053], modes)):
            selected = dict(zip(FIELDS.values(), mode if enabled else (0, 0, 0, 0)))
            model.set_param(selected)
            if iteration == 2:
                model.reset()
            values = np.nan_to_num(X[:rows]) if iteration % 2 else X[:rows]
            data = xgb.DMatrix(values, label=y[:rows])
            model.update(data, iteration)
            if rows:
                assert selection(profile(model)["tuning"]) == dict(zip(FIELDS, mode if enabled else (0, 0, 0, 0)))
            trace.append(snapshot(model, [data, xgb.DMatrix(X)]))
        traces.append(trace)
    for expected, actual in zip(*traces):
        assert_identical(expected, actual)


@pytest.mark.parametrize("name,value", [("metal_histogram_packed_bins", 8),
                                        ("metal_histogram_schedule", 3)])
def test_invalid_controls_are_rejected(name, value):
    X, y = inputs(rows=257, density=0.5)
    data = xgb.DMatrix(X, label=y)
    model = xgb.Booster({"max_depth": 4, name: value}, [data])
    with pytest.raises(xgb.core.XGBoostError, match=name):
        model.update(data, 0)


@pytest.mark.parametrize("missing", [False, True])
def test_independent_cpu_reference(reference, missing):
    X, y = reference_dataset(rows=1031)
    if not missing:
        X = np.nan_to_num(X)
    options = {"max_depth": 6, "max_bin": 64 if missing else 256, "seed": 239, "base_score": 0.25,
               "eta": 0.2}
    expected, files = reference(X, y, options, rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.Booster({**options, "metal_histogram_rows": 256}, [data])
    for iteration in range(3):
        model.update(data, iteration)
        assert selection(profile(model)["tuning"]) == wanted()
    compare_reference(model, data, expected, files)
