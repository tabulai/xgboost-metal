"""Private missing-bin reductions preserve the original integer histograms."""
import numpy as np
import pytest
from scipy import sparse
import xgboost as xgb

from test_gpu_tree import assert_identical, profile, snapshot


def inputs(rows=769, density=0.05):
    rng = np.random.default_rng(7315)
    X = rng.normal(size=(rows, 7)).astype(np.float32)
    y = (X[:, 0] - X[:, 1] * X[:, 2] + X[:, 3] ** 2).astype(np.float32)
    X[rng.random(X.shape) >= density] = np.nan
    # Include explicit zero entries, an entirely missing row and column, and a
    # partial final feature group. Sparse constructors must retain those zeros.
    X[1::17, 0] = 0
    X[::29] = np.nan
    X[:, -1] = np.nan
    return X, y


def layout(X, kind):
    if kind == "dense":
        return X
    rows, columns = np.nonzero(~np.isnan(X))
    result = sparse.csr_matrix((X[rows, columns], (rows, columns)), shape=X.shape)
    return result.tocsc() if kind == "csc" else result


def fit(X, y, enabled, *, bins=16, group=4, params=None, metadata=None,
        rounds=3, quantile=False, custom=None):
    options = dict(max_depth=4, max_bin=bins, seed=17, base_score=0,
                   metal_histogram_missing=enabled, metal_features_per_group=group)
    options.update(params or {})
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    metadata = metadata or {}
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
    expected, control_stats = fit(X, y, 0, **kwargs)
    actual, stats = fit(X, y, 1, **kwargs)
    assert_identical(expected, actual)
    assert all(item["tuning"]["histogram_missing"] == 0 for item in control_stats)
    assert all(item["tuning"]["histogram_missing"] == expected_active for item in stats)
    return stats


@pytest.mark.parametrize("kind", ["dense", "csr", "csc"])
@pytest.mark.parametrize("bins", [16, 256, 513])
@pytest.mark.parametrize("group", [1, 2, 4])
def test_sparse_layouts_bins_and_feature_groups_are_exact(kind, bins, group):
    X, y = inputs()
    stats = compare(layout(X, kind), y, bins=bins, group=group)
    actual_group = group if bins * group <= 1024 else 1
    assert stats[-1]["tuning"]["features_per_group"] == actual_group


@pytest.mark.parametrize("rows", [0, 1, 31, 32, 33, 255, 256, 257, 2049])
def test_empty_partial_simd_and_partial_histogram_tiles(rows):
    X, y = inputs(rows)
    if rows == 1:
        X.fill(np.nan)
    compare(X, y, expected_active=int(rows != 0), rounds=1,
            params={"metal_histogram_rows": 256})


@pytest.mark.parametrize("group", [1, 3, 4])
def test_entirely_missing_data(group):
    X = np.full((529, 7), np.nan, np.float32)
    y = np.linspace(-1, 1, len(X), dtype=np.float32)
    compare(X, y, group=group, rounds=2)


@pytest.mark.parametrize("kind", ["dense", "csr", "csc"])
def test_dense_data_uses_original_kernel(kind):
    X = np.arange(257 * 7, dtype=np.float32).reshape(-1, 7) / 100
    y = (X[:, 0] > 3).astype(np.float32)
    compare(layout(X, kind), y, expected_active=0, rounds=2)


@pytest.mark.parametrize("density", [0.25, 0.95])
@pytest.mark.parametrize("kind", ["dense", "csr"])
def test_quantile_matrix_preserves_missing_bins(density, kind):
    X, y = inputs(density=density)
    compare(layout(X, kind), y, quantile=True, rounds=2)


@pytest.mark.parametrize("missing", [0, -999, np.inf, -np.inf])
def test_explicit_missing_sentinels(missing):
    X, y = inputs(density=0.5)
    X[np.isnan(X)] = missing
    compare(X, y, metadata={"missing": missing}, rounds=2)


EXECUTIONS = [
    {"metal_gpu_tree": 0},
    {"metal_histogram_subtraction": 0},
    {"max_cached_hist_node": 1},
    {"grow_policy": "lossguide", "max_depth": 0, "max_leaves": 7},
    {"metal_partition_scan": 0, "metal_fused_partition": 0},
    {"metal_compute_fill": 0, "metal_coalesce_encoders": 0},
    {"subsample": 0.6},
    {"subsample": 0.6, "sampling_method": "gradient_based"},
    {"colsample_bytree": 0.7, "colsample_bylevel": 0.8, "colsample_bynode": 0.8},
    {"num_parallel_tree": 3},
]


@pytest.mark.parametrize("params", EXECUTIONS)
def test_partition_sampling_rebuild_and_encoder_paths(params):
    X, y = inputs(density=0.2)
    weights = np.linspace(0, 3, len(X), dtype=np.float32)
    compare(X, y, params=params, metadata={"weight": weights}, rounds=2)


@pytest.mark.parametrize("strategy", ["multiclass", "one_output_per_tree", "multi_output_tree"])
def test_multiple_gradient_groups(strategy):
    X, y = inputs(density=0.3)
    if strategy == "multiclass":
        y = (np.arange(len(X)) % 3).astype(np.float32)
        params = {"objective": "multi:softprob", "num_class": 3}
    else:
        y = np.column_stack([y, -y, y * 0.25]).astype(np.float32)
        params = {"multi_strategy": strategy}
    compare(X, y, params=params, rounds=2)


@pytest.mark.parametrize("group", [1, 4])
def test_signed_fixed_point_carries_and_cancellation(group):
    X, y = inputs(rows=2053, density=0.3)
    gradient = np.resize(np.asarray([2**40, -2**40, 3 * 2**-14, -0.25, 0], np.float32), len(X))
    hessian = np.resize(np.asarray([1, 2, 1, 0, 2**-20], np.float32), len(X))
    compare(X, y, group=group, rounds=2,
            custom=lambda prediction, matrix: (gradient, hessian))


def test_unsorted_sparse_duplicates_and_explicit_zeros():
    rows, columns = 257, 7
    indices = np.tile(np.asarray([4, 1, 4, 0], np.int32), rows)
    values = np.empty((rows, 4), np.float32)
    values[:, 0] = np.arange(rows) % 11
    values[:, 1] = 0
    values[:, 2] = -values[:, 0]
    values[::3, 2] = np.nan
    values[:, 3] = np.arange(rows) % 3
    X = sparse.csr_matrix((values.ravel(), indices, np.arange(rows + 1, dtype=np.int32) * 4),
                          shape=(rows, columns))
    # Compare the two kernels on the same duplicate semantics; do not normalize
    # sparse duplicates by summing them or discard explicit zero entries.
    compare(X, (np.arange(rows) % 5).astype(np.float32), rounds=2)


def test_categorical_missing_bins():
    X, y = inputs(density=0.3)
    present = ~np.isnan(X[:, 0])
    X[present, 0] = np.arange(np.count_nonzero(present)) % 5
    compare(X, y, metadata={"feature_types": ["c"] + ["q"] * 6},
            params={"max_cat_to_onehot": 2}, rounds=2)


def test_uint32_bin_storage_and_global_histogram_atomics():
    # histogram_bins is capped by row count: both must exceed 65534 to exercise
    # uint32 bins rather than merely requesting a large max_bin on tiny data.
    X = np.arange(65535, dtype=np.float32)[:, None]
    y = ((X[:, 0] % 31) > 15).astype(np.float32)
    X[::3] = np.nan
    compare(X, y, bins=65535, group=1, params={"max_depth": 1}, rounds=1)


def test_default_enablement_and_reused_model_toggle():
    X, y = inputs(density=0.4)
    dense = np.nan_to_num(X)
    variants = [(X, y), (dense[:513], y[:513]), (X[:257], y[:257]), (X, y)]
    results = []
    for enabled in [0, 1]:
        matrices = [xgb.DMatrix(values, label=labels) for values, labels in variants]
        model = xgb.Booster(dict(max_depth=3, max_bin=16, seed=17, base_score=0), matrices)
        for iteration, data in enumerate(matrices):
            # The first experimental update deliberately leaves the default on.
            if not enabled or iteration:
                model.set_param("metal_histogram_missing", enabled if iteration != 2 else 0)
            if iteration == 2:
                model.reset()
            model.update(data, iteration)
            active = enabled and iteration not in (1, 2)
            assert profile(model)["tuning"]["histogram_missing"] == int(active)
        results.append(snapshot(model, matrices))
    assert_identical(*results)
