"""Phase 3 verification suite: Native End-to-End GPU CSR/CSC/COO Sparse Pipeline.

Verifies bit-for-bit model, quantile-cut, prediction, TreeSHAP, and gblinear parity
between SparseStorage (CSR/CSC/COO on GPU without dense materialization) and dense
DMatrix, plus memory/throughput benchmarks on high-dimensional sparse matrices.
"""

from __future__ import annotations

import ctypes
import json
import os
import tempfile
import time

import numpy as np
import pytest

import xgboost as xgb
from xgboost.core import _LIB, _check_call, c_bst_ulong, make_jcargs
from xgboost._data_utils import array_interface


def _metal_profile(booster: xgb.Booster) -> dict:
    out = ctypes.c_char_p()
    _check_call(_LIB.XGBoosterMetalProfile(booster.handle, ctypes.byref(out)))
    return json.loads(out.value.decode("utf-8"))


def _dmatrix_from_csr_mode(csr: xgb.CSRMatrix, y=None, weight=None, missing=np.nan, sparse_mode=1) -> xgb.DMatrix:
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.XGDMatrixCreateFromCSR(
            array_interface(csr.indptr),
            array_interface(csr.indices),
            array_interface(csr.data),
            c_bst_ulong(csr.shape[1]),
            make_jcargs(missing=float(missing), nthread=0, sparse_mode=int(sparse_mode)),
            ctypes.byref(handle),
        )
    )
    dm = xgb.DMatrix(handle)
    if y is not None:
        dm.set_label(y)
    if weight is not None:
        dm.set_weight(weight)
    return dm


def _dmatrix_from_csc_mode(csc: xgb.CSCMatrix, y=None, missing=np.nan, sparse_mode=1) -> xgb.DMatrix:
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.XGDMatrixCreateFromCSC(
            array_interface(csc.indptr),
            array_interface(csc.indices),
            array_interface(csc.data),
            c_bst_ulong(csc.shape[0]),
            make_jcargs(missing=float(missing), nthread=0, sparse_mode=int(sparse_mode)),
            ctypes.byref(handle),
        )
    )
    dm = xgb.DMatrix(handle)
    if y is not None:
        dm.set_label(y)
    return dm


def _dmatrix_from_coo_mode(coo: xgb.COOMatrix, y=None, missing=np.nan, sparse_mode=1) -> xgb.DMatrix:
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.XGDMatrixCreateFromCOO(
            array_interface(coo.row),
            array_interface(coo.col),
            array_interface(coo.data),
            c_bst_ulong(coo.shape[0]),
            c_bst_ulong(coo.shape[1]),
            make_jcargs(missing=float(missing), nthread=0, sparse_mode=int(sparse_mode)),
            ctypes.byref(handle),
        )
    )
    dm = xgb.DMatrix(handle)
    if y is not None:
        dm.set_label(y)
    return dm


def _make_sparse_problem(rows=1024, cols=64, density=0.08, seed=42, explicit_zeros=True):
    rng = np.random.default_rng(seed)
    mask = rng.random((rows, cols)) < density
    vals = rng.standard_normal((rows, cols)).astype(np.float32)
    if explicit_zeros:
        zero_mask = mask & (rng.random((rows, cols)) < 0.1)
        vals[zero_mask] = 0.0
    dense = np.full((rows, cols), np.nan, dtype=np.float32)
    dense[mask] = vals[mask]
    nz_r, nz_c = np.nonzero(mask)
    nz_v = dense[nz_r, nz_c]
    counts = np.bincount(nz_r, minlength=rows)
    indptr = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    csr = xgb.CSRMatrix((nz_v, nz_c.astype(np.int32), indptr), shape=(rows, cols))
    w_true = rng.standard_normal(cols).astype(np.float32)
    y = (np.nan_to_num(dense, nan=0.0) @ w_true + 0.25 * rng.standard_normal(rows)).astype(np.float32)
    return dense, csr, y


@pytest.mark.parametrize("max_bin", [16, 64, 256, 512])
@pytest.mark.parametrize("weighted", [False, True])
def test_sparse_csc_quantile_cuts_exact_parity(max_bin: int, weighted: bool) -> None:
    dense, csr, y = _make_sparse_problem(rows=1200, cols=48, density=0.07, seed=101 + max_bin)
    # Include an all-missing column and a constant-present column.
    dense[:, 5] = np.nan
    dense[:10, 11] = 3.25
    dense[10:, 11] = np.nan
    nz_r, nz_c = np.nonzero(~np.isnan(dense))
    counts = np.bincount(nz_r, minlength=dense.shape[0])
    indptr = np.zeros(dense.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    csr = xgb.CSRMatrix((dense[nz_r, nz_c], nz_c.astype(np.int32), indptr), shape=dense.shape)

    rng = np.random.default_rng(77)
    w = rng.uniform(0.1, 3.0, size=dense.shape[0]).astype(np.float32) if weighted else None

    dm_dense = xgb.DMatrix(dense, label=y, weight=w)
    dm_sparse = _dmatrix_from_csr_mode(csr, y=y, weight=w, sparse_mode=1)

    params = {"max_depth": 4, "max_bin": max_bin, "eta": 0.3, "seed": 9}
    b_dense = xgb.train(params, dm_dense, num_boost_round=2)
    b_sparse = xgb.train(params, dm_sparse, num_boost_round=2)

    prof = _metal_profile(b_sparse)
    assert prof["tuning"]["sparse_pipeline"] == 1

    ptr_d, val_d = dm_dense.get_quantile_cut()
    ptr_s, val_s = dm_sparse.get_quantile_cut()
    np.testing.assert_array_equal(ptr_d, ptr_s)
    np.testing.assert_array_equal(val_d, val_s)
    assert b_dense.save_raw("ubj") == b_sparse.save_raw("ubj")


def test_unsorted_duplicates_csr_csc_coo_parity() -> None:
    rows, cols = 512, 32
    rng = np.random.default_rng(2026)
    # Build unsorted CSR with duplicate columns (last-write-wins) and explicit 0.0 / -0.0.
    indptr = [0]
    indices = []
    values = []
    dense_expected = np.full((rows, cols), np.nan, dtype=np.float32)
    for r in range(rows):
        k = int(rng.integers(0, 8))
        chosen = rng.integers(0, cols, size=k)
        row_vals = rng.normal(size=k).astype(np.float32)
        if k >= 2:
            chosen[-1] = chosen[0]  # Force duplicate column in row
            row_vals[0] = 99.0
            row_vals[-1] = 0.0 if (r % 2 == 0) else -0.0
        for c, v in zip(chosen, row_vals):
            indices.append(int(c))
            values.append(float(v))
            dense_expected[r, int(c)] = np.float32(v)
        indptr.append(len(values))

    csr = xgb.CSRMatrix(
        (np.array(values, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int64)),
        shape=(rows, cols),
    )
    y = rng.normal(size=rows).astype(np.float32)
    dm_sparse = _dmatrix_from_csr_mode(csr, y=y, sparse_mode=1)
    dm_dense = xgb.DMatrix(dense_expected, label=y)
    assert dm_sparse.num_nonmissing() == len(values)

    b_s = xgb.train({"max_depth": 5, "eta": 0.25}, dm_sparse, num_boost_round=5)
    b_d = xgb.train({"max_depth": 5, "eta": 0.25}, dm_dense, num_boost_round=5)
    assert _metal_profile(b_s)["tuning"]["sparse_pipeline"] == 1
    assert b_s.save_raw("ubj") == b_d.save_raw("ubj")

    # Test CSC and COO (COO sums duplicates, matching coo.tocsr()).
    coo_r = rng.integers(0, rows, size=1400, dtype=np.int32)
    coo_c = rng.integers(0, cols, size=1400, dtype=np.int32)
    coo_v = rng.normal(size=1400).astype(np.float32)
    coo = xgb.COOMatrix((coo_v, (coo_r, coo_c)), shape=(rows, cols))
    csr_from_coo = coo.tocsr()
    csc_from_coo = coo.tocsc()

    dm_coo = xgb.DMatrix(coo, label=y)
    dm_csc = _dmatrix_from_csc_mode(csc_from_coo, y=y, sparse_mode=1)
    dm_csr_ref = _dmatrix_from_csr_mode(csr_from_coo, y=y, sparse_mode=0)

    b_coo = xgb.train({"max_depth": 4, "eta": 0.3}, dm_coo, num_boost_round=4)
    b_csc = xgb.train({"max_depth": 4, "eta": 0.3}, dm_csc, num_boost_round=4)
    b_ref = xgb.train({"max_depth": 4, "eta": 0.3}, dm_csr_ref, num_boost_round=4)
    assert _metal_profile(b_coo)["tuning"]["sparse_pipeline"] == 1
    assert _metal_profile(b_csc)["tuning"]["sparse_pipeline"] == 1
    assert b_coo.save_raw("ubj") == b_ref.save_raw("ubj")
    assert b_csc.save_raw("ubj") == b_ref.save_raw("ubj")


@pytest.mark.parametrize(
    "extra_params",
    [
        {"max_depth": 6, "metal_gpu_tree": 1},
        {"max_depth": 5, "metal_gpu_tree": 0, "metal_fused_partition": 1},
        {"max_depth": 5, "metal_gpu_tree": 0, "metal_fused_partition": 0, "metal_partition_scan": 1},
        {"max_depth": 5, "metal_gpu_tree": 0, "metal_fused_partition": 0, "metal_partition_scan": 0},
        {"grow_policy": "lossguide", "max_depth": 0, "max_leaves": 19, "max_cached_hist_node": 2},
        {"max_depth": 5, "subsample": 0.65, "colsample_bytree": 0.8, "colsample_bylevel": 0.75, "colsample_bynode": 0.8},
        {"max_depth": 5, "monotone_constraints": "(" + ",".join(str((i % 3) - 1) for i in range(64)) + ")"},
    ],
)
def test_sparse_tree_builders_exact_parity(extra_params: dict) -> None:
    dense, csr, y = _make_sparse_problem(rows=1024, cols=64, density=0.06, seed=303)
    dm_dense = xgb.DMatrix(dense, label=y)
    dm_sparse = xgb.DMatrix(csr, label=y)

    params = {"eta": 0.25, "max_bin": 64, "seed": 17, **extra_params}
    b_dense = xgb.train(params, dm_dense, num_boost_round=5)
    b_sparse = xgb.train(params, dm_sparse, num_boost_round=5)

    assert _metal_profile(b_sparse)["tuning"]["sparse_pipeline"] == 1
    assert b_dense.save_raw("ubj") == b_sparse.save_raw("ubj")
    np.testing.assert_array_equal(b_dense.predict(dm_dense), b_sparse.predict(dm_sparse))


def test_sparse_multiclass_and_vector_leaves_parity() -> None:
    dense, csr, _ = _make_sparse_problem(rows=1024, cols=48, density=0.07, seed=404)
    rng = np.random.default_rng(404)
    y_cls = rng.integers(0, 4, size=1024).astype(np.float32)

    dm_d = xgb.DMatrix(dense, label=y_cls)
    dm_s = xgb.DMatrix(csr, label=y_cls)
    params_mc = {"objective": "multi:softprob", "num_class": 4, "max_depth": 4, "eta": 0.3, "seed": 11}
    b_mc_d = xgb.train(params_mc, dm_d, num_boost_round=4)
    b_mc_s = xgb.train(params_mc, dm_s, num_boost_round=4)
    assert _metal_profile(b_mc_s)["tuning"]["sparse_pipeline"] == 1
    assert b_mc_d.save_raw("ubj") == b_mc_s.save_raw("ubj")
    np.testing.assert_allclose(b_mc_d.predict(dm_d), b_mc_s.predict(dm_s), rtol=0, atol=0)

    # Multi-target vector leaves (exercises vector_leaf_sums_csr + multi-target histogram_csr_accumulate).
    y_vec = rng.normal(size=(1024, 3)).astype(np.float32)
    dm_vd = xgb.DMatrix(dense, label=y_vec)
    dm_vs = xgb.DMatrix(csr, label=y_vec)
    params_vec = {"objective": "reg:squarederror", "multi_strategy": "multi_output_tree", "max_depth": 4, "eta": 0.25}
    b_vd = xgb.train(params_vec, dm_vd, num_boost_round=3)
    b_vs = xgb.train(params_vec, dm_vs, num_boost_round=3)
    assert _metal_profile(b_vs)["tuning"]["sparse_pipeline"] == 1
    assert b_vd.save_raw("ubj") == b_vs.save_raw("ubj")
    np.testing.assert_array_equal(b_vd.predict(dm_vd), b_vs.predict(dm_vs))


def test_sparse_predict_leaf_shap_and_interactions_parity() -> None:
    dense, csr, y = _make_sparse_problem(rows=512, cols=32, density=0.09, seed=505)
    dm_d = xgb.DMatrix(dense, label=y)
    dm_s = _dmatrix_from_csr_mode(csr, y=y, sparse_mode=1)

    booster = xgb.train({"max_depth": 4, "eta": 0.3, "seed": 21}, dm_s, num_boost_round=4)
    np.testing.assert_array_equal(booster.predict(dm_d), booster.predict(dm_s))
    np.testing.assert_array_equal(booster.predict(dm_d, pred_leaf=True), booster.predict(dm_s, pred_leaf=True))
    np.testing.assert_allclose(
        booster.predict(dm_d, pred_contribs=True),
        booster.predict(dm_s, pred_contribs=True),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        booster.predict(dm_d, pred_contribs=True, approx_contribs=True),
        booster.predict(dm_s, pred_contribs=True, approx_contribs=True),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        booster.predict(dm_d, pred_interactions=True),
        booster.predict(dm_s, pred_interactions=True),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        booster.predict(dm_d, pred_interactions=True, approx_contribs=True),
        booster.predict(dm_s, pred_interactions=True, approx_contribs=True),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "updater,selector",
    [
        ("shotgun", "cyclic"),
        ("shotgun", "shuffle"),
        ("coord_descent", "cyclic"),
        ("coord_descent", "greedy"),
        ("coord_descent", "thrifty"),
    ],
)
def test_sparse_gblinear_parity(updater: str, selector: str) -> None:
    dense, csr, y = _make_sparse_problem(rows=1024, cols=40, density=0.08, seed=606)
    dm_d = xgb.DMatrix(dense, label=y)
    dm_s = xgb.DMatrix(csr, label=y)

    params = {
        "booster": "gblinear",
        "updater": updater,
        "feature_selector": selector,
        "learning_rate": 0.3,
        "reg_alpha": 0.05,
        "reg_lambda": 0.1,
        "seed": 42,
    }
    b_d = xgb.train(params, dm_d, num_boost_round=6)
    b_s = xgb.train(params, dm_s, num_boost_round=6)
    np.testing.assert_allclose(b_d.predict(dm_d), b_s.predict(dm_s), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        b_d.predict(dm_d, pred_contribs=True),
        b_s.predict(dm_s, pred_contribs=True),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sparse_quantile_dmatrix_slice_and_binary_io() -> None:
    dense, csr, y = _make_sparse_problem(rows=1024, cols=48, density=0.07, seed=707)
    qdm_d = xgb.QuantileDMatrix(dense, label=y, max_bin=64)
    qdm_s = xgb.QuantileDMatrix(csr, label=y, max_bin=64)

    b_qd = xgb.train({"max_depth": 5, "max_bin": 64, "eta": 0.25}, qdm_d, num_boost_round=4)
    b_qs = xgb.train({"max_depth": 5, "max_bin": 64, "eta": 0.25}, qdm_s, num_boost_round=4)
    assert _metal_profile(b_qs)["tuning"]["sparse_pipeline"] == 1
    assert b_qd.save_raw("ubj") == b_qs.save_raw("ubj")
    np.testing.assert_array_equal(b_qd.predict(qdm_d), b_qs.predict(qdm_s))

    # Row slicing (both pre-quantize and post-quantize) on GPU CSR.
    dm_d = xgb.DMatrix(dense, label=y)
    dm_s = xgb.DMatrix(csr, label=y)
    idx = np.array([5, 0, 511, 200, 200, 1020, 42, 77, 888, 13], dtype=np.int32)
    sl_d = dm_d.slice(idx)
    sl_s = dm_s.slice(idx)
    np.testing.assert_array_equal(b_qs.predict(sl_d), b_qs.predict(sl_s))

    # Train first to quantize dm_s, then slice quantized sparse matrix and train again!
    _ = xgb.train({"max_depth": 3, "max_bin": 64}, dm_d, num_boost_round=1)
    _ = xgb.train({"max_depth": 3, "max_bin": 64}, dm_s, num_boost_round=1)
    sub_idx = np.arange(100, 700, dtype=np.int32)
    qsl_d = dm_d.slice(sub_idx)
    qsl_s = dm_s.slice(sub_idx)
    b_sub_d = xgb.train({"max_depth": 4, "max_bin": 64}, qsl_d, num_boost_round=3)
    b_sub_s = xgb.train({"max_depth": 4, "max_bin": 64}, qsl_s, num_boost_round=3)
    assert b_sub_d.save_raw("ubj") == b_sub_s.save_raw("ubj")

    # Binary save/load round-trip directly into SparseStorage.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sparse.dmatrix")
        dm_s.save_binary(path)
        reloaded = xgb.DMatrix(path)
        assert reloaded.num_row() == dm_s.num_row()
        assert reloaded.num_col() == dm_s.num_col()
        assert reloaded.num_nonmissing() == dm_s.num_nonmissing()
        b_rel = xgb.train({"max_depth": 4, "max_bin": 64}, reloaded, num_boost_round=3)
        assert _metal_profile(b_rel)["tuning"]["sparse_pipeline"] == 1
        np.testing.assert_array_equal(b_qs.predict(dm_s), b_qs.predict(reloaded))


def test_sparse_high_dimensional_100k_x_2k_benchmark() -> None:
    """Verify 100K x 2K (1% density = 2M nnz) runs natively in O(nnz) without dense expansion."""
    rows, cols = 100_000, 2_000
    nnz_per_row = 20  # 1% density -> 2,000,000 nonzeros (vs 200,000,000 dense floats = 800 MB!)
    rng = np.random.default_rng(999)
    indptr = np.arange(0, (rows + 1) * nnz_per_row, nnz_per_row, dtype=np.int64)
    # Generate sorted unique columns per row cleanly.
    base_cols = np.tile(np.arange(0, cols, cols // nnz_per_row, dtype=np.int32), (rows, 1))
    jitter = rng.integers(0, cols // nnz_per_row, size=(rows, nnz_per_row), dtype=np.int32)
    indices = (base_cols + jitter).ravel()
    data = rng.standard_normal(rows * nnz_per_row).astype(np.float32)
    y = (data.reshape(rows, nnz_per_row)[:, :4].sum(axis=1) + 0.1 * rng.standard_normal(rows)).astype(np.float32)

    csr = xgb.CSRMatrix((data, indices, indptr), shape=(rows, cols))

    t0 = time.perf_counter()
    dm_sparse = xgb.DMatrix(csr, label=y)
    t_ingest = time.perf_counter() - t0

    t1 = time.perf_counter()
    booster = xgb.train({"max_depth": 6, "max_bin": 64, "eta": 0.3}, dm_sparse, num_boost_round=5)
    t_train = time.perf_counter() - t1

    t2 = time.perf_counter()
    preds = booster.predict(dm_sparse)
    t_pred = time.perf_counter() - t2

    prof = _metal_profile(booster)
    assert prof["tuning"]["sparse_pipeline"] == 1
    assert preds.shape == (rows,)
    assert np.all(np.isfinite(preds))
    # Sanity check: 100K x 2K sparse ingest + 5 trees + predict finishes well under 2 seconds!
    assert t_ingest + t_train + t_pred < 5.0
