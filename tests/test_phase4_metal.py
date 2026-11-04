"""Phase 4 verification suite:
- M5a: Threadgroup-cached shallow-tree GPU inference (dense & CSR, numeric & categorical)
- M5b: Incremental GPU SHAP upload + register-resident TreeSHAP & cached interactions
- M5c: Fused GPU gblinear coordinate descent + GPU thrifty argsort
- M6 + D2 + D3: MetalArray (16 KiB unified memory), ingest_strided_typed (all dtypes,
  IEEE-754 RNE, Fortran/negative strides, validity bitmaps), Arrow C Data Interface,
  DLPack, predict_to_buffer, GPU CV folds, and zero-NumPy + zero-SciPy subprocess tests.
"""

from __future__ import annotations

import array
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import xgboost as xgb
from xgboost._c_api import _LIB, _check_call
from xgboost._metal_array import (
    INGEST_F32,
    INGEST_F64,
    INGEST_MASK_VALID_BITS,
    PAGE_SIZE,
    XGBufferDescriptor,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# M5a: Threadgroup-Cached Shallow-Tree Prediction (Dense & CSR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_depth", [1, 3, 5, 6, 8])
def test_m5a_shallow_vs_deep_tree_prediction_dense_and_csr(max_depth: int) -> None:
    rng = np.random.default_rng(42 + max_depth)
    X = rng.normal(size=(512, 18)).astype(np.float32)
    X[rng.random(X.shape) < 0.12] = np.nan
    y = (
        np.nan_to_num(X[:, 0], nan=-0.25) * 1.5
        - np.nan_to_num(X[:, 3], nan=0.5)
        + rng.normal(scale=0.2, size=512)
    ).astype(np.float32)

    dmat = xgb.DMatrix(X, label=y)
    bst = xgb.train(
        {"max_depth": max_depth, "eta": 0.25, "subsample": 0.85, "seed": 7},
        dmat,
        num_boost_round=12,
    )

    p_dmat = bst.predict(dmat)
    p_inplace = bst.inplace_predict(X)
    np.testing.assert_allclose(p_dmat, p_inplace, rtol=1e-6, atol=1e-6)

    # Partial iteration range & raw margin
    p_sub = bst.predict(dmat, iteration_range=(2, 9), output_margin=True)
    p_sub_in = bst.inplace_predict(X, iteration_range=(2, 9), predict_type="margin")
    np.testing.assert_allclose(p_sub, p_sub_in, rtol=1e-6, atol=1e-6)

    # Sparse CSR parity (missing=NaN -> explicit zeros vs missing entries)
    X_zero_sparse = rng.normal(size=(384, 18)).astype(np.float32)
    X_zero_sparse[rng.random(X_zero_sparse.shape) < 0.75] = 0.0
    csr = xgb.CSRMatrix(X_zero_sparse)
    dm_csr = xgb.DMatrix(csr)
    dm_dense_missing0 = xgb.DMatrix(X_zero_sparse, missing=0.0)
    np.testing.assert_allclose(
        bst.predict(dm_csr),
        bst.predict(dm_dense_missing0),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        bst.inplace_predict(csr),
        bst.predict(dm_csr),
        rtol=1e-6,
        atol=1e-6,
    )


def test_m5a_cached_tree_categorical_and_multiclass() -> None:
    rng = np.random.default_rng(101)
    n = 320
    cats = rng.integers(0, 12, size=(n, 2)).astype(np.float32)
    nums = rng.normal(size=(n, 4)).astype(np.float32)
    X = np.hstack([cats, nums])
    y = ((cats[:, 0].astype(int) + (nums[:, 0] > 0).astype(int)) % 3).astype(np.float32)

    ftypes = ["c", "c", "q", "q", "q", "q"]
    dm = xgb.DMatrix(X, label=y, feature_types=ftypes, enable_categorical=True)
    bst = xgb.train(
        {
            "max_depth": 4,
            "eta": 0.3,
            "objective": "multi:softprob",
            "num_class": 3,
            "max_cat_to_onehot": 1,
        },
        dm,
        num_boost_round=8,
    )
    probs = bst.predict(dm)
    assert probs.shape == (n, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(n, dtype=np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(probs, bst.inplace_predict(X), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# M5b: Incremental GPU SHAP Upload & Fast Register-Resident TreeSHAP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_cols", [14, 140])
def test_m5b_incremental_shap_and_interactions_dense_and_csr(n_cols: int) -> None:
    rng = np.random.default_rng(202 + n_cols)
    n_rows = 96 if n_cols > 50 else 192
    X = rng.normal(size=(n_rows, n_cols)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    y = (
        np.nan_to_num(X[:, 0], nan=0.2)
        - 0.8 * np.nan_to_num(X[:, 1], nan=0.0)
        + rng.normal(scale=0.1, size=n_rows)
    ).astype(np.float32)

    dm = xgb.DMatrix(X, label=y)
    bst = xgb.train({"max_depth": 5, "eta": 0.25, "seed": 11}, dm, num_boost_round=5)

    # First SHAP query populates Booster GPU SHAP cache for trees [0..5)
    margin_5 = bst.predict(dm, output_margin=True)
    contribs_5 = bst.predict(dm, pred_contribs=True)
    assert contribs_5.shape == (n_rows, n_cols + 1)
    np.testing.assert_allclose(contribs_5.sum(axis=1), margin_5, rtol=1e-5, atol=1e-5)

    if n_cols <= 20:
        inter_5 = bst.predict(dm, pred_interactions=True)
        assert inter_5.shape == (n_rows, n_cols + 1, n_cols + 1)
        np.testing.assert_allclose(inter_5.sum(axis=2), contribs_5, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(inter_5.sum(axis=1), contribs_5, rtol=1e-5, atol=1e-5)

    # Continue training ON THE SAME BOOSTER HANDLE to exercise incremental upload_shap()
    bst = xgb.train({"max_depth": 6, "eta": 0.2, "seed": 12}, dm, num_boost_round=4, xgb_model=bst)
    assert bst.num_boosted_rounds() == 9

    margin_9 = bst.predict(dm, output_margin=True)
    contribs_9 = bst.predict(dm, pred_contribs=True)
    np.testing.assert_allclose(contribs_9.sum(axis=1), margin_9, rtol=1e-5, atol=1e-5)

    # Partial iteration_range (0, 7) and sliced model [2:7] must match exact margins
    margin_prefix = bst.predict(dm, output_margin=True, iteration_range=(0, 7))
    contribs_prefix = bst.predict(dm, pred_contribs=True, iteration_range=(0, 7))
    np.testing.assert_allclose(contribs_prefix.sum(axis=1), margin_prefix, rtol=1e-5, atol=1e-5)

    bst_sliced = bst[2:7]
    np.testing.assert_allclose(
        bst_sliced.predict(dm, pred_contribs=True).sum(axis=1),
        bst_sliced.predict(dm, output_margin=True),
        rtol=1e-5,
        atol=1e-5,
    )

    if n_cols <= 20:
        inter_9 = bst.predict(dm, pred_interactions=True)
        np.testing.assert_allclose(inter_9.sum(axis=2), contribs_9, rtol=1e-5, atol=1e-5)

        # CSR sparse TreeSHAP parity against dense missing=0.0
        X_sp = rng.normal(size=(80, n_cols)).astype(np.float32)
        X_sp[rng.random(X_sp.shape) < 0.7] = 0.0
        dm_csr = xgb.DMatrix(xgb.CSRMatrix(X_sp))
        dm_d0 = xgb.DMatrix(X_sp, missing=0.0)
        np.testing.assert_allclose(
            bst.predict(dm_csr, pred_contribs=True),
            bst.predict(dm_d0, pred_contribs=True),
            rtol=1e-5,
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# M5c: Fused GPU gblinear Coordinate Descent & GPU Thrifty Sort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("selector", ["cyclic", "shuffle", "random", "thrifty"])
@pytest.mark.parametrize("is_sparse", [False, True])
@pytest.mark.parametrize("n_rows", [384, 2304])
def test_m5c_gblinear_fused_vs_unfused_parity(selector: str, is_sparse: bool, n_rows: int) -> None:
    rng = np.random.default_rng(303 + n_rows)
    n, p = n_rows, 24
    X = rng.normal(size=(n, p)).astype(np.float32)
    if is_sparse:
        X[rng.random(X.shape) < 0.65] = 0.0
    true_w = rng.normal(size=p).astype(np.float32)
    y = (X @ true_w + 0.5 + rng.normal(scale=0.1, size=n)).astype(np.float32)
    weights = rng.uniform(0.5, 2.0, size=n).astype(np.float32)

    dm = xgb.DMatrix(xgb.CSRMatrix(X) if is_sparse else X, label=y, weight=weights)
    base_params = {
        "booster": "gblinear",
        "updater": "coord_descent",
        "objective": "reg:squarederror",
        "eta": 0.2,
        "alpha": 0.05,
        "lambda": 0.5,
        "feature_selector": selector,
        "top_k": 8 if selector == "thrifty" else 0,
        "seed": 99,
    }
    bst_fused = xgb.train(
        {**base_params, "metal_linear_fused": 1, "metal_linear_gpu_sort": 1},
        dm,
        num_boost_round=10,
    )
    bst_unfused = xgb.train(
        {**base_params, "metal_linear_fused": 0, "metal_linear_gpu_sort": 0},
        dm,
        num_boost_round=10,
    )

    np.testing.assert_allclose(
        bst_fused.predict(dm),
        bst_unfused.predict(dm),
        rtol=1e-4,
        atol=1e-4,
    )


def test_m5c_gblinear_multiclass_and_narrower_continuation() -> None:
    rng = np.random.default_rng(404)
    X = rng.normal(size=(256, 12)).astype(np.float32)
    y = (np.argmax(X[:, :3], axis=1)).astype(np.float32)
    dm = xgb.DMatrix(X, label=y)
    params = {
        "booster": "gblinear",
        "updater": "coord_descent",
        "objective": "multi:softprob",
        "num_class": 3,
        "eta": 0.25,
        "feature_selector": "thrifty",
        "lambda": 0.2,
    }
    b1 = xgb.train({**params, "metal_linear_fused": 1}, dm, num_boost_round=8)
    b0 = xgb.train({**params, "metal_linear_fused": 0, "metal_linear_gpu_sort": 0}, dm, num_boost_round=8)
    np.testing.assert_allclose(b1.predict(dm), b0.predict(dm), rtol=1e-4, atol=1e-4)

    # Continuation training on the same booster + narrower-DMatrix prediction & guard
    b_cont = xgb.train({**params, "metal_linear_fused": 1}, dm, num_boost_round=3, xgb_model=b1)
    b_full = xgb.train({**params, "metal_linear_fused": 1}, dm, num_boost_round=11)
    np.testing.assert_allclose(b_cont.predict(dm), b_full.predict(dm), rtol=1e-4, atol=1e-4)

    dm_narrow = xgb.DMatrix(X[:, :8])
    X_padded = np.hstack([X[:, :8], np.zeros((256, 4), dtype=np.float32)])
    np.testing.assert_allclose(
        b1.predict(dm_narrow),
        b1.predict(xgb.DMatrix(X_padded)),
        rtol=1e-5,
        atol=1e-5,
    )
    with pytest.raises(xgb.core.XGBoostError):
        xgb.train({**params, "metal_linear_fused": 1}, xgb.DMatrix(X[:, :8], label=y), num_boost_round=1, xgb_model=b1)


# ---------------------------------------------------------------------------
# M6 + D2 + D3: MetalArray, Zero-Copy, Strided/Typed Ingest, Arrow & DLPack
# ---------------------------------------------------------------------------


def test_m6_metal_array_alignment_zero_copy_and_protocols() -> None:
    arr = xgb.MetalArray.from_list([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    assert arr.data_ptr % PAGE_SIZE == 0
    assert arr.is_page_aligned

    dm = xgb.DMatrix(arr, label=xgb.MetalArray.from_list([0.0, 1.0, 0.0]))
    assert dm.was_zero_copied is True
    assert dm.num_row() == 3 and dm.num_col() == 4

    # Sub-row view of MetalArray shares page-aligned mmap backing -> still zero-copied!
    sub = arr[1:]
    assert sub.shape == (2, 4)
    dm_sub = xgb.DMatrix(sub)
    assert dm_sub.was_zero_copied is True

    # Zero-copy __array_interface__ & __dlpack__ round-trip with NumPy
    np_view = np.asarray(arr)
    assert np_view.ctypes.data == arr.data_ptr
    dl_view = np.from_dlpack(arr)
    assert dl_view.ctypes.data == arr.data_ptr
    np.testing.assert_array_equal(np_view, dl_view)

    # Large NumPy array (>= 64 KiB) also zero-copies via VM page-spanning
    big_np = np.ones((4096, 8), dtype=np.float32)
    dm_big = xgb.DMatrix(big_np)
    assert dm_big.was_zero_copied is True


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.bool_,
    ],
)
def test_m6_ingest_all_dtypes_and_strided_views(dtype: np.dtype) -> None:
    rng = np.random.default_rng(505)
    if dtype == np.bool_:
        base = rng.integers(0, 2, size=(48, 12)).astype(np.bool_)
    elif np.issubdtype(dtype, np.unsignedinteger):
        base = rng.integers(0, 100, size=(48, 12)).astype(dtype)
    elif np.issubdtype(dtype, np.signedinteger):
        base = rng.integers(-50, 50, size=(48, 12)).astype(dtype)
    else:
        base = rng.normal(size=(48, 12)).astype(dtype)

    expected = base.astype(np.float32)
    dm_c = xgb.DMatrix(base)
    np.testing.assert_allclose(dm_c.get_data().toarray(), expected, rtol=0, atol=0)

    # Fortran-order (column-major) strided view
    f_arr = np.asfortranarray(base)
    dm_f = xgb.DMatrix(f_arr)
    np.testing.assert_allclose(dm_f.get_data().toarray(), expected, rtol=0, atol=0)

    # Non-contiguous row + negative column stride view!
    strided = base[::2, ::-2]
    dm_s = xgb.DMatrix(strided)
    np.testing.assert_allclose(dm_s.get_data().toarray(), expected[::2, ::-2], rtol=0, atol=0)


def test_m6_f64_and_i64_ieee754_rne_exact_rounding() -> None:
    # Craft float64 values around exact halfway ties, subnormals, max finite f32, NaN, and custom missing
    vals_f64 = np.array(
        [
            [1.0 + 2.0**-24, 1.0 + 3.0 * 2.0**-24, -1.0 - 2.0**-24, 1e-40],
            [1e-45, 3.4028234e38, np.nan, -999.0],
        ],
        dtype=np.float64,
    )
    dm = xgb.DMatrix(vals_f64, missing=-999.0)
    got = dm.get_data().toarray()
    # Row 0 must match hardware IEEE-754 float64 -> float32 RNE bit-for-bit!
    np.testing.assert_array_equal(got[0], vals_f64[0].astype(np.float32))
    assert dm.num_nonmissing() == 6  # NaN and -999.0 both treated as missing

    # Finite float64 above FLT_MAX (rounds to +Inf under RNE) must be rejected by GPU validation
    overflow_f64 = np.array([[1.0, 3.4028236e38]], dtype=np.float64)
    with pytest.raises(xgb.core.XGBoostError):
        xgb.DMatrix(overflow_f64)

    # Large int64 / uint64 exact RNE rounding (> 2^24)
    vals_i64 = np.array(
        [[16777217, 16777219, -16777219, (1 << 40) + (1 << 16)]],
        dtype=np.int64,
    )
    dm_i64 = xgb.DMatrix(vals_i64)
    np.testing.assert_array_equal(dm_i64.get_data().toarray(), vals_i64.astype(np.float32))


def test_m6_arrow_zero_json_c_data_interface_with_nulls_and_sliced_batches() -> None:
    table = pa.table(
        {
            "f_f64": pa.array([1.25, None, 3.5, 4.75, -2.0, 6.0], type=pa.float64()),
            "f_i32": pa.array([10, 20, None, 40, 50, 60], type=pa.int32()),
            "f_bool": pa.array([True, False, True, None, False, True], type=pa.bool_()),
            "f_f16": pa.array([np.float16(0.5), np.float16(1.5), np.float16(2.5), np.float16(3.5), None, np.float16(5.5)], type=pa.float16()),
        }
    )
    dm_tab = xgb.DMatrix(table)
    assert dm_tab.feature_names == ["f_f64", "f_i32", "f_bool", "f_f16"]
    assert dm_tab.feature_types == ["float", "int", "i", "float"]
    assert dm_tab.num_row() == 6 and dm_tab.num_col() == 4
    assert dm_tab.num_nonmissing() == 24 - 4

    # Sliced RecordBatch with non-zero offset (tests mask_bit_offset & bit-packed bool offset!)
    rb = table.to_batches()[0].slice(1, 4)
    dm_rb = xgb.DMatrix(rb)
    assert dm_rb.num_row() == 4 and dm_rb.num_col() == 4
    expected_dense = table.slice(1, 4).to_pandas().to_numpy(dtype=np.float32, na_value=np.nan)
    dm_ref = xgb.DMatrix(expected_dense)
    np.testing.assert_allclose(dm_rb.get_data().toarray(), dm_ref.get_data().toarray(), rtol=0, atol=0)


def test_m6_columnar_buffers_c_abi_and_dlpack_and_predict_to_buffer() -> None:
    # Direct XGDMatrixCreateFromColumnarBuffers with validity bitmap
    col0 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    col1 = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    # Valid bits: 0b1011 = rows 0, 1, 3 valid; row 2 null!
    mask = np.array([0b00001011], dtype=np.uint8)

    descs = (XGBufferDescriptor * 2)()
    descs[0].data = col0.ctypes.data
    descs[0].ndim = 1
    descs[0].shape[0] = 4
    descs[0].strides[0] = 4
    descs[0].dtype_code = INGEST_F32
    descs[0].mask_data = mask.ctypes.data
    descs[0].mask_bit_offset = 0
    descs[0].mask_mode = INGEST_MASK_VALID_BITS

    descs[1].data = col1.ctypes.data
    descs[1].ndim = 1
    descs[1].shape[0] = 4
    descs[1].strides[0] = 8
    descs[1].dtype_code = INGEST_F64

    hdl = ctypes.c_void_p()
    _check_call(
        _LIB.XGDMatrixCreateFromColumnarBuffers(
            descs, ctypes.c_uint64(2), ctypes.c_float(np.nan), ctypes.byref(hdl)
        )
    )
    dm_col = xgb.DMatrix(hdl)
    assert dm_col.num_row() == 4 and dm_col.num_col() == 2
    assert dm_col.num_nonmissing() == 7

    # DLPack capsule & object ingestion (including transposed strided DLPack!)
    m_arr = xgb.MetalArray.from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    dm_dl1 = xgb.DMatrix(m_arr.__dlpack__())
    np.testing.assert_allclose(dm_dl1.get_data().toarray(), np.asarray(m_arr))

    m_T = m_arr.T
    dm_dl_T = xgb.DMatrix(m_T.__dlpack__())
    np.testing.assert_allclose(dm_dl_T.get_data().toarray(), np.asarray(m_arr).T)

    # predict_to_buffer into MetalArray, np.ndarray, and array.array('f')
    y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    dm_dl_T.set_label(y)
    bst = xgb.train({"max_depth": 2, "eta": 0.3}, dm_dl_T, num_boost_round=4)
    ref_pred = bst.predict(dm_dl_T)

    out_metal = xgb.MetalArray.empty((3,), dtype="float32")
    assert bst.predict_to_buffer(dm_dl_T, out_metal) == (3,)
    np.testing.assert_allclose(np.asarray(out_metal), ref_pred, rtol=1e-6, atol=1e-6)

    out_arr = array.array("f", [0.0, 0.0, 0.0])
    assert bst.predict_to_buffer(dm_dl_T, out_arr) == (3,)
    np.testing.assert_allclose(np.asarray(out_arr), ref_pred, rtol=1e-6, atol=1e-6)

    too_small = xgb.MetalArray.empty((2,), dtype="float32")
    with pytest.raises(xgb.core.XGBoostError):
        bst.predict_to_buffer(dm_dl_T, too_small)


def test_d3_gpu_cv_folds_stratified_and_ranking_groups() -> None:
    rng = np.random.default_rng(606)
    X = rng.normal(size=(120, 6)).astype(np.float32)
    y_bin = np.array([0.0] * 90 + [1.0] * 30, dtype=np.float32)
    X[90:, 0] += 3.0
    dm_bin = xgb.DMatrix(X, label=y_bin)

    cv_strat = xgb.cv(
        {"max_depth": 3, "eta": 0.3, "objective": "binary:logistic", "eval_metric": "logloss"},
        dm_bin,
        num_boost_round=5,
        nfold=5,
        stratified=True,
        as_pandas=False,
        seed=42,
    )
    assert len(cv_strat["test-logloss-mean"]) == 5
    assert cv_strat["test-logloss-mean"][-1] < cv_strat["test-logloss-mean"][0]

    # Ranking CV with groups and per-group weights
    groups = [20, 20, 20, 20, 20, 20]
    g_weights = [1.0, 2.0, 0.5, 1.5, 1.0, 2.5]
    y_rank = rng.integers(0, 4, size=120).astype(np.float32)
    dm_rank = xgb.DMatrix(X, label=y_rank, group=groups, weight=g_weights)
    cv_rank = xgb.cv(
        {"max_depth": 3, "eta": 0.2, "objective": "rank:ndcg", "eval_metric": "ndcg"},
        dm_rank,
        num_boost_round=4,
        nfold=3,
        as_pandas=False,
        seed=7,
    )
    assert len(cv_rank["test-ndcg-mean"]) == 4


# ---------------------------------------------------------------------------
# D2 + D3: Strict Zero-NumPy + Zero-SciPy Subprocess Isolation Test
# ---------------------------------------------------------------------------


def test_d2_zero_numpy_and_zero_scipy_subprocess_end_to_end() -> None:
    code = """
import array
import math
import sys

# Block both numpy and scipy before importing xgboost!
sys.modules["numpy"] = None
sys.modules["scipy"] = None
sys.modules["scipy.sparse"] = None

import xgboost as xgb

assert xgb._typing.HAS_NUMPY is False

# 1. MetalArray construction, slicing, reshape, transpose, memoryview, DLPack
X = xgb.MetalArray.from_list(
    [[float((i * 3 + j) % 11) * 0.25 + (0.5 if j == 0 and i % 2 else 0.0) for j in range(5)] for i in range(80)],
    dtype="float32",
)
y = xgb.MetalArray.from_list([float(i % 2) for i in range(80)], dtype="float32")
assert X.is_page_aligned and X.shape == (80, 5)
assert X[..., :-1].shape == (80, 4)
assert X[..., -1].shape == (80,)
assert X.T.shape == (5, 80)

# 2. DMatrix from MetalArray (zero-copy), memoryview, nested list, and DLPack
dm = xgb.DMatrix(X, label=y)
assert dm.was_zero_copied is True
assert len(dm.get_label()) == 80

raw_buf = array.array("f", [float(v) for row in X.tolist() for v in row])
mv_2d = memoryview(raw_buf).cast("B").cast("f", shape=(80, 5))
dm_mv = xgb.DMatrix(mv_2d, label=array.array("f", y.tolist()))
assert dm_mv.num_row() == 80 and dm_mv.num_col() == 5

dm_dl = xgb.DMatrix(X.__dlpack__(), label=y.tolist())
assert dm_dl.num_row() == 80

# 3. Train gbtree + early stopping + evaluation without NumPy
evals_result = {}
bst = xgb.train(
    {"max_depth": 4, "eta": 0.3, "objective": "binary:logistic", "eval_metric": "logloss"},
    dm,
    num_boost_round=8,
    evals=[(dm, "train")],
    early_stopping_rounds=5,
    evals_result=evals_result,
    verbose_eval=False,
)
preds = bst.predict(dm)
assert isinstance(preds, xgb.MetalArray) and preds.shape == (80,)

# 4. inplace_predict & predict_to_buffer without NumPy
in_p1 = bst.inplace_predict(X)
in_p2 = bst.inplace_predict(mv_2d)
assert in_p1.tolist() == in_p2.tolist()

out_arr = array.array("f", [0.0] * 80)
shape_out = bst.predict_to_buffer(dm, out_arr)
assert shape_out == (80,)
for a, b in zip(preds.tolist(), out_arr):
    assert abs(a - b) < 1e-6

# 5. SHAP values via xgb.interpret.shap_values without NumPy
shap_vals, shap_bias = xgb.interpret.shap_values(bst, dm)
assert isinstance(shap_vals, xgb.MetalArray) and shap_vals.shape == (80, 5)
assert isinstance(shap_bias, xgb.MetalArray) and shap_bias.shape == (80,)

# 6. Custom objective returning MetalArray gradients & hessians
def custom_logreg(predt, dtrain):
    labels = dtrain.get_label().tolist()
    p_list = predt.tolist()
    g = []
    h = []
    for p_val, y_val in zip(p_list, labels):
        prob = 1.0 / (1.0 + math.exp(-p_val))
        g.append(prob - y_val)
        h.append(max(prob * (1.0 - prob), 1e-6))
    return xgb.MetalArray.from_list(g, dtype="float32"), xgb.MetalArray.from_list(h, dtype="float32")

bst_custom = xgb.train({"max_depth": 3, "eta": 0.2}, dm, num_boost_round=4, obj=custom_logreg)
assert bst_custom.predict(dm).shape == (80,)

# 7. QuantileDMatrix & DMatrix.get_quantile_cut / slice / get_data
qdm = xgb.QuantileDMatrix(X, label=y, max_bin=64)
cuts_indptr, cuts_data = qdm.get_quantile_cut()
assert isinstance(cuts_indptr, xgb.MetalArray) and isinstance(cuts_data, xgb.MetalArray)
bst_q = xgb.train({"max_depth": 3, "eta": 0.3, "max_bin": 64}, qdm, num_boost_round=4)
assert bst_q.predict(dm).shape == (80,)

dm_sliced = dm.slice([0, 2, 4, 6])
assert dm_sliced.num_row() == 4
csr_back = dm_sliced.get_data()
assert isinstance(csr_back, xgb.CSRMatrix) and csr_back.shape == (4, 5)

# 8. Native CSRMatrix / CSCMatrix / COOMatrix without NumPy or SciPy
dense_small = [[1.0, 0.0, 2.5, 0.0], [0.0, -1.5, 0.0, 3.0], [4.0, 0.0, 0.0, 0.0], [0.0, 2.0, 1.0, 0.0]]
csr = xgb.CSRMatrix(dense_small)
csc = csr.tocsc()
coo = csc.tocoo()
assert coo.tocsr().toarray().tolist() == dense_small
stacked = xgb.compat.concat([csr, csr])
assert stacked.shape == (8, 4)
dm_sp = xgb.DMatrix( stacked, label=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
bst_sp = xgb.train({"max_depth": 2, "eta": 0.3}, dm_sp, num_boost_round=4)
assert bst_sp.inplace_predict(stacked).shape == (8,)

# 9. Cross-validation (stratified & ranking group) without NumPy or scikit-learn
cv_hist = xgb.cv(
    {"max_depth": 3, "eta": 0.3, "objective": "binary:logistic", "eval_metric": "logloss"},
    dm,
    num_boost_round=5,
    nfold=4,
    stratified=True,
    as_pandas=False,
    seed=123,
)
assert len(cv_hist["test-logloss-mean"]) == 5

dm_grp = xgb.DMatrix(X, label=y, group=[20, 20, 20, 20])
assert dm_grp.get_group().tolist() == [20, 20, 20, 20]
cv_grp = xgb.cv(
    {"max_depth": 3, "eta": 0.2, "objective": "rank:ndcg", "eval_metric": "ndcg"},
    dm_grp,
    num_boost_round=3,
    nfold=2,
    as_pandas=False,
    seed=9,
)
assert len(cv_grp["test-ndcg-mean"]) == 3
print("ALL ZERO-NUMPY ZERO-SCIPY CHECKS SUCCEEDED")
"""
    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": str(ROOT / "python-package"),
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "ALL ZERO-NUMPY ZERO-SCIPY CHECKS SUCCEEDED" in proc.stdout
