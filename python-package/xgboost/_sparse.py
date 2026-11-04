"""Native lightweight CSR, CSC, and COO sparse containers and zero-SciPy /
optional-NumPy structural sparse dispatch helpers."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

from ._metal_array import MetalArray, resolve_dtype
from ._typing import HAS_NUMPY

if HAS_NUMPY:
    import numpy as np
else:
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import numpy as _np


def _to_metal_1d(arr: Any, dtype: Any) -> MetalArray:
    spec = resolve_dtype(dtype)
    if isinstance(arr, MetalArray):
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {arr.shape}")
        if arr.dtype == spec:
            return arr.copy()
        return MetalArray.from_list(arr.tolist(), dtype=spec)
    if isinstance(arr, (bytes, bytearray, memoryview)) or hasattr(arr, "typecode"):
        return MetalArray.from_buffer(arr, dtype=spec)
    vals = list(arr)
    if not vals:
        return MetalArray.empty((0,), dtype=spec)
    return MetalArray.from_list(vals, dtype=spec)


def _dense_to_2d_list(arg1: Any) -> Tuple[List[List[Any]], Tuple[int, int]]:
    if isinstance(arg1, MetalArray):
        if arg1.ndim != 2:
            raise ValueError("Expected a 2D array for sparse matrix construction.")
        return arg1.tolist(), arg1.shape
    if isinstance(arg1, memoryview):
        if arg1.ndim != 2:
            raise ValueError("Expected a 2D memoryview for sparse matrix construction.")
        return arg1.tolist(), (int(arg1.shape[0]), int(arg1.shape[1]))
    if isinstance(arg1, (list, tuple)):
        if len(arg1) == 0:
            return [], (0, 0)
        if not isinstance(arg1[0], (list, tuple)):
            raise ValueError("Expected a 2D sequence for sparse matrix construction.")
        ncols = len(arg1[0])
        rows: List[List[Any]] = []
        for r in arg1:
            if not isinstance(r, (list, tuple)) or len(r) != ncols:
                raise ValueError("Inconsistent row lengths in 2D sequence.")
            rows.append(list(r))
        return rows, (len(rows), ncols)
    raise ValueError("Expected a 2D array or sparse tuple.")


class CSRMatrix:
    """Lightweight Compressed Sparse Row (CSR) matrix compatible with DMatrix,
    QuantileDMatrix, Booster.inplace_predict, and scikit-learn estimators without
    requiring SciPy or NumPy."""

    format = "csr"
    ndim = 2

    def __init__(
        self,
        arg1: Any,
        shape: Optional[Tuple[int, int]] = None,
        dtype: Any = None,
    ) -> None:
        if isinstance(arg1, CSRMatrix):
            if HAS_NUMPY and np is not None and not isinstance(arg1.data, MetalArray):
                self.data: Any = np.asarray(arg1.data, dtype=dtype) if dtype is not None else arg1.data.copy()
                self.indices: Any = arg1.indices.copy()
                self.indptr: Any = arg1.indptr.copy()
            else:
                self.data = _to_metal_1d(arg1.data, dtype if dtype is not None else arg1.data.dtype)
                self.indices = _to_metal_1d(arg1.indices, "int32")
                self.indptr = _to_metal_1d(arg1.indptr, "int64")
            self.shape = shape if shape is not None else arg1.shape
            return

        if isinstance(arg1, tuple) and len(arg1) == 3:
            data, indices, indptr = arg1
            if HAS_NUMPY and np is not None and not isinstance(data, MetalArray):
                self.data = np.ascontiguousarray(data, dtype=dtype if dtype is not None else np.float32)
                self.indices = np.ascontiguousarray(indices, dtype=np.int32)
                self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
                max_col = (int(self.indices.max()) + 1) if len(self.indices) > 0 else 0
            else:
                self.data = _to_metal_1d(data, dtype if dtype is not None else "float32")
                self.indices = _to_metal_1d(indices, "int32")
                self.indptr = _to_metal_1d(indptr, "int64")
                idx_list = self.indices.tolist()
                max_col = (max(idx_list) + 1) if idx_list else 0
            if len(self.indices) != len(self.data):
                raise ValueError(f"CSR indices and data length mismatch: {len(self.indices)} vs {len(self.data)}")
            if self.indptr.ndim != 1 or len(self.indptr) < 1:
                raise ValueError("CSR indptr must be a non-empty 1D array.")
            nrows = len(self.indptr) - 1
            if shape is None:
                self.shape = (nrows, max_col)
            else:
                if len(shape) != 2 or shape[0] != nrows or shape[1] < 0:
                    raise ValueError(f"Invalid CSR shape {shape} for indptr of length {len(self.indptr)}")
                self.shape = (int(shape[0]), int(shape[1]))
            return

        if HAS_NUMPY and np is not None and not isinstance(arg1, MetalArray):
            arr = np.asarray(arg1, dtype=dtype if dtype is not None else np.float32)
            if arr.ndim != 2:
                raise ValueError("Expected a 2D array or (data, indices, indptr) tuple for CSRMatrix.")
            if shape is not None and tuple(shape) != arr.shape:
                raise ValueError(f"Shape {shape} does not match dense input shape {arr.shape}.")
            nz_rows, nz_cols = np.nonzero(arr)
            self.data = np.ascontiguousarray(arr[nz_rows, nz_cols])
            self.indices = np.ascontiguousarray(nz_cols, dtype=np.int32)
            counts = np.bincount(nz_rows, minlength=arr.shape[0])
            self.indptr = np.empty(arr.shape[0] + 1, dtype=np.int64)
            self.indptr[0] = 0
            np.cumsum(counts, out=self.indptr[1:])
            self.shape = (int(arr.shape[0]), int(arr.shape[1]))
            return

        rows_2d, dshape = _dense_to_2d_list(arg1)
        if shape is not None and tuple(shape) != dshape:
            raise ValueError(f"Shape {shape} does not match dense input shape {dshape}.")
        dt = dtype if dtype is not None else (arg1.dtype if isinstance(arg1, MetalArray) else "float32")
        vals_l: List[Any] = []
        cols_l: List[int] = []
        indptr_l: List[int] = [0]
        for row in rows_2d:
            for c, val in enumerate(row):
                if val != 0:
                    vals_l.append(val)
                    cols_l.append(c)
            indptr_l.append(len(vals_l))
        self.data = _to_metal_1d(vals_l, dt)
        self.indices = _to_metal_1d(cols_l, "int32")
        self.indptr = _to_metal_1d(indptr_l, "int64")
        self.shape = dshape

    @property
    def nnz(self) -> int:
        return int(len(self.data))

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    def copy(self) -> "CSRMatrix":
        return CSRMatrix((self.data.copy(), self.indices.copy(), self.indptr.copy()), shape=self.shape)

    def tocsr(self) -> "CSRMatrix":
        return self

    def tocoo(self) -> "COOMatrix":
        if HAS_NUMPY and np is not None and not isinstance(self.data, MetalArray):
            counts = np.diff(self.indptr)
            rows = np.repeat(np.arange(self.shape[0], dtype=np.int32), counts)
            return COOMatrix((self.data.copy(), (rows, self.indices.copy())), shape=self.shape)
        ip = self.indptr.tolist()
        rows_l: List[int] = []
        for r in range(self.shape[0]):
            rows_l.extend([r] * (ip[r + 1] - ip[r]))
        return COOMatrix((self.data.copy(), (_to_metal_1d(rows_l, "int32"), self.indices.copy())), shape=self.shape)

    def tocsc(self) -> "CSCMatrix":
        return self.tocoo().tocsc()

    def toarray(self) -> Any:
        if HAS_NUMPY and np is not None and not isinstance(self.data, MetalArray):
            out = np.zeros(self.shape, dtype=self.data.dtype)
            if self.nnz == 0:
                return out
            counts = np.diff(self.indptr)
            rows = np.repeat(np.arange(self.shape[0], dtype=np.intp), counts)
            out[rows, self.indices] = self.data
            return out
        out_m = MetalArray.zeros(self.shape, dtype=self.data.dtype)
        if self.nnz == 0:
            return out_m
        ip = self.indptr.tolist()
        idx = self.indices.tolist()
        vals = self.data.tolist()
        for r in range(self.shape[0]):
            for p in range(ip[r], ip[r + 1]):
                out_m[r, idx[p]] = vals[p]
        return out_m

    todense = toarray

    def __getitem__(self, key: Any) -> "CSRMatrix":
        use_metal = not HAS_NUMPY or np is None or isinstance(self.data, MetalArray)
        is_int = isinstance(key, int) or (HAS_NUMPY and np is not None and isinstance(key, np.integer))
        if is_int and not isinstance(key, bool):
            r = int(key)
            if r < 0:
                r += self.shape[0]
            if r < 0 or r >= self.shape[0]:
                raise IndexError("CSRMatrix row index out of range")
            start, stop = int(self.indptr[r]), int(self.indptr[r + 1])
            ip_one = _to_metal_1d([0, stop - start], "int64") if use_metal else np.array([0, stop - start], dtype=np.int64)
            return CSRMatrix(
                (self.data[start:stop], self.indices[start:stop], ip_one),
                shape=(1, self.shape[1]),
            )
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            if step == 1 and not use_metal:
                base = int(self.indptr[start])
                end = int(self.indptr[stop])
                return CSRMatrix(
                    (self.data[base:end], self.indices[base:end], self.indptr[start : stop + 1] - base),
                    shape=(max(0, stop - start), self.shape[1]),
                )
            row_idx_list = list(range(start, stop, step))
        elif use_metal:
            raw_seq = key.tolist() if isinstance(key, MetalArray) else list(key)
            if raw_seq and all(isinstance(x, bool) for x in raw_seq):
                row_idx_list = [i for i, flag in enumerate(raw_seq) if flag]
            else:
                row_idx_list = [int(x) for x in raw_seq]
        else:
            row_idx = np.asarray(key, dtype=np.intp)
            if row_idx.dtype == np.bool_:
                row_idx = np.flatnonzero(row_idx)
            row_idx_list = row_idx.tolist()

        if use_metal:
            ip = self.indptr.tolist()
            idx = self.indices.tolist()
            vals = self.data.tolist()
            out_vals: List[Any] = []
            out_idx: List[int] = []
            out_ip: List[int] = [0]
            for r in row_idx_list:
                if r < 0:
                    r += self.shape[0]
                if r < 0 or r >= self.shape[0]:
                    raise IndexError("CSRMatrix row index out of range")
                s, e = ip[r], ip[r + 1]
                out_vals.extend(vals[s:e])
                out_idx.extend(idx[s:e])
                out_ip.append(len(out_vals))
            return CSRMatrix(
                (_to_metal_1d(out_vals, self.data.dtype), _to_metal_1d(out_idx, "int32"), _to_metal_1d(out_ip, "int64")),
                shape=(len(row_idx_list), self.shape[1]),
            )

        parts_data = []
        parts_idx = []
        new_indptr = np.zeros(len(row_idx_list) + 1, dtype=np.int64)
        cursor = 0
        for i, r in enumerate(row_idx_list):
            if r < 0:
                r += self.shape[0]
            if r < 0 or r >= self.shape[0]:
                raise IndexError("CSRMatrix row index out of range")
            s, e = int(self.indptr[r]), int(self.indptr[r + 1])
            parts_data.append(self.data[s:e])
            parts_idx.append(self.indices[s:e])
            cursor += e - s
            new_indptr[i + 1] = cursor
        cat_data = np.concatenate(parts_data) if parts_data else np.empty(0, dtype=self.data.dtype)
        cat_idx = np.concatenate(parts_idx) if parts_idx else np.empty(0, dtype=np.int32)
        return CSRMatrix((cat_data, cat_idx, new_indptr), shape=(len(row_idx_list), self.shape[1]))

    def __repr__(self) -> str:
        return f"<CSRMatrix shape={self.shape} dtype={self.dtype} nnz={self.nnz}>"


class CSCMatrix:
    """Lightweight Compressed Sparse Column (CSC) matrix without SciPy or NumPy."""

    format = "csc"
    ndim = 2

    def __init__(
        self,
        arg1: Any,
        shape: Optional[Tuple[int, int]] = None,
        dtype: Any = None,
    ) -> None:
        if isinstance(arg1, CSCMatrix):
            if HAS_NUMPY and np is not None and not isinstance(arg1.data, MetalArray):
                self.data: Any = np.asarray(arg1.data, dtype=dtype) if dtype is not None else arg1.data.copy()
                self.indices: Any = arg1.indices.copy()
                self.indptr: Any = arg1.indptr.copy()
            else:
                self.data = _to_metal_1d(arg1.data, dtype if dtype is not None else arg1.data.dtype)
                self.indices = _to_metal_1d(arg1.indices, "int32")
                self.indptr = _to_metal_1d(arg1.indptr, "int64")
            self.shape = shape if shape is not None else arg1.shape
            return

        if isinstance(arg1, tuple) and len(arg1) == 3:
            data, indices, indptr = arg1
            if HAS_NUMPY and np is not None and not isinstance(data, MetalArray):
                self.data = np.ascontiguousarray(data, dtype=dtype if dtype is not None else np.float32)
                self.indices = np.ascontiguousarray(indices, dtype=np.int32)
                self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
                max_row = (int(self.indices.max()) + 1) if len(self.indices) > 0 else 0
            else:
                self.data = _to_metal_1d(data, dtype if dtype is not None else "float32")
                self.indices = _to_metal_1d(indices, "int32")
                self.indptr = _to_metal_1d(indptr, "int64")
                idx_list = self.indices.tolist()
                max_row = (max(idx_list) + 1) if idx_list else 0
            if len(self.indices) != len(self.data):
                raise ValueError(f"CSC indices and data length mismatch: {len(self.indices)} vs {len(self.data)}")
            if self.indptr.ndim != 1 or len(self.indptr) < 1:
                raise ValueError("CSC indptr must be a non-empty 1D array.")
            ncols = len(self.indptr) - 1
            if shape is None:
                self.shape = (max_row, ncols)
            else:
                if len(shape) != 2 or shape[1] != ncols or shape[0] < 0:
                    raise ValueError(f"Invalid CSC shape {shape} for indptr of length {len(self.indptr)}")
                self.shape = (int(shape[0]), int(shape[1]))
            return

        if HAS_NUMPY and np is not None and not isinstance(arg1, MetalArray):
            arr = np.asarray(arg1, dtype=dtype if dtype is not None else np.float32)
            if arr.ndim != 2:
                raise ValueError("Expected a 2D array or (data, indices, indptr) tuple for CSCMatrix.")
            if shape is not None and tuple(shape) != arr.shape:
                raise ValueError(f"Shape {shape} does not match dense input shape {arr.shape}.")
            nz_cols, nz_rows = np.nonzero(arr.T)
            self.data = np.ascontiguousarray(arr[nz_rows, nz_cols])
            self.indices = np.ascontiguousarray(nz_rows, dtype=np.int32)
            counts = np.bincount(nz_cols, minlength=arr.shape[1])
            self.indptr = np.empty(arr.shape[1] + 1, dtype=np.int64)
            self.indptr[0] = 0
            np.cumsum(counts, out=self.indptr[1:])
            self.shape = (int(arr.shape[0]), int(arr.shape[1]))
            return

        rows_2d, dshape = _dense_to_2d_list(arg1)
        if shape is not None and tuple(shape) != dshape:
            raise ValueError(f"Shape {shape} does not match dense input shape {dshape}.")
        dt = dtype if dtype is not None else (arg1.dtype if isinstance(arg1, MetalArray) else "float32")
        nrows, ncols = dshape
        vals_l: List[Any] = []
        rows_l: List[int] = []
        indptr_l: List[int] = [0]
        for c in range(ncols):
            for r in range(nrows):
                val = rows_2d[r][c]
                if val != 0:
                    vals_l.append(val)
                    rows_l.append(r)
            indptr_l.append(len(vals_l))
        self.data = _to_metal_1d(vals_l, dt)
        self.indices = _to_metal_1d(rows_l, "int32")
        self.indptr = _to_metal_1d(indptr_l, "int64")
        self.shape = dshape

    @property
    def nnz(self) -> int:
        return int(len(self.data))

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    def copy(self) -> "CSCMatrix":
        return CSCMatrix((self.data.copy(), self.indices.copy(), self.indptr.copy()), shape=self.shape)

    def tocsc(self) -> "CSCMatrix":
        return self

    def tocoo(self) -> "COOMatrix":
        if HAS_NUMPY and np is not None and not isinstance(self.data, MetalArray):
            counts = np.diff(self.indptr)
            cols = np.repeat(np.arange(self.shape[1], dtype=np.int32), counts)
            return COOMatrix((self.data.copy(), (self.indices.copy(), cols)), shape=self.shape)
        ip = self.indptr.tolist()
        cols_l: List[int] = []
        for c in range(self.shape[1]):
            cols_l.extend([c] * (ip[c + 1] - ip[c]))
        return COOMatrix((self.data.copy(), (self.indices.copy(), _to_metal_1d(cols_l, "int32"))), shape=self.shape)

    def tocsr(self) -> CSRMatrix:
        return self.tocoo().tocsr()

    def toarray(self) -> Any:
        if HAS_NUMPY and np is not None and not isinstance(self.data, MetalArray):
            out = np.zeros(self.shape, dtype=self.data.dtype)
            if self.nnz == 0:
                return out
            counts = np.diff(self.indptr)
            cols = np.repeat(np.arange(self.shape[1], dtype=np.intp), counts)
            out[self.indices, cols] = self.data
            return out
        out_m = MetalArray.zeros(self.shape, dtype=self.data.dtype)
        if self.nnz == 0:
            return out_m
        ip = self.indptr.tolist()
        idx = self.indices.tolist()
        vals = self.data.tolist()
        for c in range(self.shape[1]):
            for p in range(ip[c], ip[c + 1]):
                out_m[idx[p], c] = vals[p]
        return out_m

    todense = toarray

    def __repr__(self) -> str:
        return f"<CSCMatrix shape={self.shape} dtype={self.dtype} nnz={self.nnz}>"


class COOMatrix:
    """Lightweight Coordinate (COO) sparse matrix without SciPy or NumPy."""

    format = "coo"
    ndim = 2

    def __init__(
        self,
        arg1: Any,
        shape: Optional[Tuple[int, int]] = None,
        dtype: Any = None,
    ) -> None:
        if isinstance(arg1, COOMatrix):
            if HAS_NUMPY and np is not None and not isinstance(arg1.data, MetalArray):
                self.data: Any = np.asarray(arg1.data, dtype=dtype) if dtype is not None else arg1.data.copy()
                self.row: Any = arg1.row.copy()
                self.col: Any = arg1.col.copy()
            else:
                self.data = _to_metal_1d(arg1.data, dtype if dtype is not None else arg1.data.dtype)
                self.row = _to_metal_1d(arg1.row, "int32")
                self.col = _to_metal_1d(arg1.col, "int32")
            self.shape = shape if shape is not None else arg1.shape
            return

        if isinstance(arg1, tuple) and len(arg1) == 2:
            data, (row, col) = arg1
            if HAS_NUMPY and np is not None and not isinstance(data, MetalArray):
                self.data = np.ascontiguousarray(data, dtype=dtype if dtype is not None else np.float32)
                self.row = np.ascontiguousarray(row, dtype=np.int32)
                self.col = np.ascontiguousarray(col, dtype=np.int32)
                nrows_auto = (int(self.row.max()) + 1) if len(self.row) > 0 else 0
                ncols_auto = (int(self.col.max()) + 1) if len(self.col) > 0 else 0
            else:
                self.data = _to_metal_1d(data, dtype if dtype is not None else "float32")
                self.row = _to_metal_1d(row, "int32")
                self.col = _to_metal_1d(col, "int32")
                rl = self.row.tolist()
                cl = self.col.tolist()
                nrows_auto = (max(rl) + 1) if rl else 0
                ncols_auto = (max(cl) + 1) if cl else 0
            if not (len(self.data) == len(self.row) == len(self.col)):
                raise ValueError("COO data, row, and col arrays must have the same length.")
            if shape is None:
                self.shape = (nrows_auto, ncols_auto)
            else:
                self.shape = (int(shape[0]), int(shape[1]))
            return

        if HAS_NUMPY and np is not None and not isinstance(arg1, MetalArray):
            arr = np.asarray(arg1, dtype=dtype if dtype is not None else np.float32)
            if arr.ndim != 2:
                raise ValueError("Expected a 2D array or (data, (row, col)) tuple for COOMatrix.")
            if shape is not None and tuple(shape) != arr.shape:
                raise ValueError(f"Shape {shape} does not match dense input shape {arr.shape}.")
            nz_rows, nz_cols = np.nonzero(arr)
            self.data = np.ascontiguousarray(arr[nz_rows, nz_cols])
            self.row = np.ascontiguousarray(nz_rows, dtype=np.int32)
            self.col = np.ascontiguousarray(nz_cols, dtype=np.int32)
            self.shape = (int(arr.shape[0]), int(arr.shape[1]))
            return

        rows_2d, dshape = _dense_to_2d_list(arg1)
        if shape is not None and tuple(shape) != dshape:
            raise ValueError(f"Shape {shape} does not match dense input shape {dshape}.")
        dt = dtype if dtype is not None else (arg1.dtype if isinstance(arg1, MetalArray) else "float32")
        vals_l: List[Any] = []
        r_l: List[int] = []
        c_l: List[int] = []
        for r_idx, row in enumerate(rows_2d):
            for c_idx, val in enumerate(row):
                if val != 0:
                    vals_l.append(val)
                    r_l.append(r_idx)
                    c_l.append(c_idx)
        self.data = _to_metal_1d(vals_l, dt)
        self.row = _to_metal_1d(r_l, "int32")
        self.col = _to_metal_1d(c_l, "int32")
        self.shape = dshape

    @property
    def nnz(self) -> int:
        return int(len(self.data))

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    def copy(self) -> "COOMatrix":
        return COOMatrix((self.data.copy(), (self.row.copy(), self.col.copy())), shape=self.shape)

    def tocoo(self) -> "COOMatrix":
        return self

    def tocsr(self) -> CSRMatrix:
        nrows, ncols = self.shape
        if HAS_NUMPY and np is not None and not isinstance(self.data, MetalArray):
            if self.nnz == 0:
                return CSRMatrix(
                    (np.empty(0, dtype=self.data.dtype), np.empty(0, dtype=np.int32), np.zeros(nrows + 1, dtype=np.int64)),
                    shape=self.shape,
                )
            order = np.lexsort((self.col, self.row))
            s_row = self.row[order]
            s_col = self.col[order]
            s_data = self.data[order]
            dup = np.empty(len(s_row), dtype=bool)
            dup[0] = True
            dup[1:] = (s_row[1:] != s_row[:-1]) | (s_col[1:] != s_col[:-1])
            if not np.all(dup):
                idx = np.flatnonzero(dup)
                s_data = np.add.reduceat(s_data, idx)
                s_row = s_row[idx]
                s_col = s_col[idx]
            counts = np.bincount(s_row, minlength=nrows)
            indptr = np.empty(nrows + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])
            return CSRMatrix((s_data, s_col, indptr), shape=(nrows, ncols))

        if self.nnz == 0:
            return CSRMatrix(
                (_to_metal_1d([], self.data.dtype), _to_metal_1d([], "int32"), _to_metal_1d([0] * (nrows + 1), "int64")),
                shape=self.shape,
            )
        triplets = sorted(zip(self.row.tolist(), self.col.tolist(), self.data.tolist()), key=lambda t: (t[0], t[1]))
        merged_r: List[int] = []
        merged_c: List[int] = []
        merged_v: List[Any] = []
        for r, c, v in triplets:
            if merged_r and merged_r[-1] == r and merged_c[-1] == c:
                merged_v[-1] += v
            else:
                merged_r.append(r)
                merged_c.append(c)
                merged_v.append(v)
        counts_l = [0] * nrows
        for r in merged_r:
            counts_l[r] += 1
        indptr_l = [0] * (nrows + 1)
        acc = 0
        for i, cnt in enumerate(counts_l):
            acc += cnt
            indptr_l[i + 1] = acc
        return CSRMatrix(
            (_to_metal_1d(merged_v, self.data.dtype), _to_metal_1d(merged_c, "int32"), _to_metal_1d(indptr_l, "int64")),
            shape=(nrows, ncols),
        )

    def tocsc(self) -> CSCMatrix:
        nrows, ncols = self.shape
        if HAS_NUMPY and np is not None and not isinstance(self.data, MetalArray):
            if self.nnz == 0:
                return CSCMatrix(
                    (np.empty(0, dtype=self.data.dtype), np.empty(0, dtype=np.int32), np.zeros(ncols + 1, dtype=np.int64)),
                    shape=self.shape,
                )
            order = np.lexsort((self.row, self.col))
            s_row = self.row[order]
            s_col = self.col[order]
            s_data = self.data[order]
            dup = np.empty(len(s_col), dtype=bool)
            dup[0] = True
            dup[1:] = (s_col[1:] != s_col[:-1]) | (s_row[1:] != s_row[:-1])
            if not np.all(dup):
                idx = np.flatnonzero(dup)
                s_data = np.add.reduceat(s_data, idx)
                s_row = s_row[idx]
                s_col = s_col[idx]
            counts = np.bincount(s_col, minlength=ncols)
            indptr = np.empty(ncols + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])
            return CSCMatrix((s_data, s_row, indptr), shape=(nrows, ncols))

        if self.nnz == 0:
            return CSCMatrix(
                (_to_metal_1d([], self.data.dtype), _to_metal_1d([], "int32"), _to_metal_1d([0] * (ncols + 1), "int64")),
                shape=self.shape,
            )
        triplets = sorted(zip(self.col.tolist(), self.row.tolist(), self.data.tolist()), key=lambda t: (t[0], t[1]))
        merged_c: List[int] = []
        merged_r: List[int] = []
        merged_v: List[Any] = []
        for c, r, v in triplets:
            if merged_c and merged_c[-1] == c and merged_r[-1] == r:
                merged_v[-1] += v
            else:
                merged_c.append(c)
                merged_r.append(r)
                merged_v.append(v)
        counts_l = [0] * ncols
        for c in merged_c:
            counts_l[c] += 1
        indptr_l = [0] * (ncols + 1)
        acc = 0
        for i, cnt in enumerate(counts_l):
            acc += cnt
            indptr_l[i + 1] = acc
        return CSCMatrix(
            (_to_metal_1d(merged_v, self.data.dtype), _to_metal_1d(merged_r, "int32"), _to_metal_1d(indptr_l, "int64")),
            shape=(nrows, ncols),
        )

    def toarray(self) -> Any:
        if HAS_NUMPY and np is not None and not isinstance(self.data, MetalArray):
            out = np.zeros(self.shape, dtype=self.data.dtype)
            if self.nnz > 0:
                np.add.at(out, (self.row, self.col), self.data)
            return out
        out_m = MetalArray.zeros(self.shape, dtype=self.data.dtype)
        for r, c, v in zip(self.row.tolist(), self.col.tolist(), self.data.tolist()):
            out_m[r, c] = out_m[r, c] + v
        return out_m

    todense = toarray

    def __repr__(self) -> str:
        return f"<COOMatrix shape={self.shape} dtype={self.dtype} nnz={self.nnz}>"


def _loaded_scipy_sparse() -> Any:
    """Return scipy.sparse ONLY if already imported by user code; never imports scipy."""
    return sys.modules.get("scipy.sparse", None)


def _is_cuda_sparse(data: Any) -> bool:
    values = getattr(data, "data", None)
    return values is not None and hasattr(values, "__cuda_array_interface__")


def is_sparse_csr(data: Any) -> bool:
    """Check for native CSRMatrix, loaded scipy.sparse CSR, or structural host CSR."""
    if isinstance(data, CSRMatrix):
        return True
    if _is_cuda_sparse(data):
        return False
    sp = _loaded_scipy_sparse()
    if sp is not None:
        types = tuple(t for t in (getattr(sp, "csr_matrix", None), getattr(sp, "csr_array", None)) if t is not None)
        if types and isinstance(data, types):
            return True
    return (
        getattr(data, "format", None) == "csr"
        and hasattr(data, "indptr")
        and hasattr(data, "indices")
        and hasattr(data, "data")
        and hasattr(data, "shape")
    )


def is_sparse_csc(data: Any) -> bool:
    """Check for native CSCMatrix, loaded scipy.sparse CSC, or structural host CSC."""
    if isinstance(data, CSCMatrix):
        return True
    if _is_cuda_sparse(data):
        return False
    sp = _loaded_scipy_sparse()
    if sp is not None:
        types = tuple(t for t in (getattr(sp, "csc_matrix", None), getattr(sp, "csc_array", None)) if t is not None)
        if types and isinstance(data, types):
            return True
    return (
        getattr(data, "format", None) == "csc"
        and hasattr(data, "indptr")
        and hasattr(data, "indices")
        and hasattr(data, "data")
        and hasattr(data, "shape")
    )


def is_sparse_coo(data: Any) -> bool:
    """Check for native COOMatrix, loaded scipy.sparse COO, or structural host COO."""
    if isinstance(data, COOMatrix):
        return True
    if _is_cuda_sparse(data):
        return False
    sp = _loaded_scipy_sparse()
    if sp is not None:
        types = tuple(t for t in (getattr(sp, "coo_matrix", None), getattr(sp, "coo_array", None)) if t is not None)
        if types and isinstance(data, types):
            return True
    return (
        getattr(data, "format", None) == "coo"
        and hasattr(data, "row")
        and hasattr(data, "col")
        and hasattr(data, "data")
        and hasattr(data, "shape")
    )


def is_any_host_sparse(data: Any) -> bool:
    """True for any native, scipy.sparse, or duck-typed host sparse matrix."""
    if is_sparse_csr(data) or is_sparse_csc(data) or is_sparse_coo(data):
        return True
    sp = _loaded_scipy_sparse()
    if sp is not None:
        base_types = tuple(t for t in (getattr(sp, "spmatrix", None), getattr(sp, "sparray", None)) if t is not None)
        if base_types and isinstance(data, base_types):
            return True
    return False


def ensure_sparse_aligned(data: Any, is_csr: bool) -> Any:
    """Ensure contiguous aligned buffers for CSR or CSC inputs without importing scipy."""
    from ._data_utils import _ensure_np_dtype

    if len(data.indices) != len(data.data):
        raise ValueError(f"length mismatch: {len(data.indices)} vs {len(data.data)}")

    indptr, _ = _ensure_np_dtype(data.indptr, getattr(data.indptr, "dtype", "int64"))
    indices, _ = _ensure_np_dtype(data.indices, getattr(data.indices, "dtype", "int32"))
    values, _ = _ensure_np_dtype(data.data, getattr(data.data, "dtype", "float32"))
    if indptr is not data.indptr or indices is not data.indices or values is not data.data:
        sp = _loaded_scipy_sparse()
        if sp is not None and HAS_NUMPY and np is not None and not isinstance(values, MetalArray):
            cls = sp.csr_matrix if is_csr else sp.csc_matrix
            return cls((values, indices, indptr), shape=data.shape)
        cls_native = CSRMatrix if is_csr else CSCMatrix
        return cls_native((values, indices, indptr), shape=data.shape)
    return data


def ensure_coo_aligned(data: Any) -> Any:
    """Ensure contiguous aligned buffers for COO inputs without importing scipy."""
    from ._data_utils import _ensure_np_dtype

    if not (len(data.row) == len(data.col) == len(data.data)):
        raise ValueError("COO data, row, and col arrays must have the same length.")

    row, _ = _ensure_np_dtype(data.row, getattr(data.row, "dtype", "int32"))
    col, _ = _ensure_np_dtype(data.col, getattr(data.col, "dtype", "int32"))
    values, _ = _ensure_np_dtype(data.data, getattr(data.data, "dtype", "float32"))
    return row, col, values, data.shape


def sparse_vstack(blocks: Sequence[Any]) -> Any:
    """Concatenate sparse matrices row-wise without requiring SciPy or NumPy."""
    if not blocks:
        raise ValueError("Cannot concatenate an empty sequence of sparse matrices.")
    first = blocks[0]
    sp = _loaded_scipy_sparse()
    if sp is not None:
        base_types = tuple(t for t in (getattr(sp, "spmatrix", None), getattr(sp, "sparray", None)) if t is not None)
        if base_types and isinstance(first, base_types):
            fmt = "csc" if is_sparse_csc(first) else "csr"
            return sp.vstack(blocks, format=fmt)

    ncols = first.shape[1]
    for b in blocks[1:]:
        if b.shape[1] != ncols:
            raise ValueError("All sparse matrices in vstack must have the same column count.")

    use_metal = not HAS_NUMPY or np is None or isinstance(getattr(first, "data", None), MetalArray)

    if is_sparse_csc(first):
        coo_blocks = [b.tocoo() if hasattr(b, "tocoo") else CSRMatrix(b).tocoo() for b in blocks]
        if use_metal:
            rows_m: List[int] = []
            cols_m: List[int] = []
            vals_m: List[Any] = []
            row_offset = 0
            for cb in coo_blocks:
                rows_m.extend(r + row_offset for r in cb.row.tolist())
                cols_m.extend(cb.col.tolist())
                vals_m.extend(cb.data.tolist())
                row_offset += cb.shape[0]
            return COOMatrix(
                (_to_metal_1d(vals_m, "float32"), (_to_metal_1d(rows_m, "int32"), _to_metal_1d(cols_m, "int32"))),
                shape=(row_offset, ncols),
            ).tocsc()
        rows = []
        cols = []
        vals = []
        row_offset = 0
        for cb in coo_blocks:
            rows.append(cb.row + row_offset)
            cols.append(cb.col)
            vals.append(cb.data)
            row_offset += cb.shape[0]
        return COOMatrix(
            (np.concatenate(vals) if vals else np.empty(0, dtype=np.float32),
             (np.concatenate(rows) if rows else np.empty(0, dtype=np.int32),
              np.concatenate(cols) if cols else np.empty(0, dtype=np.int32))),
            shape=(row_offset, ncols),
        ).tocsc()

    csr_blocks = [b if is_sparse_csr(b) else b.tocsr() for b in blocks]
    if use_metal:
        vals_all: List[Any] = []
        idx_all: List[int] = []
        ip_all: List[int] = [0]
        offset = 0
        total_rows = 0
        for b in csr_blocks:
            vals_all.extend(b.data.tolist())
            idx_all.extend(b.indices.tolist())
            b_ip = b.indptr.tolist()
            if b.shape[0] > 0:
                ip_all.extend(p + offset for p in b_ip[1:])
            offset += int(b_ip[-1])
            total_rows += b.shape[0]
        return CSRMatrix(
            (_to_metal_1d(vals_all, "float32"), _to_metal_1d(idx_all, "int32"), _to_metal_1d(ip_all, "int64")),
            shape=(total_rows, ncols),
        )

    data_parts = [b.data for b in csr_blocks]
    idx_parts = [b.indices for b in csr_blocks]
    indptr_parts = [np.zeros(1, dtype=np.int64)]
    offset = 0
    total_rows = 0
    for b in csr_blocks:
        if b.shape[0] > 0:
            indptr_parts.append(np.asarray(b.indptr[1:], dtype=np.int64) + offset)
        offset += int(b.indptr[-1])
        total_rows += b.shape[0]
    return CSRMatrix(
        (
            np.concatenate(data_parts) if data_parts else np.empty(0, dtype=np.float32),
            np.concatenate(idx_parts) if idx_parts else np.empty(0, dtype=np.int32),
            np.concatenate(indptr_parts),
        ),
        shape=(total_rows, ncols),
    )
