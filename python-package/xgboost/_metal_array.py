"""Page-aligned Apple Silicon unified-memory array and zero-JSON PEP 3118 /
DLPack / Arrow C Data Interface bridges for xgboost-metal."""

from __future__ import annotations

import array as _py_array
import ctypes
import math
import mmap
import struct
from types import EllipsisType
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

try:
    import numpy as _np

    HAS_NUMPY = True
except ImportError:
    _np = None  # type: ignore[assignment]
    HAS_NUMPY = False

PAGE_SIZE: int = getattr(mmap, "PAGESIZE", 16384)

# Must match kIngest* constants in include/metalgb/shared.h
INGEST_F32 = 1
INGEST_F64 = 2
INGEST_F16 = 3
INGEST_BF16 = 4
INGEST_I8 = 5
INGEST_I16 = 6
INGEST_I32 = 7
INGEST_I64 = 8
INGEST_U8 = 9
INGEST_U16 = 10
INGEST_U32 = 11
INGEST_U64 = 12
INGEST_BOOL = 13
INGEST_BOOL_BITS = 14

INGEST_MASK_NONE = 0
INGEST_MASK_VALID_BITS = 1
INGEST_MASK_VALID_BYTES = 2
INGEST_MASK_INVALID_BYTES = 3

# XGBufferDescriptor.flags bits
XGBUF_FLAG_CONTIGUOUS = 1
XGBUF_FLAG_ZERO_COPY = 2
XGBUF_FLAG_BIT_OFFSET_IN_STRIDE0 = 4


class XGBufferDescriptor(ctypes.Structure):
    """Binary C ABI descriptor matching XGBufferDescriptor in xgboost/c_api.h."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.c_uint64 * 2),
        ("strides", ctypes.c_int64 * 2),
        ("ndim", ctypes.c_uint32),
        ("dtype_code", ctypes.c_uint32),
        ("mask_data", ctypes.c_void_p),
        ("mask_bit_offset", ctypes.c_int64),
        ("mask_mode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class _DTypeSpec:
    __slots__ = ("name", "code", "fmt", "itemsize", "typestr", "dl_code", "dl_bits", "is_float", "is_bool")

    def __init__(
        self,
        name: str,
        code: int,
        fmt: str,
        itemsize: int,
        typestr: str,
        dl_code: int,
        dl_bits: int,
        is_float: bool = False,
        is_bool: bool = False,
    ) -> None:
        self.name = name
        self.code = code
        self.fmt = fmt
        self.itemsize = itemsize
        self.typestr = typestr
        self.dl_code = dl_code
        self.dl_bits = dl_bits
        self.is_float = is_float
        self.is_bool = is_bool

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _DTypeSpec):
            return self.name == other.name
        if isinstance(other, str):
            try:
                return self.name == resolve_dtype(other).name
            except (TypeError, ValueError):
                return False
        if HAS_NUMPY and other is not None:
            try:
                return _np.dtype(self.typestr) == _np.dtype(other)
            except TypeError:
                return False
        return False

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"dtype('{self.name}')"

    def __str__(self) -> str:
        return self.name


_DTYPES = {
    "float32": _DTypeSpec("float32", INGEST_F32, "f", 4, "<f4", 2, 32, is_float=True),
    "float64": _DTypeSpec("float64", INGEST_F64, "d", 8, "<f8", 2, 64, is_float=True),
    "float16": _DTypeSpec("float16", INGEST_F16, "e", 2, "<f2", 2, 16, is_float=True),
    "int8": _DTypeSpec("int8", INGEST_I8, "b", 1, "|i1", 0, 8),
    "int16": _DTypeSpec("int16", INGEST_I16, "h", 2, "<i2", 0, 16),
    "int32": _DTypeSpec("int32", INGEST_I32, "i", 4, "<i4", 0, 32),
    "int64": _DTypeSpec("int64", INGEST_I64, "q", 8, "<i8", 0, 64),
    "uint8": _DTypeSpec("uint8", INGEST_U8, "B", 1, "|u1", 1, 8),
    "uint16": _DTypeSpec("uint16", INGEST_U16, "H", 2, "<u2", 1, 16),
    "uint32": _DTypeSpec("uint32", INGEST_U32, "I", 4, "<u4", 1, 32),
    "uint64": _DTypeSpec("uint64", INGEST_U64, "Q", 8, "<u8", 1, 64),
    "bool": _DTypeSpec("bool", INGEST_BOOL, "?", 1, "|b1", 6, 8, is_bool=True),
}

_ALIASES = {
    "float32": "float32",
    "f4": "float32",
    "<f4": "float32",
    "=f4": "float32",
    "float": "float32",
    "f": "float32",
    "float64": "float64",
    "f8": "float64",
    "<f8": "float64",
    "=f8": "float64",
    "double": "float64",
    "d": "float64",
    "float16": "float16",
    "f2": "float16",
    "<f2": "float16",
    "=f2": "float16",
    "e": "float16",
    "int8": "int8",
    "i1": "int8",
    "|i1": "int8",
    "<i1": "int8",
    "b": "int8",
    "int16": "int16",
    "i2": "int16",
    "<i2": "int16",
    "h": "int16",
    "int32": "int32",
    "i4": "int32",
    "<i4": "int32",
    "i": "int32",
    "int64": "int64",
    "i8": "int64",
    "<i8": "int64",
    "q": "int64",
    "int": "int64",
    "uint8": "uint8",
    "u1": "uint8",
    "|u1": "uint8",
    "<u1": "uint8",
    "B": "uint8",
    "uint16": "uint16",
    "u2": "uint16",
    "<u2": "uint16",
    "H": "uint16",
    "uint32": "uint32",
    "u4": "uint32",
    "<u4": "uint32",
    "I": "uint32",
    "uint64": "uint64",
    "u8": "uint64",
    "<u8": "uint64",
    "Q": "uint64",
    "bool": "bool",
    "bool_": "bool",
    "b1": "bool",
    "|b1": "bool",
    "?": "bool",
}


def resolve_dtype(dtype: Any) -> _DTypeSpec:
    """Normalize string, Python type, NumPy dtype, or _DTypeSpec to _DTypeSpec."""
    if dtype is None:
        return _DTYPES["float32"]
    if isinstance(dtype, _DTypeSpec):
        return dtype
    if dtype is float:
        return _DTYPES["float32"]
    if dtype is int:
        return _DTYPES["int64"]
    if dtype is bool:
        return _DTYPES["bool"]
    if isinstance(dtype, str):
        key = _ALIASES.get(dtype)
        if key is not None:
            return _DTYPES[key]
    if HAS_NUMPY and dtype is not None:
        np_dt = _np.dtype(dtype)
        if not np_dt.isnative:
            raise TypeError(f"Non-native byte-order dtype is not supported: {dtype}")
        key = _ALIASES.get(np_dt.str) or _ALIASES.get(np_dt.name)
        if key is not None:
            return _DTYPES[key]
    raise TypeError(f"Unsupported dtype: {dtype!r}")


def _dtype_from_pep3118(fmt: Optional[str], itemsize: int) -> Optional[_DTypeSpec]:
    if not fmt:
        return _DTYPES["uint8"] if itemsize == 1 else None
    if fmt[0] in (">", "!"):
        return None
    clean = fmt.lstrip("@=<|")
    if len(clean) != 1:
        return None
    ch = clean[0]
    if ch in ("l", "n"):
        return _DTYPES["int32"] if itemsize == 4 else _DTYPES["int64"] if itemsize == 8 else None
    if ch in ("L", "N"):
        return _DTYPES["uint32"] if itemsize == 4 else _DTYPES["uint64"] if itemsize == 8 else None
    name = _ALIASES.get(ch)
    if name is None:
        return None
    spec = _DTYPES[name]
    return spec if spec.itemsize == itemsize else None


# Stable CPython 3.11+ Py_buffer & PyCapsule C API bindings
class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_PyBUF_FORMAT = 0x0004
_PyBUF_ND = 0x0008
_PyBUF_STRIDES = 0x0010 | _PyBUF_ND

_PyObject_GetBuffer = ctypes.pythonapi.PyObject_GetBuffer
_PyObject_GetBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_PyObject_GetBuffer.restype = ctypes.c_int

_PyBuffer_Release = ctypes.pythonapi.PyBuffer_Release
_PyBuffer_Release.argtypes = [ctypes.POINTER(_PyBuffer)]
_PyBuffer_Release.restype = None

_PyCapsuleDestructorType = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _PyCapsuleDestructorType]
_PyCapsule_New.restype = ctypes.py_object

_PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
_PyCapsule_GetName.argtypes = [ctypes.py_object]
_PyCapsule_GetName.restype = ctypes.c_char_p

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_GetPointer.restype = ctypes.c_void_p

_PyCapsule_SetName = ctypes.pythonapi.PyCapsule_SetName
_PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_SetName.restype = ctypes.c_int

_PyCapsule_GetName_Raw = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p)(
    ("PyCapsule_GetName", ctypes.pythonapi)
)
_PyCapsule_GetPointer_Raw = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)


def _c_contiguous_strides(shape: Tuple[int, ...], itemsize: int) -> Tuple[int, ...]:
    if not shape:
        return ()
    strides = [itemsize] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


# DLPack C structs for MetalArray.__dlpack__()
class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_DLDeleterType = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DLDeleterType),
    ]


_LIVE_DLPACK_EXPORTS: dict[int, Tuple[Any, ...]] = {}


@_DLDeleterType
def _dlpack_managed_deleter(managed_ptr: int) -> None:
    if managed_ptr:
        _LIVE_DLPACK_EXPORTS.pop(managed_ptr, None)


@_PyCapsuleDestructorType
def _dlpack_capsule_destructor(capsule_ptr: int) -> None:
    if not capsule_ptr:
        return
    name = _PyCapsule_GetName_Raw(capsule_ptr)
    if name == b"dltensor":
        ptr = _PyCapsule_GetPointer_Raw(capsule_ptr, b"dltensor")
        if ptr:
            _LIVE_DLPACK_EXPORTS.pop(ptr, None)


class MetalArray:
    """16 KiB VM-page-aligned Apple Silicon unified-memory tensor.

    Supports zero-copy Metal `newBufferWithBytesNoCopy` ingestion, PEP 3118
    `memoryview`, `__array_interface__` (`np.asarray(arr)` zero-copy), and
    `__dlpack__` (`np.from_dlpack(arr)` / `xgb.DMatrix` zero-copy) without
    requiring NumPy or SciPy.
    """

    __slots__ = ("_mmap", "_base_addr", "_byte_offset", "_shape", "_strides", "_dtype", "__weakref__")

    def __init__(
        self,
        shape: Union[int, Sequence[int]],
        dtype: Any = "float32",
        *,
        _mmap_buf: Optional[mmap.mmap] = None,
        _base_addr: Optional[int] = None,
        _byte_offset: int = 0,
        _strides: Optional[Tuple[int, ...]] = None,
    ) -> None:
        if isinstance(shape, int):
            norm_shape: Tuple[int, ...] = (shape,)
        else:
            norm_shape = tuple(int(d) for d in shape)
        if any(d < 0 for d in norm_shape):
            raise ValueError(f"Negative dimensions are not allowed: {norm_shape}")

        spec = resolve_dtype(dtype)
        self._dtype = spec
        self._shape = norm_shape
        self._strides = _strides if _strides is not None else _c_contiguous_strides(norm_shape, spec.itemsize)
        self._byte_offset = int(_byte_offset)

        if _mmap_buf is not None:
            self._mmap = _mmap_buf
            assert _base_addr is not None
            self._base_addr = _base_addr
        else:
            raw_nbytes = math.prod(norm_shape) * spec.itemsize
            alloc_bytes = max(PAGE_SIZE, ((raw_nbytes + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE)
            self._mmap = mmap.mmap(-1, alloc_bytes)
            c_char_buf = ctypes.c_char.from_buffer(self._mmap)
            self._base_addr = ctypes.addressof(c_char_buf)

    @classmethod
    def empty(cls, shape: Union[int, Sequence[int]], dtype: Any = "float32") -> "MetalArray":
        """Allocate an uninitialized (OS zero-paged) page-aligned MetalArray."""
        return cls(shape, dtype=dtype)

    @classmethod
    def zeros(cls, shape: Union[int, Sequence[int]], dtype: Any = "float32") -> "MetalArray":
        """Allocate a zero-initialized page-aligned MetalArray."""
        # Anonymous mmap pages from macOS kernel are freshly zero-filled.
        return cls(shape, dtype=dtype)

    @classmethod
    def full(
        cls, shape: Union[int, Sequence[int]], fill_value: Union[int, float, bool], dtype: Any = "float32"
    ) -> "MetalArray":
        """Allocate a page-aligned MetalArray filled with `fill_value`."""
        arr = cls(shape, dtype=dtype)
        n = arr.size
        if n > 0:
            packed = struct.pack("<" + arr._dtype.fmt, fill_value)
            arr._mmap[: n * arr._dtype.itemsize] = packed * n
        return arr

    @classmethod
    def from_pointer(
        cls, ptr_addr: int, shape: Sequence[int], dtype: Any = "float32"
    ) -> "MetalArray":
        """Copy `prod(shape)` elements from raw C pointer `ptr_addr` into a MetalArray."""
        arr = cls(shape, dtype=dtype)
        nbytes = arr.nbytes
        if nbytes > 0:
            if not ptr_addr:
                raise ValueError("Null source pointer for non-empty MetalArray.")
            ctypes.memmove(arr.data_ptr, ptr_addr, nbytes)
        return arr

    @classmethod
    def from_buffer(
        cls,
        buf: Any,
        shape: Optional[Sequence[int]] = None,
        dtype: Any = None,
    ) -> "MetalArray":
        """Copy any PEP 3118 buffer (`array.array`, `memoryview`, `bytes`, `np.ndarray`) into a MetalArray."""
        mv = memoryview(buf)
        inferred = _dtype_from_pep3118(mv.format, mv.itemsize)
        target_spec = resolve_dtype(dtype) if dtype is not None else (inferred or _DTYPES["float32"])
        target_shape = tuple(int(d) for d in shape) if shape is not None else (mv.shape if mv.shape else (mv.nbytes // target_spec.itemsize,))
        out = cls(target_shape, dtype=target_spec)
        if out.size == 0:
            return out
        if inferred is not None and inferred.code == target_spec.code and mv.c_contiguous:
            raw = mv.cast("B")
            if len(raw) != out.nbytes:
                raise ValueError(f"Buffer byte size {len(raw)} does not match shape {target_shape}.")
            out._mmap[: out.nbytes] = raw
            return out
        flat_vals = mv.tolist()
        out._fill_from_nested(flat_vals)
        return out

    @classmethod
    def from_list(cls, data: Sequence[Any], dtype: Any = None) -> "MetalArray":
        """Construct a 1D, 2D, or ND page-aligned MetalArray from nested Python lists/tuples."""
        shape: List[int] = []
        cur: Any = data
        while isinstance(cur, (list, tuple)):
            shape.append(len(cur))
            if len(cur) == 0:
                break
            cur = cur[0]
        if not shape:
            raise TypeError("Expected a list or tuple to construct MetalArray.")

        if dtype is None:
            dtype = cls._infer_nested_dtype(data)
        arr = cls(shape, dtype=dtype)
        if arr.size > 0:
            arr._fill_from_nested(data)
        return arr

    @staticmethod
    def _infer_nested_dtype(data: Any) -> str:
        stack = [data]
        saw_bool = True
        saw_any = False
        while stack:
            item = stack.pop()
            if isinstance(item, (list, tuple)):
                stack.extend(item)
            elif item is None:
                return "float32"
            elif isinstance(item, bool):
                saw_any = True
            elif isinstance(item, float):
                return "float32"
            elif isinstance(item, int):
                saw_bool = False
                saw_any = True
            else:
                return "float32"
        if saw_any and saw_bool:
            return "bool"
        return "float32"

    def _fill_from_nested(self, data: Any) -> None:
        flat: List[Any] = []
        ndim = len(self._shape)

        def _walk(node: Any, depth: int) -> None:
            if depth == ndim:
                if isinstance(node, (list, tuple)):
                    raise ValueError("Ragged or deeper-than-expected nested sequence.")
                if node is None:
                    if not self._dtype.is_float:
                        raise ValueError("None / missing values require a floating-point dtype.")
                    flat.append(math.nan)
                elif self._dtype.is_bool:
                    flat.append(bool(node))
                elif self._dtype.is_float:
                    flat.append(float(node))
                else:
                    flat.append(int(node))
                return
            if not isinstance(node, (list, tuple)) or len(node) != self._shape[depth]:
                raise ValueError(f"Inconsistent sequence shape at axis {depth}; expected {self._shape}.")
            for child in node:
                _walk(child, depth + 1)

        _walk(data, 0)
        typecode = self._dtype.fmt
        if typecode in ("f", "d", "b", "B", "h", "H", "i", "I", "q", "Q"):
            buf = _py_array.array(typecode, flat)
            raw = memoryview(buf).cast("B")
            self._mmap[self._byte_offset : self._byte_offset + len(raw)] = raw
        else:
            struct.pack_into(f"<{len(flat)}{typecode}", self._mmap, self._byte_offset, *flat)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> Tuple[int, ...]:
        return self._strides

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def dtype(self) -> _DTypeSpec:
        return self._dtype

    @property
    def itemsize(self) -> int:
        return self._dtype.itemsize

    @property
    def size(self) -> int:
        return math.prod(self._shape)

    @property
    def nbytes(self) -> int:
        return self.size * self._dtype.itemsize

    @property
    def data_ptr(self) -> int:
        return self._base_addr + self._byte_offset

    @property
    def is_page_aligned(self) -> bool:
        return (self.data_ptr % PAGE_SIZE) == 0

    @property
    def is_c_contiguous(self) -> bool:
        if self.size <= 1:
            return True
        expected = _c_contiguous_strides(self._shape, self._dtype.itemsize)
        return all(
            s == exp or dim <= 1
            for s, exp, dim in zip(self._strides, expected, self._shape)
        )

    @property
    def T(self) -> "MetalArray":
        if self.ndim < 2:
            return self
        return MetalArray(
            self._shape[::-1],
            dtype=self._dtype,
            _mmap_buf=self._mmap,
            _base_addr=self._base_addr,
            _byte_offset=self._byte_offset,
            _strides=self._strides[::-1],
        )

    @property
    def __array_interface__(self) -> dict[str, Any]:
        return {
            "version": 3,
            "shape": self._shape,
            "typestr": self._dtype.typestr,
            "data": (self.data_ptr, False),
            "strides": self._strides if self._shape else None,
        }

    def __buffer__(self, flags: int) -> memoryview:
        _ = flags
        if not self.is_c_contiguous:
            raise BufferError("MetalArray view is not C-contiguous; call .contiguous() before memoryview().")
        raw = memoryview(self._mmap)[self._byte_offset : self._byte_offset + self.nbytes]
        if self.size == 0:
            if self.ndim == 1:
                return raw.cast(self._dtype.fmt)
            raise BufferError("Zero-sized multi-dimensional memoryview cast is unsupported by Python.")
        return raw.cast(self._dtype.fmt, shape=self._shape)

    def __dlpack_device__(self) -> Tuple[int, int]:
        return (1, 0)  # kDLCPU (Apple Silicon unified memory)

    def __dlpack__(self, *, stream: Any = None, max_version: Any = None, dl_device: Any = None, copy: Optional[bool] = None) -> Any:
        _ = (stream, max_version)
        if dl_device is not None and tuple(dl_device) not in ((1, 0), (8, 0)):
            raise BufferError(f"Unsupported dl_device {dl_device} for MetalArray.")
        src = self.copy() if copy else self
        ndim = src.ndim
        c_shape = (ctypes.c_int64 * max(ndim, 1))(*src._shape)
        elem_strides = tuple(st // src._dtype.itemsize for st in src._strides)
        if any(st % src._dtype.itemsize != 0 for st in src._strides):
            src = src.contiguous()
            elem_strides = tuple(st // src._dtype.itemsize for st in src._strides)
        c_strides = (ctypes.c_int64 * max(ndim, 1))(*elem_strides)

        managed = _DLManagedTensor()
        dev_type = 8 if (dl_device is not None and tuple(dl_device)[0] == 8) else 1
        managed.dl_tensor.data = ctypes.c_void_p(src.data_ptr)
        managed.dl_tensor.device = _DLDevice(dev_type, 0)
        managed.dl_tensor.ndim = ndim
        managed.dl_tensor.dtype = _DLDataType(src._dtype.dl_code, src._dtype.dl_bits, 1)
        managed.dl_tensor.shape = ctypes.cast(c_shape, ctypes.POINTER(ctypes.c_int64))
        managed.dl_tensor.strides = ctypes.cast(c_strides, ctypes.POINTER(ctypes.c_int64))
        managed.dl_tensor.byte_offset = 0
        managed.manager_ctx = None
        managed.deleter = _dlpack_managed_deleter

        addr = ctypes.addressof(managed)
        _LIVE_DLPACK_EXPORTS[addr] = (managed, c_shape, c_strides, src)
        return _PyCapsule_New(ctypes.c_void_p(addr), b"dltensor", _dlpack_capsule_destructor)

    def to_descriptor(self) -> XGBufferDescriptor:
        """Build a zero-JSON binary XGBufferDescriptor pointing directly to unified memory."""
        if self.ndim not in (1, 2):
            raise ValueError(f"XGBufferDescriptor supports 1D or 2D MetalArray, got {self.ndim}D {self._shape}.")
        desc = XGBufferDescriptor()
        desc.data = ctypes.c_void_p(self.data_ptr)
        desc.ndim = self.ndim
        desc.shape[0] = self._shape[0]
        desc.shape[1] = self._shape[1] if self.ndim == 2 else 1
        desc.strides[0] = self._strides[0]
        desc.strides[1] = self._strides[1] if self.ndim == 2 else 0
        desc.dtype_code = self._dtype.code
        desc.mask_data = None
        desc.mask_bit_offset = 0
        desc.mask_mode = INGEST_MASK_NONE
        desc.flags = XGBUF_FLAG_ZERO_COPY | (XGBUF_FLAG_CONTIGUOUS if self.is_c_contiguous else 0)
        return desc

    def contiguous(self) -> "MetalArray":
        """Return self if already C-contiguous, else a newly materialized page-aligned copy."""
        if self.is_c_contiguous and self._byte_offset == 0:
            return self
        return self.copy()

    def copy(self) -> "MetalArray":
        """Return an independent page-aligned C-contiguous copy."""
        out = MetalArray(self._shape, dtype=self._dtype)
        if self.size == 0:
            return out
        if self.is_c_contiguous:
            ctypes.memmove(out.data_ptr, self.data_ptr, self.nbytes)
            return out
        out._fill_from_nested(self.tolist())
        return out

    def reshape(self, *new_shape: Any) -> "MetalArray":
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            dims = list(new_shape[0])
        else:
            dims = list(new_shape)
        neg_idx = -1
        known_prod = 1
        for i, d in enumerate(dims):
            d = int(d)
            dims[i] = d
            if d < 0:
                if neg_idx != -1:
                    raise ValueError("Can only specify one unknown dimension (-1).")
                neg_idx = i
            else:
                known_prod *= d
        if neg_idx != -1:
            if known_prod == 0 or self.size % known_prod != 0:
                raise ValueError(f"Cannot reshape array of size {self.size} into shape {tuple(dims)}.")
            dims[neg_idx] = self.size // known_prod
        target = tuple(dims)
        if math.prod(target) != self.size:
            raise ValueError(f"Cannot reshape array of size {self.size} into shape {target}.")
        base = self if self.is_c_contiguous else self.copy()
        return MetalArray(
            target,
            dtype=base._dtype,
            _mmap_buf=base._mmap,
            _base_addr=base._base_addr,
            _byte_offset=base._byte_offset,
        )

    def flatten(self) -> "MetalArray":
        return self.reshape(self.size).copy()

    def _read_scalar_at(self, byte_off: int) -> Any:
        val = struct.unpack_from("<" + self._dtype.fmt, self._mmap, byte_off)[0]
        return val

    def _write_scalar_at(self, byte_off: int, value: Any) -> None:
        struct.pack_into("<" + self._dtype.fmt, self._mmap, byte_off, value)

    def tolist(self) -> List[Any]:
        """Convert MetalArray (arbitrary dimensionality and strides) to nested Python lists."""
        if self.size == 0:
            def _empty_nested(dims: Tuple[int, ...]) -> List[Any]:
                if len(dims) <= 1:
                    return []
                return [_empty_nested(dims[1:]) for _ in range(dims[0])]

            return _empty_nested(self._shape)

        if self.ndim == 1 and self.is_c_contiguous and self._dtype.fmt in ("f", "d", "b", "B", "h", "H", "i", "I", "q", "Q"):
            arr = _py_array.array(self._dtype.fmt)
            arr.frombytes(memoryview(self._mmap)[self._byte_offset : self._byte_offset + self.nbytes])
            return arr.tolist()

        fmt_unpack = "<" + self._dtype.fmt
        buf = self._mmap
        shape = self._shape
        strides = self._strides
        ndim = len(shape)

        if ndim == 2 and strides[1] == self._dtype.itemsize and self._dtype.fmt in ("f", "d", "i", "q"):
            cols = shape[1]
            row_bytes = cols * self._dtype.itemsize
            row_fmt = f"<{cols}{self._dtype.fmt}"
            return [
                list(struct.unpack_from(row_fmt, buf, self._byte_offset + r * strides[0]))
                for r in range(shape[0])
            ]

        def _rec(axis: int, off: int) -> Any:
            if axis == ndim - 1:
                st = strides[axis]
                return [struct.unpack_from(fmt_unpack, buf, off + i * st)[0] for i in range(shape[axis])]
            st = strides[axis]
            return [_rec(axis + 1, off + i * st) for i in range(shape[axis])]

        return _rec(0, self._byte_offset)

    def _normalize_index(self, key: Any) -> Tuple[int, Tuple[int, ...], Tuple[int, ...], bool]:
        items = list(key) if isinstance(key, tuple) else [key]
        ell_count = sum(1 for k in items if isinstance(k, EllipsisType))
        if ell_count > 1:
            raise IndexError("An index can only have a single ellipsis ('...').")
        if ell_count == 1:
            pos = next(i for i, k in enumerate(items) if isinstance(k, EllipsisType))
            fill = self.ndim - (len(items) - 1)
            if fill < 0:
                raise IndexError("Too many indices for MetalArray.")
            items = items[:pos] + [slice(None)] * fill + items[pos + 1 :]
        if len(items) > self.ndim:
            raise IndexError(f"Too many indices ({len(items)}) for {self.ndim}D MetalArray.")
        while len(items) < self.ndim:
            items.append(slice(None))

        byte_off = self._byte_offset
        new_shape: List[int] = []
        new_strides: List[int] = []
        for axis, idx in enumerate(items):
            dim = self._shape[axis]
            st = self._strides[axis]
            if isinstance(idx, slice):
                start, stop, step = idx.indices(dim)
                length = len(range(start, stop, step))
                if length > 0:
                    byte_off += start * st
                new_shape.append(length)
                new_strides.append(st * step)
            elif isinstance(idx, int) or (HAS_NUMPY and isinstance(idx, _np.integer)):
                i = int(idx)
                if i < 0:
                    i += dim
                if i < 0 or i >= dim:
                    raise IndexError(f"Index {idx} is out of bounds for axis {axis} with size {dim}.")
                byte_off += i * st
            else:
                raise TypeError(f"Unsupported index type: {type(idx)}")
        is_scalar = len(new_shape) == 0
        return byte_off, tuple(new_shape), tuple(new_strides), is_scalar

    def __getitem__(self, key: Any) -> Any:
        byte_off, new_shape, new_strides, is_scalar = self._normalize_index(key)
        if is_scalar:
            return self._read_scalar_at(byte_off)
        return MetalArray(
            new_shape,
            dtype=self._dtype,
            _mmap_buf=self._mmap,
            _base_addr=self._base_addr,
            _byte_offset=byte_off,
            _strides=new_strides,
        )

    def __setitem__(self, key: Any, value: Any) -> None:
        byte_off, new_shape, new_strides, is_scalar = self._normalize_index(key)
        if is_scalar:
            self._write_scalar_at(byte_off, value)
            return
        view = MetalArray(
            new_shape,
            dtype=self._dtype,
            _mmap_buf=self._mmap,
            _base_addr=self._base_addr,
            _byte_offset=byte_off,
            _strides=new_strides,
        )
        if isinstance(value, (int, float, bool)):
            for idx_pos in _nd_indices(new_shape):
                view[idx_pos] = value
        elif isinstance(value, MetalArray):
            vals = value.tolist()
            for idx_pos in _nd_indices(new_shape):
                v: Any = vals
                for p in idx_pos:
                    v = v[p]
                view[idx_pos] = v
        else:
            for idx_pos in _nd_indices(new_shape):
                v = value
                for p in idx_pos:
                    v = v[p]
                view[idx_pos] = v

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of unsized object")
        return self._shape[0]

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        preview = self.tolist() if self.size <= 16 else "..."
        return f"MetalArray({preview}, shape={self._shape}, dtype='{self._dtype.name}')"


def _nd_indices(shape: Tuple[int, ...]) -> Iterator[Tuple[int, ...]]:
    if not shape:
        yield ()
        return
    for i in range(shape[0]):
        for rest in _nd_indices(shape[1:]):
            yield (i,) + rest


def try_buffer_descriptor(obj: Any, *, allow_1d: bool = False) -> Optional[Tuple[XGBufferDescriptor, Any]]:
    """Extract `(XGBufferDescriptor, keepalive_owner)` from a MetalArray or PEP 3118 buffer.

    Returns `None` if `obj` does not support PEP 3118, has unsupported dtype/byteorder
    (e.g. object array or big-endian), or has incompatible dimensionality.
    """
    if isinstance(obj, MetalArray):
        if obj.ndim == 2 or (allow_1d and obj.ndim == 1):
            return obj.to_descriptor(), obj
        return None

    # Do not intercept MaskedArray via raw PEP 3118 since its mask would be ignored.
    if HAS_NUMPY and isinstance(obj, _np.ma.MaskedArray):
        return None

    pybuf = _PyBuffer()
    flags = _PyBUF_STRIDES | _PyBUF_FORMAT
    try:
        rc = _PyObject_GetBuffer(obj, ctypes.byref(pybuf), flags)
    except (BufferError, TypeError, ValueError):
        try:
            rc = _PyObject_GetBuffer(obj, ctypes.byref(pybuf), _PyBUF_ND | _PyBUF_FORMAT)
        except (BufferError, TypeError, ValueError):
            return None
    if rc != 0:
        return None

    try:
        if pybuf.suboffsets:
            return None
        ndim = int(pybuf.ndim)
        if ndim == 0 or ndim > 2 or (ndim == 1 and not allow_1d):
            return None
        fmt_str = pybuf.format.decode("ascii", errors="ignore") if pybuf.format else "B"
        spec = _dtype_from_pep3118(fmt_str, int(pybuf.itemsize))
        if spec is None:
            return None

        s0 = int(pybuf.shape[0]) if pybuf.shape else (int(pybuf.len) // int(pybuf.itemsize))
        s1 = int(pybuf.shape[1]) if (ndim == 2 and pybuf.shape) else 1
        if pybuf.strides:
            st0 = int(pybuf.strides[0])
            st1 = int(pybuf.strides[1]) if ndim == 2 else 0
        else:
            st1 = int(pybuf.itemsize) if ndim == 2 else 0
            st0 = (s1 * int(pybuf.itemsize)) if ndim == 2 else int(pybuf.itemsize)

        desc = XGBufferDescriptor()
        desc.data = pybuf.buf
        desc.ndim = ndim
        desc.shape[0] = s0
        desc.shape[1] = s1
        desc.strides[0] = st0
        desc.strides[1] = st1
        desc.dtype_code = spec.code
        desc.mask_data = None
        desc.mask_bit_offset = 0
        desc.mask_mode = INGEST_MASK_NONE
        desc.flags = 0
        return desc, obj
    finally:
        _PyBuffer_Release(ctypes.byref(pybuf))


def is_dlpack_capsule(obj: Any) -> bool:
    """True when `obj` is a DLPack PyCapsule (`dltensor` or `dltensor_versioned`)."""
    if type(obj).__name__ != "PyCapsule":
        return False
    try:
        name = _PyCapsule_GetName(obj)
    except Exception:  # pylint: disable=broad-except
        return False
    return name in (b"dltensor", b"dltensor_versioned", b"used_dltensor", b"used_dltensor_versioned")


def has_dlpack_protocol(obj: Any) -> bool:
    """True when `obj` is a DLPack capsule or exposes `__dlpack__`."""
    return is_dlpack_capsule(obj) or hasattr(obj, "__dlpack__")


def extract_dlpack_capsule(obj: Any) -> Tuple[ctypes.c_void_p, int, Any]:
    """Obtain `(managed_tensor_ptr, is_versioned, capsule_keepalive)` and mark capsule used."""
    cap = obj if is_dlpack_capsule(obj) else obj.__dlpack__()
    name = _PyCapsule_GetName(cap)
    if name in (b"used_dltensor", b"used_dltensor_versioned"):
        raise ValueError("DLPack tensor capsule has already been consumed.")
    if name == b"dltensor":
        ptr = _PyCapsule_GetPointer(cap, b"dltensor")
        _PyCapsule_SetName(cap, b"used_dltensor")
        return ctypes.c_void_p(ptr), 0, cap
    if name == b"dltensor_versioned":
        ptr = _PyCapsule_GetPointer(cap, b"dltensor_versioned")
        _PyCapsule_SetName(cap, b"used_dltensor_versioned")
        return ctypes.c_void_p(ptr), 1, cap
    raise TypeError(f"Unexpected DLPack PyCapsule name: {name!r}")


def extract_arrow_capsules(obj: Any) -> Tuple[ctypes.c_void_p, ctypes.c_void_p, Any]:
    """Extract `(schema_ptr, array_ptr, keepalive)` from any `__arrow_c_array__` object."""
    schema_cap, array_cap = obj.__arrow_c_array__()
    s_ptr = _PyCapsule_GetPointer(schema_cap, b"arrow_schema")
    a_ptr = _PyCapsule_GetPointer(array_cap, b"arrow_array")
    return ctypes.c_void_p(s_ptr), ctypes.c_void_p(a_ptr), (schema_cap, array_cap, obj)
