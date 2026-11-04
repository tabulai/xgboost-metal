"""Experimental support for a new objective interface with target dimension
reduction.

.. warning::

  Do not use this module unless you want to participate in development.

.. versionadded:: 3.2.0

"""

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

from ._data_utils import (
    _ensure_np_dtype,
    _is_flatten,
    array_interface,
    cuda_array_interface,
)
from ._metal_array import MetalArray
from ._typing import HAS_NUMPY, ArrayLike, NumpyOrCupy
from .compat import _is_cupy_alike

if HAS_NUMPY:
    import numpy as np
else:
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .core import DMatrix


class Objective(ABC):
    """Base class for custom objective functions.

    .. warning::

        Do not use this class unless you want to participate in development.

    .. versionadded:: 3.2.0

    """

    @abstractmethod
    def __call__(
        self, iteration: int, y_pred: ArrayLike, dtrain: "DMatrix"
    ) -> Tuple[ArrayLike, ArrayLike]: ...


class TreeObjective(Objective):
    """Base class for tree-specific custom objective functions.

    .. warning::

        Do not use this class unless you want to participate in development.

    .. versionadded:: 3.2.0

    """

    # pylint: disable=unused-argument
    def split_grad(
        self, iteration: int, grad: ArrayLike, hess: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike] | None:
        """Provide a different gradient type for finding tree structures."""
        return None


def _grad_arrinf(array: NumpyOrCupy, n_samples: int) -> bytes:
    # Can we check for __array_interface__ instead of a specific type instead?
    msg = (
        "Expecting `np.ndarray`, `MetalArray`, or `cupy.ndarray` for gradient and hessian."
        f" Got: {type(array)}"
    )
    is_np = HAS_NUMPY and np is not None and isinstance(array, np.ndarray)
    is_metal = isinstance(array, MetalArray)
    if not is_np and not is_metal and not _is_cupy_alike(array):
        raise TypeError(msg)

    if array.shape[0] != n_samples and _is_flatten(array):
        warnings.warn(
            "Since 2.1.0, the shape of the gradient and hessian is required to"
            " be (n_samples, n_targets) or (n_samples, n_classes).",
            FutureWarning,
        )
        array = array.reshape(n_samples, array.size // n_samples)

    if is_np:
        array, _ = _ensure_np_dtype(array, array.dtype)
        interface = array_interface(array)
    elif is_metal:
        interface = array_interface(array.contiguous())
    elif _is_cupy_alike(array):
        interface = cuda_array_interface(array)
    else:
        raise TypeError(msg)

    return interface
