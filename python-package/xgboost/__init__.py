"""XGBoost: eXtreme Gradient Boosting library.

Contributors: https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md
"""

from . import (
    collective,
    interpret,
    tracker,  # noqa
)
from ._c_api import _py_version
from ._metal_array import MetalArray
from ._sparse import COOMatrix, CSCMatrix, CSRMatrix
from .config import config_context, get_config, set_config
from .core import (
    Booster,
    DataIter,
    DMatrix,
    ExtMemQuantileDMatrix,
    QuantileDMatrix,
    build_info,
)
from .tracker import RabitTracker  # noqa
from .training import cv, train

try:
    from .plotting import plot_importance, plot_tree, to_graphviz
except ImportError:
    pass

try:
    from .sklearn import (
        XGBClassifier,
        XGBModel,
        XGBRanker,
        XGBRegressor,
        XGBRFClassifier,
        XGBRFRegressor,
    )
except ImportError:
    pass


__version__ = _py_version()

try:  # Some interpreters skip .pth files; see _xgboost_metal_dist.
    from _xgboost_metal_dist import install as _install_metadata_finder
except ImportError:  # pragma: no cover
    pass
else:
    _install_metadata_finder()
    del _install_metadata_finder


__all__ = [
    # core
    "DMatrix",
    "QuantileDMatrix",
    "ExtMemQuantileDMatrix",
    "MetalArray",
    "CSRMatrix",
    "CSCMatrix",
    "COOMatrix",
    "Booster",
    "DataIter",
    "train",
    "cv",
    # utilities
    "RabitTracker",
    "build_info",
    "plot_importance",
    "plot_tree",
    "to_graphviz",
    "set_config",
    "get_config",
    "config_context",
    # sklearn
    "XGBModel",
    "XGBClassifier",
    "XGBRegressor",
    "XGBRanker",
    "XGBRFClassifier",
    "XGBRFRegressor",
    # collective
    "collective",
    # interpretability
    "interpret",
]
