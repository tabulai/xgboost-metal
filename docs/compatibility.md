# Compatibility

This package implements the XGBoost Python API on Apple Metal. The API baseline
is XGBoost 3.5.0-dev at commit
`6d2dbe19a099aa046178be8cfc303bfec1a623a6`. Reporting that version does not
imply compatibility with every upstream feature.

## Version reporting

Starting with release 0.1.1, the `xgboost` package reports the baseline version
exactly as upstream XGBoost does at that commit, so version checks in other
libraries behave as they would with upstream. The distribution keeps its own release number. (Release 0.1.0
reported `0.1.0` from `xgboost.__version__`.)

| Query | Result |
| --- | --- |
| `xgboost.__version__` | `3.5.0-dev` |
| `importlib.metadata.version("xgboost")` | `3.5.0.dev0` |
| `importlib.metadata.version("xgboost-metal")` | The xgboost-metal release |

The wheel installs a startup hook, `xgboost_metal.pth`, that answers
`importlib.metadata` lookups for `xgboost` by name, before or after
`import xgboost`. pip does not see a separate `xgboost` distribution: `pip list`
and `pip freeze` show only `xgboost-metal`, and installing a package that
depends on `xgboost` still installs upstream XGBoost over this package's files.

## Supported workflows

| Area | Support |
| --- | --- |
| Platform | Apple silicon, macOS 14+, Python 3.10+ |
| Device | Metal; selected by default |
| Tree training | Histogram method, depthwise and loss-guided growth, regularization, constraints, sampling, parallel trees, and dropout |
| Linear training | `gblinear` with `shotgun` or `coord_descent` |
| Objectives | Regression, binary and multiclass classification, quantile and expectile regression, normal regression, AFT/Cox survival, and NDCG/MAP/pairwise ranking |
| Data | Dense and strided arrays, SciPy sparse input, pandas, Arrow, categorical features, missing values, weights, and in-memory iterators |
| Matrices | `DMatrix`, `QuantileDMatrix`, and ordinary DMatrix binary file exchange |
| Python workflows | `train`, `cv`, sklearn estimators, callbacks, early stopping, custom objectives, and continued training |
| Prediction | Values, raw margins, leaf indices, feature contributions, and interactions |
| Models | JSON/UBJSON, pickle, model dumps, configuration, and tree-model slicing |
| Existing trees | Refresh and prune with ordinary DMatrix input |

## Platform and execution limits

- CPU and CUDA execution are not available in this distribution.
- Tree construction uses `tree_method="hist"` or `"auto"`; exact and approximate
  tree construction are not implemented.
- Distributed training, Spark/Dask execution, and distributed trackers are not
  supported, even though the Python adapter contains those interface modules.
- External-memory paging and external-memory quantile training are not supported.
- The C ABI is exported for the Python adapter. Other language bindings have not
  been validated against this backend.

## Data and model restrictions

Sparse data retains CSR entries for export and binary exchange, but training
expands the input to dense storage. Training with duplicate sparse entries does
not have established upstream equivalence. Use canonical sparse input and size
memory for the dense representation.

Quantile, proxy, and iterator matrices do not support `save_binary()`. Refresh
and prune require ordinary `DMatrix` input. Quantile validation and test matrices
should use the training matrix as `ref` and matching `max_bin` settings.

The linear booster does not support categorical or quantile matrices, leaf
predictions, in-place prediction, or model slicing. Its learning rate defaults
to 0.5. The `shotgun` updater supports cyclic and shuffle feature selection;
`coord_descent` also supports random, greedy, and thrifty selection.
Query-weighted ranking with `gblinear` is not supported. The deterministic
`shotgun` implementation does not reproduce concurrent CPU update ordering.

Survival, ranking, expectile, and normal objectives take one label column.
Multioutput tree training supports categorical data, sampling, and exact
contributions and interactions. Reduced-gradient tree objectives do not support
monotonic constraints or the linear booster. Approximate contributions and
interactions have additional restrictions for multioutput trees.

Tree dropout accepts the standard dropout parameters and legacy DART models.
For leaf or contribution predictions starting after round zero, slice the model
before predicting.

## Numerical behavior and upgrades

The compatibility suite compares training, prediction, model exchange, and
matrix operations with the pinned XGBoost CPU build. Floating-point behavior can
depend on objective, data distribution, platform, and library version; exact
agreement in the tests is not a guarantee for every input. Ranking and survival
likelihood metrics can differ within numerical tolerances.

Before migrating a workload, compare its validation metric and representative
predictions, then test model saving, loading, and continued training if used.
Unsupported native operations report errors. Use the
[contribution guide](../CONTRIBUTING.md) to run the compatibility suite.
