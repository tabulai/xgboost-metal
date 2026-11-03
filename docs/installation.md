# Installation

xgboost-metal requires an Apple silicon Mac, macOS 14 or later, and
Python 3.10 or later. Source builds require Xcode with the Metal compiler and
a C++20 compiler. The build backend uses CMake 3.24 or later.

## Install from source

Create a dedicated environment. Upstream XGBoost and this distribution use the
same Python import name and should be installed in separate environments.

```sh
git clone https://github.com/tabulai/xgboost-metal.git
cd xgboost-metal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

NumPy and SciPy are installed as runtime dependencies. For scikit-learn
estimators and pandas dataframes, install the optional extra:

```sh
python -m pip install ".[sklearn]"
```

For Arrow input, also install `pyarrow`.

## Check the installation

```sh
python -c 'import xgboost as xgb; print(xgb.__version__); print(xgb.build_info())'
```

The build information includes `USE_METAL`, the Metal device name, and the path
to the loaded native library. `USE_METAL` should be `True`.

## Build a wheel

From the repository root, with the environment activated:

```sh
python -m pip install build
python -m build
```

This produces a source distribution and a wheel under `dist/`. Install the
wheel into a separate deployment environment using its full path:

```sh
python -m pip install /path/to/xgboost_metal-VERSION-py3-none-macosx_14_0_arm64.whl
```

Replace the example filename with the wheel produced by the build. The wheel
contains the native library and compiled Metal kernels, so deployment does not
require the Metal compiler. The deployment machine still needs Apple silicon
and macOS 14 or later.

## Troubleshooting

If Xcode reports a missing Metal toolchain, install the component and retry:

```sh
xcodebuild -downloadComponent MetalToolchain
```

If Python loads an unexpected backend, check `xgboost.__file__` and
`xgboost.build_info()`. Recreate the environment if another XGBoost distribution
was installed into it. For a development build, follow
[Contributing](../CONTRIBUTING.md) so the source adapter can find the local library.

Device errors for `cpu` or `cuda` are expected: this package supports Metal.
See [compatibility](compatibility.md) before migrating an existing application.
