# Contributing

Build and test on an Apple silicon Mac with macOS 14 or later and the Xcode
Metal toolchain. Keep runtime changes, tests, and user documentation together
in a focused pull request.

## Development environment

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy pandas pyarrow scikit-learn pytest cmake ninja build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
PYTHONPATH=python-package python -c 'import xgboost; print(xgboost.build_info())'
```

CMake writes the native library to `lib/`. The source Python adapter locates
that library when `PYTHONPATH=python-package` is set. The pytest configuration
adds the package path automatically.

## Compatibility tests

Tests compare results with a separate XGBoost CPU installation at commit
`6d2dbe19a099aa046178be8cfc303bfec1a623a6`. Prepare it at the expected location:

```sh
git clone https://github.com/dmlc/xgboost.git ref/xgboost
git -C ref/xgboost checkout 6d2dbe19a099aa046178be8cfc303bfec1a623a6
git -C ref/xgboost submodule update --init --recursive
cmake -S ref/xgboost -B build-reference -G Ninja \
  -DUSE_CUDA=OFF -DUSE_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-reference
python -m pytest -q
python tools/audit_api.py
```

This CPU build is used only by the tests. Disabling OpenMP makes the local
compatibility setup independent of an additional OpenMP installation; it is not
a multicore performance baseline.

To check GPU resource bindings and memory accesses, run:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 python -m pytest -q
```

Generated numeric constants can be checked without rewriting their files:

```sh
python tools/generate_cox_constants.py --check
python tools/generate_survival_constants.py --check
```

## Package checks

Build the wheel and source distribution, then install the wheel into a separate
environment:

```sh
python -m build
python3 -m venv .wheel-venv
.wheel-venv/bin/python -m pip install dist/*.whl
.wheel-venv/bin/python tools/smoke_wheel.py
```

The wheel smoke test exercises training, prediction, model exchange, and matrix
operations using the installed library. `tools/verify_package.py` checks the
package files and license notices; run it with `--help` for the required wheel,
installed-library, and optional source-distribution paths.

## Releases

Releases are published as GitHub release assets, not on PyPI.

1. Set the version in `pyproject.toml` and in the `project()` call in
   `CMakeLists.txt`, and update the wheel URLs in `README.md` and
   `docs/installation.md`. The wheel embeds the README, so do this first.
2. Commit, build, and run the package checks above.
3. Create a GitHub release tagged `vVERSION` at that commit, and attach the
   wheel and source distribution from `dist/`.

## Pull requests

Describe the behavior change and the relevant checks. Add regression coverage
for functional changes and document any API or compatibility impact. Preserve
third-party notices when modifying adapted code. Use the
[benchmark guide](docs/benchmarking.md) for performance changes, and include both
timings and model quality.
