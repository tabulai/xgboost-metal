# Third-party code

The Python compatibility adapter in `python-package/xgboost` and the public C API
header in `include/xgboost/c_api.h` originate from XGBoost, commit
`6d2dbe19a099aa046178be8cfc303bfec1a623a6`, under the Apache License 2.0 (see
`LICENSE`). The native XGBoost engine is not included in this library.

The Metal implementation also draws algorithmic guidance from XGBoost's CUDA
implementation under the same Apache-2.0 license: deterministic fixed-point
histograms, paired atomic accumulation, histogram subtraction, feature grouping,
segmented row partitioning, row sampling, SIMD split scans, growth scheduling,
histogram caching, weighted feature sketches, weighted radix selection and
smooth regression objectives.
`third_party/nlohmann/json.hpp` is nlohmann/json v3.12.0, under the MIT License
in `third_party/nlohmann/LICENSE.MIT`. SHA-256:
`aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63`.

`kernels/sketch_sort.metal.h` adapts libc++ 21.1.6 sorting algorithms from the
LLVM Project to Metal. It preserves the pinned Apple reference's ordering of
equal feature values when rounding their weights or retaining signed zero can
affect histogram cuts. The adapted device implementation uses an explicit stack
and a value-only comparator. See `third_party/llvm/LICENSE.TXT` for the
Apache License 2.0 with LLVM exceptions. The license text comes from
`llvmorg-21.1.6/libcxx/LICENSE.TXT` in `llvm/llvm-project`; its SHA-256 is
`539dd7aed86e8a4f12cbdd0e6c50c189c7d74847e4fecc64ce2c6ee3a01da38b`.
