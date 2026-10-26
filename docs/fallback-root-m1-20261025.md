# Fallback numeric root experiment

Measured on an Apple M1 Ultra on October 25, 2026 (EDT; receipts use October 26
UTC), against `release/clean-source` commit
`bcc18099b172d8e95497e40bf5fde3e54936212e`.

`metal_fallback_parallel_root=1` reuses the existing 256-thread numeric root
kernel in the host-driven builder. It was **off by default** in the tested
candidate, `bfe9a0254ededdf530ca53b4d92313b7682f2d2b`; the subsequent
[M4 validation](fallback-root-m4-20261025.md) supports enabling it by default.
Descendant splits, histogram construction, host scheduling, feature masks, monotone
bounds, and final root selection retain their existing implementations.
The candidate and baseline embedded Metal libraries are byte-identical.
See [eligibility and replay worker](benchmarking.md) for controls and exclusions.

## Confirmed training measurements

All workloads use 65,536 rows, 32 numeric features, 15% missing values, 20 warmup
rounds, and 100 measured rounds. Each arm runs in a fresh process with a pinned
package path and native SHA-256. Eight cycles alternate arm order, rotating
workload order, with three fresh fits per arm per cycle. Training time includes
booster creation/training but excludes matrix construction, prediction, profile
reads, hashing, and artifact writes. Shader validation is disabled for timing.

| Workload | Option off, median seconds | Option on, median seconds | Paired cycle speedup | Bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Depth 1, 256 bins | 0.18727 | 0.14315 | 1.317x | 1.302–1.339x |
| Depth 1, 32 bins | 0.13816 | 0.13738 | 1.035x | 0.984–1.103x |
| Depth 6, 256 bins, forced host builder | 1.86956 | 1.76707 | 1.056x | 1.037–1.079x |

Speedup is the geometric mean of eight paired cycle median ratios. The
bootstrap resamples whole cycles (10,000 draws, seed 42), not individual fits.
The 32-bin result does not establish a benefit. These intervals describe this
fixed synthetic workload and machine, not uncertainty across applications.

All **144 confirmation fits** have byte-identical UBJ models, original and
canonical JSON models, and predictions within each workload. The preliminary
baseline/off/on screen also matched all 27 fits. A further 36 fits covering
monotone constraints, interaction constraints, column sampling, uniform and
gradient-based row sampling, and lossguide growth matched exactly. Those
single-block timing screens are exploratory and are not used for the headline
speedups.

The OpenMP CPU performance build (`7f2e1f2242f8f0500cf4e22fab7a7ab4ee302120`,
default threads with `OMP_NUM_THREADS` unset) took median 0.16998, 0.16256, and
0.62894 seconds for the three table workloads, respectively. All nine CPU
prediction arrays matched the Metal arrays exactly. The depth-six fallback
still needs substantial improvement to outperform that CPU build.

## Correctness checks

The initial focused/regression invocation passed 274 cases with Metal debug
and shader validation enabled. After adding a sampled multiclass/parallel-tree
case, the final focused file passed all 37 cases under the same validation.
These invocations overlap. Tests cover exact model/prediction/leaf comparisons,
independent pinned CPU reference checks, constraints, sampling, missing values,
layout/bin boundaries, ties, zero Hessians, invalid roots, runtime toggles, and
ineligible paths. The API audit found all 99 declared symbols exported.

The first monotone benchmark stopped during output canonicalization because
the unchanged model contained infinite split thresholds. Its failed receipt
is preserved. The repaired worker preserves raw exports and explicitly tags
nonfinite metadata in a separate canonical comparison format; predictions must
remain finite. Failed-fit timing was not reused. The installed native library
was not replaced.

## AUC/AUCPR diagnosis

`benchmarks/auc_stages.cpp` profiles the unchanged baseline kernels with separate
sort/merge and reduction commands, plus a combined-command control. It tests
100,000 and 1,000,000 rows, one and four groups, unique scores and 32 tie buckets,
and both metrics. Weights are all one. Every warmup and measured pair compares
the entire sorted buffer, per-class float4 outputs, and host aggregate bitwise;
a separate 4,097-row multiclass run enables Metal shader validation.

Reduction accounts for **99.2–99.9%** of the separately measured GPU time. For
100,000 binary rows with unique scores, median sort/merge and reduction are
0.170 ms and 61.559 ms for AUC, and 0.170 ms and 91.083 ms for AUCPR. At one
million binary rows they are about 0.97 ms and 642 ms / 937 ms, respectively.
The combined controls support the same conclusion.

These are stage diagnostics: separating commands adds a wait and changes cache
and scheduling behavior. The profiler is not an independent CPU metric
correctness test. No production metric code was changed. A parallel reducer
is the strongest next target, with weighted sums, tie groups, and stable AUCPR
integration requiring separate correctness work. General descendant split
parallelization remains future work. The subsequent
[M4 validation](fallback-root-m4-20261025.md) covers the fallback root change.

Build the diagnostic against an already configured source tree:

```sh
clang++ -std=c++20 -O3 -mmacosx-version-min=14.0 \
  -Iinclude -Ithird_party -Ibuild benchmarks/auc_stages.cpp src/runtime.mm \
  -framework Foundation -framework Metal -o build/auc_stages
build/auc_stages --rows 100000 --groups 1 --repeats 5 --warmup 1
```

The evidence archive `xgb-fallback-numeric-m1-20261025-portable.tar.gz` contains raw
receipts, saved models/predictions, summaries, replay scripts, and a per-file
manifest. Size: 66,164,742 bytes. SHA-256:
`57137595a10a99c9597cd6da61b64e0dfa4f927bb343e2119d9f8ce774ff2362`.
