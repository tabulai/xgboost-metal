# Fallback numeric root M4 validation

Measured on an Apple M4 Max with 128 GB unified memory, macOS 26.6.2,
Python 3.13.3, and AC power on October 25, 2026 (EDT; receipts use October 26
UTC). The opt-in candidate was
`bfe9a0254ededdf530ca53b4d92313b7682f2d2b`, based on `release/clean-source`
commit `bcc18099b172d8e95497e40bf5fde3e54936212e`.

M4 validation passed and `metal_fallback_parallel_root=1` is enabled by default
for eligible roots. The final build passed 3,863 tests, all 864 benchmark fits
matched exactly within their workloads, and 256-bin stumps improved
consistently. An independent repeat did not reproduce the gradient-sampling
tail slowdown found in an earlier batch; both results are retained below.
`metal_fallback_parallel_root=0` retains the original evaluator.
See [eligibility and controls](benchmarking.md) for the unchanged exclusions.

## Training measurements

All workloads use 65,536 rows, 32 numeric features, 15% missing values, 20 warmup
rounds, and 100 measured rounds. Each arm runs in a fresh process, with three
fresh matrix/booster fits per process, a pinned package path and native SHA-256,
and `OMP_NUM_THREADS=1`. The host tree builder is forced for both arms. Arm order
alternates across cycles and workload order rotates. Timing excludes matrix
construction, prediction, profiling reads, hashing, and artifact writes; Metal
debug and shader validation are disabled. Every fit, including slow fits, is
retained. The recorded AC power setting was `powermode=0`.

The initial confirmation used eight paired cycles per workload:

| Workload | Off, median seconds | On, median seconds | Paired cycle speedup | Bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Depth 1, 256 bins | 0.09475 | 0.06667 | 1.423x | 1.383–1.459x |
| Depth 1, 32 bins | 0.06448 | 0.06525 | 0.998x | 0.965–1.029x |
| Depth 6, 256 bins, forced host builder | 0.53605 | 0.53848 | 1.059x | 0.877–1.273x |

The 256-bin stump improved in all eight paired cycles. The 32-bin result is
neutral within measurement uncertainty. The depth-six result does not establish
a speedup. Speedup is the geometric mean of paired process-median ratios; the
seconds columns are medians of all individual fits. The bootstrap resamples
whole paired cycles (10,000 draws, seed 42), not individual fits.

The initial six constraint/sampling/growth screens used only one paired cycle
each. Several showed large apparent slowdowns, including 0.673x for column
sampling and 0.707x for interaction constraints, so they were followed by
twelve fresh paired cycles for each screen and the depth-six fallback:

| Extended workload, 256 bins | Paired cycle speedup | Bootstrap 95% interval |
| --- | ---: | ---: |
| Depth 6, forced host builder | 1.106x | 0.920–1.299x |
| Monotone constraints | 1.089x | 0.964–1.239x |
| Interaction constraints | 1.152x | 1.027–1.315x |
| Tree/level/node column sampling | 1.012x | 0.880–1.159x |
| Uniform row sampling | 1.146x | 0.967–1.368x |
| Gradient-based row sampling | 0.985x | 0.918–1.034x |
| Loss-guided growth | 0.998x | 0.915–1.101x |

The extended paired-median estimates range from 0.985x to 1.152x. They do not
reproduce the initial large aggregate slowdowns, but individual cycles remain
variable and several intervals permit material regressions. These exploratory
follow-ups neither establish a general deep-tree speedup nor demonstrate
noninferiority within 5%.

Gradient-based sampling initially showed a tail risk: its all-fit mean increased
from 0.59630 to 0.65887 seconds (10.5%), and its individual-fit p95 increased
from 0.71338 to 1.06492 seconds (49.3%). The paired cycle-mean speedup is 0.911x
with bootstrap 95% interval 0.846–0.977x. Median-only summaries would conceal
this slowdown.

An independent repeat used the rebuilt default-on binary, a fixed twelve paired
cycles, and three fits per arm per cycle: 72 fits total. Off/on all-fit means
were 0.60325/0.57850 seconds and individual-fit p95s were 0.80445/0.78966 seconds.
The paired-median speedup was 1.048x (95% interval 1.023–1.073x); the paired-mean
speedup was 1.043x (0.975–1.107x). The earlier tail slowdown was not reproduced.
Both batches and all fits remain in the evidence. Taken together, they do not
establish a gradient-sampling speedup or global noninferiority within 5%.
No cause is assigned to the disagreement between batches.

The practical default decision weighs exactness, the consistent targeted stump
benefit, the expanded checks, and the independent follow-up; the opt-out is
retained. These measurements cover fixed synthetic workloads on one machine,
not performance across applications.

The final default-selection run added 108 fits comparing explicit zero,
explicit one, and an omitted parameter. It used two cycles for the three
primary workloads and one cycle for each of the six screens, with three fits
per arm per cycle. All outputs matched and dispatch profiles selected the
expected root path. This run verifies default selection and equivalence; its
small cycle counts do not establish additional performance claims.

All **864 fits across four batches** matched exactly within their nine
workloads: 180 initial opt-in fits, 504 extended opt-in fits, 72 independent
gradient-sampling fits, and 108 final default-selection fits. The cross-batch
audit checks input and effective parameter identity, UBJ models, original and
canonical JSON models, prediction bytes, and RNG state. The raw model export
remains authoritative; the canonical representation tags nonfinite metadata
explicitly. Predictions are required to be finite.

## Correctness and harness provenance

The opt-in focused suite passed **275 cases** with Metal debug and shader
validation enabled. It covers fallback roots, existing parallel roots, GPU tree
control, and sampling, including independent checks against the pinned CPU
reference. Exact model, prediction, leaf, and RNG comparisons exercise feature
masks, constraints, missing values, bin/layout boundaries, ties, invalid roots,
runtime switches, and unsupported paths.

The initial unadapted invocation recorded **111 failures and 164 passes**:
CPU-reference origin assertions compared a resolved package path with an
unresolved worktree symlink. Those logs and XML results are preserved. The
isolated validation harness adapts seven helper files only for package-path
resolution and local CPU-library identity. Two helpers replace the original
author's binary digest with the verified local pinned-build digest. All model,
prediction, data, serialization, tolerance, and byte-equality assertions remain
unchanged; the tracked source tests retain their original provenance guards.

The CPU reference source is clean at
`6d2dbe19a099aa046178be8cfc303bfec1a623a6`, with native SHA-256
`eabd28d40458db760345ec6b7e1856d2c26cb4098ebd13860d6dcbff468ab58c`.
The adaptations, per-file before/after hashes, source identity, and native
identity are recorded with each prepared harness. Test logs supply execution
results; a harness receipt marked `prepared_not_executed` is preparation
provenance alone.

The default-on rebuild changes the fallback lookup default to one and preserves
all eligibility guards. Its focused tests compare omitted, explicit-zero, and
explicit-one controls across the existing fallback coverage, and confirm that
GPU-driven trees retain their existing root path. The benchmark worker's
`--fallback-parallel-root default` omits the parameter and checks selection.

The final default-on full suite passed **3,863 cases with seven warnings in
453.33 seconds**, using four pytest workers, `OMP_NUM_THREADS=1`,
`MTL_DEBUG_LAYER=1`, and `MTL_SHADER_VALIDATION=1`. The production code did not
change after this run. The API audit passed: all 99 declared symbols are
exported, with no missing Python-referenced symbols. Both generated numeric
constant checks passed without rewriting files.

The AUC/AUCPR stage diagnostic also passed with Metal debug and shader
validation enabled: 4,097 rows, four groups, unique scores and tied scores,
one warmup and one measured pair for each metric/layout. Separate-stage and
combined-command runs matched the sorted buffer, per-class outputs, and host
metric bitwise. This checks the diagnostic's execution and comparison paths;
it is not an independent metric-reference test or a new performance result.

## Build and evidence identities

Both native builds use Release C++ and the existing Metal compiler settings.
The opt-in native SHA-256 is
`8371b20bd4f986aa3f46a598c1507d954c7cc8701a4eb005aad964fc8e602ea0`;
the default-on native SHA-256 is
`6abede6d2cf74b083063d1c68c5194cb233a0c39fcecf69cfd1124876fba201b`.
The compiled Metal library SHA-256 is
`e788f9c14daea2fcbfd4203bfe16155bc31e549d8706d37056ae38bb4f82f6a0`.
The production kernel source is unchanged by this fallback-root optimization.

Raw evidence is retained locally under
`benchmark-results/fallback-numeric-m4-20261025/`, including the
`opt-in-benchmark`, `opt-in-expanded`, `gradient-replication`, and
`default-on-benchmark` summaries, each worker receipt, saved models and
predictions, artifact audits, manifests, scripts as run, build logs, and
validation logs. `cross-batch-audit.json` records the final 864-fit comparison.
Source and native identity checks passed before and after all four timing
runs. Raw evidence is excluded from source control; this report records the
methodology, results, uncertainty, and build identities.
