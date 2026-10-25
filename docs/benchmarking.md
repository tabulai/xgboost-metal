# Benchmarking

The repository includes command-line benchmarks for training, prediction,
ingestion, and GPU memory bandwidth. Run them on the deployment hardware with
the same input data and model parameters used by the application.

## Training and prediction

After a [development build](../CONTRIBUTING.md), run from the repository root:

```sh
PYTHONPATH=python-package python benchmarks/run.py \
  --rows 100000 --features 32 --rounds 100 --repeats 5 \
  --output benchmark-results/training.json

PYTHONPATH=python-package python benchmarks/matrix_prediction.py \
  --rows 100000 --query-rows 100000 --features 32 --rounds 100 \
  --output benchmark-results/prediction.json
```

The training report records matrix construction, training, and prediction times
alongside the model quality metric. The matrix prediction benchmark distinguishes
matrix construction, uncached prediction, and cached prediction. Use `--help`
on each script for available dataset, objective, and model settings.

Additional scripts cover input ingestion, prediction, refresh
and pruning, and ranking caches. Raw reports belong in `benchmark-results/`,
which is excluded from source control.

## Reproducible comparisons

Use identical train/test splits, feature types, preprocessing, missing-value
rules, weights, bin counts, tree depth, boosting rounds, and seeds. Compare model
quality alongside time; a faster run with different quality is a different
tradeoff.

Report preparation time, training time, and prediction time separately, together
with end-to-end time. Include batch size and whether prediction reuses a cached
matrix or model result. Run warmups before timing, use repeated measurements,
and report the median and range.

For a multicore CPU baseline, use an OpenMP-enabled XGBoost build and report
the available cores, hardware threads, and effective thread setting. Include
the default all-thread configuration as the primary baseline. If thread tuning
is performed, publish it as an additional result with its selection procedure.

Record the dataset version or hash, source revision, package version, build
configuration, hardware, operating system, and power mode. Run timing without
Metal API or shader validation enabled and avoid competing workloads.

Use ordinary `xgb.train` calls for application training comparisons, including
their final reset/copy. Time matrix construction separately. Keep profile reads
outside the timed call; direct `Booster.update` loops and periodic profiling are
useful diagnostics, but have different timing boundaries and submission cadence.
Alternate backend order across independent processes and retain every repeat,
including slow runs. A short warmup compiles pipelines; it does not guarantee a
fixed GPU frequency.

Check monitoring overhead against unmonitored runs. On the local M4 Max,
`mactop` v2.0.0's repeated display-inventory queries materially accelerated the
workload being observed. An IOReport-only sampler had a much smaller effect.
Align samples with timestamps captured inside each fit. Device-wide power-state
residency is not shader utilization, and normal thermal status does not imply
constant clocks. Do not use background work to raise clocks in reported timings.

## Tree scheduling and scratch storage

Eligible built-in objectives encode gradients, gradient quantization, and the
first tree's root split into one serial command buffer. With the GPU tree
driver below, that command also contains the rest of the first tree. This removes one GPU
submission and completion wait per round. Custom gradients, row sampling,
refresh/update paths, linear models, Cox, and ranking retain their existing
synchronization boundaries. Set `metal_gradient_root_fusion=0` for a control.
This changes neither kernel arithmetic nor the order of prediction additions.

The default `metal_defer_prediction=1` path also combines the previous round's
training-margin update with the next gradient/tree command. It retains the
original prediction inputs after CPU tree assembly, then runs the unchanged
`predict` kernel before the next gradients, with the same tree range and
floating-point additions. Set `metal_defer_prediction=0` for an eager-cache
control. Runtime buffer arguments are bound in batches; encoder coalescing
defaults on for the tree portion of an eligible, fully dense GPU tree with
maximum depth two through six. Earlier prediction/gradient encoders retain
their boundaries. Other depths, missing-containing inputs and CPU-tree
fallbacks retain separate encoders. Explicit `metal_coalesce_encoders=0` or `1`
overrides this selection; one also coalesces the earlier prediction/gradients.

New deferred updates require an ordinary `DMatrix`, one scalar output, one tree
per round, gradient/root fusion, and an eligible GPU-driven tree. Squared error,
logistic/logitraw, Poisson, squared log error, gamma, Tweedie, pseudo-Huber and
hinge objectives are supported. Custom gradients, sampling, base margins,
dropout, multiclass/vector/parallel trees, refresh/update, linear models, and
`QuantileDMatrix` retain eager updates. Quantile matrices are excluded because
feature-type changes can rewrite their decoded storage in place. Existing
GPU-tree eligibility and scratch-budget guards still apply. A previously
deferred update may be consumed by a subsequent CPU-driven tree's fused root,
but that tree does not create a new deferred update.

Only an unsubmitted recipe survives an API call; GPU work is complete on return.
Public prediction/evaluation and incompatible updates flush pending recipes.
Forest upload flushes them before changing shared input buffers. Reset, model
replacement, expired matrices and cache eviction may discard unused recipes;
the CPU model is already complete. Cached margins and their round version are
published only after the consuming command and CPU tree assembly succeed.

The profile fields `prediction_cache_deferred` and `prediction_cache_consumed`
describe the last update's path, not currently pending work. `gradient_tree`
(or `gradient_root` for a CPU-driven tree) includes any consumed prediction;
separate flushes remain under `prediction_cache`. Encoder counts include the
prediction when present. An unsubmitted recipe does not increment command count.

Normal `xgb.train` timing includes its final reset, which discards the last
unused training-cache recipe. Later predictions use the completed model.
Direct update loops retain that recipe until a consumer or reset. Evaluation
after every round can remove the submission saving, so report the training API
and evaluation cadence. Fewer submissions do not imply fixed GPU frequency or
elimination of scheduling-related slow runs; retain all observations and report
both process medians and individual-fit tails.

The existing tree workspace also retains four setup buffers: monotonic
constraints, row IDs, row scratch, and partition ranks. Each tree reinitializes
its constraints and row IDs; partition kernels initialize scratch before use.
These slots remain distinct from per-level scratch storage, and cached
histograms retain their existing ownership. Set `metal_tree_fixed_workspace=0`
for a control. Existing workspace/persistence switches and booster reset still
release or disable retained storage. The additional retained capacity is
approximately three row-index arrays plus one constraint array at peak sizes.

The default `metal_gpu_tree=1` path moves depthwise expansion decisions and work
planning onto the GPU for eligible scalar-leaf numeric trees. It pre-encodes the
bounded depth loop, then GPU kernels compact accepted parents in node-ID order,
retain inherited integer sums, and generate partition/histogram/split dispatch
sizes. Split and partition arithmetic remain unchanged. There is one completion wait
per tree, followed by model assembly through the original CPU assignments;
keeping leaf scaling on the CPU preserves its handling of subnormal values.
Empty frontiers issue no histogram or split work, and histogram clearing touches
only the active prefix. Set `metal_gpu_tree=0` to use the previous driver.

This path requires depth 2–10, at most 256 histogram bins, no leaf-count limit,
no monotone or interaction constraints, and no level/node column sampling.
Tree-level column sampling retains the existing host RNG and feature mask.
Categorical/vector trees, loss-guided growth, compact sampled gradients,
disabled partition/subtraction modes, and insufficient histogram-cache budgets
retain the previous path. The GPU path uses two histogram banks and separate
control buffers; a 256 MiB estimate of requested scratch storage bounds its
selection. Buffer allocation minimums and previously retained peak capacities
can make actual allocation larger. Workspace controls and reset still apply.
At depth ten, every nonterminal expansion fits the reference driver's
256-parent batch. The final level may expand 512 parents, but only creates
terminal children; applying them in increasing node order preserves the two
reference batches' model layout. Deeper trees still use the original driver.

When quantized input contains missing values, `metal_histogram_missing=1`
accumulates their integer gradient/Hessian sums privately and reduces them
within each SIMD group. Each group publishes one sum instead of every missing
row contending on the same histogram bin. The new kernels also omit global
atomic updates when a reduced missing sum or merged present-bin sum has both
gradient and Hessian equal to zero. This preserves the existing modulo-64-bit
sums, including cancellation, grouped features, and histogram subtraction.
Fully dense input retains the original kernels. Set
`metal_histogram_missing=0` for a control; the latest-tree tuning field
`histogram_missing` reports actual use. This optimization still scans the
existing dense bin storage, including for CSR input.

`metal_histogram_present=1` skips missing rows before reading their gradients
on nonroot levels of eligible scalar GPU-driven trees with missing input and
`metal_histogram_missing=1`.
Those nodes inherit exact integer totals from their parents, so split search
reconstructs missing statistics as the inherited total minus the present-bin
sum. Newly built missing bins remain zero; subtraction can carry placeholder
missing-bin values, which this nonroot path never uses for scoring. Present
bins retain the original layout and exact integer sums. Root histograms and
ineligible drivers keep their complete missing statistics. This removes
missing-row gradient loads and reductions but still traverses dense bin
storage. Set the option to zero for a control; `histogram_present` reports the
latest tree's selected nonroot histogram path, even if pruning leaves it empty.

`metal_skip_empty_bins=1` skips redundant scalar SIMD
split candidates whose histogram gradient and Hessian sums are both zero.
Integer prefix sums remain unchanged. The first candidate in each scan
direction is always retained, so repeated cumulative sums keep their original
threshold and tie order. Categorical branches, vector leaves, tiled scalar
search, and serial split search retain their existing evaluation. Set
`metal_skip_empty_bins=0` for a control;
`split_skip_empty_bins` reports whether the new scalar kernel was selected.

Dense input also uses this optimization on nonroot levels of the fused GPU
tree path. Deep nodes can have many empty histogram bins even when every input
value is present. This control leaves root candidate enumeration unchanged;
dense CPU-driven trees and trees without fused subtraction retain their
original dense evaluator. Prefix sums and the first candidate in each direction are unchanged,
so skipping a zero gradient/Hessian pair preserves threshold and tie order.

The GPU driver also supports two independently controlled kernel fusions.
`metal_tree_control_fusion=1` combines initialization with root planning and
each expansion with planning for the next level. These operations run on one
GPU thread; even an empty or failed expansion clears the next indirect
arguments. It removes one control dispatch per configured depth without
changing model arithmetic or the number of command-buffer submissions.

`metal_fused_subtraction=1` combines sibling-histogram subtraction with numeric
SIMD split evaluation on nonroot GPU-driver levels. Each feature/node group
materializes its own derived histogram slice, synchronizes that group's device
memory, then performs the existing split search. Built-sibling slices are
read-only. The complete derived histogram is retained for later levels. This
removes one dispatch and an inter-dispatch dependency per nonroot level;
serial split search and CPU-driven trees retain separate subtraction.

Set either option to zero for an independent control. Latest-tree tuning fields
`tree_control_fusion` and `fused_subtraction` report selection of these GPU paths,
including when pruning leaves later indirect dispatches empty. Both are zero
when the original CPU tree driver is selected.

The optional `metal_fast_weight=1` shortcut changes scalar split-weight division in the fused
nonroot SIMD kernels. Native float division supplies only a quotient estimate.
Exact 128-bit integer arithmetic certifies the quotient and remainder, with up
to four corrections; an uncertified estimate falls back to the existing
restoring division. The binary64-to-float32 double-rounding rule, subnormal
handling, clamps, gain arithmetic and candidate order remain unchanged.
`metal_fast_weight=0` selects the original division and remains the default:
the certified-estimate implementation did not improve the measured workloads.
The latest-tree `fast_weight` field is one only when this path is selected;
disabling fused subtraction or selecting the CPU tree driver also disables it.

`metal_fast_gain=1` specializes the squared-weight step of scalar gain scoring.
For weights with magnitude between 2^-63 and 2^63, a 24-by-24-bit integer
product and explicit ties-to-even rounding replace general binary64 multiply
and conversion. Outside that range, the original software path remains in
use. The resulting squared-weight bits and all remaining gain arithmetic are
identical. This avoids depending on hardware floating-point rounding mode.
Set `metal_fast_gain=0` for a same-binary control; `fast_gain` reports selection
of the specialized fused SIMD kernel, independently of `fast_weight`.

`metal_fast_stats=1` specializes signed integer histogram statistics to
binary64. It forms the normalized significand directly, explicitly rounds the
discarded 1–11 bits to nearest/even, then applies the existing power-of-two
scaling. The original conversion rounding point is preserved, including at
the signed integer limits. Set it to zero to select the original conversion.

The eligible GPU loop can score a feature with eight SIMD groups (256 threads),
using exact integer carries between groups and the original global candidate
order to break ties. `metal_split_threads` selects 32 or 256 threads for the
noncompact nonroot splitter; the default is 256 for both dense and
missing-containing input. `metal_parallel_root=1` selects the 256-thread root
splitter, while retaining the existing dense-root final selection and model
statistics. Parent scoring is computed once per threadgroup and shared.

`metal_compact_splits` defaults to one for missing-containing input or maximum
depth at least eight, and zero otherwise. It packs nonzero histogram candidates
into adjacent lanes, retaining the first and last bins, original integer
prefixes, original bin IDs, and both scan directions' tie order. This path
always uses 256 threads, including when `metal_split_threads=32` is requested;
set `metal_compact_splits=0` to measure the noncompact splitter. Compaction
requires empty-bin skipping, so `metal_skip_empty_bins=0` disables it.

Fast statistics, the wider splitters, and compaction require the scalar numeric
GPU loop, fused subtraction and fast gain, with fast weight disabled. Existing
unsupported configurations retain their previous scoring paths. Latest-tree
fields `fast_stats`, `parallel_root`, `compact_splits`, and `split_threads`
describe selected paths and widths; they are not candidate or arithmetic hit
counts. Pruned later levels can have a selected path with no actual work.

For small numeric matrices containing missing values, at most 256 bins now
defaults to two features per histogram group. Existing large-data grouping
remains four. `metal_features_per_group` remains an explicit override. The
optional present-only histogram experiment stays disabled by default because
its measured timing benefit was inconsistent.

`metal_histogram_compact_scratch` defaults to one for missing-containing input
and zero for dense input. With at most 256 bins and one or two features per
histogram group, it reduces threadgroup scratch storage while retaining exact
integer accumulation. Present-only histograms and other group sizes keep their
existing paths. An explicit zero selects the original scratch layout.

`metal_histogram_clear_built=1` clears only the sibling histogram that will be
built from rows; fused subtraction overwrites the other sibling completely.
The root still receives a full clear. This requires the scalar GPU tree loop
with fused subtraction and adds no dispatch or synchronization point. Set it
to zero to clear both siblings as before.

`metal_fused_score=1` shares the rounded binary64 curvature between scalar
weight and gain evaluation. With positive-zero L1 regularization, it also
removes redundant operations when their inputs are finite, preserving signed
zero and every original rounding point. Other inputs retain the original
arithmetic. This path requires compact splits and fast statistics, so its
default use is limited to eligible missing-containing or depth-eight-and-deeper
training. Set it to zero for the original scoring arithmetic.

Latest-tree fields `histogram_compact_scratch`, `histogram_clear_built`, and
`fused_score` report selected paths, not counts of executed operations. Setting
all three parameters to zero provides a same-binary control for these changes.

The `XGBoosterMetalProfile` extension reports the combined gradient/root GPU
interval as `gradient_root`; add it to the remaining tree intervals when
comparing GPU work. A complete GPU-driven tree is reported as `gpu_tree`, or
`gradient_tree` when its command also contains gradients. `tree_encoders`
includes all encoders in that command. Latest-tree tuning fields `gpu_tree`,
`gpu_tree_levels`, and `gpu_tree_scratch_bytes` report actual path use, evaluated
depth levels, and the scratch-size estimate. The tuning fields `tree_fixed_workspace` and
`tree_fixed_buffer_allocations` describe the latest tree, not cumulative work.
`tree_buffer_allocations` includes the four setup slots when this reuse is
enabled. Ordinary same-shape training allocates these slots once and then
reuses them without changing model or prediction bits.

## Forest upload diagnostics

The optional `XGBoosterMetalProfile` extension includes a `forest_uploads`
object. Its cumulative counters survive booster reset, like the existing GPU
timings and command count:

- `full_uploads` and `append_uploads`: complete-prefix rebuilds and suffix uploads.
- `trees_uploaded` and `bytes_uploaded`: trees and bytes packed from the CPU model.
- `buffer_allocations` and `bytes_relocated`: geometric capacity growth and the
  initialized prefixes copied when buffers grow.

These counters cover tree metadata, nodes, category bitmaps, and vector leaf
weights. They exclude base margins and dropout weights; dropout can change
weights on earlier trees, so those weights are refreshed separately. Empty
forests still allocate valid Metal argument buffers.

Ordinary training appends only newly published trees into reusable buffers.
Model loading, reset, configuration changes, and update-mode transitions
invalidate the uploaded prefix. Training prediction-cache updates start at the
newest round's tree range without scanning earlier tree metadata. Neither
optimization changes split arithmetic or the order of prediction additions.

For this optimization, use direct `Booster.update` calls when measuring retained
training state: the `xgb.train` convenience function resets internal caches at
the end. Report fresh-query prediction separately from cached training margins.

## GPU memory bandwidth

```sh
cmake -S . -B build -DMETALGB_BUILD_BENCHMARKS=ON
cmake --build build
./build/metal_bench 256 11
```

This checks a GPU copy workload and reports observed bandwidth. It is a
microbenchmark, so use the training and prediction measurements when evaluating
application performance.
