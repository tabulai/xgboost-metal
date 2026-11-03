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

`benchmarks/deep_trees.py` compares pinned native builds on dense and
missing-valued data at depths 6, 8, and 10, using both tree drivers. Supply a
JSON arm file with two entries, for example:

```json
{
  "baseline": {
    "package_root": "baseline/python-package",
    "native_path": "baseline/lib/libxgboost.dylib",
    "native_sha256": "<SHA256 of baseline library>",
    "params": {}
  },
  "candidate": {
    "package_root": "candidate/python-package",
    "native_path": "candidate/lib/libxgboost.dylib",
    "native_sha256": "<SHA256 of candidate library>",
    "params": {}
  }
}
```

Paths resolve relative to the arm file. Use separate builds, or point both arms
at one build and set explicit optimization controls in `params`. An optional
`source_root` records tracked Git source provenance; untracked files, such as
an output directory inside the source tree, are not recorded. Then run:

```sh
python benchmarks/deep_trees.py --arms-file arms.json \
  --cycles 8 --repeats 3 --rounds 100 --warmup-rounds 20 \
  --output benchmark-results/deep-trees
```

The runner alternates arm order, rotates cases, and runs fresh processes
serially. Its report includes paired process medians, bootstrap intervals,
means, p95, and every individual fit. It checks saved UBJ and raw/canonical
JSON bytes, prediction bits, data, RNG state, and RMSE across arms. Profiling
reads and artifact writes occur outside training timers. Use `--plan-only`
to inspect the schedule without training. `--compare-baseline` can additionally
check an earlier run's `baseline-artifacts.json`; those exports retain absolute
paths, so preserve their original process directories. Outputs require a fresh
directory and include the scripts as run and a SHA256 manifest.

Optional cases `dense-d12-host` and `missing-d12-host` exercise deeper host
trees. Add `--common-params '{"metal_gpu_tree":1}'` to request normal automatic
driver selection instead of forcing the host, and set `expected_gpu_tree: 0`
in both arm definitions to verify fallback beyond the GPU depth limit.
The [M4 deep-tree measurements](deep-trees-m4.md) record the validation of
the default descendant and terminal-histogram optimizations.

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

Only an unsubmitted recipe survives an API call; GPU work is complete on return,
except for the opt-in asynchronous rounds described below.
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

The default `metal_fallback_parallel_root=1` path also lets the original
host-driven builder use the existing 256-thread numeric root splitter. Set it
to zero for the original root evaluator. Selection requires scalar numeric input
with at most 256 bins, untiled parallel split evaluation,
`metal_parallel_root=1`, `metal_fast_gain=1`, and `metal_fast_weight=0`.
The host still supplies the
original feature mask and monotone bounds, schedules subsequent nodes, and
assembles the model. Categorical, vector, tiled, and larger-bin roots retain
their previous root evaluation.

The latest-tree `fallback_parallel_root` field reports actual selection;
`parallel_root` is also one when this root kernel runs. `gpu_tree` remains zero
and `split_threads` continues to describe the nonroot splitter. This option
does not remove command-buffer waits or histogram allocations. Measure its
total-fit benefit separately for shallow and deep trees, using an explicit
zero as the same-binary control.

The `metal_fallback_parallel_splits` path reuses the
256-thread numeric split kernels for host-driven descendants. It fuses sibling
histogram subtraction with scoring while preserving the host's inherited sums,
feature masks, monotone bounds, expansion order, and histogram cache. Selection
requires numeric scalar input, at most 256 bins, untiled parallel scoring,
histogram subtraction, `metal_fused_subtraction=1`, `metal_split_threads=256`,
`metal_fast_gain=1`, and `metal_fast_weight=0`. Root controls remain independent.
Each threadgroup keeps one lane per bin, so the path wins only when most lanes
hold a cut. It defaults on when the mean number of cuts per numeric feature
exceeds 192 for dense input or 160 with missing values; below that, including
low-cardinality features at the default `max_bin`, the original 32-lane
evaluator was faster on an M4 Max (up to 1.8x at 64 bins). An explicit value
overrides this selection.

This path honors `metal_skip_empty_bins` for dense and missing-valued input.
With skipping enabled, `metal_compact_splits` defaults on for missing-valued
host input; dense host input can opt in explicitly. `metal_fast_stats` and
`metal_fused_score` retain their existing controls. Every derived histogram is
materialized, including masked features, so eviction, delayed expansion, and
loss-guided growth keep their original inputs. The latest-tree
`fallback_parallel_splits` field reports actual selection; `compact_splits`,
`fused_score`, `fast_stats`, and `split_skip_empty_bins` describe its selected
scorer. Set `metal_fallback_parallel_splits=0` for the original host descendant
path. Command-buffer waits and histogram allocations are unchanged.

`benchmarks/fallback_numeric.py` runs one pinned baseline, candidate, or CPU
worker and saves each repeat's predictions, original JSON/UBJ model, canonical
comparison JSON, timings, and dispatch profile. Its required package path and
native SHA-256 prevent accidental use of an installed library. Run separate
processes with the option off and on, alternate their order, and compare saved
outputs before interpreting timings. Use `--fallback-parallel-root default`
to omit the parameter and validate default selection. Use `OMP_NUM_THREADS=1`
with the default `--nthread 1` for Metal comparisons; the CPU performance
baseline can use `--nthread 0` with `OMP_NUM_THREADS` unset.
See `--help` for workload and output options.
The [M1 experiment](fallback-root-m1-20261025.md) and
[M4 validation](fallback-root-m4-20261025.md) record measured benefits,
exactness checks, and the limits of this root-only optimization.

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
value is present. This control leaves root candidate enumeration unchanged.
Dense CPU-driven trees that select `fallback_parallel_splits` also skip empty
bins; other dense CPU-driven trees and trees without fused subtraction retain
their original dense evaluator. Prefix sums and the first candidate in each direction are unchanged,
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
serial split search and CPU-driven trees without `fallback_parallel_splits`
retain separate subtraction. Setting `metal_fused_subtraction=0` also disables
that host path.

Set either option to zero for an independent control. Latest-tree tuning fields
`tree_control_fusion` and `fused_subtraction` report selection of these GPU paths,
including when pruning leaves later indirect dispatches empty. Both are zero
when the CPU tree driver is selected, even when `fallback_parallel_splits` fuses
host subtraction.

Several GPU-driver control kernels replace single-thread loops with parallel
forms; each has its own switch and same-named latest-tree tuning field.
`metal_parallel_select=1` selects each node's split with one SIMD group: lanes
scan features in order, then a SIMD maximum and the smallest matching feature
reproduce the original first-strictly-greater scan, including its signed-zero
ties and its handling of NaN gains. `metal_deferred_parent_statistics` leaves
the applied parent weight and Hessian of selected splits to one pass per
tree over expanded nodes. Only the CPU replay and asynchronous node emission
read these values, so the pass runs before emission. It
applies the unchanged binary64 helpers to the same recorded child sums, so
each level no longer waits on a serial software division. It defaults on from
two per-level selections (a dense root keeps its own ordered-sum statistics),
adds one dispatch per tree, and requires parallel selection.
`metal_parallel_tree_control=1` applies a frontier in one threadgroup. An
exclusive prefix over accepted slots reproduces the serial node numbering and
parent compaction, and the first error in slot order stops the same writes as
before. Node flags are cleared when children are created instead of all at
initialization; the CPU replay reads only created nodes. This path requires
`metal_tree_control_fusion=1`. `metal_parallel_partition_plan=1` writes each
parent's partition tasks with its own threadgroup instead of one thread.
None of these changes model arithmetic, and only deferral changes the number
of dispatches.

`metal_partition_single_pass=1` replaces the GPU driver's flag, serial tile
scan and scatter passes with one scatter. Each 256-row tile reserves a run at
the front (left rows) or the back (right rows) of its parent's segment with
one relaxed atomic per side; the final cursors are the children's row counts,
and an existing per-parent pass derives the same offsets as the stable scan.
Every child receives the same row set and counts, so histogram planning and
the smaller-sibling choice are unchanged. Row order inside a child now depends
on tile scheduling and can vary between runs. Its consumers do not depend on
it: histograms add exact modulo-2^64 integers, planners use counts, the next
partition tests each row independently, and the CPU never reads row
positions. This removes one dispatch per nonroot level.

With single-pass partitioning and parallel planning,
`metal_partition_bin_predicate=1` compares feature-major bins instead of
gathering row-major float values. The planner finds a cut equal to the
parent's threshold. Because quantization stores the clamped count of cuts at
most equal to a value, `value < cut[k]` holds exactly when the bin is at
most `k`, for every cut but the last. Last-cut and minimum thresholds and
unsorted cuts keep the float comparison. The path requires the matrix's own
float data, from which (or, for `QuantileDMatrix`, into which) its bins were
quantized. Inputs recoded after a feature-type change keep the float
comparison. The CPU-driven builder keeps the stable partition. Set these
options to zero for same-binary controls.

The default `metal_terminal_histogram_elision=1` path omits derived-sibling
histogram stores and their device barrier at the configured final GPU-tree depth.
It subtracts the parent and built-sibling bins directly into the unchanged
integer prefix and split-scoring path. That level only creates terminal leaves;
earlier levels retain complete histogram materialization. This requires fused
subtraction and a 256-thread parallel or compact splitter. Other paths are
unchanged. `terminal_histogram_elision` reports selection, including an empty
terminal dispatch after pruning. Set it to zero to materialize those final
histograms as a control. Host-driven trees always materialize their histograms.

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

These GPU-loop variants require scalar numeric input, fused subtraction and
fast gain, with fast weight disabled. The separately controlled fallback root
splitter described above has its own narrower eligibility checks. Existing
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

The default `metal_split_screening=1` path runs the software scorer only for
candidates that can still win. Each lane encloses the exact float32 gain of its
forward and backward candidates using native float32 arithmetic. Every native
result steps one or more ulps outward, which holds under round-to-nearest-even
or round-toward-zero. The 24-bit truncation of each integer sum brackets its
binary64 value. The child gain then lies within `G^2 / (H + lambda)` times
`1 +- 2^-22`: the error from rounding the weight cancels at the optimum.
Magnitude guards keep every operand normal. A candidate is skipped only when
its upper bound is below the best lower bound certified in its node, so its
exact gain is strictly smaller and it can neither win nor tie. Uncertain cases
always run the exact scorer, for example a Hessian next to `min_child_weight`
or an out-of-range statistic. Certainly invalid candidates have no effect on
the result. Survivors are packed into the first lanes, so exact scoring
usually occupies one SIMD group. Each lane scores at most one survivor, and
the reduction keeps the original gain, then candidate order, tie rule.
Selection requires zero L1 regularization, no `max_delta_step` clamp, finite
`reg_lambda` and `min_child_weight`, and no monotone constraints. Kernels also
recheck node bounds and each feature's constraint. Screened twins replace the
skip-empty sparse and compact nonroot scorers in both tree drivers. Without the
node cutoff below, each per-feature `Split` is byte-identical to its
unscreened twin.

`metal_split_screening_root` applies the same screening to the parallel root
scorer of either driver. It defaults to one for missing-containing input,
where each lane otherwise scores forward and backward candidates, and to zero
for dense input. A dense root scores one candidate per lane in a few
latency-bound threadgroups, and screening lengthened that dispatch in local
measurements.

`metal_split_node_cutoff=1` shares certified lower bounds between the feature
groups of one node. Each GPU-driver level has its own cells, updated with a
32-bit atomic maximum of an order-preserving key and cleared once per tree.
Any value read is a valid bound, including a stale one. Cutoffs avoid the
subnormal and zero band, because comparisons may flush subnormal gains to
zero. Every candidate tied with the node maximum survives, so features that
can win write byte-identical records. Other features may skip candidates, or
write a losing `-inf` record when nothing survives. Split selection
(`select_splits`, or `select_splits_simd` with `metal_parallel_select`), the
only consumer, keeps the first strictly greater gain, so these records cannot
be selected. The exact parent statistics are computed only when a feature group
has survivors or has seen no certified bound. Host-driven trees use only the
per-feature cutoff.

With screening, `metal_split_inherit_parent=1` takes a GPU-driver child's exact
weight and cover from its parent's selected split. The GPU apply step has
already written that split, and its scorer computed both values from the
child's integer sums. The child then recomputes only its gain and gradient
total, with the same functions, avoiding the parent division and a barrier.
The GPU driver's node bounds are infinite, so these weights are unclamped.
The root and host-driven trees compute parent statistics as before.
`split_inherit_parent` reports selection; set it to zero for a control.

Latest-tree tuning fields `split_screening`, `root_screening` and
`split_node_cutoff` report selection. `metal_split_screening_stats=1` adds
cumulative counters to the profile's `split_screening` object; asynchronous
rounds (`metal_async_rounds`) do not collect them. They record evaluated candidates,
certified and uncertain candidates, exactly scored survivors, feature groups,
and groups with survivors. Set `metal_split_screening=0` for a same-binary
control. Models and predictions are unchanged. `tests/split_screening_probe.metal`
and `tests/split_screening_bounds.cpp` (CMake option `METALGB_BUILD_TESTS`)
check enclosures against every exact scorer family on adversarial integer
sums. They also compare screened and unscreened kernel records byte for byte.

Four GPU-tree histogram dataflow options change only how bin codes, tasks and
sums move. Every row still adds the same exact integer gradient and Hessian to
the same bin, so histograms, models and predictions are unchanged. They apply
to dense input, or to missing-valued input with `metal_histogram_missing=1`;
`metal_histogram_missing=0` with missing values keeps the original kernels.
Each option has a same-binary zero control, and latest-tree fields
`histogram_packed_bins`, `histogram_schedule`, `histogram_single_owner` and
`histogram_unit_hessian` report selection. Host-driven trees keep the
original histogram kernels.

`metal_histogram_packed_bins` (default 16) reads a derived uint8 copy of the
bin codes. Four features share one 32-bit word. With 16, each row stores 16
features in one 16-byte chunk; with 4, each word group is a separate array.
Narrower matrices use only the words they need. A grouped task then
loads one 4-byte word per row instead of four scattered 2-byte column gathers.
At deep levels, a node's rows are a sparse subset of all rows, so this greatly
reduces the memory sectors fetched. The copy is built once per quantization
and chunk width, and the canonical uint16 bins remain authoritative. It needs
one, two or four features per group and every code within 8 bits: at most
256 bins for dense input, or 255 with code 255 marking missing values. Other
inputs, including 256-bin data with missing values, keep the uint16 bins. The
copy adds one byte per row and padded feature. With packing enabled, dense
numeric scalar inputs with at least 16384 rows also default to four features
per histogram group, like the existing large-data grouping. This default also
applies to host-driven depthwise trees, where it measured neutral to slightly
faster, but not to lossguide trees, whose small per-leaf histograms ran slower
grouped; `metal_features_per_group` remains an explicit override.

`metal_histogram_schedule` replaces `plan_histograms`' per-group task records
with one record per row tile. Each threadgroup derives its feature group from
its index, so planning writes up to 64 times fewer records. Value 1 orders
threadgroups group-major, keeping one feature column hot across a level's
nodes. Value 2 orders them tile-major, running the groups that share a tile's
row IDs, gradients and 16-byte packed chunk back to back. The default is 2
with 16-feature chunks and 1 otherwise. Zero selects the original records.

`metal_histogram_single_owner=1` lets the only threadgroup building a node's
feature group publish it with plain 16-byte stores. Planning tags nodes that
fit in one row tile, and `fill_histogram_built` skips their slices. The store
writes every bin, including zero bins, and 0 + total is the same modulo-2^64
sum. A missing bin is cleared before its SIMD-group sums are added. Zero-row
built nodes still receive the fill. Shared slices no longer add zero pairs.

`metal_histogram_unit_hessian=1` targets unweighted squared error. Every row
then has the same quantized Hessian, so a bin's Hessian sum is its row count
times that constant, modulo 2^64. Threadgroups add only gradients plus a 32-bit
count, then multiply at flush. A GPU check per tree compares every row's
quantized Hessian with the first row's; on any difference, the tree uses the
original sums. The check, its flag clear on the GPU and the node-cutoff clear
share the tree's initialization encoder. Custom objectives that return varying Hessians therefore fall
back automatically. `histogram_unit_hessian` reports whether the counts were
used after this check; asynchronous rounds, which never carry caller-supplied
gradients, report the selection.

With per-dispatch timestamps (`METALGB_DISPATCH_PROFILE`) on an M4 Max, the
800k-row by 128-feature dense depth-six case spent about 11 ms per tree in
histogram kernels with all four options set to zero, and about 3.6 ms with
the defaults. Before, each level from depth two onward cost 1.0-3.4 ms, more
than the first two levels together, because scattered 2-byte gathers
amplified memory traffic. With packed bins and tile-major records, the first
two levels took about 1.5 ms and each deeper level 0.5-0.8 ms.

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

## Host-side round streamlining

Each boosting round also repeats host work outside the GPU kernels. The
following exact trims are independently controlled; all default to one and
change neither kernel arithmetic, dispatch order nor model bytes.

- `metal_round_buffers` keeps the gradient pairs, partial sums, scales and
  quantized gradients alive across rounds instead of allocating fresh shared
  buffers. Each is fully overwritten by its producing kernel before any read,
  and the host never touches them. Custom gradients, linear models and the
  absolute, quantile, expectile, distribution, survival and ranking objectives
  keep fresh gradient-pair storage.
  Training-margin outputs come from a pool: a buffer is reused only when the
  pool holds its sole host reference, so margins returned to a caller,
  retained by a cache or recipe, or read by a round still in flight are never
  rewritten. Reset releases the pool.
- `metal_gpu_row_init` writes the initial row identities of a GPU-driven tree
  with one dispatch in the driver's initialization encoder, instead of a host
  loop over every row. Host-driven trees keep the host loop.
- `metal_tree_constant_inputs` skips rewriting the retained monotone, feature
  mask and bound slots when their bytes equal the slot's previous host write
  and no other request has used the slot since. These inputs are read-only on
  the GPU; changed bytes are copied as before.
- `metal_replay_path_elision` omits interaction-path bookkeeping (a set copy
  per expansion) from the host replay of GPU-driven trees, which exclude
  interaction constraints. It mainly helps depth-ten trees.
- `metal_commit_early` commits the prediction/gradient/quantization prologue
  before the host builds and encodes the tree, continuing in a second command
  buffer on the same queue. One wait covers both; `gradient_tree` reports the
  sum of their GPU intervals and the command count is unchanged. Explicitly
  coalesced prologues (`metal_coalesce_encoders=1`) and asynchronous rounds
  keep one command buffer.
- `metal_parameter_snapshot` skips re-parsing the purely parameter-dependent
  training checks while the parameters equal the last validated set. Any
  change, including an invalid value, is validated again.

Latest-tree fields `round_buffers`, `gpu_row_init`, `tree_constant_inputs`,
`replay_path_elision` and `commit_early` report selection. On the local M4 Max,
per-round paired measurements (toggling one switch every round) attributed the
largest savings to round buffers on shallow trees and to path elision on
depth-ten trees; row initialization and constant inputs mainly remove host
stores that asynchronous rounds would otherwise have to avoid.

## Asynchronous boosting rounds

`metal_async_rounds=K` (default zero) lets up to K eligible rounds remain in
flight when `Booster.update` returns, in the range one to four. Each round's
command ends with one extra dispatch that emits the host replay's Node table
for the finished tree. Its leaf values use an exact software binary32
multiplication with round-to-nearest-even and gradual underflow, because Metal
arithmetic flushes subnormals. The next round's deferred margin update runs the
unchanged `predict` kernel over that single-tree table, with the same tree range
and floating-point additions as the forest recipe. No round therefore waits
for the host model, and host setup, Python and encoding overlap GPU work.

Rounds retire in order. Retirement waits for the command if needed, replays
the tree from the driver history with the original host assignments, compares
the emitted Node bytes with the replay and only then publishes the tree. A
Metal error, invalid driver state or any byte difference rolls the model back
to the last retired round, discards later rounds and training-margin caches,
and raises the error from that call. Errors are therefore reported up to K
calls late. The CPU replay remains the only source of the model.

Eligibility is the deferred-prediction eligibility above plus an eligible
GPU-driven tree. Other updates retire committed rounds first; a tree that turns
out to need the host driver retires them before its first host workspace
store. Prediction, evaluation with data, model/config save and load, dumps,
slicing, feature scores, `XGBoosterBoostedRounds`, `XGBoosterMetalProfile`,
reset, model-changing configuration and booster destruction also retire every
round. Attribute and feature-metadata calls, parameter changes and evaluation
without data do not, so an ordinary `xgb.train` loop waits only in its final
reset. Evaluation sets, early stopping or per-round profile reads drain every
round and remove the benefit.

While rounds are in flight, retained host-written tree inputs are never
overwritten: unchanged constant inputs are reused, changed inputs and host row
identities use new allocations, and driver readback and emitted tables come
from a pool that keeps a round's buffers until it retires. GPU-only scratch is
shared by consecutive rounds and ordered by the single command queue. The
latest-tree field `async_rounds` reports K for an asynchronous round, and
retirement order defines the remaining latest-tree fields.

Leaf scaling can be checked exhaustively with the optional extension
`XGBMetalValidateLeafScaling(eta, begin, count, &mismatches, &first)`, which
compares the GPU multiplication with the host product for consecutive binary32
weight patterns. All 2^32 weights matched for every learning rate tested. For
tests only, `metal_async_fault` (1: emitted metadata, 2: driver state) with
`metal_async_fault_round` corrupts one round after emission to exercise
rollback. On the local M4 Max, same-process alternating `xgb.train` runs with
K=2 removed nearly all of the host and synchronization gap between rounds;
shallow trees benefited most. Report asynchronous timings together with the
training API and evaluation cadence used.

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
