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

## GPU memory bandwidth

```sh
cmake -S . -B build -DMETALGB_BUILD_BENCHMARKS=ON
cmake --build build
./build/metal_bench 256 11
```

This checks a GPU copy workload and reports observed bandwidth. It is a
microbenchmark, so use the training and prediction measurements when evaluating
application performance.
