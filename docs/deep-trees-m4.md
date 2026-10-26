# Deep-tree M4 measurements

The fixed primary, secondary and depth-twelve runs completed **720 fits across 240 processes**, preserving exact outputs. Depth-eight/ten GPU trees and most host cases improved against release. Depth-six GPU and dense depth-ten host results are inconclusive. The fixed secondary reproduced the dense depth-eight GPU benefit but **did not establish a dense depth-eight host benefit within the same binary**. All **3,999 tests passed** with Metal API and shader validation enabled. The validated descendant and terminal-histogram optimizations are enabled by default.

Measured October 26, 2026 (EDT) on Apple M4 Max, 128 GB unified memory, macOS 26.6.2, Python 3.13.3 and NumPy 2.5.3. Release baseline: `b85e0080bb4d9256ced8491512d020e4b2346e8a`, native SHA-256 `6abede6d2cf74b083063d1c68c5194cb233a0c39fcecf69cfd1124876fba201b`. Frozen default candidate: native SHA-256 `02573046cb66bdca93da34d0342930949b7da7fe342765a7fc695ec416ea3db3`; source core commit `e9423184f29d6a8acae6d9d3313c0bf1047b27e6`. Separate source inventories/diffs are recorded in `baseline-provenance.json` and `default-provenance.json`.

Build configuration was Release with a macOS 14.0 deployment target and Metal 3.1 kernels compiled with `-O3 -fno-fast-math`.

## Method

The candidate omits unused derived-histogram stores at the configured final evaluated GPU-tree depth and reuses wider exact scoring for host descendants. A selected terminal-elision flag does not prove nonempty work when pruning removes that frontier. Host compaction defaults on only with missing values; **dense host compaction remains off** after mixed screening results. Confirmation omits the new parameters to exercise their defaults.

Data comprise 65,536 rows × 32 features, seeded integers in `[-128,128)/16`, either dense or with 15% missing values. Labels are `x0 − 0.5*x1 + 0.125*x2*x3`, computed before missingness. Data seed is 271828; training uses squared error, 256 bins, seed 271, base score zero and learning rate 0.1. Cases cross depths 6/8/10 with dense/missing data and GPU/forced-host drivers. Host means CPU scheduling of Metal kernels. Training and OpenMP thread counts are pinned to one.

Each fresh arm/case process warms up for 20 rounds and performs three fresh 100-round fits. Timings cover ordinary `xgb.train`, including final reset, excluding matrix preparation, prediction, profile reads and artifact writes. Timing runs clear Metal validation/debug/trace controls. Eight paired cycles alternate arms and rotate cases; all fits and tails remain. The separate, fixed dense-depth-eight off/default comparison uses the same schedule. Neither run was extended for significance.

Primary speedup is the geometric mean of paired process-median ratios. Secondary sensitivity uses process means. Both bootstrap intervals resample whole paired cycles: 10,000 draws, seed 42. Pooled mean/p95/max use every fit. These intervals concern the fixed synthetic workloads, not applications generally.

Saved UBJ/raw-JSON/canonical-JSON models, prediction bytes and dtypes/shapes, data hashes, RNG state and RMSE must match exactly. Profiles record actual selected paths. The runner verified unchanged Python-package/native fingerprints. Separate postflight checks verified 72 compiled-source files and 75 test files against their recorded snapshots, along with both native libraries and the clean pinned CPU reference. Numerical parity supplements the regression tests; it is not an independent CPU oracle.

## Primary release/default confirmation

All **576 planned fits completed: zero failed or unattempted**. Every saved-output comparison passed, including the external baseline. Independent recalculation reproduced all fit distributions and paired intervals. Time triplets below are pooled **median / mean / p95 seconds**; each arm/case includes 24 fits. Speedup above one favors the default.

| Case | Release median / mean / p95 | Default median / mean / p95 | Paired median speedup [95%] | Paired mean speedup [95%] |
|---|---|---|---|---|
| Dense d6 GPU | 0.1798 / 0.1833 / 0.2017 | 0.1788 / 0.1803 / 0.1891 | 1.001× [0.980, 1.027] | 1.015× [0.991, 1.049] |
| Dense d6 host | 0.4358 / 0.4740 / 0.7092 | 0.3889 / 0.3932 / 0.5311 | 1.171× [1.085, 1.282] | 1.203× [1.108, 1.319] |
| Missing d6 GPU | 0.2051 / 0.2044 / 0.2090 | 0.2034 / 0.2034 / 0.2118 | 1.008× [0.996, 1.018] | 1.005× [0.996, 1.014] |
| Missing d6 host | 0.5779 / 0.6172 / 0.9520 | 0.4197 / 0.4559 / 0.6598 | 1.335× [1.127, 1.587] | 1.340× [1.168, 1.545] |
| Dense d8 GPU | 0.3452 / 0.3466 / 0.3673 | 0.3352 / 0.3326 / 0.3443 | 1.026× [1.015, 1.040] | 1.041× [1.023, 1.068] |
| Dense d8 host | 1.3696 / 1.3331 / 1.6236 | 1.1754 / 1.1898 / 1.4324 | 1.160× [1.085, 1.246] | 1.121× [1.067, 1.184] |
| Missing d8 GPU | 0.4369 / 0.4493 / 0.5436 | 0.4250 / 0.4248 / 0.4451 | 1.054× [1.016, 1.119] | 1.055× [1.018, 1.116] |
| Missing d8 host | 1.6819 / 1.6828 / 1.9248 | 1.3440 / 1.3521 / 1.7870 | 1.296× [1.204, 1.400] | 1.246× [1.191, 1.305] |
| Dense d10 GPU | 0.7784 / 0.7774 / 0.7957 | 0.7588 / 0.7625 / 0.7973 | 1.023× [1.014, 1.030] | 1.020× [1.011, 1.027] |
| Dense d10 host | 3.2393 / 3.1542 / 3.6902 | 3.2150 / 3.0850 / 3.6437 | 1.041× [0.947, 1.169] | 1.027× [0.954, 1.122] |
| Missing d10 GPU | 1.0126 / 1.0118 / 1.0274 | 0.9899 / 0.9878 / 0.9989 | 1.026× [1.024, 1.029] | 1.024× [1.021, 1.027] |
| Missing d10 host | 4.0175 / 4.0013 / 4.3298 | 3.1185 / 3.1407 / 3.4561 | 1.295× [1.252, 1.336] | 1.274× [1.239, 1.306] |

Profiles confirm terminal elision only on GPU trees, parallel descendants only on host trees, dense-host compaction zero and missing-host compaction one. GPU d8/d10 paired median benefits are 1.023–1.054×; missing host benefits are 1.295–1.335×. Both d6 GPU intervals and dense d10 host intervals cross one. Tails are not uniformly better: missing d6 GPU p95 rose 1.4%, dense d10 GPU p95 rose 0.2%, and the worst missing d8 host fit was 2.024 s versus release's 1.940 s despite improved median/mean/p95.

## Fixed same-binary secondary

All **96 planned fits across 32 processes passed**, with exact outputs and stable package/native fingerprints. Timing triplets and analysis match the primary definition; these results are not pooled with it.

| Case | Off median / mean / p95 | Default median / mean / p95 | Paired median speedup [95%] | Paired mean speedup [95%] |
|---|---|---|---|---|
| Dense d8 GPU | 0.3495 / 0.3478 / 0.3632 | 0.3418 / 0.3392 / 0.3546 | 1.022× [1.017, 1.028] | 1.025× [1.021, 1.030] |
| Dense d8 host | 1.4791 / 1.4975 / 1.8146 | 1.4351 / 1.3930 / 1.8301 | 1.103× [0.960, 1.279] | 1.081× [0.994, 1.186] |

Dense d8 GPU reproduced its benefit. Dense d8 host remains **inconclusive**, despite the primary release/default 1.160× result: both secondary paired intervals cross one. Its pooled mean ratio is 1.075×, while p95 is 0.9% worse (off/default ratio 0.992×); worst fits are 1.995 s default versus 1.937 s off. Those tails and all earlier losses remain in the record.

## Depth-twelve extension

The separately fixed extension completed **48 fits in 16 processes**, with no failures or unattempted fits. Four paired cycles provide 12 fits per arm/case; data and training length match the primary. Both arms request `metal_gpu_tree=1`, and every profile confirms automatic host fallback (`gpu_tree=0`) beyond the GPU depth cap. New optimization controls are omitted. Exact model/prediction/data/RNG/RMSE checks passed, with unchanged package/native fingerprints.

| Case | Release median / mean / p95 | Default median / mean / p95 | Paired median speedup [95%] | Paired mean speedup [95%] |
|---|---|---|---|---|
| Dense d12 host | 7.1799 / 7.1251 / 7.5814 | 6.5746 / 6.6278 / 6.9435 | 1.073× [1.026, 1.126] | 1.075× [1.021, 1.135] |
| Missing d12 host | 9.6117 / 9.4016 / 9.8357 | 7.1496 / 7.1416 / 7.3611 | 1.345× [1.324, 1.373] | 1.316× [1.262, 1.364] |

Default profiles use 256-thread host descendant scoring; dense compaction remains off and missing compaction is on. Terminal elision is not selected on this host path. These four-cycle intervals are separate from the primary and secondary analyses. All three runs' fit distributions and paired intervals were independently reproduced from their saved receipts. The validated defaults retain explicit opt-outs: `metal_fallback_parallel_splits=0` for host descendants and `metal_terminal_histogram_elision=0` for final GPU histograms. These measurements establish workload-specific gains; they do not establish an application-wide speedup.

## Exploration retained, including losses

| Experiment | Completed fits | Observation and disposition |
|---|---:|---|
| Instrumented profiling | 6 | Identified split scoring/final-depth work as dominant. Single-fit instrumented timings are excluded from production speedup claims. |
| Initial combined screen | 72 | Dense d8 host was **31.3% slower on/off**, despite 3.1% faster versus release. Dense d8 GPU was **3.4% slower versus release**, p95 8.6% higher. The fixed secondary run addresses these observations. |
| Dense empty-bin skipping | 36 | Two-cycle paired speedups at host d6/d8/d10: 1.402×/1.028×/1.094×; d8 inconclusive. |
| Host compact versus parallel scoring | 48 | Missing d8/d10 paired speedups: 1.301×/1.139×. Dense d10: **0.946×**, mean 7.1% and p95 5.8% slower; dense d8 inconclusive. Dense compaction was excluded from defaults. Missing d10 p95 also worsened 0.9%. |

The frozen default candidate passed **435 focused tests**. Earlier overlapping suites are not summed. Compatibility fixtures use the verified local pinned CPU reference; recorded hash/path adapters retain their functional assertions. The API audit passed with 99 declared exports and 84 Python references; Cox and survival generated-constant checks also passed. The full suite passed **3,999 tests in 468.12 seconds**, with four pytest workers, `MTL_DEBUG_LAYER=1`, `MTL_SHADER_VALIDATION=1`, and `OMP_NUM_THREADS=1`. The CPU reference is pinned to `6d2dbe19a099aa046178be8cfc303bfec1a623a6`.

## Reproduction and evidence

Replay follows the [benchmark guide](benchmarking.md) using `benchmarks/deep_trees.py` and `deep_tree_worker.py`, pinned named-arm configurations and fresh output directories. Receipts retain every fit, model, array, profile, native identity and command. Relative arm paths resolve beside the configuration; saved baseline-artifact paths remain absolute. Raw evidence is retained under ignored `benchmark-results/deep-trees-m4-20261026/`, following the repository guide; each run includes its own SHA-256 manifest.
