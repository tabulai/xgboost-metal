"""Measure tree updates from one shared, pinned-reference forest.

Prepare the forest once using the reference Python package, then run measurements
from the reference and installed Metal environments with the same arguments.
"""
from pathlib import Path
import argparse
import ctypes
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time

import numpy as np
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def provenance():
    build = xgb.build_info()
    result = dict(build=build, library_sha256=sha256(Path(build["libxgboost"]).read_bytes()),
                  xgboost_version=xgb.__version__, python_version=platform.python_version(),
                  numpy_version=np.__version__)
    if build.get("USE_METAL", False):
        digest = hashlib.sha256()
        for path in sorted([*ROOT.glob("src/*"), *ROOT.glob("include/metalgb/*"),
                            *ROOT.glob("kernels/*.metal*"), ROOT / "CMakeLists.txt"]):
            if path.is_file():
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
        result["native_source_sha256"] = digest.hexdigest()
    else:
        checkout = Path(xgb.__file__).resolve().parents[2]
        result["reference_commit"] = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    return result


def dataset(rows, features):
    rng = np.random.default_rng(43)
    X = rng.normal(size=(rows, features)).astype(np.float32)
    noise = rng.normal(0, 0.1, rows)
    second = min(1, features - 1)
    original = (2 * X[:, 0] + X[:, second] ** 2 + noise).astype(np.float32)
    updated = (2.2 * X[:, 0] + 0.9 * X[:, second] ** 2 + 0.25 + noise).astype(np.float32)
    return X, original, updated


def model_details(model):
    saved = json.loads(model.save_raw(raw_format="json"))["learner"]
    forest = saved["gradient_booster"]["model"]
    structure, values = hashlib.sha256(), hashlib.sha256()
    active_leaves = active_nodes = deleted = 0
    for tree in forest["trees"]:
        for name in ("left_children", "right_children", "parents", "split_indices", "default_left", "split_type",
                     "categories", "categories_nodes", "categories_segments", "categories_sizes"):
            structure.update(np.asarray(tree[name], dtype="<i4").tobytes())
        for name in ("split_conditions", "base_weights", "sum_hessian", "loss_changes"):
            values.update(np.asarray(tree[name], dtype="<f4").tobytes())
        deleted += int(tree["tree_param"]["num_deleted"])
        stack = [0]
        while stack:
            node = stack.pop()
            active_nodes += 1
            left = tree["left_children"][node]
            if left < 0:
                active_leaves += 1
            else:
                stack.extend([left, tree["right_children"][node]])
    return dict(model_sha256=sha256(model.save_raw()), structure_sha256=structure.hexdigest(),
                float32_statistics_sha256=values.hexdigest(), active_nodes=active_nodes,
                active_leaves=active_leaves, deleted_nodes=deleted,
                stored_nodes=sum(len(t["left_children"]) for t in forest["trees"]),
                trees=len(forest["trees"]), boosted_rounds=model.num_boosted_rounds())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true", help="Create the shared forest with the pinned CPU reference")
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--depth", type=int, default=6, help="Depth of the initial forest")
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--updater", choices=("refresh", "prune", "refresh,prune"), default="refresh")
    parser.add_argument("--keep-leaves", action="store_true", help="Refresh statistics while retaining existing leaf predictions")
    parser.add_argument("--prune-depth", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--coalesce-encoders", type=int, choices=(0, 1))
    parser.add_argument("--compute-fill", type=int, choices=(0, 1))
    parser.add_argument("--refresh-columns", type=int, choices=(0, 1), help="Compare cached column storage with direct row-major feature reads")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-result", action="store_true", help="Save the last model and complete prediction array outside timing")
    args = parser.parse_args()
    if min(args.rows, args.features, args.rounds, args.depth, args.bins, args.repeats) < 1:
        parser.error("Dataset sizes, depth, bins, rounds and repeats must be positive.")
    if args.prune_depth < 0 or args.gamma < 0:
        parser.error("Pruning depth and gamma must be nonnegative.")
    if not args.prepare and args.output is None:
        parser.error("Measurements require --output so all trials are retained.")
    build = provenance()
    is_metal = build["build"].get("USE_METAL", False)
    X, original, target = dataset(args.rows, args.features)
    metadata_path = args.model.with_suffix(args.model.suffix + ".json")
    shared = dict(rows=args.rows, features=args.features, rounds=args.rounds, depth=args.depth, bins=args.bins)
    if args.prepare:
        assert not is_metal and Path(xgb.__file__).resolve().is_relative_to(ROOT / "ref/xgboost")
        params = dict(tree_method="hist", max_depth=args.depth, max_bin=args.bins, eta=0.3, seed=9,
                      nthread=1, verbosity=0)
        started = time.perf_counter()
        model = xgb.train(params, xgb.DMatrix(X, label=original), args.rounds, verbose_eval=False)
        elapsed = time.perf_counter() - started
        args.model.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(args.model)
        record = dict(**shared, parameters=params, **build, model_file_sha256=sha256(args.model.read_bytes()),
                      preparation_seconds=elapsed, data_seed=43, **model_details(model))
        metadata_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"model": str(args.model), "metadata": str(metadata_path), "preparation_seconds": elapsed}))
        return

    payload = args.model.read_bytes()
    original_metadata = json.loads(metadata_path.read_text())
    assert original_metadata["model_file_sha256"] == sha256(payload)
    assert all(original_metadata[key] == value for key, value in shared.items()), "Shared forest dimensions differ"
    params = dict(process_type="update", updater=args.updater, refresh_leaf=not args.keep_leaves,
                  eta=0.3, max_depth=args.prune_depth, max_leaves=0, gamma=args.gamma,
                  nthread=1, seed=9, verbosity=0)
    if is_metal:
        for option, key in [(args.coalesce_encoders, "metal_coalesce_encoders"),
                            (args.compute_fill, "metal_compute_fill"), (args.refresh_columns, "metal_refresh_columns")]:
            if option is not None:
                params[key] = option
    warmup_rows = min(1024, len(X))
    warmup = xgb.DMatrix(X[:warmup_rows], label=target[:warmup_rows])
    warmup_model = xgb.train(params, warmup, 1, xgb_model=bytearray(payload), verbose_eval=False)
    warmup_model.predict(warmup)
    del warmup_model, warmup
    measurements = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        data = xgb.DMatrix(X, label=target)
        ingested = time.perf_counter()
        model = xgb.train(params, data, args.rounds, xgb_model=bytearray(payload), verbose_eval=False)
        updated = time.perf_counter()
        prediction = model.predict(data)
        predicted = time.perf_counter()
        assert np.isfinite(prediction).all(), "Prediction contains nonfinite values"
        measurement = dict(ingest_seconds=ingested - started, update_seconds=updated - ingested,
                           predict_seconds=predicted - updated,
                           rmse=float(np.sqrt(np.mean((prediction.astype(np.float64) - target) ** 2))),
                           prediction_sha256=sha256(prediction.tobytes()), prediction_sample=prediction[:16].tolist(),
                           **model_details(model))
        if is_metal:
            profile = ctypes.c_char_p()
            xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(profile)))
            measurement["gpu_profile"] = json.loads(profile.value)
        measurements.append(measurement)
    result_files = {}
    if args.save_result:
        prefix = args.output.with_suffix("")
        model_path, prediction_path = Path(str(prefix) + "-model.json"), Path(str(prefix) + "-prediction.npy")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)
        np.save(prediction_path, prediction)
        result_files = dict(model=str(model_path), prediction=str(prediction_path),
                            model_sha256=sha256(model_path.read_bytes()), prediction_sha256=sha256(prediction_path.read_bytes()))
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(), **build, **shared, parameters=params,
                  benchmark_driver_sha256=sha256(Path(__file__).read_bytes()),
                  original_model=original_metadata, original_model_path=str(args.model),
                  data_seed=43,
                  original_response=f"2 * X[:, 0] + X[:, {min(1, args.features - 1)}] ** 2 + normal(0, 0.1)",
                  updated_response=f"2.2 * X[:, 0] + 0.9 * X[:, {min(1, args.features - 1)}] ** 2 + 0.25 + same noise",
                  warmup=dict(rows=warmup_rows, rounds=1, prediction=True),
                  measurements=measurements, result_files=result_files,
                  median_update_seconds=statistics.median(m["update_seconds"] for m in measurements),
                  median_predict_seconds=statistics.median(m["predict_seconds"] for m in measurements),
                  model_deterministic=len({m["model_sha256"] for m in measurements}) == 1,
                  prediction_deterministic=len({m["prediction_sha256"] for m in measurements}) == 1,
                  metal_validation={key: os.environ.get(key) for key in ("MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION")},
                  timing_scope="Fresh DMatrix and learner each trial; update includes parsing the shared in-memory model and reset. Initial forest training and result serialization are outside timing.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("median_update_seconds", "median_predict_seconds", "model_deterministic", "prediction_deterministic")}))


if __name__ == "__main__":
    main()
