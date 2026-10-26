#!/usr/bin/env python3
"""One pinned Metal deep-tree benchmark process. All profiling is outside timing."""
import argparse
import ctypes
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical_model(raw):
    marker = "$xgboost_nonfinite_float"
    locations = []

    def normalize(value, pointer=""):
        if isinstance(value, float) and not math.isfinite(value):
            token = "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
            locations.append({"pointer": pointer, "value": token})
            return {marker: token}
        if isinstance(value, dict):
            if marker in value:
                raise ValueError("Canonical metadata reserved key collision")
            return {key: normalize(value[key], pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
                    for key in sorted(value)}
        if isinstance(value, list):
            return [normalize(v, pointer + "/" + str(i)) for i, v in enumerate(value)]
        return value
    result = json.dumps(normalize(json.loads(raw)), sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode()
    return result, locations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--native-path", type=Path, required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--depth", type=int, choices=(6, 8, 10, 12), required=True)
    parser.add_argument("--density", choices=("dense", "missing"), required=True)
    parser.add_argument("--driver", choices=("gpu", "host"), required=True)
    parser.add_argument("--params-json", default="{}", help="JSON parameter overrides, applied last")
    parser.add_argument("--expected-gpu-tree", type=int, choices=(0, 1))
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup-rounds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    overrides = json.loads(args.params_json)
    if not isinstance(overrides, dict):
        parser.error("--params-json must be an object")
    if min(args.rows, args.bins, args.rounds, args.warmup_rounds, args.repeats) < 1 or args.features < 4:
        parser.error("Positive sizes, rounds/repeats, and at least four features required")
    if sys.flags.optimize or os.environ.get("OMP_NUM_THREADS") != "1":
        parser.error("Disable Python optimization and set OMP_NUM_THREADS=1")
    controls = {key: value for key, value in os.environ.items()
                if key.startswith(("MTL_", "METALGB_", "XGB_METAL_"))}
    if any(value not in ("", "0") for value in controls.values()):
        parser.error("Remove Metal validation/debug/trace controls before timing")
    if len(args.native_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.native_sha256):
        parser.error("Supply a lowercase SHA256 pin")

    import numpy as np
    import xgboost as xgb

    package = Path(xgb.__file__).resolve()
    native = Path(xgb.core._LIB._name).resolve()
    if package.parent.parent != args.package_root.resolve():
        raise RuntimeError(f"Unexpected Python package: {package}")
    if native != args.native_path.resolve() or sha(native) != args.native_sha256:
        raise RuntimeError(f"Unexpected native library: {native}")
    if not hasattr(xgb.core._LIB, "XGBoosterMetalProfile"):
        raise RuntimeError("Metal backend required")
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def array_record(value):
        return {"shape": list(value.shape), "dtype": value.dtype.str,
                "c_order_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}

    def profile(model):
        out = ctypes.c_char_p()
        xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(out)))
        return json.loads(out.value)

    params = dict(tree_method="hist", objective="reg:squarederror", max_depth=args.depth,
                  max_bin=args.bins, base_score=0, seed=271, eta=0.1, nthread=1,
                  metal_gpu_tree=int(args.driver == "gpu"))
    params.update(overrides)
    rng = np.random.default_rng(271828)
    X = rng.integers(-128, 128, (args.rows, args.features)).astype(np.float32) / 16
    query = rng.integers(-128, 128, (min(8192, args.rows), args.features)).astype(np.float32) / 16
    y = (X[:, 0] - 0.5 * X[:, 1] + X[:, 2] * X[:, 3] * 0.125).astype(np.float32)
    query_y = (query[:, 0] - 0.5 * query[:, 1] + query[:, 2] * query[:, 3] * 0.125).astype(np.float32)
    if args.density == "missing":
        X[rng.random(X.shape) < 0.15] = np.nan
        query[rng.random(query.shape) < 0.15] = np.nan
    identity = dict(package=str(package), native=str(native), native_sha256=sha(native),
                    worker_sha256=sha(__file__), python=sys.executable,
                    python_version=platform.python_version(), xgboost_version=xgb.__version__,
                    numpy_version=np.__version__, platform=platform.platform(), build_info=xgb.build_info())
    report = dict(status="running", started_utc=now(), arm=args.arm, case=args.case,
                  requested_case=dict(depth=args.depth, density=args.density, driver=args.driver),
                  overrides=overrides, params=params, identity=identity,
                  rounds=args.rounds, warmup_rounds=args.warmup_rounds, planned_repeats=args.repeats,
                  expected_gpu_tree=args.expected_gpu_tree,
                  environment={"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), **controls},
                  data={"train": array_record(X), "label": array_record(y),
                        "query": array_record(query), "query_label": array_record(query_y)},
                  timing="ordinary xgb.train including final reset; excludes matrix construction, predictions, profiling, and artifacts",
                  measurements=[])
    save("started.json", report)
    np.save(args.output / "query-label.npy", query_y, allow_pickle=False)
    stage, repeat = "warmup", None
    try:
        warmup = xgb.DMatrix(X, label=y)
        warm_model = xgb.train(params, warmup, args.warmup_rounds, verbose_eval=False)
        warm_model.predict(warmup)
        report["warmup_profile"] = profile(warm_model)
        del warm_model, warmup
        for repeat in range(args.repeats):
            stage = "matrix"
            start = time.perf_counter()
            data = xgb.DMatrix(X, label=y)
            ingested = time.perf_counter()
            stage = "training"
            model = xgb.train(params, data, args.rounds, verbose_eval=False)
            trained = time.perf_counter()
            stage = "outputs"
            # Capture training state immediately after the timer, before query prediction.
            training_profile = profile(model)
            tuning = training_profile["tuning"]
            if args.expected_gpu_tree is not None and tuning["gpu_tree"] != args.expected_gpu_tree:
                raise RuntimeError(f"Unexpected driver selection: {tuning}")
            query_matrix = xgb.DMatrix(query)
            prediction = model.predict(query_matrix)
            if not np.isfinite(prediction).all():
                raise RuntimeError("Nonfinite predictions")
            raw = bytes(model.save_raw(raw_format="ubj"))
            raw_json = bytes(model.save_raw(raw_format="json"))
            prefix = f"repeat-{repeat:02d}"
            np.save(args.output / (prefix + ".predictions.npy"), prediction, allow_pickle=False)
            (args.output / (prefix + ".model.ubj")).write_bytes(raw)
            (args.output / (prefix + ".model.json")).write_bytes(raw_json)
            canonical, nonfinite = canonical_model(raw_json)
            (args.output / (prefix + ".model.canonical.json")).write_bytes(canonical)
            config = json.loads(model.save_config())
            record = dict(repeat=repeat, ingest_seconds=ingested - start, train_seconds=trained - ingested,
                prediction=array_record(prediction), model_ubj_sha256=hashlib.sha256(raw).hexdigest(),
                model_json_sha256=hashlib.sha256(raw_json).hexdigest(),
                canonical_json_sha256=hashlib.sha256(canonical).hexdigest(),
                nonfinite_model_metadata=nonfinite,
                rmse=float(np.sqrt(np.mean((prediction.astype(np.float64) - query_y) ** 2))),
                rng_state=config["learner"]["generic_param"].get("rng_state"),
                profile=training_profile)
            save(prefix + ".json", record)
            report["measurements"].append(record)
            print(json.dumps(dict(repeat=repeat, train_seconds=record["train_seconds"],
                                  gpu_tree=tuning["gpu_tree"], rmse=record["rmse"])), flush=True)
            del model, data, query_matrix
        stage = "postflight"
        if sha(native) != args.native_sha256 or Path(xgb.__file__).resolve() != package:
            raise RuntimeError("Package/native changed during worker")
        report.update(status="passed", median_train_seconds=statistics.median(
            m["train_seconds"] for m in report["measurements"]))
    except BaseException as error:
        report.update(status="failed", failure=dict(stage=stage, repeat=repeat,
                      error=f"{type(error).__name__}: {error}"))
        raise
    finally:
        report["finished_utc"] = now()
        save("result.json", report)


if __name__ == "__main__":
    main()
