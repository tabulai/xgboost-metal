"""One pinned, serial benchmark worker; select baseline/candidate/CPU with PYTHONPATH.

Every repeat gets a fresh matrix and booster. Training timings exclude matrix
construction, predictions, profiling reads, hashes, and artifact writes. Run this
worker in a fresh process for each arm/case; it does not schedule other workers.
"""
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


def canonical_model(raw_json):
    """Keep nonfinite model metadata explicit without emitting invalid JSON.

    Raw export bytes remain the authoritative model artifact. This canonical
    comparison format reserves a key that is rejected in the original model,
    so a tagged nonfinite float cannot collide with an existing JSON object.
    """
    marker = "$xgboost_nonfinite_float"
    nonfinite = []

    def normalize(value, path=""):
        if isinstance(value, float) and not math.isfinite(value):
            token = "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
            nonfinite.append({"pointer": path, "value": token})
            return {marker: token}
        if isinstance(value, dict):
            if marker in value:
                raise ValueError(f"Reserved canonical-model key at {path or '/'}")
            return {key: normalize(value[key], path + "/" + key.replace("~", "~0").replace("/", "~1"))
                    for key in sorted(value)}
        if isinstance(value, list):
            return [normalize(item, path + "/" + str(index)) for index, item in enumerate(value)]
        return value

    normalized = normalize(json.loads(raw_json))
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return canonical, nonfinite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("metal", "cpu"), required=True)
    parser.add_argument("--package-root", type=Path, required=True, help="Expected parent of imported xgboost")
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory; never overwritten")
    parser.add_argument("--case", choices=("root", "fallback", "monotone", "interaction", "columns",
                                         "uniform", "gradient", "lossguide"), required=True)
    parser.add_argument("--fallback-parallel-root", choices=("0", "1", "default"), default="0",
                        help="Explicit control, or default to omit the Metal parameter")
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--bins", type=int, choices=(32, 256), default=256)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup-rounds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--nthread", type=int, default=1,
                        help="Explicit thread count; CPU-only 0 selects its default with OMP_NUM_THREADS unset")
    args = parser.parse_args()
    if min(args.rows, args.rounds, args.warmup_rounds, args.repeats) < 1 or args.features < 4 or args.nthread < 0:
        parser.error("Positive rows/rounds/repeats, nonnegative threads, and at least four features are required")
    if args.backend == "cpu" and args.fallback_parallel_root != "0":
        parser.error("CPU must use --fallback-parallel-root 0")
    if len(args.native_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.native_sha256):
        parser.error("--native-sha256 must be a lowercase SHA256 digest")
    if sys.flags.optimize:
        parser.error("Disable PYTHONOPTIMIZE")
    if args.nthread == 0:
        if args.backend != "cpu" or "OMP_NUM_THREADS" in os.environ:
            parser.error("--nthread 0 requires --backend cpu and OMP_NUM_THREADS unset")
    elif os.environ.get("OMP_NUM_THREADS") != str(args.nthread):
        parser.error("Set OMP_NUM_THREADS equal to the positive --nthread value")
    controls = {key: value for key, value in os.environ.items()
                if key.startswith(("MTL_", "METALGB_", "XGB_METAL_"))}
    if any(value not in ("", "0") for value in controls.values()):
        parser.error("Clear Metal debug/validation/trace/tuning environment controls before timing")

    import numpy as np
    import xgboost as xgb

    package = Path(xgb.__file__).resolve()
    if package.parent.parent != args.package_root.resolve():
        raise RuntimeError(f"Wrong package: {package}")
    native = Path(xgb.core._LIB._name).resolve()
    if sha(native) != args.native_sha256:
        raise RuntimeError(f"Wrong native library: {native}")
    metal = hasattr(xgb.core._LIB, "XGBoosterMetalProfile")
    if metal != (args.backend == "metal"):
        raise RuntimeError("Native backend differs from --backend")

    args.output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def array_record(value):
        return {"shape": list(value.shape), "dtype": value.dtype.str,
                "c_order_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}

    params = dict(tree_method="hist", objective="reg:squarederror", max_depth=6,
                  max_bin=args.bins, base_score=0, seed=271, eta=0.1, nthread=args.nthread)
    if args.case == "root":
        params["max_depth"] = 1
    elif args.case == "monotone":
        params["monotone_constraints"] = "(" + ",".join(map(str, [1, -1] + [0] * (args.features - 2))) + ")"
    elif args.case == "interaction":
        params["interaction_constraints"] = json.dumps([list(range(0, args.features, 2)),
                                                        list(range(1, args.features, 2))])
    elif args.case == "columns":
        params.update(colsample_bytree=0.8, colsample_bylevel=0.75, colsample_bynode=0.7)
    elif args.case in ("uniform", "gradient"):
        params.update(subsample=0.63, sampling_method="uniform" if args.case == "uniform" else "gradient_based")
    elif args.case == "lossguide":
        params.update(grow_policy="lossguide", max_depth=0, max_leaves=32)
    if metal:
        # Force the host controller for every Metal arm, including the plain depth-six control.
        params["metal_gpu_tree"] = 0
        if args.fallback_parallel_root != "default":
            params["metal_fallback_parallel_root"] = int(args.fallback_parallel_root)
    expected_fallback = 1 if args.fallback_parallel_root == "default" else int(args.fallback_parallel_root)

    rng = np.random.default_rng(271828)
    X = rng.integers(-128, 128, (args.rows, args.features)).astype(np.float32) / 16
    query = rng.integers(-128, 128, (min(8192, args.rows), args.features)).astype(np.float32) / 16
    y = (X[:, 0] - 0.5 * X[:, 1] + X[:, 2] * X[:, 3] * 0.125).astype(np.float32)
    X[rng.random(X.shape) < 0.15] = np.nan
    query[rng.random(query.shape) < 0.15] = np.nan
    identity = {"package": str(package), "native": str(native), "native_sha256": sha(native),
                "python": sys.executable, "xgboost_version": xgb.__version__,
                "worker_sha256": sha(__file__), "platform": platform.platform()}
    report = {"status": "running", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "identity": identity, "backend": args.backend, "case": args.case, "params": params,
              "fallback_parallel_root_control": args.fallback_parallel_root,
              "rounds": args.rounds, "warmup_rounds": args.warmup_rounds, "planned_repeats": args.repeats,
              "environment": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), **controls},
              "thread_policy": "CPU default" if args.nthread == 0 else "explicit OMP and nthread",
              "data": {"train": array_record(X), "label": array_record(y), "query": array_record(query)},
              "canonical_json": {"version": 2, "scope": "entire model JSON", "sort_keys": True,
                                 "separators": [",", ":"], "allow_nan": False,
                                 "nonfinite_tag": "$xgboost_nonfinite_float",
                                 "nonfinite_values": ["-Infinity", "+Infinity", "NaN"],
                                 "nonfinite_locations": "RFC6901 pointers in each repeat receipt",
                                 "raw_export": "repeat-NN.model.json preserves original export bytes",
                                 "canonical_export": "repeat-NN.model.canonical.json, no trailing newline"},
              "measurements": []}
    write("started.json", report)
    stage, repeat = "warmup", None
    try:
        warmup = xgb.DMatrix(X, label=y)
        warmup_model = xgb.train(params, warmup, args.warmup_rounds, verbose_eval=False)
        warmup_model.predict(warmup)
        del warmup_model, warmup
        for repeat in range(args.repeats):
            stage = "matrix"
            start = time.perf_counter()
            data = xgb.DMatrix(X, label=y)
            ingested = time.perf_counter()
            stage = "training"
            model = xgb.train(params, data, args.rounds, verbose_eval=False)
            trained = time.perf_counter()
            stage = "outputs"
            query_matrix = xgb.DMatrix(query)
            prediction = model.predict(query_matrix)
            if not np.isfinite(prediction).all():
                raise RuntimeError("Nonfinite predictions")
            raw = bytes(model.save_raw(raw_format="ubj"))
            raw_json = bytes(model.save_raw(raw_format="json"))
            prefix = f"repeat-{repeat:02d}"
            # Preserve successful raw outputs even if later formatting/profile checks fail.
            np.save(args.output / f"{prefix}.predictions.npy", prediction, allow_pickle=False)
            (args.output / f"{prefix}.model.ubj").write_bytes(raw)
            (args.output / f"{prefix}.model.json").write_bytes(raw_json)
            canonical, nonfinite = canonical_model(raw_json)
            (args.output / f"{prefix}.model.canonical.json").write_bytes(canonical)
            record = {"repeat": repeat, "ingest_seconds": ingested - start, "train_seconds": trained - ingested,
                      "prediction": array_record(prediction), "model_ubj_sha256": hashlib.sha256(raw).hexdigest(),
                      "model_json_sha256": hashlib.sha256(raw_json).hexdigest(),
                      "canonical_json_sha256": hashlib.sha256(canonical).hexdigest(),
                      "nonfinite_model_metadata": nonfinite,
                      "rng_state": json.loads(model.save_config())["learner"]["generic_param"].get("rng_state")}
            if metal:
                output = ctypes.c_char_p()
                xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(output)))
                record["profile"] = json.loads(output.value)
                tuning = record["profile"]["tuning"]
                if tuning["gpu_tree"] or tuning.get("fallback_parallel_root", 0) != expected_fallback:
                    raise RuntimeError(f"Unexpected dispatch selection: {tuning}")
            write(f"{prefix}.json", record)
            report["measurements"].append(record)
            print(json.dumps({"repeat": repeat, "train_seconds": record["train_seconds"]}), flush=True)
            del model, data, query_matrix
        stage = "postflight"
        if sha(native) != args.native_sha256 or Path(xgb.__file__).resolve() != package:
            raise RuntimeError("Package/native changed during benchmark")
        report.update(status="passed", median_train_seconds=statistics.median(
            item["train_seconds"] for item in report["measurements"]))
    except BaseException as error:
        report.update(status="failed", failure={"stage": stage, "repeat": repeat,
                      "train_seconds": None, "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write("result.json", report)


if __name__ == "__main__":
    main()
