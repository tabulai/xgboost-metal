#!/usr/bin/env python3
"""Reproduce the three-machine regression comparison and thread sweeps.

See docs/benchmarking.md for setup, provenance and the original receipts.
Each timed fit runs in a fresh process; output directories must not exist.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = {"dense100": (100000, 32, 6), "dense262": (262144, 64, 8),
         "missing100": (100000, 32, 6)}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def worker(args: argparse.Namespace) -> None:
    # Imports are intentionally in the child process: CPU and Metal can use
    # different environments, and loading one must not contaminate the other.
    import numpy as np
    import scipy
    import xgboost as xgb

    build = xgb.build_info()
    metal = bool(build.get("USE_METAL", False))
    if metal != (args.backend == "metal"):
        raise RuntimeError(f"Wrong backend: expected {args.backend}, got {build}")
    if not metal and (not build.get("USE_OPENMP") or xgb.__version__ != args.cpu_version):
        raise RuntimeError(f"Expected OpenMP XGBoost {args.cpu_version}, got {xgb.__version__}: {build}")
    if np.__version__ != args.numpy_version:
        raise RuntimeError(f"Expected NumPy {args.numpy_version}, got {np.__version__}")
    native = Path(build["libxgboost"]).resolve()
    native_sha = file_hash(native)
    if args.native_sha and native_sha != args.native_sha:
        raise RuntimeError(f"Native library hash mismatch: {native_sha}")

    rows, features, depth = CASES[args.case]
    rng = np.random.default_rng(1337)
    X = rng.normal(size=(rows, features)).astype(np.float32)
    y = (2 * X[:, 0] + X[:, 1] ** 2 + 0.1 * rng.normal(size=rows)).astype(np.float32)
    if args.case == "missing100":
        X[rng.random(X.shape) < 0.15] = np.nan
    params = {"objective": "reg:squarederror", "tree_method": "hist", "max_depth": depth,
              "max_bin": 256, "eta": 0.3, "seed": 42, "nthread": args.threads,
              "device": args.backend}
    if args.split is not None:
        params["metal_split_threads"] = args.split
    if args.tile is not None:
        params["metal_histogram_rows"] = args.tile
    if args.group is not None:
        params["metal_features_per_group"] = args.group
    warmup_rows = 16384 if args.protocol == "gpu" else 1024
    warmup = xgb.DMatrix(X[:warmup_rows], label=y[:warmup_rows])
    xgb.train(params, warmup, num_boost_round=3).predict(warmup)
    del warmup
    start = time.perf_counter()
    data = xgb.DMatrix(X, label=y)
    ingest = time.perf_counter() - start
    cpu_start = time.process_time()
    start = time.perf_counter()
    model = xgb.train(params, data, num_boost_round=100, verbose_eval=False)
    elapsed = time.perf_counter() - start
    cpu_time = time.process_time() - cpu_start
    start = time.perf_counter()
    prediction = model.predict(data)
    prediction_time = time.perf_counter() - start
    profile = None
    if metal:
        pointer = ctypes.c_char_p()
        xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(pointer)))
        profile = json.loads(pointer.value)
    record = {"case": args.case, "backend": args.backend, "params": params,
              "rows": rows, "features": features, "rounds": 100, "warmup_rows": warmup_rows,
              "ingest_seconds": ingest, "train_seconds": elapsed, "process_cpu_seconds": cpu_time,
              "prediction_seconds": prediction_time,
              "rmse": float(np.sqrt(np.mean((prediction.astype(np.float64) - y) ** 2))),
              "input_sha256": digest(X.tobytes()), "label_sha256": digest(y.tobytes()),
              "prediction_sha256": digest(prediction.tobytes()),
              "model_sha256": digest(model.save_raw()), "native_sha256": native_sha,
              "build": build, "xgboost_version": xgb.__version__, "numpy_version": np.__version__,
              "scipy_version": scipy.__version__,
              "python_version": sys.version, "package_path": str(Path(xgb.__file__).resolve()),
              "platform": platform.platform(), "profile": profile}
    args.output.write_text(json.dumps(record, indent=2) + "\n")


def plan(args: argparse.Namespace) -> list[dict]:
    runs = []
    for cycle in range(1, args.cycles + 1):
        cases = list(args.cases)
        if args.mode != "compare":
            random.Random((103 if args.mode == "cpu-sweep" else 3800) + cycle).shuffle(cases)
        for case in cases:
            if args.mode == "compare":
                choices = [{"backend": b, "threads": args.threads} for b in
                           (["metal", "cpu"] if cycle % 2 else ["cpu", "metal"])]
            elif args.mode == "cpu-sweep":
                counts = list(args.thread_grid)
                random.Random(203 + cycle).shuffle(counts)
                choices = [{"backend": "cpu", "threads": n} for n in counts]
            else:
                if args.mode == "gpu-threads":
                    choices = [{"backend": "metal", "threads": n, "split": s}
                               for n in (1, 4, 8, 12, 16, 36) for s in (32, 256)]
                else:
                    choices = [{"backend": "metal", "threads": 1, "tile": t, "group": g}
                               for t in (1024, 2048, 4096) for g in (1, 2, 4)]
                random.Random(3900 + cycle).shuffle(choices)
            for config in choices:
                runs.append({"case": case, "cycle": cycle, **config})
    return runs


def run(args: argparse.Namespace) -> None:
    if args.cycles < 1 or args.threads < 1 or any(n < 1 for n in args.thread_grid):
        raise ValueError("Cycles and thread counts must be positive")
    if len(args.cases) != len(set(args.cases)) or len(args.thread_grid) != len(set(args.thread_grid)):
        raise ValueError("Cases and thread counts must be unique")
    schedule = plan(args)
    if any(r["backend"] == "cpu" for r in schedule) and not args.cpu_python:
        raise ValueError("--cpu-python is required for CPU runs")
    if any(r["backend"] == "metal" for r in schedule) and not args.metal_python:
        raise ValueError("--metal-python is required for Metal runs")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "runs").mkdir()
    manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "mode": args.mode, "schedule": schedule, "platform": platform.platform(),
                "cpu": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
                "logical_cpus": os.cpu_count(),
                "runner_checkout_commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                "runner_checkout_dirty": command_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]),
                "metal_checkout_commit": command_output(["git", "-C", str(args.metal_pythonpath.resolve().parent), "rev-parse", "HEAD"]),
                "metal_checkout_dirty": command_output(["git", "-C", str(args.metal_pythonpath.resolve().parent), "status", "--porcelain", "--untracked-files=no"]),
                "worker_sha256": file_hash(Path(__file__).resolve()),
                "numpy_version_required": args.numpy_version, "cpu_version_required": args.cpu_version,
                "openmp_sha256": file_hash(args.omp_library) if args.omp_library else None,
                "wait_policy": args.wait_policy,
                "note": "Training excludes data generation, matrix construction, prediction, and profiling."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for index, config in enumerate(schedule, 1):
        name = f"{index:04d}-{config['case']}-{config['backend']}"
        result_path = (args.output / "runs" / f"{name}.json").resolve()
        log_path = args.output / "runs" / f"{name}.log"
        python = args.metal_python if config["backend"] == "metal" else args.cpu_python
        cmd = [str(python), str(Path(__file__).resolve()), "worker", "--case", config["case"],
               "--backend", config["backend"], "--threads", str(config["threads"]),
               "--protocol", "gpu" if args.mode.startswith("gpu-") else "comparison",
               "--cpu-version", args.cpu_version, "--numpy-version", args.numpy_version,
               "--output", str(result_path)]
        for key in ("split", "tile", "group"):
            if key in config:
                cmd.extend(["--" + key, str(config[key])])
        expected = args.metal_sha if config["backend"] == "metal" else args.cpu_sha
        if expected:
            cmd.extend(["--native-sha", expected])
        env = os.environ.copy()
        for key in list(env):
            if key in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "KMP_BLOCKTIME", "PYTHONOPTIMIZE", "PYTHONPATH") or key.startswith(("MTL_", "METALGB_", "XGB_METAL_")):
                env.pop(key, None)
        package = args.metal_pythonpath if config["backend"] == "metal" else args.cpu_pythonpath
        if package:
            env["PYTHONPATH"] = str(package.resolve())
        if args.omp_library:
            env["DYLD_LIBRARY_PATH"] = str(args.omp_library.resolve().parent)
        if args.wait_policy == "passive":
            env.update(OMP_WAIT_POLICY="PASSIVE", KMP_BLOCKTIME="0")
        print(f"[{index}/{len(schedule)}] {name} cycle={config['cycle']}", flush=True)
        with log_path.open("w") as log:
            completed = subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Run failed ({completed.returncode}); see {log_path}")
        record = json.loads(result_path.read_text())
        results.append({"config": config, "receipt": str(result_path.relative_to(args.output.resolve())),
                        "train_seconds": record["train_seconds"], "input_sha256": record["input_sha256"],
                        "label_sha256": record["label_sha256"], "prediction_sha256": record["prediction_sha256"],
                        "model_sha256": record["model_sha256"], "native_sha256": record["native_sha256"]})
        (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
    native_hashes = {backend: sorted({r["native_sha256"] for r in results if r["config"]["backend"] == backend})
                     for backend in ("cpu", "metal")}
    summary = {"results": results, "native_hashes": native_hashes, "cases": {}}
    for case in args.cases:
        case_results = [r for r in results if r["config"]["case"] == case]
        groups = {}
        for r in case_results:
            key = json.dumps({k: v for k, v in r["config"].items() if k not in ("case", "cycle")}, sort_keys=True)
            groups.setdefault(key, []).append(r["train_seconds"])
        summary["cases"][case] = {"inputs_identical": len({r["input_sha256"] for r in case_results}) == 1,
                                  "labels_identical": len({r["label_sha256"] for r in case_results}) == 1,
                                  "predictions_identical": len({r["prediction_sha256"] for r in case_results}) == 1,
                                  "metal_models_identical": len({r["model_sha256"] for r in case_results
                                                                  if r["config"]["backend"] == "metal"}) <= 1,
                                  "median_train_seconds": {k: statistics.median(v) for k, v in groups.items()}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not all(len(hashes) <= 1 for hashes in native_hashes.values()) or not all(
            v["inputs_identical"] and v["labels_identical"] and v["predictions_identical"] and
            v["metal_models_identical"] for v in summary["cases"].values()):
        raise RuntimeError("Input, prediction, Metal model or native-library hashes differ; inspect summary.json")
    print(f"Completed {len(results)} runs: {args.output / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--mode", choices=("compare", "cpu-sweep", "gpu-threads", "gpu-hist"), default="compare")
    p.add_argument("--cpu-python", type=Path)
    p.add_argument("--metal-python", type=Path)
    p.add_argument("--cpu-pythonpath", type=Path, help="Optional parent of the stock xgboost package")
    p.add_argument("--metal-pythonpath", type=Path, default=ROOT / "python-package")
    p.add_argument("--omp-library", type=Path, help="OpenMP dylib to add to DYLD_LIBRARY_PATH and hash")
    p.add_argument("--cpu-sha", help="Require this CPU native-library SHA256")
    p.add_argument("--metal-sha", help="Require this Metal native-library SHA256")
    p.add_argument("--cpu-version", default="3.4.1")
    p.add_argument("--numpy-version", default="2.5.4")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--thread-grid", type=int, nargs="+", default=[1, 4, 8, 12, 16, 20, 24, 28, 32, 36])
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    p.add_argument("--wait-policy", choices=("default", "passive"), default="default")
    p.add_argument("--output", type=Path, required=True)
    w = sub.add_parser("worker", help=argparse.SUPPRESS)
    w.add_argument("--backend", choices=("cpu", "metal"), required=True)
    w.add_argument("--case", choices=tuple(CASES), required=True)
    w.add_argument("--threads", type=int, required=True)
    w.add_argument("--protocol", choices=("comparison", "gpu"), required=True)
    w.add_argument("--split", type=int, choices=(32, 256))
    w.add_argument("--tile", type=int)
    w.add_argument("--group", type=int)
    w.add_argument("--native-sha")
    w.add_argument("--cpu-version", required=True)
    w.add_argument("--numpy-version", required=True)
    w.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
