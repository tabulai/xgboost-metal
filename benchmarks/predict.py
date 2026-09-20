"""Compare direct and CUDA-inspired cached Metal traversal on identical models."""
from pathlib import Path
import argparse
import ctypes
import datetime
import hashlib
import json
import os
import platform
import statistics
import time

import numpy as np
import xgboost as xgb


def profile(model):
    out = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(out)))
    return json.loads(out.value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[100000, 1000000])
    parser.add_argument("--features", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--rounds", nargs="+", type=int, default=[30, 100])
    parser.add_argument("--threads", nargs="+", type=int, choices=[32, 64, 128, 256], default=[32, 64, 128])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--leaf", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.rows, *args.features, *args.rounds, args.repeats) < 1:
        parser.error("Sizes and repeats must be positive.")
    build = xgb.build_info()
    assert build.get("USE_METAL"), "This experiment requires the Metal library."
    root = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for path in sorted([*root.glob("src/*"), *root.glob("include/metalgb/*"),
                        *root.glob("kernels/*.metal*"), root / "CMakeLists.txt"]):
        if path.is_file():
            source_hash.update(str(path.relative_to(root)).encode())
            source_hash.update(path.read_bytes())
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(), build=build, xgboost_version=xgb.__version__,
                  library_sha256=hashlib.sha256(Path(build["libxgboost"]).read_bytes()).hexdigest(),
                  native_source_sha256=source_hash.hexdigest(), measurements=[],
                  metal_validation={k: os.environ.get(k) for k in ("MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION")},
                  note="Each comparison uses identical input and model bytes; exact outputs are checked outside timing. Ingestion and training are excluded.")
    rng = np.random.default_rng(43)
    for columns in args.features:
        X = rng.normal(size=(max(args.rows), columns)).astype(np.float32)
        y = (2 * X[:, 0] + X[:, min(1, columns - 1)] ** 2).astype(np.float32)
        training = xgb.DMatrix(X[:32768], label=y[:32768])
        for rounds in args.rounds:
            params = dict(max_depth=6, max_bin=256, seed=9)
            model = xgb.train(params, training, rounds)
            model_sha = hashlib.sha256(model.save_raw()).hexdigest()
            for rows in args.rows:
                data = xgb.DMatrix(X[:rows])
                model.set_param("metal_predict_shared", 0)
                expected = model.predict(data, pred_leaf=args.leaf)
                runs = []
                configurations = [(0, 256)] + [(1, t) for t in args.threads if (columns | 1) * t * 4 <= 32 * 1024]
                # Warm every pipeline before recording any samples for this input.
                for shared, threads in configurations:
                    model.set_param(dict(metal_predict_shared=shared, metal_predict_threads=threads))
                    for _ in range(2):
                        np.testing.assert_array_equal(model.predict(data, pred_leaf=args.leaf), expected)
                    runs.append(dict(shared=shared, threads=threads, wall_seconds=[], gpu_seconds=[]))
                # Alternate settings each repetition to limit order/thermal bias.
                for repeat in range(args.repeats):
                    for run in runs[::1 if repeat % 2 == 0 else -1]:
                        model.set_param(dict(metal_predict_shared=run["shared"], metal_predict_threads=run["threads"]))
                        before = profile(model)["gpu_seconds"].get("predict", 0)
                        start = time.perf_counter()
                        result = model.predict(data, pred_leaf=args.leaf)
                        elapsed = time.perf_counter() - start
                        after = profile(model)["gpu_seconds"].get("predict", 0)
                        np.testing.assert_array_equal(result, expected)
                        run["wall_seconds"].append(elapsed)
                        run["gpu_seconds"].append(after - before)
                for run in runs:
                    run["median_wall_seconds"] = statistics.median(run["wall_seconds"])
                    run["median_gpu_seconds"] = statistics.median(run["gpu_seconds"])
                measurement = dict(rows=rows, features=columns, rounds=rounds, leaf=args.leaf,
                                   model_sha256=model_sha, training_parameters=params, runs=runs)
                report["measurements"].append(measurement)
                print(json.dumps({k: measurement[k] for k in ("rows", "features", "rounds", "leaf")}),
                      [(r["shared"], r["threads"], round(r["median_wall_seconds"] * 1e3, 3)) for r in runs], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
