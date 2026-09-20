"""Measure owned DMatrix ingestion for dense, strided and columnar input."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import time

import numpy as np
import pandas as pd
import xgboost as xgb


def prepare(values, kind):
    if kind == "dense":
        return values, values
    if kind == "fortran":
        return np.asfortranarray(values), values
    if kind == "strided":
        return values[::-1, ::-1], values[::-1, ::-1]
    if kind == "float64":
        return values.astype(np.float64), values
    if kind == "dataframe":
        return pd.DataFrame(values), values
    if kind == "nullable":
        return pd.DataFrame(values, dtype="Float32"), values
    if kind == "arrow":
        import pyarrow as pa
        return pa.table({f"f{c}": pa.array(values[:, c], from_pandas=True) for c in range(values.shape[1])}), values
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--kinds", nargs="+", default=["dense", "fortran", "strided", "float64", "dataframe", "nullable", "arrow"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.features, args.repeats) < 1:
        parser.error("Dimensions and repeats must be positive.")
    root = Path(__file__).resolve().parents[1]
    build = xgb.build_info()
    source_hash = hashlib.sha256()
    for path in sorted([*root.glob("src/*"), *root.glob("include/metalgb/*"),
                        *root.glob("kernels/*.metal*"), root / "CMakeLists.txt"]):
        if path.is_file():
            source_hash.update(str(path.relative_to(root)).encode())
            source_hash.update(path.read_bytes())
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(), build=build, xgboost_version=xgb.__version__,
                  rows=args.rows, features=args.features, measurements=[],
                  numpy=np.__version__, pandas=pd.__version__,
                  library_sha256=hashlib.sha256(Path(build["libxgboost"]).read_bytes()).hexdigest(),
                  native_source_sha256=source_hash.hexdigest(),
                  metal_validation={k: os.environ.get(k) for k in ("MTL_DEBUG_LAYER", "MTL_SHADER_VALIDATION")},
                  note="Fresh DMatrix per repetition after one full warmup. Input construction and correctness checks are outside timing.")
    values = np.random.default_rng(57).normal(size=(args.rows, args.features)).astype(np.float32)
    values[::17, ::7] = np.nan
    selected = np.unique([0, args.rows // 2, args.rows - 1]).astype(np.int32)
    for kind in args.kinds:
        data, expected = prepare(values, kind)
        expected_count = int(np.count_nonzero(~np.isnan(expected)))
        # Warm compilation and the adapter's imports before recording samples.
        warmup = xgb.DMatrix(data)
        del warmup
        seconds = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            matrix = xgb.DMatrix(data)
            seconds.append(time.perf_counter() - start)
            assert matrix.num_nonmissing() == expected_count
            sample = matrix.slice(selected).get_data().toarray()
            np.testing.assert_array_equal(sample, np.nan_to_num(expected[selected], nan=0))
            del matrix
        measurement = dict(kind=kind, seconds=seconds, median_seconds=statistics.median(seconds))
        report["measurements"].append(measurement)
        print(kind, f"{measurement['median_seconds'] * 1e3:.3f} ms", flush=True)
        del data, expected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
