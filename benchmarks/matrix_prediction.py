"""Measure matrix construction, quantile reconstruction, and prediction prefixes."""
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_record(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def profile(model):
    value = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(value)))
    return json.loads(value.value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=100000)
    parser.add_argument('--query-rows', type=int, default=100000)
    parser.add_argument('--features', type=int, default=32)
    parser.add_argument('--bins', type=int, default=64)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cached-repeats', type=int, default=5)
    parser.add_argument('--mixed', action='store_true', help='Every fourth column contains category codes')
    parser.add_argument('--expected-library-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.query_rows, args.features, args.bins, args.depth,
           args.rounds, args.repeats, args.cached_repeats) < 1 or args.bins < 2:
        parser.error('Dimensions and repeat counts must be positive; bins must be at least two.')
    validators = {key: os.environ.get(key) for key in ('MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION')}
    if any(value not in (None, '0') for value in validators.values()):
        parser.error('Run timing separately from Metal validation.')
    build = xgb.build_info()
    metal = bool(build.get('USE_METAL', False))
    library = Path(build['libxgboost']).resolve()
    library_sha = digest(library)
    if args.expected_library_sha256 and library_sha != args.expected_library_sha256:
        parser.error('The loaded library does not match the requested build.')
    root = Path(__file__).resolve().parents[1]
    provenance = dict(library=str(library), library_sha256=library_sha,
                      python_package=str(Path(xgb.__file__).resolve()), xgboost_version=xgb.__version__)
    if metal:
        source = hashlib.sha256()
        for path in sorted([*root.glob('src/*'), *root.glob('include/metalgb/*'),
                            *root.glob('kernels/*.metal*'), root / 'CMakeLists.txt']):
            if path.is_file():
                source.update(str(path.relative_to(root)).encode())
                source.update(path.read_bytes())
        provenance['native_source_sha256'] = source.hexdigest()
    else:
        checkout = Path(xgb.__file__).resolve().parents[2]
        provenance['reference_commit'] = subprocess.check_output(
            ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    rng = np.random.default_rng(527)
    X = rng.normal(size=(args.rows, args.features)).astype(np.float32)
    query = (rng.normal(size=(args.query_rows, args.features)) * 1.1 + 0.25).astype(np.float32)
    types = ['q'] * args.features
    if args.mixed:
        for column in range(0, args.features, 4):
            X[:, column] = rng.integers(0, 11, len(X))
            query[:, column] = rng.integers(0, 11, len(query))
            types[column] = 'c'
    y = (2 * X[:, 0] + X[:, min(1, args.features - 1)] ** 2 +
         rng.normal(0, 0.1, len(X))).astype(np.float32)
    X[::29, -1] = np.nan
    query[::17, -1] = np.nan
    params = dict(tree_method='hist', max_bin=args.bins, max_depth=args.depth, seed=9)
    options = dict(feature_types=types, enable_categorical=True)
    modes = ('raw', 'quantile-independent', 'quantile-reference')

    def make_query(values, mode, train):
        if mode == 'raw':
            return xgb.DMatrix(values, **options)
        return xgb.QuantileDMatrix(values, max_bin=args.bins,
                                   ref=train if mode == 'quantile-reference' else None, **options)

    warm_rows = min(args.rows, 1024)
    warm = xgb.QuantileDMatrix(X[:warm_rows], label=y[:warm_rows], max_bin=args.bins, **options)
    warm_params = dict(params)
    if metal:
        warm_params['metal_features_per_group'] = 4 if args.rows >= 262144 and args.features >= 32 and args.bins <= 256 else 1
    warm_model = xgb.train(warm_params, warm, 2)
    for mode in modes:
        matrix = make_query(query[:min(1024, len(query))], mode, warm)
        warm_model.predict(matrix)
        warm_model.predict(matrix)
        matrix.get_data()
    del warm_model, warm, matrix
    measurements, predictions, cuts = [], {}, {}
    stable = None
    for repeat in range(args.repeats):
        start = time.perf_counter()
        train = xgb.QuantileDMatrix(X, label=y, max_bin=args.bins, **options)
        constructed = time.perf_counter()
        model = xgb.train(params, train, args.rounds)
        trained = time.perf_counter()
        record = dict(train_matrix_seconds=constructed - start, train_seconds=trained - constructed,
                      model_sha256=hashlib.sha256(model.save_raw()).hexdigest(), queries={})
        if metal:
            record['training_gpu_profile'] = profile(model)
        cuts['train_ptr'], cuts['train_values'] = train.get_quantile_cut()
        for mode in modes:
            start = time.perf_counter()
            matrix = make_query(query, mode, train)
            constructed = time.perf_counter()
            model.reset()
            start_predict = time.perf_counter()
            prediction = model.predict(matrix)
            cold = time.perf_counter() - start_predict
            assert np.isfinite(prediction).all()
            cached_times = []
            for _ in range(args.cached_repeats):
                start_cached = time.perf_counter()
                cached = model.predict(matrix)
                cached_times.append(time.perf_counter() - start_cached)
                np.testing.assert_array_equal(cached.view(np.uint32), prediction.view(np.uint32))
            start_export = time.perf_counter()
            csr = matrix.get_data()
            exported = time.perf_counter() - start_export
            query_record = dict(matrix_seconds=constructed - start, cold_predict_seconds=cold,
                                cached_predict_seconds=cached_times,
                                median_cached_predict_seconds=statistics.median(cached_times),
                                export_seconds=exported, prediction=array_record(prediction),
                                csr={field: array_record(getattr(csr, field))
                                     for field in ('indptr', 'indices', 'data')})
            if mode != 'raw':
                ptr, values = matrix.get_quantile_cut()
                cuts[mode + '_ptr'], cuts[mode + '_values'] = ptr, values
                query_record['cuts'] = dict(ptr=array_record(ptr), values=array_record(values))
            predictions[mode] = prediction
            record['queries'][mode] = query_record
            del matrix, csr, cached
        # Separate quantization may change query predictions; report the actual
        # differences without treating raw inputs as a quantile oracle.
        record['changed_predictions_from_raw'] = {
            mode: int(np.count_nonzero(predictions[mode].view(np.uint32) != predictions['raw'].view(np.uint32)))
            for mode in modes[1:]}
        signatures = dict(model=record['model_sha256'], queries={
            mode: {field: record['queries'][mode][field] for field in ('prediction', 'csr')}
            for mode in modes}, cuts={key: array_record(value) for key, value in cuts.items()})
        if stable is not None:
            assert signatures == stable, 'Repeated construction or training changed results'
        stable = signatures
        measurements.append(record)
        if repeat + 1 < args.repeats:
            del model, train
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for suffix, arrays in [('predictions.npz', predictions), ('cuts.npz', cuts)]:
        path = args.output.with_suffix('.' + suffix)
        np.savez(path, **arrays)
        artifacts[suffix] = dict(path=str(path), sha256=digest(path))
    model_path = args.output.with_suffix('.model.json')
    model.save_model(model_path)
    artifacts['model.json'] = dict(path=str(model_path), sha256=digest(model_path))
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(), build=build, **provenance,
                  benchmark_driver_sha256=digest(__file__), metal_validation=validators,
                  rows=args.rows, query_rows=args.query_rows, features=args.features,
                  rounds=args.rounds, repeats=args.repeats, mixed=args.mixed, parameters=params,
                  input_generation='NumPy default_rng(527); normal training features; query normal * 1.1 + 0.25; '
                                   'optional integer categories [0,11); last-column missing every 29th/17th row',
                  inputs=dict(train=array_record(X), query=array_record(query), labels=array_record(y)),
                  warmup=dict(rows=warm_rows, rounds=2, query_modes=list(modes)),
                  median_train_matrix_seconds=statistics.median(m['train_matrix_seconds'] for m in measurements),
                  median_train_seconds=statistics.median(m['train_seconds'] for m in measurements),
                  query_medians={mode: {
                      key: statistics.median(m['queries'][mode][key] for m in measurements)
                      for key in ('matrix_seconds', 'cold_predict_seconds', 'median_cached_predict_seconds', 'export_seconds')}
                      for mode in modes},
                  measurements=measurements, artifacts=artifacts,
                  note='Cold prediction follows model.reset(); cached predictions retain the same matrix. '
                       'Public-call wall times include output copies. This is not a CUDA runtime comparison.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
