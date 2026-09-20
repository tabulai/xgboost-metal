"""Measure ranking updates and repeated metrics with retained public caches."""
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(model):
    output = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(output)))
    return json.loads(output.value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', choices=['rank:ndcg', 'rank:map', 'rank:pairwise'], default='rank:ndcg')
    parser.add_argument('--method', choices=['topk', 'mean'], default='topk')
    parser.add_argument('--rows', type=int, default=65536)
    parser.add_argument('--features', type=int, default=16)
    parser.add_argument('--query-size', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--bins', type=int, default=64)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--eval-repeats', type=int, default=11)
    parser.add_argument('--expected-library-sha256', required=True)
    parser.add_argument('--build-record', type=Path, help='Source provenance for an archived library')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.query_size < 2 or args.rows < args.query_size or args.rows % args.query_size or
            args.features < 3 or min(args.rounds, args.repeats, args.eval_repeats, args.pairs) < 1):
        parser.error('Use complete queries, at least three features and positive repetition counts.')
    root = Path(__file__).resolve().parents[1]
    build = xgb.build_info()
    library = Path(build['libxgboost']).resolve()
    library_sha = digest(library)
    assert library_sha == args.expected_library_sha256, library
    metal = bool(build.get('USE_METAL'))
    provenance = {'library_sha256': library_sha}
    if args.build_record:
        record = json.loads(args.build_record.read_text())
        assert record['library_sha256'] == library_sha
        provenance['native_source_sha256'] = record['native_source_sha256']
        provenance['build_record'] = str(args.build_record)
    elif metal:
        source = hashlib.sha256()
        for path in sorted([*root.glob('src/*'), *root.glob('include/metalgb/*'),
                            *root.glob('kernels/*.metal*'), root / 'CMakeLists.txt']):
            if path.is_file():
                source.update(str(path.relative_to(root)).encode())
                source.update(path.read_bytes())
        provenance['native_source_sha256'] = source.hexdigest()
    else:
        assert not build.get('USE_CUDA') and not build.get('USE_OPENMP')
        provenance['reference_commit'] = subprocess.check_output(
            ['git', '-C', str(root / 'ref/xgboost'), 'rev-parse', 'HEAD'], text=True).strip()
    rng = np.random.default_rng(9137)
    X = rng.integers(-16, 17, (args.rows, args.features)).astype(np.float32)
    label = (2 * X[:, 0] - X[:, 1] + X[:, 2] + rng.integers(-8, 9, args.rows) > 0).astype(np.float32)
    X[::37, -1] = np.nan
    qid = np.arange(args.rows, dtype=np.uint32) // args.query_size
    params = dict(objective=args.objective, lambdarank_pair_method=args.method,
                  lambdarank_num_pair_per_sample=args.pairs, max_depth=args.depth,
                  max_bin=args.bins, base_score=0, seed=97, nthread=1, tree_method='hist',
                  eval_metric=['ndcg@3', 'ndcg@10', 'map@5', 'pre@10'])
    warm_rows = min(args.rows, max(args.query_size, (1024 // args.query_size) * args.query_size))
    warm = xgb.DMatrix(X[:warm_rows], label=label[:warm_rows], qid=qid[:warm_rows])
    warm_model = xgb.Booster(params, [warm])
    warm_model.update(warm, 0)
    warm_model.eval(warm)
    warm_model.predict(warm)
    del warm_model, warm
    measurements = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        data = xgb.DMatrix(X, label=label, qid=qid)
        ingest = time.perf_counter() - start
        model = xgb.Booster(params, [data])
        updates, evaluations, values = [], [], []
        start = time.perf_counter()
        for iteration in range(args.rounds):
            before = time.perf_counter()
            model.update(data, iteration)
            updates.append(time.perf_counter() - before)
            before = time.perf_counter()
            value = model.eval_set([(data, 'query')], iteration)
            evaluations.append(time.perf_counter() - before)
            values.append(value)
        train = time.perf_counter() - start
        repeats = []
        for _ in range(args.eval_repeats):
            before = time.perf_counter()
            value = model.eval_set([(data, 'query')], args.rounds - 1)
            repeats.append(time.perf_counter() - before)
            assert value == values[-1]
        margin = model.predict(data, output_margin=True)
        raw = model.save_raw(raw_format='ubj')
        config = json.loads(model.save_config())['learner']
        row = dict(ingest_seconds=ingest, train_with_metrics_seconds=train,
                   update_seconds=sum(updates), training_evaluation_seconds=sum(evaluations),
                   update_seconds_by_round=updates, evaluation_seconds_by_round=evaluations,
                   repeated_evaluation_seconds=repeats,
                   cached_evaluation_seconds=statistics.median(repeats),
                   metric_values_by_round=values,
                   rng_state=config['generic_param']['rng_state'],
                   metric_config=config['metrics'], objective_config=config['objective'],
                   model_sha256=hashlib.sha256(raw).hexdigest(),
                   margin_sha256=hashlib.sha256(margin.tobytes()).hexdigest())
        if metal:
            row['gpu_profile'] = profile(model)
        measurements.append(row)
    assert len({entry['model_sha256'] for entry in measurements}) == 1
    assert len({entry['margin_sha256'] for entry in measurements}) == 1
    assert all(entry['metric_values_by_round'] == measurements[0]['metric_values_by_round']
               for entry in measurements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.output.with_suffix('.model.json')
    margin_path = args.output.with_suffix('.predictions.npz')
    cuts_path = args.output.with_suffix('.cuts.npz')
    model.save_model(model_path)
    np.savez_compressed(margin_path, margin=margin)
    indptr, cuts = data.get_quantile_cut()
    np.savez_compressed(cuts_path, indptr=indptr, cuts=cuts)
    artifacts = {}
    for name, path in [('model', model_path), ('predictions', margin_path), ('cuts', cuts_path)]:
        resolved = path.resolve()
        artifacts[name] = dict(path=str(resolved.relative_to(root) if resolved.is_relative_to(root) else resolved),
                               sha256=digest(path))
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(), build=build, parameters=params, **provenance,
                  benchmark_driver_sha256=digest(Path(__file__)), rows=args.rows, features=args.features,
                  query_size=args.query_size, rounds=args.rounds, measurements=measurements,
                  warmup=dict(rows=warm_rows, rounds=1, prediction=True, metrics=params['eval_metric']),
                  inputs={name: hashlib.sha256(array.tobytes()).hexdigest()
                          for name, array in [('features', X), ('labels', label), ('qid', qid)]},
                  metal_validation={key: os.environ.get(key)
                                    for key in ['MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION']},
                  artifacts=artifacts)
    for key in ['ingest_seconds', 'train_with_metrics_seconds', 'update_seconds',
                'training_evaluation_seconds', 'cached_evaluation_seconds']:
        report['median_' + key] = statistics.median(row[key] for row in measurements)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key.startswith('median_')}))


if __name__ == '__main__':
    main()
