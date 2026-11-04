"""Benchmark the MNIST CSV files with an installed XGBoost distribution."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=Path, required=True)
parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
parser.add_argument('--threads', type=int, default=20)
parser.add_argument('--backend', choices=['metal', 'cpu'], required=True)
parser.add_argument('--trial', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()

build_info = xgb.build_info()
if bool(build_info.get('USE_METAL', False)) != (args.backend == 'metal'):
    parser.error('the installed XGBoost package does not match --backend')

base = args.input_dir
dtype = np.dtype(args.dtype).type
paths = [base / 'train.csv.gz', base / 'test.csv.gz']
frames = [pd.read_csv(p, dtype=np.uint8) for p in paths]
assert list(frames[0].columns) == list(frames[1].columns)
assert frames[0].columns[0] == 'label'
x_train = frames[0].iloc[:, 1:].to_numpy(dtype=dtype, copy=True, na_value=np.nan) / dtype(255)
y_train = frames[0].iloc[:, 0].to_numpy(dtype=np.int32)
x_test = frames[1].iloc[:, 1:].to_numpy(dtype=dtype, copy=True, na_value=np.nan) / dtype(255)
y_test = frames[1].iloc[:, 0].to_numpy(dtype=np.int32)
assert x_train.shape == (60000, 784) and x_test.shape == (10000, 784)
assert np.array_equal(np.unique(y_train), np.arange(10))
params = dict(n_estimators=100, max_depth=6, learning_rate=0.3,
              tree_method='hist', n_jobs=args.threads, random_state=42,
              device='metal' if args.backend == 'metal' else 'cpu')
model = xgb.XGBClassifier(**params)
start = time.perf_counter()
model.fit(x_train, y_train)
fit = time.perf_counter() - start
model.predict(x_test)
model.predict_proba(x_test)
times = {}
for name in ('predict', 'predict_proba'):
    measurements = []
    for _ in range(10):
        start = time.perf_counter()
        value = getattr(model, name)(x_test)
        measurements.append(time.perf_counter() - start)
    times[name] = measurements
    if name == 'predict':
        predictions = np.asarray(value)
    else:
        probabilities = np.asarray(value)
assert predictions.shape == (10000,) and probabilities.shape == (10000, 10)
assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
assert np.array_equal(predictions, np.argmax(probabilities, axis=1))
args.output.mkdir(parents=True, exist_ok=True)
stem = f'{args.backend}-{args.dtype}-{args.trial}'
np.save(args.output / (stem + '-predictions.npy'), predictions)
result = dict(backend=args.backend, trial=args.trial, input_dtype=args.dtype, python=platform.python_version(),
              xgboost=xgb.__version__, numpy=np.__version__, pandas=pd.__version__,
              sklearn=sklearn.__version__, build_info=build_info, params=params,
              train_shape=x_train.shape, test_shape=x_test.shape,
              source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              fit_seconds=fit, inference_seconds=times,
              accuracy=float(np.mean(predictions == y_test)),
              correct=int(np.count_nonzero(predictions == y_test)),
              log_loss=float(-np.log(np.clip(probabilities[np.arange(len(y_test)), y_test], 1e-15, 1)).mean()))
(args.output / (stem + '.json')).write_text(json.dumps(result, indent=2, default=str) + '\n')
print(json.dumps(result, default=str), flush=True)
