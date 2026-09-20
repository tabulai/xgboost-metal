"""Numeric-bin boundary checks shared with the XGBoost CPU reference."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import numpy as np
import xgboost as xgb


CASES = list(itertools.product([0, 1, 2, 63], [65534, 65535, 65536, 70000, 2147483647],
                               ['raw', 'quantile', 'reference'], [False], [False], [False]))
CASES += list(itertools.product([70003], [65534, 65535, 65536, 70000],
                                ['raw', 'quantile', 'reference'], [False, True], [False, True], [False]))
CASES += list(itertools.product([70003], [65536], ['raw', 'quantile', 'reference'],
                                [False], [False, True], [True]))
CASES += list(itertools.product([65533, 65534, 65535, 65536, 65537], [2147483647],
                                ['raw', 'quantile', 'reference'], [False], [False], [False]))
CASES += list(itertools.product([0, 2, 63], [65536], ['reference-small'],
                                [False], [False, True], [False]))
CASES += list(itertools.product([70003], [65534, 65536], ['raw-csr', 'quantile-csr'],
                                [True], [False, True], [False]))
CASES += list(itertools.product([0], [2, 65536, 2147483647], ['raw', 'quantile'],
                                [False], [False], [False, True], [None, 0.5]))
CASES += list(itertools.product([0], [2, 65536, 2147483647], ['raw', 'quantile'],
                                [False], [False], [True], [0]))
PARAMETERS = [None, 2, 2.0, '2', '+2', ' 2', '2 ', '2\n', '2.0', '2e0', '0x10',
              '', '+', '++2', '+-2', True, False, 0, 1, -1, 65535, 65536,
              2147483647, 2147483648]


def case_name(case):
    return '/'.join(map(str, case))


def parameter_case(value, quantile):
    X = np.arange(16, dtype=np.float32).reshape(8, 2)
    try:
        data = (xgb.QuantileDMatrix(X, label=X[:, 0], max_bin=value) if quantile
                else xgb.DMatrix(X, label=X[:, 0]))
        model = xgb.train(dict(max_bin=value, base_score=0, max_depth=1), data, 1)
        config = json.loads(model.save_config())['learner']['gradient_booster']['tree_train_param']
        return dict(max_bin=config['max_bin'])
    except xgb.core.XGBoostError:
        return dict(error=True)


def run_case(rows, bins, kind, weighted, mixed, vector, base=0):
    count = (70003 if kind == 'reference-small' else
             max(rows, 128) + 17 if kind == 'reference' else rows)
    index = np.arange(count, dtype=np.float32)
    X = np.column_stack([index, (index * 13) % 43]).astype(np.float32)
    if mixed:
        X[:, 1] = np.resize(np.array([0, 3, 65536], np.float32), count)
    X[::23, 1] = np.nan
    label = (index >= max(1, rows - 4096)).astype(np.float32)
    if vector:
        label = np.column_stack([label, (index % 7) / np.float32(7)])
    weights = np.resize(np.array([0, 0.125, 0.5, 1, 3], np.float32), count) if weighted else None
    types = ['q', 'c' if mixed else 'q']
    options = dict(label=label[:rows], weight=None if weights is None else weights[:rows], feature_types=types)
    values = X[:rows]
    if kind.endswith('-csr'):
        from scipy.sparse import csr_matrix
        values = csr_matrix(values)
    if kind.startswith('raw'):
        data = xgb.DMatrix(values, **options)
    elif kind.startswith('quantile'):
        data = xgb.QuantileDMatrix(values, max_bin=bins, **options)
    else:
        reference = xgb.QuantileDMatrix(X, label=label, weight=weights, feature_types=types, max_bin=bins)
        data = xgb.QuantileDMatrix(values, ref=reference, max_bin=bins, **options)
        del reference
    params = dict(max_bin=bins, max_depth=2, seed=11, tree_method='hist')
    if base is not None:
        params['base_score'] = base
    if vector:
        params['multi_strategy'] = 'multi_output_tree'
    model = xgb.train(params, data, 3)
    ptr, cuts = data.get_quantile_cut()
    csr = data.get_data()
    arrays = dict(prediction=model.predict(data, output_margin=True), cut_ptr=ptr, cuts=cuts,
                  csr_ptr=csr.indptr, csr_indices=csr.indices, csr_data=csr.data)
    return dict(model=json.loads(model.save_raw(raw_format='json')),
                config=json.loads(model.save_config()), array_shapes={key:list(value.shape) for key,value in arrays.items()}), arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    build = xgb.build_info()
    if args.cpu_only:
        root = Path(__file__).resolve().parents[1]
        assert Path(xgb.__file__).resolve().is_relative_to(root / 'ref/xgboost')
        assert not build.get('USE_METAL') and not build.get('USE_CUDA')
    cases = CASES
    if args.smoke:
        cases = [(63, 2147483647, 'reference', False, False, False),
                 (70003, 65536, 'raw', False, False, False),
                 (70003, 65536, 'quantile', True, True, True)]
    output, arrays = {}, {}
    for case in cases:
        key = case_name(case)
        try:
            trace, values = run_case(*case)
        except xgb.core.XGBoostError as error:
            trace = dict(error=str(error))
        else:
            arrays.update({key + '/' + name: value for name, value in values.items()})
        output[key] = trace
    build = xgb.build_info()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parameters = {case_name((index, quantile)): parameter_case(value, quantile)
                  for index, value in enumerate(PARAMETERS) for quantile in [False, True]}
    args.output.write_text(json.dumps(dict(build=build,
        library_sha256=hashlib.sha256(Path(build['libxgboost']).read_bytes()).hexdigest(),
        cases=output, parameters=parameters), indent=2) + '\n')
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    print(json.dumps(dict(cases=len(cases), errors=sum('error' in row for row in output.values()))))


if __name__ == '__main__':
    main()
