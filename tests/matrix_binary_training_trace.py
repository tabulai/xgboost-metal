"""Train and continue from exchanged binary matrices with the XGBoost CPU reference."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb

CASES = ['dense', 'weighted', 'sparse', 'uint32', 'string', 'vector', 'margins', 'queries']


def make_training(case):
    index = np.arange(64)
    values = ((index[:, None] >> np.arange(4)) & 1).astype(np.float32)
    labels = values[:, 0] * 4 - values[:, 1] * 2 + values[:, 2]
    params = dict(tree_method='hist', max_depth=2, max_bin=8, eta=0.5,
                  reg_lambda=0, min_child_weight=0, base_score=0, nthread=1, seed=19)
    meta = dict(label=labels)
    query_meta = {}
    custom = None
    if case == 'weighted':
        meta['weight'] = 1 + values[:, 3]
    elif case == 'sparse':
        values[:, 3] = np.where(index % 4 == 0, np.nan, values[:, 3])
        values = sparse.csr_matrix(values)
    elif case in ['uint32', 'string']:
        names = np.array([2**32 - 1, 0, 2**31, 1], np.uint32) if case == 'uint32' else ['茶', '', 'z', 'é']
        values = pd.DataFrame({'kind': pd.Categorical.from_codes(index % 4, names),
                               'numeric': values[:, 2]})
        params['max_cat_to_onehot'] = 1
        meta['enable_categorical'] = query_meta['enable_categorical'] = True
    elif case == 'vector':
        meta['label'] = np.column_stack([labels, -labels, labels / 2])
        params['multi_strategy'] = 'multi_output_tree'
    elif case == 'margins':
        meta['base_margin'] = labels / 8
        query_meta['base_margin'] = meta['base_margin'][::-1].copy()
    elif case == 'queries':
        meta.update(group=[16, 0, 16, 32], weight=[1, 0, 2, 1])
        params['objective'] = 'rank:pairwise'
        # Fixed square-loss derivatives isolate query metadata and weighted
        # sketch preservation from the separate ranking numerical work.
        custom = lambda prediction, data: (prediction - data.get_label(), np.ones_like(prediction))
    query = values.iloc[::-1].copy() if isinstance(values, pd.DataFrame) else values[::-1].copy()
    if isinstance(query, pd.DataFrame):
        query['kind'] = query['kind'].cat.reorder_categories(query['kind'].cat.categories[::-1])
    return xgb.DMatrix(values, **meta), xgb.DMatrix(query, **query_meta), params, custom


def run_case(case, destination, matrix=None):
    destination.mkdir()
    original, query, params, custom = make_training(case)
    matrix = original if matrix is None else matrix
    matrix.save_binary(destination / 'before.buffer', silent=True)
    query.save_binary(destination / 'query.buffer', silent=True)
    matrix = xgb.DMatrix(destination / 'before.buffer', silent=True)
    query = xgb.DMatrix(destination / 'query.buffer', silent=True)
    report = dict(stages=[])
    model = None
    for stage in range(2):
        model = xgb.train(params, matrix, 2, xgb_model=model, obj=custom)
        model.save_model(destination / f'model-{stage}.json')
        model.save_model(destination / f'model-{stage}.ubj')
        arrays = {}
        for name, data in [('train', matrix), ('query', query)]:
            options = [('prediction', {}), ('margin', {'output_margin': True})]
            if case != 'vector':
                options.append(('leaf', {'pred_leaf': True}))
            for kind, config in options:
                values = model.predict(data, **config)
                arrays[name + '-' + kind] = dict(shape=list(values.shape), bits=values.view(np.uint32).reshape(-1).tolist())
        pointer, cuts = matrix.get_quantile_cut()
        arrays['cuts'] = dict(indptr=pointer.tolist(), bits=cuts.view(np.uint32).tolist())
        report['stages'].append(dict(model=json.loads(model.save_raw(raw_format='json')), arrays=arrays))
        filename = destination / f'after-{stage}.buffer'
        matrix.save_binary(filename, silent=True)
        assert filename.read_bytes() == (destination / 'before.buffer').read_bytes()
        matrix = xgb.DMatrix(filename, silent=True)
        model = xgb.Booster(model_file=destination / f'model-{stage}.ubj')
    report['before_sha256'] = hashlib.sha256((destination / 'before.buffer').read_bytes()).hexdigest()
    report['query_sha256'] = hashlib.sha256((destination / 'query.buffer').read_bytes()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert Path(xgb.__file__).resolve().is_relative_to(root / 'ref/xgboost')
    library = Path(xgb.build_info()['libxgboost'])
    assert hashlib.sha256(library.read_bytes()).hexdigest() == '061d0e5e88a7e7282d2eab9d0b4c53a12f73f2c83a999785b20c61e82c30bfc7'
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    report = {}
    for case in ([args.case] if args.case else CASES):
        matrix = xgb.DMatrix(args.input, silent=True) if args.input else None
        report[case] = run_case(case, args.output / case, matrix)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
