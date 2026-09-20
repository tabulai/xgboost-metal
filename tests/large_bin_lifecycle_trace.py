"""Public bin-request changes, failed updates, and configuration reloads."""
from pathlib import Path
import argparse
import copy
import itertools
import json
import numpy as np
import xgboost as xgb

CASES = list(itertools.product(['raw', 'quantile', 'reference'], [False, True], [False, True]))
REFERENCE_CASES = list(itertools.product([0, 2], ['raw', 'trained', 'quantile']))


def case_name(case):
    return '/'.join(map(str, case))


def reference_case(rows, kind):
    X = np.arange(128, dtype=np.float32).reshape(64, 2)
    reference = (xgb.QuantileDMatrix(X, max_bin=65535) if kind == 'quantile'
                 else xgb.DMatrix(X, label=X[:, 0]))
    if kind == 'trained':
        xgb.train(dict(max_bin=65535, base_score=0, max_depth=1), reference, 1)
    events = []
    for bins in [65535, None, 65534, 65536, 2147483647, 65535]:
        try:
            query = xgb.QuantileDMatrix(X[:rows], ref=reference, max_bin=bins)
        except xgb.core.XGBoostError:
            events.append(dict(error=True))
        else:
            ptr, cuts = query.get_quantile_cut()
            csr = query.get_data()
            events.append(dict(cut_ptr=ptr.tolist(), cuts=cuts.view(np.uint32).tolist(),
                               csr_ptr=csr.indptr.tolist(), csr_indices=csr.indices.tolist(),
                               csr_data=csr.data.view(np.uint32).tolist()))
    return events


def run_case(kind, mixed, load_config):
    rows = 63 if kind == 'reference' else 513
    count = 70003 if kind == 'reference' else rows
    index = np.arange(count, dtype=np.float32)
    X = np.column_stack([index, index % 17]).astype(np.float32)
    if mixed:
        X[:, 1] = np.resize(np.array([0, 3, 65536], np.float32), count)
    X[::13, 1] = np.nan
    y = (index >= rows // 2).astype(np.float32)
    bins = 31 if kind == 'raw' else 65535
    types = ['q', 'c' if mixed else 'q']
    if kind == 'raw':
        data = xgb.DMatrix(X, label=y, feature_types=types)
    elif kind == 'quantile':
        data = xgb.QuantileDMatrix(X, label=y, feature_types=types, max_bin=bins)
    else:
        reference = xgb.QuantileDMatrix(X, label=y, feature_types=types, max_bin=bins)
        data = xgb.QuantileDMatrix(X[:rows], label=y[:rows], ref=reference, max_bin=bins, feature_types=types)
        del reference
    params = dict(max_bin=bins, max_depth=2, base_score=0, seed=14)
    model = xgb.Booster(params, [data])
    requests = [bins, 65536, 2147483647, bins, str(bins), str(bins) + '.0', bins]
    config = json.loads(model.save_config())
    trace, arrays = [], {}
    for iteration, request in enumerate(requests):
        step = dict(iteration=iteration, request=request)
        try:
            if load_config:
                changed = copy.deepcopy(config)
                changed['learner']['gradient_booster']['tree_train_param']['max_bin'] = str(request)
                model.load_config(json.dumps(changed))
            else:
                model.set_param('max_bin', request)
            model.update(data, iteration)
        except xgb.core.XGBoostError:
            step['error'] = True
        else:
            step['model'] = json.loads(model.save_raw(raw_format='json'))
            config = json.loads(model.save_config())
            step['max_bin'] = config['learner']['gradient_booster']['tree_train_param']['max_bin']
            step['rng_state'] = config['learner']['generic_param']['rng_state']
            arrays[str(iteration) + '/prediction'] = model.predict(data, output_margin=True)
        ptr, cuts = data.get_quantile_cut()
        csr = data.get_data()
        for field, array in dict(cut_ptr=ptr, cuts=cuts, csr_ptr=csr.indptr,
                                  csr_indices=csr.indices, csr_data=csr.data).items():
            arrays[str(iteration) + '/' + field] = array
        trace.append(step)
    return trace, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    build = xgb.build_info()
    if args.cpu_only:
        root = Path(__file__).resolve().parents[1]
        assert Path(xgb.__file__).resolve().is_relative_to(root / 'ref/xgboost')
        assert not build.get('USE_METAL') and not build.get('USE_CUDA')
    report, arrays = {}, {}
    for case in CASES:
        name = case_name(case)
        report[name], values = run_case(*case)
        arrays.update({name + '/' + key: value for key, value in values.items()})
    references = {case_name(case): reference_case(*case) for case in REFERENCE_CASES}
    args.output.write_text(json.dumps(dict(cases=report, reference=references), indent=2) + '\n')
    np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    print(json.dumps(dict(cases=len(report))))


if __name__ == '__main__':
    main()
