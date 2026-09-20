"""Public ranking metadata traces, shared by the Metal tests and CPU subprocess."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import numpy as np
import xgboost as xgb

OBJECTIVES = ['rank:ndcg', 'rank:map', 'rank:pairwise']
METHODS = ['topk', 'mean']
CHANGES = ['labels', 'label_gains', 'fractional_labels', 'label_parameter', 'weights', 'clear_weights',
           'groups', 'qid', 'group_count', 'group_count_weighted', 'ranking_parameter',
           'same_parameter', 'unrelated_parameter', 'reset', 'switch_matrix', 'metric_first']
CASES = list(itertools.product(OBJECTIVES, METHODS, CHANGES, [False, True], [False, True]))

def case_name(case):
    return '/'.join(map(str, case))

def run_case(objective, method, change, evaluate, quantile):
    rng = np.random.default_rng(217)
    X = rng.integers(-8, 9, (128, 4)).astype(np.float32)
    X[::19, 2] = np.nan
    base_labels = rng.integers(0, 4, len(X)).astype(np.float32)
    weights = np.array([0.5, 1, 2, 3], np.float32)
    labels = (base_labels > 1).astype(np.float32) if objective == 'rank:map' else base_labels
    predictions = {}
    name = case_name((objective, method, change, evaluate, quantile))
    weighted = change != 'group_count'
    matrix_type = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    options = {'max_bin': 32} if quantile else {}
    data = matrix_type(X, label=labels, group=[32] * 4,
                       weight=weights if weighted else None, **options)
    other = matrix_type(X, label=labels, group=[16, 48, 48, 16],
                        weight=weights if weighted else None, **options)
    params = dict(objective=objective, tree_method='hist', max_bin=32, max_depth=3,
                  lambdarank_pair_method=method, lambdarank_num_pair_per_sample=3,
                  base_score=0, seed=41,
                  eval_metric=['ndcg@3', 'map@3', 'pre@3'] if objective == 'rank:map'
                  else ['ndcg@3', 'ndcg@7'])
    model = xgb.Booster(params, [data])
    trace = []
    if change == 'metric_first':
        model = xgb.train(params, other, 1)
        trace.append(dict(event='pre_metric', metric=model.eval_set([(data, 'query')], 0)))
        data.set_label(np.roll(labels, 7))
        data.set_group([16, 48, 48, 16])
        data.set_weight(weights * 1.75)
    for iteration in range(4):
        active = data
        if iteration == 1:
            if change in ['labels', 'label_parameter', 'reset', 'switch_matrix']:
                data.set_label(np.roll(labels, 7))
            if change == 'label_gains':
                data.set_label(1 - labels if objective == 'rank:map' else labels * 2)
            if change == 'fractional_labels':
                data.set_label(np.where(labels > 0, np.float32(0.5), np.float32(0)))
            if change in ['weights', 'ranking_parameter', 'same_parameter', 'unrelated_parameter', 'reset']:
                data.set_weight(weights * 1.75)
            if change == 'clear_weights':
                data.set_weight(np.empty(0, np.float32))
            if change == 'groups':
                data.set_group([16, 48, 48, 16])
            if change == 'qid':
                data.set_info(qid=np.repeat(np.arange(4), [16, 48, 48, 16]))
            if change in ['group_count', 'group_count_weighted']:
                data.set_group([16] * 8)
                if weighted:
                    data.set_weight(np.tile(weights, 2))
            if change in ['ranking_parameter', 'label_parameter']:
                model.set_param('lambdarank_normalization', False)
            if change == 'same_parameter':
                model.set_param('lambdarank_bias_norm', '1.0')
            if change == 'unrelated_parameter':
                model.set_param('eta', 0.1)
            if change == 'reset':
                model.reset()
            if change == 'switch_matrix':
                active = other
        if iteration == 2 and change == 'group_count_weighted':
            data.set_group([32] * 4)
            data.set_weight(weights)
        row = dict(iteration=iteration)
        try:
            model.update(active, iteration)
        except xgb.core.XGBoostError as error:
            row['update_error'] = str(error)
        else:
            row['model'] = json.loads(model.save_raw(raw_format='json'))
            key = name + '/' + str(iteration)
            predictions[key] = model.predict(active, output_margin=True)
            row['margin_sha256'] = hashlib.sha256(predictions[key].tobytes()).hexdigest()
        if evaluate:
            try:
                row['metric'] = model.eval_set([(data, 'query')], iteration)
            except xgb.core.XGBoostError as error:
                row['metric_error'] = str(error)
        state = json.loads(model.save_config())['learner']
        row['rng_state'] = state['generic_param']['rng_state']
        row['metric_config'] = state['metrics']
        row['objective_config'] = state['objective']
        trace.append(row)

    return trace, predictions


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
    cases, predictions = {}, {}
    for case in CASES:
        trace, arrays = run_case(*case)
        cases[case_name(case)] = trace
        predictions.update(arrays)
    report = dict(build=build, version=xgb.__version__, cases=cases,
                  library_sha256=hashlib.sha256(Path(build['libxgboost']).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.output.with_suffix('.npz'), **predictions)
    print(json.dumps(dict(cases=len(cases), predictions=len(predictions),
                          errors=sum('update_error' in step for trace in cases.values() for step in trace))))


if __name__ == '__main__':
    main()
