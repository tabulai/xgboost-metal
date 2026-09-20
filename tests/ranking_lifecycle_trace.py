"""Default metrics, unbiased state and bounded metric ownership via public APIs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import itertools
import json
import numpy as np
import xgboost as xgb
from ranking_metadata_trace import OBJECTIVES, METHODS, case_name

DEFAULT_CASES = list(itertools.product(OBJECTIVES, METHODS))
METRIC_CONFIG_CASES = list(itertools.product(OBJECTIVES, METHODS))
UNBIASED_CASES = list(itertools.product(OBJECTIVES, METHODS,
    ['load_config', 'reset', 'metadata', 'disable_enable', 'objective_switch',
     'objective_switch_rank', 'objective_switch_reapply', 'failed_initial',
     'empty_bias_config']))
# The pinned CPU's unbiased MAP implementation aborts on an empty query.
UNBIASED_CASES += list(itertools.product(['rank:ndcg', 'rank:pairwise'], METHODS, ['empty_data']))
OWNERSHIP_CASES = list(itertools.product(['fifo', 'thread', 'expired'], [False, True]))

def default_trace(objective, method):
    rng = np.random.default_rng(801)
    X = rng.integers(0, 11, (96, 4)).astype(np.float32)
    y = rng.integers(0, 4, len(X)).astype(np.float32)
    if objective == 'rank:map':
        y = (y > 1).astype(np.float32)
    data = xgb.DMatrix(X, label=y, group=[24] * 4, weight=[1, 2, 3, 4])
    model = xgb.Booster(dict(objective=objective, lambdarank_pair_method=method, max_bin=32, max_depth=2,
                             lambdarank_num_pair_per_sample=3, base_score=0), [data])
    trace = []

    def save(event, evaluate=False):
        row = dict(event=event, config=json.loads(model.save_config()),
                   model=json.loads(model.save_raw(raw_format='json')),
                   margin=model.predict(data, output_margin=True).tolist())
        if evaluate:
            row['evaluation'] = model.eval_set([(data, 'q')])
            row['config_after'] = json.loads(model.save_config())
        trace.append(row)

    model.update(data, 0)
    save('before-evaluation')
    save('first-evaluation', True)
    model.set_param('lambdarank_num_pair_per_sample', 7)
    data.set_label(np.roll(y, 11))
    data.set_group([12, 36, 36, 12])
    model.update(data, 1)
    save('new-pairs-old-metric', True)
    model.set_param('eval_metric', 'map@7' if objective == 'rank:map' else 'ndcg@7')
    save('add-explicit-metric', True)
    model.set_param('disable_default_eval_metric', 1)
    save('disable-after-activation', True)
    data.set_label(y)
    data.set_group([32, 16, 32, 16])
    model.load_config(model.save_config())
    model.update(data, 2)
    save('load-config-renews-metrics', True)
    return trace


def metric_config_trace(objective, method):
    rng = np.random.default_rng(846)
    data = xgb.DMatrix(rng.integers(0, 11, (96, 4)).astype(np.float32),
                       label=rng.integers(0, 2, 96), group=[24] * 4)
    model = xgb.Booster(dict(objective=objective, lambdarank_pair_method=method,
                             lambdarank_num_pair_per_sample=3, lambdarank_normalization=False,
                             ndcg_exp_gain=False, max_bin=32, max_depth=2, base_score=0), [data])
    model.update(data, 0)
    trace = []

    def save(event):
        trace.append(dict(event=event, evaluation=model.eval(data), config=json.loads(model.save_config())))

    save('default-configuration')
    model.set_param('eval_metric', 'ndcg@7')
    save('explicit-metric-default-gain')
    model.set_param([('ndcg_exp_gain', False), ('eval_metric', 'ndcg@5')])
    save('gain-before-new-metric')
    model.set_param([('eval_metric', 'ndcg@9'), ('ndcg_exp_gain', True)])
    save('gain-after-new-metric')
    model.set_param('ndcg_exp_gain', False)
    save('change-all-ndcg-gains')
    model.set_param('objective', 'reg:squarederror')
    save('intervening-objective')
    model.set_param('objective', objective)
    save('metric-config-survives-objective-switch')
    model.set_param('eval_metric', 'ndcg@11')
    save('new-metric-after-objective-switch')
    model.load_config(model.save_config())
    save('load-metric-configurations')
    return trace


def unbiased_trace(objective, method, change):
    rng = np.random.default_rng(931)
    X = rng.integers(0, 11, (128, 4)).astype(np.float32)
    labels = rng.integers(0, 2, len(X)).astype(np.float32)
    weights = np.array([0.5, 1, 2, 3], np.float32)
    data = xgb.DMatrix(X, label=labels, group=[32] * 4, weight=weights)
    if change == 'empty_data':
        data = xgb.DMatrix(X[:0], label=labels[:0])
    params = dict(objective=objective, lambdarank_pair_method=method, lambdarank_num_pair_per_sample=3,
                  lambdarank_unbiased=True, lambdarank_bias_norm=0.5, max_depth=2, max_bin=32,
                  base_score=0, seed=17)
    model = xgb.Booster(params, [data])
    trace = []

    def attempt(event, iteration):
        row = dict(event=event)
        try:
            model.update(data, iteration)
        except xgb.core.XGBoostError as error:
            row['error'] = str(error)
        row.update(config=json.loads(model.save_config()), model=json.loads(model.save_raw(raw_format='json')),
                   margin=model.predict(data, output_margin=True).tolist())
        trace.append(row)

    if change == 'failed_initial':
        attempt('failed-start', 3)
        attempt('retry-start', 0)
        attempt('next', 1)
        return trace
    attempt('initial', 0)
    if change == 'load_config':
        model.load_config(model.save_config())
    elif change == 'empty_bias_config':
        config = json.loads(model.save_config())
        config['learner']['objective']['ti+'] = []
        config['learner']['objective']['tj-'] = []
        model.load_config(json.dumps(config))
    elif change == 'reset':
        model.reset()
    elif change == 'metadata':
        data.set_label(1 - labels)
        data.set_weight(weights * 1.25)
        data.set_group([16, 48, 48, 16])
    elif change == 'disable_enable':
        model.set_param('lambdarank_unbiased', False)
        attempt('disabled', 1)
        model.set_param('lambdarank_unbiased', True)
        attempt('enabled', 2)
        return trace
    elif change.startswith('objective_switch'):
        intervening = ('rank:map' if objective != 'rank:map' else 'rank:ndcg') \
            if change == 'objective_switch_rank' else 'reg:squarederror'
        model.set_param('objective', intervening)
        model.save_config()  # Force construction of the intervening objective.
        if change == 'objective_switch_reapply':
            model.set_param([(key, value) for key, value in params.items() if key != 'objective'] +
                            [('objective', objective)])
        else:
            model.set_param('objective', objective)
    attempt('after-change', 1)
    if change in ['reset', 'empty_bias_config', 'empty_data'] or change.startswith('objective_switch'):
        attempt('restart', 0)
    else:
        attempt('next', 2)
    return trace


def ownership_trace(kind, quantile):
    X = np.arange(16, dtype=np.float32).reshape(-1, 1)
    def matrix():
        options = dict(max_bin=8) if quantile else {}
        cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
        return cls(X, label=np.full(16, 3, np.float32), group=[8, 8],
                   base_margin=np.arange(16, dtype=np.float32), **options)
    first = matrix()
    model = xgb.train(dict(objective='rank:ndcg', eval_metric='ndcg@3', max_bin=8, base_score=0), first, 0)
    trace = [dict(event='initial', metric=model.eval(first))]
    first.set_label(np.ones(16, np.float32))
    if kind == 'thread':
        with ThreadPoolExecutor(max_workers=1) as pool:
            trace.append(dict(event='worker-first', metric=pool.submit(model.eval, first).result()))
            trace.append(dict(event='main-retained', metric=model.eval(first)))
            first.set_label(np.zeros(16, np.float32))
            trace.append(dict(event='worker-retained', metric=pool.submit(model.eval, first).result()))
            trace.append(dict(event='main-zero', metric=model.eval(first)))
        model.reset()
        trace.append(dict(event='after-reset', metric=model.eval(first)))
    elif kind == 'expired':
        for index in range(70):
            temporary = matrix()
            model.eval(temporary)
            del temporary
        trace.append(dict(event='surviving-entry', metric=model.eval(first)))
    else:
        trace.append(dict(event='retained-labels', metric=model.eval(first)))
        model.set_param('eval_metric', 'ndcg@7')
        trace.append(dict(event='new-metric', metric=model.eval(first)))
        matrices = [first]
        for index in range(1, 31):
            matrices.append(matrix())
            model.eval(matrices[-1])
        trace.append(dict(event='before-cap', metric=model.eval(first)))
        matrices.append(matrix())
        model.eval(matrices[-1])
        trace.append(dict(event='evicted-first', metric=model.eval(first)))
        matrices[30].set_label(np.ones(16, np.float32))
        trace.append(dict(event='retained-later', metric=model.eval(matrices[30])))
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if args.cpu_only:
        root = Path(__file__).resolve().parents[1]
        assert Path(xgb.__file__).resolve().is_relative_to(root / 'ref/xgboost')
        assert not xgb.build_info().get('USE_METAL')
    cases = {}
    for prefix, choices, function in [('default', DEFAULT_CASES, default_trace),
                                      ('metric-config', METRIC_CONFIG_CASES, metric_config_trace),
                                      ('unbiased', UNBIASED_CASES, unbiased_trace),
                                      ('ownership', OWNERSHIP_CASES, ownership_trace)]:
        for case in choices:
            cases[prefix + '/' + case_name(case)] = function(*case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(build=xgb.build_info(), cases=cases), indent=2) + '\n')
    print(json.dumps(dict(cases=len(cases), errors=sum('error' in row for trace in cases.values() for row in trace))))


if __name__ == '__main__':
    main()
