"""Persistent scratch storage must not retain numerical state between trees."""
import ctypes
import json
import pickle

import numpy as np
import pytest
import xgboost as xgb
from xgboost.objective import TreeObjective


def profile(model):
    output = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(output)))
    return json.loads(output.value)


MODES = [
    ('depthwise', False, 32, {}),
    ('lossguide', True, 97, {}),
    ('depthwise', True, 513, {'metal_fused_partition': 0}),
    ('lossguide', False, 32, {'metal_partition_scan': 0, 'metal_partition_aggregation': 0}),
    ('depthwise', True, 513, {'metal_partition_scan': 0, 'metal_partition_aggregation': 1}),
    ('lossguide', True, 513, {'metal_coalesce_encoders': 1, 'metal_compute_fill': 1}),
]


def matrices(outputs, quantile, categorical, bins):
    rng = np.random.default_rng(7183)
    result = []
    for rows in [513, 65, 1027, 0]:
        X = rng.normal(size=(rows, 7)).astype(np.float32)
        if categorical:
            X[:, 0] = rng.integers(0, 97, rows)
        y = (X[:, 1] - X[:, 2] * X[:, 3] + np.sin(X[:, 4])).astype(np.float32)
        if outputs > 1:
            y = np.column_stack([y + np.float32(i / 4) * X[:, i] for i in range(outputs)])
        X[rng.random(X.shape) < 0.11] = np.nan
        weight = rng.uniform(0.1, 2, rows).astype(np.float32)
        cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
        result.append(cls(X, label=y, weight=weight,
                          feature_types=['c' if categorical else 'q', *['q'] * 6],
                          **({'max_bin': bins} if quantile else {})))
    return result


def train_trace(outputs, quantile, mode, persistent):
    policy, categorical, bins, tuning = mode
    data = matrices(outputs, quantile, categorical, bins)
    params = dict(max_bin=bins, grow_policy=policy, max_depth=4, max_leaves=13,
                  max_cached_hist_node=1, subsample=0.73, colsample_bynode=0.8,
                  base_score=0, seed=37, multi_strategy='multi_output_tree' if outputs > 1 else 'one_output_per_tree',
                  metal_tree_persistent_workspace=persistent, **tuning)
    model = xgb.Booster(params, data)
    trace = []
    for iteration, index in enumerate([0, 1, 2, 3, 0, 2, 1]):
        if iteration == 4:
            model.reset()
        if iteration == 5:
            model.load_config(model.save_config())
            model.set_param('metal_tree_persistent_workspace', persistent)
        model.update(data[index], iteration)
        snapshot = dict(model=bytes(model.save_raw()),
                        rng=json.loads(model.save_config())['learner']['generic_param']['rng_state'],
                        prediction=model.predict(data[index], output_margin=True),
                        leaf=model.predict(data[index], pred_leaf=True))
        stats = profile(model)
        snapshot['commands'] = stats['command_buffers']
        assert stats['tuning']['tree_persistent_workspace'] == persistent
        trace.append(snapshot)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(data[0]), model.predict(data[0]))
    return trace


@pytest.mark.parametrize('outputs', [1, 3])
@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('mode', MODES, ids=lambda mode: f'{mode[0]}-{mode[1]}-{mode[2]}-{mode[3]}')
def test_scratch_pool_across_matrices_and_resets(outputs, quantile, mode):
    expected = train_trace(outputs, quantile, mode, 0)
    actual = train_trace(outputs, quantile, mode, 1)
    for own, wanted in zip(actual, expected):
        for field in ['model', 'rng', 'commands']:
            assert own[field] == wanted[field], field
        for field in ['prediction', 'leaf']:
            np.testing.assert_array_equal(own[field].view(np.uint32), wanted[field].view(np.uint32))


@pytest.mark.parametrize('workspace,persistent', [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_reset_releases_persistent_scratch(workspace, persistent):
    data = xgb.DMatrix(np.zeros((129, 3), np.float32), label=np.zeros(129, np.float32))
    model = xgb.Booster(dict(max_depth=1, base_score=0, metal_tree_workspace=workspace,
                             metal_tree_persistent_workspace=persistent), [data])
    allocations = []
    for iteration in range(3):
        if iteration == 2:
            model.reset()
        model.update(data, iteration)
        tuning = profile(model)['tuning']
        assert tuning['tree_persistent_workspace'] == bool(workspace and persistent)
        allocations.append(tuning['tree_buffer_allocations'])
    assert allocations[0] > 0 and allocations[2] == allocations[0]
    assert allocations[1] == (0 if workspace and persistent else allocations[0])


class ChangingSplitTargets(TreeObjective):
    def __call__(self, iteration, prediction, data):
        return prediction - data.get_label().reshape(prediction.shape), np.ones_like(prediction)

    def split_grad(self, iteration, gradient, hessian):
        targets = [3, 1, 2][iteration % 3]
        return gradient[:, :targets], hessian[:, :targets]


@pytest.mark.parametrize('sampling', ['uniform', 'gradient_based'])
@pytest.mark.parametrize('parallel', [1, 3])
def test_reduced_gradient_shapes_reuse_only_storage(sampling, parallel):
    data = matrices(3, False, True, 32)[0]
    params = dict(multi_strategy='multi_output_tree', max_depth=3, max_bin=32,
                  num_parallel_tree=parallel, subsample=0.71, sampling_method=sampling,
                  base_score=0, seed=91)
    models = [xgb.train({**params, 'metal_tree_persistent_workspace': persistent}, data, 6,
                        obj=ChangingSplitTargets()) for persistent in [0, 1]]
    assert models[0].save_raw() == models[1].save_raw()
    np.testing.assert_array_equal(models[0].predict(data).view(np.uint32), models[1].predict(data).view(np.uint32))
    configs = [json.loads(model.save_config())['learner']['generic_param'] for model in models]
    assert configs[0]['rng_state'] == configs[1]['rng_state']


@pytest.mark.parametrize('outputs', [1, 3])
@pytest.mark.parametrize('categorical', [False, True])
def test_request_and_partition_changes_reuse_only_storage(outputs, categorical):
    index = np.arange(70003, dtype=np.float32)
    X = np.column_stack([index, index % 7, index % 19]).astype(np.float32)
    if categorical:
        X[:, 1] = np.resize(np.array([0, 3, 65536], np.float32), len(X))
    y = (index >= 65536).astype(np.float32)
    if outputs > 1:
        y = np.column_stack([y, (index % 7) / np.float32(7), (index % 13) / np.float32(13)])
    X[::17, 2] = np.nan
    traces = []
    changes = [dict(max_bin=32, max_depth=2), dict(max_bin=65536, max_depth=3, metal_fused_partition=0),
               dict(max_bin=8, metal_partition_scan=0), dict(max_bin=2147483647, max_depth=1, metal_partition_scan=1)]
    for persistent in [0, 1]:
        data = xgb.DMatrix(X, label=y, feature_types=['q', 'c' if categorical else 'q', 'q'])
        model = xgb.Booster(dict(base_score=0, seed=29, multi_strategy='multi_output_tree' if outputs > 1 else 'one_output_per_tree',
                                 metal_tree_persistent_workspace=persistent), [data])
        trace = []
        for iteration, change in enumerate(changes):
            model.set_param(change)
            model.update(data, iteration)
            trace.append((bytes(model.save_raw()), model.predict(data).view(np.uint32),
                          json.loads(model.save_config())['learner']['generic_param']['rng_state']))
        traces.append(trace)
    for own, wanted in zip(*traces):
        assert own[0] == wanted[0] and own[2] == wanted[2]
        np.testing.assert_array_equal(own[1], wanted[1])


@pytest.mark.parametrize('outputs', [1, 3])
def test_failed_tree_setup_then_retry(outputs):
    traces = []
    for persistent in [0, 1]:
        data = matrices(outputs, False, True, 97)[0]
        params = dict(base_score=0, max_depth=3, max_bin=97, subsample=0.73, seed=41,
                      multi_strategy='multi_output_tree' if outputs > 1 else 'one_output_per_tree',
                      metal_tree_persistent_workspace=persistent)
        model = xgb.Booster(params, [data])
        model.update(data, 0)
        model.set_param('interaction_constraints', '[[0,9]]')
        with pytest.raises(xgb.core.XGBoostError, match='Interaction constraint feature'):
            model.update(data, 1)
        model.set_param('interaction_constraints', '[[0,1],[2,3],[4,5,6]]')
        model.update(data, 1)
        traces.append((bytes(model.save_raw()), model.predict(data).view(np.uint32),
                       json.loads(model.save_config())['learner']['generic_param']['rng_state']))
    assert traces[0][0] == traces[1][0] and traces[0][2] == traces[1][2]
    np.testing.assert_array_equal(traces[0][1], traces[1][1])
