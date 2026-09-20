"""Candidate tiling must preserve the serial feature's complete split ordering."""
import ctypes
import json

import numpy as np
import pytest
import xgboost as xgb
from xgboost.objective import TreeObjective


def profile(model):
    value = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(value)))
    return json.loads(value.value)


def numeric_data(bins, outputs):
    rng = np.random.default_rng(8729)
    rows = max(513, bins + 7)
    X = np.column_stack([rng.permutation(np.arange(rows) % bins) for _ in range(5)]).astype(np.float32)
    X[:, 1] = X[:, 0]  # Equal feature gains retain the earliest feature.
    y = np.column_stack([(X[:, 0] % (11 + target) - X[:, 2] / np.float32(bins)) / 8
                         + X[:, 3] / np.float32(bins * (target + 2))
                         for target in range(outputs)]).astype(np.float32)
    X[::11, 2] = np.nan
    X[::17, 3] = np.nan
    return X, y


def compare(X, y, params, tile, rounds=2, feature_types=None, quantile=False, objective=None, weight=None):
    results = []
    for selected in [0, tile]:
        cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
        data = cls(X, label=y, weight=weight, feature_types=feature_types,
                   **({'max_bin': params['max_bin']} if quantile else {}))
        options = dict(base_score=0, seed=117, multi_strategy='multi_output_tree', **params)
        if selected is not None:
            options['metal_vector_split_bin_tile'] = selected
        model = xgb.train(options, data, rounds, obj=objective)
        margin = model.predict(data, output_margin=True)
        leaf = model.predict(data, pred_leaf=True)
        results.append((bytes(model.save_raw()), margin.view(np.uint32), leaf.view(np.uint32),
                        json.loads(model.save_config())['learner']['generic_param']['rng_state'], profile(model)))
    actual, expected = results[1], results[0]
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])
    assert actual[3] == expected[3]
    assert actual[4]['command_buffers'] == expected[4]['command_buffers']
    assert expected[4]['tuning']['vector_split_bin_tiles'] == 1
    return actual[4]


BOUNDARIES = [(32, 2), (32, 31), (32, 32), (32, 33), (32, 65),
              (96, 95), (96, 96), (96, 97), (96, 193),
              (256, 255), (256, 256), (256, 257), (256, 513),
              (1024, 1023), (1024, 1024), (1024, 1025), (1024, 2049), (16384, 16385)]


@pytest.mark.parametrize('tile,bins', BOUNDARIES)
@pytest.mark.parametrize('outputs', [2, 5])
def test_numeric_tile_boundaries(tile, bins, outputs):
    X, y = numeric_data(bins, outputs)
    stats = compare(X, y, dict(max_bin=bins + 1, max_depth=3, reg_alpha=0.17,
                               max_delta_step=0.7, max_cached_hist_node=1), tile)
    assert stats['tuning']['vector_split_bin_tiles'] == (bins + tile - 1) // tile


@pytest.mark.parametrize('bins', [2, 32, 64, 255, 256, 257, 1024, 2049])
def test_automatic_tile_selection(bins):
    X, y = numeric_data(bins, 2)
    stats = compare(X, y, dict(max_bin=bins + 1, max_depth=3), None)
    assert stats['tuning']['vector_split_bin_tiles'] == (bins + 255) // 256
    assert stats['tuning']['vector_split_bin_tile'] == (256 if bins > 256 else 0)


@pytest.mark.parametrize('threshold,tiles', [(64, 2), (1025, 4)])
def test_partition_categories_bound_candidate_tiles(threshold, tiles):
    X, y = numeric_data(257, 2)
    X[:, 2] = np.resize(np.asarray([0, 3, 70003], np.float32), len(X))
    stats = compare(X, y, dict(max_bin=258, max_depth=3, max_cat_to_onehot=2,
                               max_cat_threshold=threshold), None,
                    feature_types=['q', 'q', 'c', 'q', 'q'])
    assert stats['tuning']['vector_split_bin_tiles'] == tiles


@pytest.mark.parametrize('quantile', [False, True])
def test_tile_changes_with_reused_workspace(quantile):
    counts = [257, 1025, 31, 513, 1025, 257]
    choices = [None, 32, 0, 16384, 96, 256]
    traces = []
    for tiled in [False, True]:
        cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
        data = [cls(X, label=y, **({'max_bin': 2050} if quantile else {}))
                for X, y in [numeric_data(count, 3) for count in counts]]
        params = dict(base_score=0, seed=19, multi_strategy='multi_output_tree',
                      max_bin=2050, max_depth=3, subsample=0.71, sampling_method='gradient_based')
        if not tiled:
            params['metal_vector_split_bin_tile'] = 0
        model = xgb.Booster(params, data)
        trace = []
        for iteration, (matrix, tile, count) in enumerate(zip(data, choices, counts)):
            if iteration == 4:
                model.reset()
            if iteration == 5:
                model.load_config(model.save_config())
            if tiled and tile is not None:
                model.set_param('metal_vector_split_bin_tile', tile)
            model.update(matrix, iteration)
            margin = model.predict(matrix, output_margin=True).view(np.uint32)
            leaf = model.predict(matrix, pred_leaf=True).view(np.uint32)
            stats = profile(model)
            effective = (256 if tile is None else tile) if tiled else 0
            expected = max(1, (count + effective - 1) // effective) if effective else 1
            assert stats['tuning']['vector_split_bin_tiles'] == expected
            trace.append((bytes(model.save_raw()), margin, leaf,
                          json.loads(model.save_config())['learner']['generic_param']['rng_state'],
                          stats['command_buffers']))
        traces.append(trace)
    for own, wanted in zip(*traces):
        assert own[0] == wanted[0] and own[3:] == wanted[3:]
        np.testing.assert_array_equal(own[1], wanted[1])
        np.testing.assert_array_equal(own[2], wanted[2])


@pytest.mark.parametrize('tile', [32, 96, 256])
@pytest.mark.parametrize('onehot', [False, True])
@pytest.mark.parametrize('policy', ['depthwise', 'lossguide'])
def test_categorical_candidate_order(tile, onehot, policy):
    rng = np.random.default_rng(533)
    X, y = numeric_data(1025, 3)
    X[:, 0] = rng.integers(0, 257, len(X))
    X[:, 1] = X[:, 0]
    X[::13, 0] = np.nan
    params = dict(max_bin=1026, max_depth=3 if policy == 'depthwise' else 0,
                  grow_policy=policy, max_leaves=9, max_cat_to_onehot=512 if onehot else 2,
                  max_cat_threshold=257, max_cached_hist_node=1, subsample=0.73,
                  sampling_method='gradient_based', colsample_bynode=0.8)
    compare(X, y, params, tile, feature_types=['c', 'c', 'q', 'q', 'q'],
            quantile=onehot, weight=rng.uniform(0.2, 2, len(X)).astype(np.float32))


@pytest.mark.parametrize('tile', [32, 96, 256, 1024])
@pytest.mark.parametrize('missing', [False, True])
def test_equal_gain_plateaus_across_tiles(tile, missing):
    rows = 4099
    code = np.arange(rows, dtype=np.float32) % 2049
    X = np.column_stack([code, code, code[::-1]]).astype(np.float32)
    gradient = np.where(code < 257, -1, np.where(code >= 1793, 1, 0)).astype(np.float32)
    hessian = (gradient != 0).astype(np.float32)
    if missing:
        X[::31, 0] = np.nan
        X[::31, 1] = np.nan
    gradient = np.column_stack([gradient, gradient * 2, -gradient])
    hessian = np.column_stack([hessian, hessian, hessian])
    compare(X, np.zeros_like(gradient), dict(max_bin=2050, max_depth=3, reg_lambda=0), tile,
            objective=lambda prediction, data: (gradient, hessian))


@pytest.mark.parametrize('tuning', [
    {'metal_fused_partition': 0},
    {'metal_partition_scan': 0, 'metal_partition_aggregation': 0},
    {'metal_partition_scan': 0, 'metal_partition_aggregation': 1},
    {'metal_coalesce_encoders': 1, 'metal_compute_fill': 1, 'metal_tree_persistent_workspace': 0},
    {'metal_tree_workspace': 0},
    {'metal_histogram_subtraction': 0},
])
def test_tiling_composes_with_tree_execution(tuning):
    X, y = numeric_data(513, 5)
    compare(X, y, dict(max_bin=514, max_depth=3, max_cached_hist_node=1,
                       monotone_constraints='(1,1,-1,0,0)', reg_alpha=0.31, **tuning), 96)


class ProjectedObjective(TreeObjective):
    def __call__(self, iteration, prediction, data):
        gradient = prediction - data.get_label().reshape(prediction.shape)
        hessian = np.where(np.arange(len(gradient))[:, None] % 7, 1.0, -0.25)
        return gradient, np.broadcast_to(hessian, gradient.shape).astype(np.float32)

    def split_grad(self, iteration, gradient, hessian):
        columns = [3, 1, 2][iteration % 3]
        return gradient[:, :columns], hessian[:, :columns]


@pytest.mark.parametrize('sampling', ['uniform', 'gradient_based'])
@pytest.mark.parametrize('parallel', [1, 3])
def test_tiling_with_reduced_gradients(sampling, parallel):
    X, y = numeric_data(1025, 4)
    compare(X, y, dict(max_bin=1026, max_depth=3, min_child_weight=0, num_parallel_tree=parallel,
                       subsample=0.71, sampling_method=sampling), 256, rounds=3, objective=ProjectedObjective())


@pytest.mark.parametrize('rows,quantile', [(70003, False), (70003, True), (0, True)])
def test_tiling_wide_and_empty_reference_bins(rows, quantile):
    index = np.arange(70003, dtype=np.float32)
    X = np.column_stack([index, index % 13, index % 19]).astype(np.float32)
    y = np.column_stack([index >= 32768, index >= 65536]).astype(np.float32)
    X[::17, 2] = np.nan
    results = []
    for tile in [0, 1024]:
        parent = xgb.QuantileDMatrix(X, label=y, max_bin=65536)
        if quantile:
            data = xgb.QuantileDMatrix(X[:rows], label=y[:rows], max_bin=65536, ref=parent)
        else:
            data = xgb.DMatrix(X[:rows], label=y[:rows])
        model = xgb.train(dict(max_bin=65536, max_depth=3, multi_strategy='multi_output_tree',
                               metal_vector_split_bin_tile=tile), data, 2)
        results.append((bytes(model.save_raw()), model.predict(data).view(np.uint32)))
    assert results[0][0] == results[1][0]
    np.testing.assert_array_equal(results[0][1], results[1][1])


@pytest.mark.parametrize('tile', [1, 31, 33, 16385, -1])
def test_invalid_vector_tile(tile):
    data = xgb.DMatrix(np.zeros((4, 2), np.float32), label=np.zeros((4, 2), np.float32))
    with pytest.raises(xgb.core.XGBoostError, match='metal_vector_split_bin_tile'):
        xgb.train(dict(multi_strategy='multi_output_tree', metal_vector_split_bin_tile=tile), data, 1)
