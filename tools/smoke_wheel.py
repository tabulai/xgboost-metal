"""Run from an isolated environment after installing the built wheel."""
from pathlib import Path
import ctypes
import json
import os
import pickle
import numpy as np
import xgboost as xgb
from category_fixture_check import check_category_fixtures

def command_count(model):
    value = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(value)))
    return json.loads(value.value)['command_buffers']

root = Path(__file__).resolve().parents[1]
library = Path(xgb.build_info()['libxgboost']).resolve()
assert 'site-packages/xgboost/lib/' in str(library), library
assert library != root / 'lib/libxgboost.dylib'
assert xgb.build_info()['USE_METAL'] is True

check_category_fixtures()

rng = np.random.default_rng(91)
X = rng.normal(size=(191, 5)).astype(np.float32)
y = (X[:, 0] - X[:, 1] > 0).astype(np.float32)
d = xgb.DMatrix(X, label=y)
b = xgb.train({'objective': 'binary:logistic', 'max_depth': 3}, d, 8)
prediction = b.predict(d)
assert np.mean((prediction > 0.5) == y) > 0.95
margin = b.predict(d, output_margin=True)
commands = command_count(b)
np.testing.assert_array_equal(b.predict(d, output_margin=True), margin)
assert command_count(b) == commands
np.testing.assert_array_equal(b.predict(d), prediction)
np.testing.assert_array_equal(pickle.loads(pickle.dumps(b)).predict(d), prediction)
np.testing.assert_allclose(b.predict(d, pred_contribs=True).sum(axis=1),
                           b.predict(d, output_margin=True), atol=2e-5)
for method in ['uniform', 'gradient_based']:
    params = {'sampling_method': method, 'subsample': 0.61, 'num_parallel_tree': 2,
              'max_depth': 3, 'max_bin': 32, 'seed': 79}
    sampled = xgb.train(params, d, 4)
    serial = xgb.train({**params, 'metal_parallel_splits': 0}, d, 4)
    assert sampled.save_raw() == serial.save_raw()
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(sampled)).predict(d), sampled.predict(d))
    np.testing.assert_array_equal(xgb.Booster(model_file=sampled.save_raw()).predict(d), sampled.predict(d))
    continued = xgb.train(params, d, 2, xgb_model=pickle.loads(pickle.dumps(sampled)))
    complete = xgb.train(params, d, 6)
    assert continued.save_raw() == complete.save_raw()
restored = xgb.Booster(model_file=b.save_raw())
np.testing.assert_array_equal(restored.predict(d), prediction)
b.set_param('metal_predict_shared', 1)
np.testing.assert_array_equal(b.predict(xgb.DMatrix(np.asfortranarray(X))), prediction)
linear_data = xgb.DMatrix(X, label=np.column_stack([X[:, 0] - X[:, 1], X[:, 2]]))
for selector in ['cyclic', 'shuffle', 'random', 'greedy', 'thrifty']:
    linear = xgb.train({'booster': 'gblinear', 'updater': 'coord_descent', 'feature_selector': selector}, linear_data, 8)
    prediction = linear.predict(linear_data)
    assert prediction.shape == (len(X), 2) and np.isfinite(prediction).all()
    np.testing.assert_array_equal(xgb.Booster(model_file=linear.save_raw()).predict(linear_data), prediction)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(linear)).predict(linear_data), prediction)
    np.testing.assert_allclose(linear.predict(linear_data, pred_contribs=True).sum(axis=-1), prediction, atol=2e-6)
    assert len(linear.get_score()) == X.shape[1]
    assert len(json.loads(linear.get_dump(dump_format='json')[0])['bias']) == 2
for method in ['uniform', 'gradient_based']:
    params = {'multi_strategy': 'multi_output_tree', 'max_depth': 3, 'max_bin': 32,
              'num_parallel_tree': 2, 'sampling_method': method, 'subsample': 0.61, 'seed': 79}
    vector = xgb.train(params, linear_data, 4)
    prediction = vector.predict(linear_data)
    assert prediction.shape == (len(X), 2) and np.isfinite(prediction).all()
    np.testing.assert_array_equal(xgb.Booster(model_file=vector.save_raw()).predict(linear_data), prediction)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(vector)).predict(linear_data), prediction)
    np.testing.assert_allclose(vector.predict(linear_data, pred_contribs=True).sum(axis=-1), prediction, atol=3e-5)
    continued = xgb.train(params, linear_data, 2, xgb_model=pickle.loads(pickle.dumps(vector)))
    assert continued.save_raw() == xgb.train(params, linear_data, 6).save_raw()
from xgboost.objective import TreeObjective
for strategy, kind in [('one_output_per_tree', 'scalar'), ('multi_output_tree', 'vector')]:
    for quantile in [False, True]:
        index = np.arange(4099, dtype=np.float32)
        tile_X = np.column_stack([index, index % 17, index % 37]).astype(np.float32)
        tile_X[::13, 2] = np.nan
        tile_y = np.column_stack([index >= 2048, index % 11, index % 23]).astype(np.float32)
        matrix_type = xgb.QuantileDMatrix if quantile else xgb.DMatrix
        tile_data = matrix_type(tile_X, label=tile_y, **({'max_bin': 65536} if quantile else {}))
        untile_data = matrix_type(tile_X, label=tile_y, **({'max_bin': 65536} if quantile else {}))
        tile_params = dict(multi_strategy=strategy, max_bin=65536, max_depth=3)
        tiled = xgb.train(tile_params, tile_data, 3)
        untiled = xgb.train({**tile_params, f'metal_{kind}_split_bin_tile': 0}, untile_data, 3)
        assert tiled.save_raw() == untiled.save_raw()
        np.testing.assert_array_equal(tiled.predict(tile_data).view(np.uint32),
                                      untiled.predict(untile_data).view(np.uint32))
        assert command_count(tiled) == command_count(untiled)
        value = ctypes.c_char_p()
        xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(tiled.handle, ctypes.byref(value)))
        tuning = json.loads(value.value)['tuning']
        assert tuning[f'{kind}_split_bin_tile'] == 256 and tuning[f'{kind}_split_bin_tiles'] == 17
saved_cut_control = os.environ.get('METALGB_SERIAL_CUTS')
try:
    for quantile in [False, True]:
        for requested in [257, 65536]:
            comparisons = []
            for serial in ['1', '0']:
                os.environ['METALGB_SERIAL_CUTS'] = serial
                matrix_type = xgb.QuantileDMatrix if quantile else xgb.DMatrix
                cut_data = matrix_type(tile_X, label=tile_y,
                                       weight=np.linspace(0.1, 2, len(tile_X), dtype=np.float32),
                                       **({'max_bin': requested} if quantile else {}))
                cut_model = xgb.train(dict(max_bin=requested, max_depth=3, seed=19), cut_data, 2)
                ptr, values = cut_data.get_quantile_cut()
                value = ctypes.c_char_p()
                xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(cut_model.handle, ctypes.byref(value)))
                tuning = json.loads(value.value)['tuning']
                assert tuning['quantile_cut_tile'] == (0 if serial == '1' else 256)
                expected_tiles = (min(len(tile_X), requested) - 1 + 255) // 256
                assert tuning['quantile_cut_tiles'] == (0 if serial == '1' else expected_tiles)
                comparisons.append((cut_model.save_raw(), ptr, values.view(np.uint32),
                                    cut_model.predict(cut_data).view(np.uint32), command_count(cut_model)))
            assert comparisons[0][0] == comparisons[1][0] and comparisons[0][-1] == comparisons[1][-1]
            for actual, expected in zip(comparisons[1][1:-1], comparisons[0][1:-1]):
                np.testing.assert_array_equal(actual, expected)
finally:
    if saved_cut_control is None:
        os.environ.pop('METALGB_SERIAL_CUTS', None)
    else:
        os.environ['METALGB_SERIAL_CUTS'] = saved_cut_control
saved_summary_control = os.environ.get('METALGB_SERIAL_SUMMARY')
saved_force_summary_control = os.environ.get('METALGB_FORCE_PARALLEL_SUMMARY')
try:
    os.environ['METALGB_FORCE_PARALLEL_SUMMARY'] = '1'
    for quantile in [False, True]:
        for weighted in [False, True]:
            comparisons = []
            for serial in ['1', '0']:
                os.environ['METALGB_SERIAL_SUMMARY'] = serial
                matrix_type = xgb.QuantileDMatrix if quantile else xgb.DMatrix
                summary_data = matrix_type(tile_X, label=tile_y,
                    **({'weight': np.linspace(0.1, 2, len(tile_X), dtype=np.float32)} if weighted else {}),
                    **({'max_bin': 257} if quantile else {}))
                summary_model = xgb.train(dict(max_bin=257, max_depth=3, seed=23), summary_data, 2)
                ptr, values = summary_data.get_quantile_cut()
                value = ctypes.c_char_p()
                xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(summary_model.handle, ctypes.byref(value)))
                tuning = json.loads(value.value)['tuning']
                active = serial == '0' and not weighted
                assert tuning['quantile_summary_group'] == (256 if active else 0)
                assert (tuning['quantile_summary_batches'] > 0) == active
                comparisons.append((summary_model.save_raw(), ptr, values.view(np.uint32),
                                    summary_model.predict(summary_data).view(np.uint32), command_count(summary_model)))
            assert comparisons[0][0] == comparisons[1][0] and comparisons[0][-1] == comparisons[1][-1]
            for actual, expected in zip(comparisons[1][1:-1], comparisons[0][1:-1]):
                np.testing.assert_array_equal(actual, expected)
    os.environ.pop('METALGB_FORCE_PARALLEL_SUMMARY', None)
    summary_rng = np.random.default_rng(227)
    summary_X = summary_rng.integers(0, 16, (40003, 4)).astype(np.float32)
    for mixed in [False, True]:
        if mixed:
            summary_X[:, 0] = summary_rng.permutation(len(summary_X)).astype(np.float32) / np.float32(len(summary_X))
        summary_y = summary_X[:, 0] + summary_X[:, 1] ** 2
        for quantile in [False, True]:
            comparisons = []
            for serial in ['1', '0']:
                os.environ['METALGB_SERIAL_SUMMARY'] = serial
                matrix_type = xgb.QuantileDMatrix if quantile else xgb.DMatrix
                summary_data = matrix_type(summary_X, label=summary_y,
                                          **({'max_bin': 8192} if quantile else {}))
                summary_model = xgb.train(dict(max_bin=8192, max_depth=2, base_score=0, seed=23), summary_data, 2)
                ptr, values = summary_data.get_quantile_cut()
                value = ctypes.c_char_p()
                xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(summary_model.handle, ctypes.byref(value)))
                tuning = json.loads(value.value)['tuning']
                assert tuning['quantile_summary_group'] == (0 if serial == '1' else 256)
                assert (tuning['quantile_summary_batches'] > 0) == (serial == '0')
                comparisons.append((summary_model.save_raw(), ptr, values.view(np.uint32),
                                    summary_model.predict(summary_data).view(np.uint32), command_count(summary_model)))
            assert comparisons[0][0] == comparisons[1][0] and comparisons[0][-1] == comparisons[1][-1]
            for actual, expected in zip(comparisons[1][1:-1], comparisons[0][1:-1]):
                np.testing.assert_array_equal(actual, expected)
finally:
    if saved_summary_control is None:
        os.environ.pop('METALGB_SERIAL_SUMMARY', None)
    else:
        os.environ['METALGB_SERIAL_SUMMARY'] = saved_summary_control
    if saved_force_summary_control is None:
        os.environ.pop('METALGB_FORCE_PARALLEL_SUMMARY', None)
    else:
        os.environ['METALGB_FORCE_PARALLEL_SUMMARY'] = saved_force_summary_control
class ReducedSquare(TreeObjective):
    def __call__(self, iteration, prediction, matrix):
        return prediction - matrix.get_label().reshape(prediction.shape), np.ones_like(prediction)
    def split_grad(self, iteration, gradient, hessian):
        return gradient[:, :1], hessian[:, :1]
reduced = xgb.train({'multi_strategy': 'multi_output_tree', 'max_depth': 3, 'max_bin': 32,
                     'sampling_method': 'gradient_based', 'subsample': 0.7}, linear_data, 4, obj=ReducedSquare())
prediction = reduced.predict(linear_data)
assert prediction.shape == (len(X), 2) and np.isfinite(prediction).all()
np.testing.assert_array_equal(xgb.Booster(model_file=reduced.save_raw()).predict(linear_data), prediction)
for tree in json.loads(reduced.save_raw(raw_format='json'))['learner']['gradient_booster']['model']['trees']:
    assert tree['tree_param']['size_leaf_vector'] == '2'
    assert len(tree['base_weights']) == len(tree['left_children'])
categorical_X = rng.integers(0, 9, size=(131, 3)).astype(np.float32)
categorical_y = (categorical_X[:, 0] % 3) - (categorical_X[:, 1] % 2)
categorical = xgb.DMatrix(categorical_X, label=categorical_y, feature_types=['c'] * 3, enable_categorical=True)
categorical_model = xgb.train({'max_depth': 3}, categorical, 5)
np.testing.assert_array_equal(xgb.Booster(model_file=categorical_model.save_raw()).predict(categorical),
                              categorical_model.predict(categorical))
np.testing.assert_allclose(categorical_model.predict(categorical, pred_contribs=True).sum(axis=1),
                           categorical_model.predict(categorical, output_margin=True), atol=3e-5)
statistics_only = xgb.train({'process_type': 'update', 'updater': 'refresh', 'refresh_leaf': False},
                            categorical, 5, xgb_model=categorical_model)
np.testing.assert_array_equal(statistics_only.predict(categorical), categorical_model.predict(categorical))
for updater in ['refresh', 'prune', 'refresh,prune', 'prune,refresh']:
    updated = xgb.train({'process_type': 'update', 'updater': updater, 'gamma': 1e6},
                        categorical, 3, xgb_model=categorical_model)
    assert updated.num_boosted_rounds() == 3
    prediction = updated.predict(categorical)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(updated)).predict(categorical), prediction)
    np.testing.assert_array_equal(xgb.Booster(model_file=updated.save_raw()).predict(categorical), prediction)
    np.testing.assert_allclose(updated.predict(categorical, pred_contribs=True).sum(axis=-1), prediction, atol=3e-5)
    if 'prune' in updater:
        saved = json.loads(updated.save_raw(raw_format='json'))['learner']['gradient_booster']['model']['trees']
        assert all(int(tree['tree_param']['num_deleted']) > 0 for tree in saved)
        assert updated.get_score() == {}
    assert updated.predict(xgb.DMatrix(categorical_X[:0])).shape == (0, 0)
dropout = xgb.train({'max_depth': 3, 'rate_drop': 0.35, 'one_drop': True,
                     'sample_type': 'weighted', 'normalize_type': 'forest'}, categorical, 5)
dropout_weights = json.loads(dropout.save_raw(raw_format='json'))['learner']['gradient_booster']['model']['weight_drop']
assert len(dropout_weights) == 5 and np.any(np.asarray(dropout_weights) < 1)
np.testing.assert_array_equal(pickle.loads(pickle.dumps(dropout)).predict(categorical), dropout.predict(categorical))
np.testing.assert_array_equal(xgb.Booster(model_file=dropout.save_raw()).predict(categorical), dropout.predict(categorical))
np.testing.assert_allclose(dropout.predict(categorical, pred_contribs=True).sum(axis=-1),
                           dropout.predict(categorical, output_margin=True), atol=3e-5)
lossguide = xgb.train({'grow_policy': 'lossguide', 'max_depth': 0, 'max_leaves': 13,
                       'max_cached_hist_node': 1}, categorical, 5)
np.testing.assert_array_equal(pickle.loads(pickle.dumps(lossguide)).predict(categorical), lossguide.predict(categorical))
np.testing.assert_array_equal(xgb.Booster(model_file=lossguide.save_raw()).predict(categorical), lossguide.predict(categorical))
np.testing.assert_allclose(lossguide.predict(categorical, pred_contribs=True).sum(axis=-1),
                           lossguide.predict(categorical, output_margin=True), atol=3e-5)
assert lossguide[:1].predict(categorical, pred_leaf=True).shape == (len(categorical_X),)
for tuning in [{'metal_tree_workspace': 0}, {'metal_tree_persistent_workspace': 0},
               {'metal_coalesce_encoders': 1, 'metal_compute_fill': 1}]:
    comparison = xgb.train({'grow_policy': 'lossguide', 'max_depth': 0, 'max_leaves': 13,
                            'max_cached_hist_node': 1, **tuning}, categorical, 5)
    assert comparison.save_raw() == lossguide.save_raw()
    np.testing.assert_array_equal(comparison.predict(categorical), lossguide.predict(categorical))
reuse_data = xgb.DMatrix(np.zeros((129, 3), np.float32), label=np.zeros(129, np.float32))
reuse_model = xgb.Booster({'max_depth': 1, 'base_score': 0}, [reuse_data])
scratch_allocations = []
for iteration in range(3):
    if iteration == 2:
        reuse_model.reset()
    reuse_model.update(reuse_data, iteration)
    value = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(reuse_model.handle, ctypes.byref(value)))
    tuning = json.loads(value.value)['tuning']
    assert tuning['tree_persistent_workspace'] == 1
    scratch_allocations.append(tuning['tree_buffer_allocations'])
assert scratch_allocations[0] > 0 and scratch_allocations[1] == 0
assert scratch_allocations[2] == scratch_allocations[0]
for objective in ['reg:absoluteerror', 'reg:quantileerror', 'reg:expectileerror', 'reg:normal']:
    params = {'objective': objective, 'max_depth': 3}
    if objective == 'reg:quantileerror':
        params['quantile_alpha'] = [0.1, 0.5, 0.9]
    if objective == 'reg:expectileerror':
        params['expectile_alpha'] = [0.1, 0.5, 0.9]
    smooth = xgb.train(params, categorical, 5)
    prediction = smooth.predict(categorical)
    assert np.isfinite(prediction).all()
    assert np.isfinite(float(smooth.eval(categorical).split(':')[-1]))
    if objective in ['reg:quantileerror', 'reg:expectileerror']:
        assert prediction.shape == (len(categorical_X), 3)
        assert np.all(np.diff(prediction, axis=1) >= 0)
    if objective == 'reg:normal':
        assert prediction.shape == (len(categorical_X), 2)
    np.testing.assert_array_equal(xgb.Booster(model_file=smooth.save_raw()).predict(categorical), prediction)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(smooth)).predict(categorical), prediction)
    np.testing.assert_allclose(smooth.predict(categorical, pred_contribs=True).sum(axis=-1),
                               smooth.predict(categorical, output_margin=True), atol=3e-5)
times = np.exp(np.float32(0.5) * categorical_y).astype(np.float32)
for distribution in [None, 'normal', 'logistic', 'extreme']:
    params = {'objective': 'survival:cox' if distribution is None else 'survival:aft', 'max_depth': 3}
    if distribution is None:
        label = times.copy()
        label[::3] *= -1
        metadata = {'label': label}
    else:
        params['aft_loss_distribution'] = distribution
        lower, upper = times.copy(), times.copy()
        upper[1::4] = np.inf
        lower[2::4] = 0
        lower[3::4] *= np.float32(0.8)
        upper[3::4] *= np.float32(1.2)
        metadata = {'label_lower_bound': lower, 'label_upper_bound': upper}
    survival_data = xgb.DMatrix(categorical_X, feature_types=['c'] * 3, enable_categorical=True, **metadata)
    survival = xgb.train(params, survival_data, 5)
    prediction = survival.predict(survival_data)
    assert np.isfinite(prediction).all() and np.all(prediction > 0)
    assert np.isfinite(float(survival.eval(survival_data).split(':')[-1]))
    np.testing.assert_array_equal(xgb.Booster(model_file=survival.save_raw()).predict(survival_data), prediction)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(survival)).predict(survival_data), prediction)
    np.testing.assert_allclose(survival.predict(survival_data, pred_contribs=True).sum(axis=-1),
                               survival.predict(survival_data, output_margin=True), atol=3e-5)
cox_X = rng.integers(0, 31, (8193, 3)).astype(np.float32)
cox_label = (np.arange(len(cox_X)) // 257 + 1).astype(np.float32)[rng.permutation(len(cox_X))]
cox_label[::3] *= -1
cox_margin = rng.uniform(-5, 5, len(cox_X)).astype(np.float32)
cox_data = xgb.DMatrix(cox_X, label=cox_label, base_margin=cox_margin)
cox_params = {'objective': 'survival:cox', 'max_depth': 3, 'max_bin': 64, 'max_delta_step': 1}
cox = xgb.train(cox_params, cox_data, 5)
cox_prediction = cox.predict(cox_data)
assert np.isfinite(cox_prediction).all()
np.testing.assert_array_equal(cox.inplace_predict(cox_X, base_margin=cox_margin), cox_prediction)
np.testing.assert_array_equal(pickle.loads(pickle.dumps(cox)).predict(cox_data), cox_prediction)
cox_first = xgb.train(cox_params, cox_data, 3)
assert xgb.train(cox_params, cox_data, 2, xgb_model=cox_first).save_raw() == cox.save_raw()
for quantile in [False, True]:
    matrix_type = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    for rows, bins in [(63, 2147483647), (65538, 65536)]:
        index = np.arange(rows, dtype=np.float32)
        wide_X = np.column_stack([index, index % 17]).astype(np.float32)
        wide_X[::13, 1] = np.nan
        wide_y = (index >= rows - 4).astype(np.float32)
        wide_data = matrix_type(wide_X, label=wide_y, **({'max_bin': bins} if quantile else {}))
        wide_model = xgb.train({'max_bin': bins, 'max_depth': 2, 'base_score': 0}, wide_data, 3)
        wide_prediction = wide_model.predict(wide_data)
        assert np.isfinite(wide_prediction).all() and wide_prediction[-1] > wide_prediction[0]
        assert json.loads(wide_model.save_config())['learner']['gradient_booster']['tree_train_param']['max_bin'] == str(bins)
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(wide_model)).predict(wide_data), wide_prediction)
        ptr, cuts = wide_data.get_quantile_cut()
        assert ptr[1] > 65535 if rows > 65536 else ptr[1] == rows + 1
        assert wide_data.get_data().nnz == int(np.isfinite(wide_X).sum())
        query = xgb.QuantileDMatrix(wide_X[:2], ref=wide_data, max_bin=bins)
        query_ptr, query_cuts = query.get_quantile_cut()
        np.testing.assert_array_equal(query_ptr, ptr)
        np.testing.assert_array_equal(query_cuts.view(np.uint32), cuts.view(np.uint32))
        for invalid in [2.0, 2147483648]:
            try:
                xgb.QuantileDMatrix(wide_X[:2], max_bin=invalid)
            except xgb.core.XGBoostError:
                pass
            else:
                raise AssertionError('QuantileDMatrix accepted an invalid max_bin')
    empty = matrix_type(np.empty((0, 2), np.float32), label=np.empty((0, 2), np.float32),
                        **({'max_bin': 65536} if quantile else {}))
    empty_model = xgb.train({'max_bin': 65536, 'multi_strategy': 'multi_output_tree'}, empty, 2)
    empty_params = json.loads(empty_model.save_config())['learner']['learner_model_param']
    assert empty_params['num_target'] == '2' and json.loads(empty_params['base_score']) == [0, 0]
    assert empty_model.predict(empty).shape == (0, 0)
for objective in ['rank:ndcg', 'rank:map', 'rank:pairwise']:
    relevance = (categorical_y > 0).astype(np.float32) if objective == 'rank:map' else categorical_y - categorical_y.min()
    qid = np.arange(len(categorical_X), dtype=np.uint32) // 16
    ranking_data = xgb.DMatrix(categorical_X, label=relevance, qid=qid, feature_types=['c'] * 3, enable_categorical=True)
    for method in ['topk', 'mean']:
        ranking = xgb.train({'objective': objective, 'lambdarank_pair_method': method,
                             'lambdarank_unbiased': True, 'max_depth': 3}, ranking_data, 5)
        prediction = ranking.predict(ranking_data)
        assert np.isfinite(prediction).all()
        assert np.isfinite(float(ranking.eval(ranking_data).split(':')[-1]))
        np.testing.assert_array_equal(xgb.Booster(model_file=ranking.save_raw()).predict(ranking_data), prediction)
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(ranking)).predict(ranking_data), prediction)
        np.testing.assert_allclose(ranking.predict(ranking_data, pred_contribs=True).sum(axis=-1),
                                   ranking.predict(ranking_data, output_margin=True), atol=3e-5)
for quantile in [False, True]:
    matrix_type = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    matrix_options = {'max_bin': 8} if quantile else {}
    ranking_data = matrix_type(np.arange(16, dtype=np.float32).reshape(-1, 1),
                               label=np.full(16, 3, np.float32), group=[8, 8], weight=[1, 3],
                               base_margin=np.arange(16, dtype=np.float32), **matrix_options)
    ranking = xgb.Booster({'objective': 'rank:ndcg', 'lambdarank_num_pair_per_sample': 3,
                          'max_depth': 2, 'max_bin': 8, 'base_score': 0}, [ranking_data])
    ranking.update(ranking_data, 0)
    assert json.loads(ranking.save_config())['learner']['metrics'] == []
    assert float(ranking.eval(ranking_data).split(':')[-1]) == 1
    ranking_data.set_label(np.ones(16, np.float32))
    ranking.update(ranking_data, 1)
    np.testing.assert_allclose(float(ranking.eval(ranking_data).split(':')[-1]), 1 / 7, atol=2e-7)
    ranking_data.set_group([4] * 4)
    ranking_data.set_weight([1, 3, 1, 3])
    try:
        ranking.update(ranking_data, 2)
    except xgb.core.XGBoostError as error:
        assert 'cached query group' in str(error)
    else:
        raise AssertionError('Ranking accepted weights with a different cached query count')
    np.testing.assert_allclose(float(ranking.eval(ranking_data).split(':')[-1]), 1 / 14, atol=2e-7)
    ranking_data.set_group([8, 8])
    ranking_data.set_weight([1, 3])
    ranking.update(ranking_data, 2)
    ranking.set_param('eval_metric', 'ndcg@7')
    scores = [float(field.rsplit(':', 1)[1]) for field in ranking.eval(ranking_data).split('\t')[1:]]
    np.testing.assert_allclose(scores, [1 / 7, 1], atol=2e-7)
    ranking_data.set_label(np.arange(16, dtype=np.float32) % 2)
    unbiased = xgb.Booster({'objective': 'rank:pairwise', 'lambdarank_pair_method': 'mean',
                           'lambdarank_unbiased': True, 'max_bin': 8, 'base_score': 0}, [ranking_data])
    for iteration in [1, 2]:
        rng_before = json.loads(unbiased.save_config())['learner']['generic_param']['rng_state']
        try:
            unbiased.update(ranking_data, iteration)
        except xgb.core.XGBoostError as error:
            assert 'iteration 0' in str(error)
        else:
            raise AssertionError('Unbiased ranking accepted a fresh workspace after iteration zero')
        assert json.loads(unbiased.save_config())['learner']['generic_param']['rng_state'] == rng_before
        unbiased.update(ranking_data, 0)
        unbiased.load_config(unbiased.save_config())
        unbiased.update(ranking_data, 1)
        unbiased.reset()
    unbiased.update(ranking_data, 0)
    empty_bias = json.loads(unbiased.save_config())
    empty_bias['learner']['objective']['ti+'] = []
    empty_bias['learner']['objective']['tj-'] = []
    unbiased.load_config(json.dumps(empty_bias))
    rng_before = json.loads(unbiased.save_config())['learner']['generic_param']['rng_state']
    try:
        unbiased.update(ranking_data, 1)
    except xgb.core.XGBoostError as error:
        assert 'iteration 0' in str(error)
    else:
        raise AssertionError('Unbiased ranking accepted a cleared bias fit after iteration zero')
    assert json.loads(unbiased.save_config())['learner']['generic_param']['rng_state'] == rng_before
    unbiased.update(ranking_data, 0)
sketch_rows = np.arange(513, dtype=np.float32)
sketch_X = np.column_stack([sketch_rows, (sketch_rows * 37) % 101 - 50]).astype(np.float32)
sketch_X[::17, 1] = np.nan
sketch_weights = np.resize(np.array([0, 0.125, 0.5, 1, 2], np.float32), len(sketch_X))
sketch_data = xgb.QuantileDMatrix(sketch_X, label=sketch_rows > 230, weight=sketch_weights, max_bin=7)
sketch_ptr, sketch_cuts = sketch_data.get_quantile_cut()
assert sketch_ptr.tolist() == [0, 8, 16]
# Public cuts from the XGBoost CPU reference for these deterministic weighted rows.
expected_cuts = np.array([-np.inf, 77, 149, 219, 294, 367, 439, 1024,
                         -np.inf, -37, -21, -7, 7, 21, 35, 100.00001525878906], np.float32)
np.testing.assert_array_equal(sketch_cuts.view(np.uint32), expected_cuts.view(np.uint32))
sketch_data.set_weight(np.ones(len(sketch_X), np.float32))
np.testing.assert_array_equal(sketch_data.get_quantile_cut()[1].view(np.uint32), sketch_cuts.view(np.uint32))
sketch_model = xgb.train({'max_bin': 7, 'max_depth': 3}, sketch_data, 3)
np.testing.assert_array_equal(xgb.Booster(model_file=sketch_model.save_raw()).predict(sketch_data),
                              sketch_model.predict(sketch_data))
try:
    xgb.train({'max_bin': 8}, sketch_data, 1)
except xgb.core.XGBoostError as error:
    assert 'Inconsistent max_bin' in str(error)
else:
    raise AssertionError('QuantileDMatrix accepted a different max_bin after a weight update')
tiny_levels = np.arange(-128, 129, dtype=np.float32)
tiny_X = (tiny_levels * np.nextafter(np.float32(0), np.float32(1))).reshape(-1, 1)
tiny_data = xgb.QuantileDMatrix(tiny_X, label=tiny_levels > 0, max_bin=8)
tiny_model = xgb.train({'max_bin': 8, 'max_depth': 2}, tiny_data, 3)
tiny_prediction = tiny_model.predict(tiny_data)
assert np.unique(tiny_prediction).size >= 2
np.testing.assert_allclose(tiny_model.predict(tiny_data, pred_contribs=True).sum(axis=-1), tiny_prediction, atol=2e-6)
representation_X = np.arange(16, dtype=np.float32).reshape(-1, 1)
representation_train = xgb.QuantileDMatrix(representation_X, label=representation_X[:, 0] < 8, max_bin=4)
representation_model = xgb.train({'max_bin': 4, 'max_depth': 1, 'base_score': 0, 'eta': 1}, representation_train, 1)
representation_query = xgb.QuantileDMatrix(np.arange(100, 108, dtype=np.float32).reshape(-1, 1),
                                          max_bin=4, feature_types=['q'])
exported = representation_query.get_data()
representatives = np.array([np.nextafter(np.float32(102), np.float32(-np.inf))] * 2 +
                           [102, 102, 104, 104, 106, 106], np.float32)
np.testing.assert_array_equal(exported.data.view(np.uint32), representatives.view(np.uint32))
representation_prediction = representation_model.predict(representation_query)
np.testing.assert_array_equal(representation_prediction, np.array([8 / 9, 8 / 9, 0, 0, 0, 0, 0, 0], np.float32))
commands = command_count(representation_model)
cached_prediction = representation_model.predict(representation_query)
np.testing.assert_array_equal(cached_prediction, representation_prediction)
cached_prediction.fill(-123)
np.testing.assert_array_equal(representation_model.predict(representation_query), representation_prediction)
np.testing.assert_array_equal(representation_model.predict(representation_query, output_margin=True), representation_prediction)
assert command_count(representation_model) == commands
np.testing.assert_array_equal(representation_model.predict(xgb.DMatrix(exported)), np.zeros(8, np.float32))
representation_query.feature_types = ['c']
np.testing.assert_array_equal(representation_model.predict(representation_query), representation_prediction)
representation_model.reset()
np.testing.assert_array_equal(representation_model.predict(representation_query), np.zeros(8, np.float32))
try:
    representation_query.slice([0, 1])
except xgb.core.XGBoostError as error:
    assert 'Slicing DMatrix is not supported' in str(error)
else:
    raise AssertionError('QuantileDMatrix accepted slicing')
type_X = np.tile(np.arange(8, dtype=np.float32), 40).reshape(-1, 1)
type_data = xgb.QuantileDMatrix(type_X, label=type_X[:, 0] % 3, max_bin=4, feature_types=['q'])
type_model = xgb.Booster({'max_bin': 4, 'max_depth': 2, 'base_score': 0}, [type_data])
for iteration in range(3):
    if iteration == 2:
        type_data.feature_types = ['c']
    type_model.update(type_data, iteration)
type_tree = json.loads(type_model.save_raw(raw_format='json'))['learner']['gradient_booster']['model']['trees'][2]
assert type_tree['categories'] == [2, 4, 14, 2, 14]
assert type_tree['sum_hessian'] == [320, 80, 240, 160, 80]
expected_type_margin = np.tile(np.array([1041368718, 1041368718, 1057374599, 1057374599,
                                        1057374599, 1057374599, 1041368718, 1041368718], np.uint32), 40)
np.testing.assert_array_equal(type_model.predict(type_data, output_margin=True).view(np.uint32), expected_type_margin)
boundary_X = np.full((129, 1), 5, np.float32)
boundary_X[::2] = np.nan
boundary_data = xgb.DMatrix(boundary_X, label=np.isnan(boundary_X[:, 0]))
boundary_model = xgb.train({'max_bin': 4, 'max_depth': 1, 'base_score': 0,
                            'monotone_constraints': '(-1)'}, boundary_data, 1)
boundary_model.set_attr(Infinity='-Infinity', NaN='escaped "Infinity" \\ NaN')
boundary_tree = json.loads(boundary_model.save_raw(raw_format='json'))['learner']['gradient_booster']['model']['trees'][0]
assert np.isneginf(boundary_tree['split_conditions'][0])
for format in ['json', 'ubj']:
    roundtrip = xgb.Booster(model_file=boundary_model.save_raw(raw_format=format))
    assert roundtrip.attributes() == boundary_model.attributes()
    np.testing.assert_array_equal(roundtrip.predict(boundary_data), boundary_model.predict(boundary_data))
print(json.dumps({'library': str(library), 'device': xgb.build_info()['METAL_DEVICE'],
                  'checks': ['train', 'predict', 'pickle', 'UBJSON', 'contributions',
                             'uniform and gradient-based row sampling, parallel trees and continuation',
                             'serial/SIMD split model equivalence',
                             'scalar and vector candidate tiles, exact models and unchanged command-buffer counts',
                             'linear booster, all five selectors and multiple outputs',
                             'vector leaves, shared row samples, contributions and continuation',
                             'reduced split gradients and full-width leaf values',
                             'refresh and prune, partial updates and deleted categorical nodes',
                             'empty prediction shapes',
                             'categorical trees', 'unsigned UBJSON names, stored category codes and explicit-reference refresh',
                             'tree dropout and weighted predictions', 'loss-guided growth and histogram eviction',
                             'tree workspace and encoder equivalence', 'single-tree leaf shape', 'Fortran input', 'cached prediction',
                             'absolute-error objective', 'multiple-quantile objective',
                             'multiple-expectile objective', 'normal objective',
                             'Cox ordered scans, ties across tiles and inplace base margins',
                             'compressed weighted cuts and metadata cache retention',
                             'parallel final cuts preserve weighted raw/quantile models and command counts',
                             'parallel batch summaries preserve raw/quantile models, cuts, commands, weighted fallback and GPU-deferred default work',
                             'quantile lower-bound prediction and distinct CSR representatives',
                             'prediction-prefix reuse, type changes and reset',
                             'complete identity/margin cache hits submit no command and preserve output ownership',
                             'inherited child statistics after feature-type changes',
                             'subnormal feature cuts, prediction and contributions',
                             'infinite split JSON/UBJSON exchange and string attributes',
                             'AFT normal/logistic/extreme survival',
                             'NDCG/MAP/pairwise ranking', 'topk/mean pairs', 'ranking position bias',
                             'ranking metadata cache ownership', 'unbiased failed-update recovery']}, indent=2))
