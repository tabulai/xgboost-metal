"""Model exchange must retain unsigned category names across signed boundaries."""
import json
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xgboost as xgb


def data_for(dtype, outputs):
    limits = np.iinfo(dtype)
    if dtype == 'uint64':
        names = [2**63 - 1, 0, 2**32, 2**63 - 2]
    elif limits.min == 0:
        middle = 2 ** (limits.bits - 1)
        names = [middle, 0, limits.max, middle - 1]
    else:
        names = [limits.min, 0, limits.max, -1]
    names = np.asarray(names, dtype=dtype)
    codes = np.arange(259) % len(names)
    frame = pd.DataFrame({'kind': pd.Categorical.from_codes(codes, names),
                          'x': (np.arange(len(codes)) % 7).astype(np.float32)})
    label = (codes * 3 - frame['x'].to_numpy(np.float32) / 4).astype(np.float32)
    if outputs > 1:
        label = np.column_stack([label, label + frame['x'], -label]).astype(np.float32)
    frame.loc[::29, 'kind'] = np.nan
    frame.loc[::31, 'x'] = np.nan
    reordered = frame.copy()
    reordered['kind'] = reordered['kind'].cat.reorder_categories(names[::-1])
    return frame, reordered, label, names


def params_for(outputs, onehot=False):
    return dict(max_depth=3, max_bin=8, base_score=0, max_cat_to_onehot=8 if onehot else 1,
                multi_strategy='multi_output_tree' if outputs > 1 else 'one_output_per_tree')


def check_names(model, names, dtype):
    encoded = json.loads(model.save_raw(raw_format='json'))['learner']['gradient_booster']['model']['cats']
    assert encoded['enc'][0]['values'] == names.tolist()
    assert encoded['sorted_idx'] == np.argsort(names, kind='stable').tolist()
    columns = model.get_categories(export_to_arrow=True).to_arrow()
    assert columns[0][1].to_pylist() == names.tolist()
    assert columns[0][1].type == pa.from_numpy_dtype(dtype)


def check_stored_data(matrix, exported):
    csr = matrix.get_data()
    for field in ['indptr', 'indices', 'data']:
        actual, expected = getattr(csr, field), exported['train_' + field]
        np.testing.assert_array_equal(actual.view(np.uint32) if field == 'data' else actual,
                                      expected.view(np.uint32) if field == 'data' else expected)
    assert matrix.num_nonmissing() == len(exported['train_data'])


@pytest.mark.parametrize('dtype', ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'])
@pytest.mark.parametrize('outputs', [1, 3])
@pytest.mark.parametrize('onehot', [False, True])
def test_declared_category_dtype_survives_model_import(reference, dtype, outputs, onehot):
    frame, reordered, label, names = data_for(dtype, outputs)
    params = params_for(outputs, onehot)
    expected, files = reference(frame, label, params, rounds=4, test=reordered)
    parent = xgb.QuantileDMatrix(frame, label=label, max_bin=8, enable_categorical=True)
    queries = [xgb.DMatrix(reordered, enable_categorical=True),
               xgb.QuantileDMatrix(reordered, max_bin=8, enable_categorical=True),
               xgb.QuantileDMatrix(reordered, max_bin=8, enable_categorical=True, ref=parent)]
    for fmt in ['json', 'ubj']:
        model = xgb.Booster(model_file=files / f'model.{fmt}')
        check_names(model, names, dtype)
        for data in queries:
            np.testing.assert_array_equal(model.predict(data).view(np.uint32), expected['prediction'].view(np.uint32))
            np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected['leaf'])
        np.testing.assert_array_equal(model.inplace_predict(frame).view(np.uint32), expected['prediction'].view(np.uint32))
        np.testing.assert_array_equal(model.inplace_predict(reordered).view(np.uint32), expected['prediction'].view(np.uint32))
        restored = pickle.loads(pickle.dumps(model))
        check_names(restored, names, dtype)
        np.testing.assert_array_equal(restored.predict(queries[0]).view(np.uint32), expected['prediction'].view(np.uint32))
        # Rejection must leave valid predictions and category metadata intact.
        unknown = reordered.iloc[:1].copy()
        unknown['kind'] = pd.Categorical(np.asarray([13], dtype=dtype), categories=np.asarray([13], dtype=dtype))
        for _ in range(2):
            with pytest.raises(xgb.core.XGBoostError, match='not in the training data'):
                model.inplace_predict(unknown)
        np.testing.assert_array_equal(model.predict(queries[0]).view(np.uint32), expected['prediction'].view(np.uint32))


@pytest.mark.parametrize('dtype', ['uint16', 'uint32'])
@pytest.mark.parametrize('outputs', [1, 3])
@pytest.mark.parametrize('quantile', [False, True])
def test_unsigned_import_continues_with_reordered_categories(reference, dtype, outputs, quantile):
    frame, reordered, label, names = data_for(dtype, outputs)
    params = params_for(outputs)
    _, files = reference(frame, label, params, rounds=4, quantile=quantile)
    expected, _ = reference(reordered, label, params, model=files / 'model.ubj',
                            continue_rounds=2, quantile=quantile)
    cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    data = cls(reordered, label=label, enable_categorical=True, **({'max_bin': 8} if quantile else {}))
    before = data.get_data().data.copy()
    imported = xgb.Booster(model_file=files / 'model.ubj')
    model = xgb.train(params, data, 2, xgb_model=imported)
    check_names(model, names, dtype)
    assert model.num_boosted_rounds() == 6
    np.testing.assert_array_equal(data.get_data().data.view(np.uint32), before.view(np.uint32))
    assert data.get_categories(export_to_arrow=True).to_arrow()[0][1].to_pylist() == names[::-1].tolist()
    np.testing.assert_array_equal(model.predict(data).view(np.uint32), expected['prediction'].view(np.uint32))
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected['leaf'])


@pytest.mark.parametrize('dtype', ['uint16', 'uint32'])
@pytest.mark.parametrize('updater', ['hist', 'refresh', 'refresh,prune'])
@pytest.mark.parametrize('custom', [False, True])
def test_explicit_category_reference_supplies_stored_codes(reference, dtype, updater, custom):
    frame, reordered, label, names = data_for(dtype, 1)
    params = params_for(1)
    _, files = reference(frame, label, params, rounds=4)
    if updater != 'hist':
        params.update(process_type='update', updater=updater)
    expected, expected_files = reference(reordered, label, params, model=files / 'model.ubj', rounds=0,
                                         continue_rounds=2, custom_square=custom, category_reference=True,
                                         matrix_data=True)
    imported = xgb.Booster(model_file=files / 'model.ubj')
    data = xgb.DMatrix(reordered, label=label, enable_categorical=True,
                       feature_types=imported.get_categories())
    exported = np.load(expected_files / 'matrix-data.npz')
    check_stored_data(data, exported)
    objective = (lambda prediction, d: (prediction - d.get_label(), np.ones_like(prediction))) if custom else None
    model = xgb.train(params, data, 2, xgb_model=imported, obj=objective)
    check_names(model, names, dtype)
    check_stored_data(data, exported)
    np.testing.assert_array_equal(model.predict(data).view(np.uint32), expected['training_prediction'].view(np.uint32))
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected['training_leaf'])
    query = xgb.DMatrix(reordered, enable_categorical=True)
    np.testing.assert_array_equal(model.predict(query).view(np.uint32), expected['prediction'].view(np.uint32))
    np.testing.assert_array_equal(model.predict(query, pred_leaf=True), expected['leaf'])


@pytest.mark.parametrize('name_kind', ['unsigned', 'string'])
@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('reorder', [False, True])
@pytest.mark.parametrize('missing', [np.nan, 0.0])
def test_category_reference_applies_before_missing_filter(reference, name_kind, quantile, reorder, missing):
    frame, reordered, label, _ = data_for('uint16', 1)
    if name_kind == 'string':
        names = ['', '茶', 'a', 'pear']
        frame['kind'] = frame['kind'].cat.rename_categories(names)
        reordered = frame.copy()
        reordered['kind'] = reordered['kind'].cat.reorder_categories(names[::-1])
    query_frame = reordered if reorder else frame
    params = params_for(1)
    _, original = reference(frame, label, params, rounds=2)
    expected, files = reference(query_frame, label, params, rounds=0, model=original / 'model.ubj',
                                continue_rounds=2, quantile=quantile, category_reference=True,
                                matrix_data=True, missing=missing)
    imported = xgb.Booster(model_file=original / 'model.ubj')
    cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    matrix = cls(query_frame, label=label, enable_categorical=True, missing=missing,
                  feature_types=imported.get_categories(), **({'max_bin': 8} if quantile else {}))
    exported = np.load(files / 'matrix-data.npz')
    check_stored_data(matrix, exported)
    model = xgb.train(params, matrix, 2, xgb_model=imported)
    check_stored_data(matrix, exported)
    np.testing.assert_array_equal(model.predict(matrix).view(np.uint32), expected['training_prediction'].view(np.uint32))
    np.testing.assert_array_equal(model.predict(matrix, pred_leaf=True), expected['training_leaf'])
    query = xgb.DMatrix(query_frame, enable_categorical=True, missing=missing)
    np.testing.assert_array_equal(model.predict(query).view(np.uint32), expected['prediction'].view(np.uint32))
    np.testing.assert_array_equal(model.predict(query, pred_leaf=True), expected['leaf'])


@pytest.mark.parametrize('dtype', ['uint16', 'uint32'])
@pytest.mark.parametrize('outputs', [1, 3])
def test_imported_unsigned_names_export_back_to_reference(reference, tmp_path, dtype, outputs):
    frame, reordered, label, names = data_for(dtype, outputs)
    params = params_for(outputs)
    expected, files = reference(frame, label, params, rounds=4, test=reordered)
    model = xgb.Booster(model_file=files / 'model.ubj')
    check_names(model, names, dtype)
    for fmt in ['json', 'ubj']:
        path = tmp_path / f'unsigned.{fmt}'
        model.save_model(path)
        loaded, _ = reference(frame, label, params, model=path, test=reordered)
        np.testing.assert_array_equal(loaded['prediction'].view(np.uint32), expected['prediction'].view(np.uint32))
        np.testing.assert_array_equal(loaded['leaf'], expected['leaf'])
