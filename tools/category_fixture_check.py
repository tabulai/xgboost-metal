"""Verify pinned category fixtures through the C ABI with only NumPy and SciPy."""
from pathlib import Path
import ctypes
import hashlib
import json
import numpy as np
import xgboost as xgb


# Exercise named columnar data through the C ABI, using only wheel dependencies.
def category_matrix(names, codes, numeric, label=None, reference=None):
    columns = [[names.__array_interface__, codes.__array_interface__], numeric.__array_interface__]
    description = {'columns': columns, 'ref_categories': reference.get_handle()} if reference is not None else columns
    handle = ctypes.c_void_p()
    xgb.core._check_call(xgb.core._LIB.XGDMatrixCreateFromColumnar(
        json.dumps(description).encode(), b'{"missing":NaN,"nthread":1}', ctypes.byref(handle)))
    matrix = xgb.DMatrix(handle)
    matrix.feature_names = ['kind', 'x']
    matrix.feature_types = ['c', 'q']
    if label is not None:
        matrix.set_label(label)
    return matrix


def check_category_fixtures():
    root = Path(__file__).resolve().parents[1]
    fixtures = root / 'tools/fixtures/category-bits'
    fixture_manifest = json.loads((fixtures / 'manifest.json').read_text())
    for name, expected_sha in fixture_manifest['files'].items():
        assert hashlib.sha256((fixtures / name).read_bytes()).hexdigest() == expected_sha, name
    with np.load(fixtures / 'expected.npz') as expected:
        for key, case in fixture_manifest['cases'].items():
            names = expected[case['dtype'] + '_names']
            codes, numeric = expected['codes'], expected['numeric']
            reordered_codes = 3 - codes
            target = expected['vector_label' if case['outputs'] > 1 else 'label']
            model = xgb.Booster(model_file=fixtures / (key + '.ubj'))
            saved_names = json.loads(model.save_raw(raw_format='json'))['learner']['gradient_booster']['model']['cats']['enc'][0]['values']
            assert saved_names == names.tolist()
            original = category_matrix(names, codes, numeric)
            reordered = category_matrix(names[::-1].copy(), reordered_codes, numeric, target)
            for query in [original, reordered]:
                np.testing.assert_array_equal(model.predict(query).view(np.uint32), expected[key + '_prediction'].view(np.uint32))
                np.testing.assert_array_equal(model.predict(query, pred_leaf=True), expected[key + '_leaf'])
            before = reordered.get_data().data.copy()
            continued = xgb.train(case['params'], reordered, 2, xgb_model=model)
            np.testing.assert_array_equal(reordered.get_data().data.view(np.uint32), before.view(np.uint32))
            np.testing.assert_array_equal(continued.predict(reordered).view(np.uint32), expected[key + '_continued'].view(np.uint32))
            np.testing.assert_array_equal(continued.predict(reordered, pred_leaf=True), expected[key + '_continued_leaf'])
            if case['outputs'] == 1:
                packed = np.column_stack([reordered_codes, numeric])
                for input_codes, input_numeric in [(reordered_codes, numeric), (packed[:, 0], packed[:, 1])]:
                    stored = category_matrix(names[::-1].copy(), input_codes, input_numeric, target, model.get_categories())
                    for field in ['indptr', 'indices', 'data']:
                        actual = getattr(stored.get_data(), field)
                        reference = expected[key + '_stored_' + field]
                        np.testing.assert_array_equal(actual.view(np.uint32) if field == 'data' else actual,
                                                      reference.view(np.uint32) if field == 'data' else reference)
                    refreshed = xgb.train({**case['params'], 'process_type': 'update', 'updater': 'refresh'},
                                          stored, 2, xgb_model=model)
                    np.testing.assert_array_equal(refreshed.predict(reordered).view(np.uint32), expected[key + '_refreshed'].view(np.uint32))
                    np.testing.assert_array_equal(refreshed.predict(reordered, pred_leaf=True), expected[key + '_refreshed_leaf'])

    return dict(fixture_cases=len(fixture_manifest['cases']), library=xgb.build_info()['libxgboost'])


if __name__ == '__main__':
    print(json.dumps(check_category_fixtures()), flush=True)
