"""Regenerate synthetic model-exchange fixtures in the pinned CPU environment.

Run with PYTHONPATH=ref/xgboost/python-package .venv/bin/python
tools/generate_category_fixtures.py. The installed-wheel smoke only reads the
resulting files and does not import or invoke this reference implementation.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
import xgboost as xgb


def main():
    root = Path(__file__).resolve().parents[1]
    reference = root / 'ref/xgboost'
    assert Path(xgb.__file__).resolve().is_relative_to(reference)
    assert xgb.__version__ == '3.5.0-dev' and not xgb.build_info().get('USE_CUDA')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=reference, text=True).strip()
    assert commit == '6d2dbe19a099aa046178be8cfc303bfec1a623a6'
    destination = root / 'tools/fixtures/category-bits'
    destination.mkdir(parents=True, exist_ok=True)
    codes = (np.arange(259) % 4).astype(np.float32)
    numeric = (np.arange(259) % 7).astype(np.float32)
    label = codes * 3 - numeric / 4
    vector_label = np.column_stack([label, label + numeric, -label])
    codes[::29] = np.nan
    numeric[::31] = np.nan
    codes_int = np.where(np.isnan(codes), -1, codes).astype(np.int64)
    payload = dict(codes=codes, numeric=numeric, label=label, vector_label=vector_label)
    cases = {}
    for dtype in ['uint16', 'uint32']:
        middle = 2 ** (np.iinfo(dtype).bits - 1)
        names = np.asarray([middle, 0, np.iinfo(dtype).max, middle - 1], dtype=dtype)
        payload[dtype + '_names'] = names
        frame = pd.DataFrame({'kind': pd.Categorical.from_codes(codes_int, names), 'x': numeric})
        reordered = frame.copy()
        reordered['kind'] = reordered['kind'].cat.reorder_categories(names[::-1])
        for outputs in [1, 3]:
            key = f'{dtype}-{outputs}'
            params = dict(max_depth=3, max_bin=8, base_score=0, max_cat_to_onehot=1,
                          multi_strategy='multi_output_tree' if outputs > 1 else 'one_output_per_tree')
            target = vector_label if outputs > 1 else label
            initial = xgb.train({'tree_method': 'hist', 'nthread': 1, **params},
                                xgb.DMatrix(frame, label=target, enable_categorical=True), 4)
            initial.save_model(destination / (key + '.ubj'))
            query = xgb.DMatrix(reordered, enable_categorical=True)
            payload[key + '_prediction'] = initial.predict(query)
            payload[key + '_leaf'] = initial.predict(query, pred_leaf=True)
            for fmt in ['json', 'ubj']:
                restored = xgb.Booster(model_file=initial.save_raw(raw_format=fmt))
                np.testing.assert_array_equal(restored.predict(query).view(np.uint32),
                                              payload[key + '_prediction'].view(np.uint32))
            continued = xgb.train({'tree_method': 'hist', 'nthread': 1, **params},
                                  xgb.DMatrix(reordered, label=target, enable_categorical=True),
                                  2, xgb_model=initial)
            payload[key + '_continued'] = continued.predict(query)
            payload[key + '_continued_leaf'] = continued.predict(query, pred_leaf=True)
            if outputs == 1:
                stored = xgb.DMatrix(reordered, label=target, enable_categorical=True,
                                     feature_types=initial.get_categories())
                for field in ['indptr', 'indices', 'data']:
                    payload[key + '_stored_' + field] = getattr(stored.get_data(), field)
                refreshed = xgb.train({**params, 'nthread': 1, 'process_type': 'update', 'updater': 'refresh'},
                                      stored, 2, xgb_model=initial)
                payload[key + '_refreshed'] = refreshed.predict(query)
                payload[key + '_refreshed_leaf'] = refreshed.predict(query, pred_leaf=True)
            cases[key] = dict(dtype=dtype, outputs=outputs, params=params)
    np.savez(destination / 'expected.npz', **payload)
    files = [destination / 'expected.npz', *sorted(destination.glob('*.ubj'))]
    manifest = dict(reference_commit=commit, reference_version=xgb.__version__,
                    reference_library_sha256=hashlib.sha256(Path(xgb.build_info()['libxgboost']).read_bytes()).hexdigest(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    cases=cases,
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
