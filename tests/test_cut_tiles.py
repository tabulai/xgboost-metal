"""Parallel final-cut queries preserve materialized cuts and complete model bits."""
import ctypes
import json

import numpy as np
import pytest
import xgboost as xgb


def profile(model):
    value = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(value)))
    return json.loads(value.value)


def numeric_data(bins, groups=1):
    rows = 4099 if bins > 65538 else max(513, bins + 17)
    rng = np.random.default_rng(181)
    index = rng.permutation(np.arange(rows, dtype=np.float32)) / np.float32(rows)
    X = np.column_stack([index, rng.normal(size=rows), rng.integers(-17, 18, rows), rng.normal(size=rows)]).astype(np.float32)
    y = (index + X[:, 1] ** 2 - X[:, 3] / 3).astype(np.float32)
    X[::17, 1] = np.nan
    X[::7, 2], X[1::7, 2] = -0.0, 0.0
    if groups > 1:
        y = np.column_stack([y, index - X[:, 3], index + X[:, 3]]).astype(np.float32)
    return X, y


def make_matrix(X, y, bins, quantile, adapter='dense', **metadata):
    if adapter == 'csr':
        from scipy.sparse import csr_matrix
        X = csr_matrix(X)
    elif adapter == 'dataframe':
        import pandas as pd
        X = pd.DataFrame(X)
    elif adapter == 'iterator':
        class Batches(xgb.DataIter):
            def __init__(self):
                self.index = 0
                super().__init__()

            def reset(self):
                self.index = 0

            def next(self, input_data):
                bounds = [0, 17, 256, 401, len(X)]
                if self.index == len(bounds) - 1:
                    return False
                begin, end = bounds[self.index:self.index + 2]
                input_data(data=X[begin:end], label=y[begin:end],
                           **{key: value[begin:end] for key, value in metadata.items()})
                self.index += 1
                return True
        return xgb.QuantileDMatrix(Batches(), max_bin=bins)
    cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    return cls(X, label=y, **metadata, **({'max_bin': bins} if quantile else {}))


def snapshot(matrix, params):
    model = xgb.train(params, matrix, 2)
    cuts = matrix.get_quantile_cut()
    data = matrix.get_data()
    arrays = [cuts[0], cuts[1].view(np.uint32), data.indptr, data.indices, data.data.view(np.uint32)]
    for options in [{'output_margin': True}, {'pred_leaf': True}, {'pred_contribs': True}]:
        arrays.append(model.predict(matrix, **options).view(np.uint32))
    return (bytes(model.save_raw()), arrays, profile(model),
            json.loads(model.save_config())['learner']['generic_param']['rng_state'])


def compare(monkeypatch, X, y, bins, quantile=False, params=None, adapter='dense', **metadata):
    results = []
    for serial in ['1', '0']:
        monkeypatch.setenv('METALGB_SERIAL_CUTS', serial)
        data = make_matrix(X, y, bins, quantile, adapter=adapter, **metadata)
        results.append(snapshot(data, dict(max_bin=bins, max_depth=3, base_score=0, seed=991, **(params or {}))))
    expected, actual = results
    assert actual[0] == expected[0] and actual[3] == expected[3]
    for own, wanted in zip(actual[1], expected[1]):
        np.testing.assert_array_equal(own, wanted)
    assert actual[2]['command_buffers'] == expected[2]['command_buffers']
    assert expected[2]['tuning']['quantile_cut_tile'] == 0
    assert expected[2]['tuning']['quantile_cut_tiles'] == 0
    return actual[2]


@pytest.mark.parametrize('bins', [2, 3, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 512, 513,
                                 4096, 65536, 65537, 65538, 2147483647])
@pytest.mark.parametrize('quantile', [False, True])
def test_cut_tile_boundaries(monkeypatch, bins, quantile):
    X, y = numeric_data(bins)
    stats = compare(monkeypatch, X, y, bins, quantile)
    capacity = min(len(X), bins)
    assert stats['tuning']['quantile_cut_tile'] == (256 if capacity >= 32 else 0)
    assert stats['tuning']['quantile_cut_tiles'] == ((capacity - 1 + 255) // 256 if capacity >= 32 else 0)


@pytest.mark.parametrize('bins', [33, 257])
@pytest.mark.parametrize('kind', ['fractional', 'front', 'tail', 'zero', 'subnormal', 'small-normal',
                                 'overflow', 'span', 'ties', 'missing', 'subnormal-values'])
def test_weighted_cut_bits(monkeypatch, reference, bins, kind):
    rng = np.random.default_rng(818)
    X = rng.normal(size=(1027, 4)).astype(np.float32)
    weight = rng.uniform(0.01, 2, len(X)).astype(np.float32)
    if kind in ['front', 'tail']:
        X[:, 0] = np.arange(len(X), dtype=np.float32)
        weight.fill(2.0**-20)
        weight[0 if kind == 'front' else -1] = 2.0**80
    elif kind == 'zero':
        weight.fill(0)
    elif kind == 'subnormal':
        weight = np.ldexp(weight, -140)
    elif kind == 'small-normal':
        weight.fill(np.finfo(np.float32).tiny)
    elif kind == 'overflow':
        weight.fill(np.float32(1e38))
    elif kind == 'span':
        weight = np.exp2(rng.integers(-149, 126, len(X)).astype(np.float64)).astype(np.float32)
    elif kind == 'ties':
        X = rng.integers(-2 * bins, 2 * bins, X.shape).astype(np.float32)
        X[::7, 1], X[1::7, 1] = -0.0, 0.0
    elif kind == 'missing':
        X[:, 0] = np.nan
        X[::3, 1:] = np.nan
    elif kind == 'subnormal-values':
        X = np.ldexp(rng.integers(-2000, 2001, X.shape).astype(np.float32), -149)
    arrays = []
    for serial in ['1', '0']:
        monkeypatch.setenv('METALGB_SERIAL_CUTS', serial)
        matrix = xgb.QuantileDMatrix(X, weight=weight, max_bin=bins)
        ptr, cuts = matrix.get_quantile_cut()
        arrays.append((ptr, cuts.view(np.uint32)))
    for own, expected in zip(arrays[1], arrays[0]):
        np.testing.assert_array_equal(own, expected)
    _, out = reference(X, np.zeros(len(X)), dict(max_bin=bins, base_score=0), rounds=0,
                       quantile=True, cuts=True, weight=weight)
    with np.load(out / 'cuts.npz') as expected:
        np.testing.assert_array_equal(arrays[1][0], expected['indptr'])
        np.testing.assert_array_equal(arrays[1][1], expected['values'].view(np.uint32))


@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('onehot', [2, 100000])
@pytest.mark.parametrize('bins', [33, 257])
def test_mixed_category_cuts(monkeypatch, bins, onehot, quantile):
    X, y = numeric_data(bins)
    X[:, 2] = np.resize(np.array([0, 3, 70003], np.float32), len(X))
    X[::19, 2] = np.nan
    stats = compare(monkeypatch, X, y, bins, quantile,
                    params=dict(max_cat_to_onehot=onehot), feature_types=['q', 'q', 'c', 'q'])
    assert stats['tuning']['quantile_cut_tiles'] == (bins - 1 + 255) // 256


@pytest.mark.parametrize('quantile', [False, True])
def test_all_categories_keep_serial_cut_generation(monkeypatch, quantile):
    X = np.column_stack([np.arange(513) % 7, np.arange(513) % 43]).astype(np.float32)
    stats = compare(monkeypatch, X, (X[:, 0] < 3).astype(np.float32), 65536, quantile,
                    params=dict(max_cat_to_onehot=2), feature_types=['c', 'c'])
    assert stats['tuning']['quantile_cut_tile'] == stats['tuning']['quantile_cut_tiles'] == 0


@pytest.mark.parametrize('rows', [0, 1, 2, 31])
def test_short_and_empty_cut_queues(monkeypatch, rows):
    X = np.column_stack([np.arange(rows, dtype=np.float32), np.full(rows, np.nan, np.float32)])
    stats = compare(monkeypatch, X, np.zeros(rows, np.float32), 65536, True)
    assert stats['tuning']['quantile_cut_tile'] == 0


@pytest.mark.parametrize('adapter', ['dense', 'csr', 'dataframe', 'iterator'])
def test_parallel_cut_adapters(monkeypatch, adapter):
    X, y = numeric_data(257)
    stats = compare(monkeypatch, X, y, 257, True, adapter=adapter,
                    weight=np.linspace(0.01, 2, len(X), dtype=np.float32))
    assert stats['tuning']['quantile_cut_tile'] == 256


@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('serial_sketch', ['0', '1'])
def test_reducer_and_cut_controls_are_independent(monkeypatch, quantile, serial_sketch):
    monkeypatch.setenv('METALGB_SERIAL_SKETCH', serial_sketch)
    X, y = numeric_data(257)
    stats = compare(monkeypatch, X, y, 257, quantile,
                    weight=np.linspace(0.01, 2, len(X), dtype=np.float32))
    assert stats['tuning']['quantile_cut_tile'] == 256


@pytest.mark.parametrize('quantile', [False, True])
def test_query_weights_and_zero_length_groups(monkeypatch, quantile):
    X, y = numeric_data(257)
    stats = compare(monkeypatch, X, (y > np.median(y)).astype(np.float32), 257, quantile,
                    params=dict(objective='rank:pairwise', lambdarank_pair_method='mean'),
                    group=np.array([17, 0, 255, len(X) - 272], np.uint32),
                    weight=np.array([0.01, 10, 1, 2], np.float32))
    assert stats['tuning']['quantile_cut_tile'] == 256


@pytest.mark.parametrize('quantile', [False, True])
def test_cut_cache_and_reference_ownership(monkeypatch, quantile):
    X, y = numeric_data(257)
    monkeypatch.setenv('METALGB_SERIAL_CUTS', '0')
    data = make_matrix(X, y, 257, quantile)
    first = snapshot(data, dict(max_bin=257, max_depth=2, base_score=0))
    assert first[2]['tuning']['quantile_cut_tile'] == 256
    monkeypatch.setenv('METALGB_SERIAL_CUTS', '1')
    data.set_weight(np.linspace(0.1, 2, len(X), dtype=np.float32))
    second = snapshot(data, dict(max_bin=257, max_depth=2, base_score=0))
    assert second[2]['tuning']['quantile_cut_tile'] == 256
    for actual, expected in zip(second[1][:2], first[1][:2]):
        np.testing.assert_array_equal(actual, expected)
    query = xgb.QuantileDMatrix(X[:67], label=y[:67], ref=data, max_bin=257)
    borrowed = snapshot(query, dict(max_bin=257, max_depth=2, base_score=0))
    assert borrowed[2]['tuning']['quantile_cut_tile'] == 256
    for actual, expected in zip(borrowed[1][:2], first[1][:2]):
        np.testing.assert_array_equal(actual, expected)
    if quantile:
        with pytest.raises(xgb.core.XGBoostError, match='Inconsistent max_bin'):
            xgb.train(dict(max_bin=513, base_score=0), data, 1)
    else:
        third = snapshot(data, dict(max_bin=513, max_depth=2, base_score=0))
        assert third[2]['tuning']['quantile_cut_tile'] == 0
        monkeypatch.setenv('METALGB_SERIAL_CUTS', '0')
        fourth = snapshot(data, dict(max_bin=65, max_depth=2, base_score=0))
        assert fourth[2]['tuning']['quantile_cut_tile'] == 256


@pytest.mark.parametrize('bins', [33, 4096])
@pytest.mark.parametrize('strategy', ['one_output_per_tree', 'multi_output_tree'])
@pytest.mark.parametrize('quantile', [False, True])
def test_full_model_matches_cpu(reference, monkeypatch, bins, strategy, quantile):
    from test_large_bins import check_model
    X, y = numeric_data(bins, 3)
    params = dict(max_bin=bins, max_depth=3, base_score=0, seed=991, multi_strategy=strategy)
    expected, out = reference(X, y, params, rounds=2, quantile=quantile, cuts=True)
    monkeypatch.setenv('METALGB_SERIAL_CUTS', '0')
    data = make_matrix(X, y, bins, quantile)
    model = xgb.train(params, data, 2)
    assert profile(model)['tuning']['quantile_cut_tile'] == 256
    check_model(json.loads(model.save_raw(raw_format='json')), json.loads((out / 'model.json').read_text()))
    for options, field in [({}, 'prediction'), ({'output_margin': True}, 'margin'), ({'pred_leaf': True}, 'leaf')]:
        np.testing.assert_array_equal(model.predict(data, **options).view(np.uint32), expected[field].view(np.uint32))
    with np.load(out / 'cuts.npz') as cuts:
        ptr, values = data.get_quantile_cut()
        np.testing.assert_array_equal(ptr, cuts['indptr'])
        np.testing.assert_array_equal(values.view(np.uint32), cuts['values'].view(np.uint32))
