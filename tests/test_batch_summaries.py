"""Batch-local integer scans preserve quantile summaries and complete models."""
import json

import numpy as np
import pytest
import xgboost as xgb

from test_cut_tiles import make_matrix, profile, snapshot


@pytest.fixture(autouse=True)
def exercise_small_parallel_batches(monkeypatch):
    # The production threshold is deliberately conservative. Algorithm tests
    # still exercise small groups, pruning boundaries and mixed fallbacks.
    monkeypatch.setenv('METALGB_FORCE_PARALLEL_SUMMARY', '1')


def inputs(rows):
    rng = np.random.default_rng(229)
    index = rng.permutation(rows).astype(np.float32) / np.float32(max(1, rows))
    X = np.column_stack([index, rng.normal(size=rows), rng.integers(-33, 34, rows), rng.normal(size=rows)]).astype(np.float32)
    y = (index + X[:, 1] ** 2 - X[:, 3] / 3).astype(np.float32)
    X[::19, 1] = np.nan
    return X, y


def compare(monkeypatch, X, y, bins=257, quantile=False, params=None, adapter='dense', **metadata):
    results = []
    for serial in ['1', '0']:
        monkeypatch.setenv('METALGB_SERIAL_SUMMARY', serial)
        data = make_matrix(X, y, bins, quantile, adapter, **metadata)
        results.append(snapshot(data, dict(max_bin=bins, max_depth=3, base_score=0, seed=67, **(params or {}))))
    expected, actual = results
    assert actual[0] == expected[0] and actual[3] == expected[3]
    for own, wanted in zip(actual[1], expected[1]):
        np.testing.assert_array_equal(own, wanted)
    assert actual[2]['command_buffers'] == expected[2]['command_buffers']
    assert expected[2]['tuning']['quantile_summary_group'] == 0
    assert expected[2]['tuning']['quantile_summary_batches'] == 0
    assert expected[2]['tuning']['quantile_weighted_summary_stages'] == 0
    assert expected[2]['tuning']['quantile_weighted_summary_batches'] == 0
    return actual


@pytest.mark.parametrize('rows', [0, 1, 2, 15, 31, 32, 33, 63, 64, 65, 255, 256, 257, 513, 4099, 70003])
@pytest.mark.parametrize('quantile', [False, True])
def test_summary_batch_boundaries(monkeypatch, rows, quantile):
    X, y = inputs(rows)
    actual = compare(monkeypatch, X, y, 65536 if rows == 70003 else 257, quantile)
    tuning = actual[2]['tuning']
    assert tuning['quantile_summary_group'] == (256 if rows >= 32 else 0)
    assert (tuning['quantile_summary_batches'] > 0) == (rows >= 32)


@pytest.mark.parametrize('rows', [31, 32, 4099, 32767, 32768, 32769])
@pytest.mark.parametrize('quantile', [False, True])
def test_default_summary_dispatch_boundary(monkeypatch, rows, quantile):
    monkeypatch.delenv('METALGB_FORCE_PARALLEL_SUMMARY')
    X, y = inputs(rows)
    actual = compare(monkeypatch, X, y, 8192, quantile)
    tuning = actual[2]['tuning']
    assert tuning['quantile_summary_group'] == (256 if rows >= 32768 else 0)
    assert (tuning['quantile_summary_batches'] > 0) == (rows >= 32768)


@pytest.mark.parametrize('rows,bins,active', [(32768, 16384, False), (32769, 16384, False),
                                           (32770, 16384, True), (70003, 65536, False)])
@pytest.mark.parametrize('quantile', [False, True])
def test_default_summary_pruning_boundary(monkeypatch, rows, bins, active, quantile):
    monkeypatch.delenv('METALGB_FORCE_PARALLEL_SUMMARY')
    X, y = inputs(rows)
    actual = compare(monkeypatch, X, y, bins, quantile)
    tuning = actual[2]['tuning']
    assert tuning['quantile_summary_group'] == (256 if active else 0)
    assert (tuning['quantile_summary_batches'] > 0) == active


@pytest.mark.parametrize('kind', ['few-values', 'mixed', 'signed-zero'])
@pytest.mark.parametrize('quantile', [False, True])
def test_default_replays_batches_without_actual_pruning(monkeypatch, kind, quantile):
    monkeypatch.delenv('METALGB_FORCE_PARALLEL_SUMMARY')
    # Adjacent repeats compact before planning. Keep enough runs above both
    # the 32768-run activation threshold and the 32769-entry sketch budget.
    X, y = inputs(40003)
    index = X[:, 0].copy()
    X = np.random.default_rng(227).integers(0, 16, X.shape).astype(np.float32)
    if kind != 'few-values':
        X[:, 0] = index
    if kind == 'signed-zero':
        X[:, 1] = np.resize(np.array([-0.0, 1, 0.0, 2], np.float32), len(X))
    actual = compare(monkeypatch, X, y, 8192, quantile)
    assert actual[2]['tuning']['quantile_summary_group'] == 256
    assert actual[2]['tuning']['quantile_summary_batches'] > 0


@pytest.mark.parametrize('kind', ['groups-31', 'groups-257', 'runs', 'signed-zero', 'negative-zero',
                                 'subnormal', 'extreme', 'missing', 'constant'])
def test_summary_group_boundaries_and_value_bits(monkeypatch, reference, kind):
    X, y = inputs(2053)
    rng = np.random.default_rng(223)
    if kind.startswith('groups-'):
        group = int(kind.split('-')[1])
        X[:, 0] = rng.permutation(np.arange(len(X)) // group).astype(np.float32)
    elif kind == 'runs':
        X[:, 0] = np.resize(np.repeat(np.arange(37, dtype=np.float32), 31), len(X))
    elif kind == 'signed-zero':
        X[:, 0] = np.resize(np.array([-0.0, 1.0, 0.0, 2.0], np.float32), len(X))
    elif kind == 'negative-zero':
        X[:, 0] = np.resize(np.array([-0.0, 1.0, -0.0, 2.0], np.float32), len(X))
    elif kind == 'subnormal':
        X[:, 0] = rng.integers(-1000, 1001, len(X)).astype(np.float32) * np.float32(2.0**-149)
    elif kind == 'extreme':
        X[:, 0] = np.ldexp(X[:, 0] - np.float32(0.5), 120)
        X[0, 0], X[1, 0] = np.finfo(np.float32).max, -np.finfo(np.float32).max
    elif kind == 'missing':
        X[:, 0] = np.nan
        X[::3, 2] = np.nan
    elif kind == 'constant':
        X[:, 0] = 7
    actual = compare(monkeypatch, X, y, 33, True)
    assert actual[2]['tuning']['quantile_summary_group'] == 256
    _, out = reference(X, y, dict(max_bin=33, base_score=0), rounds=0, quantile=True, cuts=True)
    with np.load(out / 'cuts.npz') as expected:
        np.testing.assert_array_equal(actual[1][0], expected['indptr'])
        np.testing.assert_array_equal(actual[1][1], expected['values'].view(np.uint32))


@pytest.mark.parametrize('kind', ['unit', 'fractional', 'query'])
@pytest.mark.parametrize('quantile', [False, True])
def test_explicit_weights_keep_ordered_summary_arithmetic(monkeypatch, kind, quantile):
    X, y = inputs(1027)
    metadata, params = {}, {}
    if kind == 'query':
        metadata = dict(group=np.array([31, 0, 256, len(X) - 287], np.uint32),
                        weight=np.array([0.5, 1, 2, 0.75], np.float32))
        params = dict(objective='rank:pairwise', lambdarank_pair_method='mean')
        y = (y > np.median(y)).astype(np.float32)
    else:
        metadata['weight'] = np.ones(len(X), np.float32) if kind == 'unit' else np.linspace(0.01, 2, len(X), dtype=np.float32)
    actual = compare(monkeypatch, X, y, 257, quantile, params=params, **metadata)
    assert actual[2]['tuning']['quantile_summary_group'] == 0
    assert actual[2]['tuning']['quantile_summary_batches'] == 0
    assert actual[2]['tuning']['quantile_weighted_summary_stages'] == 3
    assert actual[2]['tuning']['quantile_weighted_summary_batches'] > 0


@pytest.mark.parametrize('adapter', ['dense', 'csr', 'dataframe', 'iterator'])
def test_summary_input_adapters(monkeypatch, adapter):
    X, y = inputs(2053)
    actual = compare(monkeypatch, X, y, 257, True, adapter=adapter)
    assert actual[2]['tuning']['quantile_summary_group'] == 256


@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('mixed', [False, True])
def test_category_queues_and_summary_dispatch(monkeypatch, quantile, mixed):
    X, y = inputs(1027)
    types = ['c'] * X.shape[1]
    X = np.nan_to_num(X * 7, nan=0).astype(np.int32).astype(np.float32)
    X -= X.min(axis=0)
    X[:, 2] = np.resize(np.array([0, 3, 70003], np.float32), len(X))
    if mixed:
        types[0] = 'q'
        X[:, 0] = np.random.default_rng(83).permutation(len(X)).astype(np.float32)
    actual = compare(monkeypatch, X, y, 257, quantile, feature_types=types, params=dict(max_cat_to_onehot=2))
    assert actual[2]['tuning']['quantile_summary_group'] == (256 if mixed else 0)


@pytest.mark.parametrize('serial_sketch', ['0', '1'])
@pytest.mark.parametrize('serial_cuts', ['0', '1'])
@pytest.mark.parametrize('quantile', [False, True])
def test_summary_reducer_and_cut_controls_compose(monkeypatch, serial_sketch, serial_cuts, quantile):
    monkeypatch.setenv('METALGB_SERIAL_SKETCH', serial_sketch)
    monkeypatch.setenv('METALGB_SERIAL_CUTS', serial_cuts)
    X, y = inputs(2053)
    actual = compare(monkeypatch, X, y, 257, quantile)
    assert actual[2]['tuning']['quantile_summary_group'] == 256
    assert actual[2]['tuning']['quantile_cut_tile'] == (256 if serial_cuts == '0' else 0)


@pytest.mark.parametrize('quantile', [False, True])
def test_summary_metadata_follows_materialized_cuts(monkeypatch, quantile):
    X, y = inputs(2053)
    monkeypatch.setenv('METALGB_SERIAL_SUMMARY', '0')
    data = make_matrix(X, y, 257, quantile)
    first = snapshot(data, dict(max_bin=257, max_depth=2, base_score=0))
    assert first[2]['tuning']['quantile_summary_group'] == 256
    monkeypatch.setenv('METALGB_SERIAL_SUMMARY', '1')
    data.set_weight(np.linspace(0.1, 2, len(X), dtype=np.float32))
    second = snapshot(data, dict(max_bin=257, max_depth=2, base_score=0))
    assert second[2]['tuning']['quantile_summary_group'] == 256
    for own, expected in zip(second[1][:2], first[1][:2]):
        np.testing.assert_array_equal(own, expected)
    query = xgb.QuantileDMatrix(X[:67], label=y[:67], ref=data, max_bin=257)
    borrowed = snapshot(query, dict(max_bin=257, max_depth=2, base_score=0))
    assert borrowed[2]['tuning']['quantile_summary_group'] == 256
    assert borrowed[2]['tuning']['quantile_summary_batches'] == first[2]['tuning']['quantile_summary_batches']
    if quantile:
        with pytest.raises(xgb.core.XGBoostError, match='Inconsistent max_bin'):
            xgb.train(dict(max_bin=513, base_score=0), data, 1)
    else:
        data.set_weight(np.array([], np.float32))
        third = snapshot(data, dict(max_bin=513, max_depth=2, base_score=0))
        assert third[2]['tuning']['quantile_summary_group'] == 0
        monkeypatch.setenv('METALGB_SERIAL_SUMMARY', '0')
        fourth = snapshot(data, dict(max_bin=65, max_depth=2, base_score=0))
        assert fourth[2]['tuning']['quantile_summary_group'] == 256


@pytest.mark.parametrize('strategy', ['one_output_per_tree', 'multi_output_tree'])
@pytest.mark.parametrize('quantile', [False, True])
def test_summary_models_match_cpu(monkeypatch, reference, strategy, quantile):
    from test_large_bins import check_model
    X, y = inputs(4099)
    y = np.column_stack([y, X[:, 0], X[:, 3]]).astype(np.float32)
    params = dict(max_bin=4096, max_depth=3, base_score=0, seed=67, multi_strategy=strategy)
    expected, out = reference(X, y, params, rounds=2, quantile=quantile, cuts=True)
    monkeypatch.setenv('METALGB_SERIAL_SUMMARY', '0')
    data = make_matrix(X, y, 4096, quantile)
    model = xgb.train(params, data, 2)
    assert profile(model)['tuning']['quantile_summary_group'] == 256
    check_model(json.loads(model.save_raw(raw_format='json')), json.loads((out / 'model.json').read_text()))
    for options, key in [({}, 'prediction'), ({'output_margin': True}, 'margin'), ({'pred_leaf': True}, 'leaf')]:
        np.testing.assert_array_equal(model.predict(data, **options).view(np.uint32), expected[key].view(np.uint32))
    with np.load(out / 'cuts.npz') as expected_cuts:
        ptr, values = data.get_quantile_cut()
        np.testing.assert_array_equal(ptr, expected_cuts['indptr'])
        np.testing.assert_array_equal(values.view(np.uint32), expected_cuts['values'].view(np.uint32))
