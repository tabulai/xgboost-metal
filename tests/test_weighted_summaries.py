"""Separating weighted tie repair preserves cuts, model bytes and command counts."""
import numpy as np
import pytest
import xgboost as xgb

from test_batch_summaries import compare, inputs
from test_cut_tiles import make_matrix, snapshot


def assert_stages(result, active):
    tuning = result[2]['tuning']
    assert tuning['quantile_weighted_summary_stages'] == (3 if active else 0)
    assert (tuning['quantile_weighted_summary_batches'] > 0) == active
    # Explicit unit weights still require ordered float32 summary arithmetic.
    assert tuning['quantile_summary_group'] == tuning['quantile_summary_batches'] == 0


@pytest.mark.parametrize('rows', [0, 1, 2, 31, 255, 256, 257, 513, 4099, 70003])
@pytest.mark.parametrize('quantile', [False, True])
def test_weighted_dispatch_boundaries(monkeypatch, rows, quantile):
    X, y = inputs(rows)
    result = compare(monkeypatch, X, y, 65536, quantile,
                     weight=np.linspace(0.01, 2, rows, dtype=np.float32))
    assert_stages(result, rows >= 256)


@pytest.mark.parametrize('adapter', ['dense', 'csr', 'dataframe', 'iterator'])
def test_weighted_input_adapters(monkeypatch, adapter):
    X, y = inputs(2053)
    result = compare(monkeypatch, X, y, 257, True, adapter=adapter,
                     weight=np.linspace(0.01, 2, len(X), dtype=np.float32))
    assert_stages(result, True)


@pytest.mark.parametrize('kind', ['zero-weight', 'missing'])
@pytest.mark.parametrize('quantile', [False, True])
def test_empty_weighted_queues_skip_summary_stages(monkeypatch, kind, quantile):
    X, y = inputs(1027)
    weight = np.ones(len(X), np.float32)
    if kind == 'missing':
        X.fill(np.nan)
    else:
        weight.fill(0)
    result = compare(monkeypatch, X, y, 257, quantile, weight=weight)
    assert_stages(result, False)


@pytest.mark.parametrize('serial_sketch', ['0', '1'])
@pytest.mark.parametrize('serial_cuts', ['0', '1'])
@pytest.mark.parametrize('quantile', [False, True])
def test_weighted_flags_are_consumed_before_both_reducers(monkeypatch, serial_sketch, serial_cuts, quantile):
    monkeypatch.setenv('METALGB_SERIAL_SKETCH', serial_sketch)
    monkeypatch.setenv('METALGB_SERIAL_CUTS', serial_cuts)
    X, y = inputs(4099)
    X[:, 2] = np.resize(np.array([-0.0, 1, 0.0, 2], np.float32), len(X))
    result = compare(monkeypatch, X, y, 257, quantile,
                     weight=np.linspace(0.01, 2, len(X), dtype=np.float32))
    assert_stages(result, True)


@pytest.mark.parametrize('quantile', [False, True])
@pytest.mark.parametrize('mixed', [False, True])
def test_weighted_categories_only_schedule_numeric_batches(monkeypatch, quantile, mixed):
    X, y = inputs(1027)
    X = np.column_stack([np.arange(len(X)) % 7, np.arange(len(X)) % 43,
                         np.resize(np.array([0, 3, 70003]), len(X))]).astype(np.float32)
    types = ['c', 'c', 'c']
    if mixed:
        X[:, 0] = np.random.default_rng(241).permutation(len(X)).astype(np.float32)
        types[0] = 'q'
    result = compare(monkeypatch, X, y, 257, quantile, feature_types=types,
                     params=dict(max_cat_to_onehot=2), weight=np.linspace(0.01, 2, len(X), dtype=np.float32))
    assert_stages(result, mixed)


@pytest.mark.parametrize('quantile', [False, True])
def test_weighted_dispatch_metadata_follows_materialized_cuts(monkeypatch, quantile):
    X, y = inputs(2053)
    monkeypatch.setenv('METALGB_SERIAL_SUMMARY', '0')
    data = make_matrix(X, y, 257, quantile, weight=np.linspace(0.01, 2, len(X), dtype=np.float32))
    first = snapshot(data, dict(max_bin=257, max_depth=2, base_score=0))
    assert_stages(first, True)
    monkeypatch.setenv('METALGB_SERIAL_SUMMARY', '1')
    data.set_weight(np.ones(len(X), np.float32))
    second = snapshot(data, dict(max_bin=257, max_depth=2, base_score=0))
    assert_stages(second, True)
    for own, expected in zip(second[1][:2], first[1][:2]):
        np.testing.assert_array_equal(own, expected)
    query = xgb.QuantileDMatrix(X[:67], label=y[:67], ref=data, max_bin=257)
    borrowed = snapshot(query, dict(max_bin=257, max_depth=2, base_score=0))
    assert_stages(borrowed, True)
    assert (borrowed[2]['tuning']['quantile_weighted_summary_batches'] ==
            first[2]['tuning']['quantile_weighted_summary_batches'])
    if quantile:
        with pytest.raises(xgb.core.XGBoostError, match='Inconsistent max_bin'):
            xgb.train(dict(max_bin=513, base_score=0), data, 1)
    else:
        rebuilt = snapshot(data, dict(max_bin=513, max_depth=2, base_score=0))
        assert_stages(rebuilt, False)
        monkeypatch.setenv('METALGB_SERIAL_SUMMARY', '0')
        data.set_weight(np.array([], np.float32))
        unweighted = snapshot(data, dict(max_bin=65, max_depth=2, base_score=0))
        assert_stages(unweighted, False)
