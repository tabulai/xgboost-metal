"""Ranking cache lifetimes are visible through mutable public metadata."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from ranking_metadata_trace import CASES, case_name, run_case
from ranking_lifecycle_trace import (DEFAULT_CASES, METRIC_CONFIG_CASES, UNBIASED_CASES, OWNERSHIP_CASES,
                                     default_trace, metric_config_trace, unbiased_trace, ownership_trace)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def ranking_metadata_reference(tmp_path_factory):
    output = tmp_path_factory.mktemp('ranking-metadata-reference') / 'trace.json'
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'ref/xgboost/python-package')}
    completed = subprocess.run([sys.executable, str(ROOT / 'tests/ranking_metadata_trace.py'),
                                '--cpu-only', '--output', str(output)],
                               env=environment, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(output.read_text())['cases'], np.load(output.with_suffix('.npz'))


@pytest.fixture(scope='module')
def ranking_lifecycle_reference(tmp_path_factory):
    output = tmp_path_factory.mktemp('ranking-lifecycle-reference') / 'trace.json'
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'ref/xgboost/python-package')}
    completed = subprocess.run([sys.executable, str(ROOT / 'tests/ranking_lifecycle_trace.py'),
                                '--cpu-only', '--output', str(output)],
                               env=environment, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(output.read_text())['cases']


def check_model(own, wanted):
    own, wanted = [entry['learner']['gradient_booster']['model'] for entry in [own, wanted]]
    assert own['iteration_indptr'] == wanted['iteration_indptr']
    assert own['tree_info'] == wanted['tree_info']
    assert len(own['trees']) == len(wanted['trees'])
    floating = {'base_weights', 'leaf_weights', 'split_conditions', 'sum_hessian', 'loss_changes'}
    for index, (a, b) in enumerate(zip(own['trees'], wanted['trees'])):
        assert a.keys() == b.keys()
        for field in a:
            if field in floating:
                np.testing.assert_allclose(np.asarray(a[field], np.float32), np.asarray(b[field], np.float32),
                                           rtol=3e-5, atol=3e-5, err_msg=f'tree {index}: {field}')
            else:
                assert a[field] == b[field], (index, field)


def check_metric(own, wanted):
    actual, expected = [{part.rsplit(':', 1)[0]: float(part.rsplit(':', 1)[1])
                         for part in text.split('\t')[1:]} for text in [own, wanted]]
    assert actual.keys() == expected.keys()
    np.testing.assert_allclose(list(actual.values()), list(expected.values()), rtol=2e-7, atol=2e-7)


def check_config(own, wanted):
    own, wanted = [config['learner'] for config in [own, wanted]]
    assert own['metrics'] == wanted['metrics']
    assert own['generic_param']['rng_state'] == wanted['generic_param']['rng_state']
    a, b = own['objective'], wanted['objective']
    assert a.keys() == b.keys()
    for key in a:
        if key in ['ti+', 'tj-']:
            np.testing.assert_allclose(a[key], b[key], rtol=2e-5, atol=2e-6)
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize('case', CASES, ids=case_name)
def test_ranking_metadata_lifetime(ranking_metadata_reference, case):
    cases, arrays = ranking_metadata_reference
    name = case_name(case)
    expected = cases[name]
    actual, predictions = run_case(*case)
    assert len(actual) == len(expected)
    for own, wanted in zip(actual, expected):
        assert own.get('iteration', own.get('event')) == wanted.get('iteration', wanted.get('event'))
        for field in ['update_error', 'metric_error']:
            assert (field in own) == (field in wanted), (name, own, wanted)
        for field in ['rng_state', 'metric_config', 'objective_config']:
            assert own.get(field) == wanted.get(field), (name, field)
        if 'model' in wanted:
            check_model(own['model'], wanted['model'])
            key = name + '/' + str(own['iteration'])
            np.testing.assert_allclose(predictions[key], arrays[key], rtol=3e-5, atol=3e-5)
        if 'metric' in wanted:
            check_metric(own['metric'], wanted['metric'])


@pytest.mark.parametrize('prefix,case,function',
    [('default', case, default_trace) for case in DEFAULT_CASES] +
    [('metric-config', case, metric_config_trace) for case in METRIC_CONFIG_CASES] +
    [('unbiased', case, unbiased_trace) for case in UNBIASED_CASES] +
    [('ownership', case, ownership_trace) for case in OWNERSHIP_CASES],
    ids=[f'default/{case_name(case)}' for case in DEFAULT_CASES] +
        [f'metric-config/{case_name(case)}' for case in METRIC_CONFIG_CASES] +
        [f'unbiased/{case_name(case)}' for case in UNBIASED_CASES] +
        [f'ownership/{case_name(case)}' for case in OWNERSHIP_CASES])
def test_ranking_cache_lifecycle(ranking_lifecycle_reference, prefix, case, function):
    expected = ranking_lifecycle_reference[prefix + '/' + case_name(case)]
    actual = function(*case)
    assert len(actual) == len(expected)
    for own, wanted in zip(actual, expected):
        assert own['event'] == wanted['event']
        assert ('error' in own) == ('error' in wanted), (own, wanted)
        if 'model' in wanted:
            check_model(own['model'], wanted['model'])
            np.testing.assert_allclose(own['margin'], wanted['margin'], rtol=3e-5, atol=3e-5)
        for field in ['config', 'config_after']:
            if field in wanted:
                check_config(own[field], wanted[field])
        for field in ['evaluation', 'metric']:
            if field in wanted:
                check_metric(own[field], wanted[field])
