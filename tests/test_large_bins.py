"""Requested bin counts and physical bin indices have independent limits."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from large_bin_trace import CASES, PARAMETERS, case_name, parameter_case, run_case

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def large_bin_reference(tmp_path_factory):
    output = tmp_path_factory.mktemp('large-bin-reference') / 'trace.json'
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'ref/xgboost/python-package')}
    completed = subprocess.run([sys.executable, str(ROOT / 'tests/large_bin_trace.py'),
                                '--cpu-only', '--output', str(output)], env=environment,
                               capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with np.load(output.with_suffix('.npz')) as arrays:
        yield json.loads(output.read_text()), arrays


def check_model(actual, expected):
    actual, expected = [model['learner'] for model in [actual, expected]]
    for field in ['attributes', 'feature_names', 'feature_types', 'objective']:
        assert actual[field] == expected[field], field
    own, wanted = actual['learner_model_param'].copy(), expected['learner_model_param'].copy()
    own_base, wanted_base = json.loads(own.pop('base_score')), json.loads(wanted.pop('base_score'))
    np.testing.assert_array_equal(np.asarray(own_base, np.float32).view(np.uint32),
                                  np.asarray(wanted_base, np.float32).view(np.uint32))
    assert own == wanted
    actual, expected = [model['gradient_booster']['model'] for model in [actual, expected]]
    assert actual.keys() == expected.keys()
    for field in actual:
        if field != 'trees':
            assert actual[field] == expected[field], field
    assert len(actual['trees']) == len(expected['trees'])
    floating = {'base_weights', 'leaf_weights', 'split_conditions', 'sum_hessian', 'loss_changes'}
    for index, (own, wanted) in enumerate(zip(actual['trees'], expected['trees'])):
        assert own.keys() == wanted.keys()
        for field in own:
            if field in floating:
                np.testing.assert_array_equal(np.asarray(own[field], np.float32).view(np.uint32),
                                              np.asarray(wanted[field], np.float32).view(np.uint32),
                                              err_msg=f'tree {index}: {field}')
            else:
                assert own[field] == wanted[field], (index, field)


@pytest.mark.parametrize('case', CASES, ids=case_name)
def test_large_numeric_bins(large_bin_reference, case):
    reference, expected_arrays = large_bin_reference
    name = case_name(case)
    expected = reference['cases'][name]
    assert 'error' not in expected, expected
    actual, arrays = run_case(*case)
    check_model(actual['model'], expected['model'])
    own, wanted = [entry['config']['learner'] for entry in [actual, expected]]
    assert own['generic_param']['rng_state'] == wanted['generic_param']['rng_state']
    own, wanted = [entry['gradient_booster']['tree_train_param']['max_bin'] for entry in [own, wanted]]
    assert own == wanted == str(case[1])
    assert actual['array_shapes'] == expected['array_shapes']
    for field, own in arrays.items():
        wanted = expected_arrays[name + '/' + field]
        assert own.shape == wanted.shape, field
        if own.dtype == np.float32:
            own, wanted = own.view(np.uint32), wanted.view(np.uint32)
        np.testing.assert_array_equal(own, wanted, err_msg=field)


@pytest.mark.parametrize('index,value', list(enumerate(PARAMETERS)), ids=repr)
@pytest.mark.parametrize('quantile', [False, True])
def test_max_bin_parameter_format(large_bin_reference, index, value, quantile):
    expected = large_bin_reference[0]['parameters'][case_name((index, quantile))]
    assert parameter_case(value, quantile) == expected
