"""Bin requests remain part of cache identity even when storage sizes match."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from large_bin_lifecycle_trace import CASES, REFERENCE_CASES, case_name, reference_case, run_case
from test_large_bins import check_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def lifecycle_reference(tmp_path_factory):
    output = tmp_path_factory.mktemp('large-bin-lifecycle') / 'trace.json'
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'ref/xgboost/python-package')}
    completed = subprocess.run([sys.executable, str(ROOT / 'tests/large_bin_lifecycle_trace.py'),
                                '--cpu-only', '--output', str(output)], env=environment,
                               capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    with np.load(output.with_suffix('.npz')) as arrays:
        yield json.loads(output.read_text()), arrays


@pytest.mark.parametrize('case', CASES, ids=case_name)
def test_bin_request_lifetime(lifecycle_reference, case):
    reference, expected_arrays = lifecycle_reference
    name = case_name(case)
    actual, arrays = run_case(*case)
    expected = reference['cases'][name]
    assert len(actual) == len(expected)
    for own, wanted in zip(actual, expected):
        assert own.keys() == wanted.keys()
        for field in own:
            if field == 'model':
                check_model(own[field], wanted[field])
            else:
                assert own[field] == wanted[field], field
    assert {key for key in expected_arrays.files if key.startswith(name + '/')} == {
        name + '/' + key for key in arrays}
    for field, own in arrays.items():
        wanted = expected_arrays[name + '/' + field]
        assert own.shape == wanted.shape, field
        if own.dtype == np.float32:
            own, wanted = own.view(np.uint32), wanted.view(np.uint32)
        np.testing.assert_array_equal(own, wanted, err_msg=field)


@pytest.mark.parametrize('case', REFERENCE_CASES, ids=case_name)
def test_reference_requires_same_requested_bins(lifecycle_reference, case):
    expected = lifecycle_reference[0]['reference'][case_name(case)]
    assert reference_case(*case) == expected
