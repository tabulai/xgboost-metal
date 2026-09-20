from pathlib import Path
import json
import os
import subprocess
import sys

import pytest
import xgboost as xgb

from matrix_binary_training_trace import CASES, run_case
from test_large_bins import check_model

ROOT = Path(__file__).resolve().parents[1]


def reference_trace(destination, case=None, source=None):
    command = [sys.executable, str(ROOT / 'tests/matrix_binary_training_trace.py'), '--output', str(destination)]
    if case:
        command += ['--case', case]
    if source:
        command += ['--input', str(source)]
    completed = subprocess.run(command, env={**os.environ, 'PYTHONPATH': str(ROOT / 'ref/xgboost/python-package')},
                               capture_output=True, text=True, timeout=90)
    destination.parent.joinpath(destination.name + '.stdout').write_text(completed.stdout)
    destination.parent.joinpath(destination.name + '.stderr').write_text(completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads((destination / 'report.json').read_text())


@pytest.fixture(scope='module')
def training_reference(tmp_path_factory):
    destination = tmp_path_factory.mktemp('matrix-binary-training-reference') / 'fixtures'
    return destination, reference_trace(destination)


def check_trace(actual, expected):
    assert actual['before_sha256'] == expected['before_sha256']
    assert actual['query_sha256'] == expected['query_sha256']
    assert len(actual['stages']) == len(expected['stages'])
    for own, wanted in zip(actual['stages'], expected['stages']):
        check_model(own['model'], wanted['model'])
        assert own['arrays'] == wanted['arrays']


@pytest.mark.parametrize('case', CASES)
def test_binary_training_and_continuation(training_reference, tmp_path, case):
    files, expected = training_reference
    for origin in ['constructed', 'imported']:
        matrix = xgb.DMatrix(files / case / 'before.buffer', silent=True) if origin == 'imported' else None
        actual = run_case(case, tmp_path / origin, matrix)
        check_trace(actual, expected[case])
    # Run actual CPU training on the Metal file as well as the reverse import.
    loaded = reference_trace(tmp_path / 'cpu-reloaded', case, tmp_path / 'constructed/before.buffer')
    check_trace(loaded[case], expected[case])
