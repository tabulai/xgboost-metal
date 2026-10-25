"""CPU/Metal DMatrix file exchange, including tensor and sparse storage lifetimes."""
from pathlib import Path
import ctypes
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from matrix_binary_trace import CASES, capture, make_matrix, record_steps

ROOT = Path(__file__).resolve().parents[1]


def reference_trace(destination, case=None, source=None):
    command = [sys.executable, str(ROOT / 'tests/matrix_binary_trace.py'), '--output', str(destination)]
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
def binary_reference(tmp_path_factory):
    destination = tmp_path_factory.mktemp('matrix-binary-reference') / 'fixtures'
    return destination, reference_trace(destination)


@pytest.mark.parametrize('case', CASES)
def test_binary_exchange_and_metadata_lifecycle(binary_reference, tmp_path, case):
    files, expected = binary_reference
    # Test native construction as well as native loading of the CPU wire data.
    for origin, matrix in [('constructed', make_matrix(case)),
                           ('imported', xgb.DMatrix(files / case / 'before.buffer', silent=True))]:
        actual = record_steps(matrix, tmp_path / origin)
        assert actual == expected[case]
        for step in actual:
            assert (tmp_path / origin / (step + '.buffer')).read_bytes() == (files / case / (step + '.buffer')).read_bytes()
    # Exercise the CPU loader on an actual Metal-produced file.
    loaded = reference_trace(tmp_path / 'cpu-reloaded', case, tmp_path / 'constructed/before.buffer')
    assert loaded[case] == expected[case]


def test_binary_filename_precedes_uri_options(binary_reference, tmp_path):
    files, expected = binary_reference
    filename = tmp_path / 'features?format=csv&label_column=0'
    shutil.copyfile(files / 'metadata/before.buffer', filename)
    matrix = xgb.DMatrix(filename, silent=True)
    wanted = dict(expected['metadata']['before'])
    wanted.pop('binary_sha256')
    assert capture(matrix) == wanted
    # The removed external-memory suffix is rejected before opening a file,
    # even when the exact filename exists and contains valid binary data.
    rejected = tmp_path / 'features#cache'
    shutil.copyfile(filename, rejected)
    with pytest.raises(xgb.core.XGBoostError, match='External memory training with text input has been removed'):
        xgb.DMatrix(rejected, silent=True)


@pytest.mark.parametrize('verbosity', [1, 2])
@pytest.mark.parametrize('silent', [False, True])
def test_binary_load_information_respects_logging(binary_reference, capsys, verbosity, silent):
    files, _ = binary_reference
    filename = files / 'dense/before.buffer'
    handle = ctypes.c_void_p()
    config = json.dumps(dict(uri=str(filename), silent=int(silent))).encode()
    try:
        with xgb.config_context(verbosity=verbosity):
            xgb.core._check_call(xgb.core._LIB.XGDMatrixCreateFromURI(config, ctypes.byref(handle)))
        rows = ctypes.c_uint64()
        xgb.core._check_call(xgb.core._LIB.XGDMatrixNumRow(handle, ctypes.byref(rows)))
        assert rows.value == 6
    finally:
        if handle.value:
            xgb.core._check_call(xgb.core._LIB.XGDMatrixFree(handle))
    captured = capsys.readouterr()
    message = '6x4 matrix with 23 entries loaded from '
    assert (message in captured.out + captured.err) == (verbosity == 2 and not silent)
    # The pinned Python URI adapter omits the silent argument, even when the
    # deprecated DMatrix constructor argument is explicitly false.
    with xgb.config_context(verbosity=2):
        matrix = xgb.DMatrix(filename, silent=False)
    assert matrix.num_row() == 6
    captured = capsys.readouterr()
    assert message not in captured.out + captured.err


@pytest.mark.parametrize('dtype', ['object', 'int32', 'uint16'])
@pytest.mark.parametrize('rows', [0, 4])
@pytest.mark.parametrize('quantile', [False, True])
def test_empty_category_dictionary_is_rejected(dtype, rows, quantile):
    frame = pd.DataFrame({'empty': pd.Categorical.from_codes(np.full(rows, -1), pd.Index([], dtype=dtype)),
                          'numeric': np.arange(rows, dtype=np.float32)})
    constructor = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    with pytest.raises(xgb.core.XGBoostError, match='Categorical feature must have at least one category'):
        constructor(frame, enable_categorical=True)


class Batches(xgb.DataIter):
    def __init__(self, shapes):
        super().__init__(cache_prefix=None)
        self.shapes, self.position = shapes, 0

    def reset(self):
        self.position = 0

    def next(self, input_data):
        if self.position == len(self.shapes):
            return False
        rows, columns = self.shapes[self.position]
        self.position += 1
        values = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        arguments = dict(data=values)
        if columns is not None:
            labels = np.arange(rows * columns, dtype=np.float32).reshape(rows, columns)
            arguments.update(label=labels, base_margin=labels)
        input_data(**arguments)
        return True


@pytest.mark.parametrize('kind', ['quantile', 'iterator', 'proxy'])
def test_binary_save_rejects_non_simple_without_touching_destination(tmp_path, kind):
    if kind == 'quantile':
        matrix = xgb.QuantileDMatrix(np.arange(18, dtype=np.float32).reshape(6, 3))
    elif kind == 'iterator':
        matrix = xgb.DMatrix(Batches([(3, 2)]))
    else:
        matrix = xgb.core._ProxyDMatrix()
    destination = tmp_path / 'existing.buffer'
    destination.write_bytes(b'keep existing data')
    for filename in [destination, tmp_path / 'absent.buffer']:
        with pytest.raises(xgb.core.XGBoostError, match='binary saving only supported by SimpleDMatrix'):
            matrix.save_binary(filename, silent=True)
    assert destination.read_bytes() == b'keep existing data'
    assert not (tmp_path / 'absent.buffer').exists()


@pytest.mark.parametrize('shapes', [[(0, 2), (3, 2)], [(3, 2), (0, 2)],
                                    [(0, None), (3, 2)], [(0, 2), (0, 2)]])
def test_iterator_tensor_shapes_survive_empty_batches(shapes):
    matrix = xgb.DMatrix(Batches(shapes))
    rows = sum(rows for rows, _ in shapes)
    assert matrix.get_label().shape == (rows, 2)
    assert matrix.get_base_margin().shape == (rows, 2)
    if rows:
        expected = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
        np.testing.assert_array_equal(matrix.get_label(), expected)
        np.testing.assert_array_equal(matrix.get_base_margin(), expected)


@pytest.mark.parametrize('shapes', [[(0, 3), (3, 2)], [(3, 2), (0, 3)]])
def test_iterator_rejects_inconsistent_tensor_columns(shapes):
    with pytest.raises(xgb.core.XGBoostError):
        xgb.DMatrix(Batches(shapes))


@pytest.mark.parametrize('corruption', ['short-header', 'truncated-data', 'huge-name', 'huge-vector',
                                        'bad-version', 'bad-field', 'bad-offset', 'bad-index'])
def test_binary_rejects_corrupt_input(binary_reference, tmp_path, corruption):
    files, _ = binary_reference
    content = bytearray((files / 'dense/before.buffer').read_bytes())
    if corruption == 'short-header':
        content = content[:5]
    elif corruption == 'truncated-data':
        content = content[:-1]
    elif corruption == 'huge-name':
        struct.pack_into('<Q', content, 32, 2**64 - 1)
    elif corruption == 'bad-version':
        struct.pack_into('<i', content, 12, 2)
    elif corruption == 'bad-field':
        content[40] = ord('!')
    else:
        # Locate CSR independently: this fixture has 23 entries and 7 offsets.
        entries_begin = len(content) - 23 * 8
        offsets_begin = entries_begin - 8 - 7 * 8
        if corruption == 'huge-vector':
            struct.pack_into('<Q', content, entries_begin - 8, 2**64 - 1)
        elif corruption == 'bad-offset':
            struct.pack_into('<Q', content, offsets_begin + 8, 24)
        else:
            struct.pack_into('<I', content, entries_begin, 4)
    filename = tmp_path / (corruption + '.buffer')
    filename.write_bytes(content)
    with pytest.raises(xgb.core.XGBoostError):
        xgb.DMatrix(filename, silent=True)


def field_layout(content):
    """Locate the independently CPU-written field headers, without the codec."""
    position, fields = 32, {}
    for _ in range(struct.unpack_from('<Q', content, 24)[0]):
        length = struct.unpack_from('<Q', content, position)[0]
        position += 8
        name = content[position:position + length].decode()
        position += length
        dtype, scalar = content[position:position + 2]
        entry = dict(type=position, scalar=position + 1)
        position += 2
        if scalar:
            entry['values'] = position
            position += 8
        else:
            entry['shape'] = position
            position += 16
            entry['count'] = position
            count = struct.unpack_from('<Q', content, position)[0]
            position += 8
            entry['values'] = position
            if dtype == 5 and name != 'cats':
                for _ in range(count):
                    position += 8 + struct.unpack_from('<Q', content, position)[0]
            else:
                position += count * {1: 4, 3: 4, 4: 8, 5: 1}[dtype]
        entry['end'] = position
        fields[name] = entry
    return fields, position


VECTOR_FIELDS = ['labels', 'base_margin', 'group_ptr', 'weights', 'labels_lower_bound',
                 'labels_upper_bound', 'feature_names', 'feature_types', 'feature_weights', 'cats']


@pytest.mark.parametrize('field', VECTOR_FIELDS)
@pytest.mark.parametrize('part', ['type', 'scalar', 'shape', 'count'])
def test_binary_bounds_every_metadata_field(binary_reference, tmp_path, field, part):
    files, _ = binary_reference
    content = bytearray((files / 'metadata/before.buffer').read_bytes())
    fields, _ = field_layout(content)
    offset = fields[field][part]
    if part in ['type', 'scalar']:
        content[offset] ^= 1
    elif part == 'shape':
        struct.pack_into('<QQ', content, offset, 2**64 - 1, 2)
    else:
        struct.pack_into('<Q', content, offset, 2**64 - 1)
    filename = tmp_path / 'corrupt.buffer'
    filename.write_bytes(content)
    with pytest.raises(xgb.core.XGBoostError):
        xgb.DMatrix(filename, silent=True)


@pytest.mark.parametrize('corruption', ['short', 'negative'])
def test_binary_validates_feature_weights(tmp_path, corruption):
    # Well-formed vectors of the wrong length or sign reached column sampling,
    # which indexes feature weights by feature id.
    matrix = xgb.DMatrix(np.arange(24, dtype=np.float32).reshape(6, 4), label=np.arange(6, dtype=np.float32))
    matrix.set_info(feature_weights=np.arange(1, 5, dtype=np.float32))
    filename = tmp_path / 'weights.buffer'
    matrix.save_binary(filename)
    content = bytearray(filename.read_bytes())
    entry = field_layout(content)[0]['feature_weights']
    if corruption == 'short':
        struct.pack_into('<QQQ', content, entry['shape'], 1, 1, 1)
        del content[entry['values'] + 4:entry['end']]
    else:
        struct.pack_into('<f', content, entry['values'] + 8, -1)
    filename.write_bytes(content)
    with pytest.raises(xgb.core.XGBoostError, match='feature weights'):
        xgb.DMatrix(filename, silent=True)


@pytest.mark.parametrize('variant', ['unsorted', 'float-bits', 'future-major', 'more-fields',
                                    'trailing', 'category-search-index'])
def test_binary_accepted_payloads_follow_cpu_canonicalization(binary_reference, tmp_path, variant):
    files, _ = binary_reference
    case = 'category-uint16' if variant == 'category-search-index' else 'dense'
    content = bytearray((files / case / 'before.buffer').read_bytes())
    fields, csr_begin = field_layout(content)
    offsets = struct.unpack_from('<Q', content, csr_begin)[0]
    entries_begin = csr_begin + 8 + offsets * 8 + 8
    if variant == 'unsorted':
        content[entries_begin:entries_begin + 8], content[entries_begin + 16:entries_begin + 24] = (
            content[entries_begin + 16:entries_begin + 24], content[entries_begin:entries_begin + 8])
    elif variant == 'float-bits':
        for i, bits in enumerate([0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345]):
            struct.pack_into('<I', content, entries_begin + 8 * i + 4, bits)
    elif variant == 'future-major':
        struct.pack_into('<i', content, 12, 4)
    elif variant == 'more-fields':
        struct.pack_into('<Q', content, 24, 14)
    elif variant == 'trailing':
        content += b'ignored trailing bytes\x00\xff'
    else:
        field = fields['cats']
        payload = content[field['values']:field['end']]
        suffix = b'L' + struct.pack('>Q', 10) + b'sorted_idx[$l#L' + struct.pack('>Q', 0) + b'}'
        assert payload.endswith(suffix)
        index = [0, 2, 3, 1, 0, 1]
        replacement = payload[:-9] + struct.pack('>Q', len(index)) + struct.pack('>6i', *index) + b'}'
        content[field['values']:field['end']] = replacement
        struct.pack_into('<Q', content, field['shape'], len(replacement))
        struct.pack_into('<Q', content, field['count'], len(replacement))
    source = tmp_path / 'source.buffer'
    source.write_bytes(content)
    expected = reference_trace(tmp_path / 'cpu', case, source)[case]
    actual = record_steps(xgb.DMatrix(source, silent=True), tmp_path / 'metal')
    assert actual == expected
    for step in actual:
        assert (tmp_path / 'metal' / (step + '.buffer')).read_bytes() == (tmp_path / 'cpu' / case / (step + '.buffer')).read_bytes()
