"""Public DMatrix binary observations, run separately with the pinned CPU library."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb


DTYPES = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'string']
SHAPES = {'absent': None, 'flat-empty': (0,), 'empty-two': (0, 2),
          'three-zero': (3, 0), 'five-zero': (5, 0), 'flat-six': (6,),
          'two-three': (2, 3), 'three-two': (3, 2)}
CASES = ['dense', 'metadata', 'queries', 'empty', 'empty-multioutput', 'no-features',
         'csr-duplicates', 'csc-duplicates', 'sparse-zeros', 'empty-category-rows', 'all-missing-category']
CASES += ['category-' + dtype for dtype in DTYPES]
CASES += ['shape-' + str(rows) + '-' + name for rows in [0, 3] for name in SHAPES]


def make_matrix(case):
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    values[0, 0] = -0.0
    values[1, 1] = np.nextafter(np.float32(0), np.float32(1))
    values[2, 2] = np.nan
    label = np.array([-0.0, 1, 2, -3, 5, 7], np.float32)
    if case == 'dense':
        return xgb.DMatrix(values)
    if case == 'metadata':
        return xgb.DMatrix(values, label=np.column_stack([label, -label]),
                           base_margin=np.column_stack([label, -label, label / 4]),
                           weight=[1, 0, 2, 4, 1, -0.0],
                           label_lower_bound=[-np.inf, 0, 2, 1, 3, 7],
                           label_upper_bound=[np.inf, 0, 2, 3, 10, 7],
                           feature_names=['a', '茶', '', 'last'],
                           feature_types=['q', 'float', 'i', 'int'],
                           feature_weights=[1, -0.0, 2, 4])
    if case == 'queries':
        return xgb.DMatrix(values, label=label, group=[2, 0, 4], weight=[1, -0.0, 2])
    if case == 'empty':
        return xgb.DMatrix(np.empty((0, 3), np.float32))
    if case == 'empty-multioutput':
        return xgb.DMatrix(np.empty((0, 3), np.float32),
                           label=np.empty((0, 2), np.float32),
                           base_margin=np.empty((0, 3), np.float32))
    if case == 'no-features':
        return xgb.DMatrix(np.empty((3, 0), np.float32), label=[0, 1, 2])
    if case in ['csr-duplicates', 'csc-duplicates']:
        args = (np.array([1, 2, 3, -0.0, 4], np.float32),
                np.array([2, 0, 2, 1, 0], np.int32), np.array([0, 3, 5, 5], np.int32))
        data = (sparse.csr_matrix(args, shape=(3, 4)) if case == 'csr-duplicates'
                else sparse.csc_matrix(args, shape=(4, 3)))
        return xgb.DMatrix(data, label=np.arange(data.shape[0], dtype=np.float32))
    if case == 'sparse-zeros':
        data = sparse.csr_matrix((np.array([-0.0, 0.0, np.nan, 4], np.float32),
                                 np.array([0, 1, 3, 2]), np.array([0, 2, 3, 4])), shape=(3, 5))
        return xgb.DMatrix(data)
    if case in ['empty-category-rows', 'all-missing-category']:
        rows = 0 if case == 'empty-category-rows' else 6
        frame = pd.DataFrame({'kind': pd.Categorical.from_codes(np.full(rows, -1), ['', '茶']),
                              'numeric': np.arange(rows, dtype=np.float32)})
        return xgb.DMatrix(frame, enable_categorical=True)
    if case.startswith('category-'):
        dtype = case[len('category-'):]
        if dtype == 'string':
            names = ['茶', '', 'pear', 'é']
        else:
            limits = np.iinfo(dtype)
            names = [limits.min, min(limits.max, 2**63 - 1), 1, 7]
            names = np.asarray(names, dtype=dtype)
        frame = pd.DataFrame({'kind': pd.Categorical.from_codes([0, 1, -1, 2, 3, 0], names),
                              'numeric': values[:, 0],
                              'other': pd.Categorical(['x', 'y', 'x', None, 'y', 'x'])})
        return xgb.DMatrix(frame, label=label, enable_categorical=True)
    if case.startswith('shape-'):
        _, rows, name = case.split('-', 2)
        rows, shape = int(rows), SHAPES[name]
        matrix = xgb.DMatrix(np.arange(rows * 3, dtype=np.float32).reshape(rows, 3))
        if shape is not None:
            value = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            if value.size:
                value.reshape(-1)[0] = -0.0
            matrix.set_info(label=value, base_margin=value)
        return matrix
    raise ValueError(case)


def capture(matrix):
    csr = matrix.get_data()
    metadata = {}
    for name in ['label', 'weight', 'base_margin', 'label_lower_bound',
                 'label_upper_bound', 'feature_weights']:
        value = matrix.get_float_info(name)
        metadata[name] = dict(shape=list(value.shape), bits=value.view(np.uint32).reshape(-1).tolist())
    categories = matrix.get_categories(export_to_arrow=True).to_arrow()
    return dict(rows=matrix.num_row(), columns=matrix.num_col(), nonmissing=matrix.num_nonmissing(),
                feature_names=matrix.feature_names, feature_types=matrix.feature_types,
                group_ptr=matrix.get_uint_info('group_ptr').tolist(), metadata=metadata,
                indptr=csr.indptr.tolist(), indices=csr.indices.tolist(),
                data_bits=csr.data.view(np.uint32).tolist(),
                categories=None if categories is None else [
                    dict(name=name, dtype=None if values is None else str(values.type),
                         values=None if values is None else values.to_pylist())
                    for name, values in categories])


def record_steps(matrix, destination):
    destination.mkdir()
    report = {}

    def record(name, value):
        snapshot = capture(value)
        filename = destination / (name + '.buffer')
        value.save_binary(filename, silent=True)
        snapshot['binary_sha256'] = hashlib.sha256(filename.read_bytes()).hexdigest()
        report[name] = snapshot

    record('before', matrix)
    # The pinned CPU implementation divides by zero when slicing nonempty
    # margins attached to a zero-row matrix; that undefined path is not invoked.
    if matrix.num_row() or not matrix.get_base_margin().size:
        record('empty-slice', matrix.slice([], allow_groups=True))
        if matrix.num_row() >= 3:
            record('row-slice', matrix.slice([2, 0, 2], allow_groups=True))
    rows, cols = matrix.num_row(), matrix.num_col()
    # Keep zero-row, nonempty metadata nonempty. Clearing an allocated CPU
    # tensor to (0, 2) leaves a pointer that its array-interface exporter rejects.
    edit_rows = rows if rows or not matrix.get_label().size else 3
    labels = np.arange(edit_rows * 2, dtype=np.float32).reshape(edit_rows, 2) / 4
    if labels.size:
        labels[0, 0] = -0.0
    matrix.set_info(label=labels, base_margin=-labels,
                    feature_weights=np.arange(cols, dtype=np.float32) / 4)
    record('edited', matrix)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assert Path(xgb.__file__).resolve().is_relative_to(root / 'ref/xgboost')
    from reference_identity import check_reference
    check_reference(root, xgb)
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    report = {}
    for case in ([args.case] if args.case else CASES):
        matrix = xgb.DMatrix(args.input, silent=True) if args.input else make_matrix(case)
        report[case] = record_steps(matrix, args.output / case)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
