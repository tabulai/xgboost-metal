import numpy as np
import pandas as pd
import pytest
import xgboost as xgb


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int8, np.int16,
                                  np.int32, np.int64, np.uint8, np.uint16, np.uint32,
                                  np.uint64, np.bool_])
def test_numeric_dtype_and_layout(dtype):
    rng = np.random.default_rng(711)
    X = rng.integers(0, 100, size=(71, 5)).astype(dtype)
    y = X[:, 0].astype(np.float32)
    model = xgb.train({'max_depth': 2}, xgb.DMatrix(X, label=y), 2)
    for view in [np.asfortranarray(X), X[::-1, ::-1], X[::2]]:
        expected = xgb.DMatrix(np.ascontiguousarray(view, dtype=np.float32))
        actual = xgb.DMatrix(view)
        np.testing.assert_array_equal(actual.get_data().toarray(), expected.get_data().toarray())
        np.testing.assert_array_equal(model.predict(actual), model.predict(expected))


@pytest.mark.parametrize("shape", [(0, 3), (1, 1), (31, 7), (32, 32), (33, 35), (65, 63)])
def test_columnar_transpose_tails_missing_and_ownership(shape):
    import pyarrow as pa
    original = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    if original.size:
        original[::7, ::5] = np.nan
        original[-1, -1] = np.inf
    for kind in ["fortran", "float64_fortran", "dataframe", "reversed_dataframe", "nullable", "arrow"]:
        values = original.copy()
        if kind == "fortran":
            input_data = np.asfortranarray(values)
        elif kind == "float64_fortran":
            input_data = np.asfortranarray(values, dtype=np.float64)
        elif kind == "dataframe":
            input_data = pd.DataFrame(np.asfortranarray(values))
        elif kind == "reversed_dataframe":
            values = values[:, ::-1]
            input_data = pd.DataFrame(values)
        elif kind == "nullable":
            input_data = pd.DataFrame(values, dtype="Float32")
        else:
            input_data = pa.table({f"f{c}": pa.array(values[:, c], from_pandas=True) for c in range(shape[1])})
        expected = values.copy()
        missing = ~np.isfinite(expected)
        expected[missing] = 0
        matrix = xgb.DMatrix(input_data, missing=np.inf)
        # All paths must own their storage after returning from the C API.
        values.fill(-777)
        if isinstance(input_data, np.ndarray):
            input_data.fill(-777)
        elif isinstance(input_data, pd.DataFrame):
            input_data.iloc[:, :] = -777
        assert matrix.num_nonmissing() == int((~missing).sum())
        np.testing.assert_array_equal(matrix.get_data().toarray(), expected)


def test_transpose_dispatch_spans_multiple_grid_rows():
    rows = 65535 * 32 + 1
    X = np.empty((rows, 2), dtype=np.float32, order="F")
    X[:, 0] = np.arange(rows, dtype=np.float32)
    X[:, 1] = -X[:, 0]
    matrix = xgb.DMatrix(X)
    selected = np.array([0, 31, 32, rows - 2, rows - 1], dtype=np.int32)
    np.testing.assert_array_equal(matrix.slice(selected).get_data().toarray(), X[selected])
    assert matrix.num_nonmissing() == X.size


def test_dmatrix_owns_its_input_storage():
    X = np.arange(51, dtype=np.float32).reshape(17, 3)
    original = X.copy()
    d = xgb.DMatrix(X)
    X[:] = -999
    np.testing.assert_array_equal(d.get_data().toarray(), original)


def test_gpu_missing_value_validation():
    X = np.zeros((1031, 3), dtype=np.float32)
    X[-1, -1] = np.inf
    with pytest.raises(xgb.core.XGBoostError, match='infinity'):
        xgb.DMatrix(X)
    valid = xgb.DMatrix(X, missing=np.inf)
    assert valid.num_nonmissing() == X.size - 1
    X[-1, -1] = -999
    valid = xgb.DMatrix(X, missing=-999)
    assert valid.num_nonmissing() == X.size - 1
    X[-1, -1] = np.nan
    valid = xgb.DMatrix(X)
    assert valid.num_nonmissing() == X.size - 1


def test_nullable_dataframe_and_feature_names():
    frame = pd.DataFrame({'integer': pd.Series([1, 2, None, 4], dtype='Int64'),
                          'real': pd.Series([0.5, None, 1.5, 2.0], dtype='Float32')})
    d = xgb.DMatrix(frame, label=[1, 2, 3, 4])
    assert d.feature_names == ['integer', 'real']
    assert d.num_nonmissing() == 6
    b = xgb.train({'max_depth': 2}, d, 2)
    np.testing.assert_array_equal(b.inplace_predict(frame), b.predict(d))
    with pytest.raises(ValueError, match='feature_names mismatch'):
        b.predict(xgb.DMatrix(frame[['real', 'integer']]))


@pytest.mark.parametrize('quantile', [False, True])
def test_iterator_batches(quantile):
    rng = np.random.default_rng(821)
    X = rng.normal(size=(271, 7)).astype(np.float32)
    y = X[:, 0] - X[:, 4]
    weight = rng.uniform(0.1, 2, len(X)).astype(np.float32)

    class Batches(xgb.DataIter):
        def __init__(self):
            self.batch = 0
            super().__init__()

        def reset(self):
            self.batch = 0

        def next(self, input_data):
            if self.batch >= 3:
                return False
            begin, end = [0, 83, 174, len(X)][self.batch:self.batch + 2]
            input_data(data=X[begin:end], label=y[begin:end], weight=weight[begin:end])
            self.batch += 1
            return True

    cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    actual = cls(Batches())
    expected = cls(X, label=y, weight=weight)
    np.testing.assert_array_equal(actual.get_label(), y)
    np.testing.assert_array_equal(actual.get_weight(), weight)
    p = {'max_depth': 3, 'max_bin': 256}
    a, b = xgb.train(p, actual, 3), xgb.train(p, expected, 3)
    np.testing.assert_array_equal(a.predict(actual), b.predict(expected))
