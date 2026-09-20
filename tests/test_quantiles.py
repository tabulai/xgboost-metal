"""Public histogram cuts compared with the pinned CPU sketch and its adapters."""
import json
import re
import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize("rows,bins,kind", [
    (17, 4, "unit"), (257, 16, "unit"), (8193, 64, "unit"),
    (100003, 256, "unit"), (259, 2, "unit"), (4099, 3, "unit"),
    (8193, 31, "unit"), (8193, 33, "unit"),
    (257, 16, "fractional"), (8193, 64, "fractional"),
    (8193, 64, "integer"), (8193, 64, "zero-weights"),
    (4099, 16, "ties"), (4099, 16, "runs"),
    (4099, 16, "missing"), (4099, 16, "empty-column"),
    (4099, 16, "all-zero-weights"), (17, 64, "exact"),
])
def test_weighted_sketch_cuts_match_reference(reference, quantile, rows, bins, kind):
    rng = np.random.default_rng(941 + rows)
    X = rng.normal(size=(rows, 3)).astype(np.float32)
    X[:, 1] = np.arange(rows, dtype=np.float32)
    weight = np.ones(rows, np.float32)
    if kind in {"integer", "zero-weights"}:
        weight = rng.integers(0, 9, rows).astype(np.float32)
        if kind == "zero-weights":
            weight[::2] = 0
    elif kind != "unit":
        weight = rng.uniform(0, 2, rows).astype(np.float32)
    if kind in {"ties", "runs"}:
        X = rng.integers(-100, 101, size=X.shape).astype(np.float32)
        if kind == "runs":
            X = np.repeat(X[::31], 31, axis=0)[:rows]
    if kind == "missing":
        X[rng.uniform(size=X.shape) < 0.7] = np.nan
    if kind == "empty-column":
        X[:, 1] = np.nan
    if kind == "all-zero-weights":
        weight[:] = 0
    params = dict(max_bin=bins, max_depth=1, base_score=0)
    labels = np.zeros(rows, np.float32)
    _, out = reference(X, labels, params, rounds=1, quantile=quantile, cuts=True, weight=weight)
    expected = np.load(out / "cuts.npz")
    cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    options = dict(max_bin=bins) if quantile else {}
    d = cls(X, label=labels, weight=weight, **options)
    xgb.train(params, d, 1)
    indptr, values = d.get_quantile_cut()
    np.testing.assert_array_equal(indptr, expected["indptr"])
    np.testing.assert_array_equal(values.view(np.uint32), expected["values"].view(np.uint32))
    for begin, end in zip(indptr[:-1], indptr[1:]):
        column = values[int(begin):int(end)]
        assert np.isneginf(column[0])
        assert len(column) >= 2
        assert np.all(column[1:] > column[:-1])


@pytest.mark.parametrize("adapter", ["dense", "csr", "dataframe", "iterator"])
def test_quantile_adapters_preserve_compressed_cuts(reference, adapter):
    rng = np.random.default_rng(582)
    X = rng.normal(size=(4099, 5)).astype(np.float32)
    X[::9, 2] = np.nan
    y = np.nan_to_num(X[:, 0] - X[:, 2])
    w = rng.uniform(0.1, 2, len(X)).astype(np.float32)
    p = dict(max_bin=31, max_depth=4, base_score=0)
    expected, out = reference(X, y, p, rounds=3, quantile=True, cuts=True, weight=w)
    values = X
    kwargs = dict(label=y, weight=w)
    if adapter == "csr":
        from scipy.sparse import csr_matrix
        values = csr_matrix(X)
    elif adapter == "dataframe":
        import pandas as pd
        values = pd.DataFrame(X)
    elif adapter == "iterator":
        class Batches(xgb.DataIter):
            def __init__(self):
                self.index = 0
                super().__init__()

            def reset(self):
                self.index = 0

            def next(self, input_data):
                bounds = [0, 67, 1931, 3075, len(X)]
                if self.index == len(bounds) - 1:
                    return False
                begin, end = bounds[self.index:self.index + 2]
                input_data(data=X[begin:end], label=y[begin:end], weight=w[begin:end])
                self.index += 1
                return True
        values, kwargs = Batches(), {}
    d = xgb.QuantileDMatrix(values, max_bin=31, **kwargs)
    model = xgb.train(p, d, 3)
    cuts = np.load(out / "cuts.npz")
    actual_ptr, actual_values = d.get_quantile_cut()
    np.testing.assert_array_equal(actual_ptr, cuts["indptr"])
    np.testing.assert_array_equal(actual_values.view(np.uint32), cuts["values"].view(np.uint32))
    np.testing.assert_array_equal(model.predict(xgb.DMatrix(X, feature_names=d.feature_names)), expected["prediction"])


@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize("field", ["weight", "feature_weights", "group", "qid"])
def test_metadata_updates_preserve_materialized_cuts(reference, quantile, field):
    rng = np.random.default_rng(718)
    X = rng.normal(size=(513, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    grouped = field in {"group", "qid"}
    meta = (dict(group=[171] * 3, weight=np.array([0.01, 1, 10], np.float32)) if grouped
            else dict(weight=rng.uniform(0.1, 2, len(X)).astype(np.float32)))
    change = {
        "weight": np.linspace(0, 2, len(X), dtype=np.float32),
        "feature_weights": np.array([0.01, 1], np.float32),
        "group": np.array([100, 200, 213], np.uint32),
        "qid": np.repeat(np.arange(3, dtype=np.uint32), [100, 200, 213]),
    }[field]
    p = dict(max_bin=8, max_depth=2, base_score=0,
             objective="rank:pairwise" if grouped else "reg:squarederror")
    expected, out = reference(X, y, p, rounds=3, quantile=quantile, cuts=True,
                              matrix_updates={"1": {field: change.tolist()}}, **meta)
    cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    d = cls(X, label=y, **meta, **(dict(max_bin=8) if quantile else {}))

    class Update(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            if epoch == 1:
                before = tuple(a.copy() for a in d.get_quantile_cut())
                d.set_info(**{field: change})
                after = d.get_quantile_cut()
                for a, b in zip(before, after):
                    np.testing.assert_array_equal(a, b)
            return False

    model = xgb.train(p, d, 3, callbacks=[Update()])
    wanted = np.load(out / "cuts.npz")
    ptr, values = d.get_quantile_cut()
    np.testing.assert_array_equal(ptr, wanted["indptr"])
    np.testing.assert_array_equal(values.view(np.uint32), wanted["values"].view(np.uint32))
    if not grouped:
        np.testing.assert_allclose(model.predict(xgb.DMatrix(X)), expected["prediction"], rtol=2e-6, atol=2e-6)
    if quantile:
        with pytest.raises(xgb.core.XGBoostError, match="Inconsistent max_bin"):
            xgb.train({**p, "max_bin": 9}, d, 1)
    else:
        xgb.train({**p, "max_bin": 9}, d, 1)
        assert not np.array_equal(ptr, d.get_quantile_cut()[0])


@pytest.mark.parametrize("index,rows,bins,kind,column", [
    (25, 257, 16, "boundary", 14),
    (25, 257, 16, "boundary", 15),
    (71, 33001, 512, "fractional", 11),
    (79, 33001, 1024, "powers", 12),
])
def test_tied_weight_rounding_and_signed_zero_cuts(reference, index, rows, bins, kind, column):
    # Tied feature values have different weights. Changing their sort order
    # moves an interior cut through a rounded rank boundary in these cases.
    rng = np.random.default_rng(39281 + index)
    X = rng.normal(size=(rows, 32)).astype(np.float32)
    X[:, 8:16] = rng.integers(-2 * bins, 2 * bins + 1, (rows, 8))
    X[:, 16:24] = rng.integers(-3, 4, (rows, 8))
    X[::2, 16:24] *= np.float32(-1)
    X[::13, 2:6] = np.nan
    X[:, 24:] = np.repeat(X[::17, 24:], 17, axis=0)[:rows]
    weights = {
        "fractional": rng.uniform(0, 2, rows).astype(np.float32),
        "powers": np.ldexp(rng.uniform(1, 2, rows), rng.integers(-25, 26, rows)).astype(np.float32),
        "boundary": rng.choice(np.array([1, 2**-24, 2**-23, 2**-22, 2], np.float32), rows),
    }
    weights["boundary"][::13] = 0
    X = X[:, [column, *range(16, 24)]].copy()
    p = dict(max_bin=bins, max_depth=1, base_score=0)
    _, out = reference(X, np.zeros(rows), p, rounds=1, quantile=True, cuts=True, weight=weights[kind])
    expected = np.load(out / "cuts.npz")
    d = xgb.QuantileDMatrix(X, weight=weights[kind], max_bin=bins)
    ptr, values = d.get_quantile_cut()
    np.testing.assert_array_equal(ptr, expected["indptr"])
    np.testing.assert_array_equal(values.view(np.uint32), expected["values"].view(np.uint32))


@pytest.mark.parametrize("rows,bins", [(513, 3), (8193, 64)])
@pytest.mark.parametrize("kind", ["subnormal", "small-normal", "overflow", "span", "zero"])
def test_quantile_cuts_preserve_gradual_underflow_and_overflow(reference, rows, bins, kind):
    rng = np.random.default_rng(847 + rows)
    tiny = np.nextafter(np.float32(0), np.float32(1))
    X = rng.normal(size=(rows, 4)).astype(np.float32)
    X[:, 1] = rng.integers(-100, 101, rows).astype(np.float32) * tiny
    X[:, 2] = rng.choice(np.array([-np.finfo(np.float32).max, -1e38, -0.0, 0.0, 1e38,
                                  np.finfo(np.float32).max], np.float32), rows)
    X[::17, 3] = np.nan
    weights = np.ones(rows, np.float32)
    if kind == "subnormal":
        weights = rng.integers(0, 30, rows).astype(np.float32) * tiny
    elif kind == "small-normal":
        weights[:] = np.finfo(np.float32).tiny
    elif kind == "overflow":
        weights[:] = np.float32(1e38)
    elif kind == "span":
        weights = np.exp2(rng.integers(-149, 126, rows).astype(np.float64)).astype(np.float32)
    else:
        weights[:] = 0
    p = dict(max_bin=bins, max_depth=1, base_score=0)
    _, out = reference(X, np.zeros(rows), p, rounds=0, quantile=True, cuts=True, weight=weights)
    expected = np.load(out / "cuts.npz")
    d = xgb.QuantileDMatrix(X, weight=weights, max_bin=bins)
    ptr, values = d.get_quantile_cut()
    np.testing.assert_array_equal(ptr, expected["indptr"])
    np.testing.assert_array_equal(values.view(np.uint32), expected["values"].view(np.uint32))


@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize("missing", ["nan", "zero", "subnormal"])
def test_tiny_feature_training_prediction_and_missing_values(reference, quantile, missing):
    rng = np.random.default_rng(855)
    rows = 4099
    tiny = np.nextafter(np.float32(0), np.float32(1))
    values = rng.integers(-50, 51, rows)
    X = np.column_stack([values.astype(np.float32) * tiny, rng.normal(size=rows)]).astype(np.float32)
    X[::17, 0] = np.nan
    y = (values < -3).astype(np.float32)
    marker = {"nan": np.nan, "zero": np.float32(0), "subnormal": tiny}[missing]
    reference_X = np.where(X == marker, np.nan, X)
    p = dict(max_bin=16, max_depth=3, base_score=0)
    expected, out = reference(reference_X, y, p, rounds=5, quantile=quantile, cuts=True)
    cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    options = dict(max_bin=16) if quantile else {}
    d = cls(X, label=y, missing=float(marker), **options)
    model = xgb.train(p, d, 5)
    wanted_cuts = np.load(out / "cuts.npz")
    ptr, cuts = d.get_quantile_cut()
    np.testing.assert_array_equal(ptr, wanted_cuts["indptr"])
    np.testing.assert_array_equal(cuts.view(np.uint32), wanted_cuts["values"].view(np.uint32))
    np.testing.assert_array_equal(model.predict(d), expected["prediction"])
    np.testing.assert_array_equal(model.predict(d, pred_leaf=True), expected["leaf"])
    np.testing.assert_allclose(model.predict(d, pred_contribs=True), expected["contributions"], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("format", ["json", "ubj"])
def test_missing_only_split_uses_infinite_lower_boundary(reference, tmp_path, format):
    X = np.full((129, 1), 5.0, np.float32)
    X[::2] = np.nan
    y = np.isnan(X[:, 0]).astype(np.float32)
    p = dict(max_bin=4, max_depth=1, base_score=0, monotone_constraints="(-1)")
    expected, out = reference(X, y, p, rounds=1, cuts=True)
    d = xgb.DMatrix(X, label=y)
    model = xgb.train(p, d, 1)
    attributes = {"Infinity": "-Infinity", "NaN": 'escaped "Infinity" \\ NaN', "unicode": "\u2603Infinity"}
    model.set_attr(**attributes)
    np.testing.assert_array_equal(model.predict(d), expected["prediction"])
    actual = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"][0]
    reference_model = json.loads((out / "model.json").read_text())
    wanted = reference_model["learner"]["gradient_booster"]["model"]["trees"][0]
    np.testing.assert_array_equal(np.asarray(actual["split_conditions"], np.float32),
                                  np.asarray(wanted["split_conditions"], np.float32))
    assert np.isneginf(actual["split_conditions"][0])
    dumped = model.get_dump(dump_format="json", with_stats=True)[0]
    reference_dump = json.loads((out / "dump.json").read_text())[0]
    pattern = r'"split_condition"\s*:\s*(-?inf)'
    assert re.search(pattern, dumped).group(1) == re.search(pattern, reference_dump).group(1)
    serialized = model.save_raw(raw_format=format)
    restored = xgb.Booster(model_file=serialized)
    assert restored.attributes() == attributes
    np.testing.assert_array_equal(restored.predict(d), expected["prediction"])
    imported = xgb.Booster(model_file=out / f"model.{format}")
    np.testing.assert_array_equal(imported.predict(d), expected["prediction"])
    path = tmp_path / f"infinite-boundary.{format}"
    model.save_model(path)
    exchanged, _ = reference(X, y, p, rounds=1, model=path)
    np.testing.assert_array_equal(exchanged["prediction"], expected["prediction"])
