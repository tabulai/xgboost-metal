import json
import pickle

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb


@pytest.mark.parametrize("cardinality,onehot,bins", [(3, 4, 16), (8, 4, 16), (37, 1, 8), (301, 4, 16)])
@pytest.mark.parametrize("missing", [False, True])
def test_categorical_splits_against_reference(reference, cardinality, onehot, bins, missing):
    rng = np.random.default_rng(194)
    X = np.column_stack([rng.integers(0, cardinality, 1601), rng.integers(0, 7, 1601)]).astype(np.float32)
    y = (np.sin(X[:, 0] * 0.7) * 3 + (X[:, 1] > 3) + rng.normal(0, 0.1, len(X))).astype(np.float32)
    if missing:
        X[::7, 0] = np.nan
        y[::7] += 1.5
    weights = rng.uniform(0.2, 2, len(X)).astype(np.float32)
    params = dict(max_depth=3, max_bin=bins, max_cat_to_onehot=onehot, max_cat_threshold=17,
                  base_score=0.25, reg_lambda=1.3)
    expected, files = reference(X, y, params, rounds=4, feature_types=["c", "q"], weight=weights)
    data = xgb.DMatrix(X, label=y, weight=weights, feature_types=["c", "q"], enable_categorical=True)
    model = xgb.train(params, data, 4)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], atol=3e-5, rtol=3e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    assert any(isinstance(json.loads(tree).get("split_condition"), list) for tree in model.get_dump(dump_format="json"))
    # Model exchange also exercises arbitrary categorical membership in GPU traversal.
    for format in ["json", "ubj"]:
        imported = xgb.Booster(model_file=files / f"model.{format}")
        np.testing.assert_allclose(imported.predict(data), expected["prediction"], atol=2e-6)


@pytest.mark.parametrize("multiclass", [False, True])
def test_categorical_shap_and_model_exchange(reference, tmp_path, multiclass):
    rng = np.random.default_rng(987)
    X = rng.integers(0, 9, (201, 3)).astype(np.float32)
    y = ((X[:, 0] % 3) * (X[:, 1] - 4) + X[:, 2] % 2).astype(np.float32)
    X[::5, 0] = np.nan
    params = dict(max_depth=5, max_bin=16, base_score=0, max_cat_to_onehot=2)
    if multiclass:
        y %= 3
        params.update(objective="multi:softprob", num_class=3)
    expected, files = reference(X, y, params, rounds=3, feature_types=["c"] * 3)
    model = xgb.Booster(model_file=files / "model.json")
    data = xgb.DMatrix(X, label=y, feature_types=["c"] * 3, enable_categorical=True)
    for field, options in [
        ("contributions", dict(pred_contribs=True)),
        ("interactions", dict(pred_interactions=True)),
        ("approximate", dict(pred_contribs=True, approx_contribs=True)),
        ("approximate_interactions", dict(pred_interactions=True, approx_contribs=True)),
    ]:
        np.testing.assert_allclose(model.predict(data, **options), expected[field], atol=5e-5, rtol=5e-5)
    np.testing.assert_allclose(model.predict(data, pred_contribs=True).sum(axis=-1), model.predict(data, output_margin=True), atol=5e-5)
    trained = xgb.train(params, data, 3)
    for format in ["json", "ubj"]:
        path = tmp_path / f"metal.{format}"
        trained.save_model(path)
        reloaded, _ = reference(X, y, params, model=path, feature_types=["c"] * 3)
        np.testing.assert_allclose(reloaded["prediction"], trained.predict(data), atol=3e-6)
    atomic = xgb.train({**params, "metal_partition_scan": 0, "metal_histogram_subtraction": 0}, data, 3)
    assert trained.save_raw(raw_format="json") == atomic.save_raw(raw_format="json")


@pytest.mark.parametrize("integer_names", [False, True])
def test_dataframe_recoding_and_saved_categories(reference, tmp_path, integer_names):
    rng = np.random.default_rng(137)
    names = ["茶", "apple", "pear", ""] if not integer_names else [2**60 + i for i in range(4)]
    codes = rng.integers(0, 4, 257)
    frame = pd.DataFrame({"kind": pd.Categorical.from_codes(codes, names),
                          "count": rng.integers(0, 6, len(codes)).astype(np.float32)})
    frame.loc[::11, "kind"] = np.nan
    test = frame.copy()
    test["kind"] = test["kind"].cat.reorder_categories([names[2], names[0], names[3], names[1]])
    y = ((codes == 1) * 3 + (codes == 3) * -2 + frame["count"].to_numpy()).astype(np.float32)
    params = dict(max_depth=3, max_bin=16, base_score=0)
    expected, files = reference(frame, y, params, rounds=5, test=test)
    data = xgb.DMatrix(frame, label=y, enable_categorical=True)
    model = xgb.train(params, data, 5)
    assert not model.get_categories().empty()
    np.testing.assert_allclose(model.inplace_predict(test), expected["prediction"], atol=3e-6)
    np.testing.assert_array_equal(model.inplace_predict(test), model.inplace_predict(frame))
    recoded = xgb.DMatrix(test, enable_categorical=True, feature_types=model.get_categories())
    recoded_expected, _ = reference(frame, y, params, rounds=5, test=test,
                                    test_matrix=dict(category_reference=True))
    np.testing.assert_allclose(model.predict(recoded), recoded_expected['prediction'], atol=3e-6)
    quantile = xgb.QuantileDMatrix(frame, label=y, max_bin=16, enable_categorical=True)
    quantile_test = xgb.QuantileDMatrix(test, label=y, max_bin=16, enable_categorical=True, ref=quantile)
    np.testing.assert_array_equal(model.predict(quantile_test), model.predict(data))
    for format in ["json", "ubj"]:
        path = tmp_path / f"named.{format}"
        model.save_model(path)
        restored = xgb.Booster(model_file=path)
        np.testing.assert_array_equal(restored.inplace_predict(test), model.inplace_predict(frame))
        imported = xgb.Booster(model_file=files / f"model.{format}")
        np.testing.assert_allclose(imported.inplace_predict(test), expected["prediction"], atol=3e-6)
        loaded, _ = reference(frame, y, params, test=test, model=path)
        np.testing.assert_allclose(loaded["prediction"], model.inplace_predict(frame), atol=3e-6)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).inplace_predict(test), model.inplace_predict(frame))
    # Raw continuation retains the new matrix's codes, as the pinned updater does.
    expected, _ = reference(test, y, params, model=path, continue_rounds=2, test=frame)
    continued_data = xgb.DMatrix(test, label=y, enable_categorical=True)
    before = continued_data.get_data().data.copy()
    raw_continued = xgb.train(params, continued_data, 2, xgb_model=model)
    np.testing.assert_array_equal(continued_data.get_data().data.view(np.uint32), before.view(np.uint32))
    np.testing.assert_array_equal(raw_continued.inplace_predict(frame).view(np.uint32), expected['prediction'].view(np.uint32))
    # Explicit-reference construction also follows the adapter's missing-value conversion.
    extended_data = xgb.DMatrix(test, label=y, enable_categorical=True, feature_types=model.get_categories())
    extended = xgb.train(params, extended_data, 2, xgb_model=model)
    expected, _ = reference(test, y, params, model=path, continue_rounds=2,
                            category_reference=True, test=frame)
    np.testing.assert_array_equal(extended.inplace_predict(frame).view(np.uint32), expected['prediction'].view(np.uint32))


def test_categorical_quantiles_do_not_use_numeric_bin_limit(reference):
    X = np.column_stack([np.arange(53) % 37, np.arange(53)]).astype(np.float32)
    expected, out = reference(X, X[:, 0] % 3, dict(max_bin=4), rounds=3,
                              feature_types=["c", "q"], quantile=True, cuts=True)
    data = xgb.QuantileDMatrix(X, label=X[:, 0] % 3, feature_types=["c", "q"], enable_categorical=True, max_bin=4)
    ptr, values = data.get_quantile_cut()
    cuts = np.load(out / "cuts.npz")
    np.testing.assert_array_equal(ptr, cuts["indptr"])
    np.testing.assert_array_equal(values.view(np.uint32), cuts["values"].view(np.uint32))
    np.testing.assert_array_equal(values[ptr[0]:ptr[1]], np.arange(37, dtype=np.float32))
    assert np.isneginf(values[ptr[1]])
    assert ptr[2] - ptr[1] <= 5  # Four numeric bins plus their lower boundary.
    np.testing.assert_array_equal(xgb.train(dict(max_bin=4), data, 3).predict(data), expected["prediction"])


@pytest.mark.parametrize("invalid", [-1, 16777216])
def test_invalid_categorical_codes_rejected(invalid):
    X = np.array([[0], [invalid], [2]], dtype=np.float32)
    data = xgb.DMatrix(X, label=[0, 1, 2], feature_types=["c"], enable_categorical=True)
    with pytest.raises(xgb.core.XGBoostError, match="categorical value"):
        xgb.train({}, data, 1)


def test_unknown_category_names_rejected():
    frame = pd.DataFrame({"kind": pd.Categorical(["a", "b", "a", "b"])})
    model = xgb.train({}, xgb.DMatrix(frame, label=[0, 1, 0, 1], enable_categorical=True), 2)
    unknown = pd.DataFrame({"kind": pd.Categorical(["c"])})
    with pytest.raises(xgb.core.XGBoostError, match="not in the training data"):
        model.inplace_predict(unknown)
    for cls in [xgb.DMatrix, xgb.QuantileDMatrix]:
        query = cls(unknown, enable_categorical=True)
        for _ in range(2):
            # A rejected prediction must not publish an uninitialized cache.
            with pytest.raises(xgb.core.XGBoostError, match="not in the training data"):
                model.predict(query)


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "str"])
def test_arrow_category_export(dtype):
    pa = pytest.importorskip("pyarrow")
    names = ["", "abc", "茶"] if dtype == "str" else np.array([1, 3, 7], dtype=dtype)
    frame = pd.DataFrame({"kind": pd.Categorical.from_codes([0, 1, 2, -1, 2, 0], names), "x": np.arange(6, dtype=np.float32)})
    data = xgb.DMatrix(frame, label=[0, 1, 2, 3, 2, 0], enable_categorical=True)
    model = xgb.train(dict(max_depth=2), data, 2)
    for owner in [data, model]:
        exported = owner.get_categories(export_to_arrow=True)
        arrays = exported.to_arrow()
        assert arrays[0][0] == "kind" and arrays[1] == ("x", None)
        assert arrays[0][1].to_pylist() == list(names)
        assert arrays[0][1].type == (pa.string() if dtype == "str" else pa.from_numpy_dtype(dtype))
        # Buffers have independent ownership and remain usable after another C call.
        model.save_config()
        assert arrays[0][1].to_pylist() == list(names)


def test_arrow_dictionary_input_and_sklearn_recoding():
    pa = pytest.importorskip("pyarrow")
    rng = np.random.default_rng(23)
    names = ["red", "green", "蓝"]
    values = [names[c] if i % 11 else None for i, c in enumerate(rng.integers(0, 3, 257))]
    frame = pd.DataFrame({"color": pd.Categorical(values, categories=names), "x": rng.integers(0, 4, len(values))})
    labels = ((frame["color"] == "green") | (frame["x"] > 2)).to_numpy(dtype=np.float32)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    data = xgb.DMatrix(table, label=labels, enable_categorical=True)
    control = xgb.DMatrix(frame, label=labels, enable_categorical=True)
    params = dict(max_depth=3, objective="binary:logistic")
    model = xgb.train(params, data, 4)
    np.testing.assert_array_equal(model.predict(data), xgb.train(params, control, 4).predict(control))
    estimator = xgb.XGBClassifier(enable_categorical=True, max_depth=3, n_estimators=4)
    estimator.fit(frame, labels)
    reordered = frame.copy()
    reordered["color"] = reordered["color"].cat.reorder_categories(names[::-1])
    np.testing.assert_array_equal(estimator.predict_proba(frame), estimator.predict_proba(reordered))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(estimator)).predict_proba(reordered), estimator.predict_proba(frame))


def test_category_codes_above_uint16(reference):
    X = np.tile(np.array([0, 65535, 65536], dtype=np.float32), 23).reshape(-1, 1)
    y = np.where(X[:, 0] == 65536, 4, -1).astype(np.float32)
    params = dict(max_depth=1, max_bin=16, base_score=0)
    expected, _ = reference(X, y, params, rounds=1, feature_types=["c"])
    data = xgb.DMatrix(X, label=y, feature_types=["c"], enable_categorical=True)
    model = xgb.train(params, data, 1)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], atol=2e-6)


def test_concurrent_predictions_share_recoding_input_safely():
    from concurrent.futures import ThreadPoolExecutor
    rng = np.random.default_rng(73)
    frame = pd.DataFrame({"kind": pd.Categorical.from_codes(rng.integers(0, 4, 257), ["a", "b", "c", "d"]),
                          "x": rng.integers(0, 5, 257)})
    y = frame["kind"].cat.codes.to_numpy(dtype=np.float32) * 2 + frame["x"].to_numpy(dtype=np.float32)
    models = []
    for order in [["b", "a", "d", "c"], ["d", "c", "b", "a"], ["c", "a", "d", "b"]]:
        training = frame.copy()
        training["kind"] = training["kind"].cat.reorder_categories(order)
        models.append(xgb.train(dict(max_depth=3), xgb.DMatrix(training, label=y, enable_categorical=True), 3))
    data = xgb.DMatrix(frame, enable_categorical=True)
    expected = [model.predict(data) for model in models]
    def predict(i):
        np.testing.assert_array_equal(models[i % len(models)].predict(data), expected[i % len(models)])
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(predict, range(48)))
