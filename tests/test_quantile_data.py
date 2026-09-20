"""Quantized data access follows the CPU and CUDA bin-lower-bound contracts."""
import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb


PREDICTIONS = [
    ("prediction", {}), ("margin", {"output_margin": True}),
    ("leaf", {"pred_leaf": True}), ("contributions", {"pred_contribs": True}),
    ("approximate", {"pred_contribs": True, "approx_contribs": True}),
    ("interactions", {"pred_interactions": True}),
    ("approximate_interactions", {"pred_interactions": True, "approx_contribs": True}),
]


def check_data(matrix, files, prefix="test"):
    expected = np.load(files / "matrix-data.npz")
    csr = matrix.get_data()
    for field, actual in [("indptr", csr.indptr), ("indices", csr.indices), ("data", csr.data)]:
        wanted = expected[prefix + "_" + field]
        if field == "data":
            actual, wanted = actual.view(np.uint32), wanted.view(np.uint32)
        np.testing.assert_array_equal(actual, wanted)
    if prefix + "_cuts" in expected:
        ptr, cuts = matrix.get_quantile_cut()
        np.testing.assert_array_equal(ptr, expected[prefix + "_cut_ptr"])
        np.testing.assert_array_equal(cuts.view(np.uint32), expected[prefix + "_cuts"].view(np.uint32))
    info = json.loads((files / "matrix-info.json").read_text())[prefix]
    assert matrix.feature_types == info["feature_types"]
    assert matrix.feature_names == info["feature_names"]
    assert matrix.num_nonmissing() == info["nonmissing"]


def check_predictions(model, matrix, expected):
    for field, options in PREDICTIONS:
        actual = model.predict(matrix, **options)
        if field in {"prediction", "margin", "leaf"}:
            np.testing.assert_array_equal(actual.view(np.uint32), expected[field].view(np.uint32))
        else:
            np.testing.assert_allclose(actual, expected[field], rtol=3e-6, atol=3e-6)


def check_tree_statistics(model, reference_file):
    actual = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]
    expected = json.loads(reference_file.read_text())["learner"]["gradient_booster"]["model"]
    assert actual["iteration_indptr"] == expected["iteration_indptr"]
    assert actual["tree_info"] == expected["tree_info"]
    assert len(actual["trees"]) == len(expected["trees"])
    floating = {"base_weights", "leaf_weights", "split_conditions", "sum_hessian", "loss_changes"}
    for index, (own, wanted) in enumerate(zip(actual["trees"], expected["trees"])):
        assert own.keys() == wanted.keys()
        for field in own:
            if field in floating:
                np.testing.assert_array_equal(np.asarray(own[field], np.float32).view(np.uint32),
                                              np.asarray(wanted[field], np.float32).view(np.uint32),
                                              err_msg=f"tree {index}: {field}")
            else:
                assert own[field] == wanted[field], (index, field)


def as_csr(values):
    # Preserve explicit zeroes, including category zero.
    from scipy.sparse import csr_matrix
    rows, columns = values.shape
    return csr_matrix((values.reshape(-1), np.tile(np.arange(columns), rows),
                       np.arange(rows + 1) * columns), shape=values.shape)


@pytest.mark.parametrize("use_ref", [False, True])
@pytest.mark.parametrize("adapter", ["dense", "csr"])
@pytest.mark.parametrize("kind", ["numeric", "tiny", "mixed", "categorical"])
def test_quantile_data_and_all_prediction_modes(reference, use_ref, adapter, kind):
    rng = np.random.default_rng(7722)
    X = rng.normal(size=(513, 4)).astype(np.float32)
    query = (rng.normal(size=(137, 4)) * 1.3 + 0.2).astype(np.float32)
    feature_types = ["q"] * 4
    if kind == "tiny":
        tiny = np.nextafter(np.float32(0), np.float32(1))
        X[:, 0] = rng.integers(-128, 129, len(X)).astype(np.float32) * tiny
        query[:, 0] = rng.integers(-128, 129, len(query)).astype(np.float32) * tiny
    if kind in {"mixed", "categorical"}:
        columns = range(4) if kind == "categorical" else [0, 2]
        for column in columns:
            X[:, column] = rng.integers(0, 9, len(X))
            query[:, column] = rng.integers(0, 9, len(query))
            feature_types[column] = "c"
    y = (X[:, 0] < 0).astype(np.float32) + X[:, 1] % 3
    X[::17, 3], query[::11, 3] = np.nan, np.nan
    weight = np.resize(np.array([0.125, 0.5, 1, 2], np.float32), len(X))
    params = dict(max_bin=16, max_depth=3, base_score=0, seed=19)
    expected, files = reference(X, y, params, rounds=5, test=query, weight=weight,
                                feature_types=feature_types, quantile=True, matrix_data=True,
                                test_matrix=dict(quantile=True, ref=use_ref))
    train = xgb.QuantileDMatrix(X, label=y, weight=weight, max_bin=16,
                                feature_types=feature_types, enable_categorical=True)
    values = as_csr(query) if adapter == "csr" else query
    matrix = xgb.QuantileDMatrix(values, ref=train if use_ref else None, max_bin=16,
                                 feature_types=feature_types, enable_categorical=True)
    check_data(train, files, "train")
    check_data(matrix, files)
    model = xgb.Booster(model_file=files / "model.json")
    check_predictions(model, matrix, expected)
    model.reset()
    model.set_param({"metal_predict_shared": 1, "metal_predict_threads": 64})
    check_predictions(model, matrix, expected)
    trained = xgb.train(params, train, 5)
    np.testing.assert_array_equal(trained.predict(matrix), expected["prediction"])


def test_first_numeric_bin_has_distinct_prediction_and_export_values(reference):
    X = np.arange(16, dtype=np.float32).reshape(-1, 1)
    query = np.arange(100, 108, dtype=np.float32).reshape(-1, 1)
    params = dict(max_bin=4, max_depth=1, base_score=0, eta=1)
    expected, files = reference(X, X[:, 0] < 8, params, rounds=1, test=query,
                                quantile=True, matrix_data=True, test_matrix=dict(quantile=True))
    matrix = xgb.QuantileDMatrix(query, max_bin=4)
    model = xgb.Booster(model_file=files / "model.json")
    check_data(matrix, files)
    check_predictions(model, matrix, expected)
    exported = model.predict(xgb.DMatrix(matrix.get_data()))
    assert np.count_nonzero(model.predict(matrix) != exported) == 2


@pytest.mark.parametrize("rows,columns", [(0, 3), (1, 1), (31, 33), (33, 31), (65, 65)])
@pytest.mark.parametrize("kind", ["numeric", "all-missing", "zero-weight"])
def test_quantile_reconstruction_boundaries(reference, rows, columns, kind):
    tiny = np.nextafter(np.float32(0), np.float32(1))
    values = np.array([-np.finfo(np.float32).max, -2, -tiny, -0.0, 0.0, tiny, 2,
                       np.finfo(np.float32).max, np.nan], np.float32)
    X = np.resize(values, (rows, columns))
    if kind == "all-missing":
        X[:] = np.nan
    weight = np.zeros(rows, np.float32) if kind == "zero-weight" else np.ones(rows, np.float32)
    _, files = reference(X, np.zeros(rows), dict(max_bin=8, base_score=0), rounds=0,
                         weight=weight, quantile=True, matrix_data=True)
    matrix = xgb.QuantileDMatrix(X, weight=weight, max_bin=8)
    check_data(matrix, files, "train")


@pytest.mark.parametrize("initial,changed", [("q", "c"), ("c", "q"), ("c", None)])
def test_feature_type_changes_reinterpret_quantile_bins(reference, initial, changed):
    X = np.tile(np.arange(8, dtype=np.float32), 5).reshape(-1, 1)
    types = None if changed is None else [changed]
    params = dict(max_bin=4, max_depth=2, base_score=0)
    expected, files = reference(X, X[:, 0] % 3, params, rounds=2, feature_types=[initial],
                                quantile=True, matrix_data=True,
                                test_matrix=dict(quantile=True, feature_types_after=types))
    matrix = xgb.QuantileDMatrix(X, label=X[:, 0] % 3, max_bin=4,
                                 feature_types=[initial], enable_categorical=True)
    trained = xgb.train(params, matrix, 2)
    cached = trained.predict(matrix).copy()
    matrix.feature_types = types
    check_data(matrix, files)
    # The reference's existing training prediction cache survives this change.
    np.testing.assert_array_equal(trained.predict(matrix), cached)
    model = xgb.Booster(model_file=files / "model.json")
    check_predictions(model, matrix, expected)
    with pytest.raises(xgb.core.XGBoostError, match="Inconsistent max_bin"):
        xgb.train({**params, "max_bin": 5}, matrix, 1)


@pytest.mark.parametrize("reference_types", [None, ["q"], ["c"]])
@pytest.mark.parametrize("query_types", [None, ["q"], ["c"]])
def test_cut_reference_supplies_effective_feature_types(reference, reference_types, query_types):
    X = np.tile(np.arange(8, dtype=np.float32), 5).reshape(-1, 1)
    params = dict(max_bin=4, max_depth=2, base_score=0)
    expected, files = reference(X, X[:, 0] % 3, params, rounds=2, feature_types=reference_types,
                                quantile=True, matrix_data=True,
                                test_matrix=dict(quantile=True, ref=True, feature_types=query_types))
    train = xgb.QuantileDMatrix(X, max_bin=4, feature_types=reference_types, enable_categorical=True)
    query = xgb.QuantileDMatrix(X, max_bin=4, ref=train, feature_types=query_types, enable_categorical=True)
    check_data(query, files)
    check_predictions(xgb.Booster(model_file=files / "model.json"), query, expected)
    query.set_label(X[:, 0] % 3)
    trained = xgb.train(params, query, 2)
    check_tree_statistics(trained, files / "model.json")
    np.testing.assert_array_equal(trained.predict(query), expected["prediction"])


@pytest.mark.parametrize("reference_type,query_type", [("q", "c"), ("c", "q")])
def test_setting_same_type_names_overrides_inherited_effective_types(reference, reference_type, query_type):
    X = np.tile(np.arange(8, dtype=np.float32), 5).reshape(-1, 1)
    params = dict(max_bin=4, max_depth=2, base_score=0)
    expected, files = reference(X, X[:, 0] % 3, params, rounds=2, feature_types=[reference_type],
                                quantile=True, matrix_data=True,
                                test_matrix=dict(quantile=True, ref=True, feature_types=[query_type],
                                                 feature_types_after=[query_type]))
    train = xgb.QuantileDMatrix(X, max_bin=4, feature_types=[reference_type], enable_categorical=True)
    query = xgb.QuantileDMatrix(X, max_bin=4, ref=train, feature_types=[query_type], enable_categorical=True)
    query.feature_types = [query_type]
    check_data(query, files)
    check_predictions(xgb.Booster(model_file=files / "model.json"), query, expected)


@pytest.mark.parametrize("quantile_ref", [False, True])
def test_reference_cuts_do_not_require_query_groups_for_weights(reference, quantile_ref):
    X = np.arange(120, dtype=np.float32).reshape(40, 3)
    meta = dict(weight=np.array([1, 2, 3, 4], np.float32), group=np.array([10] * 4, np.uint32))
    _, files = reference(X, X[:, 0] % 5, dict(max_bin=8, base_score=0), rounds=0,
                         quantile=quantile_ref, matrix_data=True,
                         test_matrix=dict(quantile=True, ref=True), **meta)
    cls = xgb.QuantileDMatrix if quantile_ref else xgb.DMatrix
    train = cls(X, **meta, **(dict(max_bin=8) if quantile_ref else {}))
    query = xgb.QuantileDMatrix(X, weight=meta["weight"], max_bin=8, ref=train)
    check_data(query, files)
    np.testing.assert_array_equal(query.get_weight(), meta["weight"])


@pytest.mark.parametrize("use_ref", [False, True])
@pytest.mark.parametrize("large", [False, True])
def test_quantile_category_codes_are_reconstructed_from_bins(reference, use_ref, large):
    tiny = np.nextafter(np.float32(0), np.float32(1))
    X = np.array([0, 1, 2, 65535, 65536] if large else [0, 1, 2], np.float32).reshape(-1, 1)
    query = np.array([tiny, -0.0, 0.5, 1.5, 65535, 65536, np.nan] if large else
                     [tiny, -0.0, 0.5, 1.5, 5, np.nan], np.float32).reshape(-1, 1)
    _, files = reference(X, np.zeros(len(X)), dict(max_bin=4, base_score=0), rounds=0,
                         feature_types=["c"], quantile=True, matrix_data=True, test=query,
                         test_matrix=dict(quantile=True, ref=use_ref))
    train = xgb.QuantileDMatrix(X, max_bin=4, feature_types=["c"], enable_categorical=True)
    matrix = xgb.QuantileDMatrix(query, max_bin=4, feature_types=["c"], enable_categorical=True,
                                 ref=train if use_ref else None)
    check_data(matrix, files)


@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize("negative", [-np.nextafter(np.float32(0), np.float32(1)), -0.5])
def test_negative_categorical_training_values_are_rejected(quantile, negative):
    X = np.array([negative, 0, 1], np.float32).reshape(-1, 1)
    with pytest.raises(xgb.core.XGBoostError, match="Invalid categorical value"):
        cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
        data = cls(X, label=[0, 1, 2], feature_types=["c"], enable_categorical=True,
                    **(dict(max_bin=4) if quantile else {}))
        xgb.train(dict(max_bin=4, base_score=0), data, 1)


def test_negative_subnormal_category_prediction_matches_reference(reference):
    X = np.tile(np.arange(3, dtype=np.float32), 40).reshape(-1, 1)
    query = np.array([-np.nextafter(np.float32(0), np.float32(1)), -0.0, 0, 1, 2], np.float32).reshape(-1, 1)
    expected, files = reference(X, (X[:, 0] == 0).astype(np.float32),
                                dict(max_bin=4, max_depth=1, base_score=0), rounds=2,
                                feature_types=["c"], test=query)
    model = xgb.Booster(model_file=files / "model.json")
    check_predictions(model, xgb.DMatrix(query, feature_types=["c"], enable_categorical=True), expected)


@pytest.mark.parametrize("allow_groups", [False, True])
def test_quantile_matrix_rejects_slicing(allow_groups):
    matrix = xgb.QuantileDMatrix(np.arange(12, dtype=np.float32).reshape(6, 2), max_bin=4)
    with pytest.raises(xgb.core.XGBoostError, match="Slicing DMatrix is not supported"):
        matrix.slice([0, 3], allow_groups=allow_groups)


@pytest.mark.parametrize("kind", ["square", "logistic", "cox", "quantile", "vector", "dropout"])
def test_prediction_prefix_cache_with_ranges_and_metadata(reference, kind):
    X = np.tile(np.arange(16, dtype=np.float32), 17).reshape(-1, 1)
    query = np.arange(97, 114, dtype=np.float32).reshape(-1, 1)
    y = (X[:, 0] % 5).astype(np.float32)
    params = dict(max_bin=4, max_depth=2, base_score=0.25)
    outputs = 1
    if kind == "logistic":
        params.update(objective="binary:logistic", base_score=0.5)
        y = (y < 2).astype(np.float32)
    if kind == "cox":
        params.update(objective="survival:cox", max_delta_step=1, base_score=1)
        y = y + 1
    if kind == "quantile":
        params.update(objective="reg:quantileerror", quantile_alpha=[0.2, 0.5, 0.8])
        outputs = 3
    if kind == "vector":
        params["multi_strategy"] = "multi_output_tree"
        y = np.column_stack([y, -y * 0.75])
        outputs = 2
    if kind == "dropout":
        params.update(rate_drop=0.5, one_drop=True)
    margin = np.linspace(-0.7, 0.7, len(query) * outputs, dtype=np.float32).reshape(len(query), outputs)
    steps = [
        dict(reset=True, predict=dict(iteration_range=[0, 2])),
        dict(feature_types=["c"], predict=dict(iteration_range=[0, 4])),
        dict(feature_types=["q"], predict=dict(iteration_range=[0, 4], output_margin=True)),
        dict(predict=dict(iteration_range=[0, 1])),
        dict(feature_types=["c"], predict=dict(iteration_range=[1, 4])),
        dict(),
        dict(base_margin=margin.tolist()),
        dict(base_margin=(-margin).tolist(), predict=dict(output_margin=True)),
        dict(base_margin=[]),
        dict(feature_types=["q"], predict=dict(pred_leaf=True)),
        dict(predict=dict(output_margin=True)),
        dict(reset=True),
        dict(predict=dict(iteration_range=[5, 5], output_margin=True)),
        dict(),
    ]
    expected, files = reference(X, y, params, rounds=5, test=query, feature_types=["q"],
                                quantile=True, test_matrix=dict(quantile=True), prediction_steps=steps)
    model = xgb.Booster(model_file=files / "model.json")
    matrix = xgb.QuantileDMatrix(query, max_bin=4, feature_types=["q"])

    def check(actual, wanted, options):
        # Cache margins and leaf IDs must be exact. The existing Metal sigmoid
        # has a small exp/reciprocal rounding difference from the CPU libm.
        if kind == "logistic" and not options.get("output_margin") and not options.get("pred_leaf"):
            np.testing.assert_allclose(actual, wanted, rtol=2e-7, atol=1e-7)
        else:
            np.testing.assert_array_equal(actual.view(np.uint32), wanted.view(np.uint32))

    check(model.predict(matrix), expected["prediction"], {})
    for i, step in enumerate(steps):
        if step.get("reset"):
            model.reset()
        if "feature_types" in step:
            matrix.feature_types = step["feature_types"]
        if "base_margin" in step:
            matrix.set_base_margin(np.asarray(step["base_margin"], np.float32))
        options = step.get("predict", {})
        check(model.predict(matrix, **options), expected[f"step_{i}"], options)


@pytest.mark.parametrize("field", ["feature_types", "base_margin"])
@pytest.mark.parametrize("quantile", [False, True])
def test_training_uses_prediction_cache_after_metadata_changes(reference, field, quantile):
    X = np.tile(np.arange(8, dtype=np.float32), 40).reshape(-1, 1)
    y = X[:, 0] % 3
    params = dict(max_bin=4, max_depth=2, base_score=0)
    changes = ({"2": {field: ["c"]}, "3": {field: ["q"]}} if field == "feature_types" else
               {"2": {field: np.linspace(-1, 1, len(X)).tolist()}, "3": {field: []}})
    expected, files = reference(X, y, params, rounds=5, quantile=quantile, feature_types=["q"],
                                matrix_updates=changes, cuts=True)
    cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    matrix = cls(X, label=y, feature_types=["q"], **(dict(max_bin=4) if quantile else {}))

    class Update(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            for name, values in changes.get(str(epoch), {}).items():
                matrix.set_info(**{name: values})
            return False

    model = xgb.train(params, matrix, 5, callbacks=[Update()], evals=[(matrix, "train")], verbose_eval=False)
    np.testing.assert_array_equal(model.predict(xgb.DMatrix(X)), expected["prediction"])
    check_tree_statistics(model, files / "model.json")
    if field == "feature_types":
        assert "in feature map is numerical but the tree node is categorical" in (files / "dump-error.txt").read_text()
        for format in ["text", "json", "dot"]:
            with pytest.raises(xgb.core.XGBoostError, match="in feature map is numerical but the tree node is categorical"):
                model.get_dump(dump_format=format)


@pytest.mark.parametrize("quantile", [False, True])
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("kind", ["numeric-to-category", "category-to-numeric", "mixed", "onehot"])
def test_training_type_transitions_preserve_parent_statistics(reference, quantile, vector, kind):
    X = np.tile(np.arange(8, dtype=np.float32), 40).reshape(-1, 1)
    y = X[:, 0] % 3
    if vector:
        y = np.column_stack([y, -0.75 * y + X[:, 0] % 2]).astype(np.float32)
    initial, changed = (["c"], ["q"]) if kind == "category-to-numeric" else (["q"], ["c"])
    if kind == "mixed":
        X = np.column_stack([X, np.zeros(len(X), np.float32)])
        initial.append("c")
        changed.append("c")
    X[::13, 0] = np.nan
    params = dict(max_bin=4, max_depth=3, base_score=0,
                  multi_strategy="multi_output_tree" if vector else "one_output_per_tree",
                  max_cat_to_onehot=32 if kind == "onehot" else 2)
    changes = {"2": {"feature_types": changed}, "4": {"feature_types": initial}}
    expected, files = reference(X, y, params, rounds=5, quantile=quantile,
                                feature_types=initial, matrix_updates=changes)
    cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    matrix = cls(X, label=y, feature_types=initial, enable_categorical=True,
                 **(dict(max_bin=4) if quantile else {}))

    class Update(xgb.callback.TrainingCallback):
        def before_iteration(self, model, epoch, evals_log):
            if str(epoch) in changes:
                matrix.feature_types = changes[str(epoch)]["feature_types"]
            return False

    model = xgb.train(params, matrix, 5, callbacks=[Update()])
    query = xgb.DMatrix(X, feature_types=initial, enable_categorical=True)
    for field, options in PREDICTIONS[:3]:
        np.testing.assert_array_equal(model.predict(query, **options).view(np.uint32),
                                      expected[field].view(np.uint32))
    check_tree_statistics(model, files / "model.json")


@pytest.mark.parametrize("initial,changed", [("q", "c"), ("c", "q"), ("c", None)])
def test_regular_matrix_type_changes_preserve_materialized_cuts(initial, changed):
    X = np.tile(np.arange(8, dtype=np.float32), 5).reshape(-1, 1)
    data = xgb.DMatrix(X, label=X[:, 0] % 3, feature_types=[initial], enable_categorical=True)
    xgb.train(dict(max_bin=4, max_depth=2, base_score=0), data, 1)
    _, before = data.get_quantile_cut()
    data.feature_types = None if changed is None else [changed]
    ptr, after = data.get_quantile_cut()
    wanted = before[1:] if changed == "c" else np.concatenate([np.array([-np.inf], np.float32), before])
    np.testing.assert_array_equal(ptr, [0, len(wanted)])
    np.testing.assert_array_equal(after.view(np.uint32), wanted.view(np.uint32))
    np.testing.assert_array_equal(data.get_data().toarray(), X)


@pytest.mark.parametrize("expired", [False, True])
def test_prediction_cache_fifo_and_expired_matrices(reference, expired):
    X = np.tile(np.arange(8, dtype=np.float32), 5).reshape(-1, 1)
    params = dict(max_bin=4, max_depth=2, base_score=0)
    expected, files = reference(X, X[:, 0] % 3, params, rounds=2, feature_types=["q"],
                                quantile=True, test_matrix=dict(quantile=True),
                                prediction_steps=[dict(reset=True, feature_types=["c"])])
    model = xgb.Booster(model_file=files / "model.json")
    matrices = []
    for i in range(65):
        data = xgb.QuantileDMatrix(X, max_bin=4, feature_types=["q"])
        np.testing.assert_array_equal(model.predict(data), expected["prediction"])
        if not expired or i == 0:
            matrices.append(data)
    first = matrices[0]
    first.feature_types = ["c"]
    wanted = expected["prediction"] if expired else expected["step_0"]
    np.testing.assert_array_equal(model.predict(first), wanted)
    if not expired:
        matrices[-1].feature_types = ["c"]
        np.testing.assert_array_equal(model.predict(matrices[-1]), expected["prediction"])


def test_prediction_cache_is_per_thread(reference):
    from concurrent.futures import ThreadPoolExecutor
    X = np.tile(np.arange(8, dtype=np.float32), 5).reshape(-1, 1)
    params = dict(max_bin=4, max_depth=2, base_score=0)
    expected, files = reference(X, X[:, 0] % 3, params, rounds=2, feature_types=["q"],
                                quantile=True, test_matrix=dict(quantile=True),
                                prediction_steps=[dict(reset=True, feature_types=["c"])])
    model = xgb.Booster(model_file=files / "model.json")
    matrix = xgb.QuantileDMatrix(X, max_bin=4, feature_types=["q"])
    np.testing.assert_array_equal(model.predict(matrix), expected["prediction"])
    matrix.feature_types = ["c"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        np.testing.assert_array_equal(pool.submit(model.predict, matrix).result(), expected["step_0"])
        np.testing.assert_array_equal(model.predict(matrix), expected["prediction"])
        matrix.feature_types = ["q"]
        np.testing.assert_array_equal(pool.submit(model.predict, matrix).result(), expected["step_0"])


def test_quantile_reference_buffers_outlive_reference_changes():
    X = np.tile(np.arange(8, dtype=np.float32), 5).reshape(-1, 1)
    ref = xgb.QuantileDMatrix(X, max_bin=4, feature_types=["c"], enable_categorical=True)
    query = xgb.QuantileDMatrix(X, max_bin=4, ref=ref, feature_types=["c"], enable_categorical=True)
    original = query.get_data().data.copy()
    cuts = tuple(v.copy() for v in query.get_quantile_cut())
    ref.feature_types = ["q"]
    del ref
    np.testing.assert_array_equal(query.get_data().data, original)
    for actual, wanted in zip(query.get_quantile_cut(), cuts):
        np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("initial,changed", [("q", "c"), ("c", "q")])
def test_quantile_reference_with_reinterpreted_cut_types(reference, initial, changed):
    X = np.tile(np.arange(8, dtype=np.float32), 5).reshape(-1, 1)
    query = np.arange(0, 8, 0.5, dtype=np.float32).reshape(-1, 1)
    params = dict(max_bin=4, max_depth=2, base_score=0)
    expected, files = reference(X, X[:, 0] % 3, params, rounds=2, feature_types=[initial],
                                quantile=True, matrix_data=True, test=query,
                                test_matrix=dict(quantile=True, ref=True, reference_feature_types_after=[changed]))
    train = xgb.QuantileDMatrix(X, max_bin=4, feature_types=[initial], enable_categorical=True)
    train.feature_types = [changed]
    matrix = xgb.QuantileDMatrix(query, max_bin=4, ref=train, feature_types=[initial], enable_categorical=True)
    check_data(matrix, files)
    check_predictions(xgb.Booster(model_file=files / "model.json"), matrix, expected)


@pytest.mark.parametrize("use_ref", [False, True])
@pytest.mark.parametrize("integer_names", [False, True])
def test_quantile_named_category_prediction_keeps_input_codes(reference, use_ref, integer_names):
    rng = np.random.default_rng(447)
    names = [2**60 + i for i in range(4)] if integer_names else ["茶", "apple", "pear", ""]
    codes = rng.integers(0, 4, 257)
    frame = pd.DataFrame({"kind": pd.Categorical.from_codes(codes, names),
                          "numeric": rng.normal(size=len(codes)).astype(np.float32)})
    query = frame.iloc[:69].copy()
    query["kind"] = query["kind"].cat.reorder_categories([names[3], names[0], names[2], names[1]])
    query.loc[::7, "kind"] = np.nan
    query["numeric"] += np.float32(0.7)
    labels = codes + frame["numeric"].to_numpy()
    params = dict(max_bin=8, max_depth=3, base_score=0)
    expected, files = reference(frame, labels, params, rounds=3, test=query, quantile=True,
                                matrix_data=True, test_matrix=dict(quantile=True, ref=use_ref))
    train = xgb.QuantileDMatrix(frame, max_bin=8, enable_categorical=True)
    matrix = xgb.QuantileDMatrix(query, max_bin=8, enable_categorical=True, ref=train if use_ref else None)
    check_data(matrix, files)
    model = xgb.Booster(model_file=files / "model.json")
    check_predictions(model, matrix, expected)
    check_data(matrix, files)


@pytest.mark.parametrize("booster", ["gbtree", "dart"])
@pytest.mark.parametrize("quantile", [False, True])
def test_continuation_preserves_input_category_encoding(reference, tmp_path, booster, quantile):
    codes = np.tile(np.arange(4), 32)
    frame = pd.DataFrame({"cat": pd.Categorical.from_codes(codes, ["a", "b", "c", "d"]),
                          "num": np.sin(np.arange(len(codes))).astype(np.float32)})
    query = frame.copy()
    query["cat"] = query["cat"].cat.reorder_categories(["d", "a", "b", "c"])
    params = dict(max_bin=8, max_depth=2, base_score=0, booster=booster)
    if booster == "dart":
        params.update(rate_drop=0.5, one_drop=True)
    _, initial = reference(frame, codes, params, rounds=2, quantile=quantile)
    expected, files = reference(query, codes, params, rounds=0, model=initial / "model.json",
                                continue_rounds=2, quantile=quantile, matrix_data=True, test=frame)
    cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    matrix = cls(query, label=codes, enable_categorical=True, **({"max_bin": 8} if quantile else {}))
    before = matrix.get_data().data.copy()
    model = xgb.train(params, matrix, 2, xgb_model=initial / "model.json")
    np.testing.assert_array_equal(matrix.get_data().data.view(np.uint32), before.view(np.uint32))
    check_data(matrix, files, "train")
    check_predictions(model, xgb.DMatrix(frame, enable_categorical=True), expected)
