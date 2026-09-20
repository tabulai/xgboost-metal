import json
import pickle

import numpy as np
import pytest
import xgboost as xgb


def assert_cox_trees_match(model, reference_file):
    actual = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]
    expected = json.loads(reference_file.read_text())["learner"]["gradient_booster"]["model"]
    assert actual["iteration_indptr"] == expected["iteration_indptr"]
    assert actual["tree_info"] == expected["tree_info"]
    assert len(actual["trees"]) == len(expected["trees"])
    for index, (tree, target) in enumerate(zip(actual["trees"], expected["trees"])):
        for field in ["left_children", "right_children", "parents", "split_indices", "default_left", "split_type"]:
            assert tree[field] == target[field], (index, field)
        for field in ["base_weights", "split_conditions", "sum_hessian", "loss_changes"]:
            np.testing.assert_array_equal(np.asarray(tree[field], np.float32).view(np.uint32),
                                          np.asarray(target[field], np.float32).view(np.uint32),
                                          err_msg=f"tree {index}: {field}")


def aft_data(rows=281):
    rng = np.random.default_rng(311)
    X = rng.integers(0, 17, size=(rows, 4)).astype(np.float32)
    times = np.exp(0.1 * X[:, 0] - 0.04 * X[:, 1] + rng.normal(0, 0.2, rows)).astype(np.float32)
    lower, upper = times.copy(), times.copy()
    lower[1::5] = 0
    upper[2::5] = np.inf
    lower[3::5] *= 0.7
    upper[3::5] *= 1.3
    lower[4::5], upper[4::5] = 0, np.inf
    X[::19, 2] = np.nan
    return X, dict(label_lower_bound=lower, label_upper_bound=upper)


@pytest.mark.parametrize("distribution", ["normal", "logistic", "extreme"])
@pytest.mark.parametrize("scale", [0.5, 1.7])
@pytest.mark.parametrize("weighted", [False, True])
def test_aft_against_reference(reference, distribution, scale, weighted):
    X, meta = aft_data()
    if weighted:
        meta["weight"] = np.resize(np.asarray([0, 0.25, 0.5, 1, 2], dtype=np.float32), len(X))
    params = dict(objective="survival:aft", aft_loss_distribution=distribution,
                  aft_loss_distribution_scale=scale, max_depth=3, max_bin=32, eta=0.2,
                  eval_metric=["aft-nloglik", "interval-regression-accuracy"])
    expected, files = reference(X, None, params, rounds=6, **meta)
    data = xgb.DMatrix(X, **meta)
    history = {}
    model = xgb.train(params, data, 6, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(model.predict(data, output_margin=True), expected["margin"], rtol=5e-5, atol=5e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    np.testing.assert_allclose(model.predict(data, pred_contribs=True), expected["contributions"], rtol=8e-5, atol=8e-5)
    expected_history = json.loads((files / "history.json").read_text())["train"]
    for metric in params["eval_metric"]:
        np.testing.assert_allclose(history["train"][metric], expected_history[metric], rtol=5e-6, atol=5e-6)
    config = json.loads(model.save_config())["learner"]
    assert config["objective"]["aft_loss_param"]["aft_loss_distribution"] == distribution
    assert json.loads(config["learner_model_param"]["base_score"]) == [0.5]


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_cox_against_reference(reference, ties, weighted):
    rng = np.random.default_rng(219)
    X = rng.integers(0, 19, size=(769, 4)).astype(np.float32)
    time = np.exp(0.08 * X[:, 0] - 0.04 * X[:, 1] + rng.normal(0, 0.2, len(X))).astype(np.float32)
    if ties:
        time = np.maximum(1, np.rint(time * 3)).astype(np.float32)
    time[rng.uniform(size=len(X)) < 0.3] *= -1
    time[::73] = 0
    X[::17, 2] = np.nan
    meta = {}
    if weighted:
        meta["weight"] = np.resize(np.asarray([0, 0.125, 0.5, 1, 3], dtype=np.float32), len(X))
    params = dict(objective="survival:cox", max_depth=3, max_bin=32, eta=0.2)
    expected, files = reference(X, time, params, rounds=6, **meta)
    data = xgb.DMatrix(X, label=time, **meta)
    history = {}
    model = xgb.train(params, data, 6, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(model.predict(data, output_margin=True), expected["margin"], rtol=5e-5, atol=5e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    np.testing.assert_allclose(model.predict(data, pred_contribs=True), expected["contributions"], rtol=8e-5, atol=8e-5)
    expected_history = json.loads((files / "history.json").read_text())["train"]["cox-nloglik"]
    np.testing.assert_allclose(history["train"]["cox-nloglik"], expected_history, rtol=3e-6, atol=3e-6)
    assert json.loads(json.loads(model.save_config())["learner"]["learner_model_param"]["base_score"]) == [1]


@pytest.mark.parametrize("distribution", ["normal", "logistic", "extreme"])
def test_aft_default_metric_configuration(distribution):
    X, meta = aft_data(31)
    data = xgb.DMatrix(X, **meta)
    model = xgb.train(dict(objective="survival:aft", aft_loss_distribution=distribution,
                           aft_loss_distribution_scale=1.25), data, 1)
    config = json.loads(model.save_config())["learner"]
    assert config["metrics"] == [{"name": "aft-nloglik", "aft_loss_param": {
        "aft_loss_distribution": distribution, "aft_loss_distribution_scale": "1.25"}}]
    assert pickle.loads(pickle.dumps(model)).eval(data) == model.eval(data)


def test_aft_empty_data():
    data = xgb.DMatrix(np.empty((0, 2), dtype=np.float32), label_lower_bound=[], label_upper_bound=[])
    history = {}
    model = xgb.train(dict(objective="survival:aft"), data, 2, evals=[(data, "empty")],
                      evals_result=history, verbose_eval=False)
    assert model.num_boosted_rounds() == 2
    assert model.predict(data).shape == (0, 0)
    assert history["empty"]["aft-nloglik"] == [0, 0]


@pytest.mark.parametrize("distribution", ["normal", "logistic", "extreme"])
@pytest.mark.parametrize("bounds", [(100, 100), (0, 20), (60, np.inf), (16, 200)])
@pytest.mark.parametrize("scale", [0.2, 1, 4])
def test_aft_extreme_margins_and_censoring(reference, distribution, bounds, scale):
    # Weighted, asymmetric margins prevent cancellation from hiding incorrect
    # tail gradients. A constant-feature stump exposes the aggregate curvature.
    margin = np.asarray([-1000, -120, -40, -10, -7, -1, 0, 4, 7, 10, 40, 120, 1000], dtype=np.float32)
    X = np.zeros((len(margin), 1), dtype=np.float32)
    meta = dict(label_lower_bound=np.full(len(X), bounds[0], dtype=np.float32),
                label_upper_bound=np.full(len(X), bounds[1], dtype=np.float32),
                base_margin=margin, weight=np.linspace(0.1, 1.7, len(X), dtype=np.float32))
    params = dict(objective="survival:aft", aft_loss_distribution=distribution,
                  aft_loss_distribution_scale=scale, max_depth=1, min_child_weight=0, eta=1)
    expected, files = reference(X, None, params, rounds=1, test_margin=margin, **meta)
    data = xgb.DMatrix(X, **meta)
    model = xgb.train(params, data, 1)
    actual_margin = model.predict(data, output_margin=True)
    assert np.isfinite(actual_margin).all()
    np.testing.assert_allclose(actual_margin, expected["margin"], rtol=3e-6, atol=3e-6)
    node = json.loads(model.get_dump(dump_format="json", with_stats=True)[0])
    oracle = json.loads(json.loads((files / "dump.json").read_text())[0])
    assert node["cover"] == pytest.approx(oracle["cover"], rel=3e-6, abs=1e-6)


@pytest.mark.parametrize("objective,distribution", [
    ("survival:cox", None), ("survival:aft", "normal"),
    ("survival:aft", "logistic"), ("survival:aft", "extreme"),
])
def test_survival_exchange_and_continuation(reference, tmp_path, objective, distribution):
    X, meta = aft_data(113)
    X = np.nan_to_num(X, nan=0)
    params = dict(objective=objective, max_depth=3, max_bin=32, num_parallel_tree=2)
    label = None
    if distribution is not None:
        params.update(aft_loss_distribution=distribution, aft_loss_distribution_scale=1.5)
    else:
        label = np.exp(X[:, 0] / 10).astype(np.float32)
        label[::3] *= -1
        meta = dict(label=label)
    data = xgb.DMatrix(X, feature_types=["c"] * X.shape[1], enable_categorical=True, **meta)
    first = xgb.train(params, data, 3)
    full = xgb.train(params, data, 6)
    continued = xgb.train(params, data, 3, xgb_model=first)
    np.testing.assert_array_equal(continued.predict(data), full.predict(data))
    np.testing.assert_array_equal(full[:3].predict(data), first.predict(data))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(full)).predict(data), full.predict(data))
    for kind in ["json", "ubj"]:
        path = tmp_path / f"survival.{kind}"
        full.save_model(path)
        expected, _ = reference(X, label, params, model=path, feature_types=["c"] * X.shape[1],
                                **{key: value for key, value in meta.items() if key != "label"})
        for field, options in [("prediction", {}), ("margin", dict(output_margin=True)),
                               ("leaf", dict(pred_leaf=True)), ("contributions", dict(pred_contribs=True)),
                               ("interactions", dict(pred_interactions=True))]:
            np.testing.assert_allclose(full.predict(data, **options), expected[field], rtol=6e-5, atol=6e-5)
        np.testing.assert_array_equal(xgb.Booster(model_file=path).predict(data), full.predict(data))
    expected, files = reference(X, label, params, rounds=3, feature_types=["c"] * X.shape[1],
                                **{key: value for key, value in meta.items() if key != "label"})
    imported = xgb.Booster(model_file=files / "model.ubj")
    np.testing.assert_allclose(imported.predict(data), expected["prediction"], rtol=3e-6, atol=3e-6)


def test_cox_recursive_scan(reference):
    rng = np.random.default_rng(503)
    X = rng.integers(0, 17, size=(65537, 3)).astype(np.float32)
    label = rng.integers(1, 127, len(X)).astype(np.float32)
    label[::3] *= -1
    weight = np.resize(np.asarray([0, 0.25, 1, 2], dtype=np.float32), len(X))
    params = dict(objective="survival:cox", max_depth=3, max_bin=32, eta=0.15)
    expected, files = reference(X, label, params, rounds=3, test=X[:83], weight=weight)
    data = xgb.DMatrix(X, label=label, weight=weight)
    history = {}
    model = xgb.train(params, data, 3, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    test = xgb.DMatrix(X[:83])
    np.testing.assert_allclose(model.predict(test), expected["prediction"], rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(model.predict(test, pred_leaf=True), expected["leaf"])
    expected_history = json.loads((files / "history.json").read_text())["train"]["cox-nloglik"]
    np.testing.assert_allclose(history["train"]["cox-nloglik"], expected_history, rtol=2e-6, atol=2e-6)


def test_cox_long_training_with_bounded_updates(reference):
    # There are only 33 distinct values per feature, so this comparison does
    # not involve compressed-quantile differences. The reference's mixed float
    # exponentials and ordered double subtraction become observable late in
    # training; ordinary wider parallel sums change its later tree structures.
    rng = np.random.default_rng(751)
    X = (rng.integers(-16, 17, size=(8193, 4)) / np.float32(4)).astype(np.float32)
    response = (2 * X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 0.1, len(X))).astype(np.float32)
    label = np.exp(np.float32(0.3) * response).astype(np.float32)
    label[rng.uniform(size=len(X)) < 0.3] *= -1
    params = dict(objective="survival:cox", max_depth=6, max_bin=64, max_delta_step=1)
    expected, files = reference(X, label, params, rounds=100, test=X[:127])
    data = xgb.DMatrix(X, label=label)
    history = {}
    model = xgb.train(params, data, 100, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    test = xgb.DMatrix(X[:127])
    np.testing.assert_array_equal(model.predict(test), expected["prediction"])
    np.testing.assert_array_equal(model.predict(test, output_margin=True), expected["margin"])
    np.testing.assert_array_equal(model.predict(test, pred_leaf=True), expected["leaf"])
    assert_cox_trees_match(model, files / "model.json")
    oracle_history = json.loads((files / "history.json").read_text())["train"]["cox-nloglik"]
    np.testing.assert_allclose(history["train"]["cox-nloglik"], oracle_history, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("rows,tie_size,lower,upper", [
    (257, 1, -5, 5), (1025, 7, -5, 5), (8193, 257, -5, 5),
    (65537, 3000, -5, 5), (8193, 8193, -5, 5),
    (4097, 1, -30, 25), (1025, 7, -105, -80),
])
def test_cox_ordered_scans_and_mixed_exponentials(reference, rows, tie_size, lower, upper):
    # Ties cross SIMD, tile and recursive-scan boundaries. The last two cases
    # expose cancellation in the risk set and float32 exponential underflow.
    rng = np.random.default_rng(817)
    X = rng.integers(0, 31, (rows, 3)).astype(np.float32)
    label = (np.arange(rows) // tie_size + 1).astype(np.float32)[rng.permutation(rows)]
    label[::3] *= -1
    margin = rng.uniform(lower, upper, rows).astype(np.float32)
    weight = np.resize(np.asarray([0, 0.125, 0.75, 1, 1.5, 3], np.float32), rows)
    params = dict(objective="survival:cox", max_depth=3, max_bin=64, max_delta_step=1, eta=0.2)
    expected, files = reference(X, label, params, rounds=3, base_margin=margin, test_margin=margin, weight=weight)
    data = xgb.DMatrix(X, label=label, base_margin=margin, weight=weight)
    model = xgb.train(params, data, 3)
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    np.testing.assert_array_equal(model.predict(data, output_margin=True), expected["margin"])
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    assert_cox_trees_match(model, files / "model.json")


def test_cox_hazard_transform_matches_platform_expf(reference):
    rng = np.random.default_rng(911)
    boundaries = np.asarray([-1000, -104, -100, -90, -88, -1, -0.0, 0, 1, 88, 89, 104, 1000], np.float32)
    margin = np.concatenate([boundaries, rng.uniform(-128, 128, 4096).astype(np.float32)])
    X = np.zeros((len(margin), 1), np.float32)
    label = np.ones(len(margin), np.float32)
    params = dict(objective="survival:cox")
    expected, _ = reference(X, label, params, rounds=0, base_margin=margin, test_margin=margin)
    data = xgb.DMatrix(X, label=label, base_margin=margin)
    model = xgb.train(params, data, 0)
    np.testing.assert_array_equal(model.predict(data).view(np.uint32), expected["prediction"].view(np.uint32))
    np.testing.assert_array_equal(model.inplace_predict(X, base_margin=margin).view(np.uint32),
                                  expected["prediction"].view(np.uint32))


def test_cox_metric_weight_semantics_and_order_cache(reference, tmp_path):
    X = (np.arange(291, dtype=np.float32).reshape(97, 3) % 13)
    label = (X[:, 0] + 1).copy()
    label[::3] *= -1
    data = xgb.DMatrix(X, label=label)
    params = dict(objective="survival:cox", max_depth=3, max_bin=32)
    model = xgb.train(params, data, 4)
    model.eval(data)  # Cache the original time ordering.
    changed = np.roll(label[::-1], 7).copy()
    data.set_label(changed)
    actual = float(model.eval(data).split(":")[-1])
    data.set_weight(np.zeros(len(X)))
    assert float(model.eval(data).split(":")[-1]) == actual
    path = tmp_path / "cox.json"
    model.save_model(path)
    _, files = reference(X, changed, params, model=path, evaluation=True, weight=np.zeros(len(X)))
    expected = float((files / "evaluation.txt").read_text().split(":")[-1])
    assert actual == pytest.approx(expected, rel=2e-7)


@pytest.mark.parametrize("distribution", ["normal", "logistic", "extreme"])
@pytest.mark.parametrize("weight", [0, 1, 1e38, np.nextafter(np.float32(0), np.float32(1))])
def test_aft_metric_weight_range(reference, distribution, weight):
    X = np.zeros((1025, 1), dtype=np.float32)
    lower = np.resize(np.asarray([100, 0, 60, 16], dtype=np.float32), len(X))
    upper = np.resize(np.asarray([100, 20, np.inf, 200], dtype=np.float32), len(X))
    meta = dict(label_lower_bound=lower, label_upper_bound=upper, weight=np.full(len(X), weight, dtype=np.float32))
    params = dict(objective="survival:aft", aft_loss_distribution=distribution, base_score=64, eval_metric="aft-nloglik")
    _, files = reference(X, None, params, rounds=0, evaluation=True, **meta)
    data = xgb.DMatrix(X, **meta)
    model = xgb.train(params, data, 0)
    actual = float(model.eval(data).split(":")[-1])
    expected = float((files / "evaluation.txt").read_text().split(":")[-1])
    assert actual == pytest.approx(expected, rel=2e-7, abs=1e-12)


def test_interval_accuracy_uses_double_exponential(reference):
    X = np.zeros((7, 1), dtype=np.float32)
    meta = dict(label_lower_bound=np.full(7, 2, dtype=np.float32), label_upper_bound=np.full(7, 2, dtype=np.float32))
    params = dict(objective="survival:aft", base_score=2, eval_metric="interval-regression-accuracy")
    _, files = reference(X, None, params, rounds=0, evaluation=True, **meta)
    data = xgb.DMatrix(X, **meta)
    model = xgb.train(params, data, 0)
    # The float32 prediction rounds to 2, but exp(double(logf(2))) is above 2.
    np.testing.assert_array_equal(model.predict(data), np.full(7, 2, dtype=np.float32))
    assert float(model.eval(data).split(":")[-1]) == 0
    assert float((files / "evaluation.txt").read_text().split(":")[-1]) == 0


def test_cox_without_events():
    data = xgb.DMatrix(np.zeros((67, 2), dtype=np.float32), label=-np.arange(1, 68, dtype=np.float32))
    model = xgb.train(dict(objective="survival:cox"), data, 3)
    np.testing.assert_array_equal(model.predict(data), np.ones(67, dtype=np.float32))
    assert np.isnan(float(model.eval(data).split(":")[-1]))


@pytest.mark.parametrize("missing", ["label_lower_bound", "label_upper_bound"])
def test_aft_requires_censoring_bounds(missing):
    X, meta = aft_data(7)
    del meta[missing]
    with pytest.raises(xgb.core.XGBoostError, match="label_lower_bound.*label_upper_bound"):
        xgb.train(dict(objective="survival:aft"), xgb.DMatrix(X, **meta), 1)


@pytest.mark.parametrize("objective", ["survival:aft", "survival:cox"])
def test_survival_requires_single_output(objective):
    X, meta = aft_data(7)
    with pytest.raises(xgb.core.XGBoostError, match="Multioutput"):
        xgb.train(dict(objective=objective), xgb.DMatrix(X, label=np.ones((7, 2)), **meta), 1)


def test_aft_rejects_unknown_distribution():
    X, meta = aft_data(7)
    with pytest.raises(xgb.core.XGBoostError, match="aft_loss_distribution"):
        xgb.train(dict(objective="survival:aft", aft_loss_distribution="gaussian"), xgb.DMatrix(X, **meta), 1)


@pytest.mark.parametrize("distribution", ["normal", "logistic", "extreme"])
@pytest.mark.parametrize("scale", [0, -1])
def test_aft_reference_nonpositive_scales(reference, distribution, scale):
    # The pinned parameter parser accepts these scales. Preserve its clipping
    # behavior rather than imposing a new parameter restriction.
    X, meta = aft_data(37)
    params = dict(objective="survival:aft", aft_loss_distribution=distribution,
                  aft_loss_distribution_scale=scale, eval_metric="aft-nloglik",
                  max_depth=2, max_bin=32)
    expected, _ = reference(X, None, params, rounds=2, **meta)
    model = xgb.train(params, xgb.DMatrix(X, **meta), 2)
    np.testing.assert_allclose(model.predict(xgb.DMatrix(X)), expected["prediction"], rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("distribution", ["normal", "logistic", "extreme"])
def test_aft_adjacent_float_bounds(reference, distribution):
    lower = np.tile(np.asarray([1e-20, 0.125, 1, 100, 1e20], dtype=np.float32), 5)
    upper = np.nextafter(lower, np.float32(np.inf))
    margin = (np.log(lower.astype(np.float64)) + np.repeat([-3, -1, 0, 1, 3], 5)).astype(np.float32)
    X = np.zeros((len(lower), 1), dtype=np.float32)
    meta = dict(label_lower_bound=lower, label_upper_bound=upper, base_margin=margin,
                weight=np.linspace(0.2, 2, len(X), dtype=np.float32))
    params = dict(objective="survival:aft", aft_loss_distribution=distribution,
                  max_depth=1, min_child_weight=0, eta=1)
    expected, files = reference(X, None, params, rounds=1, test_margin=margin, **meta)
    data = xgb.DMatrix(X, **meta)
    model = xgb.train(params, data, 1)
    np.testing.assert_allclose(model.predict(data, output_margin=True), expected["margin"], rtol=3e-6, atol=3e-6)
    node = json.loads(model.get_dump(dump_format="json", with_stats=True)[0])
    oracle = json.loads(json.loads((files / "dump.json").read_text())[0])
    assert node["cover"] == pytest.approx(oracle["cover"], rel=3e-6, abs=1e-6)


def test_aft_quantile_matrix_slicing_and_cv():
    X, meta = aft_data(97)
    params = dict(objective="survival:aft", aft_loss_distribution="logistic", max_depth=2, max_bin=32)
    dense = xgb.DMatrix(X, **meta)
    quantile = xgb.QuantileDMatrix(X, max_bin=32, **meta)
    dense_model = xgb.train(params, dense, 4)
    quantile_model = xgb.train(params, quantile, 4)
    np.testing.assert_array_equal(dense_model.predict(dense), quantile_model.predict(dense))
    indices = np.arange(0, len(X), 3, dtype=np.int32)
    subset = dense.slice(indices)
    for name, values in meta.items():
        np.testing.assert_array_equal(subset.get_float_info(name), values[indices])
    history = xgb.cv(params, dense, 3, nfold=3, seed=11, shuffle=True, as_pandas=False)
    assert len(history["test-aft-nloglik-mean"]) == 3
    assert np.isfinite(history["test-aft-nloglik-mean"]).all()


def test_cox_sklearn_weighted_early_stopping():
    rng = np.random.default_rng(718)
    X = rng.integers(0, 17, size=(191, 4)).astype(np.float32)
    label = np.exp(X[:, 0] / 8 - X[:, 1] / 12).astype(np.float32)
    label[::4] *= -1
    weight = rng.uniform(0.1, 2, len(X)).astype(np.float32)
    model = xgb.XGBRegressor(objective="survival:cox", n_estimators=8, max_depth=3,
                             max_bin=32, early_stopping_rounds=3)
    model.fit(X, label, sample_weight=weight, eval_set=[(X, label)],
              sample_weight_eval_set=[weight], verbose=False)
    expected = model.get_booster().predict(xgb.DMatrix(X), iteration_range=(0, model.best_iteration + 1))
    np.testing.assert_array_equal(model.predict(X), expected)
    assert np.isfinite(model.evals_result()["validation_0"]["cox-nloglik"]).all()
