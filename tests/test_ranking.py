import json
import pickle

import numpy as np
import pytest
import xgboost as xgb


def ranking_data(binary=False):
    rng = np.random.default_rng(907)
    sizes = np.asarray([17, 13, 31, 1, 2, 57, 255, 257, 513], dtype=np.uint32)
    X = rng.integers(0, 17, (sizes.sum(), 5)).astype(np.float32)
    score = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 2, len(X))
    y = (score > 12).astype(np.float32) if binary else np.clip(score // 5, 0, 4).astype(np.float32)
    X[::23, 3] = np.nan
    return X, y, sizes


@pytest.mark.parametrize("objective", ["rank:ndcg", "rank:map", "rank:pairwise"])
@pytest.mark.parametrize("method", ["topk", "mean"])
@pytest.mark.parametrize("weighted", [False, True])
def test_ranking_against_reference(reference, objective, method, weighted):
    X, y, sizes = ranking_data(objective == "rank:map")
    meta = {"group": sizes}
    if weighted:
        meta["weight"] = np.asarray([1, 2, 0, 2, 3, 1, 2, 3, 1], dtype=np.float32)
    params = dict(objective=objective, lambdarank_pair_method=method,
                  lambdarank_num_pair_per_sample=5, max_depth=3, max_bin=32, eta=0.2, seed=71)
    expected, files = reference(X, y, params, rounds=6, **meta)
    data = xgb.DMatrix(X, label=y, **meta)
    history = {}
    model = xgb.train(params, data, 6, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    np.testing.assert_allclose(model.predict(data, pred_contribs=True), expected["contributions"], rtol=3e-5, atol=3e-5)
    expected_history = json.loads((files / "history.json").read_text())["train"]
    assert history["train"].keys() == expected_history.keys()
    for metric in expected_history:
        np.testing.assert_allclose(history["train"][metric], expected_history[metric], rtol=2e-7, atol=2e-7)
    actual_config = json.loads(model.save_config())["learner"]
    expected_config = json.loads((files / "config.json").read_text())["learner"]
    assert actual_config["objective"] == expected_config["objective"]
    assert actual_config["metrics"] == expected_config["metrics"]
    assert json.loads(actual_config["learner_model_param"]["base_score"]) == [0]


@pytest.mark.parametrize("normalization,score_normalization", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("method", ["topk", "mean"])
def test_ranking_normalization(reference, normalization, score_normalization, method):
    X, y, sizes = ranking_data()
    margin = (np.arange(len(X)) % 11 / 3).astype(np.float32)
    params = dict(objective="rank:ndcg", lambdarank_pair_method=method,
                  lambdarank_num_pair_per_sample=3, lambdarank_normalization=normalization,
                  lambdarank_score_normalization=score_normalization, max_depth=2, max_bin=32,
                  eta=0.15, seed=33)
    expected, _ = reference(X, y, params, rounds=4, group=sizes, base_margin=margin, test_margin=margin)
    data = xgb.DMatrix(X, label=y, group=sizes, base_margin=margin)
    model = xgb.train(params, data, 4)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])


@pytest.mark.parametrize("objective", ["rank:ndcg", "rank:map", "rank:pairwise"])
@pytest.mark.parametrize("method", ["topk", "mean"])
def test_unbiased_ranking(reference, objective, method):
    X, y, sizes = ranking_data(True)
    params = dict(objective=objective, lambdarank_pair_method=method, lambdarank_num_pair_per_sample=7,
                  lambdarank_unbiased=True, lambdarank_bias_norm=0.5, max_depth=2, max_bin=32, eta=0.15, seed=12)
    expected, files = reference(X, y, params, rounds=5, group=sizes)
    data = xgb.DMatrix(X, label=y, group=sizes)
    model = xgb.train(params, data, 5)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    ours = json.loads(model.save_config())["learner"]["objective"]
    theirs = json.loads((files / "config.json").read_text())["learner"]["objective"]
    assert ours["lambdarank_param"] == theirs["lambdarank_param"]
    for key in ("ti+", "tj-"):
        np.testing.assert_allclose(ours[key], theirs[key], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("metric", ["ndcg", "ndcg@1", "ndcg@7", "ndcg@32-", "ndcg-", "map", "map@1", "map@7-", "pre", "pre@7", "auc", "aucpr"])
def test_ranking_metrics(reference, metric):
    X, y, sizes = ranking_data(True)
    margin = (np.arange(len(X)) % 13).astype(np.float32)
    # Include an all-irrelevant query and tied predictions, with query weights.
    y[: sizes[0]] = 0
    weight = np.asarray([2, 3, 1, 2, 1, 0, 3, 2, 1], dtype=np.float32)
    params = dict(objective="rank:pairwise", eval_metric=metric, base_score=0)
    _, files = reference(X, y, params, rounds=0, group=sizes, weight=weight, base_margin=margin, evaluation=True)
    data = xgb.DMatrix(X, label=y, group=sizes, weight=weight, base_margin=margin)
    model = xgb.train(params, data, 0)
    actual = float(model.eval(data).rsplit(":", 1)[1])
    expected = float((files / "evaluation.txt").read_text().rsplit(":", 1)[1])
    assert actual == pytest.approx(expected, rel=2e-9, abs=2e-9)
    assert json.loads(model.save_config())["learner"]["metrics"] == json.loads((files / "config.json").read_text())["learner"]["metrics"]


@pytest.mark.parametrize("method", ["topk", "mean"])
@pytest.mark.parametrize("gain", [False, True])
def test_ranking_relevance_range(reference, method, gain):
    X, y, sizes = ranking_data()
    y = y / np.float32(3) + np.float32(0.125) if not gain else (y * 7 + 3).astype(np.float32)
    params = dict(objective="rank:ndcg", ndcg_exp_gain=gain, lambdarank_pair_method=method,
                  lambdarank_num_pair_per_sample=3, max_depth=2, max_bin=32, eta=0.15)
    expected, files = reference(X, y, params, rounds=4, group=sizes)
    data = xgb.DMatrix(X, label=y, group=sizes)
    history = {}
    model = xgb.train(params, data, 4, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    for name, values in json.loads((files / "history.json").read_text())["train"].items():
        np.testing.assert_allclose(history["train"][name], values, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize("unbiased", [False, True])
@pytest.mark.parametrize("method", ["topk", "mean"])
def test_ranking_exchange_and_continuation(reference, tmp_path, unbiased, method):
    X, y, sizes = ranking_data(True)
    data = xgb.DMatrix(X, label=y, group=sizes)
    params = dict(objective="rank:ndcg", lambdarank_pair_method=method, lambdarank_num_pair_per_sample=5,
                  lambdarank_unbiased=unbiased, max_depth=2, max_bin=32, eta=0.15, seed=67)
    model = xgb.train(params, data, 3)
    before = model.predict(data)
    for extension in ("json", "ubj"):
        path = tmp_path / f"ranking.{extension}"
        model.save_model(path)
        expected, _ = reference(X, y, params, model=path, group=sizes)
        np.testing.assert_array_equal(before, expected["prediction"])
        restored = xgb.Booster(model_file=path)
        np.testing.assert_array_equal(before, restored.predict(data))
        assert json.loads(restored.save_config())["learner"]["objective"] == json.loads(model.save_config())["learner"]["objective"]
        expected, _ = reference(X, y, params, model=path, continue_rounds=3, group=sizes)
        continued = xgb.train(params, data, 3, xgb_model=restored)
        np.testing.assert_allclose(continued.predict(data), expected["prediction"], rtol=3e-5, atol=3e-5)
        np.testing.assert_array_equal(continued.predict(data, pred_leaf=True), expected["leaf"])
    cloned = pickle.loads(pickle.dumps(model))
    assert cloned.save_config() == model.save_config()
    np.testing.assert_array_equal(cloned.predict(data), before)
    suffix = model[1:]
    np.testing.assert_allclose(suffix.predict(data), model.predict(data, iteration_range=(1, 3)), atol=1e-7)


@pytest.mark.parametrize("per_iteration", [False, True])
def test_ranking_random_state_with_column_sampling(reference, per_iteration):
    X, y, sizes = ranking_data()
    params = dict(objective="rank:ndcg", lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=3,
                  seed=271, seed_per_iteration=int(per_iteration), colsample_bytree=0.8,
                  colsample_bylevel=0.75, colsample_bynode=0.75, max_depth=3, max_bin=32)
    expected, files = reference(X, y, params, rounds=6, group=sizes)
    data = xgb.DMatrix(X, label=y, group=sizes)
    model = xgb.train(params, data, 6)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    ours = json.loads(model.save_config())["learner"]["generic_param"]["rng_state"]
    theirs = json.loads((files / "config.json").read_text())["learner"]["generic_param"]["rng_state"]
    assert ours == theirs


def test_ranker_dataframe_quantile_and_cross_validation():
    import pandas as pd

    X, y, sizes = ranking_data()
    qid = np.repeat(np.arange(len(sizes), dtype=np.uint32) * 7, sizes)
    weights = np.asarray([1, 2, 1, 2, 3, 1, 2, 3, 1], dtype=np.float32)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    frame["qid"] = qid
    ranker = xgb.XGBRanker(n_estimators=4, max_depth=2, max_bin=32, learning_rate=0.2)
    ranker.fit(frame, y, sample_weight=weights, eval_set=[(frame, y)], sample_weight_eval_set=[weights], verbose=False)
    assert ranker.n_features_in_ == X.shape[1]
    assert np.isfinite(ranker.predict(frame)).all()
    assert 0 <= ranker.score(frame, y) <= 1
    data = xgb.DMatrix(X, label=y, qid=qid, weight=weights)
    quantile = xgb.QuantileDMatrix(X, label=y, qid=qid, weight=weights, max_bin=32)
    np.testing.assert_array_equal(data.get_group(), sizes)
    np.testing.assert_array_equal(quantile.get_group(), sizes)
    params = dict(objective="rank:ndcg", max_depth=2, max_bin=32, eta=0.2)
    ordinary = xgb.train(params, data, 4)
    reduced = xgb.train(params, quantile, 4)
    np.testing.assert_array_equal(ordinary.predict(data), reduced.predict(data))
    np.testing.assert_array_equal(ranker.predict(frame), ordinary.predict(data))
    result = xgb.cv(params, data, num_boost_round=3, nfold=3, shuffle=True, seed=1)
    assert len(result) == 3
    assert np.isfinite(result.to_numpy()).all()


def test_ranking_libsvm_query_ids(tmp_path):
    path = tmp_path / "rank.libsvm"
    path.write_text("2 qid:10 0:3 1:1\n0 qid:10 0:1 1:2\n1 qid:22 0:2 1:3\n0 qid:22 0:0 1:4\n")
    data = xgb.DMatrix(f"{path}?format=libsvm")
    ordinary = xgb.DMatrix(np.asarray([[3, 1], [1, 2], [2, 3], [0, 4]], dtype=np.float32), label=[2, 0, 1, 0], group=[2, 2])
    np.testing.assert_array_equal(data.get_group(), [2, 2])
    params = dict(objective="rank:ndcg", max_depth=1, min_child_weight=0)
    np.testing.assert_array_equal(xgb.train(params, data, 3).predict(data), xgb.train(params, ordinary, 3).predict(ordinary))


@pytest.mark.parametrize("objective,labels,parameters", [
    ("rank:ndcg", [0, 0.5, 1], {}), ("rank:ndcg", [0, 32, 1], {}),
    ("rank:ndcg", [0, -1, 1], {}), ("rank:map", [0, 2, 1], {}),
    ("rank:ndcg", [0, 1, 1], {"lambdarank_num_pair_per_sample": 0}),
    ("rank:ndcg", [0, 1, 1], {"lambdarank_pair_method": "invalid"}),
    ("rank:ndcg", [0, 1, 1], {"lambdarank_bias_norm": -1}),
    ("rank:ndcg", [0, 1, 1], {"seed_per_iteration": "invalid"}),
])
def test_ranking_invalid_parameters(objective, labels, parameters):
    data = xgb.DMatrix(np.arange(6).reshape(3, 2), label=labels, group=[3])
    with pytest.raises(xgb.core.XGBoostError):
        xgb.train(dict(objective=objective, **parameters), data, 1)


@pytest.mark.parametrize("objective", ["rank:ndcg", "rank:map", "rank:pairwise"])
def test_ranking_empty_and_single_query(reference, objective):
    X = np.empty((0, 2), dtype=np.float32)
    data = xgb.DMatrix(X, label=[])
    history = {}
    model = xgb.train(dict(objective=objective), data, 2, evals=[(data, "train")], evals_result=history, verbose_eval=False)
    assert model.predict(data).shape == (0, 0)
    assert next(iter(history["train"].values())) == [1, 1]
    X = np.arange(66, dtype=np.float32).reshape(33, 2) % 7
    y = (X[:, 0] > 3).astype(np.float32)
    expected, _ = reference(X, y, dict(objective=objective, max_depth=2), rounds=3)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(dict(objective=objective, max_depth=2), data, 3)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("method", ["topk", "mean"])
def test_ranking_query_boundaries_and_bias_reductions(reference, method):
    rng = np.random.default_rng(512)
    sizes = np.asarray([1, 2, 3, 31, 32, 33, 255, 256, 257, 1025] + [7] * 257, dtype=np.uint32)
    X = rng.integers(0, 13, (sizes.sum(), 3)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 12).astype(np.float32)
    params = dict(objective="rank:map", lambdarank_pair_method=method, lambdarank_num_pair_per_sample=5,
                  lambdarank_unbiased=True, max_depth=2, max_bin=16, eta=0.2)
    expected, files = reference(X, y, params, rounds=3, group=sizes)
    data = xgb.DMatrix(X, label=y, group=sizes)
    model = xgb.train(params, data, 3)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    oracle = json.loads((files / "config.json").read_text())["learner"]["objective"]
    actual = json.loads(model.save_config())["learner"]["objective"]
    for key in ("ti+", "tj-"):
        np.testing.assert_allclose(actual[key], oracle[key], rtol=3e-5, atol=3e-6)


def test_large_ranking_query_recursive_scan(reference):
    rng = np.random.default_rng(192)
    X = rng.integers(0, 13, (65537, 2)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 11).astype(np.float32)
    params = dict(objective="rank:map", max_depth=2, max_bin=16, lambdarank_num_pair_per_sample=3)
    expected, _ = reference(X, y, params, rounds=2, group=[len(X)])
    data = xgb.DMatrix(X, label=y, group=[len(X)])
    model = xgb.train(params, data, 2)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])


@pytest.mark.parametrize("quantile", [False, True])
def test_ranking_iterator_query_weights(quantile):
    X, y, sizes = ranking_data()
    offsets = np.r_[0, np.cumsum(sizes)]
    weight = np.arange(len(sizes), dtype=np.float32) + 1

    class Batches(xgb.DataIter):
        def __init__(self):
            self.batch = 0
            super().__init__()

        def reset(self):
            self.batch = 0

        def next(self, input_data):
            if self.batch == 3:
                return False
            g = self.batch * 3
            begin, end = offsets[g], offsets[g + 3]
            input_data(data=X[begin:end], label=y[begin:end],
                       qid=np.repeat(np.arange(g, g + 3, dtype=np.uint32), sizes[g:g + 3]), weight=weight[g:g + 3])
            self.batch += 1
            return True

    cls = xgb.QuantileDMatrix if quantile else xgb.DMatrix
    kwargs = {"max_bin": 32} if quantile else {}
    data = cls(Batches(), **kwargs)
    ordinary = cls(X, label=y, group=sizes, weight=weight, **kwargs)
    np.testing.assert_array_equal(data.get_group(), sizes)
    np.testing.assert_array_equal(data.get_weight(), weight)
    params = dict(objective="rank:ndcg", max_depth=2, max_bin=32)
    np.testing.assert_array_equal(xgb.train(params, data, 3).predict(data),
                                  xgb.train(params, ordinary, 3).predict(ordinary))


@pytest.mark.parametrize("seed", [-1, 4294967294])
def test_ranking_seed_range(reference, seed):
    X, y, sizes = ranking_data()
    params = dict(objective="rank:ndcg", seed=seed, seed_per_iteration=True, lambdarank_pair_method="mean",
                  lambdarank_num_pair_per_sample=3, max_depth=2, max_bin=32)
    expected, _ = reference(X, y, params, rounds=3, group=sizes)
    data = xgb.DMatrix(X, label=y, group=sizes)
    model = xgb.train(params, data, 3)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=3e-5, atol=3e-5)
    assert json.loads(model.save_config())["learner"]["generic_param"]["seed_per_iteration"] == "1"


def test_ranking_categorical_exchange(reference, tmp_path):
    rng = np.random.default_rng(85)
    X = rng.integers(0, 7, (271, 3)).astype(np.float32)
    y = ((X[:, 0] % 3) + (X[:, 1] > 3)).astype(np.float32)
    X[::17, 0] = np.nan
    sizes = np.asarray([33, 27, 75, 136], dtype=np.uint32)
    params = dict(objective="rank:ndcg", max_depth=3, max_bin=16, max_cat_to_onehot=2)
    expected, files = reference(X, y, params, rounds=3, group=sizes, feature_types=["c"] * 3)
    data = xgb.DMatrix(X, label=y, group=sizes, feature_types=["c"] * 3, enable_categorical=True)
    model = xgb.train(params, data, 3)
    np.testing.assert_allclose(model.predict(data), expected["prediction"], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(model.predict(data, pred_interactions=True), expected["interactions"], rtol=3e-5, atol=3e-5)
    imported = xgb.Booster(model_file=files / "model.ubj")
    np.testing.assert_array_equal(imported.predict(data), expected["prediction"])
    model.save_model(tmp_path / "categorical-rank.ubj")
    reloaded, _ = reference(X, y, params, model=tmp_path / "categorical-rank.ubj", group=sizes, feature_types=["c"] * 3)
    np.testing.assert_array_equal(reloaded["prediction"], model.predict(data))


@pytest.mark.parametrize("objective", ["rank:ndcg", "rank:map", "rank:pairwise"])
def test_empty_query_segments(objective):
    # The pinned CPU MAP implementation aborts on empty interior queries.
    # Exercise safe Metal boundary handling without invoking that oracle path.
    X, y, sizes = ranking_data(True)
    expanded = np.column_stack([sizes, np.zeros_like(sizes)]).ravel()
    data = xgb.DMatrix(X, label=y, group=expanded)
    ordinary = xgb.DMatrix(X, label=y, group=sizes)
    params = dict(objective=objective, max_depth=2, max_bin=32)
    model = xgb.train(params, data, 3)
    expected = xgb.train(params, ordinary, 3)
    np.testing.assert_array_equal(model.predict(data), expected.predict(ordinary))


@pytest.mark.parametrize("objective", ["rank:ndcg", "rank:map", "rank:pairwise"])
@pytest.mark.parametrize("method", ["topk", "mean"])
def test_ranking_workspace_and_simd_aggregation_are_exact(objective, method):
    X, y, sizes = ranking_data(objective == "rank:map")
    data = xgb.DMatrix(X, label=y, group=sizes)
    params = dict(objective=objective, lambdarank_pair_method=method, lambdarank_num_pair_per_sample=5,
                  lambdarank_unbiased=True, max_depth=3, max_bin=32, seed=81)
    baseline = None
    for cache, aggregate in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        model = xgb.train({**params, "metal_ranking_cache": cache, "metal_ranking_aggregation": aggregate}, data, 4)
        raw = model.save_raw(raw_format="json")
        if baseline is None:
            baseline = raw
        assert raw == baseline
