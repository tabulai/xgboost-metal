import numpy as np
import pytest
import xgboost as xgb

@pytest.mark.parametrize("multiclass", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_shap_against_reference(reference, multiclass, missing):
    rng = np.random.default_rng(41)
    X = rng.normal(size=(101, 4)).astype(np.float32)
    y = (3 * X[:, 0] - 2 * X[:, 1] ** 2 + X[:, 2]).astype(np.float32)
    if missing:
        X[::3, 0] = np.nan
        X[::7, 2] = np.nan
    params = dict(max_depth=4, max_bin=256, base_score=0.0)
    if multiclass:
        y = np.digitize(y, np.quantile(y, [1/3, 2/3])).astype(np.float32)
        params.update(objective="multi:softprob", num_class=3)
    expected, files = reference(X, y, params, rounds=3)
    model = xgb.Booster(model_file=files / "model.json")
    data = xgb.DMatrix(X)
    kwargs = [
        ("contributions", {"pred_contribs": True}),
        ("interactions", {"pred_interactions": True}),
        ("approximate", {"pred_contribs": True, "approx_contribs": True}),
        ("approximate_interactions", {"pred_interactions": True, "approx_contribs": True}),
    ]
    for field, options in kwargs:
        actual = model.predict(data, **options)
        np.testing.assert_allclose(actual, expected[field], atol=3e-5, rtol=3e-5)
    contributions = model.predict(data, pred_contribs=True)
    interactions = model.predict(data, pred_interactions=True)
    np.testing.assert_allclose(contributions.sum(axis=-1), model.predict(data, output_margin=True), atol=3e-5)
    np.testing.assert_allclose(interactions.sum(axis=-1), contributions, atol=3e-5)
    np.testing.assert_allclose(interactions, np.swapaxes(interactions, -1, -2), atol=3e-5)

def test_prediction_shapes_and_ranges():
    rng = np.random.default_rng(61)
    X = rng.normal(size=(93, 4)).astype(np.float32)
    y = rng.integers(0, 3, len(X))
    data = xgb.DMatrix(X, label=y)
    model = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 3,
                      "num_parallel_tree": 2}, data, 4)
    assert model.predict(data, strict_shape=True).shape == (93, 3)
    assert model.predict(data, pred_leaf=True, strict_shape=True).shape == (93, 4, 3, 2)
    assert model.predict(data, pred_contribs=True, strict_shape=True).shape == (93, 3, 5)
    assert model.predict(data, pred_interactions=True, strict_shape=True).shape == (93, 3, 5, 5)
    np.testing.assert_allclose(model.predict(data, iteration_range=(1, 3)), model[1:3].predict(data))
    np.testing.assert_allclose(model.predict(data, pred_contribs=True, iteration_range=(0, 3)),
                               model[:3].predict(data, pred_contribs=True))
    np.testing.assert_allclose(model[1:3].predict(data, pred_contribs=True).sum(axis=-1),
                               model[1:3].predict(data, output_margin=True), atol=3e-5)
    for options in [dict(pred_leaf=True), dict(pred_contribs=True), dict(pred_interactions=True),
                    dict(pred_contribs=True, approx_contribs=True), dict(pred_interactions=True, approx_contribs=True)]:
        with pytest.raises(xgb.core.XGBoostError, match="iteration end"):
            model.predict(data, iteration_range=(1, 3), **options)
