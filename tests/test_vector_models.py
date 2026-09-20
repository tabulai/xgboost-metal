import json
import pickle

import numpy as np
import pytest
import xgboost as xgb


@pytest.mark.parametrize("outputs", [2, 5])
@pytest.mark.parametrize("parallel", [1, 3])
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("kind", ["json", "ubj"])
def test_vector_model_exchange_and_prediction(reference, tmp_path, outputs, parallel, categorical, kind):
    rng = np.random.default_rng(783)
    X = rng.integers(0, 7, (259, 5)).astype(np.float32)
    y = np.column_stack([X[:, i % 5] * (i + 1) - X[:, (i + 1) % 5] ** 2
                         for i in range(outputs)]).astype(np.float32)
    X[::11, 1] = np.nan
    weight = np.linspace(0.1, 1.7, len(X), dtype=np.float32)
    types = ["c" if categorical else "q", "q", "q", "q", "q"]
    params = dict(multi_strategy="multi_output_tree", max_depth=3, max_bin=16,
                  num_parallel_tree=parallel, max_cat_to_onehot=2, seed=198)
    expected, files = reference(X, y, params, rounds=4, weight=weight, feature_types=types,
                                prediction_requests=[{"iteration_range": [1, 3]}])
    model = xgb.Booster(model_file=files / f"model.{kind}")
    data = xgb.DMatrix(X, feature_types=types, enable_categorical=True)
    for shared in [0, 1]:
        model.set_param("metal_predict_shared", shared)
        np.testing.assert_array_equal(model.predict(data, output_margin=True), expected["margin"])
        np.testing.assert_array_equal(model.predict(data), expected["prediction"])
        np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    for name, request in [("contributions", {"pred_contribs": True}),
                          ("interactions", {"pred_interactions": True})]:
        np.testing.assert_allclose(model.predict(data, **request), expected[name], rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(model[1:3].predict(data), expected["request_0"])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(data), expected["prediction"])
    assert set(json.loads((files / "prediction_errors.json").read_text())) == {"approximate", "approximate_interactions"}
    with pytest.raises(xgb.core.XGBoostError, match="strict_shape"):
        model.predict(data, pred_leaf=True, strict_shape=True)
    with pytest.raises(xgb.core.XGBoostError, match="Approximate"):
        model.predict(data, pred_contribs=True, approx_contribs=True)
    own = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]
    source = json.loads((files / "model.json").read_text())["learner"]["gradient_booster"]["model"]
    assert own["iteration_indptr"] == source["iteration_indptr"]
    for a, b in zip(own["trees"], source["trees"]):
        assert a["tree_param"] == b["tree_param"]
        for key in ["base_weights", "leaf_weights", "split_conditions", "sum_hessian", "loss_changes"]:
            np.testing.assert_array_equal(np.asarray(a[key], np.float32), np.asarray(b[key], np.float32))
        for key in ["left_children", "right_children", "parents", "split_indices", "default_left",
                    "split_type", "categories", "categories_nodes", "categories_segments", "categories_sizes"]:
            assert a[key] == b[key], key
    own_dump = [json.loads(tree) for tree in model.get_dump(dump_format="json", with_stats=True)]
    expected_dump = [json.loads(tree) for tree in json.loads((files / "dump.json").read_text())]
    def compare_dump(a, b):
        assert a.keys() == b.keys()
        for key in a:
            if key == "children":
                for left, right in zip(a[key], b[key]):
                    compare_dump(left, right)
            elif isinstance(a[key], str):
                assert a[key] == b[key]
            else:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-7, atol=1e-7)
    for a, b in zip(own_dump, expected_dump):
        compare_dump(a, b)
    destination = tmp_path / f"metal-vector.{kind}"
    model.save_model(destination)
    returned, _ = reference(X, y, params, model=destination, feature_types=types)
    np.testing.assert_array_equal(returned["margin"], expected["margin"])
