import json

import numpy as np
import pytest
import xgboost as xgb


def compare_vector_models(model, reference_file):
    own = json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]
    expected = json.loads(reference_file.read_text())["learner"]["gradient_booster"]["model"]
    assert own["iteration_indptr"] == expected["iteration_indptr"]
    assert own["tree_info"] == expected["tree_info"]
    assert len(own["trees"]) == len(expected["trees"])
    for index, (actual, target) in enumerate(zip(own["trees"], expected["trees"])):
        assert actual["tree_param"] == target["tree_param"]
        for field in ["left_children", "right_children", "parents", "split_indices", "default_left",
                      "split_type", "categories", "categories_nodes", "categories_segments", "categories_sizes"]:
            assert actual[field] == target[field], (index, field)
        fields = ["base_weights", "split_conditions", "sum_hessian", "loss_changes"]
        if "leaf_weights" in target:
            fields.append("leaf_weights")
        else:
            assert "leaf_weights" not in actual
        for field in fields:
            np.testing.assert_array_equal(np.asarray(actual[field], np.float32), np.asarray(target[field], np.float32),
                                          err_msg=f"tree {index}: {field}")


def training_data(outputs, categorical=False):
    rng = np.random.default_rng(317)
    X = rng.integers(0, 7, (263, 5)).astype(np.float32)
    y = np.column_stack([X[:, i % 5] * (i + 1) - X[:, (i + 1) % 5] ** 2
                         for i in range(outputs)]).astype(np.float32)
    X[::11, 1] = np.nan
    X[::17, 0] = np.nan
    weights = np.linspace(0.1, 1.7, len(X), dtype=np.float32)
    types = ["c" if categorical else "q", "q", "q", "q", "q"]
    return X, y, weights, types


@pytest.mark.parametrize("outputs", [2, 5, 37])
@pytest.mark.parametrize("category_limit", [None, 2, 10])
def test_vector_regression(reference, outputs, category_limit):
    X, y, weights, types = training_data(outputs, category_limit is not None)
    params = dict(multi_strategy="multi_output_tree", max_depth=3, max_bin=16,
                  max_cat_to_onehot=category_limit or 4, seed=91)
    expected, files = reference(X, y, params, rounds=4, weight=weights, feature_types=types)
    data = xgb.DMatrix(X, label=y, weight=weights, feature_types=types, enable_categorical=True)
    model = xgb.train(params, data, 4)
    np.testing.assert_array_equal(model.predict(data, output_margin=True), expected["margin"])
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    compare_vector_models(model, files / "model.json")


@pytest.mark.parametrize("extra", [
    {"reg_lambda": 2.5, "reg_alpha": 0.35},
    {"max_delta_step": 0.2},
    {"min_child_weight": 20},
    {"monotone_constraints": "(1,-1,0,0,0)"},
    {"interaction_constraints": "[[0,1],[2,3]]"},
    {"grow_policy": "lossguide", "max_depth": 0, "max_leaves": 6},
    {"subsample": 0.65},
    {"subsample": 0.65, "sampling_method": "gradient_based"},
    {"num_parallel_tree": 3},
    {"colsample_bytree": 0.8, "colsample_bylevel": 0.8, "colsample_bynode": 0.8},
])
@pytest.mark.parametrize("categorical", [False, True])
def test_vector_training_parameters(reference, extra, categorical):
    X, y, weights, types = training_data(3, categorical)
    params = {"multi_strategy": "multi_output_tree", "max_depth": 3, "max_bin": 16,
              "max_cat_to_onehot": 2, "seed": 971, **extra}
    expected, files = reference(X, y, params, rounds=4, weight=weights, feature_types=types)
    data = xgb.DMatrix(X, label=y, weight=weights, feature_types=types, enable_categorical=True)
    model = xgb.train(params, data, 4)
    np.testing.assert_array_equal(model.predict(data, output_margin=True), expected["margin"])
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    compare_vector_models(model, files / "model.json")


@pytest.mark.parametrize("extra", [
    {"metal_histogram_subtraction": 0},
    {"max_cached_hist_node": 1},
    {"metal_fused_partition": 0},
    {"metal_partition_scan": 0, "metal_partition_aggregation": 0},
    {"metal_partition_scan": 0, "metal_partition_aggregation": 1},
    {"metal_features_per_group": 4},
    {"metal_tree_workspace": 0},
    {"metal_coalesce_encoders": 1, "metal_compute_fill": 1},
])
def test_vector_execution_paths(extra):
    X, y, weights, types = training_data(5, True)
    data = xgb.DMatrix(X, label=y, weight=weights, feature_types=types, enable_categorical=True)
    params = dict(multi_strategy="multi_output_tree", grow_policy="lossguide", max_depth=0,
                  max_leaves=11, max_bin=16, max_cat_to_onehot=2, seed=217, subsample=0.7)
    expected = xgb.train(params, data, 4)
    actual = xgb.train({**params, **extra}, data, 4)
    assert actual.save_raw(raw_format="json") == expected.save_raw(raw_format="json")
    np.testing.assert_array_equal(actual.predict(data), expected.predict(data))


@pytest.mark.parametrize("categories", [257, 513])
def test_vector_large_categorical_histograms(reference, categories):
    rng = np.random.default_rng(72)
    X = rng.integers(0, 7, (1029, 4)).astype(np.float32)
    X[:, 0] = np.arange(len(X)) % categories
    y = np.column_stack([X[:, 0] % 13 - X[:, 1], X[:, 0] % 7 + X[:, 2] ** 2]).astype(np.float32)
    X[::17, 0] = np.nan
    params = dict(multi_strategy="multi_output_tree", max_depth=3, max_bin=16, max_cat_to_onehot=2,
                  max_cat_threshold=31, max_cached_hist_node=1)
    types = ["c", "q", "q", "q"]
    expected, files = reference(X, y, params, rounds=3, feature_types=types)
    data = xgb.DMatrix(X, label=y, feature_types=types, enable_categorical=True)
    model = xgb.train(params, data, 3)
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    compare_vector_models(model, files / "model.json")


@pytest.mark.parametrize("extra", [{}, {"monotone_constraints": "(1,-1,0,0,0)"},
                                   {"reg_lambda": 0, "reg_alpha": 0.1}, {"max_delta_step": 0.25}])
def test_vector_signed_custom_gradients(reference, extra):
    X, y, _, _ = training_data(3)
    rng = np.random.default_rng(821)
    gradient = rng.integers(-16, 17, y.shape).astype(np.float32) / 8
    hessian = rng.integers(-4, 9, y.shape).astype(np.float32) / 4
    params = {"multi_strategy": "multi_output_tree", "max_depth": 4, "max_bin": 16,
              "min_child_weight": 0, "base_score": [0.2, 0.3, 0.4], **extra}
    expected, files = reference(X, y, params, rounds=1, grad=gradient, hess=hessian)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 1, obj=lambda p, d: (gradient, hessian))
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    compare_vector_models(model, files / "model.json")


@pytest.mark.parametrize("extra", [{"subsample": 0}, {"subsample": 0, "sampling_method": "gradient_based"},
                                   {"min_child_weight": 10000}, {"gamma": 1e10}])
def test_vector_root_only(reference, extra):
    X, y, weight, _ = training_data(5)
    params = {"multi_strategy": "multi_output_tree", "max_bin": 16, **extra}
    expected, files = reference(X, y, params, rounds=2, weight=weight)
    data = xgb.DMatrix(X, label=y, weight=weight)
    model = xgb.train(params, data, 2)
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    compare_vector_models(model, files / "model.json")
