import json

import numpy as np
import pytest
import xgboost as xgb
from xgboost.objective import TreeObjective

from test_vector_training import compare_vector_models, training_data


class ProjectedSquare(TreeObjective):
    def __init__(self, projection):
        self.projection = projection

    def __call__(self, iteration, prediction, data):
        prediction = prediction.reshape(data.num_row(), -1)
        return prediction - data.get_label().reshape(prediction.shape), np.ones_like(prediction)

    def split_grad(self, iteration, gradient, hessian):
        return gradient @ self.projection, hessian @ np.square(self.projection)


@pytest.mark.parametrize("split_outputs", [1, 2, 5])
@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("extra", [{}, {"subsample": 0.7},
                                   {"subsample": 0.7, "sampling_method": "gradient_based", "num_parallel_tree": 3}])
def test_vector_projected_gradients(reference, split_outputs, categorical, extra):
    X, y, _, types = training_data(5, categorical)
    rng = np.random.default_rng(49)
    projection = rng.integers(-3, 4, (5, split_outputs)).astype(np.float32) / 4
    params = {"multi_strategy": "multi_output_tree", "max_depth": 3, "max_bin": 16,
              "max_cat_to_onehot": 2, "seed": 189, "base_score": [0.5] * 5, **extra}
    expected, files = reference(X, y, params, rounds=4, projection=projection, feature_types=types)
    data = xgb.DMatrix(X, label=y, feature_types=types, enable_categorical=True)
    model = xgb.train(params, data, 4, obj=ProjectedSquare(projection))
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    np.testing.assert_array_equal(model.predict(data, pred_leaf=True), expected["leaf"])
    compare_vector_models(model, files / "model.json")
    imported = xgb.Booster(model_file=files / "model.ubj")
    np.testing.assert_array_equal(imported.predict(data), expected["prediction"])
    config = json.loads(model.save_config())["learner"]
    source = json.loads((files / "config.json").read_text())["learner"]
    assert config["generic_param"]["rng_state"] == source["generic_param"]["rng_state"]


@pytest.mark.parametrize("extra", [{}, {"subsample": 0}, {"subsample": 0, "sampling_method": "gradient_based"},
                                   {"subsample": 0.7, "sampling_method": "gradient_based"},
                                   {"reg_alpha": 0.17, "reg_lambda": 0.4, "max_delta_step": 0.35}])
def test_vector_zero_split_gradients(reference, extra):
    X, y, _, _ = training_data(3)
    projection = np.zeros((3, 1), np.float32)
    params = {"multi_strategy": "multi_output_tree", "max_depth": 3, "max_bin": 16, **extra}
    expected, files = reference(X, y, params, rounds=3, projection=projection)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 3, obj=ProjectedSquare(projection))
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    compare_vector_models(model, files / "model.json")


def test_reduced_gradient_with_one_output(reference):
    X, y, _, _ = training_data(1)
    projection = np.full((1, 1), 0.5, np.float32)
    params = dict(multi_strategy="multi_output_tree", max_depth=3, max_bin=16)
    expected, files = reference(X, y, params, rounds=3, projection=projection)
    data = xgb.DMatrix(X, label=y)
    model = xgb.train(params, data, 3, obj=ProjectedSquare(projection))
    np.testing.assert_array_equal(model.predict(data), expected["prediction"])
    compare_vector_models(model, files / "model.json")


@pytest.mark.parametrize("params,shape,message", [
    ({}, (3, 2), "Reduced gradient must"),
    ({"multi_strategy": "multi_output_tree", "monotone_constraints": "(1,0,0,0,0)"}, (3, 2), "Monotonic constraints"),
    ({"multi_strategy": "multi_output_tree"}, (3, 4), "split gradient must"),
    ({"multi_strategy": "multi_output_tree", "booster": "gblinear"}, (3, 2), "current booster"),
])
def test_reduced_gradient_parameter_errors(params, shape, message):
    X, y, _, _ = training_data(3)
    data = xgb.DMatrix(X, label=y)
    objective = ProjectedSquare(np.ones(shape, np.float32))
    with pytest.raises(xgb.core.XGBoostError, match=message):
        xgb.train(params, data, 1, obj=objective)
