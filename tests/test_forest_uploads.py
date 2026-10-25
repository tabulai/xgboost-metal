"""Incremental GPU forests must behave like freshly rebuilt model state."""
import ctypes
import json

import numpy as np
import pytest
import xgboost as xgb


def profile(model):
    output = ctypes.c_char_p()
    xgb.core._check_call(xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(output)))
    return json.loads(output.value)["forest_uploads"]


def forest(model):
    return json.loads(model.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]


def parameters(**extra):
    return dict(tree_method="hist", max_depth=3, max_bin=16, base_score=0.25,
                eta=0.2, seed=719, nthread=1, verbosity=0, **extra)


def dataset(outputs=1, categorical=False):
    rng = np.random.default_rng(438)
    X = rng.integers(-8, 9, (193, 5)).astype(np.float32) / np.float32(4)
    if categorical:
        # Several category bitmap words and many trees exercise global offsets.
        X[:, 0] = np.resize(np.array([0, 3, 31, 32, 65, 96], np.float32), len(X))
        signal = np.where(np.isin(X[:, 0], [3, 32, 96]), 5, -3)
    else:
        signal = 2 * X[:, 0]
    y = (signal + X[:, 1] * X[:, 2] - X[:, 3] ** 2).astype(np.float32)
    if outputs > 1:
        y = np.column_stack([y, -0.7 * y + X[:, 4], X[:, 1] ** 2 - y * 0.3]).astype(np.float32)
    X[::13, 0] = np.nan
    X[::17, 3] = np.nan
    types = ["c" if categorical else "q", "q", "q", "q", "q"]
    return X, y, types


def matrix(X, y=None, types=None, margin=None):
    return xgb.DMatrix(X, label=y, feature_types=types, enable_categorical=True,
                       base_margin=margin)


def exact(actual, expected):
    # Numeric equality alone would miss signed-zero or other bit changes.
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def reload_model(model):
    restored = xgb.Booster(model_file=model.save_raw())
    restored.load_config(model.save_config())
    return restored


def check_fresh(model, training, X, types, margin=None):
    """A newly loaded model cannot reuse an old forest or prediction prefix."""
    restored = reload_model(model)
    fresh = matrix(X, types=types, margin=margin)
    for options in ({"output_margin": True}, {"pred_leaf": True}):
        expected = restored.predict(fresh, **options)
        exact(model.predict(training, **options), expected)
        exact(model.predict(fresh, **options), expected)
    if model.num_boosted_rounds() > 2:
        options = dict(output_margin=True, iteration_range=(1, model.num_boosted_rounds() - 1))
        exact(model.predict(fresh, **options), restored.predict(fresh, **options))
        # A partial-range request must not poison the next full-range request.
        exact(model.predict(training, output_margin=True), restored.predict(fresh, output_margin=True))


GROWTH_CASES = [
    ("scalar", 1, False, 1),
    ("categorical", 1, True, 3),
    ("multiclass", 1, True, 2),
    ("vector", 3, True, 2),
    ("mixed", 3, True, 2),
]


@pytest.mark.parametrize("kind,outputs,categorical,parallel", GROWTH_CASES,
                         ids=[case[0] for case in GROWTH_CASES])
def test_append_growth_matches_rebuilding_every_round(reference, kind, outputs, categorical, parallel):
    X, y, types = dataset(outputs, categorical)
    params = parameters(num_parallel_tree=parallel, max_cat_to_onehot=2)
    if kind == "multiclass":
        y = np.digitize(y, np.quantile(y, [1 / 3, 2 / 3])).astype(np.float32)
        params.update(objective="multi:softprob", num_class=3)
    if kind == "vector":
        params["multi_strategy"] = "multi_output_tree"
    changes = ({5: {"multi_strategy": "multi_output_tree"},
                11: {"multi_strategy": "one_output_per_tree"}} if kind == "mixed" else {})
    data = matrix(X, y, types)
    retained, rebuilt = [xgb.Booster(params, [data]) for _ in range(2)]
    allocations = []
    for iteration in range(17):
        # reset drops GPU state while preserving the trained CPU model and RNG.
        # The control therefore never depends on an incrementally uploaded prefix.
        rebuilt.reset()
        for model in (retained, rebuilt):
            if iteration in changes:
                model.set_param(changes[iteration])
            model.update(data, iteration)
        assert retained.save_raw() == rebuilt.save_raw()
        exact(retained.predict(data, output_margin=True), rebuilt.predict(data, output_margin=True))
        check_fresh(retained, data, X, types)
        allocations.append(profile(retained)["buffer_allocations"])

    document = forest(retained)
    stats = profile(retained)
    # Work is proportional to the final forest, not the sum of all prefixes.
    assert stats["trees_uploaded"] == len(document["trees"])
    assert stats["append_uploads"] >= 16
    assert stats["bytes_relocated"] > 0
    assert any(a == b for a, b in zip(allocations, allocations[1:]))
    assert allocations[-1] > allocations[0]
    if categorical:
        assert any(tree["categories"] for tree in document["trees"])
    if kind == "mixed":
        sizes = {int(tree["tree_param"]["size_leaf_vector"]) for tree in document["trees"]}
        assert 3 in sizes and any(size < 2 for size in sizes)
    expected, _ = reference(X, y, params, rounds=17, feature_types=types, param_updates=changes)
    exact(retained.predict(data, output_margin=True), expected["margin"])
    exact(retained.predict(data, pred_leaf=True), expected["leaf"])


@pytest.mark.parametrize("format", ["json", "ubj"])
def test_load_replaces_a_warm_forest_of_the_same_or_smaller_size(tmp_path, format):
    X, y, types = dataset(3, True)
    params = parameters(multi_strategy="multi_output_tree", num_parallel_tree=2, max_cat_to_onehot=2)
    data = matrix(X, y, types)
    retained = xgb.Booster(params, [data])
    for iteration in range(5):
        retained.update(data, iteration)
    for rounds in (5, 2):
        old_prediction = retained.predict(data, output_margin=True)
        old_uploads = profile(retained)["full_uploads"]
        donor_data = matrix(X, -y + np.float32(0.75), types)
        donor = xgb.train(params, donor_data, rounds)
        path = tmp_path / f"replacement-{rounds}.{format}"
        donor.save_model(path)
        retained.load_model(path)
        retained.load_config(donor.save_config())
        assert retained.num_boosted_rounds() == rounds
        exact(retained.predict(data, output_margin=True), donor.predict(data, output_margin=True))
        assert not np.array_equal(retained.predict(data, output_margin=True), old_prediction)
        assert profile(retained)["full_uploads"] > old_uploads
        check_fresh(retained, data, X, types)
        for model in (retained, donor):
            model.update(data, rounds)
        assert retained.save_raw() == donor.save_raw()
        check_fresh(retained, data, X, types)


def test_slice_reset_and_base_changes_invalidate_uploaded_state():
    X, y, types = dataset(3, True)
    params = parameters(multi_strategy="multi_output_tree", num_parallel_tree=2, max_cat_to_onehot=2)
    data = matrix(X, y, types)
    model = xgb.Booster(params, [data])
    for iteration in range(9):
        model.update(data, iteration)
    full = model.predict(data, output_margin=True)
    sliced = model[2:8:2]
    check_fresh(sliced, data, X, types)
    control = reload_model(sliced)
    for candidate in (sliced, control):
        candidate.update(data, 3)
    assert sliced.save_raw() == control.save_raw()
    exact(model.predict(data, output_margin=True), full)

    for action in ("base_score", "reset", "config", "base_margin", "clear_margin"):
        previous = sliced.predict(data, output_margin=True)
        if action == "base_score":
            sliced.set_param("base_score", [1.25, -0.5, 0.75])
        elif action == "reset":
            sliced.reset()
        elif action == "config":
            sliced.load_config(sliced.save_config())
        elif action == "base_margin":
            data.set_base_margin(np.linspace(-0.8, 0.6, len(X) * 3, dtype=np.float32).reshape(len(X), 3))
        else:
            data.set_base_margin(np.empty(0, np.float32))
        margin = data.get_base_margin()
        margin = margin.reshape(len(X), 3) if margin.size else None
        check_fresh(sliced, data, X, types, margin)
        if action in ("base_score", "base_margin", "clear_margin"):
            assert not np.array_equal(sliced.predict(data, output_margin=True), previous)
        control = reload_model(sliced)
        iteration = sliced.num_boosted_rounds()
        for candidate in (sliced, control):
            candidate.update(data, iteration)
        assert sliced.save_raw() == control.save_raw()
        check_fresh(sliced, data, X, types, margin)


@pytest.mark.parametrize("updater", ["refresh", "prune", "refresh,prune"])
def test_update_existing_trees_rebuilds_a_warm_forest(reference, tmp_path, updater):
    X, y, types = dataset(categorical=True)
    params = parameters(num_parallel_tree=2, max_cat_to_onehot=2)
    data = matrix(X, y, types)
    model = xgb.Booster(params, [data])
    for iteration in range(7):
        model.update(data, iteration)
    check_fresh(model, data, X, types)
    initial = tmp_path / "initial.ubj"
    model.save_model(initial)
    changed_y = y * np.float32(0.4) + np.float32(1.1)
    changed = matrix(X, changed_y, types)
    updates = dict(process_type="update", updater=updater, refresh_leaf=True,
                   num_parallel_tree=2, eta=0.45, max_depth=2, gamma=0.3)
    model.set_param(updates)
    for iteration in range(4):
        model.update(changed, iteration)
        check_fresh(model, changed, X, types)
    assert model.num_boosted_rounds() == 4
    expected, _ = reference(X, changed_y, updates, model=initial, continue_rounds=4,
                            feature_types=types)
    np.testing.assert_allclose(model.predict(changed, output_margin=True), expected["margin"],
                               rtol=2e-5, atol=3e-5)
    exact(model.predict(changed, pred_leaf=True), expected["leaf"])
    # Switch back to appending after truncating/updating the original forest.
    model.set_param({**params, "process_type": "default"})
    control = reload_model(model)
    for candidate in (model, control):
        candidate.update(changed, 4)
    assert model.save_raw() == control.save_raw()
    check_fresh(model, changed, X, types)


@pytest.mark.parametrize("normalize", ["tree", "forest"])
def test_dropout_reuploads_changed_prefix_weights(reference, normalize):
    X, y, types = dataset(categorical=True)
    params = parameters(booster="dart", num_parallel_tree=2, max_cat_to_onehot=2,
                        rate_drop=1, one_drop=True, normalize_type=normalize)
    data = matrix(X, y, types)
    model = xgb.Booster(params, [data])
    changes = {7: {"rate_drop": 0, "one_drop": False}}
    old_weights = np.empty(0, np.float32)
    for iteration in range(9):
        if iteration in changes:
            model.set_param(changes[iteration])
        model.update(data, iteration)
        weights = np.asarray(forest(model)["weight_drop"], np.float32)
        if old_weights.size:
            if iteration < 7:
                assert np.any(weights[:len(old_weights)] != old_weights)
            else:
                np.testing.assert_array_equal(weights[:len(old_weights)], old_weights)
        check_fresh(model, data, X, types)
        old_weights = weights
    expected, _ = reference(X, y, params, rounds=9, feature_types=types, param_updates=changes)
    np.testing.assert_allclose(model.predict(data, output_margin=True), expected["margin"],
                               rtol=3e-5, atol=3e-5)
    exact(model.predict(data, pred_leaf=True), expected["leaf"])


def test_switching_to_linear_and_back_discards_old_forest_capacity():
    X, y, types = dataset()
    params = parameters()
    data = matrix(X, y, types)
    model = xgb.Booster(params, [data])
    for iteration in range(9):
        model.update(data, iteration)
    old = model.predict(data, output_margin=True)
    model.set_param(dict(booster="gblinear", updater="coord_descent", eta=0.3))
    for iteration in range(2):
        model.update(data, iteration)
    exact(model.predict(data), reload_model(model).predict(data))
    model.set_param({**params, "booster": "gbtree"})
    rebuilt = xgb.Booster(params, [data])
    for iteration in range(5):
        for candidate in (model, rebuilt):
            candidate.update(data, iteration)
        assert model.save_raw() == rebuilt.save_raw()
        check_fresh(model, data, X, types)
    assert model.num_boosted_rounds() == 5
    assert not np.array_equal(model.predict(data, output_margin=True), old)


def test_cache_includes_partial_round_after_failed_update():
    X = np.arange(32, dtype=np.float32).reshape(16, 2)
    y = np.column_stack((X[:, 0] + 1, X[:, 1] + 2))
    data = xgb.DMatrix(X, label=y)
    params = parameters()
    params.update(base_score=0, max_depth=1, max_bin=8)
    model = xgb.Booster(params, [data])
    # A valid mixed forest contains two scalar trees, one vector tree, then
    # two scalar trees. Refresh supports the scalar trees but rejects vectors.
    strategies = ["one_output_per_tree", "multi_output_tree", "one_output_per_tree"]
    for iteration, strategy in enumerate(strategies):
        model.set_param("multi_strategy", strategy)
        model.update(data, iteration)
    model.set_param(dict(process_type="update", updater="refresh",
                         multi_strategy="one_output_per_tree", num_parallel_tree=2))
    with pytest.raises(xgb.core.XGBoostError, match="multi-target"):
        model.update(data, 0)
    # Two scalar trees were published before the vector-tree rejection, but
    # the round was never completed. They must remain in the cached margin
    # when a subsequent successful growth operation completes that round.
    assert model.num_boosted_rounds() == 0
    assert len(model.get_dump()) == 2
    model.set_param(dict(process_type="default", updater="grow_quantile_histmaker",
                         num_parallel_tree=1))
    for iteration in range(2):
        model.update(data, iteration)
        fresh = xgb.DMatrix(X)
        exact(model.predict(data, output_margin=True), model.predict(fresh, output_margin=True))
