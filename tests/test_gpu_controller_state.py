"""Finite regression coverage for the intermittent fused-controller failure.

Use the frozen longdeep input and ordinary public xgb.train calls, including
their final reset. Each fit owns a fresh DMatrix and Booster, while its 1,000
rounds reuse the training workspace. Four successive fused fits exercise the
process-level lifecycle seen in the failures. Passing this finite test does
not guarantee the absence of an intermittent defect. There are no timers,
training callbacks, profiler calls, or retries.
"""

import gc
import hashlib
import json

import numpy as np
import pytest
import xgboost as xgb


def _array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


@pytest.fixture(scope="module")
def longdeep_data():
    # Preserve the generator and evaluation order in the frozen probe.py.
    rng = np.random.default_rng(42)
    features = rng.standard_normal((100000, 32), dtype=np.float32)
    noise = rng.standard_normal(100000, dtype=np.float32) * np.float32(.1)
    labels = (2 * features[:, 0] + features[:, 1] ** 2 + noise).astype(np.float32)
    # Frozen input hashes, independently recorded before the controller change.
    assert _array_sha256(features) == "784a45aaad5e96f3990896e6c1d33433f51aba41289f62ba1817e10871120fc1"
    assert _array_sha256(labels) == "2211ecbd6e1b8ebc8406f19e9ca1470314e1a4085964e7ca1397ec64790e6916"
    return features[:80000], labels[:80000], features[80000:]


def _fit(data, *, fusion, coalesce):
    # Match the benchmark's collection between fresh fits, outside xgb.train.
    gc.collect()
    features, labels, query = data
    training = xgb.DMatrix(features, label=labels, nthread=16)
    params = dict(
        tree_method="hist", device="metal", objective="reg:squarederror",
        max_depth=6, max_bin=256, eta=.1, seed=42, nthread=16,
        metal_gpu_tree=1, metal_split_threads=256,
        metal_tree_control_fusion=fusion, metal_coalesce_encoders=coalesce,
    )
    model = xgb.train(params, training, num_boost_round=1000, verbose_eval=False)
    assert model.num_boosted_rounds() == 1000
    evaluation = xgb.DMatrix(query, nthread=16)
    predictions = model.predict(evaluation)
    margins = model.predict(evaluation, output_margin=True)
    assert predictions.shape == margins.shape == (20000,)
    assert np.isfinite(predictions).all() and np.isfinite(margins).all()
    # Normalize JSON presentation only; retain every serialized model field.
    document = json.loads(model.save_raw(raw_format="json"))
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    # Return only host outputs so each following fit has fresh training objects.
    return predictions, margins, canonical


@pytest.mark.parametrize("coalesce", [0, 1], ids=["separate-encoders", "coalesced-encoders"])
def test_repeated_fused_longdeep_matches_unfused(longdeep_data, coalesce):
    assert xgb.build_info().get("USE_METAL") is True, "Run with the intended Metal package."
    expected = _fit(longdeep_data, fusion=0, coalesce=coalesce)
    for repeat in range(4):
        actual = _fit(longdeep_data, fusion=1, coalesce=coalesce)
        context = f"coalescing={coalesce}, fused repeat={repeat}"
        for name, wanted, observed in zip(("predictions", "margins"), expected[:2], actual[:2]):
            assert observed.dtype == wanted.dtype, f"{name} dtype: {context}"
            np.testing.assert_array_equal(observed, wanted, err_msg=f"{name}: {context}")
            assert observed.tobytes(order="C") == wanted.tobytes(order="C"), f"{name} bytes: {context}"
        assert actual[2] == expected[2], f"canonical model: {context}"
