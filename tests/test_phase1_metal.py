import math
import subprocess
import sys
import numpy as np
import pytest
import xgboost as xgb


def _ref_binary_auc(score: np.ndarray, label: np.ndarray, weight: np.ndarray, pr: bool) -> float:
    order = np.argsort(-score.astype(np.float64), kind="stable")
    score, label, weight = score[order], label[order], weight[order]
    positive = (label * weight).astype(np.float64)
    negative = ((np.float32(1) - label) * weight).astype(np.float64)
    total_pos = positive.sum()
    total_neg = negative.sum()
    if not (total_pos > 0 and total_neg > 0):
        return float("nan")

    def seg_area(fp_prev: float, fp: float, tp_prev: float, tp: float) -> float:
        if not pr:
            return abs(fp_prev - fp) * (tp_prev + tp) * 0.5
        if tp == tp_prev:
            return 0.0
        h = (fp - fp_prev) / (tp - tp_prev)
        a, b = h + 1.0, (fp_prev - h * tp_prev) / total_pos
        now, before = tp / total_pos, tp_prev / total_pos
        if b == 0.0:
            return (now - before) / a
        return (now - before - b / a * (math.log(a * now + b) - math.log(a * before + b))) / a

    tp = fp = tp_prev = fp_prev = area = 0.0
    for i in range(len(score)):
        if i and score[i] != score[i - 1]:
            area += seg_area(fp_prev, fp, tp_prev, tp)
            tp_prev, fp_prev = tp, fp
        tp += positive[i]
        fp += negative[i]
    area += seg_area(fp_prev, fp, tp_prev, tp)
    return area if pr else area / (tp * fp)


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
@pytest.mark.parametrize(
    "case",
    [
        "unique_multi_tile",
        "cross_tile_and_chunk_ties",
        "all_equal_scores",
        "signed_zeros_and_subnormals",
        "subnormal_weights",
    ],
)
def test_radix_auc_edge_cases(metric: str, case: str) -> None:
    rng = np.random.default_rng(1234)
    if case == "unique_multi_tile":
        n = 6500
        margin = rng.normal(size=n).astype(np.float32)
        label = rng.integers(0, 2, size=n).astype(np.float32)
        weight = rng.uniform(0.05, 5.0, size=n).astype(np.float32)
    elif case == "cross_tile_and_chunk_ties":
        # Span > 256 tiles (> 524,288 rows) with ties crossing both 2048-row tile
        # boundaries and the 256-tile (524,288-row) scan chunk boundary.
        n = 530_000
        margin = np.round(rng.uniform(-2, 2, size=n) * 3.0).astype(np.float32)
        margin[520_000:526_000] = 0.0
        label = rng.integers(0, 2, size=n).astype(np.float32)
        weight = rng.uniform(0.2, 2.0, size=n).astype(np.float32)
    elif case == "all_equal_scores":
        n = 5000
        margin = np.full(n, 0.75, dtype=np.float32)
        label = rng.integers(0, 2, size=n).astype(np.float32)
        weight = rng.uniform(0.1, 3.0, size=n).astype(np.float32)
    elif case == "signed_zeros_and_subnormals":
        margin = np.array(
            [-0.0, 0.0, np.ldexp(np.float32(1.0), -149), np.ldexp(np.float32(1.0), -140),
             -np.ldexp(np.float32(1.0), -149), -np.ldexp(np.float32(1.0), -140), 1e-6, -1e-6],
            dtype=np.float32,
        )
        label = np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=np.float32)
        weight = np.ones_like(label)
    else:
        n = 512
        margin = rng.normal(size=n).astype(np.float32)
        label = rng.integers(0, 2, size=n).astype(np.float32)
        weight = np.full(n, np.ldexp(np.float32(1.0), -145), dtype=np.float32)

    X = np.zeros((len(label), 1), dtype=np.float32)
    dm = xgb.DMatrix(X, label=label, weight=weight, base_margin=margin)
    bst = xgb.train(
        {"objective": "reg:squarederror", "eval_metric": metric, "metal_auc_radix": 1},
        dm,
        num_boost_round=0,
    )
    actual = float(bst.eval(dm).split(":")[1])
    expected = _ref_binary_auc(margin, label, weight, pr=(metric == "aucpr"))
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-12)

    if case not in ("signed_zeros_and_subnormals", "subnormal_weights", "cross_tile_and_chunk_ties"):
        bst_legacy = xgb.train(
            {"objective": "reg:squarederror", "eval_metric": metric, "metal_auc_radix": 0},
            dm,
            num_boost_round=0,
        )
        legacy = float(bst_legacy.eval(dm).split(":")[1])
        assert actual == pytest.approx(legacy, rel=2e-6, abs=1e-7)


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
def test_radix_multiclass_auc_parity(metric: str) -> None:
    rng = np.random.default_rng(77)
    X = rng.normal(size=(4096, 6)).astype(np.float32)
    y = rng.integers(0, 4, size=len(X)).astype(np.float32)
    w = rng.uniform(0.1, 3.0, size=len(X)).astype(np.float32)
    dm = xgb.DMatrix(X, label=y, weight=w)
    m1 = xgb.train(
        {"objective": "multi:softprob", "num_class": 4, "max_depth": 4, "eval_metric": metric, "metal_auc_radix": 1},
        dm,
        num_boost_round=4,
    )
    m0 = xgb.train(
        {"objective": "multi:softprob", "num_class": 4, "max_depth": 4, "eval_metric": metric, "metal_auc_radix": 0},
        dm,
        num_boost_round=4,
    )
    v1 = float(m1.eval(dm).split(":")[1])
    v0 = float(m0.eval(dm).split(":")[1])
    assert v1 == pytest.approx(v0, rel=1e-6, abs=1e-7)


def test_gpu_dmatrix_slice_and_bin_slicing() -> None:
    rng = np.random.default_rng(99)
    X = rng.normal(size=(1024, 8)).astype(np.float32)
    X[rng.random(X.shape) < 0.15] = np.nan
    X[0, 0] = np.ldexp(np.float32(1.0), -148)  # subnormal preserved bit-for-bit
    y = rng.integers(0, 2, size=1024).astype(np.float32)
    w = rng.uniform(0.5, 2.0, size=1024).astype(np.float32)

    parent = xgb.DMatrix(X, label=y, weight=w)
    idx = np.array([0, 5, 17, 511, 512, 1023, 5], dtype=np.int32)
    sliced = parent.slice(idx)
    fresh = xgb.DMatrix(X[idx], label=y[idx], weight=w[idx])

    np.testing.assert_array_equal(sliced.get_label(), fresh.get_label())
    np.testing.assert_array_equal(sliced.get_weight(), fresh.get_weight())
    np.testing.assert_array_equal(sliced.get_data().toarray(), fresh.get_data().toarray())

    # Quantize parent first (via training), then evaluate/train with sliced as validation
    # or train directly on sliced, verifying identical trees and predictions.
    bst = xgb.train(
        {"objective": "binary:logistic", "max_depth": 4, "eval_metric": "auc"},
        parent,
        num_boost_round=5,
        evals=[(parent, "train"), (sliced, "val")],
    )
    np.testing.assert_array_equal(bst.predict(sliced), bst.predict(fresh))

    with pytest.raises(xgb.core.XGBoostError, match="out of bounds"):
        parent.slice([0, 1024])
    with pytest.raises(xgb.core.XGBoostError, match="out of bounds"):
        parent.slice([-1])


@pytest.mark.parametrize("booster", ["gbtree", "gblinear"])
@pytest.mark.parametrize(
    "objective,n_classes",
    [
        ("binary:logistic", 2),
        ("binary:logitraw", 2),
        ("binary:hinge", 2),
        ("multi:softprob", 4),
        ("multi:softmax", 4),
    ],
)
def test_metal_predict_proba_all_objectives(booster: str, objective: str, n_classes: int) -> None:
    if booster == "gblinear" and objective == "binary:hinge":
        pytest.skip("gblinear does not support binary:hinge")
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(256, 6)).astype(np.float32)
    y = rng.integers(0, n_classes, size=256)

    clf = xgb.XGBClassifier(
        n_estimators=6,
        max_depth=3,
        booster=booster,
        objective=objective,
        random_state=7,
    )
    clf.fit(X, y)

    proba_dense = clf.predict_proba(X)
    assert proba_dense.shape == (256, n_classes)
    assert proba_dense.dtype == np.float32

    # Verify against raw margins computed by Booster.predict(output_margin=True).
    raw = clf.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    if n_classes == 2:
        if objective in ("binary:logistic", "binary:logitraw"):
            p1 = 1.0 / (1.0 + np.exp(-raw.astype(np.float64)))
        else:
            p1 = (raw > 0).astype(np.float64)
        expected = np.column_stack([1.0 - p1, p1]).astype(np.float32)
    else:
        shifted = raw.astype(np.float64) - raw.astype(np.float64).max(axis=1, keepdims=True)
        ex = np.exp(shifted)
        expected = (ex / ex.sum(axis=1, keepdims=True)).astype(np.float32)
    np.testing.assert_allclose(proba_dense, expected, rtol=1e-5, atol=1e-6)

    # Verify DMatrix fallback path (via CSCMatrix) matches inplace_predict path.
    X_sparse = xgb.CSCMatrix(np.where(np.abs(X) > 0.5, X, 0.0))
    X_csr = X_sparse.tocsr()
    np.testing.assert_allclose(clf.predict_proba(X_sparse), clf.predict_proba(X_csr), rtol=1e-6, atol=1e-6)

    # Verify 0-row input shape.
    empty_proba = clf.predict_proba(np.empty((0, 6), dtype=np.float32))
    assert empty_proba.shape == (0, n_classes)


@pytest.mark.parametrize("objective", ["binary:logistic", "binary:logitraw"])
def test_metal_predict_proba_multilabel(objective: str) -> None:
    rng = np.random.default_rng(55)
    X = rng.normal(size=(128, 5)).astype(np.float32)
    Y = rng.integers(0, 2, size=(128, 3)).astype(np.float32)
    clf = xgb.XGBClassifier(n_estimators=5, max_depth=3, objective=objective)
    clf.fit(X, Y)
    proba = clf.predict_proba(X)
    assert proba.shape == (128, 3)
    raw = clf.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    expected = (1.0 / (1.0 + np.exp(-raw.astype(np.float64)))).astype(np.float32)
    np.testing.assert_allclose(proba, expected, rtol=1e-5, atol=1e-6)


def test_zero_scipy_subprocess_isolation() -> None:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = "python-package"

    # 1. Pure SciPy-free (and sklearn-free) subprocess: poison scipy before importing anything.
    core_code = """
import sys
for mod in ["scipy", "scipy.sparse", "scipy.special"]:
    sys.modules[mod] = None

import numpy as np
import xgboost as xgb

rng = np.random.default_rng(11)
X_dense = rng.normal(size=(200, 8)).astype(np.float32)
X_dense[np.abs(X_dense) < 0.7] = 0.0
y_bin = rng.integers(0, 2, size=200).astype(np.float32)

csr = xgb.CSRMatrix(X_dense)
csc = xgb.CSCMatrix(X_dense)
coo = xgb.COOMatrix(X_dense)
np.testing.assert_allclose(csr.toarray(), X_dense)
np.testing.assert_allclose(csc.toarray(), X_dense)
np.testing.assert_allclose(coo.toarray(), X_dense)

dm_csr = xgb.DMatrix(csr, label=y_bin)
dm_csc = xgb.DMatrix(csc, label=y_bin)
dm_coo = xgb.DMatrix(coo, label=y_bin)
exported = dm_csr.get_data()
assert isinstance(exported, xgb.CSRMatrix)

bst = xgb.train({"objective": "binary:logistic", "eval_metric": ["auc", "aucpr"], "max_depth": 3}, dm_csr, 5)
p_csr = bst.inplace_predict(csr)
p_dm = bst.predict(dm_csc)
np.testing.assert_allclose(p_csr, p_dm, rtol=1e-6, atol=1e-6)

proba_csr = bst.inplace_predict(csr, predict_type="proba")
proba_dm = bst.predict(dm_csc, _metal_proba=True)
assert proba_csr.shape == (200, 2)
np.testing.assert_allclose(proba_csr, proba_dm, rtol=1e-6, atol=1e-6)
np.testing.assert_allclose(proba_csr.sum(axis=1), 1.0, atol=1e-6)

cv_res = xgb.cv({"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 2}, dm_coo, num_boost_round=3, nfold=3)
assert len(cv_res) == 3

assert sys.modules["scipy"] is None
print("ZERO_SCIPY_CORE_OK")
"""
    proc = subprocess.run([sys.executable, "-c", core_code], env=env, capture_output=True, text=True, check=True)
    assert "ZERO_SCIPY_CORE_OK" in proc.stdout

    # 2. Verify XGBClassifier never calls scipy.special or scipy.sparse during fit/predict_proba.
    sklearn_code = """
import scipy.special
import scipy.sparse
import numpy as np
import xgboost as xgb

def _forbidden(*args, **kwargs):
    raise RuntimeError("scipy routine must not be called by xgboost!")

scipy.special.expit = _forbidden
scipy.special.softmax = _forbidden

rng = np.random.default_rng(19)
X_dense = rng.normal(size=(200, 8)).astype(np.float32)
X_dense[np.abs(X_dense) < 0.7] = 0.0
csr = xgb.CSRMatrix(X_dense)
y_bin = rng.integers(0, 2, size=200).astype(np.float32)
y_multi = rng.integers(0, 3, size=200)

for obj, target in [("binary:logistic", y_bin), ("binary:logitraw", y_bin), ("binary:hinge", y_bin), ("multi:softmax", y_multi), ("multi:softprob", y_multi)]:
    clf = xgb.XGBClassifier(n_estimators=4, max_depth=3, objective=obj)
    clf.fit(csr, target)
    pr = clf.predict_proba(csr)
    assert pr.shape[0] == 200
    if obj != "binary:hinge":
        np.testing.assert_allclose(pr.sum(axis=1), 1.0, atol=1e-5)

print("ZERO_SCIPY_SKLEARN_OK")
"""
    proc2 = subprocess.run([sys.executable, "-c", sklearn_code], env=env, capture_output=True, text=True, check=True)
    assert "ZERO_SCIPY_SKLEARN_OK" in proc2.stdout
