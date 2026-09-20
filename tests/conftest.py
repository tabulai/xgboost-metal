from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def reference(tmp_path):
    """Run compatibility checks against the pinned XGBoost CPU build."""
    def frame_payload(frame):
        if not hasattr(frame, "dtypes"):
            return frame, None
        columns, schema = [], []
        for name in frame.columns:
            column = frame[name]
            if str(column.dtype) == "category":
                values = column.cat.codes.to_numpy(dtype=np.float32)
                values[values < 0] = np.nan
                schema.append({"names": column.cat.categories.tolist(), "dtype": str(column.cat.categories.dtype)})
            else:
                values = column.to_numpy(dtype=np.float32)
                schema.append(None)
            columns.append(values)
        return np.column_stack(columns), {"columns": list(frame.columns), "categories": schema}

    def run(X, y, params, rounds=5, test=None, model=None, custom_square=False, feature_types=None,
            post_params=None, evaluation=False, continue_rounds=0, prediction_requests=None,
            param_updates=None, matrix_updates=None, quantile=False, cuts=False, test_matrix=None,
            matrix_data=False, prediction_steps=None, category_reference=False, missing=np.nan, **meta):
        destination = tmp_path / f"reference-{len(list(tmp_path.iterdir()))}"
        destination.mkdir()
        X, train_frame = frame_payload(X)
        data = {"X": X, **meta}
        if y is not None:
            data["label"] = y
        test_frame = train_frame
        if test is not None:
            data["test"], test_frame = frame_payload(test)
        np.savez(destination / "data.npz", **data)
        (destination / "params.json").write_text(json.dumps({"params": params, "rounds": rounds, "custom_square": custom_square,
                                                           "feature_types": feature_types, "train_frame": train_frame, "test_frame": test_frame, "post_params": post_params, "evaluation": evaluation, "continue_rounds": continue_rounds, "prediction_requests": prediction_requests, "param_updates": param_updates, "matrix_updates": matrix_updates, "quantile": quantile, "cuts": cuts, "test_matrix": test_matrix, "matrix_data": matrix_data, "prediction_steps": prediction_steps, "category_reference": category_reference, "missing": missing}))
        env = {**os.environ, "PYTHONPATH": str(ROOT / "ref/xgboost/python-package")}
        cmd = [sys.executable, str(ROOT / "tools/reference_runner.py"),
               "--data", str(destination / "data.npz"), "--config", str(destination / "params.json"),
               "--output", str(destination / "result")]
        if model is not None:
            cmd += ["--model", str(model)]
        completed = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=90)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        out = destination / "result"
        return np.load(out / "predictions.npz"), out
    return run
