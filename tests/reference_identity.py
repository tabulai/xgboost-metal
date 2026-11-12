"""Validate the source and library used for CPU binary compatibility tests."""
from pathlib import Path
import subprocess

PINNED_COMMIT = "6d2dbe19a099aa046178be8cfc303bfec1a623a6"


def check_reference(root, xgb):
    reference = (root / "ref/xgboost").resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == PINNED_COMMIT, f"Expected CPU reference {PINNED_COMMIT}, got {commit}"
    library = Path(xgb.build_info()["libxgboost"]).resolve()
    assert library.is_relative_to(reference), f"CPU reference library is outside {reference}: {library}"
    assert xgb.__version__ == "3.5.0-dev", xgb.__version__
