"""Verify that the Metal library exports every declared XGBoost C ABI symbol."""
from pathlib import Path
import argparse
import ctypes
import json
import re

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--library", type=Path, default=root / "lib/libxgboost.dylib")
args = parser.parse_args()
header = (root / "include/xgboost/c_api.h").read_text()
symbols = sorted(set(re.findall(r"XGB_DLL\s+[^;]*?\b(XG\w+)\s*\(", header)))
library = ctypes.CDLL(str(args.library))
missing = [symbol for symbol in symbols if not hasattr(library, symbol)]
python_symbols = set()
for path in (root / "python-package/xgboost").rglob("*.py"):
    python_symbols.update(re.findall(r"_LIB\.(XG\w+)", path.read_text()))
python_missing = sorted(symbol for symbol in python_symbols if not hasattr(library, symbol))
print(json.dumps({"declared": len(symbols), "exported": len(symbols) - len(missing),
                  "missing": missing, "python_referenced": len(python_symbols), "python_missing": python_missing,
                  "note": "Symbol presence is not behavioral compatibility; advanced operations can return explicit unsupported errors."}, indent=2))
raise SystemExit(bool(missing or python_missing))
