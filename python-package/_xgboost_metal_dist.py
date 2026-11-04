"""Answer importlib.metadata lookups for the ``xgboost`` distribution.

xgboost-metal installs the ``xgboost`` import package under the distribution
name ``xgboost-metal``, so ``importlib.metadata.version("xgboost")`` would
otherwise raise ``PackageNotFoundError``. The finder registered by
:func:`install` answers lookups of that name with the upstream XGBoost version
the package implements. It reads ``xgboost/VERSION`` and reports it the way an
upstream build records it: ``3.5.0-dev`` becomes ``3.5.0.dev0``.

Only lookups by name are answered. Installers list packages by enumerating
distributions, so pip still sees xgboost-metal alone and ``pip freeze`` stays
installable. When the importable ``xgboost`` package is not this adapter, the
finder steps aside and that package's own metadata answers.

``xgboost_metal.pth`` calls :func:`install` at interpreter startup, and
``import xgboost`` calls it for interpreters that skip ``.pth`` files. Nothing
beyond ``sys`` is imported until a lookup for ``xgboost`` happens.
"""

import sys

_NAME = "xgboost"
_MARKER = "_metal_array.py"  # Present only in the xgboost-metal adapter.
_distribution_type = None


def _upstream_version(version: str) -> str:
    """Normalize an XGBoost VERSION string as PEP 440 build tools do."""
    if version.endswith("-dev"):
        return version[: -len("-dev")] + ".dev0"
    return version


def _make_distribution(root, version: str):
    global _distribution_type
    if _distribution_type is None:
        import importlib.metadata

        class XGBoostDistribution(importlib.metadata.Distribution):
            """Core metadata for the xgboost package installed by xgboost-metal."""

            def __init__(self, root, version: str) -> None:
                self._root = root
                self._metadata = (
                    "Metadata-Version: 2.1\n"
                    f"Name: {_NAME}\n"
                    f"Version: {version}\n"
                    "Summary: XGBoost Python package provided by xgboost-metal\n"
                )

            def read_text(self, filename):
                return self._metadata if filename == "METADATA" else None

            def locate_file(self, path):
                return self._root / path

        _distribution_type = XGBoostDistribution
    return _distribution_type(root, version)


def _find(path):
    """Return metadata if ``path`` would import the xgboost-metal adapter."""
    from importlib.machinery import PathFinder
    from pathlib import Path

    try:
        spec = PathFinder.find_spec(_NAME, list(path))
        locations = list(spec.submodule_search_locations or ()) if spec else []
        if not locations:
            return None
        package = Path(locations[0])
        if not (package / _MARKER).is_file():
            return None
        version = (package / "VERSION").read_text(encoding="ascii").strip()
    except (ImportError, OSError, ValueError):
        return None
    return _make_distribution(package.parent, _upstream_version(version))


class _Finder:
    """``sys.meta_path`` entry that imports nothing and finds one distribution."""

    _xgboost_metal_dist = True

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        return None

    @staticmethod
    def find_distributions(context=None):
        name = getattr(context, "name", None)
        if not isinstance(name, str) or name.lower() != _NAME:
            return iter(())
        distribution = _find(getattr(context, "path", sys.path))
        return iter(() if distribution is None else (distribution,))


def install() -> None:
    """Register the finder once, ahead of the standard path-based finder."""
    meta_path = sys.meta_path
    if any(getattr(finder, "_xgboost_metal_dist", False) for finder in meta_path):
        return
    position = next(
        (i for i, finder in enumerate(meta_path)
         if getattr(finder, "__name__", None) == "PathFinder"),
        len(meta_path),
    )
    meta_path.insert(position, _Finder())
