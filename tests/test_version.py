"""xgboost reports the upstream XGBoost version that this package implements."""
import importlib.metadata
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

import _xgboost_metal_dist
import xgboost as xgb
from xgboost import _c_api

PACKAGE_DIR = Path(__file__).resolve().parents[1] / 'python-package'


def test_version_matches_upstream_baseline():
    # Upstream python-package/xgboost/VERSION at the baseline commit in
    # docs/compatibility.md. The import check requires the native library to agree.
    assert xgb.__version__ == '3.5.0-dev'
    assert _c_api._lib_version(_c_api._LIB) == (3, 5, 0)


def test_metadata_reports_upstream_version():
    expected = str(Version(xgb.__version__))  # The form upstream builds record.
    assert expected == '3.5.0.dev0'
    assert importlib.metadata.version('xgboost') == expected
    assert importlib.metadata.version('XGBoost') == expected
    distribution = importlib.metadata.distribution('xgboost')
    assert distribution.metadata['Name'] == 'xgboost'
    assert Path(distribution.locate_file('xgboost/VERSION')).resolve() == PACKAGE_DIR / 'xgboost/VERSION'
    assert distribution.files is None and distribution.requires is None


def test_metadata_is_invisible_to_enumeration():
    # Installers such as pip list packages by enumerating distributions.
    context = importlib.metadata.DistributionFinder.Context()
    assert list(_xgboost_metal_dist._Finder().find_distributions(context)) == []
    assert not any(type(d).__module__ == '_xgboost_metal_dist' for d in importlib.metadata.distributions())


def test_metadata_follows_search_path(tmp_path):
    def versions(*path):
        return [d.version for d in importlib.metadata.distributions(name='xgboost', path=[str(p) for p in path])]

    other = tmp_path / 'other'
    (other / 'xgboost').mkdir(parents=True)
    (other / 'xgboost/__init__.py').write_text('')
    (other / 'xgboost/VERSION').write_text('9.9.9\n')
    (other / 'xgboost-9.9.9.dist-info').mkdir()
    (other / 'xgboost-9.9.9.dist-info/METADATA').write_text('Metadata-Version: 2.1\nName: xgboost\nVersion: 9.9.9\n')
    # Another xgboost package answers with its own metadata.
    assert versions(other) == ['9.9.9']
    # The adapter answers first when it is the package that would be imported.
    assert versions(PACKAGE_DIR, other)[0] == '3.5.0.dev0'
    assert versions(tmp_path) == []


def test_pth_file_answers_before_import():
    code = ('import importlib.metadata, site, sys; site.addsitedir(sys.argv[1]); '
            'print(importlib.metadata.version("xgboost"), "xgboost" in sys.modules)')
    result = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(PACKAGE_DIR)],
                            capture_output=True, text=True, check=True)
    assert result.stdout.split() == ['3.5.0.dev0', 'False']
