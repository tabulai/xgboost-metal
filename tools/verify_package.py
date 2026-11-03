"""Verify built package payloads without importing a development-tree learner."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tarfile
import zipfile


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--installed-library", type=Path, required=True)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    native = hashlib.sha256()
    for path in sorted([*root.glob("src/*"), *root.glob("include/metalgb/*"),
                        *root.glob("kernels/*.metal*"), root / "CMakeLists.txt"]):
        if path.is_file():
            native.update(str(path.relative_to(root)).encode())
            native.update(path.read_bytes())
    with zipfile.ZipFile(args.wheel) as wheel:
        library = wheel.read("xgboost/lib/libxgboost.dylib")
        assert library == args.installed_library.read_bytes(), "Installed library differs from wheel"
        expected_python = {}
        for path in (root / "python-package/xgboost").rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and "lib" not in path.parts:
                relative = str(path.relative_to(root / "python-package"))
                expected_python[relative] = path.read_bytes()
                assert wheel.read(relative) == expected_python[relative], relative
        actual_python = {name for name in wheel.namelist() if name.startswith("xgboost/") and not name.startswith("xgboost/lib/") and not name.endswith("/")}
        assert actual_python == expected_python.keys(), actual_python.symmetric_difference(expected_python)
        metadata = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        assert (root / "README.md").read_bytes().strip() in wheel.read(metadata), "README metadata differs"
        for path in ("LICENSE", "NOTICE", "THIRD_PARTY.md", "third_party/nlohmann/LICENSE.MIT", "third_party/llvm/LICENSE.TXT"):
            name = next(name for name in wheel.namelist() if name.endswith(".dist-info/licenses/" + path))
            assert wheel.read(name) == (root / path).read_bytes(), path
    report = {
        "wheel_bytes": args.wheel.stat().st_size,
        "wheel_sha256": sha256(args.wheel.read_bytes()),
        "library_sha256": sha256(library), "native_source_sha256": native.hexdigest(),
        "native_payload_and_notices_byte_exact": True, "python_adapter_byte_exact": True,
        "python_adapter_files": len(expected_python), "readme_embedded": True,
    }
    if args.sdist:
        workspace = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
        expected = {name for name in workspace.decode().split("\0") if name and (root / name).is_file()}
        with tarfile.open(args.sdist) as archive:
            files = {}
            for member in archive.getmembers():
                if member.isfile():
                    relative = "/".join(Path(member.name).parts[1:])
                    assert relative not in files, relative
                    files[relative] = archive.extractfile(member).read()
            assert files.keys() == expected | {"PKG-INFO"}, files.keys() ^ (expected | {"PKG-INFO"})
            for name in expected:
                assert files[name] == (root / name).read_bytes(), name
        report["sdist"] = {"bytes": args.sdist.stat().st_size, "sha256": sha256(args.sdist.read_bytes()),
                           "workspace_files_byte_exact": len(expected), "extra_files": ["PKG-INFO"]}
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
