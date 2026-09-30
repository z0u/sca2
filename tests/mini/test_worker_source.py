"""
A Modal worker gets the project packages through a source mount, which by default holds only
their `.py` files. Code that reads a package data file (a stylesheet, an mplstyle) then passes
locally and fails on a worker. `make_image` mounts the data too, and `mini` reads no data file
at import, so a worker with a stale or partial mount still gets as far as running the task.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from modal.mount import _MountedPythonModule

from mini.modal_apparatus import skip_in_source
from mini.requirements import find_project_root, project_packages

SRC = find_project_root() / "src"


@pytest.mark.parametrize("package", project_packages())
def test_mount_carries_package_data(package):
    """Every file the package tracks reaches the worker; bytecode doesn't."""
    root = SRC / package
    if not root.is_dir():
        pytest.skip(f"{package} is a single-file module")
    mounted = {
        str(remote).removeprefix(f"/root/{package}/")
        for _, remote in _MountedPythonModule(package, ignore=skip_in_source).get_files_to_upload()
    }
    tracked = subprocess.run(
        ["git", "ls-files", "--", "."], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    assert tracked, f"git lists no files under {root}"
    assert set(tracked) <= mounted
    assert not [p for p in mounted if "__pycache__" in p or p.endswith(".pyc")]


def _python_source_only(directory: str, names: list[str]) -> list[str]:
    """A `copytree` ignore that keeps what Modal's default mount keeps: `.py` files."""
    return [n for n in names if n == "__pycache__" or not (n.endswith(".py") or (Path(directory) / n).is_dir())]


def test_mini_imports_without_package_data(tmp_path):
    """Importing any `mini` module reads no data file, so a mount without them still imports."""
    shutil.copytree(SRC / "mini", tmp_path / "mini", ignore=_python_source_only)
    assert not [p for p in (tmp_path / "mini").rglob("*") if p.is_file() and p.suffix != ".py"]
    script = """
import pkgutil, sys, importlib
import mini
assert mini.__file__.startswith(sys.argv[1]), mini.__file__
failed = []
for m in pkgutil.walk_packages(mini.__path__, "mini."):
    try:
        importlib.import_module(m.name)
    except FileNotFoundError as e:
        failed.append(f"{m.name}: {e}")
print("\\n".join(failed))
sys.exit(1 if failed else 0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
