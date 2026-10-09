"""A shared cache of Markdown renders, so a report is woven once per version and every later reader reuses it (``./go render <report> --cached``).

Each thread of a project runs in its own container, and a fresh container has neither the memo cache nor, sometimes, the store credentials that a render needs. Those containers do share one mounted folder, so the cache lives there: ``<root>/<key>/<version>/`` holds ``index.md``, its ``_assets/``, and a ``meta.json`` saying what it was woven from.

The version is a digest of everything the Markdown depends on: the git trees of ``docs/`` and ``src/`` (a report can load a sibling experiment, so its own directory is not enough), the lock and project files, the report path, and the storage profile (the dev profile reads different results). A commit that touches neither tree, such as a todo or a skill edit, keeps every entry valid. A working tree with edits under any of those paths has no version, so it renders without the cache.

The mount is eventually consistent and the last write wins. An entry is written to a temporary directory and renamed into place, so a reader sees a whole entry or none; two threads that miss at once both render, and either result is the same document.
"""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
SHARED = Path("/mnt/project-files")
CACHE_ENV = "MINI_RENDER_CACHE"
INPUTS = ("docs", "src", "pyproject.toml", "uv.lock")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def cache_root() -> Path:
    """``$MINI_RENDER_CACHE`` if set, else ``rendered/`` in the project's shared folder, else ``.mini/rendered/`` (a cache for this checkout alone)."""
    if env := os.environ.get(CACHE_ENV):
        return Path(env)
    return SHARED / "rendered" if SHARED.is_dir() else ROOT / ".mini" / "rendered"


def version(report: Path, profile: str | None) -> str | None:
    """The cache version of *report* at ``HEAD``, or ``None`` when the working tree has edits the render would see."""
    if _git("status", "--porcelain", "--", *INPUTS):
        return None
    inputs = {
        "objects": _git("rev-parse", *(f"HEAD:{p}" for p in INPUTS)).split(),
        "report": report.resolve().relative_to(ROOT).as_posix(),
        "profile": profile or "",
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]


def entry(key: str, ver: str, root: Path | None = None) -> Path:
    """The directory holding the render of report *key* at version *ver*."""
    return (root or cache_root()) / key / ver


def lookup(key: str, ver: str, root: Path | None = None) -> Path | None:
    """The cached ``index.md`` of *key* at *ver*, if one has been written."""
    md = entry(key, ver, root) / "index.md"
    return md if md.is_file() else None


def store(md: Path, key: str, ver: str, report: Path, root: Path | None = None) -> Path:
    """Copy the render at *md* (and the ``_assets/`` beside it) into the cache, returning the cached ``index.md``.

    The entry is assembled under a temporary name and renamed into place. When another writer got there first, its entry stays and this copy is dropped.
    """
    dest = entry(key, ver, root)
    tmp = dest.with_name(f".{ver}.{os.getpid()}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copyfile(md, tmp / "index.md")
    if (assets := md.parent / "_assets").is_dir():
        shutil.copytree(assets, tmp / "_assets")
    meta = {
        "report": report.resolve().relative_to(ROOT).as_posix(),
        "commit": _git("rev-parse", "HEAD"),
        "woven_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    try:
        tmp.rename(dest)
    except OSError:  # a whole entry is already there
        shutil.rmtree(tmp, ignore_errors=True)
    return dest / "index.md"


def copy_out(cached: Path, out: Path) -> None:
    """Write the cached render to *out*, with its ``_assets/`` beside it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, out)
    if (assets := cached.parent / "_assets").is_dir():
        shutil.copytree(assets, out.parent / "_assets", dirs_exist_ok=True)
