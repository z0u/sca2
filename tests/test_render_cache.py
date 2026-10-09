import sys
from pathlib import Path

import pytest

from tests.conftest import load_script

REPORT = Path(__file__).resolve().parent.parent / "docs" / "m2" / "embedding-lean" / "report.py"


@pytest.fixture
def render_cache():
    yield load_script("render_cache")
    sys.modules.pop("render_cache", None)


def a_render(tmp_path: Path, text: str) -> Path:
    md = tmp_path / text / "index.md"
    (md.parent / "_assets").mkdir(parents=True)
    md.write_text(text)
    (md.parent / "_assets" / "fig.png").write_bytes(b"png")
    return md


def test_a_stored_render_is_found_with_its_figures(render_cache, tmp_path: Path):
    root = tmp_path / "cache"
    assert render_cache.lookup("m2/x", "v1", root) is None
    render_cache.store(a_render(tmp_path, "one"), "m2/x", "v1", REPORT, root)
    hit = render_cache.lookup("m2/x", "v1", root)
    assert hit is not None and hit.read_text() == "one"
    assert (hit.parent / "_assets" / "fig.png").read_bytes() == b"png"
    assert render_cache.lookup("m2/x", "v2", root) is None


def test_the_first_writer_keeps_its_entry(render_cache, tmp_path: Path):
    """Two sessions that miss at once both render; the second copy is dropped, and no temporary directory is left behind."""
    root = tmp_path / "cache"
    render_cache.store(a_render(tmp_path, "one"), "m2/x", "v1", REPORT, root)
    hit = render_cache.store(a_render(tmp_path, "two"), "m2/x", "v1", REPORT, root)
    assert hit.read_text() == "one"
    assert [p.name for p in (root / "m2" / "x").iterdir()] == ["v1"]


def test_a_cached_render_copies_out_with_its_figures(render_cache, tmp_path: Path):
    root = tmp_path / "cache"
    hit = render_cache.store(a_render(tmp_path, "one"), "m2/x", "v1", REPORT, root)
    out = tmp_path / "out" / "index.md"
    render_cache.copy_out(hit, out)
    assert out.read_text() == "one" and (out.parent / "_assets" / "fig.png").exists()


def test_the_version_follows_the_profile(render_cache, monkeypatch):
    """The dev profile reads different results, so it gets its own entries."""
    monkeypatch.setattr(render_cache, "_git", lambda *args: "" if args[0] == "status" else "a\nb\nc\nd")
    assert render_cache.version(REPORT, None) != render_cache.version(REPORT, "dev")


def test_edits_leave_no_version(render_cache, monkeypatch):
    monkeypatch.setattr(render_cache, "_git", lambda *args: " M docs/m2/x/report.py")
    assert render_cache.version(REPORT, None) is None


def test_the_cache_follows_the_env(render_cache, tmp_path: Path, monkeypatch):
    monkeypatch.setenv(render_cache.CACHE_ENV, str(tmp_path))
    assert render_cache.cache_root() == tmp_path
