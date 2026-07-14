"""Post-export HTML tidy-ups: the hide-code-by-default flip."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "clean_docs", Path(__file__).resolve().parent.parent / "scripts" / "clean_docs.py"
)
assert _SPEC and _SPEC.loader
clean_docs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(clean_docs)


# Marimo embeds the app view config mid-page in a JS wrapper; this is the shape we flip.
_EXPORT = '<div>… "view": {"showAppCode": true} … "code": "import x"</div>'


def test_flips_show_app_code(tmp_path):
    p = tmp_path / "index.html"
    p.write_text(_EXPORT, "utf-8")
    assert clean_docs.hide_code_by_default(p) is True
    out = p.read_text("utf-8")
    assert '"showAppCode": false' in out
    assert '"showAppCode": true' not in out
    assert "import x" in out  # code stays in the export, so the menu toggle can reveal it


def test_idempotent(tmp_path):
    p = tmp_path / "index.html"
    p.write_text(_EXPORT, "utf-8")
    clean_docs.hide_code_by_default(p)
    assert clean_docs.hide_code_by_default(p) is False  # already flipped — no-op


def test_noop_when_absent(tmp_path):
    """A marimo format change (or an unrelated file) simply no-ops, never corrupts."""
    p = tmp_path / "index.html"
    p.write_text("<div>no config here</div>", "utf-8")
    assert clean_docs.hide_code_by_default(p) is False
