import sys
from pathlib import Path

import pytest

from tests.conftest import load_script


@pytest.fixture
def render_report(monkeypatch):
    # The script imports its siblings (build_site, export_reports, review_base) by name.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent / "scripts"))
    yield load_script("render_report")
    for name in ("render_report", "build_site", "export_reports", "review_base"):
        sys.modules.pop(name, None)


PAGE = '<html><body><main class="lit"><p>one</p></main></body></html>'


def test_a_print_names_its_commit(render_report, tmp_path: Path):
    """With no baseline to mark against, a review print still names the commit it is of, so the next round has a ref."""
    review = render_report.Review(since=None, printed="abc1234 + uncommitted edits")
    out = review.mark(PAGE, "ex-1", root=tmp_path)
    assert "Printed from abc1234 + uncommitted edits" in out


def test_a_review_marks_a_text_only_report(render_report, tmp_path: Path, monkeypatch):
    """A report with no figures exports without an `_assets/` dir; its print is barred against a baseline like any other."""
    base = tmp_path / "base"
    base.mkdir()
    (base / "index.html").write_text(PAGE.replace("one", "zero"))
    monkeypatch.setattr(render_report, "baseline_dir", lambda sha, key: base)
    review = render_report.Review(since="f" * 40, printed="abc1234")
    out = review.mark(PAGE, "ex-1", root=tmp_path / "no-assets")
    assert 'id="rv-data"' in out and "Bars mark changes since fffffff" in out


def test_since_needs_a_pdf(render_report, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["render_report.py", "docs/x/report.py", "-o", "x.md", "--since", "HEAD"])
    with pytest.raises(SystemExit):
        render_report.main()
    assert "--since marks a PDF" in capsys.readouterr().err


def test_a_pdf_gets_no_copy_of_the_figures(render_report, tmp_path: Path, monkeypatch):
    """The PDF is printed from the bundle's own ``_assets/``; only the Markdown and HTML, which link the figures relatively, get a copy beside them."""
    bundle = tmp_path / "bundle"
    (bundle / "_assets").mkdir(parents=True)
    (bundle / "_assets" / "fig.png").write_bytes(b"png")
    (bundle / "index.html").write_text("<main class=lit>hi</main>")
    (bundle / render_report.MD_LEAF).write_text("hi")
    monkeypatch.setattr(render_report, "print_bundle", lambda bundle, out, **kw: out)
    render_report.write_from_bundle(bundle, "<html>", [tmp_path / "prints" / "r.pdf", tmp_path / "site" / "r.html"])
    assert not (tmp_path / "prints" / "_assets").exists()
    assert (tmp_path / "site" / "_assets" / "fig.png").read_bytes() == b"png"
