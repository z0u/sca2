"""The export's PDF print: stable bytes, and a quiet exit without a browser."""

import re
from pathlib import Path

import pikepdf
import pytest

from mini import report_print

_PAGE = "<html><body><h1>Hi</h1><p>A page with <a href='https://example.test/'>a link</a>.</p></body></html>"


@pytest.fixture
def browser():
    """A headless Chromium, or skip: the CI runner has none, and that is the case the exporter's fallback covers. Function-scoped, since Playwright's sync API owns one event loop per context and the next test opens its own."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=report_print.chromium_path())
        except Error as e:
            pytest.skip(f"no Chromium: {e}")
        yield browser
        browser.close()


@pytest.fixture
def chromium():
    """Skip unless a Chromium launches, for a test that lets :func:`print_bundle` open its own (two sync Playwright contexts cannot share a thread)."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as pw:
        try:
            pw.chromium.launch(executable_path=report_print.chromium_path()).close()
        except Error as e:
            pytest.skip(f"no Chromium: {e}")


def _never(route) -> None:
    """A route handler that neither fulfils nor aborts: the request stays pending for the life of the page."""


def test_two_prints_of_one_page_are_byte_equal(browser, tmp_path: Path):
    """An unchanged report must re-publish as an unchanged bundle; Chromium's dates and random ID would break that."""
    outs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for out in outs:
        page = browser.new_page()
        page.set_content(_PAGE)
        page.wait_for_timeout(1100)  # CreationDate has one-second resolution
        report_print.print_page(page, out, settle=0)
        page.close()
    assert outs[0].read_bytes() == outs[1].read_bytes()
    with pikepdf.open(outs[0]) as pdf:
        assert {"/CreationDate", "/ModDate"}.isdisjoint(pdf.docinfo.keys())


def test_print_bundle_skips_without_a_browser(tmp_path: Path, monkeypatch, caplog):
    """A publish must not fail on the PDF: no Chromium means no file, one warning, and the bundle untouched."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "index.html").write_text(_PAGE)
    monkeypatch.setattr(report_print, "chromium_path", lambda: str(tmp_path / "no-such-chromium"))
    out = report_print.print_bundle(bundle, bundle / "report.pdf")
    assert out is None
    assert not (bundle / "report.pdf").exists()
    assert not (tmp_path / ".render-bundle").exists()  # the serve root is cleaned up on the way out
    assert [r.message[:29] for r in caplog.records] == ["report PDF skipped: no Chromi"]


_FIGURED = '<html><body><main class="lit"><h1>Hi</h1><img src="https://cdn.test/never.png"></main></body></html>'


def test_wait_for_figures_gives_up_on_an_image_that_never_arrives(browser, tmp_path: Path, caplog):
    """A figure the CDN never delivers must not hold the print forever: the wait is bounded, the image is named, and the page still prints."""
    page = browser.new_page()
    page.route(re.compile(r"^https://"), _never)
    page.set_content(_FIGURED, wait_until="domcontentloaded")
    assert report_print.wait_for_figures(page, timeout=0.3) == ["https://cdn.test/never.png"]
    assert "1 figure(s) had not arrived after 0s; printing without them: https://cdn.test/never.png" in caplog.text
    report_print.print_page(page, tmp_path / "out.pdf", settle=0)
    page.close()
    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        assert len(pdf.pages) == 1


def test_print_bundle_prints_a_page_whose_figure_never_arrives(chromium, tmp_path: Path, monkeypatch, caplog):
    """The site build prints thirty reports in a row; one stalled figure costs its wait and nothing more."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "index.html").write_text(_FIGURED)
    monkeypatch.setattr(report_print, "prefetch_remote", lambda html: set())
    monkeypatch.setattr(report_print, "route_remote", lambda page, **_: page.route(re.compile(r"^https://"), _never))
    out = report_print.print_bundle(bundle, tmp_path / "report.pdf", figures=0.3, settle=0)
    assert out == tmp_path / "report.pdf" and out.is_file()
    assert "printing without them: https://cdn.test/never.png" in caplog.text


def test_print_bundle_skips_a_page_that_never_arrives(chromium, tmp_path: Path, monkeypatch, caplog):
    """Navigation is bounded like every other wait: a page that does not come is skipped with a log line, never an exception."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "index.html").write_text(_PAGE)
    monkeypatch.setattr(report_print, "route_remote", lambda page, **_: page.route("**/index.html", _never))
    assert report_print.print_bundle(bundle, tmp_path / "report.pdf", timeout=0.3, settle=0) is None
    assert not (tmp_path / "report.pdf").exists()
    assert "report PDF skipped: no main.lit on the page after 0s" in caplog.text


_SECTIONED = (
    """<html><head><style>
@page { size: 100mm 120mm; margin: 10mm }
h2 { break-before: page }
</style></head><body>
<h1>Title</h1><p>A short first page.</p>
<h2>Long</h2>"""
    + "<p>Line.</p>" * 60
    + """
<h2>Short</h2><p>One line.</p>
</body></html>"""
)

_MM = 72 / 25.4  # points per mm


def test_fit_prints_one_page_per_section_and_clips_each_to_its_ink(browser, tmp_path: Path):
    """The long section overflows the stylesheet's page; the print grows the sheet until it fits, then cuts every page to what is on it, no shorter than the reader's screen."""
    out = tmp_path / "fit.pdf"
    page = browser.new_page()
    page.set_content(_SECTIONED)
    report_print.print_page(page, out, settle=0)
    page.close()
    with pikepdf.open(out) as pdf:
        heights = [float(p.MediaBox[3]) - float(p.MediaBox[1]) for p in pdf.pages]
        width = float(pdf.pages[0].MediaBox[2]) - float(pdf.pages[0].MediaBox[0])
    assert len(heights) == 3  # title page + two sections, none broken across pages
    assert heights[1] > 120 * _MM  # the long section needed more than the stylesheet's page
    floor = (
        report_print.MIN_PAGE_ASPECT * width
    )  # the short ones were cut back, to the screen's height rather than their content
    assert heights[0] == pytest.approx(floor) and heights[2] == pytest.approx(floor)


def test_padded_height_rounds_a_short_page_up_to_whole_screens():
    screen = report_print.SCREEN_ASPECT * 100
    assert report_print.padded_height(10, 100) == screen
    assert report_print.padded_height(screen * 1.2, 100) == 2 * screen  # would zoom out to fit; scrolls at two
    assert report_print.padded_height(screen * 2.5, 100) == screen * 2.5


def test_fit_leaves_a_page_without_a_sized_page_rule_alone(browser, tmp_path: Path):
    out = tmp_path / "plain.pdf"
    page = browser.new_page()
    page.set_content(_PAGE)
    report_print.print_page(page, out, settle=0)
    page.close()
    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 1
        assert float(pdf.pages[0].MediaBox[3]) == pytest.approx(11 * 72, abs=1)  # Chromium's default letter page, uncut


_FOOTNOTED = """<html><body><h1>Hi</h1><p>A claim.<sup id="fnref:a"><a href="#fn:a">1</a></sup></p>
<div class="footnote"><ol><li id="fn:a"><p>A note. <a href="#fnref:a">back</a></p></li></ol></div></body></html>"""


def test_in_page_links_print_as_direct_destinations(browser, tmp_path: Path):
    """Chromium prints a footnote link as a named destination; the pass gives the annotation the page and position outright, so a viewer without name lookup follows it."""
    out = tmp_path / "fn.pdf"
    page = browser.new_page()
    page.set_content(_FOOTNOTED)
    report_print.print_page(page, out, settle=0)
    page.close()
    with pikepdf.open(out) as pdf:
        dests = [a.Dest for p in pdf.pages for a in (p.get("/Annots") or []) if "/Dest" in a]
        assert len(dests) == 2
        for d in dests:
            assert isinstance(d, pikepdf.Array) and d[1] == "/XYZ"
            assert d[0].objgen == pdf.pages[0].obj.objgen


def test_ink_extents_of_a_blank_page_is_none(tmp_path: Path):
    out = tmp_path / "blank.pdf"
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(200, 400))
    pdf.save(out)
    assert report_print.ink_extents(out) == [None]


def test_route_remote_serves_https_from_a_python_filled_cache(browser, tmp_path: Path):
    """The math and fonts come from CDNs the print browser may not reach; Python fetches once and the page gets the cached copy."""
    calls = []

    def fetch(url, headers):
        calls.append(url)
        return "text/css", b"body { color: rgb(1, 2, 3); }"

    for _ in range(2):
        page = browser.new_page()
        report_print.route_remote(page, cache=tmp_path, fetch=fetch)
        page.set_content(
            '<html><head><link rel="stylesheet" href="https://cdn.test/x.css"></head><body>hi</body></html>'
        )
        page.wait_for_load_state("load")
        assert page.evaluate("getComputedStyle(document.body).color") == "rgb(1, 2, 3)"
        page.close()
    assert calls == ["https://cdn.test/x.css"]  # the second page read the cache


def test_route_remote_prints_on_when_a_fetch_fails(browser, tmp_path: Path, caplog):
    def fetch(url, headers):
        raise OSError("no route to host")

    page = browser.new_page()
    report_print.route_remote(page, cache=tmp_path, fetch=fetch)
    page.set_content('<html><head><link rel="stylesheet" href="https://cdn.test/x.css"></head><body>hi</body></html>')
    page.wait_for_load_state("load")
    assert page.locator("body").inner_text() == "hi"
    assert "cdn.test unreachable" in caplog.text
    assert not list(tmp_path.iterdir())


def test_prefetch_remote_fills_the_cache_the_page_reads(browser, tmp_path: Path):
    """A report's figures are fetched before the page loads, several at once, so the route handler finds each one on disk."""
    import threading

    calls, started = [], threading.Barrier(2, timeout=5)

    def fetch(url, headers):
        calls.append(url)
        started.wait()  # both fetches are in flight together, or this times out
        return "image/svg+xml", b'<svg xmlns="http://www.w3.org/2000/svg" width="7" height="5"/>'

    html = (
        '<main><img src="https://cdn.test/a.svg?x=1&amp;y=2"><img alt="" src="https://cdn.test/b.svg">'
        '<img src="https://cdn.test/a.svg?x=1&amp;y=2"><img src="_assets/local.png"></main>'
    )
    assert report_print.prefetch_remote(html, cache=tmp_path, fetch=fetch) == set()
    assert sorted(calls) == ["https://cdn.test/a.svg?x=1&y=2", "https://cdn.test/b.svg"]

    def no_fetch(url, headers):
        raise AssertionError(f"fetched {url} again")

    page = browser.new_page()
    report_print.route_remote(page, cache=tmp_path, fetch=no_fetch)
    page.set_content(html.replace('<img src="_assets/local.png">', ""))
    page.wait_for_load_state("load")
    assert page.evaluate("[...document.images].map(i => i.naturalWidth)") == [7, 7, 7]
    page.close()


def test_prefetch_remote_names_a_failed_host_once(tmp_path: Path, caplog):
    """A host that fails is warned about once and handed back, so the route handler does not wait on it again."""

    def fetch(url, headers):
        if "down.test" in url:
            raise OSError("no route to host")
        return "image/png", b"png"

    html = "".join(f'<img src="https://{h}/{i}.png">' for h in ("down.test", "up.test") for i in range(3))
    assert report_print.prefetch_remote(html, cache=tmp_path, fetch=fetch) == {"down.test"}
    assert caplog.text.count("down.test unreachable") == 1
    assert len([p for p in tmp_path.iterdir() if p.suffix != ".type"]) == 3  # up.test's, and no partial files


_WIDE_TABLE = (
    """<html><head><style>
@page { size: 100mm 120mm; margin: 10mm }
body { font-size: 11pt }
</style></head><body>
<h1>Title</h1>
<figure><table><thead><tr>"""
    + "".join(f"<th>column heading {i}</th>" for i in range(8))
    + "</tr></thead><tbody><tr>"
    + "".join(f"<td>a value with a long range ({i}.40 to {i}.46)</td>" for i in range(8))
    + """</tr></tbody></table><figcaption>The wide one.</figcaption></figure>
<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>
</body></html>"""
)


def test_wrapped_cells_names_a_table_that_wraps_on_the_sheet_and_the_print_warns(browser, tmp_path: Path, caplog):
    page = browser.new_page()
    page.set_content(_WIDE_TABLE)
    wrapped = report_print.wrapped_cells(page, 100)
    assert [t["name"] for t in wrapped] == ["The wide one."]  # the two-column table fits
    assert wrapped[0]["columns"] == 8 and wrapped[0]["headers"] == 8 and wrapped[0]["cells"] == 8
    report_print.print_page(page, tmp_path / "wide.pdf", settle=0)
    page.close()
    assert "table 'The wide one.' (8 columns) wraps on the page: 8 header and 8 body cell(s)" in caplog.text
