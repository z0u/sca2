"""Change bars for a re-review print: which words of a page get a bar, against which baseline."""

from pathlib import Path

import pytest

from mini import report_print
from mini.review_marks import mark_changes, stamp


@pytest.fixture
def browser():
    """A headless Chromium, or skip (see ``test_report_print``)."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=report_print.chromium_path())
        except Error as e:
            pytest.skip(f"no Chromium: {e}")
        yield browser
        browser.close()


def _page(body: str) -> str:
    return f'<html><head></head><body><main class="lit">{body}</main></body></html>'


def _barred(browser, head: str, base: str, tmp_path: Path) -> list[str]:
    """The word each bar sits in on the marked *head* page (``img`` for a figure), in document order."""
    html = mark_changes(_page(head), _page(base), note="since abc", root=tmp_path, base_root=tmp_path / "base")
    page = browser.new_page()
    page.set_content(html)
    words = page.evaluate("""() => [...document.querySelectorAll('main.lit .rv')].map(bar => {
      const img = [...document.images].find(i => i.style.anchorName && bar.style.top.includes(i.style.anchorName));
      if (img) return 'img';
      if (bar.parentElement.matches('.arithmatex, mark')) return bar.parentElement.textContent;  // inside an atom
      // Inside a word, after its first character.
      return bar.previousSibling.textContent.slice(-1) + (bar.nextSibling?.textContent ?? '').split(/\\s+/)[0];
    })""")
    page.close()
    return words


_TWO = "<h2>Results</h2><p>The spread falls a little.</p><p>The mean moves up.</p>"


@pytest.mark.parametrize(
    "head, base, expected",
    [
        pytest.param(_TWO, _TWO, [], id="unchanged"),
        pytest.param(
            "<h2>Results</h2><p>The spread falls a lot.</p><p>The mean moves up.</p>", _TWO, ["lot."], id="edited word"
        ),
        pytest.param(
            "<h2>Results</h2><p>The spread falls.</p><p>The mean moves up.</p>", _TWO, ["falls."], id="removed words"
        ),
        pytest.param("<h2>Results</h2><p>The mean moves up.</p>", _TWO, ["The"], id="removed paragraph"),
        pytest.param(_TWO + "<p>A new closing line.</p>", _TWO, ["A", "new", "closing", "line."], id="new paragraph"),
        pytest.param(
            "<h2>Results</h2><p>The spread falls a little.</p><p>The mean moves <span class='arithmatex'>\\(x+1\\)</span> up.</p>",
            "<h2>Results</h2><p>The spread falls a little.</p><p>The mean moves <span class='arithmatex'>\\(x\\)</span> up.</p>",
            ["\\(x+1\\)"],
            id="edited math is one word",
        ),
    ],
)
def test_bars_mark_the_words_that_changed(browser, tmp_path: Path, head: str, base: str, expected: list[str]):
    assert _barred(browser, head, base, tmp_path) == expected


def test_a_redrawn_figure_is_barred_although_its_name_is_the_same(browser, tmp_path: Path):
    (tmp_path / "base" / "_assets").mkdir(parents=True)
    (tmp_path / "_assets").mkdir()
    (tmp_path / "base" / "_assets" / "fig.png").write_bytes(b"old pixels")
    (tmp_path / "_assets" / "fig.png").write_bytes(b"new pixels")
    fig = '<h2>Results</h2><figure><img src="_assets/fig.png"><figcaption>The spread.</figcaption></figure>'
    assert _barred(browser, fig, fig, tmp_path) == ["img"]


def _png(pixels: bytes, *ancillary: tuple[bytes, bytes]) -> bytes:
    """A PNG-shaped file: a header, *ancillary* chunks, and *pixels* as its data (no checksums; the digest skips them)."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return len(body).to_bytes(4, "big") + kind + body + b"\0\0\0\0"

    head = chunk(b"IHDR", b"1x1")
    return b"\x89PNG\r\n\x1a\n" + head + b"".join(chunk(k, v) for k, v in ancillary) + chunk(b"IDAT", pixels)


def test_a_figure_rewritten_with_new_metadata_keeps_its_digest(tmp_path: Path):
    from mini.review_marks import tag_images

    (tmp_path / "a.png").write_bytes(_png(b"pixels", (b"tEXt", b"Software")))
    (tmp_path / "b.png").write_bytes(_png(b"pixels", (b"caBX", b"manifest 1"), (b"tEXt", b"Software")))
    (tmp_path / "c.png").write_bytes(_png(b"other pixels", (b"tEXt", b"Software")))
    tags = [tag_images(f'<img src="{n}.png">', tmp_path) for n in "abc"]
    assert tags[0] == tags[1].replace("b.png", "a.png")
    assert tags[0] != tags[2].replace("c.png", "a.png")


def _counts(browser, head: str, base: str, tmp_path: Path) -> list[str]:
    """The change count at the top of each section of the marked *head* page, in document order."""
    html = mark_changes(_page(head), _page(base), note="since abc", root=tmp_path, base_root=tmp_path / "base")
    page = browser.new_page()
    page.set_content(html)
    counts = page.evaluate("() => [...document.querySelectorAll('main.lit .rv-count')].map(c => c.textContent)")
    page.close()
    return counts


def test_each_section_counts_the_words_added_and_removed_in_it(browser, tmp_path: Path):
    base = "<h1>Title</h1><p>A lede.</p>" + _TWO + "<h2>Method</h2><p>We fit a line.</p>"
    head = "<h1>Title</h1><p>A lede.</p>" + _TWO.replace("a little", "a lot") + "<h2>Method</h2><p>We fit a line.</p>"
    assert _counts(browser, head, base, tmp_path) == ["unchanged", "+1 −1", "unchanged"]


def test_a_changed_table_row_has_one_bar_as_tall_as_the_row(browser, tmp_path: Path):
    table = "<h2>Results</h2><table><tr><td>seed</td><td>{}</td></tr><tr><td>mean</td><td>0.5</td></tr></table>"
    html = mark_changes(
        _page(table.format("0.12")), _page(table.format("0.13")), note="", root=tmp_path, base_root=tmp_path
    )
    page = browser.new_page()
    page.set_content(html)
    bars, row = page.evaluate("""() => [
      [...document.querySelectorAll('main.lit .rv')].map(b => b.getBoundingClientRect().height),
      document.querySelector('tr').getBoundingClientRect().height,
    ]""")
    page.close()
    assert bars == [pytest.approx(row)]


def test_a_redrawn_inline_svg_is_barred(browser, tmp_path: Path):
    fig = '<h2>Results</h2><figure><svg width="40" height="20"><rect width="{}" height="20"/></svg><figcaption>A strip.</figcaption></figure>'
    html = mark_changes(_page(fig.format(30)), _page(fig.format(20)), note="", root=tmp_path, base_root=tmp_path)
    page = browser.new_page()
    page.set_content(html)
    bars, svg = page.evaluate("""() => [
      [...document.querySelectorAll('main.lit .rv')].map(b => b.getBoundingClientRect().height),
      document.querySelector('svg').getBoundingClientRect().height,
    ]""")
    page.close()
    assert bars == [pytest.approx(svg)]


def test_the_hidden_dark_copy_of_a_themed_figure_gets_no_bar(browser, tmp_path: Path):
    for theme, pixels in [("light", b"new"), ("dark", b"new too")]:
        (tmp_path / "base" / "_assets").mkdir(parents=True, exist_ok=True)
        (tmp_path / "_assets").mkdir(exist_ok=True)
        (tmp_path / "base" / "_assets" / f"{theme}.png").write_bytes(b"old")
        (tmp_path / "_assets" / f"{theme}.png").write_bytes(pixels)
    fig = (
        '<h2>Results</h2><figure class="mini-themed-figure"><img class="mini-themed-img-light" src="_assets/light.png">'
        '<img class="mini-themed-img-dark" src="_assets/dark.png"></figure>'
    )
    assert _barred(browser, fig, fig, tmp_path) == ["img"]


def test_an_unmarked_print_still_names_its_version():
    html = stamp(_page("<h2>Results</h2><p>Text.</p>"), note="Printed from abc1234")
    assert '<main class="lit"><div class="rv-note">Printed from abc1234</div><h2>' in html
    assert "<style>" in html


def test_a_word_at_the_start_of_a_wrapped_line_is_barred_on_its_own_line(browser, tmp_path: Path):
    """The bar sits inside the word: before it, at a line break, its static position is the end of the line above."""
    words = [f"w{i}" for i in range(40)]
    base = "<h2>R</h2><p>" + " ".join(words) + "</p>"
    html = mark_changes(_page(base.replace("w13", "edited")), _page(base), note="", root=tmp_path, base_root=tmp_path)
    page = browser.new_page()
    page.set_content("<style>main.lit { width: 300px; font: 16px sans-serif }</style>" + html)
    top, word_top = page.evaluate("""() => {
      const bar = document.querySelector('main.lit .rv');
      const r = document.createRange(); r.selectNodeContents(bar.previousSibling);
      return [bar.getBoundingClientRect().top, r.getBoundingClientRect().top];
    }""")
    page.close()
    assert top == pytest.approx(word_top, abs=2)


@pytest.mark.parametrize(
    "head, base, expected",
    [
        pytest.param(
            "<h2>R</h2><ul><li>Foo baz<ul><li>bar</li></ul></li></ul>",
            "<h2>R</h2><ul><li>Foo bar<ul><li>bar</li></ul></li></ul>",
            ["baz"],
            id="a list item's own words, before its sub-list",
        ),
        pytest.param(
            "<h2>R</h2><p>So:</p><div class='arithmatex'>\\[x+1\\]</div>",
            "<h2>R</h2><p>So:</p><div class='arithmatex'>\\[x\\]</div>",
            ["\\[x+1\\]"],
            id="display math",
        ),
        pytest.param(
            "<h2>A</h2><p>One two.</p><h2>B</h2><p>Three four.</p>",
            "<h2>A</h2><p>One two.</p><p>Gone now.</p><h2>B</h2><p>Three four.</p>",
            ["two."],
            id="a paragraph removed from the end of a section marks that section",
        ),
        pytest.param(
            "<h2>R</h2><p>One two.</p>",
            "<h2>R</h2><p>One two.</p><p>Gone now.</p>",
            ["two."],
            id="a paragraph removed from the end of the page",
        ),
    ],
)
def test_bars_reach_the_blocks_that_were_missed(browser, tmp_path: Path, head: str, base: str, expected: list[str]):
    assert _barred(browser, head, base, tmp_path) == expected


def test_a_removal_at_the_end_of_a_section_counts_on_that_section(browser, tmp_path: Path):
    head = "<h2>A</h2><p>One two.</p><h2>B</h2><p>Three four.</p>"
    base = "<h2>A</h2><p>One two.</p><p>Gone now.</p><h2>B</h2><p>Three four.</p>"
    assert _counts(browser, head, base, tmp_path) == ["+0 −2", "unchanged"]
