"""Sidenotes and glossary notes (mini.lit.notes), through the page renderer."""

import re

from mini.lit.notes import Term, load_glossary, local_glossary
from mini.lit.page import render_fragment, to_html

GLOSSARY = """
## Glossary

<dl>
<dt>Op margin</dt>
<dd>The line margin, <code>m</code>.</dd>
<dt>ᾱ (containment)</dt>
<dd>Mean alignment.</dd>
</dl>
"""


class TestSidenotes:
    def test_footnote_is_placed_beside_its_marker(self):
        html = to_html("Text.[^a] More.\n\n[^a]: A *note*.\n")
        assert re.search(
            r'<sup class="fnref" id="fnref:a">.*?</sup><span class="sidenote" role="note" tabindex="-1"><span class="sidenote-number">1</span> <span class="gloss">A <em>note</em>.</span></span> More.',
            html,
        )
        assert '<li class="sidenoted" id="fn:a" value="1">' in html
        assert 'class="footnote all-sidenoted"' in html

    def test_backlink_and_its_space_are_left_out(self):
        html = to_html("Text.[^a]\n\n[^a]: Note.\n")
        note = html.split('class="sidenote"')[1].split("</p>")[0]
        assert "footnote-backref" not in note
        assert "Note.</span>" in note

    def test_first_sentence_is_the_gloss(self):
        html = to_html("Text.[^a]\n\n[^a]: A *note*, e.g. this. Then `more`.\n\n    Second.\n")
        assert (
            '<span class="gloss">A <em>note</em>, e.g. this.</span><span class="more"> Then <code>more</code>.'
            '<span class="sidenote-p">Second.</span></span></span>' in html
        )

    def test_note_in_a_table_is_hover_only_and_stays_in_the_list(self):
        html = to_html("| a |\n|---|\n| x[^a] |\n\nText.[^b]\n\n[^a]: In a cell.\n[^b]: In prose.\n")
        assert 'class="sidenote popover"' in html
        assert '<li id="fn:a" value="1">' in html
        assert '<li class="sidenoted" id="fn:b" value="2">' in html
        assert "all-sidenoted" not in html

    def test_footnote_with_a_list_stays_in_the_list(self):
        html = to_html("Text.[^a]\n\n[^a]: Items:\n\n    - one\n    - two\n")
        assert "sidenote" not in html.replace("sidenoted", "")

    def test_notes_can_be_turned_off(self):
        assert "sidenote" not in to_html("Text.[^a]\n\n[^a]: Note.\n", notes=False)
        assert "sidenote" not in render_fragment("Caption.[^a]\n\n[^a]: Note.\n")


class TestGlossary:
    def test_local_glossary_reads_the_section(self):
        terms = local_glossary(GLOSSARY)
        assert set(terms) == {"op margin", "ᾱ (containment)"}
        assert terms["ᾱ (containment)"].forms[1:] == ("ᾱ", "containment")
        assert terms["op margin"].definition == "The line margin, <code>m</code>."

    def test_first_use_per_section_is_annotated(self):
        html = to_html(
            f"## One\n\nThe op margins and the op margin.\n\n## Two\n\nContainment and op margin.\n{GLOSSARY}"
        )
        assert html.count('<dfn class="term"') == 3
        assert (
            '<dfn class="term" tabindex="0">op margins</dfn><span class="sidenote glossnote" role="note" tabindex="-1"><span class="glossnote-term">Op margin</span> <span class="gloss">The line margin, <code>m</code>.</span></span> and the op margin.'
            in html
        )
        assert '<dfn class="term" tabindex="0">Containment</dfn>' in html

    def test_code_links_headings_and_the_glossary_are_left_alone(self):
        html = to_html(f"## Op margin\n\n`op margin` and [op margin](#x).\n{GLOSSARY}")
        assert "<dfn" not in html

    def test_whole_words_only(self):
        assert "<dfn" not in to_html(f"Scoop margins.\n{GLOSSARY}")

    def test_shared_terms_and_local_precedence(self):
        shared = {"op margin": Term("Op margin", "Shared."), "band": Term("Band", "Precision.")}
        html = to_html(f"Op margin and band.\n{GLOSSARY}", glossary=shared)
        assert "Shared." not in html
        assert '<span class="gloss">Precision.</span></span>' in html

    def test_gloss_leads_the_definition(self):
        shared = {
            "op margin": Term("Op margin", "The margin at a rate κ. Usually small."),
            "band": Term("Band", "Precision, long form.", gloss="Precision."),
        }
        html = to_html("Op margin and band.\n", glossary=shared)
        assert '<span class="gloss">The margin at a rate κ.</span><span class="more"> Usually small.</span>' in html
        assert '<span class="gloss">Precision.</span><span class="more"> Precision, long form.</span>' in html

    def test_a_colon_can_end_the_gloss(self):
        shared = {
            "ema": Term("EMA", "Exponential moving average: a running mean. See <code>a: b</code>."),
            "ctx": Term("Context", "An <em>x</em>: more."),
        }
        html = to_html("An EMA.\n", glossary=shared)
        assert (
            '<span class="gloss">Exponential moving average<span class="lead-colon">:</span></span>'
            '<span class="more"> a running mean.' in html
        )
        html = to_html("A context.\n", glossary=shared)
        assert (
            '<span class="gloss">An <em>x</em><span class="lead-colon">:</span></span><span class="more"> more.</span>'
            in html
        )
        assert to_html("An EMA.\n", glossary={"ema": Term("EMA", "Short. Then: more.")}).count(
            '<span class="gloss">Short.</span>'
        )

    def test_note_is_focusable_off_the_tab_order(self):
        assert 'class="sidenote glossnote" role="note" tabindex="-1"' in to_html(f"Op margin.\n{GLOSSARY}")

    def test_explicit_use_of_a_term_off_auto(self):
        shared = {"band": Term("Band", "Precision.", auto=False)}
        html = to_html("A band here, and [the band](term:band).\n", glossary=shared)
        assert html.count("<dfn") == 1
        assert '<dfn class="term explicit" tabindex="0">the band</dfn><span class="sidenote glossnote"' in html
        assert "term:" not in html

    def test_load_glossary(self, tmp_path):
        assert load_glossary(tmp_path / "missing.toml") == {}
        path = tmp_path / "g.toml"
        path.write_text('["op margin"]\ndefinition = "A *margin*."\ngloss = "*M*."\naliases = ["m_op"]\nauto = false\n')
        assert load_glossary(path) == {
            "op margin": Term("op margin", "A <em>margin</em>.", ("m_op",), False, "<em>M</em>.")
        }
