"""
Sidenotes and glossary notes: what a footnote or a defined term says, placed next to the text that needs it.

A python-markdown extension (:class:`NotesExtension`) that rewrites the converted tree, so the Markdown stays plain and the woven ``index.md`` is unchanged. Two things become notes:

- **Footnotes.** Each footnote reference gets a copy of its footnote's text right after it, as a ``<span class="sidenote">``. The list at the end of the document stays, with each item that has a sidenote marked ``sidenoted`` and its number pinned (``value``), so the stylesheet can hide it where the sidenote shows and the remaining items keep their numbers.
- **Glossary terms.** The first use of a defined term in each ``h2`` section is wrapped in a ``<dfn class="term">`` and followed by the definition, as a ``<span class="sidenote glossnote">``. Definitions come from the document's own ``## Glossary`` section (a ``<dl>``, which wins) and then from a shared dictionary (:func:`load_glossary`, ``docs/glossary.toml``). A term is matched case-insensitively as a whole word, with a plural ``s``/``es``, in running prose only: not in code, links, headings, tables, captions, math, or the glossary itself. ``[text](term:key)`` marks a use the matcher would miss (other wording, or a term the dictionary keeps off auto-matching).

Where a note goes is the stylesheet's call (``lit.css``): in the margin on a wide screen and on paper, and on hover or focus of its marker on a narrow screen. A note whose marker sits where a margin float cannot reach (a table cell, a caption, a heading) is marked ``popover`` and only ever shows on hover; its footnote keeps its place in the list, so paper still has it. The floats stack rather than overlap because each one clears the one before it (``clear: right``).

A footnote with block content other than paragraphs (a list, a code block) is left to the list: a note is inline markup, since it sits inside the paragraph that cites it.

Each note leads with a *gloss*, a few words to jog the reader's memory, marked ``<span class="gloss">``; the rest of it is ``<span class="more">`` (and a footnote's later paragraphs, ``sidenote-p-more``). The stylesheets show the gloss alone in the margin until the note or its marker is hovered or focused, and on paper. The gloss is a dictionary entry's ``gloss`` when it has one, and otherwise the first sentence of the note, or what comes before a colon if that is shorter (:func:`_lead_end`). So a note that leads with a short sentence or a "head: explanation" needs nothing more.
"""

from __future__ import annotations

import copy
import html
import re
import tomllib
import xml.etree.ElementTree as etree
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import markdown
import markdown.util
from markdown.extensions import Extension
from markdown.extensions.footnotes import NBSP_PLACEHOLDER
from markdown.preprocessors import Preprocessor
from markdown.treeprocessors import Treeprocessor

__all__ = ["Term", "NotesExtension", "load_glossary", "local_glossary"]


@dataclass(frozen=True)
class Term:
    """A defined term: its display name, its definition as an HTML fragment (inline), the words that match it, whether prose is scanned for it, and a short gloss (inline HTML) to lead the note with, in place of the first sentence of the definition."""

    name: str
    definition: str
    aliases: tuple[str, ...] = ()
    auto: bool = True
    gloss: str | None = None

    def parts(self) -> tuple[str, str]:
        """The note as a gloss and the rest (which starts with its space or colon, or is empty)."""
        if self.gloss:
            return self.gloss, " " + self.definition
        return _split_html(self.definition)

    @property
    def key(self) -> str:
        return _norm(self.name)

    @property
    def forms(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def load_glossary(path: Path) -> dict[str, Term]:
    """The shared dictionary at *path* (TOML), or nothing when there is no file.

    One table per term, keyed by its name; ``definition`` is inline Markdown, ``gloss`` (optional, inline Markdown) is a shorter lead than its first sentence, ``aliases`` (optional) lists other wordings, and ``auto = false`` keeps a term that is also an everyday word from being matched on its own (mark its uses with ``[text](term:key)``)::

        ["expected exact match"]
        definition = "The task score we measure: …"
        gloss = "The task score."
        aliases = ["EEM"]
    """
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text("utf-8"))
    md = markdown.Markdown(extensions=["pymdownx.tilde", "pymdownx.caret", "pymdownx.mark"])
    terms = {}
    for name, entry in data.items():
        definition = _unwrap_p(md.reset().convert(entry["definition"]))
        gloss = _unwrap_p(md.reset().convert(entry["gloss"])) if "gloss" in entry else None
        t = Term(name, definition, tuple(entry.get("aliases", ())), bool(entry.get("auto", True)), gloss)
        terms[t.key] = t
    return terms


_GLOSSARY_HEADING_RE = re.compile(r"^(#{2,6})\s+Glossary\s*$", re.MULTILINE | re.IGNORECASE)
_DT_DD_RE = re.compile(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", re.DOTALL)
_BLOCK_TAG_RE = re.compile(r"<(?:p|ul|ol|table|pre|div|dl|blockquote|figure)\b", re.IGNORECASE)


def local_glossary(text: str) -> dict[str, Term]:
    """The terms a document defines in its own ``Glossary`` section: each ``<dt>``/``<dd>`` pair of the ``<dl>`` there.

    A ``<dt>`` may name several wordings, split on commas (``Red line, non-red line``), and a parenthesised one is an alias (``ᾱ (containment)``).
    """
    terms: dict[str, Term] = {}
    for m in _GLOSSARY_HEADING_RE.finditer(text):
        level = len(m[1])
        end = re.compile(rf"^#{{1,{level}}}\s", re.MULTILINE).search(text, m.end())
        section = text[m.end() : end.start() if end else len(text)]
        for dt, dd in _DT_DD_RE.findall(section):
            words = html.unescape(re.sub(r"<[^>]+>", "", dt)).strip()
            forms = [f.strip() for f in re.split(r",|\(|\)", words) if f.strip()]
            if not forms or _BLOCK_TAG_RE.search(_unwrap_p(dd)):
                continue
            # "Red line, non-red line": the whole is the name, and each part is a form of its own.
            t = Term(words, _unwrap_p(dd.strip()), tuple(forms) if len(forms) > 1 else ())
            terms[t.key] = t
    return terms


def _unwrap_p(fragment: str) -> str:
    """Drop one wrapping ``<p>``, and join several paragraphs with a line break: a note is inline markup."""
    s = fragment.strip()
    paras = re.findall(r"<p>(.*?)</p>", s, re.DOTALL)
    if paras and re.sub(r"<p>.*?</p>", "", s, flags=re.DOTALL).strip() == "":
        return "<br>".join(p.strip() for p in paras)
    return s


# A sentence ends at a stop (and any closing quote or bracket) and a space, before a capital,
# a symbol (σ, $, a quote), the end of the text, or the element that follows it. A lead can
# also end at a colon, as in "Exponential moving average: a running average that …".
_COLON_RE = re.compile(r"(?:^|(?<=\S)):\s")
_STOP_RE = re.compile(r"[.!?][\"')\]\u201d\u2019]*(\s+)")
_ABBREV_RE = re.compile(r"(?:^|[\s(])(?:e\.g|i\.e|vs|cf|al|ex|fig|figs|eq|sec|approx)$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"<[^>]*>|[^<]+")
_VOID_TAGS = {"br", "img", "wbr", "hr", "input"}


def _lead_end(text: str) -> int | None:
    """Where the lead of *text* ends, if it ends within it: after the stop of its first sentence, or before a colon, whichever comes first."""
    colon = _COLON_RE.search(text)
    for m in _STOP_RE.finditer(text, 0, colon.start() if colon else len(text)):
        after = text[m.end() : m.end() + 1]
        if _ABBREV_RE.search(text, 0, m.start()):
            continue
        if not after or after.isupper() or not after.isascii() or after in "$\"'(`":
            return m.start(1)
    return colon.start() if colon else None


def _split_html(fragment: str) -> tuple[str, str]:
    """*fragment* (inline HTML) as its lead (:func:`_lead_end`) and the rest, looking for the end only outside elements."""
    depth = 0
    for m in _TOKEN_RE.finditer(fragment):
        token = m[0]
        if token.startswith("</"):
            depth -= 1
        elif token.startswith("<"):
            tag = re.match(r"<\s*(\w+)", token)
            if not token.endswith("/>") and not (tag and tag[1].lower() in _VOID_TAGS):
                depth += 1
        elif depth == 0 and (end := _lead_end(token)) is not None:
            cut = m.start() + end
            if fragment[cut:].strip():
                return fragment[:cut], fragment[cut:]
            break
    return fragment, ""


def _split_lead(el: etree.Element) -> None:
    """Regroup *el*'s content as a gloss (its lead, :func:`_lead_end`) and, when there is more, the rest."""
    children = list(el)
    segments = [el.text or "", *(c.tail or "" for c in children)]
    cut = next(((i, end) for i, s in enumerate(segments) if (end := _lead_end(s)) is not None), None)
    gloss = etree.Element("span", {"class": "gloss"})
    more = etree.Element("span", {"class": "more"})
    box = gloss
    el.text = None
    for i, text in enumerate(segments):
        if i:
            child = children[i - 1]
            el.remove(child)
            child.tail = None
            box.append(child)
        if cut and i == cut[0]:
            _append_text(box, text[: cut[1]])
            box = more
            text = text[cut[1] :]
        _append_text(box, text)
    el.append(gloss)
    if len(more) or (more.text or "").strip():
        el.append(more)
    elif more.text:
        _append_text(gloss, more.text)


def _append_text(box: etree.Element, text: str | None) -> None:
    """Add *text* at the end of *box*: after its last child, or as its text."""
    if not text:
        return
    if len(box):
        box[-1].tail = (box[-1].tail or "") + text
    else:
        box.text = (box.text or "") + text


class _CollectLocal(Preprocessor):
    """Read the document's own glossary before the raw HTML is stashed away."""

    def __init__(self, md: markdown.Markdown, ext: NotesExtension):
        super().__init__(md)
        self.ext = ext

    def run(self, lines: list[str]) -> list[str]:
        self.ext.local = local_glossary("\n".join(lines))
        return lines


# Where a note is never placed. Matching skips the whole subtree; a footnote marker in one
# of the float-blocking ones gets a hover-only note.
_SKIP_TAGS = {
    "code",
    "pre",
    "a",
    "sup",
    "sub",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "script",
    "style",
    "dfn",
    "abbr",
    "table",
    "figcaption",
    "dl",
    "summary",
    "kbd",
    "samp",
    "math",
}
_SKIP_CLASSES = {"arithmatex", "footnote", "sidenote", "admonition-title", "pending", "highlight"}
_NO_FLOAT_TAGS = {"table", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6", "dt", "summary", "caption"}


def _classes(el: etree.Element) -> set[str]:
    return set((el.get("class") or "").split())


def _skipped(el: etree.Element) -> bool:
    return el.tag in _SKIP_TAGS or bool(_classes(el) & _SKIP_CLASSES)


@dataclass
class _Match:
    term: Term
    start: int
    end: int


@dataclass
class _Scanner:
    """First use per section, over the text nodes of the tree in document order."""

    terms: dict[str, Term]
    pattern: re.Pattern[str] | None = None
    forms: dict[str, Term] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for t in self.terms.values():
            if t.auto:
                for f in t.forms:
                    self.forms[_norm(f)] = t
        if self.forms:
            alts = "|".join(re.escape(f).replace(r"\ ", r"\s+") for f in sorted(self.forms, key=len, reverse=True))
            self.pattern = re.compile(rf"(?<![\w-])({alts})(?:e?s)?(?![\w-])", re.IGNORECASE)

    def first(self, text: str, seen: set[str]) -> _Match | None:
        if self.pattern is None or not text:
            return None
        for m in self.pattern.finditer(text):
            t = self.forms[_norm(m[1])]
            if t.key not in seen:
                return _Match(t, m.start(), m.end())
        return None


class _NotesTree(Treeprocessor):
    def __init__(self, md: markdown.Markdown, ext: NotesExtension):
        super().__init__(md)
        self.ext = ext
        self.stash = md.htmlStash

    def run(self, root: etree.Element) -> None:
        parents = {c: p for p in root.iter() for c in p}
        self._sidenotes(root, parents)
        terms = {**self.ext.shared, **self.ext.local}
        if terms:
            self._explicit_terms(root, parents, terms)
            self._glossary(root, _Scanner(terms))

    # Footnotes -------------------------------------------------------------------------

    def _sidenotes(self, root: etree.Element, parents: dict[etree.Element, etree.Element]) -> None:
        div = next((el for el in root.iter("div") if "footnote" in _classes(el)), None)
        if div is None:
            return
        items = {li.get("id"): li for li in div.iter("li")}
        for n, li in enumerate(items.values(), start=1):
            li.set("value", str(n))
        placed: set[str] = set()
        for sup in list(root.iter("sup")):
            ref = sup.find("a")
            if ref is None or "footnote-ref" not in _classes(ref) or not sup.get("id", "").startswith("fnref:"):
                continue  # not a reference, or a second reference to one note (fnref2:…)
            fid = (ref.get("href") or "").removeprefix("#")
            li = items.get(fid)
            if li is None or fid in placed or any(c.tag != "p" for c in li):
                continue
            floats = not any(a.tag in _NO_FLOAT_TAGS for a in _ancestors(sup, parents))
            note = _footnote_note(ref.text or "", li, floats=floats)
            _insert_after(sup, note, parents)
            sup.set("class", "fnref")
            placed.add(fid)
            if floats:
                li.set("class", "sidenoted")
        if placed and all(li.get("class") == "sidenoted" for li in items.values()):
            div.set("class", "footnote all-sidenoted")

    # Glossary --------------------------------------------------------------------------

    def _explicit_terms(
        self, root: etree.Element, parents: dict[etree.Element, etree.Element], terms: dict[str, Term]
    ) -> None:
        """``[text](term:key)``: a link that says which term its text is a use of (``term:`` alone looks the text up)."""
        for a in list(root.iter("a")):
            href = a.get("href") or ""
            if not href.startswith("term:"):
                continue
            key = _norm(href.removeprefix("term:") or "".join(a.itertext()))
            term = terms.get(key) or next((t for t in terms.values() if key in map(_norm, t.forms)), None)
            a.tag = "dfn"
            a.attrib.clear()
            if term is None:
                continue
            a.set("class", "term explicit")
            a.set("data-term", term.key)

    def _glossary(self, root: etree.Element, scan: _Scanner) -> None:
        seen: set[str] = set()
        self._walk(root, scan, seen)

    def _walk(self, el: etree.Element, scan: _Scanner, seen: set[str]) -> None:
        """Wrap and annotate first uses under *el*: its own text, then each child and the text after it."""
        children = list(el)  # before the text is scanned, which may insert a term and its note
        el.text = self._scan_text(el, None, el.text, scan, seen)
        for child in children:
            if child.tag == "h2":
                seen.clear()  # a new section: each term is defined again at its first use
            last = child  # what the rest of the text follows: the child, or the note placed after it
            if child.tag == "dfn" and "explicit" in _classes(child):
                key = child.attrib.pop("data-term", None)
                if key and key not in seen:
                    seen.add(key)
                    last = self._annotate(el, child, scan.terms[key])
            elif not _skipped(child):
                self._walk(child, scan, seen)
            last.tail = self._scan_text(el, last, last.tail, scan, seen)

    def _scan_text(
        self, parent: etree.Element, before: etree.Element | None, text: str | None, scan: _Scanner, seen: set[str]
    ) -> str | None:
        """Mark the first use of each unseen term in *text* (``parent.text`` when *before* is None, else ``before.tail``); returns what stays as that string."""
        if not text or isinstance(text, markdown.util.AtomicString):
            return text
        m = scan.first(text, seen)
        if m is None:
            return text
        seen.add(m.term.key)
        dfn = etree.Element("dfn", {"class": "term"})
        dfn.text = text[m.start : m.end]
        rest = text[m.end :]
        idx = 0 if before is None else list(parent).index(before) + 1
        parent.insert(idx, dfn)
        note = self._annotate(parent, dfn, m.term)
        # The rest of the string follows the note, and may hold another term's first use.
        note.tail = self._scan_text(parent, note, rest, scan, seen)
        return text[: m.start]

    def _annotate(self, parent: etree.Element, dfn: etree.Element, term: Term) -> etree.Element:
        dfn.set("tabindex", "0")
        note = etree.Element("span", {"class": "sidenote glossnote", "role": "note", "tabindex": "-1"})
        name = etree.SubElement(note, "span", {"class": "glossnote-term"})
        name.text = term.name
        name.tail = " "
        lead, rest = term.parts()
        etree.SubElement(note, "span", {"class": "gloss"}).text = self.stash.store(lead)
        if rest:
            etree.SubElement(note, "span", {"class": "more"}).text = self.stash.store(rest)
        idx = list(parent).index(dfn)
        note.tail = dfn.tail
        dfn.tail = None
        parent.insert(idx + 1, note)
        return note


def _footnote_note(number: str, li: etree.Element, *, floats: bool) -> etree.Element:
    """The note for footnote *li* (all paragraphs): its number, then each paragraph as a span, without the ↩ back to the text; the lead of the first is the gloss."""
    note = etree.Element(
        "span", {"class": "sidenote" if floats else "sidenote popover", "role": "note", "tabindex": "-1"}
    )
    num = etree.SubElement(note, "span", {"class": "sidenote-number"})
    num.text = number
    num.tail = " "
    for i, p in enumerate(li):
        part = etree.SubElement(note, "span", {"class": "sidenote-p sidenote-p-more" if i else "sidenote-p"})
        part.text = p.text
        for child in p:
            if "footnote-backref" in _classes(child):
                _skip(child, part)
            else:
                part.append(copy.deepcopy(child))
        # The non-breaking space the list put before its ↩ (a placeholder until serialised).
        if len(part) and part[-1].tail:
            part[-1].tail = _trim(part[-1].tail)
        elif part.text:
            part.text = _trim(part.text)
        if not i:
            _split_lead(part)
    return note


def _trim(text: str) -> str:
    return text.removesuffix(NBSP_PLACEHOLDER).rstrip()


def _ancestors(el: etree.Element, parents: dict[etree.Element, etree.Element]) -> Iterator[etree.Element]:
    node = parents.get(el)
    while node is not None:
        yield node
        node = parents.get(node)


def _insert_after(el: etree.Element, new: etree.Element, parents: dict[etree.Element, etree.Element]) -> None:
    parent = parents[el]
    parent.insert(list(parent).index(el) + 1, new)
    new.tail, el.tail = el.tail, None
    parents[new] = parent


def _skip(child: etree.Element, into: etree.Element) -> None:
    """Leave *child* out of a copy being built in *into*, keeping the text that follows it."""
    _append_text(into, child.tail)


class NotesExtension(Extension):
    """Footnotes as sidenotes, and glossary terms annotated at first use per section; *glossary* is the shared dictionary (:func:`load_glossary`)."""

    def __init__(self, glossary: dict[str, Term] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.shared = glossary or {}
        self.local: dict[str, Term] = {}

    def extendMarkdown(self, md: markdown.Markdown) -> None:
        md.registerExtension(self)
        md.preprocessors.register(_CollectLocal(md, self), "lit-glossary", 35)
        # After inline (20) and the footnote tree (50, 15), so the notes copy finished markup;
        # before toc (5), which only reads headings, and unescape (0), which then covers the copies too.
        md.treeprocessors.register(_NotesTree(md, self), "lit-notes", 7)

    def reset(self) -> None:
        self.local = {}
