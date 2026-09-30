"""Change bars for a re-review: mark, in the margin of a report's print, every line that differs from an earlier version.

The reader reviews a report on paper or e-ink, rounds apart. From the second round on, most of the report is what they already annotated, so the print (``./go render <report> -o <name>.pdf --since <ref>``) shows which lines changed since the version they last saw: a bar on the paper's right edge beside each one, clear of a figure or table that reaches into the margin, and of the reMarkable's toolbar on the left. A new word, an edited one, or a removal all mark the line where it falls; how much changed on the line does not matter, because the line is re-read either way.

The comparison is between two *rendered* pages, the report's and the baseline's (:func:`mark_changes`), rather than two sources, so a number an f-string prints is compared too. It runs in the page, in the browser that prints it (:data:`_JS`): blocks (paragraphs, list items, table cells, captions, headings, display math) are aligned first, anchored on the ones that match word for word, with the rest paired by similarity; each changed pair is then diffed word by word. A block's words are its own text, so a list item keeps the words before its sub-list and the sub-list's items are blocks of their own. Inline math and figures are one word each: an ``<img>`` is compared by a digest of its file, since a redrawn figure keeps its name, and an inline ``<svg>`` (a subline strip, a swatch table) by its markup.

A bar is drawn without measuring anything: before each changed word the script inserts an empty span that is absolutely positioned with ``top: auto``, so it sits on the line the word is on, wherever the print's layout puts that line, and its ``right`` reaches across the ``@page`` margin to the edge of the paper. Each bar is a line tall and opaque, and overlaps its neighbours by a hair, so the bars of consecutive lines join into one without a seam (a PDF reader anti-aliases each edge, and two abutting edges would show a faint gap). A changed figure's bar is as tall as the image, and a table row with any change in it has one bar as tall as the row, through CSS anchor positioning. Each section's page carries a count at the top of its margin, words added and removed (``+12 −3``) or ``unchanged``, so the reader can skip a page at a glance; it is indicative, a sense of how much moved rather than a tally of the bars. A note under the first page's count, on the same edge, names the two versions; a print without a baseline carries the same note with only the version it was printed from (:func:`stamp`), so the next round has a ref to name. The ref is a commit: a PDF printed from a tree with uncommitted edits says so in its note, and a later ``--since`` that names its commit diffs against the commit, not the paper.
"""

from __future__ import annotations

import hashlib
import json
import re
from html import escape
from pathlib import Path

from mini.lit.page import MAIN_OPEN

__all__ = ["mark_changes", "stamp", "tag_images", "baseline_dir"]

_IMG_RE = re.compile(r'<img\b[^>]*?\bsrc="([^"]+)"')

_CSS = """
main.lit { position: relative; }
main.lit .rv {
  position: absolute;
  right: calc(-1 * var(--rv-edge, 0px));
  width: 6px;
  height: calc(1lh + 1px);
  margin-top: -0.5px;
  background: #f3a978;
  pointer-events: none;
}
main.lit .rv-count {
  position: absolute;
  right: calc(3mm - var(--rv-edge, 0px));
  font: 600 0.8rem/1.6 'PT Sans', sans-serif;
  font-variant-numeric: tabular-nums;
  color: #c4621f;
  white-space: nowrap;
}
main.lit .rv-count.rv-same { font-weight: 400; color: #aaa; }
main.lit > .rv-count { top: 0; }
main.lit > .rv-count ~ .rv-note { top: 8mm; }
main.lit .rv-note {
  position: absolute;
  top: 0;
  right: calc(3mm - var(--rv-edge, 0px));
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  font: 0.7rem 'PT Sans', sans-serif;
  color: #999;
  white-space: nowrap;
}
"""

_JS = r"""
(() => {
  const main = document.querySelector('main.lit');
  const data = JSON.parse(document.getElementById('rv-data').textContent);
  const base = new DOMParser().parseFromString(data.base, 'text/html').body;
  const LEAF = 'p,li,h1,h2,h3,h4,h5,h6,figcaption,caption,th,td,dt,dd,pre,summary,.admonition-title,div.arithmatex';
  const ATOM = '.arithmatex,img,svg,mark.pending';
  // Skipped: link furniture, and the dark copy of a themed figure, which the print hides.
  const SKIP = 'a.anchor-link,.footnote-backref,.mini-themed-img-dark';
  // Every block element, and a figure outside one (a figure's own inner svg elements are part of it).
  const blocks = root => {
    const all = [...root.querySelectorAll(LEAF)]
      .concat([...root.querySelectorAll('img,svg')].filter(i => !i.closest(LEAF) && !i.parentElement?.closest('svg')));
    all.sort((a, b) => a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1);
    return all.filter(b => !b.closest(SKIP));
  };
  // A block's own words, each with where it sits: text split on whitespace, and an atom (math, a
  // figure) whole. A nested block's words are that block's, so a list item's are the ones before its sub-list.
  const atomText = a => a.matches('img') ? 'img:' + (a.dataset.rv ?? a.getAttribute('src')) : a.matches('svg') ? a.outerHTML : a.textContent;
  const tokens = block => {
    if (block.matches(ATOM)) return [{ t: atomText(block), atom: block }];
    const out = [];
    const walk = node => {
      for (const c of [...node.childNodes]) {
        if (c.nodeType === Node.TEXT_NODE) {
          for (const m of c.data.matchAll(/\S+/g)) out.push({ t: m[0], node: c, off: m.index });
        } else if (c.nodeType === Node.ELEMENT_NODE && !c.matches(SKIP) && !c.matches(LEAF)) {
          if (c.matches(ATOM)) out.push({ t: atomText(c), atom: c });
          else walk(c);
        }
      }
    };
    walk(block);
    return out;
  };
  const summarize = b => { const tk = tokens(b); const w = tk.map(t => t.t); return { b, tk, w, key: w.join(' ') }; };
  // The index pairs of a longest common subsequence, where eq scores a pair (0: no match).
  const lcs = (a, b, eq) => {
    const n = a.length, m = b.length, d = Array.from({ length: n + 1 }, () => new Float64Array(m + 1));
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) {
      const s = eq(a[i], b[j]);
      d[i][j] = Math.max(d[i + 1][j], d[i][j + 1], s > 0 ? s + d[i + 1][j + 1] : 0);
    }
    const pairs = [];
    for (let i = 0, j = 0; i < n && j < m;) {
      const s = eq(a[i], b[j]);
      if (s > 0 && d[i][j] === s + d[i + 1][j + 1]) { pairs.push([i, j]); i++; j++; }
      else if (d[i + 1][j] >= d[i][j + 1]) i++; else j++;
    }
    return pairs;
  };
  const same = (p, q) => p === q ? 1 : 0;
  // Two blocks are one block edited when at least a third of their words survive in order.
  // The words two blocks share bound that from above, and are cheap, so most pairs stop there.
  const bag = x => x.bag ??= x.w.reduce((m, w) => m.set(w, (m.get(w) ?? 0) + 1), new Map());
  const similar = (x, y) => {
    const n = x.w.length + y.w.length || 1;
    let shared = 0;
    for (const [w, c] of bag(x)) shared += Math.min(c, bag(y).get(w) ?? 0);
    if (2 * shared / n < 0.35) return 0;
    const r = 2 * lcs(x.w, y.w, same).length / n;
    return r >= 0.35 ? r : 0;
  };
  const H = blocks(main).map(summarize).filter(x => x.tk.length), B = blocks(base).map(summarize).filter(x => x.tk.length);

  // Which of the page's words to mark, as (block index, word index).
  const marked = new Map();
  const mark = (bi, wi) => {
    const h = H[bi]; if (!h || !h.tk.length) return;
    (marked.get(bi) ?? marked.set(bi, new Set()).get(bi)).add(Math.max(0, Math.min(wi, h.tk.length - 1)));
  };
  // A removal marks the line it leaves a gap in: the next word, or the last one before it when
  // nothing follows or the next block is a section heading (the gap is in the section that lost it).
  const removed = (bi, wi) => bi < H.length && !(wi === 0 && bi > 0 && H[bi].b.matches('h1, h2'))
    ? mark(bi, wi) : mark(Math.min(bi, H.length) - 1, Infinity);
  // Words added and removed, per block of the page (a removal counts where it would be marked), for each section's count.
  const plus = new Array(H.length).fill(0), minus = new Array(H.length).fill(0);
  const tally = (bi, p, m) => {
    let k = Math.min(bi, H.length - 1);
    if (k > 0 && !p && H[k].b.matches('h1, h2')) k--;  // a removal before a heading, as for its bar
    if (k >= 0) { plus[k] += p; minus[k] += m; }
  };
  const exact = lcs(H, B, (x, y) => x.key === y.key ? 1 : 0);
  let hi = 0, bi = 0;
  for (const [he, be] of [...exact, [H.length, B.length]]) {
    const pairs = lcs(H.slice(hi, he), B.slice(bi, be), similar).map(([x, y]) => [x + hi, y + bi]);
    let hj = hi, bj = bi;
    for (const [x, y] of [...pairs, [he, be]]) {
      // A removal with something new in its place is marked by the new; one without, by the line it leaves.
      if (bj < y && hj === x) removed(x, 0);
      for (let k = bj; k < y; k++) tally(x, 0, B[k].w.length);
      for (; hj < x; hj++) { H[hj].tk.forEach((_, k) => mark(hj, k)); tally(hj, H[hj].w.length, 0); }  // a new block
      if (x < he) {                                                    // an edited block: its words
        const words = lcs(H[x].w, B[y].w, same);
        let i = 0, j = 0;
        for (const [wi, wj] of [...words, [H[x].w.length, B[y].w.length]]) {
          if (j < wj && i === wi) removed(x, wi);
          tally(x, wi - i, wj - j);
          for (; i < wi; i++) mark(x, i);
          i = wi + 1; j = wj + 1;
        }
      }
      hj = x + 1; bj = y + 1;
    }
    hi = he + 1; bi = be + 1;
  }

  // Insert the bars, last word first within a block, so splitting a text node never moves a word still to come.
  // A figure (img or inline svg), and a table row with a change anywhere in it, get one bar as tall as they are, through anchor positioning.
  let anchors = 0;
  const bar = () => { const s = document.createElement('span'); s.className = 'rv'; return s; };
  const spanned = new Set();
  const span = box => {
    if (spanned.has(box)) return;
    spanned.add(box);
    const name = `--rv-${anchors++}`, s = bar();
    box.style.anchorName = name;
    Object.assign(s.style, { top: `anchor(${name} top)`, bottom: `anchor(${name} bottom)`, height: 'auto', marginTop: '0' });
    main.appendChild(s);
  };
  for (const [b, words] of marked) {
    const row = H[b].b.closest('tr');
    if (row) { span(row); continue; }
    for (const k of [...words].sort((p, q) => q - p)) {
      const t = H[b].tk[k];
      // The bar goes just inside the word (after its first character, or inside an inline atom),
      // where its static position is the word's line: before the word, at a line break, it is the line above.
      if (t.atom && t.atom.matches('img,svg')) span(t.atom);
      else if (t.atom) t.atom.prepend(bar());
      else { const w = t.node.splitText(t.off); w.splitText(1); w.after(bar()); }
    }
  }
  // Each section's count, at the top of its page's margin, so an unchanged page can be skipped at a glance.
  // It counts words (+ added or edited, − removed), so it tells a touched-up page from a rewritten one.
  const heads = H.map((h, i) => h.b.matches('h1, h2') ? i : -1).filter(i => i >= 0);
  heads.forEach((start, n) => {
    const from = n ? start : 0, to = heads[n + 1] ?? H.length;
    let p = 0, m = 0;
    for (let i = from; i < to; i++) { p += plus[i]; m += minus[i]; }
    const label = document.createElement('span');
    label.className = 'rv-count' + (p + m ? '' : ' rv-same');
    label.textContent = p + m ? `+${p} \u2212${m}` : 'unchanged';
    // The title's page starts at the top of the column; a section's page starts at its heading.
    if (n) H[start].b.prepend(label); else main.prepend(label);
  });
  // A bar sits on the paper's edge, across the @page rule's right margin. Chromium paints nothing in a
  // page margin, so the margin moves inside the page as padding: the column stays where it was, and
  // the strip beside it can be drawn on.
  const pages = rules => [...rules].flatMap(r => r instanceof CSSPageRule ? [r.style] : r.cssRules ? pages(r.cssRules) : []);
  const edge = [...document.styleSheets].flatMap(s => { try { return pages(s.cssRules) } catch { return [] } })
    .map(st => st.getPropertyValue('margin-right')).filter(Boolean).at(-1);
  if (edge) {
    const style = document.createElement('style');
    style.textContent = `@page { margin-right: 0 } html { padding-right: ${edge} }`;
    document.head.append(style);
    main.style.setProperty('--rv-edge', edge);
  }
  const note = document.createElement('div');
  note.className = 'rv-note';
  note.textContent = data.note;
  main.append(note);
})();
"""


def baseline_dir(sha: str, key: str) -> Path:
    """Where the baseline export of report *key* at commit *sha* is kept (``scripts/review_base.py`` writes it; the site build reads it)."""
    from mini.runs import data_root

    return data_root() / "review" / sha / key


def tag_images(html: str, root: Path) -> str:
    """Tag each ``<img>`` in *html* with a digest of its file under *root* (``data-rv``), so a redrawn figure reads as changed although its name is stable. An image that is not a local file keeps its ``src`` as its identity."""

    def tag(m: re.Match[str]) -> str:
        path = root / m[1]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else m[1]
        return m[0].replace("<img", f'<img data-rv="{digest}"', 1)

    return _IMG_RE.sub(tag, html)


def _body(html: str) -> str:
    """The content of a rendered page's ``main.lit``, or the whole of *html* when it has none (a fragment)."""
    if MAIN_OPEN not in html:
        return html
    return html.split(MAIN_OPEN, 1)[1].rsplit("</main>", 1)[0]


def stamp(html: str, *, note: str) -> str:
    """*html*, a rendered report page, with *note* on the edge of its first page: the version an unmarked print is of, so a later print can be marked against it."""
    note_html = f'<div class="rv-note">{escape(note)}</div>'
    return html.replace(MAIN_OPEN, f"{MAIN_OPEN}{note_html}", 1).replace(
        "</main>", f"</main>\n<style>{_CSS}</style>\n", 1
    )


def mark_changes(html: str, base_html: str, *, note: str, root: Path, base_root: Path) -> str:
    """*html*, a rendered report page, with a script that bars the margin beside every line that differs from *base_html* when the page loads.

    *root* and *base_root* are the directories each page's figures resolve against (:func:`tag_images`). *note* is the text of the margin note on the first page, which names the two versions. The script runs as the page is parsed, before KaTeX renders the math, so math compares as its source.
    """
    payload = json.dumps({"base": tag_images(_body(base_html), base_root), "note": note})
    inject = (
        f"<style>{_CSS}</style>\n"
        # Inside a script element a "<" can begin an early end or a comment, so JSON with every "<" escaped is inert text.
        f'<script type="application/json" id="rv-data">{payload.replace("<", "\\u003c")}</script>\n'
        f"<script>{_JS}</script>\n"
    )
    return tag_images(html, root).replace("</main>", "</main>\n" + inject, 1)
