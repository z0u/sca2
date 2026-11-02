#!/usr/bin/env python
"""Ranked keyword search over what the project has written down: report prose, design docs, the backlogs, the decision register, and the index entries.

The question this answers is "where did we find out X?", which `rg` answers poorly in two ways. It searches a report's `.py` source, where the prose sits among loaders and figure code, so a hit names a file and leaves the reading to you. And it does not rank, so a common word comes back as fifty files in path order. Here every document is cut into passages at its headings (a report's prose is pulled out of the source statically, with `{…}` fields left as written, so nothing runs), and the passages are ranked by BM25: each query word adds to a passage's score in proportion to how rare the word is across the corpus, with diminishing returns for repeats and a correction for passage length.

The corpus is a few megabytes, so the index is built on every call (well under a second) rather than stored. There is nothing to keep in step with the repo, and a fresh container needs no setup.

Two cheap aids cover most of the gap between how a question is worded and how a report worded the answer. Words are folded (Unicode NFKC, so `e₁` matches `e1`; case; a light suffix strip, so `spills` matches `spill`). And the glossary lists other wordings for each term (`docs/glossary.md`), so a query that names one wording also matches the others, at half weight.

Like `todo.py`, this uses the standard library only, so it runs before `uv sync` does.

Usage is logged so the ranking can be tuned from evidence. When `FIND_LOG_DIR` is set (`./go` sets it to the project's shared folder in a cloud project thread), each search appends one JSON line to `queries/<session>.jsonl` there: the query, the filters, and the top paths. An agent that finds an answer the search missed records it with `--miss`, which appends to `misses/<session>.jsonl` with where the answer was and what rank that place has now. One file per session keeps concurrent threads from writing to the same file. `--log-report` summarizes both for the tuning routine (`.claude/routines/find-tuning.md`), and settled misses become cases in `tests/find_cases.json`, which `tests/test_find_cases.py` checks against the real corpus. The log lives outside the repo, so it is never indexed itself.
"""

import argparse
import ast
import json
import math
import os
import re
import shutil
import signal
import sys
import textwrap
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
KINDS = ("report", "doc", "todo", "eng", "index")
K1, B = 1.2, 0.75  # the usual BM25 constants: term-frequency saturation, length normalization
ALIAS_WEIGHT = 0.5  # how much a glossary alias counts against a word the query used
SNIPPET = 320  # characters of excerpt per hit
LOGGED_HITS = 5  # top paths kept per logged query
REWORD_SECONDS = 180  # a second query from the same session this soon after a first is counted as a rewording
STOP = frozenset(
    "a an and are as at be by for from has have how in into is it its of on or that the this to was we were what when where which with".split()
)
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$", re.MULTILINE)
WORD_RE = re.compile(r"\w+")
TAG_RE = re.compile(r"<!--.*?-->|<[^>]+>", re.DOTALL)
MARKUP_RE = re.compile(r"<!--.*?-->|<[^>]+>|^///.*$", re.DOTALL | re.MULTILINE)  # tags, comments, and admonition fences


def plain(text: str) -> str:
    """*text* as one line for display: markup dropped and whitespace collapsed."""
    return " ".join(MARKUP_RE.sub(" ", re.sub(r"^/// \w+ \|", "", text, flags=re.MULTILINE)).split())


@dataclass(frozen=True, slots=True)
class Passage:
    """One section of one document: where it is, what it is under, and its text."""

    path: str  # repo-relative
    line: int
    kind: str
    title: str  # the document's
    heading: str  # the trail of headings above the passage, " › "-joined; empty for a lead-in
    text: str
    status: str = ""  # a backlog item's

    @property
    def where(self) -> str:
        return " › ".join(p for p in (self.title, self.heading) if p)


# --- Words ----------------------------------------------------------------------------------------------------


def stem(word: str) -> str:
    """A light suffix strip, enough to join plurals and simple verb forms without merging distinct words."""
    if len(word) < 5 or word.endswith("ss"):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    for suffix in ("ing", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def words(text: str) -> list[str]:
    """The index terms of *text*, in order: folded, stemmed, stop words dropped."""
    folded = unicodedata.normalize("NFKC", text).lower()
    return [stem(w) for w in WORD_RE.findall(folded) if w not in STOP]


# --- Reading the corpus ---------------------------------------------------------------------------------------


def sections(text: str, title: str) -> Iterator[tuple[int, str, str]]:
    """Cut Markdown at its headings: (offset of the heading, heading trail, body) for each section that has a body. A heading that names the document again (its `# Title`) is dropped from the trail."""
    trail: list[tuple[int, str]] = []
    starts = [(m.start(), m.end(), len(m.group(1)), m.group(2)) for m in HEADING_RE.finditer(text)]
    bounds = [(0, 0, 0, "")] + starts
    for i, (begin, end, level, name) in enumerate(bounds):
        if level:
            trail = [t for t in trail if t[0] < level]
            if not (level == 1 and name.strip().lower() == title.strip().lower()):
                trail.append((level, name.strip()))
        body = text[end : starts[i][0] if i < len(starts) else len(text)].strip()
        if body:
            yield begin, " › ".join(n for _, n in trail), body


def field_text(part: ast.expr) -> str:
    """One piece of an f-string: literal text as is, and a field as its source in braces."""
    match part:
        case ast.Constant(value=str(text)):
            return text
        case ast.FormattedValue(value=expr):
            return "{" + ast.unparse(expr) + "}"
    return ""


def prose_of(source: str) -> Iterator[tuple[int, str]]:
    """The prose of a literate script: each top-level string, with an f-string's fields written back as `{expr}`."""
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Expr):
            continue
        match node.value:
            case ast.Constant(value=str(s)):
                yield node.lineno, s
            case ast.JoinedStr(values=parts):
                yield node.lineno, "".join(field_text(p) for p in parts)


def report_passages(path: Path, rel: str) -> Iterator[Passage]:
    source = path.read_text()
    if not (m := re.search(r"^# title:\s*(.+)$", source, re.MULTILINE)):
        return
    title = m.group(1).strip()
    try:
        pieces = list(prose_of(source))
    except SyntaxError:
        return
    # A heading opens a section that runs on through later strings until the next heading, so join the strings first and remember where each began.
    joined, origins = "", []
    for lineno, text in pieces:
        origins.append((len(joined), lineno + len(text) - len(text.lstrip("\n"))))
        joined += text.strip("\n") + "\n\n"

    def line_of(offset: int) -> int:
        start, lineno = max((o for o in origins if o[0] <= offset), default=(0, 1))
        return lineno + joined.count("\n", start, offset)

    for offset, heading, body in sections(joined, title):
        yield Passage(rel, line_of(offset), "report", title, heading, body)


def front_matter(text: str) -> tuple[dict[str, str], str, int]:
    """A backlog item's header as raw strings, the text after it, and the line the text starts on."""
    if not text.startswith("---\n") or (end := text.find("\n---\n", 4)) < 0:
        return {}, text, 1
    head = dict(re.findall(r"^(\w+):\s*(.*?)\s*$", text[4:end], re.MULTILINE))
    return head, text[end + 5 :], text.count("\n", 0, end + 5) + 1


def markdown_passages(path: Path, rel: str, kind: str) -> Iterator[Passage]:
    head, text, first = front_matter(path.read_text())
    title = m.group(1).strip() if (m := re.search(r"^#\s+(.+)$", text, re.MULTILINE)) else path.stem
    for offset, heading, body in sections(text, title):
        yield Passage(rel, first + text.count("\n", 0, offset), kind, title, heading, body, head.get("status", ""))


def index_passages(path: Path, rel: str) -> Iterator[Passage]:
    """`docs/index.md` one entry at a time, each pointing at the report it summarizes."""
    text = path.read_text()
    for m in re.finditer(r"^- \[(.+?)\]\(\./(.+?)\)\n(.*?)(?=^- \[|^#|\Z)", text, re.MULTILINE | re.DOTALL):
        body = re.sub(r"<!--.*?-->", "", m.group(3)).strip()
        yield Passage(
            f"docs/{m.group(2)}", text.count("\n", 0, m.start()) + 1, "index", m.group(1), "index entry", body
        )


def corpus(root: Path) -> Iterator[Passage]:
    """Every passage under *root*, in a stable order."""

    def files(base: str, pattern: str) -> Iterable[Path]:
        return sorted((root / base).rglob(pattern)) if (root / base).is_dir() else ()

    for path in files("docs", "*.py"):
        yield from report_passages(path, path.relative_to(root).as_posix())
    for path in files("docs", "*.md"):
        rel = path.relative_to(root).as_posix()
        if rel == "docs/index.md":
            yield from index_passages(path, rel)
        elif path.name != "README.md":
            yield from markdown_passages(path, rel, "doc")
    for path in files("todo", "*.md"):
        if path.name != "README.md":
            yield from markdown_passages(path, path.relative_to(root).as_posix(), "todo")
    for path in files("eng", "*.md"):
        yield from markdown_passages(path, path.relative_to(root).as_posix(), "eng")


# --- The glossary ---------------------------------------------------------------------------------------------


def glossary_forms(text: str) -> list[tuple[str, ...]]:
    """Each glossary entry as its wordings: the term lines above a `:   ` definition, with comma- and paren-separated parts split out as `dl_terms` does."""
    groups: list[tuple[str, ...]] = []
    for block in re.split(r"\n\s*\n", TAG_RE.sub("", text)):
        lines = block.strip().splitlines()
        if not any(line.startswith(":") for line in lines):
            continue
        names = [re.sub(r"\{[^}]*\}", "", n).strip() for n in lines if not n.startswith(":") and not n.startswith("#")]
        forms = {f.strip() for n in names for f in [n, *re.split(r",|\(|\)", n)] if f.strip()}
        if len(forms) > 1:
            groups.append(tuple(sorted(forms)))
    return groups


def expand(query: str, groups: list[tuple[str, ...]]) -> dict[str, float]:
    """The query as weighted terms: its own words at weight 1, and the words of any glossary wording it names at `ALIAS_WEIGHT`."""
    own = words(query)
    weights = dict.fromkeys(own, 1.0)
    padded = f" {' '.join(own)} "
    for forms in groups:
        if any((w := words(f)) and f" {' '.join(w)} " in padded for f in forms):
            for f in forms:
                for w in words(f):
                    weights.setdefault(w, ALIAS_WEIGHT)
    return weights


# --- Ranking --------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class Index:
    """BM25 over passages. Each passage is indexed with its heading trail, so a section titled for a topic counts for it."""

    passages: list[Passage]
    terms: list[Counter[str]] = field(init=False)
    df: Counter[str] = field(init=False)
    avg: float = field(init=False)

    def __post_init__(self) -> None:
        self.terms = [Counter(words(f"{p.where}\n{p.text}")) for p in self.passages]
        self.df = Counter(t for c in self.terms for t in c)
        self.avg = sum(c.total() for c in self.terms) / max(len(self.terms), 1)

    def idf(self, term: str) -> float:
        n, df = len(self.passages), self.df[term]
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, i: int, query: dict[str, float]) -> float:
        c = self.terms[i]
        norm = K1 * (1 - B + B * c.total() / self.avg)
        return sum(w * self.idf(t) * c[t] * (K1 + 1) / (c[t] + norm) for t, w in query.items() if c[t])

    def search(self, query: dict[str, float], limit: int, per_file: int) -> list[tuple[float, Passage]]:
        scored = sorted(
            ((s, p) for i, p in enumerate(self.passages) if (s := self.score(i, query)) > 0), key=lambda sp: -sp[0]
        )
        seen: Counter[str] = Counter()
        hits = []
        for s, p in scored:
            if seen[p.path] < per_file:
                seen[p.path] += 1
                hits.append((s, p))
            if len(hits) == limit:
                break
        return hits


def snippet(text: str, query: dict[str, float], width: int = SNIPPET) -> str:
    """The window of *text* that holds the most query weight, as one line of plain text."""
    flat = plain(text)
    if len(flat) <= width:
        return flat
    spans = [(m.start(), stem(unicodedata.normalize("NFKC", m.group()).lower())) for m in WORD_RE.finditer(flat)]
    best, best_at = -1.0, 0
    for i, (start, w) in enumerate(spans):
        if w not in query:
            continue  # a window opens on a query word, so the excerpt leads into it
        got = {w for s, w in spans[i:] if s < start + width and w in query}
        if (value := sum(query[w] for w in got)) > best:
            best, best_at = value, start
    start = max(0, min(best_at - width // 6, len(flat) - width))
    start = flat.rfind(" ", 0, start) + 1 if start else 0
    end = flat.rfind(" ", start, start + width)
    return (
        ("…" if start else "") + flat[start : end if end > start else start + width] + ("…" if end < len(flat) else "")
    )


# --- The usage log ---------------------------------------------------------------------------------------------


def session_name() -> str:
    """A stable name for this session's log files: the Claude Code session id where there is one."""
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID") or os.environ.get("CLAUDE_CODE_REMOTE_SESSION_ID")
    return re.sub(r"[^\w-]", "_", sid) if sid else f"pid-{os.getpid()}"


def append_log(log_dir: Path | None, kind: str, record: dict) -> None:
    """Append *record* as one JSON line to `<log_dir>/<kind>/<session>.jsonl`. Logging never fails a search."""
    if log_dir is None:
        return
    try:
        (folder := log_dir / kind).mkdir(parents=True, exist_ok=True)
        stamped = {"time": datetime.now(UTC).isoformat(timespec="seconds"), "session": session_name(), **record}
        with open(folder / f"{session_name()}.jsonl", "a") as f:
            f.write(json.dumps(stamped, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"find: could not write the usage log ({e})", file=sys.stderr)


def read_log(log_dir: Path, kind: str, since: str = "") -> list[dict]:
    """Every record of *kind* at or after *since* (an ISO date or time), oldest first."""
    records = []
    for path in sorted((log_dir / kind).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn line from an interrupted write
            if record.get("time", "") >= since:
                records.append(record)
    return sorted(records, key=lambda r: r.get("time", ""))


def rank_of(hits: list[tuple[float, Passage]], target: str) -> int | None:
    """Where *target* (`path` or `path:line`) first appears in *hits*, counting from 1."""
    path, _, line = target.partition(":")
    for i, (_, p) in enumerate(hits, 1):
        if p.path == path and (not line or p.line <= int(line) < p.line + p.text.count("\n") + 3):
            return i
    return None


def log_report(log_dir: Path, since: str, rank: Callable[[str, str], int | None]) -> str:
    """A plain-text summary of the log: volume, empty results, quick rewordings, and each recorded miss with its rank today."""
    queries, misses = read_log(log_dir, "queries", since), read_log(log_dir, "misses", since)
    out = [f"{len(queries)} searches from {len({q['session'] for q in queries})} sessions since {since or 'the start'}"]
    if empty := [q for q in queries if not q.get("top")]:
        out += ["", "No results:", *(f"  {q['query']!r}" for q in empty)]
    rewordings = []
    for a, b in pairwise(sorted(queries, key=lambda q: (q["session"], q["time"]))):
        gap = datetime.fromisoformat(b["time"]) - datetime.fromisoformat(a["time"])
        if a["session"] == b["session"] and gap.total_seconds() <= REWORD_SECONDS and a["query"] != b["query"]:
            rewordings.append(f"  {a['query']!r} → {b['query']!r}")
    if rewordings:
        out += ["", "Reworded within a few minutes (the first may have missed):", *rewordings]
    if misses:
        out += ["", "Recorded misses (rank of the answer in today's ranking, top 20):"]
        for m in misses:
            now = rank(m["query"], m["found"])
            note = f" · {m['note']}" if m.get("note") else ""
            out.append(f"  {m['query']!r} → {m['found']}: was {m.get('rank') or '>20'}, now {now or '>20'}{note}")
    return "\n".join(out)


# --- CLI ------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None, root: Path = ROOT, log_dir: Path | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ranked keyword search (BM25) over report prose, design docs, backlogs, the decision register and the index."
    )
    ap.add_argument("query", nargs="*", help="words to look for; order does not matter")
    ap.add_argument("-n", type=int, default=10, help="hits to show (default 10)")
    ap.add_argument(
        "--kind", action="append", choices=KINDS, help="only these kinds of document (repeatable; default all)"
    )
    ap.add_argument(
        "--in",
        dest="under",
        action="append",
        metavar="PATH",
        help="only documents under this repo-relative path (repeatable)",
    )
    ap.add_argument("--per-file", type=int, default=2, help="most passages shown from one file (default 2)")
    ap.add_argument("--full", action="store_true", help="print each passage whole, wrapped to the terminal width")
    ap.add_argument("--no-glossary", action="store_true", help="match only the words given, without glossary wordings")
    ap.add_argument("--json", action="store_true", help="emit the hits as JSON")
    ap.add_argument(
        "--miss",
        metavar="PATH[:LINE]",
        help="record that the query missed: the answer was at PATH (the query words are the search that missed)",
    )
    ap.add_argument(
        "--note", default="", help="with --miss: what the answer was worded as, or why the search missed it"
    )
    ap.add_argument("--log-report", action="store_true", help="summarize the usage log and the recorded misses")
    ap.add_argument("--since", default="", metavar="DATE", help="with --log-report: only records from this date on")
    args = ap.parse_args(argv)

    glossary = root / "docs" / "glossary.md"
    groups = [] if args.no_glossary or not glossary.exists() else glossary_forms(glossary.read_text())
    passages = [
        p
        for p in corpus(root)
        if (not args.kind or p.kind in args.kind)
        and (not args.under or any(p.path.startswith(u.rstrip("/")) for u in args.under))
    ]
    index = Index(passages)

    def rank(words: str, target: str) -> int | None:
        return rank_of(index.search(expand(words, groups), 20, 20), target)

    if args.log_report:
        if log_dir is None:
            print("find: no usage log here (FIND_LOG_DIR is not set)", file=sys.stderr)
            return 2
        print(log_report(log_dir, args.since, rank))
        return 0
    if not args.query:
        ap.error("a query is needed")
    query = expand(" ".join(args.query), groups)
    if not query:
        print("find: the query has only stop words", file=sys.stderr)
        return 2
    if args.miss:
        found = rank(" ".join(args.query), args.miss)
        append_log(
            log_dir, "misses", {"query": " ".join(args.query), "found": args.miss, "rank": found, "note": args.note}
        )
        where = "not in the top 20" if found is None else f"at rank {found}"
        print(
            f"recorded: {args.miss} is {where} for {' '.join(args.query)!r}"
            + ("" if log_dir else " (no log here, so not saved)")
        )
        return 0

    hits = index.search(query, args.n, args.per_file)
    append_log(
        log_dir,
        "queries",
        {
            "query": " ".join(args.query),
            "kind": args.kind or [],
            "in": args.under or [],
            "top": [f"{p.path}:{p.line}" for _, p in hits[:LOGGED_HITS]],
        },
    )
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "score": round(s, 3),
                        "path": p.path,
                        "line": p.line,
                        "kind": p.kind,
                        "where": p.where,
                        "status": p.status,
                        "text": p.text,
                    }
                    for s, p in hits
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if not hits:
        print("no matches", file=sys.stderr)
        return 1
    cols = shutil.get_terminal_size((120, 24)).columns
    for s, p in hits:
        status = f" ({p.status})" if p.status else ""
        print(f"{p.path}:{p.line}  [{p.kind}{status}]  {s:.1f}")
        print(f"  {p.where}")
        body = plain(p.text) if args.full else snippet(p.text, query)
        print(textwrap.fill(body, max(cols, 40), initial_indent="    ", subsequent_indent="    "))
        print()
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `| head` ends the listing quietly
    logs = os.environ.get("FIND_LOG_DIR")
    sys.exit(main(log_dir=Path(logs) if logs and not os.environ.get("FIND_NO_LOG") else None))
