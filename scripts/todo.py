#!/usr/bin/env python
"""The engineering backlog, one item per file — this is the index over it.

`todo/eng/` holds one Markdown file per item, each with a small front-matter header. The shape is deliberate: two branches adding items add two files, so the append-conflict that a single shared list generates every week doesn't arise, and a match from `rg` carries its own title and boundary instead of landing mid-item in a file thousands of lines long.

The cost of that split is the overview, which a single file gave away for free. This script buys it back — and prints it rather than writing it down, so there is no committed index to fall out of step with the files. Nothing here caches, and nothing writes.

Front matter is a fixed six-key subset of YAML (scalars and inline lists), parsed here rather than with a library: `pyyaml` is only a transitive dependency, and this tool should keep running before `uv sync` does. `--check` is the gate that keeps the schema honest, and it runs as part of `./go check --lint`.

    ---
    status: open          # open | partial | done
    tags: [cli, storage]
    opened: 2026-08-12    # optional — some inherited items carry no date
    closed: 2026-08-12    # required when status is done
    bundle: cli-devx      # optional — groups items a single session should take together
    priority: high        # optional — the shortlist, capped so it stays a shortlist
    ---
    # A title, as the first heading

    The body, as ordinary prose.

Done items stay where they are rather than moving to an archive directory: the default view already filters them out, and leaving them put keeps their inbound links and their file history intact.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
TODO_DIR = ROOT / "todo"
STATUSES = ("open", "partial", "done", "finding")
LIVE = ("open", "partial")  # what the default view shows
MARKS = {"open": " ", "partial": "~", "done": "x", "finding": "•"}
PRIORITIES = ("high",)  # absence is the default, so one level is all the vocabulary needed
BUDGET = 6  # live `priority: high` items allowed at once — see `over_budget`
KEYS = ("status", "tags", "opened", "closed", "bundle", "priority")
FENCE = "---"


class TodoError(Exception):
    """A malformed item. Carries the path and line so `--check` names what to fix."""


@dataclass(frozen=True, slots=True)
class Item:
    """One backlog item, as parsed from its file."""

    path: Path
    title: str
    status: str
    tags: tuple[str, ...]
    body: str
    opened: date | None = None
    closed: date | None = None
    bundle: str | None = None
    priority: str | None = None

    @property
    def slug(self) -> str:
        return self.path.stem

    @property
    def set(self) -> str:
        """Which backlog this belongs to — the directory under `todo/`."""
        return self.path.parent.name

    @property
    def rel(self) -> str:
        """Repo-relative where it can be, and as-given otherwise — a tmp tree in tests is neither under the repo nor broken."""
        try:
            return self.path.relative_to(ROOT).as_posix()
        except ValueError:
            return self.path.as_posix()

    def as_dict(self) -> dict:
        """The item as JSON-ready data, dates as ISO strings and the body left out."""
        return {
            "slug": self.slug,
            "set": self.set,
            "path": self.rel,
            "title": self.title,
            "status": self.status,
            "tags": list(self.tags),
            "opened": self.opened.isoformat() if self.opened else None,
            "closed": self.closed.isoformat() if self.closed else None,
            "bundle": self.bundle,
            "priority": self.priority,
        }


def _front_matter(text: str, path: Path) -> tuple[dict[str, str | list[str]], str, int]:
    """Split *text* into its front-matter mapping and the body that follows it.

    Also returns the line the body starts on, so a complaint about the title can point at a real line.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FENCE:
        raise TodoError(f"{path}:1: expected a '{FENCE}' front-matter fence on the first line")
    try:
        end = next(n for n, line in enumerate(lines[1:], start=1) if line.strip() == FENCE)
    except StopIteration:
        raise TodoError(f"{path}: front matter is never closed — no second '{FENCE}'") from None

    fields: dict[str, str | list[str]] = {}
    for n, line in enumerate(lines[1:end], start=2):
        if not line.strip():
            continue
        key, sep, raw = line.partition(":")
        key, raw = key.strip(), raw.strip()
        if not sep:
            raise TodoError(f"{path}:{n}: expected 'key: value', got {line.strip()!r}")
        if key in fields:
            raise TodoError(f"{path}:{n}: duplicate key {key!r}")
        if key not in KEYS:
            raise TodoError(f"{path}:{n}: unknown key {key!r} — known keys are {', '.join(KEYS)}")
        fields[key] = (
            [v.strip() for v in raw[1:-1].split(",") if v.strip()] if raw.startswith("[") and raw.endswith("]") else raw
        )
    return fields, "\n".join(lines[end + 1 :]).strip(), end + 2


def _as_date(value: str | list[str], key: str, path: Path) -> date:
    if not isinstance(value, str):
        raise TodoError(f"{path}: {key!r} should be a single YYYY-MM-DD date, not a list")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise TodoError(f"{path}: {key!r} is {value!r} — expected YYYY-MM-DD") from None


def parse(path: Path) -> Item:
    """Read one item file, validating its header. Raises :class:`TodoError` with a path and line."""
    fields, body, body_line = _front_matter(path.read_text(), path)

    if (status := fields.get("status")) is None:
        raise TodoError(f"{path}: no 'status' — one of {', '.join(STATUSES)}")
    if not isinstance(status, str) or status not in STATUSES:
        raise TodoError(f"{path}: status is {status!r} — expected one of {', '.join(STATUSES)}")
    tags = fields.get("tags", [])
    if not isinstance(tags, list):
        raise TodoError(f"{path}: 'tags' should be an inline list, e.g. tags: [cli, storage]")

    opened = _as_date(o, "opened", path) if (o := fields.get("opened")) else None
    closed = _as_date(c, "closed", path) if (c := fields.get("closed")) else None
    if status != "done" and closed is not None:
        raise TodoError(f"{path}: has a 'closed' date but status is {status!r}")
    if opened and closed and closed < opened:
        raise TodoError(f"{path}: closed ({closed}) is before opened ({opened})")

    first = body.splitlines()[0] if body else ""
    if not first.startswith("# "):
        raise TodoError(f"{path}:{body_line}: the body should open with a '# Title' heading")

    bundle = fields.get("bundle") or None
    if bundle is not None and not isinstance(bundle, str):
        raise TodoError(f"{path}: 'bundle' should be a single name, not a list")

    priority = fields.get("priority") or None
    if priority is not None and (not isinstance(priority, str) or priority not in PRIORITIES):
        raise TodoError(f"{path}: priority is {priority!r} — expected one of {', '.join(PRIORITIES)}")

    return Item(
        path=path,
        title=first.removeprefix("# ").strip(),
        status=status,
        tags=tuple(tags),
        body="\n".join(body.splitlines()[1:]).strip(),
        opened=opened,
        closed=closed,
        bundle=bundle,
        priority=priority,
    )


def load(root: Path = TODO_DIR) -> tuple[list[Item], list[TodoError]]:
    """Every item under *root*, plus the errors from the ones that wouldn't parse.

    Parsing continues past a broken file so `--check` reports all of them in one pass, which is the difference between one fix-up round and several.
    """
    items, errors = [], []
    for path in sorted(root.rglob("*.md")):
        # A symlink is a second name for a file already scanned — `todo/CLAUDE.md` and
        # `todo/AGENTS.md` both point at the README, so agents pick the conventions up
        # under whichever name they look for.
        if path.name == "README.md" or path.is_symlink():
            continue
        try:
            items.append(parse(path))
        except TodoError as e:
            errors.append(e)
    return items, errors


def _sets(root: Path = TODO_DIR) -> list[str]:
    """The backlogs that exist — every directory under `todo/`, so adding one needs no code change."""
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def shortlist(items: list[Item]) -> list[Item]:
    """The live items marked `priority: high` — the answer to "what should we do next".

    Settled items and findings are left out even if they still carry the key: an old priority on a closed item is history rather than a claim on attention, and dropping it on the way out would be one more thing to remember.
    """
    return [it for it in items if it.priority == "high" and it.status in LIVE]


def over_budget(items: list[Item]) -> list[TodoError]:
    """The shortlist against its cap, checked across the backlog rather than one file at a time.

    The cap is what keeps the label worth reading. Priority schemes decay when promotion is free — everything drifts upward until the top rung holds the whole backlog and means nothing. Here a seventh item costs a demotion, which is the judgement the ranking exists to record.
    """
    if len(high := shortlist(items)) <= BUDGET:
        return []
    listing = "\n".join(f"    {it.rel}" for it in high)
    return [
        TodoError(
            f"{len(high)} live items are 'priority: high', over the budget of {BUDGET} — demote one before promoting another:\n{listing}"
        )
    ]


def select(
    items: list[Item],
    tags: list[str] | None = None,
    status: str | None = None,
    bundle: str | None = None,
    sets: list[str] | None = None,
    priority: str | None = None,
) -> list[Item]:
    """The items matching every filter given, shortlisted work first, then newest first and undated last.

    Tags are conjunctive — `--tag cli --tag storage` is the intersection, which is the useful direction when narrowing a backlog. Without a status filter, only live work shows: settled items and findings are still there, and `--status` reaches them.
    """
    wanted = set(tags or ())
    keep = [
        it
        for it in items
        if (wanted <= set(it.tags))
        and (it.status == status if status else it.status in LIVE)
        and (it.bundle == bundle if bundle else True)
        and (it.set in sets if sets else True)
        and (it.priority == priority if priority else True)
    ]
    return sorted(
        keep,
        key=lambda it: (
            it.priority != "high",
            it.opened is None,
            -it.opened.toordinal() if it.opened else 0,
            it.slug,
        ),
    )


def render(items: list[Item]) -> str:
    """The items as a grouped, aligned listing — by set, then by bundle, with unbundled last in each.

    A shortlisted item carries a `!` beside its status mark and sorts to the head of its group, so the ranking is visible in the ordinary listing without needing `--priority` to find it.
    """
    if not items:
        return "(nothing matches)"
    groups: dict[tuple[str, str], list[Item]] = {}
    for it in items:
        groups.setdefault((it.set, it.bundle or ""), []).append(it)
    width = max(len(it.rel) for it in items)

    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[1] == "", k[1])):
        head = f"{key[0]} · {key[1]}" if key[1] else key[0]
        out.append(f"\n{head}:")
        for it in groups[key]:
            tags = f"  [{' '.join(it.tags)}]" if it.tags else ""
            flag = "!" if it.priority == "high" else " "
            out.append(f"  [{MARKS[it.status]}]{flag} {it.rel:<{width}}  {it.title}{tags}")
    return "\n".join(out).lstrip("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="List backlog items.")
    ap.add_argument(
        "sets", nargs="*", metavar="SET", help=f"which backlogs to list ({', '.join(_sets())}); default: all"
    )
    ap.add_argument(
        "--tag",
        action="append",
        dest="tags",
        metavar="TAG",
        help="only items carrying this tag (repeatable, conjunctive)",
    )
    ap.add_argument("--status", choices=STATUSES, help="only items with this status (default: everything but done)")
    ap.add_argument("--bundle", help="only items in this bundle")
    ap.add_argument(
        "--priority",
        nargs="?",
        const="high",
        choices=PRIORITIES,
        help=f"only shortlisted items (at most {BUDGET} are live at a time)",
    )
    ap.add_argument("--json", action="store_true", help="emit the selection as JSON")
    ap.add_argument("--check", action="store_true", help="validate every item's header and exit non-zero on a problem")
    args = ap.parse_args()

    if unknown := [s for s in args.sets if s not in _sets()]:
        ap.error(f"unknown backlog {unknown[0]!r} — expected one of {', '.join(_sets())}")

    items, errors = load()
    if args.check:
        problems = [*errors, *over_budget(items)]
        for e in problems:
            print(e, file=sys.stderr)
        ok = f"✅ {len(items)} todo items parse, {len(shortlist(items))}/{BUDGET} priority slots used"
        print(
            f"❌ {len(problems)} problem(s) in the backlog" if problems else ok,
            file=sys.stderr if problems else sys.stdout,
        )
        sys.exit(1 if problems else 0)
    if errors:
        print(f"warning: skipped {len(errors)} malformed item(s); run --check to see them", file=sys.stderr)

    chosen = select(
        items, tags=args.tags, status=args.status, bundle=args.bundle, sets=args.sets, priority=args.priority
    )
    print(json.dumps([it.as_dict() for it in chosen], indent=1) if args.json else render(chosen))


if __name__ == "__main__":
    main()
