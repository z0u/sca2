"""Tests for the backlog index — header validation, and the query that replaces a committed list."""

from pathlib import Path

import pytest

from tests.conftest import load_script

todo = load_script("todo")

VALID = """---
status: open
tags: [cli, storage]
opened: 2026-08-13
---
# A title

The body.
"""


def write(root: Path, name: str, text: str) -> Path:
    (path := root / f"{name}.md").write_text(text)
    return path


def item(root: Path, name: str = "x", **fields) -> Path:
    """An item with *fields* overriding the valid header, and `None` dropping a key."""
    head = {"status": "open", "tags": "[cli]", "opened": "2026-08-13"} | fields
    front = "\n".join(f"{k}: {v}" for k, v in head.items() if v is not None)
    root.mkdir(parents=True, exist_ok=True)
    return write(root, name, f"---\n{front}\n---\n# {name}\n\nBody.\n")


@pytest.fixture
def backlog(tmp_path: Path) -> Path:
    """Two sets: four eng items (two bundled, one done, one undated) and one science finding."""
    eng = tmp_path / "eng"
    item(eng, "newer", tags="[cli, storage]", opened="2026-08-12", bundle="cli-devx")
    item(eng, "older", tags="[cli]", opened="2026-07-01", bundle="cli-devx")
    item(eng, "shipped", tags="[storage]", status="done", opened="2026-07-02", closed="2026-07-03")
    item(eng, "undated", tags="[vis]", opened=None)
    item(tmp_path / "science", "learned", tags="[anchoring]", status="finding", opened="2026-07-05")
    (eng / "README.md").write_text("# Not an item\n")
    return tmp_path


def slugs(items) -> list[str]:
    return [it.slug for it in items]


# --- parsing -------------------------------------------------------------


def test_a_valid_item_parses(tmp_path):
    """The title comes off the heading, the slug off the filename — so the two can disagree."""
    it = todo.parse(write(tmp_path, "some-slug", VALID))
    assert (it.title, it.slug, it.status, it.tags) == ("A title", "some-slug", "open", ("cli", "storage"))
    assert it.body == "The body."
    assert it.opened.isoformat() == "2026-08-13"


@pytest.mark.parametrize(
    "text",
    [
        "# No front matter\n",
        "---\nstatus: open\n# never closed\n",
        "---\nstatus: open\ntags: [cli]\n---\nNo heading here.\n",
    ],
    ids=["no-fence", "unclosed", "no-title"],
)
def test_structural_problems_are_rejected(tmp_path, text):
    with pytest.raises(todo.TodoError):
        todo.parse(write(tmp_path, "x", text))


@pytest.mark.parametrize(
    "fields",
    [
        {"status": None},
        {"status": "wip"},
        {"tags": "cli"},
        {"opened": "13-08-2026"},
        {"bundle": "[a, b]"},
        {"priority": "urgent"},
        {"priority": "[high]"},
    ],
    ids=[
        "no-status",
        "bad-status",
        "tags-not-a-list",
        "bad-date",
        "bundle-is-a-list",
        "bad-priority",
        "priority-is-a-list",
    ],
)
def test_bad_fields_are_rejected(tmp_path, fields):
    with pytest.raises(todo.TodoError):
        todo.parse(item(tmp_path, **fields))


def test_an_unknown_key_is_rejected_rather_than_ignored(tmp_path):
    with pytest.raises(todo.TodoError, match="unknown key"):
        todo.parse(item(tmp_path, owner="alex"))


def test_a_duplicate_key_is_rejected(tmp_path):
    with pytest.raises(todo.TodoError, match="duplicate"):
        todo.parse(write(tmp_path, "x", "---\nstatus: open\nstatus: done\ntags: [cli]\n---\n# T\n"))


def test_optional_fields_fall_back_to_their_defaults(tmp_path):
    it = todo.parse(item(tmp_path, tags=None, opened=None, priority=None))
    assert (it.tags, it.opened, it.priority) == ((), None, None)
    assert todo.parse(item(tmp_path, "p", priority="high")).priority == "high"


def test_a_closed_date_is_optional_on_done(tmp_path):
    """Migrated items often record that something shipped without recording when."""
    assert todo.parse(item(tmp_path, status="done")).closed is None


def test_only_done_carries_a_closed_date(tmp_path):
    with pytest.raises(todo.TodoError, match="closed"):
        todo.parse(item(tmp_path, status="open", closed="2026-08-13"))


def test_closing_before_opening_is_rejected(tmp_path):
    with pytest.raises(todo.TodoError, match="before opened"):
        todo.parse(item(tmp_path, status="done", opened="2026-08-13", closed="2026-08-01"))


def test_an_error_names_the_file(tmp_path):
    with pytest.raises(todo.TodoError, match="broken.md"):
        todo.parse(item(tmp_path, "broken", status="nonsense"))


# --- loading -------------------------------------------------------------


def test_load_skips_the_readme(backlog):
    items, errors = todo.load(backlog)
    assert not errors
    assert "README" not in slugs(items)


def test_one_broken_item_does_not_hide_the_rest(backlog):
    write(backlog / "eng", "broken", "not an item at all\n")
    write(backlog / "eng", "also-broken", "---\nstatus: nope\ntags: [x]\n---\n# T\n")
    items, errors = todo.load(backlog)
    assert len(errors) == 2
    assert len(items) == 5


# --- selecting -----------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ({}, ["newer", "older", "undated"]),
        ({"status": "done"}, ["shipped"]),
        ({"status": "finding"}, ["learned"]),
        ({"tags": ["cli"]}, ["newer", "older"]),
        ({"tags": ["cli", "storage"]}, ["newer"]),
        ({"tags": ["cli", "vis"]}, []),
        ({"bundle": "cli-devx"}, ["newer", "older"]),
        ({"sets": ["science"], "status": "finding"}, ["learned"]),
        ({"sets": ["eng"], "status": "finding"}, []),
    ],
    ids=[
        "default-hides-settled-work-and-findings",  # newest first, undated last
        "status",
        "a-finding-is-a-status",
        "one-tag",
        "tags-are-conjunctive",
        "conjunction-with-no-members",
        "bundle",
        "a-set-narrows-the-selection",
        "the-set-wins-over-the-status",
    ],
)
def test_select_filters_and_orders(backlog, query, expected):
    items, _ = todo.load(backlog)
    assert slugs(todo.select(items, **query)) == expected


def test_sets_are_the_directories_that_exist(backlog):
    assert todo._sets(backlog) == ["eng", "science"]


# --- the shortlist -------------------------------------------------------


def test_the_shortlist_holds_live_items_only(backlog):
    """A settled item keeping its old priority is history, so it shouldn't answer "what next"."""
    item(backlog / "eng", "urgent", priority="high")
    item(backlog / "eng", "was-urgent", status="done", closed="2026-07-04", priority="high")
    items, _ = todo.load(backlog)
    assert slugs(todo.shortlist(items)) == ["urgent"]
    assert slugs(todo.select(items, priority="high")) == ["urgent"]


def test_shortlisted_work_outranks_recency(backlog):
    item(backlog / "eng", "urgent", opened="2000-01-01", priority="high")
    items, _ = todo.load(backlog)
    assert slugs(todo.select(items))[0] == "urgent"


def test_the_budget_holds_until_it_is_exceeded(backlog):
    items, _ = todo.load(backlog)
    assert todo.over_budget(items) == []
    for n in range(todo.BUDGET + 1):
        item(backlog / "eng", f"was-urgent-{n}", status="done", closed="2026-07-04", priority="high")
    items, _ = todo.load(backlog)
    assert todo.over_budget(items) == [], "settled items keeping an old priority don't spend the budget"
    for n in range(todo.BUDGET + 1):
        item(backlog / "eng", f"urgent-{n}", priority="high")
    items, _ = todo.load(backlog)
    assert len(todo.over_budget(items)) == 1


# --- rendering -----------------------------------------------------------


def test_bundles_group_together_and_unbundled_comes_last(backlog):
    items, _ = todo.load(backlog)
    out = todo.render(todo.select(items))
    assert out.index("eng \u00b7 cli-devx:") < out.index("\neng:")
    assert out.index("newer.md") < out.index("undated.md")


def test_render_says_so_when_nothing_matches():
    assert todo.render([]) == "(nothing matches)"


def test_status_shows_in_the_listing(backlog):
    items, _ = todo.load(backlog)
    assert "[x] " in todo.render(todo.select(items, status="done"))
    assert "[\u2022] " in todo.render(todo.select(items, status="finding"))


def test_the_shortlist_is_marked_in_the_listing(backlog):
    item(backlog / "eng", "urgent", priority="high")
    items, _ = todo.load(backlog)
    out = todo.render(todo.select(items))
    assert "[ ]! " in out
    assert "[ ]  " in out


def test_json_shape_is_serializable(backlog):
    import json

    items, _ = todo.load(backlog)
    data = json.loads(json.dumps([it.as_dict() for it in todo.select(items)]))
    assert data[0]["slug"] == "newer"
    assert data[0]["tags"] == ["cli", "storage"]
    assert data[0]["priority"] is None
    assert data[-1]["opened"] is None


# --- the real backlog ----------------------------------------------------


def test_the_committed_backlog_is_well_formed():
    """The validation behind `./go todo --check`, asserted here so a broken item fails the suite."""
    items, errors = todo.load()
    assert not errors, "\n".join(str(e) for e in errors)
    assert items, "no items found under todo/"


def test_the_committed_backlog_stays_within_its_priority_budget():
    items, _ = todo.load()
    assert not (over := todo.over_budget(items)), "\n".join(str(e) for e in over)
