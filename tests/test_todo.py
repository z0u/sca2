"""Tests for the backlog index — header validation, and the query that replaces a committed list."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("todo", Path(__file__).resolve().parent.parent / "scripts" / "todo.py")
assert _SPEC and _SPEC.loader
todo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(todo)

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
    it = todo.parse(write(tmp_path, "x", VALID))
    assert (it.title, it.status, it.tags) == ("A title", "open", ("cli", "storage"))
    assert it.body == "The body."
    assert it.opened.isoformat() == "2026-08-13"


def test_the_title_comes_off_the_heading_not_the_filename(tmp_path):
    assert todo.parse(write(tmp_path, "some-slug", VALID)).title == "A title"
    assert todo.parse(write(tmp_path, "some-slug", VALID)).slug == "some-slug"


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
    ],
    ids=["no-status", "bad-status", "tags-not-a-list", "bad-date", "bundle-is-a-list"],
)
def test_bad_fields_are_rejected(tmp_path, fields):
    with pytest.raises(todo.TodoError):
        todo.parse(item(tmp_path, **fields))


def test_an_unknown_key_is_rejected_rather_than_ignored(tmp_path):
    with pytest.raises(todo.TodoError, match="unknown key"):
        todo.parse(item(tmp_path, priority="high"))


def test_a_duplicate_key_is_rejected(tmp_path):
    with pytest.raises(todo.TodoError, match="duplicate"):
        todo.parse(write(tmp_path, "x", "---\nstatus: open\nstatus: done\ntags: [cli]\n---\n# T\n"))


def test_tags_are_optional(tmp_path):
    assert todo.parse(item(tmp_path, tags=None)).tags == ()


def test_a_closed_date_is_optional_on_done(tmp_path):
    """Migrated items often record that something shipped without recording when."""
    assert todo.parse(item(tmp_path, status="done")).closed is None


def test_only_done_carries_a_closed_date(tmp_path):
    with pytest.raises(todo.TodoError, match="closed"):
        todo.parse(item(tmp_path, status="open", closed="2026-08-13"))


def test_closing_before_opening_is_rejected(tmp_path):
    with pytest.raises(todo.TodoError, match="before opened"):
        todo.parse(item(tmp_path, status="done", opened="2026-08-13", closed="2026-08-01"))


def test_opened_is_optional(tmp_path):
    assert todo.parse(item(tmp_path, opened=None)).opened is None


def test_an_error_names_the_file(tmp_path):
    with pytest.raises(todo.TodoError, match="broken.md"):
        todo.parse(item(tmp_path, "broken", status="nonsense"))


# --- loading -------------------------------------------------------------


def test_a_finding_is_a_status(backlog):
    items, _ = todo.load(backlog)
    assert slugs(todo.select(items, status="finding")) == ["learned"]


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


def test_settled_work_and_findings_are_hidden_by_default(backlog):
    items, _ = todo.load(backlog)
    assert {"shipped", "learned"}.isdisjoint(slugs(todo.select(items)))


def test_a_set_narrows_the_selection(backlog):
    items, _ = todo.load(backlog)
    assert slugs(todo.select(items, sets=["science"], status="finding")) == ["learned"]
    assert "learned" not in slugs(todo.select(items, sets=["eng"], status="finding"))


def test_sets_are_the_directories_that_exist(backlog):
    assert todo._sets(backlog) == ["eng", "science"]


def test_status_filter_finds_them(backlog):
    items, _ = todo.load(backlog)
    assert slugs(todo.select(items, status="done")) == ["shipped"]


def test_tags_are_conjunctive(backlog):
    items, _ = todo.load(backlog)
    assert slugs(todo.select(items, tags=["cli"])) == ["newer", "older"]
    assert slugs(todo.select(items, tags=["cli", "storage"])) == ["newer"]
    assert todo.select(items, tags=["cli", "vis"]) == []


def test_bundle_filter(backlog):
    items, _ = todo.load(backlog)
    assert slugs(todo.select(items, bundle="cli-devx")) == ["newer", "older"]


def test_newest_first_with_undated_last(backlog):
    items, _ = todo.load(backlog)
    assert slugs(todo.select(items)) == ["newer", "older", "undated"]


def test_an_item_knows_its_set(backlog):
    items, _ = todo.load(backlog)
    assert {it.slug: it.set for it in items}["learned"] == "science"


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


def test_json_shape_is_serializable(backlog):
    import json

    items, _ = todo.load(backlog)
    data = json.loads(json.dumps([it.as_dict() for it in todo.select(items)]))
    assert data[0]["slug"] == "newer"
    assert data[0]["tags"] == ["cli", "storage"]
    assert data[-1]["opened"] is None


# --- the real backlog ----------------------------------------------------


def test_the_committed_backlog_is_well_formed():
    """The gate `./go check --lint` runs, asserted here too so a broken item fails the suite."""
    items, errors = todo.load()
    assert not errors, "\n".join(str(e) for e in errors)
    assert items, "no items found under todo/"
