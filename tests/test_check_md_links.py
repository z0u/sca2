"""Tests for the Markdown link check — relative targets and `#anchor` fragments."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_md_links", Path(__file__).resolve().parent.parent / "scripts" / "check_md_links.py"
)
assert _SPEC and _SPEC.loader
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)

ROOT = Path(__file__).resolve().parent.parent


def doc(tmp_path: Path, body: str, name: str = "doc.md") -> Path:
    """A Markdown file holding *body*, with any parent directories made."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def reasons(path: Path) -> list[str]:
    return [f.reason for f in check.findings_in(path, {})]


# --- the two failures the check exists for -------------------------------------------------


def test_a_moved_file_is_reported(tmp_path: Path):
    """The loud half: a target that isn't there any more."""
    page = doc(tmp_path, "See [the notes](./notes.md).")

    (finding,) = check.findings_in(page, {})
    assert (finding.target, finding.reason, finding.line) == ("./notes.md", "no such file", 1)


def test_a_renamed_heading_is_reported(tmp_path: Path):
    """The quiet half: the link still opens the file, just at the top, so nothing looks wrong."""
    doc(tmp_path, "# Storage\n\n## Publishing elsewhere\n", "storage.md")
    page = doc(tmp_path, "See [publishing](./storage.md#publishing-to-the-web).")

    (finding,) = check.findings_in(page, {})
    assert finding.reason.endswith("storage.md slugs to #publishing-to-the-web")


def test_a_link_that_resolves_is_not_reported(tmp_path: Path):
    doc(tmp_path, "# Storage\n\n## Publishing to the web\n", "storage.md")
    page = doc(tmp_path, "See [publishing](./storage.md#publishing-to-the-web).")

    assert check.findings_in(page, {}) == []


# --- the two strips, which is where a naive version reports nonsense -----------------------


def test_a_link_inside_a_fenced_block_is_not_a_link(tmp_path: Path):
    """`ty-loses-pep695-alias.md`'s repro has `(x: Sequence[T])`, which parses as a target."""
    page = doc(tmp_path, "Repro:\n\n```python\ndef wider[T](x: Iterable[T]) -> T: ...\n```\n")

    assert check.findings_in(page, {}) == []


def test_a_link_inside_a_code_span_is_not_a_link(tmp_path: Path):
    """`reports.md` shows this as the shape a report author writes, not as a link of its own."""
    page = doc(tmp_path, "Write `[experiment](./experiment.py)` in the report.")

    assert check.findings_in(page, {}) == []


def test_stripping_keeps_line_numbers(tmp_path: Path):
    """Blanking rather than deleting: a finding after a long fence must still name its own line."""
    page = doc(tmp_path, "```\na\nb\nc\n```\n\n[gone](./nope.md)\n")

    (finding,) = check.findings_in(page, {})
    assert finding.line == 7


def test_a_longer_fence_can_quote_a_shorter_one(tmp_path: Path):
    """The docs nest fences this way, and a closer that's shorter than its opener is content."""
    page = doc(tmp_path, "````\n```\n[gone](./nope.md)\n```\n````\n")

    assert check.findings_in(page, {}) == []


# --- slugs, against the real anchors in this repo ------------------------------------------


@pytest.mark.parametrize(
    ("heading", "slug"),
    [
        ("Publishing to the web", "publishing-to-the-web"),
        # The removed `&` leaves the two spaces that become two hyphens — easy to get wrong.
        ("Provenance & cost", "provenance--cost"),
        ("Hotfix safety: avoid double-spending", "hotfix-safety-avoid-double-spending"),
        ("D2.3: asymmetry", "d23-asymmetry"),
        ("Write cache-friendly experiments", "write-cache-friendly-experiments"),
        ("A `code` span and **bold**", "a-code-span-and-bold"),
        ("A [linked](./x.md) word", "a-linked-word"),
        # Underscores are identifiers here, not emphasis: CommonMark won't open emphasis on an
        # intra-word `_`, so reading a pair as one would eat everything between them. Both of
        # these are headings in this repo today.
        ("Keeps_underscores", "keeps_underscores"),
        (
            "`test_local_apparatus_concurrent` failed on a pristine tree",
            "test_local_apparatus_concurrent-failed-on-a-pristine-tree",
        ),
        (
            "A package `__init__.py` is evidence for deferred imports only",
            "a-package-__init__py-is-evidence-for-deferred-imports-only",
        ),
    ],
)
def test_slugify_matches_github(heading: str, slug: str):
    assert check.slugify(heading) == slug


def test_repeated_headings_take_numbered_suffixes(tmp_path: Path):
    """GitHub keeps the bare slug for the first and numbers the rest, so both forms are offered."""
    anchors = check.anchors_in("# Notes\n\n## Notes\n\n## Notes\n")

    assert {"notes", "notes-1", "notes-2"} <= anchors


def test_a_heading_inside_a_fence_offers_no_anchor():
    """A `#` at the head of a shell example is a comment, and GitHub gives it no anchor."""
    assert check.anchors_in("```sh\n# Install\n```\n") == set()


def test_an_explicit_html_anchor_counts(tmp_path: Path):
    """A hand-written target is how an anchor survives a heading rewrite."""
    doc(tmp_path, '<a id="pinned"></a>\n\n# Something else\n', "target.md")
    page = doc(tmp_path, "See [it](./target.md#pinned).")

    assert check.findings_in(page, {}) == []


# --- target forms ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "link",
    [
        "[x](https://example.com/missing.md)",  # off-repo, and a different job
        "[x](mailto:someone@example.com)",
        "[x](//example.com/missing.md)",  # protocol-relative
    ],
)
def test_external_targets_are_left_alone(tmp_path: Path, link: str):
    assert check.findings_in(doc(tmp_path, link), {}) == []


def test_a_root_absolute_path_resolves_from_the_repo_root(tmp_path: Path):
    """`/AGENTS.md` means the repo root, which is how it reads on GitHub."""
    page = doc(tmp_path, "See [agents](/AGENTS.md) and [nope](/no-such-file.md).")

    assert reasons(page) == ["no such file"]


def test_a_title_is_not_part_of_the_target(tmp_path: Path):
    doc(tmp_path, "# T\n", "target.md")
    page = doc(tmp_path, '[x](./target.md "A title")')

    assert check.findings_in(page, {}) == []


def test_an_angle_bracketed_target_is_unwrapped(tmp_path: Path):
    doc(tmp_path, "# T\n", "a b.md")
    page = doc(tmp_path, "[x](<./a b.md>)")

    assert check.findings_in(page, {}) == []


def test_a_percent_escape_is_decoded(tmp_path: Path):
    doc(tmp_path, "# T\n", "a b.md")
    page = doc(tmp_path, "[x](./a%20b.md)")

    assert check.findings_in(page, {}) == []


def test_an_image_target_is_checked(tmp_path: Path):
    """`references/` broke this way — imported prose whose figures never came with it."""
    page = doc(tmp_path, "![a chart](./figures/chart.png)")

    assert reasons(page) == ["no such file"]


def test_a_fragment_on_a_non_markdown_target_is_left_alone(tmp_path: Path):
    """`foo.py#L10` is a line reference, and belongs to whatever renders the file."""
    (tmp_path / "mod.py").write_text("x = 1\n")
    page = doc(tmp_path, "[x](./mod.py#L1)")

    assert check.findings_in(page, {}) == []


def test_a_bare_fragment_checks_the_same_file(tmp_path: Path):
    page = doc(tmp_path, "# Top\n\nJump to [top](#top), or to [nowhere](#nowhere).")

    (finding,) = check.findings_in(page, {})
    assert finding.reason == "no heading in this file slugs to #nowhere"


def test_a_directory_target_resolves(tmp_path: Path):
    (tmp_path / "style").mkdir()
    page = doc(tmp_path, "See [the set](./style/).")

    assert check.findings_in(page, {}) == []


def test_a_reference_definition_is_checked(tmp_path: Path):
    page = doc(tmp_path, "See [the notes][n].\n\n[n]: ./notes.md\n")

    (finding,) = check.findings_in(page, {})
    assert (finding.target, finding.line) == ("./notes.md", 3)


def test_a_footnote_definition_is_not_a_target(tmp_path: Path):
    """`[^id]:` opens a prose body, not a link."""
    page = doc(tmp_path, "Text.[^a]\n\n[^a]: Kim, et al. Some paper.\n")

    assert check.findings_in(page, {}) == []


def test_a_link_in_an_html_comment_is_left_alone(tmp_path: Path):
    """Nothing renders it, so a stale target there misleads no reader."""
    page = doc(tmp_path, "<!-- was: [x](./gone.md) -->\n")

    assert check.findings_in(page, {}) == []


# --- the file set ---------------------------------------------------------------------------


def test_an_uncommitted_doc_is_still_checked(tmp_path: Path):
    """The trap this avoids: writing a doc, running the check, and being told it's clean.

    Uses a real path in the repo rather than *tmp_path*, since the file set is whatever git reports — and an untracked file is exactly the case at issue.
    """
    scratch = ROOT / "eng" / "zz-test-uncommitted.md"
    scratch.write_text("[gone](./no-such-file.md)\n")
    try:
        assert scratch in check.repo_markdown([Path("eng")])
    finally:
        scratch.unlink()


def test_an_ignored_tree_is_left_out():
    """`.gitignore` does the excluding, so `.venv/` and friends need no rule of their own."""
    listed = {check.display(p) for p in check.repo_markdown([])}

    assert not [p for p in listed if p.startswith((".venv/", "_site/", ".mini/", ".claude/worktrees/"))]


def test_the_repos_own_docs_resolve():
    """The check is a gate, so this is the assertion CI is really making."""
    cache: dict[Path, set[str]] = {}
    found = [f for path in check.repo_markdown([]) for f in check.findings_in(path, cache)]

    assert [str(f) for f in found] == []
