"""Tests for the auto-publish work list — which report notebooks a branch changed."""

import importlib.util
import subprocess
from pathlib import Path

import pytest
from mini.reports import NO_AUTO_PUBLISH_MARKER

_SPEC = importlib.util.spec_from_file_location(
    "changed_reports", Path(__file__).resolve().parent.parent / "scripts" / "changed_reports.py"
)
assert _SPEC and _SPEC.loader
changed_reports = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(changed_reports)

_APP = "import marimo\napp = marimo.App()\n"


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def commit(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def selected(repo: Path, base: str = "main") -> set[str]:
    """The work list, as docs-relative names — the shape the assertions read in."""
    return {p.relative_to(repo).as_posix() for p in changed_reports.changed_reports(base, root=repo)}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo with two published reports on `main`, and a `work` branch checked out."""
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "commit.gpgsign", "false")
    (docs := tmp_path / "docs" / "ex-1").mkdir(parents=True)
    (docs / "report.py").write_text(_APP)
    (docs / "experiment.py").write_text("def main(ctx): ...\n")
    (tmp_path / "docs" / "overview.py").write_text(_APP)
    commit(tmp_path, "reports")
    git(tmp_path, "checkout", "-q", "-b", "work")
    return tmp_path


def test_nothing_changed_is_an_empty_list(repo):
    assert selected(repo) == set()


def test_picks_the_changed_report_and_leaves_the_others(repo):
    (repo / "docs" / "ex-1" / "report.py").write_text(_APP + "# edited\n")
    commit(repo, "edit one report")
    assert selected(repo) == {"docs/ex-1/report.py"}


def test_a_new_report_counts(repo):
    (docs := repo / "docs" / "ex-2").mkdir()
    (docs / "report.py").write_text(_APP)
    commit(repo, "new report")
    assert selected(repo) == {"docs/ex-2/report.py"}


def test_non_notebooks_are_not_reports(repo):
    (repo / "docs" / "ex-1" / "experiment.py").write_text("def main(ctx): pass\n")
    commit(repo, "edit the experiment definition")
    assert selected(repo) == set()


def test_the_opt_out_marker_is_left_to_a_hand_publish(repo):
    (repo / "docs" / "ex-1" / "report.py").write_text(f"# {NO_AUTO_PUBLISH_MARKER}\n{_APP}")
    commit(repo, "flag the slow report")
    assert selected(repo) == set()


def test_a_notebook_outside_docs_is_not_a_report(repo):
    """`marimo.App(` appears in plenty of files that aren't reports — this repo's own
    tests among them. Only `docs/` is the report tree."""
    (tests := repo / "tests").mkdir()
    (tests / "test_something.py").write_text(f'SAMPLE = """{_APP}"""\n')
    (repo / "notebook.py").write_text(_APP)
    commit(repo, "a test that quotes a notebook")
    assert selected(repo) == set()


def test_a_deleted_report_has_no_bundle_to_export(repo):
    (repo / "docs" / "overview.py").unlink()
    commit(repo, "drop a report")
    assert selected(repo) == set()


def test_base_branch_commits_are_not_ours(repo):
    """Three-dot: the diff is against the merge base, so main moving on doesn't count."""
    git(repo, "checkout", "-q", "main")
    (repo / "docs" / "overview.py").write_text(_APP + "# landed on main meanwhile\n")
    commit(repo, "someone else's report")
    git(repo, "checkout", "-q", "work")
    assert selected(repo) == set()


def test_unknown_base_ref_fails_loudly(repo):
    with pytest.raises(SystemExit, match="nope"):
        changed_reports.changed_reports("nope", root=repo)
