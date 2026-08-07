"""Tests for the forgotten-publish check — changed reports whose pin didn't move."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from mini.reports import MANUAL_PUBLISH_MARKER

_SPEC = importlib.util.spec_from_file_location(
    "unpublished_reports", Path(__file__).resolve().parent.parent / "scripts" / "unpublished_reports.py"
)
assert _SPEC and _SPEC.loader
unpub = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(unpub)

_APP = "import marimo\napp = marimo.App()\n"


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def commit(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


def pin(repo: Path, key: str, revision: str) -> None:
    """Repin one report, as `./go publish` would."""
    lock = repo / "docs" / "publish.lock"
    pins = json.loads(lock.read_text()) | {key: revision}
    lock.write_text(json.dumps(dict(sorted(pins.items())), indent=1) + "\n")


def flagged(repo: Path, base: str = "main") -> set[str]:
    return {p.relative_to(repo).as_posix() for p in unpub.unpublished(base, root=repo)}


def changed(repo: Path, base: str = "main") -> set[str]:
    return {p.relative_to(repo).as_posix() for p in unpub.changed_reports(base, root=repo)}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Two published reports on `main`, with a `work` branch checked out."""
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "commit.gpgsign", "false")
    (docs := tmp_path / "docs" / "ex-1").mkdir(parents=True)
    (docs / "report.py").write_text(_APP)
    (docs / "experiment.py").write_text("def main(ctx): ...\n")
    (tmp_path / "docs" / "overview.py").write_text(_APP)
    (tmp_path / "docs" / "publish.lock").write_text(json.dumps({"ex-1": "aaa", "overview": "bbb"}, indent=1) + "\n")
    commit(tmp_path, "reports")
    git(tmp_path, "checkout", "-q", "-b", "work")
    return tmp_path


# --- which notebooks count as changed reports -------------------------------------


def test_nothing_changed(repo):
    assert changed(repo) == set()


def test_the_changed_report_only(repo):
    (repo / "docs" / "ex-1" / "report.py").write_text(_APP + "# edited\n")
    commit(repo, "edit one report")
    assert changed(repo) == {"docs/ex-1/report.py"}


def test_non_notebooks_are_not_reports(repo):
    (repo / "docs" / "ex-1" / "experiment.py").write_text("def main(ctx): pass\n")
    commit(repo, "edit the experiment definition")
    assert changed(repo) == set()


def test_a_notebook_outside_docs_is_not_a_report(repo):
    """`marimo.App(` appears in plenty of files that aren't reports — this repo's own
    tests among them. Only `docs/` is the report tree."""
    (tests := repo / "tests").mkdir()
    (tests / "test_something.py").write_text(f'SAMPLE = """{_APP}"""\n')
    (repo / "notebook.py").write_text(_APP)
    commit(repo, "a test that quotes a notebook")
    assert changed(repo) == set()


def test_a_deleted_report_has_nothing_to_publish(repo):
    (repo / "docs" / "overview.py").unlink()
    commit(repo, "drop a report")
    assert changed(repo) == set()


def test_base_branch_commits_are_not_ours(repo):
    """Three-dot: the diff is against the merge base, so main moving on doesn't count."""
    git(repo, "checkout", "-q", "main")
    (repo / "docs" / "overview.py").write_text(_APP + "# landed on main meanwhile\n")
    commit(repo, "someone else's report")
    git(repo, "checkout", "-q", "work")
    assert changed(repo) == set()


# --- and of those, which were left unpublished --------------------------------------


def test_a_changed_report_without_a_new_pin_is_flagged(repo):
    (repo / "docs" / "ex-1" / "report.py").write_text(_APP + "# edited\n")
    commit(repo, "edit a report")
    assert flagged(repo) == {"docs/ex-1/report.py"}


def test_repinning_clears_it(repo):
    (repo / "docs" / "ex-1" / "report.py").write_text(_APP + "# edited\n")
    pin(repo, "ex-1", "ccc")
    commit(repo, "edit and publish a report")
    assert flagged(repo) == set()


def test_only_the_unpinned_report_is_flagged(repo):
    (repo / "docs" / "ex-1" / "report.py").write_text(_APP + "# edited\n")
    (repo / "docs" / "overview.py").write_text(_APP + "# edited\n")
    pin(repo, "ex-1", "ccc")
    commit(repo, "publish one of the two")
    assert flagged(repo) == {"docs/overview.py"}


def test_a_brand_new_report_needs_its_first_publish(repo):
    (docs := repo / "docs" / "ex-2").mkdir()
    (docs / "report.py").write_text(_APP)
    commit(repo, "new report")
    assert flagged(repo) == {"docs/ex-2/report.py"}


def test_the_marker_opts_out_of_the_reminder(repo):
    (repo / "docs" / "ex-1" / "report.py").write_text(f"# {MANUAL_PUBLISH_MARKER}\n{_APP}# edited\n")
    commit(repo, "hand-published from here on")
    assert flagged(repo) == set()


def test_a_stale_base_does_not_produce_noise(repo):
    """A container's `origin/main` is often behind. That makes more reports look changed,
    but their pins moved in the same range, so they don't read as unpublished."""
    (repo / "docs" / "ex-1" / "report.py").write_text(_APP + "# edited\n")
    pin(repo, "ex-1", "ccc")
    commit(repo, "edit and publish")
    git(repo, "tag", "stale-base", "main")  # pretend main advanced past this point
    (repo / "docs" / "overview.py").write_text(_APP + "# edited later\n")
    commit(repo, "a later edit, unpublished")
    assert flagged(repo, "stale-base") == {"docs/overview.py"}


def test_unknown_base_ref_fails_loudly(repo):
    with pytest.raises(SystemExit, match="nope"):
        unpub.changed_reports("nope", root=repo)
