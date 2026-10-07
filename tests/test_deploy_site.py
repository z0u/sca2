"""The site deploy: does one run leave `gh-pages` serving main plus a preview per open PR and nothing else, as a single commit, and does a broken preview or a repeat run cost anything?"""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import load_script

deploy_site = load_script("deploy_site")

SLUG = "z0u/sca2"


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def commit(work: Path, message: str) -> None:
    git("add", "-A", cwd=work)
    git("-c", "user.name=dev", "-c", "user.email=dev@example.invalid", "commit", "-q", "-m", message, cwd=work)


MEMOS: list[tuple[str, tuple[str | None, ...]]] = []
"""What each fake build was handed as its memos: (checkout, each memo's manifest or None)."""


def build(worktree: Path, site_url: str | None, memos: tuple[Path, ...]) -> Path:
    """A stand-in for `./go site`: names the checkout it ran in and the URL it was given, the two things the real one varies on, and records the memos it was handed."""
    who = (worktree / "WHO").read_text()
    if who == "pr-34":
        raise RuntimeError("this branch doesn't build")
    site = worktree / "_site"
    site.mkdir()
    (site / "index.html").write_text(f"<h1>{who}</h1><a href='{site_url or 'https://z0u.github.io/sca2/'}'>index</a>")
    (site / "pdfs.json").write_text(f'{{"{who}": "printed"}}')
    (site / ".nojekyll").write_text("")
    if who == "pr-12":  # two of the reports it re-pins, one printed; the third it pins didn't render
        for key in ("m2/ex-a", "m2/ex-b"):
            (site / key).mkdir(parents=True)
            (site / key / "index.html").write_text(key)
        (site / "m2/ex-a/report.pdf").write_text("%PDF")
    MEMOS.append((who, tuple((m / "pdfs.json").read_text() if (m / "pdfs.json").is_file() else None for m in memos)))
    return site


LOCK_PATCH = """\
@@ -1,6 +1,6 @@
 {
- "m2/ex-a": "1111111111111111111111111111111111111111",
+ "m2/ex-a": "2222222222222222222222222222222222222222",
- "m2/ex-b": "3333333333333333333333333333333333333333",
+ "m2/ex-b": "4444444444444444444444444444444444444444",
- "m2/ex-c": "5555555555555555555555555555555555555555",
  "m2/ex-d": "6666666666666666666666666666666666666666",
+ "m2/ex-e": "7777777777777777777777777777777777777777"
 }"""
"""PR 12's diff of the lock: it re-pins ex-a and ex-b, drops ex-c, leaves ex-d alone, and pins ex-e, which its build doesn't render."""

PR_12_REPORTS = [deploy_site.Report("m2/ex-a", pdf=True), deploy_site.Report("m2/ex-b", pdf=False)]


class FakeGitHub:
    """Three open PRs, one of them from a fork, and a record of every write. PR 12 re-pins some reports and unpins one."""

    def __init__(self, statuses: dict[str, list[dict[str, Any]]] | None = None):
        self.statuses = statuses or {}
        self.writes: list[tuple[str, str, dict[str, Any] | None]] = []

    def paged(self, path: str) -> list[dict[str, Any]]:
        if path.startswith("/pulls/12/files"):
            return [
                {"filename": "docs/m2/ex-a/report.py", "patch": '+    "m2/ex-z": "0123abc",'},
                {"filename": deploy_site.LOCK, "patch": LOCK_PATCH},
            ]
        if path.startswith("/pulls/"):
            return []
        if path.startswith("/pulls"):
            return [
                {"number": 34, "head": {"sha": "b" * 40, "repo": {"full_name": SLUG}}},
                {"number": 12, "head": {"sha": "a" * 40, "repo": {"full_name": SLUG}}},
                {"number": 56, "head": {"sha": "c" * 40, "repo": {"full_name": "someone/sca2"}}},
            ]
        raise AssertionError(f"unexpected listing {path}")

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        if method == "GET" and path.startswith("/commits/"):
            return {"statuses": self.statuses.get(path.split("/")[2], [])}
        self.writes.append((method, path, body))
        return {}

    def posted(self, sha: str) -> dict[str, dict[str, Any]]:
        """The statuses written to `sha`, by context."""
        return {body["context"]: body for _, path, body in self.writes if path == f"/statuses/{sha}" and body}


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """A bare origin: `main`, the head refs of PRs 12 and 34, and a `gh-pages` three deploys deep that still serves a preview for a PR long closed."""
    bare, work = tmp_path / "origin.git", tmp_path / "work"
    git("init", "--bare", "--initial-branch=main", str(bare), cwd=tmp_path)
    git("init", "-q", "--initial-branch=main", str(work), cwd=tmp_path)
    git("remote", "add", "origin", str(bare), cwd=work)
    (work / "WHO").write_text("main")
    commit(work, "source")
    git("push", "-q", "origin", "main", cwd=work)
    for n in (12, 34):
        (work / "WHO").write_text(f"pr-{n}")
        commit(work, f"pr {n}")
        git("push", "-q", "origin", f"HEAD:refs/pull/{n}/head", cwd=work)
        git("reset", "-q", "--hard", "origin/main", cwd=work)

    git("checkout", "-q", "--orphan", "gh-pages", cwd=work)
    git("rm", "-q", "-rf", ".", cwd=work)
    for n in range(3):
        (work / "index.html").write_text(f"<h1>old build {n}</h1>")
        (work / "pr-preview" / "pr-7").mkdir(parents=True, exist_ok=True)
        (work / "pr-preview" / "pr-7" / "index.html").write_text("<h1>a preview whose PR closed weeks ago</h1>")
        commit(work, f"Deploy {n}")
    git("push", "-q", "origin", "gh-pages", cwd=work)
    return bare


@pytest.fixture
def clone(tmp_path: Path, remote: Path) -> Path:
    """The runner as the workflow leaves it: `actions/checkout` clones the triggering ref at depth 1, and nothing has fetched `gh-pages`."""
    path = tmp_path / "runner"
    git("clone", "-q", "--depth", "1", f"file://{remote}", str(path), cwd=tmp_path)
    return path


def served(remote: Path) -> dict[str, str]:
    """Every file on `gh-pages`, path → content."""
    paths = git("ls-tree", "-r", "--name-only", "gh-pages", cwd=remote).splitlines()
    return {path: git("show", f"gh-pages:{path}", cwd=remote) for path in paths}


def test_the_branch_becomes_main_plus_a_preview_per_open_pr(clone: Path, remote: Path):
    api = FakeGitHub()
    assert deploy_site.reconcile(clone, slug=SLUG, builder=build, api=api) == 0

    site = served(remote)
    assert site["index.html"].startswith("<h1>main</h1>"), "production isn't built from main"
    assert "<h1>pr-12</h1>" in site["pr-preview/pr-12/index.html"]
    assert "https://z0u.github.io/sca2/pr-preview/pr-12/" in site["pr-preview/pr-12/index.html"], (
        "the preview's links point outside the preview"
    )
    assert ".nojekyll" in site
    assert not [path for path in site if path.startswith("pr-preview/pr-7/")], "a closed PR's preview survived"
    assert not [path for path in site if path.startswith("pr-preview/pr-56/")], "a fork got a preview"
    assert git("rev-list", "--count", "gh-pages", cwd=remote) == "1", "the deploy kept history behind it"


def test_a_broken_preview_is_reported_and_skipped(clone: Path, remote: Path):
    """PR 34's branch doesn't build. Production and the other preview deploy anyway, and its status says what happened."""
    api = FakeGitHub()
    assert deploy_site.reconcile(clone, slug=SLUG, builder=build, api=api) == 0

    site = served(remote)
    assert "pr-preview/pr-12/index.html" in site
    assert not [path for path in site if path.startswith("pr-preview/pr-34/")]
    pr_12 = git("rev-parse", "refs/pull/12/head", cwd=remote)
    assert api.posted(pr_12)["preview"]["state"] == "success"
    assert api.posted("b" * 40)["preview"]["state"] == "failure"


def test_a_repeat_run_pushes_nothing(clone: Path, remote: Path):
    """The same state builds to the same tree, so the second run leaves the tip alone — and so triggers no Pages deployment."""
    deploy_site.reconcile(clone, slug=SLUG, builder=build, api=FakeGitHub())
    tip = git("rev-parse", "gh-pages", cwd=remote)
    pr_12 = git("rev-parse", "refs/pull/12/head", cwd=remote)
    preview = deploy_site.preview_statuses("https://z0u.github.io/sca2/pr-preview/pr-12/", PR_12_REPORTS)
    api = FakeGitHub(statuses={pr_12: [{"context": context, **body} for context, body in preview.items()]})
    assert deploy_site.reconcile(clone, slug=SLUG, builder=build, api=api) == 0
    assert git("rev-parse", "gh-pages", cwd=remote) == tip
    assert not api.posted(pr_12), "an unchanged preview rewrote its statuses"


def test_each_build_reads_its_own_part_of_the_previous_deploy(clone: Path, remote: Path):
    """Production's memo is the served root; a preview's is its own directory, with the root behind it to borrow from. The first deploy here replaces a `gh-pages` without manifests, so each build is handed paths with nothing in them, and the second run is the one that finds them."""
    MEMOS.clear()
    deploy_site.reconcile(clone, slug=SLUG, builder=build, api=FakeGitHub())
    assert MEMOS == [("main", (None,)), ("pr-12", (None, None))]

    MEMOS.clear()
    deploy_site.reconcile(clone, slug=SLUG, builder=build, api=FakeGitHub())
    assert MEMOS == [
        ("main", ('{"main": "printed"}',)),
        ("pr-12", ('{"pr-12": "printed"}', '{"main": "printed"}')),
    ]
    assert len(git("worktree", "list", cwd=clone).splitlines()) == 1, "the previous deploy's worktree was left behind"


def test_a_dry_run_pushes_nothing(clone: Path, remote: Path):
    before = git("rev-parse", "gh-pages", cwd=remote)
    api = FakeGitHub()
    assert deploy_site.reconcile(clone, slug=SLUG, builder=build, api=api, dry_run=True) == 0
    assert git("rev-parse", "gh-pages", cwd=remote) == before
    assert api.writes == []


def test_the_runners_checkout_is_left_alone(clone: Path):
    """Builds happen in worktrees that are removed afterwards, and the commit goes through a temporary index."""
    deploy_site.reconcile(clone, slug=SLUG, builder=build, api=FakeGitHub())
    assert git("status", "--porcelain", cwd=clone) == ""
    assert (clone / "WHO").read_text() == "main"
    assert len(git("worktree", "list", cwd=clone).splitlines()) == 1
    assert not (clone / ".git" / "deploy-site.index").exists()


def test_previewable_keeps_same_repo_heads_in_number_order():
    pulls = FakeGitHub().paged("/pulls?state=open")
    assert deploy_site.previewable(pulls, SLUG) == [
        deploy_site.PullRequest(12, "a" * 40),
        deploy_site.PullRequest(34, "b" * 40),
    ]
    assert deploy_site.previewable([{"number": 1, "head": {"sha": "d" * 40, "repo": None}}], SLUG) == [], (
        "a PR whose fork was deleted has no head to build"
    )


def test_only_changed_statuses_are_posted():
    api = FakeGitHub(
        statuses={
            "a" * 40: [
                {"context": "preview", "state": "success", "target_url": "https://x/", "description": "d"},
                {"context": "preview: m2/ex-a", "state": "success", "target_url": "https://x/old/", "description": "d"},
            ]
        }
    )
    deploy_site.post_statuses(
        api,
        "a" * 40,
        {
            "preview": {"state": "success", "target_url": "https://x/", "description": "d"},
            "preview: m2/ex-a": {"state": "success", "target_url": "https://x/m2/ex-a/", "description": "d"},
        },
    )
    assert list(api.posted("a" * 40)) == ["preview: m2/ex-a"]


def test_the_statuses_link_the_reports_the_pr_publishes(clone: Path, remote: Path):
    """PR 12 re-pins ex-a and ex-b, which its preview renders, so it gets a status for each. It drops ex-c, leaves ex-d, and pins ex-e without rendering it: none of those are linked. A PR that moves no pins gets the preview link alone."""
    assert deploy_site.repinned(FakeGitHub(), 12) == ["m2/ex-a", "m2/ex-b", "m2/ex-e"]
    api = FakeGitHub()
    deploy_site.reconcile(clone, slug=SLUG, builder=build, api=api)
    posted = api.posted(git("rev-parse", "refs/pull/12/head", cwd=remote))
    preview = "https://z0u.github.io/sca2/pr-preview/pr-12/"
    assert sorted(posted) == ["preview", "preview: m2/ex-a", "preview: m2/ex-b"]
    assert posted["preview"]["target_url"] == preview
    assert posted["preview: m2/ex-a"]["target_url"] == f"{preview}m2/ex-a/"
    assert "with its PDF" in posted["preview: m2/ex-a"]["description"]
    assert "no PDF" in posted["preview: m2/ex-b"]["description"]
    assert list(deploy_site.preview_statuses(preview)) == ["preview"]
