"""The preview sweep: does it remove exactly the previews whose PRs are closed, leave the rest of the site byte-identical, and refuse rather than guess when the open-PR list looks wrong?"""

import subprocess
from pathlib import Path

import pytest

from tests.conftest import load_script

stale_previews = load_script("stale_previews")


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def commit(work: Path, message: str) -> None:
    git("add", "-A", cwd=work)
    git("-c", "user.name=deployer", "-c", "user.email=bot@example.invalid", "commit", "-m", message, cwd=work)


def deploy_preview(work: Path, n: int) -> None:
    """A preview as the action leaves it: a whole site build under `pr-preview/pr-<n>/`."""
    directory = work / "pr-preview" / f"pr-{n}"
    (directory / "m2").mkdir(parents=True, exist_ok=True)
    (directory / "index.html").write_text(f"<h1>preview {n}</h1>")
    (directory / ".nojekyll").write_text("")
    (directory / "m2" / "report.html").write_text(f"<p>report in preview {n}</p>")


def tear_down_preview(work: Path, n: int) -> None:
    """A preview as a completed teardown leaves it: the lone stub, and nothing else."""
    directory = work / "pr-preview" / f"pr-{n}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".nojekyll").write_text("")


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """A bare origin whose `gh-pages` carries production plus four previews: one torn down, three still holding a build."""
    bare, work = tmp_path / "origin.git", tmp_path / "work"
    git("init", "--bare", "--initial-branch=main", str(bare), cwd=tmp_path)
    git("init", "--initial-branch=main", str(work), cwd=tmp_path)
    git("remote", "add", "origin", str(bare), cwd=work)
    (work / "README.md").write_text("source")
    commit(work, "source")
    git("push", "-q", "origin", "main", cwd=work)

    git("checkout", "-q", "--orphan", "gh-pages", cwd=work)
    git("rm", "-q", "-rf", ".", cwd=work)
    (work / "index.html").write_text("<h1>the site</h1>")
    (work / "m2").mkdir()
    (work / "m2" / "ex-2.1.1.html").write_text("<p>a published report</p>")
    tear_down_preview(work, 37)
    for n in (42, 59, 135):
        deploy_preview(work, n)
    commit(work, "Deploy site 🚀")
    git("push", "-q", "origin", "gh-pages", cwd=work)
    return bare


@pytest.fixture
def clone(tmp_path: Path, remote: Path) -> Path:
    """The runner as the sweep inherits it: `actions/checkout` clones `main` at depth 1 and never touches `gh-pages`."""
    path = tmp_path / "runner"
    git("clone", "-q", "--depth", "1", f"file://{remote}", str(path), cwd=tmp_path)
    return path


def served(repo: Path, ref: str = "gh-pages") -> dict[str, str]:
    """Every file the branch serves at `ref`, path → blob sha, which is what "byte-identical" means here."""
    listing = git("ls-tree", "-r", ref, cwd=repo)
    return {line.split("\t")[1]: line.split()[2] for line in listing.splitlines()}


def test_removes_only_the_previews_whose_prs_closed(clone: Path, remote: Path):
    before = served(remote)
    assert "pr-preview/pr-42/index.html" in before, "the fixture no longer stages a preview to remove"
    assert stale_previews.sweep(clone, apply=True, open_prs={135}) == 0

    after = served(remote)
    assert {path for path in before if path.startswith("pr-preview/pr-135/")} <= set(after), (
        "an open PR lost its preview"
    )
    assert not [path for path in after if path.startswith(("pr-preview/pr-42/", "pr-preview/pr-59/"))], (
        "a closed PR kept its build"
    )
    assert "pr-preview/pr-37/.nojekyll" in after, "an already-torn-down preview lost its stub"
    site = {path: sha for path, sha in after.items() if not path.startswith("pr-preview/")}
    assert site == {path: sha for path, sha in before.items() if not path.startswith("pr-preview/")}, (
        "the sweep changed something outside the previews"
    )


def test_a_dry_run_pushes_nothing(clone: Path, remote: Path):
    """The default, because this deletes from a published site: it must be possible to see the list before agreeing to it."""
    before = git("rev-parse", "gh-pages", cwd=remote)
    assert stale_previews.sweep(clone, open_prs={135}) == 0
    assert git("rev-parse", "gh-pages", cwd=remote) == before


def test_a_torn_down_preview_is_left_alone(clone: Path):
    """The stub is what the teardown leaves, so a directory holding only that has nothing owing — even for a PR closed long ago."""
    git("fetch", "-q", "origin", "gh-pages", cwd=clone)
    tip = git("rev-parse", "FETCH_HEAD", cwd=clone)
    files = stale_previews.preview_files(clone, tip)
    assert set(files) == {37, 42, 59, 135}
    assert set(stale_previews.stale(files, open_prs=set())) == {42, 59, 135}


def test_an_oversized_sweep_is_refused(clone: Path, remote: Path):
    """The rail against the failure that matters: an open-PR list that came back wrong makes every preview look closed, and the site is two thirds previews. A cap turns that into an exit code rather than a deletion."""
    before = git("rev-parse", "gh-pages", cwd=remote)
    assert stale_previews.sweep(clone, apply=True, max_remove=2, open_prs=set()) == 1
    assert git("rev-parse", "gh-pages", cwd=remote) == before


def test_yields_to_a_deploy_that_lands_mid_sweep(clone: Path, remote: Path, tmp_path: Path):
    """The push is a fast-forward, so a preview deploy arriving between the fetch and the push rejects it. That costs nothing: the next sweep sees whatever the deploy left."""
    git("fetch", "-q", "origin", "gh-pages", cwd=clone)
    tip = git("rev-parse", "FETCH_HEAD", cwd=clone)
    new_tip = stale_previews.remove(clone, tip, ["pr-preview/pr-42/index.html"], "Remove stale previews")

    other = tmp_path / "other"
    git("clone", "-q", "--branch", "gh-pages", f"file://{remote}", str(other), cwd=tmp_path)
    deploy_preview(other, 136)
    commit(other, "Deploy preview for PR 136 🚀")
    git("push", "-q", "origin", "gh-pages", cwd=other)

    assert stale_previews.push(clone, "origin", "gh-pages", new_tip) is False
    assert "pr-preview/pr-136/index.html" in served(remote), "the deploy that won the race was dropped anyway"


def test_the_removal_leaves_the_callers_own_checkout_alone(clone: Path):
    """It writes through a temporary index, so the `main` checkout CI is sitting on keeps its index and its working files."""
    git("fetch", "-q", "origin", "gh-pages", cwd=clone)
    tip = git("rev-parse", "FETCH_HEAD", cwd=clone)
    stale_previews.remove(clone, tip, ["pr-preview/pr-42/index.html"], "Remove stale previews")
    assert git("status", "--porcelain", cwd=clone) == ""
    assert (clone / "README.md").read_text() == "source"
    assert not (clone / ".git" / "stale-previews.index").exists()


def test_reads_the_slug_from_a_remote_url(clone: Path, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    git("remote", "set-url", "origin", "https://github.com/z0u/sca2.git", cwd=clone)
    assert stale_previews.repo_slug(clone, "origin") == "z0u/sca2"
    git("remote", "set-url", "origin", "git@github.com:z0u/sca2", cwd=clone)
    assert stale_previews.repo_slug(clone, "origin") == "z0u/sca2"
    monkeypatch.setenv("GITHUB_REPOSITORY", "z0u/elsewhere")
    assert stale_previews.repo_slug(clone, "origin") == "z0u/elsewhere"
