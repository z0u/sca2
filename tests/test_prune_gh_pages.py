"""The deploy-branch pruner: does it keep the site byte-identical while cutting the history behind it, and does it yield when a preview deploy beats it to the push?"""

import re
import subprocess
from pathlib import Path

import pytest

from tests.conftest import load_script

prune_gh_pages = load_script("prune_gh_pages")
WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "publish-docs.yml"


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def deploy(work: Path, n: int) -> None:
    """One deploy commit: a site file whose content identifies the build it came from."""
    (work / "index.html").write_text(f"<h1>build {n}</h1>")
    (work / "pr-preview").mkdir(exist_ok=True)
    (work / "pr-preview" / f"pr-{n}.html").write_text(f"preview {n}")
    git("add", "-A", cwd=work)
    git(
        "-c",
        f"user.name=deployer {n}",
        "-c",
        "user.email=bot@example.invalid",
        "commit",
        "-m",
        f"Deploy {n} 🚀",
        cwd=work,
    )


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """A bare origin with `main` and a `gh-pages` branch eight deploys deep."""
    bare, work = tmp_path / "origin.git", tmp_path / "work"
    git("init", "--bare", "--initial-branch=main", str(bare), cwd=tmp_path)
    git("init", "--initial-branch=main", str(work), cwd=tmp_path)
    git("remote", "add", "origin", str(bare), cwd=work)
    (work / "README.md").write_text("source")
    git("add", "-A", cwd=work)
    git("-c", "user.name=dev", "-c", "user.email=dev@example.invalid", "commit", "-m", "source", cwd=work)
    git("push", "-q", "origin", "main", cwd=work)

    git("checkout", "-q", "--orphan", "gh-pages", cwd=work)
    git("rm", "-q", "-rf", ".", cwd=work)
    for n in range(1, 9):
        deploy(work, n)
    git("push", "-q", "origin", "gh-pages", cwd=work)
    return bare


@pytest.fixture
def clone(tmp_path: Path, remote: Path) -> Path:
    """The runner as the prune step inherits it: `actions/checkout` clones `main` at depth 1, and the deploy step then fetches `gh-pages` at depth 1 into the same repository.

    That second fetch is the part worth reproducing. It leaves the tip present as a shallow boundary, so a plain fetch pulls nothing and the branch reads as one commit deep — which is what the first CI run did, on a fixture that cloned shallow but let the pruner meet `gh-pages` fresh.
    """
    path = tmp_path / "runner"
    git("clone", "-q", "--depth", "1", f"file://{remote}", str(path), cwd=tmp_path)
    git("fetch", "-q", "--no-recurse-submodules", "--depth=1", "origin", "gh-pages", cwd=path)
    return path


def shas(repo: Path, ref: str = "gh-pages") -> list[str]:
    return git("rev-list", ref, cwd=repo).split()


def test_sees_the_true_depth_through_a_shallow_prefetch(clone: Path, remote: Path):
    """The regression from the pruner's first CI run. With `gh-pages` already present at a shallow boundary, a plain fetch pulls nothing and `rev-list` stops after one commit, so a 344-commit branch reads as 1 and the prune declines — a failure shaped exactly like the healthy no-op, which is why it needs a test of its own rather than trust."""
    assert git("rev-parse", "--is-shallow-repository", cwd=clone) == "true", (
        "the fixture no longer reproduces a shallow runner"
    )
    assert prune_gh_pages.fetch_tip(clone, "origin", "gh-pages") == git("rev-parse", "gh-pages", cwd=remote)
    assert len(shas(clone, "FETCH_HEAD")) == 8


def test_reroots_to_the_keep_window(clone: Path, remote: Path):
    """Past the threshold, the branch is cut to `keep` commits."""
    assert prune_gh_pages.prune(clone, "origin", "gh-pages", keep=3, prune_above=5) == 0
    assert len(shas(remote)) == 3


def test_the_deployed_site_survives_untouched(clone: Path, remote: Path):
    """Ancestry changes; content does not. The tip's tree — the thing Pages actually serves, previews included — is the same object it was before."""
    before = git("rev-parse", "gh-pages^{tree}", cwd=remote)
    prune_gh_pages.prune(clone, "origin", "gh-pages", keep=3, prune_above=5)
    assert git("rev-parse", "gh-pages^{tree}", cwd=remote) == before


def test_kept_commits_keep_their_messages_and_authors(clone: Path, remote: Path):
    """The window is replayed, not squashed, so recent deploys stay legible and diffable."""
    prune_gh_pages.prune(clone, "origin", "gh-pages", keep=3, prune_above=5)
    assert git("log", "--format=%s %an", "gh-pages", cwd=remote).split("\n") == [
        "Deploy 8 🚀 deployer 8",
        "Deploy 7 🚀 deployer 7",
        "Deploy 6 🚀 deployer 6",
    ]


def test_below_the_threshold_nothing_moves(clone: Path, remote: Path):
    """Eight commits under a threshold of twenty: the branch is left exactly as it was."""
    before = shas(remote)
    assert prune_gh_pages.prune(clone, "origin", "gh-pages", keep=3, prune_above=20) == 0
    assert shas(remote) == before


def test_a_second_prune_is_a_no_op(clone: Path, remote: Path):
    """Hysteresis: once cut to the window, the branch sits under the threshold and the next build leaves it alone."""
    prune_gh_pages.prune(clone, "origin", "gh-pages", keep=3, prune_above=5)
    after_first = shas(remote)
    assert prune_gh_pages.prune(clone, "origin", "gh-pages", keep=3, prune_above=5) == 0
    assert shas(remote) == after_first


def test_dry_run_pushes_nothing(clone: Path, remote: Path):
    before = shas(remote)
    assert prune_gh_pages.prune(clone, "origin", "gh-pages", keep=3, prune_above=5, dry_run=True) == 0
    assert shas(remote) == before


def test_yields_to_a_deploy_that_lands_mid_prune(clone: Path, remote: Path, tmp_path: Path, monkeypatch, capsys):
    """A preview deploy pushing between our fetch and our push takes the lease with it. We must leave that deploy standing rather than force it away."""
    racer = tmp_path / "racer"
    git("clone", "-q", str(remote), str(racer), cwd=tmp_path)
    git("checkout", "-q", "gh-pages", cwd=racer)

    real_fetch = prune_gh_pages.fetch_tip

    def fetch_then_lose_the_race(repo: Path, remote_name: str, branch: str) -> str:
        tip = real_fetch(repo, remote_name, branch)
        deploy(racer, 99)
        git("push", "-q", "origin", "gh-pages", cwd=racer)
        return tip

    monkeypatch.setattr(prune_gh_pages, "fetch_tip", fetch_then_lose_the_race)
    assert prune_gh_pages.prune(clone, "origin", "gh-pages", keep=3, prune_above=5) == 0
    assert git("log", "-1", "--format=%s", "gh-pages", cwd=remote) == "Deploy 99 🚀"
    assert len(shas(remote)) == 9
    assert "moved while we were pruning" in capsys.readouterr().out


def test_works_without_a_configured_git_identity(clone: Path, remote: Path, tmp_path: Path, monkeypatch):
    """A runner has no `user.name`: `actions/checkout` configures none, and `commit-tree` refuses to write a commit without one. Nothing here needs a fallback identity — every replayed commit carries its own — but only an unconfigured environment proves it, and a dev machine always has one."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "absent-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("EMAIL", raising=False)
    assert prune_gh_pages.prune(clone, "origin", "gh-pages", keep=3, prune_above=5) == 0
    assert len(shas(remote)) == 3


def test_the_workflow_and_the_defaults_agree():
    """The workflow passes both thresholds explicitly, so its step says what will happen without anyone opening the script. That leaves two copies of the numbers, and the point of the second one is that a dry run on a laptop predicts what CI does — so they have to match."""
    step = WORKFLOW.read_text()
    assert "scripts/prune_gh_pages.py" in step, "the workflow no longer runs the pruner"
    assert re.search(rf"--keep {prune_gh_pages.KEEP}\b", step)
    assert re.search(rf"--prune-above {prune_gh_pages.PRUNE_ABOVE}\b", step)


def test_keep_above_the_threshold_is_refused(monkeypatch, capsys):
    """A window deeper than the trigger would deepen the branch rather than trim it."""
    monkeypatch.setattr("sys.argv", ["prune_gh_pages.py", "--keep", "50", "--prune-above", "10"])
    with pytest.raises(SystemExit):
        prune_gh_pages.main()
    assert "would deepen the branch" in capsys.readouterr().err
