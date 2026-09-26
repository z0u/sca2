# title: Ex 2.2.15: lines cut short by the training window

# The design constants and the refs come from `experiment.py` beside this script (the script's directory is
# on sys.path while it runs).
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import experiment as ex
from mini.lit import memo, stop
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

POLICY_TEXT = {
    "all": "every labelled line, whatever is visible (the current behaviour)",
    "whole": "only lines wholly inside the window",
    "half": "only lines with more than half their tokens inside the window",
    "scaled": "every labelled line, its pull scaled by the share of it that is visible",
    "knowable": "only positions at or after a visible op word",
    "cut-only": "only lines the window cuts short",
}


def model_text(arm: ex.Arm) -> str:
    """What the arm changes about the model, against ex-2.2.14's primary."""
    if arm.line_mask:
        return "attention stops at each newline"
    return "tied readout" if arm.tie else "untied readout"


def conditions_html() -> str:
    """One row per arm, plus the served control."""
    head = (
        "<tr><th>condition</th><th>which labelled lines the anchor pulls</th>"
        "<th>model</th><th class=num>window</th><th class=num>pull kept</th><th class=num>seeds</th></tr>"
    )
    rows = [
        f"<tr><td><code>{a.name}</code></td><td>{POLICY_TEXT[a.policy]}</td><td>{model_text(a)}</td><td class=num>{a.block}</td>"
        f"<td class=num>{ex.pull_share(a.policy, a.block):.0%}</td><td class=num>{ex.SEEDS}</td></tr>"
        for a in ex.ARMS
    ]
    rows.append(
        f"<tr><td><code>{ex.CONTROL}</code></td><td>none: the un-anchored control from ex-2.2.11, served from the store</td><td>untied readout</td>"
        f"<td class=num>{ex.BLOCK}</td><td class=num>–</td><td class=num>{ex.CONTROL_SEEDS}</td></tr>"
    )
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def shares_html() -> str:
    """How often a line visit shows each run of roles, at both window sizes, computed from the sampler."""
    long, short = ex.visit_shares(ex.BLOCK), ex.visit_shares(ex.SHORT_BLOCK)
    head = (
        f"<tr><th>visible roles</th><th>what the anchor can see</th>"
        f"<th class=num>{ex.BLOCK}-token window</th><th class=num>{ex.SHORT_BLOCK}-token window</th></tr>"
    )

    def seen(first: int, last: int) -> str:
        if (first, last) == (0, ex.LINE_TOKENS - 1):
            return "the whole line"
        if first <= ex.OP_ROLE <= last:
            return "the op word, cut short"
        return "<strong>no op word</strong>"

    rows = [
        f"<tr><td>{' '.join(ex.ROLES[first : last + 1])}</td><td>{seen(first, last)}</td>"
        f"<td class=num>{long[(first, last)]:.1%}</td><td class=num>{short[(first, last)]:.1%}</td></tr>"
        for first, last in long
    ]
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


LONG = ex.visit_shares(ex.BLOCK)
SHORT = ex.visit_shares(ex.SHORT_BLOCK)
CUT = 1 - LONG[(0, 5)]
CUT_SHORT = 1 - SHORT[(0, 5)]
BLIND = sum(v for (first, last), v in LONG.items() if not first <= ex.OP_ROLE <= last)
OP1_ONLY = LONG[(0, 0)]
SMOKE = ex.SMOKE_RED_LEAN
SMOKE_REMOVED = (SMOKE["all"] - SMOKE["whole"]) / (SMOKE["all"] - SMOKE["control"])


# --- Loading ------------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict | None:
    return None if path is None else json.loads(path.read_text())


with tempfile.TemporaryDirectory() as _tmp:
    _files = fetch([ex.METRICS_REF, ex.TRAJ_REF], Path(_tmp))
    metrics_loaded = read_json(_files[ex.METRICS_REF])
    traj_loaded = read_json(_files[ex.TRAJ_REF])

OP = ex.ANCHORED_OP
LAST = ex.FINAL_SLICE
ARM_NAMES = tuple(a.name for a in ex.ARMS)
ARM = {a.name: a for a in ex.ARMS}
POLICY_ARMS = ("all", "whole", "half", "scaled", "knowable", "cut-only")
SHORT_ARMS = ("all-short", "whole-short")
MODEL_ARMS = ("whole-mask", "whole-tied", "all-tied")
# The roles that carry no color: the op word, `=`, and the newline.
SYNTAX_ROLES = (1, 3, 5)


@dataclass(frozen=True)
class Results:
    """Every published result the report reads: the eval records (the served control's among them, under
    `ex.CONTROL`) and the training trajectories of the anchored runs.
    """

    metrics: dict
    traj: dict

    def __memo_key__(self) -> str:
        return f"{len(self.metrics['runs'])}:{len(self.traj)}"

    def runs(self, cond: str) -> list[dict]:
        """The eval records of a condition, in seed order."""
        return sorted((r for r in self.metrics["runs"] if r["condition"] == cond), key=lambda r: r["seed"])

    def cos(self, cond: str) -> np.ndarray:
        """(seeds, L1, T): the mean cosine with e₁ over every op's probe lines (each op the same count)."""
        return np.array([np.mean([r["cos_mean"][o] for o in ex.OP_NAMES], axis=0) for r in self.runs(cond)])

    def lean(self, cond: str) -> np.ndarray:
        """Per seed, the lean: the first operand at the final slice."""
        return self.cos(cond)[:, LAST, 0]

    def control_lean(self) -> float:
        return float(self.lean(ex.CONTROL).mean())

    def excess(self, cond: str) -> float:
        """The seed-mean lean less the control's."""
        return float(self.lean(cond).mean()) - self.control_lean()

    def margin(self, cond: str) -> np.ndarray:
        return np.array([r["op_margin"] for r in self.runs(cond)], float)

    def eem(self, cond: str, op: str) -> np.ndarray:
        return np.array([r["holdout_eem"][op] for r in self.runs(cond)], float)

    def task_gap(self, cond: str, op: str) -> float:
        return float(self.eem(cond, op).mean() - self.eem(ex.CONTROL, op).mean())

    def worst_gap(self, cond: str) -> tuple[str, float]:
        """The op furthest from the control, by absolute gap, and its signed gap."""
        gaps = {o: self.task_gap(cond, o) for o in ex.OP_NAMES}
        o = max(gaps, key=lambda k: abs(gaps[k]))
        return o, gaps[o]

    def fragment(self, cond: str, ops: Sequence[str] = ex.OP_NAMES) -> np.ndarray:
        """Per seed, the mean cosine with e₁ over every position of every trailing fragment of *ops*, at the
        final slice. Each op has the same number of lines, so pooling the per-line means by position count is
        the mean over positions.
        """
        out = []
        for r in self.runs(cond):
            total = count = 0.0
            for o in ops:
                for start in ex.FRAGMENT_START_ROLES:
                    v = np.asarray(r["fragment"][o][str(start)])[LAST]
                    total += v.sum()
                    count += len(v)
            out.append(total / count)
        return np.array(out)

    def fragment_contrast(self, cond: str) -> np.ndarray:
        """Per seed, the anchored op's trailing fragments less the rest."""
        return self.fragment(cond, (OP,)) - self.fragment(cond, tuple(o for o in ex.OP_NAMES if o != OP))

    def fragment_roles(self, cond: str) -> dict[int, np.ndarray]:
        """Start role → (T − start,): the seed-mean cosine at each position of the trailing fragments, every op."""
        return {
            s: np.mean([[r["fragment"][o][str(s)][LAST] for o in ex.OP_NAMES] for r in self.runs(cond)], axis=(0, 1))
            for s in ex.FRAGMENT_START_ROLES
        }

    def landing(self, cond: str) -> np.ndarray:
        """(seeds, L1, T): where the pull lands, per role."""
        return np.array([r["pull_landing"] for r in self.runs(cond)], float)

    def readout(self, cond: str, key: str) -> np.ndarray:
        return np.array([r["readout"][key] for r in self.runs(cond)], float)

    def newline_e1(self, cond: str) -> np.ndarray:
        """Per seed, the e₁ component of the ⏎ embedding row."""
        return np.array([r["readout"]["embedding_e1"][r["readout"]["vocab"].index("\n")] for r in self.runs(cond)])

    def trajectories(self, cond: str) -> list[dict]:
        return [self.traj[r["label"]]["traj"] for r in self.runs(cond)]

    def lean_traj(self, cond: str) -> tuple[np.ndarray, np.ndarray]:
        """(points,) epochs and (seeds, points): the lean through training."""
        ts = self.trajectories(cond)
        return np.asarray(ts[0]["epoch"]), np.array([np.asarray(t["alpha_roles"])[:, LAST, 0] for t in ts])

    def fragment_traj(self, cond: str) -> np.ndarray:
        """(seeds, points): the trailing-fragment lean through training, pooled over ops."""
        return np.array([np.asarray(t["fragment"])[:, LAST].mean(axis=1) for t in self.trajectories(cond)])


def sd(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, ref_rows: frozenset[int] = frozenset()) -> str:
    """An authored result table in the shared report style; the first column is text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=ref' if r in ref_rows else ''}>"
        + "".join(f"<td{' class=num' if i else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def bold_if(text: str, ok: bool) -> str:
    return f"<b>{text}</b>" if ok else text


def verdict_md(status: str, line: str) -> str:
    """The verdict admonition that closes a hypothesis section; the site hoists its title into the heading."""
    line = line[0].upper() + line[1:]
    kind = {"pass": "success", "partial": "warning", "miss": "danger", "unresolved": "info"}[status]
    return f"/// admonition | {status.capitalize()}\n    type: {kind}\n{line}\n///"


INKS = {
    ex.CONTROL: ("#6b6b6b", "#b0b0b0"),
    "all": ("#c0392b", "#ff8a76"),
    "whole": ("#2b6cb0", "#7fb3ff"),
    "half": ("#2a9d8f", "#7fe0d2"),
    "scaled": ("#2e8b57", "#7fd8a4"),
    "knowable": ("#7b3fa0", "#cfa3ff"),
    "cut-only": ("#d9822b", "#ffc07a"),
    "all-short": ("#e07a6a", "#ffb3a6"),
    "whole-short": ("#6a9ad0", "#b3d1ff"),
    "whole-mask": ("#1f3f6e", "#a6c4ff"),
    "whole-tied": ("#3a8fb7", "#9fe0ff"),
    "all-tied": ("#8b2a1f", "#ffb09c"),
}
MARKERS = {ex.CONTROL: "s", "all": "o", "whole": "o", "cut-only": "v", "all-short": "D", "whole-short": "D"}
MARKERS |= {"whole-mask": "^", "whole-tied": "P", "all-tied": "P"}


def ink(cond: str) -> str:
    return light_dark(*INKS[cond])


def dots(
    ax: Axes, x: float, v: np.ndarray, cond: str, *, rng, ms: float = 5.0, width: float = 0.08, label=None
) -> None:
    """One column of per-seed dots, a thin bar over the seed range, and the seed mean in the condition's marker."""
    v = np.asarray(v, float)
    color, m = ink(cond), MARKERS.get(cond, "o")
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + rng.uniform(-width, width, len(v)), v, "o", ms=2.2, color=color, alpha=0.45, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=ms, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6, label=label)


def control_band(ax: Axes, centre: float, half: float) -> None:
    """The control's seed mean as a line, with the band of ±*half* around it shaded."""
    ax.axhspan(centre - half, centre + half, color=ink(ex.CONTROL), alpha=0.15, lw=0, zorder=0)
    ax.axhline(centre, color=ink(ex.CONTROL), lw=0.8, ls="--", zorder=1)


def gate_line(ax: Axes, y: float, *, fail: str) -> None:
    """A dashed gate line with the failing side hatched."""
    ax.axhline(y, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
    lo, hi = ax.get_ylim()
    span = (lo, y) if fail == "below" else (y, hi)
    ax.axhspan(*span, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, zorder=0, alpha=0.1)
    ax.set_ylim(lo, hi)


def arm_axis(ax: Axes, conds: Sequence[str]) -> None:
    ax.set_xticks(range(len(conds)), conds, rotation=30, ha="right", fontsize=8)


# --- Verdicts -----------------------------------------------------------------------------------


def removed(res: Results, cond: str) -> float:
    """The share of `all`'s excess lean that *cond* removes."""
    return 1 - res.excess(cond) / res.excess("all")


def in_band(res: Results, cond: str) -> bool:
    return abs(res.excess(cond)) <= ex.LEAN_BAND


def h1(res: Results) -> dict:
    e = res.excess("all")
    kept = res.excess("cut-only") / e
    whole = removed(res, "whole")
    if e < ex.READABLE_LEAN:
        status = "unresolved"
    elif in_band(res, "whole") and kept >= ex.CUT_ONLY_SHARE:
        status = "pass"
    elif whole >= ex.PARTIAL_SHARE and kept >= ex.PARTIAL_SHARE:
        status = "partial"
    else:
        status = "miss"
    return {
        "status": status,
        "excess": e,
        "whole_removed": whole,
        "cut_kept": kept,
        "whole_excess": res.excess("whole"),
    }


def h2_arm(res: Results, cond: str) -> dict:
    """H2 for one arm: its margin as a share of `all`'s, and its largest task gap."""
    share = float(res.margin(cond).mean() / res.margin("all").mean())
    op, gap = res.worst_gap(cond)
    return {"share": share, "op": op, "gap": gap, "ok": share >= ex.MARGIN_KEEP and abs(gap) <= ex.TASK_GATE}


def h2(res: Results) -> dict:
    arms = {c: h2_arm(res, c) for c in ARM_NAMES}
    failed = [c for c, a in arms.items() if not a["ok"]]
    return {"status": "miss" if failed else "pass", "arms": arms, "failed": failed}


def h3(res: Results) -> dict:
    f = {c: float(res.fragment(c).mean()) for c in (*ARM_NAMES, ex.CONTROL)}
    ok = f["all"] > f[ex.CONTROL] and f["whole"] < f["all"] and f["knowable"] < f["all"]
    return {"status": "pass" if ok else "miss", "lean": f}


def h4(res: Results) -> dict:
    long = float(res.lean("all").mean() - res.lean("whole").mean())
    short = float(res.lean("all-short").mean() - res.lean("whole-short").mean())
    grew, band = short > long, in_band(res, "whole-short")
    status = "pass" if grew and band else "partial" if grew or band else "miss"
    return {"status": status, "long": long, "short": short, "grew": grew, "band": band}


def paired(res: Results, a: str, b: str) -> np.ndarray:
    """Per seed, *a*'s lean less *b*'s: the arms share their windows and labels at a seed."""
    return res.lean(a) - res.lean(b)


def h5(res: Results) -> dict:
    tied = paired(res, "whole-tied", "whole")
    mask = paired(res, "whole-mask", "whole")
    return {
        "status": "pass" if tied.mean() < 0 else "miss",
        "tied": tied,
        "mask": mask,
        "tied_gap": float(paired(res, "all-tied", "whole-tied").mean()),
        "untied_gap": float(paired(res, "all", "whole").mean()),
    }


def rule(res: Results) -> dict:
    """The rule for the pilot, applied as frozen (`ex.ADOPTION`)."""
    v1, v2 = h1(res)["status"], h2(res)["arms"]
    cand: dict[str, dict] = {}
    chosen = "all"
    if v1 in ("pass", "partial"):
        bar = ex.SCALED_SHARE * removed(res, "whole")
        cand["scaled"] = {"removed": removed(res, "scaled"), "bar": bar, "h2": v2["scaled"]["ok"]}
        cand["scaled"]["ok"] = cand["scaled"]["removed"] >= bar and cand["scaled"]["h2"]
        for p in ex.MITIGATIONS[1:]:
            lean_ok = in_band(res, p) if v1 == "pass" else removed(res, p) >= ex.PARTIAL_SHARE
            cand[p] = {"removed": removed(res, p), "lean_ok": lean_ok, "h2": v2[p]["ok"], "ok": lean_ok and v2[p]["ok"]}
        if cand["scaled"]["ok"]:
            chosen = "scaled"
        else:
            # Of the two, the one that keeps more of the pull.
            ok = [p for p in ex.MITIGATIONS[1:] if cand[p]["ok"]]
            if ok:
                chosen = max(ok, key=lambda p: ex.pull_share(ARM[p].policy, ARM[p].block))
    elif v1 == "unresolved":
        ok = res.excess("all-short") >= ex.READABLE_LEAN and in_band(res, "whole-short") and v2["whole"]["ok"]
        chosen = "whole" if ok else "all"
    return {"h1": v1, "candidates": cand, "chosen": chosen}


# --- H1: the lean -------------------------------------------------------------------------------

# The dot-plot order: the control, the policies, the halved pair, and the model arms, in groups.
LEAN_ORDER = (ex.CONTROL, *POLICY_ARMS, *SHORT_ARMS, *MODEL_ARMS)


def lean_figure(res: Results) -> str:
    leans = {c: res.lean(c) for c in LEAN_ORDER}
    ctl = res.control_lean()
    alt = f"""
        A dot plot of the lean at the last block, one column per condition in the order
        {", ".join(LEAN_ORDER)}, with one small dot per seed and the seed mean as a larger mark. A grey band
        {ex.LEAN_BAND:g} either side of the control's mean {ctl:.3f} runs across the plot, and a dotted line
        marks ex-2.2.14's value of {ex.REFERENCE_OP1_LEAN:g}. The seed means are
        {", ".join(f"{c} {v.mean():.3f}" for c, v in leans.items())}.
    """
    return lean_draw({c: v.tolist() for c, v in leans.items()}, ctl, alt)


@memo
def lean_draw(leans: dict, ctl: float, alt_text: str) -> str:
    @themed(
        name="h1-lean",
        alt_text=alt_text,
        caption=f"""
            **The lean at the last block, per condition.** The mean cosine with e₁ at the first operand, over
            every op's probe lines, one small dot per seed and the seed mean as the large mark. The grey band
            is the control's seed mean ± {ex.LEAN_BAND:g}, the band `whole` has to land in for H1 to pass; the
            dotted line is the primary of ex-2.2.14 ({ex.REFERENCE_OP1_LEAN:g}). The groups are the
            policies, the halved windows, and the model arms.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.4, 3.4), layout="constrained")
        control_band(ax, ctl, ex.LEAN_BAND)
        ax.axhline(ex.REFERENCE_OP1_LEAN, color=ink("all"), lw=0.8, ls=":", zorder=1)
        ax.text(len(leans) - 0.5, ex.REFERENCE_OP1_LEAN, "ex-2.2.14 ", fontsize=7, ha="right", va="bottom")
        rng = np.random.default_rng(1)
        for i, (c, v) in enumerate(leans.items()):
            dots(ax, i, np.asarray(v), c, rng=rng)
        for edge in (len(POLICY_ARMS) + 0.5, len(POLICY_ARMS) + len(SHORT_ARMS) + 0.5):
            ax.axvline(edge, color=light_dark("#999", "#555"), lw=0.6, zorder=0)
        arm_axis(ax, list(leans))
        ax.set_xlim(-0.6, len(leans) - 0.4)
        ax.set_ylabel("lean (cos with e₁ at op1)")
        return fig

    return _plot()


def slices_figure(res: Results) -> str:
    conds = (ex.CONTROL, *POLICY_ARMS)
    per = {c: res.cos(c)[:, :, 0].mean(0).tolist() for c in conds}
    alt = f"""
        A line chart of the lean at each slice, from the embedding (slice 0) to the last block (slice
        {LAST}), one line per condition: {", ".join(conds)}. At the last block the lines end at
        {", ".join(f"{c} {v[-1]:.3f}" for c, v in per.items())}.
    """
    return slices_draw(per, alt)


@memo
def slices_draw(per: dict, alt_text: str) -> str:
    @themed(
        name="h1-slices",
        alt_text=alt_text,
        caption="""
            **The lean at each slice.** The seed mean of the cosine with e₁ at the first operand, over every
            op's probe lines, from the embedding (slice 0) to the last block, for the control and each policy.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.6, 3.2), layout="constrained")
        for c, v in per.items():
            ls = "--" if c == ex.CONTROL else "-"
            ax.plot(range(len(v)), v, ls, color=ink(c), marker=MARKERS.get(c, "o"), ms=4, lw=1.2, label=c)
        ax.set_xticks(range(LAST + 1))
        ax.set_xlabel("slice (0 = embedding)")
        ax.set_ylabel("lean (cos with e₁ at op1)")
        ax.legend(fontsize=7, ncol=2, frameon=False)
        return fig

    return _plot()


def traj_figure(res: Results) -> str:
    conds = POLICY_ARMS
    epochs, _ = res.lean_traj("all")
    lean = {c: res.lean_traj(c)[1].tolist() for c in conds}
    frag = {c: res.fragment_traj(c).tolist() for c in conds}
    weight = np.asarray(res.trajectories("all")[0]["weight"]).tolist()
    ctl, ctl_frag = res.control_lean(), float(res.fragment(ex.CONTROL).mean())
    alt = f"""
        Two stacked line charts sharing an epoch axis from 0 to {epochs[-1]:.0f}. Top: the lean at the last
        block through training, one thick line per policy ({", ".join(conds)}) for the seed mean and a
        hairline per seed, over a grey silhouette of the anchor weight's schedule, with a dashed line at the
        control's end-of-training lean {ctl:.3f}. Bottom: the same for the trailing-fragment lean, against
        the control's {ctl_frag:.3f}. At the end the seed-mean lean is
        {", ".join(f"{c} {np.mean(v, axis=0)[-1]:.3f}" for c, v in lean.items())}.
    """
    return traj_draw(epochs.tolist(), lean, frag, weight, ctl, ctl_frag, alt)


@memo
def traj_draw(epochs: list, lean: dict, frag: dict, weight: list, ctl: float, ctl_frag: float, alt_text: str) -> str:
    @themed(
        name="h1-trajectory",
        alt_text=alt_text,
        caption=f"""
            **The lean through training.** Top: the lean at the last block on the trajectory probe lines,
            recorded every {ex.TRAJ_STRIDE} steps; the thick line is the seed mean and the hairlines are the
            seeds. The grey silhouette is the anchor weight's schedule (right-hand scale, relative to its
            peak). Bottom: the trailing-fragment lean (H3), pooled over ops. The dashed lines are the
            control's values at the end of training; the control records no trajectory.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 5.0), layout="constrained", sharex=True)
        w = np.asarray(weight)
        tw = a.twinx()
        tw.fill_between(epochs, w / w.max(), color=light_dark("#000", "#fff"), alpha=0.07, lw=0, step="post")
        tw.set_ylim(0, 1.05)
        tw.set_ylabel("anchor weight (rel.)", fontsize=7)
        tw.tick_params(labelsize=7)
        a.set_zorder(tw.get_zorder() + 1)
        a.patch.set_visible(False)
        for ax, series, c0 in ((a, lean, ctl), (b, frag, ctl_frag)):
            ax.axhline(c0, color=ink(ex.CONTROL), lw=0.8, ls="--", zorder=1)
            for c, v in series.items():
                v = np.asarray(v)
                for s in v:
                    ax.plot(epochs, s, color=ink(c), lw=0.5, alpha=0.3, zorder=2)
                ax.plot(epochs, v.mean(0), color=ink(c), lw=1.6, zorder=3, label=c)
        a.set_ylabel("lean (op1)")
        b.set_ylabel("trailing-fragment lean")
        b.set_xlabel("epoch")
        a.legend(fontsize=7, ncol=3, frameon=False, loc="upper left")
        return fig

    return _plot()


def h1_table(res: Results) -> str:
    head = ["condition", "lean", "± sd", "excess over control", "share of `all`'s excess", "pull kept"]
    rows = [[f"`{ex.CONTROL}`", f"{res.control_lean():.3f}", f"{sd(res.lean(ex.CONTROL)):.3f}", "–", "–", "–"]]
    for c in LEAN_ORDER[1:]:
        a = ARM[c]
        e = res.excess(c)
        rows.append(
            [
                f"`{c}`",
                f"{res.lean(c).mean():.3f}",
                f"{sd(res.lean(c)):.3f}",
                bold_if(f"{e:+.3f}", abs(e) <= ex.LEAN_BAND),
                f"{e / res.excess('all'):.0%}",
                f"{ex.pull_share(a.policy, a.block):.0%}",
            ]
        )
    return table_html(
        head,
        rows,
        f"The lean at the last block per condition: the seed mean and its seed-to-seed spread, its excess over "
        f"the control (bold within the band of ±{ex.LEAN_BAND:g}), that excess as a share of `all`'s, and the "
        f"share of the pull the policy keeps. The halved pair's shares are against `all`, at the long window.",
        ref_rows=frozenset({0}),
    )


# --- H2: the anchor and the task -----------------------------------------------------------------


def h2_figure(res: Results) -> str:
    v = h2(res)["arms"]
    share = {c: (res.margin(c) / res.margin("all").mean()).tolist() for c in ARM_NAMES}
    gap = {c: v[c]["gap"] for c in ARM_NAMES}
    alt = f"""
        Two panels, each with the {len(ARM_NAMES)} arms along the bottom. Left: the op margin as a share of
        the seed-mean margin under `all`, one dot per seed and the seed mean as the large mark, with a dashed
        line at {ex.MARGIN_KEEP:.0%} and the region below it hatched. The seed means are
        {", ".join(f"{c} {np.mean(s):.2f}" for c, s in share.items())}. Right: the largest gap in held-out
        expected exact match from the control over the eleven ops, one bar per arm, with dashed lines at
        ±{ex.TASK_GATE:g} and the regions beyond them hatched. The gaps are
        {", ".join(f"{c} {g:+.3f}" for c, g in gap.items())}.
    """
    return h2_draw(share, gap, alt)


@memo
def h2_draw(share: dict, gap: dict, alt_text: str) -> str:
    @themed(
        name="h2-gates",
        alt_text=alt_text,
        caption=f"""
            **The anchor and the task, per arm.** Left: the op margin as a share of the seed-mean margin
            under `all`, one small dot per seed and the seed mean as the large mark; H2 asks for at least
            {ex.MARGIN_KEEP:.0%} (dashed; hatched below). Right: the gap in held-out expected exact match from
            the control, for the op furthest from it; H2 asks for at most {ex.TASK_GATE:g} either way
            (dashed; hatched beyond).
        """,
    )
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained")
        rng = np.random.default_rng(2)
        for i, (c, v) in enumerate(share.items()):
            dots(a, i, np.asarray(v), c, rng=rng)
        lo = min(ex.MARGIN_KEEP - 0.05, min(min(v) for v in share.values()) - 0.02)
        hi = max(1.1, max(max(v) for v in share.values()) + 0.02)
        a.set_ylim(lo, hi)
        gate_line(a, ex.MARGIN_KEEP, fail="below")
        a.set_ylabel("op margin / margin under all")
        g = np.array(list(gap.values()))
        b.bar(range(len(gap)), g, color=[ink(c) for c in gap], width=0.6, zorder=3)
        b.axhline(0, color=light_dark("#333", "#ddd"), lw=0.6)
        span = max(ex.TASK_GATE * 1.6, float(np.abs(g).max()) * 1.15)
        b.set_ylim(-span, span)
        for y in (ex.TASK_GATE, -ex.TASK_GATE):
            b.axhline(y, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
        for lo_, hi_ in ((ex.TASK_GATE, span), (-span, -ex.TASK_GATE)):
            b.axhspan(lo_, hi_, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, alpha=0.1)
        b.set_ylabel("largest task gap (EEM)")
        for ax, conds in ((a, share), (b, gap)):
            arm_axis(ax, list(conds))
        return fig

    return _plot()


def h2_table(res: Results) -> str:
    v = h2(res)["arms"]
    head = ["arm", "op margin", "share of `all`'s", "op of largest gap", "gap", "pull kept", "H2"]
    rows = []
    for c in ARM_NAMES:
        q, a = v[c], ARM[c]
        rows.append(
            [
                f"`{c}`",
                f"{res.margin(c).mean():.3f}",
                bold_if(f"{q['share']:.2f}", q["share"] >= ex.MARGIN_KEEP),
                f"`{q['op']}`",
                bold_if(f"{q['gap']:+.4f}", abs(q["gap"]) <= ex.TASK_GATE),
                f"{ex.pull_share(a.policy, a.block):.0%}",
                "pass" if q["ok"] else "miss",
            ]
        )
    return table_html(
        head,
        rows,
        f"The two gates of H2 per arm, on the seed means: the op margin and its share of the margin under "
        f"`all` (bold at or above {ex.MARGIN_KEEP:.0%}), and the op whose held-out expected exact match is "
        f"furthest from the control, with its gap (bold within {ex.TASK_GATE:g}).",
    )


# --- H3: trailing fragments ----------------------------------------------------------------------

FRAG_ORDER = (ex.CONTROL, *POLICY_ARMS, *SHORT_ARMS, *MODEL_ARMS)


def h3_figure(res: Results) -> str:
    lean = {c: res.fragment(c).tolist() for c in FRAG_ORDER}
    contrast = {c: res.fragment_contrast(c).tolist() for c in FRAG_ORDER}
    alt = f"""
        Two panels, each with the control and the {len(ARM_NAMES)} arms along the bottom, one small dot per
        seed and the seed mean as a larger mark, with a grey band around the control's mean. Left: the
        trailing-fragment lean; the seed means are {", ".join(f"{c} {np.mean(v):.3f}" for c, v in lean.items())}.
        Right: the contrast between the `{OP}` trailing fragments and the rest; the seed means are
        {", ".join(f"{c} {np.mean(v):+.3f}" for c, v in contrast.items())}.
    """
    return h3_draw(lean, contrast, alt)


@memo
def h3_draw(lean: dict, contrast: dict, alt_text: str) -> str:
    @themed(
        name="h3-fragments",
        alt_text=alt_text,
        caption=f"""
            **Trailing fragments at the last block.** Left: the mean cosine with e₁ over every position of
            every trailing fragment of every op. Right: the same over the `{OP}` fragments less the mean over
            the rest. One small dot per seed, the seed mean as the large mark; the grey band is the control's
            seed mean ± {ex.LEAN_BAND:g}, for scale.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.3), layout="constrained")
        rng = np.random.default_rng(4)
        for ax, series in ((a, lean), (b, contrast)):
            control_band(ax, float(np.mean(series[ex.CONTROL])), ex.LEAN_BAND)
            for i, (c, v) in enumerate(series.items()):
                dots(ax, i, np.asarray(v), c, rng=rng)
            arm_axis(ax, list(series))
            ax.set_xlim(-0.6, len(series) - 0.4)
        a.set_ylabel("trailing-fragment lean")
        b.set_ylabel(f"{OP} fragments − the rest")
        return fig

    return _plot()


def roles_figure(res: Results) -> str:
    conds = (ex.CONTROL, "all", "whole", "knowable")
    prof = {c: {s: v.tolist() for s, v in res.fragment_roles(c).items()} for c in conds}
    alt = f"""
        {len(ex.FRAGMENT_START_ROLES)} small line charts side by side, one per trailing fragment, starting at
        {", ".join(ex.ROLES[s] for s in ex.FRAGMENT_START_ROLES)}. Each has the fragment's positions along the
        bottom and the seed-mean cosine with e₁ at the last block up the side, one line per condition:
        {", ".join(conds)}. The syntax tokens `=` and ⏎ are shaded.
    """
    return roles_draw({c: {str(s): v for s, v in p.items()} for c, p in prof.items()}, alt)


@memo
def roles_draw(prof: dict, alt_text: str) -> str:
    @themed(
        name="h3-fragment-roles",
        alt_text=alt_text,
        caption="""
            **Where a trailing fragment leans.** The seed-mean cosine with e₁ at the last block at each
            position of a trailing fragment, pooled over ops, one panel per role the fragment starts at. The
            shaded columns are the syntax tokens `=` and ⏎, which carry no color.
        """,
    )
    def _plot() -> plt.Figure:
        starts = list(next(iter(prof.values())))
        fig, axes = plt.subplots(1, len(starts), figsize=(8.4, 2.6), layout="constrained", sharey=True)
        for ax, s in zip(axes, starts, strict=True):
            roles = ex.ROLES[int(s) :]
            for j, r in enumerate(roles):
                if ex.ROLES.index(r) in SYNTAX_ROLES:
                    ax.axvspan(j - 0.4, j + 0.4, color=light_dark("#000", "#fff"), alpha=0.06, lw=0)
            for c, p in prof.items():
                ls = "--" if c == ex.CONTROL else "-"
                ax.plot(range(len(roles)), p[s], ls, color=ink(c), marker=MARKERS.get(c, "o"), ms=3.5, lw=1.1, label=c)
            ax.set_xticks(range(len(roles)), roles, fontsize=8)
            ax.set_xlim(-0.5, max(len(roles) - 0.5, 1.5))
            ax.set_title(f"from {ex.ROLES[int(s)]}", fontsize=8)
        axes[0].set_ylabel("cos with e₁")
        axes[-1].legend(fontsize=7, frameon=False)
        return fig

    return _plot()


def h3_table(res: Results) -> str:
    head = ["condition", "trailing-fragment lean", "± sd", f"`{OP}` less the rest", "± sd"]
    rows = []
    for c in FRAG_ORDER:
        f, k = res.fragment(c), res.fragment_contrast(c)
        rows.append([f"`{c}`", f"{f.mean():.3f}", f"{sd(f):.3f}", f"{k.mean():+.3f}", f"{sd(k):.3f}"])
    return table_html(
        head,
        rows,
        "The trailing fragments at the last block per condition: the lean pooled over every op, and the "
        f"contrast of the `{OP}` fragments with the rest, each as a seed mean with its seed-to-seed spread.",
        ref_rows=frozenset({0}),
    )


# --- H4: halved windows --------------------------------------------------------------------------


def h4_figure(res: Results) -> str:
    x = {"all": CUT, "whole": CUT, "all-short": CUT_SHORT, "whole-short": CUT_SHORT}
    lean = {c: res.lean(c).tolist() for c in x}
    ctl = res.control_lean()
    alt = f"""
        A dot plot with the share of cut line visits along the bottom, {CUT:.0%} for the
        {ex.BLOCK}-token window and {CUT_SHORT:.0%} for the {ex.SHORT_BLOCK}-token one, and the lean at the
        last block up the side. At each window size, `all` and `whole` sit side by side with a line joining
        their seed means; one small dot per seed. A grey band marks the control's {ctl:.3f} ± {ex.LEAN_BAND:g}.
        The seed means are {", ".join(f"{c} {np.mean(v):.3f}" for c, v in lean.items())}.
    """
    return h4_draw(x, lean, ctl, alt)


@memo
def h4_draw(x: dict, lean: dict, ctl: float, alt_text: str) -> str:
    @themed(
        name="h4-halved",
        alt_text=alt_text,
        caption=f"""
            **The lean against the share of cut line visits.** At each window size, `all` (left) and
            `whole` (right), one small dot per seed and the seed mean as the large mark, with a line joining
            the two means: its drop is the lean the cut lines carry. The grey band is the control's seed mean
            ± {ex.LEAN_BAND:g}; the control trains at {ex.BLOCK} tokens only.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        control_band(ax, ctl, ex.LEAN_BAND)
        rng = np.random.default_rng(5)
        off = 0.012
        for pair in (("all", "whole"), ("all-short", "whole-short")):
            xs = [x[pair[0]] - off, x[pair[1]] + off]
            ax.plot(xs, [np.mean(lean[c]) for c in pair], "-", color=light_dark("#555", "#bbb"), lw=0.9, zorder=2)
            for xx, c in zip(xs, pair, strict=True):
                dots(ax, xx, np.asarray(lean[c]), c, rng=rng, width=0.003, label=c)
        ax.set_xticks(
            [x["all"], x["all-short"]],
            [f"{x['all']:.0%}\n({ex.BLOCK} tokens)", f"{x['all-short']:.0%}\n({ex.SHORT_BLOCK} tokens)"],
        )
        ax.set_xlim(x["all"] - 0.06, x["all-short"] + 0.06)
        ax.set_xlabel("share of line visits cut short")
        ax.set_ylabel("lean (cos with e₁ at op1)")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        return fig

    return _plot()


def h4_table(res: Results) -> str:
    v = h2(res)["arms"]
    head = ["arm", "window", "lean", "excess over control", "gap within the pair", "op margin", "largest task gap"]
    rows = []
    for a_, w_ in (("all", "whole"), ("all-short", "whole-short")):
        gap = float(res.lean(a_).mean() - res.lean(w_).mean())
        for c in (a_, w_):
            rows.append(
                [
                    f"`{c}`",
                    str(ARM[c].block),
                    f"{res.lean(c).mean():.3f}",
                    bold_if(f"{res.excess(c):+.3f}", in_band(res, c)),
                    f"{gap:+.3f}" if c == a_ else "",
                    f"{res.margin(c).mean():.3f}",
                    bold_if(f"{v[c]['gap']:+.4f} (`{v[c]['op']}`)", abs(v[c]["gap"]) <= ex.TASK_GATE),
                ]
            )
    return table_html(
        head,
        rows,
        f"The two pairs, on the seed means: the lean and its excess over the control (bold within "
        f"±{ex.LEAN_BAND:g}), the gap between `all` and `whole` at each window size, the op margin, and the "
        f"largest task gap from the control (bold within {ex.TASK_GATE:g}).",
    )


# --- H5: the model arms --------------------------------------------------------------------------

H5_ORDER = ("all", "whole", "whole-mask", "whole-tied", "all-tied")


def h5_figure(res: Results) -> str:
    lean = {c: res.lean(c).tolist() for c in H5_ORDER}
    ctl = res.control_lean()
    alt = f"""
        A dot plot of the lean at the last block for {", ".join(H5_ORDER)}, one dot per seed with the seed
        mean as a larger mark. Thin lines join each seed's dots across the columns, since the arms share
        their windows and labels at a seed. A grey band marks the control's {ctl:.3f} ± {ex.LEAN_BAND:g}. The
        seed means are {", ".join(f"{c} {np.mean(v):.3f}" for c, v in lean.items())}.
    """
    return h5_draw(lean, ctl, alt)


@memo
def h5_draw(lean: dict, ctl: float, alt_text: str) -> str:
    @themed(
        name="h5-model-arms",
        alt_text=alt_text,
        caption=f"""
            **The lean under the model arms, by seed.** The lean at the last block, one dot per seed and the
            seed mean as the large mark. The thin grey lines join the dots of one seed, which trains on the
            same windows and labels in every column. The grey band is the control's seed mean ±
            {ex.LEAN_BAND:g}.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.8, 3.4), layout="constrained")
        control_band(ax, ctl, ex.LEAN_BAND)
        v = np.array(list(lean.values()))  # (conds, seeds)
        for s in v.T:
            ax.plot(range(len(s)), s, "-", color=light_dark("#999", "#666"), lw=0.6, alpha=0.7, zorder=1)
        for i, c in enumerate(lean):
            color, m = ink(c), MARKERS.get(c, "o")
            ax.plot([i] * v.shape[1], v[i], "o", ms=3, color=color, alpha=0.7, mew=0, zorder=3)
            ax.plot(i, v[i].mean(), m, ms=7, color=color, mec=light_dark("white", "#111"), mew=0.6, zorder=4)
        arm_axis(ax, list(lean))
        ax.set_xlim(-0.5, len(lean) - 0.5)
        ax.set_ylabel("lean (cos with e₁ at op1)")
        return fig

    return _plot()


def h5_table(res: Results) -> str:
    v = h2(res)["arms"]
    head = ["arm", "lean", "less `whole`'s, paired", "± sd", "op margin", "largest task gap"]
    rows = []
    for c in ("whole", *MODEL_ARMS):
        d = paired(res, c, "whole")
        rows.append(
            [
                f"`{c}`",
                f"{res.lean(c).mean():.3f}",
                "–" if c == "whole" else f"{d.mean():+.3f}",
                "–" if c == "whole" else f"{sd(d):.3f}",
                f"{res.margin(c).mean():.3f}",
                bold_if(f"{v[c]['gap']:+.4f} (`{v[c]['op']}`)", abs(v[c]["gap"]) <= ex.TASK_GATE),
            ]
        )
    return table_html(
        head,
        rows,
        f"The model arms against `whole`: the seed-mean lean, its difference from `whole` paired by seed (with "
        f"the spread of that difference over seeds), the op margin, and the largest task gap from the control "
        f"(bold within {ex.TASK_GATE:g}).",
        ref_rows=frozenset({0}),
    )


# --- The rule ------------------------------------------------------------------------------------


def rule_table(res: Results) -> str:
    r = rule(res)
    head = ["policy", "share of the excess removed", "the lean test", "H2", "pull kept", "qualifies"]
    rows = []
    for p in ex.MITIGATIONS:
        q = r["candidates"].get(p)
        a = ARM[p]
        if q is None:
            test = "–"
            ok = "–"
        elif p == "scaled":
            test = f"at least {q['bar']:.0%}: {'yes' if q['removed'] >= q['bar'] else 'no'}"
            ok = "yes" if q["ok"] else "no"
        else:
            bar = f"within ±{ex.LEAN_BAND:g}" if r["h1"] == "pass" else f"at least {ex.PARTIAL_SHARE:.0%}"
            test = f"{bar}: {'yes' if q['lean_ok'] else 'no'}"
            ok = "yes" if q["ok"] else "no"
        rows.append(
            [
                bold_if(f"`{p}`", p == r["chosen"]),
                f"{removed(res, p):.0%}",
                test,
                "pass" if h2(res)["arms"][p]["ok"] else "miss",
                f"{ex.pull_share(a.policy, a.block):.0%}",
                ok,
            ]
        )
    return table_html(
        head,
        rows,
        f"The candidates against the rule, in the order it considers them, with H1 at *{r['h1']}*. The share "
        f"removed is of the excess lean of `all` over the control; `scaled`'s bar is {ex.SCALED_SHARE:.0%} of "
        f"what `whole` removes. The rule goes forward with **`{r['chosen']}`** (bold).",
    )


# --- Exploratory ---------------------------------------------------------------------------------


def landing_figure(res: Results) -> str:
    conds = POLICY_ARMS
    share = {c: res.landing(c).mean(0)[LAST].tolist() for c in conds}
    alt = f"""
        A grid with the {len(conds)} policies as rows and the six roles of a line ({", ".join(ex.ROLES)}) as
        columns; each cell is shaded and labelled with the share of the pull on a labelled `{OP}` line that
        lands on that role at the last block, averaged over the ways a window shows the line. At the first
        operand the shares are {", ".join(f"{c} {v[0]:.2f}" for c, v in share.items())}.
    """
    return landing_draw(share, alt)


@memo
def landing_draw(share: dict, alt_text: str) -> str:
    @themed(
        name="x-landing",
        alt_text=alt_text,
        caption=f"""
            **Where the pull lands (post hoc; planned, no gate).** The share of the pull on a labelled `{OP}`
            line that goes to each role at the last block, averaged over the ways a window shows the line,
            per policy. A row sums to the pull the policy keeps, relative to a line under `all`.
        """,
    )
    def _plot() -> plt.Figure:
        m = np.array(list(share.values()))
        fig, ax = plt.subplots(figsize=(5.8, 2.9), layout="constrained")
        ax.imshow(m, cmap="Blues", vmin=0, vmax=max(float(m.max()), 1e-6), aspect="auto")
        for (i, j), val in np.ndenumerate(m):
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="#000" if val < 0.6 * m.max() else "#fff",
            )
        ax.set_xticks(range(len(ex.ROLES)), ex.ROLES, fontsize=8)
        ax.set_yticks(range(len(share)), list(share), fontsize=8)
        ax.grid(False)
        return fig

    return _plot()


def whole_roles_figure(res: Results) -> str:
    conds = (ex.CONTROL, "all", "whole", "cut-only", "knowable")
    per = {c: res.cos(c).mean(0)[LAST].tolist() for c in conds}
    alt = f"""
        A line chart with the six roles of a line along the bottom and the seed-mean cosine with e₁ at the
        last block up the side, over every op's whole probe lines, one line per condition: {", ".join(conds)}.
    """
    return whole_roles_draw(per, alt)


@memo
def whole_roles_draw(per: dict, alt_text: str) -> str:
    @themed(
        name="x-whole-roles",
        alt_text=alt_text,
        caption="""
            **Each role on whole lines (post hoc; planned, no gate).** The seed-mean cosine with e₁ at the
            last block at each role, over every op's probe lines, which are whole lines.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.6, 3.0), layout="constrained")
        for c, v in per.items():
            ls = "--" if c == ex.CONTROL else "-"
            ax.plot(range(len(v)), v, ls, color=ink(c), marker=MARKERS.get(c, "o"), ms=4, lw=1.2, label=c)
        ax.set_xticks(range(len(ex.ROLES)), ex.ROLES)
        ax.set_ylabel("cos with e₁")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        return fig

    return _plot()


def drift(res: Results, cond: str) -> np.ndarray:
    """Per seed, the op margin at the anneal start less its peak before it: how far it drifted down first."""
    out = []
    for r, t in zip(res.runs(cond), res.trajectories(cond), strict=True):
        ep, m = np.asarray(t["epoch"]), np.asarray(t["m_line"])
        before = m[ep <= r["anneal_epoch"]]
        out.append(float(before[-1] - before.max()) if len(before) else float("nan"))
    return np.array(out)


def readout_table(res: Results) -> str:
    head = [
        "condition",
        "syntax-word log-odds at op1",
        "of it on e₁",
        "⏎ embedding on e₁",
        "margin drift before anneal",
    ]
    rows = []
    for c in (ex.CONTROL, "all", "whole", "whole-tied", "all-tied"):
        lo, le, nl = res.readout(c, "logodds_op1"), res.readout(c, "logodds_op1_e1"), res.newline_e1(c)
        dr = "–" if c == ex.CONTROL else f"{np.nanmean(drift(res, c)):+.3f}"
        rows.append([f"`{c}`", f"{lo.mean():.2f}", f"{le.mean():+.2f}", f"{nl.mean():+.3f}", dr])
    return table_html(
        head,
        rows,
        "The tied readout against the untied (post hoc; planned, no gate), seed means. The log-odds of a syntax "
        "word against a color at the first operand, and the part of them that rides on e₁ (the drop when e₁ is "
        "removed from the state, as in the op1-lean reanalysis); the e₁ component of the ⏎ embedding row; and "
        "the op margin at the start of the anneal less its peak before it, on the trajectory.",
        ref_rows=frozenset({0}),
    )


# --- The report --------------------------------------------------------------------------------

if metrics_loaded is None or traj_loaded is None:
    stop("The run has not published its results yet.")

res = Results(metrics=metrics_loaded, traj=traj_loaded)


def traj_summary(res: Results) -> dict:
    """What the trajectories say about H1's descriptive prediction, over the plateau of the anchor weight (from
    the end of its ramp to the start of its anneal), so the warm-up every arm shares stays out of it: `all`'s
    lean over each half of the plateau and at the end, and how far `whole`'s strays from the control's
    end-of-training value from the plateau on.
    """
    epochs, lean_all = res.lean_traj("all")
    w = np.asarray(res.trajectories("all")[0]["weight"])
    on = np.flatnonzero(w >= w.max())
    start, stop_ = int(on[0]), int(on[-1])
    mid = (start + stop_) // 2
    m = lean_all.mean(0)
    _, lean_whole = res.lean_traj("whole")
    ex_whole = lean_whole.mean(0)[start:] - res.control_lean()
    k = int(np.argmax(np.abs(ex_whole)))
    return {
        "plateau": (float(epochs[start]), float(epochs[stop_])),
        "all_first": float(m[start:mid].mean()),
        "all_second": float(m[mid : stop_ + 1].mean()),
        "all_end": float(m[-1]),
        "whole_max_excess": float(ex_whole[k]),
        "whole_max_epoch": float(epochs[start + k]),
    }


def fragment_peak(res: Results, cond: str) -> tuple[str, str, float]:
    """The fragment start and position where *cond*'s seed-mean trailing-fragment lean is largest."""
    best = max(
        ((s, j, float(v)) for s, p in res.fragment_roles(cond).items() for j, v in enumerate(p)), key=lambda t: t[2]
    )
    s, j, v = best
    return ex.ROLES[s], ex.ROLES[s + j], v


def numbers(res: Results) -> dict:
    """The figures the prose quotes, gathered once so every sentence reads the same data as its table."""
    v2 = h2(res)
    shares = {c: a["share"] for c, a in v2["arms"].items()}
    gaps = {c: a["gap"] for c, a in v2["arms"].items()}
    worst = max(gaps, key=lambda c: abs(gaps[c]))
    return {
        "h1": h1(res),
        "h2": v2,
        "h3": h3(res),
        "h4": h4(res),
        "h5": h5(res),
        "rule": rule(res),
        "traj": traj_summary(res),
        "ctl": res.control_lean(),
        "lean": {c: float(res.lean(c).mean()) for c in ARM_NAMES},
        "excess": {c: res.excess(c) for c in ARM_NAMES},
        "removed": {c: removed(res, c) for c in ARM_NAMES},
        "min_share": min(shares.values()),
        "min_share_arm": min(shares, key=lambda c: shares[c]),
        "worst_gap": gaps[worst],
        "worst_gap_arm": worst,
        "worst_gap_op": v2["arms"][worst]["op"],
        "contrast": {c: float(res.fragment_contrast(c).mean()) for c in (*ARM_NAMES, ex.CONTROL)},
        "frag_peak": fragment_peak(res, "all"),
    }


N = numbers(res)
V = {k: N[k]["status"] for k in ("h1", "h2", "h3", "h4", "h5")}


def pct(x: float) -> str:
    return f"{x:.0%}"


def h1_line() -> str:
    q = N["h1"]
    if q["status"] == "unresolved":
        return f"the lean of `all` exceeds the control's by {q['excess']:.3f}, under the readable {ex.READABLE_LEAN:g}."
    return (
        f"`whole` removes {pct(q['whole_removed'])} of the excess lean of `all` "
        f"({q['whole_excess']:+.3f} from the control, band ±{ex.LEAN_BAND:g}), and `cut-only` keeps "
        f"{pct(q['cut_kept'])} of it (pass at {pct(ex.CUT_ONLY_SHARE)}; partial at {pct(ex.PARTIAL_SHARE)} each)."
    )


def h2_line() -> str:
    q = N["h2"]

    def why(c: str) -> str:
        a = q["arms"][c]
        gates = [f"margin {a['share']:.2f} of `all`'s"] if a["share"] < ex.MARGIN_KEEP else []
        gates += [f"task gap {a['gap']:+.4f} on `{a['op']}`"] if abs(a["gap"]) > ex.TASK_GATE else []
        return f"`{c}` ({', '.join(gates)})"

    head = (
        "every arm clears both gates"
        if not q["failed"]
        else f"{len(q['failed'])} of {len(q['arms'])} arms miss: " + "; ".join(why(c) for c in q["failed"])
    )
    return (
        f"{head}. The gates are a margin of at least {ex.MARGIN_KEEP:g} of `all`'s and a task gap within "
        f"{ex.TASK_GATE:g}; over every arm, the lowest margin is {N['min_share']:.2f} (`{N['min_share_arm']}`) and "
        f"the largest gap {N['worst_gap']:+.4f} (`{N['worst_gap_arm']}` on `{N['worst_gap_op']}`)."
    )


def h3_line() -> str:
    f = N["h3"]["lean"]
    return (
        f"the trailing-fragment lean is {f['all']:.3f} under `all`, against {f['whole']:.3f} under `whole`, "
        f"{f['knowable']:.3f} under `knowable`, and {f[ex.CONTROL]:.3f} on the control."
    )


def h4_line() -> str:
    q = N["h4"]
    return (
        f"the gap between `all` and `whole` is {q['short']:+.3f} at {ex.SHORT_BLOCK} tokens against "
        f"{q['long']:+.3f} at {ex.BLOCK}, and `whole-short` is {N['excess']['whole-short']:+.3f} from the "
        f"control (band ±{ex.LEAN_BAND:g})."
    )


def h5_line() -> str:
    q = N["h5"]
    return (
        f"`whole-tied` leans {q['tied'].mean():+.3f} against `whole`, paired by seed, and `whole-mask` "
        f"{q['mask'].mean():+.3f}."
    )


VERDICT_WORD = {"pass": "held", "partial": "partly held", "miss": "did not hold", "unresolved": "unresolved"}


def finding(h: str) -> str:
    return f"**{VERDICT_WORD[V[h]]}**"


rf"""
# Ex 2.2.15: lines cut short by the training window

/// tip |
<!-- tl;dr -->
The anchor pulls on whole lines. But in training, the model sees the corpus through a window, and each window cuts off the lines at its edges. On a cut line, the anchor asks the visible part to carry the whole label, even when that part cannot see the op word. We retrain the anchored model under a few policies for which cut lines to pull, and track what each one does over the course of training. We do this on the current grammar, before the in-context grammar makes the problem larger.
///

This is a scouting run of {ex.N_RUNS} fresh training runs, with a few predictions and one rule: it proposes the crop policy the [in-context grammar pilot](../d2.2/design.md#the-pilot) starts from. Cut lines are a small share of any one batch, but the anchor meets them at every step, so the question is what they add up to over training. Each run records the lean and the trailing-fragment lean at every trajectory point (every {ex.TRAJ_STRIDE} training steps), as well as at the end.

## Findings

- [The first operand lean is due to cut lines (H1)](#the-first-operand-lean-is-due-to-cut-lines-h1) — {finding("h1")}: {h1_line()}
- [The anchor and the task hold under every policy (H2)](#the-anchor-and-the-task-hold-under-every-policy-h2) — {finding("h2")}: {h2_line()}
- [Trailing fragments without their op word (H3)](#trailing-fragments-without-their-op-word-h3) — {finding("h3")}: {h3_line()}
- [Halved windows, more cut lines (H4)](#halved-windows-more-cut-lines-h4) — {finding("h4")}: {h4_line()}
- [Where the rest of the lean comes from (H5)](#where-the-rest-of-the-lean-comes-from-h5) — {finding("h5")}: {h5_line()}

[The rule for the pilot](#the-rule-for-the-pilot): the pilot starts from **`{N["rule"]["chosen"]}`**.

## How to read this draft

The policies, the five predictions, and the rule were fixed before any run, at commit `5334cb9`. Everything after that commit is either results filled into their sections or exploratory work, marked as post hoc.

Each result section opens with what we expect, then what we saw. The interpretation and the discussion are still to come.

## Why this experiment

The training corpus is one long tape of six-token lines. Each training step cuts a batch of {ex.BATCH} windows of {ex.BLOCK} tokens from it, each at a random place. Most lines in a window are whole, but its two edges usually slice through a line.

The anchor does not know about the edges. It asks each labelled line to align with the axis at whichever visible position aligns most easily. So when a line is cut, the visible part alone has to carry the whole label.

In these runs the anchor targets the op `{ex.ANCHORED_OP}`, as in the primary arm of ex-2.2.14. If a `{ex.ANCHORED_OP}` line shows only its first operand, the anchor asks a color token to say "this line is `{ex.ANCHORED_OP}`" before the op word has appeared. That token cannot know, so the only way the model can satisfy the anchor there is to lean every first operand a little toward the axis.

That is what [ex-2.2.14](../ex-2.2.14/report.py) saw, after the fact. At the last block, the first operand leaned toward e₁ at {ex.REFERENCE_OP1_LEAN:.2f} on the primary arm, against {ex.REFERENCE_OP1_LEAN_CONTROL:.2f} on the control, and the arm that pulls only the op word had no lean. The [op1-lean reanalysis](../op1-lean/report.py) found the same route on the *red* runs, and proposed skipping the pull on cut lines as a test.

Before fixing the predictions, we ran that test as a smoke test on *red*: two seeds each of the conditions `all`, `whole`, and `cut-only`. `whole` took the lean from {SMOKE["all"]:.2f} to {SMOKE["whole"]:.2f}, and `cut-only` kept {SMOKE["cut-only"]:.2f}, against about zero for the control. So on *red*, cut lines carry about a third of the lean and whole lines the rest. The validation loss was the same under every policy, to three decimals.

Where the rest comes from is a question of its own. So two more arms keep `whole` and change the model. One stops attention at each newline; the other ties the readout to the embedding table (the output layer reuses the input embedding weights).

Red is a weaker test than the op, though. On a *red* line the first operand can be red itself, which is evidence the anchor can pull on; on a `{ex.ANCHORED_OP}` line the first operand says nothing about the op.

At {ex.BLOCK} tokens, {CUT:.0%} of line visits are cut short. On {BLIND:.1%} the op word is out of sight, and on {OP1_ONLY:.1%} only the first operand is visible. The shares are small, but the term is normalized per labelled line, so each of those visits gets the pull of a full line.

This matters more after the [D2.2 pivot](../d2.2/pivot.md). There, a line is a context of solved examples about 20 tokens long, and the op is inferred from the examples, so a window that cuts the start of a context can remove the examples that name the op. With fewer lines per window, the share of cut visits roughly doubles, and a cut context has lost evidence, not just one token. The current grammar is the cheap place to see what each policy does, with a clear sign (the lean) to watch.

## Glossary

<dl>
<dt>Cut line</dt>
<dd>A line visit that the training window shows only part of. A line cut at its start has lost its first tokens, so its states differ from those of the whole line. A line cut at its end shows a prefix. Under causal attention (each position sees only earlier positions), those prefix states are the same as in the whole line.</dd>
<dt>Crop policy</dt>
<dd>Which labelled lines the anchor pulls, given what the window shows of them. The label draws are the same under every policy; only the pull changes.</dd>
<dt>Pull kept</dt>
<dd>The share of the current total pull a policy keeps, averaged over every way a window shows a line. It is computed from the sampler, not measured.</dd>
<dt>The lean</dt>
<dd>The mean cosine with e₁ at the first operand, over every op's probe lines, at the last block. The first operand comes before the op word, so any lean there is shared by every line, whatever its op.</dd>
<dt>Trailing fragment</dt>
<dd>A probe line shown from its second operand, <code>=</code>, answer, or newline on, as a sequence of its own: what a window leaves of a line when it cuts off the op word.</dd>
<dt>Op margin</dt>
<dd>As in ex-2.2.14: how much closer the <code>{ex.ANCHORED_OP}</code> lines sit to e₁ than the pool of all eleven ops, at the role where the gap is largest, averaged over slices. It checks that the anchor landed.</dd>
</dl>

## Conditions

{conditions_html()}

Every arm is the primary from ex-2.2.14: `{ex.ANCHORED_OP}` on e₁, labelled at a rate of {ex.LABEL_RATE:g} per line, the pull over the whole line, and the handover recipe. Only the crop policy changes, plus the window for the short pair and the model for the last three.

**Same batches, same labels.** A policy is a weight on the pull of each labelled line, applied after the labels are drawn. So at one seed every {ex.BLOCK}-token arm trains on the same windows with the same labels, and differences between arms at a seed come from the policy. The model arms, `whole-mask`, `whole-tied`, and `all-tied`, share those windows and labels too, so each pairs with `whole` or `all` at a seed. The short-window arms draw differently and are compared with each other.

**A policy only takes pull away.** A line that a policy keeps gets the same pull it had under `all`, because the term still divides by the number of labelled lines with anything visible. We could instead divide by the number of lines the policy keeps, but then every kept pull would grow stronger as the policy drops more, which is the side effect ex-2.1.7 warned about.

The cost of our choice is that the policies differ a little in total pull, so the table gives the share each keeps. `cut-only` keeps only {ex.pull_share("cut-only"):.0%}, and that is intended: `whole` and `cut-only` split the pull of `all` in two, so if the lean follows `cut-only`, it follows the cut lines and not the total.

**The policies.** `whole`, `half`, and `scaled` need nothing but the window, so each would carry over to any grammar. They differ on spans longer than the window, as a natural-language document often is: `whole` never pulls one, `half` stops at twice the window, and `scaled` pulls every span by the share in view.

`knowable` needs to know where the evidence is. Here, that is the op word; in the in-context grammar, it would be the posterior over ops given the tokens so far (label variant (c) in the pivot). It also drops the first operand from the pull on whole lines, as the note in ex-2.2.14 suggested, and is a reference for what knowing the evidence buys.

**The halved windows.** At {ex.SHORT_BLOCK} tokens, {CUT_SHORT:.0%} of line visits are cut, about the share the in-context grammar would have. The batch doubles to {ex.SHORT_BATCH} windows, so a step sees the same number of tokens and the runs take the same steps. `all-short` against `whole-short` asks whether the effect grows with the share of cut lines, and whether `whole` still removes it.

**The model arms.** `whole-mask` and `whole-tied` each keep `whole` and change one thing about the model, to test a route for any lean that `whole` leaves on whole lines. `whole-mask` stops attention at each newline, so a position sees only its own line; on a whole line the first operand then sees nothing but itself. The [pivot](../d2.2/pivot.md#the-proposal) names this mask as optional for the in-context grammar. `whole-tied` ties the readout to the embedding table, as `handover-tied` did in ex-2.2.9. The untied readout arrived with the handover recipe, and the [op1-lean reanalysis](../op1-lean/report.py) found the lean on *red* running through it. `all-tied` pairs with `whole-tied`, so the share of the lean that cut lines carry can be measured under the tied readout as well.

## The first operand lean is due to cut lines (H1)

**What we expect.** `all` reproduces the lean from ex-2.2.14. `whole` removes it: its seed-mean lean is within {ex.LEAN_BAND:g} of the control's. And `cut-only` keeps at least {ex.CUT_ONLY_SHARE:.0%} of the excess of `all` over the control, although it has only {ex.pull_share("cut-only"):.0%} of the pull.

H1 is partial if `whole` removes at least {ex.PARTIAL_SHARE:.0%} of that excess and `cut-only` keeps at least {ex.PARTIAL_SHARE:.0%} of it. Then the cut lines carry a real share of the lean, though not all of it, and the rule can still adopt a policy that improves on `all`.

Why: the pool needs only one position of a line to align. On a whole line the op word aligns far more easily than the first operand, so the first operand gets almost no pull; only visits that show nothing else force pull onto it. `half` and `knowable` also drop those visits, so we expect them to remove the lean. `scaled` keeps a sixth of their pull, so we expect it to remove most of it.

Through training, we expect the lean of `all` to build while the anchor weight is high and persist through the anneal, and the lean of `whole` to stay near the control throughout. This prediction is descriptive, with no gate. If `whole` shows a lean early and sheds it later, or `all` builds its lean only late, that would mean the cut lines matter at a particular stage of training, which a reading taken only at the end would miss.

If `whole` keeps most of the lean, the lean comes from whole lines (perhaps through the readout, as the op1-lean reanalysis found for *red*), and cropping is a side issue on this grammar. If `whole` removes part of the lean but stays outside the band, the cut lines are one route among others, and `cut-only` says how large a share they carry. On *red* the smoke test landed here: `whole` removed {SMOKE_REMOVED:.0%} of the excess, just short of partial. The model arms (H5) test two routes for the rest.

If the lean of `all` is less than {ex.READABLE_LEAN:g} above the control's, it did not reproduce at these seeds, and H1 is unresolved.

<!-- REVIEW: the smoke test on red (two seeds) found `whole` removing 29% of the excess lean and `cut-only` keeping about a third, so on red H1 would miss its pass. The pass gate and CUT_ONLY_SHARE stay as they were, because on a red line the first operand can itself carry the label's evidence, and on a `difference` line it cannot. After Sandy's review of 2543d1b, a partial band (PARTIAL_SHARE) gives a middle result a verdict, and lets the rule adopt a policy that only improves on `all`. Verify: red lands at 29%, just under the partial share, and the share was set with that in view. -->

**What we saw.** `all` leaned {N["lean"]["all"]:.3f} at the last block, against the control's {N["ctl"]:.3f}: an excess of {N["h1"]["excess"]:.3f}, where ex-2.2.14 had {ex.REFERENCE_OP1_LEAN - ex.REFERENCE_OP1_LEAN_CONTROL:.2f}. `whole` removed {pct(N["h1"]["whole_removed"])} of that excess, leaving {N["h1"]["whole_excess"]:+.3f}, {"inside" if in_band(res, "whole") else "outside"} the band of ±{ex.LEAN_BAND:g}. `cut-only`, with {pct(ex.pull_share("cut-only"))} of the pull, kept {pct(N["h1"]["cut_kept"])} of it. Of the other policies, `half` removed {pct(N["removed"]["half"])}, `scaled` {pct(N["removed"]["scaled"])}, and `knowable` {pct(N["removed"]["knowable"])}.

{lean_figure(res)}

{slices_figure(res)}

The anchor weight holds at its peak from epoch {N["traj"]["plateau"][0]:.0f} to epoch {N["traj"]["plateau"][1]:.0f}. Over the first half of that plateau, the lean of `all` averaged {N["traj"]["all_first"]:.3f} on the trajectory probe; over the second half, {N["traj"]["all_second"]:.3f}; and it ended at {N["traj"]["all_end"]:.3f}. From the start of the plateau on, the lean of `whole` came furthest from the control's end value at epoch {N["traj"]["whole_max_epoch"]:.0f}, at {N["traj"]["whole_max_excess"]:+.3f}.

{traj_figure(res)}

{h1_table(res)}

{verdict_md(V["h1"], h1_line())}

## The anchor and the task hold under every policy (H2)

**What we expect.** No policy costs the anchor or the task, and neither do the model arms. The op margin of every arm is at least {ex.MARGIN_KEEP:.0%} of the margin under `all`, and every op's held-out expected exact match is within {ex.TASK_GATE:g} of the control (the gate from ex-2.2.11), on the seed means.

In ex-2.2.14 the margin sat at the op word and saturated early, and whole lines carry the op word, so dropping cut lines should leave it where it was. If the margin falls under `whole`, there are two possible causes. The cut lines may have been doing part of the work of the anchor. Or the policy removed some of the total pull, and λ_a would need scaling up to make up for it. The pull kept roughly tells the two apart: a fall about the size of the pull removed points to λ_a, and a larger one points to the cut lines. Either way the rule checks this gate, since a policy that moves the anchor or the task would not go forward as it stands.

**What we saw.** The lowest op margin was {N["min_share"]:.2f} of the margin under `all`, on `{N["min_share_arm"]}`. The largest task gap from the control was {N["worst_gap"]:+.4f}, on `{N["worst_gap_op"]}` under `{N["worst_gap_arm"]}`.

{h2_figure(res)}

{h2_table(res)}

{verdict_md(V["h2"], h2_line())}

## Trailing fragments without their op word (H3)

**What we expect.** Under `all`, trailing fragments lean toward e₁, and under `whole` and `knowable` they lean less. The number is the mean cosine with e₁ over every position of every trailing fragment and every op, at the last block, against the control. This is a prediction of direction, with no gate.

Why: under `all`, the anchor asks a trailing fragment like `op2 = answer ⏎` to align with nothing to go on. The pool puts the pull where alignment comes most easily, which here is likely the syntax tokens `=` and `⏎`, since they carry no color. So we expect most of the lean to sit there.

On *red*, the smoke test points to the second operand as well: on whole probe lines, the lean there came mostly from cut lines. `whole` took it from {ex.SMOKE_RED_OP2["all"]:.2f} to {ex.SMOKE_RED_OP2["whole"]:.2f}, and `cut-only` kept {ex.SMOKE_RED_OP2["cut-only"]:.2f}. But on *red* the second operand can itself be red, so that may not carry over to the op.

The model can respond with a general lean, as at the first operand, or by guessing the op from what it can see, since some pairs of second operand and answer fit `{ex.ANCHORED_OP}` better than others. So we also report the contrast between the `{ex.ANCHORED_OP}` trailing fragments and the rest, with no prediction.

A positive contrast would mean the model is learning a surface cue for the op: the shortcut the pivot names as a [failure mode](../d2.2/pivot.md#the-anchor-picks-up-a-shortcut), here in miniature.

**What we saw.** Under `all` the trailing fragments leaned {N["h3"]["lean"]["all"]:.3f}, against {N["h3"]["lean"][ex.CONTROL]:.3f} on the control. Under `whole` they leaned {N["h3"]["lean"]["whole"]:.3f}, and under `knowable` {N["h3"]["lean"]["knowable"]:.3f}. The largest lean under `all` sat at `{N["frag_peak"][1]}` in the fragments starting at `{N["frag_peak"][0]}` ({N["frag_peak"][2]:.3f}). The contrast between the `{OP}` fragments and the rest was {N["contrast"]["all"]:+.3f} under `all` and {N["contrast"][ex.CONTROL]:+.3f} on the control.

{h3_figure(res)}

{roles_figure(res)}

{h3_table(res)}

{verdict_md(V["h3"], h3_line())}

## Halved windows, more cut lines (H4)

**What we expect.** Halved windows give twice as many cut visits, so we expect the lean caused by cut lines to grow. That is, the gap between `all-short` and `whole-short` should be larger than the gap between `all` and `whole`. We also expect `whole-short` to remove the lean, bringing it to within {ex.LEAN_BAND:g} of the control. This prediction has no gate of its own; the rule falls back on it if H1 is unresolved.

We compare the gaps within each pair rather than comparing `all-short` with `all`. The doubled batch matches the steps and the anchor updates, but a model trained on {ex.SHORT_BLOCK}-token windows has seen less context per line, and the gap within a pair holds that fixed.

<!-- REVIEW: H4's statistic is the within-pair gap (all-short − whole-short against all − whole). The short arms now double the batch, so the steps and anchor updates match the long arms (Sandy's review of f2f8e41); the within-pair gap stays because the window still changes the context the model learns from. Verify: with the steps matched, a reader could argue for the raw all-short vs all comparison as a second read. -->

Suppose the gap for the halved pair is no larger than the gap for the long pair. Then the lean does not depend on how often the first operand is pulled alone, and the extrapolation to the in-context grammar becomes weaker.

**What we saw.** At {ex.SHORT_BLOCK} tokens the gap between `all` and `whole` was {N["h4"]["short"]:+.3f}, against {N["h4"]["long"]:+.3f} at {ex.BLOCK} tokens, so it {"grew" if N["h4"]["grew"] else "did not grow"}. `whole-short` sat {N["excess"]["whole-short"]:+.3f} from the control, {"inside" if N["h4"]["band"] else "outside"} the band.

{h4_figure(res)}

{h4_table(res)}

{verdict_md(V["h4"], h4_line())}

## Where the rest of the lean comes from (H5)

**What we expect.** Whatever lean `whole` leaves sits on whole lines, and each model arm removes one route it could take. The number is the seed-mean lean of each arm against that of `whole`, paired by seed. These are predictions of direction, with no gate.

`whole-tied` should lean less than `whole`. The op1-lean reanalysis found that with a readout of its own, the model makes e₁ a feature meaning "a syntax word comes next", and a first operand is always followed by one (the op word). A tied readout shares its rows with the embedding table, and on *red* the reanalysis found that feature almost absent on the tied arm.

`whole-mask` is the weaker expectation. On a whole line the first operand follows a newline and can attend to the line before it. If the lean draws on that line, the mask removes it. If the lean is built at the first operand from its own token, the mask changes nothing. We record the direction either way.

`all-tied` against `whole-tied` asks H1's question under the tied readout, with no gate: whether the cut lines still add lean once the readout is tied. If the tied readout removes the lean from whole lines only, the gap between the tied pair should be about the size of the gap between `all` and `whole`.

If H1 passes, `whole` leaves little lean to split, and these arms mostly say whether each change costs the anchor or the task (H2).

**What we saw.** Paired by seed, `whole-tied` leaned {N["h5"]["tied"].mean():+.3f} against `whole` (lower at {int((N["h5"]["tied"] < 0).sum())} of {len(N["h5"]["tied"])} seeds), and `whole-mask` {N["h5"]["mask"].mean():+.3f} (lower at {int((N["h5"]["mask"] < 0).sum())} of {len(N["h5"]["mask"])}). Under the tied readout, `all` leaned {N["h5"]["tied_gap"]:+.3f} above `whole`, against {N["h5"]["untied_gap"]:+.3f} under the untied readout.

{h5_figure(res)}

{h5_table(res)}

{verdict_md(V["h5"], h5_line())}

## The rule for the pilot

> {ex.ADOPTION}

The test for `scaled` is looser than for the others, and stated against `whole`, so it loosens with H1. `whole` may give the cleanest result, but `scaled` is the one that carries to labelled spans of any length, so the rule accepts part of the lean to get it. The model arms do not enter the rule; what they find goes to the discussion.

With H1 at *{N["rule"]["h1"]}*, the rule goes forward with **`{N["rule"]["chosen"]}`**.

{rule_table(res)}

## Exploratory analyses

Anything we think of after seeing the data goes here, marked as post hoc. Three descriptive measurements are planned, with no gate.

**Where the pull lands.** For each arm and slice, the share of the pull on a labelled line that goes to each role, averaged over the ways a window shows a line (as in E7 of op1-lean). This shows what each policy changes about where the anchor is asked to act.

{landing_figure(res)}

**Each role on whole lines.** The mean cosine with e₁ at each role over every op's probe lines, per arm and slice. A line cut before its op word puts its pull on the roles after it, so those roles may lean the way the first operand does, on whole lines as well as on trailing fragments.

{whole_roles_figure(res)}

**The tied readout.** On the tied pair against `all` and `whole`, the measurements the untied readout made hard in earlier runs: the readout gap from the op1-lean reanalysis (how much of the log-odds of a syntax word against a color at the first operand rides on e₁), the component on e₁ of the ⏎ embedding row, and whether the op margin drifts down before the anneal, which ex-2.2.10 saw only with the untied readout.

{readout_table(res)}

## Discussion

/// admonition | TODO
After the results. What we would take to the pilot: which policy, what it costs in labels there (the share of cut contexts is larger, and `whole` drops them all), whether the evidence question needs label variant (c) on top, and whether the newline mask or a tied readout belongs in the recipe.
///

## Method

### The policies

The sampler already knows the visible run of each labelled line: the offset of the window fixes the role of every position, and padding hides a prefix. Each policy turns that run into a weight in [0, 1] on the pooled term of the line (`line_weight` in `experiment.py`).

The denominator stays the count of labelled lines with any visible position, as under `all`, and the label draws are unchanged, so the random stream is consumed identically under every policy. `knowable` also removes the positions before the op word from the pool, so a whole line is pooled over roles 1 to 5.

How often a window shows each run of a line, at each window size, enumerated over every offset and padding length:

{shares_html()}

### The measurements

The lean is the mean cosine with e₁ at role 0 over the probe lines of every op, at slice {ex.FINAL_SLICE}, as ex-2.2.14 measured it. Its excess is measured against the seed mean of the control. The op margin is the one from ex-2.2.14, on the same probe sets. The task is held-out expected exact match per op, against the control from ex-2.2.11.

`whole-mask` is measured with its mask on, since the mask is part of the model; on a probe fed one line at a time it changes nothing.

The trailing fragments are the probe lines of every op, cut to start at roles {", ".join(str(r) for r in ex.FRAGMENT_START_ROLES)} and fed as sequences of their own. The trailing-fragment lean is the mean cosine over every position of every trailing fragment. The contrast is the mean over `{ex.ANCHORED_OP}` trailing fragments less the mean over the rest.

The op margin, the lean, and the trailing-fragment lean are also measured during training, every {ex.TRAJ_STRIDE} steps, on the same probe lines. These measurements run inside the training loop; no checkpoint is kept along the way.

### Budget

{ex.N_RUNS} runs at d64-L4, each as long as a run in ex-2.2.14; the halved pair takes the same steps at twice the windows per step. Ex-2.2.13 trained 160 runs of this size for about fourteen dollars on Modal. Eval adds the trailing-fragment probe to the measurements from ex-2.2.14; no intervention is scored.

### What this experiment does not do

It does not change what the model sees. The [plan in the pivot](../d2.2/pivot.md#the-proposal) for the in-context grammar keeps random crops of the packed corpus, with a block large enough to fit at least two whole contexts, so the first context in a window is usually cut short. Cut lines could be removed altogether with windows that hold whole lines only: each window starts at a line boundary and is padded after the last line that fits, with a longer block so the padding is a small share. Windows that start at a line boundary but are not padded would still cut the last line. Either changes the task data as well as the pull, so it would need a control of its own. A policy from this experiment still matters with padded windows wherever a labelled span can be longer than the window, as a natural-language document can. The newline mask leaves cut lines cut, so `whole-mask` tests it beside a crop policy, and does not replace one.

It also does not test label variant (c) on the in-context grammar; `knowable` is its version on this grammar.
"""
