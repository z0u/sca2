# title: Ex 2.2.12: what the stream holds on hue-hsv, and a small recipe sweep

# The design constants and the refs come from `experiment.py` beside this script (the script's directory
# is on sys.path while it runs).
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from mini.lit import memo, stop
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed
from sca.data.colors import swatch

import experiment as ex


def conditions_html() -> str:
    """The sweep's conditions, one row each, as a compact report table."""
    head = "<tr><th>condition</th><th class=num>seeds</th><th>what changes</th><th class=num>τ</th><th class=num>λ_a</th><th class=num>anti peak</th><th class=num>blocks</th><th>home of <em>red</em></th></tr>"
    rows = [
        f"<tr><td><code>{c.name}</code></td><td class=num>{c.seeds}</td><td>{c.title}</td><td class=num>{c.tau:g}</td>"
        f"<td class=num>{c.lam:g}</td><td class=num>{c.anti_peak:g}×</td><td class=num>{c.n_layer}</td><td>{c.subspace}</td></tr>"
        for c in ex.CONDITIONS
    ]
    return f'<table class="report-table dense"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


# --- Loading ------------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict | None:
    return None if path is None else json.loads(path.read_text())


def read_arrays(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


REFS = [ex.METRICS_REF, ex.PART1_REF, ex.EX2211_METRICS_REF, ex.EX2211_PROBE_REF]
with tempfile.TemporaryDirectory() as _tmp:
    files = fetch(REFS, Path(_tmp))
    metrics_loaded = read_json(files[ex.METRICS_REF])
    part1_loaded = read_arrays(files[ex.PART1_REF])
    ref_loaded = read_json(files[ex.EX2211_METRICS_REF])
    probes_loaded = read_arrays(files[ex.EX2211_PROBE_REF])

SWEEP = [c.name for c in ex.CONDITIONS if c is not ex.REF]
REFERENCE = ex.REF.name
CONTROL = "control"
PLANE = ex.CONTROL_PLANE
OTHER_OPS = tuple(op for op in ex.OP_NAMES if op != ex.MISSED_OP)
MARGIN_GATE = ex.MARGIN_RATIO * ex.REF_M_LINE
PROMOTION_LEVEL = ex.RED_KEPT_GATE - ex.KEPT_BAND


@dataclass(frozen=True)
class Results:
    """Every published result the report reads. `metrics` is this experiment's (the sweep, the control
    checkpoints scored on the plane, and part 1 under `part1`); `ref` is ex-2.2.11's, whose `handover` is
    the reference condition and whose `control` is the task reference; `part1` is the per-line arrays, and
    `probes` ex-2.2.11's probe set, whose tokens name each line's colors.
    """

    metrics: dict
    ref: dict
    part1: dict[str, np.ndarray]
    probes: dict[str, np.ndarray]

    def __memo_key__(self) -> str:
        return f"{len(self.metrics['runs'])}:{len(self.metrics['scores'])}:{sorted(self.part1)!r:.200}"

    def _source(self, cond: str) -> dict:
        return self.ref if cond in ex.REFERENCE_SEEDS else self.metrics

    def runs(self, cond: str) -> list[dict]:
        """The eval records of a condition, in seed order, from whichever experiment trained it."""
        return sorted((r for r in self._source(cond)["runs"] if r["condition"] == cond), key=lambda r: r["seed"])

    def scored(self, cond: str) -> list[dict]:
        return sorted((r for r in self._source(cond)["scores"] if r["condition"] == cond), key=lambda r: r["seed"])

    def n(self, cond: str) -> int:
        return len(self.runs(cond))

    def stat(self, cond: str, key: str) -> np.ndarray:
        """One value per seed of a placement statistic on the `mix` lines."""
        return np.array([r[key] for r in self.runs(cond)], float)

    def stat_slices(self, cond: str, key: str) -> np.ndarray:
        """(seeds, slices) of a per-slice placement statistic on the `mix` lines."""
        return np.array([r["per_op"][ex.PRIMARY_OP][key] for r in self.runs(cond)], float)

    def eem(self, cond: str, op: str) -> np.ndarray:
        return np.array([r["holdout_eem"][op] for r in self.runs(cond)], float)

    def score(self, cond: str, op: str, operator: str | None, key: str, group: str) -> np.ndarray:
        out = []
        for r in self.scored(cond):
            s = r["ops"][op]
            out.append((s["clean"][key] if operator is None else s["operators"][operator][key])[group])
        return np.array(out, float)

    def kept(self, cond: str, op: str, group: str = "removal") -> np.ndarray:
        """The share of clean expected exact match kept under `projection` on a group of one op's lines."""
        return self.score(cond, op, "projection", "kept", group)

    def kept_others(self, cond: str) -> np.ndarray:
        """Per seed, the mean kept share over the ten ops other than the missed one."""
        return np.mean([self.kept(cond, op) for op in OTHER_OPS], axis=0)

    def kept_worst(self, cond: str) -> tuple[str, float]:
        """The op other than the missed one with the highest seed-mean kept share, and that share."""
        means = {op: float(self.kept(cond, op).mean()) for op in OTHER_OPS}
        op = max(means, key=lambda o: means[o])
        return op, means[op]

    def deficit(self, cond: str, op: str = ex.PRIMARY_OP, group: str = "nonred") -> np.ndarray:
        return self.score(cond, op, "projection", "deficit", group)

    def task_gap(self, cond: str) -> dict[str, float]:
        """Per op, the condition's held-out expected exact match minus ex-2.2.11's control's, seed means."""
        return {op: float(self.eem(cond, op).mean() - self.eem(CONTROL, op).mean()) for op in ex.OP_NAMES}

    def task_worst(self, cond: str) -> tuple[str, float]:
        gaps = self.task_gap(cond)
        op = min(gaps, key=lambda o: gaps[o])
        return op, gaps[op]

    def axes(self, cond: str) -> tuple[int, ...]:
        r = self.runs(cond)[0]
        return tuple(r.get("axes", [ex.ANCHOR_AXIS]))

    def baseline(self, cond: str) -> str:
        """The un-anchored condition scored on the same subspace: the plane control for a plane condition."""
        return PLANE if len(self.axes(cond)) > 1 else CONTROL

    def alpha_excess(self, cond: str) -> np.ndarray:
        """ᾱ at op1 per seed, less the seed mean of its baseline scored on the same subspace."""
        return self.stat(cond, "alpha_op1") - float(self.stat(self.baseline(cond), "alpha_op1").mean())

    # -- part 1 ----------------------------------------------------------------------------

    def side_rows(self, cond: str, op: str) -> list[dict]:
        return [r for r in self.metrics["part1"]["side"] if r["condition"] == cond and r["op"] == op]

    def side_kept(self, cond: str, op: str, side: str) -> np.ndarray:
        return np.array([r["kept"] for r in self.side_rows(cond, op) if r["side"] == side], float)

    def side_n(self, cond: str, op: str, side: str) -> int:
        return next(r["n"] for r in self.side_rows(cond, op) if r["side"] == side)

    def side_lines(self, cond: str, op: str, side: str) -> np.ndarray:
        """Per line of one side, the kept share averaged over the seeds of a condition."""
        labels = sorted({r["label"] for r in self.side_rows(cond, op)})
        m = self.part1[f"{op}/removal"] & (self.part1[f"{op}/side"] == ex.SIDE_GROUPS.index(side))
        return np.nanmean([self.part1[f"{lb}/{op}/kept_line"][m] for lb in labels], axis=0)

    def bypass(self, cond: str, op: str, edit: str, group: str = "removal") -> np.ndarray:
        rows = [b for b in self.metrics["part1"]["bypass"] if b["condition"] == cond]
        return np.array(
            [b["ops"][op]["edits"][edit]["kept"][group] for b in sorted(rows, key=lambda b: b["seed"])], float
        )

    def label_curves(self, cond: str, op: str, group: str) -> np.ndarray:
        """(seeds, slices): ᾱ at op1 on one label group of one op's lines."""
        rows = [
            r
            for r in self.metrics["part1"]["label_source"]
            if r["condition"] == cond and r["op"] == op and r["group"] == group
        ]
        return np.array([r["alpha_op1_slices"] for r in sorted(rows, key=lambda r: r["seed"])], float)

    def label_n(self, op: str, group: str) -> int:
        return int(self.metrics["part1"]["label_source_n"][op][group])


# --- Shared drawing and table helpers ---------------------------------------------------------

INKS = {
    "handover": ("#c0392b", "#ff8a76"),
    "handover-slot": ("#2b6cb0", "#7fb3ff"),
    "handover-tied": ("#7b3fa0", "#cfa3ff"),
    "control": ("#6b6b6b", "#b0b0b0"),
    "control-plane": ("#6b6b6b", "#b0b0b0"),
    "tau-0.03": ("#1f77b4", "#7fb3ff"),
    "tau-0.01": ("#0b4a7a", "#4a90d9"),
    "lam-0.2": ("#d98c00", "#ffc04d"),
    "anti-5": ("#8c5a00", "#d9a24d"),
    "L6": ("#2e8b57", "#7fd4a5"),
    "plane": ("#7b3fa0", "#cfa3ff"),
    "L6-plane": ("#4b2570", "#a77fe0"),
    "plane-lam-0.2": ("#b03a9c", "#f28ae0"),
}
MARKERS = {
    "handover": "o",
    "handover-slot": "^",
    "handover-tied": "D",
    "control": "s",
    "control-plane": "s",
    "tau-0.03": "v",
    "tau-0.01": "v",
    "lam-0.2": ">",
    "anti-5": ">",
    "L6": "h",
    "plane": "P",
    "L6-plane": "P",
    "plane-lam-0.2": "X",
}


def ink(cond: str) -> str:
    return light_dark(*INKS[cond])


def dots(
    ax: Axes, x: float, v: np.ndarray, cond: str, *, rng, ms: float = 5.0, width: float = 0.06, label=None
) -> None:
    """One column of per-seed dots with the seed mean on top, in the condition's ink and marker, and a thin
    bar behind them spanning the seed range.
    """
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return
    color, m = ink(cond), MARKERS[cond]
    jit = rng.uniform(-width, width, len(v))
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + jit, v, "o", ms=2.2, color=color, alpha=0.45, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=ms, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6, label=label)


def ref_band(ax: Axes, v: np.ndarray, cond: str = REFERENCE, label: str | None = None) -> None:
    """The reference's seed range as a band across the panel, with its seed mean as a line."""
    v = np.asarray(v, float)
    ax.axhspan(v.min(), v.max(), color=ink(cond), alpha=0.12, lw=0, zorder=0, label=label)
    ax.axhline(v.mean(), color=ink(cond), lw=0.9, alpha=0.8, zorder=1)


def gate_line(ax: Axes, y: float, *, partial: float | None = None, fail: str | None = None) -> None:
    """A dashed gate line, a dotted secondary level when there is one, and the failing side hatched."""
    ax.axhline(y, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
    if partial is not None:
        ax.axhline(partial, color=light_dark("#333", "#ddd"), lw=0.7, ls=":", zorder=2)
    if fail is not None:
        lo, hi = ax.get_ylim()
        span = (lo, y) if fail == "below" else (y, hi)
        ax.axhspan(*span, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, zorder=0, alpha=0.1)
        ax.set_ylim(lo, hi)


def fig_legend(fig: plt.Figure, ax: Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7, **kwargs)


def span2(v: np.ndarray, digits: int = 3) -> str:
    """A seed mean with its range, as `0.512 (0.49–0.53)`."""
    v = np.asarray(v, float)
    if v.size == 0 or np.all(np.isnan(v)):
        return "—"
    return f"{np.nanmean(v):.{digits}f} ({np.nanmin(v):.{digits - 1}f}–{np.nanmax(v):.{digits - 1}f})"


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
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def bold_if(text: str, ok: bool) -> str:
    return f"<b>{text}</b>" if ok else text


def verdict_md(status: str, line: str) -> str:
    kind = {"held": "success", "partly": "warning", "did not hold": "danger", "unresolved": "info"}[status]
    return f"/// admonition | {status.capitalize()}\n    type: {kind}\n{line}\n///"


# --- Part 1: the side of red -----------------------------------------------------------------


def side_figure(res: Results) -> str:
    lines = {(c, s): res.side_lines(c, ex.MISSED_OP, s) for c in ex.SIDE_CONDITIONS for s in ex.SIDE_GROUPS}
    kept = {(c, s): res.side_kept(c, ex.MISSED_OP, s) for c in ex.SIDE_CONDITIONS for s in ex.SIDE_GROUPS}
    return side_draw(list(ex.SIDE_CONDITIONS), lines, kept, side_alt(res))


def side_alt(res: Results) -> str:
    m = {s: float(res.side_kept("handover", ex.MISSED_OP, s).mean()) for s in ex.SIDE_GROUPS}
    order = sorted(m, key=lambda s: m[s])
    return f"""
        Three dot panels, one per condition, with the three sides of red along the bottom of each. On
        handover the kept share is lowest where the red operand has {order[0]} and highest where it has
        {order[-1]}; the cloud of per-line dots spreads widely within every side. The two reference
        conditions show the same ordering. A dashed line marks the gate at one fifth.
    """


@memo
def side_draw(conds: list[str], lines: dict, kept: dict, alt_text: str) -> str:
    @themed(
        name="p1-side-of-red",
        alt_text=alt_text,
        caption=f"""
            **Kept share on the `{ex.MISSED_OP}` removal lines, by the side of red.** One panel per condition
            of ex-2.2.11; within each, one column per side of the red operand. The faint cloud is one dot per
            removal line, its kept share averaged over the condition's seeds (lines with no clean accuracy to
            keep are left out). The larger mark is the seed mean of the side's kept share, the ratio of group
            means ex-2.2.11 gates on, with one small dot per seed on the thin bar spanning the seed range.
            The dashed line is the
            {ex.RED_KEPT_GATE:.0%} gate, hatched above.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(conds), figsize=(8.4, 3.2), layout="constrained", sharey=True)
        axes = cast(AxesRow, axes)
        rng = np.random.default_rng(3)
        xs = np.arange(len(ex.SIDE_GROUPS))
        for ax, c in zip(axes, conds, strict=True):
            for x, s in zip(xs, ex.SIDE_GROUPS, strict=True):
                v = lines[c, s]
                v = v[~np.isnan(v)]
                ax.plot(
                    x + rng.uniform(-0.28, 0.28, len(v)),
                    v,
                    "o",
                    ms=1.4,
                    color=ink(c),
                    alpha=0.12,
                    mew=0,
                    zorder=1,
                    rasterized=True,
                )
                dots(ax, x, kept[c, s], c, rng=rng, ms=6, width=0.0)
            ax.set_title(c, fontsize=9)
            ax.set_xticks(xs, ex.SIDE_GROUPS, fontsize=8)
            ax.set_xlim(-0.6, len(xs) - 0.4)
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylim(-0.03, 1.03)
        for ax in axes:
            gate_line(ax, ex.RED_KEPT_GATE, fail="above")
        axes[0].set_ylabel("kept share")
        return fig

    return _plot()


def side_table(res: Results) -> str:
    head = ["condition", *(f"{s} (n)" for s in ex.SIDE_GROUPS), "all removal lines"]
    rows = []
    for c in ex.SIDE_CONDITIONS:
        cells = [
            f"{span2(res.side_kept(c, ex.MISSED_OP, s))} ({res.side_n(c, ex.MISSED_OP, s)})" for s in ex.SIDE_GROUPS
        ]
        rows.append([f"`{c}`", *cells, span2(res.kept(c, ex.MISSED_OP))])
    return table_html(
        head,
        rows,
        f"Kept share under `projection` on the `{ex.MISSED_OP}` removal lines by the side of red, seed mean and range, with the line count of each side. The last column is ex-2.2.11's measurement on the same lines.",
    )


def side_other_ops_table(res: Results) -> str:
    """Post hoc: the same split on every op, `handover` only."""
    head = ["op", *ex.SIDE_GROUPS, "all"]
    rows = []
    for op in ex.OP_NAMES:
        cells = [
            f"{res.side_kept('handover', op, s).mean():.3f} ({res.side_n('handover', op, s)})" for s in ex.SIDE_GROUPS
        ]
        rows.append([f"`{op}`", *cells, f"{res.kept('handover', op).mean():.3f}"])
    return table_html(
        head,
        rows,
        "Post hoc: the side split on every op's removal lines, `handover`, seed means with line counts. The rule that picks removal lines leaves some sides empty on some ops.",
    )


def side_status(res: Results) -> dict:
    m = {s: float(res.side_kept("handover", ex.MISSED_OP, s).mean()) for s in ex.SIDE_GROUPS}
    rng_ = {
        s: (
            float(res.side_kept("handover", ex.MISSED_OP, s).min()),
            float(res.side_kept("handover", ex.MISSED_OP, s).max()),
        )
        for s in ex.SIDE_GROUPS
    }
    on_axis, orange, pink = (m[s] for s in ex.SIDE_GROUPS)
    # The expectation: near zero with G = B and well above zero where G ≠ B. "Well above" is read as more
    # than the band above the on-axis side, on both off-axis sides.
    held = on_axis <= ex.RED_KEPT_GATE and min(orange, pink) > on_axis + ex.KEPT_BAND
    partly = (not held) and (max(orange, pink) > on_axis + ex.KEPT_BAND)
    return {"means": m, "ranges": rng_, "status": "held" if held else "partly" if partly else "did not hold"}


# --- Part 1: where the surviving hue is written -----------------------------------------------


def bypass_figure(res: Results) -> str:
    kept = {(op, e): res.bypass("handover", op, e) for op in ex.BYPASS_OPS for e in ex.BYPASS_EDITS}
    return bypass_draw(list(ex.BYPASS_OPS), list(ex.BYPASS_EDITS), kept, bypass_alt(res))


def bypass_alt(res: Results) -> str:
    m = {e: float(res.bypass("handover", ex.MISSED_OP, e).mean()) for e in ex.BYPASS_EDITS}
    lo, hi = min(m, key=lambda e: m[e]), max(m, key=lambda e: m[e])
    return f"""
        Two dot panels, hue-hsv on the left and mix on the right, with the five edits along the bottom.
        On hue-hsv the kept share is lowest under {lo} and highest under {hi}. On mix the two whole-line
        embedding edits remove almost everything, the blocks-only edit leaves about a third, and the two
        second-operand edits leave about half.
    """


@memo
def bypass_draw(ops: list[str], edits: list[str], kept: dict, alt_text: str) -> str:
    @themed(
        name="p1-bypass",
        alt_text=alt_text,
        caption=f"""
            **Kept share on the removal lines with the projection applied at chosen slices and positions,
            `handover`.** Left, `{ex.MISSED_OP}`; right, `{ex.PRIMARY_OP}`, the op where the full projection removes
            cleanly. Along the bottom, the five edits: `all` is ex-2.2.11's operator (every slice, every
            position), `embedding` and `blocks` split it by slice, and the two `op2` edits apply it at the
            second operand's position only. Each small dot is one of the twenty seeds, the larger mark the
            seed mean, the thin bar the seed range. The dashed line is the {ex.RED_KEPT_GATE:.0%} gate,
            hatched above.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(ops), figsize=(8.4, 3.2), layout="constrained", sharey=True)
        axes = cast(AxesRow, axes)
        rng = np.random.default_rng(4)
        xs = np.arange(len(edits))
        for ax, op in zip(axes, ops, strict=True):
            for x, e in zip(xs, edits, strict=True):
                dots(ax, x, kept[op, e], "handover", rng=rng, ms=6)
            ax.set_title(f"`{op}`".replace("`", ""), fontsize=9)
            ax.set_xticks(xs, edits, rotation=20, ha="right", fontsize=8)
            ax.set_xlim(-0.6, len(xs) - 0.4)
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylim(-0.03, 1.03)
        for ax in axes:
            gate_line(ax, ex.RED_KEPT_GATE, fail="above")
        axes[0].set_ylabel("kept share on removal lines")
        return fig

    return _plot()


def bypass_table(res: Results) -> str:
    head = ["edit", *(f"`{op}`" for op in ex.BYPASS_OPS)]
    rows = [[f"`{e}`", *(span2(res.bypass("handover", op, e)) for op in ex.BYPASS_OPS)] for e in ex.BYPASS_EDITS]
    return table_html(
        head,
        rows,
        "Kept share on the removal lines under each edit, `handover`, seed mean and range over twenty seeds.",
    )


def bypass_status(res: Results) -> dict:
    m = {e: float(res.bypass("handover", ex.MISSED_OP, e).mean()) for e in ex.BYPASS_EDITS}
    # The expectation: the embedding-only edit keeps about what the full projection keeps (within the band).
    emb_like_all = abs(m["embedding"] - m["all"]) <= ex.KEPT_BAND
    blocks_like_all = abs(m["blocks"] - m["all"]) <= ex.KEPT_BAND
    return {
        "means": m,
        "status": "held" if emb_like_all and not blocks_like_all else "partly" if emb_like_all else "did not hold",
    }


# --- Part 1: where the op1 drift comes from ---------------------------------------------------

LABEL_SHOWN = ("neither", "labelled by operand", "labelled by answer", "both")
LABEL_INKS = {
    "neither": ("#6b6b6b", "#b0b0b0"),
    "labelled by operand": ("#2b6cb0", "#7fb3ff"),
    "labelled by answer": ("#c0392b", "#ff8a76"),
    "both": ("#7b3fa0", "#cfa3ff"),
}


def label_shown(res: Results) -> list[str]:
    """The groups with at least one line; an empty group has no curve to draw."""
    return [g for g in LABEL_SHOWN if res.label_n(ex.PRIMARY_OP, g) > 0]


def label_figure(res: Results) -> str:
    shown = label_shown(res)
    curves = {(c, g): res.label_curves(c, ex.PRIMARY_OP, g) for c in ex.LABEL_CONDITIONS for g in shown}
    return label_draw(list(ex.LABEL_CONDITIONS), shown, curves, label_alt(res))


def label_alt(res: Results) -> str:
    m = {g: float(res.label_curves("handover", ex.PRIMARY_OP, g).mean()) for g in label_shown(res)}
    order = sorted(m, key=lambda g: m[g])
    return f"""
        Two line panels, handover on the left and handover-slot on the right, with the five slices along
        the bottom and one line per label group with a seed band. On handover the lines order from
        {order[0]} at the bottom to {order[-1]} at the top, and the two conditions look alike. The
        answer-labelled group has no line and is absent from both panels.
    """


@memo
def label_draw(conds: list[str], groups: list[str], curves: dict, alt_text: str) -> str:
    @themed(
        name="p1-label-source",
        alt_text=alt_text,
        caption=f"""
            **ᾱ at op1 on the `{ex.PRIMARY_OP}` probe lines whose first operand is not red, by which other color
            could earn the line's label.** One panel per condition, the slices along the bottom (`emb` is the
            token embedding). Each line is a group's seed mean of the alignment at op1 with the anchored axis,
            with the seed range as a band: lines that no color labels, lines labelled through a red second
            operand, and lines with a red second operand and a red answer. Red is at the red dose throughout.
            A group with no line (labelled through the answer alone) is left out.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(conds), figsize=(8.4, 3.2), layout="constrained", sharey=True)
        axes = cast(AxesRow, axes)
        n_slices = next(iter(curves.values())).shape[1]
        xs = np.arange(n_slices)
        for ax, c in zip(axes, conds, strict=True):
            for g in groups:
                v = curves[c, g]
                color = light_dark(*LABEL_INKS[g])
                ax.fill_between(xs, v.min(0), v.max(0), color=color, alpha=0.15, lw=0, zorder=1)
                ax.plot(xs, v.mean(0), "-o", ms=3.5, color=color, lw=1.4, zorder=3, label=g if c == conds[0] else None)
            ax.set_title(c, fontsize=9)
            ax.set_xticks(xs, ["emb", *(str(i) for i in range(1, n_slices))], fontsize=8)
            ax.set_xlabel("slice")
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("ᾱ at op1")
        fig_legend(fig, axes[0])
        return fig

    return _plot()


def label_table(res: Results) -> str:
    head = ["group", "lines", *(f"`{c}`" for c in ex.LABEL_CONDITIONS)]
    rows = [
        [
            g,
            str(res.label_n(ex.PRIMARY_OP, g)),
            *(span2(res.label_curves(c, ex.PRIMARY_OP, g).mean(1)) for c in ex.LABEL_CONDITIONS),
        ]
        for g in ex.LABEL_GROUPS
    ]
    return table_html(
        head,
        rows,
        f"ᾱ at op1 on the `{ex.PRIMARY_OP}` lines by label group, mean over slices, seed mean and range. The last group is the lines with a red first operand, which the containment line pools with the rest.",
    )


def label_status(res: Results) -> dict:
    m = {g: float(res.label_curves("handover", ex.PRIMARY_OP, g).mean()) for g in ex.LABEL_GROUPS}
    s = {g: float(res.label_curves("handover-slot", ex.PRIMARY_OP, g).mean()) for g in ex.LABEL_GROUPS}
    gap = m["labelled by answer"] - m["labelled by operand"]
    slot_gap = s["labelled by answer"] - s["labelled by operand"]
    # The expectation: the answer-labelled lines sit higher than the operand-labelled ones on handover, by
    # more than the band, and not on handover-slot.
    held = gap > ex.KEPT_BAND and slot_gap <= ex.KEPT_BAND
    # REVIEW: an empty deciding group makes the observation unresolved rather than a miss. On `mix` an
    # answer at the red dose needs a red operand, so "labelled by answer" alone has no line, and the
    # comparison the prediction rests on cannot be made. Verify: `label_source_n` in the metrics.
    empty = res.label_n(ex.PRIMARY_OP, "labelled by answer") == 0
    return {
        "handover": m,
        "slot": s,
        "gap": gap,
        "slot_gap": slot_gap,
        "status": "unresolved" if empty else "held" if held else "partly" if gap > 0 else "did not hold",
    }


# --- Exploratory: which red colors survive ---------------------------------------------------


def rgb_swatch(rgb: np.ndarray) -> str:
    """An inline swatch for a grid color that has no palette name, labelled by its hex."""
    hexcode = "#" + "".join(f"{int(round(x * 255)):02x}" for x in rgb)
    return f'<span class="sw" style="--sw: {hexcode};" aria-hidden="true"></span>&nbsp;<code>{hexcode}</code>'


def red_color_table(res: Results) -> str:
    """Post hoc: kept share on the `hue-hsv` removal lines by the red operand's color, `handover`.

    A line's colors come from its tokens; a token is read as the color it most often answers with in the
    probe set (the answer token against the modal true answer). Kept share here is the mean over lines of
    each line's own ratio, which is what part 1 stored per line, so it is not the ratio of group means the
    rest of the report quotes.
    """
    op = ex.MISSED_OP
    tok, q_idx, q_p = (res.probes[f"{op}/{k}"] for k in ("tokens", "q_idx", "q_p"))
    ans = q_idx[np.arange(len(q_idx)), q_p.argmax(1)]
    votes: dict[int, dict[int, int]] = {}
    for t, c in zip(tok[:, ex.ANSWER_POS], ans, strict=True):
        d = votes.setdefault(int(t), {})
        d[int(c)] = d.get(int(c), 0) + 1
    tok2col = {t: max(d, key=lambda c: d[c]) for t, d in votes.items()}
    grid = ex.ex229.GRID_RGB
    removal = res.part1[f"{op}/removal"].astype(bool)
    op2 = np.array([tok2col.get(int(t), -1) for t in tok[:, ex.ex229.OPERAND_POSITIONS[1]]])
    kept_line = np.stack(
        [
            res.part1[f"{lb}/{op}/kept_line"]
            for lb in sorted(
                k.split("/")[0]
                for k in res.part1
                if k.endswith("kept_line")
                and k.startswith("handover-s")
                and k.split("/")[0].removeprefix("handover-s").isdigit()
            )
        ]
    )
    rows = []
    for c in sorted(
        np.unique(op2[removal]), key=lambda c: (grid[c][1] != grid[c][2], -grid[c][0], grid[c][1], grid[c][2])
    ):
        m = removal & (op2 == c)
        v = np.nanmean(kept_line[:, m], axis=1)
        g, b = grid[c][1], grid[c][2]
        side = ex.SIDE_GROUPS[0 if g == b else 1 if g > b else 2]
        rows.append([rgb_swatch(grid[c]), side, str(int(m.sum())), span2(v)])
    return table_html(
        [f"{swatch(None)}red operand", "side", "lines", "kept ↓"],
        rows,
        f"Post hoc: kept share under `projection` on the `{op}` removal lines by the color of the red operand, `handover`, mean over lines and then over seeds, with the seed range. Every grid color at or above the red dose appears once.",
    )


# --- Part 2: the sweep -----------------------------------------------------------------------


def sweep_values(res: Results, key: str) -> dict[str, np.ndarray]:
    """One array per condition of the sweep, the reference, and the two controls, for one measurement."""
    conds = [REFERENCE, *SWEEP, CONTROL, PLANE]
    match key:
        case "kept_missed":
            return {c: res.kept(c, ex.MISSED_OP) for c in conds if c not in (CONTROL,)}
        case "kept_others":
            return {c: res.kept_others(c) for c in conds if c not in (CONTROL,)}
        case "alpha_op1":
            return {c: res.stat(c, "alpha_op1") for c in conds}
        case "m_line":
            return {c: res.stat(c, "m_line") for c in conds}
        case "deficit":
            return {c: res.deficit(c) for c in conds if c not in (CONTROL,)}
        case "task":
            return {
                c: np.array([res.eem(c, op) - res.eem(CONTROL, op).mean() for op in ex.OP_NAMES]).min(0)
                for c in conds
                if c != CONTROL
            }
        case "lead_emb" | "contrast":
            return {c: res.stat(c, key) for c in conds}
    raise KeyError(key)


PANELS = {
    # key: (y label, gate, secondary level, failing side)
    "kept_missed": (f"kept share on {ex.MISSED_OP}", ex.RED_KEPT_GATE, PROMOTION_LEVEL, "above"),
    "kept_others": ("kept share, ten other ops (mean)", ex.RED_KEPT_GATE, None, "above"),
    "alpha_op1": ("ᾱ at op1", None, None, None),
    "m_line": ("margin m_line", MARGIN_GATE, None, "below"),
    "deficit": ("non-red deficit, mix", ex.NONRED_DEFICIT_GATE, None, "above"),
    "task": ("task gap from control, worst op", -ex.TASK_GATE, None, "below"),
}


def sweep_figure(res: Results, keys: tuple[str, ...], name: str, alt_text: str, caption: str) -> str:
    values = {k: sweep_values(res, k) for k in keys}
    return sweep_draw(list(keys), name, values, alt_text, caption)


@memo
def sweep_draw(keys: list[str], name: str, values: dict, alt_text: str, caption: str) -> str:
    @themed(name=name, alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        n = len(keys)
        fig, axes = plt.subplots(n, 1, figsize=(8.4, 2.6 * n), layout="constrained", sharex=True, squeeze=False)
        axes = cast(AxesRow, axes[:, 0])
        rng = np.random.default_rng(5)
        cols = [*SWEEP, PLANE]
        xs = np.arange(len(cols))
        for ax, k in zip(axes, keys, strict=True):
            ylabel, gate, second, fail = PANELS[k]
            v = values[k]
            ref_band(ax, v[REFERENCE], label=f"{REFERENCE} (ex-2.2.11, 20 seeds)" if k == keys[0] else None)
            for x, c in zip(xs, cols, strict=True):
                if c in v:
                    dots(ax, x, v[c], c, rng=rng, ms=6)
            if CONTROL in v:
                ref_band(ax, v[CONTROL], cond=CONTROL, label="control (axis)" if k == keys[0] else None)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(axis="y", alpha=0.2)
            ax.set_xlim(-0.6, len(cols) - 0.4)
            if gate is not None:
                lo, hi = ax.get_ylim()
                ax.set_ylim(min(lo, gate - 0.02), max(hi, gate + 0.02))
                gate_line(ax, gate, partial=second, fail=fail)
        axes[-1].set_xticks(xs, cols, rotation=20, ha="right", fontsize=8)
        for x in (len(SWEEP) - 0.5,):
            for ax in axes:
                ax.axvline(x, color=light_dark("#999", "#666"), lw=0.6, ls=":", zorder=1)
        fig_legend(fig, axes[0])
        return fig

    return _plot()


def gates(res: Results, cond: str) -> dict[str, bool]:
    """Every gate the promotion rule names, on one condition's seed means."""
    _, task = res.task_worst(cond)
    return {
        "task": task >= -ex.TASK_GATE,
        "margin": float(res.stat(cond, "m_line").mean()) >= MARGIN_GATE,
        "lead": float(res.stat(cond, "lead_emb").mean()) >= ex.LEAD_GATE,
        "contrast": float(res.stat(cond, "contrast").mean()) >= ex.CONTRAST_GATE,
        "deficit": float(res.deficit(cond).mean()) <= ex.NONRED_DEFICIT_GATE,
    }


def sweep_table(res: Results) -> str:
    head = [
        "condition",
        "changes",
        f"kept `{ex.MISSED_OP}` ↓",
        "kept, others (worst op) ↓",
        "ᾱ at op1 ↓",
        "ᾱ over control",
        "margin ↑",
        "lead ↑",
        "contrast ↑",
        "non-red deficit ↓",
        "task gap (worst op) ↑",
    ]
    rows = []
    conds = [REFERENCE, *SWEEP, PLANE]
    for c in conds:
        g = gates(res, c)
        worst_op, worst = res.kept_worst(c)
        task_op, task = res.task_worst(c)
        changes = next((cc.changes for cc in ex.CONDITIONS if cc.name == c), 0)
        rows.append(
            [
                f"`{c}`",
                "—" if c == PLANE else str(changes),
                bold_if(span2(res.kept(c, ex.MISSED_OP)), float(res.kept(c, ex.MISSED_OP).mean()) <= PROMOTION_LEVEL),
                bold_if(f"{res.kept_others(c).mean():.3f} (`{worst_op}` {worst:.2f})", worst <= ex.RED_KEPT_GATE),
                span2(res.stat(c, "alpha_op1")),
                span2(res.alpha_excess(c)),
                bold_if(span2(res.stat(c, "m_line")), g["margin"]),
                bold_if(span2(res.stat(c, "lead_emb"), 2), g["lead"]),
                bold_if(span2(res.stat(c, "contrast"), 2), g["contrast"]),
                bold_if(span2(res.deficit(c)), g["deficit"]),
                bold_if(f"{task:+.3f} (`{task_op}`)", g["task"]),
            ]
        )
    return table_html(
        head,
        rows,
        f"Every condition on every measurement, seed mean and range. Bold passes its gate: kept share on `{ex.MISSED_OP}` under {PROMOTION_LEVEL:.2f} (the {ex.RED_KEPT_GATE:.0%} gate less the band), the worst other op under the gate, margin at least {MARGIN_GATE:.3f}, lead at least {ex.LEAD_GATE:g}, contrast at least {ex.CONTRAST_GATE:g}, non-red deficit at most {ex.NONRED_DEFICIT_GATE:g}, and the worst op's task gap no lower than −{ex.TASK_GATE:g}. `ᾱ over control` is ᾱ at op1 less the un-anchored control's scored on the same subspace (the axis, or the plane for `{PLANE}` and the plane conditions). `{PLANE}` is ex-2.2.11's un-anchored control scored on the plane, and gates nothing.",
        ref_rows=frozenset({0, len(conds) - 1}),
    )


# --- The proposal -------------------------------------------------------------------------------


def promotion(res: Results) -> dict:
    """The promotion rule and the τ proposal, applied to the seed means."""
    ref_alpha = float(res.alpha_excess(REFERENCE).mean())
    out: dict[str, dict] = {}
    for c in SWEEP:
        g = gates(res, c)
        kept = float(res.kept(c, ex.MISSED_OP).mean())
        alpha = float(res.alpha_excess(c).mean())
        out[c] = {
            "kept": kept,
            "kept_ok": kept <= PROMOTION_LEVEL,
            "gates_ok": all(g.values()),
            "failed": [k for k, ok in g.items() if not ok],
            "alpha": alpha,
            "alpha_ok": alpha <= ref_alpha + ex.KEPT_BAND,
            "changes": next(cc.changes for cc in ex.CONDITIONS if cc.name == c),
        }
        out[c]["qualifies"] = out[c]["kept_ok"] and out[c]["gates_ok"] and out[c]["alpha_ok"]
    qualifying = [c for c in SWEEP if out[c]["qualifies"]]
    proposed = min(qualifying, key=lambda c: (out[c]["changes"], out[c]["alpha"])) if qualifying else None
    tau = {}
    for c in (cc.name for cc in ex.TAU_CONDITIONS):
        g = gates(res, c)
        core = all(g[k] for k in ("task", "margin", "lead", "contrast"))
        lower = float(res.stat(REFERENCE, "alpha_op1").mean()) - float(res.stat(c, "alpha_op1").mean())
        tau[c] = {"lowers_by": lower, "gates_ok": core, "qualifies": lower > ex.KEPT_BAND and core}
    tau_proposed = [c for c, t in tau.items() if t["qualifies"]]
    return {
        "conditions": out,
        "qualifying": qualifying,
        "proposed": proposed,
        "tau": tau,
        "tau_proposed": tau_proposed,
        "ref_alpha": ref_alpha,
    }


def promotion_table(res: Results, p: dict) -> str:
    head = [
        "condition",
        f"kept `{ex.MISSED_OP}` ≤ {PROMOTION_LEVEL:.2f}",
        "gates",
        f"ᾱ over control ≤ ref + {ex.KEPT_BAND:g}",
        "changes",
        "qualifies",
    ]
    rows = []
    for c in SWEEP:
        r = p["conditions"][c]
        rows.append(
            [
                f"`{c}`",
                bold_if(f"{r['kept']:.3f}", r["kept_ok"]),
                "all pass" if r["gates_ok"] else "misses " + ", ".join(r["failed"]),
                bold_if(f"{r['alpha']:+.3f}", r["alpha_ok"]),
                str(r["changes"]),
                "yes" if r["qualifies"] else "no",
            ]
        )
    return table_html(
        head,
        rows,
        f"The promotion rule, clause by clause, on the seed means. The reference's ᾱ over control is {p['ref_alpha']:+.3f}.",
    )


# --- Load ---------------------------------------------------------------------------------------


def load_results() -> Results | None:
    if metrics_loaded is None or part1_loaded is None or ref_loaded is None or probes_loaded is None:
        return None
    return Results(metrics_loaded, ref_loaded, part1_loaded, probes_loaded)


res = load_results()
if res is None:
    stop(
        "No results yet. Run the experiment:\n```bash\nbin/mini run docs/m2/ex-2.2.12/experiment.py --app modal --max-containers 8 --budget 3h\n```"
    )

# --- The status of every observation, for the index and the verdict boxes ----------------------

side_st = side_status(res)
bypass_st = bypass_status(res)
label_st = label_status(res)
prop = promotion(res)
proposal_line = (
    f"`{prop['proposed']}` is proposed as the fix for `{ex.MISSED_OP}`"
    if prop["proposed"]
    else f"no condition qualifies as a fix for `{ex.MISSED_OP}`; the re-run keeps the reference recipe"
)
tau_line = (
    "and " + " and ".join(f"`{c}`" for c in prop["tau_proposed"]) + " is proposed as the τ"
    if prop["tau_proposed"]
    else "and neither τ condition qualifies"
)
red_by_side = res.metrics["part1"]["red_colors_by_side"]

rf"""
# Ex 2.2.12: what the stream holds on `hue-hsv`, and a small recipe sweep

/// tip |
<!-- tl;dr -->
Scouting, in two parts: what the stored ex-2.2.11 models kept of *red* on the one op where the projection did not remove it, and a sweep of a few changes to the recipe, including anchoring *red* to a plane, looking for clean removal on every op. The blind-spot story was wrong, the plane does not help, and nothing in the sweep qualifies, so the handover re-run keeps the reference recipe.
///

## Observations

- [Which lines survive on `hue-hsv`](#which-lines-survive-on-hue-hsv-part-1) — {side_st["status"]}. The split runs the other way. The lines that keep their answer are the ones whose red operand sits on the red axis, at a kept share of {side_st["means"]["G = B"]:.2f}; the lines on either side of it lose theirs, at {side_st["means"]["G > B"]:.2f} and {side_st["means"]["G < B"]:.2f}.
- [Where the surviving hue is written](#where-the-surviving-hue-is-written-part-1) — {bypass_st["status"]}. Removing at the blocks alone takes as much as the full projection, keeping {bypass_st["means"]["blocks"]:.2f} against {bypass_st["means"]["all"]:.2f}; removing at the embedding alone leaves more, {bypass_st["means"]["embedding"]:.2f}. Whatever survives is re-derived inside the blocks.
- [Where the op1 drift comes from](#where-the-op1-drift-comes-from-part-1) — {label_st["status"]}. One of the two groups the prediction compares is empty. On `{ex.PRIMARY_OP}`, an answer at the red dose needs a red operand, so no line is labelled through its answer alone.
- [The sweep](#the-sweep-part-2) — no condition qualifies. The plane conditions keep {min(prop["conditions"][c]["kept"] for c in ("plane", "L6-plane", "plane-lam-0.2")):.2f}–{max(prop["conditions"][c]["kept"] for c in ("plane", "L6-plane", "plane-lam-0.2")):.2f} on `{ex.MISSED_OP}`, which is inside the reference band, and every other condition keeps at least as much. Every condition passes the task, margin, lead, contrast, and deficit gates.

[The proposal](#the-proposal): {proposal_line}, {tau_line}.

## How to read this report

The measurements and the sweep's conditions were fixed before any run, at commit `007ec39`, and the promotion rule below says in advance what counts as a proposal and what happens if nothing qualifies. Everything after that commit is either observations filled into their sections or exploratory, marked as post hoc. The next preregistered experiment adopts what it needs from here and checks it on models it trains itself.

## Why this experiment

The anchored-op experiments that follow will read their results through a training setup that nearly works. Every unexplained leftover in it is a confound waiting for a later experiment, and a blind spot we can name now is one we can design around.

[Ex-2.2.11](../ex-2.2.11/report.py) put *red* on one axis of the residual stream of a small transformer,[^rs] taught it eleven color operations, and then projected the axis out. On ten of the eleven ops the model then lost *red*: it could no longer answer the lines whose answer needs the hue of the red operand. On `hue-hsv`, the op that takes its hue from the second operand, the model kept about a quarter of those answers, a little over the gate. The rule we froze then said the setup is not adopted.

[^rs]: The *residual stream* is the running vector of activations that each layer of a transformer reads from and writes back to.

Part 1 checks a candidate explanation. Picture hue as a clock face with red at twelve. The anchored axis measures how red a color is, and that is the same a little clockwise of red (toward orange) and a little anticlockwise (toward pink). So the one thing the axis cannot hold is which side of red a color sits on.

Every other op reads the red operand through its channels, and the red channel is what the axis holds. `hue-hsv` with red at op2 is the only case that needs the side. If the model keeps the side somewhere off the axis, that is what survives the projection. Red sitting at zero degrees is only a coincidence; a green anchor would have the same blind spot around green.

If that holds, pulling harder will not fix it, since one axis cannot hold a two-sided quantity and a plane can. So part 2 trains a small sweep on the handover setup: *red* anchored to a plane crossed against *red* anchored to an axis, at four and at six blocks, each of the two force factors raised by one step, the plane at the higher anchor weight, and two sharper settings of the pooling temperature τ. We expect the force factors to do nothing for `hue-hsv`; a null there is worth having.

The τ conditions answer a separate question. Ex-2.2.11 left the op1 alignment higher than its references and drifting before the anneal, and the τ conditions and one of the stored-checkpoint measurements are there to make progress on that.

The setup we sweep is the `handover` condition of ex-2.2.11: [table A+](../ex-2.2.4/report.py#the-op-set), the stochastic corpus, the whole-line labeller, the untied readout, and the recipe from ex-2.2.3. Its twenty seeds are the reference condition, and memoization makes that condition free.

## Conditions

The two parts are independent: part 2 does not wait on part 1.

### Part 1: the stored checkpoints

Part 1 trains nothing. It scores the twenty `handover` checkpoints of {ex.REFERENCE_EXPERIMENT}, with its probe set and its removal lines, under a few more edits than ex-2.2.11 scored. Where a measurement has a comparison, `handover-slot` and `handover-tied` are scored too.

### Part 2: the sweep

With the ex-2.2.11 recipe as a base, every condition changes one or two things from the reference.

{conditions_html()}

**`{ex.REF.name}`** is ex-2.2.11's candidate at its twenty seeds, and the other conditions train at its first {ex.SWEEP_SEEDS}, so every condition pairs with the reference seed for seed.

**`tau-0.03`** and **`tau-0.01`** sharpen the pool. Mellowmax at τ over the positions of a line is roughly the best position minus τ times the log of the line length, and the pull it returns is a softmax at that τ. So with τ held fixed, a longer label span spreads a little more of the pull onto positions that are not the red operand. The whole-line label made the span longer, so a sharper pool may take ᾱ at op1 back down.

**`lam-0.2`** and **`anti-5`** are the force factors: twice the anchor weight, and twice the anti-subspace peak. They are here so that a null can be read: if the plane fixes `hue-hsv` and these do not, the fix was the shape of the home and not its strength.

**`L6`**, **`plane`**, and **`L6-plane`** with the reference are a two-by-two of depth and subspace. Under the plane conditions *red* is pulled toward the span of e₁ and e₂ rather than toward e₁. Alignment with the plane is the length of a state's projection onto the pair: a cosine too,[^cos] since every state is unit-norm, but unsigned, so it says how much of the state lies in the plane and nothing about the direction within it.

[^cos]: The *cosine similarity* of two vectors is the cosine of the angle between them: one when they point the same way, zero when they are at right angles.

The anchor term pulls that length toward one on the labelled lines, the anti-subspace term is its square over every live position, and the removal projects the whole plane out. Six blocks instead of four is the other capacity factor, in case the HSV ops want more depth; it changes the number of slices too, so every slice-averaged measurement is also reported per slice.

**`plane-lam-0.2`** is the plane at the doubled anchor weight, in case the plane needs more pull than the axis did. Without it, a null on the force conditions would only say that the axis cannot be pushed harder, and a partial result on `plane` could not be told from a plane pulled too gently.

### The measurements

Two things differ from ex-2.2.11 in how the sweep is scored. First, the plane conditions are scored on the plane. An unsigned two-dimensional alignment is higher than a signed one-dimensional one even for a state that has nothing to do with *red*, so every plane measurement is compared with the control checkpoints scored the same way; those are stored and cost nothing to score. Second, every op is scored on the removal lines chosen by hue, as ex-2.2.11 did, with `{ex.MISSED_OP}` split by slot.

## Glossary

<dl>
<dt>Removal lines</dt>
<dd>The red lines whose answer needs the hue of the red operand: some permutation of the channels of that operand moves the true answer far. This is the rule from ex-2.2.11, unchanged. On <code>hue-hsv</code> they are the lines with red at op2.</dd>
<dt>Kept share</dt>
<dd>How much of its clean accuracy on the removal lines a model keeps after the projection. One means the projection did nothing. Zero means every removal-line answer changed, which is what the gate takes <em>red</em> being gone to mean. The model may still land on a near neighbour of the true answer; how far the answers move is a separate question, asked by the answer-cube figures of ex-2.2.10.</dd>
<dt>Side of red</dt>
<dd>Whether the red operand sits on the red axis of the cube, leans toward orange, or leans toward pink ({", ".join(ex.SIDE_GROUPS)}). The part of the hue the anchored axis cannot hold.</dd>
<dt>ᾱ at op1</dt>
<dd>The mean alignment with the axis over every color at the first operand position. How much the colors that are not red have drifted onto the axis.</dd>
<dt>Band</dt>
<dd>The spread a statistic showed across the twenty seeds of ex-2.2.11. A difference smaller than the band is not resolved.</dd>
</dl>

## Which lines survive on `hue-hsv` (part 1)

**What we expect.** On the `{ex.MISSED_OP}` removal lines, the kept share under the projection splits by the side of red: near zero where the red operand has G = B, and well above zero where G ≠ B. If the kept share is the same on all three sides, the axis is not the reason, and the plane conditions of part 2 lose their motivation; they run regardless, and would then be read as a capacity change with no hypothesis behind it.

[Ex-2.2.10](../ex-2.2.10/report.py#where-the-answers-go) already saw the two sides. Its answer-cube figure for `{ex.MISSED_OP}` with red at op2 shows the projected answers leaving red in two lobes, one toward orange and one toward pink. `sat-hsv` and `value-hsv` with red at op1 fan the same way, so the side survives the projection on those ops too; it just cannot rescue an answer that needs the saturation or value of red.

That figure pools every line; this measurement pairs each answer with the side of its own operand, which is what tells us whether the lobes are the side of red.
"""

side_figure(res)

f"""
**What we saw.** The opposite. The lines whose red operand sits on the red axis, with G = B, are the ones that keep their answer, at a kept share of {side_st["means"]["G = B"]:.2f} on `handover`. The lines on either side of the axis lose theirs: {side_st["means"]["G > B"]:.2f} toward orange and {side_st["means"]["G < B"]:.2f} toward pink, both under the gate.

`handover-slot` shows the same shape more strongly, and `handover-tied` more weakly. So the surviving quarter on `{ex.MISSED_OP}` is not the side of red: the side is the one thing the axis cannot hold, and it is the thing the projection removes.

The exploratory section splits the on-axis lines further, by the red operand's color: two of the three on-axis reds survive and the third does not.
"""

side_table(res)

f"""
{verdict_md(side_st["status"], "The kept share is highest where G = B and near zero on both sides of the axis, the reverse of the prediction. The blind spot is not what survives.")}

## Where the surviving hue is written (part 1)

**What we expect.** With the projection applied at the embedding only, `{ex.MISSED_OP}` keeps about what it keeps under the full projection: the side is written at the embedding, where the red operand's token is, and the blocks read it from there. If removal at the blocks alone takes as much as the full projection, the side is re-derived from the other channels inside the blocks, and the plane at the embedding would not be enough.
"""

bypass_figure(res)

f"""
**What we saw.** On `{ex.MISSED_OP}`, removing at the blocks alone keeps {bypass_st["means"]["blocks"]:.2f}, about what the full projection keeps ({bypass_st["means"]["all"]:.2f}), and removing at the embedding alone keeps more, {bypass_st["means"]["embedding"]:.2f}. The two second-operand edits sit with their whole-line counterparts, so where the projection is applied along the line does not matter, and where it is applied in depth does. The blocks-only edit has about the widest seed range of any measurement in this report: some seeds keep almost everything under it and some almost nothing.

`{ex.PRIMARY_OP}` behaves as ex-2.2.11 would predict, once the position of the edit is held fixed: applied over the whole line, an edit that touches the embedding removes nearly everything, and the blocks-only edit leaves about a third. Applied at the second operand alone it leaves about half, which on this op is the position mattering rather than the depth. That is the pattern the prediction had in mind for `{ex.MISSED_OP}` too, where it is closer to the reverse. The part of the answer that survives the full projection is not written at the embedding; or if it is, the blocks re-derive it from the other channels once the copy at the embedding is gone.
"""

# REVIEW: corrected the `mix` reading. The prose and the alt text said every edit that touches the
# embedding removes nearly everything on `mix`, but the two op2 edits touch the embedding and keep about
# half (0.489, 0.501 against 0.018 for `all`). The claim now holds the position fixed. Verify: the `mix`
# column of the bypass table.
bypass_table(res)

f"""
{verdict_md(bypass_st["status"], "Removal at the blocks alone takes as much as the full projection, and removal at the embedding alone takes less. The surviving answers are re-derived inside the blocks.")}

## Where the op1 drift comes from (part 1)

**What we expect.** On the red lines, ᾱ at op1 is higher on the lines the whole-line labeller labelled through their answer than on the lines it labelled through an operand. That is the candidate the containment item named for the labeller's half of the rise: a line whose answer draws is pulled at every position, op1 included. If the two groups read the same, the labeller's half of the rise has another cause.
"""

label_figure(res)

f"""
**What we saw.** The comparison cannot be made on `{ex.PRIMARY_OP}`. The measurement was designed for an op whose answer can be at the red dose while neither operand is, and `{ex.PRIMARY_OP}` is not one: a mean of two colors reaches the red dose only when one of them is red. So the group labelled through the answer alone is empty, the group labelled through both is small ({res.label_n(ex.PRIMARY_OP, "both")} lines), and the figure shows two groups where the prediction needed three.

The two groups we do have are alike on both conditions. On `handover`, ᾱ at op1 is about {label_st["handover"]["neither"]:.2f} for the lines no color labels and {label_st["handover"]["labelled by operand"]:.2f} for the lines with a red second operand. The eight lines with a red second operand and a red answer sit much higher, at {label_st["handover"]["both"]:.2f}, and are the same on `handover-slot`, whose labeller never reads the answer.

So on this op, what raises ᾱ at op1 is having two red colors on the line rather than which of them earned the label. The containment question needs an op such as `hue-hsv` or `darken`, where the answer can be red without a red operand; the measurement is written to take any op.
"""

label_table(res)

f"""
{verdict_md(label_st["status"], f"The answer-labelled group is empty on `{ex.PRIMARY_OP}`, so the measurement cannot say what causes the half of the rise attributed to the labeller. The lines it does have put the rise down to having two red colors on the line, whichever of them earned the label.")}

## The sweep (part 2)

**What we expect.** The plane conditions (`plane`, `L6-plane`, `plane-lam-0.2`) bring the `{ex.MISSED_OP}` kept share under the gate by more than the band, and the force conditions (`lam-0.2`, `anti-5`) do not move it. τ moves ᾱ at op1 down and nothing else. Depth on its own does little for `{ex.MISSED_OP}`, and the factorial says whether the plane needs it. Every condition keeps the task, the margin, the lead, and the contrast inside ex-2.2.11's gates; a condition that does not is reported and cannot be proposed.

In every figure of this section, each condition of the sweep is a column of {ex.SWEEP_SEEDS} seed dots with its mean, the twenty reference seeds are the band across the panel, and a gate is a dashed line with the failing side hatched. The last column is the un-anchored control scored on the plane, the comparison for the plane conditions.
"""

sweep_figure(
    res,
    ("kept_missed", "kept_others"),
    "p2-kept",
    f"""
        Two dot panels stacked, one column per condition of the sweep and one for the control on the plane.
        Top, the kept share on {ex.MISSED_OP}: every sweep column sits inside the reference band, which
        straddles the gate, and the control sits at one. The three plane columns sit lowest, at about a
        fifth, and the force and tau columns sit at about the reference or above it. Bottom, the kept share on the
        other ops: every sweep column sits well under the gate, and the control sits near one.
    """,
    f"""
        **Kept share under the projection, per condition.** Top, on the `{ex.MISSED_OP}` removal lines; the
        dashed line is the {ex.RED_KEPT_GATE:.0%} gate and the dotted line the band below it that a proposed
        condition has to clear. Bottom, the mean over the ten other ops of each op's kept share on its removal
        lines, against the same gate.
    """,
)

f"""
**What we saw: the kept share.** No condition moves `{ex.MISSED_OP}` under the gate by more than the band. The three plane conditions sit lowest, at {prop["conditions"]["plane-lam-0.2"]["kept"]:.2f} to {prop["conditions"]["plane"]["kept"]:.2f}, and the reference keeps {float(res.kept(REFERENCE, ex.MISSED_OP).mean()):.2f} with a band running from {float(res.kept(REFERENCE, ex.MISSED_OP).min()):.2f} to {float(res.kept(REFERENCE, ex.MISSED_OP).max()):.2f}, so a fifth falls inside it.

The force conditions and the sharper τ keep at least as much as the reference, and `L6` keeps the most. Given part 1, this is the shape to expect: the plane was meant to hold the side of red, and the side is not what survives.

On the ten other ops every condition removes cleanly on average. Two conditions, `tau-0.03` and `L6`, leave `darken` a little over the gate on its own, which the rule does not check and which the table below shows.
"""

sweep_figure(
    res,
    ("alpha_op1",),
    "p2-alpha",
    f"""
        One dot panel, one column per condition. The reference band spans about {float(res.stat(REFERENCE, "alpha_op1").min()):.2f} to {float(res.stat(REFERENCE, "alpha_op1").max()):.2f} with its mean
        at {float(res.stat(REFERENCE, "alpha_op1").mean()):.2f}, and the grey control band sits near zero. The force and tau columns sit in the
        lower half of the reference band and the three plane columns in its upper half; the plane control
        sits below the band, at about {float(res.stat(PLANE, "alpha_op1").mean()):.2f}.
    """,
    f"""
        **ᾱ at op1 per condition.** The mean alignment with the anchored subspace over every color at the
        first operand of the `{ex.PRIMARY_OP}` probe lines. The grey band is the un-anchored control scored on
        the axis; the plane conditions and `{PLANE}` are scored on the plane, where an unsigned two-dimensional
        alignment sits higher for every state, so they compare with `{PLANE}` and the others with the grey band.
    """,
)

f"""
**What we saw: ᾱ at op1.** The sharper τ lowers it a little: `tau-0.03` reaches {float(res.stat("tau-0.03", "alpha_op1").mean()):.3f} against {float(res.stat(REFERENCE, "alpha_op1").mean()):.3f} for the reference, a drop smaller than the band, and `tau-0.01` barely moves it, at {float(res.stat("tau-0.01", "alpha_op1").mean()):.3f}. The force conditions sit at about the same level as `tau-0.03`, which was not predicted and is also inside the band.

The plane conditions read higher, at {min(float(res.stat(c, "alpha_op1").mean()) for c in ("plane", "L6-plane", "plane-lam-0.2")):.2f} to {max(float(res.stat(c, "alpha_op1").mean()) for c in ("plane", "L6-plane", "plane-lam-0.2")):.2f}, which is the scoring rather than the models: an unsigned two-dimensional alignment is higher for every state, so the comparison that counts is with the control scored on the plane. Against that, their excess is {min(prop["conditions"][c]["alpha"] for c in ("plane", "L6-plane", "plane-lam-0.2")):.2f} to {max(prop["conditions"][c]["alpha"] for c in ("plane", "L6-plane", "plane-lam-0.2")):.2f}, below the {prop["ref_alpha"]:.2f} the reference shows over its own control.

We take that to mean the plane conditions drift no more than the axis ones, rather than that they drift less. The two excesses are over different baselines, and a state that drifts the same distance in the stream shows a smaller excess on a plane than on an axis.
"""

# REVIEW: the ᾱ excess of a plane condition is not commensurable with an axis condition's, so the prose
# reads the plane's smaller excess as "no more drift" and not as an improvement. The promotion rule
# compares excesses as written; it did not decide anything here, since no condition passes the kept
# clause. Verify: the alignment of a random unit state with a 2-plane in 64 dimensions is about √2 times
# its alignment with an axis, and `control-plane` against `control` in the table shows that ratio.

sweep_figure(
    res,
    ("m_line", "deficit", "task"),
    "p2-cost",
    """
        Three dot panels stacked, one column per condition. Top, the margin: every column sits above the
        gate, the plane columns lowest. Middle, the non-red deficit: every sweep column sits under the gate,
        with one plane seed just over it, and the control sits on it. Bottom, the task gap: every column
        sits above the gate, within a hundredth of zero.
    """,
    f"""
        **The cost side, per condition.** Top, the margin m_line on the `{ex.PRIMARY_OP}` lines, gated at
        {MARGIN_GATE:.3f} (hatched below). Middle, the non-red deficit on `{ex.PRIMARY_OP}` under the projection,
        gated at {ex.NONRED_DEFICIT_GATE:g} (hatched above). Bottom, the held-out expected exact match less the
        control's, on whichever op the condition is worst, gated at −{ex.TASK_GATE:g}.
    """,
)

f"""
**What we saw: the cost side.** Every condition passes every gate on its seed mean. The margin is lowest on `plane` and `L6-plane`, at about {float(res.stat("plane", "m_line").mean()):.2f} against {float(res.stat(REFERENCE, "m_line").mean()):.2f} for the reference, and still well over the gate. The non-red deficit on `{ex.PRIMARY_OP}` is under the gate on every seed mean, with one `plane` seed just over it. The task gap is within a hundredth of the control on every condition's worst op, so nothing in the sweep costs the task.
"""

sweep_table(res)

f"""
## The proposal

The promotion rule, frozen before the run:

> {ex.PROMOTION}

> {ex.TAU_PROPOSAL}
"""

promotion_table(res, prop)

f"""
{proposal_line[0].upper()}{proposal_line[1:]}, {tau_line}. Every condition passes the gates and fails the kept clause, so the fallback in the rule applies. The re-run trains the reference recipe, `{ex.MISSED_OP}` stays over the gate, and the next experiment treats it as a known leftover rather than a fix in waiting. The τ conditions lower ᾱ at op1 by less than the band, so the pooling temperature stays where ex-2.2.11 set it.

## Exploratory analyses

Not part of the plan. Anything we think of after seeing the data goes here, marked as post hoc.

### Which red colors survive

The side split says the surviving lines have G = B, and three grid colors do. Splitting the on-axis lines by the red operand's color says which.
"""

red_color_table(res)

f"""
Nearly all of the survival comes from two colors: the darker red and the paler one. Pure red, with the same hue and the highest redness of the seven, is removed about as cleanly as the hue-shifted reds on the seed mean, though with a wider spread: one seed keeps most of its pure-red lines. So what survives is not hue zero as such. It is not the red dose as such either, since two of the hue-shifted reds have the same redness as the surviving pair.

We do not have a mechanism for the pair. A `{ex.MISSED_OP}` answer takes its hue from the second operand and its saturation and value from the first, so the answer of a surviving line is a hue-zero color at the saturation and value of the first operand.

One candidate is that the model has learnt to answer those lines from how *achromatic* the second operand looks, meaning its G = B.[^achr] The axis does not hold that, the blocks can read it off the other channels, and the bypass measurement is consistent with it. A follow-up could test it by projecting at the blocks with the second operand replaced by a gray of the same value.

[^achr]: An *achromatic* color is one with no hue: a gray, where the three channels are equal or nearly so.

### The side split on the other ops

The same split on every op, for `handover`. The HSV ops that read the red operand's saturation or value show the same shape as `{ex.MISSED_OP}`, with the on-axis lines keeping more, and the RGB ops are flat across the three sides, each at its own level under the gate. The rule that picks removal lines leaves some sides empty on some ops.
"""

side_other_ops_table(res)

rf"""
## Discussion

The blind-spot story was a good story and it is wrong. It predicted which lines would survive on `hue-hsv`, and the lines that survive are the ones it said could not. That settles the plane: it was built to hold the one quantity the axis cannot, and that quantity is already removed. The sweep agrees, since every plane condition sits inside the reference band on `hue-hsv`.

The two force conditions and the two τ conditions give the nulls we wanted, and they are nulls on every measurement. At five seeds against ex-2.2.11's bands, that says a doubling of either force factor, or a sharper pool, moves nothing we can resolve.

What survives is narrower than we thought: two of the seven red colors, both on the red axis, on one op, re-derived inside the blocks. That makes the leftover easier to characterise for the next experiment, and harder to remove by anchoring, because the model has a route to those answers that does not run through the axis, either at the embedding or after it. Whether that route matters depends on what the anchored-op experiments will ask of the removal.

The containment question is still open, because of the op we chose. The measurement is in the code and takes any op; the next round that looks at op1 drift should run it on an op whose answer can be red on its own.

## Method

### The side of red

A grid color's side is the sign of G − B. The seven grid colors at or above the red dose split {red_by_side[0]} / {red_by_side[1]} / {red_by_side[2]} across G = B, G > B, and G < B, and the removal lines of `{ex.MISSED_OP}` carry those counts in the first figure's table.

### The plane

The home of *red* under the plane conditions is axes {ex.PLANE_AXES[0]} and {ex.PLANE_AXES[1]} of the stream, e₁ and e₂ together. Alignment is the length of the projection of a state onto the pair. The anchor and anti-subspace terms, the trajectory measurements, and the projection operator all take the pair where they took the axis. Every plane measurement is compared with the control checkpoints scored on the same pair. The concept holds two coordinates of sixty-four rather than one, and its variance share counts both.

### The label groups

The containment split groups each `{ex.PRIMARY_OP}` probe line by which of its other colors could have earned it the whole-line label: none, the second operand at the red dose, the answer at the red dose, or both. Lines whose first operand is itself red form a fifth group, reported in the table and left out of the figure. The labeller draws are stochastic, so a line is assigned to the group it could belong to under the labeller, rather than to whatever a particular epoch drew.

### Budget

{ex.NEW_RUNS} runs at d64, each as long as an ex-2.2.11 run (the two `L6` conditions about half again as long), at about three minutes a run on an L4. Scoring adds the part 1 edits on the checkpoints of three conditions, and the sweep conditions under the operators from ex-2.2.11. The reference condition is memoized from ex-2.2.11. The run cost about six dollars on Modal.
"""
