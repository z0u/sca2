# title: Ex 2.2.9: the grammar handover

import json
import math
import tempfile
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# The design constants come from `experiment.py` beside this script (the script's directory is
# on sys.path while it runs). The prose quotes the frozen gates, and that module carries the
# same numbers with each gate's wording in its docstring.
import experiment as ex
from mini.lit import memo, stop
from mini.store import project_store
from mini.vis import figure_html, light_dark, smooth_step_band, smooth_step_marks, themed
from sca.data.colors import redness
from sca.data.ops import (
    TOP,
    answer_dist,
    colors,
    commutativity,
    dose,
    is_on_grid,
    lines,
    on_grid,
    probe_lines,
    probe_partners,
    relevance,
    unordered_pairs,
)

# What a result cell shows while the run has not published yet.
RESULTS_TO_COME = "/// admonition | TODO\n    type: warning\nResults to come.\n///"


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists.

    One `get_refs` and one `get_many` for the lot: the bucket's fixed per-call latency is a couple of seconds, so resolving seven refs one at a time is most of a render.
    """
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict | None:
    return None if path is None else json.loads(path.read_text())


EX227_METRICS_REF = "reports/m2/ex-2.2.7/metrics"
EX228_METRICS_REF = "reports/m2/ex-2.2.8/metrics"
REFS = [
    ex.CALIBRATION_REF,
    ex.METRICS_REF,
    ex.GEOMETRY_REF,
    ex.ex223.METRICS_REF,
    ex.ex223.GEOMETRY_REF,
    EX227_METRICS_REF,
    EX228_METRICS_REF,
]

with tempfile.TemporaryDirectory() as _tmp:
    loaded = {r: read_json(f) for r, f in fetch(REFS, Path(_tmp)).items()}
calibration = loaded[ex.CALIBRATION_REF]


def calibration_runs() -> list[dict]:
    """The control seeds trained before the freeze, one per point looked at (corpus size × epochs × depth),
    ordered by step count and then depth.
    """
    if calibration is None:
        return []
    return sorted(calibration["runs"], key=lambda r: (calibration_steps(r), r.get("n_layer", ex.N_LAYER)))


def calibration_steps(run: dict) -> int:
    return run.get("epochs", ex.EPOCHS) * ex.steps_per_epoch(run.get("n_lines", ex.N_LINES))


def calibration_is_design(run: dict) -> bool:
    """Whether one look is the frozen design point."""
    return (run.get("n_lines", ex.N_LINES), run.get("epochs", ex.EPOCHS), run.get("n_layer", ex.N_LAYER)) == (
        ex.N_LINES,
        ex.EPOCHS,
        ex.N_LAYER,
    )


def calibration_label(run: dict) -> str:
    """One look's point, as `200k × 50 ep, L4`."""
    n_lines, e, layers = run.get("n_lines", ex.N_LINES), run.get("epochs", ex.EPOCHS), run.get("n_layer", ex.N_LAYER)
    return f"{n_lines // 1000}k × {e} ep, L{layers}"


def calibration_gaps(run: dict) -> dict[str, float]:
    return {op: run["holdout_ceiling"][op] - run["holdout_eem"][op] for op in ex.OP_NAMES}


def calibration_clears(run: dict) -> bool:
    gaps = calibration_gaps(run)
    return all(gaps[op] <= ex.CALIBRATION_FLOOR for op in (*ex.KEPT, *ex.ADDED))


def calibration_md() -> str:
    """The pre-freeze read: one control seed per point looked at, its expected exact match per op against the
    ceiling the drawn answers allow, with the gap read against `CALIBRATION_FLOOR` on the kept and added ops.
    """
    runs = calibration_runs()
    if not runs:
        return "_The calibration has not been published yet._"
    group = {
        **dict.fromkeys(ex.KEPT, "kept"),
        **dict.fromkeys(ex.ADDED, "added"),
        **dict.fromkeys(ex.ORDER_SENSITIVE, "order-sensitive"),
    }
    head = (
        "| op | group | ceiling | "
        + " | ".join(
            (f"**{calibration_label(r)}**" if calibration_is_design(r) else calibration_label(r)) + ": EEM (gap ↓)"
            for r in runs
        )
        + " |\n| --- | --- | ---: | "
        + " | ".join("---:" for _ in runs)
        + " |\n"
    )
    rows = []
    for op in ex.OP_NAMES:
        cells = []
        for r in runs:
            gap = calibration_gaps(r)[op]
            gated = op in (*ex.KEPT, *ex.ADDED)
            g = f"**{gap:.3f}**" if gated and gap <= ex.CALIBRATION_FLOOR else f"{gap:.3f}"
            cells.append(f"{r['holdout_eem'][op]:.3f} ({g})")
        rows.append(f"| `{op}` | {group[op]} | {runs[0]['holdout_ceiling'][op]:.3f} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows)


def calibration_verdict() -> str:
    """One line per point looked at on whether that control seed clears the bar."""
    gated = (*ex.KEPT, *ex.ADDED)
    clears, misses = [], []
    for r in calibration_runs():
        gaps = calibration_gaps(r)
        e = f"{calibration_label(r)} ({calibration_steps(r):,} steps{', the frozen point' if calibration_is_design(r) else ''})"
        short = [op for op in gated if gaps[op] > ex.CALIBRATION_FLOOR]
        if short:
            misses.append(f"{e} on {', '.join(f'`{op}`' for op in short)}")
        else:
            worst = max(gated, key=lambda op: gaps[op])
            clears.append(f"{e}, widest gap `{worst}` at {gaps[worst]:.3f}")
    order = max(calibration_gaps(r)[op] for r in calibration_runs() for op in ex.ORDER_SENSITIVE)
    return (
        f"- **Clears the bar** (every kept and added op within {ex.CALIBRATION_FLOOR:g} of its ceiling): "
        + "; ".join(clears)
        + ".\n"
        + "- **Misses**: "
        + "; ".join(misses)
        + f".\n- The order-sensitive ops are within {order:.2f} of their ceiling at every point."
    )


@dataclass(frozen=True)
class Results:
    """Every published result the report reads, with one accessor per shape the cells need.

    `metrics` is this experiment's (the per-line arrays under `ARRAYS_REF` are published but no cell reads them). `ex223` is the reference's metrics (the recipe's twenty seeds
    on the six-op grammar, whose placement statistics are printed beside every read in H2). `ex228` carries
    the per-run spread of the non-red deficit under `projection` at those twenty seeds, which is the σ the
    bands in H3 and H5 use. `ex227` carries the embedding-component table of the pilot, for H4's ceiling.
    """

    metrics: dict
    geometry: dict
    ex223: dict
    ex223_geometry: dict
    ex227: dict
    ex228: dict

    def runs(self, cond: str) -> list[dict]:
        """The eval records of a condition, in seed order."""
        return sorted((r for r in self.metrics["runs"] if r["condition"] == cond), key=lambda r: r["seed"])

    def scored(self, cond: str) -> list[dict]:
        return sorted((r for r in self.metrics["scores"] if r["condition"] == cond), key=lambda r: r["seed"])

    def n(self, cond: str) -> int:
        return len(self.runs(cond))

    def stat(self, cond: str, key: str, op: str | None = None) -> np.ndarray:
        """One value per seed: a gated (`mix`) placement statistic, or one op's from `per_op`."""
        return np.array([r[key] if op is None else r["per_op"][op][key] for r in self.runs(cond)], float)

    def read(self, cond: str, op: str, key: str, split: str = "holdout") -> np.ndarray:
        """One value per seed of a task read (`eem`, `ceiling`, `p_mode`, `kl`, ...) on one op's eval set."""
        return np.array([r["sets"][op][split][key] for r in self.runs(cond)], float)

    def eem(self, cond: str, op: str) -> np.ndarray:
        return np.array([r["holdout_eem"][op] for r in self.runs(cond)], float)

    def score(self, cond: str, op: str, operator: str | None, key: str, group: str = "removal") -> np.ndarray:
        """One value per seed of a scored statistic on one op's probe lines; `operator=None` is the clean pass."""
        out = []
        for r in self.scored(cond):
            s = r["ops"][op]
            v = s["clean"][key] if operator is None else s["operators"][operator][key]
            out.append(v[group])
        return np.array(out, float)

    def kept(self, cond: str, op: str, operator: str) -> np.ndarray:
        """The share of clean expected exact match the model keeps on the removal lines: H3's removal read."""
        return self.score(cond, op, operator, "kept", "removal")

    def deficit(self, cond: str, op: str, operator: str, group: str = "nonred") -> np.ndarray:
        """Clean minus intervened expected exact match on one group: H3's selectivity read on `nonred`."""
        return self.score(cond, op, operator, "deficit", group)

    def component(self, cond: str, word: str, table: str = "rows") -> np.ndarray:
        """The axis component of one token's row, per seed, on the embedding (`rows`) or the readout."""
        return np.array([r[table][word] for r in self.runs(cond) if r[table] is not None], float)

    # -- the references -------------------------------------------------------------------

    def ref_runs(self) -> list[dict]:
        """Ex-2.2.3's adopted point at its twenty seeds: the frozen five and the addendum's fifteen."""
        names = (ex.EX223_REFERENCE, f"{ex.EX223_REFERENCE}-more")
        return sorted((r for r in self.ex223["runs"] if r["condition"] in names), key=lambda r: r["seed"])

    def ref_stat(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.ref_runs()], float)

    def ref_em(self, op: str) -> np.ndarray:
        return np.array([r["holdout_em"][op] for r in self.ref_runs()], float)

    def ref_deficit_sd(self, op: str) -> float:
        """Per-run σ of the non-red deficit under `projection` at the reference (ex-2.2.8), or NaN on an op the
        six-op grammar did not have.
        """
        runs = [r for r in self.ex228["scores"] if r["condition"] == ex.EX223_REFERENCE]
        if op not in runs[0]["ops"]:
            return float("nan")
        v = np.array([r["ops"][op]["trials"]["projection"]["deficit"]["nonred"] for r in runs], float)
        return float(v.std(ddof=1))

    def ref_deficit(self, op: str) -> np.ndarray:
        runs = [r for r in self.ex228["scores"] if r["condition"] == ex.EX223_REFERENCE]
        return np.array([r["ops"][op]["trials"]["projection"]["deficit"]["nonred"] for r in runs], float)

    def ex227_component(self, cond: str, word: str, table: str = "rows") -> np.ndarray:
        """Ex-2.2.7's component for `word`, per seed; NaN for a word its grammar never had (the added ops)."""
        rows = sorted((r for r in self.ex227["rows"] if r["condition"] == cond), key=lambda r: r["seed"])
        return np.array([r[table].get(word, np.nan) for r in rows if r[table] is not None], float)


def load_results() -> Results | None:
    """Everything the results cells read, or None before the full stage has been published."""
    metrics = loaded[ex.METRICS_REF]
    if metrics is None:
        return None
    geometry, ex223, ex223_geometry, ex227, ex228 = (loaded[r] for r in REFS[2:])
    assert geometry is not None and ex223 is not None and ex223_geometry is not None
    assert ex227 is not None and ex228 is not None
    return Results(metrics, geometry, ex223, ex223_geometry, ex227, ex228)


def band(sd: float, n_a: int, n_b: int) -> float:
    """The smallest difference between two seed means the read resolves: 2σ√(1/n_a + 1/n_b)."""
    return ex.RESOLUTION_SD * sd * math.sqrt(1 / n_a + 1 / n_b)


def span2(v: np.ndarray, digits: int = 3) -> str:
    """A seed mean with its range, as `0.512 (0.49–0.53)`."""
    v = np.asarray(v, float)
    if v.size == 0 or np.all(np.isnan(v)):
        return "—"
    # The no-break space keeps the value and its range on one line when the cell wraps.
    return f"{np.nanmean(v):.{digits}f} ({np.nanmin(v):.{digits - 1}f}–{np.nanmax(v):.{digits - 1}f})"


def ink(cond: str) -> str:
    """One ink per condition, as a (light, dark) pair picked by the active theme."""
    inks = {
        "control": ("#6b6b6b", "#b0b0b0"),
        "handover": ("#c0392b", "#ff8a76"),
        "handover-slot": ("#2b6cb0", "#7fb3ff"),
        "handover-tied": ("#7b3fa0", "#cfa3ff"),
        "handover-narrow": ("#2e8b57", "#7fd8a4"),
        "reference": ("#a08a2e", "#e6d27a"),
    }
    return light_dark(*inks[cond])


def marker(cond: str) -> str:
    """One marker shape per condition, so the conditions read apart in greyscale as well as by ink."""
    return {
        "control": "s",
        "handover": "o",
        "handover-slot": "^",
        "handover-tied": "D",
        "handover-narrow": "v",
        "reference": "P",
    }[cond]


def cell_html(text: str) -> str:
    """A table cell's text with `` `code` `` spans rendered, since the cell never passes through Markdown."""
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
    """A one-line verdict box: `status` is pass, partial, miss, or unresolved."""
    kind = {"pass": "success", "partial": "warning", "miss": "danger", "unresolved": "info"}[status]
    return f"/// admonition | {status.capitalize()}\n    type: {kind}\n{line}\n///"


def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    """The within-condition per-run σ pooled over two conditions."""
    return math.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))


def dots(ax, x: float, v: np.ndarray, cond: str, *, rng, ms: float = 5.0, width: float = 0.06, label=None) -> None:
    """One column of per-seed dots with the seed mean drawn on top, in the condition's ink and marker shape.

    A thin bar behind the dots spans the seed range, so the spread reads at a glance even where the dots overlap.
    """
    v = np.asarray(v, float)
    color, m = ink(cond), marker(cond)
    jit = rng.uniform(-width, width, len(v))
    ax.plot([x, x], [np.nanmin(v), np.nanmax(v)], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + jit, v, "o", ms=2.2, color=color, alpha=0.45, zorder=3, mew=0)
    ax.plot(x, np.nanmean(v), m, ms=ms, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6, label=label)


def fig_legend(fig: plt.Figure, ax: Axes, **kwargs) -> None:
    """Move *ax*'s legend out to the top of the figure, in one row.

    Inside the axes a legend competes with the data for space, and lands on the hatched miss region on the figures
    that have one. Above the panels it reads as a key to the whole figure, which is what it is.
    """
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7, **kwargs)


def gate_line(ax: Axes, y: float, *, partial: float | None = None, fail: str | None = None) -> None:
    """A dashed gate line, with a dotted partial level under it when there is one.

    *fail* names the side that misses the gate (`below` or `above`); that side is hatched so a miss reads as a
    region rather than a line to compare against.
    """
    ax.axhline(y, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
    if partial is not None:
        ax.axhline(partial, color=light_dark("#333", "#ddd"), lw=0.7, ls=":", zorder=2)
    if fail is not None:
        lo, hi = ax.get_ylim()
        span = (lo, y) if fail == "below" else (y, hi)
        ax.axhspan(
            *span,
            facecolor="none",
            edgecolor=light_dark("#000", "#fff"),
            hatch="//",
            lw=0,
            zorder=0,
            alpha=0.1,
        )
        ax.set_ylim(lo, hi)


def h1_status(res: Results) -> dict[str, tuple[str, float, float]]:
    """Per op: (status, handover minus control seed mean, band). The band is 2σ√(1/n_h + 1/n_c) with σ the
    within-condition per-run spread of expected exact match on that op, pooled over the two conditions.
    """
    out = {}
    n_h, n_c = res.n("handover"), res.n("control")
    for op in ex.OP_NAMES:
        h, c = res.eem("handover", op), res.eem("control", op)
        sd = math.sqrt((h.var(ddof=1) * (n_h - 1) + c.var(ddof=1) * (n_c - 1)) / (n_h + n_c - 2))
        b = band(sd, n_h, n_c)
        d = float(h.mean() - c.mean())
        if d >= -ex.TASK_GATE:
            status = "pass"
        elif d >= -ex.TASK_GATE - b:
            status = "unresolved"
        elif d >= -ex.TASK_PARTIAL:
            status = "partial"
        else:
            status = "miss"
        out[op] = (status, d, b)
    return out


def h1_verdict(res: Results) -> tuple[str, str]:
    st = h1_status(res)
    worst = min(st, key=lambda op: st[op][1])
    misses = [op for op in ex.OP_NAMES if st[op][0] == "miss"]
    partials = [op for op in ex.OP_NAMES if st[op][0] == "partial"]
    unresolved = [op for op in ex.OP_NAMES if st[op][0] == "unresolved"]
    tail = f"The largest shortfall is `{worst}` at {st[worst][1]:+.3f} (band {st[worst][2]:.3f})."
    if misses:
        return (
            "miss",
            f"`handover` is more than {ex.TASK_PARTIAL:g} below the control on {', '.join(f'`{o}`' for o in misses)}. {tail}",
        )
    if partials:
        return (
            "partial",
            f"Every op is within {ex.TASK_PARTIAL:g} of the control, and {', '.join(f'`{o}`' for o in partials)} outside {ex.TASK_GATE:g}. {tail}",
        )
    if unresolved:
        return (
            "unresolved",
            f"Every op is within {ex.TASK_GATE:g} of the control or within a band of it ({', '.join(f'`{o}`' for o in unresolved)}). {tail}",
        )
    return "pass", f"On every op the `handover` seed mean is within {ex.TASK_GATE:g} of the control's. {tail}"


def h1_figure(res: Results) -> str:
    conds = [c.name for c in ex.CONDS]
    eem = {(c, op): res.eem(c, op) for c in conds for op in ex.OP_NAMES}
    ceil = {op: float(res.read("control", op, "ceiling").mean()) for op in ex.OP_NAMES}
    ctrl = {op: float(eem["control", op].mean()) for op in ex.OP_NAMES}

    return h1_draw(conds, eem, ceil, ctrl)


@memo
def h1_draw(
    conds: list[str], eem: dict[tuple[str, str], np.ndarray], ceil: dict[str, float], ctrl: dict[str, float]
) -> str:
    @themed(
        name="h1-eem-per-op",
        alt_text="""
            A dot chart with the eleven ops along the bottom and expected exact match up the side. At each op, five columns of small dots, one per condition, sit close together at the same height, with a grey strip just below the control's mean marking the gate and a bar above marking the ceiling the drawn answers allow. The handover dots sit inside the strip on every op.
        """,
        caption=f"""
            **Expected exact match on held-out lines, per op and condition.** Each small dot is one seed, the larger mark the seed mean, and the thin bar behind them the seed range. The grey strip under each op runs from the control's mean down to {ex.TASK_GATE:g} below it: a `handover` mean inside the strip passes H1 on that op. The bar above each op is the ceiling the drawn answers allow (Σq²), which is 1 on the ops that never round.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.4, 3.6), layout="constrained")
        strip = light_dark("#00000014", "#ffffff1f")
        xs = np.arange(len(ex.OP_NAMES))
        off = np.linspace(-0.3, 0.3, len(conds))
        rng = np.random.default_rng(0)
        for x, op in zip(xs, ex.OP_NAMES, strict=True):
            ax.fill_between([x - 0.45, x + 0.45], ctrl[op] - ex.TASK_GATE, ctrl[op], color=strip, lw=0, zorder=1)
            ax.plot([x - 0.45, x + 0.45], [ceil[op]] * 2, color=light_dark("#333", "#ddd"), lw=0.9, zorder=2)
            for o, c in zip(off, conds, strict=True):
                dots(ax, x + o, eem[c, op], c, rng=rng, width=0.04, label=c if x == 0 else None)
        ax.set_xticks(xs, ex.OP_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("expected exact match")
        ax.set_ylim(0.2, 1.02)
        ax.grid(axis="y", alpha=0.2)
        fig_legend(fig, ax)
        return fig

    return _plot()


def h1_table(res: Results) -> str:
    st = h1_status(res)
    conds = [c.name for c in ex.CONDS if c.name != "control"]
    rows = []
    for op in ex.OP_NAMES:
        ctrl = res.eem("control", op)
        ceil = float(res.read("control", op, "ceiling").mean())
        status, d, b = st[op]
        cells = [f"`{op}`", f"{ceil:.2f}", f"{ctrl.mean():.3f}"]
        for c in conds:
            v = res.eem(c, op)
            delta = float(v.mean() - ctrl.mean())
            text = f"{v.mean():.3f} ({delta:+.3f})"
            cells.append(bold_if(text, delta >= -ex.TASK_GATE) if c == "handover" else text)
        cells.append(f"{b:.3f}")
        rows.append(cells)
    head = ["op", "ceiling", "control", *[f"`{c}` (Δ) ↑" for c in conds], "band"]
    return table_html(
        head,
        rows,
        f"""
        **Seed-mean expected exact match on held-out lines, per op.** Each condition's column gives its mean and, in brackets, the difference from the control. A bold `handover` entry is within {ex.TASK_GATE:g} of the control, which is the H1 gate. The band is the smallest difference from the control the read resolves on that op, from the per-run spread of the two conditions.
        """,
    )


def h1_calibration_table(res: Results) -> str:
    """How the answer mass sits against the true distribution: P(mode) and KL, control against handover."""
    rows = []
    for op in ex.OP_NAMES:
        cells = [f"`{op}`"]
        for key in ("p_mode", "kl"):
            for c in ("control", "handover"):
                cells.append(f"{res.read(c, op, key).mean():.2f}")
        rows.append(cells)
    head = ["op", "P(mode) control", "P(mode) handover", "KL control ↓", "KL handover ↓"]
    return table_html(
        head,
        rows,
        """
        **How the answer mass is spread, per op.** P(mode) is the mass the model puts on the most likely true answer; KL is the divergence from the true answer distribution to the model's, in nats, averaged over held-out lines. Both are seed means. On the ops that never round, the true distribution is one color, so P(mode) is the expected exact match and KL the negative log-likelihood.
        """,
    )


def h1_prose(res: Results) -> str:
    """The plain-English read of H1, written from the numbers."""
    st = h1_status(res)
    worst = min(ex.OP_NAMES, key=lambda op: st[op][1])
    best = max(ex.OP_NAMES, key=lambda op: st[op][1])
    order = [st[op][1] for op in ex.ORDER_SENSITIVE]
    narrow = {op: float(res.eem("handover-narrow", op).mean() - res.eem("control", op).mean()) for op in ex.OP_NAMES}
    narrow_worst = min(narrow, key=lambda op: narrow[op])
    gap = {op: float(res.read("control", op, "ceiling").mean() - res.eem("control", op).mean()) for op in ex.OP_NAMES}
    widest = max(gap, key=lambda op: gap[op])
    return f"The anchored model learns the task about as well as the un-anchored one. Across the {ex.N_OPS} ops the `handover` seed mean sits between {st[worst][1]:+.3f} (`{worst}`) and {st[best][1]:+.3f} (`{best}`) of the control's. On the order-sensitive subset the differences are {', '.join(f'{d:+.3f}' for d in order)}, so reading operand order costs the anchored model nothing extra. `handover-narrow`, with lines per op held at the six-op count, sits between {min(narrow.values()):+.3f} (`{narrow_worst}`) and {max(narrow.values()):+.3f} of the control. One op sits well under its ceiling on every condition: `{widest}`, at {gap[widest]:.2f} below on the control. That is the gap the calibration look found before the freeze ([Before the freeze](#before-the-freeze)): more passes over the corpus widened it rather than closing it, and it is the same on the anchored conditions, so it is a property of the op and the corpus rather than of the anchor."


def h2_gates() -> dict[str, tuple[str, float, float | None, str]]:
    """Each H2 statistic: (label, gate, partial level, direction). Retention and latch are per-run reads."""
    return {
        "m_line": ("margin m_line", ex.MARGIN_RATIO * ex.REF_M_LINE, ex.MARGIN_PARTIAL * ex.REF_M_LINE, "↑"),
        "r2_sim": ("grading r²", ex.GRADE_R2_RATIO * ex.REF_R2_SIM, None, "↑"),
        "lead_emb": ("lead at embedding", ex.LEAD_GATE, None, "↑"),
        "contrast": ("contrast", ex.CONTRAST_GATE, ex.CONTRAST_PARTIAL, "↑"),
        "alpha_op1": ("ᾱ at op1", ex.MEAN_ALIGN_REF, None, "↓"),
    }


def h2_status(res: Results, cond: str = "handover") -> dict[str, tuple[str, float]]:
    """Per statistic on one condition: (status, seed mean). Retention and latch are read per run: `retention`
    is the share of qualifying runs that end at the gate's share of their peak, `latch` the count of latched
    runs. ᾱ is a prediction (expected above the reference level), so its status says only which side it is on.
    """
    out: dict[str, tuple[str, float]] = {}
    for key, (_, gate, partial, _direction) in h2_gates().items():
        v = float(res.stat(cond, key).mean())
        if key == "alpha_op1":
            out[key] = ("above" if v > gate else "at", v)
            continue
        ok = v >= gate
        out[key] = ("pass" if ok else "partial" if partial is not None and v >= partial else "miss", v)
    peak, ret = res.stat(cond, "m_line_peak"), res.stat(cond, "retention")
    q = peak >= ex.RETENTION_FLOOR
    kept = float((ret[q] >= ex.RETENTION_GATE).mean()) if q.any() else float("nan")
    out["retention"] = (
        "pass" if q.any() and kept == 1.0 else "miss",
        float(ret[q].mean()) if q.any() else float("nan"),
    )
    latched = int((res.stat(cond, "latch_pi") > ex.LATCH_PI).sum())
    out["latch"] = ("pass" if latched == 0 else "miss", float(latched))
    return out


def retention_detail(res: Results, cond: str) -> tuple[int, int, float]:
    """(seeds below the retention gate, qualifying seeds, the lowest retention) on one condition."""
    peak, ret = res.stat(cond, "m_line_peak"), res.stat(cond, "retention")
    q = peak >= ex.RETENTION_FLOOR
    return int((ret[q] < ex.RETENTION_GATE).sum()), int(q.sum()), float(ret[q].min()) if q.any() else float("nan")


def h2_verdict(res: Results) -> tuple[str, str]:
    st = h2_status(res)
    full = ("m_line", "r2_sim", "contrast", "lead_emb", "retention", "latch")
    misses = [k for k in full if st[k][0] == "miss"]
    partials = [k for k in full if st[k][0] == "partial"]
    names = h2_gates()
    label = lambda k: names[k][0] if k in names else k  # noqa: E731
    deficit = float(res.deficit("handover", ex.PRIMARY_OP, "projection").mean())
    alpha = f"The paired ᾱ prediction holds: ᾱ at op1 is {st['alpha_op1'][1]:.2f}, {st['alpha_op1'][0]} the {ex.MEAN_ALIGN_REF:g} the earlier experiments gated, as expected with the readout untied (the output loss no longer holds the non-red embedding rows off the axis; see the H2 results), and the non-red deficit it was paired with is {deficit:.3f}, inside H3's {ex.NONRED_DEFICIT_GATE:g} gate."
    if misses:
        clears = [k for k in full if k not in misses and k not in partials]
        detail = ""
        if "retention" in misses:
            n_below, n_q, low = retention_detail(res, "handover")
            ref = float(res.ref_stat("retention").mean())
            detail = f" {n_below} of {n_q} qualifying seeds end{'s' if n_below == 1 else ''} below {ex.RETENTION_GATE:g} of its peak (the lowest at {low:.2f}); the seed mean is {st['retention'][1]:.2f} against ex-2.2.3's {ref:.2f}."
        return (
            "miss",
            f"`handover` clears {', '.join(label(k) for k in clears)}, and misses {', '.join(label(k) for k in misses)}.{detail} {alpha}",
        )
    if partials:
        return (
            "partial",
            f"`handover` clears every gate except {', '.join(label(k) for k in partials)}, which sits in its partial band. {alpha}",
        )
    return "pass", f"`handover` clears margin, grading, contrast, lead, retention, and latch. {alpha}"


def h2_figure(res: Results) -> str:
    conds = [c.name for c in ex.CONDS]
    gates = h2_gates()
    keys = [*gates, "retention"]
    vals = {(c, k): res.stat(c, k) for c in conds for k in keys}
    refs = {k: res.ref_stat(k) for k in keys}
    titles = {k: g[0] for k, g in gates.items()} | {"retention": "retention"}

    peak = {c: res.stat(c, "m_line_peak") for c in conds} | {"reference": res.ref_stat("m_line_peak")}
    return h2_draw(conds, keys, vals, refs, peak, titles)


@memo
def h2_draw(
    conds: list[str],
    keys: list[str],
    vals: dict[tuple[str, str], np.ndarray],
    refs: dict[str, np.ndarray],
    peak: dict[str, np.ndarray],
    titles: dict[str, str],
) -> str:
    gates = h2_gates()

    @themed(
        name="h2-placement",
        alt_text="""
            Six small dot panels, one per placement statistic, each with the five conditions and the six-op reference along the bottom. On the margin, grading, lead, and contrast panels the anchored conditions sit close together and near the reference, above the dashed gate line, while the control sits at zero. On the retention panel the anchored conditions sit a little under the reference and one whole-line seed falls below the gate line. On the alignment panel every anchored condition sits above the dashed line, the two untied conditions well above the tied one; here above the line is the drift the prediction expected.
        """,
        caption=f"""
            **Placement of *red* on the `mix` lines, per condition, beside the six-op reference.** The read is on `mix` because the gates were set on `mix` at ex-2.2.3, so these numbers compare with the reference's. One panel per statistic; each small dot is one seed, the larger mark the seed mean, and the thin bar the seed range, with ex-2.2.3's twenty reference seeds at the right. The dashed line is the gate (margin {ex.MARGIN_RATIO * ex.REF_M_LINE:.3f}, grading r² {ex.GRADE_R2_RATIO * ex.REF_R2_SIM:.3f}, lead {ex.LEAD_GATE:g}, contrast {ex.CONTRAST_GATE:g}, retention {ex.RETENTION_GATE:g}), the hatched region the side that misses it, and the dotted line the partial level where there is one. The retention column for `control` is empty because there is no anchor for it to hold. On the ᾱ panel the dashed line is the {ex.MEAN_ALIGN_REF:g} the earlier experiments gated, which here is a prediction: we expected the untied conditions above it.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.0), layout="constrained")
        rng = np.random.default_rng(1)
        names = [*conds, "reference"]
        for ax, k in zip(axes.flat, keys, strict=True):
            for i, c in enumerate(names):
                v = refs[k] if c == "reference" else vals[c, k]
                if k == "retention":
                    v = v[peak[c] >= ex.RETENTION_FLOOR]
                if len(v):
                    dots(ax, i, v, c, rng=rng)
            if k == "alpha_op1":
                gate_line(ax, gates[k][1])  # a prediction, not a gate: no side to hatch
            elif k in gates:
                gate_line(ax, gates[k][1], partial=gates[k][2], fail="below")
            else:
                ax.set_ylim(min(ax.get_ylim()[0], ex.RETENTION_GATE - 0.05), None)
                gate_line(ax, ex.RETENTION_GATE, fail="below")
            ax.set_title(titles[k], fontsize=9)
            ax.set_xticks(
                range(len(names)), [n.replace("handover-", "h-") for n in names], rotation=40, ha="right", fontsize=7
            )
            ax.grid(axis="y", alpha=0.2)
        return fig

    return _plot()


def h2_profile_figure(res: Results) -> str:
    """The softmin profile over the six roles, per condition: where the labeller's weight lands."""
    conds = [c.name for c in ex.CONDS if c.name != "control"]
    prof = {c: np.asarray(res.stat(c, "pi6", ex.PRIMARY_OP), float)[:, 1:, :].mean(axis=1) for c in conds}  # (seeds, 6)
    roles = ["op1", "op", "op2", "=", "answer", "⏎"]

    return h2_profile_draw(conds, prof, roles)


@memo
def h2_profile_draw(conds: list[str], prof: dict[str, np.ndarray], roles: list[str]) -> str:
    @themed(
        name="h2-softmin-profile",
        alt_text="""
            A stepped chart with the six token roles along the bottom and softmin weight up the side, one series per anchored condition drawn as level marks joined by faint risers. Every series peaks at op1 and op2 and is near zero at the op word and the equals sign; the whole-line conditions put a little weight on the answer and the newline, the operand-only condition none.
        """,
        caption="""
            **Where the label's weight lands, by token role.** The red group's softmin weight at each of the six positions, averaged over the four post-attention slices and the seeds, one series per anchored condition, with the seed range as a band. The whole-line labeller (every condition but `handover-slot`) may place weight on the answer and the newline; the either-slot labeller only on the operands.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.0), layout="constrained")
        x = np.arange(len(roles))
        for c in conds:
            p = prof[c]
            smooth_step_band(ax, x, p.min(0), p.max(0), ramp=0.5, color=ink(c), alpha=0.15, zorder=1)
            smooth_step_marks(ax, x, p.mean(0), ramp=0.5, color=ink(c), lw=1.6, zorder=3, label=c)
        ax.set_xticks(x, roles)
        ax.set_ylabel("softmin weight")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.2)
        fig_legend(fig, ax)
        return fig

    return _plot()


def h2_table(res: Results) -> str:
    gates = h2_gates()
    keys = [*gates, "retention", "latch_pi"]
    conds = [c.name for c in ex.CONDS]
    st = {c: h2_status(res, c) for c in conds}
    rows = []
    for c in conds:
        cells = [f"`{c}`"]
        for k in keys:
            if k == "latch_pi":
                v = res.stat(c, k)
                cells.append(bold_if(f"{v.max():.2f}", v.max() <= ex.LATCH_PI) if c == "handover" else f"{v.max():.2f}")
                continue
            v = res.stat(c, k)
            if k == "retention":
                v = v[res.stat(c, "m_line_peak") >= ex.RETENTION_FLOOR]
            text = span2(v, 2 if k in ("retention",) else 3)
            ok = st[c][k][0] == "pass"
            cells.append(bold_if(text, ok) if c == "handover" and k != "alpha_op1" else text)
        rows.append(cells)
    ref = ["reference (ex-2.2.3)"]
    for k in keys:
        v = res.ref_stat(k)
        if k == "retention":
            v = v[res.ref_stat("m_line_peak") >= ex.RETENTION_FLOOR]
        ref.append(f"{v.max():.2f}" if k == "latch_pi" else span2(v, 2 if k == "retention" else 3))
    rows.append(ref)
    head = ["condition", *[f"{g[0]} {g[3]}" for g in gates.values()], "retention ↑", "latch (max) ↓"]
    bands = ", ".join(
        f"{gates[k][0]} {band(ex.NOISE_RUN[k], res.n('handover'), len(res.ref_runs())):.3f}"
        for k in ("m_line", "alpha_op1", "r2_sim", "contrast")
    )
    return table_html(
        head,
        rows,
        f"""
        **Placement statistics on the `mix` lines, per condition.** Seed mean with the seed range in brackets; retention is read on the runs whose peak m_line reached {ex.RETENTION_FLOOR:g}, and latch is the largest non-red softmin weight at op1 over the seeds, against {ex.LATCH_PI:g}. Bold `handover` entries clear their gate. The bands against the reference, from the per-run σ frozen at ex-2.2.3: {bands}.
        """,
        ref_rows=frozenset({len(rows) - 1}),
    )


def h2_prose(res: Results) -> str:
    st = h2_status(res)
    tied, slot = h2_status(res, "handover-tied"), h2_status(res, "handover-slot")
    ref_alpha = float(res.ref_stat("alpha_op1").mean())
    ref_m = float(res.ref_stat("m_line").mean())
    n_below, n_q, low = retention_detail(res, "handover")
    ref_steps = ex.ex223.EPOCHS_SHORT * ex.ex223.steps_per_epoch()
    tied_lower = tied["alpha_op1"][1] < min(st["alpha_op1"][1], slot["alpha_op1"][1])
    return (
        f"The anchor does land where we put it, holding the spot only a little less well than the references do. On the `mix` lines `handover` reaches a margin of {st['m_line'][1]:.3f} against the reference's {ref_m:.3f} (the gate is {ex.MARGIN_RATIO * ex.REF_M_LINE:.3f}), a grading r² of {st['r2_sim'][1]:.2f}, a lead at the embedding of {st['lead_emb'][1]:.2f}, and a contrast of {st['contrast'][1]:.2f}.\n\nᾱ at op1 reads {st['alpha_op1'][1]:.2f} on `handover`, {slot['alpha_op1'][1]:.2f} on `handover-slot`, and {tied['alpha_op1'][1]:.2f} on `handover-tied`, against {ref_alpha:.2f} at the reference. "
        + (
            "So the untied readout is what raises ᾱ, as ex-2.2.7 saw, and the tied condition sits with the reference. That runs against the first intuition, which is that a separate readout should leave the embedding cleaner, and we have one mechanism to offer for it. ᾱ at op1 is read on the first token, whose state at every slice is a function of that color's embedding row alone (nothing precedes it to attend to), so it is close to the mean axis component of the 216 color embeddings. With a tied table those rows are also the readout rows, so any axis component on a non-red color's row raises that color's logit whenever the state at `=` carries *red*, and the task loss pushes it back off. Untying removes that pressure: the embedding rows are then free to carry a small common component along e₁, which costs the task nothing because the readout no longer reads it. The same release is what H4 sees on the syntax words, from the other side: there the tied table had put axis onto `=` because it helped the output, and untying let it come off. Both are the output loss letting go of the embedding. The check, for the next round, is the mean axis component of the non-red embedding rows on `handover` against `handover-tied`, which the stored tables can give without a run."
            if tied_lower
            else "The tied condition does not sit below the untied ones, so untying is not on its own what moves ᾱ here. "
        )
        + f"\n\nRetention is where the hypothesis misses: the seed mean is {st['retention'][1]:.2f} on `handover` against {slot['retention'][1]:.2f} on `handover-slot`, {tied['retention'][1]:.2f} on `handover-tied`, and {float(res.ref_stat('retention').mean()):.2f} at ex-2.2.3, and {n_below} of {n_q} qualifying seeds end{'s' if n_below == 1 else ''} at {low:.2f}, under the {ex.RETENTION_GATE:g} gate."
        + f" One caveat on every ratio against the reference: those runs trained for {ref_steps:,} steps and ours for {ex.HANDOVER.steps:,}, so a margin or an ᾱ that sits above the reference could be the longer training as much as the grammar. The same difference bears on retention. The anchor weight anneals over the last tenth of training, which here is {ex.HANDOVER.steps // 10:,} steps against {ref_steps // 10:,} at the reference, so the margin has three times as many steps to drift after the weight comes down, and the seed spread on ᾱ and retention looks like the spread we saw before the schedules were tuned in D2.1, which a later anneal settled. A schedule read on this grammar (a later or shorter anneal, or one set in steps rather than as a share of training) is the first thing to try on this miss."
    )


def h3_status(res: Results) -> dict:
    """The removal read per op (kept share under `projection`, `handover`) and the selectivity read on `mix`."""
    kept = {op: float(res.kept("handover", op, "projection").mean()) for op in ex.OP_NAMES}
    d = res.deficit("handover", ex.PRIMARY_OP, "projection")
    sd = res.ref_deficit_sd(ex.PRIMARY_OP)
    b = band(sd, res.n("handover"), len(res.ref_deficit(ex.PRIMARY_OP)))
    m = float(d.mean())
    if m <= ex.NONRED_DEFICIT_GATE:
        sel = "pass"
    elif m <= ex.NONRED_DEFICIT_GATE + b:
        sel = "unresolved"
    elif m <= ex.NONRED_DEFICIT_PARTIAL:
        sel = "partial"
    else:
        sel = "miss"
    removal_misses = [op for op in ex.OP_NAMES if kept[op] > ex.RED_KEPT_GATE]
    return {
        "kept": kept,
        "removal": "pass" if not removal_misses else "miss",
        "removal_misses": removal_misses,
        "deficit": m,
        "deficit_band": b,
        "deficit_sd": sd,
        "selectivity": sel,
    }


def h3_verdict(res: Results) -> tuple[str, str]:
    st = h3_status(res)
    worst = max(st["kept"], key=lambda op: st["kept"][op])
    removal = (
        f"Removal: under `projection`, `handover` keeps at most {st['kept'][worst]:.0%} of its clean expected exact match on the removal lines (`{worst}`), inside the {ex.RED_KEPT_GATE:.0%} gate on every op."
        if st["removal"] == "pass"
        else f"Removal: on {', '.join(f'`{o}`' for o in st['removal_misses'])} `handover` still answers more than {ex.RED_KEPT_GATE:.0%} of its removal lines with the axis taken out (the most on `{worst}`, at {st['kept'][worst]:.0%} of its clean expected exact match), where the gate asks for {ex.RED_KEPT_GATE:.0%} or less."
    )
    sel = f"Selectivity {'holds' if st['selectivity'] == 'pass' else 'is ' + st['selectivity']}: the non-red `mix` deficit is {st['deficit']:.3f} against the {ex.NONRED_DEFICIT_GATE:g} gate (band {st['deficit_band']:.3f})."
    status = {("pass", "pass"): "pass"}.get((st["removal"], st["selectivity"]))
    if status is None:
        status = (
            "miss"
            if "miss" in (st["removal"], st["selectivity"])
            else st["selectivity"]
            if st["removal"] == "pass"
            else "miss"
        )
    return status, f"{removal} {sel}"


def h3_figure(res: Results) -> str:
    conds = [c.name for c in ex.CONDS if c.name != "control"]
    kept = {(c, op): res.kept(c, op, "projection") for c in conds for op in ex.OP_NAMES}
    deficit = {(c, op): res.deficit(c, op, "projection") for c in conds for op in ex.OP_NAMES}

    return h3_draw(conds, kept, deficit)


@memo
def h3_draw(
    conds: list[str], kept: dict[tuple[str, str], np.ndarray], deficit: dict[tuple[str, str], np.ndarray]
) -> str:
    @themed(
        name="h3-removal-selectivity",
        alt_text="""
            Two dot panels stacked, with the eleven ops along the bottom of each. The top panel, the share of clean accuracy kept on the removal lines, has every anchored condition's dots under the dashed gate line at one fifth on eight ops, and above it on the three ops that take one HSV attribute from their second operand, where the whole-line conditions sit highest. The bottom panel, the deficit on the non-red lines, has the dots near zero on every op, well under the dashed gate line; a few seeds of the tied condition sit higher on one or two ops.
        """,
        caption=f"""
            **Removal and selectivity under `projection`, per op and anchored condition.** Top: the share of the clean expected exact match the model keeps on each op's removal lines (red lines whose answer moves by at least {ex.FAR_MOVE:g} when the red operand loses its red); the dashed line is the {ex.RED_KEPT_GATE:.0%} gate, and the hatched region above it is the miss. Bottom: the deficit in expected exact match on each op's non-red lines; dashed at the {ex.NONRED_DEFICIT_GATE:g} gate, hatched above it, and dotted at the {ex.NONRED_DEFICIT_PARTIAL:g} partial level. The gate on the deficit is read on `mix` only. Each small dot is one seed, the larger mark the seed mean, and the thin bar the seed range.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (top, bot) = plt.subplots(2, 1, figsize=(8.4, 5.6), layout="constrained", sharex=True)
        rng = np.random.default_rng(2)
        xs = np.arange(len(ex.OP_NAMES))
        off = np.linspace(-0.3, 0.3, len(conds))
        for x, op in zip(xs, ex.OP_NAMES, strict=True):
            for o, c in zip(off, conds, strict=True):
                dots(top, x + o, kept[c, op], c, rng=rng, ms=4, width=0.04, label=c if x == 0 else None)
                dots(bot, x + o, deficit[c, op], c, rng=rng, ms=4, width=0.04)
        top.set_ylim(-0.02, max(0.5, top.get_ylim()[1]))
        gate_line(top, ex.RED_KEPT_GATE, fail="above")
        gate_line(bot, ex.NONRED_DEFICIT_GATE, partial=ex.NONRED_DEFICIT_PARTIAL, fail="above")
        top.set_ylabel("kept on removal lines")
        bot.set_ylabel("non-red deficit")
        bot.set_xticks(xs, ex.OP_NAMES, rotation=30, ha="right", fontsize=8)
        for ax in (top, bot):
            ax.grid(axis="y", alpha=0.2)
        fig_legend(fig, top)
        return fig

    return _plot()


def h3_table(res: Results) -> str:
    """Two tables, removal and selectivity, so that neither runs wider than the page."""
    return h3_removal_table(res) + "\n\n" + h3_selectivity_table(res)


def h3_removal_table(res: Results) -> str:
    conds = [c.name for c in ex.CONDS if c.name != "control"]
    rows = []
    for op in ex.OP_NAMES:
        cells = [f"`{op}`"]
        for c in conds:
            v = float(res.kept(c, op, "projection").mean())
            cells.append(bold_if(f"{v:.2f}", v <= ex.RED_KEPT_GATE) if c == "handover" else f"{v:.2f}")
        for operator in ("operands", "shaped-a0.4-p0"):
            cells.append(f"{res.kept('handover', op, operator).mean():.2f}")
        rows.append(cells)
    head = ["op", *[f"`{c}` ↓" for c in conds], "`operands` ↓", "`shaped` ↓"]
    return table_html(
        head,
        rows,
        f"""
        **Removal, per op: the share of clean expected exact match kept on the removal lines.** Seed means under `projection`, one column per anchored condition, then `handover` under the `operands` and `shaped-a0.4-p0` operators. Bold entries are inside the {ex.RED_KEPT_GATE:.0%} gate, which is read on `handover`.
        """,
    )


def h3_selectivity_table(res: Results) -> str:
    conds = [c.name for c in ex.CONDS if c.name != "control"]
    rows = []
    for op in ex.OP_NAMES:
        cells = [f"`{op}`"]
        for c in conds:
            v = float(res.deficit(c, op, "projection").mean())
            cells.append(
                bold_if(f"{v:.3f}", v <= ex.NONRED_DEFICIT_GATE)
                if c == "handover" and op == ex.PRIMARY_OP
                else f"{v:.3f}"
            )
        sd = res.ref_deficit_sd(op)
        cells.append("—" if math.isnan(sd) else f"{band(sd, res.n('handover'), len(res.ref_deficit(op))):.3f}")
        rows.append(cells)
    head = ["op", *[f"`{c}` ↓" for c in conds], "band"]
    return table_html(
        head,
        rows,
        f"""
        **Selectivity, per op: the drop in expected exact match on the non-red lines.** Seed means under `projection`, one column per anchored condition. The gate ({ex.NONRED_DEFICIT_GATE:g}) is read on `mix` only, where the bold entry is inside it. The band is the smallest difference from the reference the read resolves, from the per-run σ ex-2.2.8 measured at the reference's twenty seeds ({ex.DEFICIT_NOISE}); the ops the six-op grammar did not have carry none.
        """,
    )


def h3_no_gate_reads(res: Results) -> dict:
    """The two H3 reads with a direction and no gate: the rank correlation between the to-zero move and the
    `projection` distance across ops, and the largest gap between `operands` and `projection` on any op.
    """
    move = {op: float(np.mean([r["ops"][op]["move"]["removal"] for r in res.scored("handover")])) for op in ex.OP_NAMES}
    proj = {op: float(res.score("handover", op, "projection", "expected_dist", "removal").mean()) for op in ex.OP_NAMES}
    opnd = {op: float(res.score("handover", op, "operands", "expected_dist", "removal").mean()) for op in ex.OP_NAMES}
    rank = lambda d: np.argsort(np.argsort([d[op] for op in ex.OP_NAMES]))  # noqa: E731
    rho = float(np.corrcoef(rank(move), rank(proj))[0, 1])
    top_move = sorted(ex.OP_NAMES, key=lambda op: -move[op])[:3]
    top_proj = sorted(ex.OP_NAMES, key=lambda op: -proj[op])[:3]
    gap = {op: opnd[op] - proj[op] for op in ex.OP_NAMES}
    widest = max(gap, key=lambda op: abs(gap[op]))
    return {"rho": rho, "top_move": top_move, "top_proj": top_proj, "gap": gap[widest], "widest": widest}


def h3_distance_table(res: Results) -> str:
    """How far the answer moves under `projection` on the removal lines, beside the to-zero move."""
    floor = distance_floor()
    rows = []
    for op in ex.OP_NAMES:
        move = float(np.mean([r["ops"][op]["move"]["removal"] for r in res.scored("handover")]))
        clean = float(res.score("handover", op, None, "expected_dist", "removal").mean())
        proj = float(res.score("handover", op, "projection", "expected_dist", "removal").mean())
        opnd = float(res.score("handover", op, "operands", "expected_dist", "removal").mean())
        rows.append([f"`{op}`", f"{move:.2f}", f"{floor[op]:.2f}", f"{clean:.2f}", f"{proj:.2f}", f"{opnd:.2f}"])
    head = ["op", "true answer moves", "floor", "clean", "`projection`", "`operands`"]
    return table_html(
        head,
        rows,
        """
        **How far the answer moves on the removal lines, per op.** Distances in the unit RGB cube, `handover` seed means. *True answer moves* is how far the true answer moves when the red operand loses its red (the to-zero move, from the corpus): the distance a model that had lost *red* would be expected to miss by. The last three columns are the expected distance from the model's answer distribution to the line's answer: on the clean pass, and under the two projection operators. *Floor* is that same distance for the true answer distribution itself, which is above zero on the ops that round because the drawn answer can land on either neighbour; the clean column is read against it.
        """,
    )


@memo
def distance_floor() -> dict[str, float]:
    """Per op, on its removal lines: the expected distance from the true answer distribution to the line's rounded
    answer, in the unit cube. The value a model with the true distribution would score on `expected_dist`.
    """
    out = {}
    for op in ex.TABLE:
        pl = probe_lines(op, ex.N_PROBE, ex.PROBE_SEED, both_slots=op.name in ex.ORDER_SENSITIVE)
        removal = [
            ln for ln in pl if dose(ln.lhs, ln.rhs) >= ex.RED_DOSE and to_zero_move(op, ln.lhs, ln.rhs) >= ex.FAR_MOVE
        ]
        out[op.name] = float(
            np.mean(
                [
                    sum(
                        p * np.linalg.norm(np.subtract(c, ln.result)) / TOP
                        for c, p in answer_dist(op, ln.lhs, ln.rhs).items()
                    )
                    for ln in removal
                ]
            )
        )
    return out


def h3_prose(res: Results) -> str:
    st = h3_status(res)
    kept = st["kept"]
    above = h3_removal_misses(res, "handover")
    inside = {op: v for op, v in kept.items() if op not in above}
    best, worst = min(inside, key=lambda op: inside[op]), max(inside, key=lambda op: inside[op])
    others = {op: float(res.deficit("handover", op, "projection").mean()) for op in ex.OP_NAMES}
    hi = max(others, key=lambda op: others[op])
    opnd = float(res.deficit("handover", ex.PRIMARY_OP, "operands").mean())
    d = res.deficit("handover", ex.PRIMARY_OP, "projection")
    ng = h3_no_gate_reads(res)
    if above:
        removal = f"Taking the axis out removes *red* on {len(inside)} of the {len(kept)} ops: on their removal lines `handover` keeps between {inside[best]:.0%} (`{best}`) and {inside[worst]:.0%} (`{worst}`) of its clean expected exact match under `projection`. On {', '.join(f'`{op}`' for op in above)} it keeps {', '.join(f'{kept[op]:.0%}' for op in above)}, above the {ex.RED_KEPT_GATE:.0%} gate; the slot split under [The order-sensitive subset, by slot](#the-order-sensitive-subset-by-slot) says which lines those are."
    else:
        removal = f"Taking the axis out removes *red* on every op: on the removal lines `handover` keeps between {kept[best]:.0%} (`{best}`) and {kept[worst]:.0%} (`{worst}`) of its clean expected exact match under `projection`."
    return f"{removal} On the non-red `mix` lines the deficit is {st['deficit']:.3f} (seeds from {d.min():.3f} to {d.max():.3f}), against {float(res.ref_deficit(ex.PRIMARY_OP).mean()):.3f} at the reference; the `operands` edit, which leaves the syntax positions alone, reads {opnd:.3f}. No op's non-red deficit exceeds {others[hi]:.3f}.\n\nOf the two reads with a direction and no gate, neither went the predicted way. The rank correlation between how far the true answer moves and how far the model's answer moves under `projection` is {ng['rho']:.2f}: the largest true moves are on {', '.join(f'`{op}`' for op in ng['top_move'])} and the largest model moves on {', '.join(f'`{op}`' for op in ng['top_proj'])}. And `operands` removes as much as `projection` on every op (the widest gap is {ng['gap']:+.3f}, on `{ng['widest']}`), where the prediction had it removing less on the ops computed at `=`."


def h4_status(res: Results) -> dict:
    words = ex.SYNTAX_WORDS
    comp = lambda c, table="rows": np.array(  # noqa: E731
        [np.mean([abs(r[table][w]) for w in words]) for r in res.runs(c) if r[table] is not None], float
    )
    h, t = comp("handover"), comp("handover-tied")
    sd = pooled_sd(h, t)
    b = band(sd, len(h), len(t))
    eq_h = np.abs(res.component("handover", "="))
    eq_ceiling = np.abs(res.ex227_component(ex.EX227_CEILING, "="))
    eq_sd = pooled_sd(eq_h, eq_ceiling)
    eq_b = band(eq_sd, len(eq_h), len(eq_ceiling))
    d_h = res.deficit("handover", ex.PRIMARY_OP, "projection")
    d_t = res.deficit("handover-tied", ex.PRIMARY_OP, "projection")
    d_b = band(res.ref_deficit_sd(ex.PRIMARY_OP), len(d_h), len(d_t))
    return {
        "component": (float(h.mean()), float(t.mean()), b, sd),
        "component_holds": float(t.mean() - h.mean()) > b,
        "eq": (float(eq_h.mean()), float(eq_ceiling.mean()), eq_b),
        "eq_holds": abs(float(eq_h.mean() - eq_ceiling.mean())) <= eq_b,
        "readout": float(np.mean([abs(r["rows_readout"]["="]) for r in res.runs("handover")])),
        "readout_all": float(comp("handover", "rows_readout").mean()),
        "deficit": (float(d_h.mean()), float(d_t.mean()), d_b),
        "deficit_holds": float(d_t.mean() - d_h.mean()) > d_b,
    }


def h4_verdict(res: Results) -> tuple[str, str]:
    st = h4_status(res)
    a, e, b = st["component_holds"], st["eq_holds"], st["deficit_holds"]
    c1 = (
        f"The syntax embeddings carry less of the axis on `handover` ({st['component'][0]:.3f}) than on `handover-tied` ({st['component'][1]:.3f}), a gap well over the band ({st['component'][2]:.3f})"
        if a
        else f"The syntax embeddings do not carry less of the axis on `handover` ({st['component'][0]:.3f}) than on `handover-tied` ({st['component'][1]:.3f}; band {st['component'][2]:.3f})"
    )
    ce = (
        f"On `=` `handover` reads {st['eq'][0]:.3f}, within a band ({st['eq'][2]:.3f}) of ex-2.2.7's hard-zeroed {st['eq'][1]:.3f}"
        if e
        else f"`=` is the one word that did not reach the ceiling: `handover` reads {st['eq'][0]:.3f} there, against ex-2.2.7's hard-zeroed {st['eq'][1]:.3f} (band {st['eq'][2]:.3f})"
    ) + f", and its readout row for `=` carries {st['readout']:.3f}, so the component moved to the readout"
    c2 = f"the non-red `mix` deficit is {'lower' if b else 'not lower'} on `handover`, {st['deficit'][0]:.3f} against {st['deficit'][1]:.3f} on `handover-tied` (band {st['deficit'][2]:.3f})"
    first = "holds" if a and e else "holds except on `=`" if a else "does not hold"
    second = "holds" if b else "does not hold"
    status = "pass" if a and e and b else "partial" if a or b else "miss"
    return status, f"The first prediction {first}. {c1}. {ce}. The second {second}: {c2}."


def h4_figure(res: Results) -> str:
    words = list(ex.SYNTAX_WORDS)
    labels = [w if w != "\n" else "⏎" for w in words]
    series = {
        ("handover", "rows"): np.stack([np.abs(res.component("handover", w)) for w in words], 1),
        ("handover-tied", "rows"): np.stack([np.abs(res.component("handover-tied", w)) for w in words], 1),
        ("handover", "rows_readout"): np.stack(
            [np.abs(res.component("handover", w, "rows_readout")) for w in words], 1
        ),
    }
    ceiling = np.stack([np.abs(res.ex227_component(ex.EX227_CEILING, w)) for w in words], 1)
    d = {c: res.deficit(c, ex.PRIMARY_OP, "projection") for c in ("handover", "handover-tied")}

    return h4_draw(words, labels, series, ceiling, d)


@memo
def h4_draw(
    words: list[str],
    labels: list[str],
    series: dict[tuple[str, str], np.ndarray],
    ceiling: np.ndarray,
    d: dict[str, np.ndarray],
) -> str:
    @themed(
        name="h4-syntax-component",
        alt_text="""
            Two panels. Left, a dot chart of the axis component on each syntax word's row, with the tied condition's embedding rows sitting higher than the untied condition's on the equals sign and the newline, and the untied condition's readout rows sitting where the tied embedding rows do. Right, the non-red deficit under projection for the two conditions, as columns of seed dots.
        """,
        caption="""
            **The *red* axis on the syntax tokens, and what it costs.** Left: the absolute axis component of each syntax word's row (the op words, `=`, and `⏎`), seed mean with the seed range as a bar, for `handover-tied`'s embedding rows, `handover`'s embedding rows, and `handover`'s readout rows (the open marks). The short bars are ex-2.2.7's hard-zeroed ceiling, where the embedding component is zero by construction. Right: the deficit in expected exact match on the non-red `mix` lines under `projection`, one small dot per seed and the seed mean as the larger mark.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained", width_ratios=[3, 1])
        x = np.arange(len(words))
        for c, table, o, mfc, label in (
            ("handover-tied", "rows", -0.3, None, "handover-tied, embedding"),
            ("handover", "rows", 0.0, None, "handover, embedding"),
            ("handover", "rows_readout", 0.3, "none", "handover, readout"),
        ):
            v = series[c, table]
            m, lo, hi = v.mean(0), v.min(0), v.max(0)
            left.errorbar(
                x + o,
                m,
                yerr=[m - lo, hi - m],
                fmt=marker(c),
                ms=4.5,
                color=ink(c),
                mfc=mfc or ink(c),
                lw=0.8,
                capsize=0,
                zorder=3,
                label=label,
            )
        left.plot(
            x - 0.15, ceiling.mean(0), "_", ms=9, color=ink("reference"), mew=1.5, zorder=4, label="ex-2.2.7 ceiling"
        )
        left.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
        left.set_ylabel("|axis component|")
        left.set_ylim(0, None)
        left.grid(axis="y", alpha=0.2)
        fig_legend(fig, left)
        rng = np.random.default_rng(3)
        for i, c in enumerate(d):
            dots(right, i, d[c], c, rng=rng)
        right.set_xticks([0, 1], ["handover", "h-tied"], fontsize=8)
        right.set_ylabel("non-red mix deficit")
        right.grid(axis="y", alpha=0.2)
        return fig

    return _plot()


def h4_table(res: Results) -> str:
    st = h4_status(res)
    words = list(ex.SYNTAX_WORDS)
    label = lambda w: "`⏎`" if w == "\n" else f"`{w}`"  # noqa: E731
    rows = []
    for w in words:
        rows.append(
            [
                label(w),
                f"{np.abs(res.component('handover-tied', w)).mean():.3f}",
                f"{np.abs(res.component('handover', w)).mean():.3f}",
                f"{np.abs(res.component('handover', w, 'rows_readout')).mean():.3f}",
                "—" if np.isnan(c := np.abs(res.ex227_component(ex.EX227_CEILING, w)).mean()) else f"{c:.3f}",
            ]
        )
    rows.append(
        ["all syntax words", f"{st['component'][1]:.3f}", f"{st['component'][0]:.3f}", f"{st['readout_all']:.3f}", "—"]
    )
    head = ["word", "`handover-tied` embedding ↓", "`handover` embedding ↓", "`handover` readout", "ex-2.2.7 ceiling"]
    return table_html(
        head,
        rows,
        f"""
        **Absolute axis component per syntax word.** Seed means over each condition's runs, on the embedding table and, for `handover`, on its readout table; the last column is ex-2.2.7's hard-zeroed ceiling on the embedding. The band between the two conditions on the all-words mean is {st["component"][2]:.3f} (σ {st["component"][3]:.3f}, {ex.COMPONENT_NOISE}); on `=` the band against the ceiling is {st["eq"][2]:.3f}.
        """,
    )


def h4_prose(res: Results) -> str:
    st = h4_status(res)
    comp = {w: float(np.mean([abs(r["rows"][w]) for r in res.runs("handover")])) for w in ex.SYNTAX_WORDS}
    eol = comp["\n"]
    rest = max(v for w, v in comp.items() if w != "\n")
    where = "within" if st["eq_holds"] else "outside"
    return f"The separate readout keeps the axis off the syntax embeddings on this grammar too. Averaged over the op words, `=`, and `⏎`, the e₁ component of the embedding rows is {st['component'][0]:.3f} on `handover` and {st['component'][1]:.3f} on `handover-tied`. On `=` alone, `handover` reads {st['eq'][0]:.3f}, {where} a band ({st['eq'][2]:.3f}) of ex-2.2.7's hard-zeroed {st['eq'][1]:.3f}, while its readout row for `=` carries {st['readout']:.3f}: the component moved to the readout, as it did in the pilot. Under `projection` the non-red `mix` deficit is {st['deficit'][0]:.3f} on `handover` against {st['deficit'][1]:.3f} on `handover-tied`, with a band of {st['deficit'][2]:.3f}; both sit under the reference's {float(res.ref_deficit(ex.PRIMARY_OP).mean()):.3f}, so the cost the pilot saw on the shared table is small here on either table. One word the untied table did not clean: `⏎`, whose embedding row reads {eol:.3f} on `handover` where every other syntax word is under {rest:.2f}."


def h5_status(res: Results) -> dict:
    h = res.deficit("handover", ex.PRIMARY_OP, "projection")
    s = res.deficit("handover-slot", ex.PRIMARY_OP, "projection")
    b = band(res.ref_deficit_sd(ex.PRIMARY_OP), len(h), len(s))
    tail_h, tail_s = int((h > ex.TAIL).sum()), int((s > ex.TAIL).sum())
    return {
        "mean": (float(h.mean()), float(s.mean()), b),
        "mean_holds": abs(float(h.mean() - s.mean())) < b,
        "tail": (tail_h, tail_s),
        "tail_holds": tail_h - tail_s <= 2,
        "operands": (
            float(res.deficit("handover", ex.PRIMARY_OP, "operands").mean()),
            float(res.deficit("handover-slot", ex.PRIMARY_OP, "operands").mean()),
        ),
    }


def h5_verdict(res: Results) -> tuple[str, str]:
    st = h5_status(res)
    c1 = f"the seed-mean deficits are {st['mean'][0]:.3f} (`handover`) and {st['mean'][1]:.3f} (`handover-slot`), band {st['mean'][2]:.3f}"
    c2 = f"{st['tail'][0]} `handover` seeds and {st['tail'][1]} `handover-slot` seeds sit above {ex.TAIL:g}"
    if st["mean_holds"] and st["tail_holds"]:
        return "pass", f"Both predictions hold: {c1}; {c2}."
    if st["mean_holds"] or st["tail_holds"]:
        return "partial", f"One prediction holds: {c1}; {c2}."
    return "miss", f"Neither prediction holds: {c1}; {c2}."


def h5_figure(res: Results) -> str:
    conds = ("handover", "handover-slot")
    d = {(c, o): res.deficit(c, ex.PRIMARY_OP, o) for c in conds for o in ("projection", "operands")}

    return h5_draw(conds, d)


@memo
def h5_draw(conds: tuple[str, ...], d: dict[tuple[str, str], np.ndarray]) -> str:
    @themed(
        name="h5-per-seed-deficit",
        alt_text="""
            A dot chart with four columns: the whole-line and either-slot conditions under the full projection, then the same two under the operand-only edit. Each column is the per-seed non-red deficit. Every column sits well under the dashed gate line, with the hatched miss region and the dotted tail level above it; the projection columns spread more than the operand-only columns, and the two conditions' means sit close together.
        """,
        caption=f"""
            **The non-red `mix` deficit, seed by seed.** Each small dot is one seed's drop in expected exact match on the non-red `mix` lines, the larger mark the seed mean, `handover` (whole-line label) in red and `handover-slot` (either-slot label) in blue; the left pair is under `projection` and the right pair under the `operands` edit. The dashed line is H3's {ex.NONRED_DEFICIT_GATE:g} gate with the miss hatched above it, drawn as H3 draws it, and the dotted line inside that region is the {ex.TAIL:g} tail level ex-2.2.7 read.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.0, 3.4), layout="constrained")
        rng = np.random.default_rng(4)
        xs = [0, 1, 2.5, 3.5]
        for x, (c, o) in zip(xs, [(c, o) for o in ("projection", "operands") for c in conds], strict=True):
            dots(ax, x, d[c, o], c, rng=rng, width=0.1)
        ax.axhline(ex.TAIL, color=light_dark("#333", "#ddd"), lw=0.7, ls=":", zorder=2)
        gate_line(ax, ex.NONRED_DEFICIT_GATE, fail="above")  # the same gate as H3, drawn the same way; freezes the ylim
        ax.set_xticks(
            xs, ["handover\nprojection", "h-slot\nprojection", "handover\noperands", "h-slot\noperands"], fontsize=8
        )
        ax.set_ylabel("non-red mix deficit")
        ax.grid(axis="y", alpha=0.2)
        return fig

    return _plot()


def h5_table(res: Results) -> str:
    rows = []
    for c in ("handover", "handover-slot"):
        p, o = res.deficit(c, ex.PRIMARY_OP, "projection"), res.deficit(c, ex.PRIMARY_OP, "operands")
        rows.append([f"`{c}`", span2(p), f"{int((p > ex.TAIL).sum())} of {len(p)}", span2(o), f"{p.max():.3f}"])
    ref = res.ref_deficit(ex.PRIMARY_OP)
    rows.append(
        ["reference (ex-2.2.8)", span2(ref), f"{int((ref > ex.TAIL).sum())} of {len(ref)}", "—", f"{ref.max():.3f}"]
    )
    st = h5_status(res)
    head = [
        "condition",
        "deficit, `projection` ↓",
        f"seeds above {ex.TAIL:g} ↓",
        "deficit, `operands` ↓",
        "worst seed ↓",
    ]
    return table_html(
        head,
        rows,
        f"""
        **The non-red `mix` deficit per condition.** Seed mean with the seed range, the count of seeds above the {ex.TAIL:g} tail level, the same read under the `operands` edit, and the worst seed. The band between the two conditions is {st["mean"][2]:.3f}, from the per-run σ ex-2.2.8 measured at the reference ({ex.DEFICIT_NOISE}).
        """,
        ref_rows=frozenset({2}),
    )


def h5_prose(res: Results) -> str:
    st = h5_status(res)
    h = res.deficit("handover", ex.PRIMARY_OP, "projection")
    s = res.deficit("handover-slot", ex.PRIMARY_OP, "projection")
    if st["mean_holds"] and st["tail_holds"]:
        lead = "At twenty seeds against twenty, the whole-line label costs no selectivity we can resolve."
    elif st["tail_holds"]:
        lead = "At twenty seeds against twenty, the whole-line label costs a little selectivity on average, without the tail ex-2.2.7 saw."
    else:
        lead = "At twenty seeds against twenty, the whole-line label has a tail that the either-slot label does not."
    return f"{lead} Under `projection` the deficit is {h.mean():.3f} on `handover` (worst seed {h.max():.3f}) and {s.mean():.3f} on `handover-slot` (worst seed {s.max():.3f}); {st['tail'][0]} and {st['tail'][1]} seeds sit above {ex.TAIL:g}. Under the `operands` edit the two read {st['operands'][0]:.3f} and {st['operands'][1]:.3f}."


def decision_misses(res: Results) -> list[str]:
    """Every gate of the decision rule that `handover` does not clear, named with its status."""
    h1, _ = h1_verdict(res)
    h2 = h2_status(res)
    h3 = h3_status(res)
    misses = []
    if h1 != "pass":
        misses.append(f"H1 ({h1})")
    for k, name in (("m_line", "margin"), ("r2_sim", "grading"), ("contrast", "contrast")):
        if h2[k][0] != "pass":
            misses.append(f"H2 {name} ({h2[k][0]})")
    if h3["removal"] != "pass":
        misses.append("H3 removal (miss)")
    if h3["selectivity"] != "pass":
        misses.append(f"H3 selectivity ({h3['selectivity']})")
    return misses


def reported_misses(res: Results) -> list[str]:
    """The H2 reads outside the decision rule (lead, retention, latch) that `handover` does not clear."""
    h2 = h2_status(res)
    return [
        f"H2 {name}"
        for k, name in (("lead_emb", "lead"), ("retention", "retention"), ("latch", "latch"))
        if h2[k][0] != "pass"
    ]


def retention_note(res: Results) -> str:
    """What the references say about retention, said no more firmly than twenty and nine seeds allow."""
    ret = {c: h2_status(res, c)["retention"] for c in ("handover", "handover-slot", "handover-tied")}
    n_below, n_q, low = retention_detail(res, "handover")
    clear = [c for c in ("handover-slot", "handover-tied") if ret[c][0] == "pass"]
    refs = " and ".join(f"`{c}` ({ret[c][1]:.2f})" for c in ("handover-slot", "handover-tied"))
    return (
        f" Retention is outside the rule and is reported: {n_below} of {n_q} qualifying `handover` seeds end{'s' if n_below == 1 else ''} under the {ex.RETENTION_GATE:g} line (the lowest at {low:.2f}), and the seed mean, {ret['handover'][1]:.2f}, is under {refs}."
        + (
            f" {'Both references keep' if len(clear) == 2 else f'`{clear[0]}` keeps'} every seed above the line, which points at "
            + (
                "the label and the readout together"
                if len(clear) == 2
                else "the labeller"
                if clear[0] == "handover-slot"
                else "the readout"
            )
            + f", though at {res.n('handover-slot')} and {res.n('handover-tied')} seeds, with no band on retention, one seed in twenty does not separate that from a one-arm effect."
            if clear
            else " Neither reference keeps every seed above the line either."
        )
    )


def decision(res: Results) -> tuple[bool, str]:
    """(adopted, the decision text). The rule: H1, H2 (margin, grading, contrast in full), and H3 in full."""
    misses = decision_misses(res)
    reported = reported_misses(res)
    note = retention_note(res) if "H2 retention" in reported else ""
    if not misses:
        return (
            True,
            "`handover` clears H1, H2's margin, grading, and contrast, and H3 in full, so **the handover is adopted**: table A+ with stochastic rounding, the whole-line labeller, and the untied readout become the grammar and recipe of record for the anchored-op experiments."
            + note,
        )
    # Which reference clears what the candidate missed? `handover-slot` differs from `handover` by the labeller,
    # `handover-tied` by the readout, so a reference that clears a missed gate points at its change.
    joined = " ".join(misses)
    who = []
    if "H3 removal" in joined:
        who.append(
            attribution(
                "removal", not h3_removal_misses(res, "handover-slot"), not h3_removal_misses(res, "handover-tied")
            )
        )
    if "H3 selectivity" in joined:
        who.append(
            attribution(
                "selectivity",
                h3_status_for(res, "handover-slot") == "pass",
                h3_status_for(res, "handover-tied") == "pass",
            )
        )
    return (
        False,
        f"`handover` misses {', '.join(misses)}, so **the handover is not adopted** as it stands. "
        + " ".join(who)
        + note,
    )


def attribution(gate: str, slot_clears: bool, tied_clears: bool) -> str:
    """One sentence on what the two reference conditions say about a gate `handover` missed."""
    if slot_clears and tied_clears:
        return f"Both references clear {gate}, which points at the whole-line label and the untied readout together."
    if slot_clears:
        return f"`handover-slot` clears {gate} and `handover-tied` does not, which points at the labeller."
    if tied_clears:
        return f"`handover-tied` clears {gate} and `handover-slot` does not, which points at the readout."
    return f"Neither reference clears {gate}, which points at the table or the corpus."


def h3_removal_misses(res: Results, cond: str) -> list[str]:
    """The ops on which one condition keeps more than the removal gate under `projection`, seed mean."""
    return [op for op in ex.OP_NAMES if float(res.kept(cond, op, "projection").mean()) > ex.RED_KEPT_GATE]


def h3_status_for(res: Results, cond: str) -> str:
    """The selectivity status of one condition on `mix`, on the same terms as `h3_status`."""
    m = float(res.deficit(cond, ex.PRIMARY_OP, "projection").mean())
    b = band(res.ref_deficit_sd(ex.PRIMARY_OP), res.n(cond), len(res.ref_deficit(ex.PRIMARY_OP)))
    return (
        "pass"
        if m <= ex.NONRED_DEFICIT_GATE
        else "unresolved"
        if m <= ex.NONRED_DEFICIT_GATE + b
        else "partial"
        if m <= ex.NONRED_DEFICIT_PARTIAL
        else "miss"
    )


def findings_md(res: Results | None) -> str:
    items = [
        ("Does the model still learn the task? (H1)", "does-the-model-still-learn-the-task-h1", h1_verdict),
        ("Does *red* still land where we put it? (H2)", "does-red-still-land-where-we-put-it-h2", h2_verdict),
        ("Can we still take *red* out cleanly? (H3)", "can-we-still-take-red-out-cleanly-h3", h3_verdict),
        (
            "Does the separate readout keep the axis off the syntax tokens? (H4)",
            "does-the-separate-readout-keep-the-axis-off-the-syntax-tokens-h4",
            h4_verdict,
        ),
        ("What does the whole-line label cost? (H5)", "what-does-the-whole-line-label-cost-h5", h5_verdict),
    ]
    # The verdict word is written so it cannot read as an answer to the question in the
    # link: "pass" after "What does the label cost?" looked like one. H1–H3 are gated,
    # so their word names the gate; H4–H5 are predictions, so theirs says whether it held.
    gated = {"pass": "gate cleared", "miss": "gate missed", "partial": "partial", "unresolved": "unresolved"}
    predicted = {"pass": "held", "miss": "did not hold", "partial": "held in part", "unresolved": "unresolved"}
    lines = []
    for title, anchor, fn in items:
        if res is None:
            lines.append(f"- [{title}](#{anchor}) — _to come_")
        else:
            status, line = fn(res)
            word = (gated if fn in (h1_verdict, h2_verdict, h3_verdict) else predicted)[status]
            lines.append(f"- [{title}](#{anchor}) — **{word}.** {line}")
    if res is None:
        lines.append("- [Whether the handover is adopted](#decision) — _to come_")
    else:
        adopted, _ = decision(res)
        lines.append(f"- [Whether the handover is adopted](#decision) — **{'adopted' if adopted else 'not adopted'}.**")
    return "\n".join(lines)


def cube_r2(geometry: dict, cond: str) -> np.ndarray:
    """(seeds, slices, positions, targets, channels) strict R² of the cube probes of one condition."""
    rs = sorted((r for r in geometry["runs"] if r["condition"] == cond), key=lambda r: r["seed"])
    return np.array([r["r2_strict"] for r in rs], float)


def lines_per_op_table(res: Results) -> str:
    """The operand-cube probe scan: how decodable the operands and the answer are, by condition, beside
    ex-2.2.3's six-op control.
    """
    targets = {"op1 at op1": 0, "op2 at op2": 2, "answer at `=`": ex.DECODE_POS}
    conds = [(c, res.geometry) for c in ex.CUBE_PROBED] + [("control", res.ex223_geometry)]
    names = [f"`{c}`" for c in ex.CUBE_PROBED] + ["ex-2.2.3 `control` (six ops)"]
    rows = []
    clipped = 0
    for ti, (t, p) in enumerate(targets.items()):
        cells = [t]
        for c, geo in conds:
            r2 = cube_r2(geo, c)[:, 1:, p, ti, :]
            clipped += int((r2 < 0).sum())
            cells.append(span2(np.clip(r2, 0, 1).mean(axis=(1, 2)), 2))
        rows.append(cells)
    head = ["target, site", *[f"{n} ↑" for n in names]]
    lpo = {c.name: c.lines_per_op for c in ex.CONDS}
    clip_note = "" if clipped == 0 else f" ({clipped} negative scores clipped to zero)"
    return table_html(
        head,
        rows,
        f"""
        **How decodable each color is from the residual stream, by condition.** Strict held-out R² for the RGB of op1 and op2 at their own slot and of the answer at `=`, mean over the channels and the four post-attention slices{clip_note}, as a seed mean with the seed range. `control` and `handover` have {lpo["handover"]:,} lines per op; `handover-narrow` has {lpo["handover-narrow"]:,}, the six-op count, at the same step budget. The last column is ex-2.2.3's un-anchored six-op model on the same read.
        """,
    )


def slot_table(res: Results) -> str:
    """The order-sensitive ops, split by which slot the red operand sits in: placement on each walk, and removal
    on each slot's removal lines.
    """
    rows = []
    for op in ex.ORDER_SENSITIVE:
        m1 = res.stat("handover", "m_line", op)
        m2 = np.array([r["per_op"][op]["op2_walk"]["m_line"] for r in res.runs("handover")], float)
        k1 = res.score("handover", op, "projection", "kept", "removal_op1")
        k2 = res.score("handover", op, "projection", "kept", "removal_op2")
        n1 = res.scored("handover")[0]["ops"][op]["n"]["removal_op1"]
        n2 = res.scored("handover")[0]["ops"][op]["n"]["removal_op2"]
        rows.append([f"`{op}`", span2(m1), span2(m2), f"{k1.mean():.2f} ({n1})", f"{k2.mean():.2f} ({n2})"])
    head = ["op", "m_line, op1 walk ↑", "m_line, op2 walk ↑", "kept, red op1 ↓", "kept, red op2 ↓"]
    return table_html(
        head,
        rows,
        """
        **The order-sensitive ops by slot, on `handover`.** These ops are probed on two sets of lines: one with every palette color at op1 (the *op1 walk*) and one with every palette color at op2 (the *op2 walk*). The margin is read on each walk, and the share of clean expected exact match kept under `projection` is read on the removal lines whose red operand is op1 or op2 (line counts in brackets, pooled over both walks, so about twice the method's per-walk counts). Seed means with the seed range.
        """,
    )


def anchor_calibration_table(res: Results) -> str:
    """P(mode) and KL on the rounded held-out lines, per condition, averaged over the ops that round."""
    ops = [op for op in ex.OP_NAMES if float(res.read("control", op, "n_rounded").mean()) > 0]
    rows = []
    for c in [c.name for c in ex.CONDS]:
        pm = np.mean([res.read(c, op, "p_mode_rounded").mean() for op in ops])
        kl = np.mean([res.read(c, op, "kl_rounded").mean() for op in ops])
        worst = max(
            ops, key=lambda op: abs(res.read(c, op, "kl_rounded").mean() - res.read("control", op, "kl_rounded").mean())
        )
        rows.append(
            [
                f"`{c}`",
                f"{pm:.3f}",
                f"{kl:.3f}",
                f"`{worst}` ({res.read(c, worst, 'kl_rounded').mean() - res.read('control', worst, 'kl_rounded').mean():+.3f})",
            ]
        )
    head = ["condition", "P(mode) ↑", "KL ↓", "op furthest from control (ΔKL)"]
    return table_html(
        head,
        rows,
        f"""
        **Calibration under the anchor.** On the held-out lines that round, the mass on the most likely answer and the KL divergence from the true answer distribution to the model's (nats), averaged over the {len(ops)} ops that round and over seeds. The last column names the op whose KL moved most from the control's, with the difference.
        """,
    )


def red_answer_table(res: Results) -> str:
    """Non-red lines whose true answer is red: does the model still produce red under `projection`?"""
    rows = []
    for op in ex.OP_NAMES:
        n = res.scored("handover")[0]["ops"][op]["n"]["red_answer"]
        if n == 0:
            continue
        clean = res.score("handover", op, None, "eem", "red_answer").mean()
        proj = res.score("handover", op, "projection", "eem", "red_answer").mean()
        excl = res.score("handover", op, "projection", "deficit", "nonred_excl").mean()
        rows.append([f"`{op}`", f"{n}", f"{clean:.2f}", f"{proj:.2f}", f"{excl:.3f}"])
    head = ["op", "lines", "clean EEM", "`projection` EEM ↑", "deficit without them ↓"]
    return table_html(
        head,
        rows,
        """
        **The red-answer lines, on `handover`.** Non-red probe lines whose true answer is red, per op that has any: their count, the expected exact match on the clean pass and under `projection`, and the non-red deficit with these lines set aside. Seed means.
        """,
    )


def lines_per_op_prose(res: Results) -> str:
    r2 = {c: float(np.clip(cube_r2(res.geometry, c)[:, 1:, 0, 0, :], 0, 1).mean()) for c in ex.CUBE_PROBED}
    six = float(np.clip(cube_r2(res.ex223_geometry, "control")[:, 1:, 0, 0, :], 0, 1).mean())
    ref_steps = ex.ex223.EPOCHS_SHORT * ex.ex223.steps_per_epoch()
    if r2["handover"] - r2["handover-narrow"] > 0.05:
        reading = "So fewer lines per op does make the operands less decodable, at a matched step count."
    else:
        reading = "So at a matched step count, fewer lines per op does not make the operands less decodable: the narrow condition is at least as decodable as `handover`."
    return f"Op1 decodes at its own slot with R² {r2['control']:.2f} on the control and {r2['handover']:.2f} on `handover`, against {r2['handover-narrow']:.2f} on `handover-narrow` and {six:.2f} on ex-2.2.3's six-op control. {reading} That does not settle E4's question, though. The six-op control is far below all three, and it also trained for {ref_steps:,} steps against {ex.HANDOVER.steps:,} here, so on this read the op count is confounded with the training length, and the three eleven-op conditions are not ordered by lines per op. The clean comparison would hold the step count and vary the op count."


def slot_prose(res: Results) -> str:
    slot_kept = {
        op: [float(res.score("handover", op, "projection", "kept", f"removal_{k}").mean()) for k in ("op1", "op2")]
        for op in ex.ORDER_SENSITIVE
    }
    walk = {
        op: (
            float(res.stat("handover", "m_line", op).mean()),
            float(np.mean([r["per_op"][op]["op2_walk"]["m_line"] for r in res.runs("handover")])),
        )
        for op in ex.ORDER_SENSITIVE
    }
    return f"We expected removal in both slots at about the rates of the method's per-slot table, and the data went the other way: on `sat-hsv` and `value-hsv` the lines with red at op1 keep {slot_kept['sat-hsv'][0]:.0%} and {slot_kept['value-hsv'][0]:.0%}, and the lines with red at op2 keep {slot_kept['sat-hsv'][1]:.0%} and {slot_kept['value-hsv'][1]:.0%}; `hue-hsv` keeps {slot_kept['hue-hsv'][0]:.0%} with red at op1 and {slot_kept['hue-hsv'][1]:.0%} at op2. Placement differs by slot too: the margin on the op1 walk is {', '.join(f'{walk[op][0]:.3f}' for op in ex.ORDER_SENSITIVE)} against {', '.join(f'{walk[op][1]:.3f}' for op in ex.ORDER_SENSITIVE)} on the op2 walk ({', '.join(f'`{op}`' for op in ex.ORDER_SENSITIVE)}). The prediction had the labeller placing both slots the same, and in hindsight there was no reason to expect symmetry: the labeller pools both operands the same way, but the op's computation is not symmetric, so what the model needs from each slot is not either. What removal rate each slot should show, derived from the op rather than assumed, is the analysis the fast-follow owes; the discussion sketches the reading."


def calibration_prose(res: Results) -> str:
    ops = [op for op in ex.OP_NAMES if float(res.read("control", op, "n_rounded").mean()) > 0]
    kl = {c.name: float(np.mean([res.read(c.name, op, "kl_rounded").mean() for op in ops])) for c in ex.CONDS}
    pm = {c.name: float(np.mean([res.read(c.name, op, "p_mode_rounded").mean() for op in ops])) for c in ex.CONDS}
    anchored = [c.name for c in ex.CONDS if c.lam > 0]
    far = max(anchored, key=lambda c: kl[c])
    far_op = max(
        ops, key=lambda op: res.read(far, op, "kl_rounded").mean() - res.read("control", op, "kl_rounded").mean()
    )
    far_d = float(res.read(far, far_op, "kl_rounded").mean() - res.read("control", far_op, "kl_rounded").mean())
    same = "about where the control has it" if abs(kl["handover"] - kl["control"]) < 0.02 else "away from the control's"
    return (
        f"On the {len(ops)} ops that round, `handover` sits at a KL of {kl['handover']:.3f} nats against {kl['control']:.3f} on the control, with P(mode) {pm['handover']:.3f} against {pm['control']:.3f}: the anchor itself leaves the calibration {same}. "
        + (
            f"`{far}` is the one condition that stands out, at {kl[far]:.3f}, with `{far_op}` the op that moved most ({far_d:+.3f}): fewer lines per op costs calibration, which is the lines-per-op question again from the output side."
            if far == "handover-narrow"
            else f"`{far}` is the furthest from the control, at {kl[far]:.3f}, with `{far_op}` the op that moved most ({far_d:+.3f})."
        )
    )


def red_answer_prose(res: Results) -> str:
    ops = [op for op in ex.OP_NAMES if res.scored("handover")[0]["ops"][op]["n"]["red_answer"] > 0]
    if not ops:
        return "No op has red-answer lines on its probe set."
    drop = {
        op: float(res.score("handover", op, None, "eem", "red_answer").mean())
        - float(res.score("handover", op, "projection", "eem", "red_answer").mean())
        for op in ops
    }
    worst = max(ops, key=lambda op: drop[op])
    n = sum(res.scored("handover")[0]["ops"][op]["n"]["red_answer"] for op in ops)
    if drop[worst] > 0.3:
        reading = f"On the red-answer lines the model largely stops producing red once the axis is projected out at `=` (the largest drop is {drop[worst]:.2f} on `{worst}`): removal on the output side, which says the readout row for red carries the axis as the embedding row does."
    else:
        reading = f"On the {n} red-answer lines, expected exact match under `projection` is within {drop[worst]:.2f} of the clean pass on every op that has them (the largest drop is on `{worst}`). The model still produces red with the axis projected out of the state at `=`, so producing red at the output does not depend on that state's axis component."
    return f"{reading} The last column is the non-red deficit with these lines set aside; it differs from the gated deficit by what they contribute, which is little."


def shaped_prose(res: Results) -> str:
    shaped = {op: float(res.kept("handover", op, "shaped-a0.4-p0").mean()) for op in ex.OP_NAMES}
    shaped_misses = [op for op in ex.OP_NAMES if shaped[op] > ex.RED_KEPT_GATE]
    return (
        f"`shaped-a0.4-p0` keeps between {min(shaped.values()):.0%} and {max(shaped.values()):.0%} on the removal lines, "
        + (
            "inside the removal gate on every op."
            if not shaped_misses
            else f"outside the removal gate on {', '.join(f'`{o}`' for o in shaped_misses)}: its smaller removal does not clear the gate on the ops the projection misses either."
        )
    )


def zeroed(a, b):
    """The line with the redder operand's R set to zero: ex-2.2.4's *to-zero* rule."""
    if redness(a) >= redness(b):
        return (0, a[1], a[2]), b
    return a, (0, b[1], b[2])


def to_zero_move(op, a, b) -> float:
    """How far the true answer moves in the unit cube when the red operand loses its R."""
    return float(np.linalg.norm(np.subtract(op(*zeroed(a, b)), op(a, b))) / TOP)


@memo
def removal_counts() -> dict[str, tuple[int, int, int]]:
    """Per op: (red probe lines, removal lines, non-red lines with a red answer), on the op's probe set as
    ex-2.2.3 draws it.
    """
    out = {}
    for op in ex.TABLE:
        pl = probe_lines(op, ex.N_PROBE, ex.PROBE_SEED)
        red = [(ln.lhs, ln.rhs) for ln in pl if dose(ln.lhs, ln.rhs) >= ex.RED_DOSE]
        far = sum(to_zero_move(op, a, b) >= ex.FAR_MOVE for a, b in red)
        red_answer = sum(dose(ln.lhs, ln.rhs) <= ex.NONRED_DOSE and redness(ln.result) >= ex.RED_DOSE for ln in pl)
        out[op.name] = (len(red), far, red_answer)
    return out


@memo
def slot_counts() -> dict[str, dict[str, tuple[int, int]]]:
    """For the order-sensitive ops: (red lines, removal lines) with the red operand at op1 and at op2, on the
    both-slot probe set.
    """
    cs = colors()
    partners = probe_partners(ex.N_PROBE, ex.PROBE_SEED)[1]
    out = {}
    for name in ex.ORDER_SENSITIVE:
        op = next(o for o in ex.TABLE if o.name == name)
        out[name] = {}
        for slot in ("op1", "op2"):
            ls = [(c, b) if slot == "op1" else (b, c) for c, ps in zip(cs, partners, strict=True) for b in ps]
            red = [(a, b) for a, b in ls if dose(a, b) >= ex.RED_DOSE and (redness(a) >= redness(b)) == (slot == "op1")]
            far = sum(to_zero_move(op, a, b) >= ex.FAR_MOVE for a, b in red)
            out[name][slot] = (len(red), far)
    return out


@memo
def eps_check() -> tuple[int, int]:
    """Removal lines for a red op2 under `sat-hsv`, as the per-slot table counts them, with the red operand's
    R set to zero and to one grid level: the epsilon the review asked about.
    """
    op = next(o for o in ex.TABLE if o.name == "sat-hsv")
    partners = probe_partners(ex.N_PROBE, ex.PROBE_SEED)[1]
    ls = [(b, c) for c, ps in zip(colors(), partners, strict=True) for b in ps]
    red = [(a, b) for a, b in ls if dose(a, b) >= ex.RED_DOSE and redness(a) < redness(b)]

    def far(eps: int) -> int:
        return sum(np.linalg.norm(np.subtract(op(a, (eps, b[1], b[2])), op(a, b))) / TOP >= ex.FAR_MOVE for a, b in red)

    return far(0), far(1)


@memo
def refop_facts() -> tuple[float, float, float]:
    """`hsvmix` as a reference op: its on-grid share of unordered pairs, on-grid partners per color, and the
    expected-exact-match ceiling of its probe set (the chance that two draws from the true answer agree).
    """
    op = next(o for o in ex.TABLE if o.name == ex.SECONDARY_OP)
    cs = colors()
    partners = np.mean([sum(is_on_grid(op, c, b) for b in cs) for c in cs])
    pl = probe_lines(op, ex.N_PROBE, ex.PROBE_SEED)
    ceiling = np.mean([sum(p * p for p in answer_dist(op, ln.lhs, ln.rhs).values()) for ln in pl])
    return on_grid(op), float(partners), float(ceiling)


@memo
def refop_md() -> str:
    counts = removal_counts()
    rows = []
    for name in (ex.PRIMARY_OP, ex.SECONDARY_OP):
        op = next(o for o in ex.TABLE if o.name == name)
        pl = probe_lines(op, ex.N_PROBE, ex.PROBE_SEED)
        exact = np.mean([is_on_grid(op, ln.lhs, ln.rhs) for ln in pl])
        ceiling = np.mean([sum(p * p for p in answer_dist(op, ln.lhs, ln.rhs).values()) for ln in pl])
        red = [(ln.lhs, ln.rhs) for ln in pl if dose(ln.lhs, ln.rhs) >= ex.RED_DOSE]
        move = np.mean([to_zero_move(op, a, b) for a, b in red])
        alone = relevance(op, ex.TABLE, lines()).get(0, 0)
        n_red, far, ra = counts[name]
        rows.append(f"| `{name}` | {exact:.0%} | {ceiling:.2f} | {far / n_red:.0%} | {move:.2f} | {ra} | {alone:.0%} |")
    head = "| op | probe lines on the grid | expected exact match ceiling | red lines that are removal lines | mean to-zero move | red-answer lines | alone |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    return head + "\n".join(rows)


def coverage(lines_per_op: float) -> tuple[float, float]:
    """The share of an op's trainable lines a model sees at least once: over ordered lines, and over unordered
    pairs. Lines are drawn i.i.d. with held-out pairs rejected, so this is 1 − exp(−draws / pool).
    """
    pool_ordered = len(lines()) * (1 - ex.HOLDOUT_FRAC)
    pool_pairs = len(unordered_pairs()) * (1 - ex.HOLDOUT_FRAC)
    return 1 - math.exp(-lines_per_op / pool_ordered), 1 - math.exp(-lines_per_op / pool_pairs)


@memo
def table_md() -> str:
    rows = []
    for op in ex.TABLE:
        group = "kept" if op.name in ex.KEPT else ("added" if op.name in ex.ADDED else "order-sensitive")
        comm = "yes" if commutativity(op) > 0.99 else f"{commutativity(op):.0%}"
        rows.append(f"| `{op.name}` | {op.rule.replace('|', '\\|')} | {comm} | {group} |")
    head = "| op | rule (0..15 scale, snapped to the grid) | commutative | in A+ as |\n| --- | --- | --- | --- |\n"
    return head + "\n".join(rows)


def conds_md() -> str:
    rows = []
    for c in ex.CONDS:
        anchor = "none" if c.lam == 0 else f"λ_a = {c.lam:g}, τ = {ex.TAU:g}"
        labeller = "—" if c.lam == 0 else ("whole line" if c.keying == "line" else "either slot, prompt span")
        readout = "tied" if c.tie else "untied"
        rows.append(
            f"| **{c.name}**, {c.role}: {c.title} | {anchor} | {labeller} | {readout} | {c.lines_per_op:,} | {c.epochs} ({c.steps:,}) | {c.seeds} |"
        )
    head = "| condition | anchor | labeller | readout | lines per op | epochs (steps) | seeds |\n| --- | --- | --- | --- | ---: | ---: | ---: |\n"
    return head + "\n".join(rows)


@memo
def relevance_md() -> str:
    """Op-relevance for the two reference ops under table A+, counted over ordered pairs."""
    ordered = lines()
    rows = []
    for name in (ex.PRIMARY_OP, ex.SECONDARY_OP):
        op = next(o for o in ex.TABLE if o.name == name)
        rel = relevance(op, ex.TABLE, ordered)
        shares = [rel.get(k, 0) for k in range(3)] + [sum(v for k, v in rel.items() if k >= 3)]
        cells = " | ".join(f"{v:.0%}" for v in shares)
        rows.append(f"| `{name}` | {cells} |")
    head = "| anchored op | alone | 1 other agrees | 2 | 3+ |\n| --- | ---: | ---: | ---: | ---: |\n"
    return head + "\n".join(rows)


def removal_md() -> str:
    counts = removal_counts()
    rows = [f"| `{name}` | {red:,} | {far:,} ({far / red:.0%}) | {ra:,} |" for name, (red, far, ra) in counts.items()]
    head = "| op | red probe lines | removal lines | non-red lines with a red answer |\n| --- | ---: | ---: | ---: |\n"
    return head + "\n".join(rows)


def slot_md() -> str:
    rows = []
    for name, slots in slot_counts().items():
        cells = " | ".join(f"{red:,} | {far:,} ({far / red:.0%})" for red, far in slots.values())
        rows.append(f"| `{name}` | {cells} |")
    head = "| op | red at op1 | removal | red at op2 | removal |\n| --- | ---: | ---: | ---: | ---: |\n"
    return (
        head
        + "\n".join(rows)
        + "\n\nEach slot is counted on the walk that puts the palette color in it. The scored probe set pools both walks, so the per-slot counts in the exploratory table are about twice these."
    )


def discussion_md(res: Results | None) -> str:
    if res is None:
        return "*Written after the results.*"
    st3 = h3_status(res)
    n_below, n_q, low = retention_detail(res, "handover")
    slot_kept = {
        op: [float(res.score("handover", op, "projection", "kept", f"removal_{k}").mean()) for k in ("op1", "op2")]
        for op in ex.ORDER_SENSITIVE
    }
    ret = {c: float(h2_status(res, c)["retention"][1]) for c in ("handover", "handover-slot", "handover-tied")}
    ref_ret = float(res.ref_stat("retention").mean())
    eol = float(np.mean([abs(r["rows"]["\n"]) for r in res.runs("handover")]))
    return textwrap.dedent(f"""
    Most of what the handover was meant to settle, it settled, and the decision rule still says no. Eleven ops with drawn answers are learnable, and the anchored model learns them as well as the control does (H1). *Red* lands where the recipe put it at six ops, with the same margin, grading, and contrast (H2). The separate readout keeps the axis off the syntax embeddings on this grammar too (H4), and at twenty seeds against twenty the whole-line label costs no selectivity we can resolve (H5). The selectivity read itself is the cleanest we have had: a non-red `mix` deficit of {st3["deficit"]:.3f}, against {float(res.ref_deficit(ex.PRIMARY_OP).mean()):.3f} at the reference.

    Two reads missed, but both are narrower than "a miss" sounds. Retention, which the decision rule reports and does not count, missed on {n_below} seed of {n_q}, at {low:.2f} against the {ex.RETENTION_GATE:g} gate. The seed mean, {ret["handover"]:.2f}, is under `handover-slot`'s {ret["handover-slot"]:.2f} and `handover-tied`'s {ret["handover-tied"]:.2f} (and ex-2.2.3's {ref_ret:.2f}), and neither reference has a seed under the line. That is consistent with the label and the readout costing a little retention each. But the evidence is one seed in twenty, with no band on retention, and that does not separate a real cost from one seed's luck. The schedule is the other suspect: the anneal here runs over three times as many steps as at the reference, so the margin has longer to drift after its peak, and a later or shorter anneal is the first thing to try. Removal, which the rule does count, missed on the three order-sensitive ops. On `sat-hsv` and `value-hsv` the lines with red at op1 lose almost everything ({slot_kept["sat-hsv"][0]:.0%} and {slot_kept["value-hsv"][0]:.0%} kept), and the lines with red at op2 keep most of it ({slot_kept["sat-hsv"][1]:.0%} and {slot_kept["value-hsv"][1]:.0%}). Red at op2 is the slot where the red operand supplies only its saturation, or only its value. We had expected removal in both slots, so this is a reading made after the fact: the axis carries the *redness* of a color, and a red color's saturation and value are read from somewhere else. If that is right, it is what an anchored concept should look like, and the removal lines are the thing to fix: they were chosen by how far the true answer moves when the red operand loses its red, and on these two ops that counts lines where what moved was the value, which the model never needed *red* for. `hue-hsv` is the case that reading does not cover: red at op1 supplies the hue there, and still keeps {slot_kept["hue-hsv"][0]:.0%}.

    So the grammar and the recipe are not adopted as they stand, but it's close. Before this handover could be adopted, the removal lines for the three order-sensitive ops would need to be the lines whose answer takes the red operand's hue, so that the gate asks the model to have lost *red* rather than the value of a red color; `hue-hsv` would have to clear that gate on its own terms; and retention would need a read that twenty seeds can resolve, or a look at why the margin drifts after its peak on this arm, which the trajectories can show. Whether those changes are made, and the handover re-run, is a question for the next round. One observation goes with the readout: the untied table cleaned every syntax word except `⏎`, whose embedding row still carries {eol:.2f} on `handover`. Where that comes from, and whether it matters, is for that round too.
    """)


r"""
# Ex 2.2.9: the grammar handover

/// tip |
<!-- tl;dr -->
The last four experiments each changed one thing about the setup we anchor *red* in: more operations, answers drawn stochastically rather than rounded, a label that covers the whole line, and a readout table separate from the embeddings. Each looked fine on its own. This experiment turns all four on at once and asks whether *red* still lands where we put it and still comes out cleanly, with two reference conditions that each switch one change back.

*Red* lands. Removal is clean on eight of the eleven ops and misses the gate on the three that take one HSV attribute from their second operand, so the handover is not adopted as it stands. The whole-line label and the separate readout cost nothing the gates can resolve. One seed in twenty holds the anchor a little less well by the end of training, which the anneal schedule may explain.
///
"""

res = load_results()

# REVIEW: the Findings line quoted the whole decision rule, which the Decision section states
# again a few screens later; it is now a link, so the rule has one home. Verify: the rule is
# rendered from `ex.DECISION` under "Decision".
rf"""
## Findings

{findings_md(res)}
"""

r"""
/// admonition | How to read this draft
This is a preregistration. The conditions, the gates, and the decision rule below were written before any run and frozen at commit `7aabc72`; everything after that commit is either the results filled into the frozen sections or exploratory. Each hypothesis section opens with the background, then gives the prediction we will be scored on. Results go into each section in place once they exist. Anything we think of after seeing the data goes under [Exploratory analyses](#exploratory-analyses), marked as post hoc. Every count in the method is computed from `experiment.py` at render time.
///

## Why this experiment

Ex-2.2.3 anchored *red* on a grammar of six operations and then tried to remove it. Removal was partial on four ops, and the ops themselves were confounding. On `lighten`, say, most lines with a red operand have an answer that stays the same when the operand is a little less red, so a model that has lost *red* can still answer them.

Our investigations suggest:

1. **A wider table of operations** ([ex-2.2.4](../ex-2.2.4/report.py)). Drop `add`, add three ops whose answers spread through the color cube, and add three more that take one attribute from one operand and the rest from the other. Those last three are the first ops where the order of the operands matters.
2. **Stochastic answers instead of rounded ones** ([ex-2.2.5](../ex-2.2.5/report.py)). When an answer lands between two grid colors, the corpus picks one of them at random, rather than rounding deterministically. The model then learns a spread of possible answers rather than one biased token.
3. **A label that covers the whole line** ([ex-2.2.6](../ex-2.2.6/report.py)). Previously, only the two operands could draw a label, and the pull covered the prompt. Now the anchor is told "this line is about red", and it pulls wherever in the line it finds the most red, answer included. That is the shape a document-level label will have in M3. The first pilot found it costs nothing; a later one saw a few seeds lose selectivity under it, so it goes in with a check.
4. **A separate, untied readout table** ([ex-2.2.7](../ex-2.2.7/report.py)). The output layer of the model was sharing a table with the input embeddings, and that put part of the *red* axis onto the tokens `=` and `mix`. Giving the output its own table keeps the axis off those tokens.

In [ex-2.2.8](../ex-2.2.8/report.py), we found that the removal operator to score is the plain projection, which takes the axis out everywhere, with the operand-only edit beside it as the selective reference.

Here the four are tried together, at the same number of seeds as the reference, to inform the grammar and recipe the anchored-op experiments will use. We call that the *grammar of record*: the one setup every later D2.2 experiment is built on. It is a handover because the grammar of record changes hands here: until this runs it is the one from ex-2.2.3; after it, if the gates hold, it is this one.
"""

seen_lines, seen_pairs = coverage(ex.HANDOVER.lines_per_op)
seen_lines_six, seen_pairs_six = coverage(ex.NARROW.lines_per_op)
rf"""
## Conditions

Every condition trains fresh models on the new grammar: table A+, {ex.N_LINES:,} lines drawn uniformly over the {ex.N_OPS} ops, answers drawn stochastically, one fifth of the pairs of each op held out. That is about {ex.HANDOVER.lines_per_op:,} lines per op, drawn with replacement from the op's {len(lines()) * (1 - ex.HOLDOUT_FRAC):,.0f} trainable lines, so a model sees about {seen_lines:.0%} of them at least once ({seen_pairs:.0%} of the unordered pairs), against {seen_lines_six:.0%} ({seen_pairs_six:.0%}) at six ops. The recipe is the point adopted in ex-2.2.3 (`{ex.EX223_REFERENCE}`), unchanged: λ_a = {ex.LAM:g}, annealed over the last tenth of training as ex-2.1.10 did, τ = {ex.TAU:g}, the anti-subspace weight from {ex.ANTI_PEAK_RATIO:g}× the anchor weight down to 0.3× by {ex.ANTI_ANNEAL_END_FRAC:.0%} of training, {ex.EPOCHS} epochs ({ex.HANDOVER.steps:,} steps). *Red* is anchored to e₁ at every slice, the embedding included.

{conds_md()}

**`handover`** has everything enabled. It is the one candidate for the grammar of record, at the same twenty seeds as the reference, and the gates are read on it alone.

**`handover-slot`** switches the label back to the one from ex-2.2.3: only the two operands can draw a label, and the pull covers the prompt. Beside `handover` it is the selectivity check ex-2.2.7 asked for, at twenty seeds so that the comparison has the resolution of the gates. It is not a fallback: that labeller needs to know which tokens are the operands, and M3 will not have that.

**`handover-tied`** switches the readout back to the shared table. Beside `handover` it shows what untying does on this grammar. Nine seeds, as the pilot had.

**`control`** has no anchor at all. It sets the task bar for H1 and the calibration bar for the drawn answers.

**`handover-narrow`** is exploratory. E4 in ex-2.2.3 found the operand cube less decodable at six ops than at three, and could not tell whether the op count or the lines per op was responsible. The main corpus gives each of the eleven ops about {ex.HANDOVER.lines_per_op:,} lines, more than the {ex.NARROW.lines_per_op:,} each op had at six. This condition holds lines per op at that six-op count, using a smaller corpus for more epochs so that the step count matches.[^wide]

[^wide]: The prereg draft called it `handover-wide`: at the draft's 100k lines, holding lines per op at the six-op count meant a larger corpus. The calibration look below made the main corpus the larger one.

The reference is not retrained. It is `{ex.EX223_REFERENCE}` from ex-2.2.3, at twenty seeds on the six-op grammar, and its stored statistics are printed beside every placement read.

### The removal operators

We score every anchored checkpoint through the eval contract in [`sca.intervention`](/src/sca/intervention.py), on each op's probe lines.

- **`projection`**: the axis projected out at every slice and every position, at full strength. This is the gated operator, as ex-2.2.8 proposed.
- **`operands`**: the same edit at the two operand positions only. It is the selective reference, since it never touches the syntax tokens and so cannot cost anything there.
- **`shaped-a0.4-p0`**: a thresholded projection that leaves any state below alignment {ex.SHAPED["a"]:g} alone. Ex-2.2.8 found it removes less than the projection at no cost on the six-op grammar; what we learn here is whether that smaller removal clears the removal gate on a table built to need *red*. Scoring it is one more pass over the same checkpoints.
"""

# A definition list, written as HTML: our Markdown has no syntax for one, and the
# markup inside the block is HTML too, since Markdown is not processed there.
low = min(far / red for red, far, _ in removal_counts().values())
rf"""
## Glossary

<dl>
<dt data-forms="Red line; non-red line">Red line, non-red line</dt>
<dd>A line is <em>red</em> when its redder operand has redness at least 0.8, and <em>non-red</em> when neither operand is above 0.2. Redness is r·(1 − g/2 − b/2) on the unit scale. The redness of the redder operand is the line's <em>dose</em>.</dd>
<dt>Removal lines</dt>
<dd>The red lines on which the true answer would move a long way if the red operand had no red in it (its R channel set to zero). These are the lines where losing <em>red</em> has to show; on the others the op does not need it. On <code>mix</code> every red line is a removal line; on the other ops at least {low:.0%} of them are, and the count per op is in the method.</dd>
<dt>Red-answer lines</dt>
<dd>Non-red lines whose true answer is red (white minus cyan, under <code>difference</code>). Projecting the axis out at <code>=</code> takes <em>red</em> from the state that has to produce the answer, so a miss there is removal on the output side. They stay in the non-red deficit and are also counted on their own, and the deficit is reported with and without them. <code>mix</code> has none, so the gated read is the same either way.</dd>
<dt>Expected exact match</dt>
<dd>The task score we measure. With drawn answers, the correct answer to a line is spread over two or more colors, so a single exact-match check would penalise a model for a right answer. Instead we take the chance that an answer drawn from the model agrees with one drawn from the true distribution. Its ceiling is that same chance for the true distribution against itself (Σq²), which is below 1 on the ops that round.</dd>
<dt>m_line</dt>
<dd>How far <em>red</em> is pushed onto the axis: at the reddest position in the span, the label-weighted mean alignment minus the unweighted mean, averaged over slices. Ex-2.1.10's margin, read on the <code>mix</code> lines.</dd>
<dt data-forms="ᾱ; containment">ᾱ (containment)</dt>
<dd>The mean alignment with the axis over all 216 colors at op1. High when colors that are not red drift onto the axis at that position.</dd>
<dt>Lead</dt>
<dd>On the red lines, the largest share of the pull that any one position receives at the embedding slice. High when the pull is concentrated on one token rather than spread over the span.</dd>
<dt>Contrast</dt>
<dd>How much more the pull lands on op2 when op2 is the red operand than when it is not, averaged over the post-attention slices. It says the pull follows the red operand rather than a fixed position.</dd>
<dt>Grading r²</dt>
<dd>How well alignment with the axis tracks redness across colors, fit against the sim^1.5 target of ex-2.1.11.</dd>
<dt>Retention</dt>
<dd>Whether the placement holds to the end of training, after the anchor weight's anneal: the final m_line as a share of the run's peak.</dd>
<dt>Latch</dt>
<dd>A run in which the non-red group puts more than half of its softmin weight on one position: the pull has found a position rather than a concept. The position read is op1, the slot the earlier experiments saw it latch to.</dd>
<dt>Deficit</dt>
<dd>How much expected exact match a model loses on the non-red lines when the axis is projected out. This is a measure of the selectivity; a clean removal would cost nothing.</dd>
<dt>Band</dt>
<dd>Our measurement precision. Two seed means are told apart only when they differ by more than 2σ√(1/n_a + 1/n_b), with σ the per-run spread and n the seed counts. Smaller differences are reported as unresolved.</dd>
</dl>
"""

r"""
## Does the model still learn the task? (H1)
"""

rf"""
**Background.** Eleven ops in the same number of lines, with drawn answers, is a harder corpus than six ops with rounded ones. Before we read anything about the anchor we need to know that the anchored model learns the task as well as an un-anchored model does on this corpus.

**Prediction.** For each of the {ex.N_OPS} ops, the `handover` seed-mean expected exact match on held-out lines is within {ex.TASK_GATE:g} of the control. Partial: every op within {ex.TASK_PARTIAL:g}. Contrary: more than {ex.TASK_PARTIAL:g} below the control on some op. A comparison that misses by less than a band is reported as unresolved rather than as a miss; the band on this read is computed from the runs and printed beside the gate. The reference conditions are read on the same table without a gate, and a reference that does not learn the task has its later comparisons read with that caveat.

We expect this to hold: ex-2.2.5 saw no task cost from the drawn answers, ex-2.2.7 none from the untied readout. Whether the control itself learns the grammar is checked before the freeze, on one seed, rather than gated here. The number to watch is the order-sensitive subset, since an op that reads operand order asks the model for something the six-op grammar never did.
"""

if res is None:
    stop(RESULTS_TO_COME)
status, line = h1_verdict(res)
rf"""
**Results.** {h1_prose(res)}

{h1_figure(res)}

{h1_table(res)}

{h1_calibration_table(res)}

{verdict_md(status, line)}
"""

r"""
## Does *red* still land where we put it? (H2)
"""

rf"""
**Background.** The recipe was tuned on the six-op grammar, but here the corpus, the labeller, and the readout all change. We read the same placement statistics on the same `mix` lines, so the numbers are comparable to ex-2.2.3. The question is whether they are still inside the gates set there.

**Prediction.** On the `mix` lines, for `handover`, under its own labeller (as ex-2.2.6 read it):

- *Margin:* seed-mean m_line at least {ex.MARGIN_RATIO:.0%} of the reference's {ex.REF_M_LINE:.4f}, so {ex.MARGIN_RATIO * ex.REF_M_LINE:.3f}; partial from {ex.MARGIN_PARTIAL:.0%}.
- *Grading:* grading r² at least {ex.GRADE_R2_RATIO:.0%} of the reference's {ex.REF_R2_SIM:.3f}, so {ex.GRADE_R2_RATIO * ex.REF_R2_SIM:.3f}.
- *Concentration, attribution, retention, latch:* lead at the embedding at least {ex.LEAD_GATE:g}; contrast at least {ex.CONTRAST_GATE:g} (partial from {ex.CONTRAST_PARTIAL:g}); every run that reaches m_line {ex.RETENTION_FLOOR:g} ends at {ex.RETENTION_GATE:g} of its peak; no run latched, which the reference held at twenty seeds.

**Containment is a prediction here, not a gate.** Every experiment since ex-2.1.8 has gated ᾱ at op1 at {ex.MEAN_ALIGN_REF:g}. At nine seeds, ex-2.2.7 read 0.13 on its untied condition and 0.23 on its untied whole-line condition, against 0.08 on the reference, so we expect `handover` above {ex.MEAN_ALIGN_REF:g}, and `handover-slot` nearer to it. We do not know why untying raises it. What we do know is that the pull still lands on the red operand in those runs (lead, contrast, and latch all sat at the reference's values), so this is other colors drifting a little onto the axis at op1 rather than the pull finding a position. What that drift costs, if anything, is what H3's selectivity read measures. So the prediction is the pair: ᾱ above {ex.MEAN_ALIGN_REF:g} on `handover`, and a non-red deficit inside H3's gate all the same. If both hold, the drift is recorded for the anchored-op prereg to watch; if the deficit fails too, containment is the first place to look for why.

`handover-tied` is read on the same statistics, so that we can attribute the containment read. If the tied condition sits with the reference (more contained) and both untied conditions sit higher, the untied readout is what moved it.
"""

if res is None:
    stop(RESULTS_TO_COME)
status, line = h2_verdict(res)
rf"""
**Results.** {h2_prose(res)}

{h2_figure(res)}

{h2_table(res)}

{h2_profile_figure(res)}

{verdict_md(status, line)}
"""

r"""
## Can we still take *red* out cleanly? (H3)
"""

rf"""
**Background.** Take the *red* axis (e₁) out of every state. On the lines that need *red*, does the model fail? On the lines that never had any, does it still answer? The first is removal, the second selectivity. Ex-2.2.3 could only show removal on two of six ops, because the other four mostly did not need *red*. Table A+ was chosen so that every op has lines that do, and the removal read is scored on those lines only.

**Prediction.** Under `projection`, for `handover`:

- *Removal:* on the removal lines of every op, the model keeps at most {ex.RED_KEPT_GATE:.0%} of its clean expected exact match, seed mean. This is the red-accuracy gate of ex-2.2.3, read as a ratio, because with drawn answers the clean value sits below 1 on the ops that round.
- *Selectivity:* the seed-mean deficit on the non-red `mix` lines is at most {ex.NONRED_DEFICIT_GATE:g}; partial to {ex.NONRED_DEFICIT_PARTIAL:g}. The partial band is a reporting level, as it was in ex-2.2.3: the decision rule asks for the gate. The read stays on `mix` so that it means what it meant at the reference; `hsvmix` and every other op are reported beside it; what a switch to `hsvmix` would change is set out in the method ([The reference op](#the-reference-op)). On the reference, ex-2.2.8 read 0.040 on `mix`, which is inside the gate by less than a band. With the readout untied we expect the deficit to fall toward the `operands` row, which read 0.012. Bands on the deficit use the per-run spread ex-2.2.8 measured under `projection` at the reference's twenty seeds, per op ({ex.DEFICIT_NOISE}), frozen so that the candidate's own spread does not move its verdict. Non-red lines whose true answer is red are in the deficit and also counted on their own (see the glossary); `mix` has none.

Two further reads have an expected direction and no gate.

*How far the answer moves.* On the removal lines, we measure the distance in the unit cube between the answer the model decodes under `projection` and the true answer. Beside it we put the distance the true answer itself moves when the red operand loses its red. Ranking the ops by each distance should give the same order, with `value-hsv`, `hue-hsv`, and `sat-hsv` at the top of both.

*The operand-only edit.* On the ops whose answer is computed at `=` from both operands together, `operands` should remove less than `projection`, since the `=` state keeps its axis component. Elsewhere the two should remove the same amount.
"""

if res is None:
    stop(RESULTS_TO_COME)
status, line = h3_verdict(res)
rf"""
**Results.** {h3_prose(res)}

{h3_figure(res)}

{h3_table(res)}

{h3_distance_table(res)}

{verdict_md(status, line)}
"""

r"""
## Does the separate readout keep the axis off the syntax tokens? (H4)
"""

rf"""
**Background.** In ex-2.2.7 the readout row for `=` picked up a component along the *red* axis: after a red operand, whose state sits on the axis, that is a cheap way to raise the `=` logit. With a shared table that row is also the input embedding of `=`, so the `=` token entered the residual stream carrying some *red*, and projecting the axis out at that position took away something the model was using. Giving the output its own table moved the component onto the readout row and left the input embedding mostly clean.

That was on the six-op grammar at nine seeds. Does it carry to eleven ops, and does it give the cleaner full-line removal?

**Prediction.** On `handover` against `handover-tied`, same labeller, twenty seeds against nine (the band formula takes both counts, so it is wider than between two twenty-seed conditions):

- The axis component on the syntax embeddings (`=`, the op words, and `⏎`), read from the embedding-component table of ex-2.2.7, is lower on `handover` than on `handover-tied` by more than a band ({ex.COMPONENT_NOISE}, since ex-2.2.7 published seed means and ranges rather than a per-run spread; the σ is reported beside the comparison). On `=`, `handover` sits within a band of the hard-zeroed ceiling from ex-2.2.7 (`{ex.EX227_CEILING}`, where the component is zero by construction). The component appears on the readout table of `handover` instead.
- The non-red `mix` deficit under `projection` is lower on `handover` than on `handover-tied`, by more than a band.

Neither prediction is a gate. Ex-2.2.7 already took the readout decision, and this section either confirms it or reports that it did not carry. If the second prediction fails while the first holds, then on this grammar the syntax embeddings were not where the cost came from.
"""

if res is None:
    stop(RESULTS_TO_COME)
status, line = h4_verdict(res)
rf"""
**Results.** {h4_prose(res)}

{h4_figure(res)}

{h4_table(res)}

{verdict_md(status, line)}
"""

r"""
## What does the whole-line label cost? (H5)
"""

rf"""
**Background.** A label that covers the whole line is the shape we need for natural language (M3). Ex-2.2.6 found it costs nothing at three seeds. Ex-2.2.7 then ran nine seeds of its untied whole-line condition and saw a few of them lose a lot of non-red lines under the projection, where the operand-only labeller loses almost none. Here it is read at twenty seeds against twenty.

**Prediction.** On `handover` against `handover-slot`, under `projection`:

- The seed-mean non-red `mix` deficit differs by less than a band, and
- the count of seeds whose deficit is above {ex.TAIL:g} (the level at which ex-2.2.7 read its tail) is no more than two higher on `handover` than on `handover-slot`.

The whole-line label changes two things at once: which positions the pull can land on, and which lines get a label at all, since the answer draws at its own redness rate. So a miss here says the label as a whole costs selectivity, but not which half of it did. Neither prediction is a gate, and `handover-slot` is not a fallback: we need the whole-line label, so a cost here is something to understand and fix, and the size of the gap to `handover-slot` says how much there is to fix. We are not sure which way this goes: the tail in ex-2.2.7 was three seeds out of nine, enough to expect it and too few to be sure.
"""

if res is None:
    stop(RESULTS_TO_COME)
status, line = h5_verdict(res)
rf"""
**Results.** {h5_prose(res)}

{h5_figure(res)}

{h5_table(res)}

{verdict_md(status, line)}
"""

r"""
## Decision
"""

rf"""
H1 to H3 carry gates because one decision hangs on them: whether the anchored-op experiments run on this grammar. H4 and H5 are predictions, written down so that the result can be read against them, and what we do about a miss there is decided after reading it.

{ex.DECISION}
"""

if res is None:
    stop(RESULTS_TO_COME)
adopted, line = decision(res)
rf"""
{verdict_md("pass" if adopted else "miss", line)}
"""

r"""
## Exploratory analyses

Read without gates, and not part of the decision. Each analysis below states what it was for, then what it found.

### Lines per op

Ex-2.2.3's E4 probed how well the two operand colors can be decoded from the residual stream, and found them less decodable at six ops than at three. Was that because there were more ops, or because each op had fewer lines? `handover-narrow` gives each op as many lines as it had at six ops, where `handover` gives each about half as many again. If the operands decode better on `handover` than on `handover-narrow`, lines per op matters. If the two conditions sit together, it was the op count.
"""

if res is None:
    stop(RESULTS_TO_COME)
rf"""
{lines_per_op_prose(res)}

{lines_per_op_table(res)}
"""

r"""
### The order-sensitive subset, by slot

For `hue-hsv`, `sat-hsv`, and `value-hsv` the probe set walks every color in both slots, so every read in H2 and H3 can be split by whether the red operand is op1 or op2. The labeller pools both operands the same way, so we expected the same placement in both slots. Removal should show in both slots too, at the rates in the method's per-slot table: a red op1 under `value-hsv` supplies the hue and the saturation, and losing those moves the answer as far as losing the value does.
"""

if res is None:
    stop(RESULTS_TO_COME)
rf"""
{slot_prose(res)}

{slot_table(res)}
"""

r"""
### Calibration under the anchor

P(mode) and KL from the true distribution on the rounded held-out lines, for every anchored condition against the control, as ex-2.2.5 read them. The anchor should not change them.
"""

if res is None:
    stop(RESULTS_TO_COME)
rf"""
{calibration_prose(res)}

{anchor_calibration_table(res)}
"""

r"""
### Red-answer lines

On the ops that have them, the model's answer under `projection` on the non-red lines whose true answer is red. If the model can no longer produce red there, that is removal on the output side, and it says the readout row for red carries the axis as the embedding does.
"""

if res is None:
    stop(RESULTS_TO_COME)
rf"""
{red_answer_prose(res)}

{red_answer_table(res)}
"""

r"""
### The shaped operator

`shaped-a0.4-p0` is reported on every op beside the two gated operators: does its smaller removal still clear the removal gate on the new table?
"""

if res is None:
    stop(RESULTS_TO_COME)
rf"""
{shaped_prose(res)}
"""

r"""
## Discussion
"""

rf"""
{discussion_md(res)}
"""

hsv_on_grid, hsv_partners, hsv_ceiling = refop_facts()
sat_op2, eps_sat = eps_check()
rf"""
## Method

### The table

{table_md()}

Every rule is computed on the 0..15 scale and snapped to the grid. Where it lands between levels, the corpus draws the answer (*stochastic rounding*, ex-2.2.5). The three order-sensitive ops take one HSV attribute from op2 and the other two from op1, so each agrees with its own reverse on under 2% of pairs. Their reads are reported as a subset.

### Op-relevance under A+

For a line that uses the anchored op, how many other ops in the table give the same answer. This is the stimulus side of the per-line predictions in the anchored-op experiments, counted over ordered pairs. Widening the table makes `mix` less distinctive (ex-2.2.4 read 95% alone at six ops), which is good because the later predictions need more than one level to read.

{relevance_md()}

### The removal lines

Per op, on its probe set: the red lines (dose ≥ {ex.RED_DOSE:g}, where dose is the redness of the redder operand) and, of those, the lines on which setting the R channel of the red operand to zero moves the true answer by at least {ex.FAR_MOVE:g} in the unit cube. The removal read in H3 is scored on the second column. The last column counts the non-red lines (dose ≤ {ex.NONRED_DOSE:g}) whose true answer is red; those are in the selectivity read and also counted on their own.

{removal_md()}

Replacing the red operand with a mid-gray instead of zeroing its R channel was considered. On `mix` it counts far fewer lines (a gray partner moves the mix less than a dark one does), and elsewhere it changes the counts without changing which ops need *red*, so the to-zero rule stays.

For the order-sensitive ops, the both-slot probe set splits by where the red operand sits. The one place the rule under-counts is a red op2 under `hue-hsv` or `sat-hsv`. Zeroing the R of a pure red gives black, which HSV reads as hue 0 (red) at no saturation: under `hue-hsv` the answer keeps its hue, and under `sat-hsv` op1 only loses its saturation, which moves it far only when it was saturated. Setting R to one grid level instead of zero does not help. The operand is then a very dark red at full saturation, so `hue-hsv` moves as little as before and `sat-hsv` stops moving at all ({eps_sat} lines counted against {sat_op2} at zero). There is no hue a color should have once its red is gone, so the lines the rule drops are reported (in the red-at-op2 count) and not gated. The red-at-op2 lines that do move far stay in the gate: the removal read pools both slots, and the exploratory section splits it.

{slot_md()}

### The reference op

Every gated read is on `mix`, and `hsvmix` is reported beside it as the op a later experiment might promote. This is what the switch would change, on each op's probe set as ex-2.2.3 draws it.

{refop_md()}

`hsvmix` is the better op for the removal read: nearly every red line is a removal line, and the answer moves further when the red operand loses its red, since a hue mean loses the red hue outright where a channel mean halves it. It is also a little less distinctive, which the anchored-op predictions want. Its placement reads (H2) would look much like `mix`'s, since the label reads the colors in the line rather than the op word.

What it costs is the probe set. `mix` has {ex.N_PROBE} partners per color on which its answer lands on the grid without rounding, its probe lines are those, and so a clean model can match every answer and a deficit is a count of lines lost. `hsvmix` lands on the grid on {hsv_on_grid:.0%} of pairs, about {hsv_partners:.0f} partners per color, so its probe set is the shared random draw, and the best any model can do on it is an expected exact match of about {hsv_ceiling:.2f}. On that base the removal gate reads on a clean value near {hsv_ceiling:.1f} rather than 1, and the {ex.NONRED_DEFICIT_GATE:g} selectivity gate is an eighth of the clean value rather than a twentieth; both would need re-setting, and neither could be read against the reference's numbers. Had the program used `hsvmix` from the start, every exact-match read before ex-2.2.5 would have been on answers that round on {1 - hsv_on_grid:.0%} of lines, which is the bias that experiment found and fixed; D2.1 and ex-2.2.3 could read removal and selectivity as counts because `mix`'s probe set never rounds. The switch is open now that answers are drawn, at the price of a noisier deficit, and it is a later experiment's call, made with this experiment's `hsvmix` rows in hand.

### The corpus, the probes, the labellers

{ex.N_LINES:,} lines at seed {ex.CORPUS_SEED}, ops drawn uniformly, with {ex.HOLDOUT_FRAC:.0%} of the distinct unordered pairs of each op held out. A held-out pair is out in both orders, for every op, so the held-out share of an op's lines is the same {ex.HOLDOUT_FRAC:.0%} whether or not the op reads operand order. Probe sets follow ex-2.2.3: `mix` on its 27 on-grid partners per color, and every other op on 27 partners per color drawn once at seed {ex.PROBE_SEED} and shared across ops. The order-sensitive subset also walks every color as op2.

The two labellers are *either slot, prompt span* (each operand draws at redness⁸ × {ex.PER_SLOT_RATE:g}, and the pull covers op1, op, op2, `=`) and *whole line* (the answer draws at its redness rate too, and the pull covers all six positions). The anchor term is the per-line mellowmax over the pulled span with a conserved per-line budget, so a wider span changes where the pull can land but not how strong it is.

### Before the freeze

One seed of `control` trains first, and its expected exact match per op is recorded here, so that the gate in H1 is read against a control that learned the grammar. The bar: on every kept and added op, expected exact match within {ex.CALIBRATION_FLOOR:g} of the ceiling the drawn answers allow on that op (ex-2.2.5 saw the six-op control within 0.04 on the ops that round). A lower value on the order-sensitive subset is recorded and does not stop the run, since it says something about the grammar rather than about the anchor; a miss on a commutative op does stop it, and sends the corpus size and epoch count back for a look. Two pieces of code land with the DAG: the holdout draw, now keyed on the position of the op in this table rather than in the table of ex-2.2.3 (`sca.data.ops.holdout`), and a probe draw that walks both slots for the order-sensitive subset (`probe_partners`). Neither changes a number in the design.

**What the look found.** The first control seed, at the draft's point (100k lines, 50 epochs, 1,650 steps), missed the bar on six of the eight commutative ops, with `difference` 0.29 from its ceiling. Seen and held-out lines scored alike on every op, so this was under-training rather than a coverage problem. More epochs over the same lines fixed all but two: at 100 and 150 epochs `difference` stayed 0.07 to 0.09 short and `hsvmix` 0.07 to 0.09 short, and the `hsvmix` NLL rose with every extra pass, which says the model was memorising the drawn labels rather than learning the distribution behind them. Deeper models (six and eight layers, at 100 epochs) moved neither op. A larger corpus did: at 300k lines and 50 epochs (4,950 steps) every kept and added op is within the bar, with `hsvmix` 0.04 short and `difference` 0.03; more epochs over the same 300k lines pushed `hsvmix` back out. So the frozen point is 300k lines at 50 epochs: three times the reference's steps, from three times the lines rather than more passes. Every point looked at is in the table below, and the frozen one is in bold.

{calibration_md()}

{calibration_verdict()}

### Budget

{ex.N_RUNS} runs of {ex.HANDOVER.steps:,} steps at d64-L4, plus scoring under three operators on {ex.N_OPS} probe sets. That is three times the steps per run of ex-2.2.3's short runs, at about the same run count; on an L4 a run takes about three minutes.
"""
