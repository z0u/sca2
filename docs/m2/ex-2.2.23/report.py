# ruff: noqa: B018
# title: Ex 2.2.23: The slow seeds, trained for longer

# The design constants come from `experiment.py` beside this script (the directory of the script is on sys.path
# while it runs). The results come from the JSON this experiment publishes under its refs, and the earlier runs from
# ex-2.2.21's published trajectories.
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    """An authored table in the shared report style; the first *text_cols* columns are text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def seed_span(seeds: tuple[int, ...]) -> str:
    return f"{seeds[0]}–{seeds[-1]}"


# --- The earlier runs ---------------------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as _tmp:
    _store = project_store()
    _art = _store.get_refs([ex.ex2221.TRAJ_REF])[ex.ex2221.TRAJ_REF]
    assert _art is not None, "ex-2.2.21 has not published its trajectories"
    (_path,) = _store.get_many([(_art, Path(_tmp) / "traj21.json")])
    TRAJ21 = json.loads(_path.read_text())

HSV_IDX = [ex.OP_NAMES.index(o) for o in ex.HSV_OPS]


def hsv_skill(traj: dict) -> tuple[np.ndarray, np.ndarray]:
    """The epoch of each trajectory record, and the EEM averaged over the HSV ops there."""
    pts = [(e, np.mean([v[i] for i in HSV_IDX])) for e, v in zip(traj["epoch"], traj["eem_per_op"], strict=True) if v]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


EARLIER = {
    c.name: {s: hsv_skill(TRAJ21[f"{c.reused_as}-s{s - ex.SEED_OFFSET}"]["traj"]) for s in ex.REUSED_SEEDS}
    for c in ex.CONDITIONS
}
EARLIER_MISSED = {c: [s for s, (_, y) in runs.items() if y[-1] < ex.RISE_LEVEL] for c, runs in EARLIER.items()}


def earlier_figure() -> str:
    caption = f"""
        **The HSV skill through training, in the ex-2.2.21 runs this scout reuses.** EEM averaged over the three HSV
        ops, on a subsample of the held-out set, one line per model seed. The dashed line is the level a run has to
        pass to count as having made the second rise ({ex.RISE_LEVEL:g}).
    """
    missed = ", ".join(f"{s}" for s in EARLIER_MISSED["anchor"])
    alt = f"""
        Two panels, the control and the anchored condition, each plotting HSV skill against the epoch for five runs.
        Every run sits near 0.15 to 0.2 for at least the first quarter of training. On the control every run then
        rises past the dashed line, two of them late in training. On the anchored condition three runs rise and two, at model
        seeds {missed}, stay below the line to the end.
    """
    return earlier_draw(EARLIER, caption, alt)


@memo
def earlier_draw(data: dict, caption: str, alt_text: str) -> str:
    @themed(name="ex-2.2.23-earlier", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), layout="constrained", sharey=True)
        ink = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(ex.REUSED_SEEDS)))
        for ax, c in zip(axes, ex.CONDITIONS, strict=True):
            for color, (s, (x, y)) in zip(ink, data[c.name].items(), strict=True):
                ax.plot(x, y, color=color, lw=1.0, label=str(s))
            ax.axhline(ex.RISE_LEVEL, color=light_dark("#000", "#fff"), lw=0.8, ls="--", alpha=0.6)
            ax.set_title(c.name, fontsize=9)
            ax.set_xlim(0, ex.SHORT)
            ax.set_xlabel("epoch", fontsize=8)
        axes[0].set_ylabel("HSV skill (EEM)", fontsize=8)
        axes[1].legend(title="model seed", fontsize=7, title_fontsize=7, frameon=False, loc="upper left")
        return fig

    return _plot()


def runs_table() -> str:
    rows = []
    for c in ex.CONDITIONS:
        rows.append([f"`{c.name}`", f"{ex.SHORT}", f"{seed_span(ex.REUSED_SEEDS)}", f"ex-2.2.21 (`{c.reused_as}`)"])
        rows.append([f"`{c.name}`", f"{ex.SHORT}", f"{seed_span(ex.NEW_SEEDS)}", "new"])
        rows.append([f"`{c.name}`", f"{ex.LONG}", f"{seed_span(ex.SEEDS)}", "new"])
    return table_html(
        ["condition", "epochs", "model seeds", "runs"],
        rows,
        f"**The runs.** {ex.N_NEW_RUNS} new runs, and {len(ex.CONDITIONS) * len(ex.REUSED_SEEDS)} reused from "
        "ex-2.2.21, which trained both conditions at the same settings.",
        text_cols=4,
    )


# --- The results: fetching ----------------------------------------------------------------------------------


def fetch_json(refs: list[str]) -> dict[str, Any]:
    """Each ref's published JSON, by ref."""
    store = project_store()
    found = store.get_refs(refs)
    arts = {r: a for r, a in found.items() if a is not None}
    assert len(arts) == len(refs), f"not published yet: {sorted(set(refs) - set(arts))}"
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(arts[r], Path(tmp) / f"{i}.json") for i, r in enumerate(refs)])
        return {r: json.loads(p.read_text()) for r, p in zip(refs, paths, strict=True)}


_results = fetch_json([ex.EVAL_REF, ex.SUPPRESSION_REF, ex.TRAJ_REF])
EVAL, SUPP, TRAJ_NEW = (_results[r] for r in (ex.EVAL_REF, ex.SUPPRESSION_REF, ex.TRAJ_REF))

# --- The results: one record per run --------------------------------------------------------------------------

OPS: tuple[str, ...] = tuple(EVAL["runs"][0]["ops"])
assert OPS == tuple(ex.OP_NAMES)
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
CONTROL, ANCHOR = (c.name for c in ex.CONDITIONS)
# A run: (condition, epochs, model seed).
Key = tuple[str, int, int]


def logistic(t: np.ndarray, lo: float, hi: float, mid: float, width: float) -> np.ndarray:
    return lo + (hi - lo) / (1 + np.exp(-(t - mid) / width))


# A fitted rise shorter than this, from its floor to its ceiling, is no rise at all: the fit to a flat trajectory puts
# its midpoint anywhere.
MIN_RISE_HEIGHT = 0.1

# The logistic is fitted from this epoch on, after the first stage: on a run that never rises, a fit from the start
# would find the first stage instead.
FIT_FROM_EPOCH = 20


def logistic_midpoint(t: np.ndarray, y: np.ndarray, epochs: int) -> float | None:
    """The midpoint of a logistic curve fitted to the HSV skill through training: when the rise is half done, with no
    threshold. Bounded to twice the length of training, so a run still rising at the end gets a midpoint past it; a
    run whose fitted rise is shorter than `MIN_RISE_HEIGHT`, or whose midpoint sits on the start of the fit (a slow
    climb with no bend in it), gets None.
    """
    from scipy.optimize import curve_fit

    t, y = t[t >= FIT_FROM_EPOCH], y[t >= FIT_FROM_EPOCH]
    p0 = (float(y[: len(y) // 5].mean()), float(max(y.max(), 0.3)), float(t[np.argmax(np.gradient(y))]), epochs / 20)
    bounds = ((0.0, 0.0, FIT_FROM_EPOCH, epochs / 400), (0.4, 0.7, 2.0 * epochs, epochs / 2))
    p0 = tuple(float(np.clip(v, lo + 1e-6, hi - 1e-6)) for v, lo, hi in zip(p0, *bounds, strict=True))
    (lo, hi, mid, width), _ = curve_fit(logistic, t, y, p0=p0, bounds=bounds, maxfev=20_000)
    return float(mid) if hi - lo >= MIN_RISE_HEIGHT and mid > FIT_FROM_EPOCH + 1 else None


def traj_of(key: Key) -> dict:
    cond, epochs, seed = key
    if epochs == ex.SHORT and seed in ex.REUSED_SEEDS:
        c = next(c for c in ex.CONDITIONS if c.name == cond)
        return TRAJ21[f"{c.reused_as}-s{seed - ex.SEED_OFFSET}"]["traj"]
    return TRAJ_NEW[ex.label_of(cond, epochs, seed)]["traj"]


def build_runs() -> dict[Key, dict[str, Any]]:
    evals = {r["label"]: r for r in EVAL["runs"]}
    supps = {r["label"]: r for r in SUPP["runs"]}
    out: dict[Key, dict[str, Any]] = {}
    for c in ex.CONDITIONS:
        for epochs in ex.LENGTHS:
            for seed in ex.SEEDS:
                key = (c.name, epochs, seed)
                label = ex.label_of(*key)
                t = traj_of(key)
                pts = [(e, v) for e, v in zip(t["epoch"], t["eem_per_op"], strict=True) if v]
                epoch = np.array([p[0] for p in pts], float)
                per_op = np.array([p[1] for p in pts], float)
                hsv_t, hsv_min_t = per_op[:, HSV_IDX].mean(axis=1), per_op[:, HSV_IDX].min(axis=1)
                above = np.flatnonzero(hsv_t >= ex.RISE_LEVEL)
                final_per_op = np.array(evals[label]["task"]["eem"]["per_op"], float)
                out[key] = {
                    "label": label,
                    "epoch": epoch,
                    "hsv_t": hsv_t,
                    "hsv_min_t": hsv_min_t,
                    "rise": float(epoch[above[0]]) if len(above) else None,
                    "midpoint": logistic_midpoint(epoch, hsv_t, epochs),
                    "per_op": final_per_op,
                    "hsv": float(final_per_op[HSV_IDX].mean()),
                    "hsv_min": float(final_per_op[HSV_IDX].min()),
                    "eem": float(evals[label]["task"]["eem"]["all"]),
                    "margin": np.array(evals[label]["margin"]["by_slice"], float),
                    "supp": supps[label],
                }
    return out


RUNS = build_runs()


def edit_measurements(key: Key) -> dict[str, Any]:
    """The two edit measurements of one anchored run, net of the control at the same seed and length: the share of
    the way to the target null the drop on the anchored op gets at full dose, and the largest drop on any other op
    at any dose; with ex-2.2.21's criteria on them, for marking only.
    """
    s, c = RUNS[key]["supp"], RUNS[(CONTROL, *key[1:])]["supp"]

    def drops(r: dict) -> np.ndarray:
        return np.array(r["clean"]["eem"])[None, :] - np.array([e["eem"] for e in r["edits"]])  # (doses, ops)

    net = drops(s) - drops(c)
    gap = s["clean"]["eem"][D] - s["null"]["eem"][D]
    anchored = net[:, D]
    worst = net[:, OTHER].max(axis=1)
    rises = all(b >= a - ex.ex2221.GRADE_DIP for a, b in zip(anchored, anchored[1:], strict=False))
    share = float(anchored[-1] / gap)
    return {
        "share": share,
        "worst": float(worst.max()),
        "by_dose": net,
        "grades": bool(rises and share >= ex.GRADING_MIN_DAMAGE),
        "selective": bool(worst.max() <= ex.SELECTIVITY_GATE),
    }


def keys(cond: str | None = None, epochs: int | None = None) -> list[Key]:
    return [k for k in RUNS if (cond is None or k[0] == cond) and (epochs is None or k[1] == epochs)]


def missed(key: Key, level: float = ex.RISE_LEVEL, measure: str = "hsv") -> bool:
    """Whether a run ends below *level* of HSV skill on the whole held-out set: the average (`hsv`) or the worst
    op (`hsv_min`).
    """
    return RUNS[key][measure] < level


MISSED = {(c, e): [k[2] for k in keys(c, e) if missed(k)] for c in (CONTROL, ANCHOR) for e in ex.LENGTHS}

# --- The results: shared drawing ------------------------------------------------------------------------------


def ink_of(cond: str) -> str:
    return {CONTROL: light_dark("#555", "#bbb"), ANCHOR: light_dark("#c0392b", "#ff8a76")}[cond]


def marker_of(cond: str) -> str:
    return {CONTROL: "s", ANCHOR: "o"}[cond]


def rule(ax, y: float) -> None:
    ax.axhline(y, color=light_dark("#000", "#fff"), lw=0.8, ls="--", alpha=0.6)


def num_word(n: int) -> str:
    return ("none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")[
        n
    ]


def seeds_list(seeds: list[int] | list[str]) -> str:
    if not seeds:
        return "none"
    if len(seeds) == 1:
        return str(seeds[0])
    return ", ".join(map(str, seeds[:-1])) + f" and {seeds[-1]}"


def traj_panels(epochs: int, highlight: dict[str, list[int]], name: str, caption: str, alt: str) -> str:
    data = {
        c: {k[2]: (RUNS[k]["epoch"].tolist(), RUNS[k]["hsv_t"].tolist()) for k in keys(c, epochs)}
        for c in (CONTROL, ANCHOR)
    }
    return traj_draw(data, epochs, highlight, name, caption, alt)


@memo
def traj_draw(data: dict, epochs: int, highlight: dict, name: str, caption: str, alt_text: str) -> str:
    @themed(name=name, alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), layout="constrained", sharey=True)
        for ax, cond in zip(axes, (CONTROL, ANCHOR), strict=True):
            other = ANCHOR if cond == CONTROL else CONTROL
            for x, y in data[other].values():
                ax.plot(x, y, color=ink_of(other), lw=0.5, alpha=0.18, zorder=1)
            for seed, (x, y) in data[cond].items():
                hi = seed in highlight.get(cond, [])
                ax.plot(
                    x, y, color=ink_of(cond), lw=1.3 if hi else 0.8, alpha=0.95 if hi else 0.55, zorder=3 if hi else 2
                )
            rule(ax, ex.RISE_LEVEL)
            ax.set_title(cond, fontsize=9)
            ax.set_xlim(0, epochs)
            ax.set_xlabel("epoch", fontsize=8)
        axes[0].set_ylabel("HSV skill (EEM)", fontsize=8)
        return fig

    return _plot()


# --- E1 -----------------------------------------------------------------------------------------------------


def ops_figure() -> str:
    data = {
        f"{c}|{e}": [RUNS[(c, e, s_)]["per_op"][HSV_IDX].tolist() for s_ in ex.SEEDS]
        for e in ex.LENGTHS
        for c in (CONTROL, ANCHOR)
    }
    caption = f"""
        **The skill on each HSV op at the end of training**, on the whole held-out set. One group of three bars per
        model seed (hue, saturation, and value, left to right); one panel per condition and length. The dashed line is
        the rise level ({ex.RISE_LEVEL:g}).
    """
    alt = f"""
        Four bar charts, control and anchor at 200 and at 400 epochs, each with three bars per model seed for the hue,
        saturation, and value ops. At 200 epochs, the anchored runs at seeds {seeds_list(SLOW)} have low bars on all
        three ops, and control 700 is low on saturation and value. At 400 epochs every run has all three bars near
        0.5.
    """
    return ops_draw(data, caption, alt)


@memo
def ops_draw(data: dict, caption: str, alt_text: str) -> str:
    @themed(name="ex-2.2.23-ops", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.8), layout="constrained", sharey=True, sharex=True)
        inks = [light_dark(*pair) for pair in (("#7b3294", "#c2a5cf"), ("#1b7837", "#a6dba0"), ("#2166ac", "#92c5de"))]
        x = np.arange(len(ex.SEEDS))
        for i, e in enumerate(ex.LENGTHS):
            for j, c in enumerate((CONTROL, ANCHOR)):
                ax = axes[i][j]
                vals = np.array(data[f"{c}|{e}"])
                for o, name in enumerate(ex.HSV_OPS):
                    ax.bar(
                        x + (o - 1) * 0.27, vals[:, o], width=0.27, color=inks[o], label=name if i == j == 0 else None
                    )
                rule(ax, ex.RISE_LEVEL)
                ax.set_title(f"{c}, {e} epochs", fontsize=8)
                ax.set_xticks(x, [str(s_) for s_ in ex.SEEDS], fontsize=6.5, rotation=90)
            axes[i][0].set_ylabel("EEM", fontsize=8)
        for ax in axes[1]:
            ax.set_xlabel("model seed", fontsize=8)
        fig.legend(loc="outside upper center", ncols=3, frameon=False, fontsize=7)
        return fig

    return _plot()


def run_table(epochs: int) -> str:
    """Per model seed at one length: the rise epoch, the worst HSV op, and task skill, for both conditions."""
    rows = []
    for s_ in ex.SEEDS:
        row = [str(s_)]
        for c in (CONTROL, ANCHOR):
            r = RUNS[(c, epochs, s_)]
            row += ["—" if r["rise"] is None else f"{r['rise']:.0f}", f"{r['hsv_min']:.2f}", f"{r['eem']:.3f}"]
        rows.append(row)
    head = ["seed"] + [f"{c}: {m}" for c in (CONTROL, ANCHOR) for m in ("rise", "worst op", "task")]
    return table_html(
        head,
        rows,
        f"**Every run at {epochs} epochs.** The rise epoch (a dash where the run never rose), the skill on the worst "
        "HSV op at the end of training, and task skill, all on the whole held-out set but the rise, which is measured "
        "through training.",
    )


# --- E2 -----------------------------------------------------------------------------------------------------


def rise_figure() -> str:
    pts = {
        c: [(s, RUNS[(c, ex.SHORT, s)]["rise"], RUNS[(c, ex.LONG, s)]["rise"]) for s in ex.SEEDS]
        for c in (CONTROL, ANCHOR)
    }
    caption = f"""
        **When the rise comes, at 200 and at 400 epochs**, one dot per model seed and condition: the first epoch at
        which the HSV skill passes {ex.RISE_LEVEL:g}. A run that never passes it is drawn on the edge, past the end of
        its training. The diagonal marks the same epoch at either length; the dotted line, the same share of training.
    """
    alt = """
        A scatter of the rise epoch at 400 epochs against the rise epoch at 200 epochs, one dot per seed for each
        condition, with a diagonal for the same epoch and a steeper line for the same share of training.
    """
    return rise_draw(pts, caption, alt)


@memo
def rise_draw(pts: dict, caption: str, alt_text: str) -> str:
    @themed(name="ex-2.2.23-rise", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(3.8, 3.4), layout="constrained")
        edge_x, edge_y = ex.SHORT * 1.08, ex.LONG * 1.06
        for cond, rows in pts.items():
            xs = [edge_x if a is None else a for _, a, _ in rows]
            ys = [edge_y if b is None else b for _, _, b in rows]
            ax.plot(xs, ys, marker_of(cond), ms=4.5, color=ink_of(cond), alpha=0.8, label=cond, mew=0)
        ax.plot([0, edge_x], [0, edge_x], color=light_dark("#000", "#fff"), lw=0.7, alpha=0.5)
        ax.plot([0, edge_x], [0, 2 * edge_x], color=light_dark("#000", "#fff"), lw=0.7, alpha=0.5, ls=":")
        ax.axvline(ex.SHORT, color=light_dark("#888", "#777"), lw=0.6)
        ax.axhline(ex.LONG, color=light_dark("#888", "#777"), lw=0.6)
        ax.set_xlim(0, edge_x * 1.04)
        ax.set_ylim(0, edge_y * 1.03)
        ax.set_xlabel("rise epoch, 200-epoch run", fontsize=8)
        ax.set_ylabel("rise epoch, 400-epoch run", fontsize=8)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        return fig

    return _plot()


def rose_both_ways() -> dict[str, list[int]]:
    """Per condition, the seeds that missed at 200 epochs and made it at 400, and those that made it at 200 and
    missed at 400.
    """
    out = {}
    for c in (CONTROL, ANCHOR):
        out[f"{c} late"] = [s for s in ex.SEEDS if missed((c, ex.SHORT, s)) and not missed((c, ex.LONG, s))]
        out[f"{c} lost"] = [s for s in ex.SEEDS if not missed((c, ex.SHORT, s)) and missed((c, ex.LONG, s))]
    return out


ROSE = rose_both_ways()

# --- E3 -----------------------------------------------------------------------------------------------------


def late_rows() -> list[dict]:
    return [
        {
            "cond": k[0],
            "epochs": k[1],
            "rise": RUNS[k]["rise"],
            "mid": RUNS[k]["midpoint"],
            "eem": RUNS[k]["eem"],
            "margin_last": float(RUNS[k]["margin"][-1]),
            "margin_prev": float(RUNS[k]["margin"][-2]),
        }
        for k in RUNS
    ]


def skill_figure() -> str:
    caption = f"""
        **Task skill at the end of training, against when the run rose**, one mark per run: filled at 200 epochs,
        open at 400. The x axis is the share of training at which the rise came: on the left, the first record at the
        rise level ({ex.RISE_LEVEL:g}); on the right, the midpoint of a logistic curve fitted to the HSV skill, which
        needs no threshold. Runs that never rose sit past the end of training.
    """
    alt = """
        Two panels sharing a task-skill axis, one mark per run, against the rise time as a share of training: by the
        rise level on the left and by the logistic midpoint on the right. The 400-epoch runs cluster high on the
        left; the 200-epoch runs fall as the rise comes later, and those that never rose sit low at the right edge.
    """
    return late_draw(late_rows(), "skill", caption, alt)


def margin_figure() -> str:
    caption = """
        **The op margin of the anchored runs, against when the run rose**: at the last slice on the left and the
        second-last on the right, both against the logistic midpoint as a share of training. Filled at 200 epochs,
        open at 400; a run whose fit finds no rise sits past the end of training.
    """
    alt = """
        Two panels sharing an op-margin axis, one mark per anchored run, against the logistic midpoint as a share of
        training. The margins sit between about 0.7 and 0.8 whatever the timing, with one 200-epoch run lower on both.
    """
    return late_draw(late_rows(), "margin", caption, alt)


@memo
def late_draw(rows: list[dict], which: str, caption: str, alt_text: str) -> str:
    @themed(name=f"ex-2.2.23-late-{which}", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.6), layout="constrained", sharey=True)

        def mark(a, x, y, r):
            filled = r["epochs"] == ex.SHORT
            ink = ink_of(r["cond"])
            a.plot(x, y, marker_of(r["cond"]), ms=4.2, color=ink, mfc=ink if filled else "none",
                   mew=0 if filled else 0.9, alpha=0.8)  # fmt: skip

        for r in rows:
            mid = 1.08 if r["mid"] is None else min(r["mid"] / r["epochs"], 1.08)
            if which == "skill":
                mark(ax, 1.08 if r["rise"] is None else r["rise"] / r["epochs"], r["eem"], r)
                mark(bx, mid, r["eem"], r)
            elif r["cond"] == ANCHOR:
                mark(ax, mid, r["margin_last"], r)
                mark(bx, mid, r["margin_prev"], r)
        if which == "skill":
            ax.set_xlabel("rise epoch / length", fontsize=8)
            bx.set_xlabel("logistic midpoint / length", fontsize=8)
            ax.set_ylabel("task skill (EEM)", fontsize=8)
        else:
            ax.set_xlabel("logistic midpoint / length", fontsize=8)
            bx.set_xlabel("logistic midpoint / length", fontsize=8)
            ax.set_ylabel("op margin", fontsize=8)
            ax.set_title("last slice", fontsize=8)
            bx.set_title("second-last slice", fontsize=8)
        for a in (ax, bx):
            a.set_xlim(0, 1.12)
            a.axvline(1.0, color=light_dark("#888", "#777"), lw=0.6)
        conds = (CONTROL, ANCHOR) if which == "skill" else (ANCHOR,)
        handles = [
            *(plt.Line2D([], [], ls="", marker=marker_of(c), color=ink_of(c), mew=0, label=c) for c in conds),
            plt.Line2D([], [], ls="", marker="o", color="grey", mew=0, label="200 epochs"),
            plt.Line2D([], [], ls="", marker="o", color="grey", mfc="none", mew=0.9, label="400 epochs"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), frameon=False, fontsize=7)
        return fig

    return _plot()


def margin_table() -> str:
    rows = [
        [str(s_)] + [f"{RUNS[(ANCHOR, e, s_)]['margin'][i]:.2f}" for e in ex.LENGTHS for i in (-1, -2)]
        for s_ in ex.SEEDS
    ]
    head = ["seed"] + [f"{e}: {sl}" for e in ex.LENGTHS for sl in ("last", "second-last")]
    return table_html(head, rows, "**The op margin of every anchored run**, at the last and second-last slices.")


# --- S1 -----------------------------------------------------------------------------------------------------


MEASURES = {"hsv": "average", "hsv_min": "worst op"}


def rule_table() -> str:
    rows = []
    for measure, name in MEASURES.items():
        for level in ex.CANDIDATE_RULE_LEVELS:
            row = [name, f"{level:g}"]
            for e in ex.LENGTHS:
                for c in (CONTROL, ANCHOR):
                    row.append(str(sum(missed(k, level, measure) for k in keys(c, e))))
            rows.append(row)
    head = ["HSV skill", "level"] + [f"{c}, {e}" for e in ex.LENGTHS for c in (CONTROL, ANCHOR)]
    return table_html(
        head,
        rows,
        f"**What each candidate rule would leave out**: the number of runs, of {len(ex.SEEDS)} per column, whose HSV "
        "skill at the end of training is below the level.",
        text_cols=2,
    )


# --- E4 -----------------------------------------------------------------------------------------------------

RULE_MEASURE, RULE_LEVEL = ex.RULE
KEPT = {k: not missed(k, RULE_LEVEL, RULE_MEASURE) for k in RUNS}

for _k in RUNS:
    if _k[0] == ANCHOR:
        RUNS[_k] |= edit_measurements(_k)


def edit_figure() -> str:
    rows = [
        {
            "epochs": k[1],
            "kept": KEPT[k],
            "share": RUNS[k]["share"],
            "worst": RUNS[k]["worst"],
            "worst_by_dose": RUNS[k]["by_dose"][:, OTHER].max(axis=1).tolist(),
        }
        for k in keys(ANCHOR)
    ]
    n_out = sum(not r["kept"] for r in rows)
    caption = f"""
        **The edit on every anchored run, by whether the rule of S1 keeps it.** Left: how far the drop on
        `{ex.ANCHORED_OP}` gets toward the target null at full dose, against the largest drop on any other op at any
        dose, one dot per run (filled at 200 epochs, rings at 400), both net of the control at the same seed and
        length. The dashed lines are ex-2.2.21's criteria: at least {ex.GRADING_MIN_DAMAGE:.0%} of the way, and no
        other op down by more than {ex.SELECTIVITY_GATE:g}. Right: the largest drop on any other op at each dose, one
        line per run. Blue: runs the rule keeps; orange: the {num_word(n_out)} it leaves out.
    """
    alt = f"""
        Two panels. On the left, a scatter of the largest drop on another op against the share of the way to the target
        null, one dot per anchored run, colored by whether the rule keeps the run, with dashed lines at
        {ex.GRADING_MIN_DAMAGE:.0%} and {ex.SELECTIVITY_GATE:g}. On the right, the largest drop on another op against
        the dose, one line per run.
    """
    return edit_draw(rows, caption, alt)


@memo
def edit_draw(rows: list[dict], caption: str, alt_text: str) -> str:
    @themed(name="ex-2.2.23-edit", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.9), layout="constrained")
        kept_ink, out_ink = light_dark("#1f6fb2", "#7ab8f5"), light_dark("#d97706", "#fbbf24")
        for r in rows:
            ink = kept_ink if r["kept"] else out_ink
            filled = r["epochs"] == ex.SHORT
            ax.plot(r["share"], r["worst"], "o", ms=4.5, color=ink, mfc=ink if filled else "none",
                    mew=0 if filled else 1.0, alpha=0.85)  # fmt: skip
            bx.plot(ex.DOSE_GAMMAS, r["worst_by_dose"], color=ink, lw=0.9, ls="-" if filled else "--", alpha=0.7)  # fmt: skip
        for a in (ax, bx):
            a.axhline(ex.SELECTIVITY_GATE, color=light_dark("#000", "#fff"), lw=0.8, ls="--", alpha=0.6)
        ax.axvline(ex.GRADING_MIN_DAMAGE, color=light_dark("#000", "#fff"), lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel(f"share of the way to the target null on {ex.ANCHORED_OP}", fontsize=8)
        ax.set_ylabel("largest drop on another op", fontsize=8)
        bx.set_xlabel("dose", fontsize=8)
        bx.set_ylabel("largest drop on another op", fontsize=8)
        handles = [
            plt.Line2D([], [], ls="", marker="o", color=kept_ink, mew=0, label="kept"),
            plt.Line2D([], [], ls="", marker="o", color=out_ink, mew=0, label="left out"),
            plt.Line2D([], [], ls="", marker="o", color="grey", mew=0, label="200 epochs"),
            plt.Line2D([], [], ls="", marker="o", color="grey", mfc="none", mew=1.0, label="400 epochs"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=4, frameon=False, fontsize=7)
        return fig

    return _plot()


def edit_table() -> str:
    rows = []
    for s_ in ex.SEEDS:
        row = [str(s_)]
        for e in ex.LENGTHS:
            r = RUNS[(ANCHOR, e, s_)]
            row += ["" if KEPT[(ANCHOR, e, s_)] else "left out", f"{r['share']:.2f}", f"{r['worst']:.2f}"]
        rows.append(row)
    head = ["seed"] + [f"{e}: {m}" for e in ex.LENGTHS for m in ("rule", "share", "largest drop")]
    return table_html(
        head,
        rows,
        f"**The edit on every anchored run**: the share of the way to the target null on `{ex.ANCHORED_OP}` at full "
        "dose, and the largest drop on any other op at any dose, both net of the control at the same seed and length.",
        text_cols=2,
    )


def edit_counts() -> dict[tuple[int, bool], tuple[int, int]]:
    """Per length and group (kept or not): how many runs fall outside ex-2.2.21's selectivity criterion, of how
    many.
    """
    out = {}
    for e in ex.LENGTHS:
        for kept in (True, False):
            ks = [k for k in keys(ANCHOR, e) if KEPT[k] == kept]
            out[(e, kept)] = (sum(not RUNS[k]["selective"] for k in ks), len(ks))
    return out


# --- Numbers for the prose ----------------------------------------------------------------------------------


# The anchored seeds that miss the rise at 200 epochs.
SLOW = MISSED[(ANCHOR, ex.SHORT)]

COUNTS = edit_counts()


# %%

rf"""
# Ex 2.2.23: The slow seeds, trained for longer

/// tip |
<!-- lede -->
A scout of the runs that miss the second rise in task skill, at 200 and 400 epochs. At 200 epochs, {num_word(len(SLOW))} anchored runs missed it, and at 400 every run made it. But the edit spills onto other ops on nearly every run that made the rise, so the clean edits of earlier experiments came mostly from half-trained runs. We keep every run, at 400 epochs.
///

In ex-2.2.21 and ex-2.2.22, a few runs never made the second rise in task skill within 200 epochs, and those runs set most of the seed band of every measurement. This scout trains the recipe of record and the control at {len(ex.NEW_SEEDS)} new seeds at 200 epochs, and at all {len(ex.SEEDS)} seeds at 400 epochs, reusing ex-2.2.21's 200-epoch runs at the other {len(ex.REUSED_SEEDS)}.
"""

# %%

rf"""
## Observations

Each item below is a measurement on the runs of this scout, with no gate.

- [How often a run misses the rise (E1)](#how-often-a-run-misses-the-rise-e1): at 200 epochs, the anchor misses at {num_word(len(SLOW))} of {len(ex.SEEDS)} seeds and the control at none. The controls at those seeds rose early, which suggests the anchor slows some seeds, though the evidence is weak and the cause is not known.
- [Trained for 400 epochs (E2)](#trained-for-400-epochs-e2): every run of both conditions makes the rise at 400 epochs, including the {num_word(len(SLOW))} that missed it at 200.
- [A late rise and an early one (E3)](#a-late-rise-and-an-early-one-e3): at 400 epochs, a late rise ends close to an early one in task skill and op margin.
- [A rule for runs that miss the rise (S1)](#a-rule-for-runs-that-miss-the-rise-s1): the worst HSV op below {ex.RULE[1]:g}, which leaves out four runs at 200 epochs and none at 400.
- [The edit, with and without the rise (E4)](#the-edit-with-and-without-the-rise-e4): the edit spills onto other ops on {COUNTS[(ex.SHORT, True)][0] + COUNTS[(ex.LONG, True)][0]} of the {COUNTS[(ex.SHORT, True)][1] + COUNTS[(ex.LONG, True)][1]} runs the rule keeps, and on none of the runs it leaves out.
- [Decision](#decision): no rule for leaving runs out. The next experiment that trains runs keeps every run, at 400 epochs, and the next step is to understand the spill.
"""

# %%

rf"""
## Scope

This is a scout, with no preregistration and no gate. Each condition has {len(ex.SEEDS)} runs at 200 epochs and {len(ex.SEEDS)} at 400, so a share of runs that miss the rise is known only roughly: one run more or less moves it by about a twelfth. Pairing by seed allows comparisons between the two conditions, and between the two lengths.
"""

# %%

rf"""
## Why this experiment

The model learns the seven ops in two stages. Early in training it learns `mix`, `lighten`, `darken`, and `{ex.ANCHORED_OP}`, with skill rising quickly to a plateau. Some tens of epochs later it learns the three HSV ops, which change one aspect of a color in hue, saturation, and value space, and the skill rises a second time. The timing of the second rise varies from run to run.

A few runs never make it within 200 epochs: their HSV skill stays near where it was on the plateau, and they end well below the others, though they answer the other four ops about as well. In the 200-epoch runs this scout reuses, that happened on the anchored condition only, at two of the five seeds:

{earlier_figure()}

The two control runs that rose late also ended lower than the three that rose early, so a late rise may leave a run short of the others even when it comes within 200 epochs.

So the seed band of our measurements is mostly a record of which runs made the second rise. Ex-2.2.22 could not tell a small cost of the anchor from a slow start, and along its caps, the runs whose edit spilled onto other ops were all runs that had made the rise. If every run makes the rise when given time, a future experiment could keep the 200-epoch budget and leave out the runs that miss it, as half-trained models. Before that is safe, we need to know three things:

- How often a run misses the rise at 200 epochs, and whether the anchor changes that. Pairing by seed separates a seed that is slow under any condition from one the anchor makes slow. If the anchor decides who is slow, leaving the slow runs out could hide a cost of the anchor. If every slow run would make the rise eventually, that cost may not matter for the full recipe, but the number left out of each condition would still have to be reported wherever it differs between conditions.
- Whether a run that missed the rise at 200 epochs makes it at 400. If it doesn't, leaving it out selects a kind of seed rather than waiting for a slow one.
- Whether a run that rises late ends like one that rises early, in task skill, in how well the anchor holds, and in the edit. In the figure above, the late-rising controls end a little lower than the early ones; a run that only rises within 400 epochs could end with the early risers, or lower still.

A 400-epoch run differs from its 200-epoch twin from the start, since the learning rate decays more slowly over a longer schedule, so the two have different trajectories. A run that rises by 400 epochs and stalls in 200 would suggest that the seed can make the rise under a longer schedule, which may support a policy that drops slow runs.

Any rule for leaving runs out has to look only at the unedited model (its task skill, or how well calibrated it is), and be fixed before the edit is scored, so that it cannot select on the anchoring results. That matters here because the edit spilled only on runs that *had* made the rise.
"""

# %%

rf"""
## Parameters

The recipe of record, as the [D2.2 design](/docs/m2/d2.2/design.md#the-setup-today) currently states it: the seven-op set, three examples per context with replacement op noise of 0.3, {ex.MODEL}, the newline mask, the whole-line label on about one `{ex.ANCHORED_OP}` context in fifty, every slice pulled, and no cap. That is ex-2.2.21's `anchor-whole` condition. The control is the same without the anchor.

{runs_table()}

**The lengths.** 200 epochs is the recipe; 400 was the recipe before ex-2.2.19 halved it, and every 400-epoch run there (unanchored, three seeds) made the second rise, at about the same epoch as the 200-epoch runs. The anchor schedules scale with the length: the weight warms up over the first tenth of training and anneals over the last tenth. The learning rate peaks at the same value and warms up over the same number of epochs at either length, and its cosine decay stretches over the run.

**The seeds.** The reused runs are at model seeds {seed_span(ex.REUSED_SEEDS)}, and the new ones at {seed_span(ex.NEW_SEEDS)}, so every comparison is paired by seed: a condition with its control, and a 200-epoch run with its 400-epoch twin.

<!-- REVIEW: the open decision on seeds and lengths is resolved in review round 1: twelve seeds at 200 and 400 epochs. 300 epochs was rejected. More seeds at 200 epochs, and twelve runs at 600 epochs if 400 leaves the question open, are to be decided after the results. -->
More seeds at 200 epochs would pin down how often a run misses the rise, and runs at 600 epochs would help if some seeds have still not risen by 400. Whether either is needed depends on these results.
"""

# %%

rf"""
## Glossary

<!-- REVIEW: the measurements, as this report's glossary, from review round 1, so each shows beside its first use in every section. -->

The measurements this report uses. Each definition also shows in the margin beside the first use of its term in a section.

HSV skill
:   Skill on the three HSV ops. It is the expected exact match (EEM, the probability the model puts on the right answer) on held-out contexts, averaged over the three. Beside the average we report the worst of the three, since the ops can rise at different times. Through training it is measured every {ex.TRAJ_STRIDE_EPOCHS} epochs on a subsample of the held-out set, and at the end on the whole of it.

The rise
:   When a run learns the HSV ops. A run has made the second rise once its HSV skill passes {ex.RISE_LEVEL:g}, about halfway between the plateau and where the runs that rise end up. The *rise epoch* is the first record at or above that level. Runs that rise do so quickly, so the level matters little to the epoch; the figure above shows it against the earlier runs. On a few runs only `hue-hsv` rises within 200 epochs, which leaves the average near the level, so E1 shows the ops one at a time too. A version with no threshold, for E3: the midpoint of a logistic curve fitted to the HSV skill through training, which says when the rise is half done.

Task skill
:   Skill on all seven ops. It is the EEM on held-out contexts over the seven, compared with the control at the same seed and length.

Op margin
:   How far `{ex.ANCHORED_OP}` contexts sit along e₁ beyond the rest. It is measured at the last slice, as in ex-2.2.21. The second-last slice may say more about the op, since the last feeds only the readout; the eval measures the margin at every slice, so E3 can show both.

The edit
:   The projection of e₁ at every position. It is applied at doses from a quarter to all of the component. Two measurements per run: how far the drop on `{ex.ANCHORED_OP}` gets toward the target null at full dose, and the largest drop on any other op at any dose. Both are net of what the edit does to the control at the same seed and length. Here they are measurements: ex-2.2.21 (E2) set criteria on them (at least {ex.GRADING_MIN_DAMAGE:.0%} of the way, and no other op down by more than {ex.SELECTIVITY_GATE:g}), and E4 uses those only to say which runs fall outside them.
"""

# %%

rf"""
## How often a run misses the rise (E1)

How many runs of each condition end 200 epochs without having made the rise, paired by seed.

At 200 epochs, the anchor misses the rise at {num_word(len(SLOW))} of {len(ex.SEEDS)} seeds, and the control at none. Through training, the HSV skill of those {num_word(len(SLOW))} runs climbs slowly and never bends upward:

{traj_panels(ex.SHORT, {ANCHOR: SLOW}, "ex-2.2.23-traj-200", f"**The HSV skill through training, at 200 epochs.** One line per model seed; the runs that miss the rise are drawn heavier, and the runs of the other condition faintly behind. The dashed line is the rise level ({ex.RISE_LEVEL:g}).", f"Two panels, control and anchor, of HSV skill against epoch for twelve runs each. Every control run passes the dashed rise level; on the anchor panel, {num_word(len(SLOW))} runs stay below it to the end.")}

The pairing suggests the anchor slowed these seeds: the controls at the same seeds rose among the earliest of the twelve. But the two runs of a pair share only their seed, and once the anchor term changes their path, a small difference in where it leads could decide the timing of the rise. With {num_word(len(SLOW))} slow runs against none, the evidence that the anchor raises the share of slow runs is weak, and we don't know the cause.

{run_table(ex.SHORT)}

The figure below shows each HSV op on its own. The slow anchored runs are low on all three, though one has learned part of `value-hsv`. One control run rose late and has learned `hue-hsv` well but saturation and value only in part, so its average is near the level.

{ops_figure()}

Loosely, a run looks like its twin at the other length in the same condition, but not like its pair in the other condition: a seed that is quick under the control is no more likely to be quick under the anchor.
"""

# %%

rf"""
## Trained for 400 epochs (E2)

Whether the runs that missed the rise at 200 epochs make it at 400, or the other way round, and when the rise comes at either length. In ex-2.2.19 the rise came at about the same epoch at either length, which would mean a longer run gives a slow seed more time at a high learning rate, so the rise epoch is shown both in epochs and as a share of training.

At 400 epochs every run of both conditions makes the rise, including the {num_word(len(ROSE[f"{ANCHOR} late"]))} anchored seeds that missed it at 200, and no run that rose at 200 epochs misses at 400.

{traj_panels(ex.LONG, {ANCHOR: SLOW}, "ex-2.2.23-traj-400", f"**The HSV skill through training, at 400 epochs**, laid out as above. The heavier lines are the seeds whose anchored run missed the rise at 200 epochs ({seeds_list(SLOW)}).", f"Two panels, control and anchor, of HSV skill against epoch over 400 epochs, twelve runs each. Every run passes the dashed rise level, the last of them after epoch {max(RUNS[k]['rise'] for k in keys(ANCHOR, ex.LONG)):.0f}.")}

Those seeds rose within the first 200 epochs of their 400-epoch runs, though their 200-epoch twins never rose. We don't understand why. The two schedules differ from the start (the anchor weight warms up over twice as many epochs, and the learning rate decays more slowly), and it seems a small difference in training can tip a seed either way.

{run_table(ex.LONG)}

Across the 400-epoch runs, the rise tends to come later and over a wider spread with the anchor, as at 200 epochs.

{rise_figure()}

Many seeds rise at about the same epoch at either length, as in ex-2.2.19, but the pairing is loose, and some seeds move a long way in either direction. So a slow seed at 200 epochs need not be slow at 400.
"""

# %%

rf"""
## A late rise and an early one (E3)

Whether a run that rises late ends like one that rises early, in task skill and in the op margin: does a run that only rises within 400 epochs end with the early risers, or lower, like the late-rising controls of ex-2.2.21?

At 400 epochs a late rise ends close to an early one. At 200 epochs, a run that rises in the second half of training ends a little lower, and a run that never rises ends well below the rest. The tables of E1 and E2 have the task skill of every run.

{skill_figure()}

The op margin is less tied to the rise. It is about the same on every anchored run, early or late, at either length, except for one slow run at 200 epochs, which is lower at both slices.

{margin_figure()}

{margin_table()}
"""

# %%

rf"""
## A rule for runs that miss the rise (S1)

<!-- REVIEW: added that the reused seeds already have known edit results, so the candidate levels are fixed in experiment.py before any run. Committing the level before E4 alone does not blind it to the five reused runs. -->
A rule that marks a run as half-trained, for the next experiment to leave out and replace with the next unused seed. The rule looks only at the HSV skill on the held-out set at the end of training, as the average over the three ops or the worst of them. The measure and its level are chosen from E1 to E3 and committed before E4 is filled in, so that the choice cannot follow the edit results of this scout. The edit results at the reused seeds (700–704) are already known from ex-2.2.21, so the candidate levels were fixed with this design, before any run: {", ".join(f"{v:g}" for v in ex.CANDIDATE_RULE_LEVELS)}. The chosen level is one of them. We report what each candidate would leave out of each condition at either length.

{rule_table()}

Every candidate leaves out runs at 200 epochs only, and mostly anchored ones. We chose the worst op below {ex.RULE[1]:g}, in commit `3268c14`, before E4 was computed. It leaves out the {num_word(len(SLOW))} anchored runs that miss the rise and the control run that had learned only `hue-hsv`; the average would keep that run, though it is half-trained. The four runs it leaves out sit well below the rest on their worst op, so any level in the gap between them leaves out the same runs.
"""

# %%

rf"""
## The edit, with and without the rise (E4)

Whether the edit gets as far toward the target null on the runs the rule keeps as on the runs it leaves out, and whether it spills onto other ops more often on one group.

The edit spills onto other ops on nearly every run that has learned the HSV ops, and on none of the runs that have not. At 200 epochs, {num_word(COUNTS[(ex.SHORT, True)][0])} of the {num_word(COUNTS[(ex.SHORT, True)][1])} runs the rule keeps lower another op by more than {ex.SELECTIVITY_GATE:g}, and none of the {num_word(COUNTS[(ex.SHORT, False)][1])} it leaves out does. At 400 epochs the rule keeps every run, and {num_word(COUNTS[(ex.LONG, True)][0])} of {num_word(COUNTS[(ex.LONG, True)][1])} spill.

{edit_figure()}

The spills are larger at 400 epochs than at 200, and they grow with the dose. At 200 epochs they land mostly on the HSV ops, and on `darken` at a few seeds; at 400 epochs, on every other op, most of all `darken` and `lighten`. No seed stays within the criterion at both lengths.

{edit_table()}

The removal itself holds up: the edit gets most of the way to the target null on every kept run. Of the runs left out, it gets most of the way on two, and almost nowhere on the third, the run with the lowest op margin.
"""

# %%

r"""
## Decision

How the next experiment trains and counts its runs. The candidates:

- 200 epochs, with the rule of S1 leaving runs out and topping up from the next unused seed, and the count left out of each condition reported.
- 400 epochs, keeping every run.
- 200 epochs, keeping every run, at more seeds.

The inputs:

- E1: the rule leaves out a quarter of the anchored runs at 200 epochs, and fewer controls. The pairing suggests the anchor changes who is slow.
- E2: every run that missed at 200 epochs makes the rise at 400, so a longer run waits for a slow seed rather than selecting a kind of seed.
- E3: at 400 epochs a late rise ends with the early ones, in task skill and op margin.
- E4: the runs the rule leaves out are the runs whose edit stays clean. Leaving them out would select on the edit results after all, though the rule never looks at them, but against the anchor: it drops the runs that look best.
- Cost: with a quarter of the runs replaced in each round, a run kept under the rule costs about two-thirds of a 400-epoch run, in sequential rounds, against one round at 400 epochs.

So we leave out no runs. The next experiment that trains runs should keep every run, at 400 epochs, where all of them make the rise and end alike. The rule would save a third of the compute, but in sequential rounds, and it would still mix runs trained for different lengths. More seeds at 200 epochs would keep the mix of half-trained and fully trained runs, and the clean edits of the half-trained runs would flatter the recipe. Before further training, we would like to understand the spill, on the checkpoints this scout already has.
"""

# %%

r"""
## Discussion

The largest result here is about the edit. On the five anchored runs at 200 epochs that this scout reuses from ex-2.2.21, the edit stayed within the selectivity criterion on three, and two of those three had missed the rise. With the rise as a factor, the picture is plain: once a run has learned the HSV ops, the edit spills onto other ops on nearly every seed, and more so after 400 epochs. So the selectivity of the edit that ex-2.2.21 and ex-2.2.22 measured leaned on half-trained runs, and the recipe of record does not yet give a selective edit on a fully trained model.

Why the spill comes with the HSV skill is open. The ops it lands on (`darken`, `lighten`, and the HSV ops, `value-hsv` most of all) all deal with lightness, which suggests the fully trained model stores some lightness information along e₁, and the edit removes it with the op. The anti-subspace term keeps other information off e₁, and its weight decays with the learning rate, so a 400-epoch run spends twice as many steps learning while the term is weak. That is a guess. Editing at only some positions (the colors, or the evidence for the op), and checking whether the e₁ component of a color tracks its lightness, would test it on the checkpoints this scout already has. If it holds, a weaker anchor late in training would loosen the hold on e₁ further, and keeping the anti-subspace weight up for longer would be the change to try.

Where the anchor lands may matter too. In ex-2.2.16 and ex-2.2.21, the whole-line label put most of its alignment on the answer positions. An answer is a color, and in the examples it is also the evidence for the op, so e₁ there could carry both. A probe for the op at the answer positions of the control would say whether the op is represented there without the anchor.

The schedules are from ex-2.1.10, set when *red* was anchored in a grammar of one equation per line, and have not been tuned since the in-context grammar.

Which seeds make the rise late, and why, is open too. A slow anchored seed is quick under the control, and rises in good time at 400 epochs, so slowness is a property of a path through training rather than of a seed. We don't know what decides it.
"""

# %%

rf"""
## Method

**Reused runs.** The 200-epoch runs at model seeds {seed_span(ex.REUSED_SEEDS)} are ex-2.2.21's `control` and `anchor-whole`, resolved by ref along with the corpus, held-out set, and probes they trained and were scored on. The new runs train on the same corpus with ex-2.2.21's code, so a new 200-epoch run differs from a reused one in its seed alone.

**Scoring.** Every run, reused or new, is scored with ex-2.2.21's eval and with the suppression pass of ex-2.2.22, at every position.

**Cost.** About \$9 on Modal, under the planned budget of \${ex.BUDGET_USD}, for {ex.N_NEW_RUNS} new runs on an L4 and the scoring passes over all of them and the reused runs.
"""
