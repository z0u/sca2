# title: Ex 2.2.17: the center control plateau, a scout

# The design constants and the refs come from `experiment.py` beside this script (the script's directory is
# on sys.path while it runs); the answer table and the posterior come from ex-2.2.16 through it.
import colorsys
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed
from sca.config import SchedulerConfig
from sca.data.ops import colors
from sca.training.scheduler import configure_schedule
from sca.vis import plot_rgb_cube

X = ex.ex2216
OPS: tuple[str, ...] = tuple(X.OP_NAMES)
GROUPS: tuple[tuple[str, tuple[str, ...] | None], ...] = (
    ("all ops", None),
    ("mix, screen, multiply", ("mix", "screen", "multiply")),
    ("lighten, darken", ("lighten", "darken")),
    ("difference, exclusion", ("difference", "exclusion")),
    ("hsvmix", ("hsvmix",)),
    ("HSV channels", ("hue-hsv", "sat-hsv", "value-hsv")),
)
PEAK = ex.SWEEP_LRS[2]
# Bands of the posterior on the true op, and the confident band the answer analysis reads.
BANDS = (0.0, 0.5, 0.9, 0.99, 1.01)
BAND_LABELS = ("< 0.5", "0.5–0.9", "0.9–0.99", "> 0.99")
CONFIDENT = 0.99
# A grid color is in the support of the Bayes predictive when q gives it at least this much.
SUPPORT = 0.02
HUE_GAPS = (0, 30, 90, 150, 180.1)
CUBE_RUN = f"sweep-{PEAK:g}-s1"
RGB = np.array(colors(), dtype=float) / 15
# Squared distance on the color grid, in grid steps (the levels are 0 to 15 in steps of 3), between every pair.
GRID_D2 = ((RGB[:, None] - RGB[None]) ** 2).sum(-1) * 25
# Bins of the squared distance to the nearest support color: one step, √2, √3, 2 to √5, √6 to 3, and further.
DIST_BINS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 75))
DIST_LABELS = ("1", "1.4", "1.7", "2–2.2", "2.4–3", "over 3")


# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    """An authored result table in the shared report style; the first *text_cols* columns are text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def rule_color() -> str:
    return light_dark("#333", "#ccc")


def gate(ax: Axes, level: float) -> None:
    """The pass line as a dashed rule with the failing side hatched, drawn last so the limits hold."""
    lo, hi = ax.get_ylim()
    ax.axhline(level, ls="--", color=rule_color(), lw=1)
    ax.axhspan(lo, level, facecolor="none", edgecolor=rule_color(), hatch="//", lw=0, alpha=0.1, zorder=0)
    ax.set_ylim(lo, hi)


# --- Loading ------------------------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict:
    assert path is not None, "ex-2.2.17 has not published this yet"
    return json.loads(path.read_text())


with tempfile.TemporaryDirectory() as _tmp:
    _files = fetch([ex.EVAL_REF, ex.TRAJ_REF, ex.FINDER_REF], Path(_tmp))
    EVAL = read_json(_files[ex.EVAL_REF])
    TRAJ = read_json(_files[ex.TRAJ_REF])
    FINDER = read_json(_files[ex.FINDER_REF])
    RUNS = {r["label"]: r for r in EVAL["runs"]}
    SCORED = tuple(r["label"] for r in EVAL["runs"] if ex.scored(r))
    _refs = [ex.DETAIL_REF, *(ex.SCORE_REF.format(label=lbl) for lbl in SCORED)]
    _arrays = fetch(_refs, Path(_tmp))
    assert all(_arrays.values()), "ex-2.2.17 has not published its answer scoring yet"
    with np.load(cast(Path, _arrays[ex.DETAIL_REF])) as _z:
        DETAIL = {k: _z[k] for k in _z.files}
    ANSWERS = {
        lbl: np.load(cast(Path, _arrays[ex.SCORE_REF.format(label=lbl)]))["p"]
        for lbl in SCORED  # float16, (contexts, 216)
    }


def eem(label: str) -> float:
    return RUNS[label]["task"]["eem"]["all"]


def kl(label: str) -> float:
    return RUNS[label]["task"]["kl"]["all"]


CEILING = RUNS[SCORED[0]]["task"]["ceiling"]["all"]
EPOCH_LENGTH = round(TRAJ[SCORED[0]]["step"][-1] / RUNS[SCORED[0]]["epochs"])
PASS = CEILING - X.CEILING_MARGIN
CEILING_PER_OP = np.array(RUNS[SCORED[0]]["task"]["ceiling"]["per_op"])


def labels_of(arm: str, seeds: range = range(3)) -> list[str]:
    return [f"{arm}-s{s}" for s in seeds if f"{arm}-s{s}" in RUNS]


def seed_stats(labels: Sequence[str], f=eem) -> tuple[float, float, float]:
    v = [f(lbl) for lbl in labels]
    return float(np.mean(v)), min(v), max(v)


# --- The answer scoring --------------------------------------------------------------------------------------


@memo
def answer_table():
    return X._get_posterior().build_table(X.TABLE)


@memo
def context_info(op: np.ndarray, pair: np.ndarray, post: np.ndarray, source: np.ndarray) -> dict:
    """Per held-out context: its posterior band, its count of noisy examples, the hue gap of its operands, and
    which grid colors are an answer of some op on its query pair (`any_op`, contexts × 216).
    """
    table = answer_table()
    n = len(op)
    rows = np.arange(n)
    any_op = np.zeros((n, len(RGB)), dtype=bool)
    for o in range(X.N_OPS):
        idx, ok = table.idx[o, pair], table.prob[o, pair] > 0
        for j in range(idx.shape[1]):
            any_op[rows[ok[:, j]], idx[ok[:, j], j]] = True
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in RGB])
    a, b = pair // len(RGB), pair % len(RGB)
    gap = np.abs(hsv[a, 0] - hsv[b, 0]) * 360
    gap = np.minimum(gap, 360 - gap)
    return {
        "post_true": post[rows, op],
        "band": np.digitize(post[rows, op], BANDS) - 1,
        "noise": (source == 1).sum(axis=1),
        "gray": (hsv[a, 1] == 0) | (hsv[b, 1] == 0),
        "hue_gap": gap,
        "any_op": any_op,
        "match_idx": np.maximum(table.idx[op, pair], 0),
        "match_prob": table.prob[op, pair],
    }


INFO = context_info(DETAIL["op_ids"], DETAIL["query_pair"], DETAIL["posterior"], DETAIL["source"])
PREDICTIVE = DETAIL["predictive"].astype(float)
OP_IDS = DETAIL["op_ids"]
HO_CEILING = DETAIL["ceiling"]


def matched(p: np.ndarray, info: dict) -> np.ndarray:
    """Expected exact match per context, as `ex2216._match` computes it."""
    return (np.take_along_axis(p, info["match_idx"], axis=1) * info["match_prob"]).sum(axis=1)


@memo
def run_summary(p16: np.ndarray, q: np.ndarray, op: np.ndarray, info: dict) -> dict:
    """One run's expected exact match by posterior band, by noise count, and (hsvmix) by hue gap; and, on the
    confident contexts of each op, the mass it puts outside the Bayes support, split by whether some op gives
    that answer.
    """
    p = p16.astype(float)
    e = matched(p, info)
    off = q < SUPPORT
    conf = info["post_true"] > CONFIDENT
    hsvmix = (op == OPS.index("hsvmix")) & (info["post_true"] > 0.9) & ~info["gray"]
    gap_bin = np.digitize(info["hue_gap"], HUE_GAPS) - 1
    return {
        "eem": float(e.mean()),
        "by_band": [float(e[info["band"] == i].mean()) for i in range(len(BANDS) - 1)],
        "by_noise": [float(e[info["noise"] == k].mean()) for k in range(ex.CENTRE[0] + 1)],
        "by_gap": [float(e[hsvmix & (gap_bin == i)].mean()) for i in range(len(HUE_GAPS) - 1)],
        "off_other": [float((p * off * info["any_op"])[conf & (op == o)].sum(1).mean()) for o in range(X.N_OPS)],
        "off_none": [float((p * off * ~info["any_op"])[conf & (op == o)].sum(1).mean()) for o in range(X.N_OPS)],
    }


SUMMARY = {lbl: run_summary(ANSWERS[lbl], PREDICTIVE, OP_IDS, INFO) for lbl in SCORED}


@memo
def support_distance(q: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """On the confident contexts, the squared grid distance from every grid color to the nearest color in the
    support of the Bayes predictive (contexts × 216; zero on the support).
    """
    supp = q[conf] >= SUPPORT
    d2 = np.empty(supp.shape)
    for s in range(0, len(supp), 500):
        d2[s : s + 500] = np.where(supp[s : s + 500, :, None], GRID_D2[None], np.inf).min(1)
    return np.rint(d2)


def operand_box(pair: np.ndarray) -> np.ndarray:
    """Whether each grid color lies in the box the two operands of a query span, channel by channel (contexts × 216)."""
    a, b = RGB[pair // len(RGB)], RGB[pair % len(RGB)]
    lo, hi = np.minimum(a, b)[:, None], np.maximum(a, b)[:, None]
    return ((RGB[None] >= lo) & (RGB[None] <= hi)).all(-1)


@memo
def leak_by_distance(p16: np.ndarray, d2: np.ndarray, any_op: np.ndarray, box: np.ndarray) -> dict:
    """On the confident contexts, off the support, the mean mass per grid color on other ops' answers and on
    colors no op gives, in bins of distance to the support, over all colors and within the operand box; and the
    share of the other-op mass in each bin.
    """
    p = p16.astype(float)
    out = {"other": [], "none": [], "other_box": [], "none_box": [], "share": []}
    total = (p * (d2 > 0) * any_op).sum()
    for lo, hi in DIST_BINS:
        b = (d2 >= lo) & (d2 <= hi)
        out["other"].append(float(p[b & any_op].mean()))
        out["none"].append(float(p[b & ~any_op].mean()))
        out["other_box"].append(float(p[b & box & any_op].mean()))
        out["none_box"].append(float(p[b & box & ~any_op].mean()))
        out["share"].append(float((p * (b & any_op)).sum() / total))
    return out


_conf = INFO["post_true"] > CONFIDENT
_d2 = support_distance(PREDICTIVE, _conf)
_box = operand_box(DETAIL["query_pair"][_conf])
LEAK_DIST = {lbl: leak_by_distance(ANSWERS[lbl][_conf], _d2, INFO["any_op"][_conf], _box) for lbl in SCORED}
DIST_RATIO = np.array([np.array(v["other"]) / np.array(v["none"]) for v in LEAK_DIST.values()])
DIST_RATIO_BOX = np.array([np.array(v["other_box"]) / np.array(v["none_box"]) for v in LEAK_DIST.values()])
DIST_SHARE = np.mean([v["share"] for v in LEAK_DIST.values()], axis=0)
# Grid colors per confident context in each bin: other ops' answers, and the rest.
DIST_N_OTHER = [float(((_d2 >= lo) & (_d2 <= hi) & INFO["any_op"][_conf]).sum(1).mean()) for lo, hi in DIST_BINS]
DIST_N_NONE = [float(((_d2 >= lo) & (_d2 <= hi) & ~INFO["any_op"][_conf]).sum(1).mean()) for lo, hi in DIST_BINS]


def ceiling_by(key: str, n: int) -> list[float]:
    return [float(HO_CEILING[INFO[key] == i].mean()) for i in range(n)]


CEIL_BAND = ceiling_by("band", len(BANDS) - 1)
CEIL_NOISE = ceiling_by("noise", ex.CENTRE[0] + 1)
N_BAND = [int((INFO["band"] == i).sum()) for i in range(len(BANDS) - 1)]
N_NOISE = [int((INFO["noise"] == k).sum()) for k in range(ex.CENTRE[0] + 1)]
_hsv_sel = (OP_IDS == OPS.index("hsvmix")) & (INFO["post_true"] > 0.9) & ~INFO["gray"]
_gap_bin = np.digitize(INFO["hue_gap"], HUE_GAPS) - 1
CEIL_GAP = [float(HO_CEILING[_hsv_sel & (_gap_bin == i)].mean()) for i in range(len(HUE_GAPS) - 1)]


def mean_over_runs(key: str) -> np.ndarray:
    return np.mean([SUMMARY[lbl][key] for lbl in SCORED], axis=0)


BAND_MEAN = mean_over_runs("by_band")
NOISE_MEAN = mean_over_runs("by_noise")
GAP_MEAN = mean_over_runs("by_gap")
OFF_OTHER = mean_over_runs("off_other")
OFF_NONE = mean_over_runs("off_none")
_off = PREDICTIVE < SUPPORT
UNIFORM_SHARE = np.array(
    [
        ((_off & INFO["any_op"])[_conf & (OP_IDS == o)].sum(1) / _off[_conf & (OP_IDS == o)].sum(1)).mean()
        for o in range(X.N_OPS)
    ]
)
OTHER_SHARE = OFF_OTHER / (OFF_OTHER + OFF_NONE)

# The leak on other ops' answers against the final score, across the scored runs.
_leak = np.array([np.mean(SUMMARY[lbl]["off_other"]) for lbl in SCORED])
_final = np.array([SUMMARY[lbl]["eem"] for lbl in SCORED])
LEAK_R = float(np.corrcoef(_leak, _final)[0, 1])


# --- Figures ------------------------------------------------------------------------------------------------


def lr_curve(label: str) -> tuple[np.ndarray, np.ndarray]:
    """The learning rate a run followed, from its schedule: fine over the warm-up and coarser after it."""
    row = RUNS[label]
    total = row["epochs"] * EPOCH_LENGTH
    config = SchedulerConfig(
        epochs=row["epochs"], warmup_epochs=row["warmup_epochs"], min_lr_factor=0.01, lr_sheet=row["lr_sheet"]
    )
    schedule = configure_schedule(config, row["peak_lr"], EPOCH_LENGTH)
    steps = np.unique(np.r_[np.linspace(0, total * 0.03, 200), np.linspace(0, total, 1200)].astype(int))
    return steps, np.asarray(schedule(steps), dtype=float)


def group_lines_draw(ax_top: np.ndarray, ax_lr: np.ndarray, lines: list[tuple], k: int = 3) -> None:
    def smooth(y: np.ndarray) -> np.ndarray:
        pad = np.pad(y, (k // 2, k // 2), mode="edge")
        return np.convolve(pad, np.ones(k) / k, mode="valid")

    for ax, (name, group) in zip(ax_top, GROUPS, strict=True):
        idx = list(range(len(OPS))) if group is None else [OPS.index(o) for o in group]
        ax.axhline(CEILING if group is None else CEILING_PER_OP[idx].mean(), ls="--", color=rule_color(), lw=1)
        ax.set_title(name, fontsize=9)
        for label, color, ls, lw, _ in lines:
            t = TRAJ[label]
            y = np.array(t["eem"]) if group is None else np.array(t["eem_per_op"])[:, idx].mean(1)
            ax.plot(np.array(t["step"]) / 1e3, smooth(y), color=color, ls=ls, lw=lw)
    for label, color, ls, lw, legend in lines:
        steps, lr = lr_curve(label)
        for i, ax in enumerate(ax_lr):
            ax.plot(steps / 1e3, lr, color=color, ls=ls, lw=lw, label=legend if i == 0 else f"_{legend}")


@memo
def groups_draw(lines: list[tuple], name: str, caption: str, alt_text: str) -> str:
    @themed(name=name, alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig = plt.figure(figsize=(8.4, 6.8), layout="constrained")
        grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.6])
        top = [fig.add_subplot(grid[i // 3, i % 3]) for i in range(len(GROUPS))]
        for ax in top[1:]:
            ax.sharex(top[0])
            ax.sharey(top[0])
        lr = [fig.add_subplot(grid[2, i], sharex=top[0]) for i in range(3)]
        for ax in lr[1:]:
            ax.sharey(lr[0])
        group_lines_draw(np.array(top), np.array(lr), lines)
        top[0].set_ylim(0, 0.85)
        for ax in top[::3]:
            ax.set_ylabel("held-out EEM")
        lr[0].set_yscale("log")
        lr[0].set_ylim(1e-5, 1.5e-2)
        lr[0].set_ylabel("learning rate")
        for ax in lr:
            ax.set_xlabel("step (thousands)", fontsize=8)
        handles, labels = lr[0].get_legend_handles_labels()
        keep = [(h, lbl) for h, lbl in zip(handles, labels, strict=True) if not lbl.startswith("_")]
        fig.legend(
            *zip(*keep, strict=True), loc="outside upper center", ncols=min(len(keep), 4), frameon=False, fontsize=7
        )
        return fig

    return _plot()


@memo
def conditions_draw(
    conditions: list[tuple[str, list[str]]], values: dict, ceiling: float, passing: float, alt_text: str
) -> str:
    @themed(
        name="conditions",
        alt_text=alt_text,
        caption=f"""
            **The final score of every masked condition at peak {PEAK:g}.** Each column is one condition; faded dots
            are its seeds, the bar their range, and the diamond their mean. Left, held-out expected exact match, with
            the Bayes ceiling as a solid rule and the pass line as a dashed rule over the hatched failing side. Right,
            the calibration KL (lower is better calibrated).
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), layout="constrained")
        rng = np.random.default_rng(0)
        for ax, key in zip(axes, ("eem", "kl"), strict=True):
            for i, (_, labels) in enumerate(conditions):
                v = np.array([values[lbl][key] for lbl in labels])
                c = light_dark("#1f5fa8", "#7fb2ff")
                ax.plot([i, i], [v.min(), v.max()], color=c, lw=6, alpha=0.25, solid_capstyle="butt", zorder=1)
                ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=c, alpha=0.5, lw=0, zorder=2)
                ax.scatter([i], [v.mean()], s=40, marker="D", color=c, zorder=3)
            ax.set_xticks(range(len(conditions)), [n for n, _ in conditions], rotation=40, ha="right", fontsize=7)
            ax.set_xlim(-0.6, len(conditions) - 0.4)
        axes[0].set_ylim(0.40, 0.53)
        axes[0].axhline(ceiling, color=rule_color(), lw=1)
        gate(axes[0], passing)
        axes[0].set_ylabel("held-out EEM")
        axes[1].set_ylim(0, 1.0)
        axes[1].set_ylabel("calibration KL (nats)")
        return fig

    return _plot()


@memo
def gap_draw(summary: dict, ceil_band: list, ceil_noise: list, alt_text: str) -> str:
    @themed(
        name="gap",
        alt_text=alt_text,
        caption=f"""
            **Where the model falls short of the ceiling.** Held-out expected exact match of the {len(summary)} scored
            runs (thin lines) and their mean (thick), with the Bayes ceiling dashed. Left, contexts grouped by the
            posterior on the true op given their examples; right, by how many of their {ex.CENTRE[0]} examples carry
            replacement op noise.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained", sharey=True)
        c = light_dark("#1f5fa8", "#7fb2ff")
        for ax, key, ceil, ticks in (
            (axes[0], "by_band", ceil_band, BAND_LABELS),
            (axes[1], "by_noise", ceil_noise, [str(k) for k in range(len(ceil_noise))]),
        ):
            x = np.arange(len(ceil))
            for s in summary.values():
                ax.plot(x, s[key], color=c, lw=0.6, alpha=0.35)
            ax.plot(x, np.mean([s[key] for s in summary.values()], axis=0), color=c, lw=2, marker="o", ms=4)
            ax.plot(x, ceil, ls="--", color=rule_color(), lw=1.2, marker="_", ms=10)
            ax.set_xticks(x, ticks)
        axes[0].set_xlabel("posterior on the true op")
        axes[1].set_xlabel("noisy examples in the context")
        axes[0].set_ylabel("held-out EEM")
        axes[0].set_ylim(0, 0.8)
        return fig

    return _plot()


def cube_data(label: str, n: int = 90, seed: int = 0) -> dict:
    """On the confident contexts of each op, a fixed sample: the expected answer color under the model and under the
    Bayes predictive (cube units, 0 to 1). And the same for hsvmix, by hue gap of the operands.
    """
    rng = np.random.default_rng(seed)
    p = ANSWERS[label].astype(float)
    em = (p / p.sum(1, keepdims=True)) @ RGB
    eq = PREDICTIVE @ RGB
    per_op = {}
    for o, name in enumerate(OPS):
        idx = np.flatnonzero((OP_IDS == o) & (INFO["post_true"] > CONFIDENT))
        idx = rng.choice(idx, min(n, len(idx)), replace=False)
        per_op[name] = (em[idx], eq[idx])
    per_gap = []
    for i in range(len(HUE_GAPS) - 1):
        idx = np.flatnonzero(_hsv_sel & (_gap_bin == i))
        idx = rng.choice(idx, min(n, len(idx)), replace=False)
        per_gap.append((em[idx], eq[idx]))
    return {"per_op": per_op, "per_gap": per_gap}


@memo
def cubes_draw(per_op: dict, alt_text: str) -> str:
    @themed(
        name="answer-cubes",
        alt_text=alt_text,
        caption=f"""
            **Model answers against Bayes answers, per op.** Each panel is the color cube seen down its gray
            diagonal (white at the center). Each dot is one confident held-out context (posterior on the true op
            above {CONFIDENT:g}) of `{CUBE_RUN}`, at the expected answer color under the model, and its ring is the
            expected answer color under the Bayes predictive, joined by a stub. Up to 90 contexts per op.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 4, figsize=(8.4, 6.6), layout="constrained")
        flat = cast(np.ndarray, axes).ravel()
        for ax, (name, (em, eq)) in zip(flat, per_op.items(), strict=False):
            plot_rgb_cube(ax, em, truth=eq, s=7, view="wheel")
            ax.set_title(name, fontsize=9)
        for ax in flat[len(per_op) :]:
            ax.set_axis_off()
        return fig

    return _plot()


@memo
def hue_gap_draw(per_gap: list, eem_gap: list, ceil_gap: list, alt_text: str) -> str:
    @themed(
        name="hsvmix-hue-gap",
        alt_text=alt_text,
        caption="""
            **`hsvmix` by the hue gap between its operands.** As in the figure above, for `hsvmix` contexts with a
            posterior on the true op above 0.9 and two chromatic operands, grouped by how far apart their hues are.
            Under each panel: the expected exact match, mean over the scored runs, against the Bayes ceiling.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.6), layout="constrained")
        for i, ax in enumerate(axes):
            em, eq = per_gap[i]
            plot_rgb_cube(ax, em, truth=eq, s=7, view="wheel")
            ax.set_title(f"{HUE_GAPS[i]:.0f}°–{min(HUE_GAPS[i + 1], 180):.0f}°", fontsize=9)
            ax.text(0, -1.3, f"{eem_gap[i]:.2f} of {ceil_gap[i]:.2f}", ha="center", fontsize=8)
        return fig

    return _plot()


@memo
def leak_distance_draw(ratio: np.ndarray, ratio_box: np.ndarray, alt_text: str) -> str:
    @themed(
        name="leak-distance",
        alt_text=alt_text,
        caption=f"""
            **Other ops' answers against other colors at the same distance.** On confident contexts, off the support
            of the Bayes predictive: the mean mass per grid color on other ops' answers divided by the mean mass per
            grid color on colors no op gives, with colors grouped by their distance to the nearest support color.
            Thin lines are the {len(ratio)} scored runs, the thick line their mean. Left, all off-support colors;
            right, only those inside the box the two operands span. A ratio of 1 (dashed) would mean distance alone
            sets the mass.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained", sharey=True)
        c = light_dark("#1f5fa8", "#7fb2ff")
        x = np.arange(len(DIST_LABELS))
        for ax, r in zip(axes, (ratio, ratio_box), strict=True):
            for row in r:
                ax.plot(x, row, color=c, lw=0.6, alpha=0.35)
            ax.plot(x, r.mean(0), color=c, lw=2, marker="o", ms=4)
            ax.axhline(1, ls="--", color=rule_color(), lw=1)
            ax.set_xticks(x, DIST_LABELS)
            ax.set_xlabel("distance to the support (grid steps)")
        axes[0].set_yscale("log")
        axes[0].set_ylim(0.8, 80)
        axes[0].set_ylabel("mass ratio, other ops' answers\nagainst other colors")
        axes[0].set_title("all off-support colors", fontsize=9)
        axes[1].set_title("inside the operand box", fontsize=9)
        return fig

    return _plot()


@memo
def confusion_draw(m: np.ndarray, ops: tuple, alt_text: str) -> str:
    @themed(
        name="op-confusion",
        alt_text=alt_text,
        caption=f"""
            **Where the leak goes, by op.** On confident contexts, rows are the true op. Off the diagonal, each
            square is the mass the model puts on answers that the column op can give and the true op cannot, as a
            mean over the {len(SCORED)} scored runs; the color scale covers these entries alone. The diagonal, in
            text, is the mass on answers of the true op. The Bayes predictive puts under 0.002 in every off-diagonal
            square.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")
        n = len(ops)
        off = np.where(np.eye(n, dtype=bool), np.nan, m)
        cmap = plt.get_cmap(light_dark("Blues", "magma")).copy()
        cmap.set_bad(light_dark("#fff", "#111"))
        im = ax.imshow(off, cmap=cmap, vmin=0, vmax=float(np.nanmax(off)))
        vmax = float(np.nanmax(off))
        for i in range(n):
            for j in range(n):
                v = m[i, j]
                if i == j:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=rule_color())
                elif v >= 0.02:
                    dark_cell = (v / vmax > 0.55) == (light_dark(0, 1) == 0)
                    ax.text(
                        j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="#fff" if dark_cell else "#000"
                    )
        ax.set_xticks(range(n), ops, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n), ops, fontsize=7)
        ax.set_xlabel("answers of this op, beyond those of the true op")
        ax.set_ylabel("true op")
        fig.colorbar(im, ax=ax, shrink=0.8, label="mass")
        return fig

    return _plot()


# --- Numbers the prose quotes ----------------------------------------------------------------------------------

R1 = {arm: seed_stats(labels_of(arm)) for arm in ("low", "long", "long-low")}
R1_KL = {arm: seed_stats(labels_of(arm), kl) for arm in ("low", "long", "long-low")}
SWEEP = sorted((r for r in EVAL["runs"] if r["round"] == 2 and r["mask"]), key=lambda r: r["peak_lr"])
BEST_SWEEP = max(SWEEP, key=lambda r: r["task"]["eem"]["all"])
NOMASK = RUNS[f"sweep-nomask-{PEAK:g}-s0"]
MASKED = RUNS[f"sweep-{PEAK:g}-s0"]
FINDER_STEEP = {f["label"]: f["history"][0]["steepest_lr"] for f in FINDER["runs"]}
STEPS_8X = TRAJ[f"sweep-{PEAK:g}-s0"]["step"][-1]
STEPS_16X = TRAJ[f"sweep16x-{PEAK:g}-s0"]["step"][-1]

CONDITIONS: list[tuple[str, list[str]]] = [
    ("cosine 8×", labels_of(f"sweep-{PEAK:g}")),
    ("WSD 8×", labels_of(f"wsd-{PEAK:g}")),
    ("staircase 8×", labels_of(f"stairs-{PEAK:g}")),
    ("cosine 16×", labels_of(f"sweep16x-{PEAK:g}")),
    ("WSD 16×", labels_of(f"wsd16x-{PEAK:g}")),
    ("staircase 16×", labels_of(f"stairs16x-{PEAK:g}")),
    ("d128-L4 8×", labels_of(f"d128-sweep-{PEAK:g}")),
    ("d64-L6 8×", labels_of(f"d64L6-sweep-{PEAK:g}")),
]
COND_EEM = {name: seed_stats(labels) for name, labels in CONDITIONS}
COND_KL = {name: seed_stats(labels, kl) for name, labels in CONDITIONS}
BEST = max(SCORED, key=eem)
SPREAD_8X = [eem(lbl) for name, labels in CONDITIONS[:3] for lbl in labels]
HSVMIX = OPS.index("hsvmix")
BAND_GAP = [c - e for c, e in zip(CEIL_BAND, BAND_MEAN, strict=True)]
TOTAL_GAP = CEILING - float(np.mean([SUMMARY[lbl]["eem"] for lbl in SCORED]))
# How much of the whole gap each band carries: its share of contexts times its gap.
BAND_SHARE = [n * g / sum(N_BAND) / TOTAL_GAP for n, g in zip(N_BAND, BAND_GAP, strict=True)]

OFF_TOTAL = OFF_OTHER + OFF_NONE
LIGHTEN, DARKEN = OPS.index("lighten"), OPS.index("darken")
WIDE, DEEP = f"d128-sweep-{PEAK:g}-s0", f"d64L6-sweep-{PEAK:g}-s0"
# How often the top answer of the model is the top answer of the Bayes predictive, on confident contexts of an op.
TOP_MATCH = {
    o: float(
        np.mean(
            [
                (ANSWERS[lbl][_conf & (OP_IDS == o)].argmax(1) == PREDICTIVE[_conf & (OP_IDS == o)].argmax(1)).mean()
                for lbl in SCORED
            ]
        )
    )
    for o in (LIGHTEN, DARKEN)
}


# The op confusion between two named ops: mass on answers of *b* beyond those of *a*, on confident contexts of *a*.
def conf(a: str, b: str, m: np.ndarray | None = None) -> float:
    return float((CONFUSION if m is None else m)[OPS.index(a), OPS.index(b)])


# How much more mass a color no op gives gets inside the operand box than over all off-support colors.
BOX_LIFT = float(np.mean([np.array(v["none_box"]) / np.array(v["none"]) for v in LEAK_DIST.values()]))


@memo
def op_confusion(p16: np.ndarray, op: np.ndarray, pair: np.ndarray, exclusive: bool = True) -> np.ndarray:
    """On the given contexts, where the answer distribution *p16* puts its mass, by op. Rows are the true op. The
    diagonal is the mass on the colors the true op can give for the query operands; off the diagonal, the mass on
    the colors op *o* can give and the true op cannot. Ops share answers with each other, so the off-diagonal
    entries of a row can overlap. With *exclusive* off, an off-diagonal entry counts every color op *o* can give,
    so on the Bayes predictive it is how often the answers of the two ops coincide.
    """
    table = answer_table()
    p = p16.astype(float)
    rows = np.arange(len(p))
    gives = np.zeros((X.N_OPS, *p.shape), dtype=bool)
    for o in range(X.N_OPS):
        idx, ok = table.idx[o, pair], table.prob[o, pair] > 0
        for j in range(idx.shape[1]):
            gives[o, rows[ok[:, j]], idx[ok[:, j], j]] = True
    true = gives[op, rows]
    beyond = gives & ~true if exclusive else gives
    per_col = np.stack([(p * np.where((op == o)[:, None], true, beyond[o])).sum(1) for o in range(X.N_OPS)], 1)
    return np.stack([per_col[op == o].mean(0) for o in range(X.N_OPS)])


# The op confusion on confident contexts, for the model (mean over scored runs) and for the Bayes predictive, which
# on these contexts puts nearly all its mass on the answers of the true op.
_pair = DETAIL["query_pair"]
CONFUSION = np.mean([op_confusion(ANSWERS[lbl][_conf], OP_IDS[_conf], _pair[_conf]) for lbl in SCORED], axis=0)
CONFUSION_BAYES = op_confusion(PREDICTIVE[_conf], OP_IDS[_conf], _pair[_conf])
OVERLAP = op_confusion(PREDICTIVE[_conf], OP_IDS[_conf], _pair[_conf], exclusive=False)

rf"""
# Ex 2.2.17: the center control plateau, a scout

/// tip |
<!-- tl;dr -->
More steps, a newline mask, and a lower learning rate took ex-2.2.16's center control most of the way to its pass line, and then it leveled off. It matches the Bayes ceiling where the examples leave the op uncertain, and falls short where they settle it, keeping some of its mass on answers that the ruled-out ops would give.
///

This scout set out to lift ex-2.2.16's center control (the unanchored d64-L4 model on the corpus condition `k3-r0.3`) to its Bayes ceiling of {CEILING:.3f}. Ex-2.2.16 trained it for 50 epochs at a peak learning rate of 0.01, and it reached 0.27. More steps, a newline mask, and a lower rate took it to about 0.45. Past that, neither the shape of the schedule, nor twice the steps, nor a wider or deeper model changed the final score by more than the spread between seeds. The best run, `{BEST}`, reached {eem(BEST):.3f}.

To see where the rest of the gap sits, we scored the answer distribution of each model against the Bayes predictive, context by context. Part of the gap is in how firmly the model commits to the op its examples support, and `hsvmix` adds a separate shortfall.

## Observations

Each line is a measurement on the runs of this scout, with no gate.

- **E1** [More steps, a lower rate, and the mask](#more-steps-a-lower-rate-and-the-mask-e1): more steps and the newline mask made most of the gain, and a range of peak rates did about equally well.
- **E2** [Schedule, length, and model size](#schedule-length-and-model-size-e2): past that, neither the shape of the schedule, nor more steps, nor a wider or deeper model moved the final score.
- **E3** [Where the gap sits](#where-the-gap-sits-e3): the model matches the ceiling on contexts whose examples leave the op uncertain, and falls short on those that point to one op.
- **E4** [The leak onto other ops](#the-leak-onto-other-ops-e4): where the examples settle the op, the model keeps some of its mass on answers of other ops, mostly the op most like the true one, and more than nearness to the true answer accounts for.
- **E5** [Answers in the color cube](#answers-in-the-color-cube-e5): for ten of the eleven ops, the expected answer of the model sits close to the Bayes one, with no shared direction to the difference. `hsvmix` is the exception.
- **E6** [`hsvmix` and the hue gap](#hsvmix-and-the-hue-gap-e6): `hsvmix` falls further short of its ceiling the further apart the operand hues are, and neither width nor depth changed that.

## Scope

This is a scout, run in rounds, each designed on the results of the one before. There is no preregistration and no gate. The pass line that ex-2.2.16 set for its controls, {X.CEILING_MARGIN:g} below the ceiling at {PASS:.3f}, is a reference point here. Most conditions after round 1 have one seed, and the three-seed conditions span about 0.03, so a difference smaller than that is not a change.

## Why

The anchoring experiments on the in-context grammar compare an anchored model with an unanchored control on the same contexts. At 0.27 against a ceiling of {CEILING:.2f}, ex-2.2.16's center control had learned about half of what its examples allow, so that comparison would mix the effect of the anchor with the unfinished training of both models. We wanted a control recipe that comes close to the ceiling, or else to know where it levels off and why.

## The runs

Every run is the unanchored model of ex-2.2.16 on `k3-r0.3`, with an untied readout. Lengths are multiples of the 50 epochs ({X.EPOCHS * EPOCH_LENGTH:,} steps) that ex-2.2.16 trained for.

"""

table_html(
    ["round", "what changes", "length", "peak LR", "mask", "seeds"],
    [
        [
            "1",
            "`low`, `long`, and `long-low`: a lower rate, more steps, or both",
            "1×, 3×, 3×",
            "0.003, 0.01, 0.003",
            "no",
            "3",
        ],
        [
            "2",
            "a sweep of the peak rate, and one run without the mask",
            "8×",
            f"{SWEEP[0]['peak_lr']:g} to {SWEEP[-1]['peak_lr']:g}",
            "yes",
            "1",
        ],
        ["3", "warmup-stable-decay schedule, and two more seeds of the cosine", "8×", f"{PEAK:g}", "yes", "3"],
        ["4", "staircase schedule", "8×", f"{PEAK:g}", "yes", "3"],
        ["5", "each of the three schedules", "16×", f"{PEAK:g}", "yes", "1"],
        ["6", "a wider model, d128-L4", "8×", f"{PEAK:g}", "yes", "1"],
        ["7", "a deeper model, d64-L6", "8×", f"{PEAK:g}", "yes", "1"],
    ],
    "**The rounds.** Length is a multiple of the steps of ex-2.2.16. The schedule is the cosine unless the row says otherwise.",
    text_cols=2,
)

rf"""

The newline mask stops each position from attending past the start of its own line. Every schedule starts with a linear warm-up from 1% of the peak rate: over the first tenth of the run in round 1, and over 1,320 steps (5 epochs, ex-2.2.16's own warm-up) from round 2 on, whatever the length. The cosine then anneals to 1% of the peak by the end of the run.

## The measurements

Held-out *expected exact match* (EEM) is the probability that an answer drawn from the distribution of the model at the query `=` is a correct answer of the true op. We average it over {len(OP_IDS):,} held-out contexts, {len(OP_IDS) // X.N_OPS:,} per op.

The *Bayes ceiling* is the same score for an ideal predictor. That predictor weighs each op by how well it explains the examples of a context, then answers with the resulting mixture of ops; we call that mixture the *Bayes predictive*. The ceiling is below 1 because some ops round stochastically, and noisy examples leave the op uncertain.

The *calibration KL* is the KL divergence[^kl] from the Bayes predictive to the distribution of the model, in nats. It measures how much more the model loses on the answer than the ideal predictor does. Zero means the model holds the distribution the examples support.

[^kl]: A KL divergence measures how different one probability distribution is from another. It is zero when they match and grows as they differ.

The *posterior on the true op* is the probability the ideal predictor gives the true op after seeing the examples. Replacement op noise lowers it. We call a context *confident* when that posterior is above {CONFIDENT:g}.

The *support* of the Bayes predictive on a context is the set of grid colors to which it gives at least {SUPPORT:g}. Probability mass on any other color is *off support*. An off-support color is *another op's answer* when some op, applied to the query operands, can give it. Distances are on the color grid, in grid steps: one step is the spacing between neighboring levels of a channel, and the far corner of the cube is 5√3 ≈ 8.7 steps from black.

## More steps, a lower rate, and the mask (E1)

Round 1 tested two ideas at three seeds each, without the newline mask: that the peak rate of 0.01 was too high, or that the model needed more steps. The `low` arm trained for the length of ex-2.2.16 at 0.003, `long` for three times the length at 0.01, and `long-low` did both. Neither change alone did much, and the two together did the most.

"""

table_html(
    ["arm", "length", "peak LR", "EEM, seed mean (range)", "KL, seed mean"],
    [
        [
            f"`{arm}`",
            f"{mult}×",
            f"{lr:g}",
            f"{R1[arm][0]:.3f} ({R1[arm][1]:.3f}–{R1[arm][2]:.3f})",
            f"{R1_KL[arm][0]:.2f}",
        ]
        for arm, mult, lr in (("low", 1, 0.003), ("long", 3, 0.01), ("long-low", 3, 0.003))
    ],
    "**Round 1.** Three seeds per arm, unmasked, cosine schedule. Ex-2.2.16's control (1×, 0.01) scored 0.27.",
)

rf"""

Every round-1 curve was still rising when the cosine schedule wound down. So round 2 trained for eight times the length ({STEPS_8X:,.0f} steps), with the mask. We ran a learning-rate finder on the same configuration. It trains briefly at a rising rate and looks for where the loss falls fastest; here, the steepest descent was near {FINDER_STEEP["finder-mask"]:.4f}. Round 2 then swept the peak rate from {SWEEP[-1]["peak_lr"]:g} down to {SWEEP[0]["peak_lr"]:g} at one seed each, with one more run at {PEAK:g} without the mask. The figure below follows each run through training, one panel per group of ops, with the learning rate it followed underneath.

"""

groups_draw(
    [
        (r["label"], plt.get_cmap("viridis")(i / (len(SWEEP) - 1)), "-", 1.3, f"{r['peak_lr']:g}")
        for i, r in enumerate(SWEEP)
    ]
    + [(NOMASK["label"], "0.5", "--", 1.6, f"{PEAK:g}, no mask")],
    "round2",
    f"""
        **Round 2: the peak learning rate at eight times the length.** Each panel of the top two rows is one group of
        ops; the y-axis is held-out expected exact match on a probe set during training (a running mean over three
        points), with the Bayes ceiling of the group dashed. One seed per peak rate, shaded by rate. The gray dashed
        curve is the unmasked run at {PEAK:g}. The bottom row is the learning rate each run followed, repeated under
        each column; the near-vertical rise at the left edge is the warm-up.
    """,
    f"""
        Six line charts of held-out expected exact match against training step, one per op group, and a learning-rate
        chart under each column. The runs with peak rates from 0.00178 to 0.01 all end between 0.41 and 0.45 overall;
        the lower rates end lower. The HSV-channel panel shows sharp rises partway through training, earlier for
        mid-range rates. The unmasked run ends lowest of the mid-range rates, at {NOMASK["task"]["eem"]["all"]:.2f}.
        Each learning rate rises steeply over the first 1,320 steps, then decays along a cosine.
    """,
)

rf"""

Four peak rates from 0.00178 to 0.01 ended within 0.04 of each other, and the best was {BEST_SWEEP["peak_lr"]:g} at {BEST_SWEEP["task"]["eem"]["all"]:.3f}. The mask made the largest single difference in the scout: at {PEAK:g}, the unmasked run scored {NOMASK["task"]["eem"]["all"]:.3f} and the masked one {MASKED["task"]["eem"]["all"]:.3f}. The HSV-channel ops rose steeply once the decaying rate fell below about 0.004, which shaped the next two rounds. From round 3 on, every run trains with the mask at a peak rate of {PEAK:g}, the middle of the good range.

## Schedule, length, and model size (E2)

Round 3 tried a warmup-stable-decay (WSD) schedule, which warms up to {PEAK:g}, holds near {ex.HOLD_LR:g} until 85% of the run, and then anneals to the same floor as the cosine. Round 4 tried a staircase: holds at {", ".join(f"{s:g}" for s in ex.STAIRS)}, with short anneals between them, the last hold covering the final 11% of the run. Both ran at three seeds, beside two more seeds of the cosine.

"""

groups_draw(
    [
        (f"{arm}-{PEAK:g}-s{s}", color, "-", 1.2, legend if s == 0 else f"_{legend}")
        for s in range(3)
        for arm, color, legend in (
            ("sweep", "C0", "cosine"),
            ("wsd", "C3", "warmup-stable-decay"),
            ("stairs", "C2", "staircase"),
        )
    ],
    "schedules",
    f"""
        **Rounds 3 and 4: three schedules at eight times the length.** As in the round-2 figure, with three seeds per
        schedule, all masked, at a peak rate of {PEAK:g}.
    """,
    """
        Six line charts of held-out expected exact match against training step, one per op group, and the three
        learning-rate schedules under each column. The warmup-stable-decay and staircase runs are flat through their
        holds and rise at each drop in the rate. By the end, the three schedules sit on top of one another in every
        group, and the seeds differ more than the schedules.
    """,
)

rf"""

The schedules change the path and leave the destination in place. WSD curves stay flat through the hold and rise steeply in the final anneal; the staircase gains most at its first two drops, and almost nothing at the lowest rate. All three end together, and the seeds differ more than the schedules do.

Round 5 trained each schedule for sixteen times the length ({STEPS_16X:,.0f} steps) at one seed. Rounds 6 and 7 trained the cosine at eight times on a wider model (d128-L4) and a deeper one (d64-L6), one seed each, both at seed 0 so that they pair with the other seed-0 runs. The figure below gathers the final scores of all of these.

"""

conditions_draw(
    CONDITIONS,
    {lbl: {"eem": eem(lbl), "kl": kl(lbl)} for _, labels in CONDITIONS for lbl in labels},
    CEILING,
    PASS,
    f"""
        Two dot charts over eight conditions. Left, expected exact match: every condition sits between 0.43 and 0.47,
        below the dashed pass line at {PASS:.3f} and the solid ceiling at {CEILING:.3f}. Right, the calibration KL:
        between 0.36 and 0.56 for every condition except the wider d128-L4 model, at {kl(WIDE):.2f}.
    """,
)

rf"""

At eight times the length, the nine runs of the three schedules span {min(SPREAD_8X):.3f} to {max(SPREAD_8X):.3f}. Sixteen times the length adds between 0.002 and 0.006 to the seed-0 run of each schedule. The wider and deeper models gain 0.023 and 0.017 on their seed-0 pair, but seed 0 is the weakest seed under every schedule, and the plain model at seed 1 reaches {eem(f"sweep-{PEAK:g}-s1"):.3f}.

Two things about the wider model stand apart from its final score. It learns faster: its HSV channels make their jump near 15k steps, against 25k to 45k for the d64 runs, and it leads through the first half of training. And it is much less well calibrated, at a KL of {kl(WIDE):.2f} against about 0.5, with a lower training loss than the d64 runs. So it is about as accurate as the others while being more confident than the examples support, which fits a model that has fitted more of its training corpus.

"""

groups_draw(
    [
        (f"sweep-{PEAK:g}-s1", "C0", ":", 0.9, "d64-L4, seeds 1 and 2"),
        (f"sweep-{PEAK:g}-s2", "C0", ":", 0.9, "_d64-L4, seeds 1 and 2"),
        (f"sweep-{PEAK:g}-s0", "C0", "-", 1.6, "d64-L4"),
        (WIDE, "C1", "-", 1.6, "d128-L4 (wider)"),
        (DEEP, "C4", "-", 1.6, "d64-L6 (deeper)"),
    ],
    "model-size",
    f"""
        **Rounds 6 and 7: a wider and a deeper model.** As in the round-2 figure, for the cosine at eight times the
        length and a peak rate of {PEAK:g}, seed 0 of each model (solid). The dotted curves are seeds 1 and 2 of the
        d64-L4 model, for the seed spread.
    """,
    """
        Six line charts of held-out expected exact match against training step, one per op group, and the shared
        cosine schedule under each column. The wider model rises fastest early, most visibly in the HSV-channel panel,
        and all three models end close together, within the range of the d64-L4 seeds.
    """,
)

rf"""

## Where the gap sits (E3)

To see where the rest of the gap sits, we scored the final answer distribution of every masked run at {PEAK:g}, context by context (see [Method](#method)). The figure below groups the held-out contexts two ways: by the posterior on the true op, and by the number of examples that carry replacement op noise.

"""

gap_draw(
    SUMMARY,
    CEIL_BAND,
    CEIL_NOISE,
    f"""
        Two line charts of expected exact match, with the Bayes ceiling dashed. Left, against four bands of the
        posterior on the true op: the runs sit on the ceiling in the lowest band ({BAND_MEAN[0]:.2f} against
        {CEIL_BAND[0]:.2f}) and below it in the other three, by {BAND_GAP[1]:.2f}, {BAND_GAP[2]:.2f}, and
        {BAND_GAP[3]:.2f}. Right, against the number of noisy examples: below the ceiling with none or one, and on
        it with two or three. The {len(SCORED)} runs lie close together throughout.
    """,
)

rf"""

Where the examples leave the op uncertain (a posterior below 0.5, or two or more noisy examples out of three), the model scores what the ideal predictor scores. The gap opens once the examples point to one op: {BAND_GAP[1]:.3f} in the band from 0.5 to 0.9 and {BAND_GAP[2]:.3f} from 0.9 to 0.99, and it narrows to {BAND_GAP[3]:.3f} where the posterior is above 0.99. Those two middle bands hold {sum(N_BAND[1:3]) / sum(N_BAND):.0%} of the contexts and {sum(BAND_SHARE[1:3]):.0%} of the gap. All {len(SCORED)} runs share this shape, whatever their schedule, length, or model.

## The leak onto other ops (E4)

A model that knew the true op would lose nothing to op uncertainty. So on confident contexts, what remains of the gap is about how the model computes each op, or about the model spreading its mass over ops the examples have ruled out. Where the model puts its off-support mass tells the two apart. The table below splits that mass by whether another op would give the color on the same query.

"""

table_html(
    [
        "op",
        "ceiling, all contexts",
        "ceiling, confident",
        "mass off support",
        "on other ops' answers",
        "on no op's answer",
        "share on other ops",
        "share if spread evenly",
    ],
    [
        [
            f"`{name}`",
            f"{CEILING_PER_OP[o]:.2f}",
            f"{HO_CEILING[(OP_IDS == o) & _conf].mean():.2f}",
            f"{OFF_TOTAL[o]:.3f}",
            f"{OFF_OTHER[o]:.3f}",
            f"{OFF_NONE[o]:.3f}",
            f"{OTHER_SHARE[o]:.0%}",
            f"{UNIFORM_SHARE[o]:.0%}",
        ]
        for o, name in enumerate(OPS)
    ],
    f"""
        **The mass each op leaks on confident contexts.** The Bayes ceiling of each op over all its held-out contexts,
        and over its confident contexts (posterior on the true op above {CONFIDENT:g}). The mass columns are on the
        confident contexts, as means over the {len(SCORED)} scored runs. "Share if spread evenly" is the share that
        would land on other ops' answers if the off-support mass were spread evenly over the off-support colors.
    """,
)

rf"""

On every op, the model keeps between {OFF_TOTAL.min():.0%} and {OFF_TOTAL.max():.0%} of its mass off the support, and more than half of that sits on answers another op would give, where an even spread would put under a tenth.

`lighten` and `darken` are the clearest case. Each gives one answer per query, with no rounding, and on a confident context the examples have ruled out the other ops, so the Bayes predictive puts all its mass on that answer and the ceiling is 1. Over all their contexts, where noisy examples often leave the op uncertain, their ceilings are {CEILING_PER_OP[LIGHTEN]:.2f} and {CEILING_PER_OP[DARKEN]:.2f}. On their confident contexts the top answer of the model is the Bayes answer {min(TOP_MATCH.values()):.0%} of the time or more, and still about {OFF_OTHER[LIGHTEN]:.0%} of its mass goes to other ops' answers.

Other ops' answers are often near the true answer in the cube, and some ops give similar answers (`lighten` and `screen` both brighten), so part of this mass could be near misses. To separate the two, we compare colors at the same distance from the support. If nearness alone set the mass, a color that is another op's answer would get the same mass as one that no op gives at that distance. We also repeat the comparison inside the box the two operands span, since several ops give answers between their operands.

"""

leak_distance_draw(
    DIST_RATIO,
    DIST_RATIO_BOX,
    f"""
        Two line charts of a mass ratio on a log scale against distance to the support, from 1 to over 3 grid steps.
        Left, all off-support colors: the mean ratio is {DIST_RATIO.mean(0)[0]:.1f} at one step and rises to
        {DIST_RATIO.mean(0)[-1]:.0f} beyond three steps. Right, inside the operand box: {DIST_RATIO_BOX.mean(0)[0]:.1f}
        at one step, and {DIST_RATIO_BOX.mean(0).max():.0f} at most. Every run lies above the dashed line at 1 at every
        distance.
    """,
)

rf"""

At every distance, the colors that are other ops' answers get more mass: {DIST_RATIO.mean(0)[0]:.1f} times as much one step from the support, and more further out. Nearness does shape the leak, since {DIST_SHARE[0]:.0%} of the mass on other ops' answers sits one step from the support. The operand box also matters: at the same distance, a color no op gives gets {BOX_LIFT:.1f} times as much mass when it lies inside the box as off-support colors do on average. But inside the box, other ops' answers still get {DIST_RATIO_BOX.mean(0).min():.0f} to {DIST_RATIO_BOX.mean(0).max():.0f} times the mass of other colors at the same distance.

So the model favors the answers of the ops its examples should have ruled out, beyond what nearness or the operands account for. The runs that leak more also score lower: across the {len(SCORED)} runs, the mean leak on other ops' answers correlates with the final score at r = {LEAK_R:.2f}.

To see which ops the leak goes to, the figure below breaks it down by op. For each true op, it counts the mass on answers that another op can give and the true op cannot.

"""

confusion_draw(
    CONFUSION,
    OPS,
    f"""
        A heatmap of eleven true ops against eleven ops, with the diagonal left blank and labelled in text, from
        {np.diag(CONFUSION).min():.2f} for hsvmix to {np.diag(CONFUSION).max():.2f}. Most off-diagonal squares are pale,
        near 0.01. Five stand out: lighten onto screen ({conf("lighten", "screen"):.2f}), hsvmix onto mix
        ({conf("hsvmix", "mix"):.2f}), darken onto multiply ({conf("darken", "multiply"):.2f}), mix onto hsvmix
        ({conf("mix", "hsvmix"):.2f}), and difference onto exclusion ({conf("difference", "exclusion"):.2f}).
    """,
)

rf"""

Most of the leak goes to one other op, the one most like the true op. On confident contexts, `lighten` puts {conf("lighten", "screen"):.2f} of its mass on answers only `screen` gives, and `darken` puts {conf("darken", "multiply"):.2f} on answers only `multiply` gives. `hsvmix` and `mix` leak onto each other ({conf("hsvmix", "mix"):.2f} and {conf("mix", "hsvmix"):.2f}), and `difference` onto `exclusion` ({conf("difference", "exclusion"):.2f}). For every op, the op it leaks onto most is the one whose answers most often coincide with its own: the answer of a `lighten` query is one that `screen` can give on {conf("lighten", "screen", OVERLAP):.0%} of confident contexts, and for `darken` and `multiply` the figure is {conf("darken", "multiply", OVERLAP):.0%}. The leak runs mostly one way, from `lighten` to `screen` and from `darken` to `multiply` ({conf("screen", "lighten"):.2f} and {conf("multiply", "darken"):.2f} in the other direction).

So where two ops agree on most queries, the model keeps part of its mass on the answers of the other op even after the examples have told them apart.

## Answers in the color cube (E5)

The cubes below put the answers of the model beside the Bayes answers, one panel per op, on the confident contexts of `{CUBE_RUN}`. Each context is drawn at its expected answer color: the mean color of the answer distribution, which moves toward the middle of the cube as mass spreads.

"""

cubes_draw(
    cube_data(CUBE_RUN)["per_op"],
    """
        Eleven color-cube panels, one per op, each seen down the gray diagonal. Dots (model) sit on or very near
        their rings (Bayes) for every op except hsvmix, whose stubs are visibly longer and scattered in direction,
        with a slight pull toward the center of the cube. The deterministic ops (lighten, darken, difference) show
        almost no displacement.
    """,
)

rf"""

None of the stubs is long. For ten of the eleven ops, the expected answer of the model sits within a fraction of a grid step of the Bayes one, with no shared direction to the difference. So the model has learned these ops well, and the leak above is spread thinly around the cube. `hsvmix` is the exception, as the ops table shows: it keeps {OFF_TOTAL[HSVMIX]:.0%} of its mass off support on confident contexts, and {OFF_NONE[HSVMIX]:.0%} goes to colors no op gives. It is the one op where the computation itself falls short.

## `hsvmix` and the hue gap (E6)

`hsvmix` mixes two colors in hue, saturation, and value, with the hue taken around the color wheel. Grouping its contexts (posterior above 0.9, two chromatic operands) by how far apart the operand hues are shows where it struggles.

"""

hue_gap_draw(
    cube_data(CUBE_RUN)["per_gap"],
    list(GAP_MEAN),
    CEIL_GAP,
    f"""
        Four color-cube panels for hsvmix, by hue gap between the operands: 0 to 30, 30 to 90, 90 to 150, and 150 to
        180 degrees. The stubs lengthen as the gap grows. The score falls from {GAP_MEAN[0]:.2f} of a
        {CEIL_GAP[0]:.2f} ceiling at the smallest gap to {GAP_MEAN[3]:.2f} of {CEIL_GAP[3]:.2f} at the largest.
    """,
)

rf"""

With nearby hues the model gets {GAP_MEAN[0] / CEIL_GAP[0]:.0%} of what the ceiling allows; with near-opposite hues it gets {GAP_MEAN[3] / CEIL_GAP[3]:.0%}. Near-opposite hues are where the midpoint around the wheel moves furthest for a small change in either operand, and at exactly opposite hues the direction around the wheel is a tie. So `hsvmix` asks for a more precise computation than the other ops, and the model has not learned it. Width and depth did not help: `hsvmix` scored {RUNS[WIDE]["task"]["eem"]["per_op"][HSVMIX]:.2f} on d128-L4 and {RUNS[DEEP]["task"]["eem"]["per_op"][HSVMIX]:.2f} on d64-L6, against {RUNS[f"sweep-{PEAK:g}-s0"]["task"]["eem"]["per_op"][HSVMIX]:.2f} on their seed-0 pair.

## What we make of it

The control now sits at about 0.45 on the center condition, with the mask, a peak rate of {PEAK:g}, and eight times the steps of ex-2.2.16. That falls short of the pass line, but it *is* a model that does in-context inference over ops. It tracks the ceiling across the whole range of evidence, its answers land close to the Bayes answers on ten ops, and its shortfall is concentrated in how firmly it commits once the op is clear, plus one op it computes poorly. That seems a workable baseline for the anchoring experiments, which compare an anchored model with this control on the same contexts.

Four questions from this scout are on the backlog. Two are about the recipe: whether a curriculum over the replacement rate changes how firmly the model commits ([noise curriculum](/todo/science/noise-curriculum-center-control.md)), and whether the wider model with a tuned rate can reach the same level in half the steps ([a cheaper recipe](/todo/science/cheaper-center-control-recipe.md)); the wider model learned faster here, which is some encouragement for the second. Two are about the op set: whether to drop one op of each pair whose answers often coincide, since that is where most of the leak goes ([similar ops](/todo/science/drop-ops-with-similar-answers.md)), and whether `hsvmix` needs more training on hue, saturation, and value, or should leave the op set ([hsvmix](/todo/science/hsvmix-hue-precision.md)).

## Method

**Scoring.** For every held-out context we score the full answer distribution of the model at the query `=`, over the 216 color tokens, for the {len(SCORED)} masked runs at {PEAK:g}: every run of rounds 3 to 7 and the round-2 cosine. We then redraw the held-out contexts from their seed to recover which examples carry replacement op noise, which ex-2.2.16 did not publish; the redrawn contexts match the published ones token for token. The answer table, the posterior, and the Bayes predictive come from ex-2.2.16.

**Schedules.** WSD and the staircase are dopesheets: tables of the rate, as a multiple of the peak, at points through the run, stretched to its length. Each warms up over its first 1.25% (1,320 steps at eight times, twice that at sixteen). The learning-rate panels are drawn from the schedule of each run, which matches the rate logged during training.

**Distance to the support.** For each confident context and each off-support color, the grid distance to the nearest color in the support, binned at one step, √2, √3, 2 to √5, √6 to 3, and beyond. Within a bin, the mass per color is the mean over every pair of context and color in it. The operand box holds the grid colors that lie between the two operands on every channel.

**Op confusion.** For each confident context, the grid colors each op can give for the query operands. The diagonal counts the mass on the colors of the true op, and an off-diagonal square the mass on colors of the column op that the true op cannot give; the overlap quoted in the text counts every color of the column op, on the Bayes predictive.

**Cube figures.** Up to 90 contexts per op (or per hue-gap group), a fixed sample from `{CUBE_RUN}`; any run would do, since the runs agree closely on every summary above.

**Budget.** The scout cost about \$15 on Modal, nearly all of it L4 time for training. A step takes the same time on d64-L4 and d128-L4 (about 2,000 to 3,000 steps a minute), since at these sizes the step is bound by latency rather than arithmetic, so the cost follows the step count. An eight-times run is about 40 minutes of training.
"""
