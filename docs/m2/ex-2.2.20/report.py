# title: Ex 2.2.20: a high-rate head start before the recipe schedule

# The design constants come from `experiment.py` beside this script. The runs this draft builds on are published
# already: the 200- and 400-epoch `no-four` runs of ex-2.2.19, the 400-epoch `no-four` run of ex-2.2.18 (the
# 400-epoch run at seed 600), and the three ex-2.2.17 seeds of the recipe on the full op set, which set the per-op
# tolerances.
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

X19 = ex.ex2219
FOUR = ex.OP_SET.name
OPS = ex.OP_SET.ops
HSV_CHANNEL = ("hue-hsv", "sat-hsv", "value-hsv")
REF_E = ex.REFERENCE_EPOCHS
E = ex.EPOCHS
HS = ex.HEAD_START_EPOCHS


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


def span(values: Sequence[float], fmt: str = ".2f") -> str:
    """The range of *values*, or the one value when they all round the same."""
    lo, hi = format(min(values), fmt), format(max(values), fmt)
    return lo if lo == hi else f"{lo} and {hi}"


# --- Loading ------------------------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict:
    assert path is not None, "not published yet"
    return json.loads(path.read_text())


# The runs whose per-context arrays the op confusion (E3) needs: every schedule at the gate seeds.
HEAD = {p: {s: ex.label_of(p, s) for s in ex.SEEDS} for p in ex.SECOND_PEAKS}
_PLAIN = [X19.label_of(e, ex.LO_LR, s) for e in (ex.EPOCHS, REF_E) for s in ex.GATE_SEEDS]
_ARRAY_REFS = {
    **{ex.EVAL_ARRAYS_REF.format(label=HEAD[p][s]): HEAD[p][s] for p in ex.SECOND_PEAKS for s in ex.GATE_SEEDS},
    **{X19.EVAL_ARRAYS_REF.format(label=lbl): lbl for lbl in _PLAIN},
}
with tempfile.TemporaryDirectory() as _tmp:
    _refs = [
        ex.EVAL_REF,
        ex.TRAJ_REF,
        X19.EVAL_REF,
        X19.TRAJ_REF,
        ex.ex2218.EVAL_REF,
        ex.ex2218.TRAJ_REF,
        ex.ex2217.EVAL_REF,
        *_ARRAY_REFS,
    ]
    _files = fetch(_refs, Path(_tmp))
    EVAL = read_json(_files[ex.EVAL_REF])
    EVAL_19 = read_json(_files[X19.EVAL_REF])
    _eval18 = read_json(_files[ex.ex2218.EVAL_REF])
    _ref18 = next(r for r in _eval18["runs"] if r["label"] == FOUR)
    PRIOR = {r["label"]: r for r in read_json(_files[ex.ex2217.EVAL_REF])["runs"] if r["label"] in ex.YARDSTICK}
    # The ex-2.2.18 `no-four` run is the 400-epoch run at seed 600, so it joins the runs of ex-2.2.19.
    REF0 = X19.label_of(REF_E, ex.LO_LR, 0)
    RUNS: dict[str, dict] = {r["label"]: r for r in [*EVAL_19["runs"], *EVAL["runs"]]} | {REF0: _ref18}
    TRAJ: dict[str, dict] = (
        read_json(_files[X19.TRAJ_REF])
        | read_json(_files[ex.TRAJ_REF])
        | {REF0: read_json(_files[ex.ex2218.TRAJ_REF])[FOUR]}
    )
    ARRAYS: dict[str, dict[str, np.ndarray]] = {}
    for _ref, _lbl in _ARRAY_REFS.items():
        _f = _files[_ref]
        assert _f is not None, f"{_ref} is not published"
        with np.load(_f) as _z:
            ARRAYS[_lbl] = {k: _z[k] for k in _z.files}

STATS = EVAL_19["op_set"]
assert abs(EVAL["op_set"]["ceiling"] - STATS["ceiling"]) < 1e-9, "the head-start runs train on the corpus of ex-2.2.19"
CEILING = STATS["ceiling"]
OP_TOL = ex.op_tolerances([PRIOR[s] for s in ex.YARDSTICK], OPS)


def plain(epochs: int, seed: int) -> str:
    """The label of the ex-2.2.19 run at *epochs* on the recipe schedule."""
    return X19.label_of(epochs, ex.LO_LR, seed)


# The 50-epoch scout run of ex-2.2.19 at the higher peak rate, which the first cycle repeats.
HI50 = X19.label_of(HS, ex.HI_LR, 0)


def eem(label: str, op: str | None = None) -> float:
    r = RUNS[label]["task"]["eem"]
    return r["all"] if op is None else r["per_op"][list(RUNS[label].get("ops", OPS)).index(op)]


def plain_shortfall(seed: int) -> float:
    return eem(plain(REF_E, seed)) - eem(plain(E, seed))


PLAIN_SHORT = [plain_shortfall(s) for s in ex.GATE_SEEDS]


def traj_skill(label: str) -> np.ndarray:
    """Skill on the probe set along a run: the row for all ops, then one per op; shape (op, point)."""
    t = TRAJ[label]
    e = np.array(t["eem_per_op"]).T
    f, c = np.array(STATS["floor_per_op"])[:, None], np.array(STATS["ceiling_per_op"])[:, None]
    total = (np.array(t["eem"]) - STATS["floor"]) / (CEILING - STATS["floor"])
    return np.vstack([total, (e - f) / (c - f)])


def skill_near(label: str, epoch: float) -> float:
    """Skill over all ops at the logged point nearest *epoch*."""
    ep = np.array(TRAJ[label]["epoch"])
    return float(traj_skill(label)[0, int(np.argmin(abs(ep - epoch)))])


HSV_HALF = 0.5


def hsv_rise(label: str) -> tuple[float, float]:
    """The first logged epoch at which the mean HSV skill passes *HSV_HALF*, and the learning rate there."""
    curve = traj_skill(label)[[1 + OPS.index(op) for op in HSV_CHANNEL]].mean(0)
    i = int(np.argmax(curve >= HSV_HALF))
    assert curve[i] >= HSV_HALF, f"{label} never passes {HSV_HALF}"
    return float(TRAJ[label]["epoch"][i]), float(TRAJ[label]["lr"][i])


RISE_200 = {s: hsv_rise(plain(E, s)) for s in ex.SEEDS}
RISE_400 = {s: hsv_rise(plain(REF_E, s)) for s in ex.SEEDS}
RISE_HI200 = hsv_rise(X19.label_of(E, ex.HI_LR, 0))
assert min(r for _, r in RISE_400.values()) > max(r for _, r in RISE_200.values()), "H3 says the rates do not overlap"


def hsv_best(label: str) -> float:
    return float(traj_skill(label)[[1 + OPS.index(op) for op in HSV_CHANNEL]].mean(0).max())


# The short scout runs of ex-2.2.19 annealed to their floor without the HSV ops passing HSV_HALF.
SHORT_BEST = max(hsv_best(X19.label_of(e, lr, 0)) for e in (HS, 2 * HS) for lr in (ex.LO_LR, ex.HI_LR))
assert SHORT_BEST < HSV_HALF
LONGER_AT_50 = [skill_near(plain(e, s), HS) for e in (E, REF_E) for s in ex.SEEDS]
assert traj_skill(HI50)[0, -1] > max(LONGER_AT_50), "the Why section says the head start is ahead at epoch 50"
assert RISE_HI200[0] > max(r for r, _ in RISE_200.values()), "H3 says the high-rate run passed later than any plain run"

# --- The schedules ------------------------------------------------------------------------------------------

_t = TRAJ[plain(E, 0)]
EPOCH_LENGTH = round(_t["step"][-1] / _t["epoch"][-1])
CHECK = ex.schedule_check(EPOCH_LENGTH)
assert all(ex.schedule_check(EPOCH_LENGTH, p)["second"] < 1e-5 for p in ex.SECOND_PEAKS), (
    "the second cycle is the recipe schedule for its length and peak"
)
BAND = ex.BAND_LR


def schedule_curve(epochs: int, peak: float, sheet: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Epochs and learning rate along a run of *epochs*, from the training code itself."""
    from sca.config import SchedulerConfig
    from sca.training.scheduler import configure_schedule

    config = SchedulerConfig(
        epochs=epochs, warmup_epochs=ex.WARMUP_EPOCHS, min_lr_factor=ex.MIN_LR_FACTOR, lr_sheet=sheet
    )
    steps = np.arange(0, epochs * EPOCH_LENGTH + 1, EPOCH_LENGTH // 8)
    return steps / EPOCH_LENGTH, np.asarray(configure_schedule(config, peak, EPOCH_LENGTH)(steps))


@memo
def schedule_draw(alt_text: str, caption: str) -> str:
    @themed(name="schedule", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.0), layout="constrained")
        for (epochs, peak, sheet, name), color in zip(
            [
                (REF_E, ex.LO_LR, None, f"{REF_E} epochs"),
                (E, ex.LO_LR, None, f"{E} epochs"),
                (E, ex.HI_LR, ex.lr_sheet(), "head start"),
                (E, ex.HI_LR, ex.lr_sheet(BAND), f"head start, second peak {BAND:g}"),
            ],
            ("C7", "C0", "C1", "C3"),
            strict=True,
        ):
            x, y = schedule_curve(epochs, peak, sheet)
            ax.plot(x, y, color=color, lw=1.4, label=name)
        for rises, color in ((RISE_400, "C7"), (RISE_200, "C0")):
            ax.scatter(*zip(*rises.values(), strict=True), s=14, color=color, zorder=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.set_xlim(0, REF_E)
        fig.legend(loc="outside upper center", ncols=2, frameon=False, fontsize=8)
        return fig

    return _plot()


# --- Results ------------------------------------------------------------------------------------------------

LO = ex.LO_LR
GATE = ex.GATE_SEEDS


def short(label: str, seed: int, op: str | None = None) -> float:
    """How far *label* falls short of the plain 400-epoch run of its seed."""
    return eem(plain(REF_E, seed), op) - eem(label, op)


def gain(label: str, seed: int, op: str | None = None) -> float:
    """How far *label* scores above the plain 200-epoch run of its seed."""
    return eem(label, op) - eem(plain(E, seed), op)


def verdict_h1(p: float) -> dict:
    """The H1 rule of ex-2.2.19, applied to the head start with second peak *p*."""
    mean = float(np.mean([short(HEAD[p][s], s) for s in GATE]))
    per_op = {op: float(np.mean([short(HEAD[p][s], s, op) for s in GATE])) for op in OPS}
    over = [op for op in OPS if per_op[op] > OP_TOL[op]]
    if over or mean > ex.PARTIAL_TOL:
        v = "Miss"
    else:
        v = "Pass" if mean <= ex.SHORTFALL_TOL else "Partial"
    return {"mean": mean, "per_op": per_op, "over": over, "verdict": v}


def verdict_h2(p: float) -> dict:
    gains = [gain(HEAD[p][s], s) for s in GATE]
    mean = float(np.mean(gains))
    v = "Miss" if mean <= 0 else ("Pass" if min(gains) > 0 else "Partial")
    return {"gains": gains, "mean": mean, "wins": sum(g > 0 for g in gains), "verdict": v}


H1 = {p: verdict_h1(p) for p in ex.SECOND_PEAKS}
H2 = {p: verdict_h2(p) for p in ex.SECOND_PEAKS}
PLAIN_H1 = float(np.mean(PLAIN_SHORT))
ADOPTABLE = [p for p in ex.SECOND_PEAKS if H1[p]["verdict"] == H2[p]["verdict"] == "Pass"]
ADOPTED = max(ADOPTABLE, key=lambda p: np.mean([eem(HEAD[p][s]) for s in GATE]), default=None)

RISE = {p: {s: hsv_rise(HEAD[p][s])[0] for s in ex.SEEDS} for p in ex.SECOND_PEAKS}
SOONER = {p: [RISE_200[s][0] - RISE[p][s] for s in GATE] for p in ex.SECOND_PEAKS}
assert all(sum(d < 0 for d in SOONER[p]) == 2 and sum(d > 0 for d in SOONER[p]) == 1 for p in ex.SECOND_PEAKS), (
    "H3 and E4 say both head starts came later in two seeds and earlier in the third"
)
assert np.mean(PLAIN_SHORT) > ex.SHORTFALL_TOL, "the Why section says the plain runs fell short of the gate"
assert H1[BAND]["mean"] > H1[LO]["mean"] > PLAIN_H1, "H1 and E4 order the shortfalls"
assert H2[BAND]["wins"] == 0, "E4 says the lower-peak head start scores lower in all three seeds"


def kl(label: str) -> float:
    return RUNS[label]["task"]["kl"]["all"]


# E1: skill at the end of the first cycle and at the end of the second warmup.
CHECKPOINTS = (HS, HS + ex.WARMUP_EPOCHS)


def hsv_near(label: str, epoch: float) -> float:
    ep = np.array(TRAJ[label]["epoch"])
    return float(traj_skill(label)[[1 + OPS.index(op) for op in HSV_CHANNEL], int(np.argmin(abs(ep - epoch)))].mean())


def gate_mean(f, labels: Sequence[str]) -> float:
    return float(np.mean([f(lbl) for lbl in labels]))


E1_AT = {
    name: {
        "skill": [gate_mean(lambda lbl, e=e: skill_near(lbl, e), labels) for e in CHECKPOINTS],
        "hsv": [gate_mean(lambda lbl, e=e: hsv_near(lbl, e), labels) for e in CHECKPOINTS],
    }
    for name, labels in {
        f"head start, second peak {LO:g}": [HEAD[LO][s] for s in GATE],
        f"head start, second peak {BAND:g}": [HEAD[BAND][s] for s in GATE],
        f"plain {E} epochs": [plain(E, s) for s in GATE],
    }.items()
}
E1_HEAD, E1_BAND, E1_PLAIN = E1_AT.values()
assert E1_HEAD["skill"][1] < E1_PLAIN["skill"][1] + 0.02, "E1 says the lead is gone by the end of the second warmup"
assert short(HEAD[LO][0], 0) < 0, "H1 says seed 600 scored above its 400-epoch run"
_kl_plain = [kl(plain(E, s)) for s in GATE]
for _p in ex.SECOND_PEAKS:
    _kl_head = [kl(HEAD[_p][s]) for s in GATE]
    assert min(_kl_head) < max(_kl_plain) and min(_kl_plain) < max(_kl_head), "E2 says the KL ranges overlap"


# --- Op confusion (E3) ----------------------------------------------------------------------------------------

ALL_OPS: tuple[str, ...] = tuple(ex.ex2216.OP_NAMES)
CONFIDENT = 0.99


@memo
def answer_table():
    """The answers of all eleven ops on every pair, so the answers of a dropped op stay defined."""
    return ex.ex2216._get_posterior().build_table(ex.ex2216.TABLE)


@memo
def op_confusion(p16: np.ndarray, op_ids: np.ndarray, post: np.ndarray, pair: np.ndarray, ops: tuple[str, ...]):
    """Where a run puts its mass on confident contexts, by op, as in ex-2.2.19. Rows are the true op and columns
    all eleven ops, in the order of ALL_OPS. The diagonal is the mass on the colors the true op can give for the
    query operands; off the diagonal, the mass on the colors the column op can give and the true op cannot.
    """
    table = answer_table()
    rows = np.arange(len(p16))
    conf = post[rows, op_ids] > CONFIDENT
    p, pair = p16[conf].astype(float), pair[conf]
    true_op = np.array([ALL_OPS.index(ops[i]) for i in op_ids[conf]])
    rows = np.arange(len(p))
    gives = np.zeros((len(ALL_OPS), *p.shape), dtype=bool)
    for o in range(len(ALL_OPS)):
        idx, ok = table.idx[o, pair], table.prob[o, pair] > 0
        for j in range(idx.shape[1]):
            gives[o, rows[ok[:, j]], idx[ok[:, j], j]] = True
    true = gives[true_op, rows]
    per_col = np.stack(
        [(p * np.where((true_op == o)[:, None], true, gives[o] & ~true)).sum(1) for o in range(len(ALL_OPS))], 1
    )
    m = np.full((len(ALL_OPS), len(ALL_OPS)), np.nan)
    for o in range(len(ALL_OPS)):
        if (true_op == o).any():
            m[o] = per_col[true_op == o].mean(0)
    return m


_seven = np.ix_([ALL_OPS.index(o) for o in OPS], [ALL_OPS.index(o) for o in OPS])
SCHEDULES = {
    f"head start, second peak {LO:g}": [HEAD[LO][s] for s in GATE],
    f"head start, second peak {BAND:g}": [HEAD[BAND][s] for s in GATE],
    f"plain {E} epochs": [plain(E, s) for s in GATE],
    f"plain {REF_E} epochs": [plain(REF_E, s) for s in GATE],
}
CONFUSION = {
    name: np.mean(
        [
            op_confusion(a["p"], a["op_ids"], a["posterior"], a["query_pair"], tuple(OPS))[_seven]
            for a in (ARRAYS[lbl] for lbl in labels)
        ],
        axis=0,
    )
    for name, labels in SCHEDULES.items()
}
OFF_DIAGONAL = ~np.eye(len(OPS), dtype=bool)
OTHER_OPS = tuple(op for op in OPS if op not in HSV_CHANNEL)


def row_off(name: str, ops: Sequence[str]) -> float:
    """The mean over the rows of *ops* of the mass off the diagonal, for the schedule *name*."""
    return float(np.mean([CONFUSION[name][OPS.index(op)][OFF_DIAGONAL[OPS.index(op)]].sum() for op in ops]))


_hsv_off = [row_off(n, HSV_CHANNEL) for n in CONFUSION]
assert _hsv_off[1] > _hsv_off[0] > _hsv_off[2] > _hsv_off[3], "E3 orders the schedules by mass off the diagonal"
_other_off = [row_off(n, OTHER_OPS) for n in CONFUSION]
assert max(_other_off) - min(_other_off) < 0.015, "E3 says the other rows are about the same"


# --- Figures ------------------------------------------------------------------------------------------------


def rule_color() -> str:
    return light_dark("#333", "#ddd")


HEAD_COLOR = {LO: "C1", BAND: "C3"}
PLAIN_COLOR = {E: "C0", REF_E: "C7"}


def gate_line(ax: Axes, y: float, *, partial: float | None = None, fail: str | None = None) -> None:
    """A dashed gate line, a dotted partial level under it when there is one, and the failing side hatched."""
    ax.axhline(y, color=rule_color(), lw=0.9, ls="--", zorder=2)
    if partial is not None:
        ax.axhline(partial, color=rule_color(), lw=0.7, ls=":", zorder=2)
    if fail is not None:
        lo, hi = ax.get_ylim()
        band = (lo, y) if fail == "below" else (y, hi)
        ax.axhspan(*band, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, zorder=0, alpha=0.1)
        ax.set_ylim(lo, hi)


def dots(ax: Axes, x: float, v, color, *, rng, ms: float = 6.0) -> None:
    """One column of per-seed dots with the seed mean on top; a thin bar behind spans the seed range."""
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + rng.uniform(-0.06, 0.06, len(v)), v, "o", ms=2.8, color=color, alpha=0.6, zorder=3, mew=0)
    ax.plot(x, v.mean(), "_", ms=ms * 2, mew=2, color=color, zorder=4)


@memo
def shortfall_draw(alt_text: str, caption: str) -> str:
    @themed(name="shortfall", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.4), layout="constrained", sharey=True, width_ratios=[1, 4.5])
        rng = np.random.default_rng(0)
        cols = [
            (f"plain {E}", lambda s, op=None: short(plain(E, s), s, op), PLAIN_COLOR[E]),
            *(
                (f"head start, {p:g}", lambda s, op=None, p=p: short(HEAD[p][s], s, op), HEAD_COLOR[p])
                for p in ex.SECOND_PEAKS
            ),
        ]
        w = 0.25
        for k, (name, f, c) in enumerate(cols):
            x = (k - 1) * w * 1.6
            dots(a, x, [f(s) for s in GATE], c, rng=rng)
            a.plot([], [], "o", color=c, label=name)
            for j, op in enumerate(OPS):
                dots(b, j + (k - 1) * w, [f(s, op) for s in GATE], c, rng=rng, ms=4)
        for j, op in enumerate(OPS):
            b.plot([j - 0.42, j + 0.42], [OP_TOL[op]] * 2, "-", color=rule_color(), lw=1.4, zorder=5)
        a.set_xticks([0], ["all ops"])
        a.set_xlim(-0.8, 0.8)
        a.set_ylabel(f"EEM at {REF_E} epochs minus EEM of run")
        b.set_xticks(range(len(OPS)), OPS, fontsize=8)
        b.set_xlim(-0.6, len(OPS) - 0.4)
        for ax in (a, b):
            ax.axhline(0, color="0.6", lw=0.6)
        gate_line(a, ex.SHORTFALL_TOL, partial=ex.PARTIAL_TOL, fail="above")
        fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncols=3, frameon=False, fontsize=8)
        return fig

    return _plot()


@memo
def gain_draw(alt_text: str, caption: str) -> str:
    @themed(name="gain", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(9.0, 3.2), layout="constrained")
        rng = np.random.default_rng(1)
        w = 0.18
        for k, p in enumerate(ex.SECOND_PEAKS):
            c = HEAD_COLOR[p]
            for j, op in enumerate([None, *OPS]):
                dots(ax, j + (k - 0.5) * w * 1.6, [gain(HEAD[p][s], s, op) for s in GATE], c, rng=rng, ms=4)
            ax.plot([], [], "o", color=c, label=f"head start, second peak {p:g}")
        ax.axhline(0, color=rule_color(), lw=0.8)
        ax.axvline(0.5, color="0.7", lw=0.6)
        ax.set_xticks(range(len(OPS) + 1), ["all ops", *OPS], fontsize=8)
        ax.set_ylabel(f"EEM minus plain {E}-epoch run")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=2, frameon=False, fontsize=8)
        return fig

    return _plot()


@memo
def curves_draw(alt_text: str, caption: str) -> str:
    @themed(name="curves", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.2), layout="constrained", sharex=True, sharey=True)
        series = [
            (f"plain {E}", lambda s: plain(E, s), PLAIN_COLOR[E]),
            *((f"head start, {p:g}", lambda s, p=p: HEAD[p][s], HEAD_COLOR[p]) for p in ex.SECOND_PEAKS),
        ]
        for k, (ax, name) in enumerate(zip(axes.flat, ["all ops", *OPS], strict=True)):
            ax = cast(Axes, ax)
            for label, f, c in series:
                for i, s in enumerate(GATE):
                    lbl = f(s)
                    ax.plot(
                        TRAJ[lbl]["epoch"],
                        traj_skill(lbl)[k],
                        color=c,
                        lw=0.9,
                        alpha=0.7,
                        label=label if i == 0 else None,
                    )
            ax.axhline(1, ls="--", color=rule_color(), lw=0.8)
            ax.axvline(HS, color="0.7", lw=0.6)
            ax.set_title(name, fontsize=9)
            ax.set_ylim(-0.1, 1.05)
            ax.set_xlim(0, E)
        fig.supxlabel("epoch", fontsize=9)
        fig.supylabel("skill on the probe set", fontsize=9)
        fig.legend(
            *axes.flat[0].get_legend_handles_labels(), loc="outside upper center", ncols=3, frameon=False, fontsize=8
        )
        return fig

    return _plot()


@memo
def calibration_draw(alt_text: str, caption: str) -> str:
    @themed(name="calibration", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.0, 3.6), layout="constrained")
        groups = [
            (f"plain {REF_E}", [plain(REF_E, s) for s in ex.SEEDS], PLAIN_COLOR[REF_E]),
            (f"plain {E}", [plain(E, s) for s in ex.SEEDS], PLAIN_COLOR[E]),
            *((f"head start, {p:g}", list(HEAD[p].values()), HEAD_COLOR[p]) for p in ex.SECOND_PEAKS),
        ]
        for name, labels, c in groups:
            ax.plot([eem(lbl) for lbl in labels], [kl(lbl) for lbl in labels], "o", ms=6, color=c, label=name)
        ax.set_xlabel("held-out EEM")
        ax.set_ylabel("calibration KL")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=4, frameon=False, fontsize=8)
        return fig

    return _plot()


def seq_cmap():
    cmap = plt.get_cmap(light_dark("Blues", "magma")).copy()
    cmap.set_bad(light_dark("#fff", "#111"))
    return cmap


def cell_text_color(v: float, vmax: float) -> str:
    dark_cell = (v / vmax > 0.55) == (light_dark(0, 1) == 0)
    return "#fff" if dark_cell else "#000"


# Confusion values at or above this are printed.
PRINT_FLOOR = 0.01


@memo
def confusion_draw(alt_text: str, caption: str) -> str:
    @themed(name="confusion", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.0), layout="constrained", sharex=True, sharey=True)
        n = len(OPS)
        eye = np.eye(n, dtype=bool)
        vmax = max(float(np.nanmax(np.where(eye, np.nan, m))) for m in CONFUSION.values())
        im = None
        for ax, (name, m) in zip(axes.flat, CONFUSION.items(), strict=True):
            ax = cast(Axes, ax)
            im = ax.imshow(np.where(eye, np.nan, m), cmap=seq_cmap(), vmin=0, vmax=vmax)
            for i, j in np.ndindex(n, n):
                if i == j:
                    ax.plot(j, i, "s", ms=13, mfc="none", mec="0.6", mew=0.6)
                elif m[i, j] >= PRINT_FLOOR:
                    ax.text(
                        j,
                        i,
                        f"{m[i, j]:.2f}"[1:],
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=cell_text_color(m[i, j], vmax),
                    )
            ax.set_title(name, fontsize=9)
            ax.set_xticks(range(n), OPS, rotation=90, fontsize=8)
            ax.set_yticks(range(n), OPS, fontsize=8)
        fig.supxlabel("answers of this op, beyond those of the true op", fontsize=9)
        fig.supylabel("true op", fontsize=9)
        assert im is not None
        fig.colorbar(im, ax=axes, shrink=0.5, label="mass")
        return fig

    return _plot()


COST = len(ex.SECOND_PEAKS) * len(ex.SEEDS) * ex.cost_per_run(E)

rf"""

# Ex 2.2.20: A high-rate head start before the recipe schedule

/// tip |
<!-- lede -->
We train the seven-op set for 200 epochs on a schedule of two cycles: a short one at a high learning rate, then the recipe schedule for the rest of the run. A second condition lowers the peak of the second cycle. Does either keep more of the skill of the 400-epoch recipe than a plain 200-epoch run? Neither did, so we keep the plain schedule.
///

## Findings

- [The head start keeps most of the skill (H1)](#the-head-start-keeps-most-of-the-skill-h1) — **miss**. The head-start runs fall short of the {REF_E}-epoch runs, inside the partial band, but all three HSV ops fall short by more than their tolerance.
- [The head start beats the plain schedule (H2)](#the-head-start-beats-the-plain-schedule-h2) — **miss**. The head start scores above the plain {E}-epoch run in {H2[LO]["wins"]} of the three seeds, but below it on average. We keep the plain schedule.
- [The HSV ops come earlier (H3)](#the-hsv-ops-come-earlier-h3) — **miss**. They came later than in the plain run in two of the three seeds.
- [Where the first cycle leaves off (E1)](#where-the-first-cycle-leaves-off-e1) — the head start leads the plain run at epoch {HS}, but the second warmup takes the lead away within {ex.WARMUP_EPOCHS:g} epochs.
- [Calibration (E2)](#calibration-e2) — the head start is about as well calibrated as the plain {E}-epoch run.
- [Op confusion (E3)](#op-confusion-e3) — the head start puts more mass off the diagonal in the HSV rows than the plain {E}-epoch run does.
- [A lower second peak (E4)](#a-lower-second-peak-e4) — misses both rules by a wider margin.

/// admonition | How to read this report
This report was preregistered: the hypotheses, their gates, and the adoption rule were frozen at commit `2ea0b76`, before any run of this experiment. Each section opens with what we expect, and the results replaced the placeholders in place.
///

## Why this experiment

Ex-2.2.19 adopted 200 epochs for the seven-op set (`no-four`), half the length of the recipe, as a partial pass. On three fresh seeds the 200-epoch runs fell short of the 400-epoch runs by a little more than we would like, and nearly all of the shortfall was in the three HSV ops.

A short run at a high rate gets a long way quickly. The 50-epoch run of ex-2.2.19 at a high peak learning rate ended with more skill than any of the 200- and 400-epoch runs had at epoch {HS}. But it ends before the HSV ops are learned, which in the 200-epoch runs happened later in the run.

So we try both in one run: {HS} epochs on the schedule of that short run, then {E - HS} epochs on the recipe schedule from where it leaves off. If the second cycle builds on the first, the run should end closer to the 400-epoch runs than a plain 200-epoch run does, at the same cost.

## Parameters

Each run follows the {E}-epoch recipe of ex-2.2.19 with a different learning-rate schedule. The first cycle warms up over {ex.WARMUP_EPOCHS:g} epochs to {ex.HI_LR:g} and follows a cosine down to 1% of {ex.LO_LR:g} at epoch {HS}. The second warms up from there over {ex.WARMUP_EPOCHS:g} epochs to {ex.LO_LR:g}, and follows a cosine down to 1% of that at epoch {E}.

The second cycle is the recipe schedule of a {E - HS}-epoch run, step for step.[^sheet] The first cycle is the schedule of the ex-2.2.19 run at {ex.HI_LR:g}, except that it ends where the second warmup starts. The optimizer state is kept from one cycle to the next, though resetting it should make no measurable difference.[^adam]

The schedule differs from the plain one: it has a higher peak for the first {HS} epochs, a second warmup, and a final anneal {E - HS} epochs long instead of {E}. These runs test the three together, so they cannot tell which of them made a difference.

A second head-start condition peaks at {BAND:g} in the second cycle, about the highest rate at which a plain {E}-epoch run of ex-2.2.19 learned the HSV ops (see H3, below). So it spends longer near those rates, and less time above them, than the first head start. From about epoch 80 on, its rate is close to that of the plain {E}-epoch run, so the two differ mostly in the first 80 epochs.

[^sheet]: The schedule is a dopesheet: keyframes at the start and peak of each warmup and at the end of each cosine, joined by straight lines going up and half cosines going down. At {EPOCH_LENGTH} steps an epoch, the second cycle matches the recipe schedule of a {E - HS}-epoch run to within {CHECK["second"]:.0e} of its peak.

[^adam]: Adam keeps running averages of the gradient and of its square. These decay with time constants of about 10 and 20 steps. The second warmup lasts {ex.WARMUP_EPOCHS * EPOCH_LENGTH:,g} steps, so a reset at the end of the first cycle would be forgotten early in it, while the rate is still near its floor.

"""

schedule_draw(
    f"""
        A line chart of learning rate against epoch from 0 to {REF_E}. The {REF_E}-epoch recipe warms up to
        {ex.LO_LR:g} and falls along a cosine to near zero at {REF_E}. The {E}-epoch recipe does the same by {E}. The
        head-start schedule rises to {ex.HI_LR:g}, falls to near zero by epoch {HS}, rises again to {ex.LO_LR:g} by
        epoch {HS + ex.WARMUP_EPOCHS:g}, and falls to near zero by {E}. The second head-start schedule is the same, with a
        second peak of {BAND:g}. A dot on each recipe curve marks, for each
        seed, where the HSV ops were learned: on the {E}-epoch curve between epochs
        {span([r[0] for r in RISE_200.values()], ".0f")}, and on the {REF_E}-epoch curve between epochs
        {span([r[0] for r in RISE_400.values()], ".0f")}, where its rate is higher.
    """,
    f"""
        **The four schedules.** Learning rate against epoch. Each dot is the point at which the HSV skill of a plain
        run of ex-2.2.19 first passed {HSV_HALF:g}.
    """,
)

# %%
table_html(
    ["condition", "epochs", "schedule", "model seeds", "runs"],
    [
        [
            "head start (new)",
            f"{E}",
            f"{HS} at {ex.HI_LR:g}, then {E - HS} at {ex.LO_LR:g}",
            "600–603",
            f"{len(ex.SEEDS)}",
        ],
        [
            f"head start, second peak {BAND:g} (new)",
            f"{E}",
            f"{HS} at {ex.HI_LR:g}, then {E - HS} at {BAND:g}",
            "600–603",
            f"{len(ex.SEEDS)}",
        ],
        [f"plain {E} (ex-2.2.19)", f"{E}", f"{ex.LO_LR:g}", "600–603", "reused"],
        [f"plain {REF_E} (ex-2.2.18, ex-2.2.19)", f"{REF_E}", f"{ex.LO_LR:g}", "600–603", "reused"],
    ],
    """
        **The runs.** All train `no-four` from the same four initializations, so each head-start run pairs with a plain
        run of each length.
    """,
    text_cols=3,
)

rf"""

Held-out *expected exact match* (EEM) is the probability that an answer drawn from the model at the query `=` is a correct answer of the true op. The *Bayes ceiling* is the same score for an ideal predictor. *Skill* is EEM rescaled so that a uniform guess scores 0 and the ceiling scores 1.

A run *falls short* of another by how much lower its EEM is, compared between runs that share a model seed and averaged over seeds. The hypotheses are scored on seeds {ex.SEED_OFFSET + min(ex.GATE_SEEDS)}–{ex.SEED_OFFSET + max(ex.GATE_SEEDS)}, as H1 of ex-2.2.19 was. Seed {ex.SEED_OFFSET} is shown beside them: the observation behind this experiment came from its runs, so it could flatter the head start.

## The head start keeps most of the skill (H1)

**What we expect.** The head-start runs would fall short of the {REF_E}-epoch runs by at most {ex.SHORTFALL_TOL} on average, and no op would fall short by more than its tolerance on average. That would be a pass. A shortfall between {ex.SHORTFALL_TOL} and {ex.PARTIAL_TOL} would be a partial pass, and one beyond {ex.PARTIAL_TOL}, or an op beyond its tolerance, would be a miss. This is the H1 rule of ex-2.2.19, with the same tolerances.

**What we saw.** The head-start runs fell short of the {REF_E}-epoch runs by a little more than the plain {E}-epoch runs did, inside the partial band. The shortfall is in the HSV ops: each of the three falls short by more than its tolerance, where none did in the plain runs.

At seed {ex.SEED_OFFSET} the head start scored above the {REF_E}-epoch run, so this seed did flatter the head start.

"""

shortfall_draw(
    f"""
        Two panels of dot plots sharing a vertical axis: EEM at {REF_E} epochs minus EEM of the run. The left panel
        is all ops, the right one column per op, each with three groups of dots: the plain {E}-epoch runs, the head
        start, and the head start with a second peak of {BAND:g}, one dot per seed at seeds 601–603 with a bar for
        the mean. Over all ops the plain runs average {PLAIN_H1:.3f}, the head start {H1[LO]["mean"]:.3f}, and the
        lower-peak head start {H1[BAND]["mean"]:.3f}, against a dashed gate at {ex.SHORTFALL_TOL} and a dotted
        partial level at {ex.PARTIAL_TOL}. In the per-op panel a black bar marks each tolerance; both head starts
        go above it in all three HSV ops, and the plain runs stay below it.
    """,
    f"""
        **Shortfall from the {REF_E}-epoch runs.** Seeds 601–603; a dash marks the seed mean. Left: all ops, with
        the gate (dashed) and the edge of the partial band (dotted). Right: by op, with the tolerance of each op
        (black bar).
    """,
)

# %%
table_html(
    ["op", "tolerance", f"plain {E}", "head start", f"head start, {BAND:g}"],
    [
        [
            f"`{op}`",
            f"{OP_TOL[op]:.3f}",
            f"{np.mean([short(plain(E, s), s, op) for s in GATE]):.3f}",
            f"{H1[LO]['per_op'][op]:.3f}",
            f"{H1[BAND]['per_op'][op]:.3f}",
        ]
        for op in OPS
    ],
    "**Mean shortfall by op**, seeds 601–603.",
)

rf"""

/// admonition | Miss
The head start falls short by {H1[LO]["mean"]:.4f} on average, inside the partial band, but `hue-hsv`, `sat-hsv`, and `value-hsv` each fall short by more than their tolerance.
///

## The head start beats the plain schedule (H2)

**What we expect.** The head-start run would score a higher EEM than the plain {E}-epoch run in each of the three seeds, and so on average. That would be a pass. A higher mean with a lower EEM in one or two seeds would be a partial pass, and a mean gain at or below zero would be a miss.

A miss would mean the second cycle kept nothing of the first, or lost it in the second warmup. H2 checks only the direction of the gain: the whole gap to the {REF_E}-epoch runs is small, so H1 covers its size.

**Which schedule we adopt.** H1 and H2 score the head start with the recipe peak, and E4 scores the one with a second peak of {BAND:g} by the same rules. If both hypotheses pass for either schedule, the {E}-epoch runs that follow will use that schedule. If both schedules pass, we take the one with the higher mean EEM over seeds {ex.SEED_OFFSET + min(ex.GATE_SEEDS)}–{ex.SEED_OFFSET + max(ex.GATE_SEEDS)}. Otherwise we keep the plain schedule.

**What we saw.** The head start scored above the plain {E}-epoch run in {H2[LO]["wins"]} of the three seeds, and below it on average. The ops that lose most are the HSV ops again.

"""

gain_draw(
    f"""
        A dot plot of EEM minus that of the plain {E}-epoch run of the same seed, one column for all ops and one per
        op, with the two head-start schedules side by side in each, one dot per seed at seeds 601–603 and a dash for
        the mean. Over all ops the head start has gains of
        {", ".join(f"{g:+.3f}" for g in H2[LO]["gains"])}, and the head start with a second peak of {BAND:g} has
        {", ".join(f"{g:+.3f}" for g in H2[BAND]["gains"])}. The means are near or above zero for the four other
        ops and below zero for the three HSV ops, lowest for the lower-peak head start.
    """,
    f"""
        **Gain over the plain {E}-epoch run.** Seeds 601–603; a dash marks the seed mean. Above the line, the head
        start scored higher.
    """,
)

rf"""

/// admonition | Miss
The head start scores below the plain {E}-epoch run on average ({H2[LO]["mean"]:+.4f}), above it in {H2[LO]["wins"]} of three seeds.
///

## The HSV ops come earlier (H3)

**What we expect.** In the plain {E}-epoch runs, the mean skill of the HSV ops first passed {HSV_HALF:g} between epochs {span([r[0] for r in RISE_200.values()], ".0f")}. We expect the head-start runs to pass it about 25 epochs earlier, paired by seed (a gap estimated from the skill curves of ex-2.2.19).

The runs of ex-2.2.19 already suggest these ops do not wait for the learning rate to fall to some level. The {REF_E}-epoch runs passed {HSV_HALF:g} at about the same epochs as the {E}-epoch runs of the same seeds, but at learning rates higher than any rate at which a {E}-epoch run passed it.

The {HS}- and {2 * HS}-epoch runs fell through those rates to their floor, and their HSV skill never passed {HSV_HALF:g}. That includes the {HS}-epoch run at the higher peak rate, which is the first cycle of the head-start schedule.

So this schedule can separate elapsed time from learning progress. A rise at about the same epoch as in the plain run would say these ops wait for a number of epochs, whatever the model has learned by then. An earlier rise, as we expect, would say they build on the progress of the first cycle.

A later rise is possible too: the {E}-epoch run of ex-2.2.19 at the higher peak rate passed {HSV_HALF:g} later than any {E}-epoch run at the recipe peak, so a high early rate may hold these ops back. The predicted gap is several times the logging interval of {E // ex.ex2218.N_TRAJ_POINTS} epochs, so the logged curves can tell these outcomes apart. There is no gate, since no decision hangs on it.

**What we saw.** The HSV ops came later in the head-start runs in two of the three seeds, and earlier in the third. The curves show the first cycle lifting the other ops well above the plain run by epoch {HS}, with the HSV ops still near zero.

"""

curves_draw(
    f"""
        Eight small line charts of skill on the probe set against epoch, 0 to {E}: one for all ops and one per op.
        Each has three lines per schedule, one per seed at seeds 601–603: the plain {E}-epoch run, the head start,
        and the head start with a second peak of {BAND:g}. A vertical line marks epoch {HS}. In the all-ops panel
        both head starts climb to about {E1_HEAD["skill"][0]:.2f} by epoch {HS}, against about
        {E1_PLAIN["skill"][0]:.2f} for the plain run, then drop in the second warmup and rejoin the plain curves. Two of the
        head-start runs end lower than the rest. The other ops climb early in all schedules, and the head starts
        lift them to about 0.8 by epoch {HS} before the second warmup pulls them back. In the three HSV panels
        every run stays below about 0.15 until epoch 60 or later, then rises at seed-dependent epochs up to about
        140.
    """,
    """
        **Skill trajectories.** Seeds 601–603, one line per run. The vertical line marks the end of the first
        cycle.
    """,
)

# %%
table_html(
    ["model seed", f"plain {E}", "head start", f"head start, {BAND:g}"],
    [
        [f"{ex.SEED_OFFSET + s}", f"{RISE_200[s][0]:.0f}", f"{RISE[LO][s]:.0f}", f"{RISE[BAND][s]:.0f}"]
        for s in ex.SEEDS
    ],
    f"**The epoch at which the HSV skill first passes {HSV_HALF:g}**, logged every {E // ex.ex2218.N_TRAJ_POINTS} epochs.",
    text_cols=1,
)

rf"""

/// admonition | Miss
The HSV ops came later in two of three seeds and earlier in one. They did not build on the progress of the first cycle.
///

## Where the first cycle leaves off (E1)

Exploratory, with no prediction: how much of a head start the second cycle receives, and how much of it remains after the second warmup. We show the skill of the head-start runs at epoch {HS}, overall and per op, beside the plain runs at the same epoch and the {HS}-epoch ex-2.2.19 run at the higher peak rate (seed {ex.SEED_OFFSET} only).

The head start leads the plain run at epoch {HS} over all ops, while the HSV skill is still low in both. The second warmup takes the lead away by epoch {HS + ex.WARMUP_EPOCHS:g}. The head start with a second peak of {BAND:g} keeps more of its lead, but ends lower all the same (E4).

"""

table_html(
    [
        "schedule",
        f"skill, epoch {HS}",
        f"skill, epoch {HS + ex.WARMUP_EPOCHS:g}",
        f"HSV, epoch {HS}",
        f"HSV, epoch {HS + ex.WARMUP_EPOCHS:g}",
    ],
    [[name, *(f"{v:.2f}" for v in (*row["skill"], *row["hsv"]))] for name, row in E1_AT.items()],
    """
        **Skill at the end of the first cycle and of the second warmup**, over all ops and over the HSV ops,
        mean over seeds 601–603. The curves are in the H3 figure.
    """,
)

rf"""

## Calibration (E2)

Exploratory, with no prediction. We show the calibration KL of each run beside its EEM: the KL divergence[^kl] from the Bayes answer distribution to the answer distribution of the model. In ex-2.2.19 the {E}- and {REF_E}-epoch runs were equally well calibrated, so a head start that buys EEM at some cost to calibration would show here.

[^kl]: A measure of how far one probability distribution is from another; it is zero when they match.

The calibration KL of both head starts overlaps that of the plain {E}-epoch runs. Among the {E}-epoch runs, KL falls as EEM rises, so the runs that score lower are also less well calibrated.

"""

calibration_draw(
    f"""
        A scatter plot of calibration KL against held-out EEM, one dot per run at seeds 600–603, colored by
        schedule: plain {REF_E}, plain {E}, head start, and head start with a second peak of {BAND:g}. The {E}-epoch
        runs fall along one band from upper left to lower right, with the lower-peak head start at its upper left
        and two head-start runs at its lower right. The {REF_E}-epoch runs sit to the right of that band, at higher
        EEM and a KL between {span([kl(plain(REF_E, s)) for s in ex.SEEDS], ".2f")}.
    """,
    "**Calibration against EEM.** One dot per run, seeds 600–603.",
)

rf"""

## Op confusion (E3)

Exploratory, with no prediction. We show the op confusion matrix of each schedule, as in ex-2.2.19. For confident contexts of each true op, it gives the probability mass the model puts on the answers of each op, averaged over seeds {ex.SEED_OFFSET + min(ex.GATE_SEEDS)}–{ex.SEED_OFFSET + max(ex.GATE_SEEDS)}.

In ex-2.2.19 the {E}-epoch runs put more mass off the diagonal than the {REF_E}-epoch runs, mostly in the HSV rows. This shows whether the head start moves that mass back.

The head start puts more of that mass off the diagonal in the HSV rows than the plain {E}-epoch run, and the head start with a second peak of {BAND:g} puts more still. In the other rows the four schedules are about the same.

"""

confusion_draw(
    f"""
        Four heatmaps of op confusion, in a two-by-two grid on one color scale: the head start, the head start with
        a second peak of {BAND:g}, the plain {E}-epoch run, and the plain {REF_E}-epoch run. Rows are the true op
        and columns the op whose answers get the mass; the diagonal is left blank. The darkest squares are in the
        rows of `hue-hsv`, `sat-hsv`, and `value-hsv`, and they are darkest for the lower-peak head start, then
        the head start, then plain {E}, and lightest for plain {REF_E}.
    """,
    f"""
        **Op confusion**, mean over seeds 601–603, on confident contexts. Each square is the mass on the answers
        of the column op that the true op cannot give; values of {PRINT_FLOOR} and above are printed.
    """,
)

rf"""

## A lower second peak (E4)

Exploratory, with no prediction. We score the head start with a second peak of {BAND:g} by the rules of H1 and H2. As in H3, we also show the epoch at which its HSV skill first passes {HSV_HALF:g}, beside the same epoch for the other head start.

Its results sit beside the other head start in the H1, H2, and H3 figures and tables. It misses both H1 and H2. It falls short of the {REF_E}-epoch runs by more than the other head start, at the edge of the partial band ({H1[BAND]["mean"]:.4f}), with the three HSV ops each beyond their tolerance. It scores below the plain {E}-epoch run in all three seeds.

Its HSV ops came later than in the plain run in two seeds, and earlier in the third.

## Discussion

It seems we should keep the plain schedule from ex-2.2.19.

The head start does what the {HS}-epoch run of ex-2.2.19 suggested, while it lasts: by the end of the first cycle the model has learned most ops except the HSV ones. The second warmup then takes the lead away (E1). The second cycle seems to relearn those ops rather than build on the first.

The HSV ops came no earlier (H3), and the head-start runs ended with more of their probability mass on the answers of the wrong op (E3).

The lower second peak was meant to give those ops more time at the rates where the plain runs learned them, yet it ended furthest short (E4). So time near those rates does not seem to be what the HSV ops wait for, although the schedule changes three things at once and the three seeds disagree about the timing.

We don't know whether the second warmup itself costs the lead. A schedule that hands over from the first cycle without a second warmup may answer that, although we did already try schedules with plateaus in ex-2.2.17.

## Method

**Recipe.** The unanchored d64-L4 model with an untied readout and the newline mask, as in ex-2.2.19, on the `no-four` corpus condition that ex-2.2.18 built (`k3-r0.3`). The runs differ from the plain {E}-epoch runs of ex-2.2.19 in the learning-rate schedule alone.

**Measurements.** EEM, ceiling, floor, and calibration KL on the {ex.ex2216.HOLDOUT_CONTEXTS:,} held-out contexts per op, with the evaluation of ex-2.2.18. Skill curves are logged at about {ex.ex2218.N_TRAJ_POINTS} points per run on {ex.ex2218.N_TRAJ_EEM_PER_OP} held-out contexts per op, every {E // ex.ex2218.N_TRAJ_POINTS} epochs at {E} epochs, so an epoch of first passing is known to within that.

**Cost.** The new runs cost about \${COST:.2f} in all, scaled from the cost of an ex-2.2.18 run.

**Per-op tolerances.** As in ex-2.2.19: the seed range of the gap of each op over the three ex-2.2.17 runs, or {ex.PARTIAL_TOL}, whichever is larger.

"""

table_html(
    ["op", "tolerance"],
    [[f"`{op}`", f"{OP_TOL[op]:.3f}"] for op in OPS],
    "**Per-op tolerances.**",
)
