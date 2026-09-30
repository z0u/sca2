# title: Ex 2.2.18: dropping ops with similar answers, a scout

# The design constants and the refs come from `experiment.py` beside this script; the answer table and the posterior
# come from ex-2.2.16 through it, and the seed yardstick from ex-2.2.17's published evaluation.
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed

X = ex.ex2216
ALL_OPS: tuple[str, ...] = tuple(X.OP_NAMES)
# Each dropped op and the op whose answers it most often shares, from either side of a pair.
PAIR_OF: dict[str, str] = ex.PARTNER | ex.COUNTERPART
SINGLES: tuple[str, ...] = tuple(f"no-{op}" for op in PAIR_OF)
FOURS: tuple[str, ...] = ("no-four", "no-four-ld")
SETS: tuple[str, ...] = ("full", *SINGLES, *FOURS)
assert set(SETS) == {s.name for s in ex.OP_SETS}
YARDSTICK = tuple(f"sweep-{ex.PEAK_LR:g}-s{s}" for s in range(3))
# A context is confident when the posterior on its true op is above this, as in ex-2.2.17.
CONFIDENT = 0.99


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


def seq_cmap():
    """A sequential map that runs from the page color, so it prints legibly and holds up in the dark theme."""
    cmap = plt.get_cmap(light_dark("Blues", "magma")).copy()
    cmap.set_bad(light_dark("#fff", "#111"))
    return cmap


def cell_text_color(v: float, vmax: float) -> str:
    """Text that stays legible on a square of *seq_cmap* at *v*."""
    dark_cell = (v / vmax > 0.55) == (light_dark(0, 1) == 0)
    return "#fff" if dark_cell else "#000"


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


with tempfile.TemporaryDirectory() as _tmp:
    _refs = [ex.EVAL_REF, ex.TRAJ_REF, ex.ex2217.EVAL_REF, *(ex.EVAL_ARRAYS_REF.format(label=s) for s in SETS)]
    _files = fetch(_refs, Path(_tmp))
    EVAL = read_json(_files[ex.EVAL_REF])
    TRAJ = read_json(_files[ex.TRAJ_REF])
    RUNS = {r["label"]: r for r in EVAL["runs"]}
    STATS = EVAL["op_sets"]
    PRIOR = {r["label"]: r for r in read_json(_files[ex.ex2217.EVAL_REF])["runs"] if r["label"] in YARDSTICK}
    ARRAYS: dict[str, dict[str, np.ndarray]] = {}
    for _s in SETS:
        with np.load(cast(Path, _files[ex.EVAL_ARRAYS_REF.format(label=_s)])) as _z:
            ARRAYS[_s] = {k: _z[k] for k in _z.files}


def ops_of(s: str) -> tuple[str, ...]:
    return tuple(RUNS[s]["ops"])


def score(s: str, key: str = "eem", op: str | None = None, runs: dict | None = None) -> float:
    r = (runs or RUNS)[s]["task"][key]
    return r["all"] if op is None else r["per_op"][list((runs or RUNS)[s].get("ops", ALL_OPS)).index(op)]


def skill(s: str, op: str | None = None) -> float:
    """The share of the way from the floor to the ceiling of its own op set that a run got."""
    e, c, f = (score(s, k, op) for k in ("eem", "ceiling", "floor"))
    return (e - f) / (c - f)


def gap(s: str, op: str | None = None, runs: dict | None = None) -> float:
    return score(s, "ceiling", op, runs) - score(s, "eem", op, runs)


YARD_EEM = [score(s, runs=PRIOR) for s in YARDSTICK]
YARD_GAP = [gap(s, runs=PRIOR) for s in YARDSTICK]
YARD_SKILL = [
    (score(s, runs=PRIOR) - score(s, "floor", runs=PRIOR))
    / (score(s, "ceiling", runs=PRIOR) - score(s, "floor", runs=PRIOR))
    for s in YARDSTICK
]
YARD_SPREAD = max(YARD_EEM) - min(YARD_EEM)
# The seed range of each op's gap in ex-2.2.17's three runs of this recipe: the per-op yardstick.
YARD_OP_SPREAD = {
    op: max(gap(s, op, PRIOR) for s in YARDSTICK) - min(gap(s, op, PRIOR) for s in YARDSTICK) for op in ALL_OPS
}


# --- The leak onto the dropped op ------------------------------------------------------------------------------


@memo
def answer_table():
    """The answers of all eleven ops on every pair, so the answers of a dropped op stay defined."""
    return X._get_posterior().build_table(X.TABLE)


@memo
def op_confusion(p16: np.ndarray, op_ids: np.ndarray, post: np.ndarray, pair: np.ndarray, ops: tuple[str, ...]):
    """Where a run puts its mass on confident contexts, by op, as in ex-2.2.17. Rows are the true op and columns
    all eleven ops, in the order of ALL_OPS; a row is NaN for an op the run does not have. The diagonal is the mass
    on the colors the true op can give for the query operands; off the diagonal, the mass on the colors the column
    op can give and the true op cannot. Answers come from the full table, so a dropped op keeps its column. Also
    returns the number of confident contexts of each op.
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
    n = np.zeros(len(ALL_OPS), dtype=int)
    for o in range(len(ALL_OPS)):
        n[o] = (true_op == o).sum()
        if n[o]:
            m[o] = per_col[true_op == o].mean(0)
    return m, n


CONFUSION: dict[str, np.ndarray] = {}
N_CONFIDENT: dict[str, np.ndarray] = {}
for _s in SETS:
    _a = ARRAYS[_s]
    CONFUSION[_s], N_CONFIDENT[_s] = op_confusion(_a["p"], _a["op_ids"], _a["posterior"], _a["query_pair"], ops_of(_s))


def conf(s: str, a: str, b: str) -> float:
    """In run *s*, the mass on answers of *b* beyond those of *a*, on confident contexts of *a*."""
    return float(CONFUSION[s][ALL_OPS.index(a), ALL_OPS.index(b)])


def n_conf(s: str, op: str) -> int:
    return int(N_CONFIDENT[s][ALL_OPS.index(op)])


def unrelated(s: str, a: str, dropped: str) -> float:
    """In run *s*, the median square on the row of *a* over the ops other than *a* and *dropped*: the background."""
    row = CONFUSION[s][ALL_OPS.index(a)]
    return float(np.nanmedian([row[j] for j, o in enumerate(ALL_OPS) if o not in (a, dropped)]))


def leak(s: str, dropped: str) -> float:
    """The leak onto *dropped*: mass on its answers beyond those of its partner, on confident partner contexts."""
    return conf(s, PAIR_OF[dropped], dropped)


# --- Figures ------------------------------------------------------------------------------------------------


def probe_skill(s: str) -> np.ndarray:
    st = STATS[s]
    return (np.array(TRAJ[s]["eem"]) - st["floor"]) / (st["ceiling"] - st["floor"])


def crossing(s: str, share: float) -> int:
    """The first trajectory point at which the probe skill reached *share* of its final value."""
    y = probe_skill(s)
    return int(np.argmax(y >= share * y[-1]))


@memo
def confusion_draw(alt_text: str, caption: str) -> str:
    @themed(name="leak", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 3, figsize=(9.0, 9.6), layout="constrained", sharex=True, sharey=True)
        n = len(ALL_OPS)
        eye = np.eye(n, dtype=bool)
        vmax = max(float(np.nanmax(np.where(eye, np.nan, CONFUSION[s]))) for s in SETS)
        cmap = seq_cmap()
        im = None
        for ax, s in zip(axes.flat, SETS, strict=True):
            ax = cast(Axes, ax)
            m = CONFUSION[s]
            im = ax.imshow(np.where(eye, np.nan, m), cmap=cmap, vmin=0, vmax=vmax)
            for i, j in np.ndindex(n, n):
                v = m[i, j]
                if np.isnan(v):
                    continue
                if i == j:
                    ax.plot(j, i, "s", ms=6.5, mfc="none", mec="0.6", mew=0.6)
                elif v >= 0.03:
                    ax.text(
                        j, i, f"{v:.2f}"[1:], ha="center", va="center", fontsize=5.5, color=cell_text_color(v, vmax)
                    )
            for i in np.flatnonzero(np.isnan(m[:, 0])):
                ax.axhspan(i - 0.5, i + 0.5, facecolor="none", edgecolor=light_dark("0.7", "0.3"), hatch="///", lw=0)
                # The column of a dropped op stays: it is a baseline, the mass that lands on its colors by chance.
                ax.axvspan(
                    i - 0.5,
                    i + 0.5,
                    facecolor="none",
                    edgecolor=light_dark("0.7", "0.3"),
                    hatch="/",
                    lw=0,
                    alpha=light_dark(0.4, 0.5),
                )
            ax.set_title(s, fontsize=9)
            ax.set_xticks(range(n), ALL_OPS, rotation=90, fontsize=6.5)
            ax.set_yticks(range(n), ALL_OPS, fontsize=6.5)
        fig.supxlabel("answers of this op, beyond those of the true op", fontsize=9)
        fig.supylabel("true op", fontsize=9)
        assert im is not None
        fig.colorbar(im, ax=axes, shrink=0.5, label="mass")
        return fig

    return _plot()


@memo
def scores_draw(alt_text: str, caption: str) -> str:
    @themed(name="scores", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), layout="constrained")
        axes = cast(AxesRow, axes)
        x = np.arange(len(SETS))
        ax = axes[0]
        ax.plot(x, [score(s, "ceiling") for s in SETS], "_", ms=22, mew=2, color=rule_color(), label="Bayes ceiling")
        ax.plot(x, [score(s, "floor") for s in SETS], "_", ms=22, mew=1, color="0.6", label="floor")
        ax.plot(x, [score(s) for s in SETS], "o", color="C0", label="model")
        ax.axhspan(min(YARD_EEM), max(YARD_EEM), color="C0", alpha=0.15, lw=0, label="ex-2.2.17, three seeds")
        ax.set_ylabel("held-out EEM")
        ax.set_ylim(0, 0.7)
        fig.legend(loc="outside upper center", ncols=4, frameon=False, fontsize=8)
        ax = axes[1]
        ax.bar(x, [gap(s) for s in SETS], color="C0", width=0.6)
        ax.axhspan(min(YARD_GAP), max(YARD_GAP), color="C0", alpha=0.15, lw=0)
        ax.set_ylabel("gap to own ceiling")
        ax = axes[2]
        ax.bar(x, [skill(s) for s in SETS], color="C0", width=0.6)
        ax.axhspan(min(YARD_SKILL), max(YARD_SKILL), color="C0", alpha=0.15, lw=0)
        ax.set_ylabel("skill")
        ax.set_ylim(0, 1)
        for a in axes:
            a.set_xticks(x, SETS, rotation=30, ha="right", fontsize=8)
        return fig

    return _plot()


@memo
def gaps_draw(alt_text: str, caption: str) -> str:
    @themed(name="gaps", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        m = np.full((len(ALL_OPS), len(SETS)), np.nan)
        for j, s in enumerate(SETS):
            for op in ops_of(s):
                m[ALL_OPS.index(op), j] = gap(s, op)
        fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
        lim = float(np.nanmax(m))
        im = ax.imshow(m, cmap=seq_cmap(), vmin=0, vmax=lim, aspect="auto")
        for (i, j), v in np.ndenumerate(m):
            if np.isnan(v):
                ax.text(j, i, "dropped", ha="center", va="center", fontsize=6, color="0.5")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=cell_text_color(v, lim))
        ax.set_xticks(range(len(SETS)), SETS, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(ALL_OPS)), ALL_OPS, fontsize=8)
        fig.colorbar(im, ax=ax, label="gap to own ceiling (EEM)", shrink=0.8)
        return fig

    return _plot()


@memo
def traj_draw(alt_text: str, caption: str) -> str:
    @themed(name="trajectories", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")
        for i, s in enumerate(SETS):
            x, y = np.array(TRAJ[s]["step"]) / 1e3, probe_skill(s)
            ax.plot(x, y, color=f"C{i}", lw=2 if s == "full" or s in FOURS else 1.1, label=s)
            for share, face in ((0.9, f"C{i}"), (0.95, "none")):
                k = crossing(s, share)
                ax.plot(x[k], y[k], "o", ms=4.5, mfc=face, mec=f"C{i}", mew=1.1, zorder=3)
        ax.axhline(1, ls="--", color=rule_color(), lw=1)
        ax.set_xlabel("step (thousands)")
        ax.set_ylabel("skill on the probe set")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, frameon=False, ncols=2)
        return fig

    return _plot()


# --- Derived measurements --------------------------------------------------------------------------------------

HSV_CHANNEL = ("hue-hsv", "sat-hsv", "value-hsv")
BEST_PRIOR = max(YARD_EEM)
FULL, FOUR = "full", "no-four"
# The trajectory points are evenly spaced, so index LAST_FIFTH is 80% of the way through training.
LAST_FIFTH = (len(TRAJ[FULL]["step"]) - 1) * 4 // 5


def steps_to(s: str, share: float) -> int:
    """The first logged step at which the probe skill reached *share* of its final value."""
    return int(TRAJ[s]["step"][crossing(s, share)])


def late_gain(s: str) -> float:
    """Probe EEM gained over the last fifth of training."""
    return TRAJ[s]["eem"][-1] - TRAJ[s]["eem"][LAST_FIFTH]


def hsv_at(s: str, i: int) -> float:
    """Probe EEM on the three HSV-channel ops, averaged, at trajectory point *i*."""
    e = np.array(TRAJ[s]["eem_per_op"])[i]
    return float(np.mean([e[ops_of(s).index(op)] for op in HSV_CHANNEL]))


STEPS = TRAJ[FULL]["step"][-1]
EARLY = 10

LD = "no-four-ld"


def hsv_gap(s: str) -> float:
    """The mean gap over the three HSV-channel ops."""
    return float(np.mean([gap(s, op) for op in HSV_CHANNEL]))


# A run falls short on the HSV-channel ops when their mean gap is more than 0.1, against 0.06 in the full-set run.
HSV_SHORT = tuple(s for s in SETS if hsv_gap(s) > 0.1)
NO_HSVMIX = tuple(s for s in SETS if "hsvmix" not in ops_of(s))
WITH_HSVMIX = tuple(s for s in SETS if "hsvmix" in ops_of(s))


WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def listed(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f", and {items[-1]}"


def names(sets: Sequence[str]) -> str:
    return listed(f"`{s}`" for s in sets)


rf"""
# Ex 2.2.18: dropping ops with similar answers, a scout

/// tip |
<!-- tl;dr -->
Dropping `screen`, `multiply`, `hsvmix`, and `exclusion` together made the in-context grammar easier to solve, and the model got closer to what is solvable than in any run so far. Most of that came from dropping ops whose answers round at random. Dropping `lighten` and `darken` in place of `screen` and `multiply` breaks the same pairs, and it left the ceiling where it was and narrowed the gap by less.
///

In ex-2.2.17, the center control kept part of its mass on the answers of a similar op, even on contexts where the examples should settle the op. Four pairs of ops stood out: `lighten` with `screen`, `darken` with `multiply`, `mix` with `hsvmix`, and `difference` with `exclusion`. This scout drops one op from each pair, first one at a time and then all four together. A second round drops the other op of the first two pairs: `lighten` alone, `darken` alone, and both of them in place of `screen` and `multiply` in the four-op drop. Each op set trains the ex-2.2.17 recipe once (a single seed).

Dropping ops changes the task, so every op set has its own Bayes ceiling and floor, and we score each run against its own.

## Observations

Each item below is a measurement on the runs of this scout, with no gate.

- **E1** [Scores against each ceiling](#scores-against-each-ceiling-e1): dropping `lighten` or `darken` lowered the ceiling, and dropping any of the other four raised it. Without `screen`, `multiply`, `hsvmix`, and `exclusion`, the model came within {gap(FOUR):.3f} of its ceiling, closer than any run so far. The second four-op drop came within {gap(LD):.3f} of a ceiling near that of the full set.
- **E2** [The HSV-channel ops](#the-hsv-channel-ops-e2): {WORDS[len(HSV_SHORT)]} of the {WORDS[len(SINGLES)]} single drops fell short on the three HSV-channel ops, and the {WORDS[len(NO_HSVMIX)]} runs without `hsvmix` learned those ops earlier than the others did.
- **E3** [The leak onto a dropped op](#the-leak-onto-a-dropped-op-e3): on contexts of a partner op, the mass the model put on the answers of the dropped op mostly went away with it.
- **E4** [Training time](#training-time-e4): every run reached 95% of its final skill by {max(steps_to(s, 0.95) for s in SETS) / STEPS:.0%} of the way through training, and the last fifth of training added at most {max(late_gain(s) for s in SETS):.3f}.

## Scope

This is a scout, with no preregistration and no gate. Each op set has one run, all from the same model seed, so a difference between two runs is only a hint. For scale, ex-2.2.17 trained this recipe on the full op set at three seeds, and their gaps to the ceiling spanned {min(YARD_GAP):.3f} to {max(YARD_GAP):.3f}.

## Why

The in-context grammar asks the model to infer the op from three examples, and some pairs of ops give the same answer on many operand pairs. An example that fits `lighten` often fits `screen` too, so those examples say less about the op, and the model has two nearly interchangeable answers to choose between. If the pairs are part of why the control falls short of its ceiling, a smaller op set might make a better grammar for the anchoring experiments.

It may matter which op of a pair is dropped, too. `lighten`, `darken`, and `difference` give one answer for each pair of operands, while their partners round each channel at random between grid levels, so their answers are spread over a few colors. Dropping an op with spread-out answers raises the ceiling whether or not similarity matters. Dropping its partner instead breaks up the same pair without raising the ceiling, so comparing the two tells the effect of similarity apart from the effect of rounding.

## The runs

Every run uses the recipe that ex-2.2.17 settled on: the unanchored d64-L4 model with an untied readout and the newline mask, a cosine schedule at a peak learning rate of {ex.PEAK_LR:g} after a warm-up of {ex.WARMUP_EPOCHS:g} epochs, for {ex.EPOCHS} epochs ({STEPS:,.0f} steps). The corpus condition is `k3-r0.3`, three examples and a replacement rate of 0.3, with {ex.ex2216.N_LINES:,} contexts. Each op set gets its own corpus, holdout, and probe set, drawn from its own ops.

"""

table_html(
    ["op set", "ops dropped", "partner kept", "ops"],
    [
        [
            f"`{s}`",
            ", ".join(f"`{o}`" for o in RUNS[s]["dropped"]) or "none",
            ", ".join(f"`{PAIR_OF[o]}`" for o in RUNS[s]["dropped"]) or "",
            str(len(ops_of(s))),
        ]
        for s in SETS
    ],
    """
        **The op sets.** Each dropped op has a partner, the op ex-2.2.17 found its answers most often coincide with.
        The `no-lighten`, `no-darken`, and `no-four-ld` sets were a second round.
    """,
    text_cols=3,
)

r"""

## The measurements

Held-out *expected exact match* (EEM) is the probability that an answer drawn from the distribution of the model at the query `=` is a correct answer of the true op. The *Bayes ceiling* is the same score for an ideal predictor, which weighs each op by how well it explains the examples and answers with the resulting mixture. The *floor* is the score of a predictor that ignores the examples. Both are computed for each op set on its own ops, as in ex-2.2.16.

The *gap* is the ceiling minus EEM, and *skill* is how far a run got from the floor to the ceiling, as a fraction of that distance. Gaps compare runs whose ceilings differ, which EEM alone cannot do.

## Scores against each ceiling (E1)

The figure below puts each run beside the ceiling and floor of its op set.

"""

scores_draw(
    f"""
        Three charts over nine op sets. Left: held-out expected exact match, with a ceiling mark and a floor mark per op
        set. The ceilings run from {min(score(s, "ceiling") for s in SETS):.2f} (no-darken) to
        {score(FOUR, "ceiling"):.2f} (no-four), and the model dots sit a little under each ceiling, further below for
        no-multiply and no-darken. The full-set dot at {score(FULL):.3f} falls inside the shaded band of ex-2.2.17's
        three seeds. Right: the gap to the ceiling as bars, between {gap(FOUR):.3f} (no-four) and
        {gap("no-multiply"):.3f} (no-multiply), with the ex-2.2.17 band from {min(YARD_GAP):.3f} to
        {max(YARD_GAP):.3f}; no-four and no-four-ld are the two shortest bars. Right: skill as bars, highest for no-four
        at {skill(FOUR):.2f} and lowest for no-multiply at {skill("no-multiply"):.2f}.
    """,
    """
        **Scores against each ceiling.** Left: held-out expected exact match of each run (dots), with the Bayes ceiling
        (dark) and floor (light) of its op set. Middle: the gap between them. Right: skill, how far each run got from
        the floor to the ceiling. The shaded band on each is the range of ex-2.2.17's three seeds on the full op set.
    """,
)

rf"""

The full-set run scored {score(FULL):.3f}, inside the range of ex-2.2.17's seeds ({min(YARD_EEM):.3f} to {max(YARD_EEM):.3f}), so the recipe reproduced.

Dropping one of `screen`, `multiply`, `hsvmix`, or `exclusion` raised the ceiling by 0.02 to 0.03, and dropping `lighten` or `darken` lowered it by about 0.015. That follows from which side rounds at random. A predictor told the op, with no inference to do, would score {STATS[FULL]["told_op"]:.3f} on the full set: it can't always name the answer of an op that rounds at random, but it always can for `lighten` or `darken`. Without `screen` it would score {STATS["no-screen"]["told_op"]:.3f}, and without `lighten` {STATS["no-lighten"]["told_op"]:.3f}.

The gaps of the single drops mostly stayed near the gap of the full set, {gap(FULL):.3f}. The exceptions were `no-multiply` at {gap("no-multiply"):.3f} and `no-darken` at {gap("no-darken"):.3f}, the two sides of the darkening pair, and `no-exclusion` at {gap("no-exclusion"):.3f}; E2 shows where those fell short.

The two four-op drops break the same four pairs. Without `screen`, `multiply`, `hsvmix`, and `exclusion`, the ceiling rose to {score(FOUR, "ceiling"):.3f} and the model reached {score(FOUR):.3f}, a gap of {gap(FOUR):.3f}. Without `lighten`, `darken`, `hsvmix`, and `exclusion`, the ceiling stayed near that of the full set, at {score(LD, "ceiling"):.3f}, and the model reached {score(LD):.3f}, a gap of {gap(LD):.3f}. So most of the rise in EEM from `no-four` came from its higher ceiling.

Both four-op drops narrowed the gap, by {gap(FULL) - gap(FOUR):.3f} and {gap(FULL) - gap(LD):.3f}. But each op set has a single run, and the gaps of the three ex-2.2.17 seeds on the full set spanned {min(YARD_GAP):.3f} to {max(YARD_GAP):.3f}, a range about as wide as either change. So these narrower gaps could come from seed variation alone, and it would take more seeds to tell.

## The HSV-channel ops (E2)

The heatmap below breaks each gap down by op.

"""

gaps_draw(
    f"""
        A heatmap with the eleven ops as rows and the nine op sets as columns, each square the gap between the Bayes
        ceiling and the model on that op, darker for a larger gap, with dropped ops marked. Most squares sit between
        0.03 and 0.15. The no-multiply, no-darken, and no-exclusion columns are darkest on hue-hsv, sat-hsv, and
        value-hsv, up to {max(gap(s, op) for s in HSV_SHORT for op in HSV_CHANNEL):.2f}. The no-four and no-four-ld
        columns are the palest.
    """,
    """
        **The gap by op.** Each square is the Bayes ceiling minus held-out expected exact match, for one op in one run.
    """,
)

rf"""

Three single drops fell short on the three HSV-channel ops: {names(HSV_SHORT)}, with a mean gap on those ops of {listed(f"{hsv_gap(s):.2f}" for s in HSV_SHORT)}, against {hsv_gap(FULL):.2f} in the full-set run. The HSV-channel ops are blend modes that take one of hue, saturation, or value from one operand and the other two from the other. None of the dropped ops gives answers like theirs, so similarity doesn't explain this.

The learning curves suggest timing. Ex-2.2.17 saw these ops rise steeply partway through training, and in this scout the rise came at different times in different runs. At step {TRAJ[FULL]["step"][EARLY]:,.0f}, the mean probe EEM on the three ops was {listed(f"{hsv_at(s, EARLY):.2f}" for s in NO_HSVMIX)} in the {WORDS[len(NO_HSVMIX)]} runs without `hsvmix`, and between {min(hsv_at(s, EARLY) for s in WITH_HSVMIX):.2f} and {max(hsv_at(s, EARLY) for s in WITH_HSVMIX):.2f} in the others. The runs that fell short had not caught up when the schedule ended.

With one seed per op set, we can't tell whether dropping `multiply`, `darken`, or `exclusion` makes those ops harder to learn, or these runs were slow ones. Dropping an op also changes the corpus, since the other ops share its contexts, so each gets a little more training. The early start without `hsvmix` is firmer: three runs share it, though two of them also drop three other ops.

Away from the HSV-channel ops, both four-op columns are lower than the full set on nearly every op. `difference` went from {gap(FULL, "difference"):.2f} to {gap(FOUR, "difference"):.2f} and {gap(LD, "difference"):.2f}, and `mix` from {gap(FULL, "mix"):.2f} to {gap(FOUR, "mix"):.2f} and {gap(LD, "mix"):.2f}. In `no-four`, `lighten` and `darken` went from {gap(FULL, "lighten"):.2f} and {gap(FULL, "darken"):.2f} to {gap(FOUR, "lighten"):.2f} and {gap(FOUR, "darken"):.2f}.

## The leak onto a dropped op (E3)

Ex-2.2.17 found the model keeping mass on the answers of a similar op even where the examples should settle the op. To see where that mass goes, we take the held-out contexts whose posterior on the true op is above {CONFIDENT:g}, and for each other op measure the mass the model puts on colors that op can give and the true op cannot. The figure below shows this as one matrix per op set, the same measurement as the op confusion of ex-2.2.17. The Bayes predictive puts almost nothing off the diagonal on these contexts.

"""

confusion_draw(
    f"""
        Nine small heatmaps in a three-by-three grid, one per op set, each with the eleven true ops as rows and the
        eleven ops as columns; rows of dropped ops are densely hatched, their columns
        faintly hatched over the values, and the diagonal is left blank. Most squares are pale.
        In the full set, four squares stand out: lighten onto screen ({conf(FULL, "lighten", "screen"):.2f}), darken
        onto multiply ({conf(FULL, "darken", "multiply"):.2f}), hsvmix and mix onto each other
        ({conf(FULL, "hsvmix", "mix"):.2f} and {conf(FULL, "mix", "hsvmix"):.2f}), and difference onto exclusion
        ({conf(FULL, "difference", "exclusion"):.2f}). In each drop, the faintly hatched column of the dropped op is as pale as its neighbors. The darkest squares
        are in no-multiply, where sat-hsv and value-hsv put {conf("no-multiply", "sat-hsv", "value-hsv"):.2f} and
        {conf("no-multiply", "value-hsv", "sat-hsv"):.2f} onto each other.
    """,
    """
        **Where the mass goes, by op.** One matrix per op set, on confident held-out contexts. Rows are the true op. Densely
        hatched rows are dropped ops, which have no contexts; their columns, faintly hatched, still count the colors
        those ops would give. Off the diagonal, each square is the mass the model puts
        on colors the column op can give and the true op cannot, with values of 0.03 and above printed. The diagonal
        is outlined and left blank.
    """,
)

rf"""

In the full set, most of the mass off the diagonal sits in the squares of the four pairs. Confident `lighten` contexts put {leak(FULL, "screen"):.2f} of their mass on colors only `screen` gives, and `darken`, `mix`, and `difference` contexts put {listed(f"{leak(FULL, d):.2f}" for d in ("multiply", "hsvmix", "exclusion"))} on colors only their partner gives. The leak is lopsided: confident `screen` contexts put {leak(FULL, "lighten"):.3f} on colors only `lighten` gives. Part of that is in how it is counted. `lighten` gives one color, often one of the colors `screen` gives, so there are fewer colors for `screen` contexts to leak onto.

A dropped op keeps its column. The model answers with colors and never names an op, so the colors `screen` would give are still colors the model can put mass on, whether or not it trained on `screen`. What changes is why mass lands there. In a run that drops an op, nothing it learned points at those colors, so its column is a baseline: the mass that falls on some set of plausible colors through the general spread of the answers. It is small. On the partner row, the leak onto a dropped op is at most {max(leak(s, d) for s in SETS[1:] for d in RUNS[s]["dropped"] if PAIR_OF[d] in ops_of(s)):.3f}, about the level of the unrelated ops on the same row, whose median square runs from {min(unrelated(s, PAIR_OF[d], d) for s in SETS[1:] for d in RUNS[s]["dropped"] if PAIR_OF[d] in ops_of(s)):.3f} to {max(unrelated(s, PAIR_OF[d], d) for s in SETS[1:] for d in RUNS[s]["dropped"] if PAIR_OF[d] in ops_of(s)):.3f} across these runs.

That gives a scale for the rest of the matrix. A square means something only where it stands clear of this background, so `lighten` onto `screen` at {leak(FULL, "screen"):.2f} in the full set is mostly the pairing, with about {leak("no-screen", "screen"):.3f} of it expected without `screen` at all, while squares near 0.01 are background.

Dropping an op also makes more contexts confident: without `screen`, {n_conf("no-screen", "lighten")} held-out `lighten` contexts are confident, against {n_conf(FULL, "lighten")} in the full set.

The matrices also show where the HSV-channel ops of E2 fell short. In `no-multiply`, confident `sat-hsv` and `value-hsv` contexts put {conf("no-multiply", "sat-hsv", "value-hsv"):.2f} and {conf("no-multiply", "value-hsv", "sat-hsv"):.2f} of their mass on the answers of each other, and the other runs that fell short spread smaller amounts over several ops.

## Training time (E4)

To see whether a shorter schedule could suffice, we logged how skill on the probe set grew through training. The figure below shows each run.

"""

traj_draw(
    f"""
        A line chart of skill on the probe set against training step, one line per op set, with a dashed line at 1
        for the ceiling. Every line rises quickly at first and levels off in the last fifth of training. No-four ends
        highest, near {probe_skill(FOUR)[-1]:.2f}, and no-multiply lowest, near {probe_skill("no-multiply")[-1]:.2f}.
    """,
    """
        **Skill through training.** Skill is how far a run got from the floor to the ceiling of its op set, measured
        on a probe set of 200 contexts per op every 2% of training. Filled dots mark where each run passed 90% of its
        final skill, and open dots 95%. The full set and the two four-op sets are drawn heavier.
    """,
)

r"""

The dots on each line mark where the run passed 90% and 95% of its final skill, and the table below gives those steps.

"""

table_html(
    ["op set", "final probe skill", "step at 90%", "step at 95%", "EEM gained in the last fifth"],
    [
        [
            f"`{s}`",
            f"{probe_skill(s)[-1]:.2f}",
            f"{steps_to(s, 0.9):,}",
            f"{steps_to(s, 0.95):,}",
            f"{late_gain(s):.3f}",
        ]
        for s in SETS
    ],
    f"""
        **Training time.** The first logged step at which probe skill reached 90% and 95% of its final value, out of
        {STEPS:,.0f}, and the probe expected exact match gained over the last fifth of training.
    """,
)

rf"""

Every run reached 90% of its final skill between step {min(steps_to(s, 0.9) for s in SETS):,} and {max(steps_to(s, 0.9) for s in SETS):,}, and the last fifth added between {min(late_gain(s) for s in SETS):.3f} and {max(late_gain(s) for s in SETS):.3f} of EEM. These are on the cosine schedule, where the rate falls to 1% of its peak by the end, so a shorter run would anneal sooner too and might not lose even that much. Ex-2.2.17 saw the same shape.

## What we make of it

The op set without `screen`, `multiply`, `hsvmix`, and `exclusion` is easier (its ceiling is higher) and the model gets closer to it, at a skill of {skill(FOUR):.2f} against {skill(FULL):.2f} for the full set. The higher ceiling comes from dropping ops that round at random. The smaller gap came with both four-op drops, so breaking up the similar pairs may account for it, though one run each is not enough to be sure. Dropping `screen` and `multiply` rather than `lighten` and `darken` leaves seven ops that are easier to tell apart. Adopting that set would also change the ceiling that ex-2.2.16 and ex-2.2.17 were scored against, so their results would not carry over as they are.

The single drops say less. Dropping one op mostly moved the ceiling and the model together, except in the three runs that fell short on the HSV-channel ops; with one seed, that may be timing. The more consistent sign is that all three runs without `hsvmix` learned the HSV-channel ops early, which fits ex-2.2.17 finding `hsvmix` the hardest op to compute.

## Method

**Corpora.** Each op set has its own corpus of 300,000 contexts, its own holdout of {ex.ex2216.HOLDOUT_CONTEXTS:,} contexts per op, and its own probe set of 200 per op, drawn with the sampler of ex-2.2.16 from the answer table restricted to its ops. The posterior, ceiling, and floor are ex-2.2.16's, computed on the same restricted table.

**Leak.** The colors a dropped op would give come from the full eleven-op answer table, so its column is defined in runs that never trained on it, and serves there as a baseline for the other columns. Confident contexts are chosen on the posterior of the op set of the run, so they differ between runs.

**Cost.** The scout cost about \$2.42 on Modal, \$2.30 of it L4 time: about \$0.26 per training run. Each training run took 14 to 17 minutes on one L4, about 7,000 steps a minute.
"""
