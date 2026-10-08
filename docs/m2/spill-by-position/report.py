# ruff: noqa: B018
# title: Where the spill of the edit comes from

# A re-analysis of ex-2.2.23's checkpoints, with no new training. `experiment.py` beside this script applies the
# edit to one set of positions at a time on every run of ex-2.2.23 and publishes the per-op task scores; this script
# nets each anchored run against the control at the same seed and length, as ex-2.2.23 did, and compares the sets.
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, smooth_step, themed

ex23 = ex.ex2223

# --- The stored results ---------------------------------------------------------------------------------------


def fetch_json(refs: list[str]) -> list[Any]:
    store = project_store()
    found = store.get_refs(refs)
    missing = [r for r, a in found.items() if a is None]
    assert not missing, f"not published: {missing}"
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(found[r], Path(tmp) / f"{i}.json") for i, r in enumerate(refs)])
        return [json.loads(p.read_text()) for p in paths]


RESULTS, EVAL23 = fetch_json([ex.RESULTS_REF, ex23.EVAL_REF])
DESIGN = RESULTS["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
HSV = [OPS.index(o) for o in ex23.HSV_OPS]
FULL = max(DESIGN["dose_gammas"])
DOSES: list[float] = DESIGN["dose_gammas"]
N_READ: int = DESIGN["n_read"]
SEPARATORS = [p for p, r in enumerate(DESIGN["roles"]) if r.startswith(",")]
LENGTHS = ex23.LENGTHS

# A run: (condition, epochs, model seed).
Key = tuple[str, int, int]
BY_KEY: dict[Key, dict] = {(r["condition"], r["epochs"], r["model_seed"]): r for r in RESULTS["runs"]}
EVALS = {r["label"]: r for r in EVAL23["runs"]}


def kept(key: Key) -> bool:
    """Whether ex-2.2.23's rule keeps the run: its worst HSV op ends at or above the rule level."""
    per_op = np.array(EVALS[ex23.label_of(*key)]["task"]["eem"]["per_op"])
    measure, level = ex23.RULE
    assert measure == "hsv_min"
    return bool(per_op[HSV].min() >= level)


def measurements(key: Key) -> dict[tuple[str, float], dict[str, Any]]:
    """Per edit, on one anchored run, net of the control at the same seed and length: the removal (the share of the
    way to the target null that the drop on the anchored op gets), the spill (the largest drop on any other op), and
    the op the spill lands on.
    """
    a, c = BY_KEY[key], BY_KEY[("control", *key[1:])]

    def drops(r: dict) -> dict[tuple[str, float], np.ndarray]:
        clean = np.array(r["clean"])
        return {(e["edit"], e["dose"]): clean - np.array(e["eem"]) for e in r["edits"]}

    da, dc = drops(a), drops(c)
    gap = a["clean"][D] - a["null"][D]
    out = {}
    for k in da:
        net = da[k] - dc[k]
        worst = OTHER[int(np.argmax(net[OTHER]))]
        out[k] = {"removal": float(net[D] / gap), "spill": float(net[worst]), "onto": OPS[worst]}
    return out


ANCHORED: list[Key] = [k for k in BY_KEY if k[0] == "anchor"]
KEPT: list[Key] = [k for k in ANCHORED if kept(k)]
LEFT_OUT: list[Key] = [k for k in ANCHORED if not kept(k)]
M = {k: measurements(k) for k in KEPT}

SEP_LEVEL = 0.25
# A run "holds the op at a separator" when editing one separator alone, at full dose, takes the drop on `difference`
# at least this share of the way to the target null. Each example answer alone takes it about a fifth of the way, so
# the level sits above what any example answer does on its own.


def sep_removal(key: Key) -> float:
    return max(M[key][(f"position {p}", FULL)]["removal"] for p in SEPARATORS)


SEP_RUNS = [k for k in KEPT if sep_removal(k) >= SEP_LEVEL]


def runs_at(epochs: int, sep: bool | None = None) -> list[Key]:
    return [k for k in KEPT if k[1] == epochs and (sep is None or (k in SEP_RUNS) == sep)]


def values(keys: list[Key], edit: str, dose: float = FULL, what: str = "removal") -> np.ndarray:
    return np.array([M[k][(edit, dose)][what] for k in keys])


def mean(keys: list[Key], edit: str, dose: float = FULL, what: str = "removal") -> float:
    return float(values(keys, edit, dose, what).mean())


SET_NAMES = ["every position", "example answers", "query operands", "query =", "rest"]

# --- Drawing --------------------------------------------------------------------------------------------------


def rule_color() -> str:
    return light_dark("#333", "#ddd")


def ghost_color() -> str:
    return light_dark("#bbb", "#555")


SET_STYLE = {
    "every position": (light_dark("#333", "#ddd"), "s"),
    "example answers": (light_dark("#c0392b", "#ff8a76"), "o"),
    "rest": (light_dark("#1f6fb2", "#7ab8f5"), "^"),
    "query operands": (light_dark("#2e8b57", "#7fd8a4"), "D"),
    "query =": (light_dark("#8e44ad", "#c39bd3"), "v"),
}


def short_num(v: float) -> str:
    return f"{v:.2f}".replace("-0.00", "0.00")


def cell(v: np.ndarray) -> str:
    return f'{short_num(v.mean())} <span class="range">({short_num(v.min())} to {short_num(v.max())})</span>'


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{c}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


# --- Numbers for the prose ------------------------------------------------------------------------------------

N_KEPT = {e: len(runs_at(e)) for e in LENGTHS}
N_SEP = {e: len(runs_at(e, True)) for e in LENGTHS}
SHORT, LONG = LENGTHS


def num_word(n: int) -> str:
    words = "zero one two three four five six seven eight nine ten eleven twelve".split()
    return words[n] if n < len(words) else str(n)


r"""
# Where the spill of the edit comes from

/// tip |
<!-- lede -->
Ex-2.2.23 found that removing the anchored direction from every position also lowers the task score on other ops. Editing one set of positions at a time shows that, on most runs, the removal of `difference` and the spill onto other ops come from the same place, the example answers, so no set of positions gives one without the other. The other positions add spill of their own.
///
"""

rf"""
The question comes from the [backlog item](/todo/science/spill-by-position.md), split from the withdrawn ex-2.2.24 draft. [Ex-2.2.23](/docs/m2/ex-2.2.23/report.py) (E4) found that the edit spills onto other ops, mostly `darken`, `lighten`, and `value-hsv`, on nearly every run that has learned the HSV ops. This page applies the same edit to one set of positions at a time, on ex-2.2.23's checkpoints, with no new training.

At 400 epochs, {num_word(N_SEP[LONG])} of the {num_word(N_KEPT[LONG])} runs also hold the op at the separator after an example, and on those runs the separator is where most of the spill comes from. There, editing the example answers alone spills much less than editing every position.
"""

# %%

rf"""
## Observations

- [The sets of positions (E1)](#the-sets-of-positions-e1): editing only the example answers removes `{ex.ANCHORED_OP}` as far as editing every position, and on most runs spills about as much, mostly onto `darken` at 400 epochs. The query positions do almost nothing. The rest of the positions spill without removing, mostly onto `lighten`.
- [One position at a time (E2)](#one-position-at-a-time-e2): each example answer alone takes the drop on `{ex.ANCHORED_OP}` about a fifth of the way, and spills a little; no other single position removes anything, except the separators on a few runs at 400 epochs.
- [The runs that use the separator (E3)](#the-runs-that-use-the-separator-e3): on those {num_word(N_SEP[LONG])} runs, editing the first separator alone takes `{ex.ANCHORED_OP}` a third of the way or more and spills heavily. Editing only the example answers there spills a fraction of what the full edit does, though still more than the selectivity criterion of ex-2.2.21 allows.

## Scope

This is an exploratory re-analysis, with no preregistration and no gate. It covers every run of ex-2.2.23: the recipe of record and the control, at 200 and 400 epochs, at model seeds {ex23.SEEDS[0]} to {ex23.SEEDS[-1]}. Every measurement of an anchored run is netted against the control at the same seed and length under the same edit, so what the edit does to a model without the anchor does not count as spill. The scores are on ex-2.2.21's held-out set of {ex23.K}-example contexts, 2,000 per op.

The {num_word(len(LEFT_OUT))} anchored runs that ex-2.2.23's rule leaves out as half-trained (all at 200 epochs) are left out here too: they had not learned the HSV ops, and their edit hardly spilled. That leaves {N_KEPT[SHORT]} runs at 200 epochs and {N_KEPT[LONG]} at 400. Runs differ a lot from seed to seed, so each table gives the seed range beside the seed mean, and the figures draw every run.

## The measurements

The edit is ex-2.2.23's: at every slice, it removes a share (the dose) of the component of the state along e₁, the anchored direction, and rescales the state back onto the sphere. Here it acts on a chosen set of positions and leaves the others as they are. The answer is read at the query `=`, so only the {N_READ} positions up to and including it can matter. The sets are:

- the example answers (*y* in each of the {ex23.K} examples);
- the query operands (*a* and *b* of the query);
- the query `=`;
- the rest: the example operands, every `?` and example `=`, and the separators (`,`) between examples.

Each set is edited at each dose, and so is every position together; each set is also left out of an otherwise full edit, at full dose; and each single position is edited alone, at full dose. Two measurements per edit, both from the expected exact match on held-out contexts:

- *Removal*: how far the drop on `{ex.ANCHORED_OP}` gets toward the target null, as a share of the way. Higher is more removal; 1 means the model answers as if `{ex.ANCHORED_OP}` were not among the ops.
- *Spill*: the largest drop on any other op. Lower is better; ex-2.2.21 (E2) set a selectivity criterion of {ex23.SELECTIVITY_GATE:g}, which we draw for reference.

## Glossary

Target null
:   What the model would answer without `{ex.ANCHORED_OP}`: the ideal predictor with that op removed from the posterior over ops, so it falls back on the ops that fit the examples next best.

Separator
:   The `,` after each example answer, before the next example.
"""

# %%


def tradeoff_data() -> dict:
    out = {}
    for e in LENGTHS:
        ks = runs_at(e)
        out[e] = {
            name: {
                "removal": [mean(ks, name, g) for g in DOSES],
                "spill": [mean(ks, name, g, "spill") for g in DOSES],
                "runs_removal": values(ks, name).tolist(),
                "runs_spill": values(ks, name, what="spill").tolist(),
            }
            for name in SET_NAMES
        }
    return out


def tradeoff_figure() -> str:
    alt = f"""
        Two panels, 200 and 400 epochs, of spill against removal. Each set of positions is a line through its four
        doses, seed mean, with its runs at full dose as faint marks. The lines for every position and for the example
        answers run from near the origin to the right, ending near {mean(runs_at(LONG), "every position"):.1f}
        removal; at 200 epochs they end at about the same spill, and at 400 epochs the every-position line ends higher.
        The query operands and query `=` stay near the origin. The rest rises straight up at 200 epochs with little
        removal; at 400 epochs it also reaches to the right at full dose, from the runs that hold the op at the
        separator. A dashed line marks the selectivity criterion of {ex23.SELECTIVITY_GATE:g}.
    """
    return tradeoff_draw(tradeoff_data(), alt)


@memo
def tradeoff_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="spill-by-position-sets",
        alt_text=alt_text,
        caption=f"""
            **Spill against removal, for each set of positions.** One line per set through the doses
            {", ".join(f"{g:g}" for g in DOSES)}, the seed mean over the runs the rule keeps; the faint marks are
            those runs at full dose. Both measurements are net of the control at the same seed and length. Dashed:
            the selectivity criterion of ex-2.2.21.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained", sharex=True, sharey=True)
        for ax, (e, sets) in zip(axes, data.items(), strict=True):
            ax.axhline(ex23.SELECTIVITY_GATE, color=rule_color(), lw=0.8, ls="--", zorder=1)
            ax.axhline(0, color=rule_color(), lw=0.4, zorder=1)
            for name, d in sets.items():
                color, m = SET_STYLE[name]
                ax.plot(d["runs_removal"], d["runs_spill"], m, ms=3, color=color, alpha=0.3, mew=0, zorder=2)
                ax.plot([0, *d["removal"]], [0, *d["spill"]], "-", color=color, lw=1.2, zorder=3)
                ax.plot(d["removal"], d["spill"], m, ms=4.5, color=color, mec=light_dark("white", "#111"),
                        mew=0.6, zorder=4, label=name)  # fmt: skip
            ax.set_title(f"{e} epochs", fontsize=9)
            ax.set_xlabel(f"removal of {ex.ANCHORED_OP}", fontsize=8)
            ax.set_xlim(-0.12, 1.02)
        axes[0].set_ylabel("spill onto another op", fontsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


def mode_op(keys: list[Key], edit: str) -> str:
    (op, n), *_ = Counter(M[k][(edit, FULL)]["onto"] for k in keys).most_common(1)
    return f"<code>{op}</code> ({n})"


def sets_table() -> str:
    rows = []
    for name in [*SET_NAMES, "all but example answers"]:
        row = [name]
        for e in LENGTHS:
            ks = runs_at(e)
            row += [cell(values(ks, name)), cell(values(ks, name, what="spill")), mode_op(ks, name)]
        rows.append(row)
    head = ["edited"] + [f"{e}: {m}" for e in LENGTHS for m in ("removal ↑", "spill ↓", "spill lands on")]
    return table_html(
        head,
        rows,
        f"""
        **Each set of positions at full dose.** Removal and spill, net of the control at the same seed and length:
        the seed mean over the {N_KEPT[SHORT]} runs at 200 epochs and the {N_KEPT[LONG]} at 400, with the seed range
        in brackets. The last column of each length names the op the spill lands on most often, with its count of
        runs.
        """,
    )


rf"""
## The sets of positions (E1)

Is there a set of positions where the edit removes `{ex.ANCHORED_OP}` without lowering other ops? Each set below is edited at four doses, and the figure plots the spill against the removal as the dose grows. A set that removes without spilling would run along the bottom of the panel to the right.

{tradeoff_figure()}

None does. The removal comes almost entirely from the example answers: editing them alone gets as far as editing every position, a little further on average. The spill comes with it. At 200 epochs the two lines lie nearly on top of each other, and at 400 epochs the example answers spill less than every position, though still far above the criterion on most runs.

The query positions do almost nothing, either way. The rest is the surprise: it spills without removing anything, as its line climbs straight up at 200 epochs. At 400 epochs it also removes on a few runs at full dose, from the separators (E3).

{sets_table()}

The two kinds of spill land on different ops. Editing the example answers lowers `darken` most on most runs at 400 epochs, and editing the rest lowers `lighten`. Leaving the example answers out of the edit keeps the spill of the rest and loses nearly all of the removal at 200 epochs.

So the edit at the example answers removes `{ex.ANCHORED_OP}` and spills in the same act, and the other positions add a spill of their own. Restricting the edit to the example answers would drop the second kind and keep the first.
"""

# %%


def map_data() -> dict:
    out: dict[str, Any] = {"roles": DESIGN["roles"], "panels": {}}
    for e in LENGTHS:
        series = {}
        for sep in (False, True):
            ks = runs_at(e, sep)
            if not ks:
                continue
            series["separator runs" if sep else "other runs"] = {
                what: [mean(ks, f"position {p}", what=what) for p in range(N_READ)] for what in ("removal", "spill")
            }
        ghosts = {
            what: [[M[k][(f"position {p}", FULL)][what] for p in range(N_READ)] for k in runs_at(e)]
            for what in ("removal", "spill")
        }
        out["panels"][e] = {"series": series, "ghosts": ghosts}
    return out


def map_figure() -> str:
    alt = f"""
        A grid of four panels: removal on the top row and spill on the bottom, at 200 epochs on the left and 400 on
        the right. Each panel is a smooth step along the {N_READ} positions of a context, from the first operand to
        the query `=`, with one faint line per run behind the seed means. Removal sits near zero everywhere except at
        the three example answers, where it rises to about a fifth at each. At 400 epochs a second series, the
        {num_word(N_SEP[LONG])} runs that hold the op at the separator, also rises at the separators, highest at the
        first. Spill is small at most positions, with small bumps at the example answers and the query operands; on
        the separator runs it rises steeply at the separators.
    """
    return map_draw(map_data(), alt)


def role_tick(r: str) -> str:
    return {"a": "$a$", "b": "$b$", "y": "$y$"}.get(r.split()[0][0], r[0])


SERIES_STYLE = {
    "other runs": (light_dark("#c0392b", "#ff8a76"), "o"),
    "separator runs": (light_dark("#1f6fb2", "#7ab8f5"), "^"),
}


@memo
def map_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="spill-by-position-map",
        alt_text=alt_text,
        caption=f"""
            **Removal and spill for each position edited alone, at full dose.** Top: removal of
            `{ex.ANCHORED_OP}`; bottom: spill onto another op; both net of the control at the same seed and length.
            The markers are the seed means, over the runs that hold the op at a separator (E3) and over the other
            runs; the faint lines are single runs. The shading marks the example answers.
        """,
    )
    def _plot() -> plt.Figure:
        n = len(data["roles"])
        x = np.arange(n)
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.0), layout="constrained", sharex=True, sharey="row")
        for col, (e, panel) in enumerate(data["panels"].items()):
            for row, what in enumerate(("removal", "spill")):
                ax = axes[row, col]
                for p, r in enumerate(data["roles"]):
                    if r.startswith("y"):
                        ax.axvspan(p - 0.5, p + 0.5, facecolor=light_dark("#000", "#fff"), alpha=0.06, lw=0)
                ax.axvline(n - 4 - 0.5, color=rule_color(), lw=0.6, ls=(0, (2, 2)))
                ax.axhline(0, color=rule_color(), lw=0.4)
                if what == "spill":
                    ax.axhline(ex23.SELECTIVITY_GATE, color=rule_color(), lw=0.7, ls="--")
                for g in panel["ghosts"][what]:
                    smooth_step(ax, x, g, ramp=0.6, color=ghost_color(), lw=0.5, alpha=0.8, zorder=1)
                for name, s in panel["series"].items():
                    color, m = SERIES_STYLE[name]
                    smooth_step(ax, x, s[what], ramp=0.6, color=color, lw=1.2, zorder=3)
                    ax.plot(x, s[what], m, ls="", ms=3.5, color=color, mec=light_dark("white", "#111"), mew=0.5,
                            zorder=4, label=name)  # fmt: skip
                ax.set_xlim(-0.6, n - 0.4)
                if row == 0:
                    ax.set_title(f"{e} epochs", fontsize=9)
                    ax.text(n - 4 - 0.3, 0.97, "query", fontsize=6, color=rule_color(), va="top",
                            transform=ax.get_xaxis_transform())  # fmt: skip
                if col == 0:
                    ax.set_ylabel(what, fontsize=8)
            axes[1, col].set_xticks(x, [role_tick(r) for r in data["roles"]], fontsize=7)
        for ax in axes[0]:
            ax.set_ylim(-0.1, 0.85)
        for ax in axes[1]:
            ax.set_ylim(-0.03, 0.6)
        handles, labels = axes[0, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


ANSWER_POS = [p for p, r in enumerate(DESIGN["roles"]) if r.startswith("y")]
ONE_ANSWER = float(np.mean([mean(runs_at(e), f"position {p}") for e in LENGTHS for p in ANSWER_POS]))

rf"""
## One position at a time (E2)

The sets in E1 lump positions of different roles together. Editing each position alone, at full dose, shows where within a set the removal and the spill sit, and whether the spill of the rest is spread over many positions or comes from a few.

{map_figure()}

Each example answer alone takes the drop on `{ex.ANCHORED_OP}` about a fifth of the way, and the three are alike. Together they take it further than their sum ({mean(runs_at(LONG), "example answers"):.2f} at 400 epochs, against three times {ONE_ANSWER:.2f}), so when one answer is edited, the other two make up part of the loss. No other position removes anything, except the separators on the runs of E3.

The spill is spread thinly. Each example answer spills a little, and the operands, those of the query most, spill a little too. Few of them pass the criterion on their own, but together they do: the spill of the rest in E1 comes from many positions with a small spill each, and from the separators on a few runs. So there is no single position to leave out of the edit that would take the spill away.
"""

# %%


def sep_rows() -> list[dict]:
    rows = []
    for k in sorted(SEP_RUNS, key=lambda k: (k[1], k[2])):
        best = max(SEPARATORS, key=lambda p: M[k][(f"position {p}", FULL)]["removal"])
        rows.append(
            {
                "key": k,
                "best": best,
                "cells": {
                    name: M[k][(edit, FULL)]
                    for name, edit in (
                        ("separator", f"position {best}"),
                        ("answers", "example answers"),
                        ("every", "every position"),
                    )
                },
            }
        )
    return rows


def sep_table() -> str:
    rows = []
    for r in sep_rows():
        k = r["key"]
        row = [f"{k[2]}", f"{k[1]}", DESIGN["roles"][r["best"]].replace(",", "after example ")]
        for name in ("separator", "answers", "every"):
            c = r["cells"][name]
            row += [short_num(c["removal"]), f"{short_num(c['spill'])} <code>{c['onto']}</code>"]
        rows.append(row)
    head = ["seed", "epochs", "separator"] + [
        f"{w}: {m}" for w in ("separator", "example answers", "every position") for m in ("removal ↑", "spill ↓")
    ]
    return table_html(
        head,
        rows,
        f"""
        **The runs that hold the op at a separator.** For each, the separator whose edit removes most, and three
        edits at full dose: that separator alone, the example answers, and every position. Removal, and spill with
        the op it lands on, net of the control at the same seed and length.
        """,
        text_cols=3,
    )


def scatter_data() -> list[dict]:
    return [
        {
            "epochs": k[1],
            "sep": k in SEP_RUNS,
            "answers": M[k][("example answers", FULL)]["spill"],
            "every": M[k][("every position", FULL)]["spill"],
        }
        for k in KEPT
    ]


def scatter_figure() -> str:
    alt = f"""
        A scatter of the spill under the full edit against the spill under the example-answer edit, one mark per run,
        circles at 200 epochs and triangles at 400, with the {num_word(len(SEP_RUNS))} runs that hold the op at a
        separator in a second color. A diagonal marks equal spill. Most runs sit near the diagonal, spread from near
        zero to about 0.3. The separator runs sit well above it: low spill under the example-answer edit and high under
        the full edit.
    """
    return scatter_draw(scatter_data(), alt)


@memo
def scatter_draw(rows: list[dict], alt_text: str) -> str:
    @themed(
        name="spill-by-position-separator",
        alt_text=alt_text,
        caption=f"""
            **Spill under the full edit against spill under the example-answer edit**, one mark per run at full
            dose, net of the control at the same seed and length. Circles: 200 epochs; triangles: 400. Blue: the
            runs that hold the op at a separator. Dotted: equal spill; dashed: the selectivity criterion of ex-2.2.21.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(3.6, 3.3), layout="constrained")
        hi = max(max(r["every"], r["answers"]) for r in rows) * 1.08
        ax.plot([0, hi], [0, hi], ":", color=rule_color(), lw=0.8)
        ax.axhline(ex23.SELECTIVITY_GATE, color=rule_color(), lw=0.7, ls="--")
        ax.axvline(ex23.SELECTIVITY_GATE, color=rule_color(), lw=0.7, ls="--")
        for r in rows:
            color = SERIES_STYLE["separator runs" if r["sep"] else "other runs"][0]
            m = "o" if r["epochs"] == SHORT else "^"
            ax.plot(r["answers"], r["every"], m, ms=5, color=color, mec=light_dark("white", "#111"), mew=0.5)
        ax.set_xlim(-0.02, hi)
        ax.set_ylim(-0.02, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("spill, example answers edited", fontsize=8)
        ax.set_ylabel("spill, every position edited", fontsize=8)
        return fig

    return _plot()


SEP_LONG = runs_at(LONG, True)
OTHER_LONG = runs_at(LONG, False)

rf"""
## The runs that use the separator (E3)

At 400 epochs, the separators remove `{ex.ANCHORED_OP}` on some runs and not on others. Which runs, and what does that do to the spill? We mark a run as holding the op at a separator when editing one separator alone takes the drop on `{ex.ANCHORED_OP}` at least {SEP_LEVEL:.0%} of the way, more than any one example answer does. That marks {num_word(N_SEP[LONG])} of the {num_word(N_KEPT[LONG])} runs at 400 epochs and {num_word(N_SEP[SHORT])} at 200.

{sep_table()}

On these runs the first separator usually does most: alone, it takes the drop on `{ex.ANCHORED_OP}` a third of the way or more, and its spill comes close to that of the full edit. Editing only the example answers removes as much as ever, and spills a fraction of what the full edit does.

{scatter_figure()}

The figure puts the two kinds of run side by side. Most runs sit on the diagonal: the spill is where the removal is, and leaving positions out of the edit changes little. The separator runs sit well above it, because their spill under the full edit comes mostly from the separators. Even there, the example-answer edit spills more than the criterion allows (from {min(values(SEP_LONG, "example answers", what="spill")):.2f} to {max(values(SEP_LONG, "example answers", what="spill")):.2f}).

We can't say from this pass why some runs come to use the separator. It is the position right after an answer, so it can see the whole example, and the whole-line label pulls every position of a `{ex.ANCHORED_OP}` context toward e₁; a run that settles there may simply be one of the paths the label allows. All the separator runs are at 400 epochs, which fits a hold that forms late, though {num_word(N_SEP[LONG])} runs are too few to say so with confidence.
"""

# %%

rf"""
## Discussion

The edit removes `{ex.ANCHORED_OP}` at the example answers, and on most runs that is where it spills too, by an amount that grows with the removal. So the spill is part of what e₁ holds at the example answers, beside the op. [What the anchor follows at the example answers](/docs/m2/example-evidence/report.py) found that the alignment there follows how well each example fits `{ex.ANCHORED_OP}` on its own. A judgement like that, made one example at a time, would also fire on examples of other ops whose answers happen to fit, and it may carry what makes an answer fit, such as how dark it is: `{ex.ANCHORED_OP}` answers are darker than their operands on average, and the spill at the example answers lands mostly on `darken`. This pass does not separate those readings, since it keeps only the score per op.

The rest of the positions spill without removing, and their spill lands on `lighten`. Nothing at those positions helps the model infer the op, so whatever the edit takes from them is something other than the op. So on a fully trained model something besides the op reaches e₁ at the operands and symbols too, though the anti-subspace term is meant to keep it off.

If the edit were restricted to the example answers, it would lose the spill of the rest and of the separators, and keep the spill that comes with the removal. On most runs that leaves most of the spill. A selective edit would then need e₁ at the example answers to hold the op and little else, which is a question about training, and this pass can't answer it. The separator runs show that where the op is held differs from run to run, and at one seed between the two lengths, so the positions an edit should act on may differ too.
"""
