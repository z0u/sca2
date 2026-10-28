# ruff: noqa: B018
# title: Why editing block 1 alone removes the op

# An exploratory re-analysis of stored checkpoints, with no new training: `falloff` at seeds 704 to 711
# (split-anti-seeds), `τ_ℓ fixed` at 712 to 719 (fixed-confirm), and two ex-2.2.23 controls. `experiment.py` beside
# this script runs the forward pass by sub-step with the e₁ edit spliced in at chosen slices and positions, and stores
# the alignment, the scores, the attention, and the color probes of each run.

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

# --- The stored results ---------------------------------------------------------------------------------------


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


RESULTS = fetch_json(ex.RESULTS_REF)
RUNS: dict[str, dict] = {r["label"]: r for r in RESULTS["runs"]}
SUBSTEPS: list[str] = RESULTS["design"]["substeps"]
SUBSTEP_TICK = ["emb", *(f"{p} {i}" for i in range(1, ex.N_LAYER + 1) for p in ("attn", "mlp"))]
BLOCKS = list(range(1, ex.N_LAYER + 1))
ARMS = ("falloff", "fixed")
SHOWN = {"falloff": "falloff", "fixed": "τ_ℓ fixed", "control": "control"}
ANCHORED = [lbl for lbl, r in RUNS.items() if r["arm"] in ARMS]
CONTROLS = [lbl for lbl, r in RUNS.items() if r["arm"] == "control"]
EX_ANS = ex.EX_ANS
SINGLE = ("block 1", "block 2", "block 3", "block 4")


def at_answers(alpha: dict[str, list[float]], sub: str) -> float:
    """Mean α over the example answers at one sub-step."""
    return float(np.mean([alpha[sub][p] for p in EX_ANS]))


def profile(lbl: str, edit: str = "none") -> np.ndarray:
    """α at the example answers of `difference` contexts, by sub-step."""
    return np.array([at_answers(RUNS[lbl]["edits"][edit]["alpha_a"], s) for s in SUBSTEPS])


def reading(lbl: str) -> np.ndarray:
    """Attention from the query `=` to the example answers, by block (mean over heads and `difference` contexts)."""
    return np.asarray(RUNS[lbl]["edits"]["none"]["att_from_eq"]["example answers"])


def reader_of(lbl: str) -> int:
    return int(np.argmax(reading(lbl))) + 1


def removal(lbl: str, edit: str) -> float:
    return RUNS[lbl]["edits"][edit]["removal"]


READER = {lbl: reader_of(lbl) for lbl in ANCHORED}
N_READER = {b: sum(1 for v in READER.values() if v == b) for b in (2, 3, 4)}
READER_RUNS = {b: [RUNS[lbl]["model_seed"] for lbl in ANCHORED if READER[lbl] == b] for b in (2, 3, 4)}
CLEAN = {lbl: profile(lbl) for lbl in RUNS}
EDIT1 = {lbl: profile(lbl, "block 1") for lbl in RUNS}
MLP1, MLP4 = SUBSTEPS.index("mlp1"), SUBSTEPS.index("mlp4")
AFTER1 = np.array([CLEAN[lbl][MLP1] for lbl in ANCHORED])
AFTER4 = np.array([CLEAN[lbl][MLP4] for lbl in ANCHORED])
SHARE1 = AFTER1 / AFTER4


def arriving_at_reader(lbl: str) -> float:
    """α at the example answers under the block-1 edit, at the slice the reader block takes in. The stored state after
    the MLP of block 1 is the one the edit acts on, so a reader at block 2 takes in none.
    """
    if READER[lbl] == 2:
        return 0.0
    return EDIT1[lbl][SUBSTEPS.index(f"mlp{READER[lbl] - 1}")]


ARRIVING = {lbl: arriving_at_reader(lbl) for lbl in ANCHORED}
REBUILT = np.array([EDIT1[lbl][MLP4] for lbl in ANCHORED])  # α after block 4 under the block-1 edit

# Attention from each example answer to the two operands of its own example, by block.
OWN = {lbl: np.asarray(RUNS[lbl]["edits"]["none"]["att_answer_to_operands"]) for lbl in RUNS}
OWN_ANCHORED = np.array([OWN[lbl] for lbl in ANCHORED])  # (runs, blocks)
OWN_UNDER_EDIT = np.array([RUNS[lbl]["edits"]["block 1"]["att_answer_to_operands"] for lbl in ANCHORED])


def probe(lbl: str, name: str) -> np.ndarray:
    return np.array([RUNS[lbl]["probes"][name][s] for s in SUBSTEPS])


def operand_probe(lbl: str) -> np.ndarray:
    """R² for the two operand colors of an example at its answer, the mean of the two probes."""
    return (probe(lbl, "answer, first operand") + probe(lbl, "answer, second operand")) / 2


OPERAND_AT_ANSWER = {lbl: operand_probe(lbl) for lbl in RUNS}
OPERAND_OWN = {lbl: probe(lbl, "operand") for lbl in RUNS}
ATTN1 = SUBSTEPS.index("attn1")

# The share of the reading done at block 2, against removal under the edit at blocks 2 to 4.
SHARE_AT_2 = np.array([reading(lbl)[1] / reading(lbl).sum() for lbl in ANCHORED])
REMOVAL_2TO4 = np.array([removal(lbl, "blocks 2 to 4") for lbl in ANCHORED])
CORR_2 = float(np.corrcoef(SHARE_AT_2, REMOVAL_2TO4)[0, 1])

# Removal on the controls: what a baseline would subtract, for the edits at one slice and at several.
CONTROL_SINGLE = max(abs(removal(lbl, e)) for lbl in CONTROLS for e in (*SINGLE, "embedding"))
CONTROL_SEVERAL = max(
    abs(removal(lbl, e)) for lbl in CONTROLS for e in ("blocks 2 to 4", "blocks 1 to 4", "every slice")
)
EQ_ALPHA = max(abs(np.mean(RUNS[lbl]["edits"]["none"]["alpha_a"][s][ex.EQ])) for lbl in ANCHORED for s in SUBSTEPS)

POSITION: dict[str, np.ndarray] = {
    name: np.array([RUNS[lbl]["position_edits"][name]["removal"] for lbl in ANCHORED]) for name in ex.POSITION_EDITS
}
BLOCK1 = np.array([removal(lbl, "block 1") for lbl in ANCHORED])
POSITION_GAP = float(np.abs(POSITION["example answers"] - BLOCK1).max())
# Attention from the query `=` to the example answers at the reader block, clean and under the block-1 edit.
READER_ATT = np.array([reading(lbl)[READER[lbl] - 1] for lbl in ANCHORED])
READER_ATT_EDIT = np.array(
    [RUNS[lbl]["edits"]["block 1"]["att_from_eq"]["example answers"][READER[lbl] - 1] for lbl in ANCHORED]
)
READER_ATT_DROP = READER_ATT_EDIT / READER_ATT
HALVED = [RUNS[lbl]["model_seed"] for lbl, d in zip(ANCHORED, READER_ATT_DROP, strict=True) if d < 0.6]


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus, and no sign on one that prints as zero."""
    half = 0.5 * 10 ** -int(spec[-2])
    return f"{0.0 if abs(v) < half else v:{spec}}".replace("-", "−")


def rng(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(v.min(), spec)} to {num(v.max(), spec)}"


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    def code(c: str) -> str:
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", c)

    head, rows = [code(h) for h in head], [[code(c) for c in r] for r in rows]
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


# Theme inks are functions: `light_dark` picks by the theme active when it is called.


def ink(arm: str) -> str:
    return {
        "falloff": light_dark("#1f6fb4", "#7ab8f0"),
        "fixed": light_dark("#c0392b", "#ff8a76"),
        "control": light_dark("#999", "#777"),
    }[arm]


def rule() -> str:
    return light_dark("#aaa", "#555")


def shade() -> str:
    return light_dark("#00000010", "#ffffff14")


MARKER = {"falloff": "o", "fixed": "s", "control": "D"}
# The slices each recipe pulls: `falloff` steps the pull on at slice 2; `τ_ℓ fixed` pools it over slices 1 to 3.
PULLED = {"falloff": (2, 3, 4), "fixed": (1, 2, 3)}


def substep_axis(ax, labels: bool = True) -> None:
    x = np.arange(len(SUBSTEPS))
    ax.set_xticks(x, SUBSTEP_TICK if labels else [""] * len(x), rotation=45 if labels else 0, ha="right", fontsize=7)
    ax.set_xlim(-0.3, len(SUBSTEPS) - 0.7)
    ax.axvspan(0.5, 2.5, color=shade(), lw=0, zorder=0)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def series(ax, values: dict[str, np.ndarray], ls: str = "-", lw: float = 0.8, marker: bool = True) -> None:
    """One hairline per run, in the ink of its recipe, with the marker of the recipe on each sub-step."""
    x = np.arange(len(SUBSTEPS))
    for lbl, v in values.items():
        arm = RUNS[lbl]["arm"]
        ax.plot(x, v, ls, color=ink(arm), lw=lw, alpha=0.7, marker=MARKER[arm] if marker else None, ms=2.2, mew=0)


def pull_strip(ax) -> None:
    """Which slices each recipe pulls, marked at the sub-step that ends the slice."""
    for row, arm in enumerate(ARMS):
        for sl in range(ex.N_LAYER + 1):
            x = 0 if sl == 0 else SUBSTEPS.index(f"mlp{sl}")
            on = sl in PULLED[arm]
            ax.plot(x, row, MARKER[arm], ms=4, color=ink(arm), mfc=ink(arm) if on else "none", mew=0.8)
    ax.set_yticks([0, 1], [SHOWN[a] for a in ARMS], fontsize=7)
    ax.set_ylim(-0.6, 1.6)
    ax.set_ylabel("pulled", fontsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)


def fig_legend(fig, extra: list | None = None) -> None:
    handles = [
        plt.Line2D([], [], color=ink(a), marker=MARKER[a], ms=3.5, lw=1, label=SHOWN[a]) for a in (*ARMS, "control")
    ]
    fig.legend(handles=handles + (extra or []), loc="outside upper center", ncols=4, frameon=False, fontsize=7)


# --- E1: where the evidence is written ----------------------------------------------------------------------


@memo
def written_draw(clean: dict, operands: dict, alt_text: str) -> str:
    @themed(
        name="block1-edit-written",
        alt_text=alt_text,
        caption="""
            **Where the alignment and the operand colors arrive at the example answers**, by sub-step: the
            embedding, then the stream after the attention and after the MLP of each block. One line per run, in the
            ink and marker of its recipe; block 1 is shaded. **Top:** α at the example answers of `difference`
            contexts. **Middle:** how well a linear probe recovers the colors of the two operands of an example from
            the state at its answer (held-out $R^2$, every op). **Bottom:** the slices each recipe pulls (filled).
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(
            3, 1, figsize=(5.6, 5.0), sharex=True, layout="constrained", height_ratios=[3, 2.2, 0.7]
        )
        series(axes[0], clean)
        axes[0].set_ylabel("α at the example answers", fontsize=8)
        axes[0].set_ylim(-0.1, 0.8)
        series(axes[1], operands)
        axes[1].set_ylabel("operand colors\nat the answer, R²", fontsize=8)
        axes[1].set_ylim(0, 1)
        pull_strip(axes[2])
        for ax in axes[:2]:
            substep_axis(ax, labels=False)
        substep_axis(axes[2])
        axes[0].text(1.5, 0.78, "block 1", ha="center", va="top", fontsize=7, color=rule())
        fig_legend(fig)
        return fig

    return _plot()


def written_figure() -> str:
    alt = (
        "Top: alignment at the example answers by sub-step. Every anchored run starts near zero at the embedding, "
        "rises to about a third after the MLP of block 1, and climbs to about 0.5 to 0.7 by block 3; the two "
        "recipes overlap through block 1 and the controls stay near zero. Middle: the probe for the operand colors "
        "at the answer rises at the attention of block 1 on every run, controls included, by an amount that varies "
        "widely between runs, and then eases down. Bottom: falloff pulls slices 2 to 4, tau-l fixed pulls slices 1 to 3."
    )
    return written_draw(CLEAN, OPERAND_AT_ANSWER, alt)


def own_table() -> str:
    rows = []
    for arm in (*ARMS, "control"):
        lbls = [lbl for lbl in RUNS if RUNS[lbl]["arm"] == arm]
        own = np.array([OWN[lbl] for lbl in lbls])
        rows.append([f"`{SHOWN[arm]}`", *(rng(own[:, b]) for b in range(ex.N_LAYER))])
    return table_html(
        ["recipe", *(f"block {b}" for b in BLOCKS)],
        rows,
        f"""**Attention from each example answer to the operands of its own example**, by block: the share of its
        attention that lands on those two positions, mean over heads and over `difference` contexts, range over the
        runs of each recipe ({len(CONTROLS)} controls).""",
    )


# --- E2: what the later blocks add --------------------------------------------------------------------------


@memo
def carried_draw(clean: dict, edited: dict, own: dict, alt_text: str) -> str:
    @themed(
        name="block1-edit-carried",
        alt_text=alt_text,
        caption="""
            **What the later blocks put back after the edit at block 1.** **Top:** α at the example answers of
            `difference` contexts, clean (solid) and under the edit at block 1 (dotted), one pair of lines per
            anchored run. The edit takes effect after the MLP of block 1, so the dotted line leaves the solid one at
            the attention of block 2. **Bottom:** how well a linear probe recovers the color of an operand from the
            state at its own position (held-out $R^2$, every op, clean), for every run including the controls.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.2), sharex=True, layout="constrained", height_ratios=[3, 2])
        series(axes[0], clean)
        series(axes[0], edited, ls=":", lw=1.0, marker=False)
        axes[0].set_ylabel("α at the example answers", fontsize=8)
        axes[0].set_ylim(-0.1, 0.8)
        series(axes[1], own)
        axes[1].set_ylabel("operand color at\nthe operand, R²", fontsize=8)
        axes[1].set_ylim(0.7, 1.0)
        substep_axis(axes[0], labels=False)
        substep_axis(axes[1])
        extra = [plt.Line2D([], [], color=rule(), ls=":", lw=1.0, label="under the edit at block 1")]
        fig_legend(fig, extra)
        return fig

    return _plot()


def carried_figure() -> str:
    alt = (
        "Top: clean, the alignment at the example answers climbs from about a third after block 1 to about 0.5 to "
        "0.7. Under the edit at block 1 it falls to near zero at block 2 and comes back only partway, to about 0.2 "
        "on most runs by block 4. Bottom: the operand color stays recoverable at the operand positions through every "
        "block, falling slowly from nearly 1 at the embedding to about 0.85 to 0.9 after block 4, on the anchored "
        "runs and the controls alike."
    )
    edited = {lbl: EDIT1[lbl] for lbl in ANCHORED}
    return carried_draw({lbl: CLEAN[lbl] for lbl in ANCHORED}, edited, OPERAND_OWN, alt)


# --- E3: the reader, and the edits at one slice -------------------------------------------------------------


@memo
def reader_draw(points: dict, alt_text: str) -> str:
    @themed(
        name="block1-edit-reader",
        alt_text=alt_text,
        caption="""
            **Removal under the edit at one slice, by the block that reads the example answers.** One mark per
            anchored run and edit, its shape naming the slice edited, nudged sideways so the edits sit apart. The
            reader block is where the attention from the query `=` to the example answers is largest.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.0, 3.0), layout="constrained")
        shapes = {"block 1": "o", "block 2": "s", "block 3": "^", "block 4": "v"}
        cols = {
            "block 1": light_dark("#1b7f5e", "#5fd3a8"),
            "block 2": light_dark("#7a4fa0", "#c4a6e3"),
            "block 3": light_dark("#b5651d", "#f0b070"),
            "block 4": light_dark("#555", "#bbb"),
        }
        for i, e in enumerate(SINGLE):
            xs, ys = points[e]
            ax.scatter(
                np.asarray(xs) + (i - 1.5) * 0.12, ys, marker=shapes[e], s=26, color=cols[e], label=f"edit at {e}"
            )
        ax.set_xticks([2, 3, 4], [f"block {b}\n({N_READER[b]} runs)" for b in (2, 3, 4)], fontsize=7)
        ax.set_xlim(1.5, 4.5)
        ax.set_xlabel("the block that reads the example answers", fontsize=8)
        ax.set_ylabel("removal", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        fig.legend(loc="outside upper center", ncols=4, frameon=False, fontsize=7)
        return fig

    return _plot()


def reader_figure() -> str:
    points = {e: ([READER[lbl] for lbl in ANCHORED], [removal(lbl, e) for lbl in ANCHORED]) for e in SINGLE}
    alt = (
        "Removal against the reader block for four single-slice edits. The edit at block 1 removes between about 0.4 "
        "and 1 whatever the reader. The edit at block 2 removes nearly everything where the reader is block 3 or 4, "
        "and about a tenth where it is block 2. The edit at block 3 removes nearly everything only where the reader "
        "is block 4. The edit at block 4 removes nothing on any run."
    )
    return reader_draw(points, alt)


def runs_table() -> str:
    rows = []
    for lbl in ANCHORED:
        r = RUNS[lbl]
        rows.append(
            [
                f"`{SHOWN[r['arm']]}` {r['model_seed']}",
                str(READER[lbl]),
                num(CLEAN[lbl][MLP1]),
                num(CLEAN[lbl][MLP4]),
                *(num(removal(lbl, e)) for e in ("block 1", "block 2", "block 3", "blocks 2 to 4", "every slice")),
                num(ARRIVING[lbl]),
            ]
        )
    return table_html(
        [
            "run",
            "reader block",
            "α after block 1",
            "α after block 4",
            "edit at block 1 ↑",
            "edit at block 2",
            "edit at block 3",
            "edit at blocks 2 to 4",
            "edit at every slice",
            "α arriving at the reader, under the block-1 edit",
        ],
        rows,
        """**Each anchored run.** The reader block, α at the example answers of `difference` contexts after block 1
        and after block 4 (clean), removal under the edit at each set of slices, and α at the example answers as it
        arrives at the reader block under the edit at block 1 (none for a reader at block 2, which takes in the edited
        slice). Removal is not net of a baseline (see the scope and method).""",
    )


# %%

r"""
# Why editing block 1 alone removes the op

/// tip |
<!-- lede -->
Nothing pulls block 1 under the step recipe, yet editing it alone removes most of the op. The reason is that block 1 writes the evidence for the op onto e₁, and one later block reads it off the example answers. An edit removes the op when it sits between that writer and that reader, and block 1 is upstream of every reader.
///

Since the [late-pull experiment](/docs/m2/late-pull/report.py), the recipes have kept the pull off the early slices, so that stand-ins for the op stay out of the embedding table. Yet on every seed, editing block 1 alone removed most of the op ([slice-terms](/docs/m2/slice-terms/report.py) E1, [recipe-compare](/docs/m2/recipe-compare/report.py) E3). And editing blocks 2 to 4, where the pull acts, fell short on the same seed in every arm. Neither result had an account.

This report trains nothing. It takes the stored checkpoints of the step recipe `falloff` at seeds 704 to 711 and of the recipe of record `τ_ℓ fixed` at 712 to 719, with two controls, and follows the alignment with e₁ through the model one sub-step at a time, clean and under edits at chosen slices and positions. It also looks at where attention goes, and at what color information the states hold.
"""

# %%

rf"""
## Observations

- [Where the evidence is written (E1)](#where-the-evidence-is-written-e1): block 1 is the first place the evidence for the op can be computed, and both recipes write about half of it there, whether or not block 1 is pulled. At block 1 each example answer gathers the colors of its own operands.
- [What the later blocks add (E2)](#what-the-later-blocks-add-e2): the later blocks mostly carry the alignment forward. Under the edit at block 1 they put back only a little of it, though the operand colors are still there to compute it from.
- [The reader, and the edits at one slice (E3)](#the-reader-and-the-edits-at-one-slice-e3): on each run one block reads the example answers from the query `=`, usually block 3. An edit at one slice removes the op when it sits between block 1 and that reader, which accounts for the seeds where the edit at blocks 2 to 4 fell short.

## Scope and method

This is an exploratory re-analysis, with no preregistration and no gate. It covers 16 anchored runs, all trained for 400 epochs on the in-context grammar with `difference` anchored on e₁:

- `falloff` (seeds 704 to 711, from [split-anti-seeds](/docs/m2/split-anti-seeds/report.py)): the pull steps on at slice 2 and acts on slices 2 to 4. Nothing pulls block 1.
- `τ_ℓ fixed` (seeds 712 to 719, from [fixed-confirm](/docs/m2/fixed-confirm/report.py)): the pull is pooled over slices 1 to 3, so block 1 can take part of it.

The two controls of ex-2.2.23 at seeds 704 and 710 were trained with no pull and no anti term.

Each block adds to the residual stream twice, first with its attention and then with its MLP. So the figures follow the stream at nine points: the embedding, then after the attention and after the MLP of each of the four blocks. A slice is the stream after a whole block, so slice 1 is the point after the MLP of block 1.

Every pass uses the 14,000 held-out contexts of ex-2.2.21, 2,000 per op. The edit is the full e₁ edit, applied after the named blocks, at every position unless stated. Removal here is not net of a baseline, unlike the published scores, because the median baseline of the D2.2 design exists only for the edit at every slice. The controls show what a baseline would subtract: the same edits remove at most {num(CONTROL_SINGLE)} of `difference` on them where the edit is at one slice, and up to {num(CONTROL_SEVERAL)} where it is at several.

What block 1 computes is inferred from where the alignment appears, from attention, and from color probes. We did not fit a probe for the op itself, which would show the per-example judgement as a feature apart from e₁ (the backlog item on what e₁ holds along the context asks for one).

## The measurements

α
:   Alignment with e₁: the e₁ component of the unit-norm state. Here, the mean over the three example answers of `difference` contexts unless stated.

Removal
:   The drop in the score on `difference` under an edit, as a share of the way from the clean score to the target null, with no baseline subtracted. Higher means more of the op removed.

Reader block
:   The block whose attention from the query `=` to the example answers is largest, averaged over heads and over `difference` contexts.

Color probe
:   A ridge regression from the 64-dimensional state to the RGB of a color, scored by held-out $R^2$ over five folds of contexts (the positions of one context stay in one fold), on contexts of every op. 1 means the color can be recovered fully from the state, and 0 means no better than its mean.
"""

# %%

rf"""
## Where the evidence is written (E1)

Where along the stream does the alignment at the example answers appear? The evidence for the op is a judgement about one example: does this answer fit `difference` on these two operands? The embedding of an answer is only its token, so the judgement needs at least one attention step, from the answer to its operands, and something to compute with what it gathers. Block 1 is the first place that can happen. The main measure is α at the example answers, with the color probe beside it.

{written_figure()}

Both recipes write about half of the final alignment in block 1: a little over half of that in its attention, and the rest in its MLP. After block 1, α at the example answers is {rng(AFTER1)} across the 16 runs, {rng(SHARE1, ".1f")} of its value after block 4. The embedding has none of it, and the controls stay near zero throughout.

The profile through block 1 is about the same under `falloff`, which does not pull block 1, as under `τ_ℓ fixed`, which does. So the pull at block 1 itself seems to make little difference to what block 1 writes. The pull at slices 2 to 4 reaches block 1 anyway: the stream carries what block 1 writes to every later slice, so the gradient of the pull there flows back along the residual path to the parameters of block 1. The cheapest way to have e₁ aligned at three later slices is to write it once, at the earliest block that can, and let the stream carry it.

The attention fits this account. Block 1 is the first block where an example answer can look at its operands, and on every run it does: it puts {rng(OWN_ANCHORED[:, 0])} of its attention on the two operands of its own example. Blocks 2 and 3 look at them too, on some runs as much, but they come after the alignment has appeared.

{own_table()}

The color probe at the answer shows what that attention brings. The operand colors are partly recoverable from the answer even at the embedding, since an answer says something about its operands under most ops. At the attention of block 1 that rises on every run, the controls included, then eases down through the later blocks. How far it rises varies a great deal from run to run, and it does not separate the anchored runs from the controls. So gathering the operands at the answer is something the model does for the task anyway, and the pull puts the judgement made from them onto e₁.
"""

# %%

rf"""
## What the later blocks add (E2)

Blocks 2 and 3 raise α at the example answers from about a third to its final value on the clean pass. Do they compute the evidence again, or add to what arrives? If they computed it again, the alignment would come back after an edit at block 1 takes it out. The main measure is α at the example answers under the edit at block 1.

{carried_figure()}

Most of it does not come back. Under the edit at block 1, α at the example answers after block 4 is {rng(REBUILT)}, against {rng(AFTER4)} on the clean pass. So what blocks 2 and 3 add onto e₁ is mostly downstream of what arrives: take away the e₁ component at block 1, and their writes onto e₁ largely go with it. The part that does come back varies by run, and E3 shows it is what sets how far short of full removal the edit at block 1 falls.

Could the later blocks compute the evidence again, if they had what it takes? The operand colors are still there at the operand positions: the color probe there falls only a little through the four blocks (lower panel), and the controls look the same. And blocks 2 and 3 do attend from the example answers to their operands on many runs (table in E1). Under the edit at block 1 that attention stays about the same, or rises a little.

So the later blocks have both the colors and the attention, and still put back only part of the alignment. What they write onto e₁ seems to depend on the e₁ component that arrives, which fits blocks that add to the evidence block 1 wrote more than blocks that compute it afresh. A probe for the op itself, apart from e₁, would show whether the later blocks keep a copy of the judgement off e₁; we did not fit one.
"""

# %%

rf"""
## The reader, and the edits at one slice (E3)

If block 1 writes the evidence, where is it used? The query `=` predicts the answer, so the evidence at the example answers has to reach it through attention. And why did the edit at blocks 2 to 4 fall short on some seeds and not others? The main measure is removal under the edit at one slice, set against the block that reads the example answers.

On every run, the attention from the query `=` to the example answers is concentrated in one block: block 3 on {N_READER[3]} of the 16 runs, block 4 on {N_READER[4]} ({", ".join(map(str, READER_RUNS[4]))}), and block 2 on {N_READER[2]} ({" and ".join(map(str, READER_RUNS[2]))}). That reader takes in the stream from the slice before it. α at the query `=` itself stays low at every slice (at most {num(EQ_ALPHA, ".1f")}), so what the reader carries over lands off e₁, where no later edit reaches it.

Where the reader forms may go with how early the example answers hold their operands. The two runs whose reader is block 2 are the runs where the attention of block 1 gathers the operand colors at the answer most, each the highest of its recipe (the top lines in the middle panel of the E1 figure), and the control at seed 710 gathers them about as much. So the seed seems to set how strongly block 1 gathers, and perhaps with it how early the reading can form, though two runs are too few to say more.

{reader_figure()}

This one fact accounts for the pattern of the edits at one slice:

- The edit at block 4 removes nothing on any run, because every reader has already taken what it needs.
- The edit at block 3 removes nearly everything only where the reader is block 4.
- The edit at block 2 removes nearly everything where the reader is block 3 or 4, and little where it is block 2, which takes in slice 1.
- The edit at block 1 removes the op on every run, since every reader comes after it.

The same holds for the edit at blocks 2 to 4: it removes nearly everything unless the reader is block 2. Across the 16 runs, removal under that edit and the share of the reading done at block 2 have a correlation of {num(CORR_2)}. Seed 710 fell short in every earlier arm (split-anti-seeds, recipe-compare, and the soft-pool arms), and it is one of the two seeds whose reader formed at block 2.

{runs_table()}

How much the edit at block 1 removes depends on how much of the alignment the later blocks put back before the reader takes it in (the last column). On the runs whose reader is block 2, the reader takes in the edited slice itself, nothing has come back, and removal is nearly full. On the runs whose reader is block 4, about half of the clean alignment at slice 3 is back by then, and removal is lowest.

It is the e₁ component at the example answers that holds the op. The edit at block 1 restricted to the example answers removes as much as the edit at block 1 at every position, within {num(POSITION_GAP)} on every run, while the same edit at every other position removes nothing ({rng(POSITION["all but example answers"])}). And under the edit the reader still looks at the example answers: its attention there stays about the same on most runs, and halves on one ({", ".join(map(str, HALVED))}). What it takes in has lost the evidence.
"""

# %%

r"""
## Discussion

The residual stream makes "where the pull acts" and "where the concept is written" separate questions. The pull names slices, and training answers with the earliest block that can compute the concept, because one write there serves every pulled slice after it. On this task that is block 1, since the per-example judgement needs one attention step (the answer gathers its operands) and one MLP. The lean of [embedding-lean](/docs/m2/embedding-lean/report.py) looks like the same effect one step earlier: where the pull reached the embedding, the model answered with the nearest thing the embedding could offer. So keeping the pull off block 1 did not keep the concept off block 1. It kept the stand-ins off it, which was its purpose.

The write bound of [edit-effects](/docs/m2/edit-effects/report.py) already treats the edit at each slice as acting on what arrives there. This account adds that most of what arrives at blocks 2 to 4 on e₁ was put there by block 1. So the edit at every slice does most of its work at slice 1, and the later slices find little left to turn.

It also says why the edit at every slice is the robust choice, as the D2.2 design argues. An edit at fewer slices has to land between the writer and the reader, and the reader forms at a different block from seed to seed. The edit at every slice covers every pairing. A larger model has readers at many depths that we don't choose, so this is the property that would carry over.

The task needs three steps in sequence: each example answer judges itself against its operands, the query `=` reads those judgements, and the query applies the op it infers. That fits in two blocks. If which block becomes the reader is set by when the reading forms during training, a two-block model would leave it only block 2, and the edit at slice 1 would be the whole edit. That would make a cleaner picture of the mechanism, though without the depth that makes the reader vary.
"""

# %%

r"""
## References

[late-pull experiment](/docs/m2/late-pull/report.py)
:   With the pull on blocks 2 to 4, the color embedding table leans a quarter as far onto e₁ and block 1 comes to hold part of the concept, though nothing pulls it.

[slice-terms](/docs/m2/slice-terms/report.py)
:   Per-slice anti weights recover removal with spill about within the criterion; editing block 1 alone removes most of the op on every seed.

[recipe-compare](/docs/m2/recipe-compare/report.py)
:   `step, soft τ` stays close to `falloff`; under every recipe the edit at blocks 2 to 4 removes the op and spills nothing, and what spill there is enters with the edit at block 1.

[split-anti-seeds](/docs/m2/split-anti-seeds/report.py)
:   At fresh seeds, the `falloff` anti weight keeps the anchor and keeps spill within the criterion; it became the step recipe of record.

[fixed-confirm](/docs/m2/fixed-confirm/report.py)
:   The pool over slices 1 to 3 with $\tau_\ell$ fixed keeps the anchor on all eight runs at seeds 712 to 719 and was adopted as the recipe of record.

[embedding-lean](/docs/m2/embedding-lean/report.py)
:   Where the pull reaches the embedding, the color embeddings lean onto e₁ by lightness, a stand-in for the op that the edit then takes from every op.

[edit-effects](/docs/m2/edit-effects/report.py)
:   On `falloff`, the blocks after the edit put back no more e₁ on the states of the other ops than the clean model has there.
"""
