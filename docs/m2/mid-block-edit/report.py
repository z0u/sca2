# ruff: noqa: B018
# title: Does the edit miss the mid-block state?

# An exploratory re-analysis of stored checkpoints, with no new training: `τ_ℓ fixed` at seeds 712 to 719
# (fixed-confirm). `experiment.py` beside this script scores the full e₁ edit between blocks, at the mid-block states
# alone, and at both, along every coordinate axis, and keeps the alignment with e₁ by sub-step.

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
RUNS: list[dict] = sorted(RESULTS["runs"], key=lambda r: r["model_seed"])
SEEDS = [r["model_seed"] for r in RUNS]
OPS: tuple[str, ...] = tuple(RESULTS["design"]["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SUBSTEPS: list[str] = RESULTS["design"]["substeps"]
SUBSTEP_TICK = ["emb", *(f"{p} {i}" for i in range(1, ex.N_LAYER + 1) for p in ("attn", "mlp"))]
EDITS = list(ex.EDITS)
CRITERION = 0.02  # the selectivity criterion of the D2.2 design
E1 = 0  # the anchored axis, in code


def net_of(drop) -> np.ndarray:
    """The drop under the e₁ edit less the median drop along the other axes: *drop* is (axes, ops), e₁ first."""
    drop = np.asarray(drop)
    return drop[E1] - np.median(drop[1:], axis=0)


def scores(r: dict, edit: str) -> dict[str, Any]:
    net = net_of(r["drop"][edit])
    worst = OTHER[int(net[OTHER].argmax())]
    return {
        "removal": float(net[D] / (r["clean"][D] - r["null"][D])),
        "spill": float(net[worst]),
        "op": OPS[worst],
    }


SCORES = {e: [scores(r, e) for r in RUNS] for e in EDITS}
REMOVAL = {e: np.array([s["removal"] for s in SCORES[e]]) for e in EDITS}
SPILL = {e: np.array([s["spill"] for s in SCORES[e]]) for e in EDITS}
GAIN = REMOVAL["both"] - REMOVAL["between blocks"]


def raw_removal(r: dict, edit: str) -> float:
    """Removal with no baseline subtracted."""
    return float(np.asarray(r["drop"][edit])[E1, D] / (r["clean"][D] - r["null"][D]))


RAW = {e: np.array([raw_removal(r, e) for r in RUNS]) for e in EDITS}
# The raw drop on the other ops under the e₁ edit, seed mean of the worst op per run.
RAW_OTHER = {e: np.array([np.asarray(r["drop"][e])[E1, OTHER].max() for r in RUNS]) for e in EDITS}
OVER = {e: int((SPILL[e] > CRITERION).sum()) for e in EDITS}


def alignment(r: dict, edit: str, site: str, ops: str = "difference") -> np.ndarray:
    return np.asarray(r["alignment"][edit][site][ops])


ATTN = [SUBSTEPS.index(f"attn{b}") for b in ex.BLOCKS]
MLP = [SUBSTEPS.index(f"mlp{b}") for b in ex.BLOCKS]
# α at the example answers of `difference` contexts, clean and under the between-block edit, by sub-step: (runs, sub-steps)
CLEAN = np.array([alignment(r, "clean", "example answers") for r in RUNS])
EDITED = np.array([alignment(r, "between blocks", "example answers") for r in RUNS])
EDITED_EQ = np.array([alignment(r, "between blocks", "query =") for r in RUNS])
CLEAN_EQ = np.array([alignment(r, "clean", "query =") for r in RUNS])
MID_UNDER_EDIT = EDITED[:, ATTN]  # what the attention of each block writes onto e₁ after the edit cleared it
MID_UNDER_EDIT_EQ = EDITED_EQ[:, ATTN]
CLEAN_OTHER = np.array([alignment(r, "clean", "example answers", "other") for r in RUNS])
A1, M1 = ATTN[0], MLP[0]


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus, and no sign on one that prints as zero."""
    half = 0.5 * 10 ** -int(spec[-2])
    return f"{0.0 if abs(v) < half else v:{spec}}".replace("-", "−")


def rng(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(v.min(), spec)} to {num(v.max(), spec)}"


def fmt(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(float(v.mean()), spec)} ({rng(v, spec)})"


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


def ink(edit: str) -> str:
    return {
        "clean": light_dark("#999", "#777"),
        "between blocks": light_dark("#c0392b", "#ff8a76"),
        "mid-block": light_dark("#1f6fb4", "#7ab8f0"),
        "both": light_dark("#6c3483", "#c39bd3"),
    }[edit]


def shade() -> str:
    return light_dark("#00000010", "#ffffff14")


MARKER = {"clean": "o", "between blocks": "s", "mid-block": "^", "both": "D"}


def substep_axis(ax, labels: bool = True) -> None:
    x = np.arange(len(SUBSTEPS))
    ax.set_xticks(x, SUBSTEP_TICK if labels else [""] * len(x), rotation=45 if labels else 0, ha="right", fontsize=7)
    ax.set_xlim(-0.3, len(SUBSTEPS) - 0.7)
    for i in ATTN:
        ax.axvspan(i - 0.5, i + 0.5, color=shade(), lw=0, zorder=0)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


# --- E1: what the attention writes after the edit -----------------------------------------------------------------


@memo
def written_draw(panels: dict, alt_text: str) -> str:
    @themed(
        name="mid-block-edit-written",
        alt_text=alt_text,
        caption="""
            **Alignment with e₁ by sub-step**, clean (grey circles) and under the full edit between blocks (red
            squares): the embedding, then the stream after the attention (shaded) and after the MLP of each block. One
            line per run, on `difference` contexts. Under the edit, every point after an MLP is the state before the
            edit acts, so it shows what each block put back. **Top:** at the example answers. **Bottom:** at the query
            `=`.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.2), sharex=True, sharey=True, layout="constrained")
        x = np.arange(len(SUBSTEPS))
        for ax, (label, series) in zip(axes, panels.items(), strict=True):
            for edit, rows in series.items():
                for v in rows:
                    ax.plot(x, v, "-", color=ink(edit), lw=0.8, alpha=0.7, marker=MARKER[edit], ms=2.2, mew=0)
            ax.axhline(0, color=ink("clean"), lw=0.5, zorder=0)
            ax.set_ylabel(f"α at {label}", fontsize=8)
            substep_axis(ax, labels=ax is axes[-1])
        handles = [
            plt.Line2D([], [], color=ink(e), marker=MARKER[e], ms=3.5, lw=1, label=e)
            for e in ("clean", "between blocks")
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


def written_figure() -> str:
    panels = {
        "the example answers": {"clean": CLEAN.tolist(), "between blocks": EDITED.tolist()},
        "the query =": {"clean": CLEAN_EQ.tolist(), "between blocks": EDITED_EQ.tolist()},
    }
    alt = (
        "Two line charts of alignment with e₁ across nine sub-steps, one line per run, clean and under the full edit "
        f"between blocks. At the example answers the clean lines peak near {num(CLEAN.max(axis=1).mean(), '.1f')}; "
        f"under the edit, each attention step puts back at most {num(MID_UNDER_EDIT.max())}. At the query = both "
        "stay near zero."
    )
    return written_draw(panels, alt)


# --- E2: removal and spill with the mid-block sites -----------------------------------------------------------------


@memo
def scores_draw(removal: dict, spill: dict, alt_text: str) -> str:
    @themed(
        name="mid-block-edit-scores",
        alt_text=alt_text,
        caption="""
            **Removal and spill under the full e₁ edit at three sets of sites**, one point per run (jittered), with the
            seed mean as a bar. Both are net of the median over the other 63 axes of the run. **Left:** removal, as a
            share of the way to the target null. **Right:** the largest spill on another op; the dashed line is the
            selectivity criterion.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4), layout="constrained")
        jitter = np.linspace(-0.12, 0.12, len(next(iter(removal.values()))))
        for ax, data in zip(axes, (removal, spill), strict=True):
            for i, (edit, v) in enumerate(data.items()):
                ax.plot(i + jitter, v, MARKER[edit], color=ink(edit), ms=3, alpha=0.7, mew=0)
                ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=ink(edit), lw=1.6)
            ax.set_xticks(range(len(data)), list(data), fontsize=7)
            ax.set_xlim(-0.5, len(data) - 0.5)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("removal", fontsize=8)
        axes[0].set_ylim(0, 1)
        axes[1].set_ylabel("worst spill", fontsize=8)
        axes[1].axhline(CRITERION, color=ink("clean"), lw=0.8, ls="--")
        return fig

    return _plot()


def scores_figure() -> str:
    alt = (
        "Two dot plots over three edits: between blocks, mid-block, and both. "
        f"Removal has seed means {', '.join(num(REMOVAL[e].mean()) for e in EDITS)}; "
        f"spill stays below the criterion for {', '.join(f'{8 - OVER[e]} of 8' for e in EDITS)} runs."
    )
    return scores_draw({e: REMOVAL[e].tolist() for e in EDITS}, {e: SPILL[e].tolist() for e in EDITS}, alt)


def scores_table() -> str:
    rows = []
    for i, s in enumerate(SEEDS):
        rows.append(
            [
                str(s),
                *(num(REMOVAL[e][i]) for e in EDITS),
                num(GAIN[i]),
                *(num(SPILL[e][i], ".3f") for e in EDITS),
            ]
        )
    return table_html(
        [
            "seed",
            *(f"removal, {e} ↑" for e in EDITS),
            "added by the mid-block sites",
            *(f"spill, {e} ↓" for e in EDITS),
        ],
        rows,
        """**Each run.** Removal and the largest spill on another op under the full e₁ edit at each set of sites, net
        of the median over the other 63 axes of the run, and the removal that adding the mid-block sites to the
        between-block edit gains.""",
    )


# %%

r"""
# Does the edit miss the mid-block state?

/// tip |
<!-- lede -->
Inside each block, the attention writes to the stream and the MLP of the same block reads it before the edit acts again. Editing that mid-block state as well gains nothing: the between-block edit already takes out nearly all of the op, and the extra sites cost selectivity, because the MLP of a block uses e₁ for the other ops too.
///

The edit of record acts between blocks, at the same slices the anchor pulls. But each block adds to the residual stream twice, first with its attention and then with its MLP, and nothing touches the state in between. So a block could, in principle, write the op onto e₁ with its attention and use it in its own MLP before any edit sees it. The [block1-edit](/docs/m2/block1-edit/report.py) account says the op travels between positions, from the example answers to the query `=`, and every such move goes through attention, which takes in an edited slice. That route is covered. The route this report asks about stays within one position and one block.

This report trains nothing. It takes the stored checkpoints of the recipe of record, `τ_ℓ fixed` at seeds 712 to 719, and scores the full e₁ edit at three sets of sites: between blocks (the edit of record), at the mid-block states alone, and at both.
"""

# %%

rf"""
## Observations

- [What the attention writes after the edit (E1)](#what-the-attention-writes-after-the-edit-e1): under the between-block edit, the attention of block 2 puts some alignment back at the example answers, and the later blocks almost none. The query `=` stays near zero throughout.
- [Removal and spill with the mid-block sites (E2)](#removal-and-spill-with-the-mid-block-sites-e2): adding the mid-block sites removes a little less of the op, and half the runs then spill past the criterion. The mid-block sites alone remove less, and spill more.

## Scope and method

This is an exploratory re-analysis of eight runs, with no preregistration and no gate. The runs are the eight of [fixed-confirm](/docs/m2/fixed-confirm/report.py), trained for 400 epochs on the in-context grammar with `difference` anchored on e₁ and the pull pooled over slices 1 to 3. No controls were scored, since every score here is net of the median over the other axes of the same run.

The mid-block state of a block is the stream after its attention step and before its MLP step. The edit is the full e₁ projection at every position, followed by the re-projection onto the sphere, as in the edit of record. "Mid-block" edits the mid-block states of all four blocks, and "both" adds them to the between-block edit at all five slices.

Every pass uses the 14,000 held-out contexts of ex-2.2.21, 2,000 per op. The between-block scores reproduce those of fixed-confirm. We scored the mid-block sites of all four blocks together, so this report does not say which block the spill comes from.

## The measurements

α
:   Alignment with e₁: the e₁ component of the unit-norm state, averaged over the three example answers (or taken at the query `=`) of `difference` contexts unless stated.

Removal
:   The drop in the score on `difference` under the edit, less the median drop under the same edit along each of the other 63 axes of the run, as a share of the way from the clean score to the target null. Higher means more of the op removed.

Spill
:   The largest drop on another op under the edit, net of the same median baseline. The selectivity criterion of the D2.2 design is {CRITERION}.
"""

# %%

rf"""
## What the attention writes after the edit (E1)

If a block wrote the op onto e₁ with its attention after the edit had cleared it, the mid-block state would show alignment that the next slice edit takes away only after the MLP has used it. So the main measure here is α at the mid-block states under the between-block edit.

{written_figure()}

On the clean pass, the alignment at the example answers builds up through blocks 1 to 3, as in the block1-edit report. Under the between-block edit, the attention of block 2 puts some of it back at the example answers ({fmt(MID_UNDER_EDIT[:, 1])} across runs), the MLP of block 2 keeps it about level, and the edit at slice 2 then clears it. The attention of blocks 3 and 4 puts back almost nothing on most runs. So there is alignment for an MLP to read inside block 2, about a third of the clean value at that point.

At the query `=`, where the answer is predicted, α stays near zero on every sub-step, clean or edited. The block1-edit report found the same between blocks, and it holds inside the blocks too.

Before any edit, the attention of block 1 writes nearly as much alignment at the example answers of the other ops ({fmt(CLEAN_OTHER[:, A1])}) as at those of `difference` ({fmt(CLEAN[:, A1])}). Only the MLP of block 1 sets `difference` apart. So the mid-block state of block 1 has e₁ on every op, and most of what it holds there is something other than the op.
"""

# %%

rf"""
## Removal and spill with the mid-block sites (E2)

Does editing the mid-block states as well remove more of the op? The main measure is removal, with spill beside it.

{scores_figure()}

It does not. Adding the mid-block sites to the between-block edit lowers removal on every run, by a little ({rng(GAIN)}), and spill rises: {OVER["both"]} of the 8 runs go past the criterion, against {OVER["between blocks"]} under the between-block edit. The mid-block sites alone remove less than the between-block edit and spill the most.

Most of the gap between removal and 1 is the baseline. With no baseline subtracted, the between-block edit takes out {rng(RAW["between blocks"])} of the way to the target null, and the median edit along another axis takes out about a tenth. So there was little left for any extra site to find.

The spill fits what E1 showed of block 1. The MLP of a block seems to read the e₁ component of its mid-block state on every op, so taking it away there changes what the MLP computes for the other ops as well. With no baseline, the worst drop on another op is {fmt(RAW_OTHER["between blocks"], ".3f")} under the between-block edit and {fmt(RAW_OTHER["both"], ".3f")} with the mid-block sites added.

{scores_table()}
"""

# %%

r"""
## Discussion

The within-block route is there, in the sense that the attention of block 2 puts alignment back where its MLP could read it. But removing it gains nothing, so the MLP does not seem to carry the op past the edit through it. The between-block edit already removes nearly all of the op on its own, which fits the block1-edit account: the op reaches the query `=` through attention, from slices the edit has cleared.

The mid-block state also has e₁ on every op, before the MLP of block 1 makes it specific to `difference`. That makes it a poorer place to edit than the slice after the block, which is where the anchor pulls and where the concept has settled. The same reasoning would apply to a pull at the mid-block states: it would ask block 1 to tell the ops apart before its MLP has done so.

These results hold for this recipe and this small model. In a deeper model the attention and the MLP of one block might do more of the work between them; the scorer can now edit mid-block states, so the check is cheap to repeat there.
"""

# %%

r"""
## References

[block1-edit](/docs/m2/block1-edit/report.py)
:   Block 1 writes the evidence for the op onto e₁ at the example answers, one later block reads it from the query `=` through attention, and an edit removes the op when it sits between the two.

[fixed-confirm](/docs/m2/fixed-confirm/report.py)
:   The pool over slices 1 to 3 with $\tau_\ell$ fixed keeps the anchor on all eight runs at seeds 712 to 719 and was adopted as the recipe of record.
"""
