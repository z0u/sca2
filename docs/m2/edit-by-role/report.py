# ruff: noqa: B018
# title: What the edit changes at every position

# An exploratory re-analysis with no new training: the recipe of record (`τ_ℓ fixed`, seeds 712 to 719) scored at
# every position of the held-out contexts, clean and under the full edit along e₁, beside the controls and the
# `sans-diff` runs at the same seeds. `experiment.py` beside this script does the scoring.

import json
import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed
from mini.vis.plt import smooth_step, smooth_step_band, smooth_step_marks

# --- The stored results ---------------------------------------------------------------------------------------


def fetch() -> dict:
    store = project_store()
    art = store.get_refs([ex.RESULTS_REF])[ex.RESULTS_REF]
    assert art is not None, f"not published: {ex.RESULTS_REF}"
    with tempfile.TemporaryDirectory() as tmp:
        (path,) = store.get_many([(art, Path(tmp) / "results.json")])
        return json.loads(Path(path).read_text())


BODY = fetch()
OPS: tuple[str, ...] = tuple(BODY["design"]["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = list(BODY["design"]["seeds"])
FIXED, CONTROL, SANS = "fixed", "control", "sans-diff"
RUN = {(r["kind"], r["model_seed"]): r["summary"] for r in BODY["runs"]}
assert set(RUN) == {(k, s) for k in (FIXED, CONTROL, SANS) for s in SEEDS}, "every run is scored"

UNIT = ex.UNIT
N_POS = UNIT * (ex.K + 1) - 1
"""Positions that predict a token: every position but the last."""
ROLE_OF = [(t + 1) % UNIT for t in range(N_POS)]
"""The role of the token each position predicts, as an offset within its equation."""
SYMBOL = ("a", "?", "b", "=", "y", "⏎")
"""The roles in the symbols of P_o(y | a, b), as the figures label them."""
ANSWER_SITES = list(ex.ANSWER_SITES)
SITE_NAMES = ("example 1", "example 2", "example 3", "query")


def arr(kind: str, s: int, *path) -> np.ndarray:
    v = RUN[(kind, s)]
    for p in path:
        v = v[p]
    return np.asarray(v, float)


def change(kind: str, s: int, key: str = "s") -> dict[str, np.ndarray]:
    """Per op and position, the change under the e₁ edit from the clean pass, and the baseline: the median of that
    change over the edits along the other 63 coordinate axes. For "tv" the clean value is zero by construction.
    """
    clean = arr(kind, s, "clean", key)
    axes = np.stack([np.asarray(a[key], float) for a in RUN[(kind, s)]["axes"]])  # (axes, ops, positions)
    return {"e1": axes[0] - clean, "baseline": np.median(axes[1:], axis=0) - clean}


SURPRISAL = {(k, s): change(k, s) for k in (FIXED, CONTROL) for s in SEEDS}
MOVED = {(k, s): change(k, s, "tv") for k in (FIXED, CONTROL) for s in SEEDS}


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def span(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(float(v.mean()), spec)} <span class=range>({num(float(v.min()), spec)} to {num(float(v.max()), spec)})</span>"  # fmt: skip


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


RULE = ("#333", "#ddd")
GHOST = ("#bbb", "#555")
INK = {
    CONTROL: ("#666", "#bbb"),
    FIXED: ("#08306b", "#c6dbef"),
    SANS: ("#b35806", "#fdb863"),
    "null": ("#2a7a3b", "#8fd19e"),
}
MARKER = {CONTROL: "o", FIXED: "s", SANS: "D"}


def tex(shown: str) -> str:
    return shown.replace("τ_ℓ", "$\\tau_\\ell$")


# --- E1: where the prediction changes ----------------------------------------------------------------------------

GROUPS = {
    "operands": [t for t in range(N_POS) if ROLE_OF[t] in (0, 2)],
    "syntax (`?`, `=`, separator)": [t for t in range(N_POS) if ROLE_OF[t] in (1, 3, 5)],
    "example answers": ANSWER_SITES[:-1],
    "query answer": ANSWER_SITES[-1:],
}


def group_stat(kind: str, which: str, pos: list[int], ops: list[int], worst: bool = False, src=None) -> np.ndarray:
    """Per seed: the change in surprisal at the positions *pos*, mean over them, on the contexts of *ops* (the mean
    over those ops, or the largest change on any one of them with *worst*).
    """
    src = SURPRISAL if src is None else src
    out = []
    for s in SEEDS:
        v = src[(kind, s)][which][ops][:, pos].mean(axis=1)
        out.append(v.max() if worst else v.mean())
    return np.asarray(out)


def by_position(kind: str, which: str, ops: list[int], src=None) -> np.ndarray:
    """(seeds, positions): the change at each position, mean over the contexts of *ops*."""
    src = SURPRISAL if src is None else src
    return np.stack([src[(kind, s)][which][ops].mean(axis=0) for s in SEEDS])


def where_figure() -> str:
    data = {
        "d": {k: by_position(FIXED, k, [D]).tolist() for k in ("e1", "baseline")},
        "o": {k: by_position(FIXED, k, OTHER).tolist() for k in ("e1", "baseline")},
        "control_d": by_position(CONTROL, "e1", [D]).tolist(),
        "control_o": by_position(CONTROL, "e1", OTHER).tolist(),
    }
    d, o = np.asarray(data["d"]["e1"]).mean(0), np.asarray(data["o"]["e1"]).mean(0)
    alt = f"""
        Two panels of change in surprisal along the 23 positions of a context, under the e₁ edit on the recipe of
        record. On difference contexts (top) the line sits near zero at every operand and syntax position and rises
        only where the model predicts an example answer or the query answer: {num(d[ANSWER_SITES[0]])} at the first
        example, {num(d[ANSWER_SITES[1]])} at the second, {num(d[ANSWER_SITES[2]])} at the third, and
        {num(d[ANSWER_SITES[3]])} at the query. On the contexts of the other ops (bottom) it stays within
        {num(np.abs(o).max())} of zero everywhere, below the reference edits along the other axes, which raise the
        surprisal at every answer by about a tenth of a nat.
    """
    return where_draw(data, alt)


@memo
def where_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="edit-by-role-where",
        alt_text=alt_text,
        caption="""
            **Where the edit changes the prediction.** The change in surprisal of the next token under the full edit
            along e₁, from the clean pass, at each position of a context, labelled by the token predicted. **Top:**
            `difference` contexts; **bottom:** the contexts of the other six ops, pooled. The recipe of record is the
            dark line with its seed range as a band; the dashed line is the reference edit (the median over the other
            63 axes, seed mean), and the hairline is the e₁ edit on the controls.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 1, figsize=(6.4, 3.6), layout="constrained", sharex=True)
        x = np.arange(N_POS)
        for ax, key in zip(axes, ("d", "o"), strict=True):
            e1, base = np.asarray(data[key]["e1"]), np.asarray(data[key]["baseline"])
            ctl = np.asarray(data[f"control_{key}"])
            ax.axhline(0, color=light_dark(*GHOST), lw=0.6, zorder=0)
            smooth_step(ax, x, ctl.mean(0), ramp=0.6, color=light_dark(*INK[CONTROL]), lw=0.5, alpha=0.8, zorder=1)
            smooth_step(ax, x, base.mean(0), ramp=0.6, color=light_dark(*INK[CONTROL]), lw=0.9, ls="--", zorder=2)
            smooth_step_band(ax, x, e1.min(0), e1.max(0), ramp=0.6, color=light_dark(*INK[FIXED]), alpha=0.25, zorder=2)
            smooth_step(ax, x, e1.mean(0), ramp=0.6, color=light_dark(*INK[FIXED]), lw=1.4, zorder=3)
            ax.plot(x, e1.mean(0), MARKER[FIXED], ms=3, color=light_dark(*INK[FIXED]), zorder=4)
            ax.set_ylabel("change in surprisal\n(nats)", fontsize=7)
            ax.tick_params(labelsize=6.5)
            for t in ANSWER_SITES:
                ax.axvspan(t - 0.5, t + 0.5, color=light_dark("#eee", "#222"), lw=0, zorder=-1)
        axes[0].set_title("difference contexts", fontsize=7, loc="left")
        axes[1].set_title("other ops' contexts", fontsize=7, loc="left")
        axes[1].set_xticks(x, [SYMBOL[r] for r in ROLE_OF], fontsize=6.5)
        axes[1].set_xlabel("token predicted (example 1, 2, 3, then the query)", fontsize=7)
        return fig

    return _plot()


def where_table() -> str:
    rows = []
    for name, pos in GROUPS.items():
        rows.append([
            name,
            span(group_stat(FIXED, "e1", pos, [D])),
            span(group_stat(FIXED, "e1", pos, OTHER, worst=True)),
            span(group_stat(FIXED, "baseline", pos, OTHER, worst=True)),
            span(group_stat(CONTROL, "e1", pos, OTHER, worst=True)),
        ])  # fmt: skip
    return table_html(
        ["token predicted", "`difference`", "other ops, largest", "reference, largest", "control e₁, largest"],
        rows,
        f"""**The change in surprisal by role, in nats.** Mean over the positions of each role, seed mean and range over
        {len(SEEDS)} runs. The first two columns are the e₁ edit on the recipe of record; "largest" is the largest
        change on any one of the other six ops. The reference is the median over edits along the other 63 axes of
        the same run, and the last column is the e₁ edit on the controls.""",
    )


# --- E2: against the target null at the answers ----------------------------------------------------------------


def to_ref(kind: str, which: str, key: str, ops: list[int]) -> np.ndarray:
    """(seeds, answer sites): distance of the answer distribution from a reference predictor (`tv_null`,
    `tv_bayes`), on the clean pass or under the e₁ edit, mean over the contexts of *ops*.
    """
    out = []
    for s in SEEDS:
        v = arr(kind, s, "clean", key) if which == "clean" else np.asarray(RUN[(kind, s)]["axes"][0][key], float)
        out.append(v[ops].mean(axis=0))
    return np.stack(out)


TO_NULL = {
    "control": to_ref(CONTROL, "clean", "tv_null", [D]),
    "clean": to_ref(FIXED, "clean", "tv_null", [D]),
    "edited": to_ref(FIXED, "edited", "tv_null", [D]),
    "sans": to_ref(SANS, "clean", "tv_null", [D]),
}
OWN_GAP = to_ref(SANS, "clean", "tv_null", OTHER)
"""`sans-diff` on the other ops, where the target null is its own Bayes predictor: how close a model of this kind
gets to the ideal on contexts it trained on."""
EDITED_OTHER = to_ref(FIXED, "edited", "tv_null", OTHER)
CLEAN_OTHER_BAYES = to_ref(FIXED, "clean", "tv_bayes", OTHER)


M = {k: v.mean(axis=0) for k, v in TO_NULL.items()}
SHARE = (M["control"] - M["edited"]) / (M["control"] - M["sans"])
"""Per answer site, the share of the way from the clean control to `sans-diff` that the edited recipe covers."""
SITES_APART = [bool(TO_NULL["edited"][:, e].min() > TO_NULL["sans"][:, e].max()) for e in range(4)]
assert all(SITES_APART), "the seed ranges of the edited runs and sans-diff are apart at every site"
assert SHARE[0] < 0 < SHARE[1:].min(), "the edit moves away from the null at the first example, toward it after"

SURPRISE_EDITED = np.stack([np.asarray(RUN[(FIXED, s)]["axes"][0]["s"], float)[D, ANSWER_SITES] for s in SEEDS])
SURPRISE_NULL = np.stack([arr(FIXED, s, "surprisal_ref", "null")[D] for s in SEEDS])
"""On `difference` contexts, the surprisal of the shown answer at each answer site: under the edit, and under the
target null (the same for every run, since it depends on the contexts alone)."""


def null_figure() -> str:
    data = {k: v.tolist() for k, v in TO_NULL.items()} | {"own": OWN_GAP.tolist()}
    m = {k: np.asarray(v).mean(0) for k, v in data.items()}
    alt = f"""
        Distance from the target null at the four answer sites of difference contexts. The clean recipe of record
        moves away from it as the examples accumulate, from {num(m["clean"][0])} at the first example to
        {num(m["clean"][3])} at the query, as the control does. Under the edit it stays between {num(m["edited"].min())}
        and {num(m["edited"].max())}, close to sans-diff ({num(m["sans"].min())} to {num(m["sans"].max())}) and to the
        distance sans-diff keeps from its own ideal on the other ops ({num(m["own"].min())} to {num(m["own"].max())}).
    """
    return null_draw(data, alt)


@memo
def null_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="edit-by-role-null",
        alt_text=alt_text,
        caption="""
            **How far the answers on `difference` contexts are from the target null**, at each answer site: the total
            variation between the answer distribution and the target-null predictive for the examples seen so far.
            The recipe of record clean (hollow squares) and edited (solid), the control and `sans-diff` clean, each the
            seed mean with its seed range as a band. The dotted line is `sans-diff` on the contexts of the other ops,
            where the target null is its own ideal.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.6, 2.6), layout="constrained")
        x = np.arange(len(SITE_NAMES))
        own = np.asarray(data["own"]).mean(0)
        smooth_step(ax, x, own, color=light_dark(*INK[SANS]), lw=0.8, ls=":", zorder=1)
        series = (
            ("control", CONTROL, dict(mfc=None), "control"),
            ("clean", FIXED, dict(mfc="none"), tex("τ_ℓ fixed, clean")),
            ("sans", SANS, dict(mfc=None), "sans-diff"),
            ("edited", FIXED, dict(mfc=None), tex("τ_ℓ fixed, edited")),
        )
        for key, kind, mk, label in series:
            v = np.asarray(data[key])
            color = light_dark(*INK[kind])
            smooth_step_band(ax, x, v.min(0), v.max(0), color=color, alpha=0.18, zorder=2)
            smooth_step_marks(ax, x, v.mean(0), color=color, lw=1.3, zorder=3)
            kw = {k: w for k, w in mk.items() if w is not None}
            ax.plot(x, v.mean(0), MARKER[kind], ms=4.5, color=color, zorder=4, label=label, **kw)
        ax.set_xticks(x, SITE_NAMES, fontsize=6.5)
        ax.set_xlabel("answer site", fontsize=7)
        ax.set_ylabel("distance from the\ntarget null ↓", fontsize=7)
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=6.5)
        fig.legend(loc="outside upper center", ncols=4, frameon=False, fontsize=6.5)
        return fig

    return _plot()


def null_table() -> str:
    rows = [
        ["control, clean", *[span(TO_NULL["control"][:, e]) for e in range(4)]],
        ["`τ_ℓ fixed`, clean", *[span(TO_NULL["clean"][:, e]) for e in range(4)]],
        ["`τ_ℓ fixed`, edited", *[span(TO_NULL["edited"][:, e]) for e in range(4)]],
        ["`sans-diff`, clean", *[span(TO_NULL["sans"][:, e]) for e in range(4)]],
        ["`sans-diff` on the other ops", *[span(OWN_GAP[:, e]) for e in range(4)]],
        ["share of the way, edited", "–", *[num(float(SHARE[e])) for e in range(1, 4)]],
    ]
    return table_html(
        ["distance from the target null ↓", *SITE_NAMES],
        rows,
        f"""**Distance from the target null at each answer site**, on `difference` contexts unless named. Seed mean and
        range over {len(SEEDS)} runs. The share of the way is how far the edit moves the recipe of record from the
        control toward `sans-diff`, on the seed means: 0 at the control, 1 at `sans-diff`. At the first example the control and `sans-diff` are too close for a share to mean much.""",
    )


# %%

r"""
# What the edit changes at every position

/// tip |
<!-- lede -->
Beyond the query, the edit changes the next-token loss almost only at the answers of `difference` contexts. From the second example on, it takes those answers most of the way to where a model trained without `difference` lands. Before any example, it leaves the guess the model makes about `difference` mostly in place.
///

Every earlier score of the edit looked at one prediction, the answer at the query `=`. There, in aggregate, the edited model answers `difference` contexts about as the target null says it should ([fixed-effects](/docs/m2/fixed-effects/report.py)). But the model predicts a token at every position of a context, and the edit acts at all of them.

This re-analysis needs no new training. It scores the clean and edited model at every position of the held-out contexts, for the recipe of record, the controls, and `sans-diff` (a model trained without `difference`).

## Observations

- [Where the prediction changes (E1)](#where-the-prediction-changes-e1): the next-token loss rises at the answers of `difference` contexts, more with each example, and barely moves anywhere else. On the other ops the loss and the answers move less than under an edit along any of the other axes.
- [Against the target null at every answer (E2)](#against-the-target-null-at-every-answer-e2): from the second example on, the edit covers more than four-fifths of the way from the control to `sans-diff` at every answer site. At the first example, before any evidence, the edit keeps most of the weight the model gives `difference` answers.

## Scope and method

This is an exploratory re-analysis, with no preregistration and no gate. It covers three kinds of run at seeds 712 to 719:
(a) `τ_ℓ fixed` from the fixed-confirm experiment (the recipe of record),
(b) the controls of the anneal-confirm experiment, and
(c) the `sans-diff` runs of the filtered-corpus experiment.

Each run is scored at its final checkpoint on the 14,000 held-out contexts of ex-2.2.21 (2,000 per op), twice: clean, and under the full edit along e₁ at every slice and position. As the D2.2 design asks, the same edit along each of the other 63 coordinate axes of the run gives a reference: what the edit does to the same model along a direction the anchor did not choose.

A context is three examples and a query, each an equation of six tokens: *a* `?` *b* `=` *y* and a separator. The operands are drawn at random, so no model can predict them, and the symbols are fixed, so every model predicts them at once. Only the answers depend on the op, and the model infers the op from the examples before each one.

Surprisal
:   How unexpected the token that comes next is to the model, in nats: minus the log of the probability it gave that token. Lower is better. A change in surprisal under the edit, averaged over contexts, is the change in the next-token loss at that position.

Distance from the target null
:   The total variation between the answer distribution of the model at an answer site and the target-null predictive there: half the summed difference in probability over the colors, with mass on tokens that are not colors counted in full. 0 is the same distribution, 1 shares nothing. At an example answer the target null weighs the other ops by the examples before it, and mixes in the noise the corpus draws examples with.
"""

# %%

# REVIEW: three corrections of fact in E1, checked against the stored summary. (1) "within two hundredths" at the
# operands and symbols holds per position except at the separator after the query answer (seed mean 0.04, runs 713
# and 716 at 0.07 and 0.21), so that position is now named. (2) "an edit of the same size" became "the same edit":
# every axis gets the same full projection, and how much it removes depends on what the state has on that axis.
# (3) The control e₁ edit is at half the reference on run 712 (0.05 against 0.10, pooled over the other ops), so
# "near or above" is now seven of eight. Verify: `by_position(FIXED, "e1", [D])` and `by_position(CONTROL, "e1", OTHER)`.

rf"""
## Where the prediction changes (E1)

The target null says the edit should change nothing but the answers that depend on `difference`. The figure shows the change in surprisal at each position.

{where_figure()}

On `difference` contexts, the loss rises only at the answers. The rise grows with each example the model has seen: barely anything at the first, about two-thirds of a nat at the second, nearly a nat at the third, and four nats at the query, where the clean model was most sure of a `difference` answer (the table below).

At the operands and the symbols the loss stays within two hundredths of a nat of where it was, and on most runs within a few thousandths. The one exception is the separator after the query answer, where it rises by a few hundredths on the seed mean and by a fifth of a nat on one run.

On the contexts of the other ops, nothing moves much anywhere. The largest rise on any one op, at any answer site, is a few hundredths of a nat on most runs, and an eighth of a nat on the query of one. At the query answer the loss falls a little on most of these ops, which is the way the target null points: with `difference` gone, the other ops take its share.

The same edit along one of the other 63 axes raises the loss at every answer by about a tenth of a nat, on every op. So the e₁ edit is gentler on the other ops than one along an arbitrary direction, which fits an axis the anchor has kept clear of everything but the op.

On the controls, where e₁ is one more ordinary axis, the e₁ edit sits near that reference or above it on seven of the eight runs, and at about half of it on one.

The answers of the other ops move more than their loss does, by about a tenth in total variation at each answer site, but less than under the reference edits, and from the second example on they stay as far from the target null as they were.

{where_table()}

The rise at the first example is small. Before any example the model can only guess among all seven ops, and the edit keeps most of the weight that guess gives `difference` answers (E2).
"""

# %%

# REVIEW: three corrections of fact in E2 and the discussion. (1) The seed ranges of the edited runs and `sans-diff`
# are apart by 0.02, 0.01, 0.006, and 0.03 across the sites, so "by a few hundredths" became "narrowly at the third
# example". (2) The target null does give `difference` answers weight at an example: `predictives` keeps the anchored
# op among the replacement ops, so the median null surprisal of a shown `difference` answer is -log(ρ/6) ≈ 3.0 at
# every example. "No weight" and "expects none" were reworded. (3) "Close to it at the later ones" holds at the
# second and third examples only: at the query the null gives 91% of the shown answers zero probability, so its mean
# surprisal (629, from the 1e-300 clamp) is no yardstick. Verify: SURPRISE_EDITED and SURPRISE_NULL above.

rf"""
## Against the target null at every answer (E2)

No model here reaches the Bayes predictor at the examples (the ideal one, which weighs each op by the examples so far): each takes in less from the earlier examples than an ideal model would. So the target null is an ideal, and the yardstick is `sans-diff` instead, a model of the same kind that never learned `difference`, together with its own distance from the ideal on the contexts it trained on. The figure shows the distance from the target null on `difference` contexts.

{null_figure()}

Clean, the recipe of record is indistinguishable from the control: both move away from the target null with each example, as they grow surer of `difference`.

Under the edit the distance stays almost flat across the answer sites. From the second example on, it covers more than four-fifths of the way from the control to `sans-diff` (the table below), and the seed ranges of the two stay apart at every site, though narrowly at the third example.

At every example answer, `sans-diff` sits about as far from the target null on `difference` contexts as it sits from its own ideal on the other ops. So at the example answers, both models are near the floor that a model of this kind reaches.

{null_table()}

At the first example the edit moves the answer a little away from the target null, where `sans-diff` is slightly closer than the control. Before any example, the target null is an even guess among the six other ops, so it expects a `difference` answer only where another op gives the same color, or where the noise in the examples shows one.

`sans-diff` has learned to give these answers little weight. The clean recipe gives them as much weight as the answers of any other op, and the edit leaves that weight in place: the surprisal of the shown answer under the edit is below the target null at the first example and close to it at the second and third. So the edit takes out the evidence for `difference` as it arrives, and keeps most of the guess the model makes before seeing any.
"""

# %%

# REVIEW: narrowed after the claims review. "The prediction stays as it was" became a claim about the loss, since
# the answers of the other ops move by about a tenth in total variation (less than the reference edits). "About as
# close as sans-diff" became "most of the way", since the seed ranges are apart at every site and the surprisal of the
# shown answer ranks the two the other way. The first example now says the edit keeps most of the weight on
# `difference` answers rather than leaving the guess alone (it moves it 0.10 in total variation). "Builds up where
# the model combines the evidence" became "may show only at the query", since the gap at examples 2 and 3 equals the
# gap at example 1. Verify: MOVED, TO_NULL, SURPRISE_EDITED.

r"""
## Discussion

On `difference` contexts, the edit takes every answer the examples bear on most of the way to where a model that never learned the op lands, at the second and third examples as at the query. It leaves the loss almost as it was everywhere else, and moves the answers of the other ops less than an edit along any other axis does. So, beyond the query, the edited model behaves much like the target null, with one gap and a small remainder. This holds for one recipe on one task, under the full edit at every slice.

The remainder is that the edited model stays a little further from the target null than `sans-diff` at every answer site, by the distance from the target null. By the surprisal of the shown answer, the two rank the other way at the examples. We did not measure the distance between the two models themselves, so this report can say how far each is from the ideal but not how alike they are.

The gap is the prior. With no examples seen, the edited model still gives `difference` answers about the weight it gives any other op, where the target null gives them only what the noise in the examples does. That fits the reading of earlier reports that the anchor holds the evidence for the op at the example answers rather than the op itself, though the edit does move this guess a little.

The query stays the place where the edited model is furthest from `sans-diff`, as the filtered-corpus experiment found. At the second and third examples the gap is about what it is at the first, where there is no evidence to leave behind. So if part of the gap is evidence the edit leaves in place, it may show only at the query, where the model combines all three examples.
"""

# %%

r"""
## References

[fixed-effects](/docs/m2/fixed-effects/report.py)
:   On the recipe of record, the full edit takes `difference` contexts most of the way to the target null at the query, with no other op dropping beyond the selectivity criterion.

[filtered-corpus](/docs/m2/filtered-corpus/report.py)
:   A model trained with no loss on `difference` contexts (`sans-diff`) comes a little closer to the target null on those contexts than the edited recipe, which gets about 85% of the way to it from the control.

[fixed-confirm](/docs/m2/fixed-confirm/report.py)
:   The recipe of record (`τ_ℓ fixed`) kept the anchor on all eight runs at the fresh seeds 712 to 719.
"""
