# ruff: noqa: B018
# title: The peak alignment on other ops, at two lengths

# A re-analysis of ex-2.2.23's stored evaluation, with no new runs. It reads the alignment with e₁ by op, slice, and
# role that the evaluation stored for every run at the end of training, takes its largest value over the roles of
# the contexts of the six other ops, and compares that with the spill of the edit.
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as the ex-2.2 experiments do."""
    path = Path(__file__).resolve().parent.parent / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


ex = _load_sibling("ex-2.2.23", "schedules_at_the_rise_ex2223")
SHORT, LONG = ex.SHORT, ex.LONG
D = ex.OP_NAMES.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(ex.OP_NAMES)) if o != D]
CONTROL, ANCHOR = (c.name for c in ex.CONDITIONS)
LATCH = 0.9
# A run has latched a syntax token when that token's embedding has at least this alignment with e₁: latched
# embeddings lie on e₁ at about 1.00, and no other syntax embedding in these runs comes near it.

# --- The stored results ----------------------------------------------------------------------------------------


def fetch_json(refs: list[str]) -> dict[str, Any]:
    """Each ref's published JSON, by ref."""
    store = project_store()
    found = store.get_refs(refs)
    arts = {r: a for r, a in found.items() if a is not None}
    assert len(arts) == len(refs), f"not published: {sorted(set(refs) - set(arts))}"
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(arts[r], Path(tmp) / f"{i}.json") for i, r in enumerate(refs)])
        return {r: json.loads(p.read_text()) for r, p in zip(refs, paths, strict=True)}


_got = fetch_json([ex.EVAL_REF, ex.SUPPRESSION_REF])
EVAL, SUPP = _got[ex.EVAL_REF], _got[ex.SUPPRESSION_REF]
assert tuple(EVAL["runs"][0]["ops"]) == tuple(ex.OP_NAMES)
K = EVAL["runs"][0]["k"]

# A run: (condition, epochs, model seed).
Key = tuple[str, int, int]


def role_names(k: int) -> list[str]:
    """The roles of a context of *k* examples and a query, in the symbols of the figures: a ? b = y, then the
    separator (`,` after an example, ⏎ after the query). Examples are numbered from 1; the query is q.
    """
    out = []
    for i in range(k + 1):
        n = "q" if i == k else str(i + 1)
        out += [f"a{n}", f"?{n}", f"b{n}", f"={n}", f"y{n}", "⏎" if i == k else f",{n}"]
    return out


ROLES = role_names(K)
REACH = EVAL["runs"][0]["roles"]["query ="] + 1
# The roles that can reach the answer: up to the query `=`, where the answer is read. Later roles (the query answer
# and ⏎) cannot affect it, so the peak leaves them out, as spill-by-position's masks did.
assert ROLES[REACH - 1] == "=q"
N_SLICES = len(EVAL["runs"][0]["alignment"][0])
SLICES = ["emb", *(str(i) for i in range(1, N_SLICES))]


def worst_spill(r: dict, c: dict) -> float:
    """The largest drop the edit causes on any op other than the anchored one, at any dose, net of the control
    run *c* at the same seed and length: ex-2.2.23's spill measurement.
    """

    def drops(x: dict) -> np.ndarray:
        return np.array(x["clean"]["eem"])[None, :] - np.array([e["eem"] for e in x["edits"]])  # (doses, ops)

    return float((drops(r) - drops(c))[:, OTHER].max())


def build_runs() -> dict[Key, dict[str, Any]]:
    supps = {r["label"]: r for r in SUPP["runs"]}
    out = {}
    for r in EVAL["runs"]:
        key = (r["condition"], r["epochs"], r["model_seed"])
        a = np.array(r["alignment"], float)  # (ops, slices, roles)
        n = np.array(r["n_per_op"], float)[OTHER]
        other = np.einsum("o,olt->lt", n / n.sum(), a[OTHER])[:, :REACH]  # ᾱ over the contexts of the other ops
        latched = [t for t, v in r["syntax_embeddings"].items() if v >= LATCH]
        out[key] = {
            "peak": other.max(axis=1),
            "mean": other.mean(axis=1),
            "latched": latched[0] if latched else None,
            "spill": None,
        }
    for key in out:
        if key[0] == ANCHOR:
            label, ctrl = ex.label_of(*key), ex.label_of(CONTROL, *key[1:])
            out[key]["spill"] = worst_spill(supps[label], supps[ctrl])
    return out


RUNS = build_runs()


def keys(cond: str, epochs: int) -> list[Key]:
    return sorted(k for k in RUNS if k[0] == cond and k[1] == epochs)


def stack(cond: str, epochs: int, field: str) -> np.ndarray:
    """*field* for every run of one condition and length: (runs, slices)."""
    return np.array([RUNS[k][field] for k in keys(cond, epochs)])


GROUPS = [(c, e) for c in (CONTROL, ANCHOR) for e in ex.LENGTHS]

# --- Shared drawing --------------------------------------------------------------------------------------------


def ink_of(cond: str, epochs: int) -> str:
    return {
        (CONTROL, SHORT): light_dark("#888", "#999"),
        (CONTROL, LONG): light_dark("#444", "#ccc"),
        (ANCHOR, SHORT): light_dark("#1f6fb4", "#7ab8f0"),
        (ANCHOR, LONG): light_dark("#c0392b", "#ff8a76"),
    }[(cond, epochs)]


MARKER = {SHORT: "o", LONG: "s"}


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    """An authored table in the shared report style; the first *text_cols* columns are text, the rest numeric."""

    def cell(text: str) -> str:
        return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(text.split("`")))

    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def num_word(n: int) -> str:
    return ("none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")[
        n
    ]


# --- E1: the peak by slice -----------------------------------------------------------------------------------------


def peak_table() -> str:
    rows = []
    for c, e in GROUPS:
        p, m = stack(c, e, "peak"), stack(c, e, "mean")
        cells = [
            f"{p[:, s].mean():.2f} <span class=range>({m[:, s].mean() + 0.0:.2f})</span>".replace("(-0.00)", "(0.00)")
            for s in range(N_SLICES)
        ]
        rows.append([f"`{c}`", f"{e}", *cells])
    return table_html(
        ["condition", "epochs", *SLICES],
        rows,
        "**The peak alignment on other ops, by slice.** Seed mean of the peak of ᾱ over the roles up to the query `=`, "
        "on the contexts of the six other ops, with the seed mean of the mean over roles in brackets. Twelve runs in each row.",
        text_cols=2,
    )


LATCH_GROUPS = (",", "?", "⏎", None)
LATCH_MARKER = {",": "^", "?": "D", "⏎": "v", None: "o"}


def latch_rows() -> list[list[str]]:
    rows = []
    for t in LATCH_GROUPS:
        cells = []
        for e in ex.LENGTHS:
            ks = [k for k in keys(ANCHOR, e) if RUNS[k]["latched"] == t]
            cells.append(f"{len(ks)}")
        for e in ex.LENGTHS:
            sp = [RUNS[k]["spill"] for k in keys(ANCHOR, e) if RUNS[k]["latched"] == t]
            cells.append(f"{np.median(sp):.2f}" if sp else "–")
        rows.append(["none" if t is None else f"`{t}`", *cells])
    return rows


def latch_table() -> str:
    return table_html(
        ["latched token", f"runs, {SHORT}", f"runs, {LONG}", f"median spill, {SHORT}", f"median spill, {LONG}"],
        latch_rows(),
        f"**Which syntax token each anchored run latched.** A run has latched a token when its embedding lies on e₁ "
        f"(alignment {LATCH:g} or more); no run latched more than one. ⏎ comes after the query `=`, so it is left "
        "out of the peak. Spill as in E2.",
    )


def n_latched(epochs: int, before_answer: bool) -> int:
    toks = {",", "?", "="} if before_answer else {"⏎"}
    return sum(RUNS[k]["latched"] in toks for k in keys(ANCHOR, epochs))


def peak_figure() -> str:
    caption = """
        **The peak alignment on other ops, by slice.** One column per condition and length at each slice: the
        individual runs as small faded dots, a thin bar over their range, and the seed mean on top. Circles at 200
        epochs, squares at 400; the control in grey.
    """
    alt = f"""
        A chart of the peak alignment against the slice, from the embedding to the fourth block. The control runs
        sit between about 0.04 and 0.1 at every slice. The anchored runs spread from near zero to 1 at the embedding
        and the first block, with seed means near {stack(ANCHOR, SHORT, "peak")[:, 0].mean():.2f} at 200 epochs and
        {stack(ANCHOR, LONG, "peak")[:, 0].mean():.2f} at 400 at the embedding, a little higher at the first block,
        then fall with depth to about 0.1 at both lengths by the last two slices.
    """
    data = {f"{c}|{e}": stack(c, e, "peak").tolist() for c, e in GROUPS}
    return peak_draw(data, caption, alt)


@memo
def peak_draw(data: dict, caption: str, alt_text: str) -> str:
    @themed(name="schedules-at-the-rise-peak", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        rng = np.random.default_rng(0)
        fig, ax = plt.subplots(figsize=(6.4, 2.8), layout="constrained")
        offsets = np.linspace(-0.27, 0.27, len(GROUPS))
        for (c, e), dx in zip(GROUPS, offsets, strict=True):
            v = np.array(data[f"{c}|{e}"])
            color = ink_of(c, e)
            for s in range(v.shape[1]):
                x = s + dx
                col = v[:, s]
                ax.plot([x, x], [col.min(), col.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2,
                        solid_capstyle="butt")  # fmt: skip
                ax.plot(x + rng.uniform(-0.03, 0.03, len(col)), col, "o", ms=2.2, color=color, alpha=0.45, mew=0,
                        zorder=3)  # fmt: skip
                ax.plot(x, col.mean(), MARKER[e], ms=5, color=color, mec=light_dark("white", "#111"), mew=0.6,
                        zorder=4, label=f"{c}, {e} epochs" if s == 0 else None)  # fmt: skip
        ax.set_xticks(range(len(SLICES)), SLICES)
        ax.set_xlim(-0.5, len(SLICES) - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("slice", fontsize=8)
        ax.set_ylabel("peak ᾱ on other ops", fontsize=8)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


# --- E2: the peak against the spill ---------------------------------------------------------------------------------


def rho(epochs: int, s: int) -> float:
    ks = keys(ANCHOR, epochs)
    return float(spearmanr([RUNS[k]["peak"][s] for k in ks], [RUNS[k]["spill"] for k in ks]).statistic)


RHO = {(e, s): rho(e, s) for e in ex.LENGTHS for s in range(N_SLICES)}


def rho_table() -> str:
    rows = [[f"{e}", *(f"{RHO[(e, s)]:+.2f}" for s in range(N_SLICES))] for e in ex.LENGTHS]
    return table_html(
        ["epochs", *SLICES],
        rows,
        "**The peak against the spill.** Spearman ρ between the peak at each slice and the spill, over the twelve "
        "anchored runs at each length. With twelve runs, a ρ of about ±0.58 is needed for p < 0.05 (uncorrected).",
    )


def spill_figure() -> str:
    caption = f"""
        **The spill of the edit against the peak alignment on other ops**, at the embedding slice (left) and the
        last slice (right). One mark per anchored run, colored by length (blue at 200 epochs, red at 400) and
        shaped by the token it latched (E3): triangle for `,`, diamond for `?`, down-triangle for ⏎, circle for
        none. The dashed line is
        ex-2.2.21's selectivity criterion ({ex.SELECTIVITY_GATE:g}).
    """
    alt = """
        Two scatter panels sharing the spill axis. On the left, at the embedding slice, four 400-epoch squares sit
        at a peak of 1 with the highest spill, 0.3 to 0.47, and the other squares at peaks near 0.25 to 0.33 spill
        0.1 to 0.28; one square at a low peak barely spills. The 200-epoch circles mostly sit at low peaks with little spill; two
        at a peak of 1 and two near 0.26 spill 0.14 to 0.22. On the right, at the last slice, the peaks bunch
        between 0.03 and 0.17, with a weaker rising trend at both lengths.
        In both panels the runs that latched ⏎ (down-triangles) sit at the bottom, spilling under 0.05.
    """
    data = {
        e: [(RUNS[k]["peak"][0], RUNS[k]["peak"][-1], RUNS[k]["spill"], RUNS[k]["latched"]) for k in keys(ANCHOR, e)]
        for e in ex.LENGTHS
    }
    return spill_draw(data, caption, alt)


@memo
def spill_draw(data: dict, caption: str, alt_text: str) -> str:
    @themed(name="schedules-at-the-rise-spill", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), layout="constrained", sharey=True)
        for i, (ax, title) in enumerate(zip(axes, ("embedding slice", "last slice"), strict=True)):
            for e in ex.LENGTHS:
                for t in LATCH_GROUPS:
                    pts = np.array([p[:3] for p in data[e] if p[3] == t])
                    if not len(pts):
                        continue
                    ax.scatter(pts[:, i], pts[:, 2], marker=LATCH_MARKER[t], s=26, facecolor=ink_of(ANCHOR, e),
                               edgecolor=light_dark("#fff", "#111"), lw=0.6, zorder=3,
                               )  # fmt: skip
            ax.axhline(ex.SELECTIVITY_GATE, color=light_dark("#000", "#fff"), lw=0.8, ls="--", alpha=0.6)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("peak ᾱ on other ops", fontsize=8)
        axes[0].set_xlim(0, 1.05)
        axes[1].set_xlim(0, 0.2)
        axes[0].set_ylim(-0.02, 0.5)
        axes[0].set_ylabel("spill (largest drop on another op)", fontsize=8)
        from matplotlib.lines import Line2D

        def key(marker: str, color: str, label: str) -> Line2D:
            return Line2D([], [], ls="none", marker=marker, ms=5, color=color, label=label)

        neutral = light_dark("#666", "#bbb")
        handles = [key("s", ink_of(ANCHOR, e), f"{e} epochs") for e in ex.LENGTHS] + [
            key(LATCH_MARKER[t], neutral, f"latched {t}" if t else "no latch") for t in LATCH_GROUPS
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), frameon=False, fontsize=7)
        return fig

    return _plot()


# --- The report ----------------------------------------------------------------------------------------------------

LAST = N_SLICES - 1

rf"""
# The peak alignment on other ops, at two lengths

/// tip |
<!-- lede -->
On contexts of the other ops, the alignment of the anchored runs with e₁ is concentrated at a few roles near the embedding, and pooled by its peak it is higher at 400 epochs than at 200, as the spill is. Within each length, the runs with a higher peak tend to spill more. Much of the peak is a latched syntax token, and which token matters: the runs that latched ⏎, which comes after the query, barely spill, and at 400 epochs the runs that latched the example separator spill most.
///

The lean that ex-2.2.23 tracked through training is a mean over every state. The effect it was looking for is likely localized, on a few roles or on a latched token embedding (as [spill-by-position](/docs/m2/spill-by-position/report.py) found), so the mean washes it out. And the spill of the edit lands on the other ops, so the states that matter are those in contexts of the six ops besides `{ex.ANCHORED_OP}`. This report pools the alignment in that way: for each run and slice, ᾱ is the mean alignment with e₁ at one role over the contexts of the other ops, the peak is its largest value over the roles that can reach the answer, and the summary is the seed mean of the peak. It reads the alignment that ex-2.2.23 stored at the end of training for its {len(RUNS)} runs, with no new runs. The stored trajectories keep only the mean over every state, so this pooling is available at the end of training only.
"""

# %%

rf"""
## Observations

- [The peak by slice (E1)](#the-peak-by-slice-e1): on the anchored runs the peak is several times the mean over roles, and highest at the embedding and the first block. There it is about twice as high at 400 epochs as at 200. By the last slice it is close to the control at both lengths.
- [The peak against the spill (E2)](#the-peak-against-the-spill-e2): within each length, the runs with a higher peak tend to spill more, most clearly at 400 epochs and in the early slices.
- [The latched token (E3)](#the-latched-token-e3): most anchored runs latched one syntax token. The runs that latched ⏎ (most of them at 200 epochs) barely spill. At 400 epochs the runs that latched the separator `,` spill most.

## Scope

This is an exploratory re-analysis of stored results, with no preregistration and no gate. It covers the {len(keys(ANCHOR, SHORT)) + len(keys(ANCHOR, LONG))} anchored and {len(keys(CONTROL, SHORT)) + len(keys(CONTROL, LONG))} control runs of ex-2.2.23, twelve model seeds of each condition at each of 200 and 400 epochs, measured on its held-out contexts of {K} examples and a query. With twelve runs per length, a rank correlation has to be large to mean much.

Three of the anchored runs at 200 epochs never learned the HSV ops, and ex-2.2.23 found they do not spill. They are kept, since nothing here depends on the task score, but they weigh on any trend at 200 epochs, and all three latched ⏎.

## The measurements

ᾱ on other ops
:   At one slice and one role, the alignment with e₁ (the cosine between the state and the first basis vector) averaged over the held-out contexts of the six ops other than `{ex.ANCHORED_OP}`, weighted by their counts.

Peak
:   The largest ᾱ over the roles from the first operand to the query `=`, at one slice: the {REACH} roles whose states can reach the answer. The query answer and ⏎ come later, so they are left out. Higher means some role sits closer to e₁ on the other ops.

Spill
:   The largest drop the edit causes on any op other than `{ex.ANCHORED_OP}`, at any dose, net of the control at the same seed and length: ex-2.2.23's measurement. Lower is better; ex-2.2.21's criterion is {ex.SELECTIVITY_GATE:g}.
"""

# %%

rf"""
## The peak by slice (E1)

How far toward e₁ does the most aligned role sit, on contexts of the other ops, and does that differ between the lengths?

{peak_figure()}

{peak_table()}

On the anchored runs the peak sits several times above the mean over roles in the early slices, so the alignment on the other ops is concentrated on a few roles. It is highest at the embedding and the first block, and falls with depth to near the control by the last slice. The spread is wide: at the embedding some runs sit at 1, which E3 traces to a latched token, and others at or below the control.

In the early slices the 400-epoch runs peak about twice as high as the 200-epoch runs, and in the last two slices the lengths are alike. The control runs peak lower at 400 epochs than at 200, so the difference on the anchored runs is not a general effect of longer training.
"""

# %%

rf"""
## The peak against the spill (E2)

Do the runs whose most aligned role sits closer to e₁ spill more?

{spill_figure()}

{rho_table()}

They do, at both lengths, though less firmly at 200 epochs. The rank correlation is positive at every slice (at the embedding at 200 epochs, below the level a twelve-run test would need), and strongest in the early slices: at the first block at 200 epochs, and at the embedding at 400. At the last slice it is weaker at 200 epochs. At the embedding, though, the runs fall into three clumps (a peak of 1, about a quarter to a third, and near zero), which E3 traces to the latched token, so there the rank correlation mostly ranks the clumps. The correlation also holds in the blocks, where there are no clumps. At 200 epochs it leans on the three runs that never learned the HSV ops, which neither peak nor spill: without them (nine runs) it stays positive at the embedding, the first block, and the third, and fades at the second block and the last. With those caveats, the peak in the early slices fits being the version of the lean that follows the spill, most clearly at 400 epochs.
<!-- REVIEW: softened "is the version of the lean" to "fits being" and added the clumping caveat: at the embedding the peaks take three values set by the latch (see E3 table), so ρ there is not independent of E3. Verify: ρ within the no-latch runs alone. -->
"""

# %%

rf"""
## The latched token (E3)

The peaks of 1 at the embedding are syntax tokens whose embedding lies on e₁. Which tokens are they, and do they go with the spill?

{latch_table()}

Most anchored runs latched one syntax token, and which token goes with the spill. The shapes in the E2 figure show it at a glance, in both panels. The runs that latched ⏎ spill little: ⏎ comes after the query, so a latched ⏎ cannot reach the answer, and its peak is left out. Most runs at 200 epochs latched ⏎, though three of those are the runs that never learned the HSV ops. The runs that latched a token before the answer (the separator `,` on most, `?` on one) are {num_word(n_latched(LONG, True))} of the twelve at 400 epochs against {num_word(n_latched(SHORT, True))} at 200. At 400 epochs they spill most; at 200 epochs they and the two runs that latched nothing spill about alike. The runs that latched nothing have their peak at the example answers, at about a quarter to a third at the embedding.

So part of the difference between the lengths is which token the runs latched, and the latch on `,` fits spill-by-position, where the latched separator carried most of the spill.
"""

# %%

r"""
## Discussion

Pooled by its peak over the roles that can reach the answer, the alignment on the other ops is concentrated near the embedding, and it follows the spill: higher at 400 epochs than at 200, and higher on the runs that spill more within each length. The mean over every state, which is what the trajectories kept, washes this out, perhaps partly because the latched token can come after the answer, where it moves the mean and cannot touch the answer.

How much of this is the latch is open. The latched token seems to decide a good part of the spill, and nothing here says why a run latches one token and not another, or why the longer runs latch the separator more often. The runs that latched nothing also peak higher, and spill more, than the runs that latched ⏎, so the latch is not the whole of it.

How the peak moves through training, and whether it rises as the anti-subspace weight falls, is out of reach of the stored results, which keep the alignment by role only at the end. That would take new runs with a recorder that keeps ᾱ by op and role.
"""
