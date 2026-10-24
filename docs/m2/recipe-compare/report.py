# ruff: noqa: B018
# title: Three recipes side by side

# An exploratory re-analysis of the stored runs of `falloff`, `step, soft τ` and `pool 1–3, soft τ` at seeds 704 to
# 711, with no new training. `experiment.py` beside this script adds the write bound on the two soft conditions and
# the axis scoring on the loss-rate runs that were finished at 400 epochs; everything else was stored by
# split-anti-seeds, soft-pool, edit-effects and reference-baseline.

import json
import re
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed

# --- The stored results ---------------------------------------------------------------------------------------

FALLOFF, STEP_SOFT, POOL = ex.CONDITIONS
CONTROL = "control"
SHOWN = {FALLOFF: "falloff", STEP_SOFT: "step, soft τ", POOL: "pool 1–3, soft τ"}
SEEDS: list[int] = list(ex.SEEDS)
OPS: tuple[str, ...] = tuple(ex.OP_NAMES)
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
GAMMAS: list[float] = list(ex.rb.DOSE_GAMMAS)
FULL = len(GAMMAS) - 1
CRITERION = 0.02  # the selectivity criterion of the D2.2 design
LOST_BELOW = 0.5  # a run has lost the anchor when the full edit removes less than this
TASK_MEAN_GATE, TASK_OP_GATE = -0.02, -0.05  # the task gate of the D2.2 design
E1 = 0  # the anchored axis, in code
UNIT = 6  # tokens per equation: a ? b = y and a separator
N_POS = UNIT * (ex.ee.K + 1)
ANSWERS = [UNIT * e + 4 for e in range(ex.ee.K)]
QUERY_ANSWER = UNIT * ex.ee.K + 4
SLICE_TICK = ["emb", "1", "2", "3", "4"]
Q = 0.99  # the quantile the bound and the write are taken at, as in edit-effects
TOLERANCE = 1.25  # ex-2.2.1 allowed the write a quarter over the bound


def label_of(condition: str, seed: int) -> str:
    return f"{condition}-s{seed}"


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


def fetch_arrays(refs: dict[str, str], keep: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    """The `.npz` behind each ref, by label, with only the *keep* arrays."""
    store = project_store()
    got = store.get_refs(list(refs.values()))
    missing = [r for r, a in got.items() if a is None]
    assert not missing, f"not published: {missing}"
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(got[r], Path(tmp) / f"{lbl}.npz") for lbl, r in refs.items()])
        for lbl, p in zip(refs, paths, strict=True):
            with np.load(p) as z:
                out[lbl] = {k: z[k] for k in keep}
    return out


# Reference-baseline scored the dose edit along every coordinate axis of each run and control at 704 to 711; this
# experiment did the same for the finished loss-rate runs. Keyed here by condition label (`step-soft-s704`).
_RB = {r["label"]: r for r in fetch_json(ex.rb.AXES_REF)["runs"]}
AX = {
    label_of(c, s): _RB[label_of(c, s) if c == FALLOFF else ex.rb.SOFT_POOL + label_of(c, s)]
    for c in ex.CONDITIONS
    for s in SEEDS
}
AX |= {label_of(CONTROL, s): _RB[label_of(CONTROL, s)] for s in SEEDS}
FINISHED = {r["label"]: r for r in fetch_json(ex.AXES_REF)["runs"]}

# The measurements of split-anti-seeds and soft-pool (the edit restricted to slice sets), and their dose summaries.
LEAN = {
    r["label"]: r
    for ref in (ex.sas.RESULTS_REF, ex.sp.RESULTS_REF)
    for r in fetch_json(ref)["runs"]
    if r["arm"] in ex.CONDITIONS and r["model_seed"] in SEEDS
}
DOSE_SUMMARY = {r["label"]: r for ref in (ex.sas.DOSE_REF, ex.sp.DOSE_REF) for r in fetch_json(ref)["runs"]}


def dose_array_ref(lbl: str) -> str:
    cond = lbl.rsplit("-s", 1)[0]
    return (ex.sp.DOSE_ARRAYS_REF if cond in ex.SOFT else ex.sas.DOSE_ARRAYS_REF).format(label=lbl)


def alpha_ref(lbl: str) -> str:
    cond = lbl.rsplit("-s", 1)[0]
    return (ex.WRITE_ARRAYS_REF if cond in ex.SOFT else ex.ee.ARRAYS_REF).format(label=lbl)


LABELS = [label_of(c, s) for c in (*ex.CONDITIONS, CONTROL) for s in SEEDS]
DOSE = fetch_arrays(
    {lbl: dose_array_ref(lbl) for lbl in LABELS},
    ("op_ids", "posterior", "query_pair", "clean_eem", "null_eem", "eem"),
)
ALPHA = fetch_arrays({lbl: alpha_ref(lbl) for lbl in LABELS}, ("clean", "arriving", "op_ids"))
FITS = fetch_arrays({"fits": ex.ee.FITS_REF}, ("fits",))["fits"]["fits"]
OP_IDS = DOSE[LABELS[0]]["op_ids"].astype(int)
IS_D = OP_IDS == D
for _lbl in LABELS:
    assert (DOSE[_lbl]["op_ids"] == OP_IDS).all() and (ALPHA[_lbl]["op_ids"] == OP_IDS).all()
FITS_D = FITS[:, :, D]  # (contexts, examples + query): the answer is one `difference` can give

# --- E1: removal, spill and task ------------------------------------------------------------------------------


def axis_scores(r: dict) -> dict[str, Any]:
    """Removal at the full dose and spill at every dose for the e₁ edit on one run, net of the median drop over every
    other coordinate axis of the same run (the D2.2 rule), and removal with nothing subtracted.
    """
    drop = np.asarray(r["drop"])  # (axes, doses, ops)
    net = drop[E1] - np.median(drop[1:], axis=0)
    span = r["clean"][D] - r["null"][D]
    spill = net[:, OTHER].max(axis=1)
    worst = int(spill.argmax())
    return {
        "removal": float(net[FULL, D] / span),
        "raw": float(drop[E1][FULL, D] / span),
        "spill": spill.tolist(),
        "worst_spill": float(spill.max()),
        "worst_op": OPS[OTHER[int(net[worst, OTHER].argmax())]],
        "removal_by_dose": (net[:, D] / span).tolist(),
    }


SCORES = {lbl: axis_scores(r) for lbl, r in AX.items() if not lbl.startswith(CONTROL)}
FIN_SCORES = {lbl: axis_scores(r) for lbl, r in FINISHED.items()}


def task_change(c: str, s: int) -> np.ndarray:
    """The clean score per op less that of the control at the same seed."""
    return np.asarray(AX[label_of(c, s)]["clean"]) - np.asarray(AX[label_of(CONTROL, s)]["clean"])


def per_seed(c: str, key: str) -> np.ndarray:
    return np.array([SCORES[label_of(c, s)][key] for s in SEEDS])


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus, and no sign on one that prints as zero."""
    half = 0.5 * 10 ** -int(spec[-2])
    return f"{0.0 if abs(v) < half else v:{spec}}".replace("-", "−")


def fmt(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(v.mean(), spec)} ({num(v.min(), spec)} to {num(v.max(), spec)})"


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


REMOVAL = {c: per_seed(c, "removal") for c in ex.CONDITIONS}
RAW = {c: per_seed(c, "raw") for c in ex.CONDITIONS}
SPILL = {c: per_seed(c, "worst_spill") for c in ex.CONDITIONS}
TASK = {c: np.array([task_change(c, s) for s in SEEDS]) for c in ex.CONDITIONS}  # (seeds, ops)
N_OVER = {c: int((SPILL[c] > CRITERION).sum()) for c in ex.CONDITIONS}
GAP = {c: REMOVAL[c] - REMOVAL[FALLOFF] for c in ex.SOFT}  # paired by seed


# --- E2: the write, the grading and the landing ---------------------------------------------------------------


@memo
def turn_maps(clean: np.ndarray, arriving: np.ndarray, op_ids: np.ndarray, fits_d: np.ndarray) -> dict:
    """The bound and the write of one run as alignments, `(slices, positions)`: the 99th percentile of |α| over the
    contexts of the other ops, clean and as it arrives under the full edit, with the answers that fit `difference`
    left out at the answers, as in edit-effects. Beside it, the same at the example answers that do not fit, by slice.
    """
    other = op_ids != D
    c, a = np.abs(clean.astype(np.float32)), np.abs(arriving.astype(np.float32))
    bound, write = np.quantile(c[:, other], Q, axis=1), np.quantile(a[:, other], Q, axis=1)
    for e, p in enumerate(ANSWERS + [QUERY_ANSWER]):
        keep = other & ~fits_d[:, e]
        bound[:, p] = np.quantile(c[:, keep, p], Q, axis=1)
        write[:, p] = np.quantile(a[:, keep, p], Q, axis=1)
    cc = np.concatenate([c[:, other & ~fits_d[:, e], p] for e, p in enumerate(ANSWERS)], axis=1)
    aa = np.concatenate([a[:, other & ~fits_d[:, e], p] for e, p in enumerate(ANSWERS)], axis=1)
    return {
        "bound": bound,
        "write": write,
        "ans_bound": np.quantile(cc, Q, axis=1),
        "ans_write": np.quantile(aa, Q, axis=1),
        "alpha_d": c[:, op_ids == D][:, :, ANSWERS].mean(axis=(1, 2)),  # the anchor by slice, at the example answers
        "alpha_q": c[:, op_ids == D, QUERY_ANSWER].mean(axis=1),
    }


TURN = {lbl: turn_maps(ALPHA[lbl]["clean"], ALPHA[lbl]["arriving"], OP_IDS, FITS_D) for lbl in LABELS}


def seed_mean(c: str, key: str) -> np.ndarray:
    return np.mean([TURN[label_of(c, s)][key] for s in SEEDS], axis=0)


def angle(alpha) -> np.ndarray:
    return np.degrees(np.arcsin(np.clip(alpha, 0, 1)))


def ratio_map(c: str) -> np.ndarray:
    """The write over the bound in angle, from the seed-mean quantiles, as edit-effects takes it."""
    return np.arcsin(seed_mean(c, "write")) / np.arcsin(seed_mean(c, "bound"))


RATIO = {c: ratio_map(c) for c in ex.CONDITIONS}
for _c in ex.CONDITIONS:
    assert np.allclose(RATIO[_c][0], 1.0, atol=1e-3), "the embedding sees the clean stream"


def over_tolerance(c: str) -> list[tuple[int, int, int, float, float]]:
    """Per run, the sites past the embedding where the write goes over the tolerance: seed, slice, position, the
    ratio, and the bound there in degrees.
    """
    out = []
    for s in SEEDS:
        t = TURN[label_of(c, s)]
        r = np.arcsin(t["write"]) / np.arcsin(t["bound"])
        for sl, p in zip(*np.nonzero(r[1:] > TOLERANCE), strict=True):
            out.append((s, int(sl) + 1, int(p), float(r[sl + 1, p]), float(angle(t["bound"][sl + 1, p]))))
    return out


OVER = {c: over_tolerance(c) for c in ex.CONDITIONS}

POST_BINS = (0.0, 0.2, 0.5, 0.8, 0.95, 1.0)
BIN_LABEL = ["< 0.2", "0.2–0.5", "0.5–0.8", "0.8–0.95", "≥ 0.95"]
POST_D = DOSE[LABELS[0]]["posterior"][:, D].astype(np.float64)
BIN = np.clip(np.digitize(POST_D, POST_BINS[1:-1]), 0, len(POST_BINS) - 2)


def drop_of(lbl: str) -> np.ndarray:
    r = DOSE[lbl]
    return r["clean_eem"] - r["eem"][-1]


def shares(c: str, s: int, net: bool = True) -> np.ndarray:
    """The share of the way to the target null the full edit takes on `difference` contexts, per bin of the posterior
    on `difference` and then over all of them (last), as a ratio of means; net of the control at the seed, or not.
    """
    lbl = label_of(c, s)
    drop = drop_of(lbl) - (drop_of(label_of(CONTROL, s)) if net else 0)
    dep = DOSE[lbl]["clean_eem"] - DOSE[lbl]["null_eem"]
    groups = [(BIN == b) & IS_D for b in range(len(BIN_LABEL))] + [IS_D]
    return np.array([drop[g].mean() / dep[g].mean() for g in groups])


SHARE = {c: np.array([shares(c, s) for s in SEEDS]) for c in ex.CONDITIONS}  # (seeds, bins + all)
RAW_SHARE = {c: np.array([shares(c, s, net=False) for s in SEEDS]) for c in ex.CONDITIONS}
LANDING = {c: np.array([DOSE_SUMMARY[label_of(c, s)]["landing"] for s in SEEDS]) for c in ex.CONDITIONS}


# --- E3: where the op sits ------------------------------------------------------------------------------------

SLICE_SETS = ("block 1", "blocks 2 to 4", "block 4")  # the columns of the E3 table
# REVIEW: the embedding-only edit and the edit at blocks 1 to 4 were stored too, and the prose of E3 quotes them: the
# first spills nothing under any recipe and the second spills what the edit at every slice does, which is what puts
# the spill at block 1 by elimination rather than by the block-1 edit alone. Verify: `edits` in the stored
# measurements, keys "embedding" and "blocks 1 to 4".
ALL_SETS = (*SLICE_SETS, "embedding", "blocks 1 to 4")


# Embedding-lean measured ex-2.2.23's controls the same way, under its own labels.
LEAN_CONTROL = {
    s: r
    for r in fetch_json(ex.sas.lean.RESULTS_REF)["runs"]
    for s in SEEDS
    if r["label"] == ex.sas.ex2223.label_of(CONTROL, ex.sas.EPOCHS, s)
}
assert set(LEAN_CONTROL) == set(SEEDS)


def slice_net(c: str, s: int, name: str) -> np.ndarray:
    """The drop per op under the full edit restricted to the slice set *name*, net of the same edit on the control at
    the seed (the median baseline exists only for the edit at every slice), as split-anti-seeds scored it.
    """
    return np.asarray(LEAN[label_of(c, s)]["edits"][name]) - np.asarray(LEAN_CONTROL[s]["edits"][name])


def slice_removal(c: str, s: int, name: str) -> float:
    r = LEAN[label_of(c, s)]
    return float(slice_net(c, s, name)[D] / (r["clean"][D] - r["null"][D]))


ALPHA_D = {c: np.array([TURN[label_of(c, s)]["alpha_d"] for s in SEEDS]) for c in (*ex.CONDITIONS, CONTROL)}
ALPHA_Q = {c: np.array([TURN[label_of(c, s)]["alpha_q"] for s in SEEDS]) for c in (*ex.CONDITIONS, CONTROL)}
PEAK = {c: ALPHA_D[c][:, 1:].argmax(axis=1) + 1 for c in ex.CONDITIONS}  # the slice where α peaks, per seed
SLICE_REMOVAL = {c: {n: np.array([slice_removal(c, s, n) for s in SEEDS]) for n in ALL_SETS} for c in ex.CONDITIONS}
SLICE_SPILL = {
    c: {n: np.array([slice_net(c, s, n)[OTHER].max() for s in SEEDS]) for n in ALL_SETS} for c in ex.CONDITIONS
}
# The lean of embedding-lean in the states of the other ops, by slice (a correlation of the e₁ component with
# lightness at the color positions), for the E3 prose.
LEAN_STATES = {c: np.array([LEAN[label_of(c, s)]["r_light_states_other"] for s in SEEDS], float) for c in ex.CONDITIONS}
# The lean of embedding-lean: how the e₁ components of the color embeddings follow lightness (a correlation), and
# how widely they spread (a standard deviation).
LEAN_R = {c: np.array([LEAN[label_of(c, s)]["r_light_emb"] for s in SEEDS]) for c in ex.CONDITIONS}
LEAN_R[CONTROL] = np.array([LEAN_CONTROL[s]["r_light_emb"] for s in SEEDS])
LEAN_SD = {c: np.array([LEAN[label_of(c, s)]["emb_sd"] for s in SEEDS]) for c in ex.CONDITIONS}
LEAN_SD[CONTROL] = np.array([LEAN_CONTROL[s]["emb_sd"] for s in SEEDS])

# --- Checks and side measures for E1 -------------------------------------------------------------------------


def control_median_spill(c: str, s: int) -> float:
    """The worst spill with the median taken over the other axes of the control in place of those of the run."""
    r, ctl = AX[label_of(c, s)], AX[label_of(CONTROL, s)]
    net = np.asarray(r["drop"])[E1] - np.median(np.asarray(ctl["drop"])[1:], axis=0)
    return float(net[:, OTHER].max())


CTRL_SPILL = {c: np.array([control_median_spill(c, s) for s in SEEDS]) for c in ex.CONDITIONS}
# The finished loss-rate runs have no control at their seeds; their task is taken against the mean of the eight
# controls here, as loss-rate took it.
CONTROL_MEAN_CLEAN = np.mean([AX[label_of(CONTROL, s)]["clean"] for s in SEEDS], axis=0)
CONTROL_TASK_SPREAD = float(np.ptp([np.mean(AX[label_of(CONTROL, s)]["clean"]) for s in SEEDS]))
FIN_TASK = {lbl: np.asarray(r["clean"]) - CONTROL_MEAN_CLEAN for lbl, r in FINISHED.items()}

# --- Figure helpers --------------------------------------------------------------------------------------------
# Theme inks are functions: `light_dark` picks by the theme active when it is called.


def ink(c: str) -> str:
    return {
        FALLOFF: light_dark("#2a9d8f", "#7fd8c8"),
        STEP_SOFT: light_dark("#d07a1f", "#f2b266"),
        POOL: light_dark("#7b4fa8", "#c4a6e3"),
        CONTROL: light_dark("#999", "#777"),
    }[c]


MARKER = {FALLOFF: "o", STEP_SOFT: "s", POOL: "D", CONTROL: "x"}


def ghost() -> str:
    return light_dark("#ccc", "#444")


def rule() -> str:
    return light_dark("#333", "#ddd")


def dots(ax, x: float, v: np.ndarray, c: str, rng: np.random.Generator) -> None:
    """One column of per-seed dots over a thin bar for the seed range, with the seed mean on top in the marker of
    the condition (after `dots` in ex-2.2.9).
    """
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=ink(c), lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + rng.uniform(-0.08, 0.08, len(v)), v, "o", ms=2.4, color=ink(c), alpha=0.5, zorder=3, mew=0)
    ax.plot(x, v.mean(), MARKER[c], ms=5.5, color=ink(c), zorder=4, mec=light_dark("white", "#111"), mew=0.6)


def gate_line(ax, y: float, fail: str) -> None:
    """A dashed gate line with the failing side hatched (after `gate_line` in ex-2.2.9)."""
    ax.axhline(y, color=rule(), lw=0.9, ls="--", zorder=1)
    lo, hi = ax.get_ylim()
    span = (lo, y) if fail == "below" else (y, hi)
    ax.axhspan(*span, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, zorder=0, alpha=0.1)
    ax.set_ylim(lo, hi)


def condition_ticks(ax) -> None:
    ax.set_xticks(range(len(ex.CONDITIONS)), [SHOWN[c] for c in ex.CONDITIONS], fontsize=7, rotation=20, ha="right")
    ax.set_xlim(-0.5, len(ex.CONDITIONS) - 0.5)
    ax.tick_params(labelsize=7)


# %%

r"""
# Three recipes side by side

/// tip |
<!-- lede -->
We scored the recipe of record and two soft-start recipes the same way, on the runs we already had. The soft start on the step stayed close to the recipe of record on every measurement. The pool over slices 1 to 3 removed about as much and cost the task least, but its edit still spilled past the criterion on two seeds of eight. That spill enters with the edit at block 1, the one slice the pool pulls and the step leaves alone.
///
<!-- REVIEW: "matched ... on every measurement" became "stayed close to": the soft start removes less than `falloff` on seven seeds of eight (paired), by a couple of hundredths on the mean. Verify: the paired column of the E1 table. "the spill comes in at block 1" became "enters with the edit at block 1", which is what the restricted edits of E3 show. -->


The recipe of record, `falloff`, was chosen in [split-anti-seeds](/docs/m2/split-anti-seeds/report.py) before the pooling temperature could start soft. Two recipes from the [soft-pool experiment](/docs/m2/soft-pool/report.py) have since looked close to it. `step, soft τ` is the same recipe with a soft start. It kept the anchor on every seed of the [loss-rate screen](/docs/m2/loss-rate/report.py), and against the median baseline of [reference-baseline](/docs/m2/reference-baseline/report.py) its spill is within the criterion on every run.

`pool 1–3, soft τ` pools the pull over slices 1 to 3 as well as over positions. So the pull chooses the slice within that range, where the other recipes name the blocks. That makes it the easier recipe to explain and to carry to another model. It is worth having even if it scores a little lower, as long as we know how much lower.

So how do the three compare when every measurement the D2.2 design asks for is taken the same way on each? This report trains nothing. It scores the stored runs of all three at seeds 704 to 711, with two small new passes: the write bound on the two soft conditions, and the median baseline on the seven loss-rate runs that were trained to the end.
"""

# %%

r"""
## Observations

- [Removal, spill and task (E1)](#removal-spill-and-task-e1): the three remove about the same share of the op, and the two soft recipes remove a couple of hundredths less than `falloff` on most seeds. `falloff` and `step, soft τ` stay within the spill criterion on every run, and the pool goes over it on two. All three pass the task gate on the mean; the pool costs the least, and only `falloff` has an op past the per-op line, on one seed.
- [What the edit does (E2)](#what-the-edit-does-e2): past the embedding, the write stays near the bound in all three. The pool goes a little over it at slice 1, on positions where the bound is a few degrees. The grading and the landing barely differ. Under the pool the other ops sit closer to e₁ at the embedding and at slice 1.
- [Where the op sits (E3)](#where-the-op-sits-e3): the pool settles on slice 3 at every seed, with the seeds nearly on top of one another, and the step on slice 4. Under every recipe, the edit at blocks 2 to 4 removes the op and spills nothing, and the edit at the embedding alone spills nothing either. What spill there is enters with the edit at block 1, and most of it comes under the pool. The e₁ components of the color embeddings follow lightness more closely under the pool than under the step, but that lean is not what the edit takes away.

## Scope and method

This is an exploratory re-analysis, with no preregistration and no gate. It covers three conditions at model seeds 704 to 711, 400 epochs each, all with the `falloff` anti weight:

- `falloff`, the recipe of record: the step pull on slices 2 to 4, pooled over positions at a fixed temperature $\tau_p$ of 0.1 (the runs of split-anti-seeds).
- `step, soft τ`: the same step, with $\tau_p$ starting at 1 and sharpening to 0.1 by the end of the pull warm-up at epoch 40 (soft-pool).
- `pool 1–3, soft τ`: one pull pooled over the slice and position pairs of slices 1 to 3, with both temperatures starting soft ($\tau_p$ 1 → 0.1, and the slice temperature $\tau_\ell$ 4 → 0.4) and nothing pulling slice 4 (soft-pool).

The runs at a seed share their initialization, batches, and label draws, and the control of ex-2.2.23 at that seed (trained without the anchor) shares them too. So each comparison is paired by seed. A difference between two conditions is set against the spread of the paired differences over the eight seeds, rather than against the seed range of either condition. We report the mean paired difference from `falloff`, with its range and the count of seeds on each side of zero.

The comparison is of recipes, not of factors. `pool 1–3, soft τ` differs from `step, soft τ` in which slices it pulls as well as in pooling over them. So a gap between those two cannot be put down to either change alone. Soft-pool lists what each change moves.

Eight seeds are too few to compare how often a recipe loses the anchor. None of the 24 runs here lost it. The loss-rate screen counted losses at 40 seeds each for `falloff` (one loss) and `step, soft τ` (none), and bounded the rates at about one in nine and one in fourteen. The pool has no count beyond these eight.
<!-- REVIEW: "about a tenth" for both became the two bounds loss-rate quotes (11% and 7%). Verify: the counts table in E2 of loss-rate. --> This report quotes those counts and does not add to them.

The loss-rate screen trained seven of its runs to 400 epochs: the four `falloff` runs it did not count as kept (one of them the lost run), and three `step, soft τ` runs it finished as a check. They have no control at their seeds, so removal and spill are scored against the median baseline alone. They were picked as slow or doubtful seeds, so they are a check on the slow end and not a sample of either condition.

Their task change is taken against the mean of the eight controls at 704 to 711, as loss-rate took it. The eight controls differ from one another by less than a fiftieth on the mean score, which is about how far the choice of control could move it.
<!-- REVIEW: the assumptions review left this as an open decision (quote the task change against the seed mean of the controls, or leave the seven without one). Quoted, since loss-rate found one of them below the task gate and a recipe comparison should show it. Verify: CONTROL_TASK_SPREAD, and the task columns of the finished-runs table. -->

## The measurements

Every edit is scored on the 14,000 held-out contexts of ex-2.2.21, 2,000 per op. The edit takes a share of the e₁ component out of the state at every position and every slice. That share is the dose: a quarter, a half, three quarters, or all of it.

- Removal: the drop on `difference` under the full edit, as a share of the way from the clean score to the target null. Higher is better.
- Spill: the largest drop on any other op at any dose. A run is within the criterion when it is at most 0.02.
- Baseline: both drops have a baseline subtracted, per op and dose (the rule of the D2.2 design). The baseline is the median drop when the same edit is made along each of the other coordinate axes of the same run. Removal with nothing subtracted is shown beside it.
- Task change: the clean score per op, less that of the control at the same seed. The task gate of the D2.2 design fails if the mean over ops falls below −0.02, or any one op below −0.05.
- Write bound, grading and landing, as in [edit-effects](/docs/m2/edit-effects/report.py). The write is how much e₁ the blocks after an edit put back on the states of the other ops. It is compared with the clean model, which sets the bound. The grading is the share of the way to the target null the edit takes on `difference` contexts, by bin of the posterior on `difference`. The landing is the share of the distance to the target null that the full edit covers, over whole answer distributions. The grading is measured context by context, but the median baseline exists only per op. So the grading is netted against the control at the same seed instead, as in edit-effects, with the share before netting beside it.
- Where the op sits: the alignment α with e₁ at the example answers, by slice, on the clean pass over the held-out `difference` contexts (stored by the write pass). Beside it is the removal by the full edit restricted to block 1, to blocks 2 to 4, and to block 4. The median baseline exists only for the edit at every slice. So the restricted removal is net of the same restricted edit on the control at that seed, as in split-anti-seeds.
<!-- REVIEW: the restricted removal had no baseline named. Reference-baseline scored the other axes under the every-slice edit only, and split-anti-seeds netted its restricted edits against the control's (`slice_removal` in its report). Verify: `edits` in the stored measurements holds one raw drop per slice set and op. -->
"""

# %%

r"""
## Removal, spill and task (E1)

Do the three recipes remove as much, spill as little, and cost the task as little, at each seed? The main measure is removal at the full dose, with the spill at the worst dose beside it and the task change after that. The paired difference from `falloff` at each seed says whether a gap is larger than the variation between seeds would make by itself. The seven finished loss-rate runs come last.
"""


@memo
def e1_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="recipe-compare-scores",
        alt_text=alt_text,
        caption=f"""
            **Removal, spill and task, by condition.** One column per condition, the {len(SEEDS)} seeds as faded
            dots over a thin bar for their range, and the seed mean on top in the marker of the condition.
            **Left:** removal at the full dose, net of the median over the other axes of the run. **Middle:** spill,
            the largest net drop on another op at any dose; the dashed rule is the selectivity criterion, with the
            side that misses it hatched. **Right:** the task change, the mean clean score over ops less that of the
            control at the seed; the dashed rule is the task gate on that mean.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.6), layout="constrained")
        axes = cast(AxesRow, axes)
        rng = np.random.default_rng(0)
        for ax, key, ylabel, ylim in (
            (axes[0], "removal", "removal", (0.7, 1.0)),
            (axes[1], "spill", "spill, worst dose", (-0.015, 0.06)),
            (axes[2], "task", "task change, mean over ops", (-0.04, 0.015)),
        ):
            for i, c in enumerate(ex.CONDITIONS):
                dots(ax, i, data[key][c], c, rng)
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylabel, fontsize=8)
            condition_ticks(ax)
        gate_line(axes[1], CRITERION, "above")
        gate_line(axes[2], TASK_MEAN_GATE, "below")
        return fig

    return _plot()


def e1_figure() -> str:
    alt = (
        "Three panels of per-seed dots, one column per condition. Removal sits between about 0.77 and 0.91 in all "
        "three, with the seed means within a few hundredths of each other. Spill stays near zero for falloff and "
        "step, soft tau, while the pool has several seeds near or over the criterion line and one far over it. The "
        "task change is near zero in all three, with falloff spread lowest and the pool tightest."
    )
    return e1_draw({"removal": REMOVAL, "spill": SPILL, "task": {c: TASK[c].mean(axis=1) for c in ex.CONDITIONS}}, alt)


def e1_table() -> str:
    rows = []
    for c in ex.CONDITIONS:
        worst = np.unravel_index(TASK[c].argmin(), TASK[c].shape)
        gap = GAP.get(c)
        rows.append(
            [
                f"`{SHOWN[c]}`",
                fmt(REMOVAL[c]),
                fmt(RAW[c]),
                "—" if gap is None else f"{fmt(gap)}; {int((gap > 0).sum())} up, {int((gap < 0).sum())} down",
                fmt(SPILL[c], ".3f"),
                f"{N_OVER[c]} of {len(SEEDS)}",
                f"{int((CTRL_SPILL[c] > CRITERION).sum())} of {len(SEEDS)}",
                fmt(TASK[c].mean(axis=1), ".3f"),
                f"{num(TASK[c].min(), '.3f')} (`{OPS[worst[1]]}`, {SEEDS[worst[0]]})",
            ]
        )
    return table_html(
        [
            "condition",
            "removal ↑",
            "removal, nothing subtracted ↑",
            "removal less `falloff`, paired",
            "spill ↓",
            "over criterion ↓",
            "over, control median ↓",
            "task change ↑",
            "worst op change ↑",
        ],
        rows,
        f"""**Removal, spill and task**, seed mean and range over {len(SEEDS)} runs. Removal and spill are net of the
        median drop over the other {len(AX[label_of(FALLOFF, SEEDS[0])]["drop"]) - 1} coordinate axes of the run,
        and "over, control median" counts the runs over the criterion with the median taken in the control at the
        seed instead. The paired gap is the removal of the condition less that of `falloff` at the same seed, with
        the count of seeds on each side of zero. The task change is the mean over ops of the clean score less that of
        the control at the seed; the worst op change is the largest loss on one op at any seed.""",
    )


def per_seed_table() -> str:
    head = ["condition", "measure", *map(str, SEEDS)]
    rows = []
    for c in ex.CONDITIONS:
        rows.append([f"`{SHOWN[c]}`", "removal", *(num(v) for v in REMOVAL[c])])
        rows.append(
            ["", "spill", *(f"<strong>{num(v, '.3f')}</strong>" if v > CRITERION else num(v, ".3f") for v in SPILL[c])]
        )
    return table_html(
        head,
        rows,
        """**Removal and spill by seed**, as in the table above. Bold marks a run over the selectivity criterion.""",
    )


def finished_table() -> str:
    rows = []
    for lbl, sc in FIN_SCORES.items():
        cond, seed = lbl.rsplit("-s", 1)
        t = FIN_TASK[lbl]
        rows.append(
            [
                f"`{SHOWN[cond]}`",
                seed,
                num(sc["removal"]),
                num(sc["raw"]),
                num(sc["worst_spill"], ".3f"),
                num(t.mean(), ".3f"),
                f"{num(t.min(), '.3f')} (`{OPS[int(t.argmin())]}`)",
            ]
        )
    return table_html(
        ["condition", "seed", "removal ↑", "nothing subtracted ↑", "spill ↓", "task change ↑", "worst op change ↑"],
        rows,
        f"""**The loss-rate runs trained to 400 epochs.** Removal and spill net of the median over the other axes
        of the run, as above. These seeds have no control, so the task change is against the mean of the
        {len(SEEDS)} controls at 704 to 711, whose mean scores span {CONTROL_TASK_SPREAD:.3f} from the lowest to
        the highest.""",
    )


# %%

e1_figure()

# %%

e1_table()

# %%

rf"""
The three remove about the same share of the op. The two soft recipes come in under `falloff` on seven seeds of eight, by a couple of hundredths on the mean, a gap that is small next to the spread between seeds (the paired column of the table). With nothing subtracted the order is the same, so the gap does not come from the baseline.

On spill, the recipes differ. `falloff` and `step, soft τ` stay well within the criterion on every run. The pool goes over it on {N_OVER[POOL]} seeds: 707, and 711 by more than twice the criterion. On a third seed (705) it sits at the criterion, and which side it falls on depends on whose median is subtracted.

Under the pool, the spill lands on `value-hsv` on three seeds. That is the op on which the [spill-timing experiment](/docs/m2/spill-timing/report.py) found the late spill of this condition. On the two seeds over the criterion it lands on `mix` and `darken`.

On the task, the order reverses. All three pass the gate on the mean over ops, and the pool costs the least, with the narrowest spread. `falloff` is the one with an op past the per-op line: `hue-hsv` at seed 708, a little past it, which split-anti-seeds already noted. The worst op under the two soft recipes is `hue-hsv` too, at well under the line.
"""

# %%

per_seed_table()

# %%

finished_table()

# %%

r"""
The runs the loss-rate screen trained to the end tell the same story at other seeds. The lost `falloff` run (714) removes nothing with the median subtracted either. It sits a hair under the mean line of the task gate, by less than the controls differ among themselves. Every kept run removes about as much as the runs above and spills nothing.

One `step, soft τ` run (732) scores below both lines of the task gate against the mean of the controls, by more than the spread of the controls could account for. Loss-rate saw the same and had no control at that seed to settle it, and neither do we. Counting the eight runs at 704 to 711, the soft start has one run below the task gate in eleven, and `falloff` has one past the per-op line in twelve, besides the lost run.
<!-- REVIEW: "one in eleven" counts the 8 + 3 kept step-soft runs; "one in twelve" counts falloff's 8 + 4 finished, on the per-op line only (falloff 708). Verify: the two tables of E1. -->
<!-- REVIEW: added that falloff 714 is under the mean line too (−0.022 against −0.02, with the controls spanning 0.017), so the tally does not read as if only the soft start had a run under it. Verify: the task column of the finished-runs table and CONTROL_TASK_SPREAD. -->
"""

# %%

r"""
## What the edit does (E2)

The measurements of E1 are means over all the contexts of an op. Edit-effects showed on `falloff` that the edit also behaves well context by context. The blocks downstream put back less e₁ on the other ops than the clean model has. The drop on each `difference` context follows how much that context needed the op. And the edited answers land about half of the way to the target null. Do the soft start and the pool keep those properties? The main measure is the write as a ratio of the bound, with the grading and the landing beside it.
"""


@memo
def e2_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="recipe-compare-edit",
        alt_text=alt_text,
        caption=f"""
            **What the edit does, by condition.** **Left:** the bound and the write at the example answers of the
            other ops that do not fit `{ex.ANCHORED_OP}`, by slice: arcsin |α| at the {Q:.0%} quantile, clean (solid,
            the bound) and as it arrives under the full edit (dashed, the write), seed mean. **Right:** the share of
            the way to the target null the full edit takes on `{ex.ANCHORED_OP}` contexts, by bin of the posterior on
            `{ex.ANCHORED_OP}`, net of the control, seed mean; the dashed rule at 1 is the target null.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), layout="constrained")
        axes = cast(AxesRow, axes)
        x = np.arange(len(SLICE_TICK))
        for c in ex.CONDITIONS:
            axes[0].plot(x, data["bound"][c], "-", marker=MARKER[c], ms=4, color=ink(c), lw=1.3, label=SHOWN[c])
            axes[0].plot(x, data["write"][c], "--", marker=MARKER[c], ms=3, mfc="none", color=ink(c), lw=1.0)
        axes[0].set_xticks(x, SLICE_TICK)
        axes[0].set_xlabel("slice", fontsize=8)
        axes[0].set_ylabel("angle, degrees", fontsize=8)
        axes[0].set_ylim(0, 50)
        b = np.arange(len(BIN_LABEL))
        for c in ex.CONDITIONS:
            axes[1].plot(b, data["share"][c], "-", marker=MARKER[c], ms=4, color=ink(c), lw=1.3)
        axes[1].axhline(1, color=rule(), lw=0.7, ls="--", zorder=0)
        axes[1].set_ylim(0.5, 1.05)
        axes[1].set_xticks(b, BIN_LABEL, fontsize=7)
        axes[1].set_xlabel(f"posterior on {ex.ANCHORED_OP}", fontsize=8)
        axes[1].set_ylabel("share of the way to the null", fontsize=8)
        for ax in axes:
            ax.tick_params(labelsize=7)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False, fontsize=8)
        return fig

    return _plot()


def e2_figure() -> str:
    alt = (
        "Left: the bound at the example answers of the other ops rises from a few degrees at the embedding to about "
        "32 degrees at slice 1 under the two step recipes and about 43 under the pool, which is also wider at the "
        "embedding; the write under the edit falls to about 11 degrees at slice 2 and 3 to 5 at slices 3 and 4 in "
        "all three. Right: the share of the way to the target null is about level across the posterior bins, "
        "between about 0.73 and 0.88, with the three conditions close together and the pool lowest in the first bin."
    )
    data = {
        "bound": {c: angle(seed_mean(c, "ans_bound")) for c in ex.CONDITIONS},
        "write": {c: angle(seed_mean(c, "ans_write")) for c in ex.CONDITIONS},
        "share": {c: SHARE[c][:, :-1].mean(axis=0) for c in ex.CONDITIONS},
    }
    return e2_draw(data, alt)


def worst_site(c: str) -> tuple[int, int, float, float]:
    """Where the seed-mean write is furthest over the bound: slice, position, ratio, and the bound in degrees."""
    r = RATIO[c][1:]
    sl, p = np.unravel_index(r.argmax(), r.shape)
    return int(sl) + 1, int(p), float(r[sl, p]), float(angle(seed_mean(c, "bound"))[sl + 1, p])


ROLE = ("a", "?", "b", "=", "y", ",")


def e2_table() -> str:
    rows = []
    for c in ex.CONDITIONS:
        sl, p, r, b = worst_site(c)
        over = OVER[c]
        rows.append(
            [
                f"`{SHOWN[c]}`",
                f"{r:.2f} (slice {sl}, <em>{ROLE[p % UNIT]}</em>, {b:.1f}°)",
                f"{len(over)} (≤ {max(o[4] for o in over):.0f}°)" if over else "0",
                f"{angle(seed_mean(c, 'ans_bound'))[0]:.0f}°, {angle(seed_mean(c, 'ans_bound'))[1]:.0f}°",
                fmt(SHARE[c][:, -1]),
                fmt(RAW_SHARE[c][:, -1]),
                fmt(LANDING[c]),
            ]
        )
    return table_html(
        [
            "condition",
            "largest write over bound ↓",
            "sites over 1.25×, all runs ↓",
            "bound at example answers, emb and slice 1 ↓",
            "share removed ↑",
            "share, not net of the control ↑",
            "landing ↑",
        ],
        rows,
        f"""**The write, the grading and the landing.** The largest ratio of write to bound in angle over the
        slices past the embedding and every position, from the seed-mean quantiles as in edit-effects, with where it
        falls and the bound there. The sites over 1.25× count slice and position pairs over all {len(SEEDS)} runs,
        each run taken on its own, of {4 * N_POS * len(SEEDS)}, with the largest bound among them. The bound at the
        example answers is over those that do not fit `{ex.ANCHORED_OP}`. The share removed is over every
        `{ex.ANCHORED_OP}` context, and the landing is that of split-anti-seeds and soft-pool; seed mean and
        range.""",
    )


# %%

e2_figure()

# %%

e2_table()

# %%

rf"""
Context by context, the three behave alike. Past the embedding the write stays close to the bound under every recipe, and from slice 2 on the other ops arrive with much less e₁ than the clean model gives them, the same in all three. The grading has the same shape under each: about the same share removed in every bin of the posterior, with the pool lower in the bin of contexts that needed the op least. The landing is about half of the way to the target null in all three.

The pool is where they differ, at the first two slices. There the other ops sit closer to e₁ under the pool: the bound at their example answers is about {angle(seed_mean(POOL, "ans_bound"))[0] / angle(seed_mean(FALLOFF, "ans_bound"))[0]:.0f}× as wide at the embedding as under `falloff`, and about a third wider at slice 1.

At slice 1 under the pool, the write goes a little over the bound at the example operands, where the bound is a few degrees. That is the largest ratio of any recipe, and the only one past the tolerance of ex-2.2.1 at the seed mean, but in degrees it is small. More single sites go over the tolerance under the pool than under the step, all where the bound is 6° or less.
<!-- REVIEW: the bound at the example answers is per example answer that does not fit `difference`, as in edit-effects; the ratio at the embedding is about 3× and at slice 1 about 1.35×. Verify: the third column of the table. -->

So the pool does not change what the edit does downstream. What it changes is how close the other ops sit to e₁ at the slices before the model has read the context. The step deliberately puts no pull there.
"""

# %%

r"""
## Where the op sits (E3)

The step puts the pull on slices 2 to 4, while the pool lets the pull choose among slices 1 to 3. In [slice-terms](/docs/m2/slice-terms/report.py), a pull pooled over blocks 1 to 4 at a fixed temperature settled on block 4 on most seeds and on the query answer on one, where the op could not be removed. Soft-pool found the pool over slices 1 to 3 peaking at slice 3 on average. Does it settle on the same slice at every seed, and does the edit find it there? The main measure is α at the example answers by slice, per seed, with the removal by the edit restricted to a set of slices beside it.
"""

PULL = {FALLOFF: [0, 0, 1, 1, 1], STEP_SOFT: [0, 0, 1, 1, 1], POOL: [0, 1, 1, 1, 0]}  # by slice, emb first
ANTI_HOLD = list(ex.sas.FALLOFF_HOLDS)


@memo
def e3_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="recipe-compare-depth",
        alt_text=alt_text,
        caption=f"""
            **Where the anchor sits by depth.** **Top:** α at the example answers of the held-out
            `{ex.ANCHORED_OP}` contexts, clean, by slice: one hairline per run and the seed mean in the marker of the
            condition, with the control as the gray ghost. **Underneath:** the slices each condition pulls (for the
            pool, the slices in its pool) and the anti hold every condition anneals to.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(
            3, 1, figsize=(6.4, 4.0), layout="constrained", sharex=True, gridspec_kw={"height_ratios": [4, 1.2, 1]}
        )
        axes = cast(AxesRow, axes)
        x = np.arange(len(SLICE_TICK))
        ax = axes[0]
        ax.plot(x, data["alpha"][CONTROL].mean(axis=0), "-", color=ghost(), lw=1.0, zorder=1, label="control")
        for c in ex.CONDITIONS:
            for run in data["alpha"][c]:
                ax.plot(x, run, "-", color=ink(c), lw=0.4, alpha=0.5, zorder=2)
            ax.plot(x, data["alpha"][c].mean(axis=0), "-", marker=MARKER[c], ms=4.5, color=ink(c), lw=1.3,
                    zorder=3, label=SHOWN[c])  # fmt: skip
        ax.set_ylim(0, 0.8)
        ax.set_ylabel("α, example answers", fontsize=8)
        ax = axes[1]
        for i, c in enumerate(ex.CONDITIONS):
            on = [k for k in x if PULL[c][k]]
            ax.plot(on, [i] * len(on), MARKER[c], ms=4.5, color=ink(c), ls="-", lw=1.0)
        ax.set_yticks(range(3), [SHOWN[c] for c in ex.CONDITIONS], fontsize=6.5)
        ax.set_ylim(-0.6, 2.6)
        ax.invert_yaxis()
        ax.set_ylabel("pulled", fontsize=8)
        ax = axes[2]
        ax.plot(x, data["anti"], "-", marker="v", ms=3.5, color=rule(), lw=1.0)
        ax.set_ylim(0, 0.25)
        ax.set_ylabel("anti hold", fontsize=8)
        axes[-1].set_xticks(x, SLICE_TICK)
        axes[-1].set_xlabel("slice", fontsize=8)
        for a in axes:
            a.tick_params(labelsize=7)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False, fontsize=8)
        return fig

    return _plot()


def e3_figure() -> str:
    alt = (
        "Top: under falloff and step, soft tau the alignment at the example answers rises from near zero at the "
        "embedding to about 0.36 at slice 1, 0.6 at slice 2 and about 0.69 at slices 3 and 4, with every seed on "
        "top of the others. The pool is higher at the embedding and slice 1 (about 0.45), reaches the same level at "
        "slice 3, and falls back to about 0.53 at slice 4, again with the seeds close together. The control stays "
        "below 0.1 at every slice. Underneath, the step conditions pull slices 2 to 4 and the pool slices 1 to 3, "
        "and the anti hold falls in a straight line from 0.2 at the embedding to 0.03 at slice 4."
    )
    return e3_draw({"alpha": ALPHA_D, "anti": ANTI_HOLD}, alt)


def e3_table() -> str:
    rows = []
    for c in ex.CONDITIONS:
        peaks = {int(k): int(v) for k, v in zip(*np.unique(PEAK[c], return_counts=True), strict=True)}
        rows.append(
            [
                f"`{SHOWN[c]}`",
                ", ".join(f"slice {k}: {v}" for k, v in peaks.items()),
                *(fmt(SLICE_REMOVAL[c][n]) for n in SLICE_SETS),
                *(fmt(SLICE_SPILL[c][n], ".3f") for n in ("embedding", "block 1", "blocks 2 to 4", "blocks 1 to 4")),
                fmt(LEAN_R[c]),
            ]
        )
    rows.append(["control", *[""] * 8, fmt(LEAN_R[CONTROL])])
    return table_html(
        [
            "condition",
            "peak α, seeds",
            "removal, block 1 alone ↑",
            "removal, blocks 2 to 4 ↑",
            "removal, block 4 alone",
            "spill, embedding alone ↓",
            "spill, block 1 alone ↓",
            "spill, blocks 2 to 4 ↓",
            "spill, blocks 1 to 4 ↓",
            "lightness on e₁, embedding table",
        ],
        rows,
        f"""**Where the op sits, and where the edit finds it.** The slice at which α at the example answers of
        `{ex.ANCHORED_OP}` contexts peaks, with the count of seeds. Removal and spill under the full edit restricted
        to a set of slices, at every position, net of the same edit on the control at the seed; seed mean and range.
        The last column is the correlation, over the color embeddings, of the e₁ component with lightness (the lean
        of embedding-lean; negative means darker colors sit further along e₁).""",
    )


def e3_seed_table() -> str:
    head = ["condition", "measure", *map(str, SEEDS)]
    rows = []
    for c in ex.CONDITIONS:
        rows.append([f"`{SHOWN[c]}`", "removal, block 1 alone", *(num(v) for v in SLICE_REMOVAL[c]["block 1"])])
        rows.append(["", "removal, blocks 2 to 4", *(num(v) for v in SLICE_REMOVAL[c]["blocks 2 to 4"])])
        rows.append(["", "spill, block 1 alone", *(num(v, ".3f") for v in SLICE_SPILL[c]["block 1"])])
    return table_html(head, rows, """**The restricted edits by seed**, as in the table above.""")


# %%

e3_figure()

# %%

e3_table()

# %%

r"""
Each recipe settles in one place on every seed. Under the step, α at the example answers climbs to slice 4 and stays there, and the soft start changes nothing in that shape. The pool climbs higher at slice 1, peaks at slice 3 on every seed, and falls back at slice 4, where nothing pulls it. Its eight seeds lie nearly on top of one another, and none of them settled on the query answer.

The edit finds the op wherever it settles. The edit at blocks 2 to 4 alone removes most of it under every recipe, except at seed 710. There it falls short under all three, as it did in every arm of split-anti-seeds, and the edit at block 1 alone removes nearly all of it. The edit at block 4 alone removes nothing in any recipe, the pool included. Under the pool the edit at block 1 alone removes more, and more evenly across seeds, than under the step, which fits the pool pulling there.
"""

# %%

e3_seed_table()

# %%

rf"""
The same restricted edits show where the spill of E1 comes in. The edit at blocks 2 to 4 costs the other ops nothing under any recipe. The edit at block 1 alone costs them a little under `falloff`, more under `step, soft τ`, and most under the pool. Under the pool, the block-1 edit costs most at 707 and 711, the two seeds over the criterion in E1, and at 708. The edit at the embedding alone costs the other ops nothing under any recipe, and the edit at blocks 1 to 4 costs what the edit at every slice does. So under the pool the spill enters with block 1.

The restricted edits are not the full edit taken apart, since each later block reads what the earlier ones were given. And the block-1 edit on its own overstates the cost: under `step, soft τ` it goes over the criterion on {int((SLICE_SPILL[STEP_SOFT]["block 1"] > CRITERION).sum())} seeds whose full edit spills nothing. So it says where the other ops are open to the edit, not how much the full edit will cost them.
<!-- REVIEW: added the embedding-only and blocks-1-to-4 edits, and the step's block-1 spill, so the block-1 reading rests on an elimination rather than on the block-1 edit alone. Verify: SLICE_SPILL["embedding"] and ["blocks 1 to 4"] against ["every slice"] in the stored `edits`. -->

The embedding table leans the same way, but it does not account for the spill on its own. Under the pool, the e₁ components of the color embeddings follow lightness more closely than under the step: darker colors sit further along e₁. The pool shows this on average and closely on every seed, while under the step it varies from seed to seed. In the control the same correlation scatters around zero.

That is the lean of [embedding-lean](/docs/m2/embedding-lean/report.py), the stand-in the model offers for `difference` before it has read the context. It is also the reason the step leaves block 1 out of the pull. So the pool, by pulling slice 1, seems to bring back part of the lean.

But the edit at the embedding alone, which takes that lean out, spills nothing. And in the states at slice 1, the lean has faded to about the same level under all three recipes ({num(LEAN_STATES[POOL][:, 1].mean())} under the pool against {num(LEAN_STATES[STEP_SOFT][:, 1].mean())} under `step, soft τ`). So what the block-1 edit removes is what block 1 writes onto e₁ for the other ops, which these measurements do not resolve. The lean in the table is a sign of the pull reaching slice 1, rather than the thing the edit takes away. Within the pool the lean is about the same on every seed, so it does not say why 707 and 711 spill and 704 does not, either.
<!-- REVIEW: the lean paragraph read as the mechanism of the spill ("the edit at block 1 takes some lightness away from every op with it"). Lowered: the embedding-only edit spills nothing under the pool, and `r_light_states_other` at slice 1 is about alike under the three recipes, so the lean fits the pull reaching slice 1 but not the spill itself. Verify: LEAN_STATES[:, 1] and SLICE_SPILL["embedding"]. -->
"""

# %%

r"""
## Discussion

On these eight seeds `step, soft τ` is about as good a recipe as `falloff`. It removes a couple of hundredths less, spills no more, and costs the task about as little. The edit also acts the same way context by context. Its one run below the task gate is a loss-rate run with no control at its seed. And `falloff` has a run of its own past the per-op line, so on the task the two look about level.

The measure that could still set them apart is how often each loses the anchor, which these runs cannot add to. Loss-rate found no losses in 40 for the soft start and one for `falloff`, a difference too small at those counts to rest on.

`pool 1–3, soft τ` is close behind on removal and ahead on the task, and the op settles in the same place on every seed, which is what one would want of a recipe that chooses the slice for itself. Its shortfall is spill.

If the reading of E3 holds, that spill comes from what the pool asks of block 1, and not from pooling over slices as such. Block 1 is the one slice the pool pulls and the step leaves alone, and the edit there is where the spill enters. What block 1 writes onto e₁ for the other ops is not resolved here, though.

A pool over slices 2 to 4 would separate the two while still leaving the slice for the pull to choose. These runs do not say whether the op would then settle on block 3 or 4, or whether it would keep the anchor on every seed as the soft start did here.

The pool is the easier recipe to explain because it needs no choice of block within its range. It still has a range, slices 1 to 3, and the measurements here point at the edge of that range rather than at the pooling, though no run here separates the two.
<!-- REVIEW: "suggest the edge of that range matters more than the pooling" became "point at the edge ... though no run here separates the two": the scope says a gap between the pool and the step cannot be put down to either change alone, and the discussion should not claim more. The lean sentence in the paragraph above now follows the lowered reading in E3. -->
"""
