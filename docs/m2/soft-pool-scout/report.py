# ruff: noqa: B018
# title: Soft pool scout

# An exploratory report in two stages: sixteen schedule conditions of `pool 1–3, soft τ` at its eight soft-pool seeds,
# screened at epoch 64 for lost anchors and early spill (stage 1), then the leaders carried on to epoch 400 (stage 2).
# `experiment.py` beside this script trains and scores the runs.
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


def by_label(body: dict) -> dict[str, dict]:
    return {r["label"]: r for r in body["runs"]}


SCREEN_BODY = fetch_json(ex.SCREEN_REF)
SCREEN = by_label(SCREEN_BODY)
SCREEN_TRAJ = fetch_json(ex.SCREEN_TRAJ_REF)
# The base: spill-timing's reruns of soft-pool's `pool 1–3, soft τ`, scored at every record.
BASE_TRAJ = fetch_json(ex.spt.TRAJ_REF)
# Spill-timing's scoring of the loss-rate screen's epoch-64 checkpoints of `falloff` and `step, soft τ`.
QUIET = by_label(fetch_json(ex.spt.STORED_REF))
DESIGN = SCREEN_BODY["design"]

OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
STOP = DESIGN["stop_epoch"]
CRITERION = 0.02  # the selectivity criterion of the D2.2 design
UNIT = 6  # tokens per equation: a ? b = y and a separator
ANSWERS = [UNIT * e + 4 for e in range(ex.K)]
QUERY_ANSWER = UNIT * ex.K + 4
SLICE_TICK = ["emb", "1", "2", "3", "4"]
LOST_MAX, KEPT_MIN = ex.lr.LOST_MAX, ex.lr.KEPT_MIN
CONDITIONS = [ex.Condition(**{k: v for k, v in c.items() if k != "anchor"}) for c in DESIGN["conditions"]]
for c in CONDITIONS:  # JSON turns the warm-up tuple into a list
    object.__setattr__(c, "warmup", tuple(c.warmup))
NAMES = [c.name for c in CONDITIONS]
SHOWN = {c.name: c.shown_as for c in CONDITIONS}
BY_NAME = {c.name: c for c in CONDITIONS}


def num(v: float, spec: str = ".3f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


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


# --- Each run at the stop -------------------------------------------------------------------------------------


def reference_spill(clean, edited, ref) -> float:
    """The largest drop on another op under the full edit, less the median drop under the reference edits
    (spill-timing): the median reference score less the edit score, at its largest over the other ops.
    """
    del clean  # the clean score cancels
    net = np.median(np.asarray(ref), axis=0) - np.asarray(edited)
    return float(net[OTHER].max())


def worst_op(edited, ref) -> str:
    net = np.median(np.asarray(ref), axis=0) - np.asarray(edited)
    return OPS[OTHER[int(net[OTHER].argmax())]]


def at_stop(traj: dict) -> int:
    """The index of the first record at the stop epoch (a stopped run records it twice)."""
    return int(np.flatnonzero(np.isclose(np.asarray(traj["epoch"]), STOP))[0])


def run_at_stop(name: str, seed: int) -> dict[str, Any]:
    """Everything stage 1 measures on one run, from the screen for a new condition and from spill-timing's rerun for
    the base, so the two are scored the same way.
    """
    if name == ex.STORED:
        traj = BASE_TRAJ[ex.spt.label_of(seed)]["traj"]
        i = at_stop(traj)
        scores = {k: traj[k][i] for k in ("es_clean", "es_edit", "es_sham")}
    else:
        label = ex.label_of(name, seed)
        traj, scores = SCREEN_TRAJ[label]["traj"], SCREEN[label]
        i = at_stop(traj)
    alpha = np.asarray(traj["alpha_anchored"])[i]  # (slices, positions), on the `difference` probes
    return {
        "window": ex.lr.window_mean(traj),
        "spill": reference_spill(scores["es_clean"], scores["es_edit"], scores["es_sham"]),
        "op": worst_op(scores["es_edit"], scores["es_sham"]),
        "alpha_ans": alpha[:, ANSWERS].mean(axis=1),
        "alpha_q": alpha[:, QUERY_ANSWER],
    }


RUNS = {(n, s): run_at_stop(n, s) for n in NAMES for s in SEEDS}


def per_seed(name: str, key: str) -> np.ndarray:
    return np.array([RUNS[(name, s)][key] for s in SEEDS])


WINDOW = {n: per_seed(n, "window") for n in NAMES}
SPILL = {n: per_seed(n, "spill") for n in NAMES}
CALLS = {n: [ex.lr.call_of(v) for v in WINDOW[n]] for n in NAMES}
N_LOST = {n: sum(c != "kept" for c in CALLS[n]) for n in NAMES}
PEAK = {n: SLICE_TICK[int(per_seed(n, "alpha_ans").mean(axis=0).argmax())] for n in NAMES}

# The quiet recipes at the stop: the loss-rate screen's checkpoints, pooled over its seeds.
QUIET_SPILL = {
    arm: np.array(
        [
            reference_spill(r["es_clean"], r["es_edit"], r["es_sham"])
            for lab, r in QUIET.items()
            if lab.startswith(f"{arm}-s") and lab.endswith(f"-e{STOP}")
        ]
    )
    for arm in ("falloff", "step-soft")
}
QUIET_SHOWN = {"falloff": "falloff", "step-soft": "step, soft τ"}

# Ranking: no lost or uncertain call, then the least mean spill at the stop.
RANKED = sorted(NAMES, key=lambda n: (N_LOST[n] > 0, float(SPILL[n].mean())))
N_RUNS = len(RUNS)
N_KEPT = sum(c == "kept" for n in NAMES for c in CALLS[n])
LOWEST_CALL = min(float(WINDOW[n].min()) for n in NAMES)

# The crossing, pooled over its four timing levels at each slice temperature.
CROSSING = [c.name for c in ex.CROSSING]
LEVELS = sorted({c.slice_tau for c in ex.CROSSING})
LEVEL_RUNS = {lt: [n for n in CROSSING if BY_NAME[n].slice_tau == lt] for lt in LEVELS}
LEVEL_SPILL = {lt: np.concatenate([SPILL[n] for n in ns]) for lt, ns in LEVEL_RUNS.items()}
LEVEL_OVER = {lt: int((v > CRITERION).sum()) for lt, v in LEVEL_SPILL.items()}
LEVEL_ALPHA1 = {lt: float(np.concatenate([per_seed(n, "alpha_ans")[:, 1] for n in ns]).mean()) for lt, ns in LEVEL_RUNS.items()}  # fmt: skip
OVER_OPS: dict[str, int] = {}
for r in RUNS.values():
    if r["spill"] > CRITERION:
        OVER_OPS[r["op"]] = OVER_OPS.get(r["op"], 0) + 1
N_OVER = sum(OVER_OPS.values())
TOP_OP = max(OVER_OPS, key=OVER_OPS.__getitem__)

# --- Encoding ---------------------------------------------------------------------------------------------------

RULE = light_dark("#333", "#ddd")
ZERO = light_dark("#aaa", "#555")
GHOST = light_dark("#bbb", "#555")
QUIET_INK = light_dark("#777", "#aaa")
TAU_INK = {0.1: light_dark("#08306b", "#c6dbef"), 0.2: light_dark("#2171b5", "#6baed6"), 0.4: light_dark("#6baed6", "#2171b5")}  # fmt: skip
WARMUP_INK = {True: light_dark("#d95f02", "#fdae6b"), False: light_dark("#7570b3", "#bcbddc")}  # soft τ, fixed τ


def tex(shown: str) -> str:
    """A condition name for a figure label, with its symbols as mathtext."""
    return shown.replace("τ_ℓ", "$\\tau_\\ell$").replace("λ_a", "$\\lambda_a$")


def ink(c: ex.Condition):
    if c.name in [w.name for w in ex.WARMUPS]:
        return WARMUP_INK[c.sharp_by is not None]
    return TAU_INK[c.slice_tau]


def marker(c: ex.Condition) -> str:
    if c.warmup == ex.DELAY:
        return ">"
    if c.warmup == ex.STRETCH:
        return "<"
    if not c.slice_soft:
        return "s"
    return {10.0: "v", 20.0: "D", 40.0: "o", None: "P"}[c.sharp_by]


# --- The schedules ----------------------------------------------------------------------------------------------


def schedules_figure() -> str:
    from sca.anchoring import AnchorSpec

    base = ex.st.schedules()[0]
    epoch = np.linspace(0, 100, 401)
    data = {"epoch": epoch.tolist(), "conditions": []}
    for c in CONDITIONS:
        spec = AnchorSpec(**c.anchor(base))
        taus = np.array([spec.taus(e) for e in epoch])
        data["conditions"].append(
            {"name": c.name, "weight": spec(epoch).tolist(), "tau_p": taus[:, 0].tolist(), "tau_l": taus[:, 1].tolist()}
        )
    alt = """
        Three panels over epochs 0 to 100. Top: λ_a rises from zero to 0.1 by epoch 40 for most conditions, by 60
        from a start at 20 for the delay, and by 80 for the stretch. Middle: τ_p falls from 1 to 0.1 by epoch 10,
        20 or 40, or sits at 0.1 throughout for fixed τ. Bottom: τ_ℓ falls from 4 to 0.1, 0.2 or 0.4 on the same
        windows, or sits at its end value throughout.
    """
    return schedules_draw(data, alt)


@memo
def schedules_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-scout-schedules",
        alt_text=alt_text,
        caption="""
            **The schedules the conditions change.** The anchor weight $\\lambda_a$, the position pool temperature
            $\\tau_p$ and the slice pool temperature $\\tau_\\ell$ (both on a log scale) over the first hundred
            epochs, one line per condition, in its ink and marker; where conditions share a schedule, the later line is
            drawn over the earlier. After epoch 100 every condition is the same.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.6), layout="constrained", sharex=True)
        ep = np.asarray(data["epoch"])
        for c in data["conditions"]:
            cond = BY_NAME[c["name"]]
            for a_, key in zip(axes, ("weight", "tau_p", "tau_l"), strict=True):
                a_.plot(ep, c[key], color=ink(cond), lw=0.9, marker=marker(cond), ms=3, markevery=40, alpha=0.85)
        axes[0].set_ylabel("$\\lambda_a$", fontsize=8)
        axes[0].set_ylim(0, 0.11)
        for a_, label in zip(axes[1:], ("$\\tau_p$", "$\\tau_\\ell$"), strict=True):
            a_.set_yscale("log")
            a_.set_ylabel(label, fontsize=8)
            a_.minorticks_off()
        axes[1].set_yticks([0.1, 1], ["0.1", "1"])
        axes[2].set_yticks([0.1, 0.2, 0.4, 1, 4], ["0.1", "0.2", "0.4", "1", "4"])
        axes[-1].set_xlim(0, 100)
        axes[-1].set_xlabel("epoch", fontsize=8)
        for a_ in axes:
            a_.tick_params(labelsize=7)
        return fig

    return _plot()


# --- E1 and E2: the call and the spill at the stop ---------------------------------------------------------------


def screen_figure() -> str:
    data = {
        "names": NAMES,
        "window": {n: WINDOW[n].tolist() for n in NAMES},
        "spill": {n: SPILL[n].tolist() for n in NAMES},
        "quiet": {QUIET_SHOWN[a]: v.tolist() for a, v in QUIET_SPILL.items()},
        "lost_max": LOST_MAX,
        "kept_min": KEPT_MIN,
        "criterion": CRITERION,
    }
    alt = """
        Two panels with one column per condition. Top: the call sits between about 0.5 and 0.6 for every condition,
        far above the kept line at 0.28; the lowest single run, on the stretched warm-up, is at 0.36. Bottom: the
        four τ_ℓ 0.1 conditions sit just below zero with every seed under 0.01, like the falloff and step, soft τ
        columns at the right. The τ_ℓ 0.2 conditions sit near zero, with one seed at 0.13 (τ_ℓ fixed) and two near 0.03. The τ_ℓ 0.4
        conditions and the λ_a conditions have means between about 0 and 0.02 and seeds up to 0.05 to 0.18.
    """
    return screen_draw(data, alt)


def _dots(ax, x: float, v, color, m: str, rng) -> None:
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + rng.uniform(-0.12, 0.12, len(v)), v, "o", ms=2.2, color=color, alpha=0.5, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=5.5, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6)


@memo
def screen_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-scout-screen",
        alt_text=alt_text,
        caption=f"""
            **Each condition at epoch {STOP}.** One column per condition, one small dot per seed, the seed mean as
            the large mark and the seed range as a thin bar. Top: α at the example answers at slice
            {ex.lr.TRACK_SLICE}, averaged over epochs {ex.lr.WINDOW[0]} to {ex.lr.WINDOW[1]} (the loss-rate call);
            below the dotted line a run is called lost, above the dashed line kept. Bottom: the spill at epoch
            {STOP}, the largest drop on another op under the full edit less the median drop under eight reference
            edits; the dashed line is the criterion. The gray columns at the right are the loss-rate screen's
            epoch-{STOP} checkpoints of the two quiet recipes, at {len(QUIET_SPILL["falloff"])} seeds each.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.6, 4.6), layout="constrained", sharex=True)
        rng = np.random.default_rng(0)
        for i, n in enumerate(names):
            c = BY_NAME[n]
            _dots(ax, i, data["window"][n], ink(c), marker(c), rng)
            _dots(bx, i, data["spill"][n], ink(c), marker(c), rng)
        for j, v in enumerate(data["quiet"].values()):
            _dots(bx, len(names) + 0.5 + j, v, QUIET_INK, "o", rng)
        ax.axhline(data["lost_max"], color=RULE, lw=0.6, ls=":", zorder=1)
        ax.axhline(data["kept_min"], color=RULE, lw=0.6, ls="--", zorder=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("α, epochs 44–64", fontsize=8)
        bx.axhline(0, color=ZERO, lw=0.5, zorder=0)
        bx.axhline(data["criterion"], color=RULE, lw=0.7, ls="--", zorder=1)
        bx.set_ylabel(f"spill at epoch {STOP}", fontsize=8)
        ticks = list(range(len(names))) + [len(names) + 0.5 + j for j in range(len(data["quiet"]))]
        bx.set_xticks(
            ticks,
            [tex(SHOWN[n]) + (" (base)" if n == ex.STORED else "") for n in names] + list(data["quiet"]),
            rotation=60,
            ha="right",
            fontsize=6,
        )
        bx.set_xlim(-0.6, ticks[-1] + 0.6)
        for a_ in (ax, bx):
            a_.tick_params(labelsize=7)
        return fig

    return _plot()


def screen_table() -> str:
    rows = []
    for n in RANKED:
        q = per_seed(n, "alpha_q").mean(axis=0)[ex.lr.TRACK_SLICE]
        a = per_seed(n, "alpha_ans").mean(axis=0)
        rows.append(
            [
                f"`{SHOWN[n]}`" + (" (base)" if n == ex.STORED else ""),
                f"{8 - N_LOST[n]}/8",
                num(float(WINDOW[n].min()), ".2f"),
                num(float(SPILL[n].mean())),
                f"{num(float(SPILL[n].min()))} to {num(float(SPILL[n].max()))}",
                str(int((SPILL[n] > CRITERION).sum())),
                PEAK[n],
                num(float(a.max()), ".2f"),
                num(float(q), ".2f"),
            ]
        )
    return table_html(
        ["condition", "kept ↑", "lowest call", "spill ↓", "seed range", "over", "peak slice", "α at peak", "α, query"],
        rows,
        f"""
        **The conditions at epoch {STOP}, ranked.** Ranked by whether any run was not called kept, then by the mean
        spill. Kept: the runs the loss-rate call keeps (α at the example answers at slice {ex.lr.TRACK_SLICE},
        averaged over epochs {ex.lr.WINDOW[0]} to {ex.lr.WINDOW[1]}, at least {KEPT_MIN:g}). Lowest call: the
        lowest of those averages. Spill: the largest drop on another op under the full edit at epoch {STOP}, less
        the median drop under the reference edits, as a seed mean, with its seed range and the number of runs over
        the criterion ({CRITERION:g}). Peak slice and α at peak: where the seed-mean α at the example answers is
        highest at epoch {STOP}, and its value. α, query: the seed-mean α at the query answer at slice
        {ex.lr.TRACK_SLICE}.
        """,
    )


# --- E3: where the op is heading --------------------------------------------------------------------------------


def depth_figure(names: list[str]) -> str:
    data = {
        "names": names,
        "all": [per_seed(n, "alpha_ans").mean(axis=0).tolist() for n in NAMES],
        "seeds": {n: per_seed(n, "alpha_ans").tolist() for n in names},
        "query": {n: per_seed(n, "alpha_q").tolist() for n in names},
    }
    alt = """
        Four panels of α against slice, from the embedding to slice 4. In every panel the example answers rise from
        about zero at the embedding to a peak of about 0.55 at slice 3 and fall to about 0.37 at slice 4, and the
        query answer peaks near 0.85 at slice 3. At slice 1 the example answers sit near 0.22 for τ_ℓ 0.1, against
        about 0.3 to 0.37 in the other three panels.
    """
    return depth_draw(data, alt)


@memo
def depth_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-scout-depth",
        alt_text=alt_text,
        caption=f"""
            **Where the op sits at epoch {STOP}, by slice.** α at the example answers (solid, one line per seed) and
            at the query answer (dashed), at each slice, for the conditions at the top and the bottom of the ranking.
            The hairlines behind every panel are the seed means of every condition.
        """,
    )
    def _plot() -> plt.Figure:
        names = data["names"]
        fig, axes = plt.subplots(1, len(names), figsize=(6.6, 2.4), layout="constrained", sharey=True, squeeze=False)
        x = np.arange(len(SLICE_TICK))
        for ax, n in zip(axes[0], names, strict=True):
            c = BY_NAME[n]
            for g in data["all"]:
                ax.plot(x, g, color=GHOST, lw=0.4, zorder=0)
            for a, q in zip(data["seeds"][n], data["query"][n], strict=True):
                ax.plot(x, a, color=ink(c), lw=0.9, marker=marker(c), ms=2.5, zorder=2)
                ax.plot(x, q, color=ink(c), lw=0.6, ls="--", zorder=1)
            ax.set_title(tex(SHOWN[n]), fontsize=7)
            ax.set_xticks(x, SLICE_TICK, fontsize=7)
            ax.set_ylim(-0.05, 1)
            ax.tick_params(labelsize=7)
        axes[0][0].set_ylabel("α", fontsize=8)
        return fig

    return _plot()


def slice1_figure() -> str:
    data = {
        "runs": [
            {"name": n, "alpha1": float(RUNS[(n, s)]["alpha_ans"][1]), "spill": RUNS[(n, s)]["spill"]}
            for n in NAMES
            for s in SEEDS
        ],  # fmt: skip
        "criterion": CRITERION,
    }
    alt = """
        A scatter of 136 runs: α at the example answers at slice 1 on the horizontal axis, from about 0.15 to 0.45,
        and spill at epoch 64 on the vertical, from about −0.02 to 0.18. The τ_ℓ 0.1 runs (dark blue) sit at the
        left, between about 0.16 and 0.26, and all below the criterion. The runs over the criterion all sit right of
        about 0.26, most from the τ_ℓ 0.4 and λ_a conditions.
    """
    return slice1_draw(data, alt)


@memo
def slice1_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="soft-pool-scout-slice1",
        alt_text=alt_text,
        caption=f"""
            **Spill beside the anchor at slice 1.** One mark per run at epoch {STOP}: α at the example answers at
            slice 1 (horizontal), and the spill (vertical), in the ink and marker of its condition. The dashed line
            is the criterion.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.0, 3.0), layout="constrained")
        for r in data["runs"]:
            c = BY_NAME[r["name"]]
            ax.plot(r["alpha1"], r["spill"], marker(c), color=ink(c), ms=3.5, alpha=0.75, mew=0)
        ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
        ax.axhline(data["criterion"], color=RULE, lw=0.7, ls="--", zorder=0)
        ax.set_xlabel("α at the example answers, slice 1", fontsize=8)
        ax.set_ylabel(f"spill at epoch {STOP}", fontsize=8)
        ax.tick_params(labelsize=7)
        return fig

    return _plot()


# %%
rf"""
# Soft pool scout

/// tip |
<!-- lede -->
We screened sixteen schedules of the pooled pull at epoch {STOP}, looking for one that stops the early spill of `pool 1–3, soft τ`. A sharper pool over slices did it: with the slice temperature ending at 0.1, every run spilled about as little as the quiet recipes, whenever the temperatures sharpened. Moving the anchor warm-up did not help, and fixing the temperatures from the start spilled more.
///

[Recipe-compare](/docs/m2/recipe-compare/report.py) left one thing between `pool 1–3, soft τ` and the recipe of record: two of its eight runs go over the spill criterion, and the spill enters at block 1, the one slice the pool pulls and the step leaves alone. [Spill-timing](/docs/m2/spill-timing/report.py) found that both of those runs spill early, while the pool sharpens, and that this early kind already shows at epoch {STOP}. So a schedule that keeps block 1 closed should show it at epoch {STOP}, at a sixth of the cost of a full run.

This report trains sixteen conditions of the pool at its eight seeds ({SEEDS[0]} to {SEEDS[-1]}), each changing when the two pool temperatures sharpen, how sharp the pool over slices ends, or when the anchor weight warms up. Every run stops at epoch {STOP}, where we score whether it is keeping the anchor and how much the edit spills. This is stage 1; the leaders will be carried on to epoch 400 from where they stopped, in a second stage still to run.
"""

# %%

rf"""
## Observations

- [Lost anchors (E1)](#lost-anchors-e1): no schedule lost the anchor. All {N_RUNS} runs, the base included, are called kept, the lowest well above the line, though the fixed temperatures and the stretched warm-up keep it a little less firmly.
- [Spill at epoch {STOP} (E2)](#spill-at-epoch-{STOP}-e2): the slice temperature at the end decides the spill, and when the temperatures sharpen hardly matters. At τ_ℓ 0.1 no run is over the criterion, and the four conditions sit at or below the quiet recipes. At 0.4 (the base) about a quarter of the runs are over. The delayed and stretched warm-ups spill as much as the base, and fixing the temperatures from the start spills the most.
- [Where the op is heading (E3)](#where-the-op-is-heading-e3): every condition peaks at slice 3, and none leans on the query answer. What changes is slice 1: a sharper pool over slices puts less of the anchor there, and the runs that spill are the ones with more of it at slice 1.

## Scope and method

This is an exploratory study, with no preregistration and no gate. Every condition starts from `pool 1–3, soft τ` (the base): the pull is pooled over the slice and position pairs of slices 1 to 3, the position pool temperature $\tau_p$ sharpens from 1 to 0.1 and the slice pool temperature $\tau_\ell$ from 4 to 0.4 over the first forty epochs, and the anchor weight $\lambda_a$ warms up over the same forty epochs to its peak of 0.1. The anti weight is that of `falloff`, the recipe of record, and the rest of training is as in [soft-pool](/docs/m2/soft-pool/report.py).

Twelve conditions cross two factors. One is $\tau_\ell$ at the end, with levels 0.1, 0.2 and 0.4 (the base), each starting from 4. The other is when the temperatures are sharp: by epoch 10, 20 or 40 (the base), both on the same window, or $\tau_\ell$ at its end value from epoch 0 while $\tau_p$ still sharpens by 40, which tells the two temperatures apart. Five more conditions move the $\lambda_a$ warm-up at the base temperatures: a delay (zero until epoch 20, then up by 60) and a stretch (up by 80), each with the soft temperatures and with both temperatures fixed at their end values from the start, and the fixed temperatures with the base warm-up. The fixed-temperature conditions might have lost the anchor, which is what they were there to show: if a delayed pull keeps it at a fixed τ, the soft start was a delay by other means. The schedules are drawn below.

{schedules_figure()}

Each run trains on the full 400-epoch schedule and stops at epoch {STOP}, keeping what it needs to carry on (a resumed run matches a straight run bit for bit). The base is not trained again: [spill-timing](/docs/m2/spill-timing/report.py) trained it at these seeds with the same scoring at every record, so its epoch-{STOP} record is the base here. The epoch-{STOP} checkpoints of `falloff` and `step, soft τ` left by [the loss-rate screen](/docs/m2/loss-rate/report.py) are reference points on the same scale.

A screen at epoch {STOP} can rank the conditions by the early kind of spill and by lost anchors. It cannot clear a condition: the late kind of spill, which grows on `value-hsv` after epoch 100 on seeds that learn it late, is invisible here. Spill-timing found the base about 0.02 above the quiet recipes at epoch {STOP}, pooled over seeds, so the seed mean of eight runs can tell an early-spilling condition from a quiet one at about that resolution, and a single run says little. We compare many conditions, so a mean a little apart from its neighbors is a lead only. The pattern across a factor, which pools 32 runs at each level, is firmer.

### The measurements

- The call: α (the cosine of the state with e₁) at the example answers at slice {ex.lr.TRACK_SLICE}, averaged over epochs {ex.lr.WINDOW[0]} to {ex.lr.WINDOW[1]}. As in the loss-rate screen, a run is called lost below {LOST_MAX:g}, kept from {KEPT_MIN:g}, and uncertain between.
- Spill at epoch {STOP}: the largest drop on another op when e₁ is projected out at every position and slice, less the median drop under eight reference edits (each projecting out one other basis direction), over the whole held-out set, as in spill-timing. A drop is in expected exact match. Lower is better. The criterion is {CRITERION:g}, though at epoch {STOP} it is a guide only, since it is set for a finished run.
- Where the op sits: α at the example answers and at the query answer, by slice, at epoch {STOP}.
"""

# %%

rf"""
## Lost anchors (E1)

The fixed-temperature conditions were the ones most likely to lose the anchor, since a sharp pool from the first step is what let the pull settle on the query answer on one seed in slice-terms. Did any schedule lose it by epoch {STOP}?

None did. Every run is called kept, and the lowest call ({LOWEST_CALL:.2f}, on a stretched warm-up) is still above the line by a wide margin. The figure (top) shows the calls close together across all sixteen conditions. The fixed temperatures and the stretch sit a little lower than the rest, which fits a pull that has had less time at full strength by epoch {STOP}, but none comes near the uncertain band.

{screen_figure()}

So the soft start is not what keeps the anchor in this pool, at least to epoch {STOP} and at these seeds: the pull pooled over three slices holds the anchor at a fixed temperature too. Eight seeds would miss a loss rate of one in ten or so, and a run can still lose the anchor late, so the fixed temperatures are not cleared. But the soft start was added to the pool to keep the anchor, and at epoch {STOP} the pool keeps it without one.
"""

# %%

rf"""
## Spill at epoch {STOP} (E2)

If spill enters through block 1 while the pool is soft, then sharpening sooner, sharpening the slice pool further, or holding the pull back until e₁ is cleared might each close it. Which of them lowers the spill at epoch {STOP}?

The slice temperature at the end does, and the others barely move it. The figure (bottom) groups the crossing by $\tau_\ell$: at 0.1 the four timing levels all sit at or below the quiet recipes, with no run of 32 over the criterion; at 0.2 the spill is a little higher, with {LEVEL_OVER[0.2]} runs over; at 0.4, the base level, {LEVEL_OVER[0.4]} of 32 are over. Within each level, sharpening by epoch 10, 20 or 40, or holding $\tau_\ell$ fixed from the start, makes no difference we can see at eight seeds. So it is how sharp the slice pool ends, not how soon it gets there, that matters for the early spill.

{screen_table()}

The $\lambda_a$ conditions all stay well above the $\tau_\ell$ 0.1 level. A twenty-epoch delay, which gives e₁ time to clear under the full learning rate, lowered the mean spill somewhat, at the soft temperatures and at the fixed ones. The stretched warm-up raised it, and so did fixing both temperatures with the base warm-up. Each of these means rests on one or two seeds that spill far more than the rest, so at eight seeds the differences among them could be chance. None of them comes near what sharpening the slice pool does.

Of the {N_OVER} runs over the criterion, {OVER_OPS[TOP_OP]} spill most on `{TOP_OP}`, the op on which the base spills early in spill-timing.

These are early numbers. A sharper slice pool could still spill late, on `value-hsv` as the base does on its slow seeds, and only a finished run can show that. What we take from the screen is that $\tau_\ell$ is the lever on the early kind, and that the level at 0.1 brings the pool to the quiet recipes on it.
"""

# %%

rf"""
## Where the op is heading (E3)

A sharper slice pool could lower the spill in a way we would not want: by settling the pull on one slice and one role, as the joint pool did in [slice-terms](/docs/m2/slice-terms/report.py), where one seed settled on the query answer and the op could not be removed. Where does the evidence for the op sit at epoch {STOP}?

In the same place in every condition. The seed-mean α at the example answers peaks at slice 3 in all seventeen conditions, and α at the query answer is about the same everywhere (the table above). The figure shows the τ_ℓ ladder at the base timing, beside the condition that spilled most: the shapes are nearly the same, with one difference at slice 1.

{depth_figure(["lt0.1-by40", "lt0.2-by40", ex.STORED, "fixed-stretch"])}

That difference is the one that tracks the spill. At $\tau_\ell$ 0.1, α at the example answers at slice 1 averages {LEVEL_ALPHA1[0.1]:.2f}, against {LEVEL_ALPHA1[0.4]:.2f} at 0.4. Across all {N_RUNS} runs, the runs over the criterion are the ones with more of the anchor at slice 1, though at the same α some spill and some do not.

{slice1_figure()}

This fits the account recipe-compare gave: the pool spills when it puts part of the pull on block 1, and the edit then takes something from the other ops there. A sharper slice pool takes more of the pull off slice 1 and puts it on slice 3, so block 1 holds less of the anchor and the edit has less to take. It fits without proving it; the screen shows the two moving together, and a restricted edit on the finished runs (block 1 alone) would test it.
"""

# %%

r"""
## Discussion

TODO after stage 2.

/// admonition | Open decision
Which conditions go to stage 2. The plan finishes two or three leaders, ranked by no lost anchors and the lowest mean spill, preferring the fewest changes from the base when two are close. The four conditions at $\tau_\ell$ 0.1 are close together. A proposal: `τ_ℓ 0.1, by 40` (one change from the base) and `τ_ℓ 0.1, τ_ℓ fixed` (the simplest schedule, with only $\tau_p$ moving), and `τ_ℓ 0.2, by 40` as a middle step that would show whether 0.1 is needed or 0.2 is enough once the late kind is counted. To check: whether the finished runs keep their removal, since a sharper slice pool concentrates the pull.
///
"""
