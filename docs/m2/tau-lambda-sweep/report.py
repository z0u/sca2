# ruff: noqa: B018
# title: The pool temperature and anchor weight, without the embedding

# The design comes from `experiment.py` beside this script. The results come from the JSON it publishes, and the
# controls and the every-slice runs at the recipe point from ex-2.2.23's.
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

ex23 = ex.ex2223

r"""
# The pool temperature and anchor weight, without the embedding

/// tip |
<!-- lede -->
We swept the pool temperature τ and the anchor weight λ_a together, with the in-context pull kept off the embedding slice. Leaving the embedding out keeps the latch off the syntax embeddings over most of the plane, but at the recipe point the edit still spills about as much as on the every-slice runs without a latch, and the spill grows as the pool softens. The most selective trials tend to sit near a narrow edge, just soft enough for the edit to remove the op at all.
///

[Spill-by-position](/docs/m2/spill-by-position/report.py) found that most anchored runs of the recipe latch one syntax embedding onto e₁, and that a latched separator carries much of the spill of the edit. The `no-emb` pull of [ex-2.2.21](/docs/m2/ex-2.2.21/report.py), which leaves the embedding slice out of the anchor and anti-subspace terms, never latched there and scored best on the task, but only at three seeds and 200 epochs. The [backlog item](/todo/science/softer-tau-for-the-in-context-pull.md) on a softer τ asked a related question: whether spreading the pull over more positions would move the anchor where it is needed.

This page trains `no-emb` at the recipe length over a log-log plane of τ and λ_a, one seed per trial, and fits smooth surfaces over the plane in place of repeated seeds. A few every-slice trials along τ are the reference.
"""

# --- The stored results ---------------------------------------------------------------------------------------


def fetch_json(refs: list[str]) -> list[Any]:
    store = project_store()
    found = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    assert len(found) == len(refs), f"not published: {sorted(set(refs) - set(found))}"
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(found[r], Path(tmp) / f"{i}.json") for i, r in enumerate(refs)])
        return [json.loads(p.read_text()) for p in paths]


EVAL, SUPP, TRAJ, EVAL23, SUPP23 = fetch_json(
    [ex.EVAL_REF, ex.SUPPRESSION_REF, ex.TRAJ_REF, ex23.EVAL_REF, ex23.SUPPRESSION_REF]
)
OPS: tuple[str, ...] = tuple(EVAL["runs"][0]["ops"])
assert OPS == tuple(ex.OP_NAMES)
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
HSV = [OPS.index(o) for o in ex.HSV_OPS]
K = ex.K
UNIT = 6
# Tokens per example: operand, `?`, operand, `=`, answer, separator.
EQ_ROLES = [UNIT * i + 3 for i in range(K + 1)]
# Every `=`: the examples' and the query's, the positions whose next token depends on the op.
ANSWER_ROLES = [UNIT * i + 4 for i in range(K)]
LATE = slice(2, None)
# The slices α is averaged over, as in the example-evidence report: from the output of the second block on.
RISE_MEASURE, RISE_LEVEL = ex23.RULE
assert RISE_MEASURE == "hsv_min"
LATCH_LEVEL = 0.9
# A syntax embedding is latched when its alignment with e₁ is above this, as in spill-by-position. Each one sits near 0
# or near 1, so any level between them picks out the same runs.
REMOVAL_LEVEL = ex.ex2221.GRADING_MIN_DAMAGE
SPILL_GATE = ex.ex2221.SELECTIVITY_GATE


def drops(r: dict) -> np.ndarray:
    """The drop in task score per dose and op under the edit: `(doses, ops)`."""
    return np.array(r["clean"]["eem"])[None, :] - np.array([e["eem"] for e in r["edits"]])


CONTROLS = [r for r in SUPP23["runs"] if r["condition"] == "control" and r["epochs"] == ex.EPOCHS]
CONTROL_DROP = np.mean([drops(r) for r in CONTROLS], axis=0)
CONTROL_EEM = float(
    np.mean(
        [r["task"]["eem"]["all"] for r in EVAL23["runs"] if r["condition"] == "control" and r["epochs"] == ex.EPOCHS]
    )
)
assert len(CONTROLS) == len(ex23.SEEDS)


def measure(ev: dict, sp: dict) -> dict[str, Any]:
    """Every measurement of one run, from its eval and suppression records. The edit measurements are net of the
    seed mean of ex-2.2.23's controls at 400 epochs.
    """
    per_op = np.array(ev["task"]["eem"]["per_op"], float)
    net = drops(sp) - CONTROL_DROP
    gap = sp["clean"]["eem"][D] - sp["null"]["eem"][D]
    syntax = ev["syntax_embeddings"]
    top = max(syntax, key=lambda t: syntax[t])
    align = np.array(ev["alignment"], float)[D]  # (slices, positions), on `difference` contexts
    return {
        "eem": float(ev["task"]["eem"]["all"]),
        "cost": CONTROL_EEM - float(ev["task"]["eem"]["all"]),
        "hsv_min": float(per_op[HSV].min()),
        "rose": bool(per_op[HSV].min() >= RISE_LEVEL),
        "syntax_max": float(syntax[top]),
        "syntax_top": top,
        "latched": top if syntax[top] >= LATCH_LEVEL else None,
        "removal": float(net[-1, D] / gap),
        "spill": float(net[:, OTHER].max()),
        "onto": OPS[OTHER[int(net[:, OTHER].max(axis=0).argmax())]],
        "alpha_answers": float(align[LATE][:, ANSWER_ROLES].mean()),
        "alpha_eq": float(align[LATE][:, EQ_ROLES].mean()),
        "alpha_query_eq": float(align[LATE][:, EQ_ROLES[-1]].mean()),
        "margin": float(ev["margin"]["value"]),
    }


def build() -> list[dict[str, Any]]:
    supp = {r["label"]: r for r in SUPP["runs"]}
    out = [
        {k: ev[k] for k in ("name", "arm", "tau", "lam", "model_seed")}
        | {"source": "sweep"}
        | measure(ev, supp[ev["label"]])
        for ev in EVAL["runs"]
    ]
    supp23 = {r["label"]: r for r in SUPP23["runs"]}
    for ev in EVAL23["runs"]:
        if ev["condition"] == "anchor" and ev["epochs"] == ex.EPOCHS:
            info = {"name": ev["label"], "arm": "whole", "tau": ex.TAU_DEFAULT, "lam": ex.LAMBDA_DEFAULT}
            out.append(
                info | {"model_seed": ev["model_seed"], "source": "ex-2.2.23"} | measure(ev, supp23[ev["label"]])
            )
    return out


RUNS = build()
SWEEP = [r for r in RUNS if r["arm"] == "no-emb"]
REFERENCE = [r for r in RUNS if r["arm"] == "whole"]
AT_RECIPE = [r for r in REFERENCE if r["source"] == "ex-2.2.23"]
assert len(SWEEP) == 2**ex.SOBOL_M and len(AT_RECIPE) == len(ex23.SEEDS)

# --- Fits over the plane --------------------------------------------------------------------------------------

LOG_LO = np.log10([ex.TAU_RANGE[0], ex.LAMBDA_RANGE[0]])
LOG_HI = np.log10([ex.TAU_RANGE[1], ex.LAMBDA_RANGE[1]])


def unit(tau, lam) -> np.ndarray:
    """(τ, λ_a) mapped onto the unit square of the sweep, on log scales: `(n, 2)`."""
    x = np.log10(np.column_stack([np.ravel(tau), np.ravel(lam)]))
    return (x - LOG_LO) / (LOG_HI - LOG_LO)


def gp_fit(runs: list[dict], key: str):
    """A Gaussian process over the unit square for one continuous measurement: an anisotropic squared-exponential
    kernel plus white noise, so the fit finds its own smoothness along each axis and its own seed noise.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    x = unit([r["tau"] for r in runs], [r["lam"] for r in runs])
    y = np.array([r[key] for r in runs], float)
    # A length scale at its bound only says the fit wants a sharper edge than the bound allows.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    kernel = ConstantKernel(1.0, (1e-3, 1e2)) * RBF([0.5, 0.5], (0.1, 10.0)) + WhiteKernel(0.1, (1e-4, 1.0))
    return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=8, random_state=0).fit(x, y)


EQ_LEVEL = 0.5
# A `no-emb` trial has its anchor on `=` when the `=` embedding has an alignment with e₁ above this; the trials sit
# either below 0.2 or above 0.7.
EQ_RUNS = [r for r in SWEEP if r["syntax_top"] == "=" and r["syntax_max"] >= EQ_LEVEL]
CONTINUOUS = ("removal", "spill", "cost", "alpha_answers", "alpha_eq")
FITS = {k: gp_fit([r for r in SWEEP if r not in EQ_RUNS], k) for k in CONTINUOUS}
# The trials with the anchor on `=` are left out of the fits: they are a different solution, and a smooth surface
# through both would describe neither (E2).

# --- Numbers for the prose ------------------------------------------------------------------------------------

SHARP = 0.025
EDGE = 0.07
# The three stretches of τ that E1 describes, set by eye from the figure: below SHARP the pool is close to a max; from
# EDGE up, every `no-emb` trial removes the op.
NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()


def num_word(n: int) -> str:
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)


def where(lo: float = 0.0, hi: float = np.inf, runs: list[dict] = SWEEP) -> list[dict]:
    return [r for r in runs if lo <= r["tau"] < hi]


SHARP_RUNS = where(hi=SHARP)
assert all(r in SHARP_RUNS for r in EQ_RUNS)
SOFT_RUNS = where(lo=EDGE)
assert all(r["removal"] >= REMOVAL_LEVEL for r in SOFT_RUNS)
MIDDLE_RUNS = where(lo=SHARP, hi=EDGE)
REMOVING = [r for r in SWEEP if r["removal"] >= REMOVAL_LEVEL and r not in EQ_RUNS]
SPILL_R = float(np.corrcoef([r["alpha_eq"] for r in REMOVING], [r["spill"] for r in REMOVING])[0, 1])
NEAR_RECIPE = where(lo=0.08, hi=0.2)
RECIPE_LATCHED = [r for r in AT_RECIPE if r["latched"]]
RECIPE_CLEAN = [r for r in AT_RECIPE if not r["latched"]]
RECIPE_SPILL_SD = float(np.std([r["spill"] for r in AT_RECIPE], ddof=1))
COST_MAX = max(r["cost"] for r in SWEEP)
LATCHED_REFERENCE = [r for r in REFERENCE if r["source"] == "sweep" and r["latched"]]


def propose() -> dict[str, float]:
    """The point of the plane with the lowest fitted spill among those whose fitted removal is at least
    `REMOVAL_TARGET`, on a grid over the sweep, with the fits and their standard deviations there.
    """
    g = np.linspace(0, 1, 81)
    u = np.column_stack([a.ravel() for a in np.meshgrid(g, g)])
    rem, rem_sd = FITS["removal"].predict(u, return_std=True)
    spill, spill_sd = FITS["spill"].predict(u, return_std=True)
    i = int(np.argmin(np.where(rem >= REMOVAL_TARGET, spill, np.inf)))
    tau, lam = 10 ** (LOG_LO + u[i] * (LOG_HI - LOG_LO))
    return {
        "tau": float(tau),
        "lam": float(lam),
        "removal": float(rem[i]),
        "removal_sd": float(rem_sd[i]),
        "spill": float(spill[i]),
        "spill_sd": float(spill_sd[i]),
    }


REMOVAL_TARGET = 0.7
# The fitted removal the proposal asks for: well past the edge, where the fit is still rising, and below the plateau
# near 0.9, where the spill is high.
PROPOSED = propose()
BEST = sorted((r for r in SWEEP if r["removal"] >= REMOVAL_TARGET), key=lambda r: r["spill"])[:3]
COSTLY = 0.02
# A task cost above this is well outside the spread of the recipe runs (seed standard deviation about 0.01).

# --- Drawing ----------------------------------------------------------------------------------------------------

LAM_LEVELS = (0.04, 0.1, 0.25)
# The λ_a at which the fits are drawn along τ: low, the recipe weight, and high.
TAU_GRID = np.geomspace(*ex.TAU_RANGE, 121)


def lam_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("lam", light_dark(["#a9cbe8", "#08306b"], ["#24486e", "#c6e3ff"]), N=256)


def lam_pos(lam: float) -> float:
    lo, hi = np.log10(ex.LAMBDA_RANGE)
    return float((np.log10(lam) - lo) / (hi - lo))


def ref_ink() -> str:
    return light_dark("#c2571a", "#f0a060")


def rule_color() -> str:
    return light_dark("#888", "#777")


def along_tau_data(keys: tuple[str, ...]) -> dict:
    """What `along_tau_draw` plots for each measurement: the trials, the reference trials, the runs at the recipe
    point, and the fits at `LAM_LEVELS` (when the measurement has one).
    """
    out = {}
    for key in keys:
        fits = {}
        if key in FITS:
            for lam in LAM_LEVELS:
                m, sd = FITS[key].predict(unit(TAU_GRID, np.full_like(TAU_GRID, lam)), return_std=True)
                fits[lam] = {"mean": m.tolist(), "sd": sd.tolist()}
        out[key] = {
            "sweep": [(r["tau"], r["lam"], r[key], r in EQ_RUNS) for r in SWEEP],
            "reference": sorted((r["tau"], r[key]) for r in REFERENCE if r["source"] == "sweep"),
            "recipe": [r[key] for r in AT_RECIPE],
            "fits": fits,
        }
    return out


LABELS = {
    "removal": f"removal of {ex.ANCHORED_OP}",
    "spill": "spill onto another op",
    "cost": "task cost",
    "alpha_answers": "α at the example answers",
    "alpha_eq": "α at every =",
    "syntax_max": "largest syntax embedding on e₁",
}


@memo
def along_tau_draw(data: dict, name: str, caption: str, alt_text: str) -> str:
    @themed(name=name, alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        n = len(data)
        fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.0), layout="constrained", sharex=True, squeeze=False)
        cmap = lam_cmap()
        rng = np.random.default_rng(0)
        for ax, (key, d) in zip(axes[0], data.items(), strict=True):
            ax.axhline(0, color=rule_color(), lw=0.5, zorder=0)
            for lam, f in d["fits"].items():
                m, sd = np.array(f["mean"]), np.array(f["sd"])
                c = cmap(lam_pos(lam))
                ax.fill_between(TAU_GRID, m - sd, m + sd, color=c, alpha=0.1, lw=0, zorder=1)
                ax.plot(TAU_GRID, m, "-", color=c, lw=1.2, zorder=2, label=f"fit at λₐ = {lam:g}")
            for i, (tau, lam, v, on_eq) in enumerate(d["sweep"]):
                c = cmap(lam_pos(lam))
                ax.plot(tau, v, "o", ms=5, color=c, mfc="none" if on_eq else c,
                        mec=c if on_eq else light_dark("white", "#111"), mew=1.0 if on_eq else 0.5, zorder=4,
                        label=None if i else "no-emb trial")  # fmt: skip
            rt, rv = np.array(d["reference"]).T
            ax.plot(rt, rv, "-", color=ref_ink(), lw=0.7, alpha=0.6, zorder=3)
            ax.plot(rt, rv, "^", ms=5, color=ref_ink(), mec=light_dark("white", "#111"), mew=0.5, zorder=4,
                    label="every-slice trial")  # fmt: skip
            rec = np.array(d["recipe"])
            x0 = ex.TAU_DEFAULT
            jit = x0 * 10 ** rng.uniform(-0.03, 0.03, len(rec))
            ax.plot([x0, x0], [rec.min(), rec.max()], "-", color=ref_ink(), lw=1.0, alpha=0.5, zorder=3)
            ax.plot(jit, rec, "o", ms=2.2, color=ref_ink(), alpha=0.45, mew=0, zorder=3)
            ax.plot(x0, rec.mean(), "D", ms=5, color=ref_ink(), mec=light_dark("white", "#111"), mew=0.6, zorder=5,
                    label="every-slice, recipe point (12 seeds)")  # fmt: skip
            ax.set_xscale("log")
            ax.set_xlabel("pool temperature τ", fontsize=8)
            ax.set_ylabel(LABELS[key], fontsize=8)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False, fontsize=7)
        return fig

    return _plot()


def plane_data() -> dict:
    g = np.linspace(0, 1, 81)
    u = np.column_stack([a.ravel() for a in np.meshgrid(g, g)])
    t, lam = 10 ** (LOG_LO[:, None] + u.T * (LOG_HI - LOG_LO)[:, None])
    return {
        "tau": t.reshape(81, 81).tolist(),
        "lam": lam.reshape(81, 81).tolist(),
        "removal": FITS["removal"].predict(u).reshape(81, 81).tolist(),
        "spill": FITS["spill"].predict(u).reshape(81, 81).tolist(),
        "sweep": [(r["tau"], r["lam"], r in EQ_RUNS) for r in SWEEP],
        "reference": [(r["tau"], r["lam"]) for r in REFERENCE if r["source"] == "sweep"],
        "proposed": (PROPOSED["tau"], PROPOSED["lam"]),
    }


@memo
def plane_draw(data: dict, spill_levels: tuple[float, ...], caption: str, alt_text: str) -> str:
    @themed(name="tau-lambda-plane", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.4, 3.6), layout="constrained")
        t, lam = np.array(data["tau"]), np.array(data["lam"])
        ink = light_dark("#222", "#ddd")
        cr = ax.contour(t, lam, np.array(data["removal"]), levels=[REMOVAL_LEVEL, REMOVAL_TARGET], colors=[ink],
                        linewidths=[0.8, 1.3], linestyles=["-"])  # fmt: skip
        ax.clabel(cr, fmt={REMOVAL_LEVEL: f"removal {REMOVAL_LEVEL:g}", REMOVAL_TARGET: f"{REMOVAL_TARGET:g}"},
                  fontsize=6)  # fmt: skip
        cs = ax.contour(t, lam, np.array(data["spill"]), levels=list(spill_levels), colors=[ref_ink()],
                        linewidths=1.0, linestyles=["--"])  # fmt: skip
        ax.clabel(cs, fmt={v: f"spill {v:g}" for v in spill_levels}, fontsize=6)
        for tau, lm, on_eq in data["sweep"]:
            ax.plot(tau, lm, "o", ms=4.5, color=ink, mfc="none" if on_eq else ink, mew=0.9, zorder=4)
        rt, rl = np.array(data["reference"]).T
        ax.plot(rt, rl, "^", ms=4.5, color=ref_ink(), mec=light_dark("white", "#111"), mew=0.5, zorder=4)
        ax.plot(ex.TAU_DEFAULT, ex.LAMBDA_DEFAULT, "D", ms=5.5, color=ref_ink(), mec=light_dark("white", "#111"),
                mew=0.6, zorder=5)  # fmt: skip
        ax.plot(*data["proposed"], "*", ms=12, color=light_dark("#1a7f37", "#5fd47f"), mec=light_dark("white", "#111"),
                mew=0.6, zorder=6)  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*ex.TAU_RANGE)
        ax.set_ylim(*ex.LAMBDA_RANGE)
        ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1], ["0.01", "0.03", "0.1", "0.3", "1"])
        ax.set_yticks([0.025, 0.05, 0.1, 0.2, 0.4], ["0.025", "0.05", "0.1", "0.2", "0.4"])
        ax.minorticks_off()
        ax.set_xlabel("pool temperature τ", fontsize=8)
        ax.set_ylabel("anchor weight λₐ", fontsize=8)
        return fig

    return _plot()


@memo
def spill_alpha_draw(rows: list, caption: str, alt_text: str) -> str:
    @themed(name="tau-lambda-spill-alpha", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
        cmap = lam_cmap()
        ax.axhline(0, color=rule_color(), lw=0.5, zorder=0)
        for a, s, lam, on_eq, arm in rows:
            if arm == "no-emb":
                c = cmap(lam_pos(lam))
                ax.plot(a, s, "o", ms=5, color=c, mfc="none" if on_eq else c, mec=c if on_eq else
                        light_dark("white", "#111"), mew=1.0 if on_eq else 0.5, zorder=3)  # fmt: skip
            else:
                ax.plot(a, s, "^", ms=5, color=ref_ink(), mec=light_dark("white", "#111"), mew=0.5, zorder=3)
        ax.set_xlabel(LABELS["alpha_eq"], fontsize=8)
        ax.set_ylabel(LABELS["spill"], fontsize=8)
        return fig

    return _plot()


# --- Tables -----------------------------------------------------------------------------------------------------


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i >= text_cols else ''}>{c}</td>" for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def trial_row(r: dict, name: str | None = None) -> list[str]:
    return [
        name or r["name"],
        f"{r['tau']:.3f}",
        f"{r['lam']:.3f}",
        f"{r['cost']:+.3f}",
        f"{r['hsv_min']:.2f}",
        f"<code>{r['syntax_top']}</code> {r['syntax_max']:.2f}",
        f"{r['alpha_answers']:.2f}",
        f"{r['alpha_eq']:.2f}",
        f"{r['removal']:.2f}",
        f"{r['spill']:.2f} <code>{r['onto']}</code>",
    ]


def trials_table() -> str:
    head = ["trial", "τ", "λₐ", "task cost ↓", "worst HSV op ↑", "top syntax embedding", "α, answers", "α, every =",
            "removal ↑", "spill ↓"]  # fmt: skip
    rows = [trial_row(r) for r in sorted(SWEEP, key=lambda r: r["tau"])]
    rows += [trial_row(r) for r in sorted((r for r in REFERENCE if r["source"] == "sweep"), key=lambda r: r["tau"])]
    rows += [trial_row(r, f"recipe, seed {r['model_seed']}") for r in AT_RECIPE]
    return table_html(
        head,
        rows,
        f"""
        **Every trial.** The {len(SWEEP)} `no-emb` trials in order of τ, then the {len(REFERENCE) - len(AT_RECIPE)}
        every-slice trials, then ex-2.2.23's every-slice runs at the recipe point (τ = λₐ = 0.1). Task cost is the
        seed mean of the controls minus the trial score; the worst HSV op is its score on the held-out set; the top
        syntax embedding is the one most aligned with e₁; α is averaged over slices 2 to {ex.ex2216.N_LAYER} on
        `{ex.ANCHORED_OP}` contexts. Removal and spill are net of the control seed mean, and the spill names the op
        it lands on.
        """,
        text_cols=1,
    )


rf"""
## Observations

- [Removal along τ (E1)](#removal-along-e1): below τ ≈ {EDGE:g} the edit barely removes `{ex.ANCHORED_OP}` on most trials, and from there up it removes most of the way on every trial. The weight makes little difference, and removal hardly changes past the edge.
- [Where the anchor sits (E2)](#where-the-anchor-sits-e2): past the edge the anchor sits at the example answers. As the pool softens it spreads onto every `=`, and the syntax embeddings come partway onto e₁, with no full latch. At the sharpest pool, {num_word(len(EQ_RUNS))} of the {num_word(len(SHARP_RUNS))} trials put the `=` embedding most of the way onto e₁, a latch in a new place.
- [Spill (E3)](#spill-e3): spill grows as the pool softens, together with the anchor at `=`. At the recipe τ, `no-emb` spills about as much as the every-slice runs that did not latch. The least spill, among trials that remove the op, is on the edge itself; the fits put the best point near τ = {PROPOSED["tau"]:.2f}, λ_a = {PROPOSED["lam"]:.2f}, though not with much confidence.

## Scope

This is an exploratory study, with no preregistration and no gate. It trains {len(SWEEP)} `no-emb` trials and {len(REFERENCE) - len(AT_RECIPE)} every-slice trials, each at its own model seed ({ex.SEED_OFFSET} to {ex.SEED_OFFSET + len(ex.TRIALS) - 1}) and for {ex.EPOCHS} epochs, on the recipe of record: ex-2.2.21's corpus with {K} examples per context, the whole-line label, no cap. The `no-emb` trials are scattered over τ from {ex.TAU_RANGE[0]:g} to {ex.TAU_RANGE[1]:g} and λ_a from {ex.LAMBDA_RANGE[0]:g} to {ex.LAMBDA_RANGE[1]:g}, both on log scales, by a scrambled Sobol sequence (a quasi-random pattern that covers a square more evenly than random draws). The every-slice trials are spaced evenly along τ, on a log scale, at the recipe weight of {ex.LAMBDA_DEFAULT:g}. Beside them are ex-2.2.23's {len(AT_RECIPE)} every-slice runs at 400 epochs, all at the recipe point (τ = λ_a = 0.1), which show how far runs differ by seed alone at one point.

The recipe sets the weight of the anti-subspace term as a multiple of λ_a, so the two terms move together and the ratio between them is the same on every trial. The sweep says nothing about the anti-subspace term relative to the pull.

With one seed per trial, how far a difference can be trusted comes from the fits (see below), which also estimate how much the trials scatter about the surface. At the recipe point, removal is nearly the same from seed to seed, and spill is not: its seed standard deviation is {RECIPE_SPILL_SD:.2f}, as large as most of the differences in E3. So a single trial says little about spill, and a pattern has to show across several neighboring trials before we read anything into it.

## The measurements

Each measurement is taken on ex-2.2.21's held-out set, as in ex-2.2.23. The edit is ex-2.2.22's: at every position and every slice, it removes a share (the dose) of the component of the state along e₁, the anchored direction.

- *Removal*: how far the drop on `{ex.ANCHORED_OP}` at full dose gets toward the target null, as a share of the way. Higher is better.
- *Spill*: the largest drop on any other op, at any dose. Lower is better; ex-2.2.21 set a selectivity criterion of {SPILL_GATE:g}, which no trial here meets.
- *α*: the alignment with e₁ (the cosine between a state and the anchored direction), on `{ex.ANCHORED_OP}` contexts, averaged over slices 2 to {ex.ex2216.N_LAYER}: at the example answers, and at every `=` (the examples' and the query).
- *Syntax embeddings*: the largest alignment with e₁ of any syntax token embedding (`?`, `=`, `,`, ⏎).
- *Task cost*: the seed mean of the controls minus the trial score.

Both edit measurements are net of the seed mean of ex-2.2.23's {len(CONTROLS)} controls at 400 epochs, so what the edit does to a model with no anchor is not counted. Ex-2.2.23 netted against the control at the same seed; here the trials have new seeds, and on ex-2.2.23's runs pairing by seed made little difference.

Each continuous measurement gets a Gaussian process over the plane, fit in log τ and log λ_a on the `no-emb` trials, leaving out the {num_word(len(EQ_RUNS))} whose anchor sits on `=` (E2). The design also planned logistic fits for two yes/no outcomes, but neither varied: every trial made the second rise, the late jump in task skill when the model learns the three HSV ops (its worst HSV op ends above {RISE_LEVEL:g}), and no `no-emb` trial latched a syntax embedding at spill-by-position's level of {LATCH_LEVEL:g}, though the `=` came close on four (E2). Task cost is small everywhere, at most {COST_MAX:.2f}, with no pattern over the plane; it is in the table at the end, and in E3 where it bears on the trials with the least spill.

## Glossary

Pool temperature
τ
:   The temperature of the mellowmax, a smooth stand-in for the maximum, that pools the alignment over the positions of a context: small τ is close to the largest alignment, large τ close to the mean. The recipe uses 0.1.

Target null
:   What the model would answer without `{ex.ANCHORED_OP}`: the ideal predictor with that op removed from the posterior over ops.

Latch
:   A pooled pull that settles on one cheap position that every context shares, a different one from run to run: here, a syntax token whose embedding comes to lie on e₁ ([spill-by-position](/docs/m2/spill-by-position/report.py), E3).

Gaussian process
:   A smoother that fits a surface through scattered points and reports its own uncertainty at every point of the surface. Here it fits each measurement over log τ and log λ_a, with a separate term for how much single trials scatter about the surface.
"""

# %%


def removal_figure() -> str:
    alt = f"""
        Two panels against the pool temperature τ on a log axis, from {ex.TAU_RANGE[0]:g} to {ex.TAU_RANGE[1]:g}.
        Left, removal: the no-emb trials, shaded by anchor weight, sit low or scattered below about {EDGE:g} and near
        0.9 above it; the fitted curves at the three weights lie on top of each other and rise steeply around {EDGE:g}. Four hollow
        trials at the sharpest pool sit near 0.9. The
        every-slice trials, triangles, follow the same rise, collapsing below about 0.04. The twelve every-slice
        runs at the recipe point cluster near 0.85. Right, α at the example answers: low below the edge and near 0.7
        above it, for both pulls.
    """
    return along_tau_draw(
        along_tau_data(("removal", "alpha_answers")),
        "tau-lambda-removal",
        f"""
        **Removal and the anchor at the example answers, along τ.** Circles are `no-emb` trials, shaded from light
        (λ_a = {ex.LAMBDA_RANGE[0]:g}) to dark (λ_a = {ex.LAMBDA_RANGE[1]:g}), hollow where the anchor sits on `=`
        (E2). Lines are the fits at three weights, with bands of one standard deviation; the fits leave out the
        hollow trials, and where the lines coincide, the fit found no effect of the weight. Triangles are the every-slice trials at λ_a = {ex.LAMBDA_DEFAULT:g};
        the diamond, its whisker, and the faint dots are ex-2.2.23's every-slice runs at the recipe point.
        """,
        alt,
    )


rf"""
## Removal along τ (E1)

Where on the plane does the edit remove `{ex.ANCHORED_OP}`? The figure plots removal against τ, with the weight as shade, and the anchor at the example answers beside it, since that is where [the example-evidence report](/docs/m2/example-evidence/report.py) found it.

{removal_figure()}

Removal depends mostly on τ, through a sharp edge near τ = {EDGE:g} (placed by eye). The fit rises again at the sharpest pool, but that rests on one trial. Above it, every `no-emb` trial removes the op most of the way, about as far as the every-slice runs at the recipe point, and softening the pool further changes little. Below it, most trials remove little. The every-slice trials follow the same edge, a little sharper.

The weight makes little difference. The fit finds no change with λ_a over the sixteenfold range of the sweep, and the shades mix on both sides of the edge. Just below it, the two trials that remove the op best are among the heavier ones, which is a hint at most.

The anchor at the example answers follows the same edge: below it, α there is low, and above it, α settles near 0.7 on every trial. So with a sharp pool, the pull is met somewhere the answer does not depend on, and the edit has nothing to remove. The trials at the sharpest pool that do remove the op are the exception, and E2 shows where their anchor went.
"""

# %%


def anchor_figure() -> str:
    alt = """
        Two panels against τ on a log axis. Left, α at every `=`: near zero up to about 0.2 for no-emb trials, then
        rising to about 0.5 at τ = 1; four trials at the sharpest pool sit at 0.25 to 0.66. Every-slice trials stay
        near zero until about 0.3, then rise to 0.4. Right, the largest syntax embedding on e₁: no-emb trials near
        0.1 until about 0.2, then rising to between 0.3 and 0.75, with four trials at the sharpest pool between 0.74
        and 0.9. Every-slice trials sit at 1 at several τ values, as do five of the twelve recipe runs.
    """
    return along_tau_draw(
        along_tau_data(("alpha_eq", "syntax_max")),
        "tau-lambda-anchor",
        """
        **The anchor at every `=` and on the syntax embeddings, along τ.** As in the figure of E1. The right panel
        has no fit: the syntax embedding with the largest alignment differs from trial to trial.
        """,
        alt,
    )


rf"""
## Where the anchor sits (E2)

Does the pool spread the anchor beyond the example answers as it softens, and does leaving out the embedding keep the syntax embeddings off e₁? The figure plots α at every `=` and the largest alignment of a syntax embedding.

{anchor_figure()}

At the sharpest pool, the anchor of {num_word(len(EQ_RUNS))} trials moves to `=`. On those trials the `=` embedding lies most of the way along e₁, even though the embedding slice is not pulled, and α at every `=` is high on `{ex.ANCHORED_OP}` contexts, and on the contexts of other ops too. The pull at the output of the first block is cheapest to meet by moving a token that every context shares, so leaving out the embedding slice moves the latch one slice up rather than removing it. On those trials the edit removes the op by removing `=` everywhere (E3 shows what that costs). The other {num_word(len(SHARP_RUNS) - len(EQ_RUNS))} trials at the sharpest pool have no anchor in use.

From the edge up to τ ≈ 0.2 the anchor stays at the example answers, and the syntax embeddings stay near zero, as in ex-2.2.21. As the pool softens beyond that, α rises at every `=`, and the most aligned syntax embedding comes partway onto e₁, usually the line break or the separator. None reaches a full latch, and the weight makes little difference.

The every-slice pull latches at half the values of τ, much as it did at the recipe point: {num_word(len(LATCHED_REFERENCE))} of its {len(REFERENCE) - len(AT_RECIPE)} trials have one syntax embedding on e₁, and at the softest pool its syntax embeddings rise partway too.
"""

# %%


def spill_figure() -> str:
    alt = f"""
        Spill against τ on a log axis. No-emb trials at the sharpest pool split between about 0.6 (the four with
        the anchor on `=`) and below 0.1; between {SHARP:g} and {EDGE:g} they sit near zero; above the edge they
        rise from about 0.05 to between 0.2 and 0.6, with the fits climbing from about 0.1 to 0.4. The every-slice
        trials and recipe runs sit between 0.1 and 0.5 above the edge.
    """
    return along_tau_draw(
        along_tau_data(("spill",)),
        "tau-lambda-spill",
        "**Spill along τ.** As in the figure of E1.",
        alt,
    )


def spill_alpha_figure() -> str:
    rows = [
        (r["alpha_eq"], r["spill"], r["lam"], r in EQ_RUNS, "no-emb") for r in SWEEP if r["removal"] >= REMOVAL_LEVEL
    ]
    rows += [(r["alpha_eq"], r["spill"], r["lam"], False, "whole") for r in REFERENCE if r["removal"] >= REMOVAL_LEVEL]
    alt = f"""
        Spill against α at every `=`, for the trials and runs whose removal is at least {REMOVAL_LEVEL:g}. No-emb
        trials rise from spill near 0.05 at α near zero to 0.4 to 0.6 at α of 0.4 to 0.5; the four with the anchor
        on `=`, drawn hollow, sit at the top right. Every-slice runs spread from 0 to 0.5 at α near zero.
    """
    return spill_alpha_draw(
        rows,
        f"""
        **Spill against the anchor at every `=`.** The trials whose removal is at least {REMOVAL_LEVEL:g}. Circles
        are `no-emb` trials, shaded by weight as before, hollow where the anchor sits on `=` (E2); triangles are
        every-slice trials and runs.
        """,
        alt,
    )


def plane_figure() -> str:
    alt = f"""
        The plane of τ (log, horizontal) and λ_a (log, vertical). Solid contours of the fitted removal at
        {REMOVAL_LEVEL:g} and {REMOVAL_TARGET:g} run straight up the plane between τ = 0.06 and {EDGE:g}, with a
        second {REMOVAL_LEVEL:g} contour near τ = 0.012. Dashed contours of the fitted spill at 0.15 and 0.3 curve
        across the right half, spill growing toward the upper right. The no-emb trials are
        filled dots, hollow for the four with the anchor on `=`; the every-slice trials are triangles along
        λ_a = {ex.LAMBDA_DEFAULT:g}, with a diamond at the recipe point. A star marks τ = {PROPOSED["tau"]:.2f},
        λ_a = {PROPOSED["lam"]:.2f}, on the {REMOVAL_TARGET:g} removal contour.
    """
    return plane_draw(
        plane_data(),
        (0.15, 0.3),
        f"""
        **The fitted surfaces over the plane.** Solid: removal at {REMOVAL_LEVEL:g} and {REMOVAL_TARGET:g}.
        Dashed: spill. Dots are the `no-emb` trials, hollow where the anchor sits on `=`; triangles the every-slice
        trials; the diamond the recipe point. The star is the point of least fitted spill where the fitted removal
        reaches {REMOVAL_TARGET:g}.
        """,
        alt,
    )


rf"""
## Spill (E3)

How much does the edit spill across the plane, and does it follow where the anchor sits? The figure plots spill along τ, as in E1.

{spill_figure()}

Past the edge, spill grows as the pool softens. Near the recipe τ (0.08 to 0.2), the `no-emb` trials spill about as much as the every-slice runs at the recipe point that did not latch, and less than the ones that did. So leaving out the embedding slice takes away the extra spill of the latch, and leaves the rest. The four trials with the anchor on `=` spill most of all, since their edit removes `=` in every context. Below the edge, the trials spill little, and remove little.

The spill follows the anchor at `=`. Among the trials that remove the op, leaving out the four on `=`, the correlation between spill and α at every `=` is {SPILL_R:.2f}.

{spill_alpha_figure()}

The trials that spread the anchor onto `=` spill most, and those that keep it at the example answers spill least, though even those spill well above the criterion of ex-2.2.21. The every-slice runs at the recipe point have little α at `=` and spill as widely as the `no-emb` trials, so `=` is one source of spill among others.

Putting removal and spill together, the most selective trials tend to sit near the edge, where the pool is just soft enough for the anchor to settle at the example answers and no softer. The three trials with the least spill among those that remove at least {REMOVAL_TARGET:g} of the way are {", ".join(f"`{r['name']}` (τ = {r['tau']:.2f}, λ_a = {r['lam']:.2f})" for r in BEST[:-1])}, and `{BEST[-1]["name"]}` (τ = {BEST[-1]["tau"]:.2f}, λ_a = {BEST[-1]["lam"]:.2f}). Two of them are among the few trials that cost the task something ({" and ".join(f"`{r['name']}`" for r in BEST if r["cost"] > COSTLY)}, whose worst HSV op ends lower than on most trials), so a model that learned the task less well may account for part of their low spill. The third has no task cost, and it sits on the edge.

{plane_figure()}

On the fitted surfaces, the point of least spill where removal reaches {REMOVAL_TARGET:g} is at τ = {PROPOSED["tau"]:.2f} and λ_a = {PROPOSED["lam"]:.2f}. There the fits give removal {PROPOSED["removal"]:.1f} ± {PROPOSED["removal_sd"]:.1f} and spill {PROPOSED["spill"]:.1f} ± {PROPOSED["spill_sd"]:.1f}. The edge is narrow along τ and the spill is noisy, so the region is a thin band beside a steep slope, rather than a broad plateau.
"""

# %%

# REVIEW: Scope held some method and results (round 2). The fits and the outcomes that did not vary moved to
# the measurements; Scope keeps the runs, the coupled weights as a limit, and the seed spread.
# REVIEW: softened two causal readings in the Discussion ("seems to have been the short training" ->
# "may", with no 200-epoch arm here; "no choice of slices would remove the latch" -> a conditional, since only the
# embedding slice was left out), and added that no-emb-04 and no-emb-06 have high task cost (0.03, 0.06; worst
# HSV op 0.41, 0.34), an alternative reading of their low spill. Verify against the table at the end.
# REVIEW: last sentence of the Discussion said a heavier anti term "does little"; λ_a moves both terms, so the
# ratio never changed (embedding-lean report). Now says the sweep is silent on it.
# E2: "four of its 8" is half, not "most"; "every syntax embedding" -> the top one, since only the max is measured.
r"""
## Discussion

The pull kept off the embedding slice does what spill-by-position hoped for over most of the plane: no syntax embedding latches, and the extra spill that a latched separator brings goes away. But the edit still spills at the recipe τ, about as much as on the every-slice runs without a latch. So most of the spill on a fully trained model has some other source. A later re-analysis, [where the lean comes from](/docs/m2/embedding-lean/report.py), traces it on the every-slice runs to lightness loaded onto e₁ in the color embedding table, and finds the `no-emb` trials here leaning the same way. The `no-emb` runs of ex-2.2.21 and ex-2.2.22 spilled little, and at 200 epochs on three seeds, that may have been the short training more than the pull, though this sweep has no 200-epoch arm to separate the two.

Leaving out the embedding slice also moves the latch rather than removing it, when the pool is sharp. At the sharpest τ the pull is met at the first slice it reaches, by putting the `=` embedding on e₁. That fits the account of the latch in spill-by-position, a pool that is close to a max is met most cheaply by a position every context shares, and it suggests that leaving out more slices might only move the latch further up at a sharp enough pool, though only the embedding slice was left out here.

Softening the pool spreads the anchor, as the backlog item expected, but onto `=` and the syntax tokens, and the spill grows with it. The anchor does reach the query `=`, though only together with every other `=`. So a softer pool does not help the edit here: by the time the anchor reaches the query, the spill has grown.

So within this plane, the least spill comes at the edge where the anchor first settles at the example answers. It is a narrow band beside a steep slope, and the noise in the spill is about as large as the differences along it, so we hold the proposed point loosely: with one trial per point, we cannot tell whether the spill drops on the edge or the trials there were lucky draws. Two of the three least-spilling trials also cost the task something (E3), which leaves one clean trial on the edge. If the spill does drop there, removal and spill are traded along τ, and λ_a barely enters removal; the spill fit varies with λ_a only through single trials. Since λ_a scales the anti-subspace term too, the ratio between the two terms is the same on every trial, so the sweep says nothing about a heavier anti-subspace term relative to the pull.
"""

# %%

rf"""
## All trials

{trials_table()}
"""
