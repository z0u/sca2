# ruff: noqa: B018
# title: When paired runs stop being alike

# An exploratory re-analysis, with no new runs: the correlation across seeds between two conditions trained at the
# same seeds, along training, from the stored trajectories and kept-checkpoint edits of the fixed and anneal
# confirmations, the loss-rate screen, and the late spill scout.

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from mini.lit import memo
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed


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


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


fc = _load_sibling("fixed-confirm", "paired_runs_fc")
lss = _load_sibling("late-spill-scout", "paired_runs_lss")
ac, sps, lr, rb = fc.ac, fc.sps, fc.lr, fc.rb

OPS: tuple[str, ...] = tuple(fetch_json(fc.AXES_REF)["design"]["ops"])
D = OPS.index(fc.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
ANSWERS = list(lr.lean.ANSWERS)  # the example-answer roles of the `difference` probe contexts
TRACK = lr.TRACK_SLICE  # slice 3, where the loss-rate call reads α
E1 = 0  # the anchored axis, in code


def ranks(x) -> np.ndarray:
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def r_of(x, y) -> float:
    """The rank correlation (Spearman) of two scores across seeds. Ranks, because spill is near zero on most runs
    and a single run that spills under both conditions would set a Pearson correlation on its own.
    """
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


# The standard error of Fisher's z for a rank correlation over n pairs (the Fieller, Hartley and Pearson
# approximation, sqrt(1.06 / (n − 3))).
def _z_se(n: int) -> float:
    return float(np.sqrt(1.06 / (n - 3)))


def chance(n: int) -> float:
    """The largest |r| that n pairs of unrelated scores give in about one draw of twenty (two-sided 95%)."""
    return float(np.tanh(1.96 * _z_se(n)))


def interval(r: float, n: int) -> tuple[float, float]:
    """A 95% interval for a rank correlation of *r* over *n* pairs, by Fisher's z."""
    z, h = np.arctanh(r), 1.96 * _z_se(n)
    return float(np.tanh(z - h)), float(np.tanh(z + h))


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def span(lo: float, hi: float) -> str:
    return f"{num(lo)} to {num(hi)}"


# --- Trajectories: the task score and α, recorded every four epochs during training ---------------------------


def series(traj: dict) -> dict[str, np.ndarray]:
    """The task score (expected exact match on the held-out probe contexts, mean over ops) and α at the example
    answers at slice 3, at each recorded epoch once (a run stopped at epoch 64 records it twice).
    """
    ep = np.round(np.asarray(traj["epoch"]), 3)
    _, first = np.unique(ep, return_index=True)
    alpha = np.asarray(traj["alpha_anchored"])[:, TRACK][:, ANSWERS].mean(axis=1)
    task = np.asarray(traj["eem_per_op"]).mean(axis=1)
    return {"epoch": ep[first], "alpha": alpha[first], "task": task[first]}


# From one initialization, to epoch 64: four recipes at the 32 seeds of the loss-rate screen (712 to 743).
SCREEN_SEEDS = list(lr.SEEDS)
_FC_SCREEN = fetch_json(fc.SCREEN_TRAJ_REF)
_AC_SCREEN = fetch_json(ac.SCREEN_TRAJ_REF)
_LR_SCREEN = {(v["arm"], v["model_seed"]): v["traj"] for v in fetch_json(lr.TRAJ_REF).values()}
FIXED, ANNEAL, STEP, FALLOFF, CONTROL = "fixed", "anneal", "step-soft", "falloff", "control"
SHOWN = {FIXED: "τ_ℓ fixed", ANNEAL: "τ_ℓ anneal", STEP: "step, soft τ", FALLOFF: "falloff", CONTROL: "control"}
SCREEN = {
    FIXED: {s: series(_FC_SCREEN[f"fixed-s{s}-e64"]["traj"]) for s in SCREEN_SEEDS},
    ANNEAL: {s: series(_AC_SCREEN[f"anneal-s{s}-e64"]["traj"]) for s in SCREEN_SEEDS},
    STEP: {s: series(_LR_SCREEN[(STEP, s)]) for s in SCREEN_SEEDS},
    FALLOFF: {s: series(_LR_SCREEN[(FALLOFF, s)]) for s in SCREEN_SEEDS},
}
SCREEN_EPOCHS = SCREEN[FIXED][SCREEN_SEEDS[0]]["epoch"]
assert all(np.array_equal(SCREEN[n][s]["epoch"], SCREEN_EPOCHS) for n in SCREEN for s in SCREEN_SEEDS)
PAIRS = [(FIXED, ANNEAL), (FIXED, STEP), (FIXED, FALLOFF), (STEP, FALLOFF)]


def r_along(runs: dict, a: str, b: str, key: str, seeds: list[int]) -> np.ndarray:
    """The correlation across *seeds* between conditions *a* and *b*, at each recorded epoch."""
    x = np.array([runs[a][s][key] for s in seeds])
    y = np.array([runs[b][s][key] for s in seeds])
    return np.array([r_of(x[:, j], y[:, j]) for j in range(x.shape[1])])


def r_next(runs: dict, n: str, key: str, seeds: list[int]) -> np.ndarray:
    """The correlation across seeds between each run of *n* and the same run four epochs later (from the second
    record on): how much one record of a run predicts its next one.
    """
    x = np.array([runs[n][s][key] for s in seeds])
    return np.array([r_of(x[:, j - 1], x[:, j]) for j in range(1, x.shape[1])])


R_SCREEN = {(a, b, k): r_along(SCREEN, a, b, k, SCREEN_SEEDS) for a, b in PAIRS for k in ("task", "alpha")}
NEXT_SCREEN = {k: np.mean([r_next(SCREEN, n, k, SCREEN_SEEDS) for n in SCREEN], axis=0) for k in ("task", "alpha")}
LATE = SCREEN_EPOCHS >= 44  # the window of the loss-rate call, epochs 44 to 64
LATE_R = {key: float(np.mean(v[LATE])) for key, v in R_SCREEN.items()}
LATE_NEXT = {k: float(np.mean(v[LATE[1:]])) for k, v in NEXT_SCREEN.items()}
FIRST_R = {key: float(v[0]) for key, v in R_SCREEN.items()}

# --- Early edits at forty seeds: removal and spill at epochs 16, 32 and 64, `τ_ℓ fixed` against `τ_ℓ anneal` ---


def net_of(drop) -> np.ndarray:
    """The drop under the e₁ edit less the median drop along the other axes: *drop* is (axes, ...), e₁ first."""
    drop = np.asarray(drop)
    return drop[E1] - np.median(drop[1:], axis=0)


_EARLY = {(r["kind"], r["model_seed"], r["epoch"]): net_of(r["drop"]) for r in fetch_json(fc.EARLY_REF)["runs"]}
EARLY_SEEDS = sorted({s for _, s, _ in _EARLY})
EARLY_EPOCHS = sorted({e for *_, e in _EARLY})
EARLY_R = {
    (key, e): r_of(*([f(_EARLY[(n, s, e)]) for s in EARLY_SEEDS] for n in (FIXED, ANNEAL)))
    for key, f in (("removal", lambda v: v[D]), ("spill", lambda v: v[OTHER].max()))
    for e in EARLY_EPOCHS
}

# --- The whole run, at eight seeds: `τ_ℓ fixed`, `τ_ℓ anneal` and their control at 712 to 719 -----------------

SEEDS8 = list(range(712, 720))
_FC_FIN = fetch_json(fc.FINISHED_TRAJ_REF)
_AC_FIN = fetch_json(ac.FINISHED_TRAJ_REF)
FULL_RUNS = {
    FIXED: {s: series(_FC_FIN[f"fixed-s{s}"]["traj"]) for s in SEEDS8},
    ANNEAL: {s: series(_AC_FIN[f"anneal-s{s}"]["traj"]) for s in SEEDS8},
    CONTROL: {s: series(_AC_FIN[f"control-s{s}"]["traj"]) for s in SEEDS8},
}
FULL_EPOCHS = FULL_RUNS[FIXED][SEEDS8[0]]["epoch"]
R_FULL = {(a, b): r_along(FULL_RUNS, a, b, "task", SEEDS8) for a, b in ((FIXED, ANNEAL), (FIXED, CONTROL), (ANNEAL, CONTROL))}  # fmt: skip

_ALONG = {(n, r["model_seed"], r["epoch"]): net_of(r["drop"]) for n, ref in ((FIXED, fc.ALONG_REF), (ANNEAL, ac.ALONG_REF)) for r in fetch_json(ref)["runs"]}  # fmt: skip
KEPT_EPOCHS = sorted({e for (n, _, e) in _ALONG if n == FIXED})
KEPT_R = {
    key: np.array([r_of(*([f(_ALONG[(n, s, e)]) for s in SEEDS8] for n in (FIXED, ANNEAL))) for e in KEPT_EPOCHS])
    for key, f in (("removal", lambda v: v[D]), ("spill", lambda v: v[OTHER].max()))
}
KEPT_LATE = np.asarray(KEPT_EPOCHS) >= 160

# At the end: the task change of the D2.2 design nets each run against the control at its seed. How alike is a run
# to that control? Five recipe sets, each with the control trained at its seeds.
_AX: dict[str, dict[Any, dict]] = {
    "fc": {r["label"]: r for r in fetch_json(fc.AXES_REF)["runs"]},
    "ac": {r["label"]: r for r in fetch_json(ac.AXES_REF)["runs"]},
    "rb": {r["label"]: r for r in fetch_json(rb.AXES_REF)["runs"]},
    "sps": {(r["condition"], r["model_seed"]): r for r in fetch_json(sps.AXES_REF)["runs"]},
}
OLD8 = list(range(704, 712))
END_SETS = [
    (FIXED, SEEDS8, lambda s: _AX["fc"][fc.label_of(s, None)], lambda s: _AX["ac"][ac.label_of(s, None, "control")]),
    (ANNEAL, SEEDS8, lambda s: _AX["ac"][ac.label_of(s, None)], lambda s: _AX["ac"][ac.label_of(s, None, "control")]),
    (FIXED, OLD8, lambda s: _AX["sps"][(fc.BASE, s)], lambda s: _AX["rb"][f"control-s{s}"]),
    (STEP, OLD8, lambda s: _AX["rb"][rb.SOFT_POOL + f"step-soft-s{s}"], lambda s: _AX["rb"][f"control-s{s}"]),
    (FALLOFF, OLD8, lambda s: _AX["rb"][f"falloff-s{s}"], lambda s: _AX["rb"][f"control-s{s}"]),
]


def end_task(run_of, control_of, seeds) -> dict[str, Any]:
    x = np.array([np.mean(run_of(s)["clean"]) for s in seeds])
    c = np.array([np.mean(control_of(s)["clean"]) for s in seeds])
    return {
        "r": r_of(x, c),
        "sd_run": float(x.std(ddof=1)),
        "sd_control": float(c.std(ddof=1)),
        # The spread of the task change netted per seed, and netted against the seed mean of the control.
        "sd_paired": float((x - c).std(ddof=1)),
        "sd_unpaired": float(np.sqrt(x.var(ddof=1) + c.var(ddof=1) / len(seeds))),
    }


END_TASK: list[dict[str, Any]] = [
    end_task(run, ctl, seeds) | {"name": n, "seeds": seeds} for n, seeds, run, ctl in END_SETS
]
END_R = np.array([t["r"] for t in END_TASK])

# --- From one stored state: the late spill scout, which carries each run on from its epoch-64 state ------------

_LSS_TRAJ = {(v["condition"], v["arm"], v["model_seed"]): v["traj"] for v in fetch_json(lss.TRAJ_REF).values()}
_SPS_FIN = fetch_json(sps.FINISHED_TRAJ_REF)
BASE, LT_LATE, ANTI_LATE, ANTI_HELD, GPU = "base", "lt-late", "anti-late", "anti-held", "L4"
ARMS = [LT_LATE, ANTI_LATE, ANTI_HELD]
SHOWN |= {LT_LATE: "τ_ℓ late", ANTI_LATE: "anti late", ANTI_HELD: "anti held", GPU: "base on L4"}
BRANCH = {a: {s: series(_LSS_TRAJ[(lss.CONDITION, a, s)]) for s in OLD8} for a in (BASE, *ARMS)} | {
    GPU: {s: series(_SPS_FIN[sps.label_of(lss.CONDITION, s, None)]["traj"]) for s in OLD8}
}


def after_branch(run: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep = run["epoch"] > lss.STOP_EPOCH
    return {k: v[keep] for k, v in run.items()}


BRANCH = {a: {s: after_branch(v) for s, v in runs.items()} for a, runs in BRANCH.items()}
BRANCH_EPOCHS = BRANCH[BASE][OLD8[0]]["epoch"]
assert all(np.array_equal(BRANCH[a][s]["epoch"], BRANCH_EPOCHS) for a in BRANCH for s in OLD8)
R_BRANCH = {(a, k): r_along(BRANCH, a, BASE, k, OLD8) for a in (*ARMS, GPU) for k in ("task", "alpha")}

_LSS_ALONG = {
    (r["arm"], r["model_seed"], r["epoch"]): np.median(np.asarray(r["es_sham"]), axis=0) - np.asarray(r["es_edit"])
    for r in fetch_json(lss.ALONG_REF)["runs"]
    if r["condition"] == lss.CONDITION
}  # the net drop per op under the full edit, against the median of eight reference edits (spill-timing)
LSS_EPOCHS = sorted({e for *_, e in _LSS_ALONG})
LSS_R = {
    (a, key): np.array(
        [
            r_of([f(_LSS_ALONG[(a, s, e)]) for s in OLD8], [f(_LSS_ALONG[(BASE, s, e)]) for s in OLD8])
            for e in LSS_EPOCHS
        ]
    )  # fmt: skip
    for a in ARMS
    for key, f in (("removal", lambda v: v[D]), ("spill", lambda v: v[OTHER].max()))
}
# Check: `anti late` and `anti held` have one schedule to epoch 200, so they must agree exactly to epoch 192.
_SAME = [e for e in LSS_EPOCHS if e <= 192]
assert all(np.array_equal(_LSS_ALONG[(ANTI_LATE, s, e)], _LSS_ALONG[(ANTI_HELD, s, e)]) for s in OLD8 for e in _SAME)

# At the end, `base` (A10G after epoch 64) against the finished run of the soft pool scout (L4 throughout).
_LSS_AX = {(r["condition"], r["arm"], r["model_seed"]): r for r in fetch_json(lss.AXES_REF)["runs"]}


def removal_spill(r: dict) -> tuple[float, float]:
    net = net_of(r["drop"])[len(rb.DOSE_GAMMAS) - 1]
    return float(net[D]), float(net[OTHER].max())


GPU_END = {
    key: r_of(
        [removal_spill(_LSS_AX[(lss.CONDITION, BASE, s)])[i] for s in OLD8],
        [removal_spill(_AX["sps"][(lss.CONDITION, s)])[i] for s in OLD8],
    )
    for i, key in enumerate(("removal", "spill"))
}
# Each arm against `base` at the end under the 63-axis baseline of E1 and E2, to separate the change of baseline
# (eight reference edits along training in E3) from the change of pairing.
ARM_END_AXES = {
    (a, key): r_of(
        [removal_spill(_LSS_AX[(lss.CONDITION, a, s)])[i] for s in OLD8],
        [removal_spill(_LSS_AX[(lss.CONDITION, BASE, s)])[i] for s in OLD8],
    )
    for a in ARMS
    for i, key in enumerate(("removal", "spill"))
}
BRANCH_LATE = BRANCH_EPOCHS > 288
LSS_LATE = np.asarray(LSS_EPOCHS) >= 320


def binned(epochs, r, start: float, width: float = 32) -> tuple[np.ndarray, np.ndarray]:
    """The mean of a correlation over the records of each stretch of *width* epochs after *start*, placed at the end
    of the stretch: at eight seeds one record is too noisy to draw on its own.
    """
    epochs, r = np.asarray(epochs), np.asarray(r)
    ends = np.arange(start + width, epochs.max() + 1e-6, width)
    ends = np.append(ends, epochs.max()) if ends[-1] < epochs.max() else ends
    return ends, np.array([r[(epochs > e - width) & (epochs <= e)].mean() for e in ends])


R_FULL_BINNED = {k: binned(FULL_EPOCHS, v, 0) for k, v in R_FULL.items()}
R_BRANCH_BINNED = {k: binned(BRANCH_EPOCHS, v, lss.STOP_EPOCH) for k, v in R_BRANCH.items()}

# --- Figures ---------------------------------------------------------------------------------------------------

# Inks are (light, dark) pairs, resolved with `light_dark` where a figure draws.
RULE = ("#333", "#ddd")
BAND = ("#e4e4e4", "#2a2a2a")
GHOST = ("#888", "#999")
INK = {
    (FIXED, ANNEAL): ("#08306b", "#c6dbef"),
    (FIXED, STEP): ("#1b7837", "#a6dba0"),
    (FIXED, FALLOFF): ("#a6611a", "#dfc27d"),
    (STEP, FALLOFF): ("#7570b3", "#bcbddc"),
    (FIXED, CONTROL): ("#a6611a", "#dfc27d"),
    (ANNEAL, CONTROL): ("#7570b3", "#bcbddc"),
    LT_LATE: ("#08306b", "#c6dbef"),
    ANTI_LATE: ("#1b7837", "#a6dba0"),
    ANTI_HELD: ("#a6611a", "#dfc27d"),
    GPU: ("#7570b3", "#bcbddc"),
}
MARKER = {
    (FIXED, ANNEAL): "o",
    (FIXED, STEP): "^",
    (FIXED, FALLOFF): "v",
    (STEP, FALLOFF): "D",
    (FIXED, CONTROL): "v",
    (ANNEAL, CONTROL): "D",
    LT_LATE: "o",
    ANTI_LATE: "^",
    ANTI_HELD: "v",
    GPU: "D",
}


def tex(shown: str) -> str:
    """A recipe name for a figure label, with its symbols as mathtext."""
    return shown.replace("τ_ℓ", "$\\tau_\\ell$").replace("τ", "$\\tau$")


def pair_label(key) -> str:
    if isinstance(key, tuple):
        return f"{tex(SHOWN[key[0]])} ~ {tex(SHOWN[key[1]])}"
    return f"{tex(SHOWN[key])} ~ base"


def panel_data(title: str, lines: dict, band: float, next_line=None) -> dict:
    """One panel: *lines* maps a pairing to (epochs, r); *band* is the chance bound for its seed count."""
    return {
        "title": title,
        "lines": [
            {
                "label": pair_label(k),
                "ink": INK[k],
                "marker": MARKER[k],
                "x": np.asarray(x).tolist(),
                "y": np.asarray(y).tolist(),
            }  # fmt: skip
            for k, (x, y) in lines.items()
        ],
        "band": band,
        "next": None if next_line is None else [np.asarray(v).tolist() for v in next_line],
    }


@memo
def r_draw(name: str, panels: list[dict], size: tuple[float, float], xlim: tuple[float, float], alt_text: str, caption: str) -> str:  # fmt: skip
    @themed(name=name, alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(panels), figsize=size, layout="constrained", sharey=True)
        axes = cast(AxesRow, np.atleast_1d(axes))
        seen: dict[str, Any] = {}
        for ax, p in zip(axes, panels, strict=True):
            ax.axhspan(-p["band"], p["band"], color=light_dark(*BAND), lw=0, zorder=0)
            ax.axhline(0, color=light_dark(*GHOST), lw=0.5, zorder=1)
            if p["next"] is not None:
                (h,) = ax.plot(*p["next"], color=light_dark(*GHOST), lw=1.0, ls="--", zorder=2)
                seen.setdefault("the same run, four epochs on", h)
            for line in p["lines"]:
                (h,) = ax.plot(line["x"], line["y"], color=light_dark(*line["ink"]), marker=line["marker"], ms=2.6, lw=0.9, zorder=3)  # fmt: skip
                seen.setdefault(line["label"], h)
            ax.set_title(p["title"], fontsize=8)
            ax.set_xlim(*xlim)
            ax.set_ylim(-1.0, 1.02)
            ax.set_xlabel("epoch", fontsize=7)
            ax.tick_params(labelsize=6.5)
        axes[0].set_ylabel("rank correlation across seeds", fontsize=7)
        fig.legend(list(seen.values()), list(seen), loc="outside upper center", ncols=3, frameon=False, fontsize=6.5)
        return fig

    return _plot()


def screen_figure() -> str:
    ep = SCREEN_EPOCHS
    lines = {k: {(a, b): (ep, R_SCREEN[(a, b, k)]) for a, b in PAIRS} for k in ("task", "alpha")}
    early = {(FIXED, ANNEAL): (EARLY_EPOCHS, [EARLY_R[("removal", e)] for e in EARLY_EPOCHS])}
    early_spill = {(FIXED, ANNEAL): (EARLY_EPOCHS, [EARLY_R[("spill", e)] for e in EARLY_EPOCHS])}
    panels = [
        panel_data("task score", lines["task"], chance(len(SCREEN_SEEDS)), (ep[1:], NEXT_SCREEN["task"])),
        panel_data(
            "α at the example answers", lines["alpha"], chance(len(SCREEN_SEEDS)), (ep[1:], NEXT_SCREEN["alpha"])
        ),  # fmt: skip
        panel_data("removal (40 seeds)", early, chance(len(EARLY_SEEDS))),
        panel_data("spill (40 seeds)", early_spill, chance(len(EARLY_SEEDS))),
    ]
    alt = """
        Over the first 64 epochs, every pairing falls from about 0.9 to about 0.5 in the task score, and α falls to
        the low level of a run against itself four epochs on. Removal stays just above the grey band of chance, and spill inside it.
    """
    return r_draw(
        "paired-runs-screen",
        panels,
        (6.8, 2.4),
        (0, 66),
        alt,
        f"""
        **From one initialization: how alike the runs of a seed stay under two recipes.** Each line is one pairing of
        recipes, and each point the rank correlation of a score across the {len(SCREEN_SEEDS)} seeds of the loss-rate
        screen (712 to 743), at one record. **Left two:** the task score and α at the example answers, recorded every
        four epochs during training; the dashed line is each run against itself at its previous record, averaged over
        the four recipes. **Right two:** removal and spill under the full e₁ edit at the three checkpoints kept to
        epoch 64, for the one pairing that scored them there, at forty seeds (704 to 743). The grey band is where a
        correlation falls about nineteen times in twenty when the two scores are unrelated.
        """,
    )


def full_figure() -> str:
    kept = {(FIXED, ANNEAL): (KEPT_EPOCHS, KEPT_R["removal"])}
    kept_spill = {(FIXED, ANNEAL): (KEPT_EPOCHS, KEPT_R["spill"])}
    task = R_FULL_BINNED
    band = chance(len(SEEDS8))
    panels = [
        panel_data("removal", kept, band),
        panel_data("spill", kept_spill, band),
        panel_data("task score", task, band),
    ]
    alt = """
        To epoch 400 at eight seeds, removal, spill, and the task score all wander inside the wide grey band of chance.
        The one point outside it is removal at epoch 16, before the anchor forms.
    """
    return r_draw(
        "paired-runs-full",
        panels,
        (6.8, 2.5),
        (0, 405),
        alt,
        f"""
        **Over the whole run, at eight seeds.** The rank correlation across seeds 712 to 719. **Left two:** removal and
        spill under the full e₁ edit, `τ_ℓ fixed` against `τ_ℓ anneal`, at the checkpoints kept at epochs 16, 32, and
        64, then every 32 epochs, and at the end. **Right:** the task score, for the two recipes against each other and
        each against the control at the same seeds. Each point is the mean correlation over the records (every four
        epochs) of the 32 epochs up to it, or of the last 16 epochs for the final point. The grey band is where a
        correlation over eight seeds falls about nineteen times in twenty when the two scores are unrelated
        (±{band:.2f}).
        """,
    )


def branch_figure() -> str:
    removal = {a: (LSS_EPOCHS, LSS_R[(a, "removal")]) for a in ARMS}
    spill = {a: (LSS_EPOCHS, LSS_R[(a, "spill")]) for a in ARMS}
    task = {a: R_BRANCH_BINNED[(a, "task")] for a in (*ARMS, GPU)}
    band = chance(len(OLD8))
    panels = [
        panel_data("removal", removal, band),
        panel_data("spill", spill, band),
        panel_data("task score", task, band),
    ]
    alt = """
        After the branch, every arm stays well above the grey band of chance in spill to epoch 400, and in removal and
        the task score until about epoch 300. After that, removal under the two anti arms and the task score on the
        other GPU fall to about 0.5.
    """
    return r_draw(
        "paired-runs-branch",
        panels,
        (6.8, 2.5),
        (60, 405),
        alt,
        f"""
        **From one stored state: the late spill scout.** The rank correlation across seeds 704 to 711 between each arm
        and `base`, all carried on from the stored epoch-64 state of `τ_ℓ 0.1, τ_ℓ fixed` on an A10G. **Left two:**
        removal and spill under the full e₁ edit at the checkpoints kept every 32 epochs from epoch 96, and at the
        end, net of eight reference edits. **Right:** the task score, each point the mean correlation over the records
        of the 32 epochs up to it (16 for the last). One more pairing (`base` on L4) uses the run of the same schedule
        from the soft pool scout, trained on L4 throughout. The grey band is the chance band for eight seeds
        (±{band:.2f}).
        """,
    )


def end_table() -> str:
    # No figure beside this table: it is one value per row at epoch 400, and the right panel of the E2 figure already
    # draws the correlation of the first two rows along training.
    head = ["recipe", "seeds", "rank correlation", "95% interval", "spread, netted per seed ↓", "spread, netted against the control mean ↓"]  # fmt: skip
    rows = []
    for t in END_TASK:
        lo, hi = interval(t["r"], len(t["seeds"]))
        rows.append([
            f"<code>{SHOWN[t['name']]}</code>",
            f"{t['seeds'][0]}–{t['seeds'][-1]}",
            num(t["r"]),
            span(lo, hi),
            num(t["sd_paired"], ".4f"),
            num(t["sd_unpaired"], ".4f"),
        ])  # fmt: skip
    ths = "".join(f"<th{' class=num' if i > 1 else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f"<td{' class=num' if i > 1 else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)  # fmt: skip
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(
        table,
        caption="""
            **The task score at epoch 400, against the control at the same seeds.** The rank correlation across eight
            seeds between the clean score of each run (mean over ops) and that of the control, with its interval. The
            two spreads are standard deviations over seeds of the task change. The first nets each run against the
            control at the same seed, as the D2.2 design does. The second nets it against the seed mean of the control,
            and includes the uncertainty of that mean.
        """,
        class_="report-figure",
    )


# --- Values the prose quotes -----------------------------------------------------------------------------------

TASK_FIRST = [FIRST_R[(a, b, "task")] for a, b in PAIRS]
TASK_LATE = [LATE_R[(a, b, "task")] for a, b in PAIRS]
ALPHA_FIRST = [FIRST_R[(a, b, "alpha")] for a, b in PAIRS]
ALPHA_LATE = [LATE_R[(a, b, "alpha")] for a, b in PAIRS]
CHANCE = {n: chance(n) for n in (len(SCREEN_SEEDS), len(EARLY_SEEDS), len(SEEDS8))}
EARLY_REMOVAL = [EARLY_R[("removal", e)] for e in EARLY_EPOCHS if e > 16]
EARLY_SPILL = [EARLY_R[("spill", e)] for e in EARLY_EPOCHS]
KEPT_AFTER = {k: v[1:] for k, v in KEPT_R.items()}  # from epoch 32, once the anchor has formed
FC_400 = {k: float(v[-1]) for k, v in KEPT_R.items()}
FC_400_INTERVAL = interval(FC_400["removal"], len(SEEDS8))
END_PAIRED_LOWER = sum(t["sd_paired"] < t["sd_unpaired"] - 5e-4 for t in END_TASK)
END_PAIRED_HIGHER = sum(t["sd_paired"] > t["sd_unpaired"] + 5e-4 for t in END_TASK)
END_INTERVALS_HAVE_ZERO = all(interval(t["r"], len(t["seeds"]))[0] < 0 for t in END_TASK)
assert END_INTERVALS_HAVE_ZERO
LSS_SPILL_MIN = min(float(LSS_R[(a, "spill")].min()) for a in ARMS)
LSS_REMOVAL_LT = LSS_R[(LT_LATE, "removal")]
LSS_REMOVAL_ANTI_LATE = [float(LSS_R[(a, "removal")][LSS_LATE].mean()) for a in (ANTI_LATE, ANTI_HELD)]
BRANCH_TASK_LATE = {a: float(R_BRANCH[(a, "task")][BRANCH_LATE].mean()) for a in (*ARMS, GPU)}
ARM_END = {key: [ARM_END_AXES[(a, key)] for a in ARMS] for key in ("removal", "spill")}
ARM_END_ANTI = sorted(ARM_END_AXES[(a, "removal")] for a in (ANTI_LATE, ANTI_HELD))
# The prose says these sit above every E2 correlation after the anchor forms.
assert min(ARM_END["removal"]) > KEPT_AFTER["removal"].max() and min(ARM_END["spill"]) > KEPT_AFTER["spill"].max()
WORD = "no one two three four five six seven eight nine ten".split()

# %%

rf"""
# When paired runs stop being alike

/// tip |
<!-- lede -->
Two runs at one seed share their initialization, batches, and labels. Started together under two recipes, they stay partly alike in the task score through epoch 64, whether the recipes differ a little or a lot, but they are only weakly alike in removal and no more alike than chance in spill. Carried on from one stored state, they stay alike in all three scores for hundreds of epochs.
///
<!-- REVIEW: moved "whether the recipes differ a little or a lot" from the removal/spill clause to the task-score
clause. From one initialization, removal and spill were scored for one pairing only (fixed against anneal, the
smallest change); the four pairings cover the task score and α (E1). Verify: the right two panels of the E1 figure
carry one line. -->


Our design rule is to compare conditions as two independent samples. We pair runs by seed only when they branch from one stored state and are measured soon after ([eng/determinism.md](/eng/determinism.md#seeds-are-draws)). But "soon after" has no number yet.

Two earlier results set the rule. Removal under `τ_ℓ fixed` and `τ_ℓ anneal`, which share their seeds, correlated about 0.2 across eight seeds at epoch 400 (the fixed confirmation). Twin runs (the same in all but the GPU arithmetic) are as far apart in their weights as unrelated runs by epoch 16, while their task loss agrees to a hundredth (the low-runs report).

So the weights drift apart early, while some scores do not. Here we ask how alike two runs at one seed stay, score by score, as training goes on, using only the trajectories and kept checkpoints that four earlier experiments stored. For each pair of conditions trained at the same seeds, we take the correlation of a score across seeds at each record (each recorded epoch).

## Observations

- [From one initialization (E1)](#from-one-initialization-e1): over the first 64 epochs, the two runs of a seed start out ranking the seeds in nearly the same order on the task score and end at a rank correlation of about a half, under every pairing of recipes. Removal correlates by about a third, at the edge of chance, and spill by no more than chance. The alignment with e₁ (α) moves so much from record to record that one record of a run predicts its own next record about as poorly as it predicts the paired run.
- [Over the whole run (E2)](#over-the-whole-run-e2): at eight seeds, once the anchor forms, no correlation of removal or spill can be told apart from zero, including the correlation of 0.2 at epoch 400. The task score of a run is only loosely tied to that of the control at its seed. Netting the task change against that control, seed by seed, leaves its spread about where it was.
- [From one stored state (E3)](#from-one-stored-state-e3): runs carried on from one epoch-64 state stay alike in spill to the end. They stay alike in removal and the task score for at least a hundred and fifty epochs, under each change of schedule and under a change of GPU. Under the two anti arms, the pairing in removal loosens after epoch 300, though at eight seeds that is not firm.
<!-- REVIEW: was "The largest change loosens removal after epoch 300." Named the arms (E3 never says "the largest
change") and hedged: at the end, under the 63-axis baseline, the anti arms correlate with `base` in removal at
about 0.64 and 0.86 against about a half under the eight reference edits, and at eight seeds those are the same
measurement. Verify: ARM_END_ANTI in E3. -->


## Scope and method

This is an exploratory re-analysis, with no preregistration and no gate. It covers three kinds of pairing, ordered by how much the two runs of a seed share, least first.

From one initialization
:   Four recipes stopped at epoch 64, each at the {len(SCREEN_SEEDS)} seeds of the loss-rate screen (712 to 743): `τ_ℓ fixed` and `τ_ℓ anneal` from the fixed and anneal confirmations, `step, soft τ` and `falloff` from the loss-rate screen. At a seed the four share their initialization, batches, and label draws, and differ by varying amounts. `τ_ℓ anneal` is `τ_ℓ fixed` with the slice temperature sharpening (by under 1% before epoch 64), `step, soft τ` pulls at a step over slices 2 to 4 in place of the pool over slices 1 to 3, and `falloff` is that step with the position temperature held fixed. The fixed confirmation also scored the edit on both of its recipes at epochs 16, 32, and 64, at forty seeds (704 to 743).

Over the whole run
:   `τ_ℓ fixed` and `τ_ℓ anneal` at seeds 712 to 719, trained to epoch 400, with the control at the same seeds (no pull and no anti term). At the end of training only, three more recipes join them, at seeds 704 to 711 with their own control, as the reference-baseline experiment and the soft pool scout scored them.

From one stored state
:   The four arms of the late spill scout (`base`, `τ_ℓ late`, `anti late`, and `anti held`) at seeds 704 to 711. Each arm is carried on from the stored epoch-64 state of `τ_ℓ 0.1, τ_ℓ fixed` at its seed, on an A10G, and each arm other than `base` changes one schedule from epoch 64 on. The soft pool scout trained the same schedule as `base` from the same state on an L4, which gives one more pairing: the same model on another GPU.

Within each pairing, both runs trained on the same kind of GPU (L4 for the first two kinds of pairing, A10G for the third), except for that last pairing. As a check that the code pairs the right runs, `anti late` and `anti held`, which share a schedule up to epoch 200, agree exactly at every kept checkpoint to epoch 192.

A correlation between unrelated scores falls within ±{CHANCE[len(SCREEN_SEEDS)]:.2f} of zero about nineteen times in twenty at {len(SCREEN_SEEDS)} seeds, within ±{CHANCE[len(EARLY_SEEDS)]:.2f} at forty, and within ±{CHANCE[len(SEEDS8)]:.2f} at eight. Each figure shades that range grey, as the band of chance. Neighboring records of one pairing are not independent draws, so a correlation that stays above the band over many records says more than one point does.

## The measurements

Rank correlation across seeds
:   How well the order of the seeds on a score under one condition predicts their order under the other (Spearman's correlation), for two conditions trained at the same seeds, at one epoch. It is 1 when the seeds come in the same order under both, and near 0 when one order says nothing about the other. Near 1, a difference taken seed by seed cancels most of the variation between seeds; near 0, none of it. Ranks rather than values, because spill is near zero on most runs, and one run that spills under both conditions would set the correlation of the values on its own.

Task score
:   Expected exact match on the held-out probe contexts, averaged over ops, recorded every four epochs during training. Higher is better. At the end (E2), it is the clean score of the scoring suite, averaged over ops.

α at the example answers
:   How far the pull has brought the anchored op onto e₁: the mean alignment with e₁ of the states at the example answers of `difference` contexts, at slice 3, recorded every four epochs. The loss-rate call averages it over epochs 44 to 64.

Removal
:   The drop on `difference` under the full e₁ edit, applied at every position and slice. It is net of the median drop when the same edit is applied along each of the other 63 coordinate axes of the run. In E3 it is net of the median of eight reference edits instead, as spill-timing scored it. We report the net drop rather than a share of the way to the target null, since a stopped run has no end record to scale it by.

Spill
:   The largest net drop on any other op, under the same edit.
"""

# %%

rf"""
## From one initialization (E1)

Started from one initialization under two recipes, how long do the two runs of a seed stay more alike than runs at different seeds? If the seed sets most of what a run does, the correlation should stay high whatever the change. If small differences grow quickly, it should fall early, and fall faster the larger the change.

In the figure below, the task score stays partly paired through epoch 64. Removal is only weakly paired, and spill no more than chance would give.

{screen_figure()}

At epoch 4, the two runs of a seed rank the seeds in nearly the same order on the task score under every pairing (a rank correlation of {min(TASK_FIRST):.2f} to {max(TASK_FIRST):.2f}). The correlation falls slowly from there, to about a half on average over epochs 44 to 64 ({min(TASK_LATE):.2f} to {max(TASK_LATE):.2f}).

From epoch 16 on, a run correlates with itself four epochs later at about {LATE_NEXT["task"]:.1f}. So the fall comes from the two runs drifting apart, while under any one recipe the order of the seeds on the score changes little from one record to the next.

The size of the change makes no clear difference. `τ_ℓ fixed` and `falloff`, which pull at different slices with different temperatures, stay about as alike in the task score as `τ_ℓ fixed` and `τ_ℓ anneal`, which differ by under 1% in one temperature. That fits what the low-runs report found in the weights, where a small difference grows as quickly as a large one.

α behaves differently. The runs of a seed start alike in it ({min(ALPHA_FIRST):.2f} to {max(ALPHA_FIRST):.2f} at epoch 4), and by epoch 44 the correlation is about a third ({min(ALPHA_LATE):.2f} to {max(ALPHA_LATE):.2f}). But a run correlates with itself four epochs later about as weakly ({LATE_NEXT["alpha"]:.2f}). So α moves a lot from record to record, and the paired run at the same epoch is about as good a guide to it as the same run four epochs earlier.

The two runs see the same batches in the same order, which may keep some of that movement in step, though nothing here tests that.

Removal under the edit correlates by about a third at epochs 32 and 64 ({span(min(EARLY_REMOVAL), max(EARLY_REMOVAL))} over forty seeds), at the edge of chance. Spill stays inside the band at all three checkpoints. At epoch 16 the anchor has barely formed, and removal is close to zero on every run.

Pairing by seed from one initialization would cancel about half of the variation between seeds in the task score, about a third in removal if that correlation is real, and none we can detect in spill.
<!-- REVIEW: was "and little of it in removal or spill". The reading that gives "about half" at a correlation of
0.5 to 0.6 gives about a third at 0.32 to 0.38, which removal holds at all three checkpoints over forty seeds. -->
"""

# %%

rf"""
## Over the whole run (E2)

Does what is left of the pairing at epoch 64 last to the end of training? And how alike is a run to the control at its seed, which the D2.2 design nets the task change against?

The figure follows `τ_ℓ fixed` against `τ_ℓ anneal`, and each against the control, to epoch 400 at eight seeds, where the band of chance is wide.

{full_figure()}

From epoch 32 on, every correlation of removal and spill between the two recipes falls inside the wide band of chance for eight seeds, wandering from about {num(min(KEPT_AFTER["spill"].min(), KEPT_AFTER["removal"].min()), ".1f")} to {num(max(KEPT_AFTER["spill"].max(), KEPT_AFTER["removal"].max()), ".1f")} between neighboring checkpoints.[^e16]

So the correlation of about 0.2 at epoch 400 (as a rank correlation, {num(FC_400["removal"])}, with an interval of {span(*FC_400_INTERVAL)}) is consistent with anything from no pairing at all to a strong one. At epoch 64, the forty seeds of E1 pin the pairing in removal down better, to about a third.

[^e16]: The {KEPT_R["removal"][0]:.2f} in removal at epoch 16 comes before the anchor forms, when removal is noise around zero.

The pairing in the task score is as unsteady. The two recipes correlate by about a third to a half over the run, and each correlates about as well with the control, though the control has neither the pull nor the anti term. So for the task score, what the two runs of a seed share seems to matter more than how alike their recipes are.

{end_table()}

The table gives the end of training for five recipes and seed ranges. At epoch 400, the task score of a run correlates with that of the control at its seed by {min(END_R):.2f} to {max(END_R):.2f}, and every interval reaches below zero.

Netting the task change against the control seed by seed gives about the same spread as netting it against the seed mean of the control, a little smaller in {WORD[END_PAIRED_LOWER]} row{"" if END_PAIRED_LOWER == 1 else "s"} of the table and larger in {WORD[END_PAIRED_HIGHER]}. In four of the five rows the control spreads about half as much as the anchored runs, which is why the choice matters little either way.
<!-- REVIEW: was "about half as much as the anchored runs"; on `τ_ℓ fixed` at 704 to 711 the control spreads nearly
as much as the run (0.0048 against 0.0055), so the claim holds on four sets. -->


So at the end of training, eight seeds cannot show that pairing by seed helps on any of these scores.
"""

# %%

rf"""
## From one stored state (E3)

Runs carried on from one stored state share everything up to that state. By epoch 64 the runs at different seeds have already drifted apart, and both runs of a seed start from where that seed got to. How long do they stay alike after the branch, and does that depend on the change?

The figure compares each arm of the late spill scout with `base`, the arm that keeps the schedule unchanged, at the same seed.

{branch_figure()}

They stay alike for a long time. In spill, every arm keeps a rank correlation with `base` of at least {LSS_SPILL_MIN:.2f} at every checkpoint to the end. In removal, every arm stays mostly between 0.8 and 0.95 to epoch 288, with a dip to about a half at single checkpoints.

After epoch 288, `τ_ℓ late` stays near {LSS_REMOVAL_LT[LSS_LATE].mean():.2f} on average in removal, while the two anti arms fall to about a half ({num(min(LSS_REMOVAL_ANTI_LATE))} and {num(max(LSS_REMOVAL_ANTI_LATE))}), with their anti weight still above that of `base`. The late spill scout found that the anti arms lowered removal by the end, by about a quarter on one seed. So the two runs of a seed drifting apart here fits a change that by then has a real effect, though at eight seeds the loosening is not firm (below).

Along training, E3 nets removal against eight reference edits, where E1 and E2 net it against the median drop along the other 63 axes. Could the baseline, rather than the pairing, set E3 apart from E2? At the end of the run, under the 63-axis baseline, the three arms still correlate with `base` by {span(min(ARM_END["removal"]), max(ARM_END["removal"]))} in removal and {span(min(ARM_END["spill"]), max(ARM_END["spill"]))} in spill, above every correlation of E2 after the anchor forms.

Under that baseline the anti arms correlate with `base` at {num(ARM_END_ANTI[0])} and {num(ARM_END_ANTI[1])} in removal, against about a half under the reference edits, and at eight seeds those values cannot be told apart.

The task score starts at a correlation near 1 for every arm and for the change of GPU, and stays at about 0.9 or above to about epoch 250. After that it falls: least for the anti arms (about {min(BRANCH_TASK_LATE[ANTI_LATE], BRANCH_TASK_LATE[ANTI_HELD]):.2f} on average after epoch 288), more for `τ_ℓ late` ({BRANCH_TASK_LATE[LT_LATE]:.2f}), and most for the change of GPU ({BRANCH_TASK_LATE[GPU]:.2f}).

At the end, `base` on the A10G and on the L4 also correlate in removal and spill ({GPU_END["removal"]:.2f} and {GPU_END["spill"]:.2f}), and both clear the band at eight seeds.
<!-- REVIEW: was "only the second clears the band"; the band at eight seeds is 0.72 and both values are above it. -->


So from a stored state, pairing by seed helps on all three scores for at least a hundred and fifty epochs, under each change here and under a change of GPU. The branch passes the differences between seeds from the first 64 epochs on to both runs. Until late in the run, the changes after the branch add less than those differences.
"""

# %%

r"""
## Discussion

There is no one number for "soon after": what the runs share when they part seems to matter more than how long ago they parted, and in E1 the size of the change made no clear difference to the task score either. That fits the design rule, and puts the weight on the stored state rather than on the time since the branch.

If it holds beyond these experiments, arms that branch from a stored state after the anchor forms can be paired for most of a run. Conditions that share only an initialization gain only weakly from pairing on removal, and not detectably on spill, at any checkpoint we have.

The evidence for the long pairing rests on one stored state at one epoch and on eight seeds, so how far it generalizes is not settled. A stored state taken earlier, before the seeds have parted, may behave more like a shared initialization.

At epoch 400, netting the D2.2 task change against the control at each seed neither narrows nor widens its spread much (E2), so a gate set against the seed mean of the control would see about the same spread.

α at the example answers moves so much from record to record that a single record is a poor measure of a run (E1). The loss-rate call averages it over six records, and this result supports that choice.
"""

# %%

r"""
## References

[fixed confirmation](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` at the fresh seeds of the anneal confirmation. Removal under the two recipes correlated about 0.2 across eight seeds at epoch 400, and the two runs of a seed were up to 0.43 apart in removal at epoch 32.

[anneal confirmation](/docs/m2/anneal-confirm/report.py)
:   The pool over slices 1 to 3 with the slice temperature annealing from 0.1 to 0.025, at seeds 712 to 719 to the end and at forty seeds to epoch 64.

[loss-rate screen](/docs/m2/loss-rate/report.py)
:   `falloff` and `step, soft τ` at the same 32 fresh seeds to epoch 64, called kept or lost from α at the example answers over epochs 44 to 64.

[late spill scout](/docs/m2/late-spill-scout/report.py)
:   Four ways of carrying `τ_ℓ fixed` on from its stored epoch-64 states; holding the anti weight high lowered removal, by about a quarter on one seed.

[soft pool scout](/docs/m2/soft-pool-scout/report.py)
:   The pool over slices 1 to 3 at seeds 704 to 711 on L4, whose epoch-64 states the late spill scout carried on from.

[reference-baseline experiment](/docs/m2/reference-baseline/report.py)
:   Removal and spill net of the median drop along the other coordinate axes of the run itself, scored on the stored runs of `falloff`, `step, soft τ`, and the control at seeds 704 to 711.

[low-runs report](/docs/m2/low-runs/report.py)
:   Twin runs, the same in all but the GPU arithmetic, are as far apart in their weights as unrelated runs by epoch 16, while their task loss agrees to a hundredth for the whole run.

[spill-timing](/docs/m2/spill-timing/report.py)
:   Spill along training, net of the median of eight reference edits.
"""
