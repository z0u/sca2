# ruff: noqa: B018
# title: When the spill appears

# An exploratory report: `pool 1–3, soft τ` trained again at its eight soft-pool seeds, with the full edit, a reference edit
# edit and an RGB probe scored at every record, to see when along training the spill of the edit appears and whether
# an early measurement sorts the seeds as the end does. `experiment.py` beside this script trains and scores the runs.
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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


TRAJ = fetch_json(ex.TRAJ_REF)
DOSE = by_label(fetch_json(ex.DOSE_REF))
STORED = fetch_json(ex.STORED_REF)
SCORED = by_label(STORED)
# The stored full runs of this condition (soft-pool), and ex-2.2.23's controls, dose-scored by split-anti-seeds.
OLD_DOSE = by_label(fetch_json(ex.sp.DOSE_REF))
FALLOFF_DOSE = by_label(fetch_json(ex.sas.DOSE_REF))  # split-anti-seeds' `falloff` runs at 704 to 711
CONTROL_DOSE = {k: v for k, v in FALLOFF_DOSE.items() if k.startswith("control-")}
FINISHED_DOSE = by_label(fetch_json(ex.lr.DOSE_REF))  # the loss-rate runs finished to 400 epochs
DESIGN = STORED["design"]

OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
CRITERION = 0.02  # The selectivity criterion of ex-2.2.21: the largest net drop on another op the edit may cause.
SL = 3  # The slice whose anti weight the strips show, and where the probe is shown: pulled by every recipe.
SLICE_AT = DESIGN["probe_slices"].index(SL)
POSITIONS: list[str] = DESIGN["probe_positions"]
ANSWER_POS = [i for i, p in enumerate(POSITIONS) if p.startswith("c")]  # the example answers
STOP_EPOCH = 64  # where the loss-rate screen stopped its runs


def spread(ys: list[float], gap: float) -> list[float]:
    """*ys* nudged apart, in order, so neighbors are at least *gap* apart: where a column of line labels goes."""
    order = np.argsort(ys)
    out = np.asarray(ys, float).copy()
    for a, b in zip(order[:-1], order[1:], strict=True):
        out[b] = max(out[b], out[a] + gap)
    return out.tolist()


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


OVER_INK = light_dark("#c0392b", "#ff8a76")
WITHIN_INK = light_dark("#0b3c73", "#9fcbff")
CTL_INK = light_dark("#777", "#aaa")
ARM_INK = {"falloff": light_dark("#2a9d8f", "#7fd8c8"), "step-soft": light_dark("#e08e0b", "#f5b950")}
GHOST = light_dark("#bbb", "#555")
RULE = light_dark("#333", "#ddd")
ZERO = light_dark("#aaa", "#555")

# --- Spill, as the earlier reports score it and as the reference edit scores it --------------------------------------------


def control_spill(dose: dict, controls: list[dict]) -> tuple[float, str]:
    """The largest net drop on another op at any dose, net of the mean of *controls* (one control at a run's own
    seed), and the op it falls on.
    """
    ctl = np.mean([[e["drop"] for e in c["edits"]] for c in controls], axis=0)
    net = (np.array([e["drop"] for e in dose["edits"]]) - ctl)[:, OTHER]
    return float(net.max()), OPS[OTHER[int(np.unravel_index(net.argmax(), net.shape)[1])]]


def reference_net(clean, edited, ref) -> np.ndarray:
    """The drop on each op under the edit, less the median drop under the reference edits: the median reference score less the
    edit score. The last axis is the op; *ref* has the draws on the axis before it.
    """
    return np.median(np.asarray(ref), axis=-2) - np.asarray(edited)


def reference_spill(net: np.ndarray) -> np.ndarray:
    """The largest net drop on another op, on the last axis."""
    return net[..., OTHER].max(axis=-1)


def traj_of(seed: int) -> dict:
    return TRAJ[ex.label_of(seed)]["traj"]


EPOCH = np.asarray(traj_of(SEEDS[0])["epoch"])
NET = {s: reference_net(traj_of(s)["es_clean"], traj_of(s)["es_edit"], traj_of(s)["es_sham"]) for s in SEEDS}
TRACK = {s: reference_spill(NET[s]) for s in SEEDS}  # the reference-netted spill at every record
WORST_OP = {s: [OPS[OTHER[i]] for i in NET[s][:, OTHER].argmax(axis=1)] for s in SEEDS}
END = {s: control_spill(DOSE[ex.label_of(s)], [CONTROL_DOSE[f"control-s{s}"]]) for s in SEEDS}
OLD_END = {s: control_spill(OLD_DOSE[ex.label_of(s)], [CONTROL_DOSE[f"control-s{s}"]])[0] for s in SEEDS}
END_SPILL = np.array([END[s][0] for s in SEEDS])
OVER = [s for s in SEEDS if END[s][0] > CRITERION]
WITHIN = [s for s in SEEDS if s not in OVER]
REPRO_MAX = max(abs(END[s][0] - OLD_END[s]) for s in SEEDS)
SAMPLE_SE = float(np.mean([np.asarray(traj_of(s)["es_net_se"])[:, OTHER].mean() for s in SEEDS]))
SCHEDULE = {
    "epoch": EPOCH.tolist(),
    "weight": traj_of(SEEDS[0])["weight"],
    "tau": np.asarray(traj_of(SEEDS[0])["tau"])[:, 0].tolist(),
    "anti": np.asarray(traj_of(SEEDS[0])["anti_weight"])[:, SL].tolist(),
}


def at(epoch: float) -> int:
    return int(np.flatnonzero(np.isclose(EPOCH, epoch))[0])


def window(seed: int, lo: float, hi: float) -> float:
    """The reference-netted spill averaged over the records from *lo* to *hi* (each epoch once)."""
    keep = (EPOCH >= lo - 1e-6) & (EPOCH <= hi + 1e-6)
    _, first = np.unique(np.round(EPOCH[keep], 3), return_index=True)
    return float(TRACK[seed][keep][first].mean())


WINDOWS = [(24, 40), (48, 80), (100, 160), (200, 240), (360, 400)]
WIN = {w: np.array([window(s, *w) for s in SEEDS]) for w in WINDOWS}
WIN_RHO = {w: float(stats.spearmanr(WIN[w], END_SPILL)[0]) for w in WINDOWS}
LATE_SD = float(np.mean([TRACK[s][EPOCH >= 300].std() for s in SEEDS]))

# The reference edit beside the control at the end of training, on the eight runs.
END_REF = np.array([TRACK[s][-1] for s in SEEDS])
END_RHO = float(stats.spearmanr(END_REF, END_SPILL)[0])

# The same scoring, once, on the controls and on the loss-rate screen's epoch-64 checkpoints.


def scored_spill(label: str) -> float:
    r = SCORED[label]
    return float(reference_spill(reference_net(r["es_clean"], r["es_edit"], r["es_sham"])))


CTL_REF = {s: scored_spill(f"control-s{s}") for s in SEEDS}
EARLY = {
    arm: {lab: scored_spill(lab) for lab in SCORED if lab.startswith(f"{arm}-s") and lab.endswith(f"-e{STOP_EPOCH}")}
    for arm in ("falloff", "step-soft")
}
N_EARLY = sum(len(v) for v in EARLY.values())
N_EARLY_OVER = sum(v > CRITERION for a in EARLY.values() for v in a.values())
MEAN_CONTROL = list(CONTROL_DOSE.values())


def known_end(arm: str, seed: int) -> float | None:
    """The end spill of an epoch-64 checkpoint whose run was trained to the end: the stored full run at the repro
    seeds (net of its control), or the loss-rate run finished from the band (net of the mean control).
    """
    if seed in SEEDS:
        full = (FALLOFF_DOSE if arm == "falloff" else OLD_DOSE)[f"{arm}-s{seed}"]
        return control_spill(full, [CONTROL_DOSE[f"control-s{seed}"]])[0]
    if f"{arm}-s{seed}" in FINISHED_DOSE:
        return control_spill(FINISHED_DOSE[f"{arm}-s{seed}"], MEAN_CONTROL)[0]
    return None


KNOWN: list[dict[str, Any]] = []  # the epoch-64 checkpoints whose end spill is known
for arm, runs in EARLY.items():
    for lab, v in runs.items():
        seed = int(lab.rsplit("-s", 1)[1].split("-")[0])
        if (e := known_end(arm, seed)) is not None:
            KNOWN.append({"arm": arm, "seed": seed, "early": v, "end": e})
assert len(KNOWN) == 11

# The probe: held-out R² by record, context group (other ops, then the anchored op), state (whole, without e₁, e₁
# alone), slice and position.
PROBE = {s: np.asarray(traj_of(s)["probe_r2"]) for s in SEEDS}
OTHER_G, ANCHORED_G = 0, 1
WHOLE, WITHOUT, ALONE = 0, 1, 2


def probe_track(seed: int, group: int, state: int) -> np.ndarray:
    """R² at the example answers at slice `SL`, averaged over the three answers, at every record."""
    return PROBE[seed][:, group, state, SLICE_AT][:, ANSWER_POS].mean(axis=1)


GAP_MAX = max(
    float(np.max(PROBE[s][:, g, WHOLE] - PROBE[s][:, g, WITHOUT])) for s in SEEDS for g in (OTHER_G, ANCHORED_G)
)
ALONE_OTHER_END = float(np.mean([probe_track(s, OTHER_G, ALONE)[-1] for s in SEEDS]))
ALONE_ANCHORED_END = float(np.mean([probe_track(s, ANCHORED_G, ALONE)[-1] for s in SEEDS]))
FIT_RHO = {
    e: float(stats.spearmanr([probe_track(s, OTHER_G, WHOLE)[at(e)] for s in SEEDS], END_SPILL)[0])
    for e in (32, 64, 200, 400)
}
ALONE_RHO = {
    e: float(stats.spearmanr([probe_track(s, OTHER_G, ALONE)[at(e)] for s in SEEDS], END_SPILL)[0])
    for e in (32, 64, 200, 400)
}

# --- E1: the reference edit beside the control ----------------------------------------------------------------------------


def reference_figure() -> str:
    data = {
        "runs": [
            {"seed": s, "control": END[s][0], "reference": float(END_REF[i]), "over": s in OVER}
            for i, s in enumerate(SEEDS)
        ],
        "controls": list(CTL_REF.values()),
        "early": {arm: list(v.values()) for arm, v in EARLY.items()},
        "criterion": CRITERION,
    }
    alt = """
        Left: eight points, the reference-netted spill against the control-netted spill at the end of training, lying close
        to the diagonal and a little below it; seed 711 sits far up and right of the rest. Right: three columns of
        dots. The controls spread from about zero to 0.05. The epoch-64 checkpoints of falloff and step, soft τ sit
        mostly just below zero, with one step, soft τ dot just over the criterion.
    """
    return reference_draw(data, alt)


@memo
def reference_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="spill-timing-reference",
        alt_text=alt_text,
        caption=f"""
            **The reference edit beside the control.** Left: one point per rerun at the end of training. Vertical: the spill
            at the full dose, netted against the median of the eight reference edits. Horizontal: the spill as earlier
            reports score it, netted against the control at the same seed. Red where the run ends over the
            criterion, blue where it ends within it. The dashed lines are the criterion, and the dotted line is
            equality. Right: the reference-netted spill of the e₁ edit on other models, one small mark each, with the
            mean as a large mark and the range as a gray bar: the controls of ex-2.2.23 at seeds {SEEDS[0]} to
            {SEEDS[-1]}, and the checkpoints of the loss-rate screen at epoch {STOP_EPOCH}.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 3.0), layout="constrained", width_ratios=[1, 1.1])
        lim = (-0.03, 0.08)
        ax.plot(lim, lim, color=ZERO, lw=0.6, ls=":", zorder=0)
        for f in (ax.axhline, ax.axvline):
            f(data["criterion"], color=RULE, lw=0.6, ls="--", zorder=0)
        for r in data["runs"]:
            ink = OVER_INK if r["over"] else WITHIN_INK
            ax.plot(r["control"], r["reference"], "o", color=ink, ms=4.5, zorder=2)
            ax.annotate(str(r["seed"]), (r["control"], r["reference"]), xytext=(3, 2), textcoords="offset points",
                        fontsize=6, color=RULE)  # fmt: skip
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("spill, net of control", fontsize=8)
        ax.set_ylabel("spill, net of reference", fontsize=8)
        cols = [("controls", data["controls"], CTL_INK, "s")] + [
            (f"{name}\nepoch {STOP_EPOCH}", data["early"][arm], ARM_INK[arm], m)
            for arm, name, m in (("falloff", "falloff", "D"), ("step-soft", "step, soft τ", "v"))
        ]
        rng = np.random.default_rng(0)
        for i, (_, vals, ink, mark) in enumerate(cols):
            v = np.asarray(vals)
            bx.plot([i, i], [v.min(), v.max()], color=GHOST, lw=4, alpha=0.5, solid_capstyle="round", zorder=0)
            bx.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=ink, alpha=0.6, marker=mark, lw=0,
                       zorder=1)  # fmt: skip
            bx.plot(i, v.mean(), marker=mark, color=ink, ms=7, mec=RULE, mew=0.6, zorder=2)
        bx.axhline(data["criterion"], color=RULE, lw=0.6, ls="--", zorder=0)
        bx.axhline(0, color=ZERO, lw=0.5, zorder=0)
        bx.set_xticks(range(len(cols)), [c[0] for c in cols], fontsize=7)
        bx.set_xlim(-0.6, len(cols) - 0.4)
        bx.set_ylim(lim)
        bx.set_ylabel("spill, net of reference", fontsize=8)
        for a_ in (ax, bx):
            a_.tick_params(labelsize=7)
        return fig

    return _plot()


def reference_table() -> str:
    rows = [
        [
            str(s),
            num(OLD_END[s]),
            num(END[s][0]),
            END[s][1],
            num(float(END_REF[i])),
            WORST_OP[s][-1],
            num(CTL_REF[s]),
        ]
        for i, s in enumerate(SEEDS)
    ]
    return table_html(
        ["seed", "stored ↓", "rerun ↓", "op", "net of reference ↓", "op", "control, net of reference"],
        rows,
        f"""
        **Spill at the end of training, by seed.** Stored and rerun: the stored run of soft-pool and the rerun, each
        netted against the control at the same seed (the largest net drop on another op at any dose), then the op
        that drop falls on. Net of reference: the rerun netted against the median reference edit at the full dose, then its op.
        Last column: the e₁ edit on the control, netted against the median reference edit on that control. The criterion
        is {CRITERION:g}.
        """,
    )


# --- E2: spill along training -----------------------------------------------------------------------------------


def track_figure() -> str:
    data = {
        "runs": [{"seed": s, "spill": TRACK[s].tolist(), "over": s in OVER} for s in SEEDS],
        "schedule": SCHEDULE,
        "criterion": CRITERION,
        "stop": STOP_EPOCH,
    }
    alt = """
        Three red lines (seeds 707, 708 and 711) jump over the criterion between epochs 24 and 48 and swing widely
        until about epoch 100, 708 then falling back to about 0.01; red 710 and blue 705 rise slowly to about 0.02
        after epoch 100 to 150, and the other blue lines stay near zero. The strips show the warm-up ending by epoch
        40 and the anti weight annealing slowly to near zero by the end.
    """
    return track_draw(data, alt)


@memo
def track_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="spill-timing-track",
        alt_text=alt_text,
        caption=f"""
            **Spill along training.** The largest drop on another op under the full edit, less the median drop under
            the eight reference edits, at every record (every four epochs), one line per seed: red for the seeds whose
            run ends over the criterion, blue for those within it. The dotted line marks where the
            loss-rate screen stopped. The strips underneath are the anchor weight $\\lambda_a$, the position pool temperature
            $\\tau_p$ (log scale), and the anti weight at slice {SL}, $\\lambda_{{\\bar s}}$.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (ax, wax, tax, nax) = plt.subplots(
            4, 1, figsize=(6.4, 4.6), layout="constrained", sharex=True, height_ratios=[3, 0.7, 0.7, 0.7]
        )
        ep = data["schedule"]["epoch"]
        ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
        ax.axhline(data["criterion"], color=RULE, lw=0.7, ls="--", zorder=0)
        for a_ in (ax, wax, tax, nax):
            a_.axvline(data["stop"], color=ZERO, lw=0.6, ls=":", zorder=0)
        for r in data["runs"]:
            ink, mark = (OVER_INK, "o") if r["over"] else (WITHIN_INK, "s")
            ax.plot(ep, r["spill"], color=ink, lw=0.8, alpha=0.85, marker=mark, ms=1.6, zorder=2)
        ys = spread([r["spill"][-1] for r in data["runs"]], 0.0045)
        for r, y in zip(data["runs"], ys, strict=True):
            ink = OVER_INK if r["over"] else WITHIN_INK
            ax.annotate(str(r["seed"]), (ep[-1], y), xytext=(3, 0), textcoords="offset points",
                        fontsize=6, color=ink, va="center")  # fmt: skip
        ax.set_ylim(-0.03, 0.1)
        ax.set_ylabel("spill, net of reference", fontsize=8)
        sch = data["schedule"]
        wax.plot(ep, sch["weight"], color=RULE, lw=1.0)
        wax.set_ylim(0, 0.12)
        wax.set_yticks([0, 0.1])
        wax.set_ylabel("$\\lambda_a$", fontsize=8)
        tax.plot(ep, sch["tau"], color=RULE, lw=1.0)
        tax.set_yscale("log")
        tax.set_ylim(0.07, 1.4)
        tax.set_yticks([0.1, 1], ["0.1", "1"])
        tax.minorticks_off()
        tax.set_ylabel("$\\tau_p$", fontsize=8)
        nax.plot(ep, sch["anti"], color=RULE, lw=1.0)
        nax.set_ylim(0, 0.28)
        nax.set_yticks([0, 0.2])
        nax.set_ylabel("$\\lambda_{\\bar s}$", fontsize=8)
        nax.set_xlim(0, 420)
        nax.set_xlabel("epoch", fontsize=8)
        for a_ in (ax, wax, tax, nax):
            a_.tick_params(labelsize=7)
        return fig

    return _plot()


def window_table() -> str:
    rows = [
        [str(s), *(num(float(WIN[w][i])) for w in WINDOWS), num(END[s][0]), WORST_OP[s][at(64)], WORST_OP[s][-1]]
        for i, s in enumerate(SEEDS)
    ]
    rows.append(["rank correlation with the end", *(f"{WIN_RHO[w]:.2f}" for w in WINDOWS), "", "", ""])
    return table_html(
        ["seed", *(f"epochs {a}–{b}" for a, b in WINDOWS), "end, net of control ↓", "op at 64", "op at 400"],
        rows,
        f"""
        **Spill along training, by window.** The reference-netted spill averaged over the records in each window, the
        spill at the end as earlier reports score it (net of the control), and the op the reference-netted spill falls
        on at epochs {STOP_EPOCH} and 400. The last line is the rank (Spearman) correlation of each window with the
        end spill over the eight seeds.
        """,
    )


# --- E3: color on e₁ at the other ops ----------------------------------------------------------------------------


def probe_figure() -> str:
    data = {
        "epoch": EPOCH.tolist(),
        "runs": [
            {
                "seed": s,
                "over": s in OVER,
                "alone_other": probe_track(s, OTHER_G, ALONE).tolist(),
                "alone_anchored": probe_track(s, ANCHORED_G, ALONE).tolist(),
                "whole_other": probe_track(s, OTHER_G, WHOLE).tolist(),
            }
            for s in SEEDS
        ],
        "stop": STOP_EPOCH,
    }
    alt = """
        Left: from e₁ alone, R² on the other ops drops from near 0.1 to about 0.01 by epoch 30, red and blue seeds
        intermixed, while on difference (ghosted) it stays a little higher, about 0.02 to 0.05. Right: from the
        whole state, R² falls slowly from about 0.94 to 0.85 on most seeds, and further, to about 0.75 and 0.77, on
        706 (blue, after about epoch 60) and 711 (red, from the start).
    """
    return probe_draw(data, alt)


@memo
def probe_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="spill-timing-probe",
        alt_text=alt_text,
        caption=f"""
            **The color probe at the example answers**, slice {SL}: held-out $R^2$ of a ridge fit from the state to
            the RGB of the answer color, averaged over the three example answers. One line per seed, red for the
            seeds whose run ends over the criterion and blue for those within it. Left: from the e₁ coordinate
            alone, on the contexts of the other ops (solid) and on `{ex.ANCHORED_OP}` contexts (ghosted). Right: from
            the whole state, on the contexts of the other ops. The dotted line marks where the loss-rate screen
            stopped.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.8), layout="constrained", sharex=True)
        ep = data["epoch"]
        for r in data["runs"]:
            ink = OVER_INK if r["over"] else WITHIN_INK
            ax.plot(ep, r["alone_anchored"], color=GHOST, lw=0.6, zorder=1)
            ax.plot(ep, r["alone_other"], color=ink, lw=0.8, alpha=0.85, zorder=2)
            bx.plot(ep, r["whole_other"], color=ink, lw=0.8, alpha=0.85, zorder=2)
        ys = spread([r["whole_other"][-1] for r in data["runs"]], 0.011)
        for r, y in zip(data["runs"], ys, strict=True):
            bx.annotate(str(r["seed"]), (ep[-1], y), xytext=(3, 0), textcoords="offset points", fontsize=6,
                        color=OVER_INK if r["over"] else WITHIN_INK, va="center")  # fmt: skip
        for a_ in (ax, bx):
            a_.axvline(data["stop"], color=ZERO, lw=0.6, ls=":", zorder=0)
            a_.set_xlabel("epoch", fontsize=8)
            a_.tick_params(labelsize=7)
            a_.set_xlim(0, 430)
        ax.set_ylim(0, 0.12)
        ax.set_ylabel("$R^2$, e₁ alone", fontsize=8)
        bx.set_ylim(0.7, 1.0)
        bx.set_ylabel("$R^2$, whole state", fontsize=8)
        return fig

    return _plot()


# %%

rf"""
# When the spill appears

/// tip |
<!-- lede -->
We trained `pool 1–3, soft τ` again at its eight seeds and applied the edit every four epochs along training, to see when its spill onto other ops appears. On three of the four seeds that end over the spill criterion it appears early in training, and on the fourth only late. A color probe on the other ops showed nothing that tracks it.
///

`pool 1–3, soft τ` from [soft-pool](/docs/m2/soft-pool/report.py) kept the anchor on all eight of its seeds, but only four of its edits stayed within the spill criterion. A cheap early test for spill would let it, or arms tuned from it, run at many seeds, as [the loss-rate screen](/docs/m2/loss-rate/report.py) did for losing the anchor. But nothing in the stored trajectories up to epoch {STOP_EPOCH} predicted the end spill. The stored runs keep only their end checkpoints, so we couldn't tell when the spill appears.

So we trained the condition again at its seeds ({SEEDS[0]} to {SEEDS[-1]}), with three measurements at every record:[^record] the edit itself, a reference edit along another direction as its baseline, and a probe for the colors at the other ops. The reference edit replaces the control as the baseline, since there is no control trajectory to net against. We also scored the edit and the reference edit on the controls, and on the epoch-{STOP_EPOCH} checkpoints the loss-rate screen left.

[^record]: Record: a point along training where the measurements are taken, every four epochs.
"""

# %%

rf"""
## Observations

- [The reference edit beside the control (E1)](#the-reference-edit-beside-the-control-e1): the reruns match the stored runs. At the end of training, netting the edit against the reference edit sorts the eight seeds much as netting it against the control does. On the controls themselves, the e₁ edit costs some ops a few hundredths more than the reference edit does and others less, so the control adds a seed-to-seed term of its own. An edit at epoch 64 on two low-spill conditions finds no spill that isn't there at the end, but misses some that is.
- [Spill along training (E2)](#spill-along-training-e2): spill appears in two ways. On three seeds it appears during the rise, as the pool sharpens and the anchor weight reaches its plateau, and on two of them it stays. On two seeds it grows after epoch 100 on `value-hsv`, while the anti weight anneals. A window around epoch {STOP_EPOCH} sorts the seeds roughly as the end does, missing the late one; the full order settles only after epoch 200.
- [Color on e₁ at the other ops (E3)](#color-on-e1-at-the-other-ops-e3): e₁ holds almost no color information at the other ops, from about epoch 30 on, on every seed, and removing it barely changes how well the color can be read from the rest of the state. The fit from the whole state doesn't follow the spill either.

## Scope

This is an exploratory study, with no preregistration and no gate. It covers one condition, `pool 1–3, soft τ`, at eight seeds, trained exactly as in soft-pool. That condition pools the pull over slices 1 to 3, and its position and slice pool temperatures sharpen from soft to sharp over the 40-epoch warm-up. Training is deterministic, and the reruns reproduce the stored runs: the end spill matches to the last digit on every seed (largest difference {REPRO_MAX:.1g}).

Eight seeds is few for correlations: a rank correlation of 0.6 over eight seeds comes up by chance about one time in ten. The seeds also split unevenly. One seed (711) spills far more than the rest, and the other three over the criterion are only just over it. We looked at several measurements, windows and slices after the runs came in, so we take a single strong correlation here as a lead only.

The epoch-{STOP_EPOCH} checkpoints are those of `falloff` and `step, soft τ` from the loss-rate screen, {N_EARLY} in all, and these two conditions spill little. Eleven of the checkpoints belong to runs trained to the end, so their end spill is known, and three of those eleven end at or over the criterion. So the checkpoints can show whether an early edit finds spill that isn't there at the end, and whether it misses spill that is.

## The measurements

- Spill, net of control (at the end, as earlier reports score it): the largest drop on another op under the edit, less the drop on that op under the same edit on the control at the same seed, taken over the four doses. The edit projects e₁ out at every position and slice. A run is within the criterion when its spill is at most {CRITERION:g}.
- Spill, net of reference (along training): the largest drop on another op under the full edit, less the median drop on that op under eight reference edits. Each reference edit projects out one other basis direction (e₂, e₃, …), chosen at random, and the same eight are used throughout. We use the median because on some seeds one of the eight directions costs every op far more than the rest: Adam gives the residual stream a privileged basis, with the separator tokens on one coordinate that holds about half the stream's energy and `=` on another, and our draw includes the separator coordinate of three seeds (704, 705 and 708). A label-free rule for later experiments is to draw the reference axes from outside the stream's highest-energy coordinates. Each drop is measured in expected exact match over the whole held-out set, so the noise from the choice of contexts is small (a standard error of about {SAMPLE_SE:.3f} per op).
- The probe: a ridge regression (a linear fit with a small penalty on the weights) from the state at a position to the RGB of the color there. It is fitted on half of the held-out contexts and scored on the other half as $R^2$ (the share of the variance of the color it predicts). It is fitted at every color position and at slices 2 to 4, on the contexts of the other ops and on `{ex.ANCHORED_OP}` contexts. Each fit uses one of three versions of the state: the whole state, the state without its e₁ coordinate, or the e₁ coordinate alone.
"""

# %%

# REVIEW: E1 now also says the early edit misses the spill three of the eleven known-end checkpoints end with
# (step-soft 705, falloff 714 and 704, all under 0.01 at epoch 64), from the claims review. Verify: KNOWN.
rf"""
## The reference edit beside the control (E1)

Can the reference edit stand in for the control as the baseline the spill is netted against? At the end of training we have both, so the figure (left) and the table compare them on the eight reruns. We also apply the edit and the reference edit to the controls and to the epoch-{STOP_EPOCH} checkpoints, where the edit should have little to act on (right).

We first tried a random direction orthogonal to e₁ as the reference edit, as planned. On a control it cost every op far more than the e₁ edit does, so a random direction is no baseline for e₁. A single other basis direction behaves much more like e₁ on a control, though, so we used eight of those as the reference edits. The gap comes from the few high-energy coordinates: a random direction touches all of them, and a random direction kept off them costs about what an ordinary axis does.

{reference_figure()}

{reference_table()}

The two scores rank the eight seeds alike (rank correlation {END_RHO:.2f}), though they often name different ops. The reference-netted score runs a little lower, so the criterion would need restating for it. The seeds over the criterion stay at or over it, apart from 708, which falls under it.

On the controls, the e₁ edit costs some ops up to a few hundredths more than the reference edit does, and others as much less, differently on each seed (the table shows only the largest). So e₁ on an unanchored model isn't quite a typical direction, and netting against the control at the same seed adds that seed-to-seed term to the spill score. That may be part of why quiet runs scored as high as 0.018 in earlier reports.

On the epoch-{STOP_EPOCH} checkpoints of the two low-spill conditions, the edit costs the other ops a little less than the reference edit, as it should when the anti term keeps them off e₁. Only {N_EARLY_OVER} of the {N_EARLY} goes over the criterion, and on the eleven with a known end spill the early score stays below {max(k["early"] for k in KNOWN):.2f}. So an early edit doesn't seem to invent spill. But it also misses the spill that three of the eleven end with (at or over the criterion). That is the late kind of spill from E2, seen here in two other conditions.
"""

# %%

# REVIEW: the early rise was dated "between epochs 24 and 32" on all three seeds; the record shows 708 first over
# the criterion at epoch 24 and 711 at 32, but 707 at 44 (it sits at −0.012 at epoch 40 and 0.097 at 48). The lede
# and this section now say 707 rises at about epoch 44, just after the warm-up. Verify: TRACK[707] around at(40)
# to at(48).
rf"""
## Spill along training (E2)

When does the spill appear, and from when does it sort the seeds as the end does? The figure shows the reference-netted spill at every record, and the table its mean over a few windows.

{track_figure()}

{window_table()}

Spill appears in two ways. The early kind rises on seeds 708 and 711 between epochs 24 and 32, as the pool temperature reaches its sharp value and the anchor weight its plateau, and on 707 at about epoch 44, just after. It falls mostly on `darken` and `hue-hsv`, and it stays up past where the loss-rate screen stopped. After that the three seeds differ. On 711 it stays to the end, on `darken`. On 707 it moves to `mix` and settles near the criterion, and on 708 it falls back to about half.

The late kind grows on seeds 705 and 710, from about epoch 120 and 150 respectively, on `value-hsv`, while the anti weight on slice {SL} falls through its slow anneal. It ends just under the criterion on 705 and just over it on 710.

From record to record the score swings by a few hundredths during the rise, and by much less late in training (a standard deviation of about {LATE_SD:.3f} after epoch 300). Averaging over a window damps this. Over epochs 48 to 80 the window sorts the seeds roughly as the end does: 707, 708 and 711 are clearly up, 710 is lowest of all, and 705 is among the runs within the criterion. The order settles only after about epoch 200 (last line of the table).

So a screen at epoch {STOP_EPOCH} would catch the early kind of spill and miss the late kind. On these seeds it would catch three of the four runs over the criterion, though two of those three end near it.
"""

# %%

rf"""
## Color on e₁ at the other ops (E3)

Does e₁ hold color information at the positions of the other ops, which the edit would take away, and does that track the spill? This is a geometric measurement that doesn't apply the edit. The figure shows the probe at the example answers, at slice {SL}.

{probe_figure()}

Removing e₁ barely changes how well the color can be read from the state: the largest difference between the whole state and the state without e₁ is {GAP_MAX:.3f} in $R^2$, at any position, slice or record, on the other ops and on `{ex.ANCHORED_OP}` alike. The color is spread over many coordinates, so one coordinate more or less makes little difference to a linear readout.

On its own, e₁ holds little color at the example answers of the other ops. After the first few epochs its $R^2$ is about {ALONE_OTHER_END:.2f}, with the seeds that spill and those that don't intermixed (rank correlation with the end spill {num(ALONE_RHO[64], ".2f")} at epoch {STOP_EPOCH}). On `{ex.ANCHORED_OP}` contexts it holds more, about {ALONE_ANCHORED_END:.2f}, which is still little: the anchor puts redness on e₁, and RGB is more than redness.

The whole-state fit on the other ops falls slowly over training, and on two seeds it falls further: from the start on 711, which spills most, and after about epoch 60 on 706, which doesn't spill. So the fit doesn't follow the spill (rank correlation {num(FIT_RHO[64], ".2f")} at epoch {STOP_EPOCH}, {num(FIT_RHO[400], ".2f")} at the end).

So the probe doesn't see the spill. The edit takes away e₁, but what the other ops lose may be how later blocks read e₁, rather than color information stored there.
"""

# %%

r"""
## Discussion

The edit spills in two ways in this condition. The early kind appears during the rise, when the pool is still sharpening and the anchor weight reaches its plateau. That fits the reading of soft-pool that the soft pool lifts the other ops toward e₁ at that time. The late kind appears as the anti term begins its slow anneal, on `value-hsv`, which fits the anti term holding the other ops off e₁ until it lets go. Neither reading is tested here; each would need an arm that changes the schedule at that time.

A screen that edits a checkpoint around epoch 64 would see only the early kind (E2). On these seeds the late kind was the smaller, but it put one run over the criterion and another just under it, so such a screen would undercount. A screen late enough to see both, around epoch 200, would cost half a full run.

Netting against a reference edit works about as well as netting against a control at the end of training, and it needs no control run. The controls showed that e₁ isn't a neutral direction on an unanchored model either, so some of the noise in the earlier spill scores may come from the control rather than from the anchored run.
"""
