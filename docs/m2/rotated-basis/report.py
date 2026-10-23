# ruff: noqa: B018
# title: Anchoring on an axis the optimizer does not favor

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
from mini.vis import light_dark, themed

# --- The stored results ---------------------------------------------------------------------------------------


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


RESULTS = fetch_json(ex.RESULTS_REF)
TRAJ = fetch_json(ex.TRAJ_REF)
DOSE = {r["label"]: r for r in fetch_json(ex.DOSE_REF)["runs"]}
BASIS = {r["label"]: r for r in fetch_json(ex.BASIS_REF)["runs"]}
# The stored twins: split-anti-seeds' falloff runs and ex-2.2.23's controls, measured the same way.
STORED = {r["label"]: r for r in fetch_json(ex.sas.RESULTS_REF)["runs"]} | {
    r["label"]: r for r in fetch_json(ex.lean.RESULTS_REF)["runs"]
}
STORED_TRAJ = fetch_json(ex.sas.TRAJ_REF)
STORED_DOSE = {r["label"]: r for r in fetch_json(ex.sas.DOSE_REF)["runs"]}
RUNS = {r["label"]: r for r in RESULTS["runs"]}
DESIGN = RESULTS["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
GAMMAS: list[float] = DESIGN["dose_gammas"]
CRITERION = 0.02  # The selectivity criterion of ex-2.2.21: the largest net drop on another op the edit may cause.
ANSWERS = list(ex.lean.ANSWERS)  # The positions of the example answers in a context.
EVERY = ex.EVERY
C = 64  # The width of the residual stream.
SYNTAX = ["?", ",", "\n", "="]
SHOWN = {"?": "`?`", ",": "`,`", "\n": "⏎", "=": "`=`"}
FALLOFF, CONTROL = "falloff", "control"
NEW, OLD = "rotated", "stored"  # The new runs, and their stored twins.
SLICE_TICK = ["emb", "1", "2", "3", "4"]


def new_run(arm: str, seed: int) -> dict:
    return RUNS[ex.label_of(arm, seed)]


def old_run(arm: str, seed: int) -> dict:
    if arm == CONTROL:
        return STORED[ex.ex2223.label_of("control", ex.EPOCHS, seed)]
    return STORED[ex.sas.label_of(arm, seed)]


def new_dose(arm: str, seed: int) -> dict:
    return DOSE[ex.label_of(arm, seed)]


def old_dose(arm: str, seed: int) -> dict:
    return STORED_DOSE[f"control-s{seed}" if arm == CONTROL else ex.sas.label_of(arm, seed)]


def alpha_by_slice(which: str, seed: int, key: str = "alpha_anchored") -> np.ndarray:
    """α at the example answers at the end of training, per slice, for the new falloff run or its stored twin."""
    traj = TRAJ[ex.label_of(FALLOFF, seed)] if which == NEW else STORED_TRAJ[ex.sas.label_of(FALLOFF, seed)]
    return np.asarray(traj["traj"][key])[-1][:, ANSWERS].mean(axis=1)


def dose_curves(which: str, seed: int) -> dict[str, list[float]]:
    """Removal and spill at each dose (ex-2.2.22's scoring, at every position and slice), net of the control at the
    same seed, and the same without subtracting the control (raw), and what the control alone loses on the op.
    """
    get = new_dose if which == NEW else old_dose
    d, c = get(FALLOFF, seed), get(CONTROL, seed)
    span = d["clean"]["eem"][D] - d["null"]["eem"][D]
    keys = ("removal", "raw_removal", "control_drop", "control_other", "spill", "raw_spill")
    out: dict[str, list[float]] = {k: [] for k in keys}
    for e, ec in zip(d["edits"], c["edits"], strict=True):
        drop, cdrop = np.array(e["drop"]), np.array(ec["drop"])
        net = drop - cdrop
        out["removal"].append(float(net[D] / span))
        out["raw_removal"].append(float(drop[D] / span))
        out["control_drop"].append(float(cdrop[D] / span))
        out["control_other"].append(float(cdrop[OTHER].max()))
        out["spill"].append(float(net[OTHER].max()))
        out["raw_spill"].append(float(drop[OTHER].max()))
    return out


def landing(which: str, seed: int) -> float:
    return (new_dose if which == NEW else old_dose)(FALLOFF, seed)["landing"]


def op_change(which: str, seed: int) -> np.ndarray:
    """The task score of each op in the anchored run, less that of its control."""
    get = new_run if which == NEW else old_run
    return np.array(get(FALLOFF, seed)["clean"]) - np.array(get(CONTROL, seed)["clean"])


def spike_axis(label: str) -> int:
    """The rotated basis vector that holds the most energy at the last slice (code index)."""
    return int(np.argmax(BASIS[label]["energy_rotated"][-1]))


def cos_e1(seed: int, label: str) -> float:
    """The cosine between e₁ and that run's spike direction in the rotated basis."""
    return float(abs(ex.basis_of(seed, C)[0, spike_axis(label)]))


def mean_cost(label: str, key: str) -> np.ndarray:
    """The cost of projecting out each direction of *key*, averaged over the ops."""
    return np.array(BASIS[label][key]).mean(axis=1)


def token_cos(label: str, token: str, key: str = "emb_rotated_max") -> float:
    b = BASIS[label]
    return float(b[key][b["tokens"].index(token)])


def mean_range(values) -> tuple[float, float, float]:
    v = np.asarray(list(values), float)
    return float(v.mean()), float(v.min()), float(v.max())


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def fmt(values, spec: str = ".2f") -> str:
    half = 0.5 * 10 ** -int(spec[1])  # half the last printed digit, so a value that prints as zero has no sign
    m, lo, hi = (0.0 if abs(v) < half else v for v in mean_range(values))
    return f"{num(m, spec)} ({num(lo, spec)} to {num(hi, spec)})"


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    from mini.vis import figure_html

    def code(c: str) -> str:
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", c)

    head, rows = [code(h) for h in head], [[code(c) for c in r] for r in rows]
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


INK = {NEW: light_dark("#0b3c73", "#cfe6ff"), OLD: light_dark("#2a9d8f", "#7fd8c8")}
MARK = {NEW: "s", OLD: "D"}
GHOST = light_dark("#bbb", "#555")
AXIS_INK, ROT_INK, RAND_INK = light_dark("#555", "#bbb"), light_dark("#c0392b", "#ff8a76"), light_dark("#888", "#999")

# --- Summaries the prose quotes ---------------------------------------------------------------------------------

CURVES = {w: {s: dose_curves(w, s) for s in SEEDS} for w in (NEW, OLD)}
FULL = {w: {s: CURVES[w][s]["removal"][-1] for s in SEEDS} for w in (NEW, OLD)}
RAW = {w: {s: CURVES[w][s]["raw_removal"][-1] for s in SEEDS} for w in (NEW, OLD)}
CDROP = {w: {s: CURVES[w][s]["control_drop"][-1] for s in SEEDS} for w in (NEW, OLD)}
WORST_SPILL = {w: {s: max(CURVES[w][s]["spill"]) for s in SEEDS} for w in (NEW, OLD)}
N_WITHIN = {w: sum(v <= CRITERION for v in WORST_SPILL[w].values()) for w in (NEW, OLD)}
TASK = {w: {s: float(op_change(w, s).mean()) for s in SEEDS} for w in (NEW, OLD)}
LANDING = {w: {s: landing(w, s) for s in SEEDS} for w in (NEW, OLD)}
ALPHA = {w: {s: alpha_by_slice(w, s) for s in SEEDS} for w in (NEW, OLD)}
# The kept/lost rule of split-anti-seeds and the loss-rate screen: the full edit removes less than half of the op.
LOST_BY_NET = [s for s in SEEDS if FULL[NEW][s] < 0.5]
LOST_BY_RAW = [s for s in SEEDS if RAW[NEW][s] < 0.5]
# Where the control's own e₁ overlaps its syntax direction, the paired baseline drops on the anchored op too.
COS = {a: {s: cos_e1(s, ex.label_of(a, s)) for s in SEEDS} for a in (FALLOFF, CONTROL)}
E1_COST = {a: {s: float(mean_cost(ex.label_of(a, s), "cost_axes")[0]) for s in SEEDS} for a in (FALLOFF, CONTROL)}
CTRL_OTHER = {w: {s: CURVES[w][s]["control_other"][-1] for s in SEEDS} for w in (NEW, OLD)}
SHORTFALL = {s: FULL[OLD][s] - FULL[NEW][s] for s in SEEDS}
R_SHORTFALL = float(np.corrcoef([SHORTFALL[s] for s in SEEDS], [COS[CONTROL][s] for s in SEEDS])[0, 1])
OVERLAP_SEEDS = sorted(SEEDS, key=lambda s: -COS[CONTROL][s])[:2]
TOP_AXIS = {lab: float(np.max(b["energy_axes"][-1])) for lab, b in BASIS.items()}
TOP_ROT = {lab: float(np.max(b["energy_rotated"][-1])) for lab, b in BASIS.items()}
KURT = {lab: (float(np.mean(b["kurt_axes"])), float(np.mean(b["kurt_rotated"]))) for lab, b in BASIS.items()}
SYNTAX_ON_SPIKE = {
    lab: [token_cos(lab, t) for t in SYNTAX if b["emb_rotated_arg"][b["tokens"].index(t)] == spike_axis(lab)]
    for lab, b in BASIS.items()
}
MOVED = [s for s in SEEDS if spike_axis(ex.label_of(FALLOFF, s)) != spike_axis(ex.label_of(CONTROL, s))]


# %%

r"""
# Anchoring on an axis the optimizer does not favor

/// tip |
<!-- lede -->
Adam scales each coordinate of the residual stream separately, so the coordinate axes are a privileged basis: they are directions the optimizer makes cheap to hold, and e₁, the direction the anchor targets, is one of them. We trained the recipe of record and its control again with Adam's scaling applied in a random rotation of the coordinates, so that e₁ is an ordinary direction. The anchor held on every seed, and the edit removed the anchored op (`difference`) and spilled as little as before. What changed is the control. On some seeds e₁ now overlaps the direction where the un-anchored model puts its syntax tokens, so the edit costs the control something too, and the removal net of the control is low there.
///

The question comes from the memo of 2026-10-23 on whether the coordinate axes are special (in the project files, not on the site). On stored checkpoints, the memo found that every run parks its separators (`?`, `,` and ⏎) one-hot on a coordinate of their own and `=` on a second, and that a coordinate axis is cheaper to project out than a random direction. In short training runs, Adam built this basis, and Adam in a rotated basis built the same thing on the rotated axes.

The anchoring design had taken e₁ for an arbitrary direction. This report trains sixteen new runs, with the basis of the optimizer rotated and nothing else changed: `falloff` (the recipe of record of [split-anti-seeds](../split-anti-seeds/report.py)) and the un-anchored control, at the eight seeds where both are stored.
"""

# %%

rf"""
## Observations

- [Does the anchor hold? (E1)](#does-the-anchor-hold-e1): on every seed. The alignment at the example answers rises with depth as in the stored runs, and the profiles of the new and stored runs lie on top of each other.
- [Removal, spill and landing (E2)](#removal-spill-and-landing-e2): on every seed the edit removes nearly all of the op and spills within the criterion, as before. The removal net of the control falls short on {len(LOST_BY_NET)} seeds, and all of the shortfall comes from the control: where the rotation puts e₁ near the syntax direction of the control, the edit removes part of the op from the control as well.
- [Where the privileged basis went (E3)](#where-the-privileged-basis-went-e3): to the rotated axes. The energy spectrum is flat along the coordinate axes and spiked along one rotated axis, which holds the syntax tokens. In a control, e₁ costs what an ordinary direction costs, and more where it overlaps that spike.

## Scope

This is an exploratory study, with no preregistration and no gate. It covers `falloff` and the control at model seeds 704 to 711, trained for 400 epochs with Adam's per-coordinate scaling applied in a random orthogonal rotation of the residual coordinates (one rotation per seed, drawn from the model seed).

Each new run pairs with a stored twin at the same seed (the `falloff` run from split-anti-seeds or the 400-epoch control from ex-2.2.23). The pair shares its initialization, batches and label draws, and differs only in the basis of the optimizer. The model, schedules, anchoring code, measurements and dose scoring are those of split-anti-seeds, unchanged.

A difference between a new run and its twin is compared with the seed range of the stored arm. Split-anti-seeds found `falloff` within 0.002 of `split-anti` on removal, with every seed within the spill criterion.

The loss-rate screen put the loss rate near one in forty, so one loss in eight here would be unremarkable. Its upper bound is about one in nine, which leaves two losses in eight inside chance as well. So a change in the loss rate would show only at three or more losses; a smaller effect would show in α on the pulled slices, or in removal on the seeds that kept the anchor.
<!-- REVIEW: an earlier draft said two or more losses would show a change; that read the loss-rate point estimate (1 of 40) as the rate. Its one-sided bound is near one in nine, at which two of eight losses have about one chance in five, so the threshold is three. Verify: the binomial at p = 1/9, n = 8. -->

## The measurements

- Whether the anchor held: α at the example answers at the end of training, per slice, measured as in split-anti-seeds. A run that lost the anchor sits near zero on the pulled slices. It counts as lost when the full edit removes less than half of `difference`, net of the control (the rule of split-anti-seeds and the loss-rate screen).
- Removal, spill, and landing: the dose scoring of ex-2.2.22 (the projection at every slice and position, at four doses), net of the control at the same seed. Removal is the net drop on `difference` over its clean-to-null span, spill the largest net drop on another op, and landing the fraction of the distance to the target null that the full edit covers. E2 also shows removal without subtracting the control, since the control turns out to matter here.
- The task scores of every op before any edit, compared with those of the control.
- The basis measurement, new here: the share of the energy of the stream[^energy] that each direction holds, along the coordinate axes and along the rotated basis, and the mean kurtosis[^kurtosis] along each. It also records the cost of projecting out one direction at full dose, for e₁, each coordinate axis, each rotated axis, and sixteen random directions. The stored runs in the memo give the reference: one coordinate holds about half of the energy, a typical axis costs 0.03, and a random direction 0.08 to 0.13.

[^energy]: Energy: the sum of the squared projections of the states onto a direction. The share of a direction is its energy over the total.

[^kurtosis]: Kurtosis: how heavy the tails of a distribution are. A Gaussian scores 3; a direction where most values are small and a few are large scores higher.
"""

# %%


def depth_figure() -> str:
    data = {
        "holds": list(ex.FALLOFF_HOLDS),
        "series": [
            {
                "which": w,
                "name": "rotated basis" if w == NEW else "stored (split-anti-seeds)",
                "anchored": [ALPHA[w][s].tolist() for s in SEEDS],
                "other": [alpha_by_slice(w, s, "alpha_other").tolist() for s in SEEDS],
            }
            for w in (NEW, OLD)
        ],
    }
    alt = """
        Alignment at the example answers by slice, the new runs beside the stored falloff runs. In difference
        contexts both rise from near zero at the embedding to about 0.6 at slices 3 and 4, nearly on top of each
        other at every seed; the new runs sit a few hundredths higher at the embedding. In the contexts of the other
        ops both stay low, at about 0.1 to 0.2 from slice 1 on. Underneath, the hold of the anti weight by slice,
        a straight line from 0.2 at the embedding to 0.03 at slice 4.
    """
    return depth_draw(data, alt)


@memo
def depth_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="rotated-basis-depth",
        alt_text=alt_text,
        caption=f"""
            **Alignment at the example answers by slice**, at the end of training, on the probe contexts of
            `{ex.ANCHORED_OP}` (left) and of the other ops (right): the new runs and their stored twins. One
            hairline per run and the seed mean bold. Underneath is the hold of the anti weight at each slice,
            the same in both.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(
            2, 2, figsize=(6.4, 3.4), layout="constrained", sharex=True, sharey="row", height_ratios=[3, 1]
        )
        x = np.arange(len(SLICE_TICK))
        for col, key in enumerate(("anchored", "other")):
            ax, wax = axes[0, col], axes[1, col]
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            for series in data["series"]:
                w = series["which"]
                for r in series[key]:
                    ax.plot(x, r, color=INK[w], lw=0.5, alpha=0.6, zorder=2)
                ax.plot(
                    x, np.mean(series[key], axis=0), color=INK[w], lw=1.6, marker=MARK[w], ms=4.5, zorder=3,
                    label=series["name"],
                )  # fmt: skip
            wax.plot(x, data["holds"], color=light_dark("#333", "#ddd"), lw=1.2, marker="o", ms=3)
            ax.set_ylim(-0.05, 0.8)
            wax.set_ylim(0, 0.24)
            wax.set_yticks([0, 0.1, 0.2])
            wax.set_xticks(x, SLICE_TICK)
            wax.set_xlabel("slice", fontsize=8)
            for a_ in (ax, wax):
                a_.tick_params(labelsize=7)
        axes[0, 0].set_ylabel(f"α, {ex.ANCHORED_OP}", fontsize=8)
        axes[0, 1].set_ylabel("α, other ops", fontsize=8)
        axes[1, 0].set_ylabel("hold", fontsize=8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


EMB_NEW, EMB_OLD = (np.mean([ALPHA[w][s][0] for s in SEEDS]) for w in (NEW, OLD))
LATE_GAP = max(abs(ALPHA[NEW][s][1:] - ALPHA[OLD][s][1:]).max() for s in SEEDS)

rf"""
## Does the anchor hold? (E1)

With the basis rotated, is e₁ still held through training at every seed, and as firmly? If the anchor leans on e₁ being a privileged axis, some seeds should lose it, or α at the example answers should sit lower on the pulled slices than in the stored `falloff` runs. If not, the per-slice α profiles should lie within the stored seed range.

A second explanation needs keeping apart. In the stored runs the syntax spike sits on a coordinate axis, orthogonal to e₁ by construction; under the rotation it sits on a rotated axis whose cosine with e₁ is a few tenths at most, and known per seed.

So an anchor that is harder to hold could mean that the anchor needs a privileged axis, or that the anti term now has to push the syntax spike off e₁. E3 reports that cosine per seed, to show how much of any seed-to-seed difference it accounts for.

**What we saw.** The anchor held on every seed. The alignment profile of each new run lies on that of its stored twin at every slice after the embedding (the largest gap on any seed and slice is {num(LATE_GAP)}), and the eight new profiles span the same range the eight stored ones do.

{depth_figure()}

The one visible difference is at the embedding, where the new runs sit at about {num(EMB_NEW)} against {num(EMB_OLD)} in the stored runs. The anti weight is highest there, and under the rotation the syntax tokens no longer have a coordinate of their own, orthogonal to e₁ for free. So the few hundredths may be what it costs the anti term to keep them off e₁.

But the same uplift shows in the contexts of the other ops, which fits another explanation at least as well: under the rotation e₁ is a generic direction at the embedding, so every embedding has a small share of it. Past the embedding, the alignment in those contexts is unchanged.

So the anchor does not need e₁ to be a privileged axis: the pull finds and holds an ordinary direction as well as it held an axis.
"""

# %%


def dose_figure() -> str:
    data = {
        "doses": GAMMAS,
        "series": [
            {
                "which": w,
                "name": "rotated basis" if w == NEW else "stored (split-anti-seeds)",
                "runs": [CURVES[w][s] for s in SEEDS],
            }
            for w in (NEW, OLD)
        ],
    }
    alt = f"""
        Three panels against the dose: removal net of the control, removal without subtracting the control, and
        spill. Net removal grows with dose and reaches 0.8 to 0.9 at the full dose on five new runs, as on every
        stored run; three new runs end lower, at about {", ".join(num(FULL[NEW][s]) for s in sorted(LOST_BY_NET + [s for s in SEEDS if 0.5 <= FULL[NEW][s] < 0.7]))}.
        The raw removal reaches about 0.97 on every run, new and stored alike, so the three low net curves are the
        control's own drop. Spill stays near zero on every run.
    """
    return dose_draw(data, alt)


@memo
def dose_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="rotated-basis-dose",
        alt_text=alt_text,
        caption="""
            **Removal and spill by dose**, the edit at every position and slice: removal net of the control (left),
            removal without subtracting the control (middle), and spill (right). One hairline per run and the seed
            mean bold; the stored `falloff` runs beside the new ones. The dashed rule is the selectivity criterion.
        """,
    )
    def _plot() -> plt.Figure:
        keys = (("removal", "removal, net of control"), ("raw_removal", "removal, raw"), ("spill", "spill, net"))
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), layout="constrained", sharex=True)
        for ax, (key, title) in zip(axes, keys, strict=True):
            ax.axhline(0, color=light_dark("#aaa", "#555"), lw=0.5, zorder=0)
            for series in data["series"]:
                w = series["which"]
                for r in series["runs"]:
                    ax.plot(data["doses"], r[key], color=INK[w], lw=0.6, alpha=0.8, zorder=2)
                mean = np.mean([r[key] for r in series["runs"]], axis=0)
                ax.plot(
                    data["doses"], mean, color=INK[w], lw=1.7, marker=MARK[w], ms=4.5, zorder=3, label=series["name"]
                )
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("dose", fontsize=8)
            ax.set_xticks(data["doses"], [f"{g:g}" for g in data["doses"]], fontsize=7)
            ax.tick_params(labelsize=7)
            if key == "spill":
                ax.axhline(CRITERION, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=1)
                ax.set_ylim(-0.07, 0.07)
            else:
                ax.set_ylim(-0.05, 1.05)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


def seed_table() -> str:
    rows = [
        [
            str(s),
            num(RAW[NEW][s]),
            num(RAW[OLD][s]),
            num(CDROP[NEW][s]),
            num(CDROP[OLD][s]),
            num(FULL[NEW][s]),
            num(FULL[OLD][s]),
            num(WORST_SPILL[NEW][s], ".3f"),
            num(CURVES[NEW][s]["raw_spill"][-1], ".3f"),
            num(CURVES[OLD][s]["raw_spill"][-1], ".3f"),
            num(LANDING[NEW][s]),
            num(TASK[NEW][s], "+.3f"),
            num(COS[CONTROL][s]),
        ]
        for s in SEEDS
    ]
    return table_html(
        [
            "seed",
            "raw removal, rotated ↑",
            "raw removal, stored ↑",
            "control drop, rotated",
            "control drop, stored",
            "net removal, rotated ↑",
            "net removal, stored ↑",
            "spill, rotated ↓",
            "raw spill, rotated ↓",
            "raw spill, stored ↓",
            "landing, rotated ↑",
            "task change, rotated ↑",
            "cos(e₁, control spike)",
        ],
        rows,
        f"""
        **Removal, spill, landing and task by seed**, at the full dose. The raw removal is the anchored run's own
        drop on `{ex.ANCHORED_OP}` as a share of its clean-to-null span; the control drop is the same for the
        un-anchored control at that seed, under the same edit; the net removal is their difference, the usual
        measurement. "Stored" is the twin from split-anti-seeds and ex-2.2.23. Spill is the largest net drop on
        another op over the four doses, the raw spill the largest drop on another op at the full dose without
        subtracting the control, and the task change is the mean score over ops less that of the control. The
        last column is the cosine between e₁ and the rotated basis vector that holds the syntax tokens of the control
        (E3). Arrows mark which direction is better.
        """,
    )


rf"""
## Removal, spill and landing (E2)

On the seeds that kept the anchor, does the edit remove `difference` as it did, and does it spill as little? The stored `falloff` runs removed four fifths to nine tenths of the op at full dose with every seed within the spill criterion of 0.02. The comparison is paired by seed, against the new control at the same seed.

**What we saw.** Two things happened at once, and the table separates them. First, the anchored runs themselves remove the op as they did: before subtracting the control, the drop on `difference` under the full edit is {fmt(RAW[NEW].values())} of the clean-to-null span on the new runs and {fmt(RAW[OLD].values())} on the stored ones.

Every new run spills within the criterion (the largest net spill over the doses is {fmt(WORST_SPILL[NEW].values(), ".3f")}, and the raw spill at the full dose, {fmt((CURVES[NEW][s]["raw_spill"][-1] for s in SEEDS), ".3f")}, matches what the stored runs show). The landing is {fmt(LANDING[NEW].values())} against {fmt(LANDING[OLD].values())} stored, and the task change is within the seed range ({fmt(TASK[NEW].values(), ".3f")}).

{dose_figure()}

Second, the removal net of the control (the measurement the earlier reports quote) falls short on {len(LOST_BY_NET)} seeds by the kept/lost rule, and is lower on a third. The shortfall sits in the control. In the stored runs the control loses about a tenth of the op when e₁ is projected out, because e₁ there is an ordinary coordinate axis that holds little.

Under the rotation e₁ is an ordinary direction instead, which is not orthogonal to the syntax spike of the model. On seeds {" and ".join(str(s) for s in OVERLAP_SEEDS)} the cosine between e₁ and the spike direction of the control is about a quarter, and the control loses half or more of the op to the edit on its own. The shortfall in net removal tracks that cosine across the eight seeds (Pearson r = {num(R_SHORTFALL)}).

{seed_table()}

So the kept/lost rule, applied as written, would count {len(LOST_BY_NET)} of 8 new runs as having lost the anchor. That is inside chance by the reckoning in Scope, and the raw removal and the alignment both say that no run lost the anchor ({len(LOST_BY_RAW)} are below half by the raw drop).

What the edit takes from each run agrees. On seeds {" and ".join(str(s) for s in OVERLAP_SEEDS)} the control loses on every op, up to {fmt((CTRL_OTHER[NEW][s] for s in OVERLAP_SEEDS))} in score on an op other than `{ex.ANCHORED_OP}`, while the anchored run at the same seeds loses {fmt((CURVES[NEW][s]["raw_spill"][-1] for s in OVERLAP_SEEDS), ".3f")} at most on any other op. The stored controls lose at most {num(max(CTRL_OTHER[OLD].values()))} on any other op.

So the edit on the rotated control takes syntax, spread over the task, while the edit on the anchored run takes the anchored concept and little else. Netting against the control assumes the edit on the control touches nothing much, which held when e₁ was a quiet axis and does not hold for a direction that overlaps the spike of the control.

At those seeds the anchored run holds its own syntax spike on a rotated axis away from e₁ (E3), which fits the anti term steering it there. So e₁ is clean in the anchored run where it is not in the control.
<!-- REVIEW: the raw removal is reported beside the net removal here because the net measurement's baseline
  changed meaning under the rotation. Verify: the control drop on the anchored op (table) is 0.5 or more on the seeds
  where net removal falls short, and about 0.1 elsewhere and on every stored seed. The claims review added the
  per-op comparison: the control's drop on the overlap seeds is spread over every op, the anchored run's is on the
  anchored op alone (dose arrays, edits[-1]["drop"]). -->
"""

# %%


def spectrum_figure() -> str:
    data = {
        "arms": [
            {
                "arm": a,
                "name": name,
                "axes": [sorted(BASIS[ex.label_of(a, s)]["energy_axes"][-1], reverse=True) for s in SEEDS],
                "rotated": [sorted(BASIS[ex.label_of(a, s)]["energy_rotated"][-1], reverse=True) for s in SEEDS],
            }
            for a, name in ((CONTROL, "control"), (FALLOFF, "falloff"))
        ]
    }
    alt = """
        The share of the energy of the stream held by each direction at the last slice, sorted from the largest, for the
        control (left) and falloff (right). Along the rotated basis (red) one direction holds nearly half and the
        rest fall away quickly; along the coordinate axes (grey) the largest holds under a tenth and the spectrum is
        nearly flat. The same on every seed.
    """
    return spectrum_draw(data, alt)


@memo
def spectrum_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="rotated-basis-spectrum",
        alt_text=alt_text,
        caption="""
            **The energy spectrum of the stream at the last slice**, along the coordinate axes (grey) and along
            the rotated basis (red), each sorted from the largest, one hairline per seed, on a log scale. A
            share of a direction is the mean square, over positions, of the component of the state along it; the 64 shares
            along a basis sum to one, and the dotted rule is one 64th.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), layout="constrained", sharex=True, sharey=True)
        x = np.arange(1, C + 1)
        for ax, arm in zip(axes, data["arms"], strict=True):
            for key, ink, label in (("axes", AXIS_INK, "coordinate axes"), ("rotated", ROT_INK, "rotated basis")):
                for i, r in enumerate(arm[key]):
                    ax.plot(x, r, color=ink, lw=0.7, alpha=0.8, label=label if i == 0 else None)
            ax.axhline(1 / C, color=light_dark("#aaa", "#555"), lw=0.5, ls=":", zorder=0)
            ax.set_title(arm["name"], fontsize=8)
            ax.set_xlabel("direction, sorted", fontsize=8)
            ax.set_xticks([1, 16, 32, 48, 64])
            ax.set_yscale("log")
            ax.set_ylim(0.002, 0.7)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("share of energy", fontsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


def cost_figure() -> str:
    rng = np.random.default_rng(0)
    data = {
        "seeds": SEEDS,
        "arms": [
            {
                "arm": a,
                "name": name,
                "e1": [float(mean_cost(ex.label_of(a, s), "cost_axes")[0]) for s in SEEDS],
                "axes": [mean_cost(ex.label_of(a, s), "cost_axes")[1:].tolist() for s in SEEDS],
                "rotated": [mean_cost(ex.label_of(a, s), "cost_rotated").tolist() for s in SEEDS],
                "random": [mean_cost(ex.label_of(a, s), "cost_random").tolist() for s in SEEDS],
                "cos": [COS[CONTROL][s] for s in SEEDS],
            }
            for a, name in ((CONTROL, "control"), (FALLOFF, "falloff"))
        ],
        "jitter": rng.uniform(-0.12, 0.12, size=200).tolist(),
    }
    alt = f"""
        The cost of projecting out one direction, one column per seed, for the control (left) and falloff (right).
        In the control, the 63 other coordinate axes sit in a band at about 0.05 to 0.1 with a few higher, the
        rotated axes mostly lower with one at about 0.57 (the spike), and the random directions spread between.
        e₁ (the square) sits in the band on most seeds and rises to about 0.17 and 0.22 on seeds
        {" and ".join(str(s) for s in OVERLAP_SEEDS)}, where its cosine with the spike is about a quarter. In
        falloff, e₁ costs about 0.1 on every seed, the removal of the anchored op spread over the ops.
    """
    return cost_draw(data, alt)


@memo
def cost_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="rotated-basis-cost",
        alt_text=alt_text,
        caption="""
            **The cost of projecting out one direction**, at full dose at every slice and position, as the mean drop
            in the task score over the ops; one column per seed, the control on the left and `falloff` on the
            right. Grey dots are the 63 coordinate axes other than e₁, red the 64 rotated axes, and the open circles
            sixteen random directions; the square is e₁.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained", sharex=True, sharey=True)
        jit = np.asarray(data["jitter"])
        for ax, arm in zip(axes, data["arms"], strict=True):
            for i in range(len(data["seeds"])):
                for key, ink, dx, mk in (
                    ("axes", AXIS_INK, -0.22, "."),
                    ("rotated", ROT_INK, 0.0, "."),
                    ("random", RAND_INK, 0.22, "o"),
                ):
                    v = np.asarray(arm[key][i])
                    ax.scatter(
                        i + dx + jit[: len(v)] * 0.5, v, s=6 if mk == "." else 9, color=ink, alpha=0.5, marker=mk,
                        facecolors="none" if mk == "o" else ink, lw=0.6, zorder=2,
                    )  # fmt: skip
                ax.scatter([i - 0.22], [arm["e1"][i]], s=28, marker="s", color=INK[NEW], zorder=4, label="e₁")
            ax.set_xticks(range(len(data["seeds"])), [str(s) for s in data["seeds"]])
            ax.set_xlabel("seed", fontsize=8)
            ax.set_title(arm["name"], fontsize=8)
            ax.set_ylim(-0.02, 0.62)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("cost", fontsize=8)
        return fig

    return _plot()


def home_table() -> str:
    rows = []
    for s in SEEDS:
        row = [str(s)]
        for a in (CONTROL, FALLOFF):
            lab = ex.label_of(a, s)
            row += [
                num(TOP_ROT[lab]),
                num(TOP_AXIS[lab]),
                ", ".join(num(token_cos(lab, t)) for t in SYNTAX),
                num(COS[a][s]),
                "same" if spike_axis(ex.label_of(a, s)) == spike_axis(ex.label_of(CONTROL, s)) else "moved",
            ]
        rows.append(row)
    return table_html(
        [
            "seed",
            "spike share, control",
            "top axis share, control",
            "syntax on a rotated axis, control",
            "cos(e₁, spike), control",
            "spike share, `falloff`",
            "top axis share, `falloff`",
            "syntax on a rotated axis, `falloff`",
            "cos(e₁, spike), `falloff`",
            "spike axis, `falloff`",
        ],
        rows,
        f"""
        **Where the privileged basis went**, per seed. The spike share is the largest share of energy held by any
        rotated basis vector at the last slice, and the top axis share the largest held by any coordinate axis. The
        syntax columns give the largest cosine between each syntax embedding ({", ".join(SHOWN[t] for t in SYNTAX)})
        and any rotated basis vector; under Adam in the coordinate basis these are 0.86 to 1.0 against a coordinate
        axis (the memo). The cosine columns give the cosine between e₁ and the spike direction of that run, and the last column says whether `falloff` holds its spike on the same rotated basis vector as the control at that seed.
        """,
    )


ROT_MEDIAN = float(np.median([np.median(mean_cost(ex.label_of(CONTROL, s), "cost_rotated")) for s in SEEDS]))
AXIS_MEDIAN = float(np.median([np.median(mean_cost(ex.label_of(CONTROL, s), "cost_axes")[1:]) for s in SEEDS]))
E1_CTRL_QUIET = [E1_COST[CONTROL][s] for s in SEEDS if s not in OVERLAP_SEEDS]
E1_CTRL_OVERLAP = [E1_COST[CONTROL][s] for s in OVERLAP_SEEDS]

rf"""
## Where the privileged basis went (E3)

Did the rotation do what it was meant to? If so, the energy spectrum should be flat along the coordinate axes and spiked along the rotated axes, with the syntax tokens on a rotated axis. And e₁ should cost what an ordinary direction costs in the stored runs (0.08 to 0.13 in a control, where it touches the spikes), not what an ordinary axis costs (0.03).

In the anchored runs e₁ holds the anchored op, so its cost there is the removal plus the spill, and it is compared with e₁ in the stored `falloff` runs. The cleaner comparison is within a run (e₁ against the random directions of that run and its rotated axes off the spikes), since the stored reference numbers are means over draws that touch the spikes by varying amounts.

The basis measurement also records where each token embedding sits (its largest cosine with a coordinate axis and with a rotated axis), so the published data show whether the syntax tokens moved to a rotated axis.

**What we saw.** The privileged basis moved with the optimizer. Along the rotated basis one direction holds {fmt((TOP_ROT[ex.label_of(a, s)] for a in (CONTROL, FALLOFF) for s in SEEDS))} of the energy of the stream at the last slice (the spike of the memo). Along the coordinate axes the largest share is {fmt((TOP_AXIS[ex.label_of(a, s)] for a in (CONTROL, FALLOFF) for s in SEEDS))}, and the spectrum is nearly flat.

Kurtosis runs the reverse of the stored runs: near the Gaussian value of 3 along the axes ({fmt((KURT[ex.label_of(CONTROL, s)][0] for s in SEEDS), ".1f")} in the controls), and above it along the rotated basis ({fmt((KURT[ex.label_of(CONTROL, s)][1] for s in SEEDS), ".1f")}).

{spectrum_figure()}

The syntax tokens sit on that rotated spike: the `?`, `,`, ⏎ and `=` embeddings have a cosine of {fmt((token_cos(ex.label_of(a, s), t) for a in (CONTROL, FALLOFF) for s in SEEDS for t in SYNTAX))} with a rotated basis vector, and on most seeds all four share the vector that holds the spike (the table). Against the coordinate axes their largest cosine is about a third, what a random unit vector in 64 dimensions gives.

On {len(MOVED)} of 8 seeds ({", ".join(str(s) for s in MOVED)}) the spike of the anchored run sits on a different rotated axis from that of the control, and on the others on the same one. In the stored runs the memo found the spike on the same coordinate in the control and the anchored run at every seed it checked. So a move here fits the anti term steering the choice, but the two runs also differ in their loss terms, and could settle differently for that reason alone.

{home_table()}

e₁ now costs what a direction costs, plus the cost of its overlap with the spike. In the controls, the 63 other coordinate axes cost {num(AXIS_MEDIAN)} at the median, about what the random directions in the memo cost, and a rotated axis off the spike costs {num(ROT_MEDIAN)}, the ordinary-axis figure from the memo. So the cheap directions are now the rotated axes.

In the controls, e₁ costs {fmt(E1_CTRL_QUIET)} on the six seeds where its cosine with the spike is about a tenth or less, and {fmt(E1_CTRL_OVERLAP)} on the two where it is about a quarter, the overlap E2 found in the net removal. In the `falloff` runs e₁ costs {fmt(E1_COST[FALLOFF].values())} on every seed: the removal of the anchored op, which the cost spreads over the ops, with nothing of the spike in it.

{cost_figure()}
"""

# %%

rf"""
## Discussion

The anchor did not need a privileged axis. Pulled toward an ordinary direction, `falloff` aligned, removed and stayed selective as it did on the coordinate axis on all eight seeds, apart from a few hundredths of alignment at the embedding. So the assumption of the design, that e₁ stands in for any direction, holds for the anchor itself. The finding of the memo changes how we should interpret the comparisons around the anchor, not what the anchor does.

What the rotation did change is the control, which the removal and spill measurements are netted against. When e₁ was an axis, the syntax tokens of the control sat on another axis, orthogonal to e₁ for free. When e₁ is a direction, it overlaps the spike of the control by a random cosine, and on two of eight seeds that was enough for the edit to take half the op from the control, so the net measurement counts a loss the anchored run never had.

A reader of an earlier report need not revise anything, since every stored comparison kept e₁ as an axis in both runs. But the same logic applies to sham directions: a sham compared with e₁ should be another coordinate axis off the two syntax coordinates, since a random direction overstates the baseline by whatever it takes from the spike.

The anchored runs also show the anti term at work in a way the stored runs could not. With the coordinates privileged, keeping the syntax off e₁ cost nothing, because the model parked it on a different coordinate anyway. Here, on {len(MOVED)} seeds, the anchored run holds its syntax on a different rotated axis from the one the control chose.

That fits the anti term steering the choice of the model among the cheap directions of the optimizer. But one of the three is a seed where e₁ overlaps nothing, and the two runs may have settled differently for reasons of their own.

If this holds, the anti term would have a larger job still in a model with no privileged basis at all, since there would be no cheap direction to steer the syntax onto. [The optimizer scout](../optimizer-scout/report.py) shows what the stream of such a model looks like without an anchor.
"""
