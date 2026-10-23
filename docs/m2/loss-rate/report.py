# ruff: noqa: B018
# title: How often the anchor is lost

# An exploratory report: an early call on whether a run kept the anchor, checked on the stored runs of split-anti-seeds,
# anchor-ramp and soft-pool, then used to screen 32 new seeds each of `falloff` and `step, soft τ`, stopped at epoch 64.
# `experiment.py` beside this script trains the screen runs.
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


def fetch_json(ref: str, required: bool = True) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    if art is None:
        assert not required, f"not published: {ref}"
        return None
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


SP = ex.sp
SAS = ex.sas
AR = SP._load_sibling("anchor-ramp", "loss_rate_report_ar")
STORED = {"split-anti-seeds": SAS, "anchor-ramp": AR, "soft-pool": SP}
# The experiments whose runs calibrate the call, each with every run trained to 400 epochs and dose-scored.
STORED_TRAJ = {name: fetch_json(m.TRAJ_REF) for name, m in STORED.items()}
STORED_DOSE = {name: {r["label"]: r for r in fetch_json(m.DOSE_REF)["runs"]} for name, m in STORED.items()}
CONTROL_DOSE = {k: v for k, v in STORED_DOSE["split-anti-seeds"].items() if k.startswith("control-")}

SCREEN_TRAJ = fetch_json(ex.TRAJ_REF)
CALLS = fetch_json(ex.CALLS_REF)
FINISHED_DOSE = fetch_json(ex.DOSE_REF, required=False)  # absent when every screen run was called kept
FINISHED_DOSE = {r["label"]: r for r in FINISHED_DOSE["runs"]} if FINISHED_DOSE else {}
FINISHED_TRAJ = fetch_json(ex.FINISHED_TRAJ_REF, required=False) or {}
# Every run measured by embedding-lean's `measure_one`: the finished runs, the stored runs of both conditions, and
# ex-2.2.23's controls.
MEASURED = {
    r["label"]: r
    for ref in (ex.RESULTS_REF, SAS.RESULTS_REF, SP.RESULTS_REF, ex.lean.RESULTS_REF)
    for r in (fetch_json(ref, required=False) or {"runs": []})["runs"]
}
DESIGN = CALLS["design"]

OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
ANSWERS = list(ex.lean.ANSWERS)  # The positions of the example answers in a context.
CRITERION = 0.02  # The selectivity criterion of ex-2.2.21: the largest net drop on another op the edit may cause.
SL = ex.TRACK_SLICE
LOST_BELOW = 0.5  # A run whose full edit removes less than half the op lost the anchor (as in split-anti-seeds).
RISE = 0.4  # The level at which E3 calls the example answers on e₁, as in soft-pool and anchor-ramp.
ARMS = tuple(a.name for a in ex.ARMS)
FALLOFF, SOFT = ARMS
NAME = {a.name: a.shown_as for a in ex.ARMS}
SEEDS: list[int] = DESIGN["seeds"]
OLD_SEEDS = list(SAS.SEEDS)
# The seeds each condition has a stored full run at: split-anti-seeds' `falloff` and soft-pool's `step-soft`.
OLD_SOURCE = {FALLOFF: "split-anti-seeds", SOFT: "soft-pool"}


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def fmt(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    if not len(v):
        return "–"
    half = 0.5 * 10 ** -int(spec[1])  # half the last printed digit, so a value that prints as zero has no sign
    m, lo, hi = (0.0 if abs(x) < half else float(x) for x in (v.mean(), v.min(), v.max()))
    return f"{num(m, spec)} ({num(lo, spec)} to {num(hi, spec)})"


def removal_of(dose: dict, seed: int | None) -> float:
    """Removal under the full edit, net of the control at *seed*, or of the seed mean of the controls for `None`."""
    span = dose["clean"]["eem"][D] - dose["null"]["eem"][D]
    if seed is None:
        ctl = np.mean([c["edits"][-1]["drop"][D] for c in CONTROL_DOSE.values()])
    else:
        ctl = CONTROL_DOSE[f"control-s{seed}"]["edits"][-1]["drop"][D]
    return float((dose["edits"][-1]["drop"][D] - ctl) / span)


def control_drops(seed: int | None) -> np.ndarray:
    """The drop of each op under the full edit on the control at *seed*, or the seed mean of the controls for `None`,
    one row per dose.
    """
    if seed is None:
        return np.mean([[e["drop"] for e in c["edits"]] for c in CONTROL_DOSE.values()], axis=0)
    return np.array([e["drop"] for e in CONTROL_DOSE[f"control-s{seed}"]["edits"]])


def spill_of(dose: dict, seed: int | None) -> float:
    """The largest net drop on another op at any dose, net of the control as in `removal_of`."""
    net = np.array([e["drop"] for e in dose["edits"]]) - control_drops(seed)
    return float(net[:, OTHER].max())


def task_change(label: str, seed: int | None) -> tuple[float, float]:
    """The mean task score over ops less that of the control, and the largest loss on any one op."""
    ctl = [
        MEASURED[SP.ex2223.label_of("control", ex.EPOCHS, s)]["clean"]
        for s in ([seed] if seed is not None else OLD_SEEDS)
    ]
    diff = np.array(MEASURED[label]["clean"]) - np.mean(ctl, axis=0)
    return float(diff.mean()), float(diff.min())


def track(traj: dict, sl: int = SL) -> tuple[np.ndarray, np.ndarray]:
    """Epochs and α at the example answers at slice *sl*, on the `difference` probe contexts."""
    return np.asarray(traj["epoch"]), np.asarray(traj["alpha_anchored"])[:, sl][:, ANSWERS].mean(axis=1)


def first_rise(traj: dict) -> float:
    ep, a = track(traj)
    above = np.flatnonzero(a > RISE)
    return float(ep[above[0]]) if len(above) else float("nan")


# Every stored run: its experiment, arm, seed, outcome and window mean.
STORED_RUNS: list[dict[str, Any]] = []
for name, trajs in STORED_TRAJ.items():
    for label, t in trajs.items():
        arm, seed = label.rsplit("-s", 1)
        rem = removal_of(STORED_DOSE[name][label], int(seed))
        STORED_RUNS.append(
            {
                "experiment": name,
                "arm": arm,
                "seed": int(seed),
                "removal": rem,
                "lost": rem < LOST_BELOW,
                "window": ex.window_mean(t["traj"]),
                "traj": t["traj"],
            }
        )
N_STORED = len(STORED_RUNS)
N_STORED_LOST = sum(r["lost"] for r in STORED_RUNS)
LOST_HI = max(r["window"] for r in STORED_RUNS if r["lost"])
KEPT_LO = min(r["window"] for r in STORED_RUNS if not r["lost"])
SLOWEST = min((r for r in STORED_RUNS if not r["lost"]), key=lambda r: r["window"])
assert LOST_HI < ex.LOST_MAX and KEPT_LO >= ex.KEPT_MIN  # the constants the experiment called with


def held_out() -> list[dict[str, Any]]:
    """For each stored experiment, the cut chosen on the other two (the middle of their gap), and the runs of the
    held-out one it sorts wrongly.
    """
    out = []
    for name in STORED:
        rest = [r for r in STORED_RUNS if r["experiment"] != name]
        lo = max(r["window"] for r in rest if r["lost"])
        hi = min(r["window"] for r in rest if not r["lost"])
        cut = (lo + hi) / 2
        mine = [r for r in STORED_RUNS if r["experiment"] == name]
        wrong = [r for r in mine if (r["window"] < cut) != r["lost"]]
        out.append(
            {"held": name, "lo": lo, "hi": hi, "cut": cut, "n": len(mine), "n_lost": sum(r["lost"] for r in mine),
             "wrong": len(wrong)}
        )  # fmt: skip
    return out


HELD = held_out()


def repro() -> list[dict]:
    """Each early-stopped reproduction against its stored full run: the largest difference in α at the example
    answers at slice 3 over the records up to the stop.
    """
    out = []
    for arm in ARMS:
        for s in ex.REPRO_SEEDS:
            stopped = SCREEN_TRAJ[ex.label_of(arm, s)]["traj"]
            full = STORED_TRAJ[OLD_SOURCE[arm]][f"{arm}-s{s}"]["traj"]
            ep_s, a_s = track(stopped)
            ep_f, a_f = track(full)
            n = int(np.sum(ep_f <= ex.STOP_EPOCH + 1e-6))
            assert np.allclose(ep_s[:n], ep_f[:n])
            out.append({"arm": arm, "seed": s, "n": n, "diff": float(np.abs(a_s[:n] - a_f[:n]).max())})
    return out


REPRO = repro()
REPRO_MAX = max(r["diff"] for r in REPRO)

# --- The screen ------------------------------------------------------------------------------------------------

CALL = {(c["arm"], c["model_seed"]): c for c in CALLS["runs"]}


def outcome(arm: str, seed: int) -> dict[str, Any]:
    """One seed of an condition: its window mean, the call, and whether it lost the anchor, from the full edit where the
    run was trained to the end (the stored seeds, and the screen runs not called kept) and from the call otherwise.
    """
    if seed in OLD_SEEDS:
        r = next(r for r in STORED_RUNS if r["experiment"] == OLD_SOURCE[arm] and r["arm"] == arm and r["seed"] == seed)
        return {"window": r["window"], "call": ex.call_of(r["window"]), "removal": r["removal"], "lost": r["lost"],
                "traj": r["traj"], "stored": True}  # fmt: skip
    c = CALL[(arm, seed)]
    d = FINISHED_DOSE.get(ex.label_of(arm, seed, None))
    rem = removal_of(d, None) if d is not None else None
    assert c["call"] == "kept" or rem is not None, f"{arm} s{seed}: not called kept and not finished"
    lost = rem < LOST_BELOW if rem is not None else False
    return {"window": c["window_mean"], "call": c["call"], "removal": rem, "lost": lost,
            "traj": SCREEN_TRAJ[c["label"]]["traj"], "stored": False}  # fmt: skip


ALL_SEEDS = OLD_SEEDS + SEEDS
OUT = {a: {s: outcome(a, s) for s in ALL_SEEDS} for a in ARMS}
LOST = {a: [s for s in ALL_SEEDS if OUT[a][s]["lost"]] for a in ARMS}
N = len(ALL_SEEDS)


def upper(k: int, n: int) -> float:
    """The one-sided 95% Clopper–Pearson upper bound on a rate with *k* events in *n*."""
    return 1.0 if k == n else float(stats.beta.ppf(0.95, k + 1, n - k))


ONLY = {a: [s for s in LOST[a] if s not in LOST[b]] for a, b in ((FALLOFF, SOFT), (SOFT, FALLOFF))}
N_DISCORDANT = len(ONLY[FALLOFF]) + len(ONLY[SOFT])
SIGN_P = (
    float(stats.binomtest(len(ONLY[FALLOFF]), N_DISCORDANT, 0.5, alternative="greater").pvalue) if N_DISCORDANT else 1.0
)
# The one-sided sign test that `falloff` loses more often, over the seeds where exactly one condition lost.
N_FINISHED = {a: sum(1 for s in SEEDS if OUT[a][s]["removal"] is not None) for a in ARMS}
RISES = {a: [first_rise(OUT[a][s]["traj"]) for s in ALL_SEEDS] for a in ARMS}
WINDOWS = {a: [OUT[a][s]["window"] for s in ALL_SEEDS] for a in ARMS}
LATE = {
    a: [
        s
        for s in ALL_SEEDS
        if not np.isfinite(first_rise(OUT[a][s]["traj"])) or first_rise(OUT[a][s]["traj"]) > ex.STOP_EPOCH
    ]
    for a in ARMS
}
# Seeds whose example answers had not gone over `RISE` by the stop.
LATE_ONLY = {a: [s for s in LATE[a] if s not in LATE[b]] for a, b in ((FALLOFF, SOFT), (SOFT, FALLOFF))}
LATE_P = float(stats.binomtest(len(LATE_ONLY[FALLOFF]), len(LATE_ONLY[FALLOFF]) + len(LATE_ONLY[SOFT]), 0.5,
                               alternative="greater").pvalue) if LATE_ONLY[FALLOFF] or LATE_ONLY[SOFT] else 1.0  # fmt: skip
# The one-sided sign test that `falloff` is late more often, over the seeds where only one condition was late.
WINDOW_R = float(np.corrcoef([WINDOWS[FALLOFF], WINDOWS[SOFT]])[0, 1])
FINISHED: list[dict[str, Any]] = sorted((OUT[a][s] | {"arm": a, "seed": s} for a in ARMS for s in SEEDS if OUT[a][s]["removal"] is not None),
                  key=lambda o: o["seed"])  # fmt: skip
SLOW = [o for o in FINISHED if not o["lost"] and o["call"] != "kept"]
# The runs called kept and finished anyway: the lowest window means of `step, soft τ`.
CHECKED = [o for o in FINISHED if o["call"] == "kept"]


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


INK = {FALLOFF: light_dark("#2a9d8f", "#7fd8c8"), SOFT: light_dark("#e08e0b", "#f5b950")}
MARK = {FALLOFF: "D", SOFT: "v"}
EXP_INK = {
    "split-anti-seeds": light_dark("#2a9d8f", "#7fd8c8"),
    "anchor-ramp": light_dark("#0b3c73", "#9fcbff"),
    "soft-pool": light_dark("#c0392b", "#ff8a76"),
}
EXP_MARK = {"split-anti-seeds": "D", "anchor-ramp": "s", "soft-pool": "^"}
GHOST = light_dark("#bbb", "#555")
RULE = light_dark("#333", "#ddd")
ZERO = light_dark("#aaa", "#555")
BAND = light_dark("#ececec", "#262626")

# --- E1: the call on the stored runs -----------------------------------------------------------------------------


def calib_figure() -> str:
    data = {
        "runs": [{"experiment": r["experiment"], "window": r["window"], "removal": r["removal"]} for r in STORED_RUNS],
        "lost_hi": LOST_HI,
        "kept_lo": KEPT_LO,
        "cut": ex.CALL_AT,
    }
    alt = """
        The lost and kept runs separate cleanly on the window mean: every lost run is below 0.15, with removal near
        zero (one at 0.44), and every kept run is at 0.29 or above, with removal 0.55 to 0.9; no run falls between.
    """
    return calib_draw(data, alt)


@memo
def calib_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="loss-rate-calibration",
        alt_text=alt_text,
        caption=f"""
            **The early call against the outcome**, for every stored run: the window mean (α at the example answers
            at slice {SL}, averaged over epochs {ex.WINDOW[0]} to {ex.WINDOW[1]}) against removal under the full
            edit, net of the control. The shaded band is the gap between the highest lost run and the lowest kept
            one; the dashed line is the cut, at its middle, and the dotted line is the half-removal line that
            defines a loss.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
        ax.axvspan(data["lost_hi"], data["kept_lo"], color=BAND, lw=0, zorder=0)
        ax.axvline(data["cut"], color=RULE, lw=0.8, ls="--", zorder=1)
        ax.axhline(LOST_BELOW, color=RULE, lw=0.8, ls=":", zorder=1)
        ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
        for name in EXP_INK:
            rs = [r for r in data["runs"] if r["experiment"] == name]
            ax.scatter(
                [r["window"] for r in rs],
                [r["removal"] for r in rs],
                marker=EXP_MARK[name],
                s=16,
                facecolor="none",
                edgecolor=EXP_INK[name],
                lw=0.9,
                label=name,
                zorder=2,
            )
        ax.set_xlim(-0.02, 0.65)
        ax.set_ylim(-0.15, 1.0)
        ax.set_xlabel(f"window mean: α, example answers, slice {SL}, epochs {ex.WINDOW[0]}–{ex.WINDOW[1]}", fontsize=8)
        ax.set_ylabel("removal, full edit", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.legend(loc="outside upper center", ncols=3, frameon=False, fontsize=7)
        return fig

    return _plot()


def held_table() -> str:
    rows = [
        [
            f"{h['held']}",
            f"{h['n']} ({h['n_lost']})",
            num(h["lo"]),
            num(h["hi"]),
            num(h["cut"]),
            str(h["wrong"]),
        ]
        for h in HELD
    ]
    return table_html(
        ["held out", "runs (lost)", "highest lost, other two", "lowest kept, other two", "cut", "sorted wrongly ↓"],
        rows,
        """
        **The call on each experiment, with the cut chosen on the other two.** The cut is the middle of the gap
        between the highest window mean of a lost run and the lowest of a kept run in the other two experiments.
        """,
    )


# --- E2: the screen ------------------------------------------------------------------------------------------------


def count_table() -> str:
    rows = []
    for a in ARMS:
        k = len(LOST[a])
        rows.append(
            [
                f"`{NAME[a]}`",
                str(len(OLD_SEEDS)),
                str(len(SEEDS)),
                str(sum(OUT[a][s]["call"] == "kept" for s in SEEDS)),
                str(N_FINISHED[a]),
                f"{k} of {N}",
                f"{upper(k, N):.0%}",
            ]
        )
    return table_html(
        ["condition", "stored seeds", "screen seeds", "called kept", "finished", "lost ↓", "95% upper bound ↓"],
        rows,
        f"""
        **How often each condition lost the anchor.** The stored seeds ({OLD_SEEDS[0]} to {OLD_SEEDS[-1]}) trained to
        400 epochs in earlier experiments; the screen seeds ({SEEDS[0]} to {SEEDS[-1]}) stopped at epoch
        {ex.STOP_EPOCH}. Those not called kept, and the {len(ex.FINISH_ALSO)} lowest kept calls of `step, soft τ`, were
        finished at full length and scored with the full edit. The
        bound is the one-sided Clopper–Pearson bound on the loss rate.
        """,
    )


def schedule_of(arm: str) -> dict:
    """λ_a and τ_p along the first epochs, shared by every seed of an condition; a fixed τ_p is drawn at its value."""
    t = OUT[arm][SEEDS[0]]["traj"]
    tau = np.asarray(t["tau"])[:, 0] if "tau" in t else np.full(len(t["epoch"]), SP.st.ex2216.TAU)
    return {"epoch": t["epoch"], "weight": t["weight"], "tau": tau.tolist()}


def screen_figure() -> str:
    def cut(t: dict) -> dict:
        ep, a = track(t)
        keep = ep <= ex.STOP_EPOCH + 1e-6
        return {"epoch": ep[keep].tolist(), "answers": a[keep].tolist()}

    data = {
        "arms": [
            {
                "arm": a,
                "name": NAME[a],
                "schedule": schedule_of(a),
                "runs": [cut(OUT[a][s]["traj"]) | {"lost": OUT[a][s]["lost"]} for s in ALL_SEEDS],
            }
            for a in ARMS
        ],
        "window": list(ex.WINDOW),
        "lost_max": ex.LOST_MAX,
        "kept_min": ex.KEPT_MIN,
    }
    alt = """
        In both conditions most runs climb from zero to about 0.5 by epoch 30, the `step, soft τ` runs a few epochs later and
        more tightly bunched, with two dipping between the rules at single records; under falloff one run rises
        only at epoch 44, three level off near 0.2 between the rules, and the lost run stays near zero. The strips
        show the anchor weight reaching 0.1 by epoch 40 in both conditions, and the temperature falling from 1 to 0.1
        only under `step, soft τ`.
    """
    return screen_draw(data, alt)


@memo
def screen_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="loss-rate-screen",
        alt_text=alt_text,
        caption=f"""
            **The example answers through epoch {ex.STOP_EPOCH}**: α at the example answers at slice {SL} on the
            `{ex.ANCHORED_OP}` probe contexts, one column per condition and one line per seed (the stored seeds and the
            screen seeds alike), dashed where the run lost the anchor; the other condition is ghosted behind. The shaded
            window is where the call averages, and the two rules are the highest stored lost run and the lowest
            stored kept one. The strips underneath are the anchor weight $\\lambda_a$ and the position pool
            temperature $\\tau_p$ (log scale).
        """,
    )
    def _plot() -> plt.Figure:
        arms = data["arms"]
        fig, axes = plt.subplots(
            3, len(arms), figsize=(7.0, 4.0), layout="constrained", sharex=True, sharey="row",
            height_ratios=[3, 0.8, 0.8], squeeze=False,
        )  # fmt: skip
        for col, arm in enumerate(arms):
            ax, wax, tax = axes[:, col]
            ink = INK[arm["arm"]]
            ax.axvspan(*data["window"], color=BAND, lw=0, zorder=0)
            ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
            for other in arms:
                if other is not arm:
                    for r in other["runs"]:
                        ax.plot(r["epoch"], r["answers"], color=GHOST, lw=0.4, alpha=0.4, zorder=1)
            for r in arm["runs"]:
                ax.plot(r["epoch"], r["answers"], color=ink, lw=0.6, alpha=0.8, ls="--" if r["lost"] else "-",
                        zorder=2)  # fmt: skip
            for y in (data["lost_max"], data["kept_min"]):
                ax.axhline(y, color=RULE, lw=0.7, ls="--", zorder=3)
            ax.set_title(arm["name"], fontsize=8)
            ax.set_ylim(-0.1, 0.8)
            sch = arm["schedule"]
            wax.plot(sch["epoch"], sch["weight"], color=RULE, lw=1.0)
            wax.set_ylim(0, 0.12)
            wax.set_yticks([0, 0.1])
            tax.plot(sch["epoch"], sch["tau"], color=ink, lw=1.1)
            tax.set_yscale("log")
            tax.set_ylim(0.07, 1.4)
            tax.set_yticks([0.1, 1], ["0.1", "1"])
            tax.minorticks_off()
            tax.set_xlim(0, data["window"][1])
            tax.set_xlabel("epoch", fontsize=8)
            for a_ in (ax, wax, tax):
                a_.tick_params(labelsize=7)
        axes[0, 0].set_ylabel(f"α, example\nanswers, slice {SL}", fontsize=8)
        axes[1, 0].set_ylabel("$\\lambda_a$", fontsize=8)
        axes[2, 0].set_ylabel("$\\tau_p$", fontsize=8)
        return fig

    return _plot()


# --- E3: when the example answers come onto e₁ ----------------------------------------------------------------------


def settle_figure() -> str:
    data = {
        "arms": [{"arm": a, "name": NAME[a], "window": WINDOWS[a], "rise": RISES[a]} for a in ARMS],
        "lost_max": ex.LOST_MAX,
        "kept_min": ex.KEPT_MIN,
        "rise": RISE,
    }
    alt_window = """
        `step, soft τ` seeds all have window means between 0.33 and 0.58, above both rules; most falloff seeds sit in
        the same range, but three fall between the rules near 0.2 and one is at zero.
    """
    alt_rise = f"""
        Every `step, soft τ` seed first goes over {RISE:g} at epoch 24 or 28; most falloff seeds get there sooner, at 20
        or 24, but a few take until epochs 28 to 44.
    """
    return settle_draw(data, alt_window, alt_rise)


def dots_panel(arms: list[dict], key: str) -> tuple[plt.Figure, plt.Axes]:
    """One column per condition: the seeds jittered, a thin bar over their range, and the seed mean in the condition marker."""
    fig, ax = plt.subplots(figsize=(2.9, 2.8), layout="constrained")
    rng = np.random.default_rng(0)
    for i, arm in enumerate(arms):
        ink, mk = INK[arm["arm"]], MARK[arm["arm"]]
        v = np.asarray(arm[key], float)
        v = v[np.isfinite(v)]
        ax.plot([i, i], [v.min(), v.max()], color=ink, lw=2.5, alpha=0.25, solid_capstyle="butt", zorder=1)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=ink, alpha=0.5, lw=0, zorder=2, clip_on=False)
        ax.scatter([i], [v.mean()], marker=mk, s=40, color=ink, edgecolor=RULE, lw=0.6, zorder=3)
    ax.set_xticks(range(len(arms)), [a["name"] for a in arms], fontsize=7)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.tick_params(labelsize=7)
    return fig, ax


@memo
def settle_draw(data: dict, alt_window: str, alt_rise: str) -> str:
    @themed(
        name="loss-rate-settle-window",
        alt_text=alt_window,
        caption="""
            **The window mean**, with the highest stored lost run and the lowest stored kept one as rules.
        """,
    )
    def _window() -> plt.Figure:
        fig, ax = dots_panel(data["arms"], "window")
        for y in (data["lost_max"], data["kept_min"]):
            ax.axhline(y, color=RULE, lw=0.7, ls="--", zorder=0)
        ax.set_ylabel(f"window mean, epochs {ex.WINDOW[0]}–{ex.WINDOW[1]}", fontsize=8)
        ax.set_ylim(0, 0.7)
        return fig

    @themed(
        name="loss-rate-settle-rise",
        alt_text=alt_rise,
        caption=f"""
            **The first recorded epoch** at which α at the example answers at slice {SL} goes over {data["rise"]:g}
            (α is recorded every four epochs).
        """,
    )
    def _rise() -> plt.Figure:
        fig, ax = dots_panel(data["arms"], "rise")
        ax.set_ylabel(f"first epoch over {data['rise']:g}", fontsize=8)
        return fig

    return figure_html(
        _window() + _rise(),
        caption="""
            **How far above the cut, and how soon**, for every seed of each condition. Each dot is a seed, jittered
            sideways; the thin bar spans the seeds and the large marker is the seed mean.
        """,
    )


def settle_table() -> str:
    rows = [
        [
            f"`{NAME[a]}`",
            fmt(WINDOWS[a]),
            fmt([r for r in RISES[a] if np.isfinite(r)], ".0f"),
            str(len(LATE[a])),
        ]
        for a in ARMS
    ]
    return table_html(
        ["condition", "window mean ↑", f"first epoch over {RISE:g}", f"seeds not over it by epoch {ex.STOP_EPOCH}"],
        rows,
        f"""
        **How the runs settle**, seed mean and range over all {N} seeds of each condition; the first epoch is over the
        seeds that got there by epoch {ex.STOP_EPOCH}. The warm-up of the anchor weight ends at epoch 40.
        """,
    )


def scores(dose: dict, label: str, seed: int | None) -> dict[str, float]:
    task, worst = task_change(label, seed)
    return {"removal": removal_of(dose, seed), "spill": spill_of(dose, seed), "task": task, "worst": worst}


FINISHED_SCORES = {
    (o["arm"], o["seed"]): scores(FINISHED_DOSE[lb := ex.label_of(o["arm"], o["seed"], None)], lb, None)
    for o in FINISHED
}
STORED_SCORES = {
    a: [scores(STORED_DOSE[OLD_SOURCE[a]][f"{a}-s{s}"], f"{a}-s{s}", s) for s in OLD_SEEDS if not OUT[a][s]["lost"]]
    for a in ARMS
}


def finished_table() -> str:
    keys = ("removal", "spill", "task", "worst")
    rows = [
        [f"`{NAME[o['arm']]}`", str(o["seed"]), num(o["window"]), o["call"]]
        + [num(FINISHED_SCORES[(o["arm"], o["seed"])][k], ".2f" if k == "removal" else ".3f") for k in keys]
        for o in FINISHED
    ]
    rows += [
        [f"`{NAME[a]}`", f"stored, {len(STORED_SCORES[a])} kept", "", ""]
        + [fmt([r[k] for r in STORED_SCORES[a]], ".2f" if k == "removal" else ".3f") for k in keys]
        for a in ARMS
    ]
    return table_html(
        [
            "condition",
            "seed",
            "window mean",
            "call",
            "removal ↑",
            "spill, any dose ↓",
            "task change ↑",
            "worst op change ↑",
        ],  # fmt: skip
        rows,
        f"""
        **The finished runs, scored as the earlier experiments scored theirs**, beside the stored runs of each
        condition that kept the anchor (seed mean and range). Removal is at the full dose; spill is the largest net
        drop on another op at any of the four doses (the criterion is {CRITERION:g}); the task change is the mean
        score over ops less that of the control, and the worst op change the largest loss on any one op. The finished
        runs are netted against the seed mean of the controls, the stored runs against the control at their seed.
        """,
    )


def finished_figure() -> str:
    def full(t: dict) -> dict:
        ep, a = track(t)
        return {"epoch": ep.tolist(), "answers": a.tolist()}

    t0 = FINISHED_TRAJ[ex.label_of(FINISHED[0]["arm"], FINISHED[0]["seed"], None)]["traj"]
    data = {
        "runs": [
            full(FINISHED_TRAJ[ex.label_of(o["arm"], o["seed"], None)]["traj"])
            | {"arm": o["arm"], "seed": o["seed"], "lost": o["lost"]}
            for o in FINISHED
        ],
        "names": {a: NAME[a] for a in ARMS},
        "ghosts": [full(OUT[a][s]["traj"]) for a in ARMS for s in OLD_SEEDS],
        "schedule": {
            "epoch": t0["epoch"],
            "weight": t0["weight"],
            "anti": np.asarray(t0["anti_weight"])[:, SL].tolist(),
        },
        "window": list(ex.WINDOW),
    }
    alt = f"""
        The three slow runs hold near 0.2 until somewhere between epochs 200 and 280, then each jumps within a few
        epochs to about 0.6, level with the stored runs; the lost run stays near zero throughout. The three
        step, soft τ runs swing between about 0.2 and 0.6 until about epoch 200, then settle near 0.6. Underneath,
        the anchor weight is flat at 0.1 from epoch 40 to 360, while the anti weight at slice {SL} falls smoothly from
        about 0.25 to 0.07 by epoch 300.
    """
    return finished_draw(data, alt)


@memo
def finished_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="loss-rate-finished",
        alt_text=alt_text,
        caption=f"""
            **The finished runs, to the end of training**: α at the example answers at slice {SL} on the
            `{ex.ANCHORED_OP}` probe contexts, one line per finished screen run (the `falloff` runs not called kept,
            and the three lowest `step, soft τ` runs), dashed where the run lost the anchor; the stored runs of both
            conditions are ghosts behind. The shaded window is
            where the call averages. The strips underneath are the anchor weight $\\lambda_a$ and the anti weight at
            slice {SL}, $\\lambda_{{\\bar s}}$.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (ax, wax, nax) = plt.subplots(
            3, 1, figsize=(5.6, 3.8), layout="constrained", sharex=True, height_ratios=[3, 0.8, 0.8]
        )
        ax.axvspan(*data["window"], color=BAND, lw=0, zorder=0)
        ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
        for g in data["ghosts"]:
            ax.plot(g["epoch"], g["answers"], color=GHOST, lw=0.5, alpha=0.6, zorder=1)
        for r in data["runs"]:
            ax.plot(r["epoch"], r["answers"], color=INK[r["arm"]], lw=0.9, ls="--" if r["lost"] else "-", zorder=2)
        for a, name in data["names"].items():
            ax.plot([], [], color=INK[a], lw=0.9, label=name)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        ax.set_ylim(-0.1, 0.8)
        ax.set_ylabel(f"α, example\nanswers, slice {SL}", fontsize=8)
        sch = data["schedule"]
        wax.plot(sch["epoch"], sch["weight"], color=RULE, lw=1.0)
        wax.set_ylim(0, 0.12)
        wax.set_yticks([0, 0.1])
        wax.set_ylabel("$\\lambda_a$", fontsize=8)
        nax.plot(sch["epoch"], sch["anti"], color=RULE, lw=1.0)
        nax.set_ylim(0, 0.28)
        nax.set_yticks([0, 0.2])
        nax.set_ylabel("$\\lambda_{\\bar s}$", fontsize=8)
        nax.set_xlim(0, 415)
        nax.set_xlabel("epoch", fontsize=8)
        for a_ in (ax, wax, nax):
            a_.tick_params(labelsize=7)
        return fig

    return _plot()


# %%

rf"""
# How often the anchor is lost

/// tip |
<!-- lede -->
We trained 32 new seeds each of `falloff` and `step, soft τ`, stopped each run early, and called whether it kept the anchor from how far its example answers had come onto e₁. `falloff` lost the anchor on one seed of 40, and `step, soft τ` on none. A few `falloff` runs came onto e₁ only late in training, which `step, soft τ` never did.
///

The recipe of record, `falloff` from [the split-anti-seeds experiment](/docs/m2/split-anti-seeds/report.py), had kept the anchor on all eight of its seeds, which fits a loss rate as high as three in ten. `step, soft τ` from [soft-pool](/docs/m2/soft-pool/report.py), which starts with a soft pool, also kept it on all eight.

Several close variants lost the anchor on one to three seeds of eight, in split-anti-seeds, [anchor-ramp](/docs/m2/anchor-ramp/report.py), and soft-pool. With the recipe known this loosely, a condition that loses one seed in eight couldn't be told apart from it.

Full runs at enough seeds to narrow this are costly. But in those three experiments the lost runs had parted from the kept ones within the first sixty or so epochs. So we first check, on the {N_STORED} stored runs whose outcome we know, whether an early call made from the alignment agrees with the full edit.

Then we train {len(SEEDS)} new seeds each of `falloff` and `step, soft τ` on the full-length schedule, stop them at epoch {ex.STOP_EPOCH}, and call each one. The runs the call can't settle are trained to the end and scored, and so are the three `step, soft τ` runs it was least sure of among those it called kept.
"""

# %%

rf"""
## Observations

- [The early call on the stored runs (E1)](#the-early-call-on-the-stored-runs-e1): how well the example answers had come onto e₁ by epochs {ex.WINDOW[0]} to {ex.WINDOW[1]} sorts every stored run the way the full edit does, with a clear gap between the lost runs and the kept ones. It still does when each experiment is held out of choosing the cut, though one slow run then clears the cut by only a few hundredths.
- [How often each condition loses the anchor (E2)](#how-often-each-condition-loses-the-anchor-e2): `falloff` lost the anchor on one seed of {N}, and `step, soft τ` on none. That puts the loss rate of `falloff` under about one in nine and that of `step, soft τ` under about one in fourteen. The two can't be told apart on losses alone. The runs finished to check the call all kept the anchor, with removal and spill like those of the stored runs.
- [The slow runs (E3)](#the-slow-runs-e3): under `step, soft τ` the example answers came onto e₁ within a few epochs of each other on every seed. Under `falloff` most runs were a little quicker, but three hung partway until after epoch 200. They came up once the anti weight had fallen to about half, and kept the anchor.

## Method

This is an exploratory study, with no preregistration and no gate. It covers two conditions, both with the step pull on slices 2 to 4 and the `falloff` anti weight on every slice. `falloff` itself has a fixed pooling temperature $\tau_p$ = 0.1, and `step, soft τ` sharpens $\tau_p$ from 1 to 0.1 over the 40-epoch warm-up of the anchor weight. Everything else is the training of split-anti-seeds.

The new runs are at seeds {SEEDS[0]} to {SEEDS[-1]} in both conditions, so the two are paired by initialization, batches and label draws. Each follows the full 400-epoch schedule (learning rate and both term weights as in a full run) but stops at epoch {ex.STOP_EPOCH}. With the stored runs at seeds {OLD_SEEDS[0]} to {OLD_SEEDS[-1]}, each condition has {N} seeds.

Stopping early leaves the run up to the stop as it was in the stored full run. Two seeds of each condition ({" and ".join(map(str, ex.REPRO_SEEDS))}) were trained again and stopped at epoch {ex.STOP_EPOCH}, and α at every recorded epoch up to the stop matches the stored run to the last digit in all four (largest difference {REPRO_MAX:g}).

A screen run that is not firmly kept (called lost, or between the stored lost and kept runs) was trained again to the end and scored with the full edit, so every loss counted is a measured one. A kept call is mostly not checked this way, and rests on the timing of the stored losses.

One stored loss shows how a kept call could be wrong. In soft-pool, seed 708 of `ramp 1–3` lost the anchor late: its example answers rose with the other runs, fell back a few epochs later while its query answer stayed on e₁, and only climbed back after about epoch 100, to about four fifths of the kept level. Yet the full edit at the end removed almost nothing, so its alignment came back without the op coming with it. That run was still low in the window, so the call caught it. A lost run that came back sooner, by epochs {ex.WINDOW[0]} to {ex.WINDOW[1]}, would be called kept. No stored run did that, and to check the kept call where it is least sure we also finished the {len(ex.FINISH_ALSO)} `step, soft τ` runs with the lowest window means.

The finished runs have no control at their seed, so their removal is netted against the seed mean of the eight controls of [ex-2.2.23](/docs/m2/ex-2.2.23/report.py). The controls differ by up to about an eighth in removal, which matters only for a run near the half-removal line, and no finished run came near it.

We don't measure spill or the task score on the screen runs, or removal on most of those called kept: the question is the loss rate, and those are known for both conditions from the earlier experiments.

## The measurements

- Kept or lost (the outcome, on stored and finished runs): a run lost the anchor when the full edit, along e₁ at every position and slice, removes less than half of `difference`, net of the control, as in split-anti-seeds.
- The window mean (the call): α (the cosine of the state with e₁) at the example answers of the `difference` probe contexts at slice {SL}, averaged over the six records (one every four epochs) at epochs {ex.WINDOW[0]} to {ex.WINDOW[1]}. Slice {SL} is the deepest slice every stored condition pulls, both screen conditions included. A run is called lost below {ex.LOST_MAX:g}, the highest window mean of any stored lost run (rounded up), and kept at {ex.KEPT_MIN:g} or above, the lowest of any stored kept run (rounded down). Every run in between was finished, so the full edit decides.
- The loss rate: losses over seeds, with the one-sided 95% upper bound from the binomial (Clopper–Pearson[^cp]). The conditions are compared on the seeds where one kept the anchor and the other lost it, with a one-sided exact (sign) test. At these counts it takes about five such seeds all one way to reach p < 0.05, so the screen can bound each rate well but can show a difference between the conditions only if it is large.

[^cp]: Clopper–Pearson: an exact interval for a proportion, built from the binomial itself, so it holds at small counts and at zero losses.
"""

# %%

rf"""
## The early call on the stored runs (E1)

Does the window mean tell the lost runs from the kept ones? We plot it against removal under the full edit for all {N_STORED} stored runs, {N_STORED_LOST} of them lost.

{calib_figure()}

It does, with room to spare. Every lost run has a window mean near zero, every kept run one well above it, and no run falls in the gap between them. Most lost runs ended with almost no removal.

The lost run near the half-removal line is seed 710 of `ramp from 1`, which the anchor-ramp report counted as borderline; on the window mean it groups with the other lost runs. The kept run nearest the gap is seed {SLOWEST["seed"]} of `ramp from emb`, whose example answers came onto e₁ only after epoch 200.

A cut chosen on the same runs it sorts could fit them by chance, so we chose it on two experiments and sorted the third:

{held_table()}

Each held-out experiment is sorted with no error. The closest call is the slow kept run of anchor-ramp, which clears the cut chosen on the other two experiments by a few hundredths.

Windows that end later separate the runs less well, because the late loss of soft-pool brought its example answers partway back by epoch 100 without the removal coming back. Windows that end at epoch 52 or earlier leave little or no gap, since some kept runs were still rising. So the stored runs separate best with the window ending at epoch {ex.STOP_EPOCH}, and we took that stop as given for the screen.
"""

# %%

rf"""
## How often each condition loses the anchor (E2)

How many of the {N} seeds of each condition lost the anchor, how high could the loss rate be, and do the two conditions differ? The figure shows the example answers through epoch {ex.STOP_EPOCH} on every seed of each condition, and the table counts the calls and the losses.

{screen_figure()}

{count_table()}

`falloff` lost the anchor on one seed, {LOST[FALLOFF][0]}, the way the earlier losses did, in anchor-ramp and in the runs of split-anti-seeds with a uniform anti weight on every slice. Its query answer came onto e₁ by epoch 20 and stayed there, its example answers never left zero, and the full edit removed nothing. `step, soft τ` kept the anchor on that seed and every other.

So with 40 seeds each, the loss rate of `falloff` is under about {upper(len(LOST[FALLOFF]), N):.0%} and that of `step, soft τ` under about {upper(len(LOST[SOFT]), N):.0%}. One seed where only one condition lost is far from enough to say which loses more (p = {SIGN_P:.2g}).

Three more `falloff` runs fell between the stored lost and kept runs. All three kept the anchor when finished, with removal {fmt([o["removal"] for o in SLOW])}, close to that of the other kept runs. Two had window means below the middle of the gap, so a single cut there would have counted three losses in place of one.

{finished_table()}

The finished runs scored like the stored runs of their condition. The three slow `falloff` runs removed as much, stayed well within the spill criterion, and lost no more task score than the stored runs. The three lowest kept calls of `step, soft τ` ({fmt([o["window"] for o in CHECKED])} on the window mean) all kept the anchor, with removal {fmt([o["removal"] for o in CHECKED])} and spill within the criterion, so the kept call held where it was least sure.

One of them, seed 732, scored lower on the task than any stored run: a few hundredths lower on every op, and about a tenth lower on the HSV ops. The edit isn't involved, since this is the score before editing, and its removal and spill are ordinary. It has no control at its seed, so we can't tell whether the seed learns the task less well or the condition did; the screen stopped its `falloff` run at epoch {ex.STOP_EPOCH}, so there is nothing to compare with.
"""

# %%

rf"""
## The slow runs (E3)

What set those three `falloff` runs apart, and does `step, soft τ` have runs like them? The figure and table show the window mean, and the first recorded epoch at which α at the example answers goes over {RISE:g} (as in soft-pool).

{settle_figure()}

{settle_table()}

Under `step, soft τ` the example answers came onto e₁ within a few epochs of each other on every seed, between epochs {np.nanmin(RISES[SOFT]):.0f} and {np.nanmax(RISES[SOFT]):.0f}, as in soft-pool. Under `falloff` most runs got there a few epochs sooner, but the spread is wider: one run jumped up only at epoch 44, and {len(LATE[FALLOFF])} had not got there by the stop, the lost run and the three slow ones.

All four are seeds where `step, soft τ` got there on time. If the conditions didn't differ, a split of four seeds to none would come up about one time in sixteen (p = {LATE_P:.2g}), and we chose this measure after seeing the runs. So this fits `step, soft τ` being more even, without being conclusive.

A run that crosses {RISE:g} late also has a low window mean, so these four late runs are the lost run and the three that E2 finished from the band. The timing picks out the same runs as the call, and the split adds little to E2 beyond showing that `step, soft τ` had none of them.

The evenness of `step, soft τ` is in when its example answers cross {RISE:g}. How high they then sit through the window varies from seed to seed as much as on the `falloff` runs above the band, a little more if anything. Its lowest run, seed 724, dips into the band at two records late in the window, though its window mean is above it.

Seeds that were slow under one condition tended to be a little slow under the other (the window means of the two conditions are correlated over seeds, r = {WINDOW_R:.2f}). The `step, soft τ` run with the lowest window mean is seed 724, one of the slow `falloff` runs.

The finished runs show what the slow ones did after the stop:

{finished_figure()}

The three slow runs held their example answers about a third of the way to the level of the other runs until after epoch 200. Each then jumped up within a few epochs, at different times between epochs 200 and 280, and kept the anchor. The lost run stayed off e₁ to the end. The three `step, soft τ` runs swung widely, between about 0.2 and 0.6, until about epoch 200, then settled at the level of the stored runs. None of them hung low and then jumped as the slow `falloff` runs did.

The jumps come once the anti weight on slice {SL} has fallen to about half its starting value, which fits the anti term holding them back. But the anti weight falls over most of training, so the timing alone doesn't single it out. The jumps are also abrupt where the fall is smooth, and nothing here tests the idea.
"""

# %%

r"""
## Discussion

The screen bounds the loss rates of both conditions well below the three in ten that eight seeds allowed. The one loss was of the kind seen before: the query answer takes the pull and the example answers never come onto e₁. The conditions can't be told apart on losses, and telling apart rates this low would take several times as many seeds.

In these runs `step, soft τ` was more even in when the example answers came onto e₁, and every `falloff` seed that was slow or lost was on time under it. That fits the explanation offered in soft-pool: a soft pool holds the pull back until the example answers can share it. The slow runs and the losses of `falloff` may also be two degrees of one thing. If so, the slow runs would be the cheaper measure of it, since there are more of them than losses.

The early call held up on the screen as far as we checked it: every run it was unsure of was finished and sorted cleanly by removal, and the three kept calls it was least sure of kept the anchor. But the slow `falloff` runs fell in the band, where the stored runs had only one run nearby, and a single cut would have miscounted two of them.
"""
