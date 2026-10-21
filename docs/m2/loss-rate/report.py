# ruff: noqa: B018
# title: How often the anchor is lost

# An exploratory report: an early call on whether a run kept the anchor, checked on the stored runs of split-anti-seeds,
# anchor-ramp and soft-pool, then used to screen 32 new seeds each of `falloff` and the soft start, stopped at epoch 64.
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
DESIGN = CALLS["design"]

OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
ANSWERS = list(ex.lean.ANSWERS)  # The positions of the example answers in a context.
SL = ex.TRACK_SLICE
LOST_BELOW = 0.5  # A run whose full edit removes less than half the op lost the anchor (as in split-anti-seeds).
RISE = 0.4  # The level at which E3 calls the example answers on e₁, as in soft-pool and anchor-ramp.
ARMS = tuple(a.name for a in ex.ARMS)
FALLOFF, SOFT = ARMS
NAME = {a.name: a.shown_as for a in ex.ARMS}
SEEDS: list[int] = DESIGN["seeds"]
OLD_SEEDS = list(SAS.SEEDS)
# The seeds each arm has a stored full run at: split-anti-seeds' `falloff` and soft-pool's `step-soft`.
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


def track(traj: dict, sl: int = SL) -> tuple[np.ndarray, np.ndarray]:
    """Epochs and α at the example answers at slice *sl*, on the `difference` probe contexts."""
    return np.asarray(traj["epoch"]), np.asarray(traj["alpha_anchored"])[:, sl][:, ANSWERS].mean(axis=1)


def first_rise(traj: dict) -> float:
    ep, a = track(traj)
    above = np.flatnonzero(a > RISE)
    return float(ep[above[0]]) if len(above) else float("nan")


# Every stored run: its experiment, arm, seed, outcome and window mean.
STORED_RUNS = []
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


def held_out() -> list[dict]:
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


def outcome(arm: str, seed: int) -> dict:
    """One seed of an arm: its window mean, the call, and whether it lost the anchor, from the full edit where the
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
# The one-sided sign test that `falloff` loses more often, over the seeds where exactly one arm lost.
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
# The one-sided sign test that `falloff` is late more often, over the seeds where only one arm was late.
WINDOW_R = float(np.corrcoef([WINDOWS[FALLOFF], WINDOWS[SOFT]])[0, 1])
FINISHED = sorted((OUT[a][s] | {"arm": a, "seed": s} for a in ARMS for s in SEEDS if OUT[a][s]["removal"] is not None),
                  key=lambda o: o["seed"])  # fmt: skip
SLOW = [o for o in FINISHED if not o["lost"]]
WRONG_PROVISIONAL = [o for o in FINISHED if (o["call"] == "lost?") != o["lost"] and o["call"].endswith("?")]


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
    alt = f"""
        Scatter of full-edit removal against the window mean for the {N_STORED} stored runs, marked by experiment.
        The kept runs sit in a cloud at removal 0.6 to 0.95 and window means 0.29 to 0.6; the lost runs sit near
        zero removal (one at 0.44) with window means below 0.15. A shaded gap between 0.15 and 0.29 holds no run.
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
        ["arm", "stored seeds", "screen seeds", "called kept", "finished", "lost ↓", "95% upper bound ↓"],
        rows,
        f"""
        **How often each arm lost the anchor.** The stored seeds ({OLD_SEEDS[0]} to {OLD_SEEDS[-1]}) trained to
        400 epochs in earlier experiments; the screen seeds ({SEEDS[0]} to {SEEDS[-1]}) stopped at epoch
        {ex.STOP_EPOCH}, and those not called kept were finished at full length and scored with the full edit. The
        bound is the one-sided Clopper–Pearson bound on the loss rate.
        """,
    )


def schedule_of(arm: str) -> dict:
    """λ_a and τ_p along the first epochs, shared by every seed of an arm; a fixed τ_p is drawn at its value."""
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
    alt = f"""
        α at the example answers at slice {SL} against epoch, up to epoch {ex.STOP_EPOCH}, one column per arm,
        {N} runs each, the other arm ghosted. Under falloff most runs rise from near zero to about 0.5 between
        epochs 16 and 24; one jumps up only at epoch 44, three level off near 0.2 inside the band between the two
        dashed rules, and one dashed run stays near zero. Under the soft start every run rises a few epochs later,
        between epochs 20 and 28, tightly bunched, and all sit above the upper rule through most of the window,
        with two runs dipping to about 0.22 at single records late in it. Underneath,
        the anchor weight rises from 0 to 0.1 by epoch 40 in both arms; the temperature stays at 0.1 under falloff
        and falls from 1 to 0.1 by epoch 40 under the soft start.
    """
    return screen_draw(data, alt)


@memo
def screen_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="loss-rate-screen",
        alt_text=alt_text,
        caption=f"""
            **The example answers through epoch {ex.STOP_EPOCH}**: α at the example answers at slice {SL} on the
            `{ex.ANCHORED_OP}` probe contexts, one column per arm and one line per seed (the stored seeds and the
            screen seeds alike), dashed where the run lost the anchor; the other arm is ghosted behind. The shaded
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
    alt = f"""
        Two dot plots, one dot per seed for each arm. Left, the window mean: soft start seeds lie between 0.33 and
        0.58; falloff seeds mostly between 0.42 and 0.58, with three near 0.2 between the two dashed rules and one
        at zero. Right, the first epoch over {RISE:g}: soft start seeds all at 24 or 28; falloff seeds mostly at 20
        or 24, with a few at 28 to 44.
    """
    return settle_draw(data, alt)


@memo
def settle_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="loss-rate-settle",
        alt_text=alt_text,
        caption=f"""
            **How far above the cut, and how soon.** Left: the window mean of every seed of each arm, with the
            highest stored lost run and the lowest stored kept one as rules. Right: the first recorded epoch at
            which α at the example answers at slice {SL} goes over {RISE:g} (α is recorded every four epochs). Each
            dot is a seed, jittered sideways; the thin bar spans the seeds and the large marker is the seed mean.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (wax, rax) = plt.subplots(1, 2, figsize=(5.6, 2.8), layout="constrained")
        rng = np.random.default_rng(0)
        for i, arm in enumerate(data["arms"]):
            ink, mk = INK[arm["arm"]], MARK[arm["arm"]]
            for ax, key in ((wax, "window"), (rax, "rise")):
                v = np.asarray(arm[key], float)
                v = v[np.isfinite(v)]
                ax.plot([i, i], [v.min(), v.max()], color=ink, lw=2.5, alpha=0.25, solid_capstyle="butt", zorder=1)
                ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=ink, alpha=0.5, lw=0, zorder=2)
                ax.scatter([i], [v.mean()], marker=mk, s=40, color=ink, edgecolor=RULE, lw=0.6, zorder=3)
        for y in (data["lost_max"], data["kept_min"]):
            wax.axhline(y, color=RULE, lw=0.7, ls="--", zorder=0)
        wax.set_ylabel(f"window mean, epochs {ex.WINDOW[0]}–{ex.WINDOW[1]}", fontsize=8)
        rax.set_ylabel(f"first epoch over {data['rise']:g}", fontsize=8)
        wax.set_ylim(0, 0.7)
        for ax in (wax, rax):
            ax.set_xticks(range(len(data["arms"])), [a["name"] for a in data["arms"]], fontsize=7)
            ax.set_xlim(-0.6, len(data["arms"]) - 0.4)
            ax.tick_params(labelsize=7)
        return fig

    return _plot()


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
        ["arm", "window mean ↑", f"first epoch over {RISE:g}", f"seeds not over it by epoch {ex.STOP_EPOCH}"],
        rows,
        f"""
        **How the runs settle**, seed mean and range over all {N} seeds of each arm; the first epoch is over the
        seeds that got there by epoch {ex.STOP_EPOCH}. The warm-up of the anchor weight ends at epoch 40.
        """,
    )


def finished_figure() -> str:
    def full(t: dict) -> dict:
        ep, a = track(t)
        return {"epoch": ep.tolist(), "answers": a.tolist()}

    t0 = FINISHED_TRAJ[ex.label_of(FINISHED[0]["arm"], FINISHED[0]["seed"], None)]["traj"]
    data = {
        "runs": [
            full(FINISHED_TRAJ[ex.label_of(o["arm"], o["seed"], None)]["traj"]) | {"seed": o["seed"], "lost": o["lost"]}
            for o in FINISHED
        ],
        "ghosts": [full(OUT[FALLOFF][s]["traj"]) for s in OLD_SEEDS],
        "schedule": {
            "epoch": t0["epoch"],
            "weight": t0["weight"],
            "anti": np.asarray(t0["anti_weight"])[:, SL].tolist(),
        },
        "window": list(ex.WINDOW),
    }
    alt = f"""
        α at the example answers at slice {SL} against epoch over the full 400 epochs, for the {len(FINISHED)}
        finished `falloff` runs, with the eight stored `falloff` runs as ghosts that rise to about 0.5 by epoch 30
        and to about 0.6 by the end. Three solid lines sit near 0.2 until about epoch 200, then climb to about 0.6 by
        epoch 300, level with the ghosts. One dashed line stays near zero throughout. Underneath, the anchor weight
        is flat at 0.1 from epoch 40 to 360, and the anti weight at slice {SL} falls from about 0.25 to 0.07 over
        the same span, most steeply between epochs 100 and 250, then to zero by epoch 400.
    """
    return finished_draw(data, alt)


@memo
def finished_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="loss-rate-finished",
        alt_text=alt_text,
        caption=f"""
            **The finished runs, to the end of training**: α at the example answers at slice {SL} on the
            `{ex.ANCHORED_OP}` probe contexts, one line per `falloff` screen run that was not called kept, dashed
            where the run lost the anchor; the eight stored `falloff` runs are ghosts behind. The shaded window is
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
            ax.plot(r["epoch"], r["answers"], color=INK[FALLOFF], lw=0.9, ls="--" if r["lost"] else "-", zorder=2)
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
We trained 32 new seeds each of `falloff` and the soft start, stopped each run early, and judged it by whether the example answers had come onto e₁. `falloff` lost the anchor on one seed of 40, and the soft start on none. A few `falloff` runs came onto e₁ only late in training, which the soft start never did.
///

The recipe of record, `falloff` from [the split-anti-seeds experiment](/docs/m2/split-anti-seeds/report.py), had kept the anchor on all eight of its seeds, which fits a loss rate as high as three in ten. The soft start of [soft-pool](/docs/m2/soft-pool/report.py) (`step, soft τ`) also kept it on all eight.

Several close variants lost the anchor on one to three seeds of eight, in split-anti-seeds, [anchor-ramp](/docs/m2/anchor-ramp/report.py), and soft-pool. With the recipe known this loosely, an arm that loses one seed in eight couldn't be told apart from it.

Full runs at enough seeds to narrow this are costly. But in those three experiments the lost runs had parted from the kept ones within the first sixty or so epochs. So we first check, on the {N_STORED} stored runs whose outcome we know, whether an early call made from the alignment agrees with the full edit.

Then we train {len(SEEDS)} new seeds each of `falloff` and the soft start on the full-length schedule, stop them at epoch {ex.STOP_EPOCH}, and call each one. The runs the call can't settle are trained to the end and scored.
"""

# %%

rf"""
## Observations

- [The early call on the stored runs (E1)](#the-early-call-on-the-stored-runs-e1): how well the example answers had come onto e₁ by epochs {ex.WINDOW[0]} to {ex.WINDOW[1]} sorts every stored run the way the full edit does, with a clear gap between the lost runs and the kept ones. It still does when each experiment is held out of choosing the cut, though one slow run then lands close to the cut.
- [How often each arm loses the anchor (E2)](#how-often-each-arm-loses-the-anchor-e2): `falloff` lost the anchor on one seed of {N}, and the soft start on none. That puts the loss rate of `falloff` under about one in nine and that of the soft start under about one in fourteen. The two can't be told apart on losses alone.
- [The slow runs (E3)](#the-slow-runs-e3): under the soft start the example answers came onto e₁ within a few epochs of each other on every seed. Under `falloff` most runs were a little quicker, but three hung partway until after epoch 200. They came up once the anti weight had fallen to about half, and kept the anchor.

## Scope

This is an exploratory study, with no preregistration and no gate. It covers two arms, both with the step pull on slices 2 to 4 and the `falloff` anti weight on every slice. `falloff` itself has a fixed pooling temperature $\tau_p$ = 0.1.[^temp] The soft start sharpens $\tau_p$ from 1 to 0.1 over the 40-epoch warm-up of the anchor weight. Everything else is the training of split-anti-seeds.

[^temp]: Pooling temperature: the pull on a context is a softmin-weighted mix over its positions. At 0.1 nearly all the weight goes to the position closest to e₁; at 1 it is spread over many.

The new runs are at seeds {SEEDS[0]} to {SEEDS[-1]} in both arms, so the two are paired by initialization, batches and label draws. Each follows the full 400-epoch schedule (learning rate and both term weights as in a full run) and stops at epoch {ex.STOP_EPOCH}. With the stored runs at seeds {OLD_SEEDS[0]} to {OLD_SEEDS[-1]}, each arm has {N} seeds.

Stopping early leaves the run up to the stop as it was in the stored full run. Two seeds of each arm ({" and ".join(map(str, ex.REPRO_SEEDS))}) were trained again and stopped at epoch {ex.STOP_EPOCH}, and α at every recorded epoch up to the stop matches the stored run to the last digit in all four (largest difference {REPRO_MAX:g}).

A screen run that is not firmly kept (called lost, or between the stored lost and kept runs) was trained again to the end and scored with the full edit, so every loss counted is a measured one. A kept call is not checked this way, and rests on the timing of the stored losses.

The late loss of soft-pool brought its example answers back to about 0.4 by epoch 100 and still removed almost nothing. So a lost run whose example answers came back by epochs {ex.WINDOW[0]} to {ex.WINDOW[1]} would be wrongly called kept. No stored run did that.

The finished runs have no control at their seed, so their removal is netted against the seed mean of the eight controls of ex-2.2.23. The controls differ by up to about an eighth in removal, which matters only for a run near the half-removal line, and no finished run came near it.

We don't measure removal, spill or the task score on the screen runs: the question is the loss rate, and those are known for both arms from the earlier experiments.

## The measurements

- Kept or lost (the outcome, on stored and finished runs): a run lost the anchor when the full edit, along e₁ at every position and slice, removes less than half of `difference`, net of the control, as in split-anti-seeds.
- The window mean (the call): α (the cosine of the state with e₁) at the example answers of the `difference` probe contexts at slice {SL}, averaged over the six records (one every four epochs) at epochs {ex.WINDOW[0]} to {ex.WINDOW[1]}. Slice {SL} is the deepest slice every stored arm pulls, both screen arms included. A run is called lost below {ex.LOST_MAX:g}, the highest window mean of any stored lost run, and kept at {ex.KEPT_MIN:g} or above, the lowest of any stored kept run. Every run in between was finished, so the full edit decides.
- The loss rate: losses over seeds, with the one-sided 95% upper bound from the binomial (Clopper–Pearson[^cp]). The arms are compared on the seeds where one kept the anchor and the other lost it, with a one-sided exact (sign) test. At these counts it takes about five such seeds all one way to reach p < 0.05, so the screen can bound each rate well but can show a difference between the arms only if it is large.

[^cp]: Clopper–Pearson: an exact interval for a proportion, built from the binomial itself, so it holds at small counts and at zero losses.
"""

# %%

rf"""
## The early call on the stored runs (E1)

Does the window mean tell the lost runs from the kept ones? We plot it against removal under the full edit for all {N_STORED} stored runs, {N_STORED_LOST} of them lost.

{calib_figure()}

It does, with room to spare. Every lost run sits near zero on the window mean, every kept run well above it, and no run falls in the gap between them. Most lost runs ended with almost no removal.

The lost run near the half-removal line is seed 710 of `ramp from 1`, which the anchor-ramp report counted as borderline; on the window mean it sits with the other lost runs. The kept run nearest the gap is seed {SLOWEST["seed"]} of `ramp from emb`, whose example answers came onto e₁ only after epoch 200.

A cut chosen on the same runs it sorts could fit them by chance, so we chose it on two experiments and sorted the third:

{held_table()}

Each held-out experiment is sorted with no error. The closest call is the slow kept run of anchor-ramp, which clears the cut chosen on the other two experiments by a few hundredths.

Windows that end later separate the runs less well, because the late loss of soft-pool brought its example answers partway back by epoch 100 without the removal coming back. Windows that end at epoch 52 or earlier leave little or no gap, since some kept runs were still rising. So the stop at epoch {ex.STOP_EPOCH} sits where the stored runs separate best, and we took it as given for the screen.
"""

# %%

rf"""
## How often each arm loses the anchor (E2)

How many of the {N} seeds of each arm lost the anchor, how high could the loss rate be, and do the two arms differ? The figure shows the example answers through epoch {ex.STOP_EPOCH} on every seed of each arm.

{screen_figure()}

{count_table()}

`falloff` lost the anchor on one seed, {LOST[FALLOFF][0]}, the way the anchor-ramp losses and the uniform holds of split-anti-seeds did. Its query answer came onto e₁ by epoch 20 and stayed there, its example answers never left zero, and the full edit removed nothing. The soft start kept the anchor on that seed and every other.

So with 40 seeds each, the loss rate of `falloff` is under about {upper(len(LOST[FALLOFF]), N):.0%} and that of the soft start under about {upper(len(LOST[SOFT]), N):.0%}. One seed where only one arm lost is far from enough to say which loses more (p = {SIGN_P:.2g}).

Three more `falloff` runs fell between the stored lost and kept runs. All three kept the anchor when finished, with removal close to that of the other kept runs ({fmt([o["removal"] for o in SLOW])}). Two sat below the middle of the gap, so a single cut there would have counted three losses in place of one.
"""

# %%

rf"""
## The slow runs (E3)

What set those three `falloff` runs apart, and does the soft start have runs like them? We look at the window mean, and at the first recorded epoch at which α at the example answers goes over {RISE:g} (as in soft-pool).

{settle_figure()}

{settle_table()}

Under the soft start the example answers came onto e₁ within a few epochs of each other on every seed, between epochs {np.nanmin(RISES[SOFT]):.0f} and {np.nanmax(RISES[SOFT]):.0f}, as in soft-pool. Under `falloff` most runs got there a few epochs sooner, but the spread is wider: one run jumped up only at epoch 44, and {len(LATE[FALLOFF])} had not got there by the stop, the lost run and the three slow ones.

All four are seeds where the soft start got there on time. If the arms didn't differ, a split of four seeds to none would come up about one time in sixteen (p = {LATE_P:.2g}), and we chose this measure after seeing the runs. So this fits the soft start being more even, without showing it.

These are the same four runs that E2 finished, so the split restates that result by timing rather than adding to it. The evenness is also only in when the example answers arrive. The level they hold through the window spreads about as widely under the soft start, whose lowest run dips into the band at single records.

Seeds that were slow under one arm tended to be a little slow under the other (the window means of the two arms are correlated over seeds, r = {WINDOW_R:.2f}). The soft start run with the lowest window mean is seed 724, one of the slow `falloff` runs.

The finished runs show what the slow ones did after the stop:

{finished_figure()}

The three slow runs held their example answers about a third of the way to the level of the other runs until after epoch 200. Each then jumped up within a few epochs, at different times between epochs 200 and 280, and kept the anchor. The lost run stayed off e₁ to the end.

The jumps come once the anti weight on slice {SL} has fallen to about half its plateau, which fits the anti term holding them back. But the anti weight falls over most of training, so the timing alone doesn't single it out. The jumps are also abrupt where the fall is smooth, and nothing here tests the idea.
"""

# %%

r"""
## Discussion

The screen bounds the loss rate of `falloff` at about one in nine, down from three in ten with eight seeds, and that of the soft start at about one in fourteen. The one loss was of the kind seen before: the query answer takes the pull and the example answers never come onto e₁. The arms can't be told apart on losses, and telling apart rates this low would take several times as many seeds.

The soft start was more even in how the example answers came onto e₁, and every `falloff` seed that was slow or lost was on time under it. That fits the explanation offered in soft-pool: a soft pool holds the pull back until the example answers can share it. The slow runs and the losses of `falloff` may also be two degrees of one thing. If so, the slow runs would be the cheaper measure of it, since there are more of them than losses.

The early call held up on the screen as far as we checked it: every run it was unsure of was finished, and the finished ones sorted cleanly by removal. But the slow `falloff` runs sat in the band, where the stored runs had only one run nearby, and a single cut would have miscounted two of them. A later window would let a late loss like the one in soft-pool pass as kept.
"""
