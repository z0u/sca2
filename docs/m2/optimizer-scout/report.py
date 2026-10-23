# ruff: noqa: B018
# title: Optimizers with no privileged basis

# An exploratory scout: the un-anchored control under three optimizers whose update is the same in every
# orthonormal basis of the residual coordinates, at three learning rates and two seeds. `experiment.py` beside this
# script finds each rate, trains, and takes the basis measurement of rotated-basis; the Adam controls at the same
# seeds come from ex-2.2.23.
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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


FINDER = {r["label"]: r for r in fetch_json(ex.FINDER_REF)["runs"]}
TRAJ = fetch_json(ex.TRAJ_REF)
BASIS_BODY = fetch_json(ex.BASIS_REF)
BASIS = {r["label"]: r for r in BASIS_BODY["runs"]}
DESIGN = BASIS_BODY["design"]
PRINCIPAL = {r["label"]: r for r in fetch_json(ex.PRINCIPAL_REF)["runs"]}
# The validation loss of ex-2.2.23's Adam controls at the same seeds, for the reference line in E1.
ADAM_LOSS = {
    s: fetch_json(ex.ex2223.TRAJ_REF)[ex.ex2223.label_of("control", ex.EPOCHS, s)]["val_loss"] for s in ex.SEEDS
}

OPS: tuple[str, ...] = tuple(DESIGN["ops"])
SEEDS: tuple[int, ...] = ex.SEEDS
ARMS = tuple(a.name for a in ex.ARMS)
NAME = {"adam": "Adam"} | {a.name: a.shown_as.replace("SGD with momentum", "SGD") for a in ex.ARMS}
FACTORS: tuple[float, ...] = ex.RATE_FACTORS
TOP = FACTORS[0]  # The top rung, three times the finder's rate.
FACTOR_SHOWN = {3.0: "3×", 1.0: "1×", 1 / 3: "⅓×"}
FOUND = {a: FINDER[f"finder-{a}"]["best_lr"] for a in ARMS}
C = 64  # The width of the residual stream.
SYNTAX = ["?", ",", "\n", "="]
SEPARATORS = ["?", ",", "\n"]
SHOWN = {"?": "`?`", ",": "`,`", "\n": "⏎", "=": "`=`"}
SEED_RANGE = 0.02  # The seed range of the stored Adam controls in the task score (split-anti-seeds).
MEMO_SGD_LR = 0.3  # The lowest rate the short SGD runs of the memo used.


def label_of(arm: str, factor: float, seed: int) -> str:
    return f"adam-s{seed}" if arm == "adam" else ex.label_of(arm, factor, seed)


def basis(arm: str, factor: float, seed: int) -> dict:
    return BASIS[label_of(arm, factor, seed)]


def clean(arm: str, factor: float, seed: int) -> np.ndarray:
    """The expected exact match of each op before any edit, as the basis measurement takes it."""
    return np.array(basis(arm, factor, seed)["clean"])


def change(arm: str, factor: float, seed: int) -> np.ndarray:
    """The task score of each op, less that of the Adam control at the same seed."""
    return clean(arm, factor, seed) - clean("adam", 1.0, seed)


def val_loss(arm: str, factor: float, seed: int) -> list[float]:
    return ADAM_LOSS[seed] if arm == "adam" else TRAJ[label_of(arm, factor, seed)]["val_loss"]


def top_share(arm: str, factor: float, seed: int, key: str = "energy_axes") -> float:
    """The share of the stream energy that the largest direction holds at the last slice."""
    return float(np.max(basis(arm, factor, seed)[key][-1]))


def kurtosis(arm: str, factor: float, seed: int, key: str = "kurt_axes") -> float:
    """The mean kurtosis along the directions of a basis, averaged over slices."""
    return float(np.mean(basis(arm, factor, seed)[key]))


def axis_costs(arm: str, factor: float, seed: int) -> np.ndarray:
    """The cost of projecting out each coordinate axis at full dose, as the mean drop over ops."""
    return np.array(basis(arm, factor, seed)["cost_axes"]).mean(axis=1)


def random_costs(arm: str, factor: float, seed: int) -> np.ndarray:
    """The same for each of the random directions."""
    return np.array(basis(arm, factor, seed)["cost_random"]).mean(axis=1)


def token_cosine(arm: str, factor: float, seed: int, token: str, key: str = "emb_axis_max") -> float:
    b = basis(arm, factor, seed)
    return float(b[key][b["tokens"].index(token)])


def token_axis(arm: str, factor: float, seed: int, token: str) -> int:
    b = basis(arm, factor, seed)
    return int(b["emb_axis_arg"][b["tokens"].index(token)])


def principal(arm: str, factor: float, seed: int) -> dict:
    return PRINCIPAL[label_of(arm, factor, seed)]


def principal_top(arm: str, factor: float, seed: int, key: str = "shares") -> float:
    """The share of the stream energy that the top principal direction holds at the last slice."""
    return float(principal(arm, factor, seed)[key][-1][0])


def on_top(arm: str, factor: float, seed: int, token: str) -> float:
    """The cosine of a syntax token embedding with the top principal direction at the last slice."""
    p = principal(arm, factor, seed)
    return float(p["syntax_on_top"][p["syntax"].index(token)])


def pair_cosine(arm: str, factor: float, seed: int, a: str, b: str) -> float:
    """The cosine between two syntax token embeddings."""
    p = principal(arm, factor, seed)
    return float(p["syntax_gram"][p["syntax"].index(a)][p["syntax"].index(b)])


def mean_range(values) -> tuple[float, float, float]:
    v = np.asarray(list(values), float)
    return float(v.mean()), float(v.min()), float(v.max())


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def fmt(values, spec: str = ".2f") -> str:
    half = 0.5 * 10 ** -int(spec[-2])  # half the last printed digit, so a value that prints as zero has no sign
    m, lo, hi = (0.0 if abs(v) < half else v for v in mean_range(values))
    return f"{num(m, spec)} ({num(lo, spec)} to {num(hi, spec)})"


def span(values, spec: str = ".2f") -> str:
    """The range of *values*, as "lo to hi", or one number where both ends print the same."""
    _, lo, hi = mean_range(values)
    return num(lo, spec) if num(lo, spec) == num(hi, spec) else f"{num(lo, spec)} to {num(hi, spec)}"


def count_of(k: int, n: int) -> str:
    """*k* of *n* in words where it reads better."""
    return "none" if k == 0 else "all" if k == n else f"{k} of {n}"


def smooth(v, width: int = 9) -> np.ndarray:
    """A centered running mean over *width* epochs, shorter at the ends."""
    v = np.asarray(v, float)
    k = np.ones(width)
    return np.convolve(v, k, "same") / np.convolve(np.ones_like(v), k, "same")


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    def code(c: str) -> str:
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", c)

    head, rows = [code(h) for h in head], [[code(c) for c in r] for r in rows]
    ths = "".join(f"<th{' class=num' if i > 1 else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i > 1 else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def ink(arm: str, factor: float = TOP) -> str:
    """An optimizer in its own hue, with the rate as a shade: the top rung strongest. Call inside a plot."""
    if arm == "adam":
        return light_dark("#333333", "#dddddd")
    shades = {
        "adam-pooled": (("#0b3c73", "#3f7fc4", "#9cc0e8"), ("#a9cdf5", "#5f95d0", "#35587f")),
        "muon": (("#8c1d12", "#d0533f", "#f2a99b"), ("#ffb3a5", "#e8705c", "#97423a")),
        "sgd": (("#0b5048", "#2a9d8f", "#9bd6cd"), ("#9fe6d8", "#4fb3a2", "#2f6b62")),
    }
    return light_dark(*shades[arm])[FACTORS.index(factor)]


def hue(arm: str) -> str:
    """An optimizer in its middle shade, for figures where color names the optimizer alone."""
    return ink(arm, FACTORS[1]) if arm != "adam" else ink(arm)


MARK = {"adam": "D", "adam-pooled": "o", "muon": "^", "sgd": "s"}
RATE_MARK = {3.0: "o", 1.0: "s", 1 / 3: "v"}


def ghost() -> str:
    return light_dark("#bbbbbb", "#555555")


# --- Summaries the prose quotes ---------------------------------------------------------------------------------

TASK = {a: {f: [float(change(a, f, s).mean()) for s in SEEDS] for f in FACTORS} for a in ARMS}
WORST = {a: {f: [float(change(a, f, s).min()) for s in SEEDS] for f in FACTORS} for a in ARMS}
FINAL = {"adam": {1.0: [val_loss("adam", 1.0, s)[-1] for s in SEEDS]}} | {
    a: {f: [val_loss(a, f, s)[-1] for s in SEEDS] for f in FACTORS} for a in ARMS
}
# The run with the lowest validation loss at the end, seed mean; the prose names it.
LOWEST = min(((a, f) for a in FINAL for f in FINAL[a]), key=lambda k: np.mean(FINAL[k[0]][k[1]]))
# The rung with the best task score, per optimizer and seed.
BEST_RUNG = {a: {s: max(FACTORS, key=lambda f: clean(a, f, s).mean()) for s in SEEDS} for a in ARMS}
TOP_BEST = all(BEST_RUNG[a][s] == TOP for a in ARMS for s in SEEDS)
# In the last zoom of the finder, the rate where the smoothed loss was lowest, against the top of the zoom range.
FINDER_LOW_AT_TOP = all(
    np.argmin(FINDER[f"finder-{a}"]["history"][-1]["losses"]) == len(FINDER[f"finder-{a}"]["history"][-1]["lrs"]) - 1
    for a in ARMS
)
# E1 rests on these: the top rung did best for every optimizer at both seeds, the finder's last zoom had its lowest
# loss at the top of its range, and Muon at the top rung has the lowest validation loss of any run.
assert TOP_BEST and FINDER_LOW_AT_TOP and LOWEST == ("muon", TOP)
SGD_TOP_LR = FOUND["sgd"] * TOP

EQUIVARIANT = [(a, f, s) for a in ARMS for f in FACTORS for s in SEEDS]
ADAM_RUNS = [("adam", 1.0, s) for s in SEEDS]


def runs_of(arm: str, factors=FACTORS) -> list[tuple[str, float, int]]:
    return [(arm, f, s) for f in factors for s in SEEDS]


# Where a run puts the separators: the runs where at least two of `?`, `,` and ⏎ have their largest component on
# the same axis, and the runs whose largest coordinate at the last slice is the axis of one of the syntax tokens.
SHARED = [r for r in EQUIVARIANT if len({token_axis(*r, t) for t in SEPARATORS}) < len(SEPARATORS)]


def top_on_syntax(r: tuple[str, float, int]) -> bool:
    return int(np.argmax(basis(*r)["energy_axes"][-1])) in {token_axis(*r, t) for t in SYNTAX}


TOP_ON_SYNTAX = {a: [r for r in runs_of(a) if top_on_syntax(r)] for a in ARMS} | {
    "adam": [r for r in ADAM_RUNS if top_on_syntax(r)]
}
# By chance: two of three tokens on the same one of C axes; and the top axis among four syntax axes.
P_SHARED = 1 - (C - 1) * (C - 2) / C**2
P_TOP_ON_SYNTAX = len(SYNTAX) / C

# The largest cosine of a random unit vector with any of the C axes: the 5th and 95th percentiles.
_v = np.random.default_rng(0).standard_normal((100_000, C))
RANDOM_COS = np.percentile(np.abs(_v / np.linalg.norm(_v, axis=1, keepdims=True)).max(axis=1), [5, 95]).tolist()
del _v

# The spike along the stream's own principal directions: its share of the energy at the last slice, for the Adam
# controls and for every run of the new optimizers.
ADAM_P_TOP = [principal_top(*r) for r in ADAM_RUNS]
EQ_P_TOP = [principal_top(*r) for r in EQUIVARIANT]
EQ_P_TOP_CENTERED = [principal_top(*r, key="shares_centered") for r in EQUIVARIANT]
# The separators `,` and ⏎: their cosine with each other, and with the top principal direction at the top rung.
SEP_PAIR = {
    a: [pair_cosine(*r, ",", "\n") for r in (ADAM_RUNS if a == "adam" else runs_of(a))] for a in ("adam", *ARMS)
}
SEP_ON_TOP = {
    a: [on_top(*r, t) for r in (ADAM_RUNS if a == "adam" else runs_of(a, (TOP,))) for t in (",", "\n")]
    for a in ("adam", *ARMS)
}
EQ_ON_TOP = {a: [on_top(*r, "=") for r in (ADAM_RUNS if a == "adam" else runs_of(a))] for a in ("adam", *ARMS)}
EQ_PAIR = {a: [pair_cosine(*r, "?", "=") for r in (ADAM_RUNS if a == "adam" else runs_of(a))] for a in ("adam", *ARMS)}


# %%

rf"""
# Optimizers with no privileged basis

/// tip |
<!-- lede -->
Adam's per-coordinate scaling makes the coordinate axes of the residual stream a privileged basis, and the model parks its syntax tokens one-hot on them. This scout trains the un-anchored control under three optimizers whose update is the same in every orthonormal basis of the residual coordinates. Under all three the axes became ordinary directions, and the stream was no more isotropic for it: along its own principal directions the same spike is there, holding two fifths or more of the energy at the last slice, with the separator tokens on it. Muon learned the task as well as Adam once its rate was raised; pooled Adam came close, and SGD was never tried at a rate high enough to judge.
///

The question comes from the memo of 2026-10-23 on whether the coordinate axes are special (in the project files), and from the discussion after it: the prior is that an isotropic stream would be easier to reason about, and the open question is what it costs. The scout runs eighteen new 400-epoch control runs and three learning-rate finder runs, at the two seeds where ex-2.2.23's Adam controls are stored as the reference. No anchored runs: an optimizer that keeps the stream free of a privileged basis and learns the task as well is a candidate for an anchored arm later, which would need its own tuning of the term weights.

The stream is isotropic under none of them. The energy spectrum along the stream's own principal directions, taken for every run, has one direction holding {span(EQ_P_TOP)} of the energy at the last slice under the new optimizers, against {span(ADAM_P_TOP)} under Adam, and the separators `,` and ⏎ lie on it in every run. So the spike is a property of the model and the task, and the optimizer decides only whether it sits on a coordinate axis. One cost follows: with the axes ordinary, projecting out a coordinate axis costs the task more than an ordinary axis costs under Adam.
<!-- REVIEW: the plan asked whether the stream comes out isotropic. The basis measurement (energy and kurtosis along the axes and along one random basis, the removal cost of axes and random directions) can show a privileged basis but not a spike off both bases: Adam's spectrum along the random basis, a known spike seen from a basis it is not aligned with, looks even. So a spectrum along the stream's own principal directions was added to the experiment after the first draft, and the isotropy claims rest on it. Verify: `principal_one` in experiment.py; E2's second figure and the last column of its table. -->
"""

# %%

rf"""
## Observations

- [Does the task suffer? (E1)](#does-the-task-suffer-e1): Muon at three times the finder's rate matches Adam within the seed range, and pooled Adam at that rate trails by about a hundredth. Both did best at the top rung on both seeds, so the bracket may sit below their best rates. SGD trails by about half of the score at every rate, and was still improving with the rate at the top of the bracket.
- [Is the stream isotropic? (E2)](#is-the-stream-isotropic-e2): no coordinate holds more of the energy than the largest direction of a random basis does, and the kurtosis and the removal cost are about the same along the axes as along random directions. So the axes are ordinary directions. Along the stream's own principal directions, though, the top direction holds {span(EQ_P_TOP)} of the energy at the last slice, as much as under Adam ({span(ADAM_P_TOP)}): the spike has moved off the axes, and the stream is not isotropic. Removing an axis now costs what a random direction does, which under Muon at the top rung is two to three times what an ordinary axis costs under Adam.
- [Where the syntax tokens go (E3)](#where-the-syntax-tokens-go-e3): the syntax tokens leave the axes; their largest cosine with any axis is what a random unit vector gives. The separators `,` and ⏎ still share a direction, at a cosine of {span(v for a in ARMS for v in SEP_PAIR[a])} in every run, and it is the top principal direction. Under Muon `=` leans on it too, where under Adam `=` sits apart.

## Scope

This is an exploratory scout, with no preregistration and no gate. Three optimizers, each on the control at model seeds 704 and 705 for the full 400-epoch schedule, at three learning rates: the rate the finder of ex-2.2.17 suggests for that optimizer at seed 704, three times it, and a third of it. The finder's 300-step suggestion sat above the rate the ladders chose for Adam, and for SGD it may sit below the rate a long run wants (the memo's short SGD runs used rates of 0.3 to 1.0), so the bracket reaches both ways and the three rungs say how sensitive each optimizer is. Eighteen runs in all, against ex-2.2.23's two Adam controls at the same seeds, which share the initialization, the batches, and the model. Two seeds separate an optimizer from Adam only where the gap is larger than the seed range of the stored controls (about 0.02 in the task score; see split-anti-seeds' control columns); the scout is for ruling optimizers in or out, not for ranking close ones.

The three optimizers:

- Pooled Adam: Adam with its second moment pooled over the residual axis of each matrix that reads or writes the stream, so the scaling is one number per row (of a reading matrix) or column (of a writing matrix) and the same whichever orthonormal basis the residual coordinates are written in. The gains and the other axis of each matrix are scaled per coordinate as in Adam.
- Muon: each matrix's momentum is orthogonalized (a Newton–Schulz iteration toward the nearest orthogonal matrix) before the step, which is equivariant to a rotation on either side of the matrix. Every matrix gets this, the embedding and readout tables included, where the usual Muon recipe leaves the tables to Adam; the gains are updated with Adam at the recipe's rate.
- SGD with momentum 0.9, the plain reference, with no per-coordinate scaling at all.

A test in the repo checks that the rotated-basis trick changes nothing for these three and does change Adam.

## The measurements

- The task: the expected exact match of each op before any edit, as the basis measurement takes it on held-out contexts, against the Adam control at the same seed; and the validation loss along training.
- Isotropy, from the basis measurement of rotated-basis: the share of the stream's energy that each coordinate axis holds, sorted, against the same along a random orthonormal basis; the mean kurtosis along each (a privileged basis has a higher kurtosis along its axes than along random directions; a Gaussian has 3); and the cost of projecting out one direction at full dose, for each coordinate axis and for sixteen random directions. The memo's Adam reference is one coordinate at about half of the energy, a kurtosis of 4 to 5 along the axes against 3 along random directions, and a typical axis at a cost of 0.03 against 0.08 to 0.13 for a random direction.
- Where the tokens sit: each token embedding's largest cosine with a coordinate axis, and which axis that is. Under Adam the syntax tokens `?`, `,` and `⏎` sit at a cosine near 1 on one coordinate and `=` on another.
- The spectrum along the stream's own principal directions, added for this report: the eigenvalues of the second-moment matrix of the states at each slice, over the positions of 1000 held-out contexts, as shares of the total. A spike shows here whichever direction it sits on. With it, the cosine of each syntax token embedding with the top direction at the last slice, and the pairwise cosines among the syntax embeddings.
<!-- REVIEW: the task score was planned as embedding-lean's measurement, but the published results of `measure_one` (edits off) hold no task score, so the report takes the basis measurement's `clean`, the same expected exact match on held-out contexts, for the new runs and the Adam controls alike. Verify: `clean` in `basis_one` of rotated-basis. -->
"""

# %%


def loss_figure() -> str:
    epochs = len(ADAM_LOSS[SEEDS[0]])
    data = {
        "epochs": epochs,
        "adam": [ADAM_LOSS[s] for s in SEEDS],
        "arms": [
            {"arm": a, "name": NAME[a], "rates": {str(f): [val_loss(a, f, s) for s in SEEDS] for f in FACTORS}}
            for a in ARMS
        ],
    }
    alt = f"""
        Validation loss against epoch, one panel each for pooled Adam, Muon and SGD, with three rates per panel and the
        Adam control as a dashed line. In the pooled Adam and Muon panels the top rate ends on or a little under the
        Adam line, near {num(np.mean(FINAL["muon"][TOP]))}, the finder's rate just above it, and a third of it higher
        still. In the SGD panel all three rates stay well above the Adam line, ending between
        {num(min(min(FINAL["sgd"][f]) for f in FACTORS))} and {num(max(max(FINAL["sgd"][f]) for f in FACTORS))},
        lower for a higher rate.
    """
    return loss_draw(data, alt)


@memo
def loss_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="optimizer-scout-loss",
        alt_text=alt_text,
        caption="""
            **Validation loss along training**, one panel per optimizer. Each rate is a shade of the optimizer's hue,
            strongest for three times the finder's rate; the two seeds are hairlines, and the seed mean, smoothed over nine
            epochs, is drawn over them with a marker every 50 epochs. The dashed line is the seed mean of the Adam
            control at the recipe's rate, smoothed the same way.
        """,
    )
    def _plot() -> plt.Figure:
        arms = data["arms"]
        x = np.arange(1, data["epochs"] + 1)
        fig, axes = plt.subplots(1, len(arms), figsize=(7.2, 2.6), layout="constrained", sharex=True, sharey=True)
        adam = np.mean(data["adam"], axis=0)
        for ax, arm in zip(axes, arms, strict=True):
            a = arm["arm"]
            ax.plot(x, smooth(adam), color=ink("adam"), lw=1.0, ls="--", zorder=4)
            for f in FACTORS:
                runs = np.array(arm["rates"][str(f)])
                for r in runs:
                    ax.plot(x, r, color=ink(a, f), lw=0.4, alpha=0.6, zorder=2)
                ax.plot(
                    x, smooth(runs.mean(0)), color=ink(a, f), lw=1.4, marker=RATE_MARK[f], ms=3.5, markevery=(49, 50)
                )
            ax.set_title(arm["name"], fontsize=8)
            ax.set_xlabel("epoch", fontsize=8)
            ax.set_xlim(0, data["epochs"])
            ax.set_ylim(2.15, 2.55)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("validation loss", fontsize=8)
        handles = [
            Line2D([], [], color=ghost(), lw=1.4, marker=RATE_MARK[f], ms=3.5, label=f"{FACTOR_SHOWN[f]} finder rate")
            for f in FACTORS
        ] + [Line2D([], [], color=ink("adam"), lw=1.0, ls="--", label="Adam control")]
        fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), frameon=False, fontsize=7)
        return fig

    return _plot()


def task_table() -> str:
    rows = []
    for a in ARMS:
        for i, f in enumerate(FACTORS):
            cells = [NAME[a] if i == 0 else "", FACTOR_SHOWN[f], f"{FOUND[a] * f:.2g}"]
            for k in range(len(SEEDS)):
                cells += [num(TASK[a][f][k], "+.3f"), num(WORST[a][f][k], "+.3f")]
            cells.append(num(float(np.mean(FINAL[a][f])), ".3f"))
            rows.append(cells)
    rows.append(
        ["Adam (control)", "", f"{ex.ADAM_LR:.2g}"]
        + ["·"] * (2 * len(SEEDS))
        + [num(np.mean(FINAL["adam"][1.0]), ".3f")]
    )
    head = ["optimizer", "rate", "learning rate"]
    for s in SEEDS:
        head += [f"{s}: mean ↑", f"{s}: worst op ↑"]
    head.append("val loss ↓")
    return table_html(
        head,
        rows,
        f"""
        **The task change against Adam, per run.** The rate is a multiple of the finder's suggestion for that
        optimizer, and the learning rate is the peak rate it gives. For each seed, "mean" is the expected exact match
        averaged over the {len(OPS)} ops less that of the Adam control at the same seed, and "worst op" is the largest
        drop on any one op. The last column is the validation loss at the end of training, seed mean. Arrows mark which
        direction is better.
        """,
    )


rf"""
## Does the task suffer? (E1)

Is the task learned as well without a privileged basis, at any of the three rates? If an optimizer trails Adam by more than the seed range at every rate, it is out for now; if it matches at one rate, it is a candidate. One case to keep apart: if an end rung does best at both seeds, the bracket may have missed the rate rather than the optimizer, and the finder's history (whether the loss was still falling at the end of its range) says which.

The main measure is the task change: the expected exact match averaged over the ops, less that of the Adam control at the same seed. With two seeds, a change inside the seed range of the stored controls (about 0.02) is one we cannot tell from zero. The validation loss along training shows the same order, and where each run gets there.

{loss_figure()}

Muon at three times the finder's rate matches Adam: the change is within a hundredth at both seeds, and no op falls by more than the seed range. It also ends with the lowest validation loss of any run here, though only by a few thousandths. At the finder's rate it is nearly as close, and at a third of it, it trails clearly. Pooled Adam trails a little at the top rung, by about a hundredth on the mean and twice that on its worst op, more at the finder's rate, and far at a third of it. SGD trails by about half of the score at every rate.

{task_table()}

For all three optimizers the top rung did best at both seeds, so by the reading rule above the bracket may have missed the best rate. The finder fits that: in its last zoom the loss was still falling at the top of its range for every optimizer, and its suggestion is the rate where the loss fell fastest, which sits below the rate where it was lowest. A finder ramps the rate up over one short run, so the later, higher rates have also had more steps; that alone could put the lowest loss at the top, and the finder does not place the best rate by itself. What it does say is that the suggestion was low for every optimizer here. For pooled Adam it was {FOUND["adam-pooled"]:.2g}, against the {ex.ADAM_LR:.2g} the recipe uses for plain Adam.

For pooled Adam and Muon this changes little in a scout. Muon already matches Adam, and pooled Adam is close enough that a higher rate might close the gap, which these runs do not test.

For SGD it matters more. The validation loss falls steadily as the rate rises (see the table), and the top rung, at {SGD_TOP_LR:.2g}, is still about a tenth of the lowest rate the short SGD runs of the memo used. So SGD was not tested at a useful rate, and these runs say nothing against it. They do mean that the SGD runs in E2 and E3 are models that learned little of the task.
"""

# %%


def basis_figure() -> str:
    order = ("adam", *ARMS)
    runs = {a: ADAM_RUNS if a == "adam" else [(a, TOP, s) for s in SEEDS] for a in order}
    data = {
        "seeds": list(SEEDS),
        "panels": [
            {
                "arm": a,
                "name": NAME[a] if a == "adam" else f"{NAME[a]}, {FACTOR_SHOWN[TOP]}",
                "axes": [sorted(basis(*r)["energy_axes"][-1], reverse=True) for r in runs[a]],
                "rotated": [sorted(basis(*r)["energy_rotated"][-1], reverse=True) for r in runs[a]],
                "cost_axes": [axis_costs(*r).tolist() for r in runs[a]],
                "cost_random": [random_costs(*r).tolist() for r in runs[a]],
            }
            for a in order
        ],
    }

    def am(a: str) -> list[np.ndarray]:
        return [axis_costs(*r) for r in runs[a]]

    alt = f"""
        Two rows of four panels: Adam, then pooled Adam, Muon and SGD at three times the finder's rate. Top row: the
        share of the stream energy per direction at the last slice, sorted, on a log scale, along the coordinate axes
        (solid) and along a random basis (dashed). Under Adam the first axis holds about
        {num(np.mean([top_share(*r) for r in ADAM_RUNS]))} and the dashed curve stays near
        {num(np.mean([top_share(*r, key="energy_rotated") for r in ADAM_RUNS]))}; in the other three panels the solid
        and dashed curves lie together, with the largest direction near 0.05 to 0.1. Bottom row: the cost of projecting
        out each axis (dots on the left of each seed pair) and each random direction (right). Under Adam the axes
        cluster near {num(np.median(np.concatenate(am("adam"))))}, with one outlier near
        {num(max(a.max() for a in am("adam")), ".1f")}, and the random directions sit higher. Under the other three,
        axes and random directions overlap.
    """
    return basis_draw(data, alt)


@memo
def basis_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="optimizer-scout-basis",
        alt_text=alt_text,
        caption="""
            **The energy spectrum and the cost of a direction**, Adam and each optimizer at three times the finder's
            rate. Top: the share of the stream energy at the last slice that each direction holds, sorted from the
            largest, along the coordinate axes (solid) and along a random orthonormal basis (dashed); one line per seed.
            The faint level is an even spread, 1/64. Bottom: the drop in the task score, averaged over ops, when one
            direction is projected out at full dose at every slice and position. For each seed, the 64 axes (left) and
            the 16 random directions (right) as faded dots, a thin bar over their range, and the median (axes) or mean
            (random) as the marker.
        """,
    )
    def _plot() -> plt.Figure:
        panels = data["panels"]
        fig, axes = plt.subplots(
            2, len(panels), figsize=(7.2, 4.6), layout="constrained", sharey="row", height_ratios=(1.0, 1.1)
        )
        rng = np.random.default_rng(1)
        rank = np.arange(1, C + 1)
        for col, p in enumerate(panels):
            a = p["arm"]
            color = hue(a)
            ax = axes[0, col]
            ax.axhline(1 / C, color=ghost(), lw=0.6, zorder=0)
            for sa, sr in zip(p["axes"], p["rotated"], strict=True):
                ax.plot(rank, sa, color=color, lw=1.1, zorder=3)
                ax.plot(rank, sr, color=color, lw=1.0, ls="--", alpha=0.8, zorder=2)
            ax.set_yscale("log")
            ax.set_ylim(2e-3, 0.7)
            ax.set_xlim(0, C + 1)
            ax.set_title(p["name"], fontsize=8)
            ax.set_xlabel("direction, by rank", fontsize=8)
            ax.tick_params(labelsize=7)

            ax = axes[1, col]
            ticks, labels = [], []
            for k, (ca, cr) in enumerate(zip(p["cost_axes"], p["cost_random"], strict=True)):
                for j, (v, center, alpha) in enumerate(((ca, np.median, 0.35), (cr, np.mean, 0.55))):
                    x = 2.6 * k + j
                    v = np.asarray(v)
                    ax.plot([x, x], [v.min(), v.max()], color=color, lw=1.0, alpha=0.5, zorder=2)
                    ax.plot(
                        x + rng.uniform(-0.22, 0.22, len(v)), v, "o", ms=2.0, color=color, alpha=alpha, mew=0, zorder=3
                    )
                    ax.plot(
                        x, center(v), MARK[a], ms=5, color=color, mec=light_dark("white", "#111"), mew=0.6, zorder=4
                    )
                    ticks.append(x)
                    labels.append("axes" if j == 0 else "random")
                ax.annotate(
                    f"seed {data['seeds'][k]}" if k == 0 else str(data["seeds"][k]),
                    (2.6 * k + 0.5, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -2),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    color=light_dark("#666", "#999"),
                )
            ax.set_xticks(ticks, labels, fontsize=6.5, rotation=90)
            ax.set_xlim(-0.7, 2.6 * (len(p["cost_axes"]) - 1) + 1.7)
            ax.set_ylim(0, 0.68)
            ax.tick_params(axis="y", labelsize=7)
        axes[0, 0].set_ylabel("share of energy", fontsize=8)
        axes[1, 0].set_ylabel("drop in task score", fontsize=8)
        return fig

    return _plot()


def principal_figure() -> str:
    order = ("adam", *ARMS)
    runs = {a: ADAM_RUNS if a == "adam" else [(a, TOP, s) for s in SEEDS] for a in order}
    data = {
        "panels": [
            {
                "arm": a,
                "name": NAME[a] if a == "adam" else f"{NAME[a]}, {FACTOR_SHOWN[TOP]}",
                "principal": [principal(*r)["shares"][-1] for r in runs[a]],
                "axes": [sorted(basis(*r)["energy_axes"][-1], reverse=True) for r in runs[a]],
            }
            for a in order
        ],
    }
    top = {a: [principal_top(*r) for r in runs[a]] for a in order}
    alt = f"""
        Four panels, Adam then pooled Adam, Muon and SGD at three times the finder's rate: the share of the stream
        energy per direction at the last slice, sorted, on a log scale, along the stream's principal directions
        (solid) and along the coordinate axes (dashed), one line per seed. In every panel the solid curve starts near
        {num(min(min(top[a]) for a in order))} to {num(max(max(top[a]) for a in order))} and falls steeply. Under Adam
        the dashed curve lies on the solid one for the first few directions; in the other three panels the dashed curve starts near 0.05 to 0.1 and
        lies well under the solid one for the first few directions.
    """
    return principal_draw(data, alt)


@memo
def principal_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="optimizer-scout-principal",
        alt_text=alt_text,
        caption="""
            **The energy spectrum along the stream's own principal directions**, Adam and each optimizer at three
            times the finder's rate. The share of the stream energy at the last slice that each direction holds, sorted
            from the largest, along the principal directions (solid) and, for comparison, along the coordinate axes
            (dashed, as in the figure above); one line per seed. The faint level is an even spread, 1/64.
        """,
    )
    def _plot() -> plt.Figure:
        panels = data["panels"]
        fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.4), layout="constrained", sharey=True)
        rank = np.arange(1, C + 1)
        for ax, p in zip(axes, panels, strict=True):
            color = hue(p["arm"])
            ax.axhline(1 / C, color=ghost(), lw=0.6, zorder=0)
            for sp, sa in zip(p["principal"], p["axes"], strict=True):
                ax.plot(rank, sa, color=color, lw=1.0, ls="--", alpha=0.8, zorder=2)
                ax.plot(rank, sp, color=color, lw=1.1, zorder=3)
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 0.9)
            ax.set_xlim(0, C + 1)
            ax.set_title(p["name"], fontsize=8)
            ax.set_xlabel("direction, by rank", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("share of energy", fontsize=8)
        handles = [
            Line2D([], [], color=ghost(), lw=1.1, label="principal directions"),
            Line2D([], [], color=ghost(), lw=1.0, ls="--", label="coordinate axes"),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


def isotropy_table() -> str:
    rows = []
    for a in ("adam", *ARMS):
        for i, f in enumerate((1.0,) if a == "adam" else FACTORS):
            rs = runs_of(a, (f,))
            rows.append(
                [
                    (NAME[a] if i == 0 else ""),
                    "" if a == "adam" else FACTOR_SHOWN[f],
                    fmt(top_share(*r) for r in rs),
                    fmt(top_share(*r, key="energy_rotated") for r in rs),
                    fmt(principal_top(*r) for r in rs),
                    fmt(kurtosis(*r) for r in rs),
                    fmt(kurtosis(*r, key="kurt_rotated") for r in rs),
                    fmt(float(np.median(axis_costs(*r))) for r in rs),
                    fmt(float(random_costs(*r).mean()) for r in rs),
                ]
            )
    return table_html(
        [
            "optimizer",
            "rate",
            "top share, axes",
            "top share, random basis",
            "top share, principal",
            "kurtosis, axes",
            "kurtosis, random basis",
            "cost, median axis",
            "cost, random direction",
        ],
        rows,
        """
        **Energy, kurtosis and cost along the axes and along random directions**, seed mean and range over the two
        seeds. The top share is the share of the stream energy that the largest direction holds at the last slice,
        along the coordinate axes, along a random orthonormal basis, and along the stream's own principal directions.
        The kurtosis is the mean over the directions of a basis, averaged over slices; a Gaussian has 3. The cost is
        the drop in the task score, averaged over ops, from projecting out one direction at full dose: the median over
        the 64 axes, and the mean over the 16 random directions.
        """,
    )


ADAM_TOP = [top_share(*r) for r in ADAM_RUNS]
EQ_TOP = [top_share(*r) for r in EQUIVARIANT]
EQ_TOP_ROT = [top_share(*r, key="energy_rotated") for r in EQUIVARIANT]
ADAM_TOP_ROT = [top_share(*r, key="energy_rotated") for r in ADAM_RUNS]
N_TOP_ON_SYNTAX = sum(len(TOP_ON_SYNTAX[a]) for a in ("adam-pooled", "muon"))

rf"""
## Is the stream isotropic? (E2)

Does the energy spectrum flatten and the kurtosis gap close? The equivariance says the optimizer has no privileged basis, not that the stream is isotropic: the model may still build a spike along some direction. A spike along a direction that is not an axis spreads over the axes and over a random basis alike, so these measurements would show it only as a stream that is uneven in both bases, with every direction touching it.
<!-- REVIEW: the plan said a spike off the axes would show in the spectrum along a random basis. It would not: a random basis is not aligned with it either, and Adam's rotated spectrum (a known spike of half the energy, seen from a basis it is not aligned with) has a top share of 0.04 to 0.07. Reworded to say what these measurements can and cannot show. Verify: `energy_rotated` of the Adam controls. -->

The main measure here is the share of the stream energy that the largest coordinate holds at the last slice, set beside the share that the largest direction of a random orthonormal basis holds. If the axes are a privileged basis the first is much larger; if they are ordinary directions the two are about the same.

{basis_figure()}

Under Adam one coordinate holds nearly half of the energy at the last slice, and the next holds about a twentieth. Under all three equivariant optimizers that spike is gone from the axes: at every rate the top coordinate holds about what the top direction of the random basis holds ({span(EQ_TOP)} against {span(EQ_TOP_ROT)}), and the solid and dashed spectra lie together. The kurtosis agrees: along the axes it is about what it is along random directions, near the Gaussian value, where under Adam the axes sit well above.

{isotropy_table()}

The cost of projecting out one direction says the same. Under Adam a typical axis is cheap, and a random direction costs a few times more because every random direction touches the spike. Under pooled Adam and Muon a typical axis costs about what a random direction costs. Under SGD both are cheap, but SGD learned little, so there is little to lose. So the axes are ordinary directions under all three optimizers.

That is less than isotropy. The spectrum of the Adam controls along the random basis is the reference for what a large spike looks like from a basis it is not aligned with, and it looks like the new spectra: the top direction holds {span(ADAM_TOP_ROT)} under Adam, in the same range as the new runs along either basis. So the measurements above cannot separate an even stream from a spike that has moved off the axes. The spectrum along the stream's own principal directions can: the eigenvectors of the second-moment matrix of the states are the directions where the energy sits, so a spike shows there wherever it points.

{principal_figure()}

The spike is still there. At the last slice the top principal direction holds {span(EQ_P_TOP)} of the energy under the new optimizers, against {span(ADAM_P_TOP)} under Adam, and the rest of the spectrum has the same shape. Under Adam the two spectra agree at the top, since the spike is an axis; under the other three the principal spectrum keeps the spike and the axes spectrum spreads it over the coordinates. The spike is no artifact of the mean either: with the mean state taken out first, the top direction still holds {span(EQ_P_TOP_CENTERED)}. Pooled Adam at its top rung has the smallest spike here ({span(principal_top(*r) for r in runs_of("adam-pooled", (TOP,)))}) and Muon at a third of the finder's rate the largest ({span(principal_top(*r) for r in runs_of("muon", (1 / 3,)))}); the table has the rest. So the new runs have a spike that has moved off the axes, and the stream is no more isotropic than under Adam. The spike is a property of the model and the task, and the optimizer decides only whether it lands on a coordinate axis.

The largest coordinate had hinted at this. In {N_TOP_ON_SYNTAX} of {len(runs_of("muon")) + len(runs_of("adam-pooled"))} pooled Adam and Muon runs the largest coordinate at the last slice is the axis that one of the syntax token embeddings leans on most, which chance would give in about one run in {round(1 / P_TOP_ON_SYNTAX)}, and Adam puts its spike on such an axis at {"both" if len(TOP_ON_SYNTAX["adam"]) == len(ADAM_RUNS) else count_of(len(TOP_ON_SYNTAX["adam"]), len(ADAM_RUNS))} seeds. E3 follows the tokens to the spike.

The cost of a direction has a consequence. With no privileged basis there are no cheap axes. Under Muon at the top rung the median axis costs {span(float(np.median(axis_costs(*r))) for r in runs_of("muon", (TOP,)))} of the task score, against {span(float(np.median(axis_costs(*r))) for r in ADAM_RUNS)} for an ordinary axis under Adam, and at the lower rates it costs more still. An anchor on e₁ under such an optimizer would have to make room on a direction the task already uses, where under Adam it can take a quiet axis. A quiet direction orthogonal to the spike may still exist, but it would not be an axis, and this scout does not look for it.
"""

# %%


def token_figure() -> str:
    order = ("adam", *ARMS)
    runs = {a: ADAM_RUNS if a == "adam" else [(a, TOP, s) for s in SEEDS] for a in order}
    data = {
        "random": RANDOM_COS,
        "tokens": [SHOWN[t].strip("`") for t in SYNTAX],
        "panels": [
            {
                "arm": a,
                "name": NAME[a] if a == "adam" else f"{NAME[a]}, {FACTOR_SHOWN[TOP]}",
                "axes": [[token_cosine(*r, t) for r in runs[a]] for t in SYNTAX],
                "rotated": [[token_cosine(*r, t, key="emb_rotated_max") for r in runs[a]] for t in SYNTAX],
            }
            for a in order
        ],
    }
    eq = [token_cosine(a, TOP, s, t) for a in ARMS for s in SEEDS for t in SYNTAX]
    alt = f"""
        Largest cosine of each syntax token embedding with any coordinate axis (filled markers) and with any direction
        of a random basis (open markers), one column per optimizer, a marker per seed. Under Adam the filled markers sit
        between {num(min(token_cosine(*r, t) for r in ADAM_RUNS for t in SYNTAX))} and 1 and the open ones near 0.3.
        Under pooled Adam, Muon and SGD at three times the finder's rate both kinds sit between {num(min(eq))} and
        {num(max(eq))}, inside the shaded band for a random unit vector.
    """
    return token_draw(data, alt)


@memo
def token_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="optimizer-scout-tokens",
        alt_text=alt_text,
        caption="""
            **Where the syntax tokens sit**, Adam and each optimizer at three times the finder's rate. For each syntax
            token, the largest absolute cosine of its embedding with any coordinate axis (filled) and with any direction
            of the random basis (open), one marker per seed. The shaded band holds 90% of the same cosine for a random
            unit vector in 64 dimensions.
        """,
    )
    def _plot() -> plt.Figure:
        panels, tokens = data["panels"], data["tokens"]
        fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.4), layout="constrained", sharey=True)
        lo, hi = data["random"]
        for ax, p in zip(axes, panels, strict=True):
            color = hue(p["arm"])
            ax.axhspan(lo, hi, color=light_dark("#e4e4e4", "#2a2a2a"), lw=0, zorder=0)
            for i, (ca, cr) in enumerate(zip(p["axes"], p["rotated"], strict=True)):
                ax.plot([i - 0.12] * len(ca), ca, MARK[p["arm"]], ms=4.5, color=color, zorder=3)
                ax.plot([i + 0.12] * len(cr), cr, MARK[p["arm"]], ms=4.5, mfc="none", mec=color, mew=0.9, zorder=3)
            ax.set_xticks(range(len(tokens)), tokens, fontsize=8)
            ax.set_xlim(-0.6, len(tokens) - 0.4)
            ax.set_ylim(0, 1.05)
            ax.set_title(p["name"], fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
        axes[0].set_ylabel("largest cosine", fontsize=8)
        handles = [
            Line2D([], [], ls="none", marker="o", ms=4.5, color=ghost(), label="with an axis"),
            Line2D(
                [], [], ls="none", marker="o", ms=4.5, mfc="none", mec=ghost(), label="with a random-basis direction"
            ),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=2, frameon=False, fontsize=7)
        return fig

    return _plot()


EQ_COS = [token_cosine(*r, t) for r in EQUIVARIANT for t in SYNTAX]
EQ_COS_ROT = [token_cosine(*r, t, key="emb_rotated_max") for r in EQUIVARIANT for t in SYNTAX]
N_SHARED = {a: sum(r[0] == a for r in SHARED) for a in ARMS}

rf"""
## Where the syntax tokens go (E3)

With no cheap one-hot coordinates, where do the separators and `=` sit? Spread over many coordinates, on a direction of their own that is not an axis, or still one-hot on an axis for a reason the optimizer does not supply.

The measure is the largest cosine of each syntax token embedding with any coordinate axis: near 1 for a token one-hot on an axis, and about {num(np.mean(RANDOM_COS), ".1f")} for a unit vector pointing nowhere in particular among 64 dimensions.

{token_figure()}

The syntax tokens leave the axes under all three optimizers. At every rate their largest cosine with any axis falls between {num(min(EQ_COS))} and {num(max(EQ_COS))}, about what a random unit vector gives, and their largest cosine with a direction of the random basis is the same size ({span(EQ_COS_ROT)}). So they sit on no particular direction of either basis, and the third option of the question is out.

The largest cosine cannot tell the first two options apart: a token spread over many coordinates and a token on a direction of its own both give a cosine of random size with every axis. The axis where each token leans most gives a hint. In {len(SHARED)} of {len(EQUIVARIANT)} runs, at least two of `?`, `,` and ⏎ lean most on the same axis ({N_SHARED["adam-pooled"]} under pooled Adam, {N_SHARED["muon"]} under Muon, {N_SHARED["sgd"]} under SGD), where three tokens pointing independently would do so in about one run in {round(1 / P_SHARED)}. Under Adam all three share one axis at both seeds.

The cosines between the tokens themselves settle it: the separators keep a direction of their own. In every run `,` and ⏎ have a cosine of {span(v for a in ARMS for v in SEP_PAIR[a])} with each other (under Adam, {span(SEP_PAIR["adam"])}), and that direction is the spike of E2: at the top rung their cosine with the top principal direction at the last slice is {span(v for a in ARMS for v in SEP_ON_TOP[a])}, against {span(SEP_ON_TOP["adam"])} under Adam. `?` joins them under Muon and SGD and less so under pooled Adam. `=` is the one that moves. Under Adam it sits apart, on a coordinate of its own, at a cosine of {span(EQ_ON_TOP["adam"])} with the spike and {span(EQ_PAIR["adam"])} with `?`; under Muon it leans on the spike too, at {span(EQ_ON_TOP["muon"])} with the top principal direction and {span(EQ_PAIR["muon"])} with `?`, and under SGD and pooled Adam it sits between. So the second option holds: the syntax tokens sit on a direction of their own, which is where the stream's energy sits, and Adam's contribution was to make that direction an axis and to give `=` an axis apart.
"""

# %%

r"""
## Discussion

Muon is a candidate for an anchored arm. At three times the finder's rate it learns the task as well as Adam, at both seeds, and its coordinate axes are ordinary directions. Pooled Adam is close behind, and since its best rung was also the top one, it may match Adam at a higher rate than this bracket reached. SGD was not tested at a useful rate, so the scout says nothing against it.

All three optimizers show that the privileged basis under Adam comes from the optimizer, as the memo proposed. The spike does not. Along the stream's own principal directions it is about as large under every optimizer here as under Adam, with the separators on it, so it is a property of the model and the task, and Adam's part was to turn it to face an axis. The stream is not isotropic under any of these optimizers, and an optimizer is not the lever for making it so; whatever makes the model put the separators on one large direction (the task's syntax, the unit-norm stream, the width) is a question for a different experiment.

The cost of a direction rises when no direction is privileged. Under Adam most axes are quiet, and an anchor on e₁ takes one of them at small cost to the task; under Muon every axis the scout measured holds part of the task, because each one touches the spike. If that holds for anchored runs, an anchor on e₁ under Muon would have to make room for itself on a direction the task already uses. Whether the anchor term can clear a direction as cheaply as Adam leaves one quiet is a question for anchored runs, and none ran here.
"""
