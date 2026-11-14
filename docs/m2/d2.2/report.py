# title: D2.2: figures for the post

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patheffects
import numpy as np

from mini.lit import memo
from mini.reports import current_publisher
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


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "body.json")).read_text())


# The recipe of record is `τ_ℓ fixed`, drawn without its end anneal (the cooldown): the `no-cooldown` arm of the
# cooldown experiment (docs/m2/cooldown), which carries the fixed confirmation's runs at seeds 712 to 719 on from
# their epoch-64 resume states with the anneal taken out. That experiment found the edit and the task unchanged.
fc = _load_sibling("fixed-confirm", "d22_figures_fc")
SEEDS: tuple[int, ...] = fc.SEEDS
OPS: tuple[str, ...] = fc.OP_NAMES
D = OPS.index(fc.ANCHORED_OP)
SLICE_NAMES = ["emb", "1", "2", "3", "4"]
PULLED = (1, 2, 3)  # the slices the pull acts on (design doc, the recipe of record)
FALLOFF_HOLDS: tuple[float, ...] = fc.sps.FALLOFF_HOLDS
COOLDOWN_TRAJ_REF = "reports/m2/cooldown/trajectories"  # `TRAJ_REF` of docs/m2/cooldown/experiment.py
_TRAJ = fetch_json(COOLDOWN_TRAJ_REF)


def ceilings() -> list[float]:
    """The Bayes ceiling of each op on the held-out contexts the training record scores: the first contexts of each
    op in ex-2.2.21's held-out set, as the trajectory picks them. The ceiling is the expected exact match of an ideal
    predictor that weighs each op by how well it explains the examples.
    """
    x21 = fc.ac.ex2222.ex2221
    ref = x21.HOLDOUT_REF.format(key=x21.MAIN_KEY)
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp, np.load(store.get(art, Path(tmp) / "holdout.npz")) as z:
        op, ceiling = z["op_ids"], z["ceiling"]
    sub = [np.flatnonzero(op == o)[: x21.N_TRAJ_EEM_PER_OP] for o in range(len(OPS))]
    return [float(ceiling[i].mean()) for i in sub]


CEILINGS = ceilings()


def traj(s: int) -> dict[str, np.ndarray]:
    """The training record of the run at seed *s*, one row per record. The last record repeats epoch 400 (the end of
    training is recorded twice, with the same values), so it is dropped.
    """
    t = _TRAJ[f"cooldown-no-cooldown-s{s}"]["traj"]
    ep = np.asarray(t["epoch"], float)
    keep = np.r_[True, ep[1:] != ep[:-1]]
    return {k: np.asarray(v, float)[keep] for k, v in t.items() if k in KEYS}


KEYS = ("epoch", "lr", "weight", "anti_weight", "tau", "val_loss", "eem_per_op", "alpha_anchored", "alpha_other")
TRAJS = {s: traj(s) for s in SEEDS}
EPOCHS = TRAJS[SEEDS[0]]["epoch"]
for _s in SEEDS:  # the schedules are the same for every seed, so the figure draws them once
    assert all(
        np.array_equal(TRAJS[_s][k], TRAJS[SEEDS[0]][k]) for k in ("epoch", "lr", "weight", "anti_weight", "tau")
    )

# α by slice at the end of training: the max over positions of the mean over probe contexts, per run.
ALPHA_MAX = {k: np.array([TRAJS[s][k][-1].max(axis=1) for s in SEEDS]) for k in ("alpha_anchored", "alpha_other")}
ARGMAX = np.array([TRAJS[s]["alpha_anchored"][-1].argmax(axis=1) for s in SEEDS])
QUERY_ANSWER = fc.lean.UNIT * fc.K + 4  # the position of the query answer: the fourth equation, after a ? b =
assert (ARGMAX[:, 1:] == QUERY_ANSWER).all(), "the prose says the max is at the query answer from slice 1 on"


def ink(i: int, n: int) -> tuple[float, float, float, float]:
    """The *i*-th of *n* ordered shades, kept off the end that vanishes into the background."""
    return plt.cm.viridis(light_dark(0.0, 0.3) + light_dark(0.8, 0.7) * i / (n - 1))


GHOST = ("#999", "#666")


def alpha_slice_figure() -> str:
    data = {
        "seeds": list(SEEDS),
        "anchored": ALPHA_MAX["alpha_anchored"].tolist(),
        "other": ALPHA_MAX["alpha_other"].tolist(),
        "lam_a": float(TRAJS[SEEDS[0]]["weight"].max()),
        "pulled": list(PULLED),
        "holds": list(FALLOFF_HOLDS),
        "anchored_op": OPS[D],
    }
    m = ALPHA_MAX["alpha_anchored"].mean(axis=0)
    alt = f"""
        α of the anchored op rises from near zero at the embedding to {m[1]:.2f} at slice 1, {m[2]:.2f} at slice 2
        and {m[3]:.2f} at slice 3, then falls to {m[4]:.2f} at slice 4. The other ops are highest at slice 1, near 0.2, and lower after.
        Below, the pull acts on slices 1 to 3 and the anti weight falls from 0.2 at the embedding to 0.03 at slice 4.
    """
    return alpha_slice_draw(data, alt)


@memo
def alpha_slice_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="d22-alpha-by-slice",
        alt_text=alt_text,
        caption=f"""
            **Where the op sits on the anchored axis, by slice, at the end of training.** **Top:** α, the max over
            positions of the mean over probe contexts, for the contexts of `{data["anchored_op"]}` (circles) and of
            the other six ops (squares): the mean over the eight runs of the recipe of record, with each run as a
            faint dot and a bar spanning them. **Bottom:** the weights at each slice: the peak pull weight
            $\\lambda_a$, and the anti-subspace weight $\\lambda_{{\\bar s}}$ at its hold.
        """,
    )
    def _plot() -> plt.Figure:
        x = np.arange(len(SLICE_NAMES))
        fig, (ax, aw) = plt.subplots(
            2, 1, figsize=(3.4, 3.2), layout="constrained", sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        rng = np.random.default_rng(0)
        series = (("anchored", "o", light_dark("#c2185b", "#f06292"), data["anchored_op"]), ("other", "s", light_dark(*GHOST), "other ops"))  # fmt: skip
        for key, marker, color, label in series:
            v = np.asarray(data[key])
            jit = rng.uniform(-0.08, 0.08, v.shape)
            ax.scatter(x + jit, v, s=5, color=color, alpha=0.35, lw=0, zorder=2)
            ax.vlines(x, v.min(0), v.max(0), color=color, lw=2.5, alpha=0.25, zorder=1)
            ax.plot(x, v.mean(0), color=color, marker=marker, ms=4, lw=1, label=label, zorder=3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("α", fontsize=7)
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        lam = [data["lam_a"] if i in data["pulled"] else 0.0 for i in x]
        aw.plot(x, lam, color=light_dark("#c2185b", "#f06292"), marker="D", ms=3, lw=0.8, label="$\\lambda_a$")
        aw.plot(x, data["holds"], color=light_dark("#333", "#ccc"), marker="D", mfc="none", ms=3, lw=0.8, label="$\\lambda_{\\bar s}$")  # fmt: skip
        aw.set_ylim(0, 0.25)
        aw.set_ylabel("weight", fontsize=7)
        aw.legend(fontsize=6, frameon=False, ncols=2, loc="upper right")
        aw.set_xticks(x, SLICE_NAMES)
        aw.set_xlabel("slice", fontsize=7)
        for a in (ax, aw):
            a.tick_params(labelsize=6)
        return fig

    return _plot()


def training_data() -> dict:
    t0 = TRAJS[SEEDS[0]]
    return {
        "epoch": EPOCHS.tolist(),
        "loss": [TRAJS[s]["val_loss"].tolist() for s in SEEDS],
        "eem": [TRAJS[s]["eem_per_op"].tolist() for s in SEEDS],
        "ops": list(OPS),
        "ceilings": CEILINGS,
        "anchored": D,
        "lr": t0["lr"].tolist(),
        "weight": t0["weight"].tolist(),
        "tau": t0["tau"].tolist(),
        "anti": t0["anti_weight"].tolist(),
    }


SCHEDULE_ALT = """
    The pull weight λ_a ramps up to its peak of 0.1 by about epoch 40 and holds there. The anti weight starts at 0.25
    on every slice and anneals to a hold that falls with depth, from 0.2 at the embedding to 0.03 at slice 4, so the
    pull weight sits above the anti weight at slices 3 and 4 by the end. The position temperature τ_p sharpens from 1
    to 0.1 by epoch 40, while τ_ℓ stays at 0.1.
"""


def training_figure(view: str) -> str:
    """The task score along training with the schedules under it. *view* is `per-op` (a line for each op) or `mean`
    (the mean over ops, with every op of every run behind it as a hairline).
    """
    data = training_data() | {"view": view}
    eem_end = np.array([TRAJS[s]["eem_per_op"][-1] for s in SEEDS])
    if view == "per-op":
        task = f"""
            The task score of each op rises at its own pace: the HSV ops learn later than the others, and `mix` stays
            lowest, ending near {eem_end.mean(0)[OPS.index("mix")]:.2f}, close to its ceiling, where the best op ends
            near {eem_end.mean(0).max():.2f}.
        """
    else:
        task = f"""
            The task score, averaged over the ops, rises quickly over the first 40 epochs and slowly after, ending
            near {eem_end.mean():.2f}, a little under the mean ceiling of {np.mean(CEILINGS):.2f}. The ops of every
            run spread out behind it between about 0.3 and 0.75.
        """
    return training_draw(data, f"Three stacked panels over 400 epochs. {task} {SCHEDULE_ALT}")


TASK_CAPTION = {
    "per-op": """the task score (expected exact match) of each op on held-out contexts, as the mean over the eight
        runs, with each run behind it as a hairline in the same color. The ticks past the end of
        training mark the Bayes ceiling of each op, the score of an ideal predictor on the same contexts.""",
    "mean": """the task score (expected exact match) on held-out contexts: the mean over ops and runs, with a band
        spanning the runs, and each op of each run behind it as a hairline. The tick past the end of training marks
        the Bayes ceiling, the score of an ideal predictor on the same contexts.""",
}


def draw_phases(phase_ax: plt.Axes, end: float, bg: Any, note: dict) -> None:
    """The phases of training in *phase_ax*, as a ramp with no hard edges, over epochs 0 to *end*."""
    # The keyframes of the schedules are at 10% and 90% of
    # the run (the pull at full weight and τ_p sharp by epoch 40; the anti weight settled by about 360).
    keys = [(0.0, light_dark("#f77", "#c55")), (0.1, bg), (1.0, light_dark("#88f", "#66c"))]
    ramp = mcolors.LinearSegmentedColormap.from_list("phase", keys)
    phase_ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=ramp, aspect="auto", extent=(0, end, 0, 1))
    phase_ax.set_yticks([])
    for spine in ("left", "right", "top"):
        phase_ax.spines[spine].set_visible(False)
    for x, label in ((0.0, "nascent"), (0.1, "malleable"), (0.9, "crystallized")):
        phase_ax.text(x * end, 1.25, label, ha="center", va="bottom", clip_on=False, **note)


@memo
def training_draw(data: dict, alt_text: str) -> str:
    view = data["view"]

    @themed(
        name=f"d22-training-{view}",
        alt_text=alt_text,
        caption=f"""
            **Training the recipe of record.** **Top:** {TASK_CAPTION[view]} **Middle:** the pull weight
            $\\lambda_a$ and the anti-subspace weight $\\lambda_{{\\bar s}}$ at each slice, with the middle slices
            thinner. **Bottom:** the pooling temperatures $\\tau_p$ (over positions) and $\\tau_\\ell$ (over slices).
            The schedules are the same for every run. The ramp underneath names the phases of training, with labels at
            the keyframes of the schedules.
        """,
    )
    def _plot() -> plt.Figure:
        ep = np.asarray(data["epoch"])
        fig, (eem_ax, lam_ax, tau_ax, phase_ax) = plt.subplots(
            4,
            1,
            figsize=(5.6, 4.6),
            layout="constrained",
            sharex=True,
            gridspec_kw={"height_ratios": [2.6, 1.3, 0.8, 0.12]},
        )
        main = light_dark("#222", "#ddd")
        bg = plt.rcParams["axes.facecolor"]
        eem = np.asarray(data["eem"])  # (seed, record, op)
        ceil = np.asarray(data["ceilings"])
        tick = [ep[-1] + 6, ep[-1] + 20]  # a ceiling, as a short tick past the end of training
        if view == "per-op":
            marks = "osDpPXh"  # no triangles: in this project they mark regularizer targets
            bar_at = [int(np.abs(ep - e).argmin()) for e in range(50, int(ep[-1]) + 1, 50)]
            # The ops keep their colors and markers, but draw and list in the order of their final score.
            for o in np.argsort(-eem[:, -1].mean(0)):
                color = f"C{o}"
                anchored = o == data["anchored"]
                eem_ax.plot(ep, eem[:, :, o].mean(0), color=color, lw=1.6 if anchored else 0.9, marker=marks[o], ms=3, markevery=bar_at, label=data["ops"][o], path_effects=[patheffects.Stroke(linewidth=(1.6 if anchored else 0.9) + 1.6, foreground=bg), patheffects.Normal()], zorder=3 if anchored else 2)  # fmt: skip
                for r in eem[:, :, o]:  # each run of the op, as a hairline in its color
                    eem_ax.plot(ep, r, color=color, lw=0.35, alpha=0.2, zorder=1)
                eem_ax.plot(tick, [ceil[o]] * 2, color=color, lw=1.6, clip_on=False, zorder=3)
            leg = eem_ax.legend(fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.06, 0.5))
            for h in leg.get_lines():  # the halo is for crossings in the plot, not the key
                h.set_path_effects([])
        else:
            for s in range(eem.shape[0]):
                for o in range(eem.shape[2]):
                    eem_ax.plot(ep, eem[s, :, o], color=light_dark(*GHOST), lw=0.3, alpha=0.6, zorder=1)
            mean = eem.mean(axis=2)  # (seed, record): the mean over ops of each run
            eem_ax.fill_between(ep, mean.min(0), mean.max(0), color=light_dark(*GHOST), alpha=0.45, lw=0, zorder=2)
            eem_ax.plot(ep, mean.mean(0), color=main, lw=1.4, zorder=3, label="mean over ops")
            eem_ax.plot(tick, [ceil.mean()] * 2, color=main, lw=1.6, clip_on=False, zorder=3)
            eem_ax.plot([], [], color=light_dark(*GHOST), lw=0.6, label="each op, each run")
            eem_ax.legend(fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.06, 0.5))
        eem_ax.set_ylim(0, 1)
        eem_ax.set_ylabel("task score\n(EEM)", fontsize=7)

        anti = np.asarray(data["anti"])
        n = len(SLICE_NAMES)
        for i, name in enumerate(SLICE_NAMES):
            edge = i in (0, n - 1)
            lam_ax.plot(ep, anti[:, i], color=ink(i, n), lw=1.2 if edge else 0.5, label=f"$\\lambda_{{\\bar s}}$, {name}", zorder=1)  # fmt: skip
        halo = [patheffects.Stroke(linewidth=4.2, foreground=bg), patheffects.Normal()]
        lam_ax.plot(ep, data["weight"], color=light_dark("#c2185b", "#f06292"), lw=1.4, path_effects=halo, label="$\\lambda_a$", zorder=3)  # fmt: skip
        lam_ax.set_ylim(0, 0.27)
        note = {"fontsize": 6, "color": main}
        lam_ax.text(14, 0.238, "push", va="top", **note)
        lam_ax.text(38, 0.108, "pull", va="bottom", **note)
        lam_ax.set_ylabel("weight", fontsize=7)
        handles, labels = lam_ax.get_legend_handles_labels()
        lam_leg = lam_ax.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1], fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))  # fmt: skip
        for h in lam_leg.get_lines():  # the halo is for crossings in the plot, not the key
            h.set_path_effects([])

        tau = np.asarray(data["tau"])
        tau_ax.plot(ep, tau[:, 0], color=main, lw=1, label="$\\tau_p$")
        tau_ax.plot(ep, tau[:, 1], color=main, lw=1, ls="--", label="$\\tau_\\ell$")
        tau_ax.set_ylim(0, 1.05)
        tau_ax.text(14, 0.92, "(soft)", va="center", **note)
        tau_ax.text(38, 0.14, "(sharp)", va="bottom", **note)
        tau_ax.set_ylabel("τ", fontsize=7)
        tau_ax.legend(fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        draw_phases(phase_ax, ep[-1], bg, note)
        phase_ax.set_xlabel("epoch", fontsize=7)
        phase_ax.set_xlim(0, ep[-1])
        for ax in (eem_ax, lam_ax, tau_ax, phase_ax):
            ax.tick_params(labelsize=6)
        return fig

    return _plot()


def basics_figure() -> str:
    data = training_data()
    alt = """
        Two stacked panels over 400 epochs. Validation loss falls steeply in the first 30 epochs and slowly after,
        and the eight runs stay close together. The learning rate warms up over the first few epochs and then decays
        on a cosine to near zero.
    """
    return basics_draw(data, alt)


@memo
def basics_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="d22-loss-lr",
        alt_text=alt_text,
        caption="""
            **Loss and learning rate.** **Top:** validation loss, the mean over the eight runs with a band spanning
            them. **Bottom:** the learning rate, the same for every run.
        """,
    )
    def _plot() -> plt.Figure:
        ep = np.asarray(data["epoch"])
        fig, (loss_ax, lr_ax) = plt.subplots(
            2, 1, figsize=(4.6, 2.4), layout="constrained", sharex=True, gridspec_kw={"height_ratios": [1.6, 1]}
        )
        main = light_dark("#222", "#ddd")
        loss = np.asarray(data["loss"])
        loss_ax.fill_between(ep, loss.min(0), loss.max(0), color=light_dark(*GHOST), alpha=0.35, lw=0, zorder=1)
        loss_ax.plot(ep, loss.mean(0), color=main, lw=1, zorder=2)
        loss_ax.set_ylabel("val. loss", fontsize=7)
        lr_ax.plot(ep, np.asarray(data["lr"]) * 1e3, color=main, lw=1)
        lr_ax.set_ylabel("LR (×10⁻³)", fontsize=7)
        lr_ax.set_xlabel("epoch", fontsize=7)
        lr_ax.set_xlim(0, ep[-1])
        for ax in (loss_ax, lr_ax):
            ax.tick_params(labelsize=6)
        return fig

    return _plot()


r"""
# D2.2: figures for the post

Figures for the post on D2.2, drawn here so the post can upload them from the bundle of this page. This page is not an experiment. The results behind each figure are in the reports it names, which hold the claims and their evidence; the [design doc](./design.md) has the recipe the figures describe.

## Pooling the pull

The anchor term pulls the states of each labeled context toward the anchor axis. It pools that pull twice before averaging over the batch: a mellowmax over the positions of the context, and then a mellowmax over slices. The mellowmax is a soft minimum, so each pool lets the term choose which states to pull, within a context and across depth. The [ex-2.1.9 report](../ex-2.1.9/report.py) introduced the pool over positions and drew the first version of this diagram; the pool over slices is new in D2.2.
"""

# A static SVG beside this script, revised from the ex-2.1.9 one; the publisher copies it into the bundle's assets.
pooling_svg = Path(__file__).parent / "public" / "pooling.svg"
pooling_url = pub.asset_url(pooling_svg, name="pooling.svg") if (pub := current_publisher()) else str(pooling_svg)
figure_html(
    f'<img src="{pooling_url}" alt="A training batch drawn as a cube with axes for position, slice, and batch. Two labeled sequences are shaded. Of five slices, the middle three each have a magenta line pooling over the positions of a labeled sequence, a magenta line along the slice edge pools those results, and an orange line down the batch edge takes the mean.">',
    caption="The anchor term pools the pull over positions, then over slices, then averages over the batch.",
)

r"""
## Where the op sits

The figures from here on are drawn from eight runs of the recipe of record at seeds 712 to 719, without the end anneal of the pull and anti weights: the [cooldown](../cooldown/report.py) experiment carried the runs of the [fixed confirmation](../fixed-confirm/report.py) on with the anneal taken out, and found that it changes neither the edit nor the task. α is the cosine between a state and the anchored axis e₁, averaged over the probe contexts of an op. Below, it is taken at whichever position is highest at each slice; from slice 1 on, that is the query answer on every run.
"""

alpha_slice_figure()

r"""
## Training

The training record of the same runs, with the schedules underneath on the same epoch axis.
"""

training_figure("mean")

r"""
The same figure with a line for each op:
"""

training_figure("per-op")

r"""
Loss and learning rate, which the training figure leaves out:
"""

basics_figure()
