# title: D2.2: figures for the post

import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patheffects
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import numpy as np

from mini.lit import memo
from mini.reports import current_publisher
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed
from mini.vis.plt import smooth_step_marks


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
        training mark the score ideal₇ would give for each op on the same contexts, where ideal₇ is the
        Bayes-optimal predictor over all seven ops.""",
    "mean": """the task score (expected exact match) on held-out contexts: the mean over ops and runs, with a band
        spanning the runs, and each op of each run behind it as a hairline. The tick past the end of training marks
        the score ideal₇ would give on the same contexts, where ideal₇ is the Bayes-optimal predictor over all seven
        ops.""",
}


def plain_key(legend) -> None:
    """Drop the halos from a legend: they separate crossing lines in the plot, and the key has none."""
    for h in legend.get_lines():
        h.set_path_effects([])


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
    for x, label in ((0.0, "nascent"), (0.1, "malleable"), (0.9, "settling")):
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
            plain_key(eem_ax.legend(fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.06, 0.5)))
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
        plain_key(lam_ax.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1], fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5)))  # fmt: skip

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


# --- Confusion matrices -------------------------------------------------------------------------------------------
# The cooldown experiment kept no answer distributions, so these come from the runs of the fixed confirmation at the
# same seeds, with the end anneal (it changes neither the edit nor the task), and their partners in the filtered-corpus
# experiment: the control of anneal-confirm and sans-diff. The matrices are taken as ex-2.2.22 and filtered-corpus
# take them.

fcx = _load_sibling("filtered-corpus", "d22_figures_fcx")
# The post-hoc edits re-scored on every held-out context: the configurations the post-hoc comparison chose, fitted
# on the controls without role labels ("pooled"), with nothing tuned again.
phc = _load_sibling("post-hoc-compare", "d22_figures_phc")
POSTHOC = ("leace", "inlp")


def fetch_posthoc() -> dict[int, dict[str, np.ndarray]]:
    store = project_store()
    refs = {s: phc.CONFUSION_ARRAYS_REF.format(label=f"control-s{s}") for s in SEEDS}
    found = {r: a for r, a in store.get_refs(list(refs.values())).items() if a is not None}
    missing = [r for r in refs.values() if r not in found]
    assert not missing, f"not published: {missing}"
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(found[r], Path(tmp) / f"{s}.npz") for s, r in refs.items()])
        for s, path in zip(refs, paths, strict=True):
            with np.load(path) as z:
                out[s] = {k: z[k] for k in ("op_ids", "query_pair", *(f"pooled/{m}/p_full" for m in POSTHOC))}
    return out


CONF_KINDS = ("control", "anchored", "sans-diff")


def conf_dose_ref(kind: str, s: int) -> str:
    return {
        "sans-diff": lambda: fcx.DOSE_ARRAYS_REF.format(label=fcx.label_of(s)),
        "anchored": lambda: fcx.fc.DOSE_ARRAYS_REF.format(label=fcx.fc.label_of(s, None)),
        "control": lambda: fcx.ac.DOSE_ARRAYS_REF.format(label=fcx.ac.label_of(s, None, "control")),
    }[kind]()


def fetch_doses() -> dict[tuple[str, int], dict[str, np.ndarray]]:
    store = project_store()
    keys = [(k, s) for k in CONF_KINDS for s in SEEDS]
    refs = [conf_dose_ref(*k) for k in keys]
    found = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    missing = [r for r in refs if r not in found]
    assert not missing, f"not published: {missing}"
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        paths = store.get_many([(found[r], Path(tmp) / f"{i}.npz") for i, r in enumerate(refs)])
        for k, path in zip(keys, paths, strict=True):
            with np.load(path) as z:
                out[k] = {n: z[n] for n in ("op_ids", "posterior", "query_pair", "p_clean", "p_full")}
    return out


@memo
def confusion_matrices() -> dict[str, list]:
    """Seed-mean op confusion (rows: the op of the context; the diagonal is the mass on the answers the row op can
    give, and off it the mass on the answers the column op can give and the row op cannot) for the Bayes predictor,
    the target null, and the runs, clean and edited.
    """
    dose = fetch_doses()
    d0 = dose[(CONF_KINDS[0], SEEDS[0])]
    op_ids, pairs = d0["op_ids"].astype(np.int64), d0["query_pair"].astype(np.int64)
    for k, v in dose.items():
        assert (v["op_ids"] == op_ids).all() and (v["query_pair"] == pairs).all(), k
    post = fcx.ex2221.ex2216._get_posterior()
    table = post.build_table(fcx.ex2222.ex2218.table_of(OPS))
    empty = np.zeros((len(op_ids), 0), np.int64)
    ctx = post.Contexts(op_ids, empty, empty, pairs)
    prior = d0["posterior"].astype(np.float64)
    null = fcx.ex2221.ex2216._target_null(post, table, ctx, prior, D)
    rows = np.arange(len(op_ids))
    gives = np.zeros((len(OPS), len(op_ids), 217), bool)
    for o in range(len(OPS)):
        idx = table.idx[o, pairs]
        gives[o, rows[:, None], np.where(idx < 0, 216, idx)] = table.prob[o, pairs] > 0
    gives = gives[:, :, :216]
    true = gives[op_ids, rows]

    def confusion(p: np.ndarray) -> list:
        cols = [(p * np.where((op_ids == o)[:, None], true, gives[o] & ~true)).sum(1) for o in range(len(OPS))]
        per_col = np.stack(cols, 1)
        return np.stack([per_col[op_ids == o].mean(0) for o in range(len(OPS))]).tolist()

    def seed_mean(kind: str, key: str) -> list:
        return np.mean([confusion(dose[(kind, s)][key].astype(np.float64)) for s in SEEDS], axis=0).tolist()

    posthoc = fetch_posthoc()
    for s, v in posthoc.items():
        assert (v["op_ids"] == op_ids).all() and (v["query_pair"] == pairs).all(), s

    def posthoc_mean(method: str) -> list:
        return np.mean(
            [confusion(posthoc[s][f"pooled/{method}/p_full"].astype(np.float64)) for s in SEEDS], axis=0
        ).tolist()

    return {
        **{m: posthoc_mean(m) for m in POSTHOC},
        "ideal": confusion(post.predictive(table, ctx, prior)),
        "null": confusion(post.predictive(table, ctx, null)),
        "control": seed_mean("control", "p_clean"),
        "anchored": seed_mean("anchored", "p_clean"),
        "sans-diff": seed_mean("sans-diff", "p_clean"),
        "edited": seed_mean("anchored", "p_full"),
        "control-edited": seed_mean("control", "p_full"),
    }


CONFUSION = {k: np.asarray(v) for k, v in confusion_matrices().items()}
CONF_PANELS = {
    "clean": ("ideal", "control", "anchored"),
    "edited": ("null", "control-edited", "edited", "sans-diff"),
    "post-hoc": ("null", "edited", "leace", "inlp"),
}
CONF_WIDEST = max(len(v) for v in CONF_PANELS.values())  # every figure sizes its bars for the widest panel
# What each model was before `difference` was taken out, drawn faintly behind its bar.
CONF_BEFORE = {"null": "ideal", "control-edited": "control", "edited": "anchored", "sans-diff": "control", "leace": "control", "inlp": "control"}  # fmt: skip
# Bayes grey, control blue, anchored orange, retrained (sans-diff) purple.
CONF_STYLE = {
    "ideal": ("ideal₇", "grey"),
    "control": ("control", "blue"),
    "anchored": ("anchored", "orange"),
    "null": ("ideal₆", "grey"),
    "control-edited": ("control, edited", "blue"),
    "sans-diff": ("retrained", "purple"),
    "edited": ("anchored, edited", "orange"),
    "leace": ("control, LEACE", "green"),
    "inlp": ("control, INLP", "red"),
}
EYE = np.eye(len(OPS), dtype=bool)
OP_SHORT = {"difference": "diff", "hue-hsv": "hue", "sat-hsv": "sat", "value-hsv": "value"}


def confusion_figure(view: str) -> str:
    keys = CONF_PANELS[view]
    m = {k: CONFUSION[k] for k in keys}
    outside = ~EYE & (np.arange(len(OPS)) != D)[:, None]  # off the diagonal and the `difference` row
    off = {k: float(v[outside].mean()) for k, v in m.items()}
    names = "; ".join(CONF_STYLE[k][0] for k in keys)
    lo, hi = f"{min(off.values()):.2f}", f"{max(off.values()):.2f}"
    spread = f"about {lo}" if lo == hi else f"{lo} to {hi}"
    head = f"""
        A grid with the op of the context on the rows and the op of the answer on the columns; each cell holds one
        bar per model ({names}). Off the diagonal, the bars are short and about even ({spread} on average outside the `difference` row).
    """
    if view == "clean":
        tail = f"""
            The models look alike everywhere. On the diagonal, ideal₇ puts about {np.diag(m["ideal"]).mean():.2f}
            on the answers of the row op, and the two models about {np.diag(m["control"]).mean():.2f}.
        """
    elif view == "post-hoc":
        a = max((o for o in range(len(OPS)) if o != D), key=lambda o: m["null"][D, o])
        row = "; ".join(f"{CONF_STYLE[k][0]} {m[k][D, D]:.2f}, with {m[k][D, a]:.2f} on {OPS[a]}" for k in keys)
        tail = f"""
            In the `difference` row, all four put little on `difference` answers and spread the rest over the other
            ops in about the same proportions: {row}.
        """
    else:
        a, b = sorted((o for o in range(len(OPS)) if o != D), key=lambda o: -m["null"][D, o])[:2]
        row = "; ".join(
            f"{CONF_STYLE[k][0]} {m[k][D, D]:.2f}, with {m[k][D, a]:.2f} on {OPS[a]}"
            for k in keys
            if k != "control-edited"
        )
        tail = f"""
            In the `difference` row, the diagonal bars are near zero for the three models without `difference`, and
            the mass goes to the other six ops instead, most to {OPS[a]} and {OPS[b]}: {row}. The edited control
            keeps `difference`, with {m["control-edited"][D, D]:.2f} on the diagonal.
        """
    before = {k: CONFUSION[CONF_BEFORE[k]].tolist() for k in keys if k in CONF_BEFORE}
    return confusion_draw(view, {k: v.tolist() for k, v in m.items()}, before, head + tail)


CONF_CAPTION = {
    "clean": """
        **Where the answers go.** Rows: the op of the context. Columns: the mass on the colors the column op can give
        for the query operands and the row op cannot, so a color several ops can give counts in each of their
        columns. On the diagonal (outlined), the mass on the answers of the row op. A full cell is a mass of 1. ideal₇ is the Bayes-optimal predictor over all seven ops; the models are seed means over eight runs.
    """,
    "edited": """
        **Where the answers go, with `difference` taken out.** Laid out as the earlier figure. ideal₆ is the
        Bayes-optimal predictor with `difference` taken out (the target null); retrained is the control trained from scratch with
        the loss masked on `difference` contexts (sans-diff in the earlier reports). Both edited models are under the same edit of the anchored axis;
        the control holds nothing there, so its edit leaves the op in place. Faint bars behind show each one before
        `difference` was taken out: ideal₇ for ideal₆, the control for retrained, and the unedited
        model for the edited ones.
    """,
    "post-hoc": """
        **Where the answers go, with `difference` taken out after training.** Laid out as the earlier figures. LEACE
        and INLP are post-hoc erasers fitted to the control without role labels, at the settings the post-hoc
        comparison chose; beside them, the anchored model under its edit and ideal₆. Faint bars behind show
        each one before the edit (for ideal₆, ideal₇). Seed means over eight runs.
    """,
}


# The bar grids are laid out in whole pixels of the saved PNG, so cell and bar edges land on pixel edges and stay
# crisp; only the tops of the bars fall between pixels. The pixels assume the save settings of `themed`: 192 dpi, a
# tight crop, and a pad that is itself a whole number of pixels (set below), around a figure the crop keeps whole.
DPI = 192
plt.rcParams["savefig.pad_inches"] = 24 / DPI


@memo
def confusion_draw(view: str, data: dict, before: dict, alt_text: str) -> str:
    @themed(name=f"d22-confusion-{view}", alt_text=alt_text, caption=CONF_CAPTION[view])
    def _plot() -> plt.Figure:
        n, k_n = len(OPS), len(data)
        pitch, gap, frame = 116, 8, 1  # cell pitch, gap between cells, diagonal frame (pixels)
        bar_gap, side = 2, 6  # between bars, and from the cell edge to the outer bars
        left, top, bottom, right = 84, 56, 44, 4
        w, h = left + n * pitch + right, top + n * pitch + bottom
        cell = pitch - gap
        bar_w = (cell - 2 * side - (CONF_WIDEST - 1) * bar_gap) // CONF_WIDEST
        x_pad = (cell - k_n * bar_w - (k_n - 1) * bar_gap) // 2
        inks = {
            "grey": light_dark("#555", "#aaa"),
            "blue": light_dark("#2a6fbb", "#6aa6e8"),
            "orange": light_dark("#d9731a", "#f0a05a"),
            "purple": light_dark("#7b4fb0", "#b493e0"),
            "green": light_dark("#3a8f4f", "#6cc184"),
            "red": light_dark("#b8435a", "#e57a8e"),
        }
        paper = light_dark("#fff", "#111")
        well = light_dark("#f3f3f3", "#1e1e1e")
        edge = light_dark("#999", "#666")
        fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # y in pixels from the top
        ax.set_axis_off()
        # A clear patch over the whole figure, so the tight crop keeps the figure whole and the pixels stay put.
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fc="none", ec="none", lw=0))

        def box(x: float, y: float, bw: float, bh: float, fc: Any, alpha: float = 1.0) -> Rectangle:
            patch = Rectangle((x, y), bw, bh, fc=fc, alpha=alpha, lw=0, snap=False)
            ax.add_patch(patch)
            return patch

        handles = {}
        for i, j in np.ndindex(n, n):
            x0, y0 = left + j * pitch + gap // 2, top + i * pitch + gap // 2
            if i == j:
                box(x0 - frame, y0 - frame, cell + 2 * frame, cell + 2 * frame, fc=edge)
            box(x0, y0, cell, cell, fc=paper if i == j else well)
            for c, (k, v) in enumerate(data.items()):
                bar_h = cell * float(v[i][j])
                bx = x0 + x_pad + c * (bar_w + bar_gap)
                if k in before:
                    was = cell * float(before[k][i][j])
                    box(bx, y0 + cell - was, bar_w, was, fc=inks[CONF_STYLE[k][1]], alpha=light_dark(0.15, 0.25))
                handles[k] = box(bx, y0 + cell - bar_h, bar_w, bar_h, fc=inks[CONF_STYLE[k][1]])
        ink = light_dark("#222", "#ddd")
        for i, op in enumerate(OPS):
            mid = top + i * pitch + pitch / 2
            ax.text(left - 10, mid, OP_SHORT.get(op, op), ha="right", va="center", fontsize=7, color=ink)
            ax.text(
                left + i * pitch + pitch / 2,
                top + n * pitch + 6,
                OP_SHORT.get(op, op),
                ha="center",
                va="top",
                fontsize=7,
                color=ink,
            )
        fig.legend(
            list(handles.values()), [CONF_STYLE[k][0] for k in handles], ncols=len(handles), frameon=False, fontsize=7,
            loc="upper center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2, handlelength=1.4,
        )  # fmt: skip
        return fig

    return _plot()


# --- α by position ------------------------------------------------------------------------------------------------
# The low-runs experiment scored α at every position of the held-out contexts for the per-visit runs of the fixed
# confirmation: the recipe of record at seeds 712 to 719, with the end anneal (it changes neither the edit nor the task).

lr = _load_sibling("low-runs", "d22_figures_lr")
POSITION_LABELS = [
    *(t for i in "₁₂₃" for t in (f"a{i}", "&", f"b{i}", "=", f"y{i}", ",")),
    "a",
    "&",
    "b",
    "=",
    "y",
    "⏎",
]


def alpha_by_position() -> dict[str, np.ndarray]:
    """(runs, slices, positions) α on `difference` contexts and on the other ops, for the per-visit runs."""
    ev = {r["label"]: r for r in fetch_json(lr.EVIDENCE_REF)["runs"]}
    labels = [r["label"] for r in lr.runs() if r["kind"] == "per-visit"]
    assert len(labels) == len(SEEDS), labels
    return {k: np.array([ev[lab]["alpha_by_position"][k] for lab in labels]) for k in ("difference", "other")}


ALPHA_POS = alpha_by_position()
assert ALPHA_POS["difference"].shape[1:] == (len(SLICE_NAMES), len(POSITION_LABELS))


def positions_figure() -> str:
    dif, oth = ALPHA_POS["difference"].mean(0), ALPHA_POS["other"].mean(0)
    ans = list(lr.ANSWERS)
    alt = f"""
        Five stacked step profiles, one per slice from the embedding at the bottom to slice 4 at the top, showing
        the cosine with e₁ at each position of a context. On `difference` contexts it is near zero everywhere at
        the embedding. From slice 1 on it rises at the three example answers ({dif[1, ans].mean():.2f} at slice 1,
        {dif[3, ans].mean():.2f} at slice 3) and higher still at the query answer ({dif[1, QUERY_ANSWER]:.2f} at
        slice 1, {dif[3, QUERY_ANSWER]:.2f} at slice 3), and stays near zero at every other position. On the other
        ops it stays under {oth.max():.2f}.
    """
    data = {"difference": dif.tolist(), "other": oth.tolist(), "runs": ALPHA_POS["difference"].tolist()}
    return positions_draw(data, alt)


@memo
def positions_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="d22-alpha-by-position",
        alt_text=alt_text,
        caption="""
            **Where the anchored axis is used.** The cosine between the state and e₁ at each position of a held-out
            context, one row per slice, the embedding at the bottom. Red: `difference` contexts, the mean over the
            eight runs, with each run behind it as a hairline; grey: the other ops. Shaded columns are the `=` positions,
            where each answer is predicted. The bars sit on the positions,
            and the thin lines between them only join them up.
        """,
    )
    def _plot() -> plt.Figure:
        dif, oth, runs = (np.asarray(data[k]) for k in ("difference", "other", "runs"))
        n_slices, n_pos = dif.shape
        x = np.arange(n_pos)
        ramp = 0.35  # the share of each gap the riser takes
        main = light_dark("#c2185b", "#f06292")  # the anchored op, as in the α-by-slice figure
        other = light_dark("#777", "#999")
        fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
        for p in [i for i, t in enumerate(POSITION_LABELS) if t == "="]:  # where each answer is predicted
            ax.axvspan(p - 0.5, p + 0.5, color=light_dark("#eee", "#1c1c1c"), lw=0, zorder=0)
        for si in range(n_slices):
            shift = Affine2D().translate(0, si) + ax.transData
            ax.axhline(si, color=light_dark("#bbb", "#444"), lw=0.5, zorder=0)
            for r in runs:
                smooth_step_marks(
                    ax,
                    x,
                    r[si],
                    ramp=ramp,
                    fillet=2,
                    color=main,
                    lw=0.4,
                    alpha=0.25,
                    transform=shift,
                    clip_on=False,
                    zorder=1,
                )
            smooth_step_marks(
                ax,
                x,
                oth[si],
                ramp=ramp,
                fillet=2,
                color=other,
                lw=0.9,
                transform=shift,
                clip_on=False,
                zorder=2,
                label="other ops" if si == 0 else None,
            )
            smooth_step_marks(
                ax,
                x,
                dif[si],
                ramp=ramp,
                fillet=2,
                color=main,
                lw=1.5,
                transform=shift,
                clip_on=False,
                zorder=3,
                label="difference" if si == 0 else None,
            )
        ax.set_xlim(-0.5, n_pos - 0.5)
        ax.set_ylim(-0.05, n_slices)
        ax.set_xticks(x, POSITION_LABELS, fontsize=6.5)
        ax.set_yticks(np.arange(n_slices) + 0.5, SLICE_NAMES, fontsize=7)
        ax.tick_params(axis="x", bottom=False)
        ax.tick_params(axis="y", left=False)
        ax.spines[:].set(color=light_dark("#ccc", "#444"), lw=0.6)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], ncols=2, frameon=False, fontsize=7, loc="outside upper center")
        sec = ax.secondary_yaxis("right")
        sec.set_yticks([0, 1], ["0", "1"])
        sec.tick_params(labelsize=6, direction="out")
        sec.spines["right"].set_bounds(0, 1)
        sec.set_ylabel("cosine with e₁", fontsize=7, y=0.1, va="top")
        return fig

    return _plot()  # fmt: skip


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
The same runs by position in a context, at every slice:
"""

positions_figure()


r"""
The states at the example answers, in the plane of the two class means:
"""
# The arc: the states at the example answers in the plane of the two class means (arc-in-controls probe, its
# pass over every slice, with the anchored runs from the cooldown experiment's `no-cooldown` arm).
aic = _load_sibling("arc-in-controls", "d22_figures_aic")
ARC_SEED = 715  # the run drawn; the angles between the means are given for all eight seeds
ARC_SPAN = (-25.0, 100.0)  # the part of the disc the cropped figure shows, in degrees from the non-fitting mean


def arc_data() -> dict:
    runs = fetch_json(aic.DISC_REF)["runs"]
    assert len(runs) == 2 * len(SEEDS), len(runs)
    slices = [str(s) for s in range(1, len(SLICE_NAMES))]
    out: dict[str, Any] = {"seed": ARC_SEED, "panels": {}, "angles": {}}
    for kind in ("control", "anchored"):
        mine = [r for r in runs if r["kind"] == kind]
        (one,) = [r for r in mine if r["label"].endswith(f"s{ARC_SEED}")]
        out["panels"][kind] = {s: {"angle": one["slices"][s]["angle_between_means"]} for s in slices}
        out["angles"][kind] = {s: [r["slices"][s]["angle_between_means"] for r in mine] for s in slices}
    return out


def arc_every() -> dict:
    """Every example answer of the two runs at `aic.EVERY_SEED`, by kind and slice: points and fit."""
    store = project_store()
    labels = {"control": aic.ac.label_of(aic.EVERY_SEED, None, "control"), "anchored": aic.cd.label_of("no-cooldown", aic.EVERY_SEED)}  # fmt: skip
    refs = {k: aic.EVERY_REF.format(label=lab) for k, lab in labels.items()}
    arts = store.get_refs(list(refs.values()))
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for k, ref in refs.items():
            art = arts.get(ref)
            assert art is not None, f"not published: {ref}"
            z = np.load(store.get(art, Path(tmp) / f"{k}.npz"))
            out[k] = {"alone": z["alone"].round(3).tolist(), "dist": z["dist"].tolist()} | {
                s: z[f"points/{s}"].round(4).tolist() for s in ("2", "3")
            }
    return out


ARC = arc_data()
assert aic.EVERY_SEED == ARC_SEED
ARC_EVERY = arc_every()


DIST_FLAT = 4  # grid steps from |a − b| at which a tick lies flat, in the variant turned by distance
FIT_RANGE = (
    0.05,
    0.69,
)  # the one-example posterior of an example that fits no op but by noise, and of one only `difference` fits


def arc_panel(ax: plt.Axes, m: dict, anchored: bool, angle_by: str) -> None:
    """One geometry panel (style-fig): the part of the unit disc the states lie in, with no drawn radial edges,
    every example answer on it as a short tick, a tick on the rim at each class mean, and the angle between them. In an anchored run the
    vertical axis is e₁, marked with the solid triangle of the anchor.
    """
    from matplotlib.patches import Arc, Wedge

    t0, t1 = ARC_SPAN
    fill = light_dark("#eee", "#1d1d1d")
    rim = light_dark("#0006", "#fff6")
    red = light_dark("#c2185b", "#f06292")
    grey = light_dark(*GHOST)
    ax.add_patch(Wedge((0, 0), 1, t0, t1, facecolor=fill, lw=0, zorder=-10))
    ax.add_patch(Arc((0, 0), 2, 2, theta1=t0, theta2=t1, color=rim, lw=0.6, zorder=10))
    th = np.radians(m["angle"])
    for a, c in ((0.0, grey), (th, red)):
        ax.plot([np.cos(a), 1.09 * np.cos(a)], [np.sin(a), 1.09 * np.sin(a)], color=c, lw=1.4, solid_capstyle="butt", zorder=11, clip_on=False)  # fmt: skip
    # The angle between the means, as a thin arc just outside the ticks.
    r = 1.15
    ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=0, theta2=m["angle"], color=rim, lw=0.5, zorder=10))
    mid = th / 2
    ax.text(1.2 * np.cos(mid), 1.2 * np.sin(mid), f"{m['angle']:.0f}°", ha="left", va="bottom" if mid > 0.2 else "center", fontsize=6.5)  # fmt: skip
    if anchored:
        ax.plot([0], [1.07], marker="v", ms=5, color=light_dark("#222", "#ddd"), clip_on=False, zorder=11)
    # Each example by how well it fits `difference` (its one-example posterior), from grey and flat to red and
    # upright, so a tick points at 90° × fit from the horizontal; the best fits are drawn last.
    # With `angle_by="dist"` the tick turns instead by how far the answer is from |a − b|: upright on it, flat at
    # DIST_FLAT grid steps or more.
    pts, fit = np.asarray(m["points"]), np.asarray(m["alone"])
    t = np.clip((fit - FIT_RANGE[0]) / np.subtract(*FIT_RANGE[::-1]), 0, 1).round(2)
    u = t if angle_by == "fit" else np.clip(1 - np.asarray(m["dist"]) / DIST_FLAT, 0, 1).round(2)
    for v, w in sorted(set(zip(t.tolist(), u.tolist(), strict=True))):
        k = (t == v) & (u == w)
        ax.scatter(pts[k, 0], pts[k, 1], s=22, marker=(2, 0, round(-90 * (1 - w))), color=fit_ink(v), alpha=0.2, lw=0.6, zorder=2 + v + w / 10)  # fmt: skip
    ax.set_aspect("equal")
    ax.set_xlim(np.cos(np.radians(t1)) - 0.05, 1.45)
    ax.set_ylim(np.sin(np.radians(t0)) - 0.05, 1.22)
    ax.set_axis_off()


def fit_ink(v: float) -> tuple[float, float, float]:
    """Grey at no fit to the anchored red at a full fit."""
    lo, hi = np.array(mcolors.to_rgb(light_dark(*GHOST))), np.array(mcolors.to_rgb(light_dark("#c2185b", "#f06292")))
    return tuple(lo + v * (hi - lo))


def arcs_alt() -> str:
    c, a = ARC["angles"]["control"], ARC["angles"]["anchored"]
    seen = ", ".join(f"slice {s}: {np.median(c[s]):.0f}° and {np.median(a[s]):.0f}°" for s in ("3", "2"))
    alone = np.asarray(ARC_EVERY["anchored"]["alone"])
    return f"""
        Every example answer of one control and one anchored run at seed {ARC_SEED} ({len(alone):,} per panel),
        as faint ticks, at slices 3 (top) and 2, in the plane of the two class means: the non-fitting mean to the
        right and the part of the fitting mean at right angles to it straight up, which in the anchored run is e₁.
        Each tick is colored and turned by how well its example fits `difference`: flat and grey for no fit,
        upright and red for a full fit. In the control the red ticks sit in a short group just above the grey
        ones. In the anchored run they form a band along the rim from the grey group up to e₁, pink and leaning in
        the middle, where the partial fits lie, and red and upright near the top. The median angle between the
        means over eight seeds, control and anchored, is {seen}.
    """


ARC_CAPTION = """
    Every example answer ({n:,} per panel) of one run of each kind at seed {seed}, in the plane of the two class
    means: the non-fitting mean points right, and up is the part of the fitting mean at right angles to it, which in
    an anchored run is e₁ (▼). Each tick is turned and colored by how well its example fits `difference` (the
    posterior on `difference` given that example alone), from flat and grey to upright and red. The ticks on the rim
    mark the class means, and the arc the angle between them. **Columns:** control, anchored. **Rows:** slices 3 and
    2.
"""


def arcs_figure(angle_by: str = "fit") -> str:
    def panel(k: str, s: str) -> dict:
        return ARC["panels"][k][s] | {
            "points": ARC_EVERY[k][s],
            "alone": ARC_EVERY[k]["alone"],
            "dist": ARC_EVERY[k]["dist"],
        }

    data = {"panels": {k: {s: panel(k, s) for s in ("3", "2")} for k in ARC["panels"]}, "seed": ARC_SEED, "angle_by": angle_by}  # fmt: skip
    return arcs_draw(data, arcs_alt())


@memo
def arcs_draw(data: dict, alt_text: str) -> str:
    rows = list(data["panels"]["control"])

    @themed(
        name="d22-arcs" + ("" if data["angle_by"] == "fit" else "-by-distance"),
        alt_text=alt_text,
        caption=ARC_CAPTION.format(seed=data["seed"], n=len(data["panels"]["control"][rows[0]]["alone"])),
    )
    def _plot() -> plt.Figure:
        fig, axs = plt.subplots(len(rows), 2, figsize=(6.4, 5.6), layout="constrained", squeeze=False)
        for i, s in enumerate(rows):
            for j, kind in enumerate(("control", "anchored")):
                ax = axs[i, j]
                arc_panel(ax, data["panels"][kind][s], anchored=kind == "anchored", angle_by=data["angle_by"])
                if i == len(rows) - 1:
                    ax.text(0.5, -0.02, kind, transform=ax.transAxes, ha="center", va="top", fontsize=8)
                if j == 0:
                    ax.text(-0.02, 0.5, f"slice {s}", transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=8)  # fmt: skip
        if data["angle_by"] == "fit":
            key = ((1.0, 1.0, "fits only difference"), (0.5, 0.5, "partial fit"), (0.0, 0.0, "no fit"))
        else:
            key = ((1.0, 1.0, "fits only difference"), (0.5, 1.0, "partial fit"), (0.0, 0.75, "no fit, 1 step off"), (0.0, 0.0, f"{DIST_FLAT}+ steps off"))  # fmt: skip
        handles = [plt.Line2D([], [], color=fit_ink(v), marker=(2, 0, round(-90 * (1 - w))), ms=6, mew=1, ls="", label=lab) for v, w, lab in key]  # fmt: skip
        fig.legend(handles=handles, ncols=len(key), frameon=False, fontsize=7, loc="outside upper center")
        return fig

    return _plot()


arcs_figure()

r"""
The same, with each tick turned by how far its answer is from the one `difference` gives, |a − b|, in grid steps
summed over the channels, and still colored by fit:
"""

arcs_figure("dist")

r"""
How far along the arc each kind of example sits, by its distance from |a − b|:
"""


def arc_distance_rows() -> list[list[str]]:
    groups = [("fits only `difference`", lambda d, f: (d == 0) & (f > FIT_RANGE[1] - 0.04)),
              ("partial fit", lambda d, f: (d == 0) & (f <= FIT_RANGE[1] - 0.04))]  # fmt: skip
    groups += [(f"{n} step{'s' if n > 1 else ''} off", lambda d, f, n=n: d == n) for n in range(1, DIST_FLAT + 1)]
    groups += [(f"{DIST_FLAT + 1}+ steps off", lambda d, f: d > DIST_FLAT)]
    cols = [(k, s) for k in ("control", "anchored") for s in ("3", "2")]
    dist, fit = {k: np.asarray(ARC_EVERY[k]["dist"]) for k in ARC_EVERY}, {k: np.asarray(ARC_EVERY[k]["alone"]) for k in ARC_EVERY}  # fmt: skip
    rows = []
    for name, pick in groups:
        row = [name, f"{pick(dist['anchored'], fit['anchored']).sum():,}"]
        for k, s in cols:
            p = np.asarray(ARC_EVERY[k][s])
            frac = np.degrees(np.arctan2(p[:, 1], p[:, 0])) / ARC["panels"][k][s]["angle"]
            row.append(f"{np.median(frac[pick(dist[k], fit[k])]):.2f}")
        rows.append(row)
    return rows


ARC_DIST_ROWS = arc_distance_rows()
table_html(
    ["example answers", "n", "control, slice 3", "control, slice 2", "anchored, slice 3", "anchored, slice 2"],
    ARC_DIST_ROWS,
    caption=f"""
        **How far along the arc, by distance from `difference`'s answer.** Median position of the example answers
        between the two class means (0 at the non-fitting mean, 1 at the fitting mean, so the angle from the
        non-fitting mean over the angle between them), by how far each answer is from |a − b|, in grid steps summed
        over the channels. Every example answer of one run of each kind at seed {ARC_SEED}; the same examples in
        both runs. The means lie {ARC["panels"]["control"]["3"]["angle"]:.0f}° and {ARC["panels"]["control"]["2"]["angle"]:.0f}°
        apart in the control (slices 3 and 2), {ARC["panels"]["anchored"]["3"]["angle"]:.0f}° and
        {ARC["panels"]["anchored"]["2"]["angle"]:.0f}° in the anchored run.
    """,
)

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


r"""
## Where the answers go

Op confusion: for the contexts of each op (rows), how much mass a model puts on the answers each op could give (columns), with one bar per model in each cell. The first figure has the models with every op, beside ideal₇, the Bayes-optimal predictor over all seven ops; the second has the models with `difference` taken out, by the edit or in training, beside ideal₆, the same predictor with `difference` taken out (the target null). These come from the same seeds before the cooldown experiment took the end anneal out (the [filtered-corpus](../filtered-corpus/report.py) report draws the same matrices as heatmaps), since that experiment kept no answer distributions.
"""

confusion_figure("clean")

r"""
With `difference` taken out:
"""

confusion_figure("edited")

r"""
The same with the two post-hoc erasers that removed the op best without role labels in the [post-hoc comparison](../post-hoc-compare/report.py):
"""

confusion_figure("post-hoc")
