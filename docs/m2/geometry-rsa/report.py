# title: Geometry under anchoring: a whole-geometry read over the stored runs

r"""
# Geometry under anchoring: a whole-geometry read over the stored runs

/// tip |
<!-- tl;dr -->
A reanalysis of 131 stored checkpoints from three experiments. Past the first block, an anchored model does not arrange its colours the way an un-anchored one does. By the last block, the colour geometry of an anchored run correlates with that of a control at half the level two controls reach, in every experiment. Dropping the anchor axis brings the recipe at 50 epochs back to the control band. The heavier anchors, the longer-trained recipe, and the handover arms stay under it, and their seeds agree with each other more closely than controls do. So a light anchor seems to add an axis to a control's geometry; a heavier or longer one also moves the deep geometry to a different arrangement, one that seeds reproduce and one less like the RGB cube.
///

## Observations

Each line below is a read on stored checkpoints, measured against the spread between un-anchored controls. None of them is a result; the closing section says what a preregistered follow-up would test.

- [Whole geometry](#whole-geometry-per-experiment): at the embedding, anchored runs are about as close to a control as controls are to each other. The gap opens with depth. At the last block, the colour geometry of an anchored run correlates with that of a control at roughly half the control-against-control level. That holds for the D2.1 recipe, the six-op recipe at both lengths, the three heavier survey points, and all three handover arms.
- [The anchor axis](#whole-geometry-per-experiment): with e₁ dropped from every run, the six-op recipe at 50 epochs comes back inside the control band at the last two blocks. The D2.1 recipe and the two mid-weight survey points come back most of the way, the 100-epoch recipe and the heaviest point about a third of the way, and the handover arms barely move. So what differs in those last cases is the geometry of the other 63 coordinates. At the embedding, the same read moves every anchored condition a little *under* the band, because it removes *red* from the anchored side only.
- [Agreement among seeds](#agreement-among-seeds): at depth, anchored runs of one condition agree with each other more closely than controls do. With e₁ dropped, the extra agreement stays for the heavier anchors, the 100-epoch recipe, and the handover arms. It goes away for the λ=0.1 recipes at 50 epochs, which are the conditions the axis read put back in the band. So where the geometry differs beyond the axis, seeds reproduce that difference.
- [Dose](#dose): in the full read, the drop at the last block is a step at λ=0.1, with little further change up to λ=0.557. With e₁ dropped, the heavier anchors keep less of the control's geometry than the recipe does.
- [The colour cube](#against-the-colour-cube): at the last two blocks, anchored runs arrange the colours less like the RGB cube than controls do. The embedding and first block are alike.
- [Procrustes](#a-second-statistic-procrustes) agrees with RSA on every read, so a rotation and a rescaling would not remove the difference.

## Scope

This is a reanalysis of stored checkpoints, planned in the [backlog item](/todo/science/global-structure-preserved-under-anchoring.md): no training, no gates, no verdicts. Every anchored model we have has already been scored for the axis it was given (alignment, margin, containment) and for its task. But whether the *rest* of its colour geometry matches what an un-anchored model builds has only been read through per-channel probe R² (the H2 of ex-2.1.12), which came back unresolved. Here we ask that of the geometry as a whole.

## Why

The D2.1 post claims that anchoring guides one concept to a known place and leaves the rest of the representation to form as it would have. That claim rests on the task metric (an anchored model still answers as well as a control) and on probes for the other channels.

Neither one looks at the geometry itself. A model can score the same while arranging its colours differently, and a linear probe can find green wherever it is put. So the claim needs a statistic that compares the *shape* of the representation between an anchored run and an un-anchored one, with the ordinary variation in that shape between un-anchored seeds as the yardstick.

Representational similarity analysis (RSA) is that statistic.[^rsa] For each run it takes the distance between every pair of the 216 grid colours, giving a matrix of distances; it then correlates the matrices of two runs. Two runs that place the colours the same way up to a rotation correlate at 1, whatever basis each of them chose; runs that arrange them differently score lower. The baseline is control against control: runs that differ only by seed set the correlation a faithful anchored run should reach.

[^rsa]: Those matrices of distances are *representational dissimilarity matrices*. Correlating two of them asks whether both runs find the same colours near each other and far apart, without requiring them to use the same coordinates. The measure does not change if you rotate the state space, so it is blind to *where* the anchor put red.

Anchoring is meant to move something: *red* is asked to lie along the e₁ axis. So a second variant of every read drops e₁ from the states of every run before the distances are taken, anchored and control alike. That variant asks whether the geometry *other than the anchor axis* is what a control builds.

## The runs read

Every run comes from a checkpoint a published experiment left in the store. Three experiments, four groups, each with its own un-anchored control seeds.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import experiment as ex
from mini.lit import stop
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

# The five residual slices: the embedding and the output of each block.
SLICES = ("emb", "L1", "L2", "L3", "L4")

SITES = {"op1": "operand 1", "op2": "operand 2"}
VARIANT_TITLE = {"full": "as it is", "axis": "e₁ dropped"}
LAST = ex.N_SLICES - 1


def load_json(ref: str) -> dict | None:
    """A published JSON result as a dict, or None before it exists."""
    store = project_store()
    art = store.get_refs([ref])[ref]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "data.json")])
        return json.loads(path.read_text())


def load_npz(ref: str) -> dict[str, np.ndarray] | None:
    """A published npz as a dict of arrays, or None before it exists."""
    store = project_store()
    art = store.get_refs([ref])[ref]
    if art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "arrays.npz")])
        with np.load(path) as z:
            return {k: z[k] for k in z.files}


@dataclass(frozen=True)
class Group:
    """One experiment's conditions that share a control: the unit every figure is drawn per."""

    exp: str
    name: str
    sources: tuple[ex.Source, ...]

    @property
    def control(self) -> ex.Source:
        return next(s for s in self.sources if s.lam == 0.0)

    @property
    def title(self) -> str:
        return self.exp if self.name == "main" else f"{self.exp}, 100 epochs"


GROUPS: tuple[Group, ...] = tuple(
    Group(exp, name, tuple(s for s in ex.SOURCES if s.exp == exp and s.group == name))
    for exp, name in dict.fromkeys((s.exp, s.group) for s in ex.SOURCES)
)


@dataclass
class Results:
    metrics: dict
    arrays: dict[str, np.ndarray]

    def runs(self, exp: str) -> list[dict]:
        return self.metrics["experiments"][exp]

    def idx(self, src: ex.Source) -> np.ndarray:
        """The row indices of one condition's runs in its experiment's matrices."""
        return np.array([i for i, r in enumerate(self.runs(src.exp)) if r["cond"] == src.cond])

    def mat(self, exp: str, variant: str, stat: str, pos: str, si: int) -> np.ndarray:
        return self.arrays[f"{exp}/{variant}/{stat}/{pos}/{si}"]

    def per_run(self, exp: str, stat: str, pos: str, si: int) -> np.ndarray:
        return self.arrays[f"{exp}/{stat}/{pos}/{si}"]

    def to_control(self, group: Group, src: ex.Source, variant: str, stat: str, pos: str, si: int) -> np.ndarray:
        """Per run of *src*: its mean over the group's control runs, leaving itself out when it is one."""
        m = self.mat(group.exp, variant, stat, pos, si)
        ci = self.idx(group.control)
        return np.array([m[i, ci[ci != i]].mean() for i in self.idx(src)])

    def within(self, src: ex.Source, variant: str, stat: str, pos: str, si: int) -> np.ndarray:
        """Every pair of runs within one condition."""
        m = self.mat(src.exp, variant, stat, pos, si)
        ci = self.idx(src)
        return np.array([m[a, b] for k, a in enumerate(ci) for b in ci[k + 1 :]])

    def control_pairs(self, group: Group, variant: str, stat: str, pos: str, si: int) -> np.ndarray:
        """Every control-against-control pair, the yardstick."""
        return self.within(group.control, variant, stat, pos, si)

    def cond_values(self, src: ex.Source, stat: str, pos: str, si: int) -> np.ndarray:
        return self.per_run(src.exp, stat, pos, si)[self.idx(src)]


def load_results() -> Results | None:
    metrics = load_json(ex.METRICS_REF)
    arrays = load_npz(ex.ARRAYS_REF)
    if metrics is None or arrays is None:
        return None
    return Results(metrics, arrays)


# --- Figure style --------------------------------------------------------------------------------

INKS = {
    "control": ("#6b6b6b", "#b0b0b0"),
    "recipe": ("#c0392b", "#ff8a76"),
    "t12": ("#d98a00", "#ffc04d"),
    "t48": ("#7b3fa0", "#cfa3ff"),
    "t00": ("#2e8b57", "#7fd8a4"),
    "handover-slot": ("#2b6cb0", "#7fb3ff"),
    "handover-tied": ("#7b3fa0", "#cfa3ff"),
}
MARKERS = {
    "control": "s",
    "recipe": "o",
    "t12": "^",
    "t48": "D",
    "t00": "v",
    "handover-slot": "^",
    "handover-tied": "D",
}


def style_key(src: ex.Source) -> str:
    """The style a condition draws in: every control grey, every λ=0.1 recipe red, the rest by name."""
    if src.lam == 0.0:
        return "control"
    if src.cond in INKS:
        return src.cond
    return "recipe"


def ink(src: ex.Source) -> str:
    return light_dark(*INKS[style_key(src)])


def marker(src: ex.Source) -> str:
    return MARKERS[style_key(src)]


def legend_label(src: ex.Source) -> str:
    return f"{src.cond} (λ={src.lam:g})" if src.lam > 0 else src.cond


def dots(ax: Axes, x: float, v: np.ndarray, src: ex.Source, *, rng, ms: float = 5.0, width: float = 0.05, label=None):
    """One column of per-seed dots with the seed mean drawn on top, in the condition's ink and marker."""
    v = np.asarray(v, float)
    color, m = ink(src), marker(src)
    jit = rng.uniform(-width, width, len(v))
    ax.plot([x, x], [np.nanmin(v), np.nanmax(v)], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + jit, v, "o", ms=2.2, color=color, alpha=0.45, zorder=3, mew=0)
    ax.plot(x, np.nanmean(v), m, ms=ms, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6, label=label)


def band(ax: Axes, x: float, v: np.ndarray, *, half: float = 0.45) -> None:
    """The control–control strip: the range of every control pair at one slice."""
    ax.fill_between([x - half, x + half], v.min(), v.max(), color=light_dark("#00000014", "#ffffff1f"), lw=0, zorder=1)


def fig_legend(fig: plt.Figure, ax: Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)


def offsets(n: int) -> np.ndarray:
    return np.linspace(-0.3, 0.3, n) if n > 1 else np.zeros(1)


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str) -> str:
    ths = "".join(f"<th{' class=num' if i else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i else ''}>{cell_html(c)}</td>" for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def sources_table() -> str:
    rows = [[f"`{s.exp}`", f"`{s.cond}`", s.group, f"{s.lam:g}", f"{len(s.seeds)}", s.title] for s in ex.SOURCES]
    return table_html(
        ["experiment", "condition", "group", "λ", "seeds", "what it is"],
        rows,
        "**The runs read.** Every seed of each condition is one checkpoint from the store. Conditions in one group share the control they are read against. ex-2.2.3's `recipe-short` and `recipe` include the fifteen addendum seeds each; ex-2.2.9's `handover-tied` has nine.",
    )


sources_table()

"""
## The read

For each run, the residual state at one site is a `216 × 64` matrix, one row per grid colour. At *operand 1*, the state above the first token depends on that token alone, since attention is causal, so it is the context-free representation of the colour. At *operand 2*, the state depends on the whole prompt so far, and we average it over every first operand under the reference op (`mix`, or `+` on the D2.1 grammar). Both sites are read at all five residual slices.

Each figure below shows, per condition and slice, a column of seed dots: the mean RSA of each run to the controls of its group. For a control run, that mean is taken over the *other* controls. The grey strip behind each slice is the range of control-against-control pairs, so an anchored condition whose dots sit in the strip is as close to a control as controls are to each other.

The lower row of each figure drops e₁, the anchor axis, from the states of every run first. The [checks](#checks-on-the-anchor-axis) below say how much of the variance of each run that coordinate accounts for, and whether it is where a probe reads *red* from.
"""


def rsa_figure(res: Results, group: Group, stat: str = "rsa") -> str:
    ylabel = {"rsa": "RSA to controls", "procrustes": "Procrustes disparity to controls"}[stat]
    conds = group.sources
    what = "correlation of colour-distance matrices" if stat == "rsa" else "residual after the best rotation and scale"
    direction = "higher is more alike" if stat == "rsa" else "lower is more alike"

    @themed(
        name=f"{stat}-{group.exp}-{group.name}",
        alt_text=f"""
            A two-by-two grid of dot charts for {group.title}, rows for the geometry as it is and with e₁ dropped, columns operand 1 and operand 2. Along the bottom of each panel the five residual slices, and up the side {ylabel}. At each slice a grey strip spans the control-against-control pairs and one column of dots per condition sits beside it.
        """,
        caption=f"""
            **{ylabel} for {group.title}, per slice and site.** {ylabel.split(" to ")[0]} is the {what} between one run and each of the group's control runs ({direction}); each small dot is one seed's mean over the controls, the larger mark the seed mean, and the bar the seed range. The grey strip is the range of control-against-control pairs. Top row: the geometry as it is. Bottom row: the anchor axis e₁ dropped from every run first.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), layout="constrained", sharex=True, sharey=True)
        xs = np.arange(ex.N_SLICES)
        off = offsets(len(conds))
        rng = np.random.default_rng(0)
        for r, variant in enumerate(ex.VARIANTS):
            for c, pos in enumerate(ex.POSITIONS):
                ax = axes[r, c]
                for x in xs:
                    band(ax, x, res.control_pairs(group, variant, stat, pos, int(x)))
                    for o, s in zip(off, conds, strict=True):
                        v = res.to_control(group, s, variant, stat, pos, int(x))
                        label = legend_label(s) if (x, r, c) == (0, 0, 0) else None
                        dots(ax, x + o, v, s, rng=rng, width=0.03, label=label)
                ax.set_xticks(xs, SLICES)
                ax.grid(axis="y", alpha=0.2)
                if c == 0:
                    ax.set_ylabel(f"{VARIANT_TITLE[variant]}: {ylabel}", fontsize=8)
                if r == 0:
                    ax.set_title(SITES[pos], fontsize=9)
        fig_legend(fig, axes[0, 0])
        return fig

    return _plot()


def summary_table(res: Results) -> str:
    """The last block at operand 1, per condition: RSA to controls and within the condition, both variants."""
    rows = []
    for g in GROUPS:
        for s in g.sources:
            cells = [f"`{s.exp}`" if s is g.sources[0] else "", f"`{s.cond}`"]
            for variant in ex.VARIANTS:
                cc = res.control_pairs(g, variant, "rsa", "op1", LAST)
                tc = res.to_control(g, s, variant, "rsa", "op1", LAST)
                wi = res.within(s, variant, "rsa", "op1", LAST)
                cells += [f"{cc.min():.2f}–{cc.max():.2f}", f"{tc.mean():.2f}", f"{wi.mean():.2f}"]
            rows.append(cells)
    head = ["", "condition"]
    for variant in ex.VARIANTS:
        head += [f"{VARIANT_TITLE[variant]}: control band", "to controls", "within"]
    return table_html(
        head,
        rows,
        "**RSA at the last block, operand 1.** Per condition: the range of control-against-control pairs, the seed mean of each run's RSA to the controls, and the mean over every pair of runs within the condition, for the geometry as it is and with e₁ dropped. A control's *to controls* leaves each run out of its own comparison, so its two columns are the same number.",
    )


res = load_results()
if res is None:
    stop("_Results are not published yet; the figures render once they are._")


"""
## Whole geometry, per experiment

**What we expected.** If anchoring only moves *red* and leaves the rest alone, an anchored run should sit in the control band once e₁ is dropped, at every slice, and near it before.

**What we saw.** At the embedding, every anchored condition is within a few hundredths of the control band. The gap then opens block by block. At the last block, every anchored condition sits well under the band at both sites: the D2.1 recipe, the six-op recipe at either length, the three heavier survey points, and all three handover arms.

Dropping e₁ closes the gap for the six-op recipe at 50 epochs, which returns to the band at the last two blocks. It closes most of the gap for the D2.1 recipe and the two mid-weight survey points, about a third of it for the 100-epoch recipe and the heaviest point, and almost none on the handover grammar, where the three arms stay far under the band even with e₁ gone.

At the embedding, the e₁-dropped read goes the other way: every anchored condition falls a little under the band. That is what we would expect, since e₁ holds *red* for an anchored run and nothing in particular for a control, so the read removes red from one side of the comparison only. The deeper blocks come back despite that handicap.
"""

rsa_figure(res, GROUPS[0])

"""
ex-2.1.10 has three control seeds, so the band is only three pairs wide and should be read loosely. The recipe falls under the band from the second block on. Dropping e₁ recovers most of the gap at the last two blocks, to within a few hundredths of the band.
"""

rsa_figure(res, GROUPS[1])

"""
The short conditions of ex-2.2.3 are the clearest case for the anchor axis. At the last two blocks, the recipe is back inside the band once e₁ is dropped, `t48` nearly so, and `t12` most of the way. `t00` stays well under it.
"""

rsa_figure(res, GROUPS[2])

"""
The same recipe trained for 100 epochs sits further under its band than the 50-epoch one, and dropping e₁ recovers less of the gap. So the difference seems to grow with training.
"""

rsa_figure(res, GROUPS[3])

"""
On the handover grammar, every arm is far under the band at the last two blocks, and dropping e₁ changes little. The tied readout (`handover-tied`) keeps the most, the slot labeller the least. We read the arms at operand 1 under `mix`. The control shares the eleven-op grammar and stochastic rounding with them, so the grammar itself is not what makes the difference.
"""

summary_table(res)

"""
## Agreement among seeds

**What we expected.** A run whose deep geometry is far from every control could get there two ways. Either anchoring adds seed-to-seed variation, in which case the run is far from every other run as well; or anchoring replaces one arrangement with another, in which case the run is close to the other runs of its own condition. Comparing runs within a condition tells the two apart.

**What we saw.** The second one, in the full read. Past the first block, anchored runs of one condition agree with each other more closely than controls do, in every group. The agreement among controls falls with depth, while the agreement within an anchored condition holds.

With e₁ dropped, that extra agreement stays for the heavier anchors, the 100-epoch recipe, and the handover arms. It goes away for the two λ=0.1 recipes at 50 epochs; in ex-2.1.10 it falls a little under the control level. Those are the same conditions the axis read put back in the band, so the two reads tell one story.

The seeds of a light anchor share an axis and otherwise vary as controls do. The seeds of a heavier or longer anchor share an arrangement beyond the axis, and they reproduce it more closely than controls reproduce theirs.
"""


def within_figure(res: Results) -> str:
    @themed(
        name="within-condition",
        alt_text="""
            Two rows of four dot charts, one column per group, rows for the geometry as it is and with e₁ dropped. Along the bottom the five residual slices, up the side the RSA between pairs of runs within one condition. At each slice a grey strip spans the control pairs and columns of dots sit beside it, one per anchored condition; in the top row the anchored columns sit above the strip from the second block on; in the bottom row the λ=0.1 recipes at 50 epochs drop back to it and the rest stay above.
        """,
        caption="""
            **RSA within each condition, at operand 1.** Each small dot is one pair of runs of the same condition, the larger mark the mean over pairs, and the grey strip the range of control pairs (the same strip as in the figures above). Top row: the geometry as it is. Bottom row: e₁ dropped from every run.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, len(GROUPS), figsize=(10.5, 5.0), layout="constrained", sharex=True, sharey=True)
        xs = np.arange(ex.N_SLICES)
        rng = np.random.default_rng(4)
        for c, group in enumerate(GROUPS):
            anchored = [s for s in group.sources if s.lam > 0]
            off = offsets(len(anchored))
            for r, variant in enumerate(ex.VARIANTS):
                ax = axes[r, c]
                for x in xs:
                    band(ax, x, res.control_pairs(group, variant, "rsa", "op1", int(x)))
                    for o, s in zip(off, anchored, strict=True):
                        v = res.within(s, variant, "rsa", "op1", int(x))
                        dots(
                            ax,
                            x + o,
                            v,
                            s,
                            rng=rng,
                            width=0.03,
                            ms=4,
                            label=legend_label(s) if (x, r) == (0, 0) else None,
                        )
                ax.set_xticks(xs, SLICES, fontsize=7)
                ax.grid(axis="y", alpha=0.2)
                if r == 0:
                    ax.set_title(group.title, fontsize=9)
                    ax.legend(fontsize=6, frameon=False, loc="lower left")
                if c == 0:
                    ax.set_ylabel(f"{VARIANT_TITLE[variant]}: RSA within condition", fontsize=8)
        return fig

    return _plot()


within_figure(res)

"""
## Dose

**What we expected.** If the difference scales with the anchor, the three heavier survey points should sit further from the controls than the recipe, in order of λ.

**What we saw.** In the full read the drop at the last block is a step rather than a slope. The recipe at λ=0.1 is already most of the way down, and the heavier points scatter around it. At the earlier blocks the conditions do fall in order of λ.

With e₁ dropped, the last block is graded too: the recipe returns to the band, `t48` nearly reaches it, and `t12` and `t00` stay further off. So the anchor axis seems to account for a fixed part of the distance at any λ, while the rest grows with the weight.
"""


def dose_figure(res: Results) -> str:
    group = next(g for g in GROUPS if g.exp == "ex-2.2.3" and g.name == "main")
    conds = sorted(group.sources, key=lambda s: s.lam)

    @themed(
        name="dose-ex-2.2.3",
        alt_text="""
            Two rows of two line charts, rows for the geometry as it is and with e₁ dropped, columns operand 1 and operand 2. Along the bottom the anchor weight from 0 to 0.56, and up the side RSA to controls. One line per residual slice joins the seed means at each weight, with the seed dots behind them; the lines for the deeper slices sit lower and fall steeply between 0 and 0.1.
        """,
        caption="""
            **RSA to controls against anchor weight, on ex-2.2.3's short conditions.** Each line is one residual slice (the embedding lightest, the last block darkest), joining the seed means; the dots behind them are seeds. λ=0 is the control, read against the other controls. Top row: the geometry as it is. Bottom row: e₁ dropped from every run.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), layout="constrained", sharex=True, sharey=True)
        cmap = plt.get_cmap("viridis")
        rng = np.random.default_rng(1)
        for r, variant in enumerate(ex.VARIANTS):
            for c, pos in enumerate(ex.POSITIONS):
                ax = axes[r, c]
                for si in range(ex.N_SLICES):
                    color = cmap(light_dark(0.85, 0.95) - 0.7 * si / (ex.N_SLICES - 1))
                    means, lams = [], []
                    for s in conds:
                        v = res.to_control(group, s, variant, "rsa", pos, si)
                        jit = rng.uniform(-0.006, 0.006, len(v))
                        ax.plot(s.lam + jit, v, "o", ms=2.0, color=color, alpha=0.4, mew=0, zorder=2)
                        means.append(v.mean())
                        lams.append(s.lam)
                    ax.plot(
                        lams,
                        means,
                        "-o",
                        ms=4,
                        color=color,
                        lw=1.2,
                        zorder=3,
                        label=SLICES[si] if (r, c) == (0, 0) else None,
                    )
                ax.grid(axis="y", alpha=0.2)
                if c == 0:
                    ax.set_ylabel(f"{VARIANT_TITLE[variant]}: RSA to controls", fontsize=8)
                if r == 0:
                    ax.set_title(SITES[pos], fontsize=9)
                if r == 1:
                    ax.set_xlabel("anchor weight λ")
        fig_legend(fig, axes[0, 0])
        return fig

    return _plot()


dose_figure(res)

"""
## Against the colour cube

A control does not arrange the colours as the RGB cube does either, but it comes closer than an anchored run does. This read correlates each run's colour distances with the straight-line distances between the same colours in RGB.

At the embedding and the first block, the conditions are alike, with the embedding of the tied readout as the one exception. From the second block on, the anchored conditions fall further. At the last block on the handover grammar, a control correlates with the cube at about a half and the handover arms at about a third. The heavier ex-2.2.3 anchors and `handover-tied` sit between.
"""


def cube_figure(res: Results) -> str:
    @themed(
        name="cube-rsa",
        alt_text="""
            Two rows of four dot charts, one column per group and rows for operand 1 and operand 2. Along the bottom the five residual slices, up the side the RSA between each run's geometry and the RGB cube's own distances. Columns of seed dots per condition at each slice; the control columns sit highest at the deeper slices.
        """,
        caption="""
            **RSA between each run's colour geometry and the RGB cube itself, per slice and site.** The cube's distances are the Euclidean distances between the 216 grid colours in RGB. Each dot is one seed, the larger mark the seed mean. A run that arranges the colours as the cube does scores 1.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, len(GROUPS), figsize=(10.5, 4.8), layout="constrained", sharex=True, sharey=True)
        xs = np.arange(ex.N_SLICES)
        rng = np.random.default_rng(2)
        for c, group in enumerate(GROUPS):
            off = offsets(len(group.sources))
            for r, pos in enumerate(ex.POSITIONS):
                ax = axes[r, c]
                for x in xs:
                    for o, s in zip(off, group.sources, strict=True):
                        v = res.mat(group.exp, "full", "cube_rsa", pos, int(x))[res.idx(s)]
                        dots(
                            ax,
                            x + o,
                            v,
                            s,
                            rng=rng,
                            width=0.03,
                            ms=4,
                            label=legend_label(s) if (x, r) == (0, 0) else None,
                        )
                ax.set_xticks(xs, SLICES, fontsize=7)
                ax.grid(axis="y", alpha=0.2)
                if r == 0:
                    ax.set_title(group.title, fontsize=9)
                    ax.legend(fontsize=6, frameon=False, loc="lower left")
                if c == 0:
                    ax.set_ylabel(f"{SITES[pos]}: RSA to cube", fontsize=8)
        return fig

    return _plot()


cube_figure(res)

"""
## Checks on the anchor axis

Two reads of e₁ itself, so that the `e₁ dropped` rows above can be interpreted.

The first is how much of a run's variance sits along e₁. In an anchored run it is twice the control level at the embedding, and it grows with depth, reaching ten to eighteen times the control level at the last block. For a control the share is one part in 64, the same as any other coordinate. So the axis is where anchoring put its variance, as intended.

At the last block on the handover grammar, dropping e₁ removes up to a quarter of an anchored run's total variance, against under two percent for a control.

The second is how closely e₁ lines up with the direction a ridge probe reads *red* from. For anchored runs the cosine between them is about a half at the embedding and the first block, and it falls with depth; for controls it is about a tenth. So the probe finds red mostly along e₁ early and less so late, which is how ex-2.1.12 read it.

Together these say what the `e₁ dropped` rows take away: a coordinate that is a large part of the deep variance of an anchored run and only part of where its *red* lives. The coordinate is largest on the handover grammar, and even there the last-block gap remains once it is gone.
"""


def checks_figure(res: Results) -> str:
    stats = {"e1_share": "variance share of e₁", "cos_e1": "|cos(redness, e₁)|"}

    @themed(
        name="checks",
        alt_text="""
            Two rows of four dot charts, one column per group. The top row shows the share of each run's variance along the anchor axis; the bottom row the cosine between the anchor axis and each run's ridge-fitted redness direction. Along the bottom the five residual slices. Columns of seed dots per condition; the anchored columns sit above the control's in both rows.
        """,
        caption="""
            **The anchor axis at operand 1.** Top: the share of the states' total variance along e₁. Bottom: the absolute cosine between e₁ and the direction a ridge fit (to the grading target `sim_to_red`, power 1.5) reads *red* from. Each dot is one seed, the larger mark the seed mean.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, len(GROUPS), figsize=(10.5, 4.8), layout="constrained", sharex=True, sharey="row")
        xs = np.arange(ex.N_SLICES)
        rng = np.random.default_rng(3)
        for c, group in enumerate(GROUPS):
            off = offsets(len(group.sources))
            for r, stat in enumerate(stats):
                ax = axes[r, c]
                for x in xs:
                    for o, s in zip(off, group.sources, strict=True):
                        v = res.cond_values(s, stat, "op1", int(x))
                        dots(
                            ax,
                            x + o,
                            v,
                            s,
                            rng=rng,
                            width=0.03,
                            ms=4,
                            label=legend_label(s) if (x, r) == (0, 0) else None,
                        )
                ax.set_xticks(xs, SLICES, fontsize=7)
                ax.grid(axis="y", alpha=0.2)
                if r == 0:
                    ax.set_title(group.title, fontsize=9)
                    ax.legend(fontsize=6, frameon=False, loc="upper left")
                if c == 0:
                    ax.set_ylabel(stats[stat], fontsize=8)
        return fig

    return _plot()


checks_figure(res)

"""
## A second statistic: Procrustes

RSA works from distances, so it cannot see a rotation. Procrustes disparity takes a different route to the same question. It finds the rotation, reflection, and scale that best map the states of one run onto those of another, then reports what is left over. So it also compares shape, but through the coordinates rather than the distances.

It agrees with RSA on every read. Anchored runs sit above the control band from the second block on, which here means further from the controls. Dropping e₁ changes little on the handover grammar, and the tied readout keeps the most. The figure shows the handover grammar; the other groups are in the store.
"""

rsa_figure(res, GROUPS[3], stat="procrustes")

"""
## What we make of it

The D2.1 post claims that anchoring leaves the rest of the representation to form as it would have. For a light anchor and a short run, that is what we see here once the axis is set aside: the ex-2.2.3 recipe at 50 epochs is a control geometry plus e₁, and the D2.1 recipe is close to that.

It is not what we see for a heavier anchor, a longer run, or the handover grammar. There the colours are arranged differently beyond the axis, and the difference grows with depth, with training, and with the anchor weight. Seeds reproduce that arrangement more closely than un-anchored seeds reproduce theirs, and it is less like the RGB cube.

Three things this read does not say. It does not say the task is affected: every one of these conditions matched its control on held-out exact match in its own report. It does not say the other channels are lost: ex-2.1.12 found green and blue as decodable as before. And it does not say where in training the two geometries part company, since we only read final checkpoints.

What it says is that the *arrangement* of the colours changes, which is what a whole-geometry statistic measures and a per-channel probe does not.

Two readings are open. The first is that the anchor term reshapes the deep geometry around the axis it is given, so the rest of the space organises relative to *red* rather than as an un-anchored model would have it. The falling RSA against the cube and the rising agreement within a condition both fit that.

The second is that the anchored geometry is the un-anchored one, stretched along e₁ and sheared, in a way that dropping a single coordinate cannot undo. The recipe at 50 epochs fits the stretching part, since dropping e₁ puts it back in the band; whether the handover arms fit the shearing part is open. Telling the two readings apart is a preregistered question, and the statistics here are cheap enough to run at every checkpoint of a training run.

**Next.** A preregistered experiment with one hypothesis: with e₁ dropped, the last-block RSA of an anchored run to the controls is inside the control band. The stored runs already say it holds for the six-op recipe at 50 epochs and misses for the handover grammar. So the experiment should read the handover grammar through training, at every saved checkpoint, and add an arm at a lower λ.

Two exploratory reads to carry with it: comparing each run against a *redness-only* set of distances, which would say whether the anchored arrangement organises by red; and the same figures at the answer position, where the side-effects of an intervention would land.

## Method

**Prompts.** Every ordered pair of the 216 grid colours runs as the three-token prompt `a op b`, with `op` the grammar's `mix` where it has one and `+` on the D2.1 grammar. The residual stream is read at every slice above positions 0 and 2. The state above position 0 is the same for every `b` (checked to 1e-5 in every run); the state above position 2 is averaged over `a`.

**Dissimilarity.** From a `216 × 64` state matrix we take the pairwise Euclidean distances and keep the upper triangle. The states lie on the nGPT hypersphere, so this is the chord distance, which falls as the cosine rises. RSA is then the Pearson correlation between the upper triangles of two runs.

**The anchor axis.** The `e₁ dropped` variant deletes coordinate 0 of every run's states before the distances are taken, and the states are not renormalised. The redness direction in the checks comes from a ridge fit (`l2 = 1e-2`, on centred states) from the states of a run to the grading target `sim_to_red(GRID_RGB, power=1.5)`, normalised to a unit vector.

**Procrustes.** SciPy's `procrustes`. It centres both matrices and scales them to unit Frobenius norm, finds the best orthogonal map from one to the other, and reports the sum of squared residuals as the disparity. Two geometries that differ only by a rotation, a reflection, or a scale score 0.

**Where the data is.** The run table is under the `metrics` ref of `reports/m2/geometry-rsa`, and every pairwise matrix under the `arrays` ref. The raw states of each run are under `states/{experiment}/{label}`, so a follow-up can read them without a forward pass. This pass ran on the dev storage pair, reading the source checkpoints from production. It is to be re-run on production before anything here is quoted.
"""
