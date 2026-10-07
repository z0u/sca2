# ruff: noqa: B018
# title: Ex 2.2.23: The slow seeds, trained for longer

# The design constants come from `experiment.py` beside this script (the directory of the script is on sys.path
# while it runs). This is the design draft: the only computed figure is the HSV skill through training of the
# ex-2.2.21 runs this scout reuses, from that experiment's published trajectories.
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, text_cols: int = 1) -> str:
    """An authored table in the shared report style; the first *text_cols* columns are text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        "<tr>"
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for row in rows
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def seed_span(seeds: tuple[int, ...]) -> str:
    return f"{seeds[0]}–{seeds[-1]}"


# --- The earlier runs ---------------------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as _tmp:
    _store = project_store()
    _art = _store.get_refs([ex.ex2221.TRAJ_REF])[ex.ex2221.TRAJ_REF]
    assert _art is not None, "ex-2.2.21 has not published its trajectories"
    (_path,) = _store.get_many([(_art, Path(_tmp) / "traj21.json")])
    TRAJ21 = json.loads(_path.read_text())

HSV_IDX = [ex.OP_NAMES.index(o) for o in ex.HSV_OPS]


def hsv_skill(traj: dict) -> tuple[np.ndarray, np.ndarray]:
    """The epoch of each trajectory record, and the EEM averaged over the HSV ops there."""
    pts = [(e, np.mean([v[i] for i in HSV_IDX])) for e, v in zip(traj["epoch"], traj["eem_per_op"], strict=True) if v]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


EARLIER = {
    c.name: {s: hsv_skill(TRAJ21[f"{c.reused_as}-s{s - ex.SEED_OFFSET}"]["traj"]) for s in ex.REUSED_SEEDS}
    for c in ex.CONDITIONS
}
EARLIER_MISSED = {c: [s for s, (_, y) in runs.items() if y[-1] < ex.RISE_LEVEL] for c, runs in EARLIER.items()}


def earlier_figure() -> str:
    caption = f"""
        **The HSV skill through training, in the ex-2.2.21 runs this scout reuses.** EEM averaged over the three HSV
        ops, on a subsample of the held-out set, one line per model seed. The dashed line is the level a run has to
        pass to count as having made the second rise ({ex.RISE_LEVEL:g}).
    """
    missed = ", ".join(f"{s}" for s in EARLIER_MISSED["anchor"])
    alt = f"""
        Two panels, the control and the anchored condition, each plotting HSV skill against the epoch for five runs.
        Every run sits near 0.15 to 0.2 for at least the first quarter of training. On the control every run then
        rises past the dashed line, two of them late in training. On the anchored condition three runs rise and two, at model
        seeds {missed}, stay below the line to the end.
    """
    return earlier_draw(EARLIER, caption, alt)


@memo
def earlier_draw(data: dict, caption: str, alt_text: str) -> str:
    @themed(name="ex-2.2.23-earlier", alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), layout="constrained", sharey=True)
        ink = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(ex.REUSED_SEEDS)))
        for ax, c in zip(axes, ex.CONDITIONS, strict=True):
            for color, (s, (x, y)) in zip(ink, data[c.name].items(), strict=True):
                ax.plot(x, y, color=color, lw=1.0, label=str(s))
            ax.axhline(ex.RISE_LEVEL, color=light_dark("#000", "#fff"), lw=0.8, ls="--", alpha=0.6)
            ax.set_title(c.name, fontsize=9)
            ax.set_xlim(0, ex.SHORT)
            ax.set_xlabel("epoch", fontsize=8)
        axes[0].set_ylabel("HSV skill (EEM)", fontsize=8)
        axes[1].legend(title="model seed", fontsize=7, title_fontsize=7, frameon=False, loc="upper left")
        return fig

    return _plot()


def runs_table() -> str:
    rows = []
    for c in ex.CONDITIONS:
        rows.append([f"`{c.name}`", f"{ex.SHORT}", f"{seed_span(ex.REUSED_SEEDS)}", f"ex-2.2.21 (`{c.reused_as}`)"])
        rows.append([f"`{c.name}`", f"{ex.SHORT}", f"{seed_span(ex.NEW_SEEDS)}", "new"])
        rows.append([f"`{c.name}`", f"{ex.LONG}", f"{seed_span(ex.SEEDS)}", "new"])
    return table_html(
        ["condition", "epochs", "model seeds", "runs"],
        rows,
        f"**The runs.** {ex.N_NEW_RUNS} new runs, and {len(ex.CONDITIONS) * len(ex.REUSED_SEEDS)} reused from "
        "ex-2.2.21, which trained both conditions at the same settings.",
        text_cols=4,
    )


# %%

rf"""
# Ex 2.2.23: The slow seeds, trained for longer

/// tip |
<!-- lede -->
A scout. We train the anchored recipe and the control at {len(ex.SEEDS)} seeds, for 200 epochs and for 400, to see how often a run misses the second rise in task skill, and whether a longer run makes it.
///

In ex-2.2.21 and ex-2.2.22, a few runs never made the second rise in task skill within 200 epochs, and those runs set most of the seed band of every measurement. This scout trains the recipe of record and the control at {len(ex.NEW_SEEDS)} new seeds at 200 epochs, and at all {len(ex.SEEDS)} seeds at 400 epochs, reusing ex-2.2.21's 200-epoch runs at the other {len(ex.REUSED_SEEDS)}.
"""

# %%

r"""
## Observations

Each item below is a measurement on the runs of this scout, with no gate.

- [How often a run misses the rise (E1)](#how-often-a-run-misses-the-rise-e1):
- [The same seeds at 400 epochs (E2)](#the-same-seeds-at-400-epochs-e2):
- [A late rise and an early one (E3)](#a-late-rise-and-an-early-one-e3):
- [A rule for runs that miss the rise (S1)](#a-rule-for-runs-that-miss-the-rise-s1):
- [The edit, with and without the rise (E4)](#the-edit-with-and-without-the-rise-e4):
- [Decision](#decision):
"""

# %%

rf"""
## Scope

This is a scout, with no preregistration and no gate. Each condition has {len(ex.SEEDS)} runs at 200 epochs and {len(ex.SEEDS)} at 400, so a share of runs that miss the rise is known only roughly: one run more or less moves it by about a twelfth. The pairing by seed is what makes the comparisons between the two conditions, and between the two lengths, worth more than the counts alone.
"""

# %%

rf"""
## Why this experiment

The model learns the seven ops in two stages. Early in training it learns `mix`, `lighten`, `darken`, and `{ex.ANCHORED_OP}`, and its task skill rises quickly to a plateau. Some tens of epochs later it learns the three HSV ops, which change one channel of a color in hue, saturation, and value space, and the skill rises a second time. When that second rise comes varies from run to run. A few runs never make it within 200 epochs: their HSV skill stays near where it was on the plateau, and they end well below the others, though they answer the other four ops about as well.

In the 200-epoch runs this scout reuses, that happened on the anchored condition only, at two of the five seeds:

{earlier_figure()}

So the seed band of our measurements is mostly a record of which runs made the second rise. Ex-2.2.22 could not tell a small cost of the anchor from a slow start, and along its caps, the runs whose edit spilled onto other ops were all runs that had made the rise. If every run makes the rise when given time, a future experiment could keep the 200-epoch budget and leave out the runs that miss it, as half-trained models. Before that is safe, we need to know three things:

- How often a run misses the rise at 200 epochs, and whether the anchor changes that. Pairing by seed separates a seed that is slow under any condition from one the anchor makes slow. If the anchor decides who is slow, leaving the slow runs out would hide a cost of the anchor, so the number left out of each condition would have to be reported as a measurement in its own right.
- Whether a run that missed the rise at 200 epochs makes it at 400. If it doesn't, leaving it out selects a kind of seed rather than waiting for a slow one.
- Whether a run that rises late ends like one that rises early, in task skill, in how well the anchor holds, and in the edit.

A 400-epoch run differs from its 200-epoch twin from the start, since the learning rate decays more slowly over a longer schedule, so the two are paired by seed and not by trajectory. A run that rises at 400 epochs and stalls at 200 says that the seed can make the rise under a longer schedule, which is what a policy for the next experiment needs.

Any rule for leaving runs out has to look at task skill only, and be fixed before the edit is scored, so that it cannot select on the anchoring results. That matters here because the edit spilled only on runs that had made the rise.
"""

# %%

rf"""
## Parameters

The recipe of record, as the [D2.2 design](/docs/m2/d2.2/design.md#the-setup-today) states it: the seven-op set, three examples per context with replacement op noise of 0.3, {ex.MODEL}, the newline mask, the whole-line label on about one `{ex.ANCHORED_OP}` context in fifty, every slice pulled, and no cap. That is ex-2.2.21's `anchor-whole` condition. The control is the same without the anchor.

{runs_table()}

**The lengths.** 200 epochs is the recipe; 400 was the recipe before ex-2.2.19 halved it, and every 400-epoch run there (unanchored, three seeds) made the second rise by about a third of the way through. The anchor schedules scale with the length: the weight warms up over the first tenth of training and anneals over the last tenth. The learning rate peaks at the same value and warms up over the same number of epochs at either length, and its cosine decay stretches over the run.

**The seeds.** The reused runs are at model seeds {seed_span(ex.REUSED_SEEDS)}, and the new ones at {seed_span(ex.NEW_SEEDS)}, so every comparison is paired by seed: a condition with its control, and a 200-epoch run with its 400-epoch twin.

/// admonition | Open decision
Twelve seeds and one longer length (400). The alternatives: 300 epochs, which costs less but may stop short of the rise for the slowest seeds (two of the ex-2.2.21 runs above had not risen by epoch 200); or more seeds at 200 epochs and fewer at 400, which would pin down how often a run misses the rise at the cost of fewer longer runs. To check: the planned cost is in the method, and the counts of E1 shrink by a twelfth per seed dropped.
///
"""

# %%

rf"""
## Measurements

**HSV skill.** Expected exact match (EEM, the probability the model puts on the right answer) on held-out contexts, averaged over the three HSV ops. Through training it is measured every {ex.TRAJ_STRIDE_EPOCHS} epochs on a subsample of the held-out set, and at the end on the whole of it.

**The rise.** A run has made the second rise once its HSV skill passes {ex.RISE_LEVEL:g}, about halfway between the plateau and where the runs that rise end up. The *rise epoch* is the first record at or above that level. Runs that rise do so quickly, so the level matters little to the epoch; the figure above shows it against the earlier runs. The three HSV ops can rise at different times, and on a few runs only `hue-hsv` rises within 200 epochs, which leaves the average near the level. E1 shows the ops one at a time for that reason.

**Task skill.** EEM on held-out contexts over all seven ops, compared with the control at the same seed and length.

**The anchor.** The op margin at the last slice: how far `{ex.ANCHORED_OP}` contexts sit along e₁ beyond the rest, as in ex-2.2.21.

**The edit.** The projection of e₁ at every position, at doses from a quarter to all of the component, with the two criteria of ex-2.2.21 (E2): the drop on `{ex.ANCHORED_OP}` grows with the dose and reaches at least {ex.GRADING_MIN_DAMAGE:.0%} of the way to the target null, and no other op drops by more than {ex.SELECTIVITY_GATE:g} at any dose. Both drops are net of what the edit does to the control at the same seed and length.
"""

# %%

r"""
## How often a run misses the rise (E1)

How many runs of each condition end 200 epochs without having made the rise, paired by seed.

/// admonition | TODO
The HSV skill through training at 200 epochs, one panel per condition and one line per seed, with the rise level marked. Beside it, a table of the runs that miss the rise, by seed and condition.
///
"""

# %%

r"""
## The same seeds at 400 epochs (E2)

Whether the runs that missed the rise at 200 epochs make it at 400, and when the rise comes as a share of training at either length.

/// admonition | TODO
The HSV skill through training at 400 epochs, laid out as in E1, with the seeds that missed at 200 epochs highlighted. Beside it, the rise epoch of every run, at 200 and 400 epochs, as a share of training.
///
"""

# %%

r"""
## A late rise and an early one (E3)

Whether a run that rises late ends like one that rises early, in task skill and in the op margin.

/// admonition | TODO
Task skill at the end of training against the rise epoch, one dot per run, both lengths and both conditions; the same for the op margin on the anchored runs.
///
"""

# %%

r"""
## A rule for runs that miss the rise (S1)

A rule that marks a run as half-trained, for the next experiment to leave out and replace with the next unused seed. The rule looks only at the HSV skill on the held-out set at the end of training. Its level is chosen from E1 to E3 and committed before E4 is filled in, so that the choice cannot follow the edit results. We report what each candidate level would leave out of each condition at either length.

/// admonition | TODO
For each candidate level, the number of runs it leaves out of each condition, at 200 and at 400 epochs, and the commit that fixed the chosen level.
///
"""

# %%

r"""
## The edit, with and without the rise (E4)

Whether the edit meets its two criteria on the runs the rule keeps and on the runs it leaves out, and whether it spills onto other ops more often on one group.

/// admonition | TODO
The two edit criteria for every anchored run, at both lengths, grouped by whether the rule of S1 keeps the run.
///
"""

# %%

r"""
## Decision

How the next experiment trains and counts its runs. The candidates:

- 200 epochs, with the rule of S1 leaving runs out and topping up from the next unused seed, and the count left out of each condition reported.
- 400 epochs, keeping every run.
- 200 epochs, keeping every run, at more seeds.

The criteria: how many runs the rule leaves out of each condition, and whether the anchor changes it (E1); whether the runs that missed at 200 epochs make the rise at 400 (E2); whether a late rise ends like an early one (E3); whether the edit results differ between the runs the rule keeps and the runs it leaves out (E4); and the cost.

/// admonition | TODO
Every criterion for every candidate, and the choice.
///
"""

# %%

r"""
## Discussion

/// admonition | TODO
After the results.
///
"""

# %%

rf"""
## Method

**Reused runs.** The 200-epoch runs at model seeds {seed_span(ex.REUSED_SEEDS)} are ex-2.2.21's `control` and `anchor-whole`, resolved by ref along with the corpus, held-out set, and probes they trained and were scored on. The new runs train on the same corpus with ex-2.2.21's code, so a new 200-epoch run differs from a reused one in its seed alone.

**Scoring.** Every run, reused or new, is scored with ex-2.2.21's eval and with the suppression pass of ex-2.2.22, at every position.

**Budget.** Planned at under \${ex.BUDGET_USD}: {ex.N_NEW_RUNS} runs at about \$0.13 for 200 epochs on an L4 (the cost of ex-2.2.22's runs) and twice that for 400, with the scoring passes.
"""
