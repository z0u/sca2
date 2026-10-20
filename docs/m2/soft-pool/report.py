# ruff: noqa: B018
# title: A soft pool while the anchor settles

# An exploratory report: a pooling temperature that starts soft, and a pull on slices 1 to 3 only, crossed, beside
# the `falloff` recipe of record at the split-anti-seeds seeds, and a pool over slices 1 to 3 with both temperatures
# soft. `experiment.py` beside this script trains, measures and dose-scores the new runs; the `falloff` runs and the
# controls come from split-anti-seeds and ex-2.2.23.
import re

import experiment as ex
from mini.vis import figure_html

ARMS = ("falloff", *(a.name for a in ex.ARMS))
NAME = {"falloff": "falloff"} | {a.name: a.shown_as for a in ex.ARMS}
PULL = {"falloff": list(ex.STEP)} | {a.name: list(a.pull) for a in ex.ARMS}
SOFT = {"falloff": False} | {a.name: a.soft for a in ex.ARMS}
SLICE_TICK = ["emb", "1", "2", "3", "4"]


def frac(w: float) -> str:
    """A pull weight as the fraction it was set from."""
    for d in (1, 2, 3, 4):
        if abs(w * d - round(w * d)) < 1e-9:
            n = round(w * d)
            return str(n) if d == 1 else f"{n}/{d}"
    return f"{w:.2g}"


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


def arms_table() -> str:
    def tau(a: str) -> str:
        if not SOFT[a]:
            return "0.1"
        return f"{ex.TAU_START:g} → 0.1"

    def slice_tau(a: str) -> str:
        arm = next((x for x in ex.ARMS if x.name == a), None)
        if arm is None or not arm.pool_slices:
            return "–"
        return f"{ex.SLICE_TAU_START:g} → {ex.SLICE_TAU:g}"

    rows = [[f"`{NAME[a]}`", *(frac(w) for w in PULL[a]), tau(a), slice_tau(a)] for a in ARMS]
    return table_html(
        ["arm", *SLICE_TICK, "τ_p", "τ_ℓ"],
        rows,
        r"""
        **The arms.** The pull weight on each slice, relative to the anchor weight $\lambda_a$ on each slice of the
        step, and the two pooling temperatures. An arrow is a temperature that sharpens from its first value at epoch 0
        to its second by the end of the pull warm-up (epoch 40). In `pool 1–3, soft τ` the weights only say which
        slices are in the pool. Every arm has the `falloff` anti weight on every slice.
        """,
    )


# %%

rf"""
# A soft pool while the anchor settles

/// tip |
<!-- lede -->
We tried two ways to make the anchor settle more reliably without the step in the pull weight: a pooling temperature that starts soft and sharpens over the warm-up, and a pull on slices 1 to 3 only. TODO: which way it came out.
///

In [the anchor-ramp experiment](/docs/m2/anchor-ramp/report.py), a pull that ramps with depth lost the anchor on two or three of eight seeds, where the step in [`falloff`](/docs/m2/split-anti-seeds/report.py) lost it on none. The lost runs split off from the kept ones during the pull warm-up, around epochs 16 to 24: on a kept run the example answers rise onto e₁ along with the query answer, and on a lost run they fall back. The losses also followed how little pull slices 1 to 3 had, though that ordering was picked after the counts were in.

Both readings suggest a change. The pull is pooled over the positions of each context at a sharp temperature $\tau_p$ = 0.1, so while the query answer is the closest position to e₁, the pull goes almost all to it. A soft pool early would share the pull with the example answers while they are still deciding. And a pull on slices 1 to 3 alone, with nothing on slice 4, would put the pull where the example answers are what it pulls.

The {len(ex.ARMS)} new arms, of {len(ex.SEEDS)} runs each, cross the two changes and add a pool over slices 1 to 3 with both temperatures soft. They use the seeds of the split-anti-seeds experiment, so its `falloff` runs serve as the baseline with no new training.
"""

# %%

rf"""
## Observations

- Removal and spill (E1): TODO.
- When the runs settle (E2): TODO.
- Where the anchor sits by depth (E3): TODO.

## Scope

This is an exploratory study, with no preregistration and no gate. Two factors are crossed, with `falloff` as the condition where both are at their recipe level:

- The pull shape: the step on slices 2 to 4 (the recipe), or a ramp over slices 1 to 3 at a third, two thirds and the full weight, with nothing on the embedding or slice 4. The ramp adds up to the step over slices 1 to 3 (two each), so it moves the pull of slice 4 away rather than adding to the earlier slices.
- The position pool temperature $\tau_p$: fixed at 0.1 (the recipe), or soft-to-sharp, starting at {ex.TAU_START:g} and sharpening to 0.1 by epoch 40, the end of the pull warm-up. The move is smooth in log $\tau_p$, so the pool halves its temperature at a steady pace and comes to rest at 0.1.

{arms_table()}

The fourth arm pools the pull over the slice × position pairs of slices 1 to 3 (the nested pool of [the slice-terms experiment](/docs/m2/slice-terms/report.py)), so the pull can choose the slice as well as the position. Its slice pool temperature $\tau_\ell$ starts at {ex.SLICE_TAU_START:g}, near a plain mean over slices, and sharpens to {ex.SLICE_TAU:g} over the same window as $\tau_p$. The end value rests on one seed: in E4 of slice-terms, $\tau_\ell$ from 0.2 to 1.6 at seed 701 all removed about as much and stayed within the criterion, with no trend, and the joint pool at 0.1 lost the anchor on one seed of four. So {ex.SLICE_TAU:g} sits inside the range that worked there.

A soft pool and a ramp each change more than one thing, so it helps to list them:

- A softer $\tau_p$ spreads the pull of each context over its positions, but the total pull per context stays the same at every temperature (the softmin weights sum to one). In [the τ×λ_a sweep](/docs/m2/tau-lambda-sweep/report.py), a softer fixed $\tau_p$ meant more spill, so spill is the thing to watch; here it is soft only during the warm-up.
- The ramp pulls two thirds as much in total as the step, all of it moved off slice 4. A difference between the ramp and the step can come from the shape or from the lower total.
- The pool over slices 1 to 3 pulls as much in total as the step, and lets the pull choose where it goes.

Every run trains at model seeds {ex.SEEDS[0]} to {ex.SEEDS[-1]} for 400 epochs, paired by seed with its `falloff` twin and with ex-2.2.23's control (same initialization, batches and label draws). With eight seeds per arm, one or two lost runs could come from the seeds alone, so we weigh the counts across arms together.

## The measurements

The measurements and dose scoring are those of the split-anti-seeds experiment.

- Removal: the net drop in the score on `difference` under the edit, as a share of the way from the clean score to the target null. The edit projects e₁ out of the state at every position and slice, at a dose of a quarter, a half, three quarters, or all of the e₁ component. Higher is better.
- Spill: the largest net drop on another op under the edit. A run is within the criterion when its spill is at most 0.02 at every dose.
- Both are net of the control at the same seed.
- A run lost the anchor when the full edit removes less than half of the op, as in the split-anti-seeds experiment.
- Task change: the mean score over ops, less that of the control. Zero means no damage.
- The alignment α with e₁ at the example answers and at the query answer, at every slice, along training and at the end.
"""

# %%

r"""
## Removal and spill (E1)

Does a soft pool early, or a pull on slices 1 to 3 alone, keep the anchor on every seed, and does it keep the edit as selective as `falloff`? The main measure is the removal at the full dose, with the spill at any dose beside it. The crossing separates the two factors: the soft step against `falloff` isolates the temperature, and the fixed ramp against `falloff` isolates the shape.

TODO.
"""

# %%

r"""
## When the runs settle (E2)

Does the soft pool carry the example answers onto e₁ with the query answer through the warm-up, where the lost runs of the anchor-ramp experiment fell back? The main measure is α at the example answers along training, with the temperature and the pull weight in strips underneath, and α at the query answer beside it.

TODO.
"""

# %%

r"""
## Where the anchor sits by depth (E3)

On the runs that kept the anchor, where does the alignment sit by slice? With nothing on slice 4, does slice 4 still hold the concept as the residual stream carries it up, and where does the pool over slices 1 to 3 settle, given that the pool over every block settled on block 4 in the slice-terms experiment?

TODO.
"""

# %%

r"""
## Discussion

TODO.
"""

# %%

r"""
## Method

The pooling temperatures are step inputs here, scheduled like the weights (`AnchorSpec.tau_start`, `slice_tau_start` and `tau_anneal_end`); before this experiment they were fixed when the training step was built. Training otherwise follows the split-anti-seeds experiment unchanged. The trajectory of each run also keeps α by slice and position for each op separately, and the e₁ component of every syntax embedding, for later use.
"""
