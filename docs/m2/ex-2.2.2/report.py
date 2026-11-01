# title: Ex 2.2.2: fallback control in the anchored transformer

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

# The design constants and the result refs come from `experiment.py` beside
# this notebook (the document's directory is on sys.path while it runs). The
# prose quotes the frozen gates, and that module carries the same numbers
# with each gate's wording in its docstring.
import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import light_dark, themed

POS_NAMES = ["op1", "+", "op2", "=", "ans", "⏎"]
INK = {
    ex.FALLBACK.name: ("#d40000", "#f44"),
    ex.NO_FALLBACK.name: ("#555", "#aaa"),
    ex.CONTROL.name: ("#999", "#777"),
    "fb-only": ("#e07000", "#fa4"),
    "anti-only": ("#8030c0", "#c8f"),
    "fb-w0.01": ("#f0a0a0", "#a04040"),
    "fb-w0.25": ("#800000", "#f88"),
    "recipe": ("#2060c0", "#7af"),
    "lunar": ("#108060", "#5fd0a0"),
}
# One ink per condition (or E6's fitted row) the figures compare, as (light, dark) pairs for `light_dark`.


def load_results() -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    """Resolve the metrics, the stacked per-run arrays, and the trajectories from the store, or None if unpublished."""
    store = project_store()
    arts = store.get_refs([ex.METRICS_REF, ex.ARRAYS_REF, ex.TRAJ_REF])
    m_art, a_art, t_art = arts[ex.METRICS_REF], arts[ex.ARRAYS_REF], arts[ex.TRAJ_REF]
    if m_art is None or a_art is None or t_art is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        m_path, a_path, t_path = store.get_many(
            [(m_art, Path(d) / "metrics.json"), (a_art, Path(d) / "arrays.npz"), (t_art, Path(d) / "trajectories.npz")]
        )
        with np.load(a_path) as z:
            arrays = {k: z[k] for k in z.files}
        with np.load(t_path) as z:
            traj = {k: z[k] for k in z.files}
        metrics = json.loads(m_path.read_text())
    return metrics, arrays, traj


def probe_lines() -> ex.Lines:
    """The probe set as the scorer grouped it.

    The tokenizer is a sorted vocabulary, so the one the runs used can be rebuilt from the palette alone;
    `read_lines` checks that the probe set's op1 column walks that palette in order.
    """
    from sca.config import TokenizerConfig
    from sca.data.named_colors import GRIDS, SYNTAX, WordTokenizer, grid_palette

    store = project_store()
    art = store.get_refs([ex.PROBE_REF])[ex.PROBE_REF]
    assert art is not None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "probes.npz")])
        with np.load(path) as z:
            tokens, r1, r2, redness = z["probe_tokens"], z["r1"], z["r2"], z["redness"]
    tokenizer = WordTokenizer(TokenizerConfig(vocabulary=[*SYNTAX, *grid_palette(GRIDS[ex.GRID])]))
    return ex.read_lines(tokens, r1, r2, redness, tokenizer, tokenizer.vocab_size)


def span(v: np.ndarray, fmt: str = ".3f") -> str:
    """Seed mean with the seed range beside it."""
    return f"{v.mean():{fmt}} ({v.min():{fmt}}–{v.max():{fmt}})"


def stored_trajectories() -> dict[str, dict[str, list[float]]]:
    """The training trajectories ex-2.1.10 published, by run label, for the no-fallback condition."""
    store = project_store()
    art = store.get_refs([ex.EX2110_METRICS_REF])[ex.EX2110_METRICS_REF]
    assert art is not None
    with tempfile.TemporaryDirectory() as d:
        (path,) = store.get_many([(art, Path(d) / "metrics.json")])
        cells = json.loads(path.read_text())["cells"]
    return {c["label"]: c["traj"] for c in cells}


# The loaders and the statistic helpers below are hoisted above the first prose
# that reads them (the Findings paragraph): Marimo ran cells by dependency, but
# a literate script runs top to bottom.
loaded = load_results()
assert loaded is not None, "ex-2.2.2 has not published its results"
metrics, arrays, traj = loaded
lines = probe_lines()
CONDS = (ex.CONTROL, ex.NO_FALLBACK, ex.FALLBACK, *ex.ARMS)
runs: dict[str, list[dict]] = {
    c.name: [r for r in metrics["runs"] if r["condition"] == c.name and r["exp"] == c.exp] for c in CONDS
}
assert [len(runs[c.name]) for c in CONDS] == [c.seeds for c in CONDS]
stored = stored_trajectories()


def stat(iv: str, key: str, group: str | None = None, cond: str = ex.FALLBACK.name) -> np.ndarray:
    """One value per seed of a condition, for one statistic of one intervention."""
    cells = [r["interventions"][iv][key] for r in runs[cond]]
    return np.array(cells if group is None else [c[group] for c in cells], float)


def clean(key: str, group: str | None = None, cond: str = ex.FALLBACK.name) -> np.ndarray:
    """One value per seed of a condition, for one clean-pass statistic."""
    cells = [r["clean"][key] for r in runs[cond]]
    return np.array(cells if group is None else [c[group] for c in cells], float)


def per_line(iv: str, key: str, cond: str = ex.FALLBACK.name) -> np.ndarray:
    """(seeds, ...) one intervention's per-line arrays, stacked over the seeds of a condition."""
    return np.stack([arrays[f"{r['exp']}/{r['label']}/{iv}/{key}"] for r in runs[cond]])


def floor(iv: str, key: str, group: str) -> float:
    """The resolution for one statistic: two pooled between-seed sds over the fallback and no-fallback conditions."""
    var = np.array([stat(iv, key, group, c.name).var(ddof=1) for c in (ex.FALLBACK, ex.NO_FALLBACK)])
    return ex.RESOLUTION_SD * float(np.sqrt(var.mean()))


def trajectory(key: str, cond: str = ex.FALLBACK.name) -> np.ndarray:
    """(seeds, records) one trajectory key over the seeds of a condition; the stored condition's come from ex-2.1.10."""
    labels = [r["label"] for r in runs[cond]]
    if cond == ex.NO_FALLBACK.name:
        return np.array([stored[label][key] for label in labels], float)
    return np.stack([traj[f"{label}/{key}"] for label in labels]).astype(float)


def seed_row(v: np.ndarray, fmt: str = ".3f") -> str:
    return span(v, fmt)


r"""
# Ex 2.2.2: a designed response to suppressing *red*

/// tip |
<!-- tl;dr -->
We add fallback control: a training term that teaches the blocks what to answer once the concept is gone, aiming at a designed fallback answer, with the concept's placement kept out of its gradient. Does the fallback appear, does the intervention stay effective and selective, and does seed variability fall?
The fallback appears in every seed, costs nothing on the task and a little margin, and reaches the plain projection at more than half strength. The reflection it was trained at is destructive on non-red lines in every anchored model, with or without the term, because the syntax token embeddings carry the axis.
///
"""

findings_fb = stat("redirect", "fb_acc", "red_clean").mean()
findings_fb_ref = stat("redirect", "fb_acc", "red_clean", ex.NO_FALLBACK.name).mean()
findings_gap = clean("acc", "all", ex.NO_FALLBACK.name).mean() - clean("acc", "all").mean()
findings_ratio = clean("m_span").mean() / clean("m_span", cond=ex.NO_FALLBACK.name).mean()
findings_d_red = stat("redirect", "deficit", "nonred").mean()
findings_d_red_ref = stat("redirect", "deficit", "nonred", ex.NO_FALLBACK.name).mean()
findings_d_proj = stat("projection", "deficit", "nonred").mean()
findings_d_proj_ref = stat("projection", "deficit", "nonred", ex.NO_FALLBACK.name).mean()
findings_g = [stat(f"gamma-{g}", "fb_acc", "red_clean").mean() for g in ex.GAMMAS]

rf"""
## Findings

- [The designed response (H1)](#the-designed-response-h1) — **holds.** Fallback accuracy on the red lines with a clean visible operand under `redirect`: {findings_fb:.3f}, against {findings_fb_ref:.3f} without the term; gate {ex.FALLBACK_ACC_GATE:g}, and the margin above the reference clears the {floor("redirect", "fb_acc", "red_clean"):.3f} floor.
- [Task and placement intact (H2)](#task-and-placement-intact-h2) — **holds.** Clean accuracy gap from the no-fallback condition: {findings_gap:.4f} (gate {ex.TASK_GATE:g}). Margin at the end of training: {findings_ratio:.3f} of the no-fallback value (gate {ex.MARGIN_RATIO:g}).
- [Selectivity (H3)](#selectivity-h3) — **partial.** Non-red deficit under `redirect`: {findings_d_red:.3f} (gate {ex.TASK_GATE:g}; the no-fallback condition loses {findings_d_red_ref:.3f} under the same edit). Under `projection`: {findings_d_proj:.3f} against {findings_d_proj_ref:.3f}. That difference is smaller than the {floor("projection", "deficit", "nonred"):.3f} resolution floor, so the second clause holds, but as an unresolved difference.
- [Transfer from the antipode to zero (H4)](#transfer-from-the-antipode-to-zero-h4) — **holds.** Fallback accuracy at γ = 1 is {findings_g[1] / findings_g[3]:.2f} of its value at γ = 2 (gate {ex.TRANSFER_FRAC:g}), and the sweep has no dip.

<!-- REVIEW: a seed-agreement hypothesis (then H2) was cut: at the H1 gate it is close to implied by H1, and the case that separates them (seeds agreeing on an answer other than the fallback answer) is what E1's composition shows. Seed agreement is reported under E1, ungated. The later hypotheses moved up one number. -->
"""

r"""
/// admonition | How to read this draft
The hypotheses, their gates, and the method were frozen at commit `d72340e7`, before any run. Each hypothesis section opens with that frozen prediction, and its results follow in place.

Two corrections landed after the freeze, each with a `REVIEW` note beside the text it changed. The first is the count of red lines with a clean visible operand, now 296 rather than 298, because one color sat on the 0.5 contour to roundoff. The second is the gradient paragraph in the method, which now accounts for the tied embedding.

Anything conceived after seeing the data is under [Exploratory analyses](#exploratory-analyses), marked as post hoc (E7 and E8).
///

## Why this experiment

[Ex-2.2.1](../ex-2.2.1/report.py) showed that projecting the anchor axis out of the D2.1 checkpoints suppresses *red*. Red-line accuracy falls from 1.0 to 0.09, in proportion to how red the line is, and stays inside the bound the placed geometry set.

A suppressed red line still decodes to a color, about half the time a one-step neighbor of the true mix. Which neighbor varies by seed: at least five of the nine seeds agree on 13% of red lines. Nothing in training said what a removed concept should decode to, so the answer is whatever the untrained region of the stream produces. M1 called that spoofing, and it is where the seed spread in [ex-2.9.1](/docs/m1/ex-2.9.1/report.py) came from.

The remedy in M1 was fallback control ([ex-2.9.2](/docs/m1/ex-2.9.2/report.py)): one loss term, applied to the decoder only, that pins the antipode of the anchor axis to a designed null answer, mid-gray. Reflecting a state through the axis then collapsed the response to a tight cluster on the bound the null predicted, across 32 seeds. Under plain zeroing the term added little, and that report said a trained fallback should be paired with the redirect it was trained at.

Here we bring the term into the transformer, keeping the D2.1 grammar and recipe. This comes before the operation work in the [D2.2 plan](../d2.2/design.md) changes the grammar.

The term is one loss, but it touches the model in three places:
the edit at the embedding, where every position is reflected through the axis and then held fixed;
the loss, read at the `=` position of each red line;
the gradient, which reaches the four blocks and the readout, and stops at the reflected states, so it cannot move the placement.
<!-- REVIEW: this line said the gradient reaches "the unembedding but not the embedding"; the two are one table in nGPT, so it now says what the detach cuts. The method's "Where the term acts" paragraph carries the full note. -->
"""


@memo
def fallback_term_figure() -> str:
    """The fallback-term schematic, as figure HTML. No data dependency, so it caches on its own source alone."""

    @themed(
        name="fallback-term",
        alt_text="""
            A schematic of the residual stream as a grid: four columns for the positions of a red line (red, +, blue, =), and rows from the embedding at the bottom through blocks 1 to 4 to the unembedding, which has a logits box at the = position only. The embedding row is outlined in orange and labelled as the edit, reflecting every position; an orange dashed line above it is labelled stop-gradient. The block and unembedding rows are shaded blue. A blue arrow points down into the logits box from a label reading loss, cross-entropy at = against the fallback answer, and a second blue arrow runs down the right margin from the unembedding to the dashed line, labelled gradient reaches the blocks and unembedding.
        """,
        caption="""
            **Where the fallback term acts.** One red line, `red + blue =`, as positions (columns) against slices (rows). The numbered annotations are the three places above; the blocks run forward from the edited embedding as usual, mixing positions through attention, and the dashed line is where the gradient stops.
        """,
    )
    def plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
        ink = light_dark("#333", "#ccc")
        faint = light_dark("#0002", "#fff2")
        edit = light_dark("#b5551d", "#ffab6e")
        grad = light_dark("#2a6fdb", "#8fbaff")
        cols = ["red", "+", "blue", "="]
        rows = ["embedding\n(slice 0)", "block 1", "block 2", "block 3", "block 4", "unembedding"]
        top = len(rows) - 1
        w, h = 0.72, 0.5
        right = len(cols) - 0.5
        # The rows the gradient reaches: the blocks and the unembedding.
        ax.add_patch(Rectangle((-0.5, 0.5), len(cols), top + 0.5, color=grad, alpha=0.08, lw=0, zorder=0))
        for j, name in enumerate(rows):
            for i, tok in enumerate(cols):
                if j == top and i != 3:
                    continue  # only the `=` position is decoded
                ax.add_patch(
                    FancyBboxPatch(
                        (i - w / 2, j - h / 2),
                        w,
                        h,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        fc=light_dark("#fff", "#1c1f1e"),
                        ec=edit if j == 0 else ink,
                        lw=1.2 if j == 0 else 0.8,
                        zorder=2,
                    )
                )
                label = tok if j == 0 else ("logits" if j == top else "")
                ax.text(i, j, label, ha="center", va="center", fontsize=8, color=ink, zorder=3)
            ax.text(-0.65, j, name, ha="right", va="center", fontsize=8, color=ink)
        # The stream between rows, per position.
        for i in range(len(cols)):
            end = top if i == 3 else top - 1
            ax.plot([i, i], [h / 2, end - h / 2], color=faint, lw=3, zorder=1, solid_capstyle="butt")
        # The stop-gradient, between the embedding row and block 1.
        ax.plot([-0.5, right], [0.5, 0.5], color=edit, lw=1.4, ls=(0, (4, 2)), zorder=4)
        ax.text(right + 0.15, 0.5, "stop-gradient", ha="left", va="center", fontsize=8, color=edit)
        ax.text(
            right + 0.15,
            0,
            "1. edit: reflect every position\n(α ↦ −α), then detach",
            ha="left",
            va="center",
            fontsize=8,
            color=edit,
        )
        ax.annotate(
            "2. loss: cross-entropy at =\nagainst the fallback answer",
            xy=(3, top + h / 2),
            xytext=(3, top + 1.05),
            ha="center",
            va="center",
            fontsize=8,
            color=grad,
            arrowprops=dict(arrowstyle="-|>", color=grad, lw=1),
        )
        # The gradient runs down the right margin and stops at the bar.
        ax.annotate(
            "",
            xy=(right + 0.08, 0.62),
            xytext=(right + 0.08, top),
            arrowprops=dict(arrowstyle="-|>", color=grad, lw=1.4),
        )
        ax.text(
            right + 0.15,
            (top + 0.6) / 2,
            "3. gradient reaches the\nblocks and unembedding",
            ha="left",
            va="center",
            fontsize=8,
            color=grad,
        )
        ax.set_xlim(-2.2, len(cols) + 1.9)
        ax.set_ylim(-0.6, top + 1.7)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig

    return plot()


fallback_term_figure()

r"""
In M1 the decoder took a constant input, the antipode itself, so the term was decoder-only by construction. Here a stop-gradient[^sg] gives the reflected state that same role.

[^sg]: An identity in the forward pass with a gradient of zero (`.detach()` in PyTorch), so a loss downstream of it cannot move anything upstream of it. The other losses see the clean pass and are unaffected.

The fallback answer for a continuous concept also has to be chosen. We use the center of the operand-averaged null, which works out to be the visible operand mixed with *mid-gray*; the [method](#the-fallback-answer) derives it.

This addresses the third risk in the plan, that the response to suppression is undesigned. One mismatch carries over from M1: the response is trained at the antipode, while the ex-2.2.1 projection lands the state at zero. H4 measures how far the response transfers between the two.

The nearest published analogue is LUNAR (arXiv:2502.07218): one matrix edit after training, redirecting the activations of the data to forget into the model's own refusal region. LUNAR designs the response by choosing a region the model already produces; ours is trained at a state the model never otherwise visits, which is where the mismatch above comes from. E6 fits a LUNAR-style edit to the no-fallback checkpoints, so the two designs can be compared.

**Natural language.** The three parts all transfer to the natural language domain:
the edit would be the reflection at the first anchored slice, applied at every position;
the loss would be computed on the continuation (the tokens of the response), rather than at one `=` position;
the gradient stops before the edit.
The edit happens based on alignment of the states to the anchor, so it's grammar-agnostic and token labels are not needed.

The fallback answer still has to be chosen; some candidates:
**(a)** a template continuation, so the training data is pairs of context and designed response;
**(b)** the model's own refusal region, which is the LUNAR choice (and which could be anchored separately);
**(c)** a true null, the continuation a reference model gives when it has never seen the concept, matched with a KL term.[^kl]

Spoofing in that setting is confabulation: a plausible answer drawn from the neighbors of the concept. The *mid-gray* answer here is a trained abstention that keeps the model on task (rather than, say, emitting a syntax token like `+` or `=` where the answer should be). A natural language version would want that same property.

[^kl]: A KL term penalizes the distance between the next-token distribution of the model and that of the reference.
"""

r"""
/// details | Glossary
- **line** — one equation, `c1 + c2 = answer`, six word-level tokens. The probe set enumerates all 5,832 closed lines of the 216-color grid: every color as op1 against each of its 27 closed partners.
- **residual stream** — the running vector each token carries through the network, which every block reads from and writes to.
- **slice** ($\ell$) — a depth at which the residual stream is read: the embedding (0), plus the stream after each of the four blocks.
- **alignment** ($\alpha$) — $\cos(h, e_1)$, the cosine between a state and the anchor axis. States are unit-norm, so this is just the e₁ component.
- **dose** — how *red* a line is, taken as the larger of the two operand rednesses ($r(1 - g/2 - b/2)$ on the unit cube).
- **red lines** — dose ≥ 0.8. 365 probe lines. **Non-red lines** — dose ≤ 0.2. 1,689 lines.
- **concept operand** — the redder of the two operands in a line (ties go to op1); the **visible operand** is the other one, and is *clean* when its redness is below 0.5.
- **margin** (`m_span`) — the pooled alignment margin of ex-2.1.10: at each slice, how much more aligned the red states are than the rest, taken at the best span role, then averaged over slices.
- **fallback answer** — the designed answer for a *red* line under intervention: the visible operand mixed with *mid-gray*, rounded to the grid. This is the center of the operand-averaged null.
- **fallback accuracy** — the fraction of *red* lines whose argmax at `=` is the fallback answer.
- **response** — the probability the model puts on the correct answer, read from the log-softmax at the `=` position. **Damage** is the clean response minus the intervened response, per line. **Deficit** is the clean exact-match accuracy minus the intervened accuracy over a group of lines; it is the statistic ex-2.2.1 gated selectivity on.
- **seed agreement** — the fraction of *red* lines on which at least five of the nine seeds decode the same answer, under a given intervention.
- **redirect** — reflecting the embedding state through the axis, α → −α, at every position. The edit at a position is twice its alignment, so a state with nothing on the axis does not move. This is the state the fallback was trained at.
- **clean** — the un-intervened forward pass of the same checkpoint.
///
"""

r"""
## Conditions

Two of the three conditions are ex-2.1.10 checkpoints already in the store, re-scored here through the same code path as the new runs. The third is trained here.

| condition | training | seeds | source |
| --- | --- | --- | --- |
| **un-anchored** | the recipe with the anchor weight at zero | 3 | ex-2.1.10 `lam0` |
| **no-fallback** | the D2.1 recipe: anchor + anti-subspace term | 9 | ex-2.1.10 `either-t100` |
| **fallback** | the recipe + anti-anchor term (0.1) + fallback term (w_fb = 0.05) | 9 | new |

Every hypothesis scores against the fallback condition. The no-fallback condition is the reference for it under every intervention.

The arms below have three seeds each. We report the same statistics for them, without gates.

| arm | what changes | question |
| --- | --- | --- |
| `fb-only` | no anti-anchor term | does the fallback need the antipode hemisphere cleared for it? |
| `anti-only` | no fallback term | does the anti-anchor term alone move the response? |
| `fb-w0.01`, `fb-w0.25` | the fallback weight, a factor of five either side | is 0.05 on a plateau? |
| `recipe` | both new weights at zero | does the new code path reproduce the ex-2.1.10 checkpoints? |

### The interventions

All of these run through the projection operator of the eval contract: it removes a fraction γ of the e₁ component, then puts the state back on the sphere. At γ = 2 the operator is a reflection, flipping the component instead of shrinking it.

- **`redirect`** — γ = 2 at the embedding, at every position, with no labels. This is the same edit training applied, so H1 reads the readout the term trained. Read by H1 and the first clause of H3.
- **`projection`** — the ex-2.2.1 intervention: γ = 1 at every slice and every position. The fallback was never trained under it. Read by the second clause of H3 and the projection row of H4.
- **The transfer sweep** — γ ∈ {0.5, 1, 1.5, 2} at the embedding, every position. Its γ = 2 point is `redirect`, and its γ = 1 point is the `embedding` arm of ex-2.2.1, which differs from `projection` in leaving the later slices alone. Read by H4.
- **Ride-along rows** — the `operands`, `shaped`, and `ablate` arms from ex-2.2.1, run on the fallback condition without gates. That way the intervention-tuning pass in the design has the fallback rows to hand (E5). `operands` is the one row here that needs position labels; it stays so we can compare with ex-2.2.1.

<!-- REVIEW: earlier drafts restricted `redirect` to operand positions, then to the concept operand of each line, to match a training term that reflected one state per line. Neither is an edit a model without labelled positions can run. Now `redirect` reflects every position at the embedding, training reflects the same, and the red lines with no defined fallback answer (a visible operand that is itself red) leave the term and the gate of H1 rather than the edit. Verify: the training paragraph in the method, and VISIBLE_RED_DOSE in experiment.py. -->
"""

r"""
## The designed response (H1)

<!-- REVIEW: 298 clean / 67 both → 296 / 69. The frozen count read `redness < 0.5` without roundoff slack, so one color on the 0.5 contour (cf96, whose redness evaluates to 0.49999999999999994) landed on the clean side while its five siblings at the same redness landed on the red side. The rule is unchanged ("≥ 0.5 is red"); applying it to all six moves two lines. Verify: `read_lines` in experiment.py asserts the counts. -->
**H1.** Under `redirect`, the model's answer on red lines is the fallback answer. Seed-mean fallback accuracy over the red lines with a clean visible operand (296 of the 365) is at least 0.8; partial: between 0.5 and 0.8. The other 69 red lines have no defined fallback answer, so we report them beside the gated figure, unscored. The reference row is the no-fallback condition under the same intervention. The fallback figure has to sit above that reference by a resolved margin, meaning two pooled between-seed standard deviations. A figure that clears 0.8 without a resolved margin counts as partial. Contrary: fallback accuracy at the no-fallback level, which would mean the term did not train the readout. The fallback loss over training (see the method) says whether the term was ever active.
"""

h1_fb = stat("redirect", "fb_acc", "red_clean")
h1_ref = stat("redirect", "fb_acc", "red_clean", ex.NO_FALLBACK.name)
h1_floor = floor("redirect", "fb_acc", "red_clean")
h1_rows = "\n".join(
    f"| {c.title} | {seed_row(stat('redirect', 'fb_acc', 'red_clean', c.name))} "
    f"| {seed_row(stat('redirect', 'acc', 'red_clean', c.name))} "
    f"| {seed_row(stat('redirect', 'fb_acc', 'red_both', c.name), '.2f')} "
    f"| {seed_row(stat('redirect', 'acc', 'red_both', c.name), '.2f')} |"
    for c in CONDS
)
h1_p_fb = {c.name: per_line("redirect", "p_fb", c.name)[:, lines.red_clean] for c in (ex.FALLBACK, ex.NO_FALLBACK)}


@memo
def fallback_strip_figure(p_fb: dict[str, np.ndarray]) -> str:
    """The fallback-probability strip chart, as figure HTML. Memoized on the arrays, so a re-render after a prose edit skips the draw."""

    @themed(
        name="fallback-strip",
        alt_text="""
            A strip chart of the probability on the fallback answer, one column per seed: nine fallback seeds on the left and nine no-fallback seeds on the right. Every fallback column is a dense band at the top of the panel, at one; every no-fallback column is a dense band at the bottom, at zero, with a scattering of dots rising up the panel, a few of them near one.
        """,
        caption="""
            **The probability on the fallback answer, per seed.** Under `redirect`, each dot is one of the 296 red lines with a clean visible operand; a column is one seed. Left, the fallback condition; right, the no-fallback condition. A split response, with some lines at the fallback answer and some elsewhere, would show as a column with dots at both ends.
        """,
    )
    def plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6.0, 2.4), layout="constrained")
        rng = np.random.default_rng(0)
        offset = 0
        for cond in (ex.FALLBACK, ex.NO_FALLBACK):
            p = p_fb[cond.name]
            for s in range(len(p)):
                x = offset + s + rng.uniform(-0.3, 0.3, p.shape[1])
                ax.scatter(x, p[s], s=2, color=light_dark(*INK[cond.name]), alpha=0.5, lw=0)
            offset += len(p) + 1
        ax.set_xticks([4, 14])
        ax.set_xticklabels([ex.FALLBACK.title, ex.NO_FALLBACK.title], fontsize=8)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("p(fallback answer)", fontsize=8)
        ax.tick_params(labelsize=7)
        return fig

    return plot()


rf"""
| condition | fallback acc., clean visible | true-answer acc., clean visible | fallback acc., red visible | true-answer acc., red visible |
|---|---|---|---|---|
{h1_rows}

Seed mean with the seed range, under `redirect`. The "clean visible" columns are the {lines.red_clean.sum()} lines H1 gates. The "red visible" columns are the other {lines.red_both.sum()} red lines, which have no defined fallback answer; the scorer reads it as the visible operand mixed with gray all the same, so that column says whether the model emits that anyway.

{fallback_strip_figure(h1_p_fb)}

Under `redirect`, the fallback condition emits the fallback answer on {h1_fb.mean():.3f} of the red lines with a clean visible operand, in every seed; the gate is {ex.FALLBACK_ACC_GATE:g}. Its margin over the no-fallback condition, which emits it on {h1_ref.mean():.3f}, clears the {h1_floor:.3f} resolution floor.

The probability the model puts on the fallback answer is {stat("redirect", "p_fb", "red_clean").mean():.3f}, averaged over seeds and lines, and the strip shows no split: every seed puts nearly every line at the top. The un-anchored condition keeps its true answer under the same edit ({stat("redirect", "acc", "red_clean", ex.CONTROL.name).mean():.3f}), so the reflection does its work only where a concept was placed. On the {lines.red_both.sum()} red lines whose visible operand is itself red, the fallback condition emits neither the true answer nor the gray mix of the visible operand.

**H1 holds.** The term trained the readout: the fallback loss over training (next section) falls to near zero, and the reflected state decodes to the designed answer on all but a handful of qualifying lines, in every seed.
<!-- REVIEW: the verdict said the loss "reaches zero" and the answer appeared "on every qualifying line"; the loss ends at 0.0003 and fallback accuracy is 0.999, so both were slightly stronger than the numbers. Verify: the fallback-loss figure in H2, and the fallback accuracy column of the table above. -->
"""

r"""
## Task and placement intact (H2)

**H2.** The term costs nothing on the task or the placement. Clean exact-match accuracy on all probe lines is within 0.02 of the no-fallback condition, and the seed-mean alignment margin at the end of training, the `m_span` of ex-2.1.10, is at least 0.8 of the no-fallback value. Partial: exactly one of the two holds, or both hold with accuracy read at the wider 0.05 band.

Contrary: the fallback term competing with the anchor for the operand states, which is what the stop-gradient is supposed to prevent.

<!-- REVIEW: H2 (then H3) gave two partial rules for the same clause ("partial: within 0.05" inline, and "one of the two holds" after), which do not decide a run whose accuracy sits between 0.02 and 0.05 with the margin intact. Merged into one rule; both bands are unchanged, and `TASK_PARTIAL` keeps its documented role. -->

<!-- REVIEW: the same merge applies to H3's first clause, which quoted only the 0.02 gate although `TASK_PARTIAL`'s docstring assigns it a 0.05 partial band there too. -->
"""

h2_acc_gap = clean("acc", "all", ex.NO_FALLBACK.name).mean() - clean("acc", "all").mean()
h2_ratio = clean("m_span").mean() / clean("m_span", cond=ex.NO_FALLBACK.name).mean()
h2_rows = "\n".join(
    f"| {c.title} | {seed_row(clean('acc', 'all', c.name))} | {seed_row(clean('acc', 'red', c.name))} "
    f"| {seed_row(clean('acc', 'nonred', c.name))} | {seed_row(clean('m_span', cond=c.name))} "
    f"| {seed_row(clean('alpha_mean_op1', cond=c.name))} |"
    for c in CONDS
)
h2_epoch = trajectory("epoch")[0]
h2_traj = {
    (key, cond): trajectory(key, cond)
    for key, conds in (
        ("m_span", (ex.NO_FALLBACK.name, ex.FALLBACK.name)),
        ("fallback", ("anti-only", ex.FALLBACK.name)),
        ("anti_anchor", ("fb-only", ex.FALLBACK.name)),
    )
    for cond in conds
}


@memo
def training_trajectories_figure(epoch: np.ndarray, traj_by_key: dict[tuple[str, str], np.ndarray]) -> str:
    """The training-trajectories figure, as figure HTML. Memoized on the arrays, so a re-render after a prose edit skips the draw."""

    @themed(
        name="training-trajectories",
        alt_text="""
            Three line charts over 100 epochs. Left, the alignment margin: nine red lines (fallback) and nine grey lines (no-fallback) all rise from zero to a plateau within the first ten epochs; the red plateau sits a little below the grey one, near 0.68 against 0.74. Middle, the fallback loss on a log scale: the red lines start near three at epoch ten, drop two orders of magnitude by epoch twenty, and keep falling to about a thousandth by the end, while the purple lines (anti-anchor only, where the term is measured and not trained) stay near ten throughout. Right, the anti-anchor loss on a log scale: the red lines fall from about 0.04 to below 0.001; the orange lines (fallback only) fall at first and then settle near 0.01.
        """,
        caption="""
            **Training, seed by seed.** Left, the alignment margin `m_span` over training, one thin line per seed: the fallback condition over the no-fallback condition. Middle, the fallback term's loss, in the fallback condition and in the `anti-only` arm, where it is measured on every step and never trained. Right, the anti-anchor hinge, in the fallback condition and in the `fb-only` arm, where it is measured and not trained. The losses are means over the crops between two trajectory records, and the fallback loss is a mean over the qualifying lines in those crops.
        """,
    )
    def plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.4), layout="constrained")
        panels = [
            ("m_span", "alignment margin (m_span)", [ex.NO_FALLBACK.name, ex.FALLBACK.name], False),
            ("fallback", "fallback loss", ["anti-only", ex.FALLBACK.name], True),
            ("anti_anchor", "anti-anchor loss", ["fb-only", ex.FALLBACK.name], True),
        ]
        titles = {c.name: c.title for c in CONDS}
        for ax, (key, label, conds, log) in zip(axes, panels, strict=True):
            for cond in conds:
                y = traj_by_key[key, cond]
                for s in range(len(y)):
                    ax.plot(
                        epoch[: y.shape[1]],
                        y[s],
                        lw=0.7,
                        alpha=0.8,
                        color=light_dark(*INK[cond]),
                        label=titles[cond] if s == 0 else None,
                    )
            if log:
                ax.set_yscale("log")
            ax.set_xlabel("epoch", fontsize=8)
            ax.set_title(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, frameon=False)
        return fig

    return plot()


rf"""
| condition | clean acc., all | red | non-red | margin `m_span` | mean α at op1 |
|---|---|---|---|---|---|
{h2_rows}

Seed mean with the seed range on the clean pass. The margin is `m_span` from ex-2.1.10: the pooled alignment contrast between the labelled and unlabelled operand states. The last column is the mean alignment of every op1 state, which is the containment statistic of ex-2.1.10.

{training_trajectories_figure(h2_epoch, h2_traj)}

Clean accuracy on all probe lines is {clean("acc", "all").mean():.3f} in the fallback condition against {clean("acc", "all", ex.NO_FALLBACK.name).mean():.3f} without the term, a gap of {h2_acc_gap:.4f} against a gate of {ex.TASK_GATE:g}. The margin at the end of training is {clean("m_span").mean():.3f} against {clean("m_span", cond=ex.NO_FALLBACK.name).mean():.3f}, a ratio of {h2_ratio:.3f} against a gate of {ex.MARGIN_RATIO:g}. Every fallback seed sits between {clean("m_span").min():.3f} and {clean("m_span").max():.3f}.

The margin is lost to the anti-anchor hinge: the `fb-only` arm ends at {clean("m_span", cond="fb-only").mean():.3f} and the `anti-only` arm at {clean("m_span", cond="anti-only").mean():.3f}. The hinge also raises the mean alignment at op1, from {clean("alpha_mean_op1", cond=ex.NO_FALLBACK.name).mean():.3f} to {clean("alpha_mean_op1").mean():.3f}.

The fallback loss switches on once the anchor has placed the concept, inside the warm-up, and reaches {trajectory("fallback")[:, -1].mean():.4f} by the end; in the `anti-only` arm, where it is measured and never trained, it stays at {trajectory("fallback", "anti-only")[:, -1].mean():.1f}. The anti-anchor hinge ends at {trajectory("anti_anchor")[:, -1].mean():.4f} where it is trained and {trajectory("anti_anchor", "fb-only")[:, -1].mean():.4f} where it is not.

**H2 holds.** Both clauses clear their gates: the term costs nothing on the task, and the margin stays above {ex.MARGIN_RATIO:g} of the no-fallback value.
"""

r"""
## Selectivity (H3)

<!-- REVIEW: the heading was "Selectivity kept", the hypothesis's short name; renamed after the results review, since the first clause failed by a wide margin and the verdict is partial. The Findings anchor follows the heading. -->

**H3.** Under `redirect`, the seed-mean non-red deficit stays at or below 0.02. Under `projection`, the non-red deficit in the fallback condition is not resolved above the no-fallback figure (0.024 in ex-2.2.1, re-measured here).

On a non-red line the operands have nearly nothing on the axis, so the first clause rests on the syntax embeddings. Zeroing their constant component is what cost the ex-2.2.1 projection its 0.024, and the reflection flips it instead. Training runs the reflected pass on every crop that carries a qualifying red line, so the blocks see flipped syntax states, and read them as syntax, throughout training. We predict that this transfers to the non-red lines the term never scored.

Partial: the first clause holds only at the 0.05 band, or exactly one clause holds.

Contrary on the first clause: a deficit at or above the ex-2.2.1 figure, which would say the flipped syntax reads as a different syntax, and would send the intervention-tuning pass toward a thresholded reflection, the `shaped` falloff at strength 2. Contrary on the second clause: the term widens the edit, which would mean the blocks now read the axis at positions or slices where they did not before.

<!-- REVIEW: H3 (then H4) was written on non-red *damage*, the probability statistic. The 0.02 width and the 0.024 reference both come from ex-2.2.1's H2, which gated the accuracy *deficit* (its H2 line: "outside the 0.02 gate but inside the 0.05 partial gate"), so the gate and its reference were being read on a statistic they were not set for. Now stated on the deficit, with the same widths, and the partial band written out. Verify: ex-2.2.1's H2 section, and the "non-red deficit" column of its arms table. Damage is still reported beside it in the table below. -->
"""

h3_d_redirect = stat("redirect", "deficit", "nonred")
h3_d_proj = stat("projection", "deficit", "nonred")
h3_d_proj_ref = stat("projection", "deficit", "nonred", ex.NO_FALLBACK.name)
h3_floor_proj = floor("projection", "deficit", "nonred")
h3_rows = "\n".join(
    f"| {c.title} | {seed_row(stat('redirect', 'deficit', 'nonred', c.name))} "
    f"| {seed_row(stat('redirect', 'damage', 'nonred', c.name))} "
    f"| {seed_row(stat('projection', 'deficit', 'nonred', c.name))} "
    f"| {seed_row(stat('projection', 'damage', 'nonred', c.name))} |"
    for c in CONDS
)
h3_alpha_clean = clean("alpha_q99_nonred").mean(0)
h3_alpha_arrive = stat("projection", "q99_alpha_nonred").mean(0)


@memo
def write_map_figure(alpha_clean: np.ndarray, alpha_arrive: np.ndarray) -> str:
    """The bound-and-write map, as figure HTML. Memoized on the arrays, so a re-render after a prose edit skips the draw."""

    @themed(
        name="write-map",
        alt_text="""
            Two line charts over the six token positions, one line per slice from light to dark. Left, the clean 99th-percentile non-red alignment in the fallback condition: between about 0.1 and 0.4 at the prompt positions, smaller at the answer and newline. Right, the same alignment arriving at the projection operator: the embedding line is identical, and the four deeper lines sit at or below their clean values at every position.
        """,
        caption="""
            **The bound and the write under `projection`, fallback condition.** The 99th-percentile alignment over the non-red lines at each (slice, position), seed mean, on the shared 0–1 scale. Left, the clean map, whose arcsine is the bound; right, the alignment arriving at the operator under `projection`, whose arcsine is the write. Slices run from the embedding (lightest) to the last block (darkest). A deeper line above its clean value would mean a block re-writes the axis after it was removed upstream, which is what a wider edit looks like.
        """,
    )
    def plot() -> plt.Figure:
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list(
            "depth", [light_dark("#f6b0b0", "#5a1a1a"), light_dark("#8a0000", "#ff7070")]
        )
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), layout="constrained", sharey=True)
        for ax, m, title in zip(
            axes, (alpha_clean, alpha_arrive), ("clean (bound)", "under projection (write)"), strict=True
        ):
            for k in range(m.shape[0]):
                ax.plot(
                    POS_NAMES, m[k], lw=1.1, color=cmap(k / (m.shape[0] - 1)), label=f"slice {k}", marker="o", ms=2.5
                )
            ax.set_ylim(0, 1)
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("99th-pct. |α|, non-red lines", fontsize=8)
        axes[1].legend(fontsize=6, frameon=False, ncol=2)
        return fig

    return plot()


rf"""
| condition | non-red deficit, `redirect` | non-red damage, `redirect` | non-red deficit, `projection` | non-red damage, `projection` |
|---|---|---|---|---|
{h3_rows}

Seed mean with the seed range. The deficit is the drop in exact-match accuracy on the {lines.nonred.sum()} non-red lines from the clean pass; damage is the drop in probability on the true answer.

**First clause.** Under `redirect`, the fallback condition loses {h3_d_redirect.mean():.3f} of its non-red accuracy, with no seed under {h3_d_redirect.min():.3f}; the gate is {ex.TASK_GATE:g} and the partial band {ex.TASK_PARTIAL:g}. The reflection is no more selective without the term: the no-fallback condition loses {stat("redirect", "deficit", "nonred", ex.NO_FALLBACK.name).mean():.3f} under it, the `recipe` arm {stat("redirect", "deficit", "nonred", "recipe").mean():.3f}, and the un-anchored condition {stat("redirect", "deficit", "nonred", ex.CONTROL.name).mean():.3f}.

So a full reflection of every embedding state is destructive on non-red lines in every anchored model, and the training-time exposure to flipped syntax states did not transfer to them. What those lines decode to is in the post-hoc row of the exploratory section.

**Second clause.** Under `projection`, the non-red deficit in the fallback condition is {h3_d_proj.mean():.3f} against {h3_d_proj_ref.mean():.3f} without the term, which is what ex-2.2.1 reported on the same checkpoints. The difference, {h3_d_proj.mean() - h3_d_proj_ref.mean():.3f}, sits under the resolution floor of {h3_floor_proj:.3f}, so by the frozen rule the deficit is not resolved above the reference.

The floor is wide because both conditions spread over seeds: the fallback seeds run from {h3_d_proj.min():.3f} to {h3_d_proj.max():.3f}, the reference seeds from {h3_d_proj_ref.min():.3f} to {h3_d_proj_ref.max():.3f}. Damage tells the same story ({stat("projection", "damage", "nonred").mean():.3f} against {stat("projection", "damage", "nonred", ex.NO_FALLBACK.name).mean():.3f}, floor {floor("projection", "damage", "nonred"):.3f}).

The weight bracket is the sharper reading: the deficit under `projection` rises with the weight, from {stat("projection", "deficit", "nonred", "fb-w0.01").mean():.3f} at 0.01 through {h3_d_proj.mean():.3f} at {ex.FALLBACK_WEIGHT:g} to {stat("projection", "deficit", "nonred", "fb-w0.25").mean():.3f} at 0.25.

{write_map_figure(h3_alpha_clean, h3_alpha_arrive)}

The write map shows no re-writing at any site: every deeper slice arrives at the operator at or below its clean alignment. Whatever the term changed under `projection`, it did not widen where the edit acts on the axis.

**H3 is partial**, by the frozen rule that exactly one clause holds. The first clause fails, and by more than the partial band allows. That is the contrary case named above: the flipped syntax does not read as syntax. The second clause holds, but as an unresolved difference, and the weight bracket suggests the difference is real and grows with the weight.

<!-- REVIEW: the verdict follows the frozen rule ("exactly one clause holds"). The second clause holds only in the sense the rule defines, not resolved at nine seeds; the weight bracket, which the rule does not consult, shows the deficit under projection growing with the weight. Verify: the `projection` columns of the table, and the E4 row. -->
"""

r"""
## Transfer from the antipode to zero (H4)

**H4.** The designed response transfers (generalizes) part of the way to zero. Along the transfer sweep, γ ∈ {0.5, 1, 1.5, 2} at the embedding, seed-mean fallback accuracy on red lines is non-decreasing in γ, allowing a dip of at most 0.02 between adjacent strengths, and at γ = 1 it is at least half its value at γ = 2. Partial: either clause holds on its own — monotone with γ = 1 below half, or γ = 1 at half or more with a dip larger than the allowance.

The γ = 1 point is a real removal, not an inert midpoint. It is the `embedding` arm of ex-2.2.1, where zeroing the axis at the embedding alone took red accuracy down to 0.16. The sweep adds how the fallback responds along the way from the trained state to that one. At γ = 1 the operator keeps whatever the state carries off the axis and rescales it by $1/\sqrt{1-\alpha^2}$, so the landing direction is defined as long as the state is not perfectly aligned, and no state is: pure red arrives at the embedding at about α = 0.9, a gain of about 2.3.

Contrary: fallback accuracy at γ = 1 sits at the no-fallback level and the rise is confined to γ > 1. That would be the mismatch showing in full, with the response living at the antipode and not reaching the projected state. The [concept swap](/todo/science/redirect-between-two-anchored-ops.md) filed for D2.3 would address it by targeting a state training already visits; widening the bracket here would not.

<!-- REVIEW: H4's (then H5's) partial band covered only the monotone-but-short case, leaving no verdict for a run that transfers to γ = 1 through a dip; both single-clause cases are now partial. The D2.3 sentence was in the present indicative ("the remedy is"), which reads as a scheduled follow-up; softened to the conditional, since that item is a backlog entry. -->

We report the `projection` row beside the sweep, without a gate: γ = 1 at every slice rather than at the embedding alone. The difference between the two says how much the projection at later slices costs the designed response.
"""

h4_fb = {
    c.name: np.array([stat(f"gamma-{g}", "fb_acc", "red_clean", c.name) for g in ex.GAMMAS])
    for c in (ex.FALLBACK, ex.NO_FALLBACK)
}
h4_acc = {
    c.name: np.array([stat(f"gamma-{g}", "acc", "red", c.name) for g in ex.GAMMAS])
    for c in (ex.FALLBACK, ex.NO_FALLBACK)
}
h4_mean = h4_fb[ex.FALLBACK.name].mean(1)
h4_dip = float(np.max(-np.diff(h4_mean)))
h4_half = h4_mean[1] / h4_mean[3]
h4_proj = stat("projection", "fb_acc", "red_clean")
h4_g1 = h4_fb[ex.FALLBACK.name][1]


@memo
def transfer_sweep_figure(
    fb: dict[str, np.ndarray], acc: dict[str, np.ndarray], proj_fb: np.ndarray, proj_acc: np.ndarray
) -> str:
    """The transfer-sweep figure, as figure HTML. Memoized on the arrays, so a re-render after a prose edit skips the draw."""

    @themed(
        name="transfer-sweep",
        alt_text="""
            Two line charts against the projection strength, from 0.5 to 2. Left, fallback accuracy: the red line (fallback condition) sits at zero at 0.5, rises to about 0.63 at 1 with a wide band from 0.23 to 0.97, and reaches one at 1.5 and 2; a red cross at strength 1 marks the projection row at 0.57, with a grey cross beside it near zero. The grey line (no-fallback) stays near zero throughout. Right, true-answer accuracy on red lines: both lines fall from about 0.7 or 0.8 at 0.5 to near zero at 1 and beyond.
        """,
        caption="""
            **The transfer sweep.** Seed mean with the seed range as a band, for the projection operator at the embedding at strength γ, every position. Left, fallback accuracy on the red lines with a clean visible operand; right, true-answer accuracy on all red lines. γ = 2 is `redirect`; γ = 1 is the `embedding` arm of ex-2.2.1. The crosses at γ = 1 are the `projection` row of each condition, which removes the axis at every slice rather than at the embedding alone.
        """,
    )
    def plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4), layout="constrained", sharey=True)
        x = np.array(ex.GAMMAS)
        for ax, series, title in zip(
            axes, (fb, acc), ("fallback accuracy, clean visible", "true-answer accuracy, red lines"), strict=True
        ):
            for cond in (ex.NO_FALLBACK, ex.FALLBACK):
                y = series[cond.name]
                ink = light_dark(*INK[cond.name])
                ax.fill_between(x, y.min(1), y.max(1), color=ink, alpha=0.15, lw=0)
                ax.plot(x, y.mean(1), color=ink, lw=1.3, marker="o", ms=3, label=cond.title)
            ax.set_xticks(x)
            ax.set_xlabel("γ at the embedding", fontsize=8)
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_ylim(-0.03, 1.03)
        axes[0].scatter([1.0], [proj_fb[0]], marker="x", s=30, color=light_dark(*INK[ex.NO_FALLBACK.name]), zorder=4)
        axes[0].scatter([1.0], [proj_fb[1]], marker="x", s=30, color=light_dark(*INK[ex.FALLBACK.name]), zorder=4)
        axes[1].scatter([1.0], [proj_acc[0]], marker="x", s=30, color=light_dark(*INK[ex.NO_FALLBACK.name]), zorder=4)
        axes[1].scatter([1.0], [proj_acc[1]], marker="x", s=30, color=light_dark(*INK[ex.FALLBACK.name]), zorder=4)
        axes[0].legend(fontsize=6, frameon=False, loc="center left")
        return fig

    return plot()


H4_FIGURE = transfer_sweep_figure(
    h4_fb,
    h4_acc,
    np.array(
        [
            stat("projection", "fb_acc", "red_clean", ex.NO_FALLBACK.name).mean(),
            stat("projection", "fb_acc", "red_clean", ex.FALLBACK.name).mean(),
        ]
    ),
    np.array(
        [
            stat("projection", "acc", "red", ex.NO_FALLBACK.name).mean(),
            stat("projection", "acc", "red", ex.FALLBACK.name).mean(),
        ]
    ),
)

rf"""
{H4_FIGURE}

Along the sweep, the seed-mean fallback accuracy in the fallback condition runs {", ".join(f"{v:.3f}" for v in h4_mean)} at γ = {", ".join(f"{g:g}" for g in ex.GAMMAS)}. It is non-decreasing: the largest drop between adjacent strengths is {max(h4_dip, 0):.3f}, against an allowance of {ex.GRADE_DIP:g}. At γ = 1 it stands at {h4_half:.2f} of its value at γ = 2, against a gate of {ex.TRANSFER_FRAC:g}.

The seed range at γ = 1 is wide, from {h4_g1.min():.2f} to {h4_g1.max():.2f}, and at γ = 1.5 every seed is above {h4_fb[ex.FALLBACK.name][2].min():.2f}. The no-fallback condition stays at {h4_fb[ex.NO_FALLBACK.name].mean(1).max():.3f} or below throughout. True-answer accuracy on red lines falls the same way in both conditions: the removal at γ = 1 is at least as complete with the term as without it ({h4_acc[ex.FALLBACK.name][1].mean():.3f} against {h4_acc[ex.NO_FALLBACK.name][1].mean():.3f}).

The `projection` row, γ = 1 at every slice, gives {h4_proj.mean():.3f} ({h4_proj.min():.2f} to {h4_proj.max():.2f}), a little under the embedding-only figure at the same strength. Removing the axis at the later slices as well costs the designed response {h4_g1.mean() - h4_proj.mean():.3f} on the seed mean.

**H4 holds.** The response reaches the projected state at more than half strength, and rises monotonically from there to the trained one. The contrary case, a rise confined to γ > 1, did not occur.
"""

e_cats = ex.COMPOSITION
e_comp = {
    (c.name, iv): np.array([r["interventions"][iv]["composition_red_clean"] for r in runs[c.name]], float)
    for c in (ex.FALLBACK, ex.NO_FALLBACK)
    for iv in ("redirect", "projection")
}
e_agree = metrics["agreement"]


@memo
def composition_figure(comp: dict[tuple[str, str], np.ndarray]) -> str:
    """The red-line composition figure, as figure HTML. Memoized on the arrays, so a re-render after a prose edit skips the draw."""

    @themed(
        name="composition",
        alt_text="""
            Two panels of stacked bars, one bar per seed, nine fallback seeds then nine no-fallback seeds. Left, under redirect: every fallback bar is entirely the fallback-answer color; the no-fallback bars are two thirds or more the shade for something else, with small neighbor, visible-operand, and true-answer segments. Right, under projection: the fallback bars are between a third and four fifths fallback answer, varying by seed, with most of the rest a neighbor; the no-fallback bars are mostly neighbor, with true-answer, visible-operand, and other segments.
        """,
        caption="""
            **What red lines decode to.** Each bar splits the 296 red lines with a clean visible operand by the decoded answer, per seed: the fallback answer, the true mix, a one-step neighbor of it, the visible operand, the red operand, or something else, tested in that order. Left, under `redirect`; right, under `projection`. The nine fallback seeds sit left of the nine no-fallback seeds in each panel.
        """,
    )
    def plot() -> plt.Figure:
        shades = [
            light_dark(*INK["lunar"]),
            light_dark("#222", "#eee"),
            light_dark("#888", "#888"),
            light_dark(*INK["fb-only"]),
            light_dark(*INK[ex.FALLBACK.name]),
            light_dark("#ddd", "#333"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), layout="constrained", sharey=True)
        for ax, iv in zip(axes, ("redirect", "projection"), strict=True):
            x0 = 0
            for cond in (ex.FALLBACK, ex.NO_FALLBACK):
                frac = comp[cond.name, iv] / comp[cond.name, iv].sum(1, keepdims=True)
                xs = x0 + np.arange(len(frac))
                bottom = np.zeros(len(frac))
                for k, cat in enumerate(e_cats):
                    ax.bar(
                        xs,
                        frac[:, k],
                        bottom=bottom,
                        color=shades[k],
                        width=0.8,
                        lw=0,
                        label=cat if (cond is ex.FALLBACK and iv == "redirect") else None,
                    )
                    bottom += frac[:, k]
                x0 += len(frac) + 1
            ax.set_xticks([4, 14])
            ax.set_xticklabels([ex.FALLBACK.title, ex.NO_FALLBACK.title], fontsize=8)
            ax.set_title(f"under `{iv}`", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("share of red lines", fontsize=8)
        fig.legend(fontsize=6, frameon=False, ncol=len(e_cats), loc="outside lower center")
        return fig

    return plot()


def e_row(c: ex.Condition, iv: str, g: str = "red_clean") -> str:
    a = e_agree[c.key][f"{iv}/{g}"]
    return f"{'–' if a['agree'] is None else format(a['agree'], '.3f')} | {a['plurality']:.3f}"


e_agree_rows = "\n".join(f"| {c.title} | {e_row(c, 'redirect')} | {e_row(c, 'projection')} |" for c in CONDS)
e_neg_rows = "\n".join(
    f"| {c.title} | "
    + " | ".join(f"{v:.3f}" for v in np.array(clean("negative_frac", cond=c.name)).mean((0, 2)))
    + f" | {np.array(clean('negative_frac', cond=c.name)).mean():.3f} |"
    for c in CONDS
)


def e_r2(iv: str, cond: str) -> np.ndarray:
    return clean("offaxis_r2", cond=cond).mean(0) if iv == "clean" else stat(iv, "offaxis_r2", cond=cond).mean(0)


def e_r2_floor(iv: str) -> np.ndarray:
    vs = [
        clean("offaxis_r2", cond=c.name) if iv == "clean" else stat(iv, "offaxis_r2", cond=c.name)
        for c in (ex.FALLBACK, ex.NO_FALLBACK)
    ]
    return ex.RESOLUTION_SD * np.sqrt(sum(v.var(0, ddof=1) for v in vs) / 2)


e_r2_rows = "\n".join(
    f"| {iv} | "
    + " | ".join(
        f"{a:.3f} / {b:.3f}" for a, b in zip(e_r2(iv, ex.FALLBACK.name), e_r2(iv, ex.NO_FALLBACK.name), strict=True)
    )
    + " | "
    + " | ".join(f"{f:.3f}" for f in e_r2_floor(iv))
    + " |"
    for iv in ("clean", "redirect", "projection")
)
e_ride_rows = "\n".join(
    f"| `{iv}` | {c.title} | {seed_row(stat(iv, 'acc', 'red', c.name))} | {seed_row(stat(iv, 'fb_acc', 'red_clean', c.name))} | {seed_row(stat(iv, 'deficit', 'nonred', c.name))} | {seed_row(stat(iv, 'damage', 'nonred', c.name))} |"
    for iv in ("operands", "shaped", "ablate")
    for c in (ex.FALLBACK, ex.NO_FALLBACK)
)
e_lunar = {
    k: stat("lunar", k, g, ex.NO_FALLBACK.name)
    for k, g in (("fb_acc", "red_clean"), ("acc", "red"), ("deficit", "nonred"), ("damage", "nonred"))
}
e_offvocab = stat("redirect", "offvocab", "red")
e_offvocab_ref = stat("redirect", "offvocab", "red", ex.NO_FALLBACK.name)

# Post hoc: what the non-red lines decode to under the reflection.
e_nr = lines.nonred
e_fbtok, e_ans, e_vis = lines.fb_token[e_nr], lines.answer[e_nr], lines.visible[e_nr]


def e_nonred_split(iv: str, cond: str) -> list[float]:
    g = per_line(iv, "guess", cond)[:, e_nr]
    is_true = g == e_ans
    is_fb = (g == e_fbtok) & ~is_true
    is_vis = (g == e_vis) & ~is_true & ~is_fb
    return [float(m.mean()) for m in (is_true, is_fb, is_vis, ~(is_true | is_fb | is_vis))]


e_nonred_rows = "\n".join(
    f"| {c.title} | "
    + " | ".join(f"{v:.3f}" for v in e_nonred_split("redirect", c.name))
    + " | "
    + " | ".join(f"{v:.3f}" for v in e_nonred_split("projection", c.name))
    + " |"
    for c in CONDS
)

e_syntax_rows = "\n".join(
    f"| {c.title} | " + " | ".join(f"{v:.2f}" for v in clean("alpha_q99_nonred", cond=c.name).mean(0)[0]) + " |"
    for c in (ex.CONTROL, ex.NO_FALLBACK, ex.FALLBACK)
)

rf"""
## Exploratory analyses

Preregistered as exploratory, no gates, except the last two rows, which are post hoc.

**E1 — composition.** {composition_figure(e_comp)}

Under `redirect`, the fallback condition decodes {e_comp[ex.FALLBACK.name, "redirect"][:, 0].sum() / e_comp[ex.FALLBACK.name, "redirect"].sum():.3f} of red lines to the fallback answer, and puts {e_offvocab.mean():.3f} of its mass outside the color vocabulary. The no-fallback condition puts {e_offvocab_ref.mean():.3f} outside it under the same edit, with a seed range from {e_offvocab_ref.min():.2f} to {e_offvocab_ref.max():.2f}. Under `projection` the fallback condition splits between the fallback answer and a one-step neighbor of the true mix, in a proportion that varies by seed.

| condition | agree, `redirect` | plurality, `redirect` | agree, `projection` | plurality, `projection` |
|---|---|---|---|---|
{e_agree_rows}

Seed agreement on the red lines with a clean visible operand. "Agree" is the fraction of lines on which at least {ex.AGREE_SEEDS} of the nine seeds decode the same answer, the statistic from ex-2.2.1, which put it at 13% under the projection; it is undefined for the three-seed conditions. "Plurality" is the mean over lines of the fraction of seeds decoding the line's plurality answer, defined for every condition.

**E2 — the antipode.** The fraction of clean states with negative alignment, per slice and averaged over positions:

| condition | emb. | block 1 | block 2 | block 3 | block 4 | all |
|---|---|---|---|---|---|---|
{e_neg_rows}

The anti-subspace term alone leaves about a quarter of clean states past the antipode plane; the hinge takes that to about {clean("negative_frac").mean():.2f}, and the `fb-only` arm, without the hinge, sits where the no-fallback condition does.

**E3 — off-axis recoverability.** Held-out R² of a ridge probe for the concept operand's redness, fitted on the operand states with the axis deleted, per slice. (A ridge probe is a linear regression with a penalty on the weights; held-out R² is the share of variance it explains on lines it was not fitted to.) Fallback / no-fallback, with the resolution floor (two pooled between-seed sds) beside each:

| states | emb. | block 1 | block 2 | block 3 | block 4 | floor: emb. | b1 | b2 | b3 | b4 |
|---|---|---|---|---|---|---|---|---|---|---|
{e_r2_rows}

The floor the task itself sets, the same ridge fit to the raw RGB values, is {metrics["rgb_floor"]:.3f}. Every clean figure is above it; under the interventions the deeper slices dip below it, as far as {min(min(e_r2(iv, c.name)) for iv in ("redirect", "projection") for c in (ex.FALLBACK, ex.NO_FALLBACK)):.3f} at the last block. Under `redirect` and under `projection`, the R² in the fallback condition sits above the no-fallback one at the deeper slices, by 0.03 to 0.05; that is under the floor at every slice. The [follow-up at more seeds](/todo/science/off-axis-probe-r2-does-bound-intervention.md) is where that would be resolved.

**E4 — arms.** The tables under H1, H2, and H3 carry every arm. Dropping the hinge (`fb-only`) leaves the fallback under `redirect` intact and the margin at the no-fallback level, and takes the transfer to γ = 1 from {stat("gamma-1.0", "fb_acc", "red_clean").mean():.3f} to {stat("gamma-1.0", "fb_acc", "red_clean", "fb-only").mean():.3f}. Dropping the fallback term (`anti-only`) leaves no designed response.

Along the weight bracket, transfer to γ = 1 rises with the weight ({stat("gamma-1.0", "fb_acc", "red_clean", "fb-w0.01").mean():.3f}, {stat("gamma-1.0", "fb_acc", "red_clean").mean():.3f}, {stat("gamma-1.0", "fb_acc", "red_clean", "fb-w0.25").mean():.3f}) and so does the non-red deficit under `projection` (see the table under H3).

The `recipe` arm reproduces the stored no-fallback checkpoints through the new code path: margin {clean("m_span", cond="recipe").mean():.3f} against {clean("m_span", cond=ex.NO_FALLBACK.name).mean():.3f}, red accuracy under `projection` {stat("projection", "acc", "red", "recipe").mean():.3f} against {stat("projection", "acc", "red", ex.NO_FALLBACK.name).mean():.3f}, non-red deficit under `projection` {stat("projection", "deficit", "nonred", "recipe").mean():.3f} against {stat("projection", "deficit", "nonred", ex.NO_FALLBACK.name).mean():.3f}. The recomputed clean alignment maps of the stored runs match the published ones from ex-2.1.10 to {max(metrics["alpha_max_diff"].values()):.4f}.

**E5 — ride-along interventions.**

| intervention | condition | red acc. | fallback acc., clean visible | non-red deficit | non-red damage |
|---|---|---|---|---|---|
{e_ride_rows}

Seed mean with the seed range, for the three ex-2.2.1 edits run without gates on both nine-seed conditions. Red accuracy and fallback accuracy are read on red lines, the deficit and damage on non-red lines.

The `operands` edit, at the operand positions only, keeps its selectivity in the fallback condition and produces the fallback answer on {stat("operands", "fb_acc", "red_clean").mean():.2f} of lines. `ablate` is the one row where the fallback condition differs from the reference by a large amount: it emits the fallback answer on about half the red lines and loses {stat("ablate", "deficit", "nonred").mean():.3f} of its non-red accuracy, against {stat("ablate", "deficit", "nonred", ex.NO_FALLBACK.name).mean():.3f} without the term.

**E6 — a LUNAR-style redirect, fitted after training.** On each frozen no-fallback checkpoint we fit one 64×64 matrix at the embedding, applied at every position, on the qualifying red lines the fallback term uses. The loss has two parts: the fallback cross-entropy at `=`, and an identity term on every other state, meaning the mean squared distance from the unedited state.

This model has no refusal region of its own, so the fallback answer stands in for it. The fitted matrix runs through the eval contract like any other operator, giving fallback accuracy {seed_row(e_lunar["fb_acc"])} on the red lines with a clean visible operand, red-line accuracy {seed_row(e_lunar["acc"])}, a non-red deficit of {seed_row(e_lunar["deficit"])}, and non-red damage of {seed_row(e_lunar["damage"])}.

Beside the fallback condition under `redirect` ({stat("redirect", "fb_acc", "red_clean").mean():.3f} and a deficit of {stat("redirect", "deficit", "nonred").mean():.3f}), the fitted edit reaches the same designed response at a smaller cost on non-red lines, with a wide seed spread. Its identity term was the only thing asking it to leave other positions alone, and at the weight used it did not do so.

**E7 — non-red lines under the reflection (post hoc).** What the non-red lines decode to, as a fraction of (seed, line) pairs: the true answer, the fallback answer of the visible operand, the visible operand itself, or something else.

| condition | true, `redirect` | fallback | visible | other | true, `projection` | fallback | visible | other |
|---|---|---|---|---|---|---|---|---|
{e_nonred_rows}

Under `redirect`, the fallback condition emits the gray mix of the visible operand on {e_nonred_split("redirect", ex.FALLBACK.name)[1]:.2f} of non-red (seed, line) pairs, so about a third of its loss on those lines is the designed response firing where no concept was removed. The rest is undesigned, as is the whole of the loss in the no-fallback condition. Under `projection` the fallback fires on {e_nonred_split("projection", ex.FALLBACK.name)[1]:.3f} of them, which accounts for about a third of the unresolved difference in the second clause of H3.
<!-- REVIEW: this line said the projection figure "is the size of" the unresolved difference in H3's second clause; the figure is 0.028 and the difference 0.090, so it accounts for part of it rather than all. Verify: the `projection` deficit columns of the H3 table against the fallback column here. -->

<!-- REVIEW: E7 is post hoc; it was added after the H3 result to say what the non-red lines decode to, using the per-line guesses the scorer already stores. The fallback answer for a non-red line is the scorer's own table (visible operand mixed with gray), with the true-answer and visible-operand coincidences removed first. Verify: the composition order in experiment.py, and the per-line `guess` arrays. -->

**E8 — the syntax embeddings on the axis (post hoc).** 99th-percentile clean alignment over the non-red lines at the embedding slice, one figure per position, averaged over seeds. Operand and answer columns pool all the color tokens, so their percentile is the tail of about two hundred tokens; a syntax column has only one token, so its percentile is that token's alignment.

| condition | {" | ".join(f"`{p}`" for p in POS_NAMES)} |
|---|---|---|---|---|---|---|
{e_syntax_rows}

In the un-anchored model the `+` and `=` embeddings sit where a random direction would. In every anchored model they carry 0.3 to 0.4. Meanwhile the anti-subspace term has flattened the non-red colors, and the hinge flattens them further. So the leak sits in two token embeddings at slice 0, before any block runs.

<!-- REVIEW: E8 is post hoc; it was added after the discussion round to say where the non-red cost of H3's first clause comes from, using the stored clean alignment maps (the same data as the write map's left panel, for three conditions instead of one). Verify: `alpha_q99_nonred` in the clean statistics. -->
"""

r"""
## Discussion

The fallback in the transformer does what it did in M1. At the edit it was trained at, every seed gives the designed answer, and the seed disagreement ex-2.2.1 found under projection is gone there.

What is new is the gap between the edit the term was trained at and the edit we want to use. The clean model never visits the antipode, and the response trained there carries over to the plain projection in most seeds, but in some barely at all. That is the mismatch the design named, and the anchored-op experiments should train toward a state that training already visits; the [concept swap](/todo/science/redirect-between-two-anchored-ops.md) filed for D2.3 gives one by construction.

The anti-anchor hinge pays for that transfer with margin: it keeps the clean states on the anchor side of the plane. That cost comes from designing at the antipode, and the swap has no antipode, so we are not adopting the hinge into the recipe. Nor does the fallback term itself carry into the operator experiments by default. We are keeping the code, for fits like E6 and for a concept with no visited state to aim at.

The selectivity cost belongs to the edit rather than to the term. Every anchored model loses most non-red lines under the reflection, because the `+` and `=` embeddings carry the axis (E8) and the reflection flips them along with the operand. We read the tied readout as what puts the axis there: the state after a red operand sits on the axis, and the cheapest way to predict the token that follows is for that token's readout to lean the same way. An [untied readout would test this](/todo/science/syntax-embeddings-carry-the-axis-via-tied-readout.md).

Until those embeddings are clean, any edit applied at every position pays the cost, including a rotation to a second anchored concept. The thresholded and operand-only edits of ex-2.2.1 avoid it.

On masking, the off-axis row cannot say whether the term keeps red readable elsewhere. The shift it shows is inside the floor, and the follow-up at more seeds is filed.

The fitted edit of E6, which reaches the designed response at a smaller non-red cost, is the LUNAR argument in miniature: if the model already produces the destination, there is no need to train at a state it never visits. The trained term still has one thing going for it, the response living in the model rather than in a fitted map. The [fit of E6 could be tightened](/todo/science/lunar-fit-retain-term-and-weight-sweep.md) before the two are compared again.
"""

r"""
## Method

### Training

We use the primary recipe from [ex-2.1.10](../ex-2.1.10/report.py) unchanged: the `v216` corpus, d64-L4, 100 epochs with a 10-epoch warm-up, the pooled either-operand labeller at τ = 0.1, the anchor at λ = 0.1 with its anneal, and the anti-subspace term at the ex-2.1.8 operating point. The [ex-2.1.11](../ex-2.1.11/report.py) survey proposed a different point. But survey numbers are proposals until they are re-measured at fresh seeds, and the design schedules that confirmation on the multi-op grammar. Keeping the ex-2.1.10 recipe also keeps its stored checkpoints as the no-fallback condition, so nothing has to be retrained for the comparison.

We also use the same seeds as the primary of that experiment, so the fallback and no-fallback conditions differ only in the two new terms. Both are weights on the existing anchored train step.

**The fallback term.** On each training crop, we reflect the embedding state of every position through the axis, $h \mapsto \mathrm{normalize}(h - 2\alpha\, e_1)$, and hold the reflected states fixed with a stop-gradient. We then run the blocks forward from there and take the cross-entropy at the `=` position of each qualifying line against its fallback token. A line qualifies on three counts: its dose is at least 0.8, the redness of its visible operand is below 0.5, and the clean embedding alignment of its concept operand is at least 0.5. The reflection singles out no positions, so it is the same edit `redirect` applies at eval, and it applies unchanged to a model whose positions are not labelled.

**Compute.** The reflected pass is a second forward and backward through the blocks, so a step that carries a qualifying line costs about twice a plain step. A crop holds about ten lines, and about one line in twenty qualifies, so roughly 40% of crops carry one. Run per batch, that is nearly every step, and the budget below counts the term as a doubling. Three things should make it cheap at scale. The pass can be skipped when no position clears the alignment threshold, which the clean pass at the edit slice already reports, so it costs nothing before the anchor has placed the concept. It can run on a subsample of the qualifying sequences, or on every k-th step, since the term is a small weight on a slowly moving target. And it reruns only the part of the model after the edit, so an edit deeper than the embedding shares the earlier layers with the clean pass.

The term is the mean over qualifying lines, at a constant weight of 0.05 from step 0. We take that weight from M1, as a starting point rather than a derived value: there it scaled a mean squared error on one decoder, and here it scales a cross-entropy through four blocks. Checking it is what the bracket arms are for.

The alignment threshold keeps the term inert until the anchor has placed the concept, which the ex-2.1.10 trajectories put inside the warm-up. If the concept never arrived at the edit slice, the term would stay inert rather than train toward the wrong state, and the margin gate of H2 would report the anchoring failure.

<!-- REVIEW: the frozen text said the embedding table gets no gradient from the term. nGPT ties the unembedding to the embedding table, so the table does see the term through the readout. Corrected to say what the detach does and does not cut; the design (reflect once, detach, train what follows) is unchanged. Verify: `reflected_logits` in src/sca/fallback.py multiplies by `wte.T`. -->
**Where the term acts.** The stop-gradient makes this similar to the decoder-only term from M1: the reflected embedding states are detached, so nothing flows back through them to the placement of *red* at the edit, while the four blocks and the readout do learn. The readout is the embedding table itself, since nGPT ties the two, so the table sees the term through the readout only: the answer tokens' readout vectors move toward the reflected pass's final state, and every other token's readout, *red*'s included, moves a little away from it. That is a decoder-side nudge on the readout table rather than a pull on where the concept sits, and the margin gate of H2 is where it would show. Together the blocks and the readout learn a map from a state at the antipode to the fallback answer. In ex-2.2.1 the concept was read from the operand states in the first two blocks, so those are the blocks with something to learn.

We reflect once, at the first anchored slice, which in this recipe is the embedding. Reflecting at every slice, as the ex-2.2.1 projection does, would need a detach at every slice, and then only the layers after the last detach would train; reflecting everywhere and detaching only the first edit is the rehearsal the design rejected.

The stop-gradient does leave one thing open. The blocks after the edit are shared with the clean pass, so the term can reshape how the concept is carried downstream of the edit, including keeping it readable off the axis. That is what the off-axis row (E3) and the margin gate watch.

**The anti-anchor term.** This is $\mathrm{mean}(\max(-\alpha, 0))$ over the same live positions and slices the anti-subspace term reads, on the clean forward pass only, at a constant weight of 0.1. The reflected states that the fallback term produces are not live positions of that pass, so the two terms do not pull against each other. The term is a hinge on negative alignment: zero when the alignment is positive, and growing linearly as it goes negative. So it keeps clean states out of the half of the space where the fallback lives, and because it acts on one side only, it does not oppose the anchor. The anti-subspace term already penalizes $\alpha^2$ there, so the arms that drop one term or the other say whether the hinge adds anything.

### The fallback answer

Removing the concept operand leaves the visible operand and an unknown partner. The least committal answer is the distribution of the mix over every partner the corpus allows, which we call the *operand-averaged null*. On this grid, each channel of the visible operand has three closed partners, and their three mixes are distinct, so the null is uniform over 27 colors and has no mode.

Its per-channel median is the mix with the middle partner. That coincides, at every level, with the visible operand mixed with mid-gray (7.5 on the 16-level channel) and rounded to the nearest grid level; `experiment.py` asserts the coincidence. That color is the fallback answer, and it is the gray fallback from M1 carried over.

The fallback answer depends on the visible operand alone, so it is defined only when that operand is clean. On a red line whose visible operand is itself red (redness ≥ 0.5), both operand states are reflected. The answer would then be the mix with both operands replaced by mid-gray, which is 7.5 on every channel, halfway between grid levels 6 and 9. With nothing visible, the null has no center on the grid; when one operand is visible, its level always breaks the tie. Those 69 lines take no fallback loss and sit outside the gate of H1.

### Measurements

We run one teacher-forced pass per run per intervention, over all 5,832 lines, through the eval contract in [`sca.intervention`](/src/sca/intervention.py). Each pass gives the log-softmax at the `=` position, from which we read the probability on the correct answer and on the fallback answer, the argmax, and the mass outside the color vocabulary.

Each pass also gives the write per (slice, line, position), as in ex-2.2.1, and the clean alignment map. Fallback accuracy, seed agreement, and the composition all come from the argmax on red lines. The training trajectory records the fallback and anti-anchor losses beside the anchor loss and the margin, every 50 steps.

**Noise floor.** As in ex-2.2.1: for each group statistic we take the pooled between-seed standard deviation per condition, computed before any comparison is read. A difference between conditions or arms smaller than twice that value is reported as not resolved. Gates score seed means against fixed thresholds and do not use the floor.

### Budget

Twenty-four training runs of the ex-2.1.10 recipe, each at about twice the cost of a plain run (see [Compute](#training)), plus the scoring of twelve stored checkpoints. The scoring is CPU work, at about a second per run per intervention. E6 adds nine fits of a 64×64 matrix, one per no-fallback seed, at a few CPU minutes each.
"""
