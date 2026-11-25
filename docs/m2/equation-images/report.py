# ruff: noqa: B018
# title: Equations as images

# A figure study for the D2.2 post, with no new training: the in-context grammar drawn with pictures for operands, and
# the `τ_ℓ fixed` runs (fixed-confirm, seeds 712 to 719) answering one context per pixel, before and after the full
# edit. `experiment.py` beside this script builds the operands and stores the answer pictures; this script draws them
# and writes each panel out as its own PNG, for an interactive figure.

import io
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import experiment as ex
from mini.lit import memo, stop
from mini.reports import current_publisher
from mini.store import project_store
from mini.vis import light_dark, themed
from sca.data.ops import colors

SCALE = 1
W, H = ex.size_of(SCALE)
PALETTE = np.asarray(colors(), float) / 15
OPS = ex.OP_NAMES
A = ex.ANCHORED_OP


def fetch(refs: list[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}.npz") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


RUNS = [(w, s) for w in ex.WALKS for s in ex.SEEDS]


def load() -> tuple[dict[str, np.ndarray], dict[tuple[str, int], dict[str, np.ndarray]]] | None:
    refs = (
        [ex.OPERANDS_REF.format(scale=SCALE)]
        + [ex.ANSWERS_REF.format(scale=SCALE, walk=w, seed=s) for w, s in RUNS]
        + [ex.RETRAINED_REF.format(scale=SCALE, walk=ex.REFERENCE_WALK, seed=s) for s in ex.SEEDS]
        + [ex.IDEAL_REF.format(scale=SCALE, walk=ex.REFERENCE_WALK)]
    )
    with tempfile.TemporaryDirectory() as tmp:
        files = fetch(refs, Path(tmp))
        if any(f is None for f in files.values()):
            return None
        out = []
        for r in refs:
            f = files[r]
            assert f is not None
            with np.load(f) as z:
                out.append({k: z[k] for k in z.files})
    # The references join the in-step answers of their seed under the keys `answer` reads: `retrained/<pick>`
    # beside `clean/<pick>` and `full edit/<pick>`, and `bayes/<pick>` and `null/<pick>` for the ideal predictors.
    runs = dict(zip(RUNS, out[1 : 1 + len(RUNS)], strict=True))
    for s, z in zip(ex.SEEDS, out[1 + len(RUNS) : -1], strict=True):
        runs[ex.REFERENCE_WALK, s] |= {retrained_key(k): v for k, v in z.items()} | out[-1]
    return out[0], runs


def retrained_key(key: str) -> str:
    """`<render>/<b>/<op>/<pick>` of the retrained answers as `<render>/<b>/<op>/retrained/<pick>`."""
    head, pick = key.rsplit("/", 1)
    return f"{head}/retrained/{pick}"


loaded = load()
if loaded is None:
    stop(
        "No results yet. Run the experiment:\n```bash\nbin/mini run docs/m2/equation-images/experiment.py --app modal\n```"
    )
operands, answers = loaded
OP_TABLES = ex.op_tables()


def truth(render: str, query: tuple[str, str], op: str) -> np.ndarray:
    """The true op answer at each pixel of a query pair, rounded to the nearest grid level."""
    a, b = operands[f"{render}/{query[0]}"], operands[f"{render}/{query[1]}"]
    return OP_TABLES[OPS.index(op), a, b]


def example_answer(render: str, pair: tuple[str, str], op: str) -> np.ndarray:
    return truth(render, pair, op)


RANDOM, IN_SYNC = ex.WALKS


def p_truth(seed: int, render: str, query: tuple[str, str], op: str, edit: str, walk: str = RANDOM) -> float:
    return float(answers[walk, seed][f"{render}/{query[1]}/{op}/{edit}/p_truth"].astype(float).mean())


# The seed shown: the median of the eight in how much of `difference` the full edit takes out, on the first query.
_q0 = ex.QUERIES[0]
_kept = {s: p_truth(s, "posterized", _q0, A, "full edit") / p_truth(s, "posterized", _q0, A, "clean") for s in ex.SEEDS}
SEED = sorted(_kept, key=lambda s: _kept[s])[len(_kept) // 2]

N_CONTEXTS = W * H * len(ex.QUERIES) * len(ex.RENDERS) * len(OPS) * len(ex.WALKS)

rf"""
# Equations as images

/// tip |
<!-- lede -->
We drew contexts of the in-context grammar with pictures for operands, so the model answers a whole image one pixel at a time. The unedited model paints a recognizable picture under every op, and the full edit turns the `difference` picture into a speckle of the other ops while leaving the others alone. Walking the examples in step with the query, as a compositor would, gathers the stray pixels into patches that follow the example images.
///

This is a figure study for the D2.2 post, with no new training. It reads the eight `τ_ℓ fixed` runs of the fixed-confirm experiment and asks each of them about {N_CONTEXTS:,} contexts, one per pixel of each query image, op and way of putting the operands on the grid, with and without the edit.
"""

# %%

rf"""
## Scope and method

Each pixel of a query image pair gets a context of its own. Its three examples are pixels from three example image pairs, each shown with its answer under the hidden op, and its query is the pixel pair itself. The model answers at the query `=`, and the answers over every pixel make a picture of the op as the model infers it. The examples are the same draws for every run and edit, so pictures from different runs and edits match pixel for pixel.

Operands
:   Photos from the Kodak Lossless True Color suite, which Kodak released for unrestricted use, and two synthetic images: a gray ramp from black to white, and a map of hue across and lightness down. Each is resized to {W} × {H} and put on the grid of 216 colors in one of two ways. Posterized rounds each channel to its nearest level. Dithered spreads the rounding error to the neighboring pixels (Floyd–Steinberg), so a patch of pixels averages to the original color.

Examples and queries
:   The examples are drawn from {", ".join(f"{a} × {b}" for a, b in ex.EXAMPLES)}. The queries are {" and ".join(f"{a} × {b}" for a, b in ex.QUERIES)}, where "huelight" is the hue and lightness map.

Example walk
:   How each example pixel is chosen: at random from its example pair (the default below), or at the query pixel's own position, so the examples walk over their images in step with the query (E4).

True answer
:   The op applied to the pixel pair, rounded to the nearest grid level, which is the most likely answer under the random rounding the corpus uses.

Argmax and T = 1
:   The color the model gives the most probability, and one color drawn from the model's distribution over colors.

Full edit
:   The e₁ component removed from the state at every position and every slice, as in the D2.2 design.

The pictures below are from seed {SEED}, the median of the eight in how much of `difference` the edit takes out. The table in E5 covers all eight.
"""

# %%


def show(ax, idx: np.ndarray, bold: bool = False) -> None:
    ax.imshow(PALETTE[idx], interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(2.5 if bold else 0.5)
        sp.set_color(light_dark("0.1", "0.9") if bold else light_dark("0.75", "0.35"))


def label(ax, text: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=16)


PANEL_IN = W / 96  # one image pixel per CSS pixel at the page's 96 px per inch
ROW_IN = PANEL_IN * H / W


def answer(walk: str, seed: int, render: str, query: tuple[str, str], op: str, col: str) -> np.ndarray:
    """One answer picture: `truth`, or `<edit>/<argmax|sample>` from the model."""
    if col == "truth":
        return truth(render, query, op)
    return answers[walk, seed][f"{render}/{query[1]}/{op}/{col}"]


def draw_context(walk: str, seed: int, render: str, query: tuple[str, str]) -> plt.Figure:
    rows = [(a, b, example_answer(render, (a, b), A)) for a, b in ex.EXAMPLES]
    rows.append((query[0], query[1], answer(walk, seed, render, query, A, "clean/sample")))
    gap, sep = 0.3, 0.25
    heights = [ROW_IN] * (len(rows) - 1) + [sep, ROW_IN]
    fig = plt.figure(figsize=(3 * PANEL_IN + 2 * gap + 0.3, sum(heights) + 0.08 * len(heights)))
    gs = fig.add_gridspec(
        len(heights),
        5,
        width_ratios=[PANEL_IN, gap, PANEL_IN, gap, PANEL_IN],
        height_ratios=heights,
        wspace=0.04,
        hspace=0.08,
    )
    grid_rows = [*range(len(rows) - 1), len(heights) - 1]
    for (an, bn, y), r in zip(rows, grid_rows, strict=True):
        is_query = r == len(heights) - 1
        show(fig.add_subplot(gs[r, 0]), operands[f"{render}/{an}"])
        show(fig.add_subplot(gs[r, 2]), operands[f"{render}/{bn}"])
        show(fig.add_subplot(gs[r, 4]), y, bold=is_query)
        label(fig.add_subplot(gs[r, 1]), "?")
        label(fig.add_subplot(gs[r, 3]), "=")
    rule = fig.add_subplot(gs[len(heights) - 2, :])
    rule.axis("off")
    rule.axhline(0.5, color=light_dark("0.6", "0.45"), lw=0.8)
    return fig


@memo
def context_figure(walk: str, seed: int, render: str, query: tuple[str, str]) -> str:
    alt = (
        f"Four equations drawn as pictures, one per row: three examples of {A}, each an image, a question mark, a "
        f"second image, an equals sign and their combination, then, below a rule, the query {query[0]} ? {query[1]} "
        "with the model answer painted pixel by pixel in a bold frame."
    )
    name = f"context-{walk.replace(' ', '-')}-{render}-{query[1]}"
    return themed(draw_context, name=name, alt_text=alt)(walk, seed, render, query)


r"""
## A context as pictures (E1)

What does one context look like when its operands are pictures? Below are the three examples of `difference`, each with its true answer, and the query, set off by a rule, with the model's answer in a bold frame. The model's answer is drawn at T = 1, and it is the only picture the model made.
"""

context_figure(RANDOM, SEED, "posterized", ex.QUERIES[0])

r"""
The examples show the op on very different inputs: a hue and lightness map, a gray ramp, and two photos. The query picture is mostly clean. Under `difference` the parrots keep their colors where the ramp is dark and turn to their complementary colors where it is light, as the true op does, with a scatter of single pixels where the drawn examples left the op in doubt or the draw landed on a less likely color.
"""

# %%

ROW_COLUMNS = (("truth", "true answer"), ("clean/sample", "unedited"), ("full edit/sample", "edited"))


def draw_rows(walk: str, seed: int, render: str, op: str) -> plt.Figure:
    gap = 0.25
    widths = [PANEL_IN, PANEL_IN, gap] + [PANEL_IN] * len(ROW_COLUMNS)
    fig = plt.figure(figsize=(sum(widths) + 0.2, len(ex.QUERIES) * (ROW_IN + 0.1) + 0.3))
    gs = fig.add_gridspec(len(ex.QUERIES), len(widths), width_ratios=widths, wspace=0.04, hspace=0.08)
    for r, query in enumerate(ex.QUERIES):
        for c, (img, title) in enumerate(
            ((operands[f"{render}/{query[0]}"], "a"), (operands[f"{render}/{query[1]}"], "b"))
        ):
            ax = fig.add_subplot(gs[r, c])
            show(ax, img)
            if r == 0:
                ax.set_title(title, fontsize=9)
        fig.add_subplot(gs[r, 2]).axis("off")
        for c, (col, title) in enumerate(ROW_COLUMNS, start=3):
            ax = fig.add_subplot(gs[r, c])
            show(ax, answer(walk, seed, render, query, op, col), bold=col != "truth")
            if r == 0:
                ax.set_title(title, fontsize=9)
    return fig


@memo
def rows_figure(walk: str, seed: int, render: str, op: str) -> str:
    alt = (
        f"Two rows of pictures, one per query: the operands a and b, then the {op} of them three ways: the true "
        "answer, the unedited model answer and the edited model answer, both drawn at T = 1."
    )
    name = f"rows-{walk.replace(' ', '-')}-{render}-{op}"
    return themed(draw_rows, name=name, alt_text=alt)(walk, seed, render, op)


r"""
## Taking out `difference` (E2)

What does the edit do to the picture? Each row is one query, with its operands, then the true answer, the unedited model, and the edited model, both models drawn at T = 1.
"""

rows_figure(RANDOM, SEED, "posterized", A)

r"""
The unedited model paints `difference` on both queries. The edited model paints something else: on the gray ramp the parrots come back mostly in their own colors, and on the hue and lightness map the picture looks at first like `mix`. Looked at closely, it is a speckle of the other ops, pixel by pixel, which E3 shows side by side.
"""

# %%

COLUMNS = (("truth", "true answer"), ("clean/argmax", "argmax"), ("clean/sample", "T = 1"))
EDITED = (("full edit/argmax", "edited, argmax"), ("full edit/sample", "edited, T = 1"))


def draw_ops(walk: str, seed: int, render: str, query: tuple[str, str]) -> plt.Figure:
    cols = COLUMNS + EDITED
    fig, axs = plt.subplots(
        len(OPS), len(cols), figsize=(len(cols) * (PANEL_IN + 0.06) + 0.9, len(OPS) * (ROW_IN + 0.08) + 0.3)
    )
    for r, op in enumerate(OPS):
        for c, (key, title) in enumerate(cols):
            ax = axs[r, c]
            show(ax, answer(walk, seed, render, query, op, key))
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(op, fontsize=9, rotation=0, ha="right", va="center")
    return fig


@memo
def ops_figure(walk: str, seed: int, render: str, query: tuple[str, str]) -> str:
    alt = (
        f"A grid of pictures, one row per op and five columns: the true answer for {query[0]} combined with "
        f"{query[1]} ({render}), the unedited model answer by argmax and by a draw at T = 1, and the edited model "
        "answer the same two ways."
    )
    name = f"ops-{walk.replace(' ', '-')}-{render}-{query[1]}"
    return themed(draw_ops, name=name, alt_text=alt)(walk, seed, render, query)


rf"""
## Every op, before and after the edit (E3)

Does the picture carry the op for every op, and what does the edit do to each? Each row is one op, with the true answer, then the unedited model, then the edited model. The query is {ex.QUERIES[0][0]} × {ex.QUERIES[0][1]}, posterized.
"""

ops_figure(RANDOM, SEED, "posterized", ex.QUERIES[0])

r"""
The unedited model paints each op recognizably. Most argmax pictures are close to the true answers, with a light speckle that the T = 1 pictures thicken. `mix` is speckled even at argmax: its answers round at random on most pairs in the corpus, so the model rightly spreads its probability over neighboring colors and the most likely one changes from pixel to pixel. Under the edit only the `difference` row changes. The argmax picture shows the op the edited model favors at each pixel, and the T = 1 picture shows the whole spread.

The same figure for the hue and lightness query:
"""

ops_figure(RANDOM, SEED, "posterized", ex.QUERIES[1])

r"""
Here the edited `difference` picture is closest to `mix`, with patches of `lighten`, `darken` and the HSV ops, which fits an edited model that falls back on the ops that best explain the examples at each pixel.
"""

# %%

r"""
## Examples in step with the query (E4)

So far each pixel drew its examples at random. A compositor works differently: it walks every image in step, so each pixel of the query sees the examples at its own position. What changes when the examples walk in step with the query? The same two figures, with the examples taken that way:
"""

context_figure(IN_SYNC, SEED, "posterized", ex.QUERIES[1])

# %%

rows_figure(IN_SYNC, SEED, "posterized", A)

rf"""
Walking in step changes little overall: the probability on the true answer is within a hundredth or two of the random walk for every op, edited or not (E5). What changes is where the stray pixels fall. With random examples they spread evenly over the picture. In step they gather into patches that follow the example images, presumably where the three examples at that position say little about the op, because several ops give them similar answers. The edited pictures show this most (on seed {SEED}, a map of the probability on the true answer, averaged over 8 × 8 blocks, varies a third to a half more in step than at random, against about a sixth more for the unedited model), which fits an edited model that leans on the examples to choose among the remaining ops.

For a reader who knows compositors, the in-step walk is the familiar one, and its patches show something true about in-context inference: each answer depends on what its own examples show. For a reader new to both, the patches may look like part of the op, so the random walk keeps the picture closer to the op alone.
"""

# %%


def table_rows() -> list[dict]:
    rows = []
    for op in OPS:
        row: dict = {"op": op}
        for walk in ex.WALKS:
            for edit in ex.EDITS:
                v = np.array([p_truth(s, "posterized", ex.QUERIES[0], op, edit, walk) for s in ex.SEEDS])
                row[walk, edit] = (float(v.mean()), float(v.min()), float(v.max()))
        rows.append(row)
    return rows


TABLE = table_rows()


def fmt(t: tuple[float, float, float]) -> str:
    return f"{t[0]:.2f} ({t[1]:.2f} to {t[2]:.2f})"


_lines = "\n".join(
    f"| `{r['op']}` | " + " | ".join(fmt(r[w, e]) for w in ex.WALKS for e in ex.EDITS) + " |" for r in TABLE
)
_diff = next(r for r in TABLE if r["op"] == A)

rf"""
## All eight runs (E5)

Do the other seeds paint the same pictures? The table gives the probability each run puts on the true answer, averaged over the pixels of the {ex.QUERIES[0][0]} × {ex.QUERIES[0][1]} query (posterized), as the seed mean with the seed range in parentheses.

| op | random, unedited | random, edited | in step, unedited | in step, edited |
| --- | ---: | ---: | ---: | ---: |
{_lines}

With random examples the edit takes the `difference` picture from {_diff[RANDOM, "clean"][0]:.2f} to {_diff[RANDOM, "full edit"][0]:.2f} on the seed mean and moves the other ops by a few hundredths at most, so the pictures of seed {SEED} are typical. These probabilities are not the D2.2 scores, which are taken on held-out corpus contexts with replacement op noise and weigh every answer the random rounding can give.
"""

# %%

REFERENCE_COLUMNS = (
    ("truth", "true answer"),
    ("bayes/argmax", "ideal₇"),
    ("clean/argmax", "unedited"),
    ("null/argmax", "ideal₆"),
    ("retrained/argmax", "retrained"),
    ("full edit/argmax", "edited"),
)


def draw_references(seed: int, render: str, op: str) -> plt.Figure:
    gap = 0.25
    groups = (REFERENCE_COLUMNS[:3], REFERENCE_COLUMNS[3:])
    widths = [PANEL_IN, PANEL_IN] + [w for g in groups for w in [gap] + [PANEL_IN] * len(g)]
    fig = plt.figure(figsize=(sum(widths) + 0.2, len(ex.QUERIES) * (ROW_IN + 0.1) + 0.3))
    gs = fig.add_gridspec(len(ex.QUERIES), len(widths), width_ratios=widths, wspace=0.04, hspace=0.08)
    for r, query in enumerate(ex.QUERIES):
        cells: list[tuple[np.ndarray, str] | None] = [
            (operands[f"{render}/{query[0]}"], "a"),
            (operands[f"{render}/{query[1]}"], "b"),
        ]
        for g in groups:
            cells += [None] + [(answer(IN_SYNC, seed, render, query, op, col), title) for col, title in g]
        for c, cell in enumerate(cells):
            ax = fig.add_subplot(gs[r, c])
            if cell is None:
                ax.axis("off")
                continue
            show(ax, cell[0])
            if r == 0:
                ax.set_title(cell[1], fontsize=9)
    return fig


@memo
def references_figure(seed: int, render: str, op: str) -> str:
    alt = (
        f"Two rows of pictures, one per query: the operands a and b, then the {op} of them six ways, each at the most "
        "likely color: the true answer, ideal₇ (the ideal predictor) and the unedited model; then ideal₆ (the ideal predictor without "
        "difference), the model retrained without difference and the edited model."
    )
    return themed(draw_references, name=f"references-{render}-{op}", alt_text=alt)(seed, render, op)


AGREE_PAIRS = (
    ("clean/argmax", "bayes/argmax"),
    ("full edit/argmax", "null/argmax"),
    ("retrained/argmax", "null/argmax"),
)


def agreement(seed: int, query: tuple[str, str], op: str, col: str, ref: str) -> float:
    """The share of pixels where two argmax pictures (posterized, examples in step) name the same color."""
    pic = answer(IN_SYNC, seed, "posterized", query, op, col)
    return float((pic == answer(IN_SYNC, seed, "posterized", query, op, ref)).mean())


def agreement_cell(query: tuple[str, str], op: str, col: str, ref: str) -> tuple[float, float, float]:
    v = np.array([agreement(s, query, op, col, ref) for s in ex.SEEDS])
    return float(v.mean()), float(v.min()), float(v.max())


AGREE = {(q[1], op, col): agreement_cell(q, op, col, ref) for q in ex.QUERIES for op in OPS for col, ref in AGREE_PAIRS}
_agree_lines = "\n".join(
    f"| `{op}` | " + " | ".join(fmt(AGREE[q[1], op, col]) for q in ex.QUERIES for col, _ in AGREE_PAIRS) + " |"
    for op in OPS
)
_ramp, _hue = (q[1] for q in ex.QUERIES)

r"""
## Beside the ideal and the retrained model (E6)

How close do the edited pictures come to what a model that never knew `difference` should paint? Two references answer the same contexts, with the examples in step.

Ideal predictors
:   Ideal₇ is the Bayes predictive over all seven ops: the posterior over ops given a pixel's three examples, under the corpus noise model, weighting each op's answer distribution on the query pair. Ideal₆ takes `difference` out of the posterior and renormalizes the other six (the target null), which is what an edited model should aim at.

Retrained model
:   The model trained without `difference` in the filtered-corpus experiment: the same recipe with its loss masked on `difference` contexts, at the same seed.

Every picture below is the most likely color at each pixel (T = 0), so the ideal pictures are as clean as the models' and differences come from the answers rather than the draw.
"""

references_figure(SEED, "posterized", A)

rf"""
The table gives the share of pixels where two of these pictures name the same color, as the mean over the eight seeds with the seed range in parentheses, posterized, for the ramp query and then the hue and lightness query.

| op | unedited ~ ideal₇ | edited ~ ideal₆ | retrained ~ ideal₆ | unedited ~ ideal₇ | edited ~ ideal₆ | retrained ~ ideal₆ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{_agree_lines}

On `difference`, the edited model names the same color as ideal₆ on {AGREE[_ramp, A, "full edit/argmax"][0]:.0%} of the ramp pixels and the retrained model on {AGREE[_ramp, A, "retrained/argmax"][0]:.0%}; on the hue and lightness query, {AGREE[_hue, A, "full edit/argmax"][0]:.0%} and {AGREE[_hue, A, "retrained/argmax"][0]:.0%}. These shares are low for every model because ideal₆ is often close to a tie: once `difference` is out, several ops explain a pixel's examples about equally, and a small shift in probability moves the most likely color from one op's answer to another's. The retrained model, which never knew `difference`, sets the level a model can reach here, and the edited model sits about five hundredths below it on both queries. In the pictures, all three paint the same kind of patchwork of the other ops.
"""

# %%

MANIFEST: dict[str, str] = {}


def slug(text: str) -> str:
    return text.replace("/", "-").replace(" ", "-").replace("full-edit", "edited")


def png(idx: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((PALETTE[idx] * 255).round().astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# The reference pictures as panels. Ideal₆ (without `difference`) takes the name `ideal`, the comparison the
# widget offers beside the edited and retrained models; the full-task ideal is `bayes`.
REFERENCE_PANELS = (("retrained", "retrained"), ("null", "ideal"), ("bayes", "bayes"))


def query_panels(seed: int, render: str, query: tuple[str, str], op: str) -> list[tuple[str, np.ndarray]]:
    """The panels of one query under one op, named: the true answer, the model answers, and the references."""
    out = [(f"truth-{render}-{query[1]}-{op}", truth(render, query, op))]
    for walk in ex.WALKS:
        for col, _ in COLUMNS[1:] + EDITED:
            out.append(
                (
                    f"answer-{slug(walk)}-{render}-{query[1]}-{op}-{slug(col)}",
                    answer(walk, seed, render, query, op, col),
                )
            )
    for ref, label in REFERENCE_PANELS:
        for pick in ("argmax", "sample"):
            key = f"answer-{slug(ex.REFERENCE_WALK)}-{render}-{query[1]}-{op}-{label}-{pick}"
            out.append((key, answer(ex.REFERENCE_WALK, seed, render, query, op, f"{ref}/{pick}")))
    return out


def write_panels(seed: int) -> str | None:
    """Each panel of the shown seed as its own PNG at its native size, a manifest naming them, and the widget page
    that shows them. Returns the widget URL.
    """
    pub = current_publisher()
    if pub is None:
        return None
    for render in ex.RENDERS:
        for name in sorted({n for pair in ex.EXAMPLES + ex.QUERIES for n in pair}):
            key = f"operand-{render}-{name}"
            MANIFEST[key] = pub.asset_url(png(operands[f"{render}/{name}"]), name=f"{key}.png")
        for op in OPS:
            for pair in ex.EXAMPLES:
                key = f"example-{render}-{pair[0]}-{pair[1]}-{op}"
                MANIFEST[key] = pub.asset_url(png(example_answer(render, pair, op)), name=f"{key}.png")
            for query in ex.QUERIES:
                for key, img in query_panels(seed, render, query, op):
                    MANIFEST[key] = pub.asset_url(png(img), name=f"{key}.png")
    MANIFEST["manifest"] = pub.asset_url(json.dumps(MANIFEST, indent=1).encode(), name="panels.json")
    body = (Path(__file__).parent / "widget.html").read_text()
    page = (
        '<!doctype html>\n<html lang="en" data-panels="">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "<style>*,*::before,*::after{box-sizing:border-box}body{margin:0}img{max-width:100%}"
        "[hidden]{display:none!important}</style>\n</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )
    return pub.asset_url(page.encode(), name="widget.html")


WIDGET = write_panels(SEED)

rf"""
## The panels as files

Every panel of seed {SEED} is also a PNG at its native size ({W} × {H}, one pixel per grid color), named in [panels.json]({MANIFEST.get("manifest", "_assets/panels.json")}). A [small page]({WIDGET or "_assets/widget.html"}) shows them as one context at a time, with the examples walked in step, and a choice of op, colors, second image, temperature and answer (the unedited, edited and retrained models, the ideal predictors ideal₇ and ideal₆, and the true answer), for embedding in a post.

## References

filtered-corpus
:   [A model trained without `difference`](../filtered-corpus/report.py): the `sans-diff` reference, the same recipe with its loss masked on `difference` contexts, at seeds 712 to 719.

fixed-confirm
:   [Fixed confirmation](../fixed-confirm/report.py): `τ_ℓ fixed` kept the anchor on all eight fresh seeds, 712 to 719, and became the recipe of record.
"""
