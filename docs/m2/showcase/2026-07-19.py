import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    app_title="M2 showcase: iteration 1 — a transformer's own color geometry",
    css_file="../../report.css",
    layout_file="layouts/2026-07-19.slides.json",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import importlib.util
    import json
    import tempfile
    from pathlib import Path

    import marimo as mo  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np

    from mini.reports import report_bundle, use_publisher
    from mini.store import project_store
    from mini.vis import light_dark, themed
    from sca.data import named_colors as nc
    from sca.data.colors import N_LEVELS, swatch

    use_publisher(report_bundle(__file__))

    def _experiment(rel: str):
        """Import docs/m2/<rel>/experiment.py under a unique module name.

        Every experiment module is called ``experiment.py``, so the usual
        ``from experiment import METRICS_REF`` trick only works for a report that
        sits beside one of them. A showcase reads from several.
        """
        path = Path(__file__).parent.parent / rel / "experiment.py"
        spec = importlib.util.spec_from_file_location(f"exp_{rel.replace('-', '_').replace('.', '_')}", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    ex211, ex212, ex213 = (_experiment(e) for e in ("ex-2.1.1", "ex-2.1.2", "ex-2.1.3"))

    SEEDS = ex213.SEEDS
    GRIDS = list(nc.GRIDS)
    PALETTES = {g: nc.grid_palette(nc.GRIDS[g]) for g in GRIDS}
    VOCAB_RGB = {g: np.array(list(p.values()), dtype=np.float32) / (N_LEVELS - 1) for g, p in PALETTES.items()}

    def load_results() -> tuple[list, dict, dict, dict[str, np.ndarray]] | None:
        """Resolve every experiment's durable results, or None if any are unpublished."""
        store = project_store()
        m211_a = store.get_ref(ex211.METRICS_REF)
        m212_a = store.get_ref(ex212.METRICS_REF)
        m213_a = store.get_ref(ex213.METRICS_REF)
        arr_a = store.get_ref(ex213.ARRAYS_REF)
        if m211_a is None or m212_a is None or m213_a is None or arr_a is None:
            return None
        with tempfile.TemporaryDirectory() as d:
            m211 = json.loads(store.get(m211_a, Path(d) / "m211.json").read_text())
            m212 = json.loads(store.get(m212_a, Path(d) / "m212.json").read_text())
            m213 = json.loads(store.get(m213_a, Path(d) / "m213.json").read_text())
            with np.load(store.get(arr_a, Path(d) / "arrays.npz")) as z:
                arrays = {k: z[k] for k in z.files}
        return m211, m212, m213, arrays

    def rows_of(cells: list[dict], **match: str) -> list[dict]:
        """The metric rows whose fields equal every given value."""
        return [r for r in cells if all(r[k] == v for k, v in match.items())]

    def grid_shades() -> dict[str, tuple]:
        stops = light_dark([0.82, 0.55, 0.32, 0.08], [0.88, 0.62, 0.42, 0.2])
        return dict(zip(GRIDS, plt.cm.viridis(stops), strict=True))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # M2 showcase: iteration 1

    *A collated report for people following the project: three experiments, one
    story. Each section links to the full experiment report; here we retell the
    story at walking pace. This edition covers iteration 1 of milestone 2
    (June–July 2026).*

    Sparse Concept Anchoring (SCA) is a training-time method for concept
    control. The usual approach to interpretability is archaeology: train a
    network, then dig through its representations hoping to find where a
    concept ended up. SCA inverts that. A light regularizer, active during
    training, guides a chosen concept toward a location we picked in advance —
    so when we later want to suppress or delete the concept, we already know
    where it lives and can bound the side effects before intervening.

    [Milestone 1](https://arxiv.org/abs/2512.12469) established the method in
    autoencoders. Milestone 2 asks: does it transfer to transformers?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## The task: a language of color mixing

    To anchor a concept we first need a task whose concepts we know completely.
    We built a tiny synthetic language about mixing colors, where every color
    is a point on a 16-level RGB grid and mixing is exact integer arithmetic
    (the channel-wise mean). Every prompt has exactly one correct completion,
    so a wrong answer is unambiguously wrong — a negative result stays
    interpretable.

    A small transformer trains on sentences like these, one character at a time:

    | Sentence type | Example |
    |------|---------|
    | Named pair | {swatch("red")} + {swatch("blue")} = {swatch("purple")} |
    | Hex pair | `#f00 + #00f = #808` |
    | Cross-form | {swatch("red")} + `#00f` = `#808` |
    | Alias | {swatch("red")} = `#f00` |

    The same colors have two spellings — names and hex codes — tied together by
    alias sentences. That's deliberate: it mirrors, in miniature, how real
    language models meet a concept through many surface forms. The question for
    this iteration: what does the model actually learn about *color*, the
    concept behind both spellings?
    """)
    return


@app.cell(hide_code=True)
def _():
    loaded = load_results()
    return (loaded,)


@app.cell(hide_code=True)
def _(loaded):
    mo.stop(
        loaded is None,
        mo.md(r"""
        No results yet — run the experiments this showcase collates:

        ```bash
        bin/mini run docs/m2/ex-2.1.1/experiment.py --app modal
        bin/mini run docs/m2/ex-2.1.2/experiment.py --app modal
        bin/mini run docs/m2/ex-2.1.3/experiment.py --app modal
        ```
        """),
    )
    metrics211, metrics212, metrics213, arrays213 = loaded
    return metrics211, metrics212, metrics213, arrays213


@app.cell(hide_code=True)
def _(metrics211):
    _sets = ["named_seen", "hex_unseen", "cross_unseen", "named_holdout"]
    _labels = ["seen\nnamed pairs", "unseen\nhex pairs", "unseen\ncross-form", "held-out\nnamed pairs"]
    _acc = np.array(
        [[rows_of(metrics211, label=f"d64-L4-s{s}")[0]["accuracy"][es]["accuracy"] for es in _sets] for s in SEEDS]
    )

    @themed(
        name="baseline-accuracy",
        alt_text="""
            Bar chart of exact-match accuracy for four evaluation sets. Seen named
            pairs, unseen hex pairs, and unseen cross-form pairs are all at 1.0;
            held-out named pairs are at exactly zero.
        """,
        caption="""
            Completion accuracy of the baseline transformer (largest model, three
            seeds shown as dots). The model is perfect on everything it can reach
            by memorization or hex arithmetic, and never answers a held-out named
            pair correctly.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        xs = np.arange(len(_sets))
        ax.bar(xs, _acc.mean(axis=0), color=plt.cm.viridis(light_dark(0.45, 0.6)), width=0.62)
        for i in xs:
            ax.plot([i] * len(SEEDS), _acc[:, i], "o", color="#0008", ms=3, zorder=3)
        ax.set_xticks(xs, _labels)
        ax.set(ylabel="exact-match accuracy", ylim=(-0.03, 1.03))
        ax.grid(alpha=0.3, axis="y")
        ax.set_title("Baseline: perfect, except where composition is required")
        return fig

    mo.vstack(
        [
            mo.md(r"""
        ## Act 1: the baseline ([ex-2.1.1](../ex-2.1.1/))

        First we trained the transformer with no anchoring at all, as the
        baseline that anchored runs will be compared against. It learns the
        language easily — with one striking exception. Some named pairs were
        held out of training entirely: the model has seen `blue` and `black` in
        many sentences, just never mixed with each other. Answering those
        requires composing knowledge — say, recalling each operand's value via
        its alias, mixing in value space, and translating back to a name. The
        model never does it.
        """),
            mo.Html(_plot()),
        ]
    )
    return


@app.cell(hide_code=True)
def _(metrics211, metrics212):
    _fails = rows_of(metrics211, label="d64-L4-s0")[0]["accuracy"]["named_holdout"]["failures"][:3]
    _rows = "\n".join(
        f"| {swatch(a)} + {swatch(b)} = | {swatch(want)} | {swatch(got)} |"
        for (prompt, want, got) in _fails
        for (a, _plus, b, _eq) in [tuple(prompt.split())]
    )
    _conds = ["control", "rev", "open", "both"]
    _hold = {
        c: float(np.mean([r["accuracy"]["named_holdout"]["accuracy"] for r in rows_of(metrics212["cells"], cond=c)]))
        for c in _conds
    }

    mo.md(rf"""
    ## Act 2: is the corpus the problem? ([ex-2.1.2](../ex-2.1.2/))

    What does the model say instead? Its guesses are legal color names, just
    the wrong ones (baseline, seed 0):

    | Prompt | Correct | Model said |
    |--------|---------|-----------|
    {_rows}

    Maybe the corpus never made composition necessary: the named slice is small
    enough to memorize, and the alias dictionary only runs one way. So we
    changed the grammar — reversed alias sentences, named equations whose
    answers leave the palette, and both together. Every variant trains fine,
    and held-out named accuracy stays at
    {", ".join(f"{_hold[c]:.2f}" for c in _conds)} across the four conditions.

    The diagnosis got sharper, though. The failure splits in two: a spelling
    rule (the model answers with the *correct value*, written in hex, where a
    name was required), and a translation step — value to name — that never
    engages mid-equation. Probes of the network's internal state suggested the
    mixed color never fully exists at any single position. That raised a
    pointed question: can this model family learn color *geometry* from names
    at all, or has it been leaning on hex arithmetic the whole time?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Act 3: names all the way down ([ex-2.1.3](../ex-2.1.3/))

    So we removed the crutch. In the third experiment there is no hex, no
    aliases, no characters: every color is a single opaque token, and the only
    sentences are named mixing equations. Nothing in the training text says
    that colors live on a 3D grid, or that colors are values at all. The only
    structure left is which names co-occur with which.

    This is matrix completion wearing a language-model costume. The mixing
    table is a large symmetric table generated by a tiny hidden rule (each
    color is a point in a cube; mixing averages the coordinates). A model that
    discovers the rule can fill in table cells it has never seen; a model that
    memorizes entries cannot. Held-out pairs separate the two.

    We swept the vocabulary from 27 colors (a small, memorizable table) to
    4096 (a table of 8.4 million pairs, of which training shows about 1% —
    generalization or nothing).
    """)
    return


@app.cell(hide_code=True)
def _(metrics213):
    _sets = ["named_seen", "named_holdout"]

    def _acc(g: str, es: str) -> float:
        return float(np.mean([r["sets"][es]["accuracy"] for r in rows_of(metrics213["cells"], grid=g)]))

    @themed(
        name="vocab-accuracy",
        alt_text="""
            Two bar panels of exact-match accuracy against vocabulary size (27, 64,
            216, 4096 colors): seen pairs and held-out pairs, three seeds as dots.
            Seen accuracy is 1.0 except at 4096 (0.85). Held-out accuracy rises from
            0.27 at 27 colors to essentially 1.0 at 216, then drops to 0.65 at 4096.
        """,
        caption="""
            Exact-match accuracy by vocabulary size (bars: mean of three seeds;
            dots: individual seeds). Held-out pairs never appear in training, so
            they can only be answered by inferring the hidden geometry.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(_sets), figsize=(8.4, 3.2), sharey=True)
        shades = grid_shades()
        xs = np.arange(len(GRIDS))
        for ax, es, title in zip(axes, _sets, ["seen pairs", "held-out pairs"], strict=True):
            per_seed = np.array(
                [[r["sets"][es]["accuracy"] for r in rows_of(metrics213["cells"], grid=g)] for g in GRIDS]
            )
            ax.bar(xs, per_seed.mean(axis=1), color=[shades[g] for g in GRIDS], width=0.62)
            for i in range(len(GRIDS)):
                ax.plot([xs[i]] * per_seed.shape[1], per_seed[i], "o", color="#0008", ms=3, zorder=3)
            ax.set(title=title, ylim=(-0.03, 1.03))
            ax.set_xticks(xs, [g.removeprefix("v") + " colors" for g in GRIDS])
            ax.grid(alpha=0.3, axis="y")
        axes[0].set_ylabel("exact-match accuracy")
        return fig

    mo.vstack(
        [
            mo.md(rf"""
        The answer is yes: the model infers the geometry from co-occurrence
        alone. With the same 27 colors that sat at zero through all of act 2,
        held-out accuracy comes up off the floor ({_acc("v27", "named_holdout"):.2f});
        at 216 colors the held-out pairs are essentially solved
        ({_acc("v216", "named_holdout"):.2f}); and at 4096, where memorizing is
        hopeless, the model still answers {_acc("v4096", "named_holdout"):.0%}
        of pairs it has never seen.
        """),
            mo.Html(_plot()),
        ]
    )
    return


@app.cell(hide_code=True)
def _(metrics213):
    def _mean(g: str, es: str, key: str) -> float:
        return float(np.mean([r["sets"][es][key] for r in rows_of(metrics213["cells"], grid=g)]))

    mo.md(rf"""
    Exact match understates the result. When the 4096-color model misses a
    held-out pair, its guess lands a mean distance of
    {_mean("v4096", "named_holdout", "guess_dist"):.3f} from the true color, in
    unit-cube units where a random guess would land at
    {_mean("v4096", "named_holdout", "chance_dist"):.2f}. The misses are
    immediate neighbors on the color grid, not confusions.

    Some operand pairs mix to a color with no name in the vocabulary at all,
    so an exact answer is impossible and "close" is the only kind of correct.
    There, the 216-color model's guesses land
    {_mean("v216", "open", "guess_dist"):.3f} from the true mix, where
    {_mean("v216", "open", "floor_dist"):.3f} is the best any answer could
    achieve (the nearest available name) and random guessing again sits near
    {_mean("v216", "open", "chance_dist"):.2f}. The model isn't recalling
    answers; it's computing with values it was never shown.
    """)
    return


@app.cell(hide_code=True)
def _(arrays213, metrics213):
    @themed(
        name="embedding-pca",
        alt_text="""
            Four scatter panels, one per vocabulary size, projecting each model's
            color-token embeddings onto their first two principal components, every
            point drawn in the color it names. Small vocabularies show loose color
            grouping; 216 colors shows a clear hue gradient; 4096 colors forms a
            smooth color wheel with hues around the rim and darker colors inward.
        """,
        caption="""
            Each model's embedding table projected onto its own two leading
            principal components (seed 0), every token drawn in the color it
            names. Nearby points having nearby colors means the model's internal
            layout mirrors color space. R²: how well true RGB can be read off the
            full embedding by a linear probe (mean of three seeds).
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(GRIDS), figsize=(11.5, 3.1))
        for ax, g in zip(axes, GRIDS, strict=True):
            emb = arrays213[f"{g}-s0/embeddings"]
            centered = emb - emb.mean(0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            z = centered @ vt[:2].T
            z /= np.abs(z).max() + 1e-9
            n = len(emb)
            ax.scatter(z[:, 0], z[:, 1], c=VOCAB_RGB[g], s=float(np.clip(6_000 / n, 4, 50)), lw=0)
            r2 = np.mean([r["emb_r2"] for r in rows_of(metrics213["cells"], grid=g)])
            ax.set_title(f"{g.removeprefix('v')} colors   (R² {r2:.2f})", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_axis_off()
        return fig

    mo.vstack(
        [
            mo.md(r"""
        ## The geometry the model built

        We can also look at the learned structure directly. Every color is a
        single token, so everything the model knows about a color lives in that
        token's embedding vector. Projecting each embedding table onto its two
        leading directions of variation, and painting every token with the
        color it names, makes the structure visible: no axis below means
        anything by itself, but if the model has organized colors by their
        values, nearby points will have nearby colors.
        """),
            mo.Html(_plot()),
            mo.md(r"""
        At 4096 colors the model has arranged tokens it only ever saw as
        arbitrary symbols into a recognizable color wheel. Nobody told it
        colors have hues; that came entirely from which names appear together
        in mixing equations.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What this means for anchoring, and what's next

    We set out to build a baseline and instead learned something about the
    substrate. The concept we want to anchor (color-as-value) does form in
    this model family, and it forms from co-occurrence statistics alone, with
    no help from a value-revealing spelling. That is good news for SCA: the
    anchoring regularizer needs a geometrically organized representation to
    steer, and this one organizes itself. The sweep also settled a design
    question: 216 colors is the size at which the task is both solved and
    solved by geometry rather than memory, so that's the vocabulary the
    anchored runs will build on.

    Next, iteration 2: a character-level twin of the names-only experiment
    (to close the loop with acts 1 and 2), and then the main event — training
    the same models with an anchor on *redness*, and measuring what the
    anchor costs and what it buys.

    Full reports: [ex-2.1.1](../ex-2.1.1/) · [ex-2.1.2](../ex-2.1.2/) ·
    [ex-2.1.3](../ex-2.1.3/), and the [experiment index](../../index.md).
    """)
    return


if __name__ == "__main__":
    app.run()
