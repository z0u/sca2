# title: Ex 2.2.5: a pilot of stochastic rounding

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import experiment as ex
import matplotlib.pyplot as plt
import numpy as np

from mini.lit import memo, stop
from mini.store import project_store
from mini.vis import AxesRow, figure_html, light_dark, themed

SLICE_NAMES = ["emb", "1", "2", "3", "4"]
POS_NAMES = ["op1", "op", "op2", "=", "ans", "⏎"]
ROUNDED_OPS = ("mix", "screen", "multiply")
# The ops whose answers round at all; `add`, `lighten` and `darken` land on the grid on every pair.

INK = {
    "control-stoch": ("#d0461b", "#f07a50"),
    "recipe-stoch": ("#1f6fb4", "#5fa8dd"),
    "control-near": ("#777", "#999"),
    "control-short": ("#bbb", "#666"),
    "recipe-short": ("#9ec3e6", "#3d6a8f"),
    "control": ("#bbb", "#666"),
}
# One ink per condition, as (light, dark) pairs; production arms draw pale.


REFS = [ex.METRICS_REF, ex.ARRAYS_REF, ex.GEOMETRY_REF, ex.EX223_METRICS_REF, ex.EX223_GEOMETRY_REF]


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists.

    One `get_refs` and one `get_many` for the lot: the bucket's per-call latency is most of a render if the refs are resolved one at a time.
    """
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict | None:
    return None if path is None else json.loads(path.read_text())


def read_npz(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def span2(v: np.ndarray, fmt: str = ".3f") -> str:
    """Seed mean with half the seed range beside it, in the shared `.range` style."""
    v = np.asarray(v, float)
    if len(v) == 1:
        return f"{v[0]:{fmt}}"
    return f"{v.mean():{fmt}} <span class='range'>±{(v.max() - v.min()) / 2:{fmt}}</span>"


def ink(cond: str):
    return light_dark(*INK[cond])


def table_html(head: list[str], rows: list[list[str]], caption: str, *, ref_rows: frozenset[int] = frozenset()) -> str:
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=ref' if r in ref_rows else ''}>"
        + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


@dataclass(frozen=True)
class Results:
    """The pilot's published results, and the production ex-2.2.3 arms they are read against."""

    metrics: dict
    arrays: dict[str, np.ndarray]
    geometry: dict
    prod: dict
    prod_geometry: dict

    def runs(self, cond: str) -> list[dict]:
        return sorted((r for r in self.metrics["runs"] if r["condition"] == cond), key=lambda r: r["seed"])

    def rounding(self, cond: str) -> list[dict]:
        return sorted((r for r in self.metrics["rounding"] if r["condition"] == cond), key=lambda r: r["seed"])

    def scored(self, cond: str) -> list[dict]:
        return sorted((r for r in self.metrics["scores"] if r["condition"] == cond), key=lambda r: r["seed"])

    def prod_runs(self, cond: str) -> list[dict]:
        """A production arm's eval records; `recipe-short` pools its addendum seeds (twenty in all)."""
        names = (cond, f"{cond}-more")
        return sorted((r for r in self.prod["runs"] if r["condition"] in names), key=lambda r: r["seed"])

    def prod_scored(self, cond: str) -> list[dict]:
        names = (cond, f"{cond}-more")
        return sorted((r for r in self.prod["scores"] if r["condition"] in names), key=lambda r: r["seed"])

    @staticmethod
    def stat(runs: list[dict], key: str, op: str | None = None) -> np.ndarray:
        return np.array([r[key] if op is None else r["per_op"][op][key] for r in runs], float)

    @staticmethod
    def em(runs: list[dict], op: str) -> np.ndarray:
        return np.array([r["holdout_em"][op] for r in runs], float)

    def round_stat(self, cond: str, op: str, split: str, key: str) -> np.ndarray:
        return np.array([r["per_op"][op][split][key] for r in self.rounding(cond)], float)

    def lines(self, cond: str) -> dict[str, np.ndarray]:
        """The per-line rounding table of a condition, seeds concatenated."""
        keys = [k.split("/")[-1] for k in self.arrays if k.startswith(f"{cond}-s0/lines/")]
        return {k: np.concatenate([self.arrays[f"{r['label']}/lines/{k}"] for r in self.rounding(cond)]) for k in keys}

    @staticmethod
    def score(scored: list[dict], op: str, iv: str | None, key: str, group: str) -> np.ndarray:
        out = []
        for r in scored:
            s = r["ops"][op]
            v = s["clean"][key] if iv is None else s["interventions"][iv][key]
            out.append(v[group])
        return np.array(out, float)

    def r2(self, cond: str, prod: bool = False) -> np.ndarray:
        """(seeds, slices, positions, targets, channels) strict-holdout R² of a control arm's cube probes."""
        src = self.prod_geometry if prod else self.geometry
        return np.array([r["r2_strict"] for r in src["runs"] if r["condition"] == cond], float)


with tempfile.TemporaryDirectory() as _tmp:
    files = fetch(REFS, Path(_tmp))
    metrics = read_json(files[ex.METRICS_REF])
    arrays_, geometry_ = read_npz(files[ex.ARRAYS_REF]), read_json(files[ex.GEOMETRY_REF])
    prod_metrics, prod_geometry = read_json(files[ex.EX223_METRICS_REF]), read_json(files[ex.EX223_GEOMETRY_REF])
if metrics is None:
    stop("_Results are not published yet; the result cells render once they are._")
assert metrics is not None
assert arrays_ and geometry_ and prod_metrics and prod_geometry, "a result is missing from the store"
res: Results = Results(metrics, arrays_, geometry_, prod_metrics, prod_geometry)


def _hs(c: str, op: str, k: str) -> float:
    return float(res.round_stat(c, op, "holdout", k).mean())


ceil_ = {op: _hs("control-near", op, "ceiling") for op in ROUNDED_OPS}
gap = max(abs(_hs(c, op, "expected_em") - ceil_[op]) for c in ("control-stoch", "recipe-stoch") for op in ROUNDED_OPS)
draw_se = max(np.sqrt(ceil_[op] * (1 - ceil_[op]) / _hs("control-stoch", op, "n")) for op in ROUNDED_OPS)
pm_ = {
    op: (_hs("control-stoch", op, "p_mode_rounded"), _hs("control-stoch", op, "ceiling_rounded")) for op in ROUNDED_OPS
}
kl_ = [_hs(c, op, "kl_rounded") for c in ("control-stoch", "recipe-stoch") for op in ROUNDED_OPS]
kl_near = [_hs("control-near", op, "kl_rounded") for op in ROUNDED_OPS]
supp = min(_hs(c, op, "support_rounded") for c in ("control-stoch", "recipe-stoch") for op in ROUNDED_OPS)
mode_mix = _hs("control-stoch", "mix", "em_mode_rounded")
mode_rest = [_hs("control-stoch", op, "em_mode") for op in ("screen", "multiply")]
m_line_stoch, m_line_prod = (
    res.stat(res.runs("recipe-stoch"), "m_line"),
    res.stat(res.prod_runs("recipe-short"), "m_line"),
)
projection_iv = ex.ex223.PROJECTION.name
scored_stoch, scored_prod = res.scored("recipe-stoch"), res.prod_scored("recipe-short")
clean_mul = res.score(scored_stoch, "multiply", None, "acc", "red").mean()
redder = {
    n: (
        res.score(s, "multiply", None, "acc", "redder").mean(),
        res.score(s, "multiply", projection_iv, "acc", "redder").mean(),
    )
    for n, s in (("stoch", scored_stoch), ("prod", scored_prod))
}
grammar = res.metrics["grammar"]
dredder = {op: grammar[op]["redder_stochastic"] / grammar[op]["redder_nearest"] - 1 for op in ROUNDED_OPS}
r2_answer = {n: res.r2(n)[:, :, 3:5, 2].mean((0, 3)).max(0) for n in ("control-stoch", "control-near")}

rf"""
# Ex 2.2.5: a pilot of stochastic rounding of off-grid answers

/// tip |
<!-- tl;dr -->
A scouting run on a corpus where an answer between grid levels rounds at random, up with probability equal to how far up it sits: the control for the six-op grammar, and the adopted recipe. The model learns the distribution of answers rather than a rounding, so exact match against a drawn answer sits at a ceiling the rule sets, and the expected exact match and the calibration are the measurements to use. Anchoring is unchanged.
///

## Observations

- **The models reach the ceiling a stochastic target sets.** Expected exact match on the held-out pairs sits on the holdout ceiling on `mix`, and within {gap:.2f} of it on `screen` and `multiply`, for the control and the recipe alike ([table](#exact-match-against-a-moving-target)). Exact match against the drawn answer adds the noise of a single draw, about ±{draw_se:.2f} over {int(_hs("control-stoch", "mix", "n"))} lines. Model seeds share that noise, so it does not average away.
- **The answer distribution is calibrated to the rule.** On the rounded held-out lines the control puts {pm_["mix"][0]:.2f} / {pm_["screen"][0]:.2f} / {pm_["multiply"][0]:.2f} of its mass on the mode for `mix` / `screen` / `multiply`, against {pm_["mix"][1]:.2f} / {pm_["screen"][1]:.2f} / {pm_["multiply"][1]:.2f} for the rule. KL from the rule[^kl] runs {min(kl_):.2f}–{max(kl_):.2f} nats on the stochastic arms, against {min(kl_near):.1f}–{max(kl_near):.1f} on the nearest-rounded control, which commits to one level. At least {supp:.3f} of the mass is on the candidate colors ([figure](#exact-match-against-a-moving-target)).
- **Exact match against the mode is an argmax read, and on `mix` a coin toss.** It is {mode_rest[0]:.2f} / {mode_rest[1]:.2f} on `screen` / `multiply`, and {mode_mix:.2f} on the rounded pairs of `mix`, where half-way ties make the mode arbitrary.
- **Anchoring does not notice the corpus.** The m_line of `recipe-stoch` is {m_line_stoch.mean():.3f} <span class='range'>±{(m_line_stoch.max() - m_line_stoch.min()) / 2:.3f}</span>, against {m_line_prod.mean():.3f} <span class='range'>±{(m_line_prod.max() - m_line_prod.min()) / 2:.3f}</span> in production. Every other placement statistic is inside the seed band of production ([table](#anchoring-on-the-stochastic-corpus)).
- **Suppression reads shift through their clean baseline.** On the nearest-rounded probe lines of `multiply`, the clean red-line accuracy of the recipe is {clean_mul:.2f}: the argmax of a calibrated model lands on the nearest level only where the mode is clear. The redder lines go {redder["stoch"][0]:.2f} → {redder["stoch"][1]:.2f} under `projection`, against {redder["prod"][0]:.2f} → {redder["prod"][1]:.2f} in production. On `mix`, whose probe lines are on-grid, nothing moves.
- **Redder-than-both counts rise on the ops that round up.** With no model in the loop, the expected count changes by {dredder["screen"]:+.1%} on `screen`, {dredder["multiply"]:+.1%} on `multiply`, and {dredder["mix"]:+.1%} on `mix` ([table](#the-grammar-under-each-rounding)).
- **No clearer gradedness at the answer, on probes that could not show it.** Peak strict R² of the answer RGB over the slices is {r2_answer["control-stoch"][0]:.2f} at `=` and {r2_answer["control-stoch"][1]:.2f} at the answer on the stochastic control, against {r2_answer["control-near"][0]:.2f} and {r2_answer["control-near"][1]:.2f} on the nearest one. At the deep slices, the two nearest-rounded seeds differ from each other by more than the corpora differ. The probe lines come from `mix`, which never rounds ([figure](#the-answers-geometry)).

[^kl]: Kullback–Leibler divergence, in nats: how much surprise you take on average by predicting with the model's distribution when the rule's is the truth. Zero means the two match.
"""

rf"""
## Why, and what we ran

Ex-2.2.3's grammar computes each answer channel on the 0..15 scale and snaps it to the nearer grid level, so `screen`, `multiply`, and (off its on-grid pairs) `mix` are step functions of their raw value. The [stochastic-rounding item](https://github.com/z0u/sca2/blob/main/todo/science/stochastic-rounding-of-off-grid-answers.md) asks what a variant that rounds by coin flip would settle: (a) the exact-match ceiling, and whether exact match is then the wrong statistic; (b) whether the answer's representation becomes more graded; (c) whether the redder-than-both counts, and so E3, move.

The variant is `sca.data.ops.Rounding = "stochastic"`: per channel and per line, the upper neighbour is drawn with probability equal to how far up the interval the raw value sits, so the mean target is the raw value itself and a tie is a coin flip. The (op, pair) sequence of the corpus and the eval lines are the production ones at the same seed; only the answers of rounded lines differ. Three arms, all at the adopted point's length ({ex.EPOCHS} epochs), all under ex-2.2.3's either-slot labeller:

| condition | seeds | corpus | anchor |
| --- | ---: | --- | --- |
{chr(10).join(f"| `{c.name}` | {c.seeds} | {ex.ROUNDING[c.name]} | {'λ = ' + str(c.lam) if c.lam else 'none'} — {c.title} |" for c in ex.CONDITIONS)}

`control-near` is two fresh seeds of production's `control-short`, trained here so the rounding readout and the cube probes have a same-code nearest-rounded comparison. The anchored arm reads against production's `recipe-short` at twenty seeds. Every task but the corpus build and the rounding readout is ex-2.2.3's code, loaded from its module unchanged, so the placement statistics mean what they meant there.
"""

r"""
## The grammar under each rounding

With no model in the loop: how much of each op rounds, the exact-match ceiling a stochastic target imposes (the mean over unordered pairs of the mode's probability, which is the most a predictor that knows the rule can match a drawn answer), and the redder-than-both count of each op's lines, where under stochastic rounding an answer counts with its probability.
"""

grammar_rows = [
    [
        f"<code>{op}</code>",
        f"{g['rounded_frac']:.1%}",
        f"{g['ceiling']:.3f}",
        f"{g['ceiling_rounded']:.3f}",
        f"{g['redder_nearest']:,}",
        f"{g['redder_stochastic']:,.0f}",
        f"{g['redder_stochastic'] / g['redder_nearest'] - 1:+.1%}" if g["redder_nearest"] else "—",
    ]
    for op, g in grammar.items()
]
grammar_head = [
    "op",
    "pairs that round",
    "EM ceiling",
    "ceiling on rounded pairs",
    "redder (nearest)",
    "redder (stochastic, expected)",
    "change",
]
table_html(
    grammar_head,
    grammar_rows,
    "The six ops under the two rounding rules. The ceiling is 1 on the three ops that never round; on the other three it is the mean mode probability over all unordered pairs, and over the pairs that round. Redder-than-both counts are over each op's 46,656 lines.",
)

r"""
## Exact match against a moving target

Ex-2.2.3 reads holdout exact match against the corpus answer. On a stochastic corpus that answer is one draw, so the statistic has a ceiling below 1 on the rounded ops however well the model knows the rule. Three readings of the same eval pass, all on held-out pairs: exact match against the drawn answer (what the production statistic is), against the rule's mode (the nearest answer, which a stochastic model should still name), and the *expected* exact match, which is the chance a fresh draw of the line would match the model's guess and is what the drawn-answer statistic converges to over many draws.
"""

conds_ = [c.name for c in ex.CONDITIONS]
ops_ = list(ex.ex223.OP_NAMES)
x_ops = np.arange(len(ops_))
ceiling_ = np.array([res.metrics["grammar"][op]["ceiling"] for op in ops_])
em_ = {c: np.array([res.em(res.runs(c), op) for op in ops_]) for c in conds_}  # (ops, seeds)
mode_ = {c: np.array([res.round_stat(c, op, "holdout", "em_mode") for op in ops_]) for c in conds_}
exp_ = {c: np.array([res.round_stat(c, op, "holdout", "expected_em") for op in ops_]) for c in conds_}
prod_em = np.array([res.em(res.prod_runs("control-short"), op) for op in ops_])


@memo
@themed(
    name="em-readings",
    alt_text="""
        Three panels side by side, each with the six ops on the horizontal axis and accuracy from 0 to 1 on the vertical. Left: exact match against the drawn answer; the two stochastic-corpus arms sit near a short black ceiling tick on mix, screen and multiply and near 1 elsewhere, while the nearest-rounded arms sit near 1 on every op. Middle: exact match against the rule's mode, where every arm sits near 1. Right: expected exact match, where the stochastic arms again meet the ceiling ticks.
    """,
    caption="""
        **Three readings of holdout accuracy, per op and condition.** Each mark is one seed. Left: exact match against the corpus answer, the production statistic. The black tick is the stochastic ceiling of the op (its mean mode probability); pale marks are production's `control-short`, on the nearest-rounded corpus. Middle: exact match against the rule's mode, the nearest answer. Right: the exact match a model would score on average over fresh draws of each line, which is what the left panel estimates with one draw.
    """,
)
def em_plot() -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6), sharey=True, layout="constrained")
    axes = cast(AxesRow, axes)
    off = np.linspace(-0.27, 0.27, len(conds_))
    for ax, data, title in zip(
        axes, (em_, mode_, exp_), ("vs the drawn answer", "vs the mode", "expected over draws"), strict=True
    ):
        for i, c in enumerate(conds_):
            v = data[c]
            for s in range(v.shape[1]):
                ax.plot(x_ops + off[i], v[:, s], "o", ms=3.5, color=ink(c), alpha=0.85, label=c if s == 0 else None)
        if data is em_:
            for s in range(prod_em.shape[1]):
                ax.plot(x_ops - 0.4, prod_em[:, s], "o", ms=3, color=ink("control-short"), alpha=0.7)
        for xi, cl in zip(x_ops, ceiling_, strict=True):
            ax.hlines(cl, xi - 0.42, xi + 0.42, color=light_dark("#000", "#fff"), lw=1.2)
        ax.set_xticks(x_ops, ops_, rotation=30, ha="right")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylim(0.3, 1.03)
    axes[0].set_ylabel("holdout accuracy")
    axes[0].legend(fontsize=7, loc="lower left", frameon=False)
    return fig


em_plot()

# %%

em_rows = []
for c in conds_:
    for op in ROUNDED_OPS:
        em_rows.append(
            [
                f"<code>{c}</code>",
                f"<code>{op}</code>",
                f"{res.round_stat(c, op, 'holdout', 'ceiling').mean():.3f}",
                span2(res.em(res.runs(c), op)),
                span2(res.round_stat(c, op, "holdout", "expected_em")),
                span2(res.round_stat(c, op, "holdout", "em_mode")),
                span2(res.round_stat(c, op, "holdout", "em_mode_rounded")),
            ]
        )
em_head = [
    "condition",
    "op",
    "holdout ceiling",
    "EM vs drawn",
    "expected EM",
    "EM vs mode",
    "EM vs mode, rounded pairs",
]
table_html(
    em_head,
    em_rows,
    "The three readings on the ops that round, held-out pairs. Seed means with half the seed range. The last column restricts the mode reading to the pairs that round at all, where the two corpora differ.",
)

r"""
## Where the answer mass goes

A model trained on a stochastic target has a reason to spread its answer probability over the two (or up to eight) candidate colors of a line in proportion to their probabilities, which is what "reads the effective geometry rather than the snap" meant in the item. The read is on the rounded eval lines only, pooled over ops and splits: the model's probability on the mode against the mode's true probability (a calibrated model sits on the diagonal; a nearest-trained one sits near 1 whatever the ceiling), and the model's total mass on the line's candidate set (near 1 for any model that has learned the rule, whichever way it rounds).
"""

edges_ = np.linspace(0.1, 1.0, 10)
binned: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
for c in conds_:
    t_ = res.lines(c)
    r_ = t_["rounded"]
    b_ = np.clip(np.digitize(t_["ceiling"][r_], edges_) - 1, 0, len(edges_) - 2)
    xs, pm, sup, n = [], [], [], []
    for k in range(len(edges_) - 1):
        m = b_ == k
        if m.sum() >= 20:
            xs.append(t_["ceiling"][r_][m].mean())
            pm.append(t_["p_mode"][r_][m].mean())
            sup.append(t_["support"][r_][m].mean())
            n.append(m.sum())
    binned[c] = (np.array(xs), np.array(pm), np.array(sup), np.array(n))


@memo
@themed(
    name="calibration",
    alt_text="""
        Two panels. Left: the model's probability on the mode answer against the mode's true probability, from about 0.15 to 1, with a dashed diagonal. The stochastic-corpus arms run close to the diagonal; the nearest-rounded arm stays near the top whatever the true probability. Right: the model's total mass on the candidate colors against the same axis; every arm sits near 1.
    """,
    caption="""
        **Calibration of the answer distribution on rounded lines.** Lines binned by the true probability of their mode answer (the line's exact-match ceiling), pooled over ops, splits and seeds; each mark is a bin with at least twenty lines, placed at the bin's mean ceiling. Left: mean model probability on the mode; the dashed line is perfect calibration. Right: mean model mass on the line's candidate set, the colors the stochastic rule can produce.
    """,
)
def calibration_plot() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), layout="constrained")
    axes = cast(AxesRow, axes)
    axes[0].plot([0, 1], [0, 1], "--", lw=0.8, color=light_dark("#888", "#aaa"))
    for c in conds_:
        xs, pm, sup, _ = binned[c]
        axes[0].plot(xs, pm, "o-", ms=3.5, lw=1, color=ink(c), label=c)
        axes[1].plot(xs, sup, "o-", ms=3.5, lw=1, color=ink(c))
    axes[0].set(xlabel="P(mode) under the rule", ylabel="model P(mode)", xlim=(0, 1.02), ylim=(0, 1.02))
    axes[1].set(
        xlabel="P(mode) under the rule", ylabel="model mass on the candidates", xlim=(0, 1.02), ylim=(0.5, 1.02)
    )
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    return fig


calibration_plot()

# %%

kl_rows = []
for c in conds_:
    for op in ROUNDED_OPS:
        kl_rows.append(
            [
                f"<code>{c}</code>",
                f"<code>{op}</code>",
                span2(res.round_stat(c, op, "holdout", "kl_rounded")),
                span2(res.round_stat(c, op, "holdout", "p_mode_rounded")),
                span2(res.round_stat(c, op, "holdout", "ceiling_rounded")),
                span2(res.round_stat(c, op, "holdout", "support_rounded")),
            ]
        )
kl_head = ["condition", "op", "KL(rule ‖ model) ↓", "model P(mode)", "rule P(mode)", "mass on candidates ↑"]
table_html(
    kl_head,
    kl_rows,
    "The same readings as means over the rounded held-out lines of each op. KL is from the rule's answer distribution to the model's, over the palette; it is zero for a model that reproduces the rule's spread and grows as the model commits to one level.",
)

r"""
## Anchoring on the stochastic corpus

The labeller and the anchor are ex-2.2.3's; only the corpus changed. The placement statistics of the anchored arm against production's `recipe-short` at twenty seeds, on the `mix` probe lines, plus the two task-cost reads: holdout exact match on `mix` (whose held-out lines include rounded pairs) and against the mode.
"""

prod_runs_ = res.prod_runs("recipe-short")
stoch_runs = res.runs("recipe-stoch")
anchor_stats = [
    ("m_line", "m_line ↑"),
    ("r2_sim", "r² grading ↑"),
    ("contrast", "contrast ↑"),
    ("lead_emb", "lead ↓"),
    ("latch_pi", "latch π ↓"),
    ("alpha_op1", "ᾱ op1 ↓"),
    ("retention", "retention ↑"),
]
anchor_rows = [
    ["<code>recipe-stoch</code>", str(len(stoch_runs))]
    + [span2(res.stat(stoch_runs, k)) for k, _ in anchor_stats]
    + [span2(res.em(stoch_runs, "mix"))],
    ["<code>recipe-short</code> (production)", str(len(prod_runs_))]
    + [span2(res.stat(prod_runs_, k)) for k, _ in anchor_stats]
    + [span2(res.em(prod_runs_, "mix"))],
]
anchor_head = ["condition", "seeds", *[t for _, t in anchor_stats], "holdout EM mix"]
table_html(
    anchor_head,
    anchor_rows,
    "Ex-2.2.3's gated statistics on the recipe, stochastic corpus against production. Arrows are the direction the gates read as good. Seed means with half the seed range.",
    ref_rows=frozenset({1}),
)

# %%

suppression_rows = []
for name, sc in (
    ("recipe-stoch", res.scored("recipe-stoch")),
    ("recipe-short (production)", res.prod_scored("recipe-short")),
):
    for op in ("mix", "multiply"):
        suppression_rows.append(
            [
                f"<code>{name}</code>",
                f"<code>{op}</code>",
                span2(res.score(sc, op, None, "acc", "red"), ".2f"),
                span2(res.score(sc, op, projection_iv, "acc", "red"), ".2f"),
                span2(
                    res.score(sc, op, None, "acc", "nonred") - res.score(sc, op, projection_iv, "acc", "nonred"), ".3f"
                ),
                span2(res.score(sc, op, None, "acc", "redder"), ".2f"),
                span2(res.score(sc, op, projection_iv, "acc", "redder"), ".2f"),
            ]
        )
suppression_head = [
    "condition",
    "op",
    "red acc clean",
    "red acc projected ↓",
    "non-red deficit ↓",
    "redder acc clean",
    "redder acc projected",
]
table_html(
    suppression_head,
    suppression_rows,
    "The eval contract's suppression reads under the full projection, on the probe lines (nearest-rounded, as production's). Red lines have dose ≥ 0.8; non-red ≤ 0.2; redder lines have an answer redder than both operands.",
    ref_rows=frozenset({2, 3}),
)

r"""
## The answer's geometry

Question (b) of the item: does a stochastic target make the answer's representation more graded? The cube probes are ex-2.2.3's: strict-holdout ridge R² of each slot's RGB at every (slice, position) site, on `mix`'s on-grid probe lines, where no rounding ever happens. The read is the answer's own RGB at `=` (the prediction site) and at the answer position, un-anchored control on each corpus, with production's full-length `control` pale.
"""

r2_stoch, r2_near, r2_prod = res.r2("control-stoch"), res.r2("control-near"), res.r2("control", prod=True)
answer_target = 2  # the answer target
x_slices = np.arange(len(SLICE_NAMES))


@memo
@themed(
    name="answer-r2",
    alt_text="""
        Two panels, one for the equals position and one for the answer position, each with the five residual-stream slices on the horizontal axis and strict-holdout R-squared from 0 to 1 on the vertical. In the equals panel the stochastic-corpus control sits at or above the nearest-rounded control at every slice, peaking near 0.9 at slice 2. In the answer panel the two overlap through slice 2 and fall together after it, with the two nearest-rounded seeds far apart at the deepest slice. Production's full-length control is pale beside them, lower in both panels.
    """,
    caption="""
        **Strict-holdout R² of the answer's RGB, by slice, at `=` and at the answer position.** Mean over the three channels; one line per seed. Production's `control` ran twice as long as the two pilot arms.
    """,
)
def answer_r2_plot() -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True, layout="constrained")
    axes = cast(AxesRow, axes)
    for ax, pos in zip(axes, (3, 4), strict=True):
        for s in r2_prod:
            ax.plot(x_slices, s[:, pos, answer_target].mean(-1), "-", lw=1, color=ink("control"), alpha=0.8)
        for name, arr in (("control-near", r2_near), ("control-stoch", r2_stoch)):
            for i, s in enumerate(arr):
                ax.plot(
                    x_slices,
                    s[:, pos, answer_target].mean(-1),
                    "o-",
                    ms=3,
                    lw=1.2,
                    color=ink(name),
                    label=name if i == 0 else None,
                )
        ax.set_xticks(x_slices, SLICE_NAMES)
        ax.set_title(f"answer RGB at {POS_NAMES[pos]}", fontsize=9)
    axes[0].set(ylabel="strict R²", ylim=(0, 1.02))
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    return fig


answer_r2_plot()

r"""
## What we make of it

A model trained on a coin-flip target learns the coin. Its answer distribution mirrors the one the rule defines, so the ceiling the item asked about is a fact about the corpus, and the model sits on it. That settles question (a).

Exact match against the drawn answer is the wrong statistic twice over: it is capped, and it carries the noise of a single draw that every seed shares. Read instead the expected exact match against the holdout ceiling, the calibration of the answer mass (P(mode) against the same quantity for the rule, or the KL), and, on ops without ties, exact match against the mode. The off-grid channels of `mix` are always half-way ties, so there the mode read is a coin toss and the calibration read is the one to keep.

The corpus changes nothing on the prompt side. The labeller reads operands, the anchor pulls the prompt span, and the placement statistics on the stochastic corpus match production to the third decimal.

The answer side does move. Every clean accuracy on a rounded line falls to the level a calibrated argmax can reach, and the suppression reads follow it down. They are still a deficit measured from clean, which the contract already handles, and the redder lines go on holding their answers under `projection` at the same ratio as production.

Redder-than-both counts on `screen` and `multiply` rise by a fifth, because a line that would round down under nearest rounding sometimes rounds up. An E3 on those ops would have more lines to read and the same finding.

Question (b), whether the representation of the answer becomes more graded, is still open. The cube probes come from ex-2.2.3, which runs them on the on-grid lines of `mix`, where nothing rounds. A pilot that could answer it would probe the rounded lines of `screen` or `multiply` and compare the spread of the answer distribution against the raw value: a graded representation would show up as a smooth function of the odds of the coin.

For D2.2 we would keep nearest rounding: it keeps every accuracy read simple, and the anchored-op experiments read the prompt side, where the corpus makes no difference. What the variant adds, a calibrated answer distribution, is not something those experiments ask about. It earns a place when a question is about the representation of the answer itself, and then it comes with the reads above and probes on the lines that round.

## Method notes

- **Corpus.** `sca.data.ops.sample_corpus(..., rounding="stochastic")` and `eval_sets(..., rounding="stochastic")`: the (op, pair) sequence and the eval lines are the production ones at seed 0; each rounded line's answer is one draw from its own stream. The probe lines keep the nearest answer: they read the prompt's geometry, and on `mix` they are on-grid lines that never round.
- **Runs.** The d64-L4 nGPT and the ex-2.2.3 training step, fifty epochs (1,650 steps), the either-slot labeller at the per-slot rate; the recipe is λ = 0.1, τ = 0.1, anti-subspace 2.5 → 0.3 by 90%, annealed anchor. Seeds 0–1 (controls) and 0–2 (recipe).
- **Rounding readout.** Per eval line, the rule's answer distribution is the product of its per-channel two-point distributions (`answer_dist`); the ceiling is the mode's probability (`mode_prob`). The model's distribution is its softmax at the pre-answer position restricted to the 216 palette tokens, renormalized for the KL only.
- **Everything else** is ex-2.2.3's: the eval pass, the eval contract's scorer, and the cube probes, run from that module's functions on this pilot's checkpoints.
"""
