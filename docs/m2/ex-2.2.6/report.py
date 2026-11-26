# title: Ex 2.2.6: a pilot of the whole-span labeller

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

# The document's directory is on sys.path while it runs, so the sibling module comes
# from the definition module beside this one.
import experiment as ex
from mini.lit import memo, read_npz, stop
from mini.store import project_store
from mini.vis import AxesGrid, AxesRow, figure_html, light_dark, smooth_step, smooth_step_area, themed
from sca.anchoring import softmin_weights

ROLES = ["op1", "op", "op2", "=", "ans", "⏎"]  # the six roles of a line; ex-2.2.3's profiles stop at `=`
ARMS = [a.name for a in ex.ARMS] + [ex.REFERENCE]
KEYING = {a.name: a.keying for a in ex.ARMS} | {ex.REFERENCE: "either"}
SPAN = {a.name: a.span for a in ex.ARMS} | {ex.REFERENCE: ex.PROMPT}
MIX = ex.ex223.PRIMARY_OP.name
PROJECTION = ex.ex223.PROJECTION.name
# The ops with lines whose answer is redder than both operands; `lighten` has none.
E3_OPS = [op for op in ex.ex223.OP_NAMES if ex.ex223.line_counts(ex.ex223.OP_BY_NAME[op])["redder"] > 0]
DOSE_BINS = np.linspace(0.0, 1.0, 11)

# One ink per arm, as (light, dark) pairs; the production corner draws grey.
INK = {
    "line-whole": ("#1f6fb4", "#5fa8dd"),
    "line-prompt": ("#1a8f7a", "#4fc3ac"),
    "either-whole": ("#d0461b", "#f07a50"),
    "recipe-short": ("#9a9a9a", "#7a7a7a"),
}


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many` for the lot."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict | None:
    return None if path is None else json.loads(path.read_text())


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
    table = f'<div class="report-table-scroll"><table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table></div>'
    return figure_html(table, caption=caption, class_="report-figure")


def matched_diff(v: np.ndarray, redder: np.ndarray, dose: np.ndarray) -> float:
    """Ex-2.2.3's E3 contrast: the mean over dose bins, weighted by the redder count, of (redder mean − other
    mean) of a per-line value.
    """
    b = np.clip(np.digitize(dose, DOSE_BINS) - 1, 0, len(DOSE_BINS) - 2)
    num = den = 0.0
    for k in np.unique(b[redder]):
        a, o = v[redder & (b == k)], v[~redder & (b == k)]
        if len(o):
            num += len(a) * (a.mean() - o.mean())
            den += len(a)
    return num / den if den else float("nan")


def matched_weights(redder: np.ndarray, dose: np.ndarray) -> np.ndarray:
    """Per-line weights that give the non-redder lines the redder lines' dose histogram."""
    b = np.clip(np.digitize(dose, DOSE_BINS) - 1, 0, len(DOSE_BINS) - 2)
    wt = np.zeros(len(dose))
    for k in np.unique(b[redder]):
        m = ~redder & (b == k)
        if m.any():
            wt[m] = (redder & (b == k)).sum() / m.sum()
    return wt / wt.sum()


@dataclass(frozen=True)
class Results:
    """The pilot's three arms and the production corner they complete, with one accessor per shape the cells need.

    `probes` is the line-keyed probe table: `{op}/line_p` is P(labelled) when the answer draws too and
    `{op}/line_p_either` the operand-only version, on the same probe lines every arm (production included) was
    measured on.
    """

    metrics: dict
    arrays: Mapping[str, np.ndarray]
    probes: Mapping[str, np.ndarray]
    prod: dict
    prod_arrays: Mapping[str, np.ndarray]

    def is_prod(self, cond: str) -> bool:
        return cond == ex.REFERENCE

    def runs(self, cond: str) -> list[dict]:
        """The eval records of an arm in seed order; the production corner pools its addendum seeds."""
        if self.is_prod(cond):
            names = (cond, f"{cond}-more")
            return sorted((r for r in self.prod["runs"] if r["condition"] in names), key=lambda r: r["seed"])
        return sorted((r for r in self.metrics["runs"] if r["condition"] == cond), key=lambda r: r["seed"])

    def scored(self, cond: str) -> list[dict]:
        if self.is_prod(cond):
            names = (cond, f"{cond}-more")
            return sorted((r for r in self.prod["scores"] if r["condition"] in names), key=lambda r: r["seed"])
        return sorted((r for r in self.metrics["scores"] if r["condition"] == cond), key=lambda r: r["seed"])

    def arr(self, cond: str, label: str, name: str) -> np.ndarray:
        src = self.prod_arrays if self.is_prod(cond) else self.arrays
        return src[f"{label}/eval/{name}"]

    def stat(self, cond: str, key: str, op: str | None = None) -> np.ndarray:
        return np.array([r[key] if op is None else r["per_op"][op][key] for r in self.runs(cond)], float)

    def em(self, cond: str, op: str) -> np.ndarray:
        return np.array([r["holdout_em"][op] for r in self.runs(cond)], float)

    def score(self, cond: str, op: str, iv: str | None, key: str, group: str) -> np.ndarray:
        out = []
        for r in self.scored(cond):
            s = r["ops"][op]
            v = s["clean"][key] if iv is None else s["interventions"][iv][key]
            out.append(v[group])
        return np.array(out, float)

    def deficit(self, cond: str, op: str, iv: str, group: str = "nonred") -> np.ndarray:
        return self.score(cond, op, None, "acc", group) - self.score(cond, op, iv, "acc", group)

    def line_w(self, op: str, keying: str) -> np.ndarray:
        """Normalized P(labelled) per probe line under one labeller."""
        p = self.probes[f"{op}/line_p" if keying == "line" else f"{op}/line_p_either"]
        return p / p.sum()

    def alpha_lines(self, cond: str) -> list[tuple[dict, np.ndarray]]:
        """Per seed, the eval record and its (slices, N, T) alignment map on the `mix` probe lines."""
        return [(r, self.arr(cond, r["label"], f"{MIX}/alpha_lines").astype(np.float32)) for r in self.runs(cond)]

    def m_line(self, cond: str, keying: str, roles: int) -> np.ndarray:
        """m_line recomputed per seed: the labeller's line weighting, and the margin maxed over `roles` roles."""
        w = self.line_w(MIX, keying)
        out = []
        for _, cos in self.alpha_lines(cond):
            m = np.einsum("n,lnt->lt", w, cos) - cos.mean(axis=1)
            out.append(m[:, :roles].max(axis=1).mean())
        return np.array(out, float)

    def profile(self, cond: str, keying: str) -> np.ndarray:
        """(seeds, slices, 6) softmin profile at each run's τ over all six roles, weighted by P(labelled)."""
        w = self.line_w(MIX, keying)
        return np.stack(
            [np.einsum("lnt,n->lt", softmin_weights(1.0 - cos, r["tau"]), w) for r, cos in self.alpha_lines(cond)]
        )

    def e3(self, cond: str, op: str) -> dict[str, np.ndarray]:
        """Per seed on one op: Δα at `=`, the answer and the newline (deep slices, dose-matched), and the deep-slice
        six-role softmin profiles of the redder lines and of their dose-matched comparison lines.
        """
        d_eq, d_ans, d_nl, prof_r, prof_m = [], [], [], [], []
        for r in self.runs(cond):
            cos = self.arr(cond, r["label"], f"{op}/alpha_lines").astype(np.float32)
            redder = self.arr(cond, r["label"], f"{op}/redder")
            dose = self.arr(cond, r["label"], f"{op}/dose")
            deep = cos[1:].mean(axis=0)
            d_eq.append(matched_diff(deep[:, 3], redder, dose))
            d_ans.append(matched_diff(deep[:, ex.ANSWER_POS], redder, dose))
            d_nl.append(matched_diff(deep[:, 5], redder, dose))
            w = softmin_weights(1.0 - cos[1:], r["tau"]).mean(axis=0)
            prof_r.append(w[redder].mean(axis=0))
            prof_m.append(matched_weights(redder, dose) @ w)
        return {
            "eq": np.array(d_eq),
            "ans": np.array(d_ans),
            "nl": np.array(d_nl),
            "prof_redder": np.stack(prof_r),
            "prof_matched": np.stack(prof_m),
        }


with tempfile.TemporaryDirectory() as _tmp:
    files = fetch([ex.METRICS_REF, ex.ARRAYS_REF, ex.PROBE_REF, ex.EX223_METRICS_REF, ex.EX223_ARRAYS_REF], Path(_tmp))
    metrics = read_json(files[ex.METRICS_REF])
    arrays = read_npz(files[ex.ARRAYS_REF])
    probes = read_npz(files[ex.PROBE_REF])
    prod = read_json(files[ex.EX223_METRICS_REF])
    prod_arrays = read_npz(files[ex.EX223_ARRAYS_REF])
if metrics is None or arrays is None or probes is None:
    stop("No results published yet. Run the experiment (see `experiment.py`) and re-open this report.")
if prod is None or prod_arrays is None:
    stop("Ex-2.2.3's production results are not reachable.")
res: Results = Results(metrics, arrays, probes, prod, prod_arrays)

e3: dict[str, dict[str, dict[str, np.ndarray]]] = {c: {op: res.e3(c, op) for op in E3_OPS} for c in ARMS}

ops = list(ex.ex223.OP_NAMES)
em_floor = min(res.em(c, op).mean() for c in ARMS for op in ops)
ml = {c: res.stat(c, "m_line") for c in ARMS}
pilot_ml = [ml[c].mean() for c in ARMS if c != ex.REFERENCE]
ref_ml = ml[ex.REFERENCE]
ans_w = {c: float(res.profile(c, KEYING[c])[:, 1:, 4].mean()) for c in ARMS}
d_ans = {c: e3[c]["multiply"]["ans"] for c in ARMS}
proj_mul = [res.score(c, "multiply", PROJECTION, "acc", "redder").mean() for c in ARMS]
proj_mix = [res.score(c, MIX, PROJECTION, "acc", "redder").mean() for c in ARMS]
deficit = [res.deficit(c, MIX, PROJECTION).mean() for c in ARMS]

rf"""
# Ex 2.2.6: a pilot of the whole-span labeller

/// tip |
<!-- tl;dr -->
The labeller of ex-2.2.3 reads only the operands and pulls only the prompt, so on a line whose answer is the reddest thing in it, nothing pulls the answer position. We retrained under labellers that also read the answer, also pull the whole line, or both. Pulling the whole line doubles the alignment at the answers of the redder lines, at no cost to the task or placement, and reading the answer changes nothing visible. Neither makes the answers depend on the axis, because the answer is read out at `=`, one position before the pull lands.
///

## Observations

- **No task cost.** Held-out exact match is at least {em_floor:.3f} on every op of every arm, the same level as the twenty production seeds ([table](#task-cost)).
- **Placement stays in the production band.** Under the labeller of each arm, m_line runs {min(pilot_ml):.3f}–{max(pilot_ml):.3f}, against {ref_ml.mean():.3f} <span class='range'>±{(ref_ml.max() - ref_ml.min()) / 2:.3f}</span> on `{ex.REFERENCE}`. Lead, contrast, grading, latch and retention behave the same way ([table](#where-the-pull-lands)).
- **The span moves the pull; the keying does not.** With a whole-line pull, the softmin weight on the answer role, averaged over the four residual slices, is {ans_w["line-whole"]:.2f} on `line-whole` and {ans_w["either-whole"]:.2f} on `either-whole`, against {ans_w[ex.REFERENCE]:.2f} on production. `line-prompt` pulls only the prompt span, and sits at {ans_w["line-prompt"]:.2f} with the production profile ([figure](#where-the-pull-lands)).
- **The redder lines carry more redness at the answer under the whole-line pull.** On the redder lines of `multiply`, Δα at the answer is {d_ans["line-whole"].mean():.2f} and {d_ans["either-whole"].mean():.2f} on the two whole-line arms, {d_ans["line-prompt"].mean():.2f} on `line-prompt`, and {d_ans[ex.REFERENCE].mean():.2f} <span class='range'>±{(d_ans[ex.REFERENCE].max() - d_ans[ex.REFERENCE].min()) / 2:.2f}</span> on production. At `=` and at the newline, nothing moves ([figure](#the-redder-than-both-lines)).
- **Projection still leaves those answers intact on every arm.** Across the four arms, redder-line accuracy under `projection` is {min(proj_mul):.2f}–{max(proj_mul):.2f} on `multiply` and {min(proj_mix):.2f}–{max(proj_mix):.2f} on `mix`. Red-line removal and the non-red `mix` deficit ({min(deficit):.3f}–{max(deficit):.3f}) are unchanged within the noise of two or three seeds ([table](#suppression-and-selectivity)).
"""

rf"""
## Why, and what we ran

A document-level label, the M3 shape, says a document is about *red* and nothing about where. Ex-2.2.3's labeller is closer to a token label: each operand draws at its own redness rate, and the pull covers the four prompt roles. Its E3 section found the blind span that implies: on lines whose answer is redder than both operands the stream carries the answer's extra redness at the answer position, the labeller never keys there, and under `projection` those lines keep their answers. The [labelling-span item](https://github.com/z0u/sca2/blob/main/todo/science/labeling-pull-span-variants-ex-2-1.md) files the fix as a fast-follow; this pilot is that follow.

Two changes, each one step toward the document shape. **Keying** `line` (`sca.anchoring.LabelSpec`): the answer slot draws at its redness rate too, so P(labelled) becomes 1 − (1 − p₁)(1 − p₂)(1 − p₃) and the redder lines draw more often. **Span** 6: the pull covers the answer and the newline as well as the prompt. The anchor term is the per-line mellowmax over the pulled span with a conserved per-line budget, so the wider span gives the softmin more places to put the same pull and changes nothing else about its strength. Everything else is ex-2.2.3's `recipe-short` (λ = {ex.ex223.SCORING_LAMBDA:g}, {ex.EPOCHS} epochs), on the same corpus and the same probe lines, and the production seeds of that arm are the fourth corner:

| arm | seeds | the answer draws | pull covers | |
| --- | ---: | :---: | --- | --- |
{chr(10).join(f"| `{a.name}` | {a.seeds} | {'yes' if a.keying == 'line' else 'no'} | {'whole line' if a.span == ex.WHOLE else 'prompt span'} | {a.title} |" for a in ex.ARMS)}
| `{ex.REFERENCE}` | 20 | no | prompt span | production, ex-2.2.3's adopted point |

Every statistic below is measured on the same probe lines the production arm was measured on. Where a statistic weights lines by P(labelled), the table says which labeller's weighting it uses, since the two differ on the redder lines by construction.
"""

r"""
## Task cost

Exact match on the held-out pairs of each op. The reference is production's twenty seeds; the pilot arms have two or three each, so the half-ranges are rough.
"""

ops = list(ex.ex223.OP_NAMES)
rows = [
    [f"<code>{c}</code>", f"{len(res.runs(c))}"]
    + [span2(res.em(c, op), ".3f") for op in ops]
    + [f"{res.em(c, MIX).mean() - res.em(ex.REFERENCE, MIX).mean():+.3f}"]
    for c in ARMS
]
caption = "Held-out exact match per op, seed mean with half the seed range; the last column is the `mix` gap from the production corner. Ex-2.2.3's task gate was a `mix` gap within 0.02 of its own control."
table_html(
    ["arm", "seeds"] + [f"<code>{op}</code>" for op in ops] + ["Δ mix"],
    rows,
    caption,
    ref_rows=frozenset({len(ARMS) - 1}),
)

r"""
## Where the pull lands

Ex-2.2.3's placement statistics on the `mix` probe lines, then m_line recomputed four ways: over the four prompt roles or all six, weighting probe lines by the operand-only labeller's P(labelled) or by the line-keyed one. A stored m_line is the four-role margin under the arm's own labeller.
"""

rows = []
for c in ARMS:
    rows.append(
        [
            f"<code>{c}</code>",
            span2(res.stat(c, "m_line")),
            span2(res.m_line(c, "either", 4)),
            span2(res.m_line(c, "line", 4)),
            span2(res.m_line(c, "either", 6)),
            span2(res.m_line(c, "line", 6)),
            span2(res.stat(c, "alpha_op1")),
            span2(res.stat(c, "lead_emb"), ".2f"),
            span2(res.stat(c, "contrast"), ".2f"),
            span2(res.stat(c, "r2_sim")),
            span2(res.stat(c, "latch_pi"), ".2f"),
            span2(res.stat(c, "retention"), ".2f"),
        ]
    )
head = [
    "arm",
    "m_line (stored)",
    "m_line 4 · either",
    "m_line 4 · line",
    "m_line 6 · either",
    "m_line 6 · line",
    "ᾱ op1",
    "lead (emb)",
    "contrast",
    "r² sim",
    "latch π",
    "retention",
]
caption = """
Placement on the <code>mix</code> probe lines, seed means with half the seed range. The four recomputed m_line columns read the same alignment maps with the role count and the line weighting varied; a six-role margin can only match or exceed the four-role one on the same weighting. ᾱ op1 is the mean alignment at op1 over every slice and color (ex-2.2.3's containment read); lead is the G1 group's softmin weight on op1 at the embedding; contrast is the deep-slice op2 weight of G2 minus G1; r² sim the grading of the op1 response against the similarity target; latch π the larger of the non-red group's deep-slice weights on the op word and on <code>=</code>; retention the final m_line over its running peak.
"""
table_html(head, rows, caption, ref_rows=frozenset({len(ARMS) - 1}))

# %%

prof = {c: res.profile(c, KEYING[c]) for c in ARMS}
x = np.arange(6)


@memo
@themed(
    name="profiles",
    alt_text="""
        A grid of small panels, one column per arm (line-whole, line-prompt, either-whole, and production's recipe-short), five residual slices per column with the embedding at the bottom, each spanning the six roles op1, op, op2, equals, answer, newline. The shaded smooth-step is the seed-mean softmin weight and a hairline follows each seed.
    """,
    caption=rf"""
        **Softmin profiles over all six roles, per arm, on the `mix` probe lines.** Columns are arms, rows residual slices with the embedding at the bottom; each panel is the seed-mean softmin weight at the arm's τ, over the six roles, with lines weighted by the arm's own labeller's P(labelled) (production's operand-only weighting on `{ex.REFERENCE}` and `either-whole`); one hairline per seed. A dotted rule separates the prompt roles from the answer and the newline: ex-2.2.3's profiles stop at that rule, and a whole-line pull is free to cross it.
    """,
)
def profiles_plot() -> plt.Figure:
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    hair = light_dark("#00000055", "#ffffff55")
    rule = light_dark("#999", "#777")
    fig, axes = plt.subplots(5, len(ARMS), figsize=(5.2, 3.9), sharex=True, sharey=True)
    axes = cast(AxesGrid, axes)
    engine = fig.get_layout_engine()
    assert isinstance(engine, ConstrainedLayoutEngine)
    engine.set(hspace=0, h_pad=0.01, wspace=0.05)
    for col, c in enumerate(ARMS):
        color = ink(c)
        mean = prof[c].mean(axis=0)
        for row in range(5):
            sl = 4 - row
            ax = axes[row][col]
            smooth_step_area(ax, x, mean[sl], ramp=0.5, color=color, alpha=light_dark(0.22, 0.28))
            smooth_step(ax, x, mean[sl], ramp=0.5, color=color, lw=1.5)
            for sp in prof[c]:
                smooth_step(ax, x, sp[sl], ramp=0.5, color=hair, lw=0.5)
            ax.axvline(3.5, color=rule, lw=0.6, ls=":")
            ax.set(ylim=(-0.08, 1.08), xlim=(-0.4, 5.4), yticks=[0.0, 0.5, 1.0])
            ax.spines[:].set_visible(False)
            ax.grid(axis="y", which="major", c="#888", alpha=0.2)
            ax.tick_params(axis="x", length=0, labelbottom=False)
            ax.tick_params(axis="y", left=True, right=True, direction="in", labelleft=False, labelright=False)
            if col == 0:
                ax.set_ylabel("emb" if sl == 0 else f"{sl}", fontsize=7.5)
        axes[0][col].set_title(c, fontsize=8, color=color)
        axes[4][col].set_xticks(x, ROLES)
        axes[4][col].tick_params(axis="x", labelbottom=True, labelsize=7)
    axes[4][-1].tick_params(axis="y", labelright=True, labelsize=6.5, pad=2)
    return fig


profiles_plot()

r"""
## The redder-than-both lines

Ex-2.2.3's E3 read, repeated on every arm and extended to the newline. Δα is the deep-slice alignment on the lines whose answer is redder than both operands, minus the alignment on lines of the same op in the same dose bin whose answer is not, at one position. Under the operand-only labeller the answer position was blind; a labeller that reads the answer has a reason to put alignment there, and a whole-line pull has somewhere to put it.
"""

x = np.arange(len(E3_OPS))


@memo
@themed(
    name="e3-delta-alpha",
    alt_text="""
        Three panels side by side, for the equals sign, the answer, and the newline. In each, the ops with redder-than-both lines run along the x axis and delta alpha along the y axis, with one marker per arm at each op and a vertical bar for its seed range. A grey line at zero marks no difference from the dose-matched lines.
    """,
    caption=rf"""
        **Δα on the redder-than-both lines, by position and arm.** Each marker is an arm's seed mean of Δα at one position, with the seed range as a bar, per op; the panels are `=`, the answer and the newline. A positive value says the stream is more aligned with the anchor on the redder lines than on dose-matched lines of the same op at that position. Ex-2.2.3 read the first two positions on `{ex.REFERENCE}` alone.
    """,
)
def e3_delta_alpha_plot() -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.0), sharey=True)
    axes = cast(AxesRow, axes)
    offs = np.linspace(-0.27, 0.27, len(ARMS))
    for ax, key, title in zip(axes, ("eq", "ans", "nl"), ("at =", "at the answer", "at ⏎"), strict=True):
        ax.axhline(0, color=light_dark("#999", "#777"), lw=0.6)
        for c, off in zip(ARMS, offs, strict=True):
            m = np.array([e3[c][op][key].mean() for op in E3_OPS])
            lo = np.array([e3[c][op][key].min() for op in E3_OPS])
            hi = np.array([e3[c][op][key].max() for op in E3_OPS])
            ax.errorbar(x + off, m, yerr=[m - lo, hi - m], fmt="o", ms=3, lw=0.8, color=ink(c), label=c)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(x, E3_OPS, fontsize=7, rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", c="#888", alpha=0.2)
    axes[0].set_ylabel("Δα (deep slices)", fontsize=8)
    axes[-1].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


e3_delta_alpha_plot()

# %%

n = int(res.runs(ex.REFERENCE)[0]["per_op"][MIX]["n_redder"])
rows = []
for c in ARMS:
    r_ = e3[c][MIX]
    rows.append(
        [
            f"<code>{c}</code>",
            span2(r_["eq"]),
            span2(r_["ans"]),
            span2(r_["nl"]),
            span2(r_["prof_redder"][:, 4], ".2f"),
            span2(r_["prof_matched"][:, 4], ".2f"),
            span2(res.score(c, MIX, None, "acc", "redder"), ".2f"),
            span2(res.score(c, MIX, PROJECTION, "acc", "redder"), ".2f"),
        ]
    )
head = [
    "arm",
    "Δα at =",
    "Δα at the answer",
    "Δα at ⏎",
    "answer weight, redder",
    "answer weight, matched",
    "clean acc",
    "projection acc ↓",
]
caption = f"""
The {n:,} redder-than-both <code>mix</code> probe lines, per arm. Δα as in the figure; the two weight columns are the deep-slice softmin weight on the answer role over all six roles, on the redder lines and on their dose-matched comparison lines. The right pair restricts H4 to these lines: exact-match accuracy clean and under <code>projection</code>. Under the operand-only labeller ex-2.2.3 found these lines keep their answers when the axis is projected out.
"""
table_html(head, rows, caption, ref_rows=frozenset({len(ARMS) - 1}))

r"""
## Suppression and selectivity

Ex-2.2.3's H4 statistics under `projection`, per arm: accuracy on the red lines of each op (the removal read, lower is more complete), the non-red deficit on `mix` (the selectivity read), and accuracy on the redder-than-both lines of each op that has them.
"""

ops = list(ex.ex223.OP_NAMES)
rows = []
for c in ARMS:
    red = [span2(res.score(c, op, PROJECTION, "acc", "red"), ".2f") for op in ops]
    redder = [span2(res.score(c, op, PROJECTION, "acc", "redder"), ".2f") if op in E3_OPS else "—" for op in ops]
    rows.append([f"<code>{c}</code>", "red"] + red + [span2(res.deficit(c, MIX, PROJECTION), ".3f")])
    rows.append(["", "redder"] + redder + [""])
head = ["arm", "lines"] + [f"<code>{op}</code>" for op in ops] + ["non-red deficit, mix"]
caption = f"""
Exact-match accuracy under <code>projection</code> per op, on the red lines (dose ≥ {ex.ex223.RED_DOSE:g}) and on the redder-than-both lines, with the non-red (dose ≤ {ex.ex223.NONRED_DOSE:g}) deficit on <code>mix</code> in the last column. Ex-2.2.3's gates were red accuracy at most {ex.ex223.RED_ACC_GATE:g} on every op and a deficit at most {ex.ex223.NONRED_DEFICIT_GATE:g}.
"""
table_html(head, rows, caption, ref_rows=frozenset({2 * len(ARMS) - 2, 2 * len(ARMS) - 1}))

r"""
## What we make of it

The blind span has two halves, and the pilot separates them: the keying (does a line whose answer is the reddest thing in it draw a label at all?) and the span (can the answer position be pulled once the line has drawn?). The profiles say the span is what moves the pull. With six roles the softmin puts between a tenth and a quarter of its weight on the answer at every depth, whichever slots draw, and answer-drawing keying with a prompt-span pull reproduces the production profile.

On the `mix` probe lines that is what we would expect the mellowmax to do: a labelled line has a red operand and so a reddish answer, and the answer position is about as cheap to align as the operand.

The extra alignment does not make the answer depend on the axis. Under `projection` the redder lines keep their answers on every arm, at the production rate. The reason is the causal structure of the model rather than the labeller: the answer token is predicted from the stream at `=`, and the stream at the answer position feeds only the newline. So an anchor at the answer position sits downstream of the readout it might have changed, and no labeller that pulls there can close the gap E3 found.

Δα at `=` could close it, but it is small on every arm and unchanged. The redder lines lose nothing under `projection` because the operands that compute the answer are not red, and so not on the axis; ex-2.2.3 gave the same reading of this table.

For D2.2 the whole-line pull is harmless: no task cost, placement within band, and free to adopt when the document shape calls for it. But nothing we read here recommends it for the anchored-op experiments. There the prompt-span pull keeps the pulled positions away from the positions the answer is read from, and that separation is what makes the `=` and operand reads interpretable. So we keep the production labeller.

If the whole-line pull is adopted later, the number to watch is containment: ᾱ at op1 is a little higher on both whole-line arms than on production, inside the twenty-seed band but on its upper side. For M3 the lesson is that a document-level label will put alignment on positions that carry the concept as an output, and an intervention that aims to change behaviour has to reach the positions whose stream feeds the readout.

What we would do differently: a single arm, `line-whole` against production, answers the question. Also, the one place an answer-position pull could act causally is the prediction of the newline, and the scorer in ex-2.2.3 does not read that; a follow-up that cares would add it.

## Method notes

- The arms and the production corner share the corpus, the eval sets and the probe lines (the same seeds through ex-2.2.3's `prepare_corpus`); the line-keyed arms train against a probe table whose `line_p` is recomputed for the answer's draw, and that table is what this report weights by when a column says `line`.
- `line` keying draws the answer slot from a separate random stream, after the operands' draws, so an `either` arm's labels are unchanged by the code path (see `tests/sca/test_anchoring.py`).
- The softmin profiles run over all six roles at the arm's τ; ex-2.2.3's stored `pi` and `w_group` stop at `=`. The stored m_line is the four-role margin under the arm's own labeller, and the recomputed columns read the stored per-line alignment maps.
- The redder-lines comparison and the dose-matched weights are ex-2.2.3's E3 construction, applied unchanged.
- This is a pilot: three arms with two or three seeds each, and production's twenty for the fourth corner. Nothing here is gated, and nothing in it should be quoted as a result.
"""
