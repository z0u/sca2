# title: Ex 2.2.3: the multi-op grammar, with red anchored again

import itertools
import json
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D

# The design constants and the result refs come from `experiment.py` beside this document
# (the document's directory is on sys.path while it runs). The prose quotes the frozen
# gates, and that module carries the same numbers with each gate's wording in its docstring.
import experiment as ex
from mini.lit import memo, read_npz, stop
from mini.store import Artifact, project_store
from mini.vis import (
    AxesGrid,
    AxesRow,
    figure_html,
    light_dark,
    smooth_step,
    smooth_step_area,
    smooth_step_band,
    themed,
)
from sca.anchoring import softmin_weights
from sca.vis import CUBE_VIEWS, draw_cube_bound, grid_diameter, project_cube
from sca.vis_probes import draw_traces

SURVEY_STATS = ("m_line", "r2_sim", "contrast", "alpha_op1", "holdout_em", "retention")
# The statistics the survey scored its trials on, in the order the H3 table prints them.

SLICE_NAMES = ["emb", "1", "2", "3", "4"]
POS_NAMES = ["op1", "op", "op2", "=", "ans", "⏎"]
ROLES = POS_NAMES[: ex.SPAN]
# The span roles the softmin profiles run over; `op` is the op word, where D2.1 had `+`.

EX221_METRICS_REF = "reports/m2/ex-2.2.1/metrics"
EX2110_PRIMARY = "either-t100"
EX2110_CONTROL = "lam0"
EX221_PROJECTION = "primary"
# Ex-2.2.1 named its full projection `primary`; the same row is `projection` here.

INK = {
    "control": ("#555", "#aaa"),
    "control-short": ("#999", "#777"),
    "recipe": ("#1f6fb4", "#5fa8dd"),
    "recipe-short": ("#7fb3d8", "#4f7ea6"),
    "t00": ("#d0461b", "#f07a50"),
    "t48": ("#8030c0", "#c48cff"),
    "t12": ("#1e8a5a", "#5ccf98"),
    "ex-2.1.10": ("#0000001a", "#ffffff22"),
}
# One ink per condition, as (light, dark) pairs for `light_dark`; the D2.1 reference draws as a ghost.


def fetch(refs: Sequence[str], into: Path) -> dict[str, tuple[Artifact, Path] | None]:
    """Each ref's artifact and published file under *into*, or None before it exists.

    One `get_refs` and one `get_many` for the lot: the bucket's per-call latency is most of a render if the refs are resolved one at a time. The artifact rides along because `Results` keys the figure cache by the hashes of what it was built from.
    """
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | {r: (a, p) for (r, a), p in zip(have.items(), paths, strict=True)}


def read_json(got: tuple[Artifact, Path] | None) -> dict | None:
    return None if got is None else json.loads(got[1].read_text())


def read_arrays(got: tuple[Artifact, Path] | None) -> Mapping[str, np.ndarray] | None:
    return None if got is None else read_npz(got[1])


def order_stat(a: np.ndarray, q: float, axis: int) -> np.ndarray:
    """The *q*-th percentile as an order statistic, as `experiment.top_quantile` reads it: no interpolation,
    so a monotone map of the values commutes with the quantile.
    """
    n = a.shape[axis]
    k = int(np.ceil(n * (1 - q / 100)))
    return np.take(np.sort(a, axis=axis), n - k, axis=axis)


def span2(v: np.ndarray, fmt: str = ".3f") -> str:
    """Seed mean with half the seed range beside it, in the shared `.range` style."""
    return f"{v.mean():{fmt}} <span class='range'>±{(v.max() - v.min()) / 2:{fmt}}</span>"


def ink(cond: str) -> str:
    return light_dark(*INK[cond])


def control_of(c: ex.Condition) -> ex.Condition:
    """The un-anchored arm of the same length, which H1 reads a condition against."""
    return ex.CONTROL if c.epochs == ex.EPOCHS else ex.CONTROL_SHORT


def table_html(head: list[str], rows: list[list[str]], caption: str, *, ref_rows: frozenset[int] = frozenset()) -> str:
    """A result table in the shared classes: numeric cells right-aligned, and `ref_rows` styled as references."""
    ths = "".join(f"<th{' class=num' if i else ''}>{h}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=ref' if r in ref_rows else ''}>"
        + "".join(f"<td{' class=num' if i else ''}>{c}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


@memo
def mix_counts() -> dict[str, int]:
    """The line counts the prose quotes (the same for every op by dose; `mix` stands for all)."""
    return ex.line_counts(ex.MIX)


@memo
def op_table_md() -> str:
    """The op table: each rule, how often it lands on the grid unrounded, and its redder-than-both lines."""
    rows = []
    for op in ex.OPS:
        n = ex.line_counts(op)
        rows.append(
            f"| `{op.name}` | {op.rule} | {ex.on_grid(op):.1%} | {n['redder']:,} ({n['redder'] / n['lines']:.1%}) |"
        )
    head = "| op | per channel | on grid | redder than both |\n| --- | --- | ---: | ---: |\n"
    return head + "\n".join(rows)


@memo
def agreement_md() -> str:
    """Pairwise agreement: the share of pairs on which two ops give the same answer."""
    names = [op.name for op in ex.OPS]
    cells = {}
    for p, q in itertools.combinations(ex.OPS, 2):
        cells[(p.name, q.name)] = cells[(q.name, p.name)] = f"{ex.agreement(p, q):.1%}"
    head = "| | " + " | ".join(f"`{n}`" for n in names) + " |\n|" + " ---: |" * (len(names) + 1) + "\n"
    rows = ["| `" + p + "` | " + " | ".join(cells.get((p, q), "—") for q in names) + " |" for p in names]
    return head + "\n".join(rows)


@memo
def relevance_md() -> str:
    """Per candidate anchored op, the share of its lines on which k other ops give the same answer."""
    dists = {op.name: ex.relevance(op) for op in ex.OPS}
    levels = sorted({k for d in dists.values() for k, v in d.items() if v >= 0.0005})
    head = "| anchored op | " + " | ".join(str(k) for k in levels) + " |\n|" + " ---: |" * (len(levels) + 1) + "\n"
    rows = [
        f"| `{name}` | " + " | ".join(f"{d[k]:.1%}" if k in d else "·" for k in levels) + " |"
        for name, d in dists.items()
    ]
    return head + "\n".join(rows)


def conditions_md() -> str:
    rows = []
    for c in ex.CONDITIONS:
        if c.lam == 0:
            anchor = "none"
        else:
            shape = "annealed" if c.anchor_anneal else "flat"
            anchor = f"λ_a = {c.lam:.3f} ({shape}), τ = {c.tau:.3f}, anti {c.anti_peak_ratio:.2f} → 0.30 by {c.anti_anneal_end_frac:.0%}"
        ops = "all six" if c.ops == ex.OP_NAMES else ", ".join(f"`{o}`" for o in c.ops)
        rows.append(
            f"| **{c.name}** | {c.title} | {anchor} | {ops} | {c.lines_per_op:,} | {c.epochs} ({c.steps:,}) | {c.seeds} |"
        )
    head = (
        "| condition | what | anchor | ops | lines per op | epochs (steps) | seeds |\n"
        "| --- | --- | --- | --- | ---: | ---: | ---: |\n"
    )
    return head + "\n".join(rows)


def calibration_md(cal: dict | None) -> str:
    """The pre-freeze read: holdout exact match per op for one seed of each control arm."""
    if cal is None:
        return "_The calibration has not been published yet._"
    head = "| run | " + " | ".join(f"`{o}`" for o in ex.OP_NAMES) + " |\n|" + " ---: |" * (len(ex.OP_NAMES) + 1) + "\n"
    rows = [
        f"| `{r['label']}` | " + " | ".join(f"{r['holdout_em'][o]:.3f}" for o in ex.OP_NAMES) + " |"
        for r in cal["runs"]
    ]
    return head + "\n".join(rows)


@dataclass(frozen=True)
class Results:
    """Every published result the report reads, with one accessor per shape the cells need.

    `metrics`, `traj`, `arrays` and `geometry` are this experiment's; `ex2110` and `ex221` are the D2.1
    references (the recipe's nine seeds, and the suppression figures on the one-op grammar); `survey` is
    ex-2.1.11's, whose proposals H3 re-measures.
    """

    metrics: dict
    traj: dict
    arrays: Mapping[str, np.ndarray]
    geometry: dict
    ex2110: dict
    ex221: dict
    survey: dict | None
    sources: tuple[Artifact | None, ...]  # what the fields were read from, in field order

    def __memo_key__(self) -> list[str | None]:
        """The figure cache keys a `Results` by the hashes of its sources rather than digesting every array it holds."""
        return [a.sha256 if a is not None else None for a in self.sources]

    def runs(self, cond: str) -> list[dict]:
        """The eval records of a condition, in seed order."""
        return sorted((r for r in self.metrics["runs"] if r["condition"] == cond), key=lambda r: r["seed"])

    def scored(self, cond: str) -> list[dict]:
        return sorted((r for r in self.metrics["scores"] if r["condition"] == cond), key=lambda r: r["seed"])

    def stat(self, cond: str, key: str, op: str | None = None) -> np.ndarray:
        """One value per seed: a gated (`mix`) statistic, or one op's from `per_op`."""
        return np.array([r[key] if op is None else r["per_op"][op][key] for r in self.runs(cond)], float)

    def em(self, cond: str, op: str) -> np.ndarray:
        """Holdout exact match per seed on one op; NaN where the arm never trained on that op."""
        return np.array([r["holdout_em"].get(op, np.nan) for r in self.runs(cond)], float)

    def arr(self, cond: str, kind: str, name: str) -> np.ndarray:
        """One per-run array stacked over seeds: `kind` is `eval` or `score`, `name` is `{op}/{array}`."""
        return np.stack([self.arrays[f"{r['label']}/{kind}/{name}"] for r in self.runs(cond)])

    def trajectory(self, cond: str, key: str) -> np.ndarray:
        """(seeds, points) of one trajectory key, recorded every `TRAJ_STRIDE` steps."""
        return np.array([self.traj[r["label"]]["traj"][key] for r in self.runs(cond)], float)

    def val_loss(self, cond: str) -> np.ndarray:
        """(seeds, epochs) validation loss."""
        return np.array([self.traj[r["label"]]["val_loss"] for r in self.runs(cond)], float)

    def score(self, cond: str, op: str, iv: str | None, key: str, group: str | None = "red") -> np.ndarray:
        """One value per seed of a scored statistic on one op's probe lines; `iv=None` is the clean pass."""
        out = []
        for r in self.scored(cond):
            s = r["ops"][op]
            v = s["clean"][key] if iv is None else s["interventions"][iv][key]
            out.append(v if group is None else v[group])
        return np.array(out, float)

    def deficit(self, cond: str, op: str, iv: str, group: str = "nonred") -> np.ndarray:
        """Clean exact-match accuracy minus intervened accuracy, per seed, the statistic ex-2.2.1 gated on."""
        return self.score(cond, op, None, "acc", group) - self.score(cond, op, iv, "acc", group)

    def pooled(self, cond: str, key: str) -> np.ndarray:
        """A recipe arm's frozen seeds and its addendum seeds together: twenty values of one statistic."""
        v = (
            self.em(c, ex.PRIMARY_OP.name) if key == "holdout_em" else self.stat(c, key) for c in (cond, f"{cond}-more")
        )
        return np.concatenate(list(v))

    def pooled_score(self, cond: str, op: str, iv: str | None, key: str, group: str = "red") -> np.ndarray:
        return np.concatenate([self.score(c, op, iv, key, group) for c in (cond, f"{cond}-more")])

    def pooled_deficit(self, cond: str, op: str, iv: str, group: str = "nonred") -> np.ndarray:
        return np.concatenate([self.deficit(c, op, iv, group) for c in (cond, f"{cond}-more")])

    def ref_cells(self, cond: str = EX2110_PRIMARY) -> list[dict]:
        return [c for c in self.ex2110["cells"] if c["condition"] == cond]

    def ref_stat(self, key: str, cond: str = EX2110_PRIMARY) -> np.ndarray:
        """One value per seed of the D2.1 reference: nine on the primary, three on the control."""
        return np.array([c[key] for c in self.ref_cells(cond)], float)

    def ref_trajectory(self, key: str, cond: str = EX2110_PRIMARY) -> np.ndarray:
        return np.array([c["traj"][key] for c in self.ref_cells(cond)], float)

    def ref_retention(self, cond: str = EX2110_PRIMARY) -> np.ndarray:
        t = self.ref_trajectory("m_line", cond)
        return t[:, -1] / np.maximum.accumulate(t, axis=1).max(axis=1)

    def ex221_stat(self, iv: str | None, key: str, group: str | None = "red", cond: str = EX2110_PRIMARY) -> np.ndarray:
        """One value per seed of ex-2.2.1's one-op suppression figures; `iv=None` is its clean pass."""
        out = []
        for r in sorted((r for r in self.ex221["runs"] if r["condition"] == cond), key=lambda r: r["seed"]):
            v = r["clean"][key] if iv is None else r["interventions"][iv][key]
            out.append(v if group is None else v[group])
        return np.array(out, float)

    def ex221_deficit(self, iv: str, group: str = "nonred") -> np.ndarray:
        return self.ex221_stat(None, "acc", group) - self.ex221_stat(iv, "acc", group)

    def survey_point(self, c: ex.Condition) -> tuple[int, dict | None]:
        """A candidate's one-op numbers: the survey's own re-run of the recipe, or its five-seed trial."""
        if c.survey_trial is None:
            return ex.SURVEY_RECIPE[c.name]
        return 5, (self.survey["scored"].get(str(c.survey_trial)) if self.survey else None)

    def nonred_profile(self, cond: str, op: str = ex.PRIMARY_OP.name) -> np.ndarray:
        """(seeds, slices, roles) softmin profile of the non-red group, at each run's τ: the latch reads it."""
        out = []
        for r in self.runs(cond):
            cos = self.arrays[f"{r['label']}/eval/{op}/alpha_lines"].astype(np.float32)
            nonred = self.arrays[f"{r['label']}/eval/{op}/dose"] <= ex.NONRED_DOSE + 1e-9
            out.append(softmin_weights(1.0 - cos[:, nonred, : ex.SPAN], r["tau"]).mean(axis=1))
        return np.stack(out)

    def clean_alpha_band(self, cond: str, op: str, q: float = 99.0) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) signed tails of the clean alignment over the non-red lines, each (slices, positions)
        and seed-meaned: the *q*-th percentile and its reflection.

        The scorer stores only `alpha_q99_nonred`, the same percentile of |α|, which folds the two tails
        together; the per-line alignments the eval stage stores keep the sign, on the same probe lines.
        """
        hi, lo = [], []
        for r in self.runs(cond):
            cos = self.arrays[f"{r['label']}/eval/{op}/alpha_lines"].astype(np.float32)
            nonred = self.arrays[f"{r['label']}/eval/{op}/dose"] <= ex.NONRED_DOSE + 1e-9
            a = cos[:, nonred]
            hi.append(order_stat(a, q, axis=1))
            lo.append(-order_stat(-a, q, axis=1))
        return np.mean(lo, axis=0), np.mean(hi, axis=0)

    def latched(self, cond: str) -> np.ndarray:
        """Per run: the non-red group's op1 weight over the post-attention slices, above `LATCH_PI`."""
        return self.nonred_profile(cond)[:, 1:, 0].mean(axis=1) > ex.LATCH_PI


@dataclass(frozen=True)
class Verdict:
    """One hypothesis's read: `status` is holds, partial, contrary, or unscored; `line` is the Findings entry."""

    status: str
    line: str

    @property
    def md(self) -> str:
        return f"**{self.status}.** {self.line}"


def alpha_at(res: Results, cond: str, op: str, position: int) -> np.ndarray:
    """(seeds, slices) mean alignment at one position of one op's probe lines."""
    return np.array([r["per_op"][op]["alpha_pos"] for r in res.runs(cond)], float)[:, :, position]


def syntax_table(res: Results, cond: str) -> str:
    """E2's table for one condition: the alignment of each op word, `=`, and the newline, per slice."""
    rows = []
    for op in ex.OP_NAMES:
        a = alpha_at(res, cond, op, 1)
        rows.append([f"<code>{op}</code>"] + [span2(a[:, s], ".2f") for s in range(len(SLICE_NAMES))])
    for name, p in (("=", 3), ("⏎", 5)):
        a = np.mean([alpha_at(res, cond, op, p) for op in ex.OP_NAMES], axis=0)
        rows.append([f"<code>{name}</code>"] + [span2(a[:, s], ".2f") for s in range(len(SLICE_NAMES))])
    head = ["token"] + [f"slice {s}" for s in SLICE_NAMES]
    caption = f"""
    Alignment (cos to e₁) of the syntax tokens on <code>{cond}</code>, per slice: each op word read on its own op's probe lines, and <code>=</code> and the newline averaged over the six ops. Seed mean with half the seed range. The embedding column is the token's embedding itself; deeper columns are the residual stream at that position.
    """
    return table_html(head, rows, caption)


def removal_table(res: Results, cond: str) -> str:
    """H4's removal read on one condition: red-line accuracy per op under each operator, with ex-2.2.1's one-op column."""
    ivs = [ex.PROJECTION, *ex.RIDE_ALONG]
    ex221 = {"projection": EX221_PROJECTION, "operands": "operands", "shaped": "shaped", "ablate": "ablate"}

    def b(s: str, ok: bool) -> str:
        return f"<b>{s}</b>" if ok else s

    rows = []
    for iv in ivs:
        gated = iv.name == ex.PROJECTION.name
        row = [f"{cond}, {iv.name}"]
        for op in ex.OP_NAMES:
            v = res.score(cond, op, iv.name, "acc", "red")
            row.append(b(span2(v, ".2f"), bool(gated and v.mean() <= ex.RED_ACC_GATE)))
        row.append(span2(res.ex221_stat(ex221[iv.name], "acc", "red"), ".2f"))
        rows.append(row)
    rows.append(
        [f"{cond}, clean"]
        + [span2(res.score(cond, op, None, "acc", "red"), ".2f") for op in ex.OP_NAMES]
        + [span2(res.ex221_stat(None, "acc", "red"), ".2f")]
    )
    head = ["condition, operator"] + [f"{op} red acc ↓" for op in ex.OP_NAMES] + ["ex-2.2.1 (mix)"]
    caption = f"""
    Removal on the red lines of each op, for <code>{cond}</code>. Each value is seed-mean exact-match accuracy on the red lines (dose ≥ {ex.RED_DOSE:g}) under the operator, with half the seed range; the last column is the same operator on ex-2.2.1's one-op grammar (nine seeds of the D2.1 recipe). Bold marks a <code>projection</code> value inside the removal gate, at or below {ex.RED_ACC_GATE:g}. The ride-along rows carry no gate. The last row is the clean pass.
    """
    return table_html(head, rows, caption, ref_rows=frozenset({len(rows) - 1}))


def selectivity_table(res: Results, cond: str) -> str:
    """H4's selectivity read on one condition: the non-red deficit per op under each operator."""
    ivs = [ex.PROJECTION, *ex.RIDE_ALONG]
    ex221 = {"projection": EX221_PROJECTION, "operands": "operands", "shaped": "shaped", "ablate": "ablate"}
    mix = ex.PRIMARY_OP.name

    def cell(v: np.ndarray, gated: bool) -> str:
        s = span2(v)
        return f"<b>{s}</b>" if gated and v.mean() <= ex.NONRED_DEFICIT_GATE else s

    rows = []
    for iv in ivs:
        row = [f"{cond}, {iv.name}"]
        for op in ex.OP_NAMES:
            row.append(cell(res.deficit(cond, op, iv.name), iv.name == ex.PROJECTION.name and op == mix))
        row.append(span2(res.ex221_deficit(ex221[iv.name])))
        rows.append(row)
    head = ["condition, operator"] + [f"{op} non-red deficit ↓" for op in ex.OP_NAMES] + ["ex-2.2.1 (mix)"]
    d = res.deficit(cond, mix, ex.PROJECTION.name).mean()
    caption = f"""
    Selectivity on the non-red lines of each op, for <code>{cond}</code>. The deficit is clean exact-match accuracy minus intervened accuracy on the non-red lines (dose ≤ {ex.NONRED_DOSE:g}), seed mean with half the seed range; the last column is ex-2.2.1's one-op value for the same operator. The gate reads <code>projection</code> on <code>mix</code> only, at or below {ex.NONRED_DEFICIT_GATE:g} (bold); the 0.02 read that ex-2.2.1 gated at is {"met" if d <= 0.02 else "not met"} there ({d:.3f}). The other ops are reported for E2.
    """
    return table_html(head, rows, caption)


def draw_probe_grid(ax: plt.Axes, stack: np.ndarray, *, own: int | None, row_names: bool, scale: bool) -> None:
    """Ex-2.1.12's probe panel: one row per residual slice, the six positions across, one trace per RGB channel.

    *stack* is (seed, slice, position, channel) strict R² for one target. Each row draws the seed mean over the grey area of the channel mean, with the seed envelope as hairlines; rows share one Axes, each offset by a translate, so every row spans the same 0–1 scale. Negative scores clip to the floor. *own* is the position whose color the panel decodes, drawn bold.
    """
    sw = 0.7
    _, n_slices, n_pos, _ = stack.shape
    stack = np.clip(stack, 0, 1)
    for si in range(n_slices):
        shift = Affine2D().translate(0, si) + ax.transData
        draw_traces(
            ax, stack.mean(0)[si], spread=stack[:, si], ramp=1 - sw, faint_risers=True, transform=shift, clip_on=False
        )
        ax.axhline(si, color=light_dark("#aaaa", "#333a"), lw=0.5)
    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_ylim(0, n_slices)
    ax.set_xticks(range(n_pos), POS_NAMES)
    ax.set_yticks(np.arange(n_slices) + 0.5, SLICE_NAMES if row_names else [""] * n_slices)
    ax.tick_params(axis="x", labelsize=6, bottom=False)
    ax.tick_params(axis="y", labelsize=7, left=False)
    if own is not None:
        ax.get_xticklabels()[own].set(fontweight="bold")
    if scale:
        sec = ax.secondary_yaxis("right")
        sec.set_yticks(np.arange(n_slices + 1), ["0", "1"] + [""] * (n_slices - 1))
        sec.tick_params(labelsize=6, direction="out")
        sec.set_ylabel("R², 0–1 per row", fontsize=7)


REFS = [
    ex.METRICS_REF,
    ex.TRAJ_REF,
    ex.ARRAYS_REF,
    ex.GEOMETRY_REF,
    ex.EX2110_METRICS_REF,
    EX221_METRICS_REF,
    ex.SURVEY_REF,
]
with tempfile.TemporaryDirectory() as _tmp:
    got = fetch([*REFS, ex.CALIBRATION_REF], Path(_tmp))
    loaded = read_json(got[ex.METRICS_REF])
    traj_loaded, arrays_loaded, geometry_loaded = (
        read_json(got[ex.TRAJ_REF]),
        read_arrays(got[ex.ARRAYS_REF]),
        read_json(got[ex.GEOMETRY_REF]),
    )
    ex2110_loaded, ex221_loaded = read_json(got[ex.EX2110_METRICS_REF]), read_json(got[EX221_METRICS_REF])
    survey_loaded, calibration = read_json(got[ex.SURVEY_REF]), read_json(got[ex.CALIBRATION_REF])
if loaded is None:
    stop("_Results are not published yet; the result cells render once they are._")
metrics_loaded: dict = loaded
assert traj_loaded and arrays_loaded and geometry_loaded and ex2110_loaded and ex221_loaded, (
    "a reference result is missing from the store"
)
res: Results = Results(
    metrics_loaded,
    traj_loaded,
    arrays_loaded,
    geometry_loaded,
    ex2110_loaded,
    ex221_loaded,
    survey_loaded,
    tuple(g[0] if g is not None else None for g in (got[r] for r in REFS)),
)
# The design the DAG ran is the one this document quotes.
assert res.metrics["design"]["n_runs"] == ex.N_RUNS
assert len(res.metrics["runs"]) == ex.N_RUNS, len(res.metrics["runs"])
assert {r["condition"] for r in res.metrics["scores"]} == {c.name for c in (*ex.CANDIDATES, *ex.ADDENDUM)}
# The reference the margin gates quote, to three decimals.
assert round(float(res.ref_stat("m_line").mean()), 3) == round(ex.REF_M_LINE, 3)

# The hypothesis verdicts and the selection-rule decisions are computed here, ahead of the Findings
# paragraph that reads them; the narration for each stays in place further down.

# H1: every scored condition against the control of its own length, op by op.
h1_gaps: dict[tuple[str, str], float] = {
    (c.name, op): float(res.em(c.name, op).mean() - res.em(control_of(c).name, op).mean())
    for c in ex.CANDIDATES
    for op in ex.OP_NAMES
}
h1_abs = {k: abs(v) for k, v in h1_gaps.items()}
h1_worst_key = max(h1_abs, key=h1_abs.__getitem__)
h1_n_gate = sum(v <= ex.TASK_GATE for v in h1_abs.values())
h1_n_partial = sum(v <= ex.TASK_PARTIAL for v in h1_abs.values())
h1_above = [k for k, v in h1_gaps.items() if v > ex.TASK_GATE]
task_ok: dict[str, bool] = {
    c.name: all(h1_abs[c.name, op] <= ex.TASK_GATE for op in ex.OP_NAMES) for c in ex.CANDIDATES
}
# Per candidate, the H1 gate as the selection rule reads it: within the gate on every op.
h1_n = len(h1_abs)
if h1_n_gate == h1_n:
    h1_status = "holds"
elif h1_n_partial == h1_n and h1_n_gate >= h1_n - 1:
    h1_status = "partial"
elif h1_n_partial == h1_n:
    h1_status = "partial"
else:
    h1_status = "contrary"
h1_worst_cond, h1_worst_op = h1_worst_key
h1_line = (
    f"All {h1_n} condition-op comparisons are within {ex.TASK_GATE:g} of the control of the same length; "
    f"the largest gap is {h1_gaps[h1_worst_key]:+.4f}, `{h1_worst_cond}` on `{h1_worst_op}`."
    if h1_status == "holds"
    else f"{h1_n_gate} of {h1_n} comparisons are within {ex.TASK_GATE:g} and {h1_n_partial} within {ex.TASK_PARTIAL:g}; "
    f"the largest gap is {h1_gaps[h1_worst_key]:+.4f}, `{h1_worst_cond}` on `{h1_worst_op}`."
)
if h1_above:
    h1_n_above = f"{len(h1_above)} of them sit" if len(h1_above) > 1 else "One of them sits"
    h1_line += f" {h1_n_above} more than {ex.TASK_GATE:g} *above* the control, an anomaly the design did not predict."
h1: Verdict = Verdict(h1_status, h1_line)


# H2's criteria on the recipe's mix lines. Seed-mean criteria first, then the two per-run ones.
h2_cond = ex.RECIPE.name
h2_m_line, h2_a_op1, h2_lead = (
    res.stat(h2_cond, "m_line"),
    res.stat(h2_cond, "alpha_op1"),
    res.stat(h2_cond, "lead_emb"),
)
h2_contrast, h2_r2 = res.stat(h2_cond, "contrast"), res.stat(h2_cond, "r2_sim")
h2_ret, h2_peak = res.stat(h2_cond, "retention"), res.stat(h2_cond, "m_line_peak")
h2_ret_scored = h2_ret[h2_peak >= ex.RETENTION_FLOOR]
h2_latched = res.latched(h2_cond)
h2_checks: dict[str, tuple[float, bool]] = {
    "containment": (float(h2_a_op1.mean()), bool(h2_a_op1.mean() <= ex.MEAN_ALIGN_GATE)),
    "lead": (float(h2_lead.mean()), bool(h2_lead.mean() >= ex.LEAD_GATE)),
    "contrast": (float(h2_contrast.mean()), bool(h2_contrast.mean() >= ex.CONTRAST_GATE)),
    "grading": (float(h2_r2.mean()), bool(h2_r2.mean() >= ex.REF_R2_SIM - ex.GRADE_R2_DROP)),
    "margin": (float(h2_m_line.mean()), bool(h2_m_line.mean() >= ex.MARGIN_RATIO * ex.REF_M_LINE)),
    "retention": (
        float(h2_ret_scored.min()) if len(h2_ret_scored) else float("nan"),
        bool(len(h2_ret_scored) == 0 or h2_ret_scored.min() >= ex.RETENTION_GATE),
    ),
    "latch": (float(res.nonred_profile(h2_cond)[:, 1:, 0].mean(axis=1).max()), bool(not h2_latched.any())),
}
# Each H2 criterion on the recipe: (the statistic as the gate reads it, whether it holds).
h2_misses = [k for k, (_, ok) in h2_checks.items() if not ok]
h2_m_band = ex.MARGIN_PARTIAL * ex.REF_M_LINE <= h2_m_line.mean() < ex.MARGIN_RATIO * ex.REF_M_LINE
h2_c_band = ex.CONTRAST_PARTIAL <= h2_contrast.mean() < ex.CONTRAST_GATE
h2_ratio = h2_m_line.mean() / ex.REF_M_LINE
if not task_ok[h2_cond]:
    h2: Verdict = Verdict(
        "unscored", "The recipe missed the H1 gate at full length, so H2 is not scored; the refuted H1 is the finding."
    )
elif not h2_misses:
    h2 = Verdict(
        "holds",
        f"Every placement criterion holds on the `mix` lines at five fresh seeds: seed-mean m_line {h2_m_line.mean():.3f}, {h2_ratio:.2f} of its ex-2.1.10 value, grading r² {h2_r2.mean():.3f}, contrast {h2_contrast.mean():.2f}, containment {h2_a_op1.mean():.3f}, no latched run.",
    )
elif h2_misses in (["margin"], ["contrast"]) and (h2_m_band or h2_c_band):
    h2 = Verdict(
        "partial",
        f"Every criterion holds except {h2_misses[0]}, which sits in its partial band: m_line {h2_m_line.mean():.3f} ({h2_ratio:.2f} of the reference), contrast {h2_contrast.mean():.2f}.",
    )
elif "latch" in h2_misses or "containment" in h2_misses:
    h2 = Verdict(
        "contrary",
        f"The recipe misses {', '.join(h2_misses)}: containment {h2_a_op1.mean():.3f}, {int(h2_latched.sum())} latched run(s). Something in the new grammar changed what the pull finds.",
    )
else:
    h2 = Verdict(
        "contrary",
        f"The recipe misses {', '.join(h2_misses)}: m_line {h2_m_line.mean():.3f} ({h2_ratio:.2f} of the reference), grading r² {h2_r2.mean():.3f}, contrast {h2_contrast.mean():.2f}, lead {h2_lead.mean():.2f}, retention {h2_checks['retention'][0]:.2f}.",
    )


# H3: feasibility as the selection rule reads it, then the rule itself.
h3_r2_floor = float(res.stat(ex.RECIPE.name, "r2_sim").mean()) - ex.GRADE_R2_DROP
h3_r2_margin = ex.GRADE_MARGIN_SD * ex.NOISE_RUN["r2_sim"]
h3_band = ex.equiv_band("m_line", 5, 5)

feas: dict[str, dict[str, bool]] = {}
# Per candidate, each feasibility check of the selection rule, in the order the table prints them.
for h3c in ex.CANDIDATES:
    h3_name = h3c.name
    h3_ret, h3_peak = res.stat(h3_name, "retention"), res.stat(h3_name, "m_line_peak")
    h3_scored = h3_ret[h3_peak >= ex.RETENTION_FLOOR]
    feas[h3_name] = {
        "task": task_ok[h3_name],
        "containment": bool(res.stat(h3_name, "alpha_op1").mean() <= ex.MEAN_ALIGN_GATE),
        "contrast ≥ partial": bool(res.stat(h3_name, "contrast").mean() >= ex.CONTRAST_PARTIAL),
        "grading": bool(res.stat(h3_name, "r2_sim").mean() >= h3_r2_floor + h3_r2_margin),
        "retention": bool(len(h3_scored) == 0 or h3_scored.min() >= ex.RETENTION_GATE),
        "no latch": bool(not res.latched(h3_name).any()),
        "m_line floor": bool(res.stat(h3_name, "m_line").min() >= ex.MARGIN_PARTIAL * ex.REF_M_LINE),
    }
h3_feasible = [c for c in ex.CANDIDATES if all(feas[c.name].values())]
h3_full = [c for c in h3_feasible if res.stat(c.name, "contrast").mean() >= ex.CONTRAST_GATE]
h3_m = {c.name: float(res.stat(c.name, "m_line").mean()) for c in ex.CANDIDATES}
h3_grade = {c.name: float(res.stat(c.name, "r2_sim").mean()) - h3_r2_floor for c in ex.CANDIDATES}
h3_order = {c.name: i for i, c in enumerate((ex.RECIPE, ex.RECIPE_SHORT))}  # then the proposals, unranked

if h3_full:
    h3_best = max(h3_m[c.name] for c in h3_full)
    h3_tied = [c for c in h3_full if h3_m[c.name] >= h3_best - h3_band]
    h3_win = max(h3_tied, key=lambda c: (h3_grade[c.name], -h3_order.get(c.name, 9)))
    adopted: str = h3_win.name
    h3_how = (
        f"{len(h3_tied)} candidate(s) within one band ({h3_band:.3f}) of the highest feasible m_line ({h3_best:.3f})"
        + ("; the tie went to the larger grading margin" if len(h3_tied) > 1 else "")
    )
else:
    adopted = ex.RECIPE.name
    h3_missed = [k for k, ok in feas[adopted].items() if not ok]
    h3_how = "no feasible candidate clears the full contrast gate, so the full-length recipe is adopted" + (
        f"; it misses {', '.join(h3_missed)} itself" if h3_missed else ""
    )
adoption_note: str = h3_how

# The verdict reads the proposals against recipe-short.
h3_short = h3_m[ex.RECIPE_SHORT.name]
h3_feasible_props = [c for c in ex.PROPOSALS if c in h3_feasible]
h3_unstable = [c.name for c in ex.PROPOSALS if [k for k, ok in feas[c.name].items() if not ok] == ["m_line floor"]]
h3_grading_only = [
    c
    for c in ex.PROPOSALS
    if [k for k, ok in feas[c.name].items() if not ok] == ["grading"] and h3_m[c.name] > h3_short + h3_band
]
h3_lead = {c.name: h3_m[c.name] - h3_short for c in ex.PROPOSALS}
h3_summary = ", ".join(f"`{c.name}` {h3_lead[c.name]:+.3f}" for c in ex.PROPOSALS)
if any(h3_lead[c.name] > h3_band for c in h3_feasible_props):
    h3_v = Verdict(
        "holds",
        f"A feasible proposal exceeds `recipe-short` ({h3_short:.3f}) by more than a band ({h3_band:.3f}): {h3_summary}. Adopted: `{adopted}`.",
    )
elif any(abs(h3_lead[c.name]) <= h3_band for c in h3_feasible_props) or h3_grading_only:
    h3_v = Verdict(
        "partial",
        f"The feasible proposals sit within a band ({h3_band:.3f}) of `recipe-short` ({h3_short:.3f}): {h3_summary}. Adopted: `{adopted}`.",
    )
elif not h3_feasible_props:
    h3_v = Verdict(
        "contrary",
        f"No proposal is feasible at fresh seeds ({h3_summary} against `recipe-short` at {h3_short:.3f}). Adopted: `{adopted}`.",
    )
else:
    h3_v = Verdict(
        "contrary",
        f"Every feasible proposal sits more than a band ({h3_band:.3f}) below `recipe-short` ({h3_short:.3f}): {h3_summary}. Adopted: `{adopted}`.",
    )
if h3_unstable:
    h3_v = Verdict(
        h3_v.status,
        h3_v.line
        + f" {', '.join(f'`{n}`' for n in h3_unstable)} missed only the per-run m_line floor, so it reads as unstable at fresh seeds rather than weak.",
    )
h3: Verdict = h3_v


# E6's twenty-seed comparison of the two recipe lengths is computed here as well, since the post hoc
# amendment below breaks its tie with it; its table and figure stay in place under Exploratory analyses.
e6_stats = ["m_line", "contrast", "r2_sim", "alpha_op1", "lead_emb", "holdout_em"]
e6_pool: dict[tuple[str, str], np.ndarray] = {
    (c, k): res.pooled(c, k) for c in (ex.RECIPE.name, ex.RECIPE_SHORT.name) for k in e6_stats
}
e6_n = len(e6_pool[ex.RECIPE.name, "m_line"])
e6_reads: dict[str, tuple[float, float, float]] = {}
for e6k in e6_stats:
    e6a, e6b = e6_pool[ex.RECIPE.name, e6k], e6_pool[ex.RECIPE_SHORT.name, e6k]
    e6d = float(e6b.mean() - e6a.mean())
    e6sd = float(np.sqrt((e6a.var(ddof=1) + e6b.var(ddof=1)) / 2))
    e6frozen = ex.equiv_band(e6k, e6_n, e6_n) if e6k in ex.NOISE_RUN else float("nan")
    e6fresh = ex.equiv_band(e6k, e6_n, e6_n, noise={e6k: e6sd})
    e6_reads[e6k] = (e6d, e6frozen, e6fresh)


# Post hoc, after the results were read (2026-09-10). The frozen selection rule carries every H2
# gate but the lead weight at the embedding, and none of H4; the point it adopts misses the first
# and, under the plain projection, fails the second. This re-runs the rule with each gate added,
# and the Findings entry names the outcome beside the frozen one; the narration sits under
# "Post hoc: the selection rule, with the gates it left out", in its original place.
ph_mix = ex.PRIMARY_OP.name
post_hoc_m = {c.name: float(res.stat(c.name, "m_line").mean()) for c in ex.CANDIDATES}
post_hoc_lead: dict[str, float] = {c.name: float(res.stat(c.name, "lead_emb").mean()) for c in ex.CANDIDATES}
post_hoc_cost: dict[str, float] = {
    c.name: float(res.deficit(c.name, ph_mix, ex.PROJECTION.name).mean()) for c in ex.CANDIDATES
}
ph_band = ex.equiv_band("m_line")
ph_lead_ok = [c for c in ex.CANDIDATES if all(feas[c.name].values()) and post_hoc_lead[c.name] >= ex.LEAD_GATE]
post_hoc_pick_lead: str = max(ph_lead_ok, key=lambda c: post_hoc_m[c.name]).name if ph_lead_ok else ex.RECIPE.name
ph_sel_ok = [c for c in ph_lead_ok if post_hoc_cost[c.name] <= ex.NONRED_DEFICIT_GATE]
ph_top = max(post_hoc_m[c.name] for c in ph_sel_ok) if ph_sel_ok else float("nan")
ph_tied = [c for c in ph_sel_ok if post_hoc_m[c.name] >= ph_top - ph_band]
post_hoc_margin_diff, ph_fb, ph_ff = e6_reads["m_line"]  # short minus full, at twenty seeds, and the two bands
post_hoc_wide_band: float = max(ph_fb, ph_ff)
if len(ph_tied) <= 1:
    adopted_post: str = ph_tied[0].name if ph_tied else ex.RECIPE.name
    post_hoc_tie_note: str = "no tie to break"
elif -post_hoc_margin_diff > post_hoc_wide_band:
    adopted_post = ex.RECIPE.name
    post_hoc_tie_note = f"at twenty seeds the full recipe leads by {-post_hoc_margin_diff:.3f}, more than the {post_hoc_wide_band:.3f} band, so it takes the tie"
elif post_hoc_margin_diff > post_hoc_wide_band:
    adopted_post = ex.RECIPE_SHORT.name
    post_hoc_tie_note = f"at twenty seeds the short recipe leads by {post_hoc_margin_diff:.3f}, more than the {post_hoc_wide_band:.3f} band, so it takes the tie"
else:
    adopted_post = ex.RECIPE_SHORT.name
    post_hoc_tie_note = f"at twenty seeds the two are within the {post_hoc_wide_band:.3f} band ({post_hoc_margin_diff:+.3f}), so the tie goes to the shorter arm, at half the compute"
post_hoc_leads: str = ", ".join(f"`{c.name}` {post_hoc_lead[c.name]:.2f}" for c in ex.CANDIDATES)
post_hoc_costs: str = ", ".join(f"`{c.name}` {post_hoc_cost[c.name]:.3f}" for c in ex.CANDIDATES)
# The decision (2026-09-10, after E6 was read): D2.2 builds on the short arm. The rule above ranks on
# margin and names the full recipe; the reasons for the short arm are in the paragraph below.
carried: str = ex.RECIPE_SHORT.name
post_hoc_r2: dict[str, float] = {
    c: float(res.pooled(c, "r2_sim").mean()) for c in (ex.RECIPE.name, ex.RECIPE_SHORT.name)
}
post_hoc_lead_emb: dict[str, float] = {
    c: float(res.pooled(c, "lead_emb").mean()) for c in (ex.RECIPE.name, ex.RECIPE_SHORT.name)
}


# H4's two statistics under `projection`, on the recipe and on the adopted point.
h4_conds: list[str] = [ex.RECIPE.name] + ([adopted] if adopted != ex.RECIPE.name else [])
# The conditions H4 gates: the recipe, and the adopted point when it is a different one.
h4_iv = ex.PROJECTION.name
h4_mix = ex.PRIMARY_OP.name
h4_reads = {}
for h4c in h4_conds:
    h4_red = {op: float(res.score(h4c, op, h4_iv, "acc", "red").mean()) for op in ex.OP_NAMES}
    h4_def = float(res.deficit(h4c, h4_mix, h4_iv).mean())
    h4_reads[h4c] = (h4_red, h4_def)
h4_removal_ok = {c: all(v <= ex.RED_ACC_GATE for v in r.values()) for c, (r, _) in h4_reads.items()}
h4_removal_misses = {c: [op for op, v in r.items() if v > ex.RED_ACC_GATE] for c, (r, _) in h4_reads.items()}
h4_select_ok = {c: d <= ex.NONRED_DEFICIT_GATE for c, (_, d) in h4_reads.items()}
h4_select_band = {c: ex.NONRED_DEFICIT_GATE < d <= ex.NONRED_DEFICIT_PARTIAL for c, (_, d) in h4_reads.items()}
h4_txt = "; ".join(
    f"`{c}`: red accuracy under projection {min(r.values()):.2f}–{max(r.values()):.2f} across the ops, `mix` non-red deficit {d:.3f}"
    for c, (r, d) in h4_reads.items()
)
if not task_ok[ex.RECIPE.name] and adopted == ex.RECIPE.name:
    h4: Verdict = Verdict("unscored", "No candidate is feasible, so H4 is not scored.")
elif all(h4_removal_ok.values()) and all(h4_select_ok.values()):
    h4 = Verdict("holds", f"Projection removes *red* on every op at ex-2.2.1's selectivity. {h4_txt}.")
elif all(
    (h4_removal_ok[c] or len(h4_removal_misses[c]) == 1) and (h4_select_ok[c] or h4_select_band[c]) for c in h4_conds
) and any(not h4_removal_ok[c] or not h4_select_ok[c] for c in h4_conds):
    h4 = Verdict(
        "partial",
        f"{h4_txt}. "
        + " ".join(
            [f"Removal misses on `{', '.join(h4_removal_misses[c])}`." for c in h4_conds if h4_removal_misses[c]]
            + [
                f"The deficit sits in the {ex.NONRED_DEFICIT_GATE:g}–{ex.NONRED_DEFICIT_PARTIAL:g} band on `{c}`."
                for c in h4_conds
                if h4_select_band[c]
            ]
        ),
    )
else:
    h4 = Verdict(
        "contrary",
        f"{h4_txt}. "
        + " ".join(
            [
                f"Removal misses on `{', '.join(h4_removal_misses[c])}` for `{c}`."
                for c in h4_conds
                if h4_removal_misses[c]
            ]
            + [
                f"The `mix` deficit on `{c}` is above {ex.NONRED_DEFICIT_PARTIAL:g}."
                for c in h4_conds
                if h4_reads[c][1] > ex.NONRED_DEFICIT_PARTIAL
            ]
        ),
    )

rf"""
# Ex 2.2.3: the multi-op grammar, with *red* anchored again

/// tip |
<!-- tl;dr -->
The grammar grows from one op to six (`mix`, `add`, `screen`, `multiply`, `lighten`, `darken`), and we retrain the control, the ex-2.1.10 recipe, and three proposals from the ex-2.1.11 survey. The recipe carried over as it was. The proposals place *red* with a wider margin but put the axis on the syntax embeddings, so projecting it out breaks most lines with no red; the frozen rule adopted one, but a look with the gates the rule left out kept the recipe. Removal looks partial on the saturating ops, because their answers mostly do not depend on how red the red operand is.
///

## Findings

- [Task cost on the new grammar (H1)](#task-cost-on-the-new-grammar-h1) — {h1.md}
- [Placement under the recipe reproduces (H2)](#placement-under-the-recipe-reproduces-h2) — {h2.md}
- [The plateau from the survey transfers (H3)](#the-plateau-from-the-survey-transfers-h3) — {h3.md} The frozen rule adopts **`{adopted}`**; a [post hoc amendment](#post-hoc-the-selection-rule-with-the-gates-it-left-out) that adds H2's lead gate and H4's selectivity gate, with the tie between the recipe lengths broken at twenty seeds (E6), adopts **`{adopted_post}`**. D2.2 builds on **`{carried}`** by decision: it grades better and leads better at the embedding, with the same H4 read, for half the compute.
- [Suppression transfers (H4)](#suppression-transfers-h4) — {h4.md}

/// admonition | How to read this draft
The hypotheses, the gates, the selection rule, and the method were frozen at commit `5a3233b`, before any anchored run, with the calibration recorded under [the method](#before-the-freeze). Each hypothesis section opens with its frozen prediction; its results follow in place. Anything we thought of after seeing the data is under [Exploratory analyses](#exploratory-analyses), marked as post hoc. The op table and every count in the method are computed from `experiment.py` at render time.
///

## Why this experiment

The claims of D2.2 are about an operation, and the D2.1 grammar has only one: `mix`, spelled `+`. The [D2.2 plan](../d2.2/design.md) makes the operation a variable. But every number D2.1 produced was measured on the one-op grammar: the recipe in [ex-2.1.10](../ex-2.1.10/report.py), the operating-point plateau in the [ex-2.1.11](../ex-2.1.11/report.py) survey, and the suppression figures in [ex-2.2.1](../ex-2.2.1/report.py). The risk table in the plan names this ("recipe is grammar-specific") and sends it here.

So this is a regression check with a decision attached. We measure the same statistics on the same `mix` lines, and we freeze a rule that turns the proposals from the survey into the operating point the anchored-op experiments adopt.

## Conditions

Every condition is trained here at fresh seeds. All but the richer-op arms use the six-op corpus at D2.1's size and epoch count, so the recipe's step count is unchanged.
"""

rf"""
{conditions_md()}

The recipe is the D2.1 primary, `either-t100`. The only thing we changed is the grammar. The short arm of the recipe runs for as long as the proposals do, so H3 compares runs of equal length; the control has a short arm so that H1 can do the same.

The three proposals are the operating point from the survey plus the two promoted trials the survey read as its twin and its knee. `t48` sits within one m_line band of `t00`, with more grading and contrast. `t12` holds the most grading of the promoted set, at a margin the survey could not tell apart from the one for `t00`.

The richer-op arms are un-anchored models on one and three ops at the recipe's step count, in two matchings. The `corpus` arms keep D2.1's line count, and so have more lines per op. The `per-op` arms keep the six-op lines per op, and so have a smaller corpus, repeated over more epochs. Corpus size, lines per op, and step count cannot all be held while the op set changes, and the two matchings put the remaining difference on opposite sides, so the pair lets E4 read in both directions. They serve E4 only.

### The interventions

We score the checkpoints of every candidate through the eval contract in [`sca.intervention`](/src/sca/intervention.py), on the probe lines of each op.
"""

rf"""
- **`projection`** — the primary from ex-2.2.1: the axis projected out at every slice and position, at full strength. Read by H4.
- **Ride-along rows** — `operands` (the projection at the two operand positions only), `shaped` (the shaped suppression from M1, a = {ex.SHAPED["a"]:g}, b = {ex.SHAPED["b"]:g}, p = {ex.SHAPED["p"]:g}), and `ablate` (the axis weights zeroed). Reported without gates, so the intervention-tuning pass the plan schedules has the multi-op figures beside the one-op figures from ex-2.2.1.

## Task cost on the new grammar (H1)
"""

rf"""
**H1.** Anchoring costs nothing on the task, on any op. We score the recipe at both lengths and the three proposals against the control of the same length, op by op, so the gate reads {len(ex.CANDIDATES) * len(ex.OPS)} condition-op comparisons. In each, seed-mean holdout exact-match accuracy is within {ex.TASK_GATE:g} of its control's. Partial: every comparison within {ex.TASK_PARTIAL:g}, or all but one within {ex.TASK_GATE:g} and that one within {ex.TASK_PARTIAL:g}. Contrary: a condition more than {ex.TASK_PARTIAL:g} below its control on some op, which would say that the anchor and reading the op compete with each other. A condition that far *above* its control on some op misses the gate too, and we would report it as an anomaly, since nothing in the design predicts one.

This gate only means something if the controls learn the grammar in the first place. We check that at both lengths in the calibration runs described in the [method](#before-the-freeze), before the freeze, rather than gating it here.

If the recipe misses this gate at full length, H2 goes unscored and the refuted H1 is the finding, as ex-2.1.10 ruled for its primary. H3 still runs, since its selection rule already requires the task gate of every candidate, and H4 is then scored on the adopted point alone. If no candidate is feasible, H4 goes unscored too.
"""


def _table() -> str:
    head = ["condition"] + [f"{op} EM ↑" for op in ex.OP_NAMES]
    rows, refs = [], set()
    for i, c in enumerate(ex.CONDITIONS):
        rows.append([c.name] + [span2(res.em(c.name, op)) if op in c.ops else "·" for op in ex.OP_NAMES])
        if c not in ex.CANDIDATES:
            refs.add(i)
    caption = f"""
    Holdout exact match per op, by condition: the seed mean with half the seed range beside it. The greyed rows are the two controls ({ex.EPOCHS} and {ex.EPOCHS_SHORT} epochs) and the richer-op arms, which carry no gate; a dot means the arm never trained on that op.
    """
    return table_html(head, rows, caption, ref_rows=frozenset(refs))


_table()

# %%


def _table() -> str:
    head = ["condition", "control"] + [f"{op} gap" for op in ex.OP_NAMES] + ["worst"]
    rows = []
    for c in ex.CANDIDATES:
        g = [h1_gaps[c.name, op] for op in ex.OP_NAMES]
        worst = max(g, key=abs)
        rows.append(
            [c.name, control_of(c).name]
            + [f"<b>{v:+.4f}</b>" if abs(v) <= ex.TASK_GATE else f"{v:+.4f}" for v in g]
            + [f"<b>{worst:+.4f}</b>" if abs(worst) <= ex.TASK_GATE else f"{worst:+.4f}"]
        )
    caption = f"""
    The H1 read: each scored condition's seed-mean holdout exact match minus that of the control of the same length, per op. Bold marks a gap within the gate of {ex.TASK_GATE:g}; the partial band runs to {ex.TASK_PARTIAL:g}.
    """
    return table_html(head, rows, caption)


_table()

# %%


@memo
@themed(
    name="h1-holdout-dots",
    alt_text="""
        Six small panels, one per op, each listing the seven conditions down the side with a dot at the seed-mean holdout exact match and a bar for the seed range. A grey strip behind each row marks the gate around that row's control. Every dot sits inside its strip, near the right edge at accuracy 1.
    """,
    caption=f"""
        **Holdout exact match per op, against the gate.** One panel per op; conditions down the side, in their own ink. Each dot is the seed mean, the bar the seed range. The grey strip behind a row is ±{ex.TASK_GATE:g} around the seed-mean accuracy of the control of the same length, so a dot outside its strip misses the H1 gate; the two control rows carry their own strips for reference.
    """,
)
def h1_holdout_dots(res: Results) -> plt.Figure:
    conds = [c for c in ex.CONDITIONS if c not in ex.RICHER_OP_ARM and c not in ex.ADDENDUM]
    em = {(c.name, op): res.em(c.name, op) for c in conds for op in ex.OP_NAMES}
    ctrl = {c.name: {op: float(res.em(control_of(c).name, op).mean()) for op in ex.OP_NAMES} for c in conds}
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 3.6), sharex=True, sharey=True, layout="constrained")
    axes = cast(AxesGrid, axes)
    strip = light_dark("#00000014", "#ffffff1a")
    ys = np.arange(len(conds))[::-1]
    for ax, op in zip([a for row in axes for a in row], ex.OP_NAMES, strict=True):
        for y, c in zip(ys, conds, strict=True):
            lo, hi = ctrl[c.name][op] - ex.TASK_GATE, ctrl[c.name][op] + ex.TASK_GATE
            ax.fill_betweenx([y - 0.42, y + 0.42], lo, hi, color=strip, lw=0, zorder=1)
            v = em[c.name, op]
            ax.plot([v.min(), v.max()], [y, y], color=ink(c.name), lw=1.2, zorder=2, solid_capstyle="round")
            ax.plot(v.mean(), y, "o", ms=3.6, color=ink(c.name), zorder=3)
        ax.set_title(op, fontsize=9)
        ax.set_yticks(ys, [c.name for c in conds], fontsize=7)
        ax.set_xlim(0.9, 1.005)
        ax.set_xticks([0.9, 0.95, 1.0])
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.2)
    return fig


h1_holdout_dots(res)

# %%


@memo
@themed(
    name="h1-val-loss",
    alt_text="""
        One log-scale line chart of validation loss over 3,300 training steps. Five thin blue lines for the recipe and five grey lines for the control fall together from about 3 to below 0.1 and stay interleaved throughout; a faint band of ghost lines from the D2.1 runs lies behind them on the same axis, reaching a lower floor.
    """,
    caption="""
        **Validation loss over training.** Five thin lines per condition on the step axis: the control in grey and the recipe in blue. The ghost lines behind them are the ex-2.1.10 primary (nine seeds) and its control (three), which trained on the one-op grammar at the same step count; their floor sits lower because their corpus held about seventeen copies of each distinct line, where this one holds most lines once. Log scale.
    """,
)
def h1_val_loss(res: Results) -> plt.Figure:
    spe = ex.steps_per_epoch()
    fresh = {c.name: res.val_loss(c.name) for c in (ex.CONTROL, ex.RECIPE)}
    ref = np.array([c["val_loss"] for c in res.ref_cells()], float)
    ref_ctrl = np.array([c["val_loss"] for c in res.ref_cells(EX2110_CONTROL)], float)
    fig, ax = plt.subplots(figsize=(6.4, 2.8), layout="constrained")
    ghost = ink("ex-2.1.10")
    x_ref = (np.arange(ref.shape[1]) + 1) * spe
    for row in ref_ctrl:
        ax.plot(x_ref, row, color=ghost, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for row in ref:
        ax.plot(x_ref, row, color=ghost, lw=0.8, zorder=1)
    for cond, rows in fresh.items():
        x = (np.arange(rows.shape[1]) + 1) * spe
        for row in rows:
            ax.plot(x, row, color=ink(cond), lw=0.8, zorder=3 if cond == "recipe" else 2, alpha=0.85)
        ax.plot([], [], color=ink(cond), lw=1.4, label=cond)
    ax.plot([], [], color=ghost, lw=1.2, label="ex-2.1.10 primary")
    ax.set_yscale("log")
    ax.set_xlim(0, ex.RECIPE.steps)
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel("validation loss (nats per token)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    return fig


h1_val_loss(res)

rf"""
**H1, {h1.status}.** {h1.line}

## Placement under the recipe reproduces (H2)
"""

# REVIEW: H2's contrary clause read a latch or containment miss as the op word alone,
# while the paragraph under it names exposure as a second reading; the clause now says
# "something in the new grammar" and the paragraph carries both. The added arithmetic
# (6.4% of all pairs against 6.3% of the on-grid ones are red by dose) is what says the
# anchor's own exposure is near unchanged. Verify: `line_counts` against ex-2.2.1's
# 365 red of 5,832 mix probe lines.
# The gate list is ordered seed-mean criteria first, then the two per-run ones
# (retention, latch), so the "unless the criterion says otherwise" carries; ex-2.1.10
# aggregated the same way. Don't reword to imply every criterion is scored at every run.
rf"""
**H2.** The ex-2.1.10 recipe places *red* on the new grammar as it did on the old one. On the `mix` probe lines, with five fresh seeds, all of the ex-2.1.10 placement gates hold. Each statistic is a seed mean unless the criterion says otherwise: containment $\bar\alpha$ at op1 at most {ex.MEAN_ALIGN_GATE:g}; a leading softmin weight of the red group at the embedding of at least {ex.LEAD_GATE:g}; contrast at least {ex.CONTRAST_GATE:g}; grading $r^2$ no more than {ex.GRADE_R2_DROP:g} below the {ex.REF_R2_SIM:.3f} the survey measured on this recipe; m_line at least {ex.MARGIN_RATIO:g} of its ex-2.1.10 value of {ex.REF_M_LINE:.3f}; and, read per run rather than on the seed mean, retention at least {ex.RETENTION_GATE:g} of the running peak for every run whose peak reaches {ex.RETENTION_FLOOR:g}, and no latched run.

Partial: every criterion holds except one of m_line in the {ex.MARGIN_PARTIAL:g}–{ex.MARGIN_RATIO:g} band or contrast in the {ex.CONTRAST_PARTIAL:g}–{ex.CONTRAST_GATE:g} band. Contrary: a latch or a containment miss. That would mean something in the new grammar changed what the pull finds. If instead the margin fell below the partial band while the other criteria held, the placement would be there but weaker, and the H3 selection would lean toward the proposals.

Two things in the new grammar could do that, and we read a miss against both. The first is the op word. The labeller never sees it, and reads the same two operand roles on every op.

The second is exposure. Only a sixth of the corpus is `mix`, and those lines are drawn from every pair rather than from the on-grid ones. So over a run the model sees a given `mix` probe pair about a fiftieth as often as in D2.1, and never sees most of them (see [the corpus](#the-corpus)). What the anchor sees is closer to unchanged. The corpus is the same size, and the share of lines the labeller fires on follows the operand redness: 6.4% of all pairs, against 6.3% of the on-grid ones in D2.1. So the exposure that fell is the exposure of the task, counted per pair, rather than the exposure of the anchor, counted per step.

Two measurements separate the readings: the task accuracy of the control on `mix` (H1), and the per-op placement in E1.
"""


def _table() -> str:
    ref_ret = res.ref_retention()
    sv_seeds, sv = ex.SURVEY_RECIPE["recipe"]

    def band(stat: str, n_b: int) -> str:
        return f"{ex.equiv_band(stat, 5, n_b):.3f}" if stat in ex.NOISE_RUN else "—"

    def row(
        name: str,
        fresh: np.ndarray,
        ref: np.ndarray | None,
        ref_note: str,
        gate: str,
        ok: bool,
        stat: str,
        n_b: int,
        fmt: str = ".3f",
    ) -> list[str]:
        v = f"<b>{span2(fresh, fmt)}</b>" if ok else span2(fresh, fmt)
        if ref is None:
            return [name, v, "—", "—", "—", "—", gate]
        d = fresh.mean() - ref.mean()
        b = ex.equiv_band(stat, 5, n_b) if stat in ex.NOISE_RUN else float("nan")
        resolved = "" if np.isnan(b) else (" ✓" if abs(d) > b else " ·")
        return [name, v, span2(ref, fmt) if len(ref) > 1 else f"{ref.mean():{fmt}}", ref_note, f"{d:+{fmt}}{resolved}", band(stat, n_b), gate]  # fmt: skip

    c = ex.RECIPE.name
    m = res.stat(c, "m_line")
    rows = [
        row(
            "m_line ↑",
            m,
            res.ref_stat("m_line"),
            "9",
            f"≥ {ex.MARGIN_RATIO * ex.REF_M_LINE:.3f}",
            m.mean() >= ex.MARGIN_RATIO * ex.REF_M_LINE,
            "m_line",
            9,
        ),  # fmt: skip
        row("m_span", res.stat(c, "m_span"), res.ref_stat("m_span"), "9", "—", False, "m_span", 9),
        row(
            "containment ᾱ ↓",
            res.stat(c, "alpha_op1"),
            res.ref_stat("alpha_mean_op1"),
            "9",
            f"≤ {ex.MEAN_ALIGN_GATE:g}",
            res.stat(c, "alpha_op1").mean() <= ex.MEAN_ALIGN_GATE,
            "alpha_op1",
            9,
        ),  # fmt: skip
        row(
            "lead weight at emb ↑",
            res.stat(c, "lead_emb"),
            None,
            "",
            f"≥ {ex.LEAD_GATE:g}",
            res.stat(c, "lead_emb").mean() >= ex.LEAD_GATE,
            "lead",
            9,
            ".2f",
        ),  # fmt: skip
        row(
            "contrast ↑",
            res.stat(c, "contrast"),
            np.array([sv["contrast"]]),
            f"{sv_seeds} (survey)",
            f"≥ {ex.CONTRAST_GATE:g}",
            res.stat(c, "contrast").mean() >= ex.CONTRAST_GATE,
            "contrast",
            sv_seeds,
        ),  # fmt: skip
        row(
            "grading r² ↑",
            res.stat(c, "r2_sim"),
            np.array([ex.REF_R2_SIM]),
            f"{sv_seeds} (survey)",
            f"≥ {ex.REF_R2_SIM - ex.GRADE_R2_DROP:.3f}",
            res.stat(c, "r2_sim").mean() >= ex.REF_R2_SIM - ex.GRADE_R2_DROP,
            "r2_sim",
            sv_seeds,
        ),  # fmt: skip
        row(
            "retention, min over runs ↑",
            res.stat(c, "retention"),
            ref_ret,
            "9",
            f"≥ {ex.RETENTION_GATE:g}",
            res.stat(c, "retention").min() >= ex.RETENTION_GATE,
            "retention",
            9,
        ),  # fmt: skip
        row(
            "non-red op1 weight, max over runs ↓",
            res.nonred_profile(c)[:, 1:, 0].mean(axis=1),
            None,
            "",
            f"≤ {ex.LATCH_PI:g}",
            not res.latched(c).any(),
            "latch",
            9,
            ".2f",
        ),  # fmt: skip
    ]
    caption = """
    The placement statistics of the recipe on the <code>mix</code> probe lines, five fresh seeds, beside the D2.1 reference. The reference is the ex-2.1.10 primary's nine seeds where its stored metrics carry the statistic, and the survey's three-seed re-run of the same recipe (its <code>ref</code> arm) for contrast and grading, which ex-2.1.10 did not store per run; the lead weight and the latch have no stored reference. Δ is fresh minus reference; a ✓ beside it means the difference clears the band (2σ√(1/5 + 1/n) with the frozen per-run σ), a dot that it does not. Bold marks a value inside its gate. Retention and the latch are read per run, so their rows print the worst run.
    """
    head = ["statistic", "fresh (5 seeds)", "D2.1 reference", "ref. seeds", "Δ (✓ clears band)", "band", "gate"]
    return table_html(head, rows, caption)


_table()

# %%


@memo
@themed(
    name="h2-m-line-trajectory",
    alt_text="""
        A line chart of m_line over 3,300 training steps. Five thin blue lines for the recipe rise steeply in the first few hundred steps, level off, and dip slightly after step 3,000; nine faint ghost lines from the D2.1 runs follow the same path behind them. Five grey lines for the control lie flat near zero.
    """,
    caption=f"""
        **m_line over training.** The trajectory instrument reads one line per color on the <code>mix</code> probe set, every {ex.TRAJ_STRIDE} steps. Five thin blue lines are the recipe's fresh seeds; the ghost lines behind them are the nine seeds of the ex-2.1.10 primary on the one-op grammar, on the same step axis. The grey lines are the un-anchored control. The dip over the last tenth is the anchor anneal.
    """,
)
def h2_m_line_trajectory(res: Results) -> plt.Figure:
    spe = ex.steps_per_epoch()
    fresh = res.trajectory(ex.RECIPE.name, "m_line")
    fresh_x = res.trajectory(ex.RECIPE.name, "epoch") * spe
    ctrl = res.trajectory(ex.CONTROL.name, "m_line")
    ctrl_x = res.trajectory(ex.CONTROL.name, "epoch") * spe
    ref = res.ref_trajectory("m_line")
    ref_x = res.ref_trajectory("step")
    fig, ax = plt.subplots(figsize=(6.4, 2.8), layout="constrained")
    for x, y in zip(ref_x, ref, strict=True):
        ax.plot(x, y, color=ink("ex-2.1.10"), lw=1.0, zorder=1)
    for x, y in zip(ctrl_x, ctrl, strict=True):
        ax.plot(x, y, color=ink("control"), lw=0.8, zorder=2, alpha=0.8)
    for x, y in zip(fresh_x, fresh, strict=True):
        ax.plot(x, y, color=ink("recipe"), lw=0.9, zorder=3, alpha=0.9)
    for c, name in (("recipe", "recipe"), ("control", "control"), ("ex-2.1.10", "ex-2.1.10 primary")):
        ax.plot([], [], color=ink(c), lw=1.4, label=name)
    ax.set_xlim(0, ex.RECIPE.steps)
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel("m_line", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    return fig


h2_m_line_trajectory(res)

# %%


@memo
@themed(
    name="h2-group-profiles",
    alt_text="""
        Ten small panels in two columns, G1 (op1 drew) and G2 (op2 drew), five residual slices per column with the embedding at the bottom, each spanning the roles op1, op, op2, and equals. In the G1 column the weight sits at op1 on every slice; in the G2 column it sits at op2. Five per-seed hairlines hug the mean in every panel.
    """,
    caption=rf"""
        **Per-group softmin profiles for the recipe on the `mix` lines.** Columns are the label groups (G1 weights probe lines by P(only op1 drew), G2 by P(only op2 drew)); rows are residual slices with the embedding at the bottom; the shaded smooth-step is the seed-mean softmin weight over the span roles, with a hairline per seed. On the embedding row, the caret on each column's outer axis marks the lead gate ({ex.LEAD_GATE:g}; uniform is 0.25), and the number above each group's own operand is the seed-mean weight there. Deep-slice op2 contrast {h2_contrast.mean():+.2f} <span class='range'>±{h2_contrast.std(ddof=1):.2f}</span> (gate ≥ {ex.CONTRAST_GATE:g}). The layout is ex-2.1.10's, whose primary put 0.77 on the own operand at the embedding and 0.9 or more at depth.
    """,
)
def h2_group_profiles(res: Results) -> plt.Figure:
    from matplotlib.layout_engine import ConstrainedLayoutEngine

    pg = res.arr(ex.RECIPE.name, "eval", f"{ex.PRIMARY_OP.name}/w_group")  # (seeds, group, slices, roles)
    mean = pg.mean(axis=0)
    x = np.arange(ex.SPAN)
    color = ink("recipe")
    hair = light_dark("#00000055", "#ffffff55")
    gate = light_dark("#555", "#bbb")
    fig, axes = plt.subplots(5, 2, figsize=(3.1, 3.9), sharex=True, sharey=True)
    axes = cast(AxesGrid, axes)
    engine = fig.get_layout_engine()
    assert isinstance(engine, ConstrainedLayoutEngine)
    engine.set(hspace=0, h_pad=0.01, wspace=0.05)
    for col, gname in enumerate(("G1: op1 drew", "G2: op2 drew")):
        for row in range(5):
            sl = 4 - row
            ax = axes[row][col]
            smooth_step_area(ax, x, mean[col, sl], ramp=0.5, color=color, alpha=light_dark(0.22, 0.28))
            smooth_step(ax, x, mean[col, sl], ramp=0.5, color=color, lw=1.5)
            for sp in pg:
                smooth_step(ax, x, sp[col, sl], ramp=0.5, color=hair, lw=0.5)
            ax.set(ylim=(-0.08, 1.08), xlim=(-0.4, ex.SPAN - 0.6), yticks=[0.0, 0.5, 1.0])
            ax.spines[:].set_visible(False)
            ax.grid(axis="y", which="major", c="#888", alpha=0.2)
            ax.tick_params(axis="x", length=0, labelbottom=False)
            ax.tick_params(axis="y", left=True, right=True, direction="in", labelleft=False, labelright=False)
            if col == 0:
                ax.set_ylabel("emb" if sl == 0 else f"{sl}", fontsize=7.5)
        axes[0][col].set_title(gname, fontsize=8.5, color=color)
        own = 0 if col == 0 else 2
        emb = axes[4][col]
        emb.plot(
            col,
            ex.LEAD_GATE,
            marker=9 if col == 0 else 8,
            ms=3,
            color=gate,
            clip_on=False,
            zorder=6,
            transform=emb.get_yaxis_transform(),
        )
        val = float(mean[col, 0, own])
        emb.annotate(
            f"{val:.2f}",
            (own + 0.35 if col == 0 else own, min(val + 0.14, 0.92)),
            ha="left" if col == 0 else "center",
            fontsize=7,
            color=light_dark("#444", "#bbb"),
        )
        axes[4][col].set_xticks(x, ROLES)
        axes[4][col].tick_params(axis="x", labelbottom=True, labelsize=7.5)
    axes[4][1].tick_params(axis="y", labelright=True, labelsize=6.5, pad=2)
    return fig


h2_group_profiles(res)

rf"""
**H2, {h2.status}.** {h2.line}

## The plateau from the survey transfers (H3)
"""

rf"""
**H3.** The plateau found in the survey belongs to the recipe rather than to the one-op grammar. At least one proposal is feasible at fresh seeds on the new grammar, and its seed-mean m_line on the `mix` probe lines exceeds `recipe-short` (the recipe at the same step count) by more than one band. Feasibility is read as the survey read it, with the constraints listed once in the selection rule below. The grading margin is one of them, and so is a floor on the worst seed: every run's m_line at least {ex.MARGIN_PARTIAL:g} of its ex-2.1.10 value, since the bands use the frozen per-run σ and a candidate's own spread would otherwise not count against it.

Partial: a proposal is feasible and its m_line sits within a band of `recipe-short`; or a proposal clears every survey constraint and exceeds `recipe-short` by more than a band, but misses the grading margin. Contrary: no proposal is feasible, or every feasible proposal sits more than a band below `recipe-short`. Either would say the plateau was specific to the one-op grammar, and the recipe would carry D2.2. A proposal that misses only the per-run floor is reported as unstable at fresh seeds rather than as weak, with the seed range printed.

**The selection rule.** {ex.SELECTION_RULE} The Findings line for this section names the adopted point, and it lands in `experiment.py` for the D2.2 experiments that follow.

**Winner's curse.** The numbers from the survey are proposals. So the table prints the one-op value for each candidate beside the fresh one, with the per-run σ behind the bands and the band a difference has to clear. The survey re-ran the recipe at both lengths, and those runs sit in the same table, since they are the one-op values the fresh recipe arms are read against.
"""


def _table() -> str:
    head = ["candidate", "statistic", "survey", "seeds", "fresh (5 seeds)", "Δ (✓ clears band)", "σ per run", "band"]
    rows = []
    for c in ex.CANDIDATES:
        seeds, sv = res.survey_point(c)
        for i, stat in enumerate(SURVEY_STATS):
            fresh = res.em(c.name, ex.PRIMARY_OP.name) if stat == "holdout_em" else res.stat(c.name, stat)
            sv_v = sv[stat] if sv and stat in sv else None
            b = ex.equiv_band(stat, 5, seeds) if stat in ex.NOISE_RUN else None
            d = (
                "—"
                if sv_v is None
                else f"{fresh.mean() - sv_v:+.3f}"
                + ("" if b is None else (" ✓" if abs(fresh.mean() - sv_v) > b else " ·"))
            )
            rows.append(
                [
                    f"<b><code>{c.name}</code></b>" if i == 0 else "",
                    stat,
                    f"{sv_v:.3f}" if sv_v is not None else "—",
                    str(seeds),
                    span2(fresh),
                    d,
                    f"{ex.NOISE_RUN[stat]:.4f}" if stat in ex.NOISE_RUN else "—",
                    f"{b:.3f}" if b is not None else "—",
                ]
            )
    caption = """
    Each candidate's one-op survey numbers beside the fresh ones on the six-op grammar. The survey column is the trial's five-seed mean (for the proposals) or the survey's own three-seed re-run of the recipe at that length; fresh is the seed mean here with half the seed range. Δ is fresh minus survey, with a ✓ where the difference clears the band for those seed counts and a dot where it does not. σ is the frozen per-run spread from ex-2.1.10 behind every band. <code>holdout_em</code> is read on <code>mix</code>.
    """
    return table_html(head, rows, caption)


_table()

# %%


@memo
@themed(
    name="h3-survey-vs-fresh",
    alt_text="""
        Four small panels, one per statistic, listing the five candidates down the side. In each row a hollow marker is the survey value and a filled dot with a range bar is the fresh five-seed read; a grey strip around the survey value is the band. The fresh m_line dots sit left of their survey markers on every proposal.
    """,
    caption="""
        **Survey against fresh, per candidate.** The same numbers as the table above, drawn: the hollow marker is the survey value, the filled dot the fresh seed mean with a bar for the seed range, and the grey strip the band around the survey value for those seed counts. A dot outside its strip is a difference the resolution rule counts.
    """,
)
def h3_survey_vs_fresh(res: Results) -> plt.Figure:
    stats = [k for k in SURVEY_STATS if k in ex.NOISE_RUN]
    fig, axes = plt.subplots(1, len(stats), figsize=(7.6, 1.8), sharey=True, layout="constrained")
    axes = cast(AxesRow, axes)
    strip = light_dark("#00000014", "#ffffff1a")
    ys = np.arange(len(ex.CANDIDATES))[::-1]
    for ax, k in zip(axes, stats, strict=True):
        for y, c in zip(ys, ex.CANDIDATES, strict=True):
            n, sv = res.survey_point(c)
            v = res.em(c.name, ex.PRIMARY_OP.name) if k == "holdout_em" else res.stat(c.name, k)
            if sv and k in sv:
                b = ex.equiv_band(k, 5, n)
                ax.fill_betweenx([y - 0.42, y + 0.42], sv[k] - b, sv[k] + b, color=strip, lw=0, zorder=1)
                ax.plot(sv[k], y, "o", ms=4.2, mfc="none", mec=ink(c.name), mew=1.0, zorder=3)
            ax.plot([v.min(), v.max()], [y, y], color=ink(c.name), lw=1.2, zorder=2, solid_capstyle="round")
            ax.plot(v.mean(), y, "o", ms=3.6, color=ink(c.name), zorder=4)
        ax.set_yticks(ys, [c.name for c in ex.CANDIDATES], fontsize=7)
        ax.set_title(k, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.2)
    return fig


h3_survey_vs_fresh(res)

# %%


def _table() -> str:
    checks = list(next(iter(feas.values())))
    head = ["candidate", *checks, "feasible", f"contrast ≥ {ex.CONTRAST_GATE:g}", "m_line ↑", "grading r² ↑"]
    rows = []
    for c in ex.CANDIDATES:
        f = feas[c.name]
        ok = all(f.values())
        full = ok and res.stat(c.name, "contrast").mean() >= ex.CONTRAST_GATE
        name = c.name + (" ★" if c.name == adopted else "")
        rows.append(
            [name, *["✓" if v else "✗" for v in f.values()], "<b>✓</b>" if ok else "✗", "✓" if full else "✗", span2(res.stat(c.name, "m_line")), span2(res.stat(c.name, "r2_sim"))]
        )  # fmt: skip
    caption = f"""
    Feasibility, as the selection rule reads it. Seed-mean checks: task within {ex.TASK_GATE:g} of the same-length control on every op; containment ᾱ ≤ {ex.MEAN_ALIGN_GATE:g}; contrast ≥ {ex.CONTRAST_PARTIAL:g}; grading r² at least {ex.GRADE_MARGIN_SD:g}σ ({ex.GRADE_MARGIN_SD * ex.NOISE_RUN["r2_sim"]:.3f}) above its floor of {h3_r2_floor:.3f}, the full-length recipe's fresh value less {ex.GRADE_R2_DROP:g}. Per-run checks: retention ≥ {ex.RETENTION_GATE:g} of the running peak for every run whose peak reaches {ex.RETENTION_FLOOR:g}; no run with more than {ex.LATCH_PI:g} of the non-red group's weight on op1; every run's m_line at least {ex.MARGIN_PARTIAL:g} of {ex.REF_M_LINE:.3f}. The selection then requires contrast at the full gate and takes the highest m_line, ties within one band ({ex.equiv_band("m_line"):.3f}) going to the larger grading margin. ★ marks the adopted point: {adoption_note}.
    """
    return table_html(head, rows, caption)


_table()

# %%


@memo
@themed(
    name="h3-margin-vs-grading",
    alt_text="""
        A scatter of m_line against grading r squared for the five candidates, each in its own color: five small filled dots per candidate with a ring at the seed mean, and one hollow marker for the survey's value. A grey horizontal strip marks the band around recipe-short, and a dotted vertical line marks the grading floor.
    """,
    caption=f"""
        **m_line against grading r² for the five candidates.** Filled dots are single seeds, the ring is the seed mean, and the hollow diamond is the same candidate's one-op survey value. The grey strip is one band ({h3_band:.3f}) either side of <code>recipe-short</code>'s seed mean, which the H3 verdict reads the proposals against; the dotted line is the grading floor of the selection rule ({h3_r2_floor:.3f}, before its 1σ margin). The adopted point is labelled ★.
    """,
)
def h3_margin_vs_grading(res: Results, adopted: str) -> plt.Figure:
    pts = {c.name: (res.stat(c.name, "r2_sim"), res.stat(c.name, "m_line")) for c in ex.CANDIDATES}
    sv = {}
    for c in ex.CANDIDATES:
        _, s = res.survey_point(c)
        if s:
            sv[c.name] = (s["r2_sim"], s["m_line"])
    short = float(res.stat(ex.RECIPE_SHORT.name, "m_line").mean())
    band = ex.equiv_band("m_line", 5, 5)
    r2_floor = float(res.stat(ex.RECIPE.name, "r2_sim").mean()) - ex.GRADE_R2_DROP
    fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
    ax.axhspan(short - band, short + band, color=light_dark("#00000012", "#ffffff18"), lw=0, zorder=0)
    ax.axvline(r2_floor, color=light_dark("#888", "#999"), lw=0.8, ls=(0, (1, 2)), zorder=1)
    for name, (r2, m) in pts.items():
        color = ink(name)
        ax.scatter(r2, m, s=12, color=color, zorder=3, lw=0)
        ax.scatter(r2.mean(), m.mean(), s=90, facecolor="none", edgecolor=color, lw=1.3, zorder=4)
        if name in sv:
            ax.scatter(*sv[name], s=40, marker="D", facecolor="none", edgecolor=color, lw=1.0, zorder=4)
        ax.annotate(
            name + (" ★" if name == adopted else ""),
            (r2.mean(), m.mean()),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )
    ax.set_xlabel("grading r² (op1 alignment vs sim^1.5)", fontsize=8)
    ax.set_ylabel("m_line, mix probe lines", fontsize=8)
    ax.tick_params(labelsize=7)
    return fig


h3_margin_vs_grading(res, adopted)

# %%


@memo
@themed(
    name="h3-grading-clouds",
    alt_text="""
        Five small scatter panels, one per candidate. Each has 216 points, one per grid color drawn in that color, with similarity to red along the bottom and the alignment at op1 up the side. On every panel the points rise from left to right, tightest on the recipe and looser on the proposals, with the reds at the top right.
    """,
    caption="""
        **What grading looks like.** One panel per candidate: each point is one of the 216 grid colors, drawn in its own color, at its similarity-to-red target (sim^1.5) along the bottom and its seed-mean alignment at op1 up the side, averaged over the four post-attention slices. The r² in each title is the squared correlation of these two axes on the seed-mean cloud; the tables report the per-seed r² averaged, which is a little lower. A cloud that rises in a line is graded; one that steps from a floor to a ceiling is thresholded.
    """,
)
def h3_grading_clouds(res: Results) -> plt.Figure:
    stride = ex.N_PROBE  # every color sits at op1 once per partner; op1 precedes the partner, so one read per color
    x = ex.SIM_TARGET

    def op1(cond: str) -> np.ndarray:
        """(seeds, colors) mean post-attention alignment at op1 on the `mix` lines."""
        a = res.arr(cond, "eval", f"{ex.PRIMARY_OP.name}/alpha_lines").astype(np.float32)  # (seeds, slices, N, T)
        return a[:, 1:, ::stride, 0].mean(axis=1)

    fig, axes = plt.subplots(1, len(ex.CANDIDATES), figsize=(7.6, 1.9), sharex=True, sharey=True, layout="constrained")
    axes = cast(AxesRow, axes)
    for ax, c in zip(axes, ex.CANDIDATES, strict=True):
        y = op1(c.name).mean(axis=0)
        ax.scatter(x, y, s=9, c=ex.GRID_RGB, edgecolors=light_dark("#00000033", "#ffffff44"), linewidths=0.3)
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        ax.set_title(f"{c.name}  r² {r2:.2f}", fontsize=8.5, color=ink(c.name))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("α at op1", fontsize=7.5)
    axes[len(axes) // 2].set_xlabel("similarity to red, sim^1.5", fontsize=7.5)
    return fig


h3_grading_clouds(res)

rf"""
**H3, {h3.status}.** {h3.line}

### Post hoc: the selection rule, with the gates it left out

This is a deviation, written after the numbers were read. The selection rule above was frozen with two holes in it, and the point it adopts falls through both.

The first hole is H2's lead gate. Feasibility carries every H2 gate except the lead weight of the red group at the embedding, which H2 gates at {ex.LEAD_GATE:g}; per candidate that weight is {post_hoc_leads}. Add the gate, and the highest feasible m_line is `{post_hoc_pick_lead}`.

The second hole is H4. The rule ranks on placement and never looks at what the plain projection costs, and on every proposal that cost is most of the non-red `mix` lines: the deficit under `projection` is {post_hoc_costs}, against H4's gate of {ex.NONRED_DEFICIT_GATE:g}. The cost sits on the syntax embeddings (E2, E7), and the operand-only edit routes around it, but an operating point that needs a routed edit before its first intervention is a poor base for the anchored-op experiments. Add H4's selectivity gate as well, and only the two recipe lengths remain; at five seeds they sit within a band of each other, and E6's twenty-seed read breaks the tie: {post_hoc_tie_note}.

The amended rule adopts **`{adopted_post}`**. The frozen H3 verdict stands as written, and the proposals keep their reads: they place *red* with a larger margin, and they pay for it on the syntax embeddings. Whether that cost can be removed rather than routed around is the untied-readout question in the Discussion.

**D2.2 builds on `{carried}`, by decision rather than by rule.** Both rules rank on margin, and at twenty seeds the full recipe leads on margin by {-post_hoc_margin_diff:.3f}, about two bands. The short arm is ahead on the rest of what the anchored-op experiments will lean on: it grades better (r² {post_hoc_r2[carried]:.3f} against {post_hoc_r2[adopted_post]:.3f}), puts more of the pull on the drawing operand at the embedding ({post_hoc_lead_emb[carried]:.2f} against {post_hoc_lead_emb[adopted_post]:.2f}), has a little more contrast, gives the same H4 read, and costs half as much to train. Those experiments read the grading and the operand lead more than they read the margin, so the graded point is the better base. The decision was taken on 2026-09-10, after E6 was read, and this paragraph is its record.

## Suppression transfers (H4)
"""

rf"""
**H4.** Projecting the axis out removes *red* on every op, at the selectivity ex-2.2.1 reached. On the recipe and on the adopted point, under `projection`, seed-mean accuracy on the red lines of each op falls to at most {ex.RED_ACC_GATE:g}, and the seed-mean non-red deficit on the `mix` lines is at most {ex.NONRED_DEFICIT_GATE:g}. Partial: exactly one clause holds, or removal holds on all but one op, or the deficit sits in the {ex.NONRED_DEFICIT_GATE:g}–{ex.NONRED_DEFICIT_PARTIAL:g} band.

The deficit gate is the band ex-2.2.1 landed in (0.024, partial at its 0.02 gate), and we report the 0.02 read beside it. That cost came from the syntax embeddings, and E2 reads the embeddings of the six op words.

Contrary on removal: red accuracy near the clean value on one of the new ops. That would say the model reads *red* off the axis on lines of that op, and the per-op alignment in E1 should show it first. Contrary on selectivity: a deficit above {ex.NONRED_DEFICIT_PARTIAL:g}. That would send the intervention-tuning pass to the `operands` and `shaped` rows before anything is anchored.
"""

removal_table(res, ex.RECIPE.name)

# %%

removal_table(res, adopted) if adopted != ex.RECIPE.name else ""

# %%

selectivity_table(res, ex.RECIPE.name)

# %%

selectivity_table(res, adopted) if adopted != ex.RECIPE.name else ""

# %%


def _table() -> str:
    iv = ex.PROJECTION.name
    rows = []
    for c in h4_conds:
        for key, title in (("argmax_dist", "decoded answer"), ("expected_dist", "expected")):
            for pas, ivn in (("projection", iv), ("clean", None)):
                rows.append(
                    [f"{c}, {title}, {pas}"] + [span2(res.score(c, op, ivn, key, "red"), ".2f") for op in ex.OP_NAMES]
                )
    ref = res.ex221_stat(EX221_PROJECTION, "guess_dist_red", None)
    head = ["condition, distance, pass"] + [f"{op} ↓" for op in ex.OP_NAMES]
    caption = f"""
    Answer distance on the red lines under <code>projection</code>, in the unit cube. The decoded row is the distance from the argmax answer to the true one; the expected row takes the expectation of that distance under the whole answer distribution over the color vocabulary. Each value is the seed mean with half the seed range; the clean pass sits under each projected row. On ex-2.2.1's one-op grammar the decoded distance under the projection was {ref.mean():.2f} <span class='range'>±{(ref.max() - ref.min()) / 2:.2f}</span>; a one-level step on this grid is {1 / (len(ex.LEVELS) - 1):.2f}.
    """
    return table_html(head, rows, caption)


_table()

# %%

h4_arr = {
    op: res.score(ex.RECIPE.name, op, ex.PROJECTION.name, "q99_write_nonred", None).mean(0)
    / np.arcsin(
        np.array([r["ops"][op]["clean"]["alpha_q99_nonred"] for r in res.scored(ex.RECIPE.name)], float).mean(0)
    )
    for op in ex.OP_NAMES
}
h4_over = {op: int((h4_arr[op][1:] > 1).sum()) for op in ex.OP_NAMES}


@memo
@themed(
    name="h4-write-bound-maps",
    alt_text="""
        A grid of thirty small panels, five rows for the residual slices with the embedding at the bottom and six columns for the ops, each over the six token positions, drawn around a zero line. The shaded band spans the clean alignment's two signed tails over the non-red lines; at the color positions it straddles zero, leaning negative at the embedding and positive on op2 in the deeper slices, while at the op word and the equals sign it sits wholly above zero and pinches to a line at the embedding, where every non-red line shares one token. The dashed pair is the arriving alignment's magnitude envelope under projection, mirrored about zero. The six columns look alike.
    """,
    caption=rf"""
        **The bound and the write, per site and per op, for the recipe under <code>projection</code>, signed.** For each op's probe lines, at each (slice, position), seed mean. The shaded band runs from the 1st to the 99th percentile of the signed clean alignment over the non-red lines; the scorer's bound is the arcsine of the 99th-percentile |α|, which is at least the further edge and equal to it where one tail holds the extreme, so this is close to the map the earlier version drew folded onto one side. The dashed pair is the 99th-percentile |α| arriving at the operator, whose arcsine is the write, mirrored about zero because the scorer keeps only its magnitude. The embedding row matches the clean band by construction. One row per slice with the embedding at the bottom, one column per op, and every panel on the same scale, so a write that pokes above its bound reads as a dashed line clearing the further band edge. Sites where the seed-mean write exceeds the seed-mean bound, per op: {", ".join(f"<code>{op}</code> {n}" for op, n in h4_over.items())} of 24 post-embedding sites.
    """,
)
def h4_write_bound_maps(res: Results) -> plt.Figure:
    from matplotlib.layout_engine import ConstrainedLayoutEngine
    from matplotlib.lines import Line2D

    c = ex.RECIPE.name
    iv = ex.PROJECTION.name
    clean = {
        op: np.array([r["ops"][op]["clean"]["alpha_q99_nonred"] for r in res.scored(c)], float).mean(0)
        for op in ex.OP_NAMES
    }
    band = {op: res.clean_alpha_band(c, op) for op in ex.OP_NAMES}
    arr = {op: res.score(c, op, iv, "q99_alpha_nonred", None).mean(0) for op in ex.OP_NAMES}
    top = min(1.0, max(float(m.max()) for m in clean.values()) * 1.15)
    bound_c, write_c = ink("recipe"), light_dark("#8a0000", "#ff7070")
    fig, axes = plt.subplots(len(SLICE_NAMES), len(ex.OP_NAMES), figsize=(7.6, 4.2), sharex=True, sharey=True)
    axes = cast(AxesGrid, axes)
    engine = fig.get_layout_engine()
    assert isinstance(engine, ConstrainedLayoutEngine)
    engine.set(hspace=0, h_pad=0.01, wspace=0.04)
    x = np.arange(len(POS_NAMES))
    for col, op in enumerate(ex.OP_NAMES):
        for row in range(len(SLICE_NAMES)):
            sl = len(SLICE_NAMES) - 1 - row  # the embedding at the bottom, as the profile figures draw it
            ax = axes[row][col]
            lo, hi = band[op][0][sl], band[op][1][sl]
            ax.axhline(0, c="#888", lw=0.5, alpha=0.5)
            smooth_step_band(ax, x, lo, hi, ramp=0.3, color=bound_c, alpha=light_dark(0.2, 0.26), fillet=1.5)
            for edge in (lo, hi):
                smooth_step(ax, x, edge, ramp=0.3, color=bound_c, lw=1.1, fillet=1.5)
            for sign in (1, -1):
                smooth_step(ax, x, sign * arr[op][sl], ramp=0.3, color=write_c, lw=1.0, ls=(0, (2, 1.5)), fillet=1.5)
            ax.set(
                ylim=(-top, top), xlim=(-0.5, len(POS_NAMES) - 0.5), yticks=[-round(top / 2, 2), 0, round(top / 2, 2)]
            )
            ax.spines[:].set_visible(False)
            ax.grid(axis="y", c="#888", alpha=0.2)
            ax.tick_params(axis="x", length=0, labelsize=7)
            ax.tick_params(axis="y", left=True, direction="in", labelleft=col == 0, labelsize=6)
            if col == 0:
                ax.set_ylabel(SLICE_NAMES[sl], fontsize=7.5)
        axes[0][col].set_title(op, fontsize=9)
        axes[-1][col].set_xticks(x, POS_NAMES, rotation=60, fontsize=6.5)
    handles = [
        Line2D([], [], color=bound_c, lw=1.1, label="clean α, 1st–99th percentile"),
        Line2D([], [], color=write_c, lw=1.0, ls=(0, (2, 1.5)), label="±|α| arriving (projected)"),
    ]
    fig.legend(handles=handles, fontsize=6.5, frameon=False, ncol=2, loc="outside upper left")
    return fig


h4_write_bound_maps(res)

rf"""
**H4, {h4.status}.** {h4.line}

## Exploratory analyses

Preregistered as exploratory, no gates.

### E1 — per-op statistics

Every H2 statistic read on the probe lines of each op, rather than on the `mix` lines alone, for every candidate. The labeller never sees the op, so a placement that differs by op would mean the blocks carry *red* differently under different rules. Only m_line and contrast can differ: they involve op2 and the answer, which follow the op word, while grading, containment, and the lead weight are read at or before op1, where causal attention has not yet seen which rule the line uses.

The five statistics, in a phrase each: **m_line** is how much more the labelled lines lean on the axis than the average line does (the margin the survey ranked on); **grading r²** is whether that lean rises smoothly with redness rather than switching on; **contrast** is how much the pull prefers the operand that drew the label over the other one, at depth; **containment ᾱ** is the mean lean of every color at op1, which should stay near zero; and the **lead weight** is the share of the pull that sits on the drawing operand at the embedding.
"""


def _table() -> tuple[str, dict[tuple[str, str], float]]:
    stats = (
        ("m_line", "m_line ↑"),
        ("r2_sim", "grading r² ↑"),
        ("contrast", "contrast ↑"),
        ("alpha_op1", "containment ᾱ ↓"),
        ("lead_emb", "lead weight ↑"),
    )
    head = ["candidate", "statistic"] + [f"<code>{op}</code>" for op in ex.OP_NAMES] + ["spread"]
    rows = []
    spread = {}
    for c in ex.CANDIDATES:
        for i, (k, title) in enumerate(stats):
            means = np.array([res.stat(c.name, k, op).mean() for op in ex.OP_NAMES])
            rows.append(
                [f"<b><code>{c.name}</code></b>" if i == 0 else "", title]
                + [span2(res.stat(c.name, k, op)) for op in ex.OP_NAMES]
                + [f"{means.max() - means.min():.3f}"]
            )
            spread[c.name, k] = float(means.max() - means.min())
    e1_spread_out: dict[tuple[str, str], float] = spread
    caption = """
    The placement statistics per op, for every candidate: seed mean with half the seed range. The <code>mix</code> column repeats the gated values. Spread is the range of the seed means across the six ops, to read against the band for that statistic in the H3 table.
    """
    return table_html(head, rows, caption), e1_spread_out


e1_table_html, e1_spread = _table()
e1_table_html

# %%


@memo
@themed(
    name="e1-per-op-dots",
    alt_text="""
        A two-by-two grid, m_line in the left column and contrast in the right, with the six ops along the
        bottom and a dashed divider separating mix from the other five. The tall upper row plots the
        statistic itself: the candidates sit in well-separated groups, each near-flat across the ops. The
        short lower row plots the same points as distances from the candidate's mean over the five, on a
        scale about ten times finer, over a grey stripe marking the equivalence band. There the contrast
        points sit inside the stripe with mix far below it, furthest on the two recipe arms; the m_line
        points scatter around the stripe with bars taller than it, darken high and add low on most
        candidates.
    """,
    caption="""
        **The two per-op placement statistics, whole and magnified.** m_line and contrast are the only
        statistics that can vary by op. One line per candidate in its ink, one dot per op at the seed mean
        with a bar for the seed range, offset a little so the candidates do not overlap. The upper row
        plots each statistic on its own scale, where the gaps between candidates set the axis. The lower
        row magnifies what that hides: the same points as a distance from the candidate's own mean over
        the five ops that share a partner draw, per seed, so the ops can be compared within a run. The
        grey stripe there is that statistic's equivalence band, the smallest seed-mean difference the
        resolution rule may call a difference. <code>mix</code> sits left of the divider, and out of the
        baseline, because its probe lines are D2.1's on-grid partners rather than the shared draw: its
        distance from the five carries the partner set as well as the op.
    """,
)
def e1_per_op_dots(res: Results) -> plt.Figure:
    stats = (("m_line", "m_line"), ("contrast", "contrast"))
    x = np.arange(len(ex.OP_NAMES))
    # The five ops that share one partner draw; `mix` has its own, so it is drawn apart from them.
    shared = [o for o in ex.OP_NAMES if o != ex.PRIMARY_OP.name]
    fig, axes = plt.subplots(2, len(stats), figsize=(6.4, 3.9), height_ratios=(3, 2), sharex=True, layout="constrained")
    axes = cast(AxesGrid, axes)
    off = np.linspace(-0.22, 0.22, len(ex.CANDIDATES))
    rule = light_dark("#555555", "#aaaaaa")
    for col, (k, title) in enumerate(stats):
        hi, lo = axes[0][col], axes[1][col]
        half = ex.equiv_band(k) / 2
        lo.axhspan(-half, half, color=rule, alpha=0.13, lw=0, zorder=0)
        lo.axhline(0, color=rule, lw=0.6, alpha=0.5, zorder=1)
        for d, c in zip(off, ex.CANDIDATES, strict=True):
            v = np.stack([res.stat(c.name, k, op) for op in ex.OP_NAMES])  # (ops, seeds)
            dev = v - np.stack([res.stat(c.name, k, op) for op in shared]).mean(axis=0)
            for ax, y in ((hi, v), (lo, dev)):
                ax.plot(x[1:] + d, y[1:].mean(axis=1), "-", color=ink(c.name), lw=0.8, alpha=0.6, zorder=2)
                ax.vlines(x + d, y.min(axis=1), y.max(axis=1), color=ink(c.name), lw=1.0, zorder=2)
                ax.plot(x + d, y.mean(axis=1), "o", ms=3, color=ink(c.name), zorder=3, label=c.name if (k == "m_line" and ax is hi) else None)  # fmt: skip
        hi.set_title(title, fontsize=9)
        lo.set_ylabel("Δ from the five", fontsize=6.5)
        lo.set_xticks(x, ex.OP_NAMES, fontsize=6.5, rotation=30)
        for ax in (hi, lo):
            ax.axvline(0.5, color=rule, lw=0.6, ls=(0, (2, 2)), alpha=0.5, zorder=1)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", alpha=0.2)
    fig.legend(fontsize=6, frameon=False, ncol=len(ex.CANDIDATES), loc="outside lower center")
    return fig


e1_per_op_dots(res)

# %%


def _e1_five(c: str, k: str) -> np.ndarray:
    """The seed means of one statistic over the five ops that share a partner draw."""
    shared = [o for o in ex.OP_NAMES if o != ex.PRIMARY_OP.name]
    return np.array([res.stat(c, k, op).mean() for op in shared])


e1_sp = {(c.name, k): float(np.ptp(_e1_five(c.name, k))) for c in ex.CANDIDATES for k in ("m_line", "contrast")}
e1_k5 = max(e1_sp[c.name, "contrast"] for c in ex.CANDIDATES)
e1_gk = [
    float(res.stat(c.name, "contrast", ex.PRIMARY_OP.name).mean() - _e1_five(c.name, "contrast").mean())
    for c in ex.CANDIDATES
]
e1_m6 = max(e1_spread[c.name, "m_line"] for c in ex.CANDIDATES)
e1_k6 = max(e1_spread[c.name, "contrast"] for c in ex.CANDIDATES)
e1_l6 = max(e1_spread[c.name, "lead_emb"] for c in ex.CANDIDATES)

rf"""
Read across all six ops, the widest spread of seed-mean m_line on any candidate is {e1_m6:.3f} (band {ex.equiv_band("m_line"):.3f}) and of contrast {e1_k6:.3f} (band {ex.equiv_band("contrast"):.3f}), both wider than the band. But the six ops are not six comparable reads. `mix` draws its probe lines from its own on-grid partners, D2.1's set, while the other five share one draw, so only those five differ from each other in the op alone. Split that way the two statistics separate.

**Contrast is op-blind among the five.** Its spread there is no wider than the band on any candidate ({e1_k5:.3f} at most, band {ex.equiv_band("contrast"):.3f}). All of the six-op spread is `mix`, which sits below the five on every candidate, by {min(-g for g in e1_gk):.3f} to {max(-g for g in e1_gk):.3f} — furthest on the two recipe arms, whose contrast is highest to begin with. Since `mix` is also the op with the different partner set, that step reads as the partner set rather than as the rule, and a `mix` probe drawn like the others would be needed to tell the two apart.

**m_line is less settled.** Among the five it stays inside the band on the recipe arms ({e1_sp[ex.RECIPE.name, "m_line"]:.3f} and {e1_sp[ex.RECIPE_SHORT.name, "m_line"]:.3f}) and sits at it on `t00` ({e1_sp["t00", "m_line"]:.3f}), reaching {e1_sp["t48", "m_line"]:.3f} on `t48` and {e1_sp["t12", "m_line"]:.3f} on `t12`, the two proposals with the lowest anchor weight, against a band of {ex.equiv_band("m_line"):.3f}. The ordering repeats across candidates — `darken` high on all five, `add` low on all but `recipe-short` — but the per-seed deviations are wider than the band, so five seeds do not settle whether the ordering is real. If it is, the probe set is the first place to look: m_line is a margin over the average line of that op's set, and the five sets share their op1 colors and partners but not their answers, so an op whose answers sit differently against the red axis would move it with the placement unchanged. Either way the effect is small, and every gated read is on `mix`.

Grading r² and containment ᾱ are identical on all six ops to every printed digit. Both are read at op1, which precedes the op word, so under causal attention they cannot see which rule the line uses; the table prints them for completeness. The lead weight is read at op1 too, but its softmin normalizer runs over the whole span, so the op word's own embedding enters it: it moves by {e1_l6:.3f} at most, again with `mix` apart from the rest.

### E2 — the op words

The alignment of the six op-word embeddings and of `=` and the newline, per slice, on the recipe and the control. Ex-2.2.1 found a constant component on `+` and `=` of about 0.3–0.4 in the published map, which is where its non-red cost came from. The second question is whether the alignment of an op word predicts the non-red deficit on its lines under `projection`.
"""

syntax_table(res, ex.RECIPE.name)

# %%

syntax_table(res, adopted) if adopted != ex.RECIPE.name else ""

# %%

syntax_table(res, ex.CONTROL.name)

# %%

# Added after the results were read: the two proposals the adoption did not name, for the same rows.
"".join(syntax_table(res, c.name) for c in ex.PROPOSALS if c.name != adopted)

# %%

e2_x = np.array([np.abs(alpha_at(res, ex.RECIPE.name, op, 1)[:, 1:]).mean() for op in ex.OP_NAMES])
e2_y = np.array([res.deficit(ex.RECIPE.name, op, ex.PROJECTION.name).mean() for op in ex.OP_NAMES])
e2_r = float(np.corrcoef(e2_x, e2_y)[0, 1]) if e2_x.std() > 0 and e2_y.std() > 0 else float("nan")
e2_pairs = ", ".join(f"`{op}` {x:.2f} / {y:.3f}" for op, x, y in zip(ex.OP_NAMES, e2_x, e2_y, strict=True))

rf"""
On the recipe, the mean |α| of each op word over the four post-attention slices, against the non-red deficit under `projection` on its lines: {e2_pairs}. Pearson r over the six ops is {e2_r:+.2f}. Six points cannot carry much, and the sign is what to read: a positive r says the ops whose word sits further along the axis lose more on their non-red lines when the axis is projected out.

### E3 — redder than both

Every op but `lighten` has lines whose answer is redder than either operand (the [op table](#the-grammar) counts them). On those lines the strongest evidence for *red* sits at the answer, and the labeller never keys on that position. This is the blind-span case named in the scope note of ex-2.1.10. We read the alignment at `=` and at the answer position against lines of the same op and dose whose answer is not redder than both, plus the softmin profile on each group, and then restrict the H4 statistics to these lines.

<!-- REVIEW: a labeller that reads the answer as well as the operands (so `magenta darken yellow = red` draws a *red* label from its answer) is the M3-shaped labelling, and is filed as a fast-follow under (c) of todo/science/labeling-pull-span-variants-ex-2-1.md rather than changed here. E3 reads the blind span under the operand-only labeller first, which is what makes the answer position blind. Verify: that item's 2026-09-09 note. -->
"""


@memo
def e3_table() -> tuple[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    c = ex.RECIPE.name
    bins = np.linspace(0.0, 1.0, 11)
    ops = [op for op in ex.OP_NAMES if ex.line_counts(ex.OP_BY_NAME[op])["redder"] > 0]

    def matched_diff(v: np.ndarray, redder: np.ndarray, dose: np.ndarray) -> float:
        """Mean over dose bins, weighted by the redder count, of (redder mean − other mean) of a per-line value."""
        b = np.clip(np.digitize(dose, bins) - 1, 0, len(bins) - 2)
        num = den = 0.0
        for k in np.unique(b[redder]):
            a, o = v[redder & (b == k)], v[~redder & (b == k)]
            if len(o):
                num += len(a) * (a.mean() - o.mean())
                den += len(a)
        return num / den if den else float("nan")

    rows = []
    profiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for op in ops:
        d_eq, d_ans, prof_r, prof_m = [], [], [], []
        for r in res.runs(c):
            cos = res.arrays[f"{r['label']}/eval/{op}/alpha_lines"].astype(np.float32)  # (slices, N, T)
            redder = res.arrays[f"{r['label']}/eval/{op}/redder"]
            dose = res.arrays[f"{r['label']}/eval/{op}/dose"]
            deep = cos[1:].mean(axis=0)  # (N, T)
            d_eq.append(matched_diff(deep[:, 3], redder, dose))
            d_ans.append(matched_diff(deep[:, ex.ANSWER_POS], redder, dose))
            w = softmin_weights(1.0 - cos[1:, :, : ex.SPAN], r["tau"]).mean(axis=0)  # (N, roles)
            b = np.clip(np.digitize(dose, bins) - 1, 0, len(bins) - 2)
            wt = np.zeros(len(dose))
            for k in np.unique(b[redder]):
                m = ~redder & (b == k)
                if m.any():
                    wt[m] = (redder & (b == k)).sum() / m.sum()
            prof_r.append(w[redder].mean(axis=0))
            prof_m.append((w * wt[:, None]).sum(axis=0) / wt.sum())
        profiles[op] = (np.array(prof_r), np.array(prof_m))
        n = int(res.runs(c)[0]["per_op"][op]["n_redder"])
        clean = res.score(c, op, None, "acc", "redder")
        proj = res.score(c, op, ex.PROJECTION.name, "acc", "redder")
        comp = np.array([r["ops"][op]["interventions"][ex.PROJECTION.name]["composition_redder"] for r in res.scored(c)], float)  # fmt: skip
        miss = comp[:, 1:]  # everything but the true answer
        share = miss / np.maximum(miss.sum(axis=1, keepdims=True), 1)
        rows.append(
            [
                f"<code>{op}</code>", f"{n:,}", span2(np.array(d_eq), ".3f"), span2(np.array(d_ans), ".3f"),
                span2(clean, ".2f"), span2(proj, ".2f"), span2(share[:, 0], ".2f"), span2(share[:, 2], ".2f"),
            ]
        )  # fmt: skip
    head = ["op", "redder lines", "Δα at =", "Δα at the answer", "clean acc", "projection acc ↓", "misses → red operand", "misses → neighbor"]  # fmt: skip
    caption = """
    The redder-than-both lines of each op on the recipe. Δα is the alignment on those lines minus the alignment on lines of the same op in the same dose bin whose answer is not redder than both, averaged over the four post-attention slices and weighted by the redder lines' dose histogram; a positive value at the answer says the stream carries the answer's extra redness there, where the labeller never keyed. The right half restricts H4 to these lines: exact-match accuracy clean and under <code>projection</code>, and, of the projected answers that are wrong, the share that are the redder operand and the share that are a one-step neighbor of the truth. Seed means with half the seed range; <code>lighten</code> has no such lines.
    """
    return table_html(head, rows, caption), profiles


e3_table_html, e3_profiles = e3_table()
e3_table_html

# %%


@memo
@themed(
    name="e3-redder-profiles",
    alt_text="""
        A row of small panels, one per op with redder-than-both lines, each showing two softmin profiles over the roles op1, op, op2, and equals: a solid line for the redder lines and a dashed line for dose-matched lines whose answer is not redder than both. The two lines lie close together in every panel.
    """,
    caption="""
        **Softmin profiles on the redder-than-both lines, recipe, post-attention slices.** One panel per op that has such lines. Solid is the seed-mean profile over the span roles on the redder lines; dashed is the dose-matched comparison group of the same op. The profile is the share of the pull each role would receive at the run's τ, read from the clean alignment, so a difference between the two says the answer's redness moved where the pull lands within the span. Hairlines are seeds.
    """,
)
def e3_redder_profiles(profiles: dict[str, tuple[np.ndarray, np.ndarray]]) -> plt.Figure:
    ops = list(profiles)
    x = np.arange(ex.SPAN)
    fig, axes = plt.subplots(1, len(ops), figsize=(1.55 * len(ops), 1.9), sharey=True, layout="constrained")
    axes = cast(AxesRow, axes)
    hair = light_dark("#00000033", "#ffffff33")
    for ax, op in zip(axes, ops, strict=True):
        pr, pm = profiles[op]
        for s in pr:
            smooth_step(ax, x, s, ramp=0.5, color=hair, lw=0.5)
        smooth_step(ax, x, pr.mean(0), ramp=0.5, color=ink("recipe"), lw=1.5)
        smooth_step(ax, x, pm.mean(0), ramp=0.5, color=ink("recipe"), lw=1.1, ls=(0, (2, 1.5)))
        ax.set(ylim=(-0.05, 1.05), xlim=(-0.4, ex.SPAN - 0.6), yticks=[0, 0.5, 1])
        ax.set_xticks(x, ROLES)
        ax.tick_params(labelsize=6.5)
        ax.set_title(op, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


e3_redder_profiles(e3_profiles)

r"""
### E4 — a richer op set

The cube probes of ex-2.1.12 (ridge, ℓ₂ = 10⁻², strict per-value holdout) on the un-anchored models at one, three, and six ops: held-out $R^2$ for op1, op2, and the RGB of the answer, per slice and position. This asks whether more rules give the model a better operand geometry at the same compute. The two matchings under [the conditions](#conditions) are read as a pair, with the control as the six-op point of both. In the `corpus` arms the fewer-op models see more lines per op, so a cube that improves with the op count there is a clean positive. In the `per-op` arms the fewer-op models see the same lines per op, repeated more often, so a cube that worsens with the op count there is a clean negative. A trend that holds in both arms is read as the op count; one that holds in only one is read as its confound, lines per op or repetition.
"""


def _e4_r2(cond: str) -> np.ndarray:
    """(seeds, slices, positions, targets, channels) strict R² of one arm."""
    rs = sorted((r for r in res.geometry["runs"] if r["condition"] == cond), key=lambda r: r["seed"])
    return np.array([r["r2_strict"] for r in rs], float)


e4_by_name = {c.name: c for c in ex.RICHER_OP_ARM}
e4_arms: dict[str, list[ex.Condition]] = {
    "corpus": sorted([c for c in e4_by_name.values() if c.name.endswith("-corpus")], key=lambda c: len(c.ops))
    + [ex.CONTROL],  # fmt: skip
    "per-op": sorted([c for c in e4_by_name.values() if c.name.endswith("-per-op")], key=lambda c: len(c.ops))
    + [ex.CONTROL],  # fmt: skip
}
e4_r2: dict[str, np.ndarray] = {c.name: _e4_r2(c.name) for arm in e4_arms.values() for c in arm}


def _table() -> str:
    targets = {"op1": 0, "op2": 2, "ans": 3}
    sites = {"op1": "own slot", "op2": "own slot", "ans": "="}
    counts = sorted({len(c.ops) for arm in e4_arms.values() for c in arm})
    head = ["target, site", "arm"] + [f"{n} op{'s' if n > 1 else ''}" for n in counts]
    rows = []
    for ti, (t, p) in enumerate(targets.items()):
        for ai, (arm, conds) in enumerate(e4_arms.items()):
            row = [f"{t} at {sites[t]}" if ai == 0 else "", arm]
            by_count = {len(c.ops): np.clip(e4_r2[c.name][:, 1:, p, ti, :], 0, 1).mean(axis=(1, 2)) for c in conds}
            row += [span2(by_count[n], ".2f") if n in by_count else "·" for n in counts]
            rows.append(row)
    caption = """
    Strict held-out R² of the un-anchored models by op count, at the site that carries each target: op1 and op2 at their own slot, the answer at <code>=</code>, where it is first decodable. Each value is the mean over the RGB channels and the four post-attention slices, negative scores clipped to zero first, as a seed mean with half the seed range. The <code>corpus</code> arm keeps D2.1's line count as the op set narrows (more lines per op at fewer ops); the <code>per-op</code> arm keeps the six-op lines per op (a smaller corpus, repeated). The control is the six-op point of both.
    """
    return table_html(head, rows, caption)


_table()

# %%


@memo
@themed(
    name="e4-answer-probes",
    alt_text="""
        Two rows of three probe grids. Each grid has one row per residual slice with the embedding at the bottom and the six token positions across, with a trace per RGB channel over a grey area. The answer's color rises at the equals slot in the upper slices in every grid; the top row is the corpus-matched arm and the bottom the per-op-matched arm, at one, three, and six ops from left to right.
    """,
    caption="""
        **Where the answer's color is decodable, by op count.** Strict held-out R² for the RGB of the answer at every site, ex-2.1.12's layout: rows are residual slices with the embedding at the bottom, columns the six positions, one trace per channel over the grey area of their mean, with the seed envelope as hairlines. The bold slot is the answer itself. Top row, the <code>corpus</code> matching; bottom row, the <code>per-op</code> matching; the six-op panel is the control in both. Negative scores clip to the floor.
    """,
)
def e4_answer_probes(e4_arms: dict[str, list[ex.Condition]], e4_r2: dict[str, np.ndarray]) -> plt.Figure:
    t = 2  # the answer's RGB
    n = max(len(a) for a in e4_arms.values())
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 5.2), squeeze=False)
    axes = cast(AxesGrid, axes)
    for row, (arm, conds) in zip(axes, e4_arms.items(), strict=True):
        for i, (ax, c) in enumerate(zip(row, conds, strict=False)):
            draw_probe_grid(
                ax, e4_r2[c.name][:, :, :, t, :], own=ex.ANSWER_POS, row_names=i == 0, scale=i == len(conds) - 1
            )
            ax.set_title(f"{arm}: {len(c.ops)} op{'s' if len(c.ops) > 1 else ''} ({c.name})", fontsize=7.5, pad=6)
        for ax in row[len(conds) :]:
            ax.set_axis_off()
    return fig


e4_answer_probes(e4_arms, e4_r2)

r"""
### E5 — the noise floor, re-measured

The per-run σ of every gated statistic on the new grammar, from the five seeds of the recipe and of the control, beside the ex-2.1.10 values the bands used. The H3 verdict and the selection rule are scored with the frozen ex-2.1.10 bands either way. Where the fresh σ is larger, we report which H3 comparisons the wider band would leave unresolved, as a robustness read.
"""


def _e5_sd(cond: str, stat: str) -> float:
    mix = ex.PRIMARY_OP.name
    v = res.em(cond, mix) if stat == "holdout_em" else res.stat(cond, stat)
    return float(v.std(ddof=1))


def _table() -> tuple[str, str]:
    head = ["statistic", "σ, ex-2.1.10 (frozen)", "σ, recipe (fresh)", "σ, control (fresh)", "band, frozen", "band, fresh recipe"]  # fmt: skip
    rows = []
    fresh_band = {}
    for stat, sigma in ex.NOISE_RUN.items():
        r, c = _e5_sd(ex.RECIPE.name, stat), _e5_sd(ex.CONTROL.name, stat)
        fresh_band[stat] = ex.equiv_band(stat, noise={stat: r})
        rows.append(
            [
                stat,
                f"{sigma:.4f}",
                f"<b>{r:.4f}</b>" if r > sigma else f"{r:.4f}",
                f"{c:.4f}",
                f"{ex.equiv_band(stat):.3f}",
                f"{fresh_band[stat]:.3f}",
            ]  # fmt: skip
        )
    short = float(res.stat(ex.RECIPE_SHORT.name, "m_line").mean())
    reads = []
    for p in ex.PROPOSALS:
        d = float(res.stat(p.name, "m_line").mean()) - short
        frozen, fresh = abs(d) > ex.equiv_band("m_line"), abs(d) > fresh_band["m_line"]
        reads.append(
            f"`{p.name}` {d:+.3f}: {'resolved' if frozen else 'unresolved'} at the frozen band, {'resolved' if fresh else 'unresolved'} at the fresh one"
        )
    md: str = "; ".join(reads)
    caption = """
    The per-run spread of each gated statistic: the frozen ex-2.1.10 value behind every band in this report, and the sample standard deviation over the five fresh seeds of the recipe and of the control. Bold marks a fresh recipe σ above the frozen one. The placement statistics on the control are near zero by construction, so its σ there is the instrument's noise rather than a run-to-run spread of a placement. The last two columns are the band for a five-against-five comparison under each σ.
    """
    return table_html(head, rows, caption), md


e5_table_html, e5_md = _table()
e5_table_html

rf"""
The H3 comparisons of each proposal's m_line against `recipe-short`, under both bands: {e5_md}.

### E6 — the two recipe lengths at twenty seeds

Added after the results were read. The H3 read compared `recipe` and `recipe-short` at five seeds each, and the post hoc amendment above adopts one of them; five seeds is a thin basis for a choice the next experiments build on, so fifteen more seeds of each length were run (the `-more` arms under [the conditions](#conditions)). The frozen arms keep their five seeds and their verdicts; this section reads all twenty of each length together. The question is plain: at twenty seeds, does the full-length recipe still place *red* better than the short one, by more than the band, and does either length change its H4 read?
"""


def _table() -> str:
    head = ["statistic", "recipe, 5 seeds", f"recipe, {e6_n} seeds", f"recipe-short, {e6_n} seeds", "Δ (short − full)", "band, frozen σ", "band, fresh σ"]  # fmt: skip
    rows = []
    for k in e6_stats:
        a, b = e6_pool[ex.RECIPE.name, k], e6_pool[ex.RECIPE_SHORT.name, k]
        d, frozen, fresh = e6_reads[k]
        five = res.em(ex.RECIPE.name, ex.PRIMARY_OP.name) if k == "holdout_em" else res.stat(ex.RECIPE.name, k)
        rows.append(
            [
                k,
                f"{five.mean():.3f}",
                f"{a.mean():.3f} <span class='range'>&nbsp;σ {a.std(ddof=1):.3f}</span>",
                f"{b.mean():.3f} <span class='range'>&nbsp;σ {b.std(ddof=1):.3f}</span>",
                f"<b>{d:+.3f}</b>" if abs(d) > fresh else f"{d:+.3f}",
                "·" if np.isnan(frozen) else f"{frozen:.3f}",
                f"{fresh:.3f}",
            ]
        )
    caption = f"""
    The two recipe lengths at {e6_n} seeds each: the five frozen seeds and the fifteen addendum seeds pooled. Values are seed means; σ is the sample standard deviation over the {e6_n} seeds. Δ is short minus full. The frozen band is 2σ√(2/{e6_n}) with the ex-2.1.10 per-run σ; the fresh band uses the pooled σ of the two arms in this table. Bold marks a difference outside the fresh band. The lead weight has no frozen σ.
    """
    return table_html(head, rows, caption)


_table()

# %%


@memo
@themed(
    name="e6-recipe-lengths",
    alt_text="""
        Four small panels, one per placement statistic, each with two rows, recipe and recipe-short. Every seed is a small dot along the row; the larger marker is the twenty-seed mean, and the five frozen seeds are drawn hollow.
    """,
    caption="""
        **The two recipe lengths, seed by seed.** One panel per placement statistic on the `mix` lines; the top row of each is `recipe`, the bottom `recipe-short`. Each small dot is one seed (hollow for the five frozen seeds, filled for the fifteen addendum seeds), jittered a little so they do not stack; the large marker is the twenty-seed mean, with a bar of one σ either side.
    """,
)
def e6_recipe_lengths(res: Results) -> plt.Figure:
    stats = ["m_line", "contrast", "r2_sim", "lead_emb"]
    conds = (ex.RECIPE.name, ex.RECIPE_SHORT.name)
    fig, axes = plt.subplots(1, len(stats), figsize=(7.6, 1.9), layout="constrained")
    axes = cast(AxesRow, axes)
    rng = np.random.default_rng(0)
    for ax, k in zip(axes, stats, strict=True):
        for y, c in zip((1.0, 0.0), conds, strict=True):
            five, more = res.stat(c, k), res.stat(f"{c}-more", k)
            allv = np.concatenate([five, more])
            jit = rng.uniform(-0.18, 0.18, len(allv))
            ax.plot(five, y + jit[: len(five)], "o", ms=3.2, mfc="none", mec=ink(c), mew=0.9, zorder=3)
            ax.plot(more, y + jit[len(five) :], "o", ms=3.2, color=ink(c), alpha=0.75, zorder=3)
            m, sd = allv.mean(), allv.std(ddof=1)
            ax.plot([m - sd, m + sd], [y - 0.42, y - 0.42], color=ink(c), lw=1.4, solid_capstyle="round", zorder=2)
            ax.plot(m, y - 0.42, "D", ms=4, color=ink(c), zorder=4)
        ax.set_yticks([1.0, 0.0], conds, fontsize=7)
        ax.set_ylim(-0.8, 1.5)
        ax.set_title(k, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.2)
    return fig


e6_recipe_lengths(res)

# %%


def _table() -> tuple[str, dict[str, tuple[float, float]]]:
    mix = ex.PRIMARY_OP.name
    head = ["condition"] + [f"{op} red acc ↓" for op in ex.OP_NAMES] + ["mix non-red deficit ↓"]
    rows = []
    h4_20: dict[str, tuple[float, float]] = {}
    for c in (ex.RECIPE.name, ex.RECIPE_SHORT.name):
        red = [res.pooled_score(c, op, ex.PROJECTION.name, "acc") for op in ex.OP_NAMES]
        deficit = res.pooled_deficit(c, mix, ex.PROJECTION.name)
        h4_20[c] = (float(max(v.mean() for v in red)), float(deficit.mean()))
        rows.append(
            [f"{c}, projection, {len(deficit)} seeds"]
            + [f"<b>{span2(v, '.2f')}</b>" if v.mean() <= ex.RED_ACC_GATE else span2(v, ".2f") for v in red]
            + [f"<b>{span2(deficit)}</b>" if deficit.mean() <= ex.NONRED_DEFICIT_GATE else span2(deficit)]
        )
    caption = f"""
    H4's two statistics under <code>projection</code> at twenty seeds per length: seed-mean exact-match accuracy on the red lines of each op (gate ≤ {ex.RED_ACC_GATE:g}, bold) and the non-red deficit on <code>mix</code> (gate ≤ {ex.NONRED_DEFICIT_GATE:g}, bold), each with half the seed range.
    """
    return table_html(head, rows, caption), h4_20


e6_h4_table_html, e6_h4 = _table()
e6_h4_table_html

# %%

e6_d, e6_frozen, e6_fresh = e6_reads["m_line"]
e6_lead = e6_reads["lead_emb"][0]
e6_full, e6_short = e6_h4[ex.RECIPE.name], e6_h4[ex.RECIPE_SHORT.name]
e6_verdict = (
    "the full-length recipe keeps its lead"
    if -e6_d > max(e6_frozen, e6_fresh)
    else "the two lengths are within a band of each other"
    if abs(e6_d) <= max(e6_frozen, e6_fresh)
    else "the short recipe leads"
)

rf"""
At twenty seeds, `recipe-short` sits {e6_d:+.3f} of m_line from `recipe`, against a band of {e6_frozen:.3f} (frozen σ) or {e6_fresh:.3f} (fresh σ): {e6_verdict}. The lead weight at the embedding moves {e6_lead:+.3f} between the lengths. Under `projection`, the worst-op red accuracy is {e6_full[0]:.2f} on the full recipe and {e6_short[0]:.2f} on the short one, and the `mix` non-red deficit is {e6_full[1]:.3f} and {e6_short[1]:.3f}. So the choice between the two lengths does not turn on H4. It turns on placement, and the two lengths split it: the full recipe has the larger margin, and the short one grades better and puts more of the pull on the drawing operand at the embedding. The post hoc rule ranks on the margin, as the frozen one did, and names the full recipe; the decision recorded beside it takes the short arm, on grading, the operand lead, and cost.

### E7 — the other arms under projection

Added after the results were read. The H4 tables above read the recipe and `{adopted}`; this reads the same statistics on the remaining scored arms, `recipe-short` and the two proposals the adoption did not name, so the removal picture (complete on `mix`, partial on the saturating ops) can be checked across every operating point rather than two. The E2 syntax-embedding tables for those proposals sit at the end of E2.
"""


def _table() -> str:
    mix = ex.PRIMARY_OP.name
    conds = [ex.RECIPE_SHORT.name] + [c.name for c in ex.PROPOSALS if c.name != adopted]
    head = ["condition"] + [f"{op} red acc ↓" for op in ex.OP_NAMES] + ["mix non-red deficit ↓", "|α| of = (deep)"]
    rows = []
    for c in conds:
        red = [res.score(c, op, ex.PROJECTION.name, "acc", "red") for op in ex.OP_NAMES]
        deficit = res.deficit(c, mix, ex.PROJECTION.name)
        eq = np.abs(np.mean([alpha_at(res, c, op, 3) for op in ex.OP_NAMES], axis=0)[:, 1:]).mean(axis=1)
        rows.append(
            [f"{c}, projection"]
            + [f"<b>{span2(v, '.2f')}</b>" if v.mean() <= ex.RED_ACC_GATE else span2(v, ".2f") for v in red]
            + [
                f"<b>{span2(deficit)}</b>" if deficit.mean() <= ex.NONRED_DEFICIT_GATE else span2(deficit),
                span2(eq, ".2f"),
            ]
        )
    caption = f"""
    H4's statistics under <code>projection</code> on the arms the H4 section does not table, five seeds each: red-line accuracy per op (gate ≤ {ex.RED_ACC_GATE:g}, bold), the <code>mix</code> non-red deficit (gate ≤ {ex.NONRED_DEFICIT_GATE:g}, bold), and, as the E2 correlate, the mean |α| of <code>=</code> over the four post-attention slices. Seed mean with half the seed range.
    """
    return table_html(head, rows, caption)


_table()

r"""
### E8 — which red lines survive projection

Added after the results were read. H4's removal statistic asks whether the answer is still right once the axis is projected out; on ops that saturate, many red lines have an answer that does not depend on how red the red operand is, so a model that has lost *red* can still answer them. This sorts each op's red lines by whether the answer *depends* on the red operand's redness (lowering that operand's R by one grid step changes the snapped answer) and reads accuracy on the two groups separately, clean and under `projection`. Removal that reads as partial in H4 should read as complete on the dependent lines if the axis carries the redness the answer needs.
"""


@memo
def dependent_table() -> str:
    from sca.config import TokenizerConfig
    from sca.data.colors import redness
    from sca.data.named_colors import WordTokenizer
    from sca.data.ops import NAMES, probe_lines, vocabulary

    stoi = WordTokenizer(TokenizerConfig(vocabulary=vocabulary())).stoi  # the corpus tokenizer, rebuilt
    step = ex.LEVELS[1] - ex.LEVELS[0]

    def dependent(op: str) -> tuple[np.ndarray, np.ndarray]:
        """(red, dependent) masks over the op's probe lines, in the stored order."""
        red, dep = [], []
        for ln in probe_lines(ex.OP_BY_NAME[op], ex.N_PROBE, ex.PROBE_SEED):
            a, b = (ln.lhs, ln.rhs) if redness(ln.lhs) >= redness(ln.rhs) else (ln.rhs, ln.lhs)
            red.append(max(redness(ln.lhs), redness(ln.rhs)) >= ex.RED_DOSE - 1e-9)
            a2 = (max(a[0] - step, 0), a[1], a[2])
            dep.append(ex.OP_BY_NAME[op](a2, b) != ln.result)
        return np.array(red), np.array(dep)

    def answers(op: str) -> np.ndarray:
        """The answer token id of each probe line, in the stored order."""
        return np.array([stoi[NAMES[ln.result]] for ln in probe_lines(ex.OP_BY_NAME[op], ex.N_PROBE, ex.PROBE_SEED)])

    def acc(g: np.ndarray, m: np.ndarray, ans: np.ndarray) -> np.ndarray:
        return (g[:, m] == ans[m]).mean(axis=1) if m.any() else np.full(len(g), np.nan)

    head = ["op", "condition", "dependent red lines", "other red lines", "clean, dependent", "clean, other", "projection, dependent ↓", "projection, other"]  # fmt: skip
    rows = []
    for op in ex.OP_NAMES:
        red, dep = dependent(op)
        for c in (ex.RECIPE.name, adopted):
            dose = res.arr(c, "score", f"{op}/dose")[0]
            assert np.array_equal(dose >= ex.RED_DOSE - 1e-9, red), (
                "the stored probe order and the rebuilt one disagree"
            )
            clean = res.arr(c, "score", f"{op}/clean/guess")
            proj = res.arr(c, "score", f"{op}/{ex.PROJECTION.name}/guess")
            ans = answers(op)
            all_mask = np.ones(len(ans), bool)
            stored = res.score(c, op, None, "acc", "all")
            assert np.allclose(acc(clean, all_mask, ans), stored, atol=1e-6), (
                "the rebuilt answers disagree with the stored accuracy"
            )
            d, o = red & dep, red & ~dep
            rows.append(
                [
                    f"<code>{op}</code>", c, f"{int(d.sum()):,}", f"{int(o.sum()):,}",
                    span2(acc(clean, d, ans), ".2f"), span2(acc(clean, o, ans), ".2f") if o.any() else "·",
                    span2(acc(proj, d, ans), ".2f"), span2(acc(proj, o, ans), ".2f") if o.any() else "·",
                ]
            )  # fmt: skip
    caption = f"""
    The red lines (dose ≥ {ex.RED_DOSE:g}) of each op split by whether the answer depends on the red operand's redness: a line is <em>dependent</em> if lowering that operand's R by one grid level changes the snapped answer. Exact-match accuracy on each group, clean and under <code>projection</code>, seed mean with half the seed range, on the recipe and on the point the frozen rule adopted. On <code>mix</code> the other lines are those where a one-level drop rounds back to the same answer.
    """
    return table_html(head, rows, caption)


dependent_table()

r"""
## Discussion

Here is what we think happened, in plain terms.

**The anchoring recipe does not care about the grammar.** We trained on six rules instead of one, with the labeller still only ever looking at the two operands, and *red* landed on the axis the same way it did in D2.1: no task cost on any op (H1), every placement gate met (H2), and the same numbers on the lines of every op (E1). The recipe needs its own stream of labelled operands each step, the corpus kept that, and the rest of the line can do whatever it likes. So the "recipe is grammar-specific" row of the [D2.2 risk table](../d2.2/design.md) closes.

**The survey's proposals are real, and smaller than advertised.** All three came in about 0.05 of m_line below their survey values, the recipe about 0.02 below its own. That is the winner's curse we priced in: when you pick the best of many noisy trials you tend to pick one that got lucky, so a fresh run scores lower.[^curse] About two thirds of the advantage over `recipe-short` survived, the proposals grade better than the survey said (the other side of the same luck), and they spread more across seeds than the recipe does. E5 says the H3 differences hold up under the wider band too.

[^curse]: The winner's curse: when you pick the top result out of many noisy trials, you tend to pick one whose noise ran in its favor, so a fresh run of it usually scores lower.

**The selection rule picked `t00`, and we are not carrying it.** The rule ranked on margin and checked most of the H2 gates, and `t00` won. But it misses the one gate the rule left out, the lead weight at the embedding (0.33 against 0.4), and its contrast is 0.36 where the recipe's is 0.86. More to the point, it costs something the rule never looked at. At five times the recipe's anchor weight, the axis ends up on the `=` and op-word embeddings of the shared embedding table (E2: `=` at 0.93 against 0.40), and projecting the axis out of every position then breaks 0.6 of the non-red `mix` lines, with some seeds losing nearly all of them. The same is true of `t48` and `t12` (E7). Editing only the operand positions gets around it (0.017 on `mix`), and the prereg's contrary clause already sends the intervention-tuning pass to those tokens. But an operating point that needs a routed edit before its first intervention is a weak base for the anchored-op experiments, so the [post hoc read](#post-hoc-the-selection-rule-with-the-gates-it-left-out) adds the two missing gates and adopts the recipe, and D2.2 builds on its short arm. Whether the syntax embeddings can be made clean rather than routed around is the [tied-readout question](/todo/science/syntax-embeddings-carry-the-axis-via-tied-readout.md), and it wants settling before the D2.3 swap, which rotates every position.

**Full length or half?** We ran fifteen more seeds of each recipe length to answer this properly (E6). At twenty seeds the full recipe keeps a 0.012 margin lead, about twice the band, and the short one grades better (r² 0.89 against 0.82) and leads better at the embedding (0.83 against 0.79). Neither is wrong. The post hoc rule ranks on margin, so it names the full recipe; we chose the short arm anyway, and the choice is recorded in the post hoc section. The anchored-op experiments lean on grading and on the operand lead more than on margin, the H4 read is the same, and the short arm is half the compute.

**Removal looked partial on four ops, and mostly is not.** H4's removal statistic is whether the model still gets the answer right once the axis is projected out. On the recipe that held on `mix` (0.06 red-line accuracy) and missed on four of the new ops (0.17 to 0.32). The reason is the ops, not the anchor. On `add`, `screen`, `multiply`, `lighten`, and `darken`, most red lines have an answer that would be the same if the red operand were a little less red, because the op saturates or picks the other operand's channel; a model that has lost *red* can still answer those. E8 splits each op's red lines by whether the answer depends on the red operand's redness at all, and on the dependent lines the projection removes *red* on every op (accuracy 0.00 to 0.18 on the recipe, 0.02 or less on `t00`). The misses in the H4 table are the lines where the correct answer never needed *red*. For the next grammar we want both kinds: some ops whose answers depend on both operands and spread through the cube, and some anisotropic ones like these, whose answers sit on a face or an edge, so the E8 split has both to read; that is the [diverse-op item](/todo/science/diverse-operator-set-hue-saturation-brightness.md), and it should be checked with the answer clouds and the E8 split before anything is anchored on it.

**Selectivity on the recipe transfers.** The `mix` non-red deficit under projection is the 0.024 from ex-2.2.1 at five seeds and 0.026 at twenty, and the other ops lose nothing to speak of. The write stays inside the bound at most sites (the stacked figure under H4), and where it pokes above, it is at the op word, which is the one new token.

**The blind span behaves.** On the lines whose answer is redder than either operand, the answer's own redness shows on the axis at `=` and at the answer position, by 0.15 at most, and the projection leaves most of those lines intact (E3). The anchor never sees the answer, so that redness is computed off the axis, as the ex-2.1.10 scope note said it would be. `multiply` has the most such lines and the largest shift, so it is where a labeller that also reads the answer would show up first; that labeller is filed under [the span variants](/todo/science/labeling-pull-span-variants-ex-2-1.md).

**More rules did not make a straighter cube, on this probe.** At six ops the operand's RGB is less linearly decodable at its own slot than at three (0.52 against 0.85 to 0.87), in both matchings, though the six-op seeds spread widely (±0.14) and the one-op arm of the `per-op` matching is in a memorization regime, so the read is a trend over three points rather than a result. The answer at `=` gives no clean read either way. Whether the model builds op-specific readings of its operands is a probe question for the anchored-op experiments, which read those sites anyway (E4).

**Which op to anchor next.** The [relevance table](#the-grammar) favors `mix` (94% of its lines have an answer no other op gives) and, among the new ops, `multiply`. E3 adds that `multiply` is where the answer's redness shows on the axis most, and E8 that its red lines are about half dependent. The suppression prereg can choose on what it wants to show: `mix` for the sharpest test, `multiply` for a rule the anchor has never seen.

## Method

Two things change for the model against D2.1: the corpus carries six rules at the same total size, and a token between the operands decides the answer.[^const] The [dependency named in the design](../d2.2/design.md#deps) lands before the DAG, in `sca.data`, and the table below is its specification.

[^const]: Whereas previously, the operation was constant (`+`), so the model had no reason to use that position for anything other than spare compute.
"""

# REVIEW: review asked whether the snap is biased. `screen` and `multiply` never land halfway
# between two levels, and their mean signed rounding error over the pairs is zero. `mix` did
# round up on every off-grid channel sum (D2.1's round-half-up rule, then the snap), a
# lightening bias on 87% of its pairs; it is now the plain mean, with ties sent to the even
# level index, per the REVIEW note on `snap` in experiment.py. Stochastic rounding was
# considered and not taken: it would put the exact-match ceiling below 1 on the rounded ops,
# which the calibration and H1 read, and is filed as a grammar variant in
# todo/science/stochastic-rounding-of-off-grid-answers.md. Verify: `snap`'s docstring, and the
# on-grid column of the op table, unchanged for `mix` at 12.9%.
r"""
### The grammar

Six ops on the `v216` grid (six levels per channel: 0, 3, 6, 9, 12, 15), with every color one token and every op one token. Each op is computed per channel on the 0..15 scale and snapped to the nearest grid level. So every op answers every pair with a color in the vocabulary, and a line can be written for any pair. Where the rule lands on the grid by itself, no rounding happens, and `mix` on those pairs is the D2.1 op.

Across the pairs, the snap rounds up about as often as down. `screen` and `multiply` never land halfway between two levels, so their mean signed rounding error is zero. `mix` lands halfway on half of its channel sums, and those ties go to the even level (0, 6, or 12), which sends ten of the eighteen cases down and eight up.[^bias]

[^bias]: If the snap always broke ties the same way, every rounded `mix` answer would shift in that direction, and the model would learn the shift as part of the rule. The `mix` of D2.1 rounded half up, which on this grid meant up by a whole level on every off-grid pair. D2.1 never trained on those pairs, so the change costs nothing there, and the probe set is the on-grid pairs, which are untouched.
"""

# REVIEW: the op table was drafted under D2.1's closed-pair rule, which admits only
# pairs whose answer is already a grid color; `screen` and `multiply` are degenerate
# under it, and were dropped. The design's deps section asks for a "grid function with
# defined rounding" instead, so every op is now computed on the 0..15 scale and snapped
# to the nearest level, which makes all six total and restores the two. Verify: the
# on-grid column below shows how much each op rounds, and the agreement table that
# the six are distinct rules.
# REVIEW: the sentence reading the agreement table named the wrong pairs — `add`
# agrees with `screen` (37.6%) but with `lighten` on only 17.1%; the third pair at
# about a third is `screen`-`lighten` (37.5%), as the design's deps section has it.
# Verify: the rendered agreement table above the sentence.
rf"""
Each op has {mix_counts()["lines"]:,} lines, of which {mix_counts()["red"]:,} are red and {mix_counts()["nonred"]:,} non-red by dose, the larger of the two operand rednesses. Those counts are the same for every op, since dose reads the operands.

{op_table_md()}

The five named rules are the blend modes of the same names in Photoshop and Krita, on the 0..15 scale; `add` is what Photoshop calls *linear dodge*. `mix` is a normal blend at half opacity, which is the per-channel mean.

All six ops are commutative, as `mix` was in D2.1, so operand order carries no information and the two operand roles stay interchangeable.

The set leans light. `add`, `screen`, and `lighten` can only raise a channel; `multiply` and `darken` can only lower one; `mix` sits between its operands. A `subtract` op would balance `add`, but it would also be the first op whose operand order mattered, so the set stays as it is.

**Agreement.** How often each pair of ops gives the same answer. On a line where two ops agree, reading the op word gains nothing:

{agreement_md()}

`screen` agrees with `add` on about a third of pairs and with `lighten` on about a third as well, at the light end of the range where all three saturate; `multiply` agrees with `darken` at the dark end. `add` and `lighten` agree on a sixth of pairs. Every other pair agrees on a few percent at most, at the ends of the range.

**Op-relevance.** Take the lines of one op and ask of each line: how many of the other five ops would have given the same answer for this pair? Call that count k. Where k = 0, no other op matches, so the model has to read the op word to get the line right. Where k is larger, the op word narrows the six ops to k + 1, all of which give the same answer, so the model could answer the line without reading it.

The table gives, for each op we might anchor, the share of its lines at each k. The suppression experiments rest on this: when the op concept is removed, only the lines at k = 0 can show it, so an op with most of its lines there makes a sharper test. Which op we anchor stays open until the suppression prereg:

{relevance_md()}

### The corpus
"""

rf"""
{ex.N_LINES:,} lines, the same count as D2.1. Ops are drawn uniformly, and within an op the training pairs are drawn uniformly with random operand order, the way ex-2.1.10 drew the `mix` pairs. For each op, a fifth of its distinct pairs is held out, keyed on (op, pair), so a pair held out under `add` may still be trained under `mix`. Corpus seed {ex.CORPUS_SEED}.

D2.1 drew its {ex.N_LINES:,} lines from 5,832 distinct ones, so the corpus held about seventeen copies of each. Here the {ex.N_LINES:,} lines come from six times {mix_counts()["lines"]:,} distinct ones, so most distinct lines appear at most once, and most pairs of an op never appear under that op. In E4, the `corpus` arms keep the {ex.N_LINES:,} lines and narrow the op set, while the `per-op` arms narrow the op set and keep {ex.CONTROL.lines_per_op:,} lines per op.

The operands are the whole grid for every op, as they were in D2.1; what differs by op is where the answers land. `add` sends a fifth of its pairs to white, `screen` crowds the light half of the cube and `multiply` the dark half, and `mix`, `lighten`, and `darken` spread their answers through it.
"""

# Where each op's answers land: the number of unordered pairs whose answer is each grid color.
# Computed outside the plot function so `themed` renders both themes from one pass.
_grid = np.array(ex.colors(), dtype=float) / ex.TOP
_index = {c: i for i, c in enumerate(ex.colors())}
_pairs = ex.unordered_pairs()


@memo
def count_answers() -> dict[str, np.ndarray]:
    """Per op, how many unordered operand pairs land on each grid color."""
    out = {}
    for op in ex.OPS:
        n = np.zeros(len(_grid))
        for c, k in Counter(op(a, b) for a, b in _pairs).items():
            n[_index[c]] = k
        out[op.name] = n
    return out


answer_counts = count_answers()
FULL_CELL = 600
# Pairs at which a mark fills its grid cell. Area is proportional to the count, so `add`'s white
# corner (a fifth of all pairs) overflows its cell, which is the point.


@memo
@themed(
    name="answer-clouds",
    alt_text="""
        Six hexagonal color-cube panels, one per op, with a mark on each grid color sized by how many pairs answer there. mix, lighten, and darken spread their answers through the whole cube; add sends a fifth of its pairs to one large white mark at the top, screen crowds the light half, and multiply the dark half.
    """,
    caption=f"""
        **Where each op's answers land.** One mark per grid color, with area proportional to the number of unordered pairs whose answer is that color, on one scale for all six panels: a mark that fills its grid cell stands for {FULL_CELL} pairs, out of {len(_pairs):,}.
    """,
)
def answer_clouds(grid: np.ndarray, counts: dict[str, np.ndarray]) -> plt.Figure:
    from matplotlib.collections import EllipseCollection

    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.8))
    # Drawn by hand rather than through `plot_rgb_cube`, whose sized marks carry no edge: the
    # white corner of `add` and the black corner of `multiply` are the largest marks on the
    # page, and each vanishes against the cube's fill in one theme without a faint outline.
    order = np.argsort(grid @ CUBE_VIEWS["solid"].toward, kind="stable")  # nearer draws last
    xy = project_cube(grid[order])
    for ax, op in zip(axes.flat, ex.OPS, strict=True):
        draw_cube_bound(ax)
        dia = grid_diameter(len(ex.LEVELS)) * np.sqrt(counts[op.name][order] / FULL_CELL)
        ax.add_collection(
            EllipseCollection(
                widths=dia,
                heights=dia,
                angles=0,
                units="xy",
                offsets=xy,
                offset_transform=ax.transData,
                facecolors=grid[order],
                edgecolors=light_dark("#00000033", "#ffffff55"),
                linewidths=0.5,
                zorder=3,
                clip_on=False,
            )
        )
        # Lifted clear of the overflowing corner marks, which are unclipped.
        ax.set_title(op.name, y=1.12)
    return fig


answer_clouds(_grid, answer_counts)

r"""
### Training
"""

rf"""
D64-L4 nGPT with the ex-2.1.3 data config. A step is a batch of {ex.BATCH} random crops of {ex.BLOCK} tokens (about ten lines each) from the tokenized corpus. An epoch is enough batches to sample a quarter of the training tokens, which the loader sets with an `oversample` of {ex.OVERSAMPLE} over the batch size. At the corpus size of D2.1 that comes to {ex.steps_per_epoch()} steps per epoch, as it did there.

Peak LR is {ex.PEAK_LR:g}. Anchoring uses the D2.1 code path: the pooled either-operand labeller (each operand draws at redness⁸ × {ex.PER_SLOT_RATE:g}), the anchor on e₁ at every slice, and the anti-subspace term.

Conditions are matched on compute, at the step count of D2.1, rather than on samples per op. So the per-step regime of the anchor matches D2.1, and how often the task sees each pair becomes a covariate that H1 reads.

The recipe runs for the {ex.EPOCHS} epochs of ex-2.1.10; the proposals and its short arm run for the {ex.EPOCHS_SHORT} of the survey. The `per-op` arms of E4 keep the step count and shrink the corpus, so they get more and shorter epochs; the conditions table prints them.

Every schedule keyframe is a fraction of training, as ex-2.1.11 restated them: LR and anchor warm-up over the first {ex.WARMUP_FRAC:.0%}, the anchor anneal for the recipe over the last {1 - ex.ANNEAL_START_FRAC:.0%} down to a {ex.ANNEAL_FLOOR:g} floor, and the anti-subspace weight annealing from its peak ratio to {ex.ANTI_HOLD_RATIO:g} of the anchor weight by its own end fraction. The parameters for the proposals are the sampled values from the survey, unrounded, in `experiment.py`.

### The probe set
"""

rf"""
The probe set is a fixed list of equations that every run is scored on, separately from training: {ex.N_PROBE} lines per color as op1, per op. For `mix` these are its on-grid partners, so the `mix` probe set is the D2.1 one with the op word changed. For the other ops we draw {ex.N_PROBE} partners per color once with seed {ex.PROBE_SEED}, and share them across every run.

Each op then has {len(ex.mix_probe_lines()):,} probe lines, red and non-red by dose. We keep the infix frame,[^infix] so the probe positions in `sca.compute.evaluation` still read correctly.

[^infix]: An infix operator sits between its operands, as in `c1 mix c2`, rather than before or after them. The frame is the fixed six-token layout of a line: op1, the op word, op2, `=`, the answer, and the newline sit at the same positions in every line, so the probe code can read a role by its position.

### Measurements
"""

rf"""
The placement statistics are the ones from ex-2.1.10, computed by its code on the probe lines of each op from one clean pass per run: m_line and the softmin profiles, containment, the lead weight, contrast, grading against the sim^1.5 target, and the latch. The trajectory records m_line every {ex.TRAJ_STRIDE} steps. Holdout exact match is read on a held-out set per op, of the size that ex-2.1.10 used.

The interventions run one teacher-forced pass per run per operator over the six probe sets, through the eval contract, giving the log-softmax at `=` and the write per (slice, line, position) as in ex-2.2.1. Red-line accuracy and the non-red deficit are read per op.

**Answer distance.** Exact match scores a one-step miss and a far miss the same way. Under suppression, ex-2.2.1 found that most misses on red lines were one-step neighbors of the true answer. So beside each accuracy we read the distance in the unit cube from the decoded answer to the true one, as ex-2.2.1 read it on the red lines.

A second version takes the expected value of that distance under the answer distribution, using the whole log-softmax rather than only its argmax. Both are computed per line and averaged per group, for every operator row, and neither is gated.

**Noise floor.** The bands use the per-run σ of each statistic at the reference recipe, from the nine seeds of ex-2.1.10, printed in the H3 table; E5 re-measures them here. Gates score seed means against fixed thresholds and do not use the floor. Every difference between two conditions is quoted with its band, and a difference inside the band is reported as not resolved.

### Budget
"""

rf"""
{ex.N_RUNS} training runs: {ex.CONTROL.seeds + ex.RECIPE.seeds} at the full length of the recipe, {sum(c.seeds for c in ex.PROPOSALS) + ex.RECIPE_SHORT.seeds + ex.CONTROL_SHORT.seeds} at half of it, and {sum(c.seeds for c in ex.RICHER_OP_ARM)} for the richer-op arms at the full length as well, each at a plain D2.1 step count on an L4. Scoring is one clean pass plus four operator passes over six probe sets per run, taking CPU seconds each, and the cube probes of E4 are ridge fits on 216 rows. That is well under the cost of ex-2.2.2, which trained 24 runs at twice the step cost. The run took 41 minutes of wall-clock on Modal at twelve containers, 15 of them training, and cost $2.33, with no task failed or retried.

### Before the freeze

Before the freeze we run a calibration: one seed of each control arm on the six-op corpus. It checks that the grammar is learned at both lengths, which means holdout exact match per op near 1, the level the D2.1 control reached. The rounded ops are the ones to watch. If the grammar is not learned at one length or the other, we change the corpus size, the step count, or the set of arms before the freeze, and record the change with a `REVIEW` note. Nothing anchored runs before the freeze.

**The calibration ran.** One seed of each control arm, published under `{ex.CALIBRATION_REF}`; the table below reads it. Both lengths learn all six ops: holdout exact match is 1.0 on every op but one at each length, and that one misses a single line of its {ex.N_EVAL}. The rounded ops are as clean as the others, and held-out surprisal is below 0.02 nats everywhere. So the corpus, the step counts, and the set of arms stay as designed, and `control-short` stays too, since its numbers say it will read the proposals' task cost at their own length without a training deficit of its own. Nothing about the arms or the gates changed after this read.

{calibration_md(calibration)}

/// details | Glossary
- **line** — one equation, `c1 ‹op› c2 = answer`, six word-level tokens. The op word is one token, like each color. Bare *op* is the operation; *op1* and *op2* are the operand roles.
- **on-grid pair** — a pair whose answer under an op lands on the grid without rounding. The on-grid pairs of `mix` are the closed pairs of D2.1, and its probe lines.
- **residual stream** — the running vector each token carries through the network, which every block reads from and writes to.
- **slice** ($\ell$) — a depth at which the residual stream is read: the embedding (0), plus the stream after each of the four blocks.
- **alignment** ($\alpha$) — $\cos(h, e_1)$, the cosine between a state and the anchor axis. States are unit-norm, so this is the e₁ component.
- **dose** — how *red* a line is, the larger of the two operand rednesses ($r(1 - g/2 - b/2)$ on the unit cube). **Red lines** have dose ≥ {ex.RED_DOSE:g}; **non-red lines** dose ≤ {ex.NONRED_DOSE:g}. The labeller draws at dose⁸, so a dark red such as (12,0,0) draws at about a sixth of pure red's rate and (9,0,0) at under 2%: dark reds are barely labelled, and on the grid the red lines are the seven operands with R = 15 or (12,0,0).
- **softmin weight** ($\pi$) — the share of the pooled pull a span position receives; sums to 1 over the span. A profile is these shares over roles (op1, the op word, op2, …) at one slice.
- **m_line** — the margin from ex-2.1.10: at the best span role, the label-weighted mean alignment minus the unweighted mean, averaged over slices. The one statistic tight enough to rank operating points on.
- **containment** ($\bar\alpha$) — the mean alignment over all 216 colors at op1. A pull that latches onto op1 as a position rather than a concept drives it up.
- **retention** — the final m_line as a fraction of its running peak over training.
- **grading** ($r^2$) — the squared Pearson correlation between op1 alignment and the sim^1.5 target across the 216 colors. It says whether alignment rises in proportion to redness rather than switching at a threshold.
- **contrast** — the difference in op2 softmin weight between the op2-triggered and op1-triggered line groups, averaged over the four post-attention slices. It says whether the pull found the operand that carried the label.
- **latch** — a run whose non-red group puts more than {ex.LATCH_PI:g} of its softmin weight on op1; vetoed per run.
- **deficit** — clean exact-match accuracy minus intervened accuracy over a group of lines. This is the statistic ex-2.2.1 gated selectivity on.
- **band** — the smallest seed-mean difference the resolution rule may call a difference: $2\sigma\sqrt{{1/n_a + 1/n_b}}$, with σ the per-run spread.
///
"""
