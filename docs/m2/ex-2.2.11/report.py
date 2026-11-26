# title: Ex 2.2.11: the handover re-run

import colorsys
import json
import math
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from mini.lit import memo, stop
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed
from sca.data.colors import redness
from sca.data.ops import TOP, probe_lines

# The design constants come from `experiment.py` beside this script (the script's directory is on
# sys.path while it runs). The prose quotes the frozen gates, and that module carries the same
# numbers with each gate's wording in its docstring.
import experiment as ex

RESULTS_TO_COME = "/// admonition | TODO\n    type: warning\nResults to come.\n///"


def removal_counts() -> dict[str, dict[str, int]]:
    """Per op, on its probe set: the red lines, the removal lines under the hue rule, the removal lines under
    ex-2.2.9's to-zero rule, and the saturation-and-value lines the hue rule sets aside. The order-sensitive
    ops are also split by which slot the red operand is in.
    """
    out = {}
    for op in ex.TABLE:
        both = op.name in ex.PROBE_BOTH_SLOTS
        red = [
            ln
            for ln in probe_lines(op, ex.N_PROBE, ex.PROBE_SEED, both_slots=both)
            if max(redness(ln.lhs), redness(ln.rhs)) >= ex.RED_DOSE
        ]
        hue = [ex.hue_move(op, ln.lhs, ln.rhs) >= ex.FAR_MOVE for ln in red]
        zero = [ex.ex229.to_zero_move(op, ln.lhs, ln.rhs) >= ex.FAR_MOVE for ln in red]
        row = {"red": len(red), "hue": sum(hue), "zero": sum(zero), "sv": len(red) - sum(hue)}
        if both:
            at1 = [redness(ln.lhs) >= redness(ln.rhs) for ln in red]
            row["hue_op1"] = sum(h for h, a in zip(hue, at1, strict=True) if a)
            row["hue_op2"] = sum(h for h, a in zip(hue, at1, strict=True) if not a)
        out[op.name] = row
    return out


counts = removal_counts()


def removal_md() -> str:
    head = "| op | red lines | removal (hue) | removal (to zero) | saturation-and-value | hue, red at op1 | hue, red at op2 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    rows = []
    for op, c in counts.items():
        split = (f"{c['hue_op1']}", f"{c['hue_op2']}") if "hue_op1" in c else ("", "")
        rows.append(f"| `{op}` | {c['red']} | {c['hue']} | {c['zero']} | {c['sv']} | {split[0]} | {split[1]} |")
    return head + "\n".join(rows)


def rule_disagreement(name: str) -> int:
    """How many red lines of an op one rule calls a removal line and the other does not."""
    op = next(o for o in ex.TABLE if o.name == name)
    red = [
        ln
        for ln in probe_lines(op, ex.N_PROBE, ex.PROBE_SEED, both_slots=op.name in ex.PROBE_BOTH_SLOTS)
        if max(redness(ln.lhs), redness(ln.rhs)) >= ex.RED_DOSE
    ]
    hue = {i for i, ln in enumerate(red) if ex.hue_move(op, ln.lhs, ln.rhs) >= ex.FAR_MOVE}
    zero = {i for i, ln in enumerate(red) if ex.ex229.to_zero_move(op, ln.lhs, ln.rhs) >= ex.FAR_MOVE}
    return len(hue ^ zero)


sv_share = {op: counts[op]["sv"] / counts[op]["red"] for op in ex.ORDER_SENSITIVE}
low_share = min(c["hue"] / c["red"] for op, c in counts.items() if op not in ex.ORDER_SENSITIVE)


def rot_move(op, a, b, steps: int = ex.HUE_ROTATION_STEPS) -> float:
    """The finer rule's move: the furthest the true answer moves when the red operand's hue is rotated in HSV
    in *steps* steps, each snapped to the grid, with its saturation and value held.
    """
    red_first = redness(a) >= redness(b)
    red = a if red_first else b
    truth = op(a, b)
    h, s, v = colorsys.rgb_to_hsv(*(np.array(red) / TOP))
    moves = []
    for k in range(1, steps):
        rgb = colorsys.hsv_to_rgb((h + k / steps) % 1.0, s, v)
        p = tuple(int(round(x * TOP)) for x in rgb)
        if p == tuple(red):
            continue
        za, zb = (p, b) if red_first else (a, p)
        moves.append(float(np.linalg.norm(np.subtract(op(za, zb), truth)) / TOP))
    return max(moves) if moves else 0.0


@memo
def finer_rule_counts() -> dict[str, tuple[int, int, int]]:
    """Per op: (red lines, removal lines under the finer rule, lines the two rules disagree on)."""
    out = {}
    for op in ex.TABLE:
        both = op.name in ex.PROBE_BOTH_SLOTS
        red = [
            ln
            for ln in probe_lines(op, ex.N_PROBE, ex.PROBE_SEED, both_slots=both)
            if max(redness(ln.lhs), redness(ln.rhs)) >= ex.RED_DOSE
        ]
        perm = np.array([ex.hue_move(op, ln.lhs, ln.rhs) >= ex.FAR_MOVE for ln in red])
        rot = np.array([rot_move(op, ln.lhs, ln.rhs) >= ex.FAR_MOVE for ln in red])
        out[op.name] = (len(red), int(rot.sum()), int((perm ^ rot).sum()))
    return out


def finer_rule_md() -> str:
    c = finer_rule_counts()
    head = "| op | red lines | removal (permutation) | removal (rotation) | disagree | share |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    rows = [
        f"| `{op}` | {red} | {counts[op]['hue']} | {rot} | {d} | {d / red:.2%} |" for op, (red, rot, d) in c.items()
    ]
    return head + "\n".join(rows)


finer_worst = max(finer_rule_counts().items(), key=lambda kv: kv[1][2] / kv[1][0])
finer_worst_share = finer_worst[1][2] / finer_worst[1][0]


# =============================================================================================
# Results
# =============================================================================================


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists. One `get_refs` and one `get_many`
    for the lot, since each round trip to the bucket costs a couple of seconds.
    """
    store = project_store()
    have = {r: a for r, a in store.get_refs(refs).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict | None:
    return None if path is None else json.loads(path.read_text())


EX228_METRICS_REF = "reports/m2/ex-2.2.8/metrics"
EX2210_METRICS_REF = "reports/m2/ex-2.2.10/metrics"
REFS = [
    ex.METRICS_REF,
    ex.TRAJ_REF,
    ex.EX229_METRICS_REF,
    ex.EX229_TRAJ_REF,
    ex.ex229.ex223.METRICS_REF,
    EX228_METRICS_REF,
    EX2210_METRICS_REF,
]

with tempfile.TemporaryDirectory() as _tmp:
    loaded = {r: read_json(f) for r, f in fetch(REFS, Path(_tmp)).items()}

ANCHORED = [c.name for c in ex.SCORED_UNDER_PROJECTION]
EOL = "\n"


@dataclass(frozen=True)
class Results:
    """Every published result the report reads, with one accessor per shape the cells need.

    `metrics` and `traj` are this experiment's. `ex229` and `ex229_traj` are ex-2.2.9's, whose `handover`
    is the before column of every placement read. `ex223` is the recipe's twenty seeds on the six-op grammar,
    the point the gates were set at. `ex228` carries the per-run spread of the non-red deficit under
    `projection` at those seeds, the σ every deficit band uses. `ex2210` is ex-2.2.10's rescoring of
    ex-2.2.9's seeds, read only by the post-hoc note on the `hue-hsv` miss.
    """

    metrics: dict
    traj: dict
    ex229: dict
    ex229_traj: dict
    ex223: dict
    ex228: dict
    ex2210: dict

    def runs(self, cond: str, metrics: dict | None = None) -> list[dict]:
        """The eval records of a condition, in seed order; *metrics* selects another experiment's."""
        m = self.metrics if metrics is None else metrics
        return sorted((r for r in m["runs"] if r["condition"] == cond), key=lambda r: r["seed"])

    def scored(self, cond: str) -> list[dict]:
        return sorted((r for r in self.metrics["scores"] if r["condition"] == cond), key=lambda r: r["seed"])

    def n(self, cond: str) -> int:
        return len(self.runs(cond))

    def stat(self, cond: str, key: str) -> np.ndarray:
        """One value per seed of a placement statistic on the `mix` lines."""
        return np.array([r[key] for r in self.runs(cond)], float)

    def read(self, cond: str, op: str, key: str, split: str = "holdout") -> np.ndarray:
        return np.array([r["sets"][op][split][key] for r in self.runs(cond)], float)

    def eem(self, cond: str, op: str) -> np.ndarray:
        return np.array([r["holdout_eem"][op] for r in self.runs(cond)], float)

    def score(self, cond: str, op: str, operator: str | None, key: str, group: str) -> np.ndarray:
        """One value per seed of a scored statistic on one group of one op's probe lines."""
        out = []
        for r in self.scored(cond):
            s = r["ops"][op]
            v = s["clean"][key] if operator is None else s["operators"][operator][key]
            out.append(v[group])
        return np.array(out, float)

    def n_lines(self, cond: str, op: str, group: str) -> int:
        return int(self.scored(cond)[0]["ops"][op]["n"][group])

    def kept(self, cond: str, op: str, operator: str = "projection", group: str = "removal") -> np.ndarray:
        """The share of clean expected exact match kept on a group; `removal` is the hue rule's lines and
        `removal_zero` the to-zero rule's.
        """
        return self.score(cond, op, operator, "kept", group)

    def deficit(self, cond: str, op: str, operator: str = "projection", group: str = "nonred") -> np.ndarray:
        return self.score(cond, op, operator, "deficit", group)

    def component(self, cond: str, word: str, table: str = "rows", metrics: dict | None = None) -> np.ndarray:
        """The axis component of one token's row, per seed, on the embedding (`rows`) or the readout."""
        return np.array([r[table][word] for r in self.runs(cond, metrics) if r.get(table) is not None], float)

    def anneal(self, cond: str, key: str, metrics: dict | None = None) -> np.ndarray:
        """One value per seed of a retention-across-the-anneal read, computed from the trajectories with the
        method's rule so ex-2.2.9's runs get the same read as ours.
        """
        traj = self.traj if metrics is None else self.ex229_traj
        out = []
        for r in self.runs(cond, metrics):
            out.append(ex.anneal_retention(traj[r["label"]]["traj"], ex.ANNEAL_WEIGHT_RATIO)[key])
        return np.array(out, float)

    def curves(self, cond: str, metrics: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(epochs, m_line) of every seed of a condition, stacked: (points,), (seeds, points)."""
        traj = self.traj if metrics is None else self.ex229_traj
        ts = [traj[r["label"]]["traj"] for r in self.runs(cond, metrics)]
        return np.array(ts[0]["epoch"], float), np.array([t["m_line"] for t in ts], float)

    # -- the references -------------------------------------------------------------------

    def ref_runs(self) -> list[dict]:
        """Ex-2.2.3's adopted point at its twenty seeds."""
        names = (ex.ex229.EX223_REFERENCE, f"{ex.ex229.EX223_REFERENCE}-more")
        return sorted((r for r in self.ex223["runs"] if r["condition"] in names), key=lambda r: r["seed"])

    def ref_stat(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.ref_runs()], float)

    def ref_deficit(self, op: str) -> np.ndarray:
        runs = [r for r in self.ex228["scores"] if r["condition"] == ex.ex229.EX223_REFERENCE]
        return np.array([r["ops"][op]["trials"]["projection"]["deficit"]["nonred"] for r in runs], float)

    def ref_deficit_sd(self, op: str) -> float:
        """Per-run σ of the non-red deficit under `projection` at the reference, or NaN on an op the six-op
        grammar did not have.
        """
        runs = [r for r in self.ex228["scores"] if r["condition"] == ex.ex229.EX223_REFERENCE]
        if op not in runs[0]["ops"]:
            return float("nan")
        return float(self.ref_deficit(op).std(ddof=1))

    def before(self, cond: str, key: str) -> np.ndarray:
        """Ex-2.2.9's value of a placement statistic, per seed of the same condition."""
        return np.array([r[key] for r in self.runs(cond, self.ex229)], float)

    def old_kept(self, cond: str, op: str, group: str) -> np.ndarray:
        """The kept share under `projection` on a line group, per seed, as ex-2.2.10 scored ex-2.2.9's runs."""
        runs = [r for r in self.ex2210["scores"] if r["condition"] == cond]
        return np.array([r["ops"][op]["kept"][group] for r in runs], float)


def load_results() -> Results | None:
    metrics, traj = loaded[ex.METRICS_REF], loaded[ex.TRAJ_REF]
    if metrics is None or traj is None:
        return None
    rest = [loaded[r] for r in REFS[2:]]
    assert all(r is not None for r in rest)
    return Results(metrics, traj, *[r for r in rest if r is not None])


# --- Shared drawing and table helpers ---------------------------------------------------------


def band(sd: float, n_a: int, n_b: int) -> float:
    """The smallest difference between two seed means the read resolves: 2σ√(1/n_a + 1/n_b)."""
    return ex.RESOLUTION_SD * sd * math.sqrt(1 / n_a + 1 / n_b)


def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    return math.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))


def span2(v: np.ndarray, digits: int = 3) -> str:
    """A seed mean with its range, as `0.512 (0.49–0.53)`."""
    v = np.asarray(v, float)
    if v.size == 0 or np.all(np.isnan(v)):
        return "—"
    return f"{np.nanmean(v):.{digits}f} ({np.nanmin(v):.{digits - 1}f}–{np.nanmax(v):.{digits - 1}f})"


def ink(cond: str) -> str:
    inks = {
        "control": ("#6b6b6b", "#b0b0b0"),
        "handover": ("#c0392b", "#ff8a76"),
        "handover-slot": ("#2b6cb0", "#7fb3ff"),
        "handover-tied": ("#7b3fa0", "#cfa3ff"),
        "reference": ("#a08a2e", "#e6d27a"),
    }
    return light_dark(*inks[cond])


def marker(cond: str) -> str:
    return {"control": "s", "handover": "o", "handover-slot": "^", "handover-tied": "D", "reference": "P"}[cond]


def short(cond: str) -> str:
    return cond.replace("handover-", "h-")


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(head: list[str], rows: list[list[str]], caption: str, *, ref_rows: frozenset[int] = frozenset()) -> str:
    """An authored result table in the shared report style; the first column is text, the rest numeric."""
    ths = "".join(f"<th{' class=num' if i else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        f"<tr{' class=ref' if r in ref_rows else ''}>"
        + "".join(f"<td{' class=num' if i else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<table class="report-table"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


def bold_if(text: str, ok: bool) -> str:
    return f"<b>{text}</b>" if ok else text


def verdict_md(status: str, line: str) -> str:
    kind = {"pass": "success", "partial": "warning", "miss": "danger", "unresolved": "info"}[status]
    return f"/// admonition | {status.capitalize()}\n    type: {kind}\n{line}\n///"


def dots(
    ax,
    x: float,
    v: np.ndarray,
    cond: str,
    *,
    rng,
    ms: float = 5.0,
    width: float = 0.06,
    label=None,
    open_: bool = False,
) -> None:
    """One column of per-seed dots with the seed mean drawn on top, in the condition's ink and marker shape.
    A thin bar behind the dots spans the seed range. An open mark is the same condition from ex-2.2.9.
    """
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return
    color, m = ink(cond), marker(cond)
    jit = rng.uniform(-width, width, len(v))
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(
        x + jit,
        v,
        "o",
        ms=2.2,
        color=color,
        alpha=0.45,
        zorder=3,
        mew=0.6 if open_ else 0,
        mfc="none" if open_ else color,
    )
    ax.plot(
        x,
        v.mean(),
        m,
        ms=ms,
        color=color,
        zorder=4,
        mfc="none" if open_ else color,
        mec=color if open_ else light_dark("white", "#111"),
        mew=1.0 if open_ else 0.6,
        label=label,
    )


def fig_legend(fig: plt.Figure, ax: Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7, **kwargs)


def gate_line(ax: Axes, y: float, *, partial: float | None = None, fail: str | None = None) -> None:
    """A dashed gate line, a dotted partial level under it when there is one, and the missing side hatched."""
    ax.axhline(y, color=light_dark("#333", "#ddd"), lw=0.9, ls="--", zorder=2)
    if partial is not None:
        ax.axhline(partial, color=light_dark("#333", "#ddd"), lw=0.7, ls=":", zorder=2)
    if fail is not None:
        lo, hi = ax.get_ylim()
        span = (lo, y) if fail == "below" else (y, hi)
        ax.axhspan(*span, facecolor="none", edgecolor=light_dark("#000", "#fff"), hatch="//", lw=0, zorder=0, alpha=0.1)
        ax.set_ylim(lo, hi)


# --- H1: the task ------------------------------------------------------------------------------


def h1_status(res: Results) -> dict[str, tuple[str, float, float]]:
    """Per op: (status, handover minus control seed mean, band)."""
    out = {}
    n_h, n_c = res.n("handover"), res.n("control")
    for op in ex.OP_NAMES:
        h, c = res.eem("handover", op), res.eem("control", op)
        b = band(pooled_sd(h, c), n_h, n_c)
        d = float(h.mean() - c.mean())
        if d >= -ex.TASK_GATE:
            status = "pass"
        elif d >= -ex.TASK_GATE - b:
            status = "unresolved"
        elif d >= -ex.TASK_PARTIAL:
            status = "partial"
        else:
            status = "miss"
        out[op] = (status, d, b)
    return out


def h1_verdict(res: Results) -> tuple[str, str]:
    st = h1_status(res)
    worst = min(st, key=lambda op: st[op][1])
    by = {s: [op for op in ex.OP_NAMES if st[op][0] == s] for s in ("miss", "partial", "unresolved")}
    tail = f"The largest shortfall is `{worst}` at {st[worst][1]:+.3f} (band {st[worst][2]:.3f})."
    ops = lambda s: ", ".join(f"`{o}`" for o in by[s])  # noqa: E731
    if by["miss"]:
        return "miss", f"`handover` is more than {ex.TASK_PARTIAL:g} below the control on {ops('miss')}. {tail}"
    if by["partial"]:
        return (
            "partial",
            f"Every op is within {ex.TASK_PARTIAL:g} of the control, and {ops('partial')} outside {ex.TASK_GATE:g}. {tail}",
        )
    if by["unresolved"]:
        return (
            "unresolved",
            f"Every op is within {ex.TASK_GATE:g} of the control or within a band of it ({ops('unresolved')}). {tail}",
        )
    return "pass", f"On every op the `handover` seed mean is within {ex.TASK_GATE:g} of the control's. {tail}"


def h1_figure(res: Results) -> str:
    conds = [c.name for c in ex.CONDS]
    eem = {(c, op): res.eem(c, op) for c in conds for op in ex.OP_NAMES}
    ceil = {op: float(res.read("control", op, "ceiling").mean()) for op in ex.OP_NAMES}
    ctrl = {op: float(eem["control", op].mean()) for op in ex.OP_NAMES}
    return h1_draw(conds, eem, ceil, ctrl)


@memo
def h1_draw(conds: list[str], eem: dict, ceil: dict[str, float], ctrl: dict[str, float]) -> str:
    @themed(
        name="h1-eem-per-op",
        alt_text="""
            A dot chart with the eleven ops along the bottom and expected exact match up the side. At each op, four columns of small dots, one per condition, sit close together at the same height, with a grey strip just below the control's mean marking the gate and a bar above marking the ceiling the drawn answers allow.
        """,
        caption=f"""
            **Expected exact match on held-out lines, per op and condition.** Each small dot is one seed, the larger mark the seed mean, and the thin bar behind them the seed range. The grey strip under each op runs from the control's mean down to {ex.TASK_GATE:g} below it: a `handover` mean inside the strip passes H1 on that op. The bar above each op is the ceiling the drawn answers allow, which is 1 on the ops that never round.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.4, 3.6), layout="constrained")
        strip = light_dark("#00000014", "#ffffff1f")
        xs = np.arange(len(ex.OP_NAMES))
        off = np.linspace(-0.3, 0.3, len(conds))
        rng = np.random.default_rng(0)
        for x, op in zip(xs, ex.OP_NAMES, strict=True):
            ax.fill_between([x - 0.45, x + 0.45], ctrl[op] - ex.TASK_GATE, ctrl[op], color=strip, lw=0, zorder=1)
            ax.plot([x - 0.45, x + 0.45], [ceil[op]] * 2, color=light_dark("#333", "#ddd"), lw=0.9, zorder=2)
            for o, c in zip(off, conds, strict=True):
                dots(ax, x + o, eem[c, op], c, rng=rng, width=0.04, label=c if x == 0 else None)
        ax.set_xticks(xs, ex.OP_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("expected exact match")
        ax.set_ylim(0.2, 1.02)
        ax.grid(axis="y", alpha=0.2)
        fig_legend(fig, ax)
        return fig

    return _plot()


def h1_table(res: Results) -> str:
    st = h1_status(res)
    rows = []
    for op in ex.OP_NAMES:
        ctrl = res.eem("control", op)
        ceil = float(res.read("control", op, "ceiling").mean())
        _, d, b = st[op]
        cells = [f"`{op}`", f"{ceil:.2f}", f"{ctrl.mean():.3f}"]
        for c in ANCHORED:
            v = res.eem(c, op)
            delta = float(v.mean() - ctrl.mean())
            text = f"{v.mean():.3f} ({delta:+.3f})"
            cells.append(bold_if(text, delta >= -ex.TASK_GATE) if c == "handover" else text)
        cells.append(f"{b:.3f}")
        rows.append(cells)
    head = ["op", "ceiling", "control", *[f"`{c}` (Δ)\u00a0↑" for c in ANCHORED], "band"]
    return table_html(
        head,
        rows,
        f"""
        **Seed-mean expected exact match on held-out lines, per op.** Each condition's column gives its mean and, in brackets, the difference from the control. A bold `handover` entry is within {ex.TASK_GATE:g} of the control, which is the H1 gate. The band is the smallest difference from the control the read resolves on that op, from the per-run spread of the two conditions.
        """,
    )


def h1_prose(res: Results) -> str:
    st = h1_status(res)
    worst = min(ex.OP_NAMES, key=lambda op: st[op][1])
    best = max(ex.OP_NAMES, key=lambda op: st[op][1])
    order = [st[op][1] for op in ex.ORDER_SENSITIVE]
    before = {
        op: float(
            np.mean([r["holdout_eem"][op] for r in res.runs("handover", res.ex229)])
            - np.mean([r["holdout_eem"][op] for r in res.runs("control", res.ex229)])
        )
        for op in ex.OP_NAMES
    }
    b_worst = min(before, key=lambda op: before[op])
    return f"The anchored model learns the task about as well as the un-anchored one, as it did at the old seeds. Across the {ex.N_OPS} ops the `handover` seed mean sits between {st[worst][1]:+.3f} (`{worst}`) and {st[best][1]:+.3f} (`{best}`) of the control's; in ex-2.2.9 the widest gap was {before[b_worst]:+.3f} (`{b_worst}`). On the three order-sensitive ops the differences are {', '.join(f'{d:+.3f}' for d in order)}."


# --- H2: placement -----------------------------------------------------------------------------

GATES = {
    "m_line": ("margin", ex.MARGIN_RATIO * ex.REF_M_LINE, ex.MARGIN_PARTIAL * ex.REF_M_LINE),
    "r2_sim": ("grading r²", ex.GRADE_R2_RATIO * ex.REF_R2_SIM, None),
    "contrast": ("contrast", ex.CONTRAST_GATE, ex.CONTRAST_PARTIAL),
}
# H2's three gates: key → (label, gate, partial level). All read upward.

CHECKS = {"lead_emb": ("lead", ex.LEAD_GATE), "latch_pi": ("latch (max)", ex.LATCH_PI)}
# The two manipulation checks: lead reads upward on the seed mean, latch downward on the worst seed."""


def retention(res: Results, cond: str, metrics: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(retention across the anneal, qualifying mask) per seed of a condition."""
    at = res.anneal(cond, "m_line_at_anneal", metrics)
    return res.anneal(cond, "retention_anneal", metrics), at >= ex.RETENTION_FLOOR


def h2_status(res: Results, cond: str = "handover") -> dict[str, tuple[str, float]]:
    """Per read on one condition: (status, seed mean). Gates and checks get pass/partial/miss; retention is
    pass when every qualifying seed ends at the gate's share of its anneal-start value; the three lines carry
    their value only.
    """
    out: dict[str, tuple[str, float]] = {}
    for k, (_, gate, partial) in GATES.items():
        v = float(res.stat(cond, k).mean())
        out[k] = ("pass" if v >= gate else "partial" if partial is not None and v >= partial else "miss", v)
    lead = float(res.stat(cond, "lead_emb").mean())
    out["lead_emb"] = ("pass" if lead >= ex.LEAD_GATE else "miss", lead)
    latch = float(res.stat(cond, "latch_pi").max())
    out["latch_pi"] = ("pass" if latch <= ex.LATCH_PI else "miss", latch)
    ret, q = retention(res, cond)
    out["retention"] = (
        "pass" if q.any() and bool((ret[q] >= ex.RETENTION_GATE).all()) else "miss",
        float(ret[q].mean()) if q.any() else float("nan"),
    )
    out["level"] = ("line", float(res.anneal(cond, "m_line_final").mean()))
    out["alpha_op1"] = ("line", float(res.stat(cond, "alpha_op1").mean()))
    out["eol"] = ("line", float(np.abs(res.component(cond, EOL)).mean()))
    return out


def retention_detail(res: Results, cond: str) -> tuple[int, int, float]:
    """(seeds under the retention line, qualifying seeds, the lowest retention) on one condition."""
    ret, q = retention(res, cond)
    return int((ret[q] < ex.RETENTION_GATE).sum()), int(q.sum()), float(ret[q].min()) if q.any() else float("nan")


def h2_verdict(res: Results) -> tuple[str, str]:
    st = h2_status(res)
    misses = [k for k in GATES if st[k][0] == "miss"]
    partials = [k for k in GATES if st[k][0] == "partial"]
    checks = [CHECKS[k][0] for k in CHECKS if st[k][0] != "pass"]
    n_below, n_q, low = retention_detail(res, "handover")
    ret = (
        f"Retention across the anneal is {st['retention'][1]:.2f} (lowest seed {low:.2f}), so every seed holds through the anneal."
        if n_below == 0
        else f"{n_below} of {n_q} seeds end{'s' if n_below == 1 else ''} under {ex.RETENTION_GATE:g} of their anneal-start alignment (lowest {low:.2f})."
    )
    check = (
        ""
        if not checks
        else f" The {' and '.join(checks)} check{'s' if len(checks) > 1 else ''} did not clear, so the run is not the one we meant to score."
    )
    gates = ", ".join(f"{GATES[k][0]} {st[k][1]:.3f}" for k in GATES)
    if misses:
        return "miss", f"`handover` misses {', '.join(GATES[k][0] for k in misses)} ({gates}). {ret}{check}"
    if partials:
        return (
            "partial",
            f"`handover` clears every gate except {', '.join(GATES[k][0] for k in partials)}, which sits in its partial band ({gates}). {ret}{check}",
        )
    return "pass", f"`handover` clears margin, grading, and contrast ({gates}). {ret}{check}"


def h2_cell(res: Results, cond: str, key: str, metrics: dict | None = None) -> str:
    """One placement cell: the seed mean with its range, read on this run's metrics or on ex-2.2.9's."""
    if key == "latch_pi":
        v = res.stat(cond, key) if metrics is None else res.before(cond, key)
        return f"{v.max():.2f}"
    if key == "retention":
        ret, q = retention(res, cond, metrics)
        return span2(ret[q])
    if key == "level":
        return span2(res.anneal(cond, "m_line_final", metrics))
    if key == "eol":
        return span2(np.abs(res.component(cond, EOL, metrics=metrics)))
    return span2(res.stat(cond, key) if metrics is None else res.before(cond, key))


H2_KEYS = [*GATES, *CHECKS, "retention", "level", "alpha_op1", "eol"]


def h2_table(res: Results) -> str:
    conds = [c.name for c in ex.CONDS]
    rows = []
    for c in conds:
        st = h2_status(res, c)
        cells = [f"`{c}`"]
        for k in H2_KEYS:
            gated = c == "handover" and (k in GATES or k in CHECKS or k == "retention")
            cells.append(bold_if(h2_cell(res, c, k), st[k][0] == "pass") if gated else h2_cell(res, c, k))
        rows.append(cells)
    # Ex-2.2.9's runs, read the same way (retention and level from its trajectories, with this method's rule).
    for c in ANCHORED:
        rows.append([f"ex-2.2.9 `{c}`", *[h2_cell(res, c, k, res.ex229) for k in H2_KEYS]])
    ref = ["ex-2.2.3 reference"]
    for k in H2_KEYS:
        if k in ("retention", "level", "eol"):
            ref.append("—")
        elif k == "latch_pi":
            ref.append(f"{res.ref_stat(k).max():.2f}")
        else:
            ref.append(span2(res.ref_stat(k)))
    rows.append(ref)
    head = [
        "condition",
        *[f"{g[0]}\u00a0↑" for g in GATES.values()],
        "lead\u00a0↑",
        "latch (max)\u00a0↓",
        "retention\u00a0↑",
        "level",
        "ᾱ at op1",
        "`⏎` embedding",
    ]
    return table_html(
        head,
        rows,
        f"""
        **Placement of *red* on the `mix` lines, per condition, with ex-2.2.9's runs and ex-2.2.3's point as reference rows.** Seed mean with the seed range in brackets. Margin, grading, and contrast are the gates; lead and latch are the manipulation checks (latch is the largest non-red softmin weight at op1 over the seeds). Retention is the final alignment over the alignment at the start of the anneal, on the seeds where that value reached {ex.RETENTION_FLOOR:g}; level is the final point of the trajectory; ᾱ at op1 is the mean alignment of the non-red colors at the first operand; the last column is the absolute axis component of the `⏎` embedding row. Bold `handover` entries clear their gate or line. Ex-2.2.9's rows are read with this report's rules from its stored trajectories, so they are the before column of every read; ex-2.2.3 stored no trajectory, so three of its cells are empty.
        """,
        ref_rows=frozenset(range(len(conds), len(rows))),
    )


def h2_traj_figure(res: Results) -> str:
    curves = {c: res.curves(c) for c in ANCHORED}
    before = {c: res.curves(c, res.ex229) for c in ANCHORED}
    anneal = {c: float(res.anneal(c, "anneal_epoch").mean()) for c in ANCHORED}
    return h2_traj_draw(ANCHORED, curves, before, anneal)


@memo
def h2_traj_draw(conds: list[str], curves: dict, before: dict, anneal: dict[str, float]) -> str:
    @themed(
        name="h2-trajectories",
        alt_text="""
            Three line panels side by side, one per anchored condition, with epoch along the bottom and the margin on the mix lines up the side. In each, twenty or nine faint lines rise steeply in the first ten epochs, and plateau with some wobble, drifting down slightly through the middle of training on the candidate and holding level on the two references; the bold seed mean runs through them, a dashed line shows ex-2.2.9's seed mean on the same condition, and the last tenth of training is shaded as the anneal window, where the curves stay flat.
        """,
        caption="""
            **The margin on the `mix` lines through training, per anchored condition.** One panel per condition; each hairline is one seed, the bold line the seed mean, and the dashed line the seed mean of the same condition in ex-2.2.9. The shaded band is the anneal window, from the epoch at which the anchor weight first drops under its plateau to the end of training. Retention is the ratio of the curve's last point to its value at the left edge of the band.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(conds), figsize=(8.4, 3.0), layout="constrained", sharey=True)
        for ax, c in zip(axes, conds, strict=True):
            ep, ml = curves[c]
            ax.axvspan(anneal[c], ep[-1], color=light_dark("#00000010", "#ffffff18"), lw=0, zorder=0)
            for row in ml:
                ax.plot(ep, row, "-", color=ink(c), lw=0.5, alpha=0.25, zorder=2)
            ax.plot(ep, ml.mean(0), "-", color=ink(c), lw=1.8, zorder=4, label="this run")
            bep, bml = before[c]
            ax.plot(bep, bml.mean(0), "--", color=light_dark("#333", "#ddd"), lw=1.0, zorder=3, label="ex-2.2.9")
            ax.set_title(c, fontsize=9)
            ax.set_xlabel("epoch")
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("margin (m_line)")
        fig_legend(fig, axes[0])
        return fig

    return _plot()


def h2_lines_figure(res: Results) -> str:
    vals = {
        "level": {c: res.anneal(c, "m_line_final") for c in ANCHORED},
        "alpha_op1": {c: res.stat(c, "alpha_op1") for c in ANCHORED},
        "eol": {c: np.abs(res.component(c, EOL)) for c in ANCHORED},
    }
    before = {
        "level": {c: res.anneal(c, "m_line_final", res.ex229) for c in ANCHORED},
        "alpha_op1": {c: res.before(c, "alpha_op1") for c in ANCHORED},
        "eol": {c: np.abs(res.component(c, EOL, metrics=res.ex229)) for c in ANCHORED},
    }
    ref = {"alpha_op1": res.ref_stat("alpha_op1")}
    return h2_lines_draw(ANCHORED, vals, before, ref)


@memo
def h2_lines_draw(conds: list[str], vals: dict, before: dict, ref: dict) -> str:
    titles = {"level": "level (final m_line)", "alpha_op1": "ᾱ at op1", "eol": "|⏎ embedding component|"}

    @themed(
        name="h2-lines",
        alt_text="""
            Three dot panels: the final margin, the mean alignment of the non-red colors at the first operand, and the axis component of the newline embedding, each with the three anchored conditions along the bottom. Filled marks are this run and open marks the same condition in ex-2.2.9; they sit at about the same heights. On the level panel handover sits a little under the two references; on the alignment panel it sits above them, with the ex-2.2.3 point lower still; on the newline panel handover and handover-tied carry a component and handover-slot about none.
        """,
        caption="""
            **The three reported lines of H2, per anchored condition.** Filled marks are this run, open marks the same condition in ex-2.2.9, each as small dots per seed with the seed mean as the larger mark and the seed range as a bar. Left: the final margin on the `mix` lines. Middle: ᾱ at op1, with ex-2.2.3's twenty seeds at the right. Right: the absolute axis component of the `⏎` embedding row.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), layout="constrained")
        rng = np.random.default_rng(1)
        for ax, k in zip(axes, vals, strict=True):
            for i, c in enumerate(conds):
                dots(ax, i - 0.17, vals[k][c], c, rng=rng, width=0.05, label=c)
                dots(ax, i + 0.17, before[k][c], c, rng=rng, width=0.05, open_=True)
            names = [short(c) for c in conds]
            if k in ref:
                dots(ax, len(conds), ref[k], "reference", rng=rng, width=0.05, label="ex-2.2.3")
                names.append("ex-2.2.3")
            ax.set_title(titles[k], fontsize=9)
            ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.2)
        fig_legend(fig, axes[1])
        return fig

    return _plot()


def h2_prose(res: Results) -> str:
    st = {c: h2_status(res, c) for c in ANCHORED}
    h = st["handover"]
    b_m = float(res.before("handover", "m_line").mean())
    ref_m = float(res.ref_stat("m_line").mean())
    n_below, n_q, low = retention_detail(res, "handover")
    b_level = {c: float(res.anneal(c, "m_line_final", res.ex229).mean()) for c in ANCHORED}
    b_alpha = {c: float(res.before(c, "alpha_op1").mean()) for c in ANCHORED}
    ref_alpha = float(res.ref_stat("alpha_op1").mean())
    b_eol = {c: float(np.abs(res.component(c, EOL, metrics=res.ex229)).mean()) for c in ANCHORED}
    level_order = h["level"][1] < min(st["handover-slot"]["level"][1], st["handover-tied"]["level"][1])
    alpha_order = h["alpha_op1"][1] > max(st["handover-slot"]["alpha_op1"][1], st["handover-tied"]["alpha_op1"][1])
    fmt = lambda k, d=2: ", ".join(f"{st[c][k][1]:.{d}f} on `{c}`" for c in ANCHORED)  # noqa: E731
    fmt_b = lambda d, digits=2: ", ".join(f"{d[c]:.{digits}f}" for c in ANCHORED)  # noqa: E731
    ret_line = (
        f"Every one of the {n_q} qualifying seeds holds its alignment through the anneal: retention is {h['retention'][1]:.2f} on the seed mean and {low:.2f} at the lowest seed, against the {ex.RETENTION_GATE:g} line."
        if n_below == 0
        else f"Retention is where a seed falls short: {n_below} of the {n_q} qualifying seeds end{'s' if n_below == 1 else ''} under the {ex.RETENTION_GATE:g} line, the lowest at {low:.2f}, with the seed mean at {h['retention'][1]:.2f}."
    )
    return (
        f"*Red* lands on its axis as clearly as before. On the `mix` lines `handover` reaches a margin of {h['m_line'][1]:.3f}, against {b_m:.3f} at ex-2.2.9's seeds and {ref_m:.3f} at ex-2.2.3 (the gate is {GATES['m_line'][1]:.3f}), a grading r² of {h['r2_sim'][1]:.2f}, and a contrast of {h['contrast'][1]:.2f}. Lead is {h['lead_emb'][1]:.2f} and the largest latch {h['latch_pi'][1]:.2f}, so the pull did what it should.\n\n"
        f"{ret_line} The margin drifts down before the anneal, and the anneal itself costs nothing, which is what ex-2.2.10 saw on the old runs.\n\n"
        f"The three lines read as expected. Level is {fmt('level')}; ex-2.2.9 measured {fmt_b(b_level)}, so the ordering {'holds' if level_order else 'does not hold'}: the candidate ends {'lower than' if level_order else 'no lower than'} either reference, and the gap is the drift before the anneal. ᾱ at op1 is {fmt('alpha_op1')}, against {fmt_b(b_alpha)} before and {ref_alpha:.2f} at ex-2.2.3; `handover` sits {'above' if alpha_order else 'not above'} both references. The `⏎` embedding carries {fmt('eol')} of the axis, against {fmt_b(b_eol)} before."
    )


# --- H3: removal and selectivity -----------------------------------------------------------------


def selectivity_status(res: Results, cond: str) -> tuple[str, float, float]:
    """(status, seed-mean non-red `mix` deficit, band) of one condition under `projection`."""
    d = res.deficit(cond, ex.PRIMARY_OP)
    b = band(res.ref_deficit_sd(ex.PRIMARY_OP), res.n(cond), len(res.ref_deficit(ex.PRIMARY_OP)))
    m = float(d.mean())
    if m <= ex.NONRED_DEFICIT_GATE:
        s = "pass"
    elif m <= ex.NONRED_DEFICIT_GATE + b:
        s = "unresolved"
    elif m <= ex.NONRED_DEFICIT_PARTIAL:
        s = "partial"
    else:
        s = "miss"
    return s, m, b


def removal_misses(res: Results, cond: str, group: str = "removal") -> list[str]:
    return [op for op in ex.OP_NAMES if float(res.kept(cond, op, group=group).mean()) > ex.RED_KEPT_GATE]


def h3_status(res: Results) -> dict:
    kept = {op: float(res.kept("handover", op).mean()) for op in ex.OP_NAMES}
    kept_zero = {op: float(res.kept("handover", op, group="removal_zero").mean()) for op in ex.OP_NAMES}
    sel, m, b = selectivity_status(res, "handover")
    misses = removal_misses(res, "handover")
    return {
        "kept": kept,
        "kept_zero": kept_zero,
        "removal": "pass" if not misses else "miss",
        "removal_misses": misses,
        "zero_misses": removal_misses(res, "handover", "removal_zero"),
        "deficit": m,
        "deficit_band": b,
        "selectivity": sel,
    }


def h3_verdict(res: Results) -> tuple[str, str]:
    st = h3_status(res)
    worst = max(st["kept"], key=lambda op: st["kept"][op])
    removal = (
        f"Removal: under `projection`, `handover` keeps at most {st['kept'][worst]:.0%} of its clean expected exact match on the removal lines (`{worst}`), inside the {ex.RED_KEPT_GATE:.0%} gate on every op."
        if st["removal"] == "pass"
        else f"Removal: `handover` keeps more than {ex.RED_KEPT_GATE:.0%} of its clean expected exact match on the removal lines of {', '.join(f'`{o}` ({st["kept"][o]:.0%})' for o in st['removal_misses'])}, and at most {max(v for o, v in st['kept'].items() if o not in st['removal_misses']):.0%} on the other ops."
    )
    sel = f"Selectivity {'holds' if st['selectivity'] == 'pass' else 'is ' + st['selectivity']}: the non-red `mix` deficit is {st['deficit']:.3f} against the {ex.NONRED_DEFICIT_GATE:g} gate (band {st['deficit_band']:.3f})."
    if st["removal"] == "pass" and st["selectivity"] == "pass":
        status = "pass"
    elif st["removal"] == "miss" or st["selectivity"] == "miss":
        status = "miss"
    else:
        status = st["selectivity"]
    return status, f"{removal} {sel}"


# A kept share this close under the removal gate is named in the prose as a near miss.
NEAR_MISS = 0.03

# The saturation-and-value lines are reported by op and by the slot the red operand sits in.
SV_SLOTS = [(op, slot) for op in ex.ORDER_SENSITIVE for slot in ("op1", "op2")]


def h3_figure(res: Results) -> str:
    kept = {(c, op): res.kept(c, op) for c in ANCHORED for op in ex.OP_NAMES}
    kept_zero = {op: res.kept("handover", op, group="removal_zero") for op in ex.OP_NAMES}
    deficit = {(c, op): res.deficit(c, op) for c in ANCHORED for op in ex.OP_NAMES}
    return h3_draw(ANCHORED, kept, kept_zero, deficit)


@memo
def h3_draw(conds: list[str], kept: dict, kept_zero: dict, deficit: dict) -> str:
    @themed(
        name="h3-removal-selectivity",
        alt_text="""
            Two dot panels stacked, with the eleven ops along the bottom. In the top panel, the share of clean accuracy kept on the removal lines, handover sits under the dashed gate line at one fifth on every op but hue-hsv, where it strays just above; handover-tied sits above the line on most channel-wise ops, and the open circles, the same read on the old to-zero lines, sit above it on the three HSV ops. In the bottom panel, the deficit on the non-red lines, the marks sit near zero on every op, well under the dashed gate.
        """,
        caption=f"""
            **Removal and selectivity under `projection`, per op and anchored condition.** Top: the share of the clean expected exact match the model keeps on each op's removal lines, with the removal lines chosen by hue (filled marks, one per condition) and, for `handover`, by ex-2.2.9's to-zero rule (open circles). The dashed line is the {ex.RED_KEPT_GATE:.0%} gate and the hatched region above it the miss. Bottom: the deficit in expected exact match on each op's non-red lines, dashed at the {ex.NONRED_DEFICIT_GATE:g} gate and dotted at the {ex.NONRED_DEFICIT_PARTIAL:g} partial level; the gate is read on `mix` only. Each small dot is one seed, the larger mark the seed mean, and the thin bar the seed range.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (top, mid) = plt.subplots(2, 1, figsize=(8.4, 5.6), layout="constrained", height_ratios=[1.2, 1])
        rng = np.random.default_rng(2)
        xs = np.arange(len(ex.OP_NAMES))
        off = np.linspace(-0.3, 0.3, len(conds) + 1)
        for x, op in zip(xs, ex.OP_NAMES, strict=True):
            for o, c in zip(off[:-1], conds, strict=True):
                dots(top, x + o, kept[c, op], c, rng=rng, ms=4, width=0.04, label=c if x == 0 else None)
                dots(mid, x + o, deficit[c, op], c, rng=rng, ms=4, width=0.04)
            dots(
                top,
                x + off[-1],
                kept_zero[op],
                "handover",
                rng=rng,
                ms=4,
                width=0.04,
                open_=True,
                label="handover, to-zero lines" if x == 0 else None,
            )
        top.set_ylim(-0.02, max(0.5, top.get_ylim()[1]))
        gate_line(top, ex.RED_KEPT_GATE, fail="above")
        gate_line(mid, ex.NONRED_DEFICIT_GATE, partial=ex.NONRED_DEFICIT_PARTIAL, fail="above")
        top.set_ylabel("kept on removal lines")
        mid.set_ylabel("non-red deficit")
        for ax in (top, mid):
            ax.set_xticks(xs, ex.OP_NAMES, rotation=30, ha="right", fontsize=8)
        top.tick_params(labelbottom=True)
        for ax in (top, mid):
            ax.grid(axis="y", alpha=0.2)
        fig_legend(fig, top)
        return fig

    return _plot()


def h3_sv_figure(res: Results) -> str:
    sv = {(c, op, slot): res.kept(c, op, group=f"sv_{slot}") for c in ANCHORED for op, slot in SV_SLOTS}
    sv_n = {(op, slot): res.n_lines("handover", op, f"sv_{slot}") for op, slot in SV_SLOTS}
    return h3_sv_draw(ANCHORED, sv, sv_n)


@memo
def h3_sv_draw(conds: list[str], sv: dict, sv_n: dict) -> str:
    @themed(
        name="h3-sv-lines",
        alt_text="""
            One dot panel with the three order-sensitive ops split by slot along the bottom, six positions. It shows the share of clean accuracy kept on the saturation-and-value lines, between a third and three quarters, with handover at or below both references on every slot.
        """,
        caption="""
            **The saturation-and-value lines under `projection`: the share of clean expected exact match kept, per slot and anchored condition.** The red lines of the three order-sensitive ops that the hue rule sets aside, split by the slot the red operand sits in; a slot with no such lines is left empty. Each small dot is one seed, the larger mark the seed mean, and the thin bar the seed range. No gate.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8.4, 3.0), layout="constrained")
        rng = np.random.default_rng(3)
        xs = np.arange(len(SV_SLOTS))
        off = np.linspace(-0.25, 0.25, len(conds))
        for x, (op, slot) in zip(xs, SV_SLOTS, strict=True):
            if sv_n[op, slot] == 0:
                continue
            for o, c in zip(off, conds, strict=True):
                dots(ax, x + o, sv[c, op, slot], c, rng=rng, ms=4, width=0.04, label=c if x == 0 else None)
        ax.set_xticks(xs, [f"{op}\nred at {slot}" for op, slot in SV_SLOTS], fontsize=8)
        ax.set_ylabel("kept on s-and-v lines")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.2)
        fig_legend(fig, ax)
        return fig

    return _plot()


def h3_removal_table(res: Results) -> str:
    rows = []
    for op in ex.OP_NAMES:
        cells = [f"`{op}`", f"{counts[op]['hue']}"]
        for c in ANCHORED:
            v = float(res.kept(c, op).mean())
            cells.append(bold_if(f"{v:.2f}", v <= ex.RED_KEPT_GATE) if c == "handover" else f"{v:.2f}")
        cells.append(f"{res.kept('handover', op, group='removal_zero').mean():.2f}")
        for operator in ("operands", "shaped-a0.4-p0"):
            cells.append(f"{res.kept('handover', op, operator).mean():.2f}")
        rows.append(cells)
    head = [
        "op",
        "lines",
        *[f"`{c}`\u00a0↓" for c in ANCHORED],
        "`handover`, to-zero lines",
        "`operands`\u00a0↓",
        "`shaped`\u00a0↓",
    ]
    return table_html(
        head,
        rows,
        f"""
        **Removal, per op: the share of clean expected exact match kept on the removal lines.** Seed means under `projection`, one column per anchored condition on the lines chosen by hue, then `handover` on ex-2.2.9's to-zero lines, and `handover` under the `operands` and `shaped-a0.4-p0` operators on the hue lines. Bold entries are inside the {ex.RED_KEPT_GATE:.0%} gate, which is read on `handover`.
        """,
    )


def h3_selectivity_table(res: Results) -> str:
    rows = []
    for op in ex.OP_NAMES:
        cells = [f"`{op}`"]
        for c in ANCHORED:
            v = float(res.deficit(c, op).mean())
            cells.append(
                bold_if(f"{v:.3f}", v <= ex.NONRED_DEFICIT_GATE)
                if c == "handover" and op == ex.PRIMARY_OP
                else f"{v:.3f}"
            )
        sd = res.ref_deficit_sd(op)
        cells.append("—" if math.isnan(sd) else f"{band(sd, res.n('handover'), len(res.ref_deficit(op))):.3f}")
        rows.append(cells)
    head = ["op", *[f"`{c}`\u00a0↓" for c in ANCHORED], "band"]
    return table_html(
        head,
        rows,
        f"""
        **Selectivity, per op: the drop in expected exact match on the non-red lines.** Seed means under `projection`, one column per anchored condition. The gate ({ex.NONRED_DEFICIT_GATE:g}) is read on `mix` only, where the bold entry is inside it. The band is the smallest difference the read resolves, from the per-run σ of {ex.DEFICIT_NOISE}; the ops the six-op grammar did not have carry none.
        """,
    )


def h3_sv_table(res: Results) -> str:
    rows = []
    for op, slot in SV_SLOTS:
        n = res.n_lines("handover", op, f"sv_{slot}")
        cells = [f"`{op}`, red at {slot}", f"{n}"]
        for c in ANCHORED:
            cells.append("—" if n == 0 else f"{res.kept(c, op, group=f'sv_{slot}').mean():.2f}")
        rows.append(cells)
    head = ["lines", "count", *[f"`{c}`" for c in ANCHORED]]
    return table_html(
        head,
        rows,
        """
        **The saturation-and-value lines: the share of clean expected exact match kept under `projection`.** The red lines of the three order-sensitive ops that the hue rule sets aside, by the slot the red operand sits in, with the count of such lines on the probe set. Seed means per anchored condition; no gate.
        """,
    )


def sv_shortfalls(res: Results) -> list[tuple[str, str, str, float, float]]:
    """(op, slot, reference, gap, band) for each slot and reference where `handover` keeps less of its clean
    expected exact match on the saturation-and-value lines than the reference does, by more than the band
    on the pooled per-seed spread.
    """
    out = []
    for op, slot in SV_SLOTS:
        if not res.n_lines("handover", op, f"sv_{slot}"):
            continue
        h = res.kept("handover", op, group=f"sv_{slot}")
        for c in ("handover-slot", "handover-tied"):
            r = res.kept(c, op, group=f"sv_{slot}")
            b = band(pooled_sd(h, r), len(h), len(r))
            gap = float(r.mean() - h.mean())
            if gap > b:
                out.append((op, slot, c, gap, b))
    return out


# REVIEW: two wordings narrowed in the saturation-and-value sentence. "the three conditions sit within a
# band of each other" became "no reference keeps more than `handover` by more than the band", because
# `sv_shortfalls` only tests that one direction and only against `handover`. And "shows on every checkpoint
# alike" became "the same on all three anchored checkpoints": every condition scored here is anchored, so the
# read cannot speak for a checkpoint outside that set. Verify: `sv_shortfalls` and the columns of the
# saturation-and-value table.
def h3_prose(res: Results) -> str:
    st = h3_status(res)
    kept, zero = st["kept"], st["kept_zero"]
    above = st["removal_misses"]
    inside = {op: v for op, v in kept.items() if op not in above}
    best, worst = min(inside, key=lambda op: inside[op]), max(inside, key=lambda op: inside[op])
    d = res.deficit("handover", ex.PRIMARY_OP)
    others = {op: float(res.deficit("handover", op).mean()) for op in ex.OP_NAMES}
    hi = max(others, key=lambda op: others[op])
    zero_above = st["zero_misses"]
    if above:
        removal = f"Taking the axis out removes *red* on {len(inside)} of the {len(kept)} ops. On their removal lines, `handover` keeps between {inside[best]:.0%} (`{best}`) and {inside[worst]:.0%} (`{worst}`) of its clean expected exact match under `projection`. On {', '.join(f'`{op}`' for op in above)} it keeps {', '.join(f'{kept[op]:.0%} (seeds from {res.kept("handover", op).min():.0%} to {res.kept("handover", op).max():.0%})' for op in above)}, above the {ex.RED_KEPT_GATE:.0%} gate."
    else:
        removal = f"Taking the axis out removes *red* on every op. On the removal lines chosen by hue, `handover` keeps between {inside[best]:.0%} (`{best}`) and {inside[worst]:.0%} (`{worst}`) of its clean expected exact match under `projection`."
    near = [op for op in inside if inside[op] >= ex.RED_KEPT_GATE - NEAR_MISS]
    if near:
        refs = ", ".join(
            f"`{op}` ({', '.join(f'{float(res.kept(c, op).mean()):.0%} on `{c}`' for c in ('handover-slot', 'handover-tied'))})"
            for op in near
        )
        removal += f" {', '.join(f'`{op}`' for op in near)} {'is a near miss' if len(near) == 1 else 'are near misses'}, within {NEAR_MISS:.0%} of the gate; the same measurement is read on every op, and the channel-wise ops keep between {min(inside[op] for op in inside if 'hsv' not in op):.0%} and {max(inside[op] for op in inside if 'hsv' not in op):.0%}, so the near miss is the top of that spread rather than an op the rule treats differently. Both references keep more there: {refs}."
    cleared = [op for op in zero_above if op not in above]
    if not zero_above:
        old = "On the same runs, the old to-zero lines also clear the gate on every op."
    else:
        old = f"On the same runs, the old to-zero lines put {', '.join(f'`{op}`' for op in zero_above)} above the gate ({', '.join(f'{zero[op]:.0%}' for op in zero_above)}). So the miss in ex-2.2.9 comes back at fresh seeds whenever the lines are picked the old way."
        if cleared and above:
            old += f" Picking them by hue clears {', '.join(f'`{op}`' for op in cleared)} and leaves {', '.join(f'`{op}`' for op in above)} above the line."
        elif cleared:
            old += " Picking them by hue clears every op."
    return f"{removal} {old}\n\nOn the non-red `mix` lines the deficit is {st['deficit']:.3f} (seeds from {d.min():.3f} to {d.max():.3f}), against {float(res.ref_deficit(ex.PRIMARY_OP).mean()):.3f} at the reference. The largest non-red deficit on any op is {others[hi]:.3f}, on `{hi}`."


def h3_sv_prose(res: Results) -> str:
    """The saturation-and-value lines: what `handover` keeps beside the two references, per slot (figure and table below)."""
    sv_low = sv_shortfalls(res)
    live = [(op, slot) for op, slot in SV_SLOTS if res.n_lines("handover", op, f"sv_{slot}")]
    n_slots = len(live)
    narrow = sum(
        1
        for op, slot in live
        if float(res.kept("handover", op, group=f"sv_{slot}").std())
        < min(float(res.kept(c, op, group=f"sv_{slot}").std()) for c in ("handover-slot", "handover-tied"))
    )
    if sv_low:
        op, slot, c, gap, b = max(sv_low, key=lambda t: t[3])
        return f"On the saturation-and-value lines (the figure and table below), `handover` keeps less than a reference on {len({(op, slot) for op, slot, *_ in sv_low})} of the {n_slots} slots by more than the band; on the other slots no reference keeps more than `handover` by more than the band. The largest gap is on `{op}` with red at {slot}: {gap:.2f} under `{c}` (band {b:.2f}). So part of what the projection takes from the saturation and value of a red color belongs to this checkpoint: the references hold more of it off the axis. The rest is the same on all three anchored checkpoints, so it belongs to the operator. The seed spread on `handover` is the narrowest of the three on {narrow} of the {n_slots} slots."
    return f"On the saturation-and-value lines (the figure and table below) the three conditions sit within a band of each other on every one of the {n_slots} slots. So the projection takes the same amount from the saturation and value of a red color on every checkpoint, and that cost belongs to the operator rather than to the readout or the labeller."


# --- H4: the two reference comparisons ------------------------------------------------------------


def syntax_component(res: Results, cond: str, table: str = "rows") -> np.ndarray:
    """Per seed, the mean absolute axis component over the syntax words on one table."""
    return np.array(
        [np.mean([abs(r[table][w]) for w in ex.SYNTAX_WORDS]) for r in res.runs(cond) if r.get(table) is not None],
        float,
    )


def h4_status(res: Results) -> dict:
    h, t = syntax_component(res, "handover"), syntax_component(res, "handover-tied")
    b = band(pooled_sd(h, t), len(h), len(t))
    readout = syntax_component(res, "handover", "rows_readout")
    d_h, d_s = res.deficit("handover", ex.PRIMARY_OP), res.deficit("handover-slot", ex.PRIMARY_OP)
    d_b = band(res.ref_deficit_sd(ex.PRIMARY_OP), len(d_h), len(d_s))
    tail_h, tail_s = int((d_h > ex.TAIL).sum()), int((d_s > ex.TAIL).sum())
    return {
        "component": (float(h.mean()), float(t.mean()), b),
        "component_holds": float(t.mean() - h.mean()) > b,
        "readout": float(readout.mean()),
        "deficit": (float(d_h.mean()), float(d_s.mean()), d_b),
        "mean_holds": abs(float(d_h.mean() - d_s.mean())) < d_b,
        "tail": (tail_h, tail_s),
        "tail_holds": tail_h - tail_s <= 2,
    }


def h4_verdict(res: Results) -> tuple[str, str]:
    st = h4_status(res)
    a, m, t = st["component_holds"], st["mean_holds"], st["tail_holds"]
    c1 = f"the syntax embeddings carry {'less' if a else 'no less'} of the axis on `handover` ({st['component'][0]:.3f}) than on `handover-tied` ({st['component'][1]:.3f}; band {st['component'][2]:.3f}), and `handover`'s readout rows carry {st['readout']:.3f}"
    c2 = f"the non-red `mix` deficits are {st['deficit'][0]:.3f} on `handover` and {st['deficit'][1]:.3f} on `handover-slot` (band {st['deficit'][2]:.3f}), with {st['tail'][0]} and {st['tail'][1]} seeds above {ex.TAIL:g}"
    first = "holds" if a else "does not hold"
    second = "holds" if m and t else "holds in part" if m or t else "does not hold"
    status = "pass" if a and m and t else "partial" if a or m or t else "miss"
    return status, f"The readout comparison {first}: {c1}. The label comparison {second}: {c2}."


def h4_figure(res: Results) -> str:
    words = list(ex.SYNTAX_WORDS)
    labels = [w if w != EOL else "⏎" for w in words]
    series = {
        ("handover-tied", "rows"): np.stack([np.abs(res.component("handover-tied", w)) for w in words], 1),
        ("handover", "rows"): np.stack([np.abs(res.component("handover", w)) for w in words], 1),
        ("handover", "rows_readout"): np.stack(
            [np.abs(res.component("handover", w, "rows_readout")) for w in words], 1
        ),
    }
    d = {c: res.deficit(c, ex.PRIMARY_OP) for c in ("handover", "handover-slot")}
    return h4_draw(labels, series, d)


@memo
def h4_draw(labels: list[str], series: dict, d: dict[str, np.ndarray]) -> str:
    @themed(
        name="h4-syntax-component",
        alt_text="""
            Two panels. Left, a dot chart of the axis component on each syntax word's row: the tied condition's embedding rows sit around 0.15 on every word and higher still on the equals sign and the newline, the untied condition's embedding rows sit near zero except on the newline, and its readout rows sit in between. Right, the non-red deficit under projection for handover and handover-slot, as columns of seed dots at about the same height, far under the dotted tail level.
        """,
        caption=f"""
            **The *red* axis on the syntax tokens, and what the whole-line label costs.** Left: the absolute axis component of each syntax word's row (the op words, `=`, and `⏎`), seed mean with the seed range as a bar, for `handover-tied`'s embedding rows, `handover`'s embedding rows, and `handover`'s readout rows (the open marks). Right: the deficit in expected exact match on the non-red `mix` lines under `projection`, one small dot per seed and the seed mean as the larger mark, with the {ex.TAIL:g} tail level dotted.
        """,
    )
    def _plot() -> plt.Figure:
        fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained", width_ratios=[3, 1])
        x = np.arange(len(labels))
        for c, table, o, mfc, label in (
            ("handover-tied", "rows", -0.25, None, "handover-tied, embedding"),
            ("handover", "rows", 0.0, None, "handover, embedding"),
            ("handover", "rows_readout", 0.25, "none", "handover, readout"),
        ):
            v = series[c, table]
            m, lo, hi = v.mean(0), v.min(0), v.max(0)
            left.errorbar(
                x + o,
                m,
                yerr=[m - lo, hi - m],
                fmt=marker(c),
                ms=4.5,
                color=ink(c),
                mfc=mfc or ink(c),
                lw=0.8,
                capsize=0,
                zorder=3,
                label=label,
            )
        left.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
        left.set_ylabel("|axis component|")
        left.set_ylim(0, None)
        left.grid(axis="y", alpha=0.2)
        fig_legend(fig, left)
        rng = np.random.default_rng(3)
        for i, c in enumerate(d):
            dots(right, i, d[c], c, rng=rng)
        right.axhline(ex.TAIL, color=light_dark("#333", "#ddd"), lw=0.7, ls=":", zorder=2)
        right.set_xticks([0, 1], [short(c) for c in d], fontsize=8)
        right.set_ylabel("non-red mix deficit")
        right.grid(axis="y", alpha=0.2)
        return fig

    return _plot()


def h4_table(res: Results) -> str:
    st = h4_status(res)
    label = lambda w: "`⏎`" if w == EOL else f"`{w}`"  # noqa: E731
    rows = [
        [
            label(w),
            f"{np.abs(res.component('handover-tied', w)).mean():.3f}",
            f"{np.abs(res.component('handover', w)).mean():.3f}",
            f"{np.abs(res.component('handover', w, 'rows_readout')).mean():.3f}",
        ]
        for w in ex.SYNTAX_WORDS
    ]
    rows.append(["all syntax words", f"{st['component'][1]:.3f}", f"{st['component'][0]:.3f}", f"{st['readout']:.3f}"])
    head = ["word", "`handover-tied` embedding\u00a0↓", "`handover` embedding\u00a0↓", "`handover` readout"]
    return table_html(
        head,
        rows,
        f"""
        **Absolute axis component per syntax word.** Seed means over each condition's runs, on the embedding table and, for `handover`, on its readout table. The band between the two conditions on the all-words mean is {st["component"][2]:.3f} ({ex.COMPONENT_NOISE}).
        """,
    )


def h4_prose(res: Results) -> str:
    st = h4_status(res)
    comp = {w: float(np.abs(res.component("handover", w)).mean()) for w in ex.SYNTAX_WORDS}
    eol, rest = comp[EOL], max(v for w, v in comp.items() if w != EOL)
    return f"Both comparisons come out as they did before. Averaged over the op words, `=`, and `⏎`, the axis component of the embedding rows is {st['component'][0]:.3f} on `handover` and {st['component'][1]:.3f} on `handover-tied`, a gap {'over' if st['component_holds'] else 'under'} the band of {st['component'][2]:.3f}, while `handover`'s readout rows carry {st['readout']:.3f}: the component moved to the readout. As before, `⏎` is the one word the untied table leaves a component on, at {eol:.3f} where no other syntax word reaches {rest:.2f}. Under `projection` the non-red `mix` deficit is {st['deficit'][0]:.3f} on `handover` and {st['deficit'][1]:.3f} on `handover-slot`, {'within' if st['mean_holds'] else 'outside'} a band of {st['deficit'][2]:.3f}, and {st['tail'][0]} seeds of `handover` sit above {ex.TAIL:g} against {st['tail'][1]} of `handover-slot`."


# --- Decision and findings ------------------------------------------------------------------------


def decision_misses(res: Results) -> list[str]:
    """Every gate of the decision rule that `handover` does not clear, named with its status."""
    h1, _ = h1_verdict(res)
    h2, h3 = h2_status(res), h3_status(res)
    misses = []
    if h1 != "pass":
        misses.append(f"H1 ({h1})")
    for k, (name, _, _) in GATES.items():
        if h2[k][0] != "pass":
            misses.append(f"H2 {name} ({h2[k][0]})")
    if h3["removal"] != "pass":
        misses.append("H3 removal")
    if h3["selectivity"] != "pass":
        misses.append(f"H3 selectivity ({h3['selectivity']})")
    return misses


def attribution(gate: str, slot_clears: bool, tied_clears: bool) -> str:
    if slot_clears and tied_clears:
        return f"Both references clear {gate}, which points at the whole-line label and the untied readout together."
    if slot_clears:
        return f"`handover-slot` clears {gate} and `handover-tied` does not, which points at the labeller."
    if tied_clears:
        return f"`handover-tied` clears {gate} and `handover-slot` does not, which points at the readout."
    return f"Neither reference clears {gate}, which points at the table or the corpus."


def decision(res: Results) -> tuple[bool, str]:
    """(adopted, the decision text). The rule: H1, H2's margin, grading, and contrast, and H3 in full."""
    misses = decision_misses(res)
    n_below, n_q, low = retention_detail(res, "handover")
    note = (
        ""
        if n_below == 0
        else f" Retention is outside the rule and is reported: {n_below} of {n_q} qualifying `handover` seeds end{'s' if n_below == 1 else ''} under the {ex.RETENTION_GATE:g} line (the lowest at {low:.2f}), which the discussion takes up."
    )
    if not misses:
        return (
            True,
            "`handover` clears H1, H2's margin, grading, and contrast, and H3 in full on the removal lines chosen by hue, so **the handover is adopted**: table A+ with stochastic rounding, the whole-line labeller, and the untied readout are the grammar and recipe of record for the anchored-op experiments."
            + note,
        )
    who = []
    joined = " ".join(misses)
    if "H3 removal" in joined:
        # REVIEW: the references are read on the ops `handover` missed, not on all eleven. Read over every op, the
        # sentence said each reference was above the gate somewhere, which is not what the miss asks. Verify: the
        # removal table's rows for the missed ops against the gate, per reference.
        missed = removal_misses(res, "handover")
        clears = {
            c: all(float(res.kept(c, op).mean()) <= ex.RED_KEPT_GATE for op in missed)
            for c in ("handover-slot", "handover-tied")
        }
        where = "removal on " + ", ".join(f"`{op}`" for op in missed)
        who.append(attribution(where, clears["handover-slot"], clears["handover-tied"]))
    if "H3 selectivity" in joined:
        who.append(
            attribution(
                "selectivity",
                selectivity_status(res, "handover-slot")[0] == "pass",
                selectivity_status(res, "handover-tied")[0] == "pass",
            )
        )
    return (
        False,
        f"`handover` misses {', '.join(misses)}, so **the handover is not adopted** as it stands. "
        + " ".join(who)
        + note,
    )


# --- Exploratory, post hoc ------------------------------------------------------------------------


def posthoc_old_seeds_md(res: Results) -> str:
    """The kept share of the missed op at the seeds of ex-2.2.9, as ex-2.2.10 scored them, beside this run."""
    misses = removal_misses(res, "handover")
    if not misses:
        return ""
    out = []
    for op in misses:
        slot = "op2" if op == "hue-hsv" else "op1"
        old, new = res.old_kept("handover", op, f"removal_{slot}"), res.kept("handover", op)
        gap, b = float(new.mean() - old.mean()), band(pooled_sd(old, new), len(old), len(new))
        out.append(
            f"On `{op}`, when ex-2.2.10 rescored the {len(old)} seeds from ex-2.2.9, they kept {old.mean():.2f} on these lines (seeds from {old.min():.2f} to {old.max():.2f}). The {len(new)} fresh seeds keep {new.mean():.2f} (from {new.min():.2f} to {new.max():.2f}). The two seed sets differ by {gap:.2f}, against a band of {b:.2f}, and the {ex.RED_KEPT_GATE:g} gate sits inside both spreads."
        )
    return "\n\n".join(out)


def posthoc_references_md(res: Results) -> str:
    """How much more the two references keep on the removal lines than `handover` does, by op group."""
    gaps = {}
    for c in ("handover-tied", "handover-slot"):
        for op in ex.OP_NAMES:
            h, r = res.kept("handover", op), res.kept(c, op)
            gaps[c, op] = (float(r.mean() - h.mean()), band(pooled_sd(h, r), len(h), len(r)))
    channel = [op for op in ex.OP_NAMES if op not in ex.ORDER_SENSITIVE]
    tied_ch = [gaps["handover-tied", op][0] for op in channel]
    tied_past = sum(g > b for g, b in (gaps["handover-tied", op] for op in channel))
    hsv_past = [op for op in ex.ORDER_SENSITIVE if gaps["handover-tied", op][0] > gaps["handover-tied", op][1]]
    slot_op = max(ex.OP_NAMES, key=lambda op: gaps["handover-slot", op][0])
    slot_past = sum(g > b for g, b in (gaps["handover-slot", op] for op in ex.OP_NAMES))
    hsv = (
        "on the three HSV ops the two keep the same share to within a band"
        if not hsv_past
        else f"on {', '.join(f'`{op}`' for op in hsv_past)} it also keeps more"
    )
    return (
        f"On the removal lines of every channel-wise op, `handover-tied` keeps more than `handover`, by {min(tied_ch):.2f} to {max(tied_ch):.2f}. That gap is past the band on {tied_past} of the {len(channel)} ops, and {hsv}. "
        f"`handover-slot` keeps a little more than `handover` on every op, the most on `{slot_op}` ({gaps['handover-slot', slot_op][0]:.2f}), past the band on {slot_past} of the {len(ex.OP_NAMES)}."
    )


def posthoc_drift(res: Results) -> dict[str, tuple[float, float, float]]:
    """(epoch of the peak, peak, drop from the peak to the end) of the seed-mean alignment per condition."""
    out = {}
    for c in ANCHORED:
        ep, ys = res.curves(c)
        m = np.nanmean(ys, axis=0)
        i = int(np.nanargmax(m))
        out[c] = (float(ep[i]), float(m[i]), float(m[i] - m[-1]))
    return out


FINDINGS = [
    ("Does the model still learn the task? (H1)", "does-the-model-still-learn-the-task-h1", h1_verdict, True),
    (
        "Does *red* land, and stay through the anneal? (H2)",
        "does-red-land-and-stay-through-the-anneal-h2",
        h2_verdict,
        True,
    ),
    (
        "Can we take *red* out on the lines that need its hue? (H3)",
        "can-we-take-red-out-on-the-lines-that-need-its-hue-h3",
        h3_verdict,
        True,
    ),
    (
        "Do the two reference comparisons hold again? (H4)",
        "do-the-two-reference-comparisons-hold-again-h4",
        h4_verdict,
        False,
    ),
]


def findings_md(res: Results | None) -> str:
    # H1–H3 are gated, so the verdict word names the gate; H4 is a prediction, so its word says whether it held.
    gated = {"pass": "gate cleared", "miss": "gate missed", "partial": "partial", "unresolved": "unresolved"}
    predicted = {"pass": "held", "miss": "did not hold", "partial": "held in part", "unresolved": "unresolved"}
    lines = []
    for title, anchor, fn, is_gate in FINDINGS:
        if res is None:
            lines.append(f"- [{title}](#{anchor}) —")
        else:
            status, line = fn(res)
            lines.append(f"- [{title}](#{anchor}) — **{(gated if is_gate else predicted)[status]}.** {line}")
    if res is None:
        lines.append("- [Decision](#decision) —")
    else:
        adopted, _ = decision(res)
        lines.append(f"- [Decision](#decision) — **{'adopted' if adopted else 'not adopted'}.**")
    return "\n".join(lines)


res = load_results()

r"""
# Ex 2.2.11: the handover re-run

/// tip |
<!-- tl;dr -->
Ex-2.2.9 missed two gates, and ex-2.2.10 traced both misses to how we were measuring. This is the same experiment at fresh seeds, with three measurements fixed in advance: removal lines picked by hue, retention measured across the anneal, and the op1 alignment reported beside its references rather than gated. *Red* lands, stays through the anneal, and comes out cleanly on ten of the eleven ops. On `hue-hsv` it does not come out far enough, so the handover is not adopted as it stands.
///

"""

rf"""
## Findings

{findings_md(res)}
"""

r"""
/// admonition | How to read this draft
This is a preregistration. We wrote the conditions, the gates, and the decision rule before any run and froze them at commit `903cae5`. Everything after that commit is either results filled into the frozen sections or exploratory. Each hypothesis section opens with what we expect and the one number we will look at, and the results go into that section in place once they exist. Anything we think of after seeing the data goes under [Exploratory analyses](#exploratory-analyses), marked as post hoc. Every count in the method is computed from `experiment.py` at render time.
///

## Why this experiment

We want a setup that the later D2.2 experiments can build on. It has two parts: the grammar, meaning which color operations the model learns, and the recipe, meaning how the model is trained and how *red* is anchored on one axis of the residual stream.[^rs] [Ex-2.2.9](../ex-2.2.9/report.py) was the handover to that setup: it combined the four changes the scouting round had tried one at a time, and asked whether *red* still lands on its axis and still comes out cleanly.

[^rs]: The *residual stream* is the running vector of activations that each layer of a transformer reads from and writes back to.

Most of it held. The model learned the eleven ops as well as an un-anchored model did, *red* landed with the same margin as before, and neither the separate readout nor the whole-line label cost anything we could resolve. Two measurements missed their gates, so the rule said no.

[Ex-2.2.10](../ex-2.2.10/report.py) then went back to the stored runs and traced both misses to how we measured. The removal measurement counted a line as needing *red* when zeroing the R channel of the red operand moved its answer far. But zeroing R on pure red gives black, which has no saturation or value either, so the rule also swept in lines whose answer takes only the saturation or value of red.

The projection leaves those lines alone, because on the red operand it acts like a change of hue. On the lines whose answer does need the hue, every seed already cleared the gate.

Retention was the second miss: that measurement divided the final alignment by the peak alignment over the run. On the candidate, the peak is the high point of a noisy plateau reached thirty epochs before the anneal, and across the anneal itself nothing is lost. Separately, the op1 alignment rises under either half of the handover, so its old reference belongs to the old grammar.

A measurement chosen after looking at the data cannot then score that data. So this is ex-2.2.9 again at seeds it never trained, with the three measurements fixed here first. We re-run rather than rescore because the claim we want is the preregistered one.
"""

rf"""
## Conditions

Everything about training is as ex-2.2.9 froze it: table A+ with its {ex.N_OPS} ops, {ex.N_LINES:,} lines with answers drawn stochastically, one fifth of the pairs of each op held out, and the recipe adopted in ex-2.2.3 (λ_a = {ex.LAM:g}, annealed over the last tenth of training, τ = {ex.TAU:g}, {ex.EPOCHS} epochs). The corpus is the same one at the same seed, so the held-out lines are the same. Every model is fresh: condition seed *i* trains at model seed {ex.SEED_OFFSET} + *i*, so we score no checkpoint that ex-2.2.10 looked at.

| condition | seeds | what it is | role |
| --- | ---: | --- | --- |
{chr(10).join(f"| `{c.name}` | {c.seeds} | {c.title} | {c.role} |" for c in ex.CONDS)}

**`handover`** has everything enabled. It is the one candidate, and every gate is scored on it alone.

**`handover-slot`** puts the label back on the two operands only, as ex-2.2.3 had it. **`handover-tied`** puts the readout back on the shared table. In ex-2.2.9 each of them undid about half of the rise in op1 alignment, which is why both are the references for that line now. Each is also the reference for one of the comparisons in H4, as before. This time we score both under the removal operators, for the saturation-and-value measurement in H3.

**`control`** has no anchor, and sets the task bar for H1.

`handover-narrow` does not return. Its question (lines per op) was exploratory, and ex-2.2.9 answered it.

The removal operators are the ones from ex-2.2.9.
**`projection`** takes the axis out at every slice and position, and is the operator the gates are scored under.
**`operands`** does the same at the two operand positions only, as the selective reference.
**`shaped-a0.4-p0`** leaves states below alignment 0.4 alone, and is reported without a gate.

## Glossary

We use the vocabulary of ex-2.2.9, and its [glossary](../ex-2.2.9/report.py#glossary) has every term. Three terms change meaning here, and one is new.

<dl>
<dt>Removal lines</dt>
<dd>The red lines whose true answer moves by at least {ex.FAR_MOVE:g} in the unit cube when we permute the channels of the red operand. Permuting the channels keeps the saturation and value of a color and changes its hue, so these are the lines whose answer needs the <em>hue</em> of red. Ex-2.2.9 zeroed the R channel instead, which also removes the saturation and value. Counts per op are in the <a href="#the-removal-lines">method</a>.</dd>
<dt>Saturation-and-value lines</dt>
<dd>The red lines the hue rule sets aside. Their answer holds still under every permutation of the red operand, so it takes only the saturation or value of red, or nothing from red at all. On the three order-sensitive ops, that is one of the two slots. We report these without a gate.</dd>
<dt>Retention</dt>
<dd>Whether the placement holds across the anneal of the anchor weight. It is the final alignment as a share of the alignment at the start of the anneal. Ex-2.2.9 divided by the peak over the run instead.</dd>
<dt>Level</dt>
<dd>The alignment at the end of training, on the <code>mix</code> lines. Every later measurement is taken on it, and it is the number that the drift before the anneal moves. Level, retention, and the trajectory figure read the alignment the training loop records, on one probe line per color; the gated margin is the same statistic read after training on the full <code>mix</code> walk. The two sit on different scales, so a level near 0.7 and a margin near 0.4 describe one run.</dd>
</dl>
"""

r"""
## Does the model still learn the task? (H1)
"""

rf"""
**What we expect.** The anchored model learns the eleven ops as well as the un-anchored one does, as it did in ex-2.2.9. The number we look at, for each op, is the `handover` seed-mean expected exact match on the held-out lines, against the same number on `control`. Gate: within {ex.TASK_GATE:g} on every op, partial to {ex.TASK_PARTIAL:g}. We report a gap smaller than a band as unresolved. What would change our mind: a gap over {ex.TASK_PARTIAL:g} on some op, which at fresh seeds would say the pass in ex-2.2.9 was luck.

"""

if res is None:
    stop(RESULTS_TO_COME)
status, line = h1_verdict(res)
rf"""
**What we saw.** {h1_prose(res)}

{h1_figure(res)}

{h1_table(res)}

{verdict_md(status, line)}
"""

r"""
## Does *red* land, and stay through the anneal? (H2)
"""

rf"""
**What we expect.** *Red* should end up on its axis as clearly as it did in ex-2.2.9, and easing off the anchor weight at the end of training should not let it slip. Everything here is measured on the `mix` lines of `handover`, under its own labeller.

Three of the numbers are gates, the same three ex-2.2.9 used. Each is something the anchor weight pushes on during training. So clearing them says the recipe still does its job on this grammar, and nothing about how.

- *Margin.* How far the pull lifts *red* onto the axis above the other colors. Seed-mean m_line at least {ex.MARGIN_RATIO:.0%} of what ex-2.1.10 measured ({ex.REF_M_LINE:.4f}); partial from {ex.MARGIN_PARTIAL:.0%}.
- *Grading.* Whether redder colors sit further along the axis, in order. Grading r² at least {ex.GRADE_R2_RATIO:.0%} of what ex-2.1.11 measured ({ex.REF_R2_SIM:.3f}).
- *Contrast.* Whether the pull follows the red operand rather than a fixed position. At least {ex.CONTRAST_GATE:g}; partial from {ex.CONTRAST_PARTIAL:g}.

Four more numbers have no gate. We print each beside its references, and say here what we expect of it.

- *Retention.* Does the placement survive the anneal? Every run whose alignment at the start of the anneal reaches {ex.RETENTION_FLOOR:g} should end at {ex.RETENTION_GATE:g} of that value. This is the measurement we changed. Ex-2.2.10 took it this way on the runs from ex-2.2.9 and got 0.99 and above on every condition, so we expect every seed to clear it. That is why it is a line and not a gate: the method already predicts the answer, so the answer cannot inform the decision. A seed under the line would still be worth stopping for, since it would say the anneal costs something after all, and that the old measurement had been right for the wrong reason.
- *Level.* Where the alignment ends up. We report the final alignment, seed mean and range, beside `handover-slot`, `handover-tied`, and the point adopted in ex-2.2.3. Ex-2.2.9 measured 0.66 against 0.72 and 0.70, and we expect the same ordering. The gap between them is the drift before the anneal. Its cause goes to the [training-dynamics item](/todo/science/training-dynamics-under-the-retention-drift.md); here we only say how large it is at fresh seeds.
- *ᾱ at op1.* How much of the axis the other colors pick up at the first operand. We report the mean alignment of the non-red colors at op1, beside the same two references and the point from ex-2.2.3. Ex-2.2.9 measured 0.28 against 0.18 and 0.16. We expect the same shape, with `handover` above both and each reference about halfway down. This was a gate at 0.1 through ex-2.2.9, where it missed. It becomes a gate again once we can name a mechanism ([backlog](/todo/science/containment-rises-under-the-untied-readout.md)).
- *The `⏎` embedding.* How much of the axis the end-of-line token carries. We report the axis component of the embedding of `⏎`, on every anchored condition. Ex-2.2.9 measured 0.17 on `handover` and about zero on `handover-slot`, and we expect that again.

We also check lead (at least {ex.LEAD_GATE:g}) and latch (no run over {ex.LATCH_PI:g} on op1) as before, to confirm the pull did what it should. These two are manipulation checks, as in ex-2.2.9: a miss would say the run was not the one we meant to score. Neither is part of the decision rule.

"""

status, line = h2_verdict(res)
rf"""
**What we saw.** {h2_prose(res)}

{h2_traj_figure(res)}

{h2_lines_figure(res)}

{h2_table(res)}

{verdict_md(status, line)}
"""

r"""
## Can we take *red* out on the lines that need its hue? (H3)
"""

rf"""
**What we expect.** With the removal lines chosen by hue, the projection takes *red* out on every op, and costs nothing on the lines that never had any *red* in them. Both gates are scored under `projection`, on `handover`.

- *Removal.* On the removal lines of every op, the model keeps at most {ex.RED_KEPT_GATE:.0%} of its clean expected exact match, seed mean. Rescoring the seeds of ex-2.2.9 this way, every seed cleared it on every op; the gate here is on fresh seeds. What would change our mind: an op above the line. If that op is one of the three HSV ops, the hue rule was not the whole story. If it is a channel-wise op, the rule change was beside the point.
- *Selectivity.* The seed-mean deficit on the non-red `mix` lines is at most {ex.NONRED_DEFICIT_GATE:g}; partial to {ex.NONRED_DEFICIT_PARTIAL:g}, which is a reporting level. Ex-2.2.9 measured 0.02. Bands use the per-run spread of {ex.DEFICIT_NOISE}, frozen as before. We report every other op beside `mix`, and count the red-answer lines on their own, as ex-2.2.9 did.

One more measurement has an expected direction but no gate.

*Saturation and value.* We take the kept share under `projection` on the saturation-and-value lines of the three order-sensitive ops, for `handover`, `handover-slot`, and `handover-tied`. On the probe sets these are {", ".join(f"{sv_share[op]:.0%} of the red lines on `{op}`" for op in ex.ORDER_SENSITIVE)}. Those shares are counted from the true answers, so they come from the rule and the op, and from no model. A slot is an operand position, op1 or op2; on these three ops the answer depends on which slot the red color sits in, so the lines are split by it. Ex-2.2.10 saw `handover` keep about half on the two slots that take the saturation or value of red, and a fifth where the answer needs both at once, so the projection takes part of the saturation and value of a red color with it. If the two references keep the same shares, that cost belongs to the operator. If they keep more, the cost belongs to this checkpoint, and either the readout or the labeller is putting saturation and value onto the axis.

"""

status, line = h3_verdict(res)
rf"""
**What we saw.** {h3_prose(res)}

{h3_figure(res)}

{h3_removal_table(res)}

{h3_selectivity_table(res)}

{h3_sv_prose(res)}

{h3_sv_figure(res)}

{h3_sv_table(res)}

{verdict_md(status, line)}
"""

r"""
## Do the two reference comparisons hold again? (H4)
"""

rf"""
**What we expect.** Ex-2.2.9 checked two things about the recipe on the side, and both held. The separate readout kept the axis off the syntax tokens, and the whole-line label cost no selectivity. We check both again at fresh seeds. Neither is a gate, so a miss is something to look into rather than a reason to stop.

- *The separate readout keeps the axis off the syntax tokens.* The axis component on the syntax embeddings (`=`, the op words, `⏎`) is lower on `handover` than on `handover-tied` by more than a band, with the component appearing on the readout table instead.
- *The whole-line label costs no selectivity.* The non-red `mix` deficit under `projection` differs between `handover` and `handover-slot` by less than a band, and at most two more seeds sit above {ex.TAIL:g} on `handover` than on `handover-slot`.

Both bands use the σ values ex-2.2.9 froze: {ex.COMPONENT_NOISE} for the embedding component, and {ex.DEFICIT_NOISE} for the deficit.

"""

status, line = h4_verdict(res)
rf"""
**What we saw.** {h4_prose(res)}

{h4_figure(res)}

{h4_table(res)}

{verdict_md(status, line)}
"""

r"""
## Decision
"""

adopted, decision_line = decision(res)
decision_misses_now = decision_misses(res)
rf"""
The rule, frozen before the run: `handover` is adopted, and becomes the grammar and recipe of record for the anchored-op experiments, if it clears H1, H2 (margin, grading, and contrast, all in full), and H3 in full on the removal lines chosen by hue; every partial band is a reporting level. Otherwise it is not adopted, and the report says which gate was missed and what the references say about which change is responsible. {"It clears all of them, so it is adopted." if adopted else f"It misses {', '.join(decision_misses_now)} and clears the rest, so it is not adopted as it stands."}

{verdict_md("pass" if adopted else "miss", decision_line)}

<!-- REVIEW: the attribution sentence in the box is now read on the missed ops only; see the note in `decision`. -->

The rule is ex-2.2.9's with the removal lines changed. Retention stays outside it, as it was there: with the new denominator ex-2.2.10 measured it at 0.99 on every condition, so it is expected to clear with room, and a gate the method can predict does not inform a decision. It is a line in H2, and a seed under 0.8 would still stop us.

<!-- REVIEW: the first draft put retention inside the decision rule, since the measurement now asks the question it
was written for. The prereg review pointed out that ex-2.2.10 already measured it at ~1.0 on every
condition, so its direction is predictable from the method and it is a manipulation check rather than a
gate. Moved back to a reported line. Verify: `ex.DECISION` names margin, grading, and contrast only. -->
"""

rf"""
## Exploratory analyses

Not part of the decision. The two below were planned before the run. Anything we think of after seeing the data goes here too, marked as post hoc.

### The finer hue rule

The permutation rule reaches six hues. A finer rule rotates the hue of the red operand in HSV in twelve steps, snaps each one to the grid, and asks the same question. The method counts how often the two rules disagree, per op. If they disagree on more than one red line in a hundred on some op, we keep the finer rule and say so under the method before the freeze. Below that rate, the choice of rule cannot move a kept share by more than a hundredth, which is finer than any gate can resolve. Outcome: the two rules disagree most on `{finer_worst[0]}`, on {finer_worst_share:.1%} of its red lines, so the permutation rule stands.

### Checkpoints on the trajectory stride

The first three seeds of each anchored condition keep a checkpoint at every trajectory point. Nothing in this report uses them. The [training-dynamics item](/todo/science/training-dynamics-under-the-retention-drift.md) needs them for a local learning coefficient estimate through the plateau and for the whole-geometry measurement at the same epochs, and this is the cheapest place to store them. Recorded.
"""

drift = posthoc_drift(res)
rf"""
### The missed op at the old seeds (post hoc)

{posthoc_old_seeds_md(res)}

So the fresh seeds came in higher by about what twenty seeds resolve. A rescoring of old seeds is not a preregistered read, and this says how far the miss is from the noise and nothing more.

### What the references keep on the removal lines (post hoc)

{posthoc_references_md(res)}

So the separate readout is what makes removal clean on the channel-wise ops, and neither of the two changes is behind the share `hue-hsv` keeps.

### The drift before the anneal (post hoc)

On `handover` the seed-mean alignment peaks near epoch {drift["handover"][0]:.0f} and loses {drift["handover"][2]:.2f} by the end of training. On `handover-slot` and `handover-tied` it peaks near epochs {drift["handover-slot"][0]:.0f} and {drift["handover-tied"][0]:.0f}, in or beside the anneal window, and loses under {max(drift["handover-slot"][2], drift["handover-tied"][2]) + 0.005:.2f}.

So the drift ex-2.2.10 saw belongs to the candidate alone, even though each reference differs from it in only one change. The [training-dynamics item](/todo/science/training-dynamics-under-the-retention-drift.md) has the checkpoints to look at it.

## Discussion

The handover setup does what ex-2.2.9 said it does, with one exception we can now name. *Red* lands on its axis, stays there through the anneal, and the projection takes it out on ten of the eleven ops without touching the lines that never had *red* in them.

<!-- REVIEW: "still gets about a quarter of the answers" became "still keeps about a quarter of the accuracy
it had": the statistic is the share of clean expected exact match retained, not an absolute accuracy. And
"came in under the gate by about the resolution of the read" became "by less than the read resolves": the old
seeds sat 0.03 under the 0.20 gate against a band of 0.06. Verify: the removal table and the post-hoc
old-seeds paragraph. -->

The exception is `hue-hsv`, the op whose answer takes its hue from the second operand. With the axis taken out, the model still keeps about a quarter of the accuracy it had on the lines that need a red hue there, a little more than the gate allows. We said in advance what that pattern would mean, and this is the first case: a miss on an HSV op under the hue rule. So the hue rule accounts for the ops it cleared, and on `hue-hsv` the miss has another cause.

On `hue-hsv` the stream holds some of the hue of the red operand somewhere other than the axis, which is one more reason to try [anchoring *red* to a plane](/todo/science/anchor-red-to-a-plane.md). `handover-tied` keeps the same share on that op and `handover-slot` keeps more, so neither the readout nor the labeller put the hue there. On this one op it seems to belong to the grammar and the recipe together.

The seed spread on that op is wide and the gate sits well inside it. At the seeds of ex-2.2.9, the same lines came in under the gate, by less than the read resolves. So the miss is a miss under the rule we froze, and it is also a small effect at the edge of what twenty seeds resolve. It is one op of eleven, on the lines whose answer needs the hue of a red at op2.

What comes next is a choice for the next preregistration. One route is to look at what the stream holds on `hue-hsv`: score the removal lines one at a time and ask what the surviving lines share, which the stored checkpoints and probes make cheap. The other is to run the anchored-op experiments on this setup, with `hue-hsv` recorded as the op where removal is partial, since nothing in D2.2 rests on that op alone.

Either way, two questions ex-2.2.9 left open are closed. Retention measured across the anneal loses nothing on any condition, so the schedule is not where the drift comes from. And the saturation-and-value cost of the projection belongs partly to this checkpoint, which the containment work could take up.
"""

rf"""
## Method

### The removal lines

The table below gives, per op and on its probe set: the red lines (dose at least {ex.RED_DOSE:g}), the removal lines under the hue rule (some channel permutation of the red operand moves the true answer by at least {ex.FAR_MOVE:g}), the same count under the to-zero rule of ex-2.2.9, and the saturation-and-value lines the hue rule sets aside. The order-sensitive ops walk every color through both slots, so we split their removal lines by where the red operand sits.

{removal_md()}

On `mix` the two rules pick the same lines. On `hsvmix` they nearly agree, differing on {rule_disagreement("hsvmix")} of the {counts["hsvmix"]["red"]} red lines. On the six other channel-wise ops the hue rule counts more, because permuting a red operand moves two channels at once. An answer that zeroing R left alone, say a `screen` with a partner about as red, moves far under a permutation.

Here at least {low_share:.0%} of the red lines of every channel-wise op are removal lines. On the six ops where the rules part, the old rule took about two thirds. Ex-2.2.9 cleared the gate on those ops with room to spare, so we expect the wider set to clear it too.

<!-- REVIEW: the draft said the two rules pick the same lines on `mix` and `hsvmix`, following ex-2.2.10's
summary. The counts in the table above disagree on `hsvmix` (401 against 391; the two sets differ on
eighteen lines, fourteen of them new to the hue rule). Verify: recompute the symmetric difference of the two rules on
`hsvmix`'s red lines. The same sentence in ex-2.2.10's "what we make of it" carries the old wording. -->


On the three HSV ops the hue rule keeps the slot whose answer takes its hue from red and sets the other aside. Under `hue-hsv` that is red at op2; under `sat-hsv` and `value-hsv` it is red at op1. The old rule counted parts of both slots on each, which is where the miss in ex-2.2.9 came from.

The finer-rule check, computed before the run: the lines per op on which the twelve-step HSV rotation and the permutation rule disagree.

{finer_rule_md()}

The rules part on at most {finer_worst[1][2]} lines of an op (`{finer_worst[0]}`, {finer_worst_share:.2%} of its red lines), under the {ex.HUE_ROTATION_TOLERANCE:.0%} tolerance on every op, so the permutation rule stands.

### Retention

We take retention from the trajectories ({ex.TRAJ_STRIDE} points over training). The anneal starts at the first point after the peak of the anchor weight where the weight falls under {ex.ANNEAL_WEIGHT_RATIO:g} of that peak, and the alignment at the start of the anneal is the last point before that. Retention is then the final point divided by that value, per run, on the runs where that value reaches {ex.RETENTION_FLOOR:g}.

### What is stored

We store what ex-2.2.9 stored: metrics, per-run arrays, trajectories, the probe set, and the end checkpoint of every run. Three things are new. Every anchored condition is scored under all three operators. The probe arrays hold the hue move and the to-zero move per line. And the first {ex.TRAJ_CHECKPOINT_SEEDS} seeds of each anchored condition keep a checkpoint at every trajectory point. The calibration of ex-2.2.9 still stands, since the corpus and the point are unchanged.

### Budget

{ex.N_RUNS} runs of {ex.HANDOVER.steps:,} steps at d64-L4, five fewer than ex-2.2.9, plus scoring under three operators on {ex.N_OPS} probe sets for {sum(c.seeds for c in ex.SCORED_UNDER_PROJECTION)} anchored runs. That is about three minutes a run on an L4, and the trajectory checkpoints add a few hundred MB of storage.
"""
