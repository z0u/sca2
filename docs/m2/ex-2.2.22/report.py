# title: Ex 2.2.22: localized by depth, various pull caps, and contexts of varying length

# The design constants come from `experiment.py` beside this script (the directory of the script is on sys.path
# while it runs). The figure in Parameters computes the posterior on the anchored op at each example count from the
# seven-op table alone; the result sections read this experiment's published eval, suppression pass, by-count pass,
# and trajectories, and ex-2.2.21's trajectories for the runs reused from it.
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import experiment as ex
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

P = ex.ex2216._POSTERIOR_MODULE

# --- Helpers -------------------------------------------------------------------------------------------------


def cell_html(text: str) -> str:
    parts = text.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def table_html(
    head: list[str],
    rows: list[list[str]],
    caption: str,
    *,
    text_cols: int = 1,
    muted: frozenset[int] = frozenset(),
) -> str:
    """An authored table in the shared report style; the first *text_cols* columns are text, the rest numeric.

    Rows whose index is in *muted* are greyed out, for context rows such as runs reused from another experiment.
    """
    ths = "".join(f"<th{' class=num' if i >= text_cols else ''}>{cell_html(h)}</th>" for i, h in enumerate(head))
    body = "".join(
        ('<tr style="opacity: 0.5">' if r in muted else "<tr>")
        + "".join(f"<td{' class=num' if i >= text_cols else ''}>{cell_html(c)}</td>" for i, c in enumerate(row))
        + "</tr>"
        for r, row in enumerate(rows)
    )
    table = f'<table class="report-table dense"><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table>'
    return figure_html(table, caption=caption, class_="report-figure")


# --- The design tables --------------------------------------------------------------------------------------


SETTING_CELLS = {
    "slices": lambda c: c.slices if c.anchored else "—",
    "weight": lambda c: f"{c.weight:.2g}" if c.anchored else "—",
    "cap": lambda c: "—" if c.cap is None else f"{c.cap:g}",
    "counts": lambda c: "–".join(map(str, (c.counts[0], c.counts[-1]))) if len(c.counts) > 1 else str(c.counts[0]),
}


def conditions_table() -> str:
    """One row per condition, with the reused runs of ex-2.2.21 greyed out. Bold marks what a condition changes."""
    head = ["condition", "reference", "seeds", "slices", "λ_a", "cap", "examples (k)"]
    first, last = ex.SEED_OFFSET, ex.SEED_OFFSET + ex.SEEDS - 1
    reused = {a.name: a.seeds for a in ex.ex2221.ARMS}
    rows = [
        [
            f"`{c.name}`",
            "ex-2.2.21",
            f"{first}–{first + reused[c.name] - 1}",
            *(fmt(c) for fmt in SETTING_CELLS.values()),
        ]
        for c in ex.REFERENCES
    ]
    for c in ex.CONDITIONS:
        ref = ex.by_name(c.reference)
        cells = [
            f"<b>{fmt(c)}</b>" if ex.settings(c)[k] != ex.settings(ref)[k] else fmt(c)
            for k, fmt in SETTING_CELLS.items()
        ]
        rows.append([f"`{c.name}`", f"`{c.reference}`", f"{first}–{last}", *cells])
    base = ex.by_name(ex.BASE)
    seeds = f"{ex.REPLICATE_SEEDS[0]}–{ex.REPLICATE_SEEDS[-1]}"
    rows.append([f"`{ex.BASE}`", "replicate", f"<b>{seeds}</b>", *(fmt(base) for fmt in SETTING_CELLS.values())])
    return table_html(
        head,
        rows,
        f"The conditions. The greyed rows are runs of ex-2.2.21, reused here. Below them are the "
        f"{len(ex.CONDITIONS)} new conditions; each changes the setting in bold from its reference, and is paired "
        f"with it by model seed. The last row repeats `{ex.BASE}` at new seeds, for H2. That makes {ex.N_RUNS} new "
        f"runs. The base is the hinge condition of ex-2.2.21 (`{ex.BASE}`): every slice pulled, "
        f"λ_a = {ex.LAMBDA_A:g}, the pull capped at {ex.HINGE_CAP:g}, and {ex.K} examples per context.",
        text_cols=7,
        muted=frozenset(range(len(ex.REFERENCES))),
    )


# --- The posterior at each example count --------------------------------------------------------------------


TABLE7 = P.build_table(ex.ex2221.ex2218.table_of(ex.OP_NAMES))


@memo
def posterior_at_count(k: int, n: int, seed: int) -> np.ndarray:
    """The posterior on the anchored op over *n* contexts of that op with *k* examples, at the corpus noise rates."""
    d = list(ex.OP_NAMES).index(ex.ANCHORED_OP)
    rng = np.random.default_rng([seed, k])
    ctx = P.sample_contexts(TABLE7, n, k, ex.RHO, ex.ex2221.CUBE_RATE, rng, true_op=d)
    return P.posterior(TABLE7, ctx, ex.RHO, ex.ex2221.CUBE_RATE)[:, d].astype(np.float32)


N_POST = 20_000
POST_SEED = 2222
POST = {k: posterior_at_count(k, N_POST, POST_SEED) for k in ex.MIXED_COUNTS}
MID_LO, MID_HI = ex.ex2221.MIDDLE_BAND


def middle_share(post: np.ndarray) -> float:
    return float(((post >= MID_LO) & (post <= MID_HI)).mean())


# The posterior given a whole context over the `k-mixed` corpus: the counts drawn uniformly, so equal parts of each.
POOLED = np.concatenate([POST[k] for k in ex.MIXED_COUNTS])
MID_FIXED = middle_share(POST[ex.K])
MID_MIXED = middle_share(POOLED)


def counts_figure() -> str:
    alt = f"""
        Cumulative distributions of the posterior on `{ex.ANCHORED_OP}` over `{ex.ANCHORED_OP}` contexts, one curve
        per example count from one to five, and a dashed curve for the counts pooled as in `k-mixed`. Each count's
        curve rises in a few steps at its own places, and with more examples more of the contexts sit near 1. The
        pooled curve has the steps of every count, so it rises in many small steps. It puts {MID_MIXED:.0%} of
        contexts in the middle band, against {MID_FIXED:.0%} at three examples alone.
    """
    return counts_draw({k: POST[k] for k in ex.MIXED_COUNTS}, POOLED, alt)


@memo
def counts_draw(post: dict[int, np.ndarray], pooled: np.ndarray, alt_text: str) -> str:
    @themed(
        name="posterior-by-count",
        alt_text=alt_text,
        caption=f"""
            **The posterior on `{ex.ANCHORED_OP}` at each example count.** The share of `{ex.ANCHORED_OP}` contexts
            whose posterior is at most the value on the x-axis, on the seven-op table at ρ = {ex.RHO:g}. The heavy
            curve is three examples, the corpus of ex-2.2.21, and the dashed curve is the counts pooled in equal
            parts, the corpus of `k-mixed`. The shaded band is the "middle band" of ex-2.2.16. {N_POST:,} sampled
            contexts per count.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.0, 2.8), layout="constrained")
        ax.axvspan(MID_LO, MID_HI, facecolor=light_dark("#000", "#fff"), alpha=0.06, lw=0)
        x = np.linspace(0, 1, 401)

        def cdf(p: np.ndarray) -> np.ndarray:
            return np.searchsorted(np.sort(p), x, side="right") / len(p)

        shades = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, len(post)))
        for (k, p), color in zip(post.items(), shades, strict=True):
            lw = 2.0 if k == ex.K else 1.0
            ax.plot(x, cdf(p), color=color, lw=lw, label=f"k = {k}")
        ax.plot(x, cdf(pooled), color=light_dark("#000", "#fff"), lw=2.0, ls="--", label="k-mixed")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"posterior on {ex.ANCHORED_OP}")
        ax.set_ylabel("share of contexts")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        return fig

    return _plot()


# --- The results: fetching ----------------------------------------------------------------------------------


def fetch(refs: Sequence[str], into: Path) -> dict[str, Path | None]:
    """Each ref's published file under *into*, or None before it exists: one `get_refs` and one `get_many`."""
    store = project_store()
    have = {r: a for r, a in store.get_refs(list(refs)).items() if a is not None}
    paths = store.get_many([(a, into / f"{i}-{Path(r).name}") for i, (r, a) in enumerate(have.items())])
    return dict.fromkeys(refs) | dict(zip(have, paths, strict=True))


def read_json(path: Path | None) -> dict:
    assert path is not None, "not published yet"
    return json.loads(path.read_text())


SUPP_KEEP = ("op_ids", "posterior", "query_pair", "p_clean", "p_full")
# The per-context arrays of the suppression pass that the confusion matrices of H2 read.

with tempfile.TemporaryDirectory() as _tmp:
    _json_refs = [ex.EVAL_REF, ex.SUPPRESSION_REF, ex.BY_COUNT_REF, ex.TRAJ_REF, ex.ex2221.TRAJ_REF]
    _files = fetch(_json_refs, Path(_tmp))
    EVAL, SUPP, BY_COUNT, TRAJ_NEW, TRAJ21 = (read_json(_files[r]) for r in _json_refs)
    _array_refs = {
        **{
            ex.SUPPRESSION_ARRAYS_REF.format(label=r["label"]): ("supp", r["label"])
            for r in SUPP["runs"]
            if r["condition"] == ex.BASE and r["replicate"]
        },
        **{ex.BY_COUNT_ARRAYS_REF.format(label=r["label"]): ("count", r["label"]) for r in BY_COUNT["runs"]},
    }
    _files = fetch(list(_array_refs), Path(_tmp))
    SUPP_ARRAYS: dict[str, dict[str, np.ndarray]] = {}
    COUNT_ARRAYS: dict[str, dict[str, np.ndarray]] = {}
    for _ref, (_kind, _lbl) in _array_refs.items():
        _f = _files[_ref]
        assert _f is not None, f"{_ref} is not published"
        with np.load(_f) as _z:
            if _kind == "supp":
                SUPP_ARRAYS[_lbl] = {k: _z[k] for k in SUPP_KEEP}
            else:
                # Only the `difference` contexts are read for E3, so the alignment arrays keep only those.
                COUNT_ARRAYS[_lbl] = {k: _z[k] for k in _z.files}

# --- The results: shared helpers ----------------------------------------------------------------------------

OPS: tuple[str, ...] = tuple(EVAL["runs"][0]["ops"])
assert OPS == tuple(ex.OP_NAMES)
D = OPS.index(ex.ANCHORED_OP)
SLOW = ex.SEED_OFFSET + 2
# Model seed 702, which took the slow path through training on most anchored conditions of ex-2.2.21.
PAIRED = tuple(range(ex.SEED_OFFSET, ex.SEED_OFFSET + ex.SEEDS))
SLICES = tuple(range(ex.N_SLICES))
CONTROL = "control"
WHOLE = "anchor-whole"
NO_EMB_UNCAPPED = "anchor-no-emb"
SLICE_GROUPS = {s: (s, f"{s}-matched") for s in ("no-emb", "no-last", "middle")}
CAP_LADDER = {ex.HINGE_CAP: ex.BASE, 0.9: "cap-0.9", 0.95: "cap-0.95", None: WHOLE}
ANCHORED = [c.name for c in (*ex.REFERENCES, *ex.CONDITIONS) if c.anchored]
ALL_CONDITIONS = [CONTROL, *ANCHORED]


def slice_name(s: int) -> str:
    """The label a table or figure gives slice *s*: "emb" for the token embedding."""
    return "emb" if s == 0 else f"slice {s}"


def short(name: str) -> str:
    """The name a figure or table gives a condition."""
    return {ex.BASE: "hinge", WHOLE: "whole-line", NO_EMB_UNCAPPED: "no-emb, no cap"}.get(name, name)


def runs(records: list[dict], cond: str, *, replicate: bool = False) -> list[dict]:
    """The runs of one condition, in model-seed order; the replicate runs of `anchor-hinge` only on request."""
    got = [r for r in records if r["condition"] == cond and bool(r["replicate"]) == replicate]
    return sorted(got, key=lambda r: r["model_seed"])


def paired(records: list[dict], cond: str) -> list[dict]:
    """The runs of one condition at the paired model seeds, 700 to 702."""
    return [r for r in runs(records, cond) if r["model_seed"] in PAIRED]


def eem(r: dict) -> float:
    return r["task"]["eem"]["all"]


CONTROL_EEM = {r["model_seed"]: eem(r) for r in runs(EVAL["runs"], CONTROL)}


def net_eem(cond: str) -> np.ndarray:
    """Held-out EEM less the control's at the same model seed, at the paired seeds."""
    return np.array([eem(r) - CONTROL_EEM[r["model_seed"]] for r in paired(EVAL["runs"], cond)])


def band(x: np.ndarray, y: np.ndarray) -> float:
    """The seed band between two conditions, with the seed standard deviation pooled over the two."""
    n1, n2 = len(x), len(y)
    var = ((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2)
    return float(ex.SEED_BAND_SD * np.sqrt(var) * np.sqrt(1 / n1 + 1 / n2))


def span(values, fmt: str = ".3f") -> str:
    return f"{min(values):{fmt}} to {max(values):{fmt}}"


def bold_if(ok, text: str) -> str:
    return f"<b>{text}</b>" if bool(ok) else text


def rule_color() -> str:
    return light_dark("#333", "#ddd")


def gate_line(ax: Axes, y: float, *, fail: str, xs: tuple[float, float] | None = None, ls: str = "--") -> None:
    """A gate rule with its failing side hatched; across the panel, or over the x-range *xs* only."""
    ink = rule_color()
    lo, hi = ax.get_ylim()
    band_ = (lo, y) if fail == "below" else (y, hi)
    edge = light_dark("#000", "#fff")
    if xs is None:
        ax.axhline(y, color=ink, lw=0.9, ls=ls, zorder=2)
        ax.axhspan(*band_, facecolor="none", edgecolor=edge, hatch="//", lw=0, zorder=0, alpha=0.1)
    else:
        ax.plot(xs, [y, y], color=ink, lw=0.9, ls=ls, zorder=2)
        ax.fill_between(xs, *band_, facecolor="none", edgecolor=edge, hatch="//", lw=0, zorder=0, alpha=0.1)
    ax.set_ylim(lo, hi)


def dots(
    ax: Axes, x: float, v, color, *, rng, ms: float = 5.0, slow: int | None = None, centre: float | None = None
) -> None:
    """One column of per-seed dots with the seed mean on top (or *centre*, where given); a thin bar behind spans the
    seed range. With *slow*, the dot at that index is drawn as a ring.
    """
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    jitter = x + rng.uniform(-0.08, 0.08, len(v))
    for i, (jx, y) in enumerate(zip(jitter, v, strict=True)):
        ring = i == slow
        ax.plot(
            jx,
            y,
            "o",
            ms=3.4 if ring else 2.6,
            color=color,
            alpha=0.8 if ring else 0.55,
            zorder=3,
            mfc="none" if ring else color,
            mew=0.8 if ring else 0,
        )
    ax.plot(
        x,
        v.mean() if centre is None else centre,
        "o",
        ms=ms,
        color=color,
        zorder=4,
        mec=light_dark("white", "#111"),
        mew=0.6,
    )


def ink_of(kind: str) -> str:
    """The colors of the report: the control, the base, a plain restriction, its matched twin, and context."""
    return {
        "control": light_dark("#555", "#bbb"),
        "base": light_dark("#c0392b", "#ff8a76"),
        "plain": light_dark("#1f6fb2", "#7ab8f5"),
        "matched": light_dark("#2e8b57", "#7fd8a4"),
        "context": light_dark("#8e6bb8", "#c9a8f0"),
        "slow": light_dark("#d97706", "#fbbf24"),
        "grey": light_dark("#aaa", "#666"),
        "replicate": light_dark("#0e7490", "#67e8f9"),
    }[kind]


def cond_ink(cond: str) -> str:
    if cond == CONTROL:
        return ink_of("control")
    if cond == ex.BASE:
        return ink_of("base")
    if cond.endswith("-matched"):
        return ink_of("matched")
    if cond in (WHOLE, NO_EMB_UNCAPPED):
        return ink_of("context")
    return ink_of("plain")


# --- The results: the edit criteria, per condition ---------------------------------------------------------


def supp_arrays(rs: list[dict]) -> dict[str, np.ndarray]:
    """Per run: the clean and null EEM by op, and the EEM, total variation, and KL from the target null by dose and
    op; stacked over the runs on the first axis.
    """
    return {
        "clean": np.array([r["clean"]["eem"] for r in rs]),
        "null": np.array([r["null"]["eem"] for r in rs]),
        "edits": np.array([[e["eem"] for e in r["edits"]] for r in rs]),
        "tv_clean": np.array([r["clean"]["tv_null"][D] for r in rs]),
        "tv_full": np.array([r["edits"][-1]["tv_null"][D] for r in rs]),
        "kl_clean": np.array([r["clean"]["kl_null"][D] for r in rs]),
        "kl_full": np.array([r["edits"][-1]["kl_null"][D] for r in rs]),
        "landing": np.array([r["landing"] for r in rs]),
        "model_seed": np.array([r["model_seed"] for r in rs]),
    }


assert [e["dose"] for e in SUPP["runs"][0]["edits"]] == list(ex.DOSE_GAMMAS)
_ctrl = supp_arrays(runs(SUPP["runs"], CONTROL))
CONTROL_DROP = (_ctrl["clean"][:, None, :] - _ctrl["edits"]).mean(axis=0)
# The control's seed-mean drop in EEM by dose and op, under the same edit; every drop here is net of it.


def edit_criteria(s: dict[str, np.ndarray]) -> dict[str, Any]:
    """Ex-2.2.21's two edit criteria and the selective reach, on the seed means of a set of runs, as there."""
    drop = (s["clean"][:, None, :] - s["edits"]).mean(axis=0) - CONTROL_DROP  # (doses, ops)
    gap = float((s["clean"][:, D] - s["null"][:, D]).mean())
    anchored = drop[:, D]
    worst = np.delete(drop, D, axis=1).max(axis=1)
    rises = all(b >= a - ex.ex2221.GRADE_DIP for a, b in zip(anchored, anchored[1:], strict=False))
    within = worst <= ex.SELECTIVITY_GATE
    return {
        "anchored": anchored,
        "worst_by_dose": worst,
        "gap": gap,
        "full": float(anchored[-1] / gap),
        "grades": bool(rises and anchored[-1] / gap >= ex.GRADING_MIN_DAMAGE),
        "worst": float(worst.max()),
        "selective": bool(within.all()),
        "reach": float(max((a / gap for a, w in zip(anchored, within, strict=True) if w), default=0.0)),
    }


def per_seed_criteria(s: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    return [edit_criteria({k: v[i : i + 1] for k, v in s.items()}) for i in range(len(s["clean"]))]


SUPP_PAIRED = {c: supp_arrays(paired(SUPP["runs"], c)) for c in ANCHORED}
CRIT = {c: edit_criteria(s) for c, s in SUPP_PAIRED.items()}
CRIT_SEEDS = {c: per_seed_criteria(s) for c, s in SUPP_PAIRED.items()}
SLICE_RUNS = [r for pair in SLICE_GROUPS.values() for c in pair for r in CRIT_SEEDS[c]]
# The runs of every slice restriction, for a count of those within the gate (post hoc).
NUM_WORDS = ("none", "one", "two", "three", "four", "five", "six")
NET = {c: net_eem(c) for c in ALL_CONDITIONS}
MARGIN = {c: np.array([r["margin"]["by_slice"] for r in paired(EVAL["runs"], c)]) for c in ALL_CONDITIONS}
TASK_BAND = {
    c: band(NET[c] + np.array([CONTROL_EEM[s] for s in PAIRED]), np.array([CONTROL_EEM[s] for s in PAIRED]))
    for c in ANCHORED
}


# --- E1 and E2: the task score -------------------------------------------------------------------------------

NET_ORDER = [
    ex.BASE,
    *(c for pair in SLICE_GROUPS.values() for c in pair),
    NO_EMB_UNCAPPED,
    "cap-0.9",
    "cap-0.95",
    WHOLE,
    "k-mixed",
]


SLOW_PATH = [short(c) for c in NET_ORDER if c != "k-mixed" and NET[c][SLOW - ex.SEED_OFFSET] < -0.1]
# The conditions whose run at model seed 702 ends far below the control (post hoc).


def net_figure() -> str:
    data = {short(c): {"v": NET[c].tolist(), "ink": cond_ink(c)} for c in NET_ORDER}
    hi = max(NET_ORDER, key=lambda c: NET[c].mean())
    alt = f"""
        Held-out expected exact match less the control at the same seed, one column per condition, three seeds each
        with the seed mean on top; the ring marks model seed {SLOW}. Seed {SLOW} sits about 0.13 below the control on
        {", ".join(SLOW_PATH[:-1])}, and {SLOW_PATH[-1]}, and within a few hundredths of it on the other restricted
        conditions; on whole-line both of the last two seeds sit about 0.1 below. The highest seed mean is
        {short(hi)}, at {NET[hi].mean():+.3f}. k-mixed is below the control at every seed,
        from {NET["k-mixed"].min():+.3f} to {NET["k-mixed"].max():+.3f}.
    """
    return net_draw(data, alt)


@memo
def net_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="task-net",
        alt_text=alt_text,
        caption=f"""
            **The task score net of the control.** Held-out expected exact match on the three-example held-out set,
            less that of the control at the same model seed. One small dot per seed and the seed mean on top; the
            ring marks model seed {SLOW}. Colors: the base (hinge) condition, the plain slice restrictions, their
            matched twins, and the conditions of ex-2.2.21 shown for context (whole-line is the uncapped end of the
            cap ladder). The dashed rule is the task tolerance of the decision, {ex.TASK_COST_TOL:g} below the
            control.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.2, 3.0), layout="constrained")
        rng = np.random.default_rng(0)
        ax.axhline(0, color=rule_color(), lw=0.7)
        ax.axhline(-ex.TASK_COST_TOL, color=rule_color(), lw=0.7, ls="--")
        for i, d in enumerate(data.values()):
            dots(ax, i, d["v"], d["ink"], rng=rng, slow=SLOW - ex.SEED_OFFSET)
        ax.set_xticks(range(len(data)), list(data), rotation=40, ha="right", fontsize=8)
        ax.set_xlim(-0.6, len(data) - 0.4)
        ax.set_ylabel("EEM less the control")
        return fig

    return _plot()


# --- E1 and E2: tables of the criteria --------------------------------------------------------------------------


def criteria_row(c: str) -> list[str]:
    k = CRIT[c]
    worst = [s["worst"] for s in CRIT_SEEDS[c]]
    reach = [s["reach"] for s in CRIT_SEEDS[c]]
    return [
        f"`{short(c)}`",
        f"{NET[c].mean():+.3f}",
        f"{span(NET[c], '+.3f')}",
        bold_if(k["grades"], "yes" if k["grades"] else "no"),
        bold_if(k["selective"], f"{k['worst']:+.3f}"),
        f"{span(worst, '+.3f')}",
        f"{k['reach']:.2f}",
        f"{span(reach, '.2f')}",
    ]


CRITERIA_HEAD = [
    "condition",
    "net EEM ↑",
    "seeds",
    "grades",
    "worst other ↓",
    "seeds",
    "selective reach ↑",
    "seeds",
]


def criteria_table(conds: list[str], caption: str) -> str:
    return table_html(CRITERIA_HEAD, [criteria_row(c) for c in conds], caption)


CRITERIA_NOTE = (
    "Net EEM is the held-out score less the control at the same seed. *Grades*: the net drop on `difference` "
    "rises with the dose and reaches half the way to the target null at full dose. *Worst other*: the largest net "
    f"drop of any other op at any dose, bold where it stays within the gate of {ex.SELECTIVITY_GATE:g}. *Selective "
    "reach*: the share of the way to the target null at the strongest dose within the gate. As in ex-2.2.21, the "
    "criteria are read off the seed-mean drops; the seed columns give the range of the same quantity per seed."
)


def margin_table(conds: list[str], caption: str) -> str:
    head = ["condition", *(slice_name(s) for s in SLICES)]
    rows = []
    for c in conds:
        pulled = set(ex.SLICE_SETS[ex.by_name(c).slices]) if c != CONTROL else set()
        m = MARGIN[c].mean(axis=0)
        rows.append([f"`{short(c)}`", *(f"{v:.2f}" if s in pulled else f"<i>{v:.2f}</i>" for s, v in enumerate(m))])
    return table_html(head, rows, caption)


# --- E1: the edit by slice set -----------------------------------------------------------------------------------

E1_PANELS = {
    "all": [ex.BASE],
    **{s: [*pair, *([NO_EMB_UNCAPPED] if s == "no-emb" else [])] for s, pair in SLICE_GROUPS.items()},
}


def e1_edit_figure() -> str:
    data = {
        s: {
            short(c): {
                "anchored": CRIT[c]["anchored"].tolist(),
                "worst": CRIT[c]["worst_by_dose"].tolist(),
                "half": ex.GRADING_MIN_DAMAGE * CRIT[c]["gap"],
                "ink": cond_ink(c),
            }
            for c in conds
        }
        for s, conds in E1_PANELS.items()
    }
    alt = f"""
        Four panels, one per slice set (all, no-emb, no-last, middle), each plotting the seed-mean net drop in
        expected exact match against the dose of the projection, from {min(ex.DOSE_GAMMAS):g} to
        {max(ex.DOSE_GAMMAS):g}. Solid lines are the drop on `{ex.ANCHORED_OP}`, dashed lines the worst other op;
        one color for the plain restriction and one for its matched twin, with the uncapped no-emb condition of
        ex-2.2.21 in the no-emb panel. In every panel the solid lines rise past the halfway rule. The dashed line of
        hinge stays near zero; in the restricted panels the dashed lines rise above the gate of
        {ex.SELECTIVITY_GATE:g}, highest on middle-matched, at {CRIT["middle-matched"]["worst"]:.2f}.
    """
    return e1_edit_draw(data, alt)


@memo
def e1_edit_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="e1-edit",
        alt_text=alt_text,
        caption=f"""
            **The edit on each slice set.** The projection at every position: the seed-mean drop in held-out
            expected exact match, less the control's drop under the same edit, against the dose. Solid: on
            `{ex.ANCHORED_OP}` contexts. Dashed: the worst of the other six ops. The dotted rule is the selectivity
            gate ({ex.SELECTIVITY_GATE:g}), and the dash-dot rule is half the way from the clean score on
            `{ex.ANCHORED_OP}` to the target null, for the base condition.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(data), figsize=(8.4, 2.8), layout="constrained", sharey=True)
        doses = list(ex.DOSE_GAMMAS)
        ink = rule_color()
        half = data["all"]["hinge"]["half"]
        for ax, (s, conds) in zip(axes, data.items(), strict=True):
            ax.axhline(ex.SELECTIVITY_GATE, color=ink, lw=0.8, ls=":")
            ax.axhline(half, color=ink, lw=0.8, ls="-.")
            ax.axhline(0, color=ink, lw=0.5)
            for name, d in conds.items():
                ax.plot(doses, d["anchored"], "-o", ms=3, color=d["ink"], lw=1.2, label=name)
                ax.plot(doses, d["worst"], "--", color=d["ink"], lw=1.0)
            ax.set_title(f"slices: {s}", fontsize=9)
            ax.set_xlabel("dose γ", fontsize=8)
            ax.set_xticks(doses)
            ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        axes[0].set_ylabel("net drop in EEM")
        return fig

    return _plot()


# --- E1: the alignment by slice --------------------------------------------------------------------------------

EX_ANSWERS = EVAL["runs"][0]["roles"]["example answers"]


def alignment_at_answers(c: str) -> tuple[np.ndarray, np.ndarray]:
    """Seed-mean α at the example answers by slice: on `difference` contexts, and the mean over the other ops."""
    a = np.mean([np.asarray(r["alignment"]) for r in paired(EVAL["runs"], c)], axis=0)  # ops × slices × positions
    at = a[:, :, EX_ANSWERS].mean(axis=2)
    return at[D], np.delete(at, D, axis=0).mean(axis=0)


def e1_align_figure() -> str:
    data = {
        s: {
            short(c): {
                "anchored": alignment_at_answers(c)[0].tolist(),
                "other": alignment_at_answers(c)[1].tolist(),
                "ink": cond_ink(c),
            }
            for c in conds
        }
        | {"_pulled": [list(ex.SLICE_SETS[s])]}
        for s, conds in E1_PANELS.items()
    }
    alt = f"""
        Four panels, one per slice set, each plotting the seed-mean alignment with e₁ at the example answers against
        the slice, 0 to {ex.N_LAYER}. Solid lines are `{ex.ANCHORED_OP}` contexts, dashed lines the other ops; one color
        per condition. Unpulled slices are shaded. On every condition the `{ex.ANCHORED_OP}` line rises from the
        embedding to its highest value in the middle or late slices; the dashed line peaks at slice 1, between 0.2 and
        0.4, and falls back to about 0.1 by the last slice.
    """
    return e1_align_draw(data, alt)


@memo
def e1_align_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="e1-alignment",
        alt_text=alt_text,
        caption=f"""
            **Where the anchor sits, by slice.** The seed-mean alignment α with e₁ at the example answers. Solid:
            `{ex.ANCHORED_OP}` contexts; dashed: the mean over the other ops. Shaded slices are left out of the pull
            in that panel.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, len(data), figsize=(8.4, 2.6), layout="constrained", sharey=True)
        shade = light_dark("#000", "#fff")
        for ax, (s, conds) in zip(axes, data.items(), strict=True):
            pulled = set(conds["_pulled"][0])
            for sl in SLICES:
                if sl not in pulled:
                    ax.axvspan(sl - 0.5, sl + 0.5, facecolor=shade, alpha=0.06, lw=0)
            ax.axhline(0, color=rule_color(), lw=0.5)
            for name, d in conds.items():
                if name == "_pulled":
                    continue
                ax.plot(SLICES, d["anchored"], "-o", ms=3, color=d["ink"], lw=1.2, label=name)
                ax.plot(SLICES, d["other"], "--", color=d["ink"], lw=0.9)
            ax.set_title(f"slices: {s}", fontsize=9)
            ax.set_xticks(SLICES, ["emb", *map(str, SLICES[1:])])
            ax.set_xlabel("slice", fontsize=8)
            ax.set_xlim(-0.5, SLICES[-1] + 0.5)
            ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        axes[0].set_ylabel("α at the example answers")
        return fig

    return _plot()


# --- E2: the cap ladder -----------------------------------------------------------------------------------------

CAP_X = {c: i for i, c in enumerate(CAP_LADDER)}


def e2_figure() -> str:
    data = {
        "labels": [f"{c:g}" if c is not None else "none" for c in CAP_LADDER],
        "worst": [[s["worst"] for s in CRIT_SEEDS[c]] for c in CAP_LADDER.values()],
        "reach": [[s["reach"] for s in CRIT_SEEDS[c]] for c in CAP_LADDER.values()],
        "net": [NET[c].tolist() for c in CAP_LADDER.values()],
        "centre": {
            "worst": [CRIT[c]["worst"] for c in CAP_LADDER.values()],
            "reach": [CRIT[c]["reach"] for c in CAP_LADDER.values()],
            "net": [float(NET[c].mean()) for c in CAP_LADDER.values()],
        },
        "margin": [MARGIN[c].mean(axis=0).tolist() for c in CAP_LADDER.values()],
    }
    alt = f"""
        Four panels against the cap ({", ".join(data["labels"])}), each with one dot per seed and the seed mean
        joined by a line; the ring marks model seed {SLOW}. The worst other op: seed means
        {", ".join(f"{CRIT[c]['worst']:+.3f}" for c in CAP_LADDER.values())}, with one seed of hinge and one of
        cap-0.95 well above the gate of {ex.SELECTIVITY_GATE:g}. The selective reach: seed means
        {", ".join(f"{CRIT[c]['reach']:.2f}" for c in CAP_LADDER.values())}. The task score net of the control:
        about the same at every cap, with the third seed far below the others, and on the uncapped condition the second seed too. The op margin by slice:
        one line per slice, each a little higher at 0.95 than at 0.8, with the uncapped condition just below 0.95.
    """
    return e2_draw(data, alt)


@memo
def e2_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="e2-caps",
        alt_text=alt_text,
        caption=f"""
            **The four caps.** Per seed (small dots, the ring is model seed {SLOW}), and the line through the large
            dots. From the left: the worst net drop of any other op at any dose of the edit, with the selectivity gate
            dotted, and the selective reach, where the line is read off the seed-mean drops as in the tables (so it
            need not be the mean of the dots); the task score net of the control; and the seed-mean op margin at each
            slice, one line per slice, darker for deeper slices. 0.8 is the hinge condition and "none" is the
            whole-line condition of ex-2.2.21.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.6), layout="constrained")
        ink = ink_of("base")
        x = np.arange(len(data["labels"]))
        rng = np.random.default_rng(1)
        for ax, key, label in zip(
            axes[:3],
            ("worst", "reach", "net"),
            ("worst other op", "selective reach", "EEM less the control"),
            strict=True,
        ):
            centre = data["centre"][key]
            for i, v in enumerate(data[key]):
                dots(ax, i, v, ink, rng=rng, ms=4, slow=SLOW - ex.SEED_OFFSET, centre=centre[i])
            ax.plot(x, centre, "-", color=ink, lw=1.0, zorder=1)
            ax.set_ylabel(label, fontsize=8)
        axes[0].axhline(ex.SELECTIVITY_GATE, color=rule_color(), lw=0.8, ls=":")
        axes[2].axhline(0, color=rule_color(), lw=0.5)
        shades = plt.get_cmap("viridis")(np.linspace(0.85, 0.1, len(SLICES)))
        m = np.array(data["margin"])
        for s, color in zip(SLICES, shades, strict=True):
            axes[3].plot(x, m[:, s], "-o", ms=2.5, lw=1.0, color=color, label=slice_name(s))
        axes[3].set_ylabel("op margin", fontsize=8)
        axes[3].legend(frameon=False, fontsize=6, loc="center right")
        for ax in axes:
            ax.set_xticks(x, data["labels"], fontsize=8)
            ax.set_xlabel("cap", fontsize=8)
            ax.set_xlim(-0.5, len(x) - 0.5)
        return fig

    return _plot()


# --- H1 ---------------------------------------------------------------------------------------------------------

HINGE_EEM = np.array([eem(r) for r in paired(EVAL["runs"], ex.BASE)])
MIXED_EEM = np.array([eem(r) for r in paired(EVAL["runs"], "k-mixed")])
CTRL_PAIRED = np.array([CONTROL_EEM[s] for s in PAIRED])
H1: dict[str, Any] = {
    "short": float((HINGE_EEM - MIXED_EEM).mean()),
    "by_seed": (HINGE_EEM - MIXED_EEM).tolist(),
    "band": band(HINGE_EEM, MIXED_EEM),
}
H1["verdict"] = (
    "Unresolved"
    if max(H1["by_seed"]) > ex.REGRESSION_TOL and min(H1["by_seed"]) < -ex.REGRESSION_TOL
    else "Pass"
    if H1["short"] <= ex.REGRESSION_TOL
    else "Partial"
    if H1["short"] <= H1["band"]
    else "Miss"
)
_t = runs(EVAL["runs"], ex.BASE)[0]["task"]
CEIL3, FLOOR3 = _t["ceiling"]["all"], _t["floor"]["all"]


def skill_at(r: dict, k: int) -> float:
    c = r["counts"][str(k)]
    return (c["eem"]["all"] - c["floor"]["all"]) / (c["ceiling"]["all"] - c["floor"]["all"])


SKILL_CONDITIONS = ("k-mixed", ex.BASE, CONTROL)
SKILL = {
    c: np.array([[skill_at(r, k) for k in ex.MIXED_COUNTS] for r in paired(BY_COUNT["runs"], c)])
    for c in SKILL_CONDITIONS
}
LONG_DROP = int(SKILL[ex.BASE][:, -1].argmin())
# The `hinge` run with the lowest skill at five examples (post hoc).


def h1_figure() -> str:
    data = {
        "eem": {
            short(c): v.tolist() for c, v in ((CONTROL, CTRL_PAIRED), (ex.BASE, HINGE_EEM), ("k-mixed", MIXED_EEM))
        },
        "inks": [cond_ink(CONTROL), cond_ink(ex.BASE), ink_of("plain")],
        "hinge": float(HINGE_EEM.mean()),
        "band": H1["band"],
        "ceiling": CEIL3,
        "floor": FLOOR3,
    }
    alt = f"""
        Held-out expected exact match on the three-example held-out set, one column each for the control, hinge, and
        k-mixed, three seeds each with the seed mean on top; the ring marks model seed {SLOW}. Over the k-mixed
        column a dashed gate at the hinge mean less {ex.REGRESSION_TOL:g}, {HINGE_EEM.mean() - ex.REGRESSION_TOL:.3f},
        hatched below, and a dotted mark at the hinge mean less the seed band, {HINGE_EEM.mean() - H1["band"]:.3f}.
        k-mixed seeds run from {MIXED_EEM.min():.3f} to {MIXED_EEM.max():.3f}, all below the dashed gate and
        {["none", "one", "two", "all three"][int((MIXED_EEM < HINGE_EEM.mean() - H1["band"]).sum())]} below the dotted mark; hinge from
        {HINGE_EEM.min():.3f} to {HINGE_EEM.max():.3f}.
    """
    return h1_draw(data, alt)


@memo
def h1_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="h1-recipe",
        alt_text=alt_text,
        caption=f"""
            **`k-mixed` against `hinge` on three examples.** Held-out expected exact match on ex-2.2.21's held-out set,
            one small dot per seed and the seed mean on top; the ring is model seed {SLOW}. The right axis gives the
            same scale as skill, from the floor (0) to the Bayes ceiling (1, the dashed rule at the top). Over the
            `k-mixed` column, the dashed rule is the `hinge` mean less {ex.REGRESSION_TOL:g}, with the failing side
            hatched, and the dotted rule the `hinge` mean less the seed band of three seeds against three.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.2, 3.2), layout="constrained")
        rng = np.random.default_rng(0)
        cols = list(data["eem"])
        ax.axhline(data["ceiling"], color=rule_color(), lw=0.9, ls="--")
        for i, (name, ink) in enumerate(zip(cols, data["inks"], strict=True)):
            dots(ax, i, data["eem"][name], ink, rng=rng, slow=SLOW - ex.SEED_OFFSET)
        ax.set_ylim(data["floor"], data["ceiling"] + 0.02)
        i = cols.index("k-mixed")
        ax.plot([i - 0.3, i + 0.3], [data["hinge"] - data["band"]] * 2, ":", color=rule_color(), lw=0.9)
        gate_line(ax, data["hinge"] - ex.REGRESSION_TOL, fail="below", xs=(i - 0.3, i + 0.3))
        floor, ceil = data["floor"], data["ceiling"]
        ax.secondary_yaxis(
            "right", functions=(lambda y: (y - floor) / (ceil - floor), lambda s: floor + s * (ceil - floor))
        ).set_ylabel("skill")
        ax.set_ylabel("held-out EEM, three examples")
        ax.set_xticks(range(len(cols)), cols, fontsize=8)
        ax.set_xlim(-0.6, len(cols) - 0.4)
        return fig

    return _plot()


def h1_table() -> str:
    rows = [
        [
            "`k-mixed` short of `hinge`",
            f"{MIXED_EEM.mean():.4f} vs. {HINGE_EEM.mean():.4f}",
            ", ".join(f"{v:+.3f}" for v in H1["by_seed"]),
            f"{H1['short']:.4f}",
            f"≤ {ex.REGRESSION_TOL:g}",
            f"{H1['band']:.4f}",
            bold_if(H1["verdict"] == "Pass", H1["verdict"].lower()),
        ]
    ]
    return table_html(
        ["criterion", "seed means", "by seed (700–702)", "shortfall ↓", "tolerance", "seed band", "verdict"],
        rows,
        "**The criterion of H1.** Held-out expected exact match on the three-example held-out set, paired by model "
        "seed. The seed band is for three seeds against three, with the seed standard deviation pooled over the two "
        "conditions; a shortfall above the tolerance but inside the band is a partial pass.",
        text_cols=3,
    )


def skill_figure() -> str:
    data = {
        short(c): {"v": SKILL[c].tolist(), "ink": e3_ink(c), "marker": E3_MARKERS[E3_CONDITIONS.index(c)]}
        for c in SKILL_CONDITIONS
    }
    km = SKILL["k-mixed"].mean(axis=0)
    hg = SKILL[ex.BASE].mean(axis=0)
    alt = f"""
        Skill against the number of examples, one to five, for k-mixed, hinge, and the control, one thin line per
        seed and a heavy line for the seed mean. k-mixed rises with the count, from {km[0]:.2f} at one example to
        {km[-1]:.2f} at five. hinge and the control peak at three examples, the count they were trained on (hinge
        {hg[2]:.2f}), and are lower on either side: hinge {hg[0]:.2f} at one example and {hg[-1]:.2f} at five. The
        k-mixed mean is below hinge at three examples and above it at one and at five.
    """
    return skill_draw(data, alt)


@memo
def skill_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="h1-skill-by-count",
        alt_text=alt_text,
        caption=f"""
            **Skill at each example count.** The share of the way from the floor to the Bayes ceiling at that count,
            on a held-out set of {ex.HOLDOUT_CONTEXTS:,} contexts per op at each count. Thin lines: one per seed;
            the hollow marker marks model seed {SLOW}. Heavy lines: the seed mean. `hinge` and the control trained on three
            examples only.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.6, 3.0), layout="constrained")
        x = np.array(ex.MIXED_COUNTS)
        for name, d in data.items():
            v = np.array(d["v"])
            for i, row in enumerate(v):
                ax.plot(x, row, "-", color=d["ink"], lw=0.6, alpha=0.5)
                if i == SLOW - ex.SEED_OFFSET:
                    ax.plot(x, row, d["marker"], ms=3, mfc="none", color=d["ink"], mew=0.7)
            ax.plot(x, v.mean(axis=0), "-", marker=d["marker"], ms=3.5, color=d["ink"], lw=1.6, label=name)
        ax.set_xticks(x)
        ax.set_xlabel("examples per context (k)")
        ax.set_ylabel("skill")
        ax.legend(frameon=False, fontsize=7)
        return fig

    return _plot()


# --- E3 ----------------------------------------------------------------------------------------------------------

E3_CONDITIONS = ("k-mixed", ex.BASE, WHOLE, CONTROL)
POST_EDGES = np.linspace(0, 1, 6)
MIN_BIN = 30
# A bin with fewer points than this in a run is left out of that run's line.


def answer_points(c: str, counts: Sequence[int]) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Per paired seed, per count: the posterior on `difference` at each answer, `(n, k + 1)`, and α there by slice,
    `(n, slices, k + 1)`, over the held-out `difference` contexts.
    """
    out = []
    for r in paired(BY_COUNT["runs"], c):
        a = COUNT_ARRAYS[r["label"]]
        per = []
        for k in counts:
            keep = a[f"k{k}_op_ids"] == D
            per.append((a[f"k{k}_posterior_at_answer"][keep], a[f"k{k}_alpha_at_answer"][keep].astype(np.float32)))
        out.append(per)
    return out


def binned(post: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Mean α by slice in each posterior bin, `(bins, slices)`; NaN where a bin has fewer than `MIN_BIN` points."""
    b = np.clip(np.digitize(post, POST_EDGES[1:-1]), 0, len(POST_EDGES) - 2)
    out = np.full((len(POST_EDGES) - 1, alpha.shape[1]), np.nan)
    for i in range(len(POST_EDGES) - 1):
        if (b == i).sum() >= MIN_BIN:
            out[i] = alpha[b == i].mean(axis=0)
    return out


def e3_counts(c: str) -> tuple[int, ...]:
    return ex.MIXED_COUNTS if c == "k-mixed" else (ex.K,)


def e3_curves(c: str) -> dict[str, np.ndarray]:
    """Seed-mean α by posterior bin and slice, at the example answers and at the query answer."""
    seeds = answer_points(c, e3_counts(c))
    out = {}
    for site in ex.GRADING_SITES:
        per_seed = []
        for per in seeds:
            if site == "example answers":
                post = np.concatenate([p[:, :-1].ravel() for p, _ in per])
                alpha = np.concatenate([a[:, :, :-1].transpose(0, 2, 1).reshape(-1, a.shape[1]) for _, a in per])
            else:
                post = np.concatenate([p[:, -1] for p, _ in per])
                alpha = np.concatenate([a[:, :, -1] for _, a in per])
            per_seed.append(binned(post, alpha))
        out[site] = np.nanmean(np.stack(per_seed), axis=0)
    return out


def e3_by_index(c: str) -> np.ndarray:
    """On the three-example held-out set: seed-mean α by answer index, posterior bin, and slice."""
    seeds = answer_points(c, (ex.K,))
    return np.nanmean(np.stack([[binned(p[:, j], a[:, :, j]) for j in range(ex.K + 1)] for ((p, a),) in seeds]), axis=0)


E3 = {c: e3_curves(c) for c in E3_CONDITIONS}
E3_INDEX = {c: e3_by_index(c) for c in E3_CONDITIONS}
BIN_MID = (POST_EDGES[:-1] + POST_EDGES[1:]) / 2


def e3_ink(c: str) -> str:
    return ink_of("plain") if c == "k-mixed" else cond_ink(c)


E3_MARKERS = ("s", "o", "^", "x")
# One marker per condition of `E3_CONDITIONS`, so the E3 figures read without color.


def e3_figure() -> str:
    data = {
        "curves": {short(c): {s: v.tolist() for s, v in E3[c].items()} for c in E3_CONDITIONS},
        "inks": [e3_ink(c) for c in E3_CONDITIONS],
        "markers": list(E3_MARKERS),
    }
    alt = f"""
        A grid of two rows (the example answers, the query answer) and {len(SLICES)} columns (slices 0 to
        {ex.N_LAYER}). Each panel plots the seed-mean alignment with e₁ against the posterior on `{ex.ANCHORED_OP}`
        given the pairs so far, in five bins, one line per condition: k-mixed (pooled over its counts), hinge,
        whole-line, and the control. The control stays near zero in every panel. At slice 0 every line is flat near
        zero. From slice 2 on, at the example answers, the anchored lines rise with the posterior, peaking in the bin
        from 0.6 to 0.8. At the query answer the three-example conditions have points in the top bin only, where the
        anchored ones sit near 0.9 past slice 1; k-mixed has a point in the middle bin too, lower.
    """
    return e3_draw(data, alt)


@memo
def e3_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="e3-alignment-by-posterior",
        alt_text=alt_text,
        caption=f"""
            **α against the posterior on `{ex.ANCHORED_OP}`.** At each answer of a held-out `{ex.ANCHORED_OP}`
            context, the alignment with e₁ against the posterior on `{ex.ANCHORED_OP}` given the pairs up to and
            including that answer, in five equal bins: the mean within each bin per run, then the mean over the three
            paired seeds. A bin with fewer than {MIN_BIN} points in a run is left out of it. `k-mixed` pools its
            held-out sets at every count; the other conditions are read at three examples.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(2, len(SLICES), figsize=(8.4, 3.8), layout="constrained", sharex=True, sharey=True)
        for row, site in enumerate(ex.GRADING_SITES):
            for s in SLICES:
                ax = axes[row][s]
                ax.axhline(0, color=rule_color(), lw=0.5)
                for (name, curves), ink, mk in zip(data["curves"].items(), data["inks"], data["markers"], strict=True):
                    ax.plot(
                        BIN_MID, np.array(curves[site])[:, s], "-", marker=mk, ms=3.2, lw=1.0, color=ink, label=name
                    )
                if row == 0:
                    ax.set_title(slice_name(s), fontsize=9)
                if row == 1:
                    ax.set_xlabel("posterior", fontsize=8)
            axes[row][0].set_ylabel(f"α, {site}", fontsize=8)
        axes[0][0].legend(frameon=False, fontsize=6.5, loc="upper left")
        return fig

    return _plot()


def e3_index_figure() -> str:
    data = {
        "curves": {short(c): E3_INDEX[c].tolist() for c in E3_CONDITIONS},
        "inks": [e3_ink(c) for c in E3_CONDITIONS],
        "markers": list(E3_MARKERS),
    }
    alt = f"""
        A grid of four rows (the answers of the first, second, and third example, then the query answer) and
        {len(SLICES)} columns (slices 0 to {ex.N_LAYER}), on the three-example held-out set. Each panel plots the
        seed-mean alignment with e₁ against the posterior on `{ex.ANCHORED_OP}` in bins, one line per condition. The
        first answer has no points in the top bin, and the query answer has points in the top bin only. Within a row
        the anchored lines rise with the posterior from slice 2 on, and at the same posterior they sit higher in
        earlier rows.
    """
    return e3_index_draw(data, alt)


@memo
def e3_index_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="e3-by-answer",
        alt_text=alt_text,
        caption="""
            **The same at each answer, on three examples.** As the figure above, with one row per answer: the
            answers of the three examples, then the query answer. Every condition is read on the three-example
            held-out set, `k-mixed` included.
        """,
    )
    def _plot() -> plt.Figure:
        rows = ex.K + 1
        fig, axes = plt.subplots(rows, len(SLICES), figsize=(8.4, 6.0), layout="constrained", sharex=True, sharey=True)
        names = [f"example {j + 1}" for j in range(ex.K)] + ["query"]
        for j in range(rows):
            for s in SLICES:
                ax = axes[j][s]
                ax.axhline(0, color=rule_color(), lw=0.5)
                for (name, v), ink, mk in zip(data["curves"].items(), data["inks"], data["markers"], strict=True):
                    ax.plot(BIN_MID, np.array(v)[j, :, s], "-", marker=mk, ms=3.2, lw=1.0, color=ink, label=name)
                if j == 0:
                    ax.set_title(slice_name(s), fontsize=9)
                if j == rows - 1:
                    ax.set_xlabel("posterior", fontsize=8)
            axes[j][0].set_ylabel(f"α, {names[j]}", fontsize=8)
        axes[0][0].legend(frameon=False, fontsize=6.5, loc="upper left")
        return fig

    return _plot()


def posterior_points(counts: Sequence[int]) -> np.ndarray:
    """The posterior on `difference` at every answer of the held-out `difference` contexts at *counts*; the same for
    every run, so read from the first `k-mixed` run.
    """
    a = COUNT_ARRAYS[paired(BY_COUNT["runs"], "k-mixed")[0]["label"]]
    return np.concatenate([a[f"k{k}_posterior_at_answer"][a[f"k{k}_op_ids"] == D].ravel() for k in counts])


E3_POINTS = {"k-mixed": posterior_points(ex.MIXED_COUNTS), "three examples": posterior_points((ex.K,))}


def e3_points_figure() -> str:
    data = {k: v.tolist() for k, v in E3_POINTS.items()}
    alt = f"""
        Cumulative curves of the posterior on `{ex.ANCHORED_OP}` at every answer of the held-out
        `{ex.ANCHORED_OP}` contexts, for k-mixed (all counts) and for three examples. Both climb in steps, the largest
        near 0.05, 0.42, 0.69, and 0.91, with about half of the points above 0.9 and about a quarter at 1. The two
        curves nearly overlap; the k-mixed curve has a few more, smaller steps.
    """
    return e3_points_draw(data, alt)


@memo
def e3_points_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="e3-points",
        alt_text=alt_text,
        caption=f"""
            **Where the points of the figures below lie.** The posterior on `{ex.ANCHORED_OP}` at every answer of the
            held-out `{ex.ANCHORED_OP}` contexts, the example answers and the query answer together: the share of
            points at or below the posterior on the x-axis.
        """,
    )
    def _plot() -> plt.Figure:
        fig, b = plt.subplots(figsize=(4.2, 2.6), layout="constrained")
        inks = [ink_of("plain"), ink_of("base")]
        x = np.linspace(0, 1, 401)
        for (name, v), ink, ls in zip(data.items(), inks, ("--", "-"), strict=True):
            v = np.asarray(v)
            b.plot(x, np.searchsorted(np.sort(v), x, side="right") / len(v), color=ink, lw=1.4, ls=ls, label=name)
        b.set_xlabel("posterior", fontsize=8)
        b.set_ylabel("share at or below", fontsize=8)
        b.set_xlim(0, 1)
        b.set_ylim(0, 1)
        b.legend(frameon=False, fontsize=7, loc="upper left")
        return fig

    return _plot()


# --- H2 ----------------------------------------------------------------------------------------------------------

REPLICATE = supp_arrays(runs(SUPP["runs"], ex.BASE, replicate=True))
STORED = SUPP_PAIRED[ex.BASE]
H2_PASS = REPLICATE["landing"] >= ex.LANDING_FRACTION
H2_VERDICT = "Pass" if H2_PASS.all() else "Miss" if not H2_PASS.any() else "Unresolved"
LANDING = {c: supp_arrays(paired(SUPP["runs"], c))["landing"] for c in ALL_CONDITIONS}
REPLICATE_CRIT = edit_criteria(REPLICATE)
HINGE_WITHIN = sum(r["selective"] for r in [*CRIT_SEEDS[ex.BASE], *per_seed_criteria(REPLICATE)])
# How many of the six `hinge` runs (paired and replicate) edit within the gate (post hoc).
# Ex-2.2.21's edit criteria on the replicate runs (post hoc: H2 scores the landing on them, and nothing else).


def hinge_edit_rows() -> list[list[str]]:
    """Per `hinge` run, paired seeds then replicate: the worst other op under the edit, net of the control, and the
    selective reach.
    """
    rows = []
    for label, arrays in (("paired", STORED), ("replicate", REPLICATE)):
        for i, k in enumerate(per_seed_criteria(arrays)):
            drop = arrays["clean"][i] - arrays["edits"][i] - CONTROL_DROP  # (doses, ops)
            others = np.delete(np.arange(len(OPS)), D)
            op = OPS[others[np.unravel_index(drop[:, others].argmax(), drop[:, others].shape)[1]]]
            rows.append(
                [
                    f"{int(arrays['model_seed'][i])}",
                    label,
                    f"`{op}`",
                    bold_if(k["selective"], f"{k['worst']:+.3f}"),
                    f"{k['reach']:.2f}",
                ]
            )
    return rows


def hinge_edit_table() -> str:
    return table_html(
        ["model seed", "runs", "worst op", "worst other ↓", "selective reach ↑"],
        hinge_edit_rows(),
        "**The edit on each `hinge` run**, at the paired seeds and on the replicate. *Worst other*: the largest drop "
        "in EEM of any other op at any dose, net of the seed-mean drop of the control, bold where it stays within "
        f"the gate of {ex.SELECTIVITY_GATE:g}; *worst op* is the op it falls on. *Selective reach* as in the tables of E1.",
        text_cols=3,
    )


def h2_figure() -> str:
    data = {
        "replicate": {k: REPLICATE[k].tolist() for k in ("tv_clean", "tv_full", "kl_clean", "kl_full", "model_seed")},
        "stored": {k: STORED[k].tolist() for k in ("tv_clean", "tv_full", "kl_clean", "kl_full", "model_seed")},
        "landing": {short(c): LANDING[c].tolist() for c in [ex.BASE, *(c for c in NET_ORDER if c != ex.BASE), CONTROL]},
        "inks": [cond_ink(c) for c in [ex.BASE, *(c for c in NET_ORDER if c != ex.BASE), CONTROL]],
        "repl_landing": REPLICATE["landing"].tolist(),
    }
    alt = f"""
        Three panels. Left: the total variation distance from the target null on `{ex.ANCHORED_OP}` contexts, clean
        and under the full edit, as paired points joined by a line, for the three replicate runs and, fainter, the
        three stored hinge runs. Each run's clean distance is about {span(REPLICATE["tv_clean"], ".2f")} on the
        replicate and its edited distance {span(REPLICATE["tv_full"], ".2f")}; a tick on each line marks half the
        clean distance, which every replicate run stays above. Middle: the KL divergence of the target null from
        the model, clean and edited, for the same runs; it falls under the edit on every run. Right: the landing per
        run for every condition, with the gate at {ex.LANDING_FRACTION:g}. The replicate runs land at
        {", ".join(f"{v:.2f}" for v in REPLICATE["landing"])}; most anchored runs land between 0.35 and 0.5, and the
        control near zero.
    """
    return h2_draw(data, alt)


@memo
def h2_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="h2-landing",
        alt_text=alt_text,
        caption=f"""
            **The edit and the target null.** On held-out `{ex.ANCHORED_OP}` contexts, the full edit (the projection
            at every position, γ = 1). Left: the mean total variation distance of the model from the target null,
            clean and edited, for the replicate runs (solid) and the stored `hinge` runs (faint); the tick on each
            line is half the clean distance, the point a run has to reach to pass. Middle: the mean KL divergence
            KL(target null ‖ model), for the same runs. Right: the landing, one dot per run at the paired seeds, with
            the replicate beside `hinge` as squares and the gate of H2 dashed.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), layout="constrained", width_ratios=[1, 1, 2.6])
        repl, stored = ink_of("replicate"), ink_of("base")
        for ax, key in zip(axes[:2], ("tv", "kl"), strict=True):
            for runs_, ink, alpha in ((data["stored"], stored, 0.35), (data["replicate"], repl, 1.0)):
                for c, f in zip(runs_[f"{key}_clean"], runs_[f"{key}_full"], strict=True):
                    ax.plot([0, 1], [c, f], "-o", ms=3.5, color=ink, alpha=alpha, lw=1.0)
                    if key == "tv":
                        ax.plot([0.92, 1.08], [c / 2] * 2, "-", color=ink, alpha=alpha, lw=1.2)
            ax.set_xticks([0, 1], ["clean", "edited"], fontsize=8)
            ax.set_xlim(-0.3, 1.3)
            ax.set_ylim(0, None)
        axes[0].set_ylabel("TV from the target null", fontsize=8)
        axes[1].set_ylabel("KL(target null ‖ model)", fontsize=8)
        ax = axes[2]
        rng = np.random.default_rng(2)
        names = list(data["landing"])
        for i, (name, ink) in enumerate(zip(names, data["inks"], strict=True)):
            dots(ax, i, data["landing"][name], ink, rng=rng, ms=4, slow=SLOW - ex.SEED_OFFSET)
        ax.plot([0.25] * 3, data["repl_landing"], "s", ms=3.5, color=repl, zorder=4)
        ax.axhline(ex.LANDING_FRACTION, color=rule_color(), lw=0.9, ls="--")
        ax.axhline(0, color=rule_color(), lw=0.5)
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=7)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylabel("landing", fontsize=8)
        return fig

    return _plot()


def h2_table() -> str:
    rows = [
        [
            f"{int(s)}",
            f"{tc:.3f}",
            f"{tf:.3f}",
            bold_if(land >= ex.LANDING_FRACTION, f"{land:.3f}"),
            f"{kc:.2f}",
            f"{kf:.2f}",
        ]
        for s, tc, tf, land, kc, kf in zip(
            REPLICATE["model_seed"],
            REPLICATE["tv_clean"],
            REPLICATE["tv_full"],
            REPLICATE["landing"],
            REPLICATE["kl_clean"],
            REPLICATE["kl_full"],
            strict=True,
        )
    ]
    return table_html(
        ["model seed", "TV, clean", "TV, edited", "landing ↑", "KL, clean", "KL, edited"],
        rows,
        f"**The replicate runs of H2.** Means over the held-out `{ex.ANCHORED_OP}` contexts. The landing is one less "
        f"the ratio of the two total variation columns; bold would mark a landing of at least "
        f"{ex.LANDING_FRACTION:g}.",
    )


# --- H2: where the answers go ---------------------------------------------------------------------------------

CONF_BINS = (0.0, MID_LO, MID_HI, 1.0)
# The bins of the posterior on the true op: below the middle band, the middle band, and above it.


@memo
def gives_table(pair: np.ndarray) -> np.ndarray:
    """For each op and context, which colors that op can give on the query pair: `(ops, n, colors)`."""
    n = len(pair)
    out = np.zeros((TABLE7.n_ops, n, P.N_COLORS), dtype=bool)
    rows = np.arange(n)
    for o in range(TABLE7.n_ops):
        idx, ok = TABLE7.idx[o, pair], TABLE7.prob[o, pair] > 0
        for j in range(idx.shape[1]):
            out[o, rows[ok[:, j]], idx[ok[:, j], j]] = True
    return out


def confusion(
    p: np.ndarray, op_ids: np.ndarray, post: np.ndarray, pair: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """Where a distribution puts its mass, by op, as in ex-2.2.20, over the contexts whose posterior on the true op
    lies in [lo, hi). Rows are the true op. The diagonal is the mass on the colors the true op can give for the
    query operands; off it, the mass on the colors the column op can give and the true op cannot.
    """
    rows = np.arange(len(p))
    on_true = post[rows, op_ids]
    sel = (on_true >= lo) & ((on_true < hi) if hi < 1 else (on_true <= hi))
    gives = gives_table(pair)[:, sel]
    p, true_op = p[sel].astype(float), op_ids[sel]
    rows = np.arange(len(p))
    true = gives[true_op, rows]
    per_col = np.stack(
        [(p * np.where((true_op == o)[:, None], true, gives[o] & ~true)).sum(1) for o in range(len(OPS))], 1
    )
    m = np.full((len(OPS), len(OPS)), np.nan)
    for o in range(len(OPS)):
        if (true_op == o).sum() >= MIN_BIN:
            m[o] = per_col[true_op == o].mean(0)
    return m


def target_null_dist(a: dict[str, np.ndarray]) -> np.ndarray:
    """The answer distribution of the target null on each context: the posterior with `difference` removed."""
    ctx = P.Contexts(
        a["op_ids"].astype(np.int64),
        np.zeros((len(a["op_ids"]), 0), np.int64),
        np.zeros((len(a["op_ids"]), 0), np.int64),
        a["query_pair"],
    )
    return P.predictive(TABLE7, ctx, ex.ex2216._target_null(P, TABLE7, ctx, a["posterior"].astype(float), D))


@memo
def confusion_set(arrays: dict[str, dict[str, np.ndarray]]) -> dict[str, list[np.ndarray]]:
    """Per bin of the posterior on the true op, the seed-mean confusion matrix of the clean model, the edited model,
    and the target null, over the replicate runs.
    """
    out: dict[str, list[np.ndarray]] = {"clean": [], "edited": [], "target null": []}
    for lo, hi in zip(CONF_BINS, CONF_BINS[1:], strict=False):
        per = {k: [] for k in out}
        for a in arrays.values():
            args = (a["op_ids"].astype(np.int64), a["posterior"], a["query_pair"], lo, hi)
            per["clean"].append(confusion(a["p_clean"], *args))
            per["edited"].append(confusion(a["p_full"], *args))
            per["target null"].append(confusion(target_null_dist(a), *args))
        for k in out:
            out[k].append(np.nanmean(np.stack(per[k]), axis=0))
    return out


CONFUSION = confusion_set(SUPP_ARRAYS)
PRINT_FLOOR = 0.01


def seq_cmap():
    cmap = plt.get_cmap(light_dark("Blues", "magma")).copy()
    cmap.set_bad(light_dark("#fff", "#111"))
    return cmap


def cell_text_color(v: float, vmax: float) -> str:
    dark_cell = (v / vmax > 0.55) == (light_dark(0, 1) == 0)
    return "#fff" if dark_cell else "#000"


def bin_label(lo: float, hi: float) -> str:
    return f"posterior {lo:g} to {hi:g}"


def confusion_figure() -> str:
    data = {k: [m.tolist() for m in v] for k, v in CONFUSION.items()}
    d_row = {k: [float(np.nanmean(np.delete(np.array(m)[D], D))) for m in v] for k, v in data.items()}
    alt = f"""
        A grid of {len(CONF_BINS) - 1} rows, one per bin of the posterior on the true op, and three columns: the clean
        model, the edited model, and the target null, each a {len(OPS)} by {len(OPS)} matrix with the true op on the
        rows and, on the columns, the mass on the answers of each op that the true op cannot give. The diagonal is
        outlined, with its value printed in grey and no color. Outside the `{ex.ANCHORED_OP}` row the edited matrices put a little more mass
        off the diagonal than the clean ones, in the two upper bins. In the `{ex.ANCHORED_OP}` row the clean model puts little mass off the diagonal, and the edited model
        and the target null put much more there; the mean cell off the diagonal in that row, by bin from low to high, is
        {", ".join(f"{v:.2f}" for v in d_row["clean"])} clean, {", ".join(f"{v:.2f}" for v in d_row["edited"])}
        edited, and {", ".join(f"{v:.2f}" for v in d_row["target null"])} for the target null.
    """
    return confusion_draw(data, alt)


@memo
def confusion_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="h2-confusion",
        alt_text=alt_text,
        caption=f"""
            **Where the answers go, by how sure the context is.** Op confusion, as in ex-2.2.20, over the held-out
            contexts of every op, for the replicate runs (seed mean). Rows: the true op. Columns: the mass on the
            colors the column op can give for the query operands and the true op cannot. The diagonal (the mass on
            the answers of the true op) is outlined, printed in grey italics, and left out of the color scale. One row of panels per bin of the
            posterior on the true op; a row of a matrix with fewer than {MIN_BIN} contexts in a run is blank. Values
            of at least {PRINT_FLOOR:g} are printed.
        """,
    )
    def _plot() -> plt.Figure:
        nb = len(CONF_BINS) - 1
        fig, axes = plt.subplots(nb, 3, figsize=(8.4, 2.7 * nb + 0.6), layout="constrained", sharex=True, sharey=True)
        n = len(OPS)
        eye = np.eye(n, dtype=bool)
        mats = [np.array(m) for v in data.values() for m in v]
        vmax = max(float(np.nanmax(np.where(eye, np.nan, m))) for m in mats)
        im = None
        for c, (name, ms) in enumerate(data.items()):
            for b, m in enumerate(ms):
                m = np.array(m)
                ax = axes[b][c]
                im = ax.imshow(np.where(eye, np.nan, m), cmap=seq_cmap(), vmin=0, vmax=vmax)
                for i, j in np.ndindex(n, n):
                    if i == j:
                        ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec="0.6", lw=0.6))
                        if np.isfinite(m[i, j]):
                            ax.text(
                                j,
                                i,
                                f"{m[i, j]:.2f}".removeprefix("0"),
                                ha="center",
                                va="center",
                                fontsize=6,
                                color=ink_of("grey"),
                                fontstyle="italic",
                            )
                    elif np.isfinite(m[i, j]) and m[i, j] >= PRINT_FLOOR:
                        ax.text(
                            j,
                            i,
                            f"{m[i, j]:.2f}"[1:],
                            ha="center",
                            va="center",
                            fontsize=6,
                            color=cell_text_color(m[i, j], vmax),
                        )
                if b == 0:
                    ax.set_title(name, fontsize=9)
                ax.set_xticks(range(n), OPS, rotation=90, fontsize=7)
                ax.set_yticks(range(n), OPS, fontsize=7)
        for b in range(nb):
            axes[b][0].set_ylabel(bin_label(CONF_BINS[b], CONF_BINS[b + 1]), fontsize=8)
        assert im is not None
        fig.colorbar(im, ax=axes, shrink=0.4, label="mass")
        return fig

    return _plot()


# --- E4: where the remaining distance is (post hoc) ------------------------------------------------------------

SPLIT_GROUPS = ("colors another op gives", "colors only `difference` gives", "colors no op gives")


def ideal_dist(a: dict[str, np.ndarray]) -> np.ndarray:
    """The answer distribution of the ideal predictor with every op, `difference` included, on each context."""
    n = len(a["op_ids"])
    empty = np.zeros((n, 0), np.int64)
    ctx = P.Contexts(a["op_ids"].astype(np.int64), empty, empty, a["query_pair"])
    return P.predictive(TABLE7, ctx, a["posterior"].astype(float))


def tv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(x - y).sum(axis=1) / 2


@memo
def distance_split(arrays: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    """Per replicate run, on the held-out `difference` contexts: the mean total variation distance from the target
    null, clean and edited, split by the colors it falls on; how closely the weight the edited model puts on each other
    op follows the target null from one context to the next; and, as a reference, how far the clean model is from the
    ideal predictor with every op.
    """
    out = []
    for a in arrays.values():
        keep = a["op_ids"] == D
        q = target_null_dist(a)[keep].astype(float)
        gives = gives_table(a["query_pair"])[:, keep]
        others = np.delete(gives, D, axis=0)
        groups = (others.any(0), gives[D] & ~others.any(0), ~gives.any(0))
        row: dict[str, Any] = {"model_seed": int(a["model_seed"]) if "model_seed" in a else None}
        for name in ("p_clean", "p_full"):
            pm = a[name][keep].astype(float)
            half = np.abs(pm - q) / 2
            row[name] = [float((half * g).sum(1).mean()) for g in groups]
        pm = a["p_full"][keep].astype(float)
        row["op_corr"] = float(np.mean([np.corrcoef((pm * o).sum(1), (q * o).sum(1))[0, 1] for o in others]))
        ideal = ideal_dist(a)
        row["ideal_d"] = float(tv(a["p_clean"].astype(float), ideal)[a["op_ids"] == D].mean())
        row["ideal_other"] = float(tv(a["p_clean"].astype(float), ideal)[a["op_ids"] != D].mean())
        out.append(row)
    return out


SPLIT = distance_split(SUPP_ARRAYS)
for _row, _seed in zip(SPLIT, REPLICATE["model_seed"], strict=True):
    _row["model_seed"] = int(_seed)


SPLIT_CLEAN = np.array([r["p_clean"] for r in SPLIT])
SPLIT_FULL = np.array([r["p_full"] for r in SPLIT])
SPLIT_ONLY_D = float((SPLIT_CLEAN[:, 1] / SPLIT_CLEAN.sum(1)).mean())
SPLIT_OTHER_EDITED = float((SPLIT_FULL[:, 0] / SPLIT_FULL.sum(1)).mean())
SPLIT_IDEAL = float(np.mean([r["ideal_d"] for r in SPLIT]))
SPLIT_GAP_SHARE = float(np.mean([r["ideal_d"] / sum(r["p_full"]) for r in SPLIT]))


def split_figure() -> str:
    clean = np.array([r["p_clean"] for r in SPLIT])
    full = np.array([r["p_full"] for r in SPLIT])
    alt = f"""
        A stacked bar chart, two bars per replicate run (model seeds {", ".join(str(r["model_seed"]) for r in SPLIT)}):
        the mean total variation distance from the target null on `{ex.ANCHORED_OP}` contexts, clean and edited,
        split into the part on colors another op gives, on colors only `{ex.ANCHORED_OP}` gives, and on colors no op
        gives. Clean, the distance is about {clean.sum(1).mean():.2f}, of which about {clean[:, 1].mean():.2f} is on
        colors only `{ex.ANCHORED_OP}` gives. Edited, it is about {full.sum(1).mean():.2f}, almost all on colors
        another op gives, with the `{ex.ANCHORED_OP}`-only part near zero; that part is hatched. A dashed line on each
        run marks the distance of the clean model from the ideal predictor on the same contexts, about {np.mean([r["ideal_d"] for r in SPLIT]):.2f}.
    """
    return split_draw(
        {
            "clean": clean.tolist(),
            "full": full.tolist(),
            "seeds": [r["model_seed"] for r in SPLIT],
            "ideal": [r["ideal_d"] for r in SPLIT],
        },
        alt,
    )


@memo
def split_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="e4-split",
        alt_text=alt_text,
        caption=f"""
            **Where the distance from the target null falls.** On held-out `{ex.ANCHORED_OP}` contexts, for the
            replicate runs of `hinge`: the mean total variation distance from the target null, clean and under the
            full edit, split by the answers it falls on. The dashed line is the distance of the clean model from
            the ideal predictor behind the Bayes ceiling, which keeps all seven ops, on the same contexts.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5.0, 2.8), layout="constrained")
        inks = (ink_of("plain"), ink_of("base"), ink_of("grey"))
        xs = []
        for i in range(len(data["seeds"])):
            for j, key in enumerate(("clean", "full")):
                x = i * 2.6 + j
                bottom = 0.0
                for g, (v, ink) in enumerate(zip(data[key][i], inks, strict=True)):
                    # The middle group is hatched, so the three stay apart in greyscale.
                    ax.bar(
                        x,
                        v,
                        bottom=bottom,
                        facecolor=to_rgba(ink, 0.2) if g == 1 else ink,
                        edgecolor=ink,
                        hatch="////" if g == 1 else None,
                        lw=0,
                        width=0.8,
                        label=SPLIT_GROUPS[g].replace("`", "") if i == j == 0 else None,
                    )
                    bottom += v
                xs.append(x)
            ax.plot([i * 2.6 - 0.45, i * 2.6 + 1.45], [data["ideal"][i]] * 2, color=rule_color(), lw=1.0, ls="--")
        ax.set_xticks(xs, ["clean" if k % 2 == 0 else "edited" for k in range(len(xs))], fontsize=7)
        for i, seed in enumerate(data["seeds"]):
            ax.text(
                i * 2.6 + 0.5,
                -0.13,
                f"model seed {seed}",
                ha="center",
                va="top",
                fontsize=7.5,
                transform=ax.get_xaxis_transform(),
            )
        ax.set_ylabel("TV from the target null", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncols=3, handlelength=1.2)
        return fig

    return _plot()


def split_table() -> str:
    rows = [
        [
            f"{r['model_seed']}",
            *(f"{v:.3f}" for v in r["p_clean"]),
            *(f"{v:.3f}" for v in r["p_full"]),
            f"{r['op_corr']:.2f}",
            f"{r['ideal_d']:.3f}",
        ]
        for r in SPLIT
    ]
    return table_html(
        [
            "model seed",
            "clean: other ops",
            "only `difference`",
            "no op",
            "edited: other ops",
            "only `difference`",
            "no op",
            "op weights, r",
            "clean from ideal",
        ],
        rows,
        f"**The distance from the target null by answer**, on held-out `{ex.ANCHORED_OP}` contexts, for the replicate "
        "runs: the parts of the mean total variation distance on colors another op gives, on colors only "
        f"`{ex.ANCHORED_OP}` gives, and on colors no op gives. *Op weights, r*: the correlation over contexts between "
        "the mass the edited model puts on the answers of each other op and the mass the target null puts there, "
        "averaged over the six ops. *Clean from ideal*: the mean distance of the clean model from the ideal predictor "
        "behind the Bayes ceiling, on the same contexts.",
    )


# --- E5: training trajectories -------------------------------------------------------------------------------

TRAJ_GRID = [
    [CONTROL, ex.BASE, "cap-0.9", "cap-0.95", WHOLE],
    ["no-emb", "no-last", "middle", NO_EMB_UNCAPPED, "k-mixed"],
    ["no-emb-matched", "no-last-matched", "middle-matched", None, None],
]


def traj_runs() -> list[dict]:
    """Every run's trajectory at the paired seeds and, for `hinge`, the replicate: the new runs' own, and ex-2.2.21's
    for the reused runs.
    """
    new = [
        {"cond": r["condition"], "seed": r["model_seed"], "replicate": bool(r["replicate"]), "traj": r["traj"]}
        for r in TRAJ_NEW.values()
    ]
    reused = [
        {"cond": r["arm"], "seed": r["model_seed"], "replicate": False, "traj": r["traj"]}
        for r in TRAJ21.values()
        if r["arm"] in ex.REUSED_SEEDS and r["model_seed"] in PAIRED
    ]
    return new + reused


TRAJ = traj_runs()


def traj_series(key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in TRAJ:
        t = r["traj"]
        pts = [(x, y) for x, y in zip(t["epoch"], t[key], strict=True) if y is not None]
        out.setdefault(r["cond"], []).append(
            {"seed": r["seed"], "replicate": r["replicate"], "x": [p[0] for p in pts], "y": [p[1] for p in pts]}
        )
    return out


def traj_figure(key: str, name: str, ylabel: str, caption: str, alt: str) -> str:
    return traj_draw(traj_series(key), name, ylabel, caption, alt)


@memo
def traj_draw(data: dict, name: str, ylabel: str, caption: str, alt_text: str) -> str:
    @themed(name=name, alt_text=alt_text, caption=caption)
    def _plot() -> plt.Figure:
        nr, nc = len(TRAJ_GRID), len(TRAJ_GRID[0])
        fig, axes = plt.subplots(nr, nc, figsize=(8.4, 1.75 * nr + 0.4), layout="constrained", sharex=True, sharey=True)
        grey, hi, repl = ink_of("grey"), ink_of("slow"), ink_of("replicate")
        for i, row in enumerate(TRAJ_GRID):
            for j, cond in enumerate(row):
                ax = axes[i][j]
                if cond is None:
                    ax.set_visible(False)
                    continue
                for r in sorted(data[cond], key=lambda r: (r["replicate"], r["seed"] == SLOW)):
                    ink = repl if r["replicate"] else hi if r["seed"] == SLOW else grey
                    ax.plot(r["x"], r["y"], color=ink, lw=0.9 if ink != grey else 0.7, alpha=0.9)
                ax.set_title(short(cond), fontsize=8)
                ax.set_xlim(0, ex.EPOCHS)
                if i == nr - 1 or TRAJ_GRID[min(i + 1, nr - 1)][j] is None:
                    ax.set_xlabel("epoch", fontsize=7)
                    ax.tick_params(labelbottom=True)
            axes[i][0].set_ylabel(ylabel, fontsize=7)
        handles = [
            plt.Line2D([], [], color=hi, lw=0.9, label=f"model seed {SLOW}"),
            plt.Line2D([], [], color=grey, lw=0.7, label=f"model seeds {PAIRED[0]}, {PAIRED[1]}"),
            plt.Line2D(
                [], [], color=repl, lw=0.9, label=f"replicate ({ex.REPLICATE_SEEDS[0]}–{ex.REPLICATE_SEEDS[-1]})"
            ),
        ]
        fig.legend(handles=handles, loc="outside upper center", ncols=3, frameon=False, fontsize=7)
        return fig

    return _plot()


# --- Decision ------------------------------------------------------------------------------------------------


def task_gate(c: str) -> bool:
    """The hard gate of the decision: a seed-mean shortfall within `TASK_COST_TOL` or within the seed band."""
    short_ = -float(NET[c].mean())
    return short_ <= ex.TASK_COST_TOL or short_ <= TASK_BAND[c]


def decision_table() -> str:
    rows = [
        [
            f"`{short(c)}`",
            bold_if(task_gate(c), f"{NET[c].mean():+.3f}"),
            f"{TASK_BAND[c]:.3f}",
            f"{LANDING[c].mean():.2f}",
            span(LANDING[c], ".2f"),
        ]
        for c in NET_ORDER
    ]
    return table_html(
        ["condition", "net EEM ↑", "seed band", "landing ↑", "seeds"],
        rows,
        "**The task gate and the landing.** Net EEM as in the tables of E1, bold where the candidate clears the "
        f"task gate (a seed-mean shortfall within {ex.TASK_COST_TOL:g} or within the seed band). The landing as in "
        "H2, at the paired seeds.",
    )


def final(cond: str, key: str, seed: int) -> float:
    r = next(r for r in TRAJ if r["cond"] == cond and r["seed"] == seed and not r["replicate"])
    return [y for y in r["traj"][key] if y is not None][-1]


def e4_eem_figure() -> str:
    caption = f"""
        **The task score through training.** EEM on a subsample of the held-out set, one line per run, one panel per
        condition. The run at model seed {SLOW} is highlighted, and on `hinge` the replicate runs are drawn too.
    """
    alt = f"""
        A grid of panels, one per condition, each plotting EEM against the epoch for each run. On `hinge`,
        `cap-0.9`, `cap-0.95`, `no-emb`, and `k-mixed`, the run at model seed {SLOW} ends below the others; on
        whole-line two runs do. Most runs rise to a first plateau and later rise again; the low runs stay near the
        plateau and climb slowly. On the other slice conditions and on the uncapped `no-emb`, the runs end together.
    """
    return traj_figure("eem", "e5-eem", "EEM", caption, alt)


def e4_margin_figure() -> str:
    caption = """
        **The op margin through training**, at the last slice, as in ex-2.2.21; same layout and highlighting as the
        figure above.
    """
    alt = f"""
        The same grid, plotting the op margin at the last slice against the epoch. On most anchored conditions the
        margin rises early and the runs end close together, the run at model seed {SLOW} with them or above. On
        `k-mixed` that run ends lowest, at {final("k-mixed", "m_context", SLOW):.2f}. The control stays near zero.
    """
    return traj_figure("m_context", "e5-margin", "op margin, last slice", caption, alt)


# %%

rf"""
# Ex 2.2.22: Localized by depth, various pull caps, and contexts of varying length

/// tip |
<!-- lede -->
A scout. We retrain the anchored `{ex.ANCHORED_OP}` condition with the pull kept off the embedding or readout, with the hinge cap raised, and on contexts whose numbers of examples vary. None of the three changes improved the recipe: leaving slices out made the edit spill onto other ops, the cap made no steady difference, and varying the number of examples cost a little skill. Pooled over contexts, the edited model answers about as an ideal predictor that had lost `{ex.ANCHORED_OP}` would, though context by context it gets less than half of the way there.
///

Ex-2.2.21 found that an edit applied at every position can take `{ex.ANCHORED_OP}` out of an anchored model gradually while the other ops stay as they were, at least when the pull is capped. This scout trains {len(ex.CONDITIONS)} new conditions at {ex.SEEDS} seeds each, and reuses ex-2.2.21's runs as references, paired by model seed.
"""

# %%

r"""
## Findings

- [Localized by depth (E1)](#localized-by-depth-e1) — every slice restriction lets the edit spill onto other ops past the gate, on nearly every run.
- [Various pull caps (E2)](#various-pull-caps-e2) — the selectivity does not fall off steadily with the cap; runs past the gate turn up at every cap from 0.8 to 0.95.
- [Mixed counts keep the recipe near its ceiling (H1)](#mixed-counts-keep-the-recipe-near-its-ceiling-h1) — **partial**. `k-mixed` falls short of `hinge` by a little more than the tolerance, inside the seed band.
- [The anchor and the posterior (E3)](#the-anchor-and-the-posterior-e3) — at the example answers, α rises with the posterior on every anchored condition, at a fixed answer index too.
- [The edit lands on the target null (H2)](#the-edit-lands-on-the-target-null-h2) — **miss**. The edit moves each replicate run a little under half of the way to the target null.
- [Where the distance from the target null remains (E4, post hoc)](#where-the-distance-from-the-target-null-remains-e4-post-hoc) — the edit removes nearly all the mass on answers only `difference` gives; what remains is on answers other ops give, shared out differently from the target null.
- [Training trajectories (E5, post hoc)](#training-trajectories-e5-post-hoc) — most runs rise to a plateau and then rise again; on several conditions the run at model seed 702 never makes the second rise.
- [Decision](#decision) — keep every slice in the pull and three examples per context, drop the cap, and look into the seeds before confirming it.

/// admonition | How to read this report
The predictions and the criteria for the decision were frozen at commit `94873f2`, before any run of this experiment. Each section opens with what we expected, and the results replace the placeholders in place.
///
"""

# %%

rf"""
## Why this experiment

In the in-context grammar the model never sees the name of an op. It reads a few solved examples, works out which op must be in use, and applies that op to the query. So when we anchor `{ex.ANCHORED_OP}`, we ask the residual stream to lie along one direction, e₁, on contexts whose examples follow `{ex.ANCHORED_OP}`. Ex-2.2.21 found that the anchor settles on the example answers, where the context shows the most about its op, and that an edit removing the e₁ component at every position takes `{ex.ANCHORED_OP}` out. The edited model seemed to answer `{ex.ANCHORED_OP}` contexts much as an ideal predictor would if it no longer knew that op; H2 scores that more formally.

We now consider three changes to our recipe.

First, which slices the pull acts on. A *slice* is one of the depths along the residual stream where we can read the state: the token embedding, then the output of each block, the last of which feeds the readout. The current recipe pulls every slice. A condition that left the embedding slice out had the best task score of any, but its edit spilled onto other ops. An op is *inferred* from the whole context, so it seems likely to live in the middle of the stack, which suggests leaving out the last slice too. But leaving slices out changes the anchor weight in effect as well, since the term averages over the slices it pulls, so each new restriction comes with a condition at the weight that keeps the effective pull as it was.

Second, how hard the pull is on nearly aligned states. The `hinge` condition stops pulling once a state reaches an alignment of 0.8 with e₁, and its edit stayed within the selectivity gate; the uncapped condition *just* missed it. The cap of 0.8 was arbitrary, so two caps between 0.8 and no cap ask whether the selectivity falls off gradually.

Third, how many examples a context has. Recent models trained on three examples per context. We would like to be able to test whether the anchor grades with the evidence, and given a whole context of three examples, the posterior on `{ex.ANCHORED_OP}` takes only a few distinct values. Varying the count from one context to the next gives it more, and training on that lets the model learn that contexts vary in length. That changes the recipe, so it has to show it still gets near its ceiling.
"""

# %%

rf"""
## Parameters

Every new condition changes one setting from the `hinge` condition of ex-2.2.21, our best recipe so far. The base recipe is: the seven-op set; replacement op noise of ρ = {ex.RHO:g} (the chance that an example follows some other op); the newline mask; {ex.EPOCHS} epochs; {ex.MODEL}; and the whole-line label. Like the `hinge` condition, no condition has verification lines (examples whose answer the model judges as right or wrong); ex-2.2.21 found they leave completion unchanged.

{conditions_table()}

**The slice conditions.** The slices are numbered from 0, the embedding, to {ex.N_LAYER}, the output of the last block. `no-emb` leaves out slice 0, `no-last` slice {ex.N_LAYER} (the input to the readout), and `middle` both, so only the three slices between them are pulled. The anchor term and the anti-subspace term both average over the slices they act on, so at a fixed weight, pulling fewer slices pulls each one harder: by a quarter for `no-emb` and `no-last`, and by two thirds for `middle`. Each `-matched` twin scales the weight down by that factor. So the plain condition and its matched twin bracket the two readings of a restriction: a change in where the anchor is, and a change in how strongly each slice is pulled. Leaving a slice out also drops the anti-subspace term there, so the states of other ops at that slice are free to sit on e₁. No condition separates that from the anchor term, so E1 compares the two terms left out together.

**The cap conditions.** The pull capped at 0.9 and at 0.95. With the `hinge` condition at {ex.HINGE_CAP:g} and ex-2.2.21's whole-line condition uncapped, that makes four levels.

**The count condition.** `k-mixed` trains on a corpus where each context draws its number of examples, k, uniformly from {", ".join(map(str, ex.MIXED_COUNTS))}. The mean is three, as in the fixed corpus, so an epoch has about as many tokens and steps. The held-out set has every count. The figure below shows the posterior on `{ex.ANCHORED_OP}` given a whole context, at each count and pooled.

{counts_figure()}

Each count gives a few steps of its own. One example leaves most contexts in some doubt, and puts about a quarter near zero, where the one example fits another op better. Two examples is the least decisive count: about half the contexts sit near 0.4, where the two examples fit some other op about as well. From three examples on, more and more contexts are near-certain. Pooled, the curve has the steps of every count, so it rises in many small steps. The share in the "middle band" (a posterior from {MID_LO:g} to {MID_HI:g}, where the examples favour `{ex.ANCHORED_OP}` and still leave some doubt) is lower pooled than at three examples ({MID_MIXED:.0%} against {MID_FIXED:.0%}), so what the mix adds is more distinct levels.

Drawing the counts uniformly keeps the mean at three, though, as above, it puts fewer contexts in the middle band than three examples alone. One example contributes the most contexts in doubt, and pooled over the counts, the share of `{ex.ANCHORED_OP}` contexts whose examples favour some other op is about what it is at three examples, so the short contexts do not add much ambiguity to the label overall. H1 reports the skill at each count, which would show whether the short contexts are too hard to learn from.

**The seeds.** The model seeds of ex-2.2.21's three-seed conditions, {ex.SEED_OFFSET} to {ex.SEED_OFFSET + ex.SEEDS - 1}, so every comparison is paired by seed. In ex-2.2.21 the last of these took a slow path through training on most anchored conditions, though not on `no-emb`, so a setting can change it; the pairing shows whether any here do.

**The replicate.** Three more runs of the `hinge` condition, at seeds {", ".join(map(str, ex.REPLICATE_SEEDS))}, which it was not trained at in ex-2.2.21. H2 is scored on these (that section says why). With them, {ex.N_RUNS} runs in all.
"""

# %%

rf"""
## Measurements

The measurements are the same as in ex-2.2.21, with two additions.

**Task score.** Expected exact match (EEM) on held-out contexts: the probability the model puts on the right answer under the true op. Each run is compared with *the control*, the unanchored condition of ex-2.2.21, at the same seed. For `k-mixed` it is also reported at each count, beside the Bayes ceiling at that count (the score of an ideal predictor that weighs every op by how well it fits the examples).

**Where the anchor sits.** The alignment α of the state with e₁ (their cosine) by role and slice, on `{ex.ANCHORED_OP}` contexts and on the others, and the op margin: how far `{ex.ANCHORED_OP}` contexts sit along e₁ beyond the rest. Ex-2.2.21 measured the margin at the last slice, which `no-last` and `middle` leave unpulled, so here it is reported at every slice, for every condition alike. We read the slices one at a time, with no summary over them, since the two ends differ from the rest: slice 0 sees no context, having no block before it, and slice {ex.N_LAYER} feeds only the readout at its own position, so no state there can pass the op on to the query answer. The anchor term asks for the margin, so it checks that the pull landed and tests nothing.

**The edit.** The projection at every position, at doses γ = {", ".join(f"{g:g}" for g in ex.DOSE_GAMMAS)} (the share of the e₁ component removed). Ex-2.2.21 (E2) set two criteria: the drop in EEM on `{ex.ANCHORED_OP}` grows with the dose and reaches at least {ex.GRADING_MIN_DAMAGE:.0%} of the way to the target null at full dose, and no other op drops by more than {ex.SELECTIVITY_GATE:g} in EEM at any dose. Both drops are net of the drop the control shows under the same edit. Beside the two criteria, the *selective reach*: how far toward the target null the strongest dose that stays within the gate goes.

**Landing (new).** The *target null* is the answer distribution of an ideal predictor that has lost `{ex.ANCHORED_OP}` and nothing else: it weighs the other ops by how well they fit the examples. For each held-out `{ex.ANCHORED_OP}` context, we measure how far the answer distribution of the model is from the target null, on the clean model and under the full edit, by their *total variation distance*: the share of probability that would have to move to turn one distribution into the other, from 0 when they match to 1 when they share no answers. The landing is the share of the clean distance that the edit closes, taken as a ratio of means over contexts: one less the mean edited distance over the mean clean distance. A mean of per-context shares would be dominated by the contexts the clean model already answers like the target null.

Beside it we report the KL divergence of the target null from the model, KL(target null ‖ model).[^kl] It has no upper bound, so a few contexts can dominate its mean, which is why the landing uses the total variation.

[^kl]: KL(target null ‖ model) is read "the KL divergence of the target null from the model", i.e. averaged over the answers the target null gives, how much less likely the model finds them, on a log scale. KL divergence measures how much one probability distribution differs from another; it is zero when they match, and it is not symmetric, so the order matters. Only this order is finite: the target null puts no weight on colors that no remaining op gives, and the model puts some weight on every color. A context where the model gives a likely answer almost no weight has a very large divergence.

**Alignment against the evidence (new).** Whether the anchor grades with the evidence: the measurement of (b) in the [D2.2 design](/docs/m2/d2.2/design.md) and of the [grading item](/todo/science/anchor-grades-with-the-posterior.md) on the backlog. (The edit criterion above uses "grades" in another sense: the drop grows with the dose.) At each answer in a `{ex.ANCHORED_OP}` context, α against the posterior on `{ex.ANCHORED_OP}` given the pairs up to and including that answer (an answer is one color token). That covers the example answers, where ex-2.2.21 found the anchor, and the query answer, where the state already holds the answer, so its posterior counts the query pair too. Every context gives one point per answer, so even the fixed-count corpus has several levels of evidence; `k-mixed` also contributes the levels of the other counts.
"""

# %%

rf"""
## Localized by depth (E1)

We look at the slice conditions beside the baseline (`hinge`) condition, on four measures: the task score net of the control, the op margin at each slice, whether the edit meets the two criteria (the drop on `difference` grades with the dose, and the other ops stay within the gate), and the selective reach. The question is whether a restriction raises the task score, as leaving out the embedding did on the uncapped pull in ex-2.2.21, while editing as selectively as the `hinge` condition, and whether the matched weight changes the answer. Ex-2.2.21's uncapped `no-emb` condition is shown beside its capped twin.

The figure below shows the task score for every condition, so E2 and H1 refer back to it.

{net_figure()}

**What we saw.** The task score varies more from seed to seed than from condition to condition. At model seed {PAIRED[0]} every anchored condition with three examples scores a little above the control, and at {PAIRED[1]} most score near it. At {SLOW} the {", ".join(f"`{c}`" for c in SLOW_PATH[:-1])}, and `{SLOW_PATH[-1]}` conditions end well below the control, and the others end near it (on `whole-line`, so does the run at {PAIRED[1]}). So the seed means mostly say whether the run at model seed {SLOW} took the slow path it took in ex-2.2.21 (E5 follows it through training). Of the slice sets, only `no-emb` at the plain weight put that run on the slow path; on every other slice set it trained as quickly as the runs at the other seeds.

The edit is next: the drop on `{ex.ANCHORED_OP}` and on the worst other op as the dose grows.

{e1_edit_figure()}

On the `hinge` condition the edit grades and stays within the gate. Every restriction lets the edit spill onto other ops past the gate, at both weights. The spill is largest on the slice sets that leave out the last slice at the plain weight and on `middle-matched`, and smallest on `no-emb`. The matched weight lowers it when the last slice is left out alone and raises it on the other two slice sets. Three conditions miss the grading criterion (`no-emb`, `no-emb-matched`, and `middle-matched`), each because the drop dips between the two strongest doses by a little more than the tolerance of {ex.ex2221.GRADE_DIP:g}, while still reaching half the way to the target null.

Taken run by run, `hinge` is less tidy than its seed mean, since one of its three runs spills past the gate (E2 has more). But the restrictions spill more often: only {NUM_WORDS[sum(r["selective"] for r in SLICE_RUNS)]} of their {len(SLICE_RUNS)} runs stays within the gate.

{criteria_table([ex.BASE, *(c for pair in SLICE_GROUPS.values() for c in pair), NO_EMB_UNCAPPED], "**The edit criteria by slice set.** " + CRITERIA_NOTE)}

The op margin (below) is about as large as on `hinge` at the middle slices. Where the last slice is left unpulled, the margin there is lower than on `hinge`. At the embedding it is higher on most restricted conditions, whether that slice is pulled or not.

{margin_table([ex.BASE, *(c for pair in SLICE_GROUPS.values() for c in pair), NO_EMB_UNCAPPED, CONTROL], "**The op margin at each slice**, seed means. Italics mark slices the condition does not pull.")}

The alignment at the example answers shows the states of the other ops as well as of `{ex.ANCHORED_OP}`.

{e1_align_figure()}

On `{ex.ANCHORED_OP}` contexts the alignment rises through the stack on every slice set, and sooner on the restricted conditions than on `hinge`. So do the states of other ops: at the first two slices they sit further along e₁ than on `hinge`, most of all on the slice sets that leave out the embedding. `no-last-matched` is the exception, close to `hinge` throughout.
"""

# %%

rf"""
## Various pull caps (E2)

The same four measures over the four caps: 0.8 (the `hinge` condition), 0.9, 0.95, and no cap (ex-2.2.21's whole-line condition). If the selectivity falls off gradually with the cap, a cap near where it crosses the gate gives the most anchor that still edits cleanly. If it drops at one cap, the step is where to stop. At three seeds a gradual fall smaller than the seed range would look flat, and the uncapped condition only just missed the gate in ex-2.2.21, so a flat result would say the cap matters less than the seeds vary.

{e2_figure()}

**What we saw.** The selectivity does not fall off steadily with the cap. On the seed means, the caps of 0.8 and 0.9 and the uncapped condition stay within the gate, and the cap of 0.95 is past it. That comes from one run: at model seed {PAIRED[0]} the edit on `cap-0.95` takes another op down by many times the gate, and reaches almost none of the way to the target null within it. The `hinge` run at the same seed also spills past the gate on its own, though the seed mean stays within it. The uncapped condition, which spilled just past the gate over five seeds in ex-2.2.21, stays within it at these three. Taken run by run, the order changes: two of the three `cap-0.9` runs go a little past the gate, and every uncapped run stays within it. So the cap does not put the spills in order: runs past the gate turn up at 0.8, 0.9, and 0.95, and the uncapped condition, within the gate at all three of these seeds, went past it on its seed mean over five seeds in ex-2.2.21.

Past the embedding, the op margin is a little higher at the caps of 0.9 and 0.95 than at 0.8, and the uncapped condition sits just below 0.95. The task score follows `hinge` at every cap, with the run at model seed {SLOW} on the slow path (E1), and on the uncapped condition the run at {PAIRED[1]} as well.

{criteria_table(list(CAP_LADDER.values()), "**The edit criteria by cap**, from 0.8 to no cap. " + CRITERIA_NOTE)}

**The replicate (post hoc).** The three replicate runs of `hinge` were trained for H2, and the edit criteria can be scored on them too. On their seed means the edit misses both criteria. It spills past the gate, and again the spill comes from one run: at model seed {int(REPLICATE["model_seed"][0])} the edit takes `darken` down by many times the gate. It also misses grading, on every replicate run, in the same way as the slice conditions that miss it: the drop on `{ex.ANCHORED_OP}` reaches nearly its full size by half the dose, then dips at the stronger doses by a little more than the tolerance. The table below sets the six `hinge` runs side by side. The edit stays within the gate on {NUM_WORDS[HINGE_WITHIN]} of them.

{hinge_edit_table()}
"""

# %%

rf"""
## Mixed counts keep the recipe near its ceiling (H1)

**What we expect.** On ex-2.2.21's held-out set, which has three examples per context, we expect the `k-mixed` condition to fall short of the `hinge` condition by less than {ex.REGRESSION_TOL:g} in EEM, paired by seed: a pass. A shortfall above {ex.REGRESSION_TOL:g} but inside the seed band would be a partial pass, since three seeds could not tell it from noise, and a larger one a miss. A gain would also be a pass. If the seeds disagree about the direction by more than the tolerance each way, the result would be outside the plan, and the verdict would be Unresolved.

At each example count we also report the skill of `k-mixed`, the share of the way from the floor (a predictor that ignores the examples) to the ceiling, with no gate. A count where the skill falls well below the others would say the model has not learned that length, which matters for the alignment against the evidence (E3).

{h1_figure()}

**What we saw.** `k-mixed` falls short of `hinge` at every seed, by {min(H1["by_seed"]):.2f} to {max(H1["by_seed"]):.2f} in EEM, and it is below the control at every seed too (E1). The mean shortfall is above the tolerance and just inside the seed band. The band is wide mostly because both conditions score lower at model seed {SLOW}, on the slow path described in E1.

{h1_table()}

The skill of `k-mixed` rises steadily with the number of examples, so no count stands out as one the model failed to learn. `hinge` and the control, trained on three examples alone, score higher than `k-mixed` at three examples and lower with one or two. With four or five, the control stays higher and `hinge` falls below. That fall comes from one run: at model seed {PAIRED[LONG_DROP]} the skill of `hinge` drops to {SKILL[ex.BASE][LONG_DROP, -2]:.2f} at four examples and to {SKILL[ex.BASE][LONG_DROP, -1]:.2f} at five (the floor is 0), while its other two runs lose much less. That run is not the slow one, so it seems that how well a model trained on three examples copes with longer contexts varies from seed to seed.

{skill_figure()}

/// admonition | Partial
`k-mixed` falls short of `hinge` by {H1["short"]:.3f} in EEM, above the tolerance of {ex.REGRESSION_TOL:g} and inside the seed band of {H1["band"]:.3f}.
///
"""

# %%

rf"""
## The anchor and the posterior (E3)

Whether α at the answers rises with the posterior on `difference`. The label is the same for every `difference` context however well its examples fit, so if α grades anyway, the anchor follows what the model has inferred and not just the label. Evidence builds up along a context, so the posterior also rises with the position of an answer; to tell the two apart, we also compare contexts at the same answer index, where the posterior still differs from one context to the next. We look at it on `k-mixed`, and on the `hinge` and whole-line conditions of ex-2.2.21 at three examples. This is a first look that would shape a later prediction, so it has no gate.

First, how the answers spread over the posterior. Each point is one answer of a held-out `{ex.ANCHORED_OP}` context.

{e3_points_figure()}

**What we saw.** About half the answers sit above 0.9, and the rest fall on a few steps lower down. The two corpora give nearly the same spread, so for this measurement mixing the counts added little. The query answers sit almost all in the top bin, since the query pair is counted too, so the curves below are most informative at the example answers.

{e3_figure()}

Past the embedding, α at the example answers rises with the posterior on every anchored condition, and the control stays flat near zero. The rise is uneven: the bin from 0.6 to 0.8 sits above the top bin. Most of the points in that bin are answers of the first example, and the next figure suggests why that matters. `k-mixed` sits lower than the two three-example conditions in most bins past the embedding, and at the embedding itself.

The posterior also rises along a context, so the next figure plots each answer index separately.

{e3_index_figure()}

At a fixed index, α rises with the posterior at every example answer. At the same posterior, it sits higher at earlier answers: at the first answer, α in the bin from 0.6 to 0.8 is about as high as it gets anywhere. So the bump in the pooled figure above seems to come from mixing answer indices, with the first answers crowded into that one bin. The first answer has few posterior levels: about two thirds of its points sit at 0.69 and most of the rest at 0.05, so its two middle points rest on a few dozen contexts each, and its line rises in steps.
"""

# %%

rf"""
## The edit lands on the target null (H2)

**What we expect.** In ex-2.2.21, after the fact, the edited `hinge` condition answered `{ex.ANCHORED_OP}` contexts about as the target null does. If that holds, we would not need to train a designed fallback, since the anchor alone gives the edit a predictable destination. The prediction came from looking at ex-2.2.21's `hinge` runs, so scoring those same runs would show little. We score it on the replicate, three `hinge` runs at fresh seeds, and report the stored `hinge` runs and every other condition beside it.

At full dose, with the edit at every position, on held-out `{ex.ANCHORED_OP}` contexts, we expect the edit to close at least {ex.LANDING_FRACTION:.0%} of the distance from the target null that the clean model has, in total variation, as a ratio of means over contexts. That is the same share as the edit criterion of ex-2.2.21, which asks for half the way to the target null in EEM, and on the same scale: both run from the clean model (0) to the target null (1). The landing asks for more, since mass that leaves the `{ex.ANCHORED_OP}` answer counts toward it only where it arrives on answers the target null gives. Closing that share in every seed of the replicate would be a pass, and falling short in every seed a miss. If some seeds pass and others fall short, the result would be outside the plan, and the verdict would be Unresolved.

{h2_figure()}

**What we saw.** The edit moves every replicate run a little under half of the way to the target null, and none reaches the gate. The stored `hinge` runs land in the same place, and the runs of the other anchored conditions land near them, apart from `k-mixed` at model seed {SLOW}; the control hardly moves. The KL divergence falls further, to about a quarter of its clean value. KL grows large wherever the model puts next to nothing on an answer the target null gives. The clean model, sure of `{ex.ANCHORED_OP}`, does that on many answers, which is why its KL is large. So it seems that under the edit far fewer such answers remain.

{h2_table()}

The confusion matrices show where the answers go, at three levels of how sure the context is about its op.

{confusion_figure()}

In the `{ex.ANCHORED_OP}` row, the edited model puts about as much mass on the answers of each other op as the target null does, in every bin. The mass left on the answers of `{ex.ANCHORED_OP}` itself, on the diagonal, falls close to the little the target null puts there. So, pooled over contexts, the edit puts the mass about where the target null does.

The matrices and the landing measure different things. A matrix adds up the mass on the answers of each op over all the contexts in a bin, so a model that puts too much on one op in some contexts and too little in others can still match the target null in total. The landing measures the distance on each context and then averages, so those mismatches add up instead of cancelling. E4 looks at where they fall.

In the rows of the other ops, the edit moves a little mass off the true op and onto the others in the two surer bins, where the target null puts almost none. That loss is in the EEM of the other ops, and E2 scores it net of the control. In the two surer bins, the largest entries off the diagonal are between `sat-hsv` and `value-hsv`, which the clean model already confuses with each other at about the same level; the edit adds a little to them.

/// admonition | Miss
The edit closes {", ".join(f"{v:.0%}" for v in REPLICATE["landing"])} of the distance from the target null on the three replicate runs, short of {ex.LANDING_FRACTION:.0%} in each.
///
"""

# %%

rf"""
## Where the distance from the target null remains (E4, post hoc)

H2 scores the distance from the target null context by context, and the confusion matrices, which pool over contexts, make the edited model look closer to the target null than the landing says. So here we split the distance by the answers it falls on: answers another op could give, answers only `{ex.ANCHORED_OP}` gives, and colors no op gives.

{split_figure()}

**What we saw.** On the clean model, about {SPLIT_ONLY_D:.0%} of the distance is mass on answers only `{ex.ANCHORED_OP}` gives. The edit removes nearly all of it and moves little onto colors no op gives. What remains, about {SPLIT_OTHER_EDITED:.0%} of the edited distance, is on answers another op could give: the edited model puts its mass on the right kind of answer and shares it among those answers differently from the target null.

The sharing partially follows the examples. The mass the edited model puts on the answers of each other op rises and falls with the mass the target null puts there, though loosely (the correlations over contexts are in the table below).

For scale, the clean model is some way from an ideal predictor too. On the same contexts it is about {SPLIT_IDEAL:.2f} from the ideal predictor behind the Bayes ceiling (which weighs all seven ops by how well they fit the examples), in the same measure, which is about {SPLIT_GAP_SHARE:.0%} of the distance that remains under the edit.

{split_table()}
"""

# %%

rf"""
## Training trajectories (E5, post hoc)

The run at model seed {SLOW} ends low on several conditions (E1), as it did in ex-2.2.21. To see when it parts from the others, the figure below follows the task score through training for every run, on a subsample of the held-out set, with the run at that seed highlighted.

{e4_eem_figure()}

**What we saw.** Most runs rise quickly to a first plateau, stay on it for some tens of epochs, and then rise again, at a time that varies from run to run. On the conditions where the run at model seed {SLOW} ends low, that run never makes the second rise and climbs slowly from the plateau for the rest of training. On whole-line the run at model seed {PAIRED[1]} does the same, and on the control one run makes the second rise late. On the restricted conditions, other than `no-emb`, every run makes the second rise and the three end together, as do the runs of the uncapped `no-emb`. All three replicate runs of `hinge` make it, one ending a little lower than the other two.

The op margin at the last slice, below, shows whether the slow runs also hold the anchor less firmly.

{e4_margin_figure()}

The margin shows no matching split: on most conditions the slow run ends with the others or above them. `k-mixed` is the exception, where the run at model seed {SLOW} also ends with the lowest margin.
"""

# %%

rf"""
## Decision

Three choices for the recipe: which slices the pull acts on, and at what weight; the cap; and whether the corpus mixes example counts. The tables below score every candidate (the conditions of this scout and the ex-2.2.21 conditions they pair with) on the criteria frozen with the plan, and the choice follows them.

{criteria_table(NET_ORDER, "**The edit criteria for every candidate.** " + CRITERIA_NOTE)}

{margin_table([*NET_ORDER, CONTROL], "**The op margin at each slice for every candidate**, seed means. Italics mark slices the condition does not pull.")}

{decision_table()}

**What we chose.** For the recipe, one of the three choices changes:

- Slices: every slice stays in the pull, at the plain weight. Every restriction let the edit spill past the gate more often than `hinge` did (E1), at either weight. Which slices the pull needs is still open; pooling over slices, so that training can choose where to hold the op, is the next idea to try.
- Cap: none. Runs past the gate turn up at every cap from 0.8 to 0.95, and `hinge` stays within the gate on {NUM_WORDS[HINGE_WITHIN]} of its six runs (E2). So at three seeds the cap seems not to buy the selectivity it was meant to, and leaving it out removes a setting.
- Counts: three examples per context. `k-mixed` falls a little short of `hinge` (H1), its α sits lower (E3), and three examples already give α enough spread to grade at a fixed answer index (E3).

Before confirming these at fresh seeds, we would like to understand the seeds better. One seed takes a slow path on most anchored conditions (E5), and along the caps the spills past the gate come from single runs, all of them runs that trained quickly.
"""

# %%

rf"""
## Discussion

It seems the recipe of ex-2.2.21 was already close to the best these three changes allow, and what most limits the next step is how much the runs vary from seed to seed.

Leaving slices out of the pull was meant to put the anchor where an inferred op lives. Instead, every restriction let the edit spill onto other ops. At the first slices the states of other ops sat further along e₁ than on `hinge`, most of all where the embedding was left out (E1). Leaving a slice out of the pull also leaves out the term that pushes other ops off e₁ there, so those states may simply have been free to drift onto the axis. Keeping that term on every slice while restricting the pull would tell the two apart.

The cap did less than we hoped. Ex-2.2.21 suggested that a cap keeps the edit selective, but with the runs taken one at a time, spills turn up at every cap, and at these seeds not on the uncapped condition. Along the caps, the spills come from single runs, and only from runs that trained quickly. A run on the slow path may not yet have reached the stage of training where spilling happens, which is part of why the seeds come first.

The edited model looks better pooled over contexts than one context at a time. Counted by op, the mass leaves `{ex.ANCHORED_OP}` and lands on the other ops about where the target null puts it (H2). Context by context, the edit gets a little under half of the way. Nearly all of what remains is on answers another op could give, shared among them differently from the target null, and part of it is the gap the clean model already has from an ideal predictor (E4). So "behaves like the target null, pooled over contexts" seems a fair description of the edited model. We don't yet know how close a model trained without ever seeing a `{ex.ANCHORED_OP}` context would come to the target null, which would say how much of the remaining distance any model would leave.

The anchor follows the evidence. At the example answers α rises with the posterior on every anchored condition, and at a fixed answer index too (E3). At the same posterior, α is higher at earlier answers, which we did not expect and can't yet explain.
"""

# %%

rf"""
## Method

**The corpus.** The fixed-count conditions train on ex-2.2.21's corpus of three-example contexts. `k-mixed` trains on a corpus of the same number of contexts, built by the same generator, with the number of examples drawn uniformly per context from {", ".join(map(str, ex.MIXED_COUNTS))}. Two whole contexts at five examples take 76 tokens, inside the window (block size) of 96. The held-out set for `k-mixed` has {ex.ex2221.HOLDOUT_CONTEXTS:,} contexts per op at each count. Every condition is also scored on ex-2.2.21's held-out set, so `k-mixed` compares with its reference on the same contexts.

**The edit and the landing.** The suppression pass is ex-2.2.21's, at every position only. The landing measurements compare the answer distribution of the edited model over the color vocabulary at the query `=` with the target null from the posterior over the other ops (`sca.data.incontext.target_null`).

**Budget.** Planned at under \$5: {ex.N_RUNS} runs at about \$0.13 each on an L4 (the cost of ex-2.2.21's runs), with the scoring passes. The experiment cost \$6.69 in all.
"""
