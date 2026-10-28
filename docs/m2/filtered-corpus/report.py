# ruff: noqa: B018
# title: A model trained without difference

# An exploratory study: the control trained with no loss on the `difference` contexts (`sans-diff`), at seeds 712 to
# 719, against the edited `τ_ℓ fixed` runs and the controls at the same seeds. `experiment.py` beside this script
# trains and scores the runs; the reference runs come from the fixed and anneal confirmations.

import itertools
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import nnls

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

# --- The stored results ---------------------------------------------------------------------------------------

ac, fc = ex.ac, ex.fc
SEEDS: list[int] = list(ex.SEEDS)
DROPPED_SEEDS: list[int] = list(ex.DROPPED_SEEDS)
SANS, DROPPED, FIXED, CONTROL = "sans-diff", "dropped", "fixed", "control"
KEEP_DOSE = ("op_ids", "posterior", "query_pair", "clean_eem", "null_eem", "eem", "p_clean", "p_full")


def dose_ref(kind: str, s: int) -> str:
    return {
        SANS: lambda: ex.DOSE_ARRAYS_REF.format(label=ex.label_of(s)),
        DROPPED: lambda: ex.DOSE_ARRAYS_REF.format(label=ex.label_of(s, ex.DROPPED)),
        FIXED: lambda: fc.DOSE_ARRAYS_REF.format(label=fc.label_of(s, None)),
        CONTROL: lambda: ac.DOSE_ARRAYS_REF.format(label=ac.label_of(s, None, "control")),
    }[kind]()


def fetch_all() -> tuple[dict, dict, dict, dict]:
    """The dose arrays of every run (`(kind, seed)` → arrays), the trajectories of the new runs and of the reference
    runs (label → trajectory), and the scoring along training.
    """
    store = project_store()
    keys = [(k, s) for k in (SANS, FIXED, CONTROL) for s in SEEDS] + [(DROPPED, s) for s in DROPPED_SEEDS]
    refs = [dose_ref(*k) for k in keys]
    json_refs = [ex.TRAJ_REF, ex.ALONG_REF, ac.FINISHED_TRAJ_REF, fc.FINISHED_TRAJ_REF]
    got = store.get_refs(refs + json_refs)
    missing = [r for r, a in got.items() if a is None]
    assert not missing, f"not published: {missing}"
    with tempfile.TemporaryDirectory() as tmp:
        into = Path(tmp)
        paths = store.get_many([(got[r], into / f"{i}.bin") for i, r in enumerate(refs + json_refs)])
        dose = {}
        for k, path in zip(keys, paths, strict=False):
            with np.load(path) as z:
                dose[k] = {n: z[n] for n in KEEP_DOSE}
        traj, along, ac_traj, fc_traj = (json.loads(Path(path).read_text()) for path in paths[len(keys) :])
    return dose, traj | ac_traj | fc_traj, along, {}


DOSE, TRAJ, ALONG_BODY, _ = fetch_all()
OPS: tuple[str, ...] = tuple(ALONG_BODY["design"]["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
STATS = ALONG_BODY["stats"]
d0 = DOSE[(SANS, SEEDS[0])]
for _k, _v in DOSE.items():
    assert (_v["op_ids"] == d0["op_ids"]).all() and (_v["query_pair"] == d0["query_pair"]).all(), _k
OP_IDS = d0["op_ids"].astype(np.int64)
IS_D = OP_IDS == D
PAIRS = d0["query_pair"].astype(np.int64)
P = ex.ex2221.ex2216._get_posterior()
TABLE = P.build_table(ex.ex2222.ex2218.table_of(OPS))


def contexts_of(op_ids: np.ndarray, pairs: np.ndarray) -> Any:
    empty = np.zeros((len(op_ids), 0), np.int64)
    return P.Contexts(op_ids, empty, empty, pairs)


@memo
def bayes(posterior: np.ndarray, op_ids: np.ndarray, pairs: np.ndarray) -> dict[str, np.ndarray]:
    """The Bayes predictor and the target null on every held-out context, as edit-effects takes them."""
    ctx = contexts_of(op_ids, pairs)
    post = posterior.astype(np.float64)
    null = ex.ex2221.ex2216._target_null(P, TABLE, ctx, post, D)
    return {
        "q_ideal": P.predictive(TABLE, ctx, post),
        "q_null": P.predictive(TABLE, ctx, null),
        "null_post": null,
        "ideal_eem": P.expected_match(TABLE, ctx, post),
        "null_eem": P.expected_match(TABLE, ctx, null),
    }


BAYES = bayes(d0["posterior"], OP_IDS, PAIRS)
assert np.allclose(BAYES["null_eem"], d0["null_eem"], atol=2e-4)
QN, QI = BAYES["q_null"], BAYES["q_ideal"]
POST_D = d0["posterior"][:, D].astype(np.float64)


def tv(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Total variation between a palette distribution *q* and a model distribution *p*, with the mass *p* puts off
    the palette counted as distance.
    """
    return 0.5 * (np.abs(q - p).sum(-1) + np.clip(1.0 - p.sum(-1), 0, None))


def between(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Total variation between two model distributions, each with its mass off the palette left as it is."""
    return 0.5 * np.abs(p - q).sum(-1)


def p_of(kind: str, s: int) -> np.ndarray:
    """The answer distribution a run is scored with on `difference` contexts: the full edit for `τ_ℓ fixed`, else
    the clean pass.
    """
    return DOSE[(kind, s)]["p_full" if kind == FIXED else "p_clean"].astype(np.float64)


def clean_of(kind: str, s: int) -> np.ndarray:
    return DOSE[(kind, s)]["p_clean"].astype(np.float64)


def num(v: float, spec: str = ".2f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


def span(values, spec: str = ".2f") -> str:
    v = np.asarray(list(values), float)
    return f"{num(float(v.mean()), spec)} ({num(float(v.min()), spec)} to {num(float(v.max()), spec)})"


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


RULE = light_dark("#333", "#ddd")
ZERO = light_dark("#aaa", "#555")
GHOST = light_dark("#bbb", "#555")
INK = {
    CONTROL: light_dark("#666", "#bbb"),
    FIXED: light_dark("#08306b", "#c6dbef"),
    SANS: light_dark("#b35806", "#fdb863"),
    DROPPED: light_dark("#8e6bb0", "#c4a6e3"),
}
MARKER = {CONTROL: "o", FIXED: "s", SANS: "D", DROPPED: "^"}
SHOWN = {CONTROL: "control", FIXED: "τ_ℓ fixed, edited", SANS: "sans-diff", DROPPED: "dropped"}


def tex(shown: str) -> str:
    return shown.replace("τ_ℓ", "$\\tau_\\ell$")


def dots(ax, x: float, v, color, m: str, jit) -> None:
    """The seeds as small faded dots over a thin bar for their range, and the seed mean as the large mark."""
    v = np.asarray(v, float)
    ax.plot([x, x], [v.min(), v.max()], "-", color=color, lw=1.0, alpha=0.5, zorder=2, solid_capstyle="butt")
    ax.plot(x + jit[: len(v)], v, "o", ms=2.2, color=color, alpha=0.5, zorder=3, mew=0)
    ax.plot(x, v.mean(), m, ms=5.5, color=color, zorder=4, mec=light_dark("white", "#111"), mew=0.6)


# --- E1: along training ----------------------------------------------------------------------------------------

TRAJ_EPOCHS = np.asarray(TRAJ[ex.label_of(SEEDS[0])]["traj"]["epoch"])
TRAJ_LABEL = {
    SANS: ex.label_of,
    CONTROL: lambda s: ac.label_of(s, None, "control"),
    FIXED: lambda s: fc.label_of(s, None),
}


def task_along(kind: str, s: int) -> np.ndarray:
    """(records, ops): the task score of each op along training, on the trajectory subset of held-out contexts."""
    t = TRAJ[TRAJ_LABEL[kind](s)]["traj"]
    assert np.allclose(np.asarray(t["epoch"]), TRAJ_EPOCHS)
    return np.asarray(t["eem_per_op"], float)


TASK_ALONG = {(k, s): task_along(k, s) for k in TRAJ_LABEL for s in SEEDS}
_ALONG = {(r["kind"], r["model_seed"], r["epoch"]): r for r in ALONG_BODY["runs"]}
ALONG_EPOCHS = sorted({r["epoch"] for r in ALONG_BODY["runs"]})


def along_of(kind: str, mode: str, ops: list[int]) -> dict[int, np.ndarray]:
    """Per seed, the distance from the target null along training, the mean over *ops*, on the clean pass or under
    the full edit (*mode*).
    """
    return {
        s: np.array([np.mean([_ALONG[(kind, s, e)][mode]["tv_null"][o] for o in ops]) for e in ALONG_EPOCHS])
        for s in SEEDS
    }


ALONG_SANS_D = along_of(SANS, "clean", [D])
ALONG_SANS_O = along_of(SANS, "clean", OTHER)
ALONG_FIXED_D = along_of(FIXED, "full", [D])
SETTLE = {k: np.mean(list(v.values()), axis=0) for k, v in (("sans", ALONG_SANS_D), ("fixed", ALONG_FIXED_D))}
GAP_ALONG = SETTLE["fixed"] - SETTLE["sans"]
OTHER_GAP_ALONG = np.mean([TASK_ALONG[(SANS, s)][:, OTHER].mean(1) - TASK_ALONG[(CONTROL, s)][:, OTHER].mean(1) for s in SEEDS], axis=0)  # fmt: skip

# --- E2: how close it comes ------------------------------------------------------------------------------------


def to_null(kind: str, s: int, on: np.ndarray = IS_D) -> float:
    return float(tv(QN, p_of(kind, s))[on].mean())


TO_NULL = {k: [to_null(k, s) for s in SEEDS] for k in (SANS, FIXED, CONTROL)}
TO_NULL[DROPPED] = [to_null(DROPPED, s) for s in DROPPED_SEEDS]
FIXED_CLEAN_TO_NULL = [float(tv(QN, clean_of(FIXED, s))[IS_D].mean()) for s in SEEDS]
BAYES_GAP_D = [float(tv(QI, clean_of(CONTROL, s))[IS_D].mean()) for s in SEEDS]
BAYES_GAP_O = [float(tv(QI, clean_of(CONTROL, s))[~IS_D].mean()) for s in SEEDS]
OWN_GAP_O = [float(tv(QN, clean_of(SANS, s))[~IS_D].mean()) for s in SEEDS]
SHARE = {k: [1 - a / c for a, c in zip(TO_NULL[k], TO_NULL[CONTROL], strict=True)] for k in (SANS, FIXED)}
SHARE_FIXED_OWN = [1 - a / c for a, c in zip(TO_NULL[FIXED], FIXED_CLEAN_TO_NULL, strict=True)]
RATIO_OWN = [a / b for a, b in zip(TO_NULL[SANS], OWN_GAP_O, strict=True)]

POST_BINS = (0.0, 0.2, 0.5, 0.8, 0.95, 1.0)
BIN_LABEL = ["< 0.2", "0.2–0.5", "0.5–0.8", "0.8–0.95", "≥ 0.95"]
BIN = np.clip(np.digitize(POST_D, POST_BINS[1:-1]), 0, len(POST_BINS) - 2)
N_BIN_D = [int(((BIN == b) & IS_D).sum()) for b in range(len(BIN_LABEL))]


def by_bin(kind: str, s: int) -> np.ndarray:
    t = tv(QN, p_of(kind, s))
    return np.array([t[(BIN == b) & IS_D].mean() for b in range(len(BIN_LABEL))])


BY_BIN = {k: np.array([by_bin(k, s) for s in SEEDS]) for k in (SANS, FIXED, CONTROL)}

# --- E3: whether it misses in the same way --------------------------------------------------------------------

PAIRWISE = list(itertools.combinations(SEEDS, 2))
BETWEEN = {
    "sans-fixed": [float(between(p_of(SANS, s), p_of(FIXED, s))[IS_D].mean()) for s in SEEDS],
    "sans-sans": [float(between(p_of(SANS, a), p_of(SANS, b))[IS_D].mean()) for a, b in PAIRWISE],
    "fixed-fixed": [float(between(p_of(FIXED, a), p_of(FIXED, b))[IS_D].mean()) for a, b in PAIRWISE],
    "sans-dropped": [float(between(p_of(SANS, s), p_of(DROPPED, s))[IS_D].mean()) for s in DROPPED_SEEDS],
}
ANCHOR_O = [float(between(clean_of(CONTROL, s), clean_of(FIXED, s))[~IS_D].mean()) for s in SEEDS]
CONTROL_O = [float(between(clean_of(CONTROL, a), clean_of(CONTROL, b))[~IS_D].mean()) for a, b in PAIRWISE]
# The answers each condition gives on `difference` contexts, averaged over seeds before the distance is taken.
MEAN_P = {k: np.mean([p_of(k, s)[IS_D] for s in SEEDS], axis=0) for k in (SANS, FIXED)}
BETWEEN_MEANS = float(between(MEAN_P[SANS], MEAN_P[FIXED]).mean())
MEAN_TO_NULL = {k: float(tv(QN[IS_D], MEAN_P[k]).mean()) for k in MEAN_P}
# Two averages of one condition, over the first and last half of the seeds: how far apart averages land by chance.
HALF = len(SEEDS) // 2
SPLIT_HALF = {
    k: float(
        between(*(np.mean([p_of(k, s)[IS_D] for s in half], axis=0) for half in (SEEDS[:HALF], SEEDS[HALF:]))).mean()
    )
    for k in (SANS, FIXED)
}
# On the contexts of the other ops: the edited model beside `sans-diff` at the same seed, and each beside itself.
OTHER_O = {
    "sans-fixed": [float(between(clean_of(SANS, s), DOSE[(FIXED, s)]["p_full"].astype(np.float64))[~IS_D].mean()) for s in SEEDS],
    "sans-sans": [float(between(clean_of(SANS, a), clean_of(SANS, b))[~IS_D].mean()) for a, b in PAIRWISE],
    "fixed-fixed": [float(between(*(DOSE[(FIXED, s)]["p_full"].astype(np.float64) for s in (a, b)))[~IS_D].mean()) for a, b in PAIRWISE],
}  # fmt: skip


def support(op: int, pairs: np.ndarray) -> np.ndarray:
    """`(n, 216)`: the colors *op* can give on each query pair (any nonzero probability)."""
    n = len(pairs)
    dense = np.zeros((n, 217), bool)
    idx = TABLE.idx[op, pairs]
    dense[np.arange(n)[:, None], np.where(idx < 0, 216, idx)] = TABLE.prob[op, pairs] > 0
    return dense[:, :216]


def answer_dists(pairs: np.ndarray) -> np.ndarray:
    """`(n, 216, ops)`: the answer distribution of every op on each query pair."""
    n = len(pairs)
    out = np.zeros((n, 217, len(OPS)))
    for o in range(len(OPS)):
        idx = TABLE.idx[o, pairs]
        np.put_along_axis(out[:, :, o], np.where(idx < 0, 216, idx), TABLE.prob[o, pairs], axis=1)
    return out[:, :216]


PAIRS_D = PAIRS[IS_D]
SUP = np.stack([support(o, PAIRS_D) for o in range(len(OPS))])  # (ops, n, 216)
OTHER_SUP = SUP[OTHER].any(0)
CATS = {
    "only `difference`": SUP[D] & ~OTHER_SUP,
    "another op": OTHER_SUP,
    "no op": ~SUP.any(0),
}
QN_D = QN[IS_D]


def mass(p: np.ndarray) -> list[float]:
    """The mass on answers only `difference` gives, on answers another op could give, on colors no op gives, and
    off the palette.
    """
    return [float((p * m).sum(1).mean()) for m in CATS.values()] + [float(np.clip(1 - p.sum(1), 0, None).mean())]


@memo
def nearest_mixture(p: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Per context, the mixture of the answer distributions of the ops closest to *p* (non-negative weights, least
    squares over the palette): `(n, 216)`.
    """
    A = answer_dists(pairs)
    return np.stack([A[i] @ nnls(A[i], p[i])[0] for i in range(len(p))])


@memo
def reweighted(null_post: np.ndarray, op_ids: np.ndarray, pairs: np.ndarray, power: float) -> np.ndarray:
    """The target null with its posterior on the other ops raised to *power* and renormalized: more even below 1,
    sharper above.
    """
    w = null_post**power
    return P.predictive(TABLE, contexts_of(op_ids, pairs), w / w.sum(1, keepdims=True))


POWERS = (0.5, 1.0, 2.0)
SHARPENED = {k: reweighted(BAYES["null_post"][IS_D], OP_IDS[IS_D], PAIRS_D, k) for k in POWERS}
BREAKDOWN: dict[tuple[str, int], dict[str, Any]] = {}
for _k in (SANS, FIXED):
    for _s in SEEDS:
        _p = p_of(_k, _s)[IS_D]
        _mix = nearest_mixture(_p, PAIRS_D)
        BREAKDOWN[(_k, _s)] = {
            "mass": mass(_p),
            "off_mixture": float(between(_p, _mix).mean()),
            "mixture_to_null": float(tv(QN_D, _mix).mean()),
            "power": {k: float(tv(SHARPENED[k], _p).mean()) for k in POWERS},
        }
MASS_NULL = mass(QN_D)


def breakdown(kind: str, key: str, sub=None) -> list[float]:
    return [BREAKDOWN[(kind, s)][key] if sub is None else BREAKDOWN[(kind, s)][key][sub] for s in SEEDS]


# Confusion matrices, as ex-2.2.22 takes them: rows are the op of the context; the diagonal is the mass on the
# answers that op can give, and off it, the mass on the answers the column op can give and the row op cannot.


@memo
def gives_table(pairs: np.ndarray) -> np.ndarray:
    """For each op and context, which colors that op can give on the query pair: `(ops, n, colors)`."""
    return np.stack([support(o, pairs) for o in range(len(OPS))])


def confusion(p: np.ndarray) -> np.ndarray:
    gives = gives_table(PAIRS)
    rows = np.arange(len(p))
    true = gives[OP_IDS, rows]
    per_col = np.stack(
        [(p * np.where((OP_IDS == o)[:, None], true, gives[o] & ~true)).sum(1) for o in range(len(OPS))], 1
    )
    return np.stack([per_col[OP_IDS == o].mean(0) for o in range(len(OPS))])


CONFUSION = {
    CONTROL: np.mean([confusion(clean_of(CONTROL, s)) for s in SEEDS], axis=0),
    FIXED: np.mean([confusion(p_of(FIXED, s)) for s in SEEDS], axis=0),
    SANS: np.mean([confusion(clean_of(SANS, s)) for s in SEEDS], axis=0),
    "null": confusion(QN),
}

# --- E4: the other ops --------------------------------------------------------------------------------------------

TASK_END = {k: np.array([[DOSE[(k, s)]["clean_eem"][OP_IDS == o].mean() for o in range(len(OPS))] for s in SEEDS]) for k in (SANS, CONTROL, FIXED)}  # fmt: skip
TASK_DROPPED = np.array([[DOSE[(DROPPED, s)]["clean_eem"][OP_IDS == o].mean() for o in range(len(OPS))] for s in DROPPED_SEEDS])  # fmt: skip
TASK_CHANGE = TASK_END[SANS] - TASK_END[CONTROL]  # (seeds, ops)
NULL_TASK = np.array([BAYES["null_eem"][OP_IDS == o].mean() for o in range(len(OPS))])
IDEAL_TASK = np.array([BAYES["ideal_eem"][OP_IDS == o].mean() for o in range(len(OPS))])

OTHER_BINS = (0.0, 0.05, 0.2, 0.5, 1.0)
OTHER_LABEL = ["< 0.05", "0.05–0.2", "0.2–0.5", "≥ 0.5"]
OBIN = np.clip(np.digitize(POST_D, OTHER_BINS[1:-1]), 0, len(OTHER_BINS) - 2)
N_OBIN = [int(((OBIN == b) & ~IS_D).sum()) for b in range(len(OTHER_LABEL))]


def gain_by_bin(g: np.ndarray) -> np.ndarray:
    return np.array([g[(OBIN == b) & ~IS_D].mean() for b in range(len(OTHER_LABEL))])


GAIN_NULL = gain_by_bin(BAYES["null_eem"] - BAYES["ideal_eem"])
GAIN_SANS = np.array([gain_by_bin(DOSE[(SANS, s)]["clean_eem"] - DOSE[(CONTROL, s)]["clean_eem"]) for s in SEEDS])
GAIN_EDIT = np.array([gain_by_bin(DOSE[(FIXED, s)]["eem"][-1] - DOSE[(FIXED, s)]["clean_eem"]) for s in SEEDS])
SHARE_GAIN = {
    "sans": GAIN_SANS[:, -1].mean() / GAIN_NULL[-1],
    "edit": GAIN_EDIT[:, -1].mean() / GAIN_NULL[-1],
}


# --- Figures and tables ---------------------------------------------------------------------------------------

JIT = np.random.default_rng(0).uniform(-0.12, 0.12, 64)


def along_figure() -> str:
    data = {
        "epochs": TRAJ_EPOCHS.tolist(),
        "other": {k: {s: TASK_ALONG[(k, s)][:, OTHER].mean(1).tolist() for s in SEEDS} for k in (CONTROL, FIXED, SANS)},
        "diff": {k: {s: TASK_ALONG[(k, s)][:, D].tolist() for s in SEEDS} for k in (CONTROL, FIXED, SANS)},
        "kept": list(ALONG_EPOCHS),
        "sans_d": {s: v.tolist() for s, v in ALONG_SANS_D.items()},
        "fixed_d": {s: v.tolist() for s, v in ALONG_FIXED_D.items()},
        "sans_o": {s: v.tolist() for s, v in ALONG_SANS_O.items()},
        "seeds": SEEDS,
    }
    alt = f"""
        Left: on the six other ops, the task score of every run climbs together through training, with sans-diff a
        little above the control throughout and τ_ℓ fixed a little below it for the first hundred epochs. Middle: on difference contexts the control and
        τ_ℓ fixed rise to about 0.7 while sans-diff stays near zero. Right: the distance of sans-diff from the target
        null on difference contexts falls from {SETTLE["sans"][0]:.2f} at epoch 16 to {SETTLE["sans"][-1]:.2f}, and
        settles from about epoch 300; τ_ℓ fixed under the edit falls from {SETTLE["fixed"][0]:.2f} to
        {SETTLE["fixed"][-1]:.2f} on the same path, about {GAP_ALONG[-1]:.2f} above it from epoch 128 on. The distance
        of sans-diff on the other ops falls faster, from {np.mean([v[0] for v in ALONG_SANS_O.values()]):.2f} to
        {np.mean([v[-1] for v in ALONG_SANS_O.values()]):.2f}.
    """
    return along_draw(data, alt)


@memo
def along_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="filtered-corpus-along",
        alt_text=alt_text,
        caption="""
            **How the runs learn, along training.** One thin line per run. **Left:** the task score, the mean over
            the six ops other than `difference`, on a subset of the held-out contexts, recorded every four epochs.
            **Middle:** the same on `difference` contexts. **Right:** the distance from the target null on
            `difference` contexts, at the checkpoints kept at epochs 16, 32 and 64 and then every 32 epochs, and at
            the end: `sans-diff` on the clean pass and `τ_ℓ fixed` under the full edit; the dashed lines are
            `sans-diff` on the contexts of the other ops, its distance from the predictor that is ideal for what it
            trains on.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), layout="constrained", sharex=True)
        ep = np.asarray(data["epochs"])
        for ax, key in zip(axes[:2], ("other", "diff"), strict=True):
            for k in (CONTROL, FIXED, SANS):
                for i, s in enumerate(data["seeds"]):
                    ax.plot(ep, data[key][k][s], color=INK[k], lw=0.6, alpha=0.7, label=tex(SHOWN[k]) if i == 0 else None)  # fmt: skip
            ax.set_ylim(0, 0.8)
            ax.tick_params(labelsize=6.5)
            ax.set_xlabel("epoch", fontsize=7)
        axes[0].set_title("task, other six ops ↑", fontsize=8)
        axes[1].set_title("task, `difference` ↑".replace("`", ""), fontsize=8)
        ax = axes[2]
        kept = np.asarray(data["kept"])
        for s in data["seeds"]:
            ax.plot(kept, data["sans_d"][s], color=INK[SANS], marker=MARKER[SANS], ms=1.8, lw=0.7, alpha=0.8)
            ax.plot(kept, data["fixed_d"][s], color=INK[FIXED], marker=MARKER[FIXED], ms=1.8, lw=0.7, alpha=0.8)
            ax.plot(kept, data["sans_o"][s], color=INK[SANS], lw=0.6, ls="--", alpha=0.6)
        ax.set_ylim(0, 0.8)
        ax.set_title("distance from the target null ↓", fontsize=8)
        ax.set_xlabel("epoch", fontsize=7)
        ax.tick_params(labelsize=6.5)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False, fontsize=7)
        return fig

    return _plot()


def close_figure() -> str:
    data = {
        "columns": [CONTROL, FIXED, SANS, DROPPED],
        "to_null": {k: list(v) for k, v in TO_NULL.items()},
        "bayes_d": float(np.mean(BAYES_GAP_D)),
        "own_o": float(np.mean(OWN_GAP_O)),
        "bins": BIN_LABEL,
        "by_bin": {k: BY_BIN[k].tolist() for k in (FIXED, SANS)},
    }
    bs, bf = BY_BIN[SANS].mean(0), BY_BIN[FIXED].mean(0)
    alt = f"""
        Left: on difference contexts the control is {np.mean(TO_NULL[CONTROL]):.2f} from the target null, the edited
        τ_ℓ fixed runs {np.mean(TO_NULL[FIXED]):.2f}, sans-diff {np.mean(TO_NULL[SANS]):.2f}, and the two dropped runs
        {np.mean(TO_NULL[DROPPED]):.2f}; the seed ranges of sans-diff and τ_ℓ fixed do not overlap. The Bayes gap on
        difference contexts is {data["bayes_d"]:.2f}, and sans-diff is {data["own_o"]:.2f} from the target null on the
        other ops. Right: by bin of the posterior on difference, sans-diff runs from {bs.min():.2f} to {bs.max():.2f}
        and τ_ℓ fixed from {bf.min():.2f} to {bf.max():.2f}, both furthest in the bin where the posterior is 0.95 or
        more, with τ_ℓ fixed further in every bin.
    """
    return close_draw(data, alt)


@memo
def close_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="filtered-corpus-close",
        alt_text=alt_text,
        caption="""
            **How close each model comes to the target null on `difference` contexts.** **Left:** the distance from
            the target null, one column per condition: the control on the clean pass, `τ_ℓ fixed` under the full edit,
            and `sans-diff` and `dropped` on the clean pass. The seeds are small dots over a thin bar for their range,
            and the seed mean is the large mark. The dashed line is the Bayes gap on `difference` contexts (the
            control from the Bayes predictor), and the dotted line is `sans-diff` on the contexts of the other ops,
            where the target null is its own ideal. **Right:** the distance by bin of the posterior on `difference`,
            one hairline per run and the seed mean bold.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), layout="constrained", width_ratios=(1, 1.2))
        ax = axes[0]
        cols = data["columns"]
        for i, k in enumerate(cols):
            dots(ax, i, data["to_null"][k], INK[k], MARKER[k], JIT)
        ax.axhline(data["bayes_d"], color=RULE, lw=0.7, ls="--", zorder=1)
        ax.axhline(data["own_o"], color=INK[SANS], lw=0.7, ls=":", zorder=1)
        ax.set_xticks(range(len(cols)), [tex(SHOWN[k]).replace(", ", ",\n") for k in cols], fontsize=6.5)
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("distance from the target null ↓", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax = axes[1]
        x = np.arange(len(data["bins"]))
        for k in (FIXED, SANS):
            v = np.asarray(data["by_bin"][k])
            for row in v:
                ax.plot(x, row, color=INK[k], lw=0.4, alpha=0.6)
            ax.plot(x, v.mean(0), color=INK[k], lw=1.4, marker=MARKER[k], ms=4, label=tex(SHOWN[k]))
        ax.set_xticks(x, data["bins"], fontsize=6.5)
        ax.set_xlabel("posterior on difference", fontsize=7)
        ax.set_ylim(0, 0.6)
        ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
        return fig

    return _plot()


def close_table() -> str:
    rows = [
        ["control", span(TO_NULL[CONTROL]), "0", "", ""],
        ["`τ_ℓ fixed`, edited", span(TO_NULL[FIXED]), span(SHARE[FIXED]), span(SHARE_FIXED_OWN), ""],
        ["`sans-diff`", span(TO_NULL[SANS]), span(SHARE[SANS]), "", span(OWN_GAP_O)],
        [
            "`dropped`",
            span(TO_NULL[DROPPED]),
            span(1 - np.array(TO_NULL[DROPPED]) / np.array(TO_NULL[CONTROL][: len(DROPPED_SEEDS)])),
            "",
            "",
        ],  # fmt: skip
        ["Bayes gap (control)", span(BAYES_GAP_D), "", "", span(BAYES_GAP_O)],
    ]
    return table_html(
        ["", "distance on `difference`", "share of the way", "against own clean pass", "on the other ops"],
        rows,
        f"""**How close each model comes to the target null.** Seed mean and range over {len(SEEDS)} runs
        ({len(DROPPED_SEEDS)} for `dropped`). The distance on the other ops is from the target null for `sans-diff`
        (its own ideal) and from the Bayes predictor for the control.""",
    )


def same_figure() -> str:
    data = {
        "between": {k: list(v) for k, v in BETWEEN.items()},
        "to_null": {k: float(np.mean(TO_NULL[k])) for k in (SANS, FIXED)},
    }
    alt = f"""
        The distance between sans-diff and the edited τ_ℓ fixed run at the same seed is
        {np.mean(BETWEEN["sans-fixed"]):.2f}, about as far as two edited runs at different seeds
        ({np.mean(BETWEEN["fixed-fixed"]):.2f}) and a little further than two sans-diff runs at different seeds
        ({np.mean(BETWEEN["sans-sans"]):.2f}). Sans-diff and dropped at the same seed are
        {np.mean(BETWEEN["sans-dropped"]):.2f} apart. All of these are about as large as the distance of each model
        from the target null.
    """
    return same_draw(data, alt)


@memo
def same_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="filtered-corpus-same",
        alt_text=alt_text,
        caption="""
            **How far apart the models land on `difference` contexts.** Total variation between two models, context by
            context, one column per pair: `sans-diff` and the edited `τ_ℓ fixed` run at the same seed; two `sans-diff`
            runs at different seeds (all 28 pairs); two edited `τ_ℓ fixed` runs at different seeds; and `sans-diff` and
            `dropped` at the same seed. The pairs are small dots over a thin bar for their range, and the mean is the
            large mark. The dotted lines are the distance of `sans-diff` and of the edited model from the target
            null.
        """,
    )
    def _plot() -> plt.Figure:
        fig, ax = plt.subplots(figsize=(4.2, 2.6), layout="constrained")
        cols = [("sans-fixed", "sans-diff vs\nedited", FIXED, "s"), ("sans-sans", "sans-diff vs\nsans-diff", SANS, "D"),
                ("fixed-fixed", "edited vs\nedited", FIXED, "s"), ("sans-dropped", "sans-diff vs\ndropped", DROPPED, "^")]  # fmt: skip
        jit = np.random.default_rng(1).uniform(-0.15, 0.15, 64)
        for i, (k, _, ink, m) in enumerate(cols):
            dots(ax, i, data["between"][k], INK[ink], m, jit)
        for k in (SANS, FIXED):
            ax.axhline(data["to_null"][k], color=INK[k], lw=0.7, ls=":", zorder=1)
        ax.set_xticks(range(len(cols)), [c[1] for c in cols], fontsize=6.5)
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.set_ylim(0, 0.55)
        ax.set_ylabel("distance between the models", fontsize=7)
        ax.tick_params(labelsize=6.5)
        return fig

    return _plot()


def mean_table() -> str:
    rows = [
        ["`sans-diff` from the target null", span(TO_NULL[SANS]), num(MEAN_TO_NULL[SANS])],
        ["`τ_ℓ fixed`, edited, from the target null", span(TO_NULL[FIXED]), num(MEAN_TO_NULL[FIXED])],
        ["`sans-diff` from `τ_ℓ fixed`, edited", span(BETWEEN["sans-fixed"]), num(BETWEEN_MEANS)],
        [f"`sans-diff`, seeds {SEEDS[0]}–{SEEDS[HALF - 1]} from {SEEDS[HALF]}–{SEEDS[-1]}", "", num(SPLIT_HALF[SANS])],
        [
            f"`τ_ℓ fixed`, edited, seeds {SEEDS[0]}–{SEEDS[HALF - 1]} from {SEEDS[HALF]}–{SEEDS[-1]}",
            "",
            num(SPLIT_HALF[FIXED]),
        ],
    ]
    return table_html(
        ["on `difference` contexts", "run by run", "answers averaged over seeds"],
        rows,
        f"""**Distances taken run by run and on the answers averaged over the {len(SEEDS)} seeds.** The first column is
        the seed mean and range of the distance taken on each run (or each pair at a seed); the second takes the
        distance once, on the answer distributions of each condition averaged over its seeds context by context; the last two
        rows average each half of the seeds of one condition separately.""",
    )


def confusion_figure() -> str:
    data = {k: CONFUSION[k].tolist() for k in (CONTROL, FIXED, SANS, "null")}
    m = {k: np.asarray(v) for k, v in data.items()}
    a, b = sorted(OTHER, key=lambda o: -m["null"][D, o])[:2]
    alt = f"""
        Four matrices, one per condition: the control, τ_ℓ fixed under the full edit, sans-diff, and the target null,
        each with the op of the context on the rows and, on the columns, the mass on answers of that op which the row
        op cannot give. Off the difference row the four look alike, with about 0.05 in each square. In the difference
        row the control puts {m[CONTROL][D, D]:.2f} on the answers of difference and little elsewhere; τ_ℓ fixed
        edited, sans-diff and the target null put {m[FIXED][D, D]:.2f}, {m[SANS][D, D]:.2f} and {m["null"][D, D]:.2f}
        there, and spread the rest over the other six ops in about the same proportions, most on {OPS[a]} and {OPS[b]}
        ({m["null"][D, a]:.2f} and {m["null"][D, b]:.2f} under the target null, {m[SANS][D, a]:.2f} and
        {m[SANS][D, b]:.2f} under sans-diff, {m[FIXED][D, a]:.2f} and {m[FIXED][D, b]:.2f} under the edit).
    """
    return confusion_draw(data, alt)


PRINT_FLOOR = 0.03


def seq_cmap():
    cmap = plt.get_cmap(light_dark("Blues", "magma")).copy()
    cmap.set_bad(light_dark("#fff", "#111"))
    return cmap


def cell_text_color(v: float, vmax: float) -> str:
    dark_cell = (v / vmax > 0.55) == (light_dark(0, 1) == 0)
    return "#fff" if dark_cell else "#000"


@memo
def confusion_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="filtered-corpus-confusion",
        alt_text=alt_text,
        caption=f"""
            **Where the answers go.** Op confusion as in ex-2.2.22, over the held-out contexts of every op, seed mean.
            Rows: the op of the context. Columns: the mass on the colors the column op can give for the query operands
            and the row op cannot. The diagonal (the mass on the answers of the row op) is outlined, printed in grey
            italics, and left out of the color scale. The `difference` column is hatched, since `sans-diff` never
            learned that op and the edit removes it. Values of at least {PRINT_FLOOR:g} are printed.
        """,
    )
    def _plot() -> plt.Figure:
        names = {CONTROL: "control", FIXED: "$\\tau_\\ell$ fixed, edited", SANS: "sans-diff", "null": "target null"}
        fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.9), layout="constrained", sharex=True, sharey=True)
        n = len(OPS)
        eye = np.eye(n, dtype=bool)
        mats = {k: np.asarray(v) for k, v in data.items()}
        vmax = max(float(np.nanmax(np.where(eye, np.nan, m))) for m in mats.values())
        im = None
        for ax, (k, m) in zip(axes, mats.items(), strict=True):
            im = ax.imshow(np.where(eye, np.nan, m), cmap=seq_cmap(), vmin=0, vmax=vmax)
            for i, j in np.ndindex(n, n):
                if i == j:
                    ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, ec="0.6", lw=0.6))
                    ax.text(j, i, f"{m[i, j]:.2f}".removeprefix("0"), ha="center", va="center", fontsize=5.5,
                            color=light_dark("#777", "#999"), fontstyle="italic")  # fmt: skip
                elif m[i, j] >= PRINT_FLOOR:
                    ax.text(j, i, f"{m[i, j]:.2f}"[1:], ha="center", va="center", fontsize=5.5,
                            color=cell_text_color(m[i, j], vmax))  # fmt: skip
            ax.axvspan(D - 0.5, D + 0.5, facecolor="none", edgecolor=light_dark("0.7", "0.3"), hatch="/", lw=0,
                       alpha=light_dark(0.4, 0.5))  # fmt: skip
            ax.set_title(names[k], fontsize=8)
            ax.set_xticks(range(n), OPS, rotation=90, fontsize=6)
            ax.set_yticks(range(n), OPS, fontsize=6)
        assert im is not None
        fig.colorbar(im, ax=axes, shrink=0.7, label="mass")
        return fig

    return _plot()


def breakdown_table() -> str:
    def m(kind: str, i: int) -> str:
        return span([BREAKDOWN[(kind, s)]["mass"][i] for s in SEEDS])

    rows = [
        ["mass on answers only `difference` gives", m(SANS, 0), m(FIXED, 0), num(MASS_NULL[0])],
        ["mass on answers another op could give", m(SANS, 1), m(FIXED, 1), num(MASS_NULL[1])],
        ["mass on colors no op gives", m(SANS, 2), m(FIXED, 2), num(MASS_NULL[2])],
        [
            "distance from the nearest mixture of the ops",
            span(breakdown(SANS, "off_mixture")),
            span(breakdown(FIXED, "off_mixture")),
            "",
        ],  # fmt: skip
        [
            "that mixture from the target null",
            span(breakdown(SANS, "mixture_to_null")),
            span(breakdown(FIXED, "mixture_to_null")),
            "",
        ],  # fmt: skip
        *[
            [
                f"from the target null, posterior to the power {k:g}",
                span(breakdown(SANS, "power", k)),
                span(breakdown(FIXED, "power", k)),
                "",
            ]  # fmt: skip
            for k in POWERS
        ],
    ]
    return table_html(
        ["on `difference` contexts", "`sans-diff`", "`τ_ℓ fixed`, edited", "target null"],
        rows,
        f"""**Where the mass goes on `difference` contexts, and what kind of miss it is**, as edit-effects (E3 there)
        breaks it down. Seed mean and range over {len(SEEDS)} runs. The nearest mixture is the mixture of the answer
        distributions of the seven ops closest to the answer of the model on each context. The last rows compare the
        model with a target null whose posterior on the other ops is raised to a power and renormalized: below 1
        spreads the mass more evenly over the other ops, above 1 gives more to the op the examples favor.""",
    )


def other_figure() -> str:
    data = {
        "change": TASK_CHANGE[:, OTHER].T.tolist(),
        "ops": [OPS[o] for o in OTHER],
        "bins": OTHER_LABEL,
        "null": GAIN_NULL.tolist(),
        "sans": GAIN_SANS.tolist(),
        "edit": GAIN_EDIT.tolist(),
    }
    alt = f"""
        Left: on each of the six other ops sans-diff scores a little above the control at the same seed, by
        {TASK_CHANGE[:, OTHER].mean(0).min():.3f} to {TASK_CHANGE[:, OTHER].mean(0).max():.3f} on the seed mean, with one
        seed below the control on the three HSV ops. Right: on the contexts of the other ops, the gain over the clean
        model rises with the posterior on difference; sans-diff gains about as much as the target null predicts,
        and the edited model gains less in every bin above the first.
    """
    return other_draw(data, alt)


@memo
def other_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="filtered-corpus-other",
        alt_text=alt_text,
        caption="""
            **What leaving `difference` out does to the other ops.** **Left:** the task score of `sans-diff` less that of
            the control at the same seed, per op. The seeds are small dots over a thin bar for their range, and the seed
            mean is the large mark. **Right:** on the contexts of the other ops, the gain in task score by bin of the
            posterior on `difference`: `sans-diff` over the control at the same seed, the edited `τ_ℓ fixed` run over its
            own clean pass, and the gain the target null predicts over the Bayes predictor (dashed).
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), layout="constrained", width_ratios=(1.2, 1))
        ax = axes[0]
        for i, v in enumerate(data["change"]):
            dots(ax, i, v, INK[SANS], MARKER[SANS], JIT)
        ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
        ax.set_xticks(range(len(data["ops"])), data["ops"], rotation=40, ha="right", fontsize=6.5)
        ax.set_ylabel("task, sans-diff less control", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax = axes[1]
        x = np.arange(len(data["bins"]))
        ax.plot(x, data["null"], color=RULE, lw=0.9, ls="--", marker="_", ms=8, label="target null")
        for k, key in ((SANS, "sans"), (FIXED, "edit")):
            v = np.asarray(data[key])
            for row in v:
                ax.plot(x, row, color=INK[k], lw=0.4, alpha=0.6)
            ax.plot(x, v.mean(0), color=INK[k], lw=1.4, marker=MARKER[k], ms=4, label=tex(SHOWN[k]))
        ax.axhline(0, color=ZERO, lw=0.5, zorder=0)
        ax.set_xticks(x, data["bins"], fontsize=6.5)
        ax.set_xlabel("posterior on difference", fontsize=7)
        ax.set_ylabel("gain in task score", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        return fig

    return _plot()


def other_table() -> str:
    rows = [["contexts", *[f"{n:,}" for n in N_OBIN]], ["target null", *[num(v, ".3f") for v in GAIN_NULL]]]
    rows.append(["`sans-diff`", *[span(GAIN_SANS[:, b], ".3f") for b in range(len(OTHER_LABEL))]])
    rows.append(["`τ_ℓ fixed`, edited", *[span(GAIN_EDIT[:, b], ".3f") for b in range(len(OTHER_LABEL))]])
    return table_html(
        ["posterior on `difference`", *OTHER_LABEL],
        rows,
        f"""**The gain on the contexts of the other ops, by bin of the posterior on `difference`.** Task score of
        `sans-diff` over the control at the same seed, of the edited model over its clean pass, and of the target
        null over the Bayes predictor; seed mean and range over {len(SEEDS)} runs.""",
    )


# %%

r"""
# A model trained without `difference`

/// tip |
<!-- lede -->
We trained the control with no loss on `difference` contexts, so that it never learns the op, and set it beside the edited model of the recipe of record. On `difference` contexts it lands a little closer to the target null than the edited model and still well short of it. It misses in the same kind of way: the mass goes to the other ops, but it is shared among them differently from the target null on each context, and differently again at each seed.
///

The edit-effects experiment found that the edited model sends the mass to the right ops overall, but shares it among them differently from the target null context by context. Its Discussion offers two readings: the edit blurs evidence the model uses to share the mass among the other ops, or no model with this architecture, grammar, and set of other ops learns that sharing on contexts that look like `difference`. A model trained without `difference` has had nothing removed, so how close it comes is a reference for the edit. It is also the filtered-corpus comparison of the D2.2 design, and its figures are meant to go into the D2.2 post. This study trains eight new runs, one per seed at seeds 712 to 719, and scores the stored `τ_ℓ fixed` runs and controls at the same seeds.
<!-- REVIEW: the lede and Observations are new with the results; the claim that the shortfall is mostly there without the anchor rests on E2 (sans-diff about 1.5 times its own gap) and E3 (sans-diff and the edited model as far apart as two edited runs). Verify: `RATIO_OWN`, `BETWEEN`. -->
"""

# %%

rf"""
## Observations

- [Along training (E1)](#along-training-e1): the two models close on the target null at the same pace and settle from about epoch 300, with the edited model about {num(float(GAP_ALONG[4:].mean()))} further from epoch 128 on. `sans-diff` keeps up with the control on the other six ops throughout, and never learns `difference`.
- [How close it comes (E2)](#how-close-it-comes-e2): `sans-diff` comes {num(float(np.mean(SHARE[SANS])))} of the way from the control to the target null on `difference` contexts, and the edited `τ_ℓ fixed` runs {num(float(np.mean(SHARE[FIXED])))}; the seed ranges don't overlap. Even so, `sans-diff` stays about {num(float(np.mean(RATIO_OWN)), ".1f")} times as far from the target null as on the contexts it trained on. The `dropped` runs land with `sans-diff`.
- [Whether it misses in the same way (E3)](#whether-it-misses-in-the-same-way-e3): `sans-diff` and the edited model at the same seed are as far apart as two edited runs at different seeds. Both put nearly all of the mass on answers another op could give and miss in how they share it among the ops, and much of that miss changes from seed to seed.
- [What leaving the op out does to the other ops (E4)](#what-leaving-the-op-out-does-to-the-other-ops-e4): `sans-diff` scores a little above the control on each of the other ops. The gain is on contexts that look partly like `difference`; on those that look most like it, `sans-diff` gets {SHARE_GAIN["sans"]:.0%} of the gain the target null predicts, and the edited model {SHARE_GAIN["edit"]:.0%}.
"""

# %%

r"""
## Scope and method

This is an exploratory study, with no preregistration and no gate.

The new condition, `sans-diff`, is the 400-epoch control (no anchor; the controls the anneal-confirm experiment trained at these seeds) with the language-model loss switched off on every `difference` context. The corpus, the batches, and the schedule are those of the control at the same seed: the windows still hold `difference` contexts, and the model reads them as input, but no target inside one is charged, so nothing in training asks the model to answer one. The loss is the mean over the targets that are charged, so a step weighs the other six ops as the control does. Each `sans-diff` run starts from the initialization of the control and of the `τ_ℓ fixed` run at its seed and sees the same windows in the same order, so the difference between `sans-diff` and the control at a seed comes from the masked loss alone.
<!-- REVIEW: the condition was a corpus with the `difference` lines dropped (`filtered`), which changed the batches and left a choice between matching passes (400 epochs) and matching steps (about 467). The paper review asked whether the loss could be masked instead, keeping everything else the same; the training code already has a target mask (`loss_mask` in `sca.anchoring.sample_anchored_batches`, used for the `FALSE` verification answers), so the open decision went away. Verify: the loss in `sca.training.loop` is the mean over non-padding targets, and a masked target is set to padding. -->

The contexts of the other six ops keep their replacement op noise, so about one example in twenty of theirs still shows the answer `difference` would give. That keeps the target null exact for this model. A context of another op draws its noise from the other six ops, `difference` among them, which is the noise model the target null keeps when it sets `difference` to zero. So the Bayes predictor of the contexts `sans-diff` is trained on is the target null, on every context (the cube noise too, which the posterior of record models as a term of its own), and `sans-diff` is scored against the predictor that is ideal for what it was trained on.
<!-- REVIEW: added the cube-noise aside (assumptions review). The corpus of record carries cube noise at 0.02; `sca.data.incontext.posterior_over_ops` leaves it out, but the posterior the dose scoring uses (`docs/m2/ex-2.2.16/posterior.py`) includes it, so the claim holds for the stored arrays. Verify: the likelihood in that module's `posterior`. -->

Contexts like the held-out `difference` contexts are rare among the ones `sans-diff` is trained on: a context of another op shows the answer of `difference` only on an example that is noise of that one kind, so a context whose examples mostly point at `difference` seldom occurs. So on those contexts `sans-diff` answers from what it learned on other contexts, as the edited model has to.
<!-- REVIEW: the held-out `difference` contexts carry the same replacement noise as the corpus, so most show `difference` on two or three examples rather than all three (assumptions review). Verify: the posterior bins of edit-effects (E2 there), where 695 of 2,000 held-out `difference` contexts have a posterior of 0.95 or more. -->

The reference runs are those of the fixed-confirm experiment: `τ_ℓ fixed`, the recipe of record, at seeds 712 to 719, and the controls at the same seeds. Their dose scoring is stored, with the answer distributions on the same held-out contexts, clean and under the full edit, beside the posterior over ops. `sans-diff` is scored with the same pass, so every distance below is taken on the same contexts and in the same way. Each run is measured at its final checkpoint. With eight runs per condition, a difference between conditions is trusted when it is clear of the seed range of both. Seed 718 learns the HSV ops late under `τ_ℓ fixed` and misses the task criterion of fixed-confirm on two of them; E4 shows whether the control and `sans-diff` do the same at that seed.
<!-- REVIEW: moved from the `falloff` runs and ex-2.2.23 controls at 704–711 to the recipe of record, `τ_ℓ fixed` at 712–719, with the anneal-confirm controls (paper review). Verify: `DOSE_ARRAYS_REF` in `docs/m2/fixed-confirm/experiment.py` and `docs/m2/anneal-confirm/experiment.py`; the control arrays are under the latter. -->

A check arm, `dropped`, trains two seeds (712 and 713) on the corpus with the `difference` contexts taken out, for 400 epochs, which is about six-sevenths of the steps. Under the mask, `sans-diff` still reads `difference` contexts in its window; under `dropped` it never sees one. If the two land in about the same place, reading the contexts without answering them makes little difference to what the model learns.

Every measurement here is taken on the answer distribution, which a rotation of the residual stream leaves unchanged, so the coordinate axes that Adam favors (the rotated-basis experiment) do not enter, except through the edit itself, which acts along e₁.

## The measurements

Each measurement is taken on the answer distribution at the query `=`, over the colors of the palette, and averaged over held-out contexts.

Distance from the target null
:   Total variation between the answer distribution of the model and the target null, with any mass the model puts off the palette counted as distance; 0 is a perfect match and 1 shares nothing. Lower is closer. On `difference` contexts for `τ_ℓ fixed` it is taken under the full edit, as in edit-effects; for `sans-diff` and the control it is taken on the clean pass. On the contexts of the other ops, for `sans-diff`, it is how far the model falls short of the predictor that is ideal for what it was trained on.

Bayes gap
:   The same distance, between the clean control and the Bayes predictor: how far a model of this size, trained without an anchor, falls short of the ideal on contexts like the ones it trained on. It sets the scale for the other distances. Taken on `difference` contexts, where edit-effects took its reference, and on the contexts of the other ops. edit-effects called it a comparison point rather than a floor for the edited model, since the two are distances between different pairs. For `sans-diff` the two pairs are the same: its Bayes predictor is the target null on every context, so its own gap on the contexts of the other ops is the floor that its distance on `difference` contexts can be measured against.

Share of the way
:   One minus the distance of a model from the target null over the distance of the control at the same seed, each a mean over `difference` contexts: how much of the way from the control to the target null that model has come. The control is the start of the way (0), and the target null its end (1). For the edited `τ_ℓ fixed` runs, edit-effects took this against their own clean pass, and it is reported both ways.

Distance between the models
:   Total variation between `sans-diff` and the edited `τ_ℓ fixed` run at the same seed, context by context, on `difference` contexts.

Seed-to-seed spread
:   The same distance, between two `sans-diff` runs at different seeds, over all 28 pairs: how far apart two models trained the same way land. A distance between the models near the spread means they are as alike as a model is to itself at another seed.
<!-- REVIEW: added the seed-to-seed spread and the own-gap reading of the distance on other-op contexts (assumptions review, agreed in the paper review). The paper review asked whether "the way from the control" is the floor; it is not, and the share of the way now says where the way starts and ends. Named a spread rather than a floor, since "floor" already means the no-evidence predictor (ex-2.2.21, ex-2.2.22) and edit-effects calls the Bayes gap "a comparison point rather than a floor". -->

Task score
:   Expected exact match, per op, on the clean pass.
"""

# %%

rf"""
## Along training (E1)

How does `sans-diff` learn beside the control, which sees the same windows in the same order? The trajectories every run records hold the task score of each op on held-out contexts along training, so the six ops `sans-diff` trains on can be compared with the control and with `τ_ℓ fixed`, step for step. Because the batches match, a gap between `sans-diff` and the control on one of those ops is caused by the masked loss.

The `τ_ℓ fixed` runs kept checkpoints along training, and `sans-diff` keeps the same ones. Scoring both at each kept epoch gives the distance from the target null on `difference` contexts along training, for `sans-diff` on the clean pass and for `τ_ℓ fixed` under the full edit, so we can see whether the two settle early or are still closing when training ends, and whether they close at the same pace. The controls kept no checkpoints, so this curve is not available for them; their task score on `difference` along training is.

{along_figure()}

On the six ops it trains on, `sans-diff` keeps pace with the control from the first record to the last, and stays a little ahead of it (left). `τ_ℓ fixed` falls a little behind the control in the first hundred epochs, while the anchor settles, and catches up by the end, except at seed 718, which learns the HSV ops late (E4). On `difference` contexts `sans-diff` scores near zero all the way through, as the target null does (middle). So reading the `difference` contexts as input, with no loss on them, does not teach the model to answer them.

The main measure here is the distance from the target null on `difference` contexts (right). The two models close on it together. The edited model starts further away, the gap narrows over the first hundred epochs or so, and from epoch 128 it holds at a small, steady amount while both curves keep falling. Both settle from about epoch 300, so 400 epochs is long enough for this comparison, and the extra distance the anchor and the edit leave is set early and does not close with more training. The dashed lines are `sans-diff` on the contexts of the other ops, where the target null is the ideal for what it trains on. They fall faster and further, which is the subject of E2.
"""

# %%

rf"""
## How close it comes (E2)

How far is `sans-diff` from the target null on `difference` contexts, beside the edited `τ_ℓ fixed` runs and the Bayes gap? If it comes about as close as the edited model, then landing short of the target null is near what a model of this kind does on these contexts, and the shortfall need not come from the anchor and the edit. If it comes much closer, the anchor and the edit leave a distance that a model can close. The main measure is the share of the way from the control to the target null, with the distance itself beside it.

{close_figure()}

{close_table()}

`sans-diff` comes closer than the edited model at every seed, and the seed ranges of the two are well apart. It comes about three-fifths of the way, where the edited model comes half of the way. Taken against the clean pass of the `τ_ℓ fixed` run in place of the control, the share for the edited model is the same, since the clean run is as far from the target null as the control. So the anchor and the edit do leave some distance that a model without the op can close, about a sixth of what the edited model has left.

Most of the shortfall is there without them, though. On the contexts of the other ops, where the target null is the ideal for what `sans-diff` was trained on, it falls short by about as much as the control falls short of the Bayes predictor there. On `difference` contexts it falls about half as short again. A model with this architecture, grammar, and set of other ops, which has never been asked to answer a `difference` context, lands well short of the target null on one, and the edited model lands a little further.

The two `dropped` runs land with `sans-diff` at their seeds, inside its seed range. A model that reads `difference` contexts without answering them ends up about where one that never sees them does.

Split by the posterior on `difference` (right), both models are furthest on the contexts that are nearly sure of `difference`, which are the ones least like anything `sans-diff` trained on. The extra distance of the edited model is also largest there, and smallest on the contexts whose examples point mostly at other ops. The middle bin holds fewer than a hundred contexts, so its value is loose.
"""

# %%

rf"""
## Whether it misses in the same way (E3)

Do the edited model and `sans-diff` miss the target null in the same way? If the distance between the two models is small beside the distance of each from the target null, and near the seed-to-seed spread, they share the mass among the other ops alike on a context, and the part of the shortfall they share is a property of these contexts rather than of the anchor and the edit. If the two models are as far from each other as from the target null, they miss in different ways on a context, though they may still miss in the same kind of way, which the breakdown below looks at. The main measure is the distance between the models, context by context.

{same_figure()}

Two `sans-diff` runs at different seeds land about as far from each other as each lands from the target null, and so do two edited runs. `sans-diff` and the edited model at the same seed are as far apart as two edited runs, and a little further apart than two `sans-diff` runs, clear of their range. Sharing the seed and the initialization does not make the two miss alike on a context. The edited runs also scatter more from seed to seed than the `sans-diff` runs do, so the anchor and the edit add some scatter as well as some distance. So on a given `difference` context each run shares the mass among the other ops in a way of its own, and much of its distance from the target null is that scatter from run to run.

Averaging the answers of each condition over its eight seeds, context by context, shows how much.

{mean_table()}

The averaged `sans-diff` answers are about a third closer to the target null than a single run, and the averaged edited answers too. The two averages are still about as far from each other as the `sans-diff` average is from the target null. But two averages over four seeds each, from one condition, land nearly as far apart, so most of what separates the two averages is scatter that eight seeds leave in an average, and the data show little of the miss that is shared by the runs of one condition and differs between the conditions.
<!-- REVIEW: added the split-half rows (claims review); the earlier text said part of the miss "differs between the conditions", which the split-half does not support. Verify: `SPLIT_HALF` against `BETWEEN_MEANS`. -->

{breakdown_table()}

Both models put nearly all of the mass on answers another op could give, and very little on answers only `difference` gives (the edited model a little more). Each is close to some mixture of the answers of the ops, and that mixture is nearly as far from the target null as the model is. So both miss in how they share the mass among the other ops, as edit-effects found for the earlier recipe. A target null with its posterior raised to a power, which spreads the mass more evenly over the ops or gives more of it to the op the examples favor, fits neither model better than the target null as it is, so neither leans toward even sharing or toward the favored op.

{confusion_figure()}

Pooled over contexts, the `difference` row of `sans-diff` and of the edited model looks much like that of the target null: the mass spread over the six other ops in about the same proportions, except that the edited model gives `lighten` about a third less than the target null does. The other rows look alike in all four matrices. As ex-2.2.22 found for the edited model, pooling over the contexts of a row averages away most of the context-by-context miss above, in the same way averaging over seeds does.

The two models differ in two ways: `sans-diff` never learned the op, and it trained without the anchor. On the contexts of the other ops, the clean `τ_ℓ fixed` run is as far from the control at the same seed as two controls at different seeds are from each other. So the anchor moves the answers there by about as much as a change of seed does, and on `difference` contexts too the two models differ by about as much as two runs of one condition.
<!-- REVIEW: the anchor bound is the distance between the clean τ_ℓ fixed run and the control at one seed on other-op contexts, compared with the spread between controls. Verify: `ANCHOR_O`, `CONTROL_O`. -->
"""

# %%

rf"""
## What leaving the op out does to the other ops (E4)

On contexts of the other ops, does `sans-diff` score as well as the control, and does it gain where the target null says a model without `difference` should? The target null predicts a gain on contexts of other ops whose examples point partly at `difference`, since a model that cannot answer `difference` puts that mass on the right op. edit-effects found the edited model gains most of the way the target null predicts there, and this is the same comparison for a model that never learned the op, with the edited model beside it. It also checks the training: if `sans-diff` scored well below the control on the other ops, its distance in E2 would partly be a weaker model rather than a missing op. The main measure is the task score.

{other_figure()}

`sans-diff` scores a little above the control on each of the six other ops (left). One seed, 717, scores a little below the control on the three HSV ops and above it on the rest. The two `dropped` runs score like `sans-diff` at their seeds. At seed 718, where `τ_ℓ fixed` learns the HSV ops late, the control and `sans-diff` both learn them on time, so that weakness belongs to the anchored run. None of this makes `sans-diff` a weaker model, and its distance in E2 comes from the missing op.

{other_table()}

The gain sits where the target null says it should (right). On contexts whose posterior on `difference` is below 0.05, most of the held-out contexts, `sans-diff` and the control score about the same. Above that, the gain grows with the posterior on `difference`, `sans-diff` gains about as much as the target null predicts (a little more in the second bin), and the edited model gains less in every bin above the first. The edited model also loses a little on the contexts that look least like `difference`, a small cost of the edit on contexts it would ideally leave alone. The top bin holds about a hundred contexts, so its values are loose.
"""

# %%

r"""
## Discussion

A model that never learned `difference` lands short of the target null on `difference` contexts, a little closer than the edited model, and misses in the same kind of way. Of the two readings edit-effects offered, this favors the second: most of the shortfall is what a model of this kind does on contexts that look like an op it was never asked to answer. The anchor and the edit add a smaller part on top, set early in training (E1) and largest on the contexts that are surest of `difference` (E2), and they add some scatter from seed to seed; the model they leave is as far from `sans-diff` as two edited runs at different seeds are from each other (E3).
<!-- REVIEW: the last clause said the added part was "about as large as the difference between two runs of one condition", which conflated the extra distance from the target null (about 0.07, E2) with the distance between the two models (0.42, E3); the two-runs comparison belongs to the latter. Verify: `TO_NULL[FIXED] - TO_NULL[SANS]` against `BETWEEN["sans-fixed"]` and `BETWEEN["fixed-fixed"]`. -->

The miss looks like an underdetermined choice. On a context whose examples point at `difference`, the other six ops are all poor fits, and the training data say little about how to share the mass among them, since such contexts are rare in it. Each run settles that sharing in its own way, so runs at different seeds disagree on a context by about as much as each misses the target null, and averaging over seeds, or pooling over contexts as a confusion matrix does, brings the answers much closer to it. The target null sits nearer the average over such models than any one of them does.

On the contexts of the other ops, a model trained without `difference` gains almost all that the target null predicts, and the edited model gains most of it. So the edited model of the recipe of record comes within about a run-to-run difference of a model trained without the op, on `difference` contexts and on the others, where the two models at a seed are as far apart as two edited runs and about as far as two `sans-diff` runs.
<!-- REVIEW: the claim on the other ops is backed by `OTHER_O` (claims review), the edited model under the full edit beside `sans-diff` on other-op contexts. -->

This holds for one op, at one model size, with 400 epochs of training.
"""

# %%

r"""
## References

[anneal-confirm](/docs/m2/anneal-confirm/report.py)
:   Trained the 400-epoch control, with no anchor, at seeds 712 to 719, beside the annealed recipe, and stored the dose scoring of both.

[edit-effects](/docs/m2/edit-effects/report.py)
:   On the `falloff` recipe the full edit lands about half of the way to the target null; the distance left is about 2.4 times the distance from the clean model to the Bayes predictor, nearly all of it in how the mass is shared among the other ops context by context.

[ex-2.2.22](/docs/m2/ex-2.2.22/report.py)
:   Confusion matrices of the clean and edited models against the target null, by how sure the context is about its op; pooled over contexts, they make the edited model look closer to the target null than the distance taken context by context.

[fixed-confirm](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` kept the anchor on all eight fresh seeds, 712 to 719, and stayed within the spill criterion; it became the recipe of record.

[rotated-basis](/docs/m2/rotated-basis/report.py)
:   Adam puts nearly half of the energy of the residual stream on one coordinate axis per seed; in a randomly rotated basis the anchor still holds on all eight seeds, and the energy moves to the rotated axes.

"""
