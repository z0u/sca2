# ruff: noqa: B018
# title: Cooldown

# A short exploratory report: the recipe of record at seeds 712 to 719, carried on from stored states to epoch 360 and
# branched there into two arms, with and without the end anneal of both terms. `experiment.py` beside this script
# trains and scores the runs.

import json
import re
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import experiment as ex
from mini.lit import memo
from mini.store import project_store
from mini.vis import figure_html, light_dark, themed

# --- The stored results ---------------------------------------------------------------------------------------


def fetch_json(ref: str) -> Any:
    store = project_store()
    art = store.get_refs([ref]).get(ref)
    assert art is not None, f"not published: {ref}"
    with tempfile.TemporaryDirectory() as tmp:
        return json.loads(Path(store.get(art, Path(tmp) / "results.json")).read_text())


rb, lr, fc, ac = ex.rb, ex.lr, ex.fc, ex.ac
AXES_BODY = fetch_json(ex.AXES_REF)
DESIGN = AXES_BODY["design"]
OPS: tuple[str, ...] = tuple(DESIGN["ops"])
D = OPS.index(ex.ANCHORED_OP)
OTHER = [o for o in range(len(OPS)) if o != D]
SEEDS: list[int] = DESIGN["seeds"]
ARMS = list(ex.ARMS)
ON, OFF = ARMS
SHOWN = {ON: "cooldown", OFF: "no cooldown"}
CRITERION = 0.02  # the selectivity criterion of the D2.2 design
E1 = 0  # the anchored axis, in code
FULL = len(rb.DOSE_GAMMAS) - 1
ANS = list(lr.lean.ANSWERS)
SLICE_NAMES = ["embedding", "slice 1", "slice 2", "slice 3", "slice 4"]


def num(v: float, spec: str = ".3f") -> str:
    """A number with a typographic minus."""
    return f"{v:{spec}}".replace("-", "−")


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


def net_of(drop) -> np.ndarray:
    """The drop under the e₁ edit less the median drop along the other axes: *drop* is (axes, ...), e₁ first."""
    drop = np.asarray(drop)
    return drop[E1] - np.median(drop[1:], axis=0)


_AX = {r["label"]: r for r in AXES_BODY["runs"]}
_RESTRICTED = {r["label"]: r for r in fetch_json(ex.RESTRICTED_REF)["runs"]}
_MEASURED = {r["label"]: r for r in fetch_json(ex.RESULTS_REF)["runs"]}
_TRAJ = fetch_json(ex.TRAJ_REF)
# The controls (no pull, no anti term) at these seeds are the anneal confirmation's, as for the fixed confirmation.
_AC_AX = {r["label"]: r for r in fetch_json(ac.AXES_REF)["runs"]}
CONTROL_CLEAN = np.mean([_AC_AX[ac.label_of(s, None, "control")]["clean"] for s in SEEDS], axis=0)


def scores(arm: str, s: int) -> dict[str, Any]:
    lab = ex.label_of(arm, s)
    r = _AX[lab]
    net = net_of(r["drop"])  # (doses, ops)
    t = _TRAJ[lab]["traj"]
    return {
        "removal": float(net[FULL, D] / (r["clean"][D] - r["null"][D])),
        "spill": float(net[:, OTHER].max()),
        "block1": float(net_of(_RESTRICTED[lab]["drop"]["block 1"])[OTHER].max()),
        "clean": np.asarray(r["clean"]),
        "task": np.asarray(r["clean"]) - CONTROL_CLEAN,
        "loss": float(_TRAJ[lab]["val_loss"][-1]),
        "lean": float(_MEASURED[lab]["r_light_emb"]),
        "alpha": np.asarray(t["alpha_anchored"])[-1][:, ANS].mean(axis=1),  # by slice, embedding first
        "alpha_other": np.asarray(t["alpha_other"])[-1][:, ANS].mean(axis=1),
    }


END = {(a, s): scores(a, s) for a in ARMS for s in SEEDS}


def by_seed(arm: str, key: str) -> np.ndarray:
    return np.array([END[(arm, s)][key] for s in SEEDS])


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """The mean of the differences *a* − *b* over seeds, and its 95% paired t interval."""
    d = a - b
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


OVER = {a: int((by_seed(a, "spill") > CRITERION).sum()) for a in ARMS}
OVER_B1 = {a: int((by_seed(a, "block1") > CRITERION).sum()) for a in ARMS}
REMOVAL_GAP = paired(by_seed(OFF, "removal"), by_seed(ON, "removal"))

def along(arm: str, s: int) -> dict[str, np.ndarray]:
    t = _TRAJ[ex.label_of(arm, s)]["traj"]
    return {
        "epoch": np.asarray(t["epoch"]),
        "eem": np.asarray(t["eem"]),
        "alpha": np.asarray(t["alpha_anchored"])[:, lr.TRACK_SLICE][:, ANS].mean(axis=1),
        "weight": np.asarray(t["weight"]),
        "anti": np.asarray(t["anti_weight"]),
    }


ALONG = {(a, s): along(a, s) for a in ARMS for s in SEEDS}

# --- The table and the figure ----------------------------------------------------------------------------------

RULE = ("#333", "#ddd")
ZERO = ("#aaa", "#555")
INK = {ON: ("#08306b", "#c6dbef"), OFF: ("#d95f02", "#fdb863")}
MARKER = {ON: "s", OFF: "o"}
FROM = 300  # the figure starts here, so the last 40 epochs are wide enough to see


def end_table() -> str:
    def row(name: str, a: np.ndarray, b: np.ndarray, spec: str = ".3f") -> list[str]:
        d, lo, hi = paired(b, a)
        return [name, num(float(a.mean()), spec), num(float(b.mean()), spec), f"{num(d, spec)} [{num(lo, spec)}, {num(hi, spec)}]"]  # fmt: skip

    rows = [
        row("removal ↑", by_seed(ON, "removal"), by_seed(OFF, "removal")),
        row("spill ↓", by_seed(ON, "spill"), by_seed(OFF, "spill")),
        row("spill, block 1 ↓", by_seed(ON, "block1"), by_seed(OFF, "block1")),
        row("validation loss ↓", by_seed(ON, "loss"), by_seed(OFF, "loss"), ".4f"),
        row("embedding lean", by_seed(ON, "lean"), by_seed(OFF, "lean"), ".2f"),
    ]
    rows += [
        row(f"score, `{op}` ↑", by_seed(ON, "clean")[:, o], by_seed(OFF, "clean")[:, o]) for o, op in enumerate(OPS)
    ]
    rows += [
        row(f"α, {name}", by_seed(ON, "alpha")[:, i], by_seed(OFF, "alpha")[:, i], ".2f")
        for i, name in enumerate(SLICE_NAMES)
    ]
    return table_html(
        ["measurement", "cooldown", "no cooldown", "difference [95% interval]"],
        rows,
        """
        **The two arms at epoch 400**, as seed means over eight runs, and the mean of the differences seed by seed
        (no cooldown less cooldown) with its 95% paired t interval. Removal and spill are net of the median over the
        other 63 axes of the run, and spill is the largest net drop on another op at any dose. The scores are
        expected exact match per op on the held-out contexts. α is at the example answers of the `difference`
        contexts, by slice.
        """,
    )


def along_figure() -> str:
    data = {
        "arms": ARMS,
        "shown": [SHOWN[a] for a in ARMS],
        "from": FROM,
        "branch": ex.BRANCH,
        "runs": {a: [{k: v.tolist() for k, v in ALONG[(a, s)].items()} for s in SEEDS] for a in ARMS},
    }
    alt = f"""
        Three stacked panels over epochs {FROM} to 400, one line per run, the cooldown arm in blue and the no-cooldown
        arm in orange. Top: the task score, a mean over ops, rising slowly and almost the same in both arms. Middle: α
        at the example answers at slice 3, flat near the same value in both arms. Bottom: the pull weight and the anti
        weight at the embedding; from epoch 360 the cooldown arm drops both to a tenth while the other arm holds them.
    """
    return along_draw(data, alt)


@memo
def along_draw(data: dict, alt_text: str) -> str:
    @themed(
        name="cooldown-along",
        alt_text=alt_text,
        caption="""
            **The last part of training, by arm.** One line per run. **Top:** expected exact match, a mean over ops,
            on the held-out contexts recorded along training. **Middle:** α at the example answers of the
            `difference` probe contexts at slice 3. **Bottom:** the pull weight (solid) and the anti weight at the
            embedding (dashed). The arms share every epoch before the dotted line.
        """,
    )
    def _plot() -> plt.Figure:
        fig, axes = plt.subplots(
            3, 1, figsize=(4.6, 4.2), layout="constrained", sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.7]}
        )
        for a, shown in zip(data["arms"], data["shown"], strict=True):
            ink = light_dark(*INK[a])
            for i, r in enumerate(data["runs"][a]):
                ep = np.asarray(r["epoch"])
                m = ep >= data["from"]
                lab = shown if i == 0 else None
                axes[0].plot(ep[m], np.asarray(r["eem"])[m], color=ink, lw=0.7, alpha=0.8, label=lab)
                axes[1].plot(ep[m], np.asarray(r["alpha"])[m], color=ink, lw=0.7, alpha=0.8)
                if i == 0:
                    anti = np.asarray(r["anti"])
                    anti = anti[:, 0] if anti.ndim == 2 else anti
                    axes[2].plot(ep[m], np.asarray(r["weight"])[m], color=ink, lw=1.0, marker=MARKER[a], ms=2, markevery=8)  # fmt: skip
                    axes[2].plot(ep[m], anti[m], color=ink, lw=1.0, ls="--")
        for ax in axes:
            ax.axvline(data["branch"], color=light_dark(*RULE), lw=0.6, ls=":", zorder=0)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("task score ↑", fontsize=7)
        axes[1].set_ylabel("α, slice 3", fontsize=7)
        axes[2].set_ylabel("weight", fontsize=7)
        axes[2].set_ylim(bottom=0)
        axes[2].set_xlabel("epoch", fontsize=7)
        axes[0].legend(fontsize=6.5, frameon=False, loc="lower right")
        return fig

    return _plot()


# %%

f"""
# Cooldown

/// tip |
<!-- lede -->
The recipe of record ends with a cooldown: over the last 40 epochs the pull and the anti weight both fall to a tenth. We branched each of eight runs at the start of the cooldown and trained one branch with it and one without. TODO: which way it came out.
///

The cooldown was in the schedule from the first D2.2 runs, and nothing since has tested it. By epoch 360 the learning rate is a few percent of its peak, so little should still be moving, and a recipe without it would be simpler to describe. This report carries the [fixed-confirm](/docs/m2/fixed-confirm/report.py) runs at seeds 712 to 719 on from their stored epoch-64 states to epoch 360, and branches each there into two arms that differ only in the cooldown. Because the arms share everything up to epoch 360, we compare them seed by seed.

TODO: the result paragraph.

## Scope and method

Every run is the recipe of record, `τ_ℓ fixed`, at the seeds of the fixed-confirm experiment. The `cooldown` arm runs it as it is. The `no cooldown` arm takes the end anneal out of both terms, so the pull and the anti weight hold their epoch-360 values to the end. The learning rate schedule is the same in both. The stems and both arms trained and were scored on L40S cards. The stored runs trained on L4s, so the cooldown arm is a close repeat of them rather than the same run.

The measurements are those of the fixed-confirm experiment: removal and spill under the full edit, net of the median over the other 63 axes of the run, spill under the edit at block 1 alone, the score per op, and α by slice.

## The two arms at epoch 400 (E1)

TODO: what we saw.
"""

# %%

end_table()

# %%

r"""
TODO: one line on the figure.
"""

# %%

along_figure()

# %%

r"""
## References

[fixed-confirm](/docs/m2/fixed-confirm/report.py)
:   `τ_ℓ fixed` kept the anchor on all eight runs at fresh seeds with removal near 0.9 and no run over the spill criterion, and became the recipe of record.

[paired-runs](/docs/m2/paired-runs/report.py)
:   Runs branched from one stored state stay alike seed by seed for well over a hundred epochs, so arms that share a state can be compared in pairs.
"""
