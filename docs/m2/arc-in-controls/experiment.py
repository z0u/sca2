"""Arc in controls: is the arc of the low-runs report there in a model trained without the anchor?

The low-runs report drew the states at the example answers of anchored runs in the plane of e₁ and the direction a
non-fitting answer has, and found an arc: fitting examples leave the non-fitting direction about as much as they gain
on e₁. This probe asks whether a model trained with no anchor holds the same arc, found after training. In each run
it takes the difference between the class means of fitting and non-fitting examples (the post-hoc direction `u`) as
the fitting end, finds the non-fitting direction the same way the low-runs report did with e₁ (the renormalized
class-mean difference of the states with `u` removed), and reads the same numbers: how well that direction separates
the classes, where each class sits along it, and how its coordinate moves with the `u` coordinate among the fitting
examples. The anchored runs are read both ways, with e₁ and with `u`, so the two readings can be set side by side.
No new training; the checkpoints are the anneal confirmation's controls and the fixed confirmation's per-visit runs
at seeds 712 to 719.

Results (2026-11-05, production checkpoints): the arc is there in every control, and short. Drawn in the plane of the
two class means, with no chosen axis, the fitting and non-fitting means lie a median of 33° apart at slice 2 (14° to
37°) and 21° at slice 3 (9° to 35°) in the controls, against 60° and 83° in the anchored runs, and the class-mean
direction separates the classes at AUC 0.93 or above in every run of both kinds. In a control the fitting examples sit
in a tight group just off the non-fitting group; in an anchored run they spread along a quarter circle toward e₁ (the
class-mean direction lies at cosine 0.73 to 0.77 to e₁ at slice 3). So the anchor stretches the arc and names one end
in advance; the arc itself is found after training too. The `u` reading (the arc read with the class-mean difference
as its fitting end, as the low-runs report read it with e₁) is recorded but not quoted: with that end, the remaining
class difference is mostly the shadow the end casts on the length of the rest.

    bin/mini run docs/m2/arc-in-controls/experiment.py --app modal --max-containers 8 --budget 30m
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

import numpy as np

from mini import Ctx, Experiment


def _load_sibling(name: str, alias: str):
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / name / "experiment.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


lr = _load_sibling("low-runs", "arc_in_controls_lr")
fc, ac, ex2216 = lr.fc, lr.ac, lr.ex2216
NAME = "arc-in-controls"
SEEDS = tuple(range(712, 720))
SAMPLE = 300


def runs() -> list[dict[str, Any]]:
    out = [{"label": ac.label_of(s, None, "control"), "kind": "control", "seed": s,
            "checkpoint": ac.CHECKPOINT_REF.format(label=ac.label_of(s, None, "control"))} for s in SEEDS]  # fmt: skip
    out += [{"label": fc.label_of(s, None), "kind": "anchored", "seed": s,
             "checkpoint": fc.CHECKPOINT_REF.format(label=fc.label_of(s, None))} for s in SEEDS]  # fmt: skip
    return out


def resolve(rows: list[dict]) -> dict:
    from mini.store import get_ref

    base = lr.ex2222.resolve_reused([])
    found = {r["checkpoint"]: get_ref(r["checkpoint"]) for r in rows}
    missing = [k for k, v in found.items() if v is None]
    assert not missing, f"not published: {missing}"
    return {"holdout": base["holdout"], "artifacts": found}


def _reading(x: np.ndarray, fits: np.ndarray, a: np.ndarray) -> dict:
    """The arc read with *a* as the fitting end, as the low-runs report read it with e₁."""
    along_a = x @ a
    off = x - np.outer(along_a, a)
    ren = off / np.linalg.norm(off, axis=1, keepdims=True)
    d = ren[fits].mean(0) - ren[~fits].mean(0)
    v = d / np.linalg.norm(d)
    along_v = ren @ v
    raw_v = x @ v
    top = fits & (along_a >= np.quantile(along_a[fits], 0.9))
    m = x[top].mean(0)
    rng = np.random.default_rng(0)
    pick = {k: rng.choice(np.flatnonzero(mk), size=min(SAMPLE, int(mk.sum())), replace=False) for k, mk in (("fits", fits), ("not", ~fits))}  # fmt: skip
    return {
        "direction_auc": lr._auc(along_v, fits),
        "renormalized_coordinate": {"fits": float(along_v[fits].mean()), "not": float(along_v[~fits].mean())},
        "end_coordinate": {"fits": float(along_a[fits].mean()), "not": float(along_a[~fits].mean())},
        "correlation_among_fits": float(np.corrcoef(along_a[fits], along_v[fits])[0, 1]),
        "in_plane_norm": {
            "fits": float(np.hypot(along_a, raw_v)[fits].mean()),
            "not": float(np.hypot(along_a, raw_v)[~fits].mean()),
        },
        "top_decile_resultant": float(np.linalg.norm(m)),
        "top_decile_cos_to_end": float(m @ a / np.linalg.norm(m)),
        "points": {k: np.stack([along_a[i], raw_v[i]], axis=1).round(4).tolist() for k, i in pick.items()},
    }


def _means(x: np.ndarray, fits: np.ndarray, u: np.ndarray) -> dict:
    """The arc drawn from the class means alone, with no chosen axis: the plane of the non-fitting mean direction n
    and the part of the fitting mean direction at right angles to it, the angle between the two means, and where each
    class sits in that plane.
    """
    mf, mn = x[fits].mean(0), x[~fits].mean(0)
    n = mn / np.linalg.norm(mn)
    f = mf / np.linalg.norm(mf)
    w = f - (f @ n) * n
    w /= np.linalg.norm(w)
    pn, pw = x @ n, x @ w
    ang = np.degrees(np.arctan2(pw, pn))
    rng = np.random.default_rng(0)
    pick = {k: rng.choice(np.flatnonzero(mk), size=min(SAMPLE, int(mk.sum())), replace=False) for k, mk in (("fits", fits), ("not", ~fits))}  # fmt: skip
    return {
        "auc_u": lr._auc(x @ u, fits),
        "e1_cos": {"non_fitting_axis": float(n[0]), "fitting_axis": float(w[0])},
        "angle_between_means": float(np.degrees(np.arccos(np.clip(f @ n, -1, 1)))),
        "mean_length": {"fits": float(np.linalg.norm(mf)), "not": float(np.linalg.norm(mn))},
        "angle_auc": lr._auc(ang, fits),
        "angle": {
            "fits": float(np.median(ang[fits])),
            "not": float(np.median(ang[~fits])),
            "fits_p90": float(np.quantile(ang[fits], 0.9)),
        },
        "in_plane_norm": {"fits": float(np.hypot(pn, pw)[fits].mean()), "not": float(np.hypot(pn, pw)[~fits].mean())},
        "points": {k: np.stack([pn[i], pw[i]], axis=1).round(4).tolist() for k, i in pick.items()},
    }


def measure(checkpoint, holdout, label: str, kind: str) -> dict:
    from mini import get_data_dir
    from mini.store import get

    workdir = get_data_dir() / NAME / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    scorer = lr._Scorer(get(holdout, workdir / "holdout.npz"), tok2color, color_ids)
    ho = scorer.ho
    a_id = lr.OP_NAMES.index(lr.ANCHORED_OP)
    n_colors = len(color_ids)
    y = tok2color[ho.tokens[:, list(lr.ANSWERS)]]
    a = tok2color[ho.tokens[:, [lr.UNIT * i for i in range(lr.K)]]]
    b = tok2color[ho.tokens[:, [lr.UNIT * i + 2 for i in range(lr.K)]]]
    fits = (scorer.table.lookup(a * n_colors + b, y)[a_id] > 0).ravel()
    st = lr._states(model, ho.tokens)
    out: dict[str, Any] = {"label": label, "kind": kind, "slices": {}}
    for s in (2, 3):
        x = st[s][:, list(lr.ANSWERS)].reshape(-1, st.shape[-1]).astype(np.float64)
        d = x[fits].mean(0) - x[~fits].mean(0)
        u = d / np.linalg.norm(d)
        e1 = np.zeros(x.shape[1])
        e1[0] = 1
        r = {"u": _reading(x, fits, u), "cos_u_e1": float(u[0]), "means": _means(x, fits, u)}
        if kind == "anchored":
            r["e1"] = _reading(x, fits, e1)
        out["slices"][str(s)] = r
    return out


def publish(got: list[dict]) -> dict:
    import json

    from mini.store import put, set_ref

    set_ref(f"reports/m2/{NAME}/results", put(json.dumps({"runs": got}).encode(), name=f"{NAME}.json"))
    return {"n": len(got)}


def main(ctx: Ctx) -> dict:
    rows = runs()
    res = ctx.run(resolve, rows, role="prep")
    got = ctx.map(
        measure,
        [res["artifacts"][r["checkpoint"]] for r in rows],
        [res["holdout"]] * len(rows),
        [r["label"] for r in rows],
        [r["kind"] for r in rows],
        role="measure",
    )
    return ctx.run(publish, got, role="prep")


experiment = Experiment(
    name=NAME,
    main=main,
    roles={
        "prep": dict(cpu=2, timeout=1800),
        "measure": dict(gpu="L40S", cpu=2, timeout=1800, watchdog=900, watchdog_grace=600),
    },
)
