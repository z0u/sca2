"""The recipe of record and its control with Adam's basis rotated: does anchoring on e₁ depend on e₁ being one of
the optimizer's privileged axes?

The coordinate axes of the residual stream are a privileged basis because Adam's per-coordinate scaling is written in
them (the memo of 2026-10-23 in the project files): every run parks its syntax tokens one-hot on a coordinate of its
own, and e₁ is one of the 64 directions that are cheap to hold that way. Here `falloff` (split-anti-seeds' recipe of
record) and the un-anchored control train again at the same eight seeds with `basis_seed` set, so Adam's scaling acts
in a random rotation of the residual coordinates and e₁ is a direction the optimizer does not favor. The model, the
initialization, the batches, the anchoring code and the measurements are unchanged, so each new run pairs with its
stored twin (split-anti-seeds' `falloff` and ex-2.2.23's control at the same seed) by everything but the optimizer's
basis. Beside the usual measurements and dose scoring, a basis measurement reads the energy spectrum of the stream
along the coordinate axes and along the rotated basis, and the removal cost of e₁ next to the rotated axes.

    bin/mini run docs/m2/rotated-basis/experiment.py --app modal --max-containers 16 --budget 4h
    bin/mini status rotated-basis
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir


def _load_sibling(name: str, alias: str):
    """A sibling experiment's module, by path and left out of `sys.modules`, as the ex-2.2 experiments do."""
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


sas = _load_sibling("split-anti-seeds", "rotated_basis_sas")
st = sas.st
lean = sas.lean
ex2223 = sas.ex2223
ex2222 = sas.ex2222
ex2221 = sas.ex2221
ex2216 = lean.ex2216
ex2218 = lean.ex2218

OP_NAMES = sas.OP_NAMES
ANCHORED_OP = sas.ANCHORED_OP
K = sas.K
EPOCHS = sas.EPOCHS
EVERY = sas.EVERY
LATE = sas.LATE
TRAJ_STRIDE_EPOCHS = sas.TRAJ_STRIDE_EPOCHS
FALLOFF_HOLDS = sas.FALLOFF_HOLDS
SEEDS: tuple[int, ...] = sas.SEEDS
"""Model seeds 704 to 711: split-anti-seeds' `falloff` and ex-2.2.23's 400-epoch controls are stored there."""


@dataclass(frozen=True)
class Arm:
    name: str
    anchored: bool
    shown_as: str


ARMS: tuple[Arm, ...] = (
    Arm("falloff", True, "falloff, rotated basis"),
    Arm("control", False, "control, rotated basis"),
)

N_RANDOM = 16
"""Random unit directions scored by the basis measurement beside the axes."""

BUDGET_USD = 8
"""Split-anti-seeds cost about \\$0.3 per 400-epoch run with its measurements: 16 runs come to about \\$5, and the
dose and basis scoring of 16 checkpoints about \\$1."""


def label_of(arm: str, model_seed: int) -> str:
    return f"{arm}-s{model_seed}"


def basis_of(model_seed: int, width: int) -> np.ndarray:
    """The rotation Adam acts in for the run at *model_seed*: `sca.training.optimizer.configure_optimizer` draws it
    from `basis_seed` this way, one basis vector per column.
    """
    return np.linalg.qr(np.random.default_rng(model_seed).standard_normal((width, width)))[0]


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/rotated-basis"
TRAJ_REF = PREFIX + "/trajectories"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
RESULTS_REF = PREFIX + "/results"
DOSE_REF = PREFIX + "/dose"
DOSE_ARRAYS_REF = PREFIX + "/dose-arrays/{label}"
BASIS_REF = PREFIX + "/basis"


def rows_of(meta, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """One row per run: split-anti-seeds' `falloff` row and ex-2.2.21's control row at each seed, each with Adam's
    basis rotated by a draw from the model seed.
    """
    (falloff,) = (a for a in sas.ARMS if a.name == "falloff")
    rows = []
    for a in ARMS:
        if a.anchored:
            built = sas.rows_of(meta, epochs, seeds)
            built = [r for r in built if r["arm"]["name"] == falloff.name]
            for r in built:
                r["anchor_slices"], r["variant"], r["pull"], r["anti_slices"] = LATE, "whole", "", EVERY
        else:
            from dataclasses import replace

            n_seeds = max(seeds) - ex2221.SEED_OFFSET + 1
            arm = (replace(ex2221.arm(ex2221.CONTROL), seeds=n_seeds),)
            built = [
                r
                for r in ex2221.cells(arm, {ex2221.MAIN_KEY: {"meta": meta}}, None, epochs)
                if r["model_seed"] in seeds
            ]
            assert len(built) == len(seeds), (len(built), len(seeds))
            for r in built:
                r["anti_slices"] = None
        for r in built:
            r["config"].optimizer.basis_seed = r["model_seed"]
            r["arm"] = asdict(a)
            r["label"] = label_of(a.name, r["model_seed"])
        rows.extend(built)
    return rows


def basis_one(checkpoint, holdout, label: str, model_seed: int, n_ctx: int = 4000) -> dict:
    """The stream's energy spectrum along the coordinate axes and along the rotated basis (the share of the energy
    each direction holds, at the final slice and averaged over slices), the mean kurtosis along each, and the
    removal cost (the full projection at every slice and position, as the dose scoring applies it; the drop in
    expected exact match per op) of e₁, of every coordinate axis, of the rotated basis's axes, and of random
    directions, on *n_ctx* held-out completion contexts.
    """
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get

    workdir = get_data_dir() / "basis" / label
    model, tok, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(OP_NAMES))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(ho.tokens), min(n_ctx, len(ho.tokens)), replace=False))
    tokens, op_ids = ho.tokens[pick], ho.op_ids[pick]
    picked = ex2216.Holdout(
        tokens, op_ids, ho.posterior[pick], ho.ceiling[pick], ho.verify_tokens, ho.verify_op_ids, ho.verdict
    )
    ctx = ex2216._post_queries(P, picked, K, tok2color)
    C = model.transformer.wte.shape[1]
    R = basis_of(model_seed, C)
    n_ops = len(OP_NAMES)

    def kurt(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(0)
        v = (x**2).mean(0)
        return (x**4).mean(0) / np.maximum(v, 1e-12) ** 2

    stream = np.asarray(model.residual_stream(tokens[:1000]))  # (L1, N, T, C)
    energy_axes, energy_rot, kurt_axes, kurt_rot = [], [], [], []
    for s in range(stream.shape[0]):
        h = stream[s].reshape(-1, C)
        energy_axes.append((h**2).mean(0))
        energy_rot.append(((h @ R) ** 2).mean(0))
        kurt_axes.append(kurt(h).mean())
        kurt_rot.append(kurt(h @ R).mean())

    every = tuple(range(stream.shape[0]))
    mask = ex2216.site_positions("every position", K)
    eq_role = ex2216.query_role(K, ex2216.QUERY_EQ)

    def per_op(v: np.ndarray) -> list[float]:
        return [float(v[op_ids == o].mean()) for o in range(n_ops)]

    def cost(sub: Subspace, clean: np.ndarray) -> list[float]:
        lg = logits_at(model, tokens, projection(sub, 1.0), every, eq_role, mask)
        return per_op(clean - ex2216._match(table, ctx, ex2216._color_probs(lg, color_ids)))

    lg = logits_at(model, tokens, projection(Subspace.axis(C), 0.0), (), eq_role)
    clean = ex2216._match(table, ctx, ex2216._color_probs(lg, color_ids))
    dirs = rng.standard_normal((N_RANDOM, C))
    return {
        "label": label,
        "model_seed": model_seed,
        "n_ctx": int(len(tokens)),
        "clean": per_op(clean),
        "energy_axes": np.stack(energy_axes).tolist(),  # (L1, C): each row sums to 1
        "energy_rotated": np.stack(energy_rot).tolist(),
        "kurt_axes": kurt_axes,
        "kurt_rotated": kurt_rot,
        "cost_axes": [cost(Subspace.axis(C, i), clean) for i in range(C)],  # e₁ first
        "cost_rotated": [cost(Subspace.direction(R[:, i]), clean) for i in range(C)],
        "cost_random": [cost(Subspace.direction(d), clean) for d in dirs],
        # Where each token embedding sits: its largest cosine with a coordinate axis, and with a rotated axis, and
        # which axis that is, for the tokens the corpus uses (the color words and the syntax).
        **embedding_homes(model, tok, R),
    }


def embedding_homes(model, tok, R: np.ndarray) -> dict:
    """For each token in the vocabulary, the largest absolute cosine of its (unit) embedding with a coordinate axis
    and with a column of *R*, and the index of that axis; a token one-hot on an axis has a cosine near 1.
    """
    from sca.model._shared import normalize

    emb = np.asarray(normalize(model.transformer.wte))
    used = sorted(tok.stoi.values())
    on_axes, on_rot = np.abs(emb[used]), np.abs(emb[used] @ R)
    return {
        "tokens": [tok.itos[i] for i in used],
        "emb_axis_max": on_axes.max(1).tolist(),
        "emb_axis_arg": on_axes.argmax(1).tolist(),
        "emb_rotated_max": on_rot.max(1).tolist(),
        "emb_rotated_arg": on_rot.argmax(1).tolist(),
    }


def design() -> dict[str, Any]:
    return {
        "experiment": "rotated-basis",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "arms": [asdict(a) for a in ARMS],
        "falloff_holds": list(FALLOFF_HOLDS),
        "seeds": list(SEEDS),
        "basis_seed": "the model seed",
        "slice_sets": {k: list(v) for k, v in lean.SLICE_SETS.items()},
        "dose_gammas": list(ex2222.DOSE_GAMMAS),
        "traj_stride_epochs": TRAJ_STRIDE_EPOCHS,
        "n_random": N_RANDOM,
    }


def publish(rows: list[dict], trained: list[dict], scored: list[dict], dosed: list[dict], based: list[dict]) -> dict:
    """The trajectories, every checkpoint, the measurements, the dose scoring with its per-context arrays, and the
    basis measurement, each under a ref of its own.
    """
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"arm": r["arm"]["name"], "model_seed": r["model_seed"]} for r in rows}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="rotated-basis-trajectories.json"))
    body = {"design": design(), "runs": [meta[r["label"]] | r for r in scored]}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="rotated-basis-results.json"))
    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    dose = {
        "design": design(),
        "runs": [meta[d["label"]] | {k: v for k, v in d.items() if k != "arrays"} for d in dosed],
    }
    set_ref(DOSE_REF, put(json.dumps(dose).encode(), name="rotated-basis-dose.json"))
    basis = {"design": design(), "runs": [meta[b["label"]] | b for b in based]}
    set_ref(BASIS_REF, put(json.dumps(basis).encode(), name="rotated-basis-basis.json"))
    return {"n_runs": len(scored), "n_dosed": len(dosed)}


def run(ctx: Ctx, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS, publishes: bool = True) -> dict:
    """Resolve ex-2.2.21's corpus condition, train every run, measure it, score it at every dose, take the basis
    measurement, and publish. *epochs*, *seeds* and *publishes* let a smoke run take the same path at a fraction of
    the size, and leave the refs alone.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    rows = rows_of(resolved["meta"], epochs, seeds)
    n = len(rows)
    epoch_length = [ex2222.ex2217.epoch_length_of(resolved["meta"].total_tokens, r["config"]) for r in rows]
    stride = [max(1, round(TRAJ_STRIDE_EPOCHS * e)) for e in epoch_length]
    trained = ctx.map(
        ex2221.train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r["variant"] for r in rows],
        [r["anchor_slices"] for r in rows],
        [r["pull"] for r in rows],
        [OP_NAMES] * n,
        [resolved["corpus"]] * n,
        [resolved["labels"]] * n,
        [resolved["probes"]] * n,
        [resolved["holdout"]] * n,
        [K] * n,
        stride,
        [ex2221.N_TRAJ_EEM_PER_OP] * n,
        [r["label"] for r in rows],
        [r["anti_slices"] for r in rows],
        [None] * n,
        [True] * n,
        role="train",
    )
    scored = ctx.map(
        lean.measure_one,
        [t["checkpoint"] for t in trained],
        [resolved["holdout"]] * n,
        [r["label"] for r in rows],
        [True] * n,
        role="measure",
    )
    dosed = ctx.map(
        ex2222.suppress_one,
        [t["checkpoint"] for t in trained],
        [resolved["holdout"]] * n,
        [OP_NAMES] * n,
        [K] * n,
        [True] * n,
        [r["label"] for r in rows],
        role="measure",
    )
    based = ctx.map(
        basis_one,
        [t["checkpoint"] for t in trained],
        [resolved["holdout"]] * n,
        [r["label"] for r in rows],
        [r["model_seed"] for r in rows],
        role="measure",
    )
    if not publishes:
        return {
            "labels": [s["label"] for s in scored],
            "dosed": [d["label"] for d in dosed],
            "based": [b["label"] for b in based],
        }
    return ctx.run(publish, rows, trained, scored, dosed, based, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed of each arm for eight epochs (past the five-epoch warm-up), through the whole DAG."""
    return run(ctx, epochs=8, seeds=SEEDS[:1], publishes=False)


COMPUTE = sas.COMPUTE

experiment = Experiment(name="rotated-basis", main=main, roles=COMPUTE)
