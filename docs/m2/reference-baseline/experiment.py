"""A baseline that does not depend on where e₁ sits in the control: the stored `falloff` runs and their controls,
scored again with the edit along every coordinate axis.

Removal and spill have been netted against the drop the e₁ edit causes in the control at the same seed. That is one
draw of a direction in the control, and under a rotated optimizer basis (rotated-basis, E2) it drew badly on two
seeds. The candidate baseline is the median drop over edits along other coordinate axes, off the two axes where the
stream holds the syntax tokens (spill-timing scores its reference edit the same way). This experiment trains nothing:
it takes split-anti-seeds' `falloff` runs at 704 to 711 and ex-2.2.23's 400-epoch controls at the same seeds, and
scores the dose edit along each of the 64 coordinate axes, at every dose, on the whole held-out set. Beside it, each
model's energy spectrum along the axes, which is how the syntax axes are found. The rotated-basis runs (`falloff` and
the control at the same seeds) are scored the same way, along the coordinate axes and along the rotated basis.

    bin/mini run docs/m2/reference-baseline/experiment.py --app modal --max-containers 16 --budget 1h
    bin/mini status reference-baseline
"""

from __future__ import annotations

import importlib.util
import sys
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


sas = _load_sibling("split-anti-seeds", "reference_baseline_sas")
rb = _load_sibling("rotated-basis", "reference_baseline_rb")
sp = _load_sibling("soft-pool", "reference_baseline_sp")
lean = sas.lean
ex2222 = sas.ex2222
ex2216 = lean.ex2216
ex2218 = lean.ex2218

OP_NAMES = sas.OP_NAMES
ANCHORED_OP = sas.ANCHORED_OP
K = sas.K
SEEDS: tuple[int, ...] = sas.SEEDS
"""Model seeds 704 to 711, where split-anti-seeds' `falloff` and ex-2.2.23's controls are both stored."""
DOSE_GAMMAS = ex2222.DOSE_GAMMAS
ARM = "falloff"
"""The recipe of record."""

N_ENERGY_CTX = 1000
"""Held-out contexts the energy spectrum is taken over."""

BUDGET_USD = 4
"""One dose scoring is five passes over the held-out set; this is 257 per checkpoint (clean, then 64 axes at four
doses), and 513 for a rotated-basis run, a few minutes on an L4 by spill-timing's timing, so 32 checkpoints come to
about a dollar of GPU."""


def label_of(arm: str, model_seed: int) -> str:
    return f"{arm}-s{model_seed}"


def axes_one(checkpoint, holdout, label: str, basis_seed: int | None = None) -> dict:
    """The dose edit (the projection at every position and slice, as ex-2.2.22's dose scoring applies it) along each
    coordinate axis at each dose, on every held-out completion context. Per op: expected exact match on the clean
    pass and for the target null, and the drop under each axis and dose (`drop[axis][dose][op]`, e₁ first). Beside it,
    the share of the energy of the stream that each axis holds, per slice.

    With *basis_seed*, for a run of rotated-basis, the same along the rotated basis Adam acted in (`drop_rotated`,
    `energy_rotated`), beside the coordinate axes.
    """
    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get

    workdir = get_data_dir() / "reference-baseline" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2218.table_of(OP_NAMES))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    ctx = ex2216._post_queries(P, ho, K, tok2color)
    eq_role = ex2216.query_role(K, ex2216.QUERY_EQ)
    a_id = OP_NAMES.index(ANCHORED_OP)
    width = model.transformer.wte.shape[1]
    every = tuple(range(len(model.transformer.blocks) + 1))
    mask = ex2216.site_positions("every position", K)
    n_ops = len(OP_NAMES)

    def per_op(v: np.ndarray) -> list[float]:
        return [float(v[ho.op_ids == o].mean()) for o in range(n_ops)]

    def score(op, slices) -> np.ndarray:
        lg = logits_at(model, ho.tokens, op, slices, eq_role, mask if slices else None)
        return ex2216._match(table, ctx, ex2216._color_probs(lg, color_ids))

    clean = score(projection(Subspace.axis(width), 0.0), ())
    null_post = ex2216._target_null(P, table, ctx, ho.posterior, a_id)
    null = P.expected_match(table, ctx, null_post)
    stream = np.asarray(model.residual_stream(ho.tokens[:N_ENERGY_CTX]))  # (L1, N, T, C)

    def scored(basis: np.ndarray) -> tuple[list, list]:
        """The drop along each column of *basis* at each dose, and the share of the energy each holds per slice."""
        drop = [
            [per_op(clean - score(projection(Subspace.direction(basis[:, i]), g), every)) for g in DOSE_GAMMAS]
            for i in range(width)
        ]
        energy = ((stream @ basis) ** 2).mean(axis=(1, 2))
        return drop, (energy / energy.sum(axis=1, keepdims=True)).tolist()  # (L1, C): each row sums to 1

    drop, energy = scored(np.eye(width))
    out = {
        "label": label,
        "n_ctx": int(len(ho.tokens)),
        "clean": per_op(clean),
        "null": per_op(null),
        "drop": drop,
        "energy": energy,
    }
    if basis_seed is not None:
        out["drop_rotated"], out["energy_rotated"] = scored(rb.basis_of(basis_seed, width))
    return out


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/reference-baseline"
AXES_REF = PREFIX + "/axes"


SOFT_POOL = "soft-pool-"
"""The label prefix of soft-pool's runs here."""
SOFT_POOL_ARMS = ("step-soft", "pool-123-soft")
"""Soft-pool's arms near the spill criterion (2 and 4 of 8 runs over it, against e₁ in the control)."""
ROTATED = "rotated-"
"""The label prefix of rotated-basis' runs here, which share their labels with the stored runs there."""


def resolve_checkpoints(seeds: tuple[int, ...]) -> dict:
    """The stored `falloff` checkpoints of split-anti-seeds and ex-2.2.23's controls at *seeds*, and the rotated-basis
    runs of both and soft-pool's two arms near the spill criterion at the same seeds, by ref.
    """
    from mini.store import get_ref

    refs = {label_of(ARM, s): sas.CHECKPOINT_REF.format(label=label_of(ARM, s)) for s in seeds} | {
        ROTATED + rb.label_of(a, s): rb.CHECKPOINT_REF.format(label=rb.label_of(a, s))
        for a in (ARM, "control")
        for s in seeds
    }
    refs |= {
        SOFT_POOL + sp.label_of(a, s): sp.CHECKPOINT_REF.format(label=sp.label_of(a, s))
        for a in SOFT_POOL_ARMS
        for s in seeds
    }
    found = {label: get_ref(ref) for label, ref in refs.items()}
    missing = [label for label, a in found.items() if a is None]
    assert not missing, f"not published: {missing}"
    return found | sas.resolve_controls(seeds)


def basis_seed_of(label: str) -> int | None:
    """The basis seed of a rotated-basis run (its model seed), else None."""
    return int(label.rsplit("-s", 1)[1]) if label.startswith(ROTATED) else None


def design() -> dict[str, Any]:
    return {
        "experiment": "reference-baseline",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "arm": ARM,
        "seeds": list(SEEDS),
        "rotated_prefix": ROTATED,
        "soft_pool_prefix": SOFT_POOL,
        "soft_pool_arms": list(SOFT_POOL_ARMS),
        "dose_gammas": list(DOSE_GAMMAS),
        "n_energy_ctx": N_ENERGY_CTX,
    }


def publish(scored: list[dict]) -> dict:
    import json

    from mini.store import put, set_ref

    body = {"design": design(), "runs": scored}
    set_ref(AXES_REF, put(json.dumps(body).encode(), name="reference-baseline-axes.json"))
    return {"n_scored": len(scored)}


def run(ctx: Ctx, seeds: tuple[int, ...] = SEEDS, publishes: bool = True) -> dict:
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    checkpoints = ctx.run(resolve_checkpoints, seeds, role="prep")
    labels = list(checkpoints)
    n = len(labels)
    scored = ctx.map(
        axes_one,
        [checkpoints[label] for label in labels],
        [resolved["holdout"]] * n,
        labels,
        [basis_seed_of(label) for label in labels],
        role="measure",
    )
    if not publishes:
        return {"labels": [s["label"] for s in scored]}
    return ctx.run(publish, scored, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed, both checkpoints, through the whole DAG, leaving the refs alone."""
    return run(ctx, seeds=SEEDS[:1], publishes=False)


COMPUTE = sas.COMPUTE

experiment = Experiment(name="reference-baseline", main=main, roles=COMPUTE)
