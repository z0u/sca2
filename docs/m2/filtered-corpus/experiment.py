"""A model trained without `difference`: the control with no loss on the `difference` contexts.

`sans-diff` is the 400-epoch control of the anneal confirmation (ex-2.2.23's control: no pull, no anti term) at seeds
712 to 719, trained on the corpus of record with every target inside a `difference` context masked out of the loss
(`loss_mask` of `sca.anchoring.sample_anchored_batches`). The corpus, batches, label draws and schedule are those of
the control at the seed, so the two differ only in the mask. It keeps checkpoints at the epochs the `τ_ℓ fixed` runs
of the fixed confirmation kept, so the distance from the target null can be taken along training for both.

A check arm, `dropped`, trains the same control at two seeds on the corpus with the `difference` contexts taken out,
for the same 400 epochs (about six-sevenths of the steps).

Every finished run is scored with ex-2.2.22's suppression pass, keeping the answer distributions, as the stored runs
it is compared with were. Along training, the clean pass and the full edit along e₁ are scored on each kept
checkpoint of `sans-diff` and of the stored `τ_ℓ fixed` runs.

    bin/mini run docs/m2/filtered-corpus/experiment.py --app modal --max-containers 10 --budget 3h
    bin/mini status filtered-corpus
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from mini import Ctx, Experiment


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


ac = _load_sibling("anneal-confirm", "filtered_corpus_ac")
fc = _load_sibling("fixed-confirm", "filtered_corpus_fc")
ex2222 = ac.ex2222
ex2221 = ac.ex2221
rb = ac.rb

OP_NAMES = ac.OP_NAMES
ANCHORED_OP = ac.ANCHORED_OP
K = ac.K
EPOCHS = ac.EPOCHS
NAME = "sans-diff"
DROPPED = "dropped"

SEEDS: tuple[int, ...] = ac.SEEDS
"""The seeds of the fixed confirmation and of the anneal confirmation's controls, 712 to 719."""
DROPPED_SEEDS: tuple[int, ...] = SEEDS[:2]
KEEP: tuple[int, ...] = tuple(sorted({*ac.STOP_KEEP, *ac.FINISH_KEEP}))
"""The epochs the `τ_ℓ fixed` runs kept: 16, 32 and 64, then every 32 to 384."""

BUDGET_USD = 8
"""From the anneal confirmation on L4s (about \\$0.80 an hour): ten 400-epoch control runs at about twenty minutes each
(\\$2.70); the suppression pass on ten finished runs (\\$1); the clean pass and the full edit on fifteen checkpoints of
sixteen runs (\\$2); and room for a retry."""


def label_of(model_seed: int, name: str = NAME) -> str:
    return f"{name}-s{model_seed}"


# =============================================================================================
# The corpus variants
# =============================================================================================


def prepare_variants(corpus, labels) -> dict:
    """The label arrays of the corpus of record with `loss_mask` set on every token of a `difference` context (the
    corpus itself is unchanged), and a copy of the corpus and its label arrays with those contexts taken out.
    """
    import numpy as np

    from sca.compute.data_pipelines import load_data, save_data
    from mini.store import get, put

    ex2216 = ex2221.ex2216
    workdir = ex2221.get_data_dir() / "filtered-corpus" / "prep"
    tokens, meta = load_data(get(corpus, workdir / "corpus"))
    with np.load(get(labels, workdir / "labels.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    op_ids, context_len = arrays["op_ids"], arrays["context_len"]
    assert int(context_len.sum()) == len(tokens), "the corpus is its contexts end to end"
    diff_ctx = op_ids == OP_NAMES.index(ANCHORED_OP)
    diff_tok = np.repeat(diff_ctx, context_len)

    per_token = {k for k, v in arrays.items() if v.shape == tokens.shape}
    per_context = {k for k, v in arrays.items() if v.shape == op_ids.shape}
    assert per_token | per_context == set(arrays), "every label array is per token or per context"
    masked = arrays | {"loss_mask": diff_tok | arrays.get("loss_mask", np.zeros(len(tokens), dtype=bool))}

    kept = {k: v[~diff_tok] if k in per_token else v[~diff_ctx] for k, v in arrays.items()}
    dropped_tokens = tokens[~diff_tok]
    dropped_meta = meta.model_copy(update={"total_tokens": len(dropped_tokens)})
    save_data(dropped_tokens, dropped_meta, workdir / "dropped")

    return {
        "masked_labels": put(ex2216._npz(**masked), name="filtered-corpus-masked-labels.npz"),
        "dropped_corpus": put(workdir / "dropped", name="filtered-corpus-dropped-corpus"),
        "dropped_labels": put(ex2216._npz(**kept), name="filtered-corpus-dropped-labels.npz"),
        "dropped_meta": dropped_meta,
        "stats": {
            "tokens": len(tokens),
            "masked_tokens": int(diff_tok.sum()),
            "contexts": len(op_ids),
            "difference_contexts": int(diff_ctx.sum()),
            "dropped_tokens": len(dropped_tokens),
        },
    }


def control_rows(meta, seeds: tuple[int, ...], name: str, epochs: int = EPOCHS) -> list[dict]:
    """The anneal confirmation's control rows at *seeds*, relabelled as *name*."""
    return [r | {"kind": name, "label": label_of(r["model_seed"], name)} for r in ac.control_rows(meta, seeds, epochs)]


# =============================================================================================
# Along training
# =============================================================================================


def along_one(kept_checkpoints, checkpoint, holdout, label: str, epochs: tuple[int, ...], end: int) -> list[dict]:
    """On each checkpoint a run kept (after *epochs*) and on its end checkpoint (after *end*): per op, expected exact
    match and the total variation from the target null, on the clean pass and under the full edit along e₁ at every
    position and slice, as ex-2.2.22's suppression pass takes them.
    """
    import numpy as np

    from sca.intervention import Subspace, logits_at, projection
    from mini.store import get

    ex2216 = ex2221.ex2216
    workdir = ex2221.get_data_dir() / "filtered-corpus" / "along" / label
    end_model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2222.ex2218.table_of(OP_NAMES))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    ctx = ex2216._post_queries(P, ho, K, tok2color)
    eq_role = ex2216.query_role(K, ex2216.QUERY_EQ)
    mask = ex2216.site_positions("every position", K)
    a_id = OP_NAMES.index(ANCHORED_OP)
    q_null = P.predictive(table, ctx, ex2216._target_null(P, table, ctx, ho.posterior, a_id))
    width = end_model.transformer.wte.shape[1]
    every = tuple(range(len(end_model.transformer.blocks) + 1))
    full = rb.DOSE_GAMMAS[-1]

    def per_op(v: np.ndarray) -> list[float]:
        return [float(v[ho.op_ids == o].mean()) for o in range(len(OP_NAMES))]

    def scored(model, epoch: int) -> dict:
        out: dict[str, Any] = {"label": label, "epoch": epoch}
        for name, op, slices in (("clean", projection(Subspace.axis(width), 0.0), ()),
                                 ("full", projection(Subspace.axis(width), full), every)):  # fmt: skip
            lg = logits_at(model, ho.tokens, op, slices, eq_role, mask if slices else None)
            p = ex2216._color_probs(lg, color_ids)
            out[name] = {
                "eem": per_op(ex2216._match(table, ctx, p)),
                "tv_null": per_op(ex2222.total_variation(q_null, p)),
            }
        return out

    out = [scored(ex2221.kept_checkpoint(kept_checkpoints, e, workdir / "kept")[0], e) for e in epochs]
    return out + [scored(end_model, end)]


def train(ctx: Ctx, sources: list[dict], rows: list[dict], keep: list[tuple[int, ...]]) -> list[dict]:
    """The anneal confirmation's `train` on every row in one fan-out, each row with its own corpus and label arrays
    (*sources*, one resolved condition per row).
    """
    n = len(rows)
    epoch_length = [ex2222.ex2217.epoch_length_of(s["meta"].total_tokens, r["config"]) for s, r in zip(sources, rows, strict=True)]  # fmt: skip
    stride = [max(1, round(ac.sps.TRAJ_STRIDE_EPOCHS * e)) for e in epoch_length]
    return ctx.map(
        ex2221.train_one,
        [r["config"] for r in rows],
        [r["anchor"] for r in rows],
        [r["anti"] for r in rows],
        [r.get("variant", "whole") for r in rows],
        [r["anchor_slices"] for r in rows],
        [r.get("pull", "") for r in rows],
        [OP_NAMES] * n,
        [s["corpus"] for s in sources],
        [s["labels"] for s in sources],
        [s["probes"] for s in sources],
        [s["holdout"] for s in sources],
        [K] * n,
        stride,
        [ex2221.N_TRAJ_EEM_PER_OP] * n,
        [r["label"] for r in rows],
        [ac.sps.EVERY] * n,
        [None] * n,
        [True] * n,
        [None] * n,
        [None] * n,
        keep,
        [None] * n,
        role="train",
    )


def resolve_fixed(seeds: tuple[int, ...]) -> list[dict]:
    """The end checkpoints, kept checkpoints and kept epochs of the fixed confirmation's `τ_ℓ fixed` runs at *seeds*."""
    from mini.store import get_ref

    out = []
    for s in seeds:
        label = fc.label_of(s, None)
        checkpoint, kept = get_ref(fc.CHECKPOINT_REF.format(label=label)), get_ref(fc.KEPT_REF.format(label=label))
        assert checkpoint is not None and kept is not None, f"fixed-confirm has not published {label}"
        out.append({"label": label, "model_seed": s, "checkpoint": checkpoint, "kept_checkpoints": kept})
    return out


# =============================================================================================
# The DAG
# =============================================================================================

PREFIX = "reports/m2/filtered-corpus"
VARIANTS_REF = PREFIX + "/variants"
CHECKPOINT_REF = PREFIX + "/checkpoints/{label}"
KEPT_REF = PREFIX + "/kept/{label}"
TRAJ_REF = PREFIX + "/trajectories"
DOSE_REF = PREFIX + "/dose"
DOSE_ARRAYS_REF = PREFIX + "/dose-arrays/{label}"
ALONG_REF = PREFIX + "/along"


def design() -> dict[str, Any]:
    return {
        "experiment": "filtered-corpus",
        "masked_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "epochs": EPOCHS,
        "seeds": list(SEEDS),
        "dropped_seeds": list(DROPPED_SEEDS),
        "keep": list(KEEP),
        "control_of": "anneal-confirm",
        "reference": "fixed-confirm",
    }


def publish(rows: list[dict], trained: list[dict], dosed: list[dict], along: list[list[dict]], stats: dict) -> dict:
    """Trajectories, checkpoints, the suppression pass and the scoring along training, under refs of their own."""
    import json

    from mini.store import put, set_ref

    meta = {r["label"]: {"kind": r["kind"], "model_seed": r["model_seed"]} for r in rows}
    meta |= {fc.label_of(s, None): {"kind": "fixed", "model_seed": s} for s in SEEDS}
    for t in trained:
        set_ref(CHECKPOINT_REF.format(label=t["label"]), t["checkpoint"])
        if "kept_checkpoints" in t:
            set_ref(KEPT_REF.format(label=t["label"]), t["kept_checkpoints"])
    traj = {t["label"]: meta[t["label"]] | {k: t[k] for k in ("traj", "val_loss", "train_loss")} for t in trained}
    set_ref(TRAJ_REF, put(json.dumps(traj).encode(), name="filtered-corpus-trajectories.json"))

    def body(runs: list[dict]) -> bytes:
        return json.dumps({"design": design(), "stats": stats, "runs": [meta[r["label"]] | r for r in runs]}).encode()

    for d in dosed:
        set_ref(DOSE_ARRAYS_REF.format(label=d["label"]), d["arrays"])
    set_ref(DOSE_REF, put(body([{k: v for k, v in d.items() if k != "arrays"} for d in dosed]), name="filtered-corpus-dose.json"))  # fmt: skip
    set_ref(ALONG_REF, put(body([r for a in along for r in a]), name="filtered-corpus-along.json"))
    return {"n_trained": len(trained), "n_along": len(along)}


def run(ctx: Ctx, epochs: int = EPOCHS, seeds: tuple[int, ...] = SEEDS, dropped: tuple[int, ...] = DROPPED_SEEDS,
        keep: tuple[int, ...] = KEEP, fixed: tuple[int, ...] = SEEDS, publishes: bool = True) -> dict:  # fmt: skip
    """Build the two corpus variants, train `sans-diff` and `dropped`, score their end checkpoints with the
    suppression pass, and score the kept checkpoints of `sans-diff` and of the `τ_ℓ fixed` runs at *fixed*.
    """
    resolved = ctx.run(ex2222.resolve_reused, [], role="prep")
    variants = ctx.run(prepare_variants, resolved["corpus"], resolved["labels"], role="prep")
    masked = resolved | {"labels": variants["masked_labels"]}
    cut = resolved | {
        "corpus": variants["dropped_corpus"],
        "labels": variants["dropped_labels"],
        "meta": variants["dropped_meta"],
    }
    sans_rows = control_rows(resolved["meta"], seeds, NAME, epochs)
    dropped_rows = control_rows(variants["dropped_meta"], dropped, DROPPED, epochs)
    rows = sans_rows + dropped_rows
    sources = [masked] * len(sans_rows) + [cut] * len(dropped_rows)
    done = train(ctx, sources, rows, [keep] * len(sans_rows) + [()] * len(dropped_rows))
    sans = done[: len(sans_rows)]

    n = len(rows)
    labels = [r["label"] for r in rows]
    checkpoints = [t["checkpoint"] for t in done]
    dosed = ctx.map(ex2222.suppress_one, checkpoints, [resolved["holdout"]] * n, [OP_NAMES] * n, [K] * n, [True] * n, labels, role="measure")  # fmt: skip

    ref = ctx.run(resolve_fixed, fixed, role="prep")
    along_runs = [(t["kept_checkpoints"], t["checkpoint"], t["label"], tuple(t["kept_epochs"])) for t in sans]
    along_runs += [(f["kept_checkpoints"], f["checkpoint"], f["label"], KEEP) for f in ref]
    m = len(along_runs)
    along = ctx.map(
        along_one,
        [a[0] for a in along_runs],
        [a[1] for a in along_runs],
        [resolved["holdout"]] * m,
        [a[2] for a in along_runs],
        [a[3] for a in along_runs],
        [epochs] * len(sans) + [EPOCHS] * len(ref),
        role="measure",
    )
    if not publishes:
        return {"trained": labels, "dosed": [d["label"] for d in dosed], "along": [len(a) for a in along], "stats": variants["stats"]}  # fmt: skip
    return ctx.run(publish, rows, done, dosed, along, variants["stats"], role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed of each arm for eight epochs, keeping two checkpoints, scored, with no `τ_ℓ fixed` runs and no refs."""
    return run(ctx, epochs=8, seeds=SEEDS[:1], dropped=SEEDS[:1], keep=(2, 4), fixed=(), publishes=False)


COMPUTE = ac.COMPUTE

experiment = Experiment(name="filtered-corpus", main=main, roles=COMPUTE)
