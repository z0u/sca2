"""How the edit changes the prediction at every position of a context, on the recipe of record, with no new training.

Every published score of the edit reads one site, the query `=`. This pass reads them all: for each held-out completion
context it runs the clean model and the full edit along e₁ (every slice, every position), and keeps the surprisal of
the token that comes next at every position, and how far the edited next-token distribution moves from the clean one
(total variation over the whole vocabulary). At the four answer sites (the `=` of each example and of the query) it
also keeps the distance of the answer distribution from the Bayes predictive and from the target null, where the
posterior over ops is the one the examples so far support. At an example answer the predictive includes the noise the
corpus draws examples with (replacement op noise and cube noise); the query answer is always clean.

Three kinds of run at seeds 712 to 719, all on ex-2.2.21's held-out set: `τ_ℓ fixed` of the fixed confirmation (the
recipe of record), the anneal confirmation's controls at the same seeds, and `sans-diff` of the filtered-corpus
experiment (the control trained with no loss on `difference` contexts). Each is scored clean and under the same edit.

    bin/mini run docs/m2/edit-by-role/experiment.py
    bin/mini status edit-by-role
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


fcorp = _load_sibling("filtered-corpus", "edit_by_role_fcorp")
ac, fc, ex2222, ex2221 = fcorp.ac, fcorp.fc, fcorp.ex2222, fcorp.ex2221
ex2216 = ex2222.ex2216

OP_NAMES = fcorp.OP_NAMES
ANCHORED_OP = fcorp.ANCHORED_OP
K = fcorp.K
SEEDS = fcorp.SEEDS
RHO = ex2222.RHO
CUBE_RATE = ex2222.CUBE_RATE
N_SLICES = ex2222.N_LAYER + 1
UNIT = 6
"""Tokens per equation: the first operand, `?`, the second operand, `=`, the answer, and the separator."""
ROLES = ("first operand", "?", "second operand", "=", "answer", "separator")
"""The role of the token predicted, by its offset within an equation."""
BATCH = 1000
"""Contexts per forward pass."""
ANSWER_SITES = np.array([UNIT * e + 3 for e in range(K + 1)])
"""The positions that predict an answer: the `=` of each example, then of the query."""

KINDS = ("fixed", "control", "sans-diff")

PREFIX = "reports/m2/edit-by-role"
RESULTS_REF = PREFIX + "/results"
ARRAYS_REF = PREFIX + "/arrays/{label}"


def runs() -> list[dict]:
    """The three kinds of run at each seed, with the ref of each end checkpoint."""
    out = []
    for s in SEEDS:
        for kind, ref in (
            ("fixed", fc.CHECKPOINT_REF.format(label=fc.label_of(s, None))),
            ("control", ac.CHECKPOINT_REF.format(label=ac.label_of(s, None, "control"))),
            ("sans-diff", fcorp.CHECKPOINT_REF.format(label=fcorp.label_of(s))),
        ):
            out.append({"label": f"{kind}-s{s}", "kind": kind, "model_seed": s, "ref": ref})
    return out


def resolve(refs: list[str]) -> dict:
    """The held-out set of ex-2.2.21 and the checkpoint behind each ref."""
    from mini.store import get_ref

    holdout = get_ref(ex2221.HOLDOUT_REF.format(key=ex2222.MAIN_KEY))
    checkpoints = {r: get_ref(r) for r in refs}
    missing = [r for r, a in checkpoints.items() if a is None]
    assert holdout is not None and not missing, f"not published: {missing}"
    return {"holdout": holdout, "checkpoints": checkpoints}


def predictives(P, table, ho, tok2color: np.ndarray, a_id: int) -> dict[str, np.ndarray]:
    """The Bayes predictive and the target null at each answer site, dense over the palette: `(N, K + 1, 216)` each,
    with the shown answers and the posterior before each answer.

    Before example e the posterior is the one the e examples so far support (uniform before the first). An example
    answer is drawn with replacement op noise and cube noise, so its predictive mixes those in as the corpus does;
    the query answer is drawn from the true op alone. The target null zeroes the anchored op in the posterior and
    renormalizes (a posterior that underflowed to all on the anchored op goes to the uniform over the others).
    """
    n_colors = len(tok2color[tok2color >= 0])
    answers = ANSWER_SITES + 1
    y = tok2color[ho.tokens[:, answers]]
    a, b = tok2color[ho.tokens[:, answers - 4]], tok2color[ho.tokens[:, answers - 2]]
    assert (y >= 0).all() and (a >= 0).all() and (b >= 0).all(), "operands and answers are colors"
    pair = a * n_colors + b
    examples = P.Contexts(ho.op_ids.astype(np.int64), pair[:, :K], y[:, :K], pair[:, K])
    prefix = P.prefix_posteriors(table, examples, RHO, CUBE_RATE)  # (N, K, ops)
    assert np.allclose(prefix[:, -1], ho.posterior, atol=1e-5), "the prefix posterior ends at the stored one"
    n_ops = table.n_ops
    uniform = np.full((len(y), 1, n_ops), 1.0 / n_ops)
    before = np.concatenate([uniform, prefix], axis=1)  # (N, K + 1, ops): the posterior before each answer

    def null_of(post: np.ndarray) -> np.ndarray:
        out = post.copy()
        out[:, a_id] = 0.0
        total = out.sum(axis=1, keepdims=True)
        lost = total[:, 0] <= 0
        out[lost] = 1.0
        out[lost, a_id] = 0.0
        return out / out.sum(axis=1, keepdims=True)

    bayes, null = [], []
    for e in range(K + 1):
        ctx = P.Contexts(examples.true_op, examples.ex_pair, examples.ex_color, pair[:, e].astype(np.int64))
        every = P.predictive(table, ctx, np.ones((len(y), n_ops)))  # Σ_o P_o(y | pair)
        qs = []
        for post in (before[:, e], null_of(before[:, e])):
            q = P.predictive(table, ctx, post)
            if e < K:
                q = (1 - RHO - CUBE_RATE) * q + RHO * (every - q) / (n_ops - 1) + CUBE_RATE / n_colors
            qs.append(q)
        bayes.append(qs[0])
        null.append(qs[1])
    return {"bayes": np.stack(bayes, axis=1), "null": np.stack(null, axis=1), "answer": y, "before": before}


def measure_one(checkpoint, holdout, label: str) -> dict:
    """The clean pass, and the full edit along each coordinate axis at every slice and position, on every held-out
    completion context. Axis 0 is e₁, the anchored axis; the median over the others is the baseline of the D2.2 design.

    For each pass, per op and position: the mean surprisal (nats) of the next token; the mean total variation between
    the clean and the edited next-token distributions; and at the answer sites, the mean total variation of the answer
    distribution from the Bayes predictive and from the target null (mass on tokens that are not colors counts as
    distance). The per-context arrays of the clean pass and the e₁ edit go to the store, with the surprisal of each
    shown answer under the two predictors and the posterior on the anchored op before each answer.
    """

    import jax
    import jax.numpy as jnp

    from mini.store import get, put
    from sca.intervention import Subspace, _forward_jit, projection

    workdir = get_data_dir() / "edit-by-role" / label
    model, _, color_ids, tok2color = ex2216._load_run(checkpoint, workdir)
    P = ex2216._get_posterior()
    table = P.build_table(ex2222.ex2218.table_of(OP_NAMES))
    ho = ex2216.load_holdout(get(holdout, workdir / "holdout.npz"), K)
    a_id = OP_NAMES.index(ANCHORED_OP)
    assert ho.tokens.shape[1] == UNIT * (K + 1), "a context is K examples and a query, each one equation"

    width = model.transformer.wte.shape[1]
    every = jnp.ones(ho.tokens.shape[1], dtype=jnp.float32)
    slices = tuple(range(N_SLICES))
    colors = jnp.asarray(color_ids)
    sites = jnp.asarray(ANSWER_SITES)
    ref = predictives(P, table, ho, tok2color, a_id)
    batches = range(0, len(ho.tokens), BATCH)

    @jax.jit
    def reduce(lc, logits, nxt):
        """The surprisal of the next token, the total variation from the clean distribution *lc* (log-probs), and
        the answer distribution over the palette at the answer sites.
        """
        lp = jax.nn.log_softmax(logits, -1)
        s = -jnp.take_along_axis(lp, nxt[..., None], -1)[..., 0]
        tv = 0.5 * jnp.abs(jnp.exp(lc) - jnp.exp(lp)).sum(-1)
        return s, tv, jnp.exp(lp[:, sites][..., colors])

    def passes(op, sl) -> dict[str, np.ndarray]:
        out: dict[str, list] = {"s": [], "tv": [], "p": []}
        for j, i in enumerate(batches):
            tokens = jnp.asarray(ho.tokens[i : i + BATCH])
            _, _, lg = _forward_jit(model, tokens, op, sl, every if sl else None, ())
            for key, v in zip(out, reduce(clean_lp[j], lg[:, :-1], tokens[:, 1:]), strict=True):
                out[key].append(np.asarray(v, dtype=np.float64))
        arr = {k: np.concatenate(v) for k, v in out.items()}
        for r in ("bayes", "null"):
            arr[f"tv_{r}"] = ex2222.total_variation(
                ref[r].reshape(-1, ref[r].shape[-1]), arr["p"].reshape(-1, arr["p"].shape[-1])
            ).reshape(arr["p"].shape[:2])
        del arr["p"]
        return arr

    identity = projection(Subspace.axis(width), 0.0)
    clean_lp = [
        jax.nn.log_softmax(_forward_jit(model, jnp.asarray(ho.tokens[i : i + BATCH]), identity, (), None, ())[2][:, :-1], -1)
        for i in batches
    ]  # fmt: skip
    clean = passes(identity, ())
    assert np.abs(clean["tv"]).max() < 1e-5, "the clean pass matches itself"

    def per_op(x: np.ndarray) -> list:
        return [x[ho.op_ids == o].mean(axis=0).tolist() for o in range(len(OP_NAMES))]

    def means(arr: dict[str, np.ndarray]) -> dict[str, list]:
        return {k: per_op(arr[k]) for k in ("s", "tv", "tv_bayes", "tv_null")}

    edited = passes(projection(Subspace.axis(width), 1.0), slices)
    axes = [means(edited)] + [means(passes(projection(Subspace.axis(width, a), 1.0), slices)) for a in range(1, width)]
    n, e = np.arange(len(ho.tokens))[:, None], np.arange(K + 1)[None, :]
    s_ref = {r: -np.log(np.maximum(ref[r][n, e, ref["answer"]], 1e-300)) for r in ("bayes", "null")}

    summary = {
        "clean": means(clean),
        "axes": axes,
        "surprisal_ref": {k: per_op(v) for k, v in s_ref.items()},
        "n_per_op": [int((ho.op_ids == o).sum()) for o in range(len(OP_NAMES))],
    }
    arrays = {
        "op_ids": ho.op_ids.astype(np.int8),
        "s_clean": clean["s"].astype(np.float16),
        "s_edited": edited["s"].astype(np.float16),
        "tv_edit": edited["tv"].astype(np.float16),
        "posterior_before": ref["before"][:, :, a_id].astype(np.float32),
    }
    arrays |= {f"{k}_{p}": run[k].astype(np.float16) for p, run in (("clean", clean), ("edited", edited)) for k in ("tv_bayes", "tv_null")}  # fmt: skip
    arrays |= {f"s_{k}": v.astype(np.float16) for k, v in s_ref.items()}
    return {"label": label, "summary": summary, "arrays": put(ex2216._npz(**arrays), name=f"edit-by-role-{label}.npz")}


def design() -> dict[str, Any]:
    return {
        "experiment": "edit-by-role",
        "anchored_op": ANCHORED_OP,
        "ops": list(OP_NAMES),
        "k": K,
        "seeds": list(SEEDS),
        "kinds": list(KINDS),
        "roles": list(ROLES),
        "answer_sites": ANSWER_SITES.tolist(),
        "rho": RHO,
        "cube_rate": CUBE_RATE,
    }


def publish(meta: list[dict], measured: list[dict]) -> dict:
    import json

    from mini.store import put, set_ref

    by_label = {m["label"]: {k: v for k, v in m.items() if k != "ref"} for m in meta}
    rows = []
    for r in measured:
        set_ref(ARRAYS_REF.format(label=r["label"]), r["arrays"])
        rows.append(by_label[r["label"]] | {k: v for k, v in r.items() if k != "arrays"})
    body = {"design": design(), "runs": rows}
    set_ref(RESULTS_REF, put(json.dumps(body).encode(), name="edit-by-role-results.json"))
    return {"n_runs": len(rows)}


def run(ctx: Ctx, seeds: tuple[int, ...] = SEEDS, publishes: bool = True) -> dict:
    meta = [m for m in runs() if m["model_seed"] in seeds]
    resolved = ctx.run(resolve, [m["ref"] for m in meta], role="prep")
    measured = ctx.map(
        measure_one,
        [resolved["checkpoints"][m["ref"]] for m in meta],
        [resolved["holdout"]] * len(meta),
        [m["label"] for m in meta],
        role="measure",
    )
    if not publishes:
        return {"labels": [m["label"] for m in measured]}
    return ctx.run(publish, meta, measured, role="prep")


def main(ctx: Ctx) -> dict:
    return run(ctx)


def smoke(ctx: Ctx) -> dict:
    """One seed of each kind, with no refs."""
    return run(ctx, seeds=SEEDS[:1], publishes=False)


COMPUTE = {
    "prep": dict(cpu=2, timeout=900),
    # Two passes over the held-out contexts of a small model, keeping the logits: a few minutes on a CPU.
    "measure": dict(cpu=2, timeout=1800),
}

experiment = Experiment(name="edit-by-role", main=main, roles=COMPUTE)
