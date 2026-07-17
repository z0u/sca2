"""Experiment 2.1.2: making composition pay — corpus fixes for the named-holdout failure.

Ex-2.1.1's baseline never solves ``named_holdout``: the named slice is
memorizable, the alias dictionary is one-way, and hex answers factorize per
channel, so nothing ever forces the model to compose. Its garden-path
walkthrough showed the failure is *close*, though — on ``lime + black`` the
arithmetic half-corrects the answer but loses to a trained lookup — so the
diagnosis predicts that small corpus changes tip the computation over the line.
This experiment tests that prediction with a 2 × 2 factorial over two grammar
interventions, holding the ex-2.1.1 backbone (d64-L4) and named-pair holdout
split fixed:

- **rev** — reverse alias lines (``#f00 = red``): supervise the hex → name
  readout that the one-way dictionary leaves untrained (the reversal-curse leg).
- **open** — named operands whose mix is off-palette, answered in hex
  (``red + navy = #804``): make name + name prompts engage the arithmetic,
  because the answer's *form* now depends on the mix's value (the lookup-table
  leg).

Each intervention carves its token share out of the (over-saturated) hex slice;
everything else — example count, seeds, LR, architecture — is fixed. Per cell
(condition × seed) we measure:

- **Completion accuracy** on ex-2.1.1's four eval sets (identical splits) plus
  three new ones: seen and held-out *open* pairs, and reverse-alias prompts.
- **Name margins**: teacher-forced log-probability of every palette name (and
  the relevant hex codes) as a complete answer to each named prompt. The margin
  of the true answer over the best competitor is the graded compute-vs-lookup
  measure; ex-2.1.1 could only sample it anecdotally.
- **Calibration** (mean answer surprisal, entropy, and s₂) per eval set — the
  graded early-warning metric queued in the todo list.
- **Probes**: ex-2.1.1's per-layer suite (comparability), a per-answer-position ×
  per-channel schedule probe on hex answers (the stair-step hypothesis), and a
  result-color probe fit on open-pair prompts and transferred to the named eval
  sets ("computed but outvoted" vs "never computed").

    bin/mini run docs/m2/ex-2.1.2/experiment.py --app modal --max-containers 9
    bin/mini status ex-2.1.2
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir

# The ex-2.1.1 backbone: the smallest cell that saturated the unseen-pair sets.
WIDTH, DEPTH = 64, 4
SEEDS = [0, 1, 2]
PEAK_LR = 1e-2  # carried over unchanged

# Identical to ex-2.1.1, so the named holdout pairs are the same ten.
CORPUS_SEED = 0
N_EXAMPLES = 40_000
HOLDOUT_FRAC = 0.2
OPEN_HOLDOUT_FRAC = 0.2  # of the 302 off-palette named pairs
N_EVAL = 256
N_PROBE = 2048
N_SCHEDULE = 2048
N_SURPRISAL = 2

# Form weights per condition. Interventions displace only the hex slice: it is
# far past saturation (hex_unseen = 1.0 at weight 0.6 in ex-2.1.1), so the
# factorial reads as "what was added", not "what was starved".
CONDITIONS: dict[str, dict] = {
    "control": {"hex": 0.60, "named": 0.15, "cross": 0.15, "alias": 0.10},
    "rev": {"hex": 0.55, "named": 0.15, "cross": 0.15, "alias": 0.10, "alias_rev": 0.05},
    "open": {"hex": 0.50, "named": 0.15, "cross": 0.15, "alias": 0.10, "open": 0.10},
    "both": {"hex": 0.45, "named": 0.15, "cross": 0.15, "alias": 0.10, "alias_rev": 0.05, "open": 0.10},
}

METRICS_REF = "reports/m2/ex-2.1.2/metrics"
MARGINS_REF = "reports/m2/ex-2.1.2/margins"
WEIGHTS_REF = "reports/m2/ex-2.1.2/probe-weights"
CKPT_REF = "reports/m2/ex-2.1.2/checkpoints"  # + f"/{label}"

MARGIN_SETS = ["named_seen", "named_holdout", "open_holdout"]


def _splits():
    from sca.data import colors

    named_train, named_holdout = colors.split_named_pairs(CORPUS_SEED, HOLDOUT_FRAC)
    open_train, open_holdout = colors.split_open_pairs(CORPUS_SEED, OPEN_HOLDOUT_FRAC)
    return named_train, named_holdout, open_train, open_holdout


def _corpus(cond: str) -> list:
    from sca.data import colors

    named_train, _, open_train, _ = _splits()
    return colors.sample_corpus(N_EXAMPLES, CORPUS_SEED, named_train, CONDITIONS[cond], open_pairs=open_train)


def prepare_corpus(cond: str, weights: dict) -> dict:
    """Tokenize one condition's corpus onto the volume; *weights* keys the memo."""
    import numpy as np

    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import colors
    from sca.data.tokenizer import CharTokenizer

    corpus = colors.sample_corpus(
        N_EXAMPLES, CORPUS_SEED, _splits()[0], weights, open_pairs=_splits()[2] if "open" in weights else None
    )
    text = "".join(ex.text for ex in corpus)
    tokenizer_config = TokenizerConfig(vocabulary=colors.alphabet())  # fixed a priori, as in ex-2.1.1
    tokens = np.asarray(CharTokenizer(tokenizer_config).encode([text])[0], dtype=np.int32)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=len(text),
        sources=[DatasetMetadata(title=f"color-mixing corpus ({cond})", fixes=[], total_chars=len(text))],
    )
    save_data(tokens, meta, get_data_dir() / "corpora" / cond)
    counts: dict[str, int] = {}
    for ex in corpus:
        form = _classify(ex)
        counts[form] = counts.get(form, 0) + 1
    return {"cond": cond, "meta": meta, "counts": counts}


def _classify(ex) -> str:
    if ex.rhs is None:
        return "alias_rev" if ex.prompt.startswith("#") else "alias"
    named_operands = "#" not in ex.prompt
    if named_operands:
        return "named" if "#" not in ex.answer else "open"
    return "hex" if ex.prompt.count("#") == 2 else "cross"


def prepare_eval_sets() -> dict:
    """Eval, probe, and margin sets, shared by every condition.

    The unseen hex/cross sets avoid the union of pairs any condition trained
    on, so one artifact serves all four conditions.
    """
    import numpy as np

    from sca.data import colors
    from mini.store import put

    named_train, named_holdout, open_train, open_holdout = _splits()
    seen = {p for cond in CONDITIONS for ex in _corpus(cond) if (p := ex.pair) is not None}
    rng = np.random.default_rng(8)
    evals = {
        "named_seen": colors.as_named(named_train, seed=1),  # ex-2.1.1's sets, verbatim
        "named_holdout": colors.as_named(named_holdout, seed=2),
        "hex_unseen": colors.sample_unseen("hex", N_EVAL, 3, seen),
        "cross_unseen": colors.sample_unseen("cross", N_EVAL, 4, seen),
        "open_seen": colors.as_form(open_train, "open", seed=3),
        "open_holdout": colors.as_form(open_holdout, "open", seed=4),
        "alias_rev": [colors.make_example("alias_rev", c, None, rng) for c in colors.PALETTE.values()],
    }
    probes = {
        # Same recipe as ex-2.1.1's probe set, for comparable classic probes.
        "probe": colors.sample_corpus(N_PROBE, 5, named_train, {"hex": 0.5, "named": 0.25, "cross": 0.25}),
        # Hex-only: fixed prompt/answer lengths for the schedule probe.
        "schedule": colors.sample_corpus(N_SCHEDULE, 6, [], {"hex": 1.0}),
        # Name + name prompts for the transfer probe: fit on open-train pairs
        # (two renders each — operand order varies), score on the eval sets.
        "transfer_fit": colors.as_form(open_train, "open", seed=7) + colors.as_form(open_train, "open", seed=8),
    }
    return {
        "evals": put(colors.dump_example_sets(evals), name="ex-2.1.2-evals.json"),
        "probes": put(colors.dump_example_sets(probes), name="ex-2.1.2-probes.json"),
    }


def _make_config(seed: int):
    """The frozen d64-L4 backbone; identical to ex-2.1.1's cell."""
    from sca.config import (
        DataConfig,
        ModelConfig,
        OptimizerConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    return TrainingConfig(
        model=ModelConfig(
            vocab_size=64,
            block_size=64,
            n_embd=WIDTH,
            n_head=8,
            n_head_dim=8,
            n_ff=4 * WIDTH,
            n_layer=DEPTH,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(batch_size=64, oversample=16, train_split=0.9, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=PEAK_LR, betas=(0.9, 0.95)),
        scheduler=SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01),
        seed=seed,
    )


def build_sweep(meta) -> list[tuple]:
    """(config, condition, label) cells; cheap and deterministic per wake."""
    from sca.utils import align

    cells = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            config = _make_config(seed)
            config.tokenizer = meta.tokenizer_config.model_copy()
            config.model.vocab_size = align(meta.tokenizer_config.vocab_size, 64)
            cells.append((config, cond, f"{cond}-s{seed}"))
    return cells


def train_one(config, cond: str, label: str) -> dict:
    """Train one cell on its condition's corpus; checkpoint into the store."""
    from sca.compute.training import train_model
    from mini.store import put

    ckpt_dir = get_data_dir() / "cells" / label
    _, metrics = train_model(config, get_data_dir() / "corpora" / cond, checkpoint_dir=ckpt_dir)
    return {
        "label": label,
        "cond": cond,
        "val_loss": [m.val_loss for m in metrics],
        "checkpoint": put(ckpt_dir / "model", name=f"ex-2.1.2-{label}-ckpt"),
    }


def eval_one(trained: dict, evals, probes) -> dict:
    """Accuracy, margins, calibration, and the three probe suites for one cell."""
    import io

    import numpy as np

    from sca.compute.evaluation import (
        answer_calibration,
        candidate_logprobs,
        completion_accuracy,
        greedy_completions,
        probe_answer_schedule,
        probe_residual_stream,
        probe_transfer,
    )
    from sca.compute.model import load_checkpoint
    from sca.data import colors
    from sca.data.colors import load_example_sets
    from sca.data.tokenizer import CharTokenizer
    from mini.store import get, put

    label = trained["label"]
    workdir = get_data_dir() / "eval" / label
    get(trained["checkpoint"], workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    tokenizer = CharTokenizer(config.tokenizer)

    eval_sets = load_example_sets(get(evals, workdir / "evals.json").read_bytes())
    accuracy = {name: completion_accuracy(model, tokenizer, exs) for name, exs in eval_sets.items()}
    calibration = {name: answer_calibration(model, tokenizer, exs) for name, exs in eval_sets.items()}

    # Greedy completions of the held-out named prompts — precomputed here, where the
    # model's already loaded, so the report reads them from metrics instead of pulling
    # every checkpoint and re-decoding on each edit.
    holdout_completions = greedy_completions(model, tokenizer, [ex.prompt for ex in eval_sets["named_holdout"]], 12)

    # Margins: score every palette name — plus the hex codes this set's true
    # mixes render to — as a complete answer to each prompt.
    margins = {}
    for name in MARGIN_SETS:
        exs = eval_sets[name]
        candidates = list(colors.PALETTE) + sorted({colors.to_hex(ex.result) for ex in exs})
        lp = candidate_logprobs(model, tokenizer, [ex.prompt for ex in exs], candidates)
        margins[name] = {"candidates": candidates, "logprobs": lp}

    probe_sets = load_example_sets(get(probes, workdir / "probes.json").read_bytes())
    probe = probe_residual_stream(model, tokenizer, probe_sets["probe"])
    schedule = probe_answer_schedule(model, tokenizer, probe_sets["schedule"])
    transfer = probe_transfer(
        model,
        tokenizer,
        probe_sets["transfer_fit"],
        {k: eval_sets[k] for k in ("open_holdout", "named_seen", "named_holdout")},
    )

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        **probe["weights"],
        **{f"margins/{name}": m["logprobs"] for name, m in margins.items()},
        **{"schedule/r2": schedule["r2"]},
    )
    return {
        "label": label,
        "cond": trained["cond"],
        "val_loss": trained["val_loss"],
        "accuracy": accuracy,
        "calibration": calibration,
        "holdout_completions": holdout_completions,
        "margin_candidates": {name: m["candidates"] for name, m in margins.items()},
        "probe_r2": probe["r2"],
        "schedule_offsets": schedule["offsets"],
        "transfer_r2": transfer,
        "arrays": put(buf.getvalue(), name=f"ex-2.1.2-{label}-arrays.npz"),
    }


def surprisal_one(checkpoint, evals, label: str) -> dict:
    """Per-character surprisal and entropy over the first lines of each eval set.

    CPU-friendly, and separate from `eval_one` so subline tweaks don't
    invalidate the GPU eval memos (same split as ex-2.1.1).
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sca.compute.model import load_checkpoint
    from sca.data.colors import load_example_sets
    from sca.data.tokenizer import CharTokenizer
    from mini.store import get

    workdir = get_data_dir() / "surprisal" / label
    get(checkpoint, workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    model = eqx.nn.inference_mode(model)
    tokenizer = CharTokenizer(config.tokenizer)

    sets: dict[str, list[dict]] = {}
    for name, exs in load_example_sets(get(evals, workdir / "evals.json").read_bytes()).items():
        rows = []
        for ex in exs[:N_SURPRISAL]:
            text = ex.prompt + ex.answer
            ids = np.asarray(tokenizer.encode([text])[0])
            logp = jax.nn.log_softmax(model(jnp.asarray(ids[None]))[0], axis=-1)[: len(ids) - 1]
            nll = np.asarray(-logp[jnp.arange(len(ids) - 1), ids[1:]])
            entropy = np.asarray(-(jnp.exp(logp) * logp).sum(axis=-1))
            rows.append(
                {
                    "text": text,
                    "answer_start": len(ex.prompt),
                    "nll": [float(v) for v in nll],
                    "entropy": [float(v) for v in entropy],
                }
            )
        sets[name] = rows
    return {"label": label, "surprisal": sets}


def publish_results(results: list[dict], surprisals: list[dict], checkpoints: dict, counts: dict) -> dict:
    """Publish metrics (JSON), stacked arrays (npz), and per-cell checkpoints."""
    import io
    import json

    import numpy as np

    from mini.store import get, put, set_ref

    by_label = {s["label"]: s["surprisal"] for s in surprisals}
    cells = [{k: v for k, v in r.items() if k != "arrays"} | {"surprisal": by_label[r["label"]]} for r in results]
    metrics = {"cells": cells, "corpus_counts": counts}
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.1.2-metrics.json"))
    for label, ckpt in checkpoints.items():
        set_ref(f"{CKPT_REF}/{label}", ckpt)

    weights, margins = {}, {}
    for r in results:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}.npz")
        with np.load(path) as z:
            for name in z.files:
                # Margins and schedule R² travel together; probe weights mirror ex-2.1.1.
                dest = margins if name.startswith(("margins/", "schedule/")) else weights
                dest[f"{r['label']}/{name}"] = z[name]
    for ref, arrays, fname in [(WEIGHTS_REF, weights, "probe-weights"), (MARGINS_REF, margins, "margins")]:
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        set_ref(ref, put(buf.getvalue(), name=f"ex-2.1.2-{fname}.npz"))
    return {"n_cells": len(results)}


def main(ctx: Ctx) -> dict:
    conds = list(CONDITIONS)
    preps = ctx.map(prepare_corpus, conds, [CONDITIONS[c] for c in conds], role="prep")
    sets = ctx.run(prepare_eval_sets, role="prep")
    configs, cell_conds, labels = zip(*build_sweep(preps[0]["meta"]), strict=True)
    trained = ctx.map(train_one, configs, cell_conds, labels, role="train")
    n = len(trained)
    evaled = ctx.map(eval_one, trained, [sets["evals"]] * n, [sets["probes"]] * n, role="eval")
    ckpts = [t["checkpoint"] for t in trained]
    surprisals = ctx.map(surprisal_one, ckpts, [sets["evals"]] * n, labels, role="surprisal")
    counts = {p["cond"]: p["counts"] for p in preps}
    summary = ctx.run(publish_results, evaled, surprisals, dict(zip(labels, ckpts, strict=True)), counts, role="prep")

    def mean_acc(cond: str, es: str) -> float:
        accs = [r["accuracy"][es]["accuracy"] for r in evaled if r["cond"] == cond]
        return round(sum(accs) / len(accs), 3)

    return {
        **summary,
        "named_holdout": {cond: mean_acc(cond, "named_holdout") for cond in CONDITIONS},
        "hex_unseen": {cond: mean_acc(cond, "hex_unseen") for cond in CONDITIONS},
    }


experiment = Experiment(
    name="ex-2.1.2",
    main=main,
    roles={
        "prep": {},  # CPU-only: corpus generation + tokenize
        # Same cells as ex-2.1.1's backbone: an L4 clears one in a few minutes.
        "train": dict(gpu="L4", timeout=1800),
        # Eval now also scores margins and two extra probe suites; still small.
        "eval": dict(gpu="L4", timeout=1200),
        "surprisal": dict(cpu=2, timeout=900),
    },
)
