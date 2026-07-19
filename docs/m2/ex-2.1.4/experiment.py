"""Experiment 2.1.4: spelling it out — the char-level twin of ex-2.1.3.

Ex-2.1.3 removed the hex scaffolding and found that word-level models infer
the color-space geometry from co-occurrence alone — but every color was a
single token, so a name *was* an embedding row and the answer had one fixed
home position. This experiment keeps the language identical (same pairs, same
splits, same operand orders, same equation count) and changes only the
tokenizer's view: every color is now an opaque fixed-length letter string
(`sca.data.char_names`), read and written one character at a time::

    tkzk + qwfd = hjnp

Two questions. First, is one-token-per-concept load-bearing for geometry
inference, or does the value subspace survive when name → value binding must
be assembled across characters? Second, does ex-2.1.2's answer schedule —
just-in-time computation with eviction — return with multi-token answers?
There it *could* evict: hex digit k is channel k, so an emitted channel is
done. Here no character maps to a channel; the whole mix must stay live until
the name is fully spelled, so the schedule probe becomes a test of holistic
vs per-channel emission.

Grids v27 and v216 (the memorizable extreme and the solved-at-word-level sweet
spot), three seeds each, on the frozen d64-L4 backbone. Per cell:

- **Exact match** two ways: greedy decode (with well-formedness accounting)
  and argmax over teacher-forced candidate names — plus the full candidate
  log-probability matrix, saved so the report can compare the model's answer
  distribution against *distance-shaped* targets (not just the one-hot truth).
- **Distance metrics** mirroring ex-2.1.3: guess vs nearest-name floor vs
  chance, on held-out and open pairs.
- **Probes**: per-depth operand/result decodability (`probe_residual_stream`),
  transfer of the pre-answer result probe to held-out and open prompts, and
  the per-offset × per-channel answer-schedule probe from ex-2.1.2.
- **Calibration** (mean answer surprisal, entropy, s₂) per closed eval set.

    bin/mini run docs/m2/ex-2.1.4/experiment.py --app modal --max-containers 9
    bin/mini status ex-2.1.4
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir

# The frozen backbone; the tokenizer is tiny (30 chars) so the whole model is shared.
WIDTH, DEPTH = 64, 4
SEEDS = [0, 1, 2]
PEAK_LR = 1e-2
GRID_NAMES = ["v27", "v216"]

# Same corpus seed and split fractions as ex-2.1.3: the operand pairs, holdout
# membership, and equation sequence are bit-identical; only the rendering differs.
CORPUS_SEED = 0
NAME_SEED = 0
N_EXAMPLES = 100_000  # matched *equations* (not tokens): same mixing-table evidence
HOLDOUT_FRAC = 0.2
N_EVAL = 256
N_PROBE = 2048  # train-distribution lines for the probe suites

METRICS_REF = "reports/m2/ex-2.1.4/metrics"
ARRAYS_REF = "reports/m2/ex-2.1.4/arrays"
EVALS_REF = "reports/m2/ex-2.1.4/evals"  # + f"/{grid}"
CKPT_REF = "reports/m2/ex-2.1.4/checkpoints"  # + f"/{label}"


def prepare_corpus(grid: str, levels: tuple) -> dict:
    """Re-render one grid's ex-2.1.3 corpus at char level; build eval + probe sets."""
    import numpy as np

    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import char_names as ch
    from sca.data import named_colors as nc
    from sca.data.colors import dump_example_sets
    from sca.data.tokenizer import CharTokenizer
    from mini.store import put

    names = {v: k for k, v in ch.opaque_names(levels, NAME_SEED).items()}
    base = nc.sample_corpus(N_EXAMPLES, CORPUS_SEED, levels, HOLDOUT_FRAC)
    corpus = [ch.rename(ex, names) for ex in base]

    text = "".join(ex.text for ex in corpus)
    tokenizer_config = TokenizerConfig(vocabulary=ch.alphabet())  # fixed a priori
    tokens = np.asarray(CharTokenizer(tokenizer_config).encode([text])[0], dtype=np.int32)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=len(text),
        sources=[DatasetMetadata(title=f"char-level named-only corpus ({grid})", fixes=[], total_chars=len(text))],
    )
    save_data(tokens, meta, get_data_dir() / "corpora" / grid)

    # Eval pairs: same construction (and rng stream) as ex-2.1.3, so the
    # per-pair comparison across the two experiments is exact.
    held = nc.holdout_test(levels, CORPUS_SEED, HOLDOUT_FRAC)
    seen = sorted({p for ex in base if (p := ex.pair) is not None})
    holdout = [p for p in nc.closed_pairs(levels) if held(*p)]
    opens = nc.open_pairs(levels)

    rng = np.random.default_rng(CORPUS_SEED + 1)
    pick = lambda pairs, r: [pairs[i] for i in r.permutation(len(pairs))[:N_EVAL]]  # noqa: E731
    sets = {
        "named_seen": [nc.make_example(a, b, names, rng) for a, b in pick(seen, rng)],
        "named_holdout": [nc.make_example(a, b, names, rng) for a, b in pick(holdout, rng)],
        "open": [nc.make_example(a, b, names, rng) for a, b in pick(opens, rng)],
        # Probe suite input: in-training lines (the schedule probe teacher-forces
        # prompt + answer, and all equations share one shape by construction).
        "probe": corpus[:N_PROBE],
    }

    stats = {
        "n_colors": len(names),
        "n_seen_distinct": len(seen),
        "n_holdout": len(holdout),
        "n_open": len(opens),
        "total_tokens": int(len(tokens)),
        "eval_n": {k: len(v) for k, v in sets.items()},
        "sample_names": sorted(names.values())[:4],
    }
    return {
        "grid": grid,
        "meta": meta,
        "stats": stats,
        "evals": put(dump_example_sets(sets), name=f"ex-2.1.4-{grid}-evals.json"),
    }


def _make_config(vocab_size: int, seed: int):
    """The frozen d64-L4 backbone, verbatim from the siblings."""
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
            vocab_size=vocab_size,
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


def build_sweep(preps: list[dict]) -> list[tuple]:
    """(config, grid, label) cells; cheap and deterministic per wake."""
    from sca.utils import align

    cells = []
    for prep in preps:
        tc = prep["meta"].tokenizer_config
        for seed in SEEDS:
            config = _make_config(align(tc.vocab_size, 64), seed)
            config.tokenizer = tc.model_copy()
            cells.append((config, prep["grid"], f"{prep['grid']}-s{seed}"))
    return cells


def train_one(config, grid: str, label: str) -> dict:
    """Train one cell on its grid's corpus; checkpoint into the store."""
    from sca.compute.training import train_model
    from mini.store import put

    ckpt_dir = get_data_dir() / "cells" / label
    _, metrics = train_model(config, get_data_dir() / "corpora" / grid, checkpoint_dir=ckpt_dir)
    return {
        "label": label,
        "grid": grid,
        "val_loss": [m.val_loss for m in metrics],
        "checkpoint": put(ckpt_dir / "model", name=f"ex-2.1.4-{label}-ckpt"),
    }


def eval_one(trained: dict, evals, grid: str, levels: tuple) -> dict:
    """Behavioral metrics, candidate scoring, and the three probe suites for one cell."""
    import io

    import numpy as np

    from sca.compute.evaluation import (
        answer_calibration,
        candidate_logprobs,
        greedy_completions,
        probe_answer_schedule,
        probe_residual_stream,
        probe_transfer,
    )
    from sca.compute.model import load_checkpoint
    from sca.data import char_names as ch
    from sca.data.colors import N_LEVELS, load_example_sets
    from sca.data.tokenizer import CharTokenizer
    from mini.store import get, put

    label = trained["label"]
    workdir = get_data_dir() / "eval" / label
    get(trained["checkpoint"], workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    tokenizer = CharTokenizer(config.tokenizer)

    palette = ch.opaque_names(levels, NAME_SEED)  # name → rgb, as in prepare_corpus
    names_list = list(palette)
    name_idx = {n: i for i, n in enumerate(names_list)}
    vocab_rgb = np.array(list(palette.values()), dtype=np.float32) / (N_LEVELS - 1)

    all_sets = load_example_sets(get(evals, workdir / "evals.json").read_bytes())
    probe_exs = all_sets.pop("probe")

    sets, arrays = {}, {}
    for set_name, exs in all_sets.items():
        prompts = [ex.prompt for ex in exs]
        # Greedy decode: does free emission produce a well-formed name at all?
        got = greedy_completions(model, tokenizer, prompts, ch.NAME_LEN + 1)
        well = np.array([g in palette for g in got])
        # Candidate scoring: log P(name + "\n") for every vocabulary name. The
        # argmax is the model's best *well-formed* answer; the full matrix is
        # saved for the report's distribution-vs-distance analysis.
        lp = candidate_logprobs(model, tokenizer, prompts, names_list)
        cand_guess = lp.argmax(axis=1)
        p = np.exp(lp)
        pn = p / p.sum(1, keepdims=True)

        result = np.array([ex.result for ex in exs], dtype=np.float32) / (N_LEVELS - 1)
        dists = np.linalg.norm(vocab_rgb[None] - result[:, None], axis=2)  # (N, V)
        guess_dist = dists[np.arange(len(exs)), cand_guess]
        floor = dists.min(axis=1)
        stats: dict = {
            "n": len(exs),
            "p_malformed": float(1 - well.mean()),
            "guess_dist": float(guess_dist.mean()),
            "exp_dist": float((pn * dists).sum(1).mean()),
            "floor_dist": float(floor.mean()),
            "chance_dist": float(dists.mean()),
            "nearest_acc": float((guess_dist <= floor + 1e-9).mean()),
            "mass_names": float(p.sum(1).mean()),  # probability of *some* complete name answer
        }
        if exs[0].answer:  # closed sets: the true answer is a vocabulary name
            true_idx = np.array([name_idx[ex.answer] for ex in exs])
            stats["accuracy"] = float(np.mean([g == ex.answer for g, ex in zip(got, exs, strict=True)]))
            stats["cand_accuracy"] = float((cand_guess == true_idx).mean())
            stats["nll"] = float(-lp[np.arange(len(exs)), true_idx].mean())
            stats["calibration"] = answer_calibration(model, tokenizer, exs)
            stats["failures"] = [(ex.prompt, ex.answer, g) for g, ex in zip(got, exs, strict=True) if g != ex.answer][
                :8
            ]
        sets[set_name] = stats
        arrays[f"logp/{set_name}"] = lp.astype(np.float32)

    # Probe suites on in-training lines: per-depth operand/result decodability,
    # transfer of the pre-answer result probe, and the answer-emission schedule.
    probe = probe_residual_stream(model, tokenizer, probe_exs)
    transfer = probe_transfer(model, tokenizer, probe_exs, all_sets)
    schedule = probe_answer_schedule(model, tokenizer, probe_exs)
    arrays["schedule/r2"] = schedule["r2"]
    arrays |= {f"probe_weights/{k}": w for k, w in probe["weights"].items()}

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return {
        "label": label,
        "grid": grid,
        "val_loss": trained["val_loss"],
        "sets": sets,
        "probe_r2": probe["r2"],
        "transfer_r2": transfer,
        "schedule_offsets": schedule["offsets"],
        "arrays": put(buf.getvalue(), name=f"ex-2.1.4-{label}-arrays.npz"),
    }


def publish_results(results: list[dict], stats: dict, checkpoints: dict, evals: dict) -> dict:
    """Publish metrics (JSON), stacked per-cell arrays (npz), eval sets, and checkpoints."""
    import io
    import json

    import numpy as np

    from mini.store import get, put, set_ref

    cells = [{k: v for k, v in r.items() if k != "arrays"} for r in results]
    metrics = {"cells": cells, "corpus_stats": stats}
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.1.4-metrics.json"))
    for grid, art in evals.items():
        set_ref(f"{EVALS_REF}/{grid}", art)
    for label, ckpt in checkpoints.items():
        set_ref(f"{CKPT_REF}/{label}", ckpt)

    arrays = {}
    for r in results:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{name}": z[name] for name in z.files}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="ex-2.1.4-arrays.npz"))
    return {"n_cells": len(results)}


def main(ctx: Ctx) -> dict:
    from sca.data.named_colors import GRIDS

    levels = [GRIDS[g] for g in GRID_NAMES]
    preps = ctx.map(prepare_corpus, GRID_NAMES, levels, role="prep")
    configs, cell_grids, labels = zip(*build_sweep(preps), strict=True)
    trained = ctx.map(train_one, configs, cell_grids, labels, role="train")
    evals_by_grid = {p["grid"]: p["evals"] for p in preps}
    evaled = ctx.map(
        eval_one,
        trained,
        [evals_by_grid[g] for g in cell_grids],
        cell_grids,
        [GRIDS[g] for g in cell_grids],
        role="eval",
    )
    ckpts = dict(zip(labels, [t["checkpoint"] for t in trained], strict=True))
    stats = {p["grid"]: p["stats"] for p in preps}
    summary = ctx.run(publish_results, evaled, stats, ckpts, evals_by_grid, role="prep")

    def mean_over_seeds(grid: str, key: str) -> float:
        vals = [r["sets"]["named_holdout"][key] for r in evaled if r["grid"] == grid]
        return round(sum(vals) / len(vals), 3)

    return {
        **summary,
        "holdout_accuracy": {g: mean_over_seeds(g, "accuracy") for g in GRID_NAMES},
        "holdout_cand_accuracy": {g: mean_over_seeds(g, "cand_accuracy") for g in GRID_NAMES},
        "holdout_guess_dist": {g: mean_over_seeds(g, "guess_dist") for g in GRID_NAMES},
    }


experiment = Experiment(
    name="ex-2.1.4",
    main=main,
    roles={
        # Corpus re-rendering + a 1.9M-char tokenize: plain CPU work.
        "prep": dict(cpu=2, timeout=900),
        # Same backbone as the siblings, but ~3× their token count per epoch.
        "train": dict(gpu="L4", timeout=2700),
        # Candidate scoring is the big item: |prompts| × |names| forced sequences.
        "eval": dict(gpu="L4", timeout=1200),
    },
)
