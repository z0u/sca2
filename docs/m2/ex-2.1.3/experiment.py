"""Experiment 2.1.3: names all the way down — geometry from co-occurrence alone.

The base color-mixing language grounds names in hex codes (alias lines, hex
arithmetic), and ex-2.1.2 showed the value → name direction is the part that
never engages. This experiment removes the crutch entirely: every color is a
single opaque token, the only sentences are ``name + name = name`` equations,
and nothing in the stream reveals that colors live on a 3D grid. Completing a
*held-out* pair then requires inferring the latent geometry from the mixing
table's co-occurrence statistics — tensor completion, in effect. The todo
item's framing: can the model infer the color-space geometry, and when it
guesses, is the guess *close*?

We sweep the vocabulary size over level sub-grids of the 16-level RGB cube
(`sca.data.named_colors.GRIDS`): 27, 64, 216, and 4096 colors. Small grids
have few closed pairs (49 distinct at 27 colors) so the table is memorizable;
the full grid's 8.4M pairs can only be covered ~1%, so the task is
generalization or nothing. Per cell (grid × seed, the fixed d64-L4 architecture):

- **Exact-match accuracy and answer NLL** on seen and held-out closed pairs
  (single-token answers, so cross-entropy over the vocabulary is exact).
- **Distance metrics**: the RGB distance from the model's guessed color to the
  true mix — for held-out pairs and for *open* pairs whose mix has no name,
  where "close" is the only possible kind of correct. Baselines per prompt:
  the nearest-vocabulary floor and the vocabulary-mean (chance) distance.
- **Geometry probes**: a ridge probe from the color-token embeddings to RGB
  (did the embedding table become a color cube?), the top-3 PCA explained
  variance of those embeddings, and per-layer probes from the pre-answer
  residual stream to the mix's RGB (is the answer computed in value space?).

Results inform the vocabulary design for the anchored D2.1.x experiments.

    bin/mini run docs/m2/ex-2.1.3/experiment.py --app modal --max-containers 9
    bin/mini status ex-2.1.3
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir

# The ex-2.1.1/2.1.2 architecture, unchanged. The embedding table grows with the
# vocabulary (4160 × 64 at the full grid) but the stack stays fixed.
WIDTH, DEPTH = 64, 4
SEEDS = [0, 1, 2]
PEAK_LR = 1e-2

CORPUS_SEED = 0
N_EXAMPLES = 100_000  # lines of 6 tokens ≈ the sibling experiments' token budget
HOLDOUT_FRAC = 0.2  # of distinct closed pairs
N_EVAL = 256  # cap per eval set (small grids have fewer; we take what exists)

METRICS_REF = "reports/m2/ex-2.1.3/metrics"
ARRAYS_REF = "reports/m2/ex-2.1.3/arrays"
EVALS_REF = "reports/m2/ex-2.1.3/evals"  # + f"/{grid}"
CKPT_REF = "reports/m2/ex-2.1.3/checkpoints"  # + f"/{label}"


def prepare_corpus(grid: str, levels: tuple) -> dict:
    """Sample and tokenize one grid's corpus; build its eval sets and stats."""
    import numpy as np

    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import named_colors as nc
    from sca.data.colors import dump_example_sets
    from mini.store import put

    palette = nc.grid_palette(levels)
    names = {v: k for k, v in palette.items()}
    corpus = nc.sample_corpus(N_EXAMPLES, CORPUS_SEED, levels, HOLDOUT_FRAC)

    words = [w for ex in corpus for w in nc.as_words(ex)]
    tokenizer_config = TokenizerConfig(vocabulary=[*nc.SYNTAX, *palette])
    tokens = np.asarray(nc.WordTokenizer(tokenizer_config).encode_words(words), dtype=np.int32)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=sum(map(len, words)),
        sources=[
            DatasetMetadata(title=f"named-only color corpus ({grid})", fixes=[], total_chars=sum(map(len, words)))
        ],
    )
    save_data(tokens, meta, get_data_dir() / "corpora" / grid)

    held = nc.holdout_test(levels, CORPUS_SEED, HOLDOUT_FRAC)
    seen = sorted({p for ex in corpus if (p := ex.pair) is not None})
    if len(palette) <= 1000:
        closed = nc.closed_pairs(levels)
        holdout = [p for p in closed if held(*p)]
        opens = nc.open_pairs(levels)
    else:  # the full grid: everything is closed; sample the held-out side by hash
        colors_arr = list(palette.values())
        rng_h = np.random.default_rng(CORPUS_SEED + 7)
        picked: set = set()
        while len(picked) < N_EVAL:
            a, b = (colors_arr[int(i)] for i in rng_h.integers(len(colors_arr), size=2))
            if held(a, b):
                picked.add(nc.pair_key(a, b))
        holdout, opens = sorted(picked), []

    rng = np.random.default_rng(CORPUS_SEED + 1)
    pick = lambda pairs, r: [pairs[i] for i in r.permutation(len(pairs))[:N_EVAL]]  # noqa: E731
    sets = {
        "named_seen": [nc.make_example(a, b, names, rng) for a, b in pick(seen, rng)],
        "named_holdout": [nc.make_example(a, b, names, rng) for a, b in pick(holdout, rng)],
    }
    if opens:
        sets["open"] = [nc.make_example(a, b, names, rng) for a, b in pick(opens, rng)]

    n_colors = len(palette)
    olc = sum(1 for a in levels for b in levels if (a + b + 1) // 2 in set(levels))
    stats = {
        "n_colors": n_colors,
        "n_closed_distinct": (olc**3 - n_colors) // 2,
        "n_seen_distinct": len(seen),
        "n_holdout": len(holdout),
        "n_open": len(opens),
        "total_tokens": int(len(tokens)),
        "eval_n": {k: len(v) for k, v in sets.items()},
    }
    return {
        "grid": grid,
        "meta": meta,
        "stats": stats,
        "evals": put(dump_example_sets(sets), name=f"ex-2.1.3-{grid}-evals.json"),
    }


def _make_config(vocab_size: int, seed: int):
    """The d64-L4 config from the siblings; only the (tied) embedding table scales."""
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
        "checkpoint": put(ckpt_dir / "model", name=f"ex-2.1.3-{label}-ckpt"),
    }


def eval_one(trained: dict, evals, grid: str) -> dict:
    """Behavioral and geometric metrics for one cell.

    Answers are single tokens, so one teacher-forced forward pass per eval set
    yields everything: the full next-token distribution at the pre-answer
    position (accuracy, NLL, distance metrics) and the residual stream at the
    same position (value-space probes).
    """
    import io

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    from sca.compute.evaluation import ridge_probe
    from sca.compute.model import load_checkpoint
    from sca.data.colors import N_LEVELS, load_example_sets
    from sca.data.named_colors import GRIDS, WordTokenizer, grid_palette
    from mini.store import get, put

    label = trained["label"]
    workdir = get_data_dir() / "eval" / label
    get(trained["checkpoint"], workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    model = eqx.nn.inference_mode(model)
    tokenizer = WordTokenizer(config.tokenizer)

    palette = grid_palette(GRIDS[grid])
    names = list(palette)
    name_idx = {n: i for i, n in enumerate(names)}
    color_ids = np.array([tokenizer.stoi[n] for n in names])
    vocab_rgb = np.array(list(palette.values()), dtype=np.float32) / (N_LEVELS - 1)

    forward = eqx.filter_jit(model.__call__)
    stream = eqx.filter_jit(model.residual_stream)

    def at_answer(exs) -> tuple[np.ndarray, np.ndarray]:
        """(full-vocab logprobs, residual stream) at the pre-answer position."""
        seq = np.array([tokenizer.encode_words(ex.prompt.split()) for ex in exs])
        lps, acts = [], []
        for i in range(0, len(seq), 256):
            s = jnp.asarray(seq[i : i + 256])
            lps.append(np.asarray(jax.nn.log_softmax(forward(s)[:, -1], axis=-1)))
            acts.append(np.asarray(stream(s))[:, :, -1])
        return np.concatenate(lps), np.concatenate(acts, axis=1)

    def r2(y_pred, y_true) -> float:
        ss_res = ((y_true - y_pred) ** 2).sum(0)
        ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
        return float((1 - ss_res / np.maximum(ss_tot, 1e-12)).mean())

    eval_sets = load_example_sets(get(evals, workdir / "evals.json").read_bytes())
    sets, arrays, probe_data = {}, {}, {}
    for set_name, exs in eval_sets.items():
        logp, acts = at_answer(exs)
        lp_colors = logp[:, color_ids]  # (N, V_colors); columns follow palette order
        result = np.array([ex.result for ex in exs], dtype=np.float32) / (N_LEVELS - 1)
        dists = np.linalg.norm(vocab_rgb[None] - result[:, None], axis=2)  # (N, V_colors)
        guess = lp_colors.argmax(axis=1)
        guess_dist = dists[np.arange(len(exs)), guess]
        p = np.exp(lp_colors)
        floor = dists.min(axis=1)
        stats: dict = {
            "n": len(exs),
            "guess_dist": float(guess_dist.mean()),
            "exp_dist": float((p / p.sum(1, keepdims=True) * dists).sum(1).mean()),
            "floor_dist": float(floor.mean()),
            "chance_dist": float(dists.mean()),
            "nearest_acc": float((guess_dist <= floor + 1e-9).mean()),
            "p_offvocab": float((1 - p.sum(1)).mean()),
        }
        if exs[0].answer:  # closed sets: the true answer is a vocabulary name
            true_idx = np.array([name_idx[ex.answer] for ex in exs])
            stats["accuracy"] = float((guess == true_idx).mean())
            stats["nll"] = float(-lp_colors[np.arange(len(exs)), true_idx].mean())
            stats["failures"] = [
                (ex.prompt, ex.answer, names[g]) for ex, g, t in zip(exs, guess, true_idx, strict=True) if g != t
            ][:8]
        sets[set_name] = stats
        arrays[f"logp/{set_name}"] = lp_colors.astype(np.float32)
        probe_data[set_name] = (acts, result)

    # Value-space probes: ridge from the pre-answer residual stream (per depth)
    # to the mix's RGB, fit on half of named_seen, scored on everything else.
    x, y = probe_data["named_seen"]
    half = x.shape[1] // 2
    fitted = [ridge_probe(x[d, :half], y[:half], x[d, half:], y[half:]) for d in range(x.shape[0])]
    probe_r2 = {"named_seen": [float(r) for *_, r in fitted]}
    for set_name, (x_t, y_t) in probe_data.items():
        if set_name != "named_seen":
            probe_r2[set_name] = [r2(x_t[d] @ w + b, y_t) for d, (w, b, _) in enumerate(fitted)]

    # Embedding geometry: did the (tied) token embeddings become a color cube?
    emb = np.asarray(model.transformer.wte)[color_ids]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    perm = np.random.default_rng(0).permutation(len(emb))
    h = len(emb) // 2
    *_, emb_r2 = ridge_probe(emb[perm[:h]], vocab_rgb[perm[:h]], emb[perm[h:]], vocab_rgb[perm[h:]])
    sv = np.linalg.svd(emb - emb.mean(0), compute_uv=False)
    arrays["embeddings"] = emb.astype(np.float32)

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return {
        "label": label,
        "grid": grid,
        "val_loss": trained["val_loss"],
        "sets": sets,
        "probe_r2": probe_r2,
        "emb_r2": float(emb_r2),
        "emb_evr3": float((sv[:3] ** 2).sum() / (sv**2).sum()),
        "arrays": put(buf.getvalue(), name=f"ex-2.1.3-{label}-arrays.npz"),
    }


def publish_results(results: list[dict], stats: dict, checkpoints: dict, evals: dict) -> dict:
    """Publish metrics (JSON), stacked per-cell arrays (npz), eval sets, and checkpoints."""
    import io
    import json

    import numpy as np

    from mini.store import get, put, set_ref

    cells = [{k: v for k, v in r.items() if k != "arrays"} for r in results]
    metrics = {"cells": cells, "corpus_stats": stats}
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.1.3-metrics.json"))
    for grid, art in evals.items():
        set_ref(f"{EVALS_REF}/{grid}", art)  # per-example prompts/answers for the report
    for label, ckpt in checkpoints.items():
        set_ref(f"{CKPT_REF}/{label}", ckpt)

    arrays = {}
    for r in results:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{name}": z[name] for name in z.files}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="ex-2.1.3-arrays.npz"))
    return {"n_cells": len(results)}


def main(ctx: Ctx) -> dict:
    from sca.data.named_colors import GRIDS

    grids = list(GRIDS)
    preps = ctx.map(prepare_corpus, grids, [GRIDS[g] for g in grids], role="prep")
    configs, cell_grids, labels = zip(*build_sweep(preps), strict=True)
    trained = ctx.map(train_one, configs, cell_grids, labels, role="train")
    evals_by_grid = {p["grid"]: p["evals"] for p in preps}
    evaled = ctx.map(eval_one, trained, [evals_by_grid[g] for g in cell_grids], cell_grids, role="eval")
    ckpts = dict(zip(labels, [t["checkpoint"] for t in trained], strict=True))
    stats = {p["grid"]: p["stats"] for p in preps}
    summary = ctx.run(publish_results, evaled, stats, ckpts, evals_by_grid, role="prep")

    def mean_over_seeds(grid: str, key: str) -> float:
        vals = [r["sets"]["named_holdout"][key] for r in evaled if r["grid"] == grid]
        return round(sum(vals) / len(vals), 3)

    return {
        **summary,
        "holdout_accuracy": {g: mean_over_seeds(g, "accuracy") for g in grids},
        "holdout_guess_dist": {g: mean_over_seeds(g, "guess_dist") for g in grids},
    }


experiment = Experiment(
    name="ex-2.1.3",
    main=main,
    roles={
        # Corpus sampling is a plain-numpy loop: give it real cores + headroom.
        "prep": dict(cpu=2, timeout=900),
        # Same architecture as the siblings; the full-grid cells are barely bigger.
        "train": dict(gpu="L4", timeout=1800),
        "eval": dict(gpu="L4", timeout=900),
    },
)
