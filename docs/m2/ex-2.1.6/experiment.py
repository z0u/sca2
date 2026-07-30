"""Experiment 2.1.6: anchoring *red* in the residual stream of a transformer.

The first anchored transformer: one regularizer term pulls the residual stream
of *red-labeled* equations toward a fixed direction, and we ask where the
concept landed and what it cost the task. See `report.py` for the
preregistration.

The testbed is the `v216` cell of ex-2.1.3 — 216 grid colors, one token each,
`name + name = name` equations, d64-L4 — so the only difference from a solved,
un-anchored baseline is the anchor. Peak anchor weight λ ∈ {0, 0.03, 0.1, 0.3}
crossed with three seeds, plus a star arm at λ=0.1 whose anneal starts 40 epochs
earlier. Per cell:

- **Behavior**: exact match and answer NLL on seen and held-out closed pairs,
  plus the distance metrics on `open` pairs, against the `sca.baselines` nulls.
- **Alignment**: cos(h, e₀) per layer × line position on an exhaustive probe
  set (216 colors × their 27 closed partners), contracted to the label-affinity
  margin the hypotheses score.
- **Leakage**: held-out ridge R² for redness at op1 with the anchor direction
  projected out — how much of *red* lives somewhere other than the axis.
- **Trajectory**: the margin at op1 every 50 steps, against both schedules.

The anchoring machinery itself is `sca.anchoring`; this file is the design and
the DAG.

    bin/mini run docs/m2/ex-2.1.6/experiment.py --app modal --max-containers 8
    bin/mini status ex-2.1.6
"""

from __future__ import annotations

import numpy as np

from sca.anchoring import LINE_TOKENS, PROMPT_SPAN, anchor_weight as _anchor_weight
from sca.colorcube import redness, sim_to_red
from sca.config import SchedulerConfig
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS
from sca.training.scheduler import configure_schedule
from mini import Ctx, Experiment, get_data_dir

GRID = "v216"  # the ex-2.1.3 cell this builds on: 216 colors, one word each
SEEDS = [0, 1, 2]
PEAK_LR = 1e-2  # carried from ex-2.1.3, unchanged

CORPUS_SEED = 0
N_EXAMPLES = 100_000  # as ex-2.1.3, so the control is comparable to its published v216 cell
HOLDOUT_FRAC = 0.2  # of distinct closed pairs
N_EVAL = 256  # cap per eval set

LAMBDAS = [0.0, 0.03, 0.1, 0.3]
"""Peak anchor weight, the swept hyperparameter. λ=0 is the in-experiment control."""

SCORING_LAMBDA = 0.1
"""The rung hypotheses are reported on; H2's gates read *any* task-clean rung."""

H4_FLOOR = 0.2
"""Running-max floor below which an H4 run is reported but not scored.

A run the anchor never took hold of has a noise-level running maximum. Simulated
on unrelated 64-d unit states — α_c a mean over the 5 residual slices, m_op1 the
label-affinity margin over them — the margin's per-checkpoint noise is ≈0.021 and
the maximum over a full trajectory sits near 0.06, so a ratio of noise-level
margins fails the 0.8× gate about as often as not. The floor is ~3× that maximum
and well under H2(a)'s 0.5.

The 27 probe lines of a color do not average that noise down, because at op1 they
share a state: position 0 of a causal model attends to itself alone, so every
probe line of a color gives the same op1 alignment (`tests/sca/test_anchoring.py`).
Averaging over partners bites at the later positions, which H3 reads.
"""

RED_RATE = 0.08
"""P(labeled) for a pure-red first operand — the ceiling of the label distribution."""

SCHEDULER = SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01)
ANNEAL_START = 90.0  # epoch at which the anchor weight starts coming down
ANNEAL_END = 100.0  # and reaches the floor — the end of training, for every condition
ANNEAL_FLOOR = 0.1  # fraction of peak it settles at; never zero (M1/ex-2.9.3)
STAR_ANNEAL_START = 50.0  # the λ=0.1* star arm: starts down 40 epochs earlier, lands in the same place

TRAJ_STRIDE = 50  # steps between alignment measurements during training

LEVELS = np.asarray(GRIDS[GRID], dtype=float) / (N_LEVELS - 1)
GRID_RGB = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), axis=-1).reshape(-1, 3)
REDNESS = redness(GRID_RGB)
LABEL_P = REDNESS**8 * RED_RATE
"""P(line labeled red) for each of the 216 colors, as a function of its first operand."""

LABEL_W = LABEL_P / LABEL_P.sum()
"""The same distribution normalized: each color's share of all labeled lines.

Also the weight $u_c$ that every alignment statistic scores with.
"""

SIM_TARGET = sim_to_red(GRID_RGB, power=1.5)
"""M1's response shape mapped to alignment: angular HSV similarity to red, power 3/2.

M1's labels used the same `redness⁸` affinity as ours, yet the damage it
measured graded as sim³ under ablation and sim² under suppression — far
flatter than the affinity. Damage is quadratic in alignment (MSE ∝ align²),
so those map to alignment ∝ sim^1.5 and sim¹. The 3/2 power is the ablation
result on the alignment side, and it is also the family-covering choice:
R² ≥ 0.86 against every power of sim from 1 to 3, ≥ 0.96 against both
M1-derived alignment shapes. H2(b)'s second track scores against it.
"""

N_PROBE = 27
"""Probe lines per color: all 27 closed partners, so the probe set is exhaustive."""

BLOCK, BATCH = 64, 64  # ex-2.1.3's DataConfig, unchanged
LINES_PER_BATCH = BLOCK * BATCH / LINE_TOKENS
"""Lines a batch carries. Samples are packed token blocks, so a batch is ~680 equations, not 64."""

LABELED_PER_BATCH = float(LINES_PER_BATCH * LABEL_P.mean())
BATCHES_WITH_A_LABEL = float(1.0 - np.exp(-LABELED_PER_BATCH))
"""Poisson approximation; the exact per-line draws are independent."""

ANCHOR_SPAN = ("op1", "+", "op2", "=")
"""Positions the anchor term pulls: the prompt span. The answer and newline are measured, not pulled."""
assert len(ANCHOR_SPAN) == PROMPT_SPAN

METRICS_REF = "reports/m2/ex-2.1.6/metrics"
ARRAYS_REF = "reports/m2/ex-2.1.6/arrays"
EVALS_REF = "reports/m2/ex-2.1.6/evals"
PROBE_REF = "reports/m2/ex-2.1.6/probes"
CKPT_REF = "reports/m2/ex-2.1.6/checkpoints"  # + f"/{label}"
GEOMETRY_REF = "reports/m2/ex-2.1.6/geometry"  # post-hoc, published beside the preregistered metrics


EFFECTIVE_COLORS = float(np.exp(-(LABEL_W[LABEL_W > 0] * np.log(LABEL_W[LABEL_W > 0])).sum()))
"""Perplexity of the label distribution — how many colors it behaves like, against 216 real ones."""


def _midranks(v: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties sharing the mean rank of their group."""
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inv]


def step_rho(threshold: float) -> float:
    """Expected Spearman ρ of a pure step response `α = [redness > threshold]` against redness.

    The measured α is continuous (a mean of cosines), so a pure step orders the
    colors within each of its two levels arbitrarily: each α's expected rank is
    its level's midrank, while its rank variance is that of a full untied
    ranking. The expectation is over that arbitrary ordering — deterministic,
    no RNG — and Monte Carlo with within-level noise agrees to 3 decimals.
    """
    n = REDNESS.size
    rx, ry = _midranks(REDNESS), _midranks(REDNESS > threshold)
    return float(np.cov(rx, ry)[0, 1] / (np.std(rx, ddof=1) * np.sqrt(n * (n + 1) / 12.0)))


STEP_RHO_CEILING = max(step_rho(t) for t in np.unique(REDNESS)[:-1])
"""The best expected Spearman ρ any pure step response can score, over all step locations.

Below the H2(b) gate of 0.8, so a step cannot pass as a grade unless it also
carries residual within-level ordering — see the H2 partial in the report.
"""


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ with midranks for ties: Pearson correlation of the rank vectors."""
    return float(np.corrcoef(_midranks(a), _midranks(b))[0, 1])


SIM_RHO_CEILING = spearman(SIM_TARGET, REDNESS)
"""The noiseless Spearman ρ of a sim-family-shaped response against redness.

Every positive power of sim shares one ranking, so this ceiling covers the
whole family. The two notions of red order the 216 colors differently, so a
sim-graded response grades poorly on the rank track. Note this constant is
noise-free and uses midranks: sim takes only 23 distinct values here, and
crediting the ties lifts it just over the 0.8 gate. Break the ties at random
and it is ≈0.81; with measurement noise (unit amplitude, sd ≈0.056 per seed or
≈0.032 on the three-seed mean the gate is scored on) it is ≈0.76. This is why
H2(b) has a second track, where the same response scores R² ≈ 0.97.
"""


def r2_sim(alpha: np.ndarray) -> float:
    """Pearson R² between a per-color response and SIM_TARGET — M1's proportionality score.

    M1 scored interventions with `corrcoef(SIM3, mse)²` (`sca.colorcube
    .score_interventions`); this is the same statistic read on alignment
    instead of damage, with the target's exponent mapped accordingly (see
    `SIM_TARGET`). Pearson is scale- and offset-invariant, so "∝" needs no
    tuned constant, only the shape.
    """
    return float(np.corrcoef(alpha, SIM_TARGET)[0, 1] ** 2)


STEP_R2_CEILING = max(r2_sim((v > t).astype(float)) for v in (REDNESS, SIM_TARGET) for t in np.unique(v)[:-1])
"""The best R² any pure step response can score against SIM_TARGET, over step
locations in either redness or similarity order. Below the H2(b) gate of 0.8,
as with `STEP_RHO_CEILING`. (Pearson ignores within-level ordering, so this
one is deterministic with no expectation to take.)
"""


def anchor_weight(epoch, *, peak: float = 1.0, anneal_start: float = ANNEAL_START) -> np.ndarray:
    """The anchor weight λ at (fractional) *epoch*: ramp, hold, anneal to a floor.

    The design's own schedule, with the shared implementation in
    `sca.anchoring`; the ramp takes the LR warmup window, so the anchor arrives
    with the optimizer rather than ahead of it.
    """
    return _anchor_weight(
        epoch,
        peak=peak,
        warmup_epochs=SCHEDULER.warmup_epochs,
        anneal_start=anneal_start,
        anneal_end=ANNEAL_END,
        floor=ANNEAL_FLOOR,
    )


def learning_rate(epoch, *, peak: float = PEAK_LR, steps_per_epoch: int = 1000) -> np.ndarray:
    """The ex-2.1.3 LR schedule, evaluated at (fractional) *epoch*.

    The schedule itself is written in steps, so this is the real `optax` object
    sampled on an epoch axis rather than a re-description of it. The step count
    only sets the resolution of the warmup ramp; the curve is the same at any
    epoch length.
    """
    schedule = configure_schedule(SCHEDULER, peak, steps_per_epoch)
    return np.asarray(schedule(np.asarray(epoch, dtype=float) * steps_per_epoch))


CONDITIONS: list[dict] = [
    *({"name": f"lam{lam:g}", "lam": lam, "anneal_start": ANNEAL_START} for lam in LAMBDAS),
    # The star arm: same peak, protection withdrawn from epoch 50 instead of 90.
    {"name": "lam0.1-star", "lam": 0.1, "anneal_start": STAR_ANNEAL_START},
]

__all__ = ["experiment"]


def prepare_corpus(grid: str, red_rate: float, n_examples: int, holdout_frac: float) -> dict:
    """Build the corpus, the eval sets, the alignment probe set, and the label map.

    The design constants arrive as arguments rather than being read off the
    module, so that changing one re-runs this step instead of silently reusing
    a corpus built under the old value.
    """
    import numpy as np

    from sca import baselines
    from sca.colorcube import redness as colorcube_redness
    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import named_colors as nc
    from sca.data.colors import N_LEVELS, dump_example_sets, mix
    from mini.store import put

    levels = GRIDS[grid]
    palette = nc.grid_palette(levels)
    names = {v: k for k, v in palette.items()}
    corpus = nc.sample_corpus(n_examples, CORPUS_SEED, levels, holdout_frac)

    words = [w for ex in corpus for w in nc.as_words(ex)]
    tokenizer_config = TokenizerConfig(vocabulary=[*nc.SYNTAX, *palette])
    tokenizer = nc.WordTokenizer(tokenizer_config)
    tokens = np.asarray(tokenizer.encode_words(words), dtype=np.int32)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=sum(map(len, words)),
        sources=[DatasetMetadata(title=f"named-only color corpus ({grid})", fixes=[], total_chars=len(words))],
    )
    save_data(tokens, meta, get_data_dir() / "corpora" / grid)

    # Eval sets, built exactly as ex-2.1.3 built them, so the control is
    # comparable with its published v216 cell.
    held = nc.holdout_test(levels, CORPUS_SEED, holdout_frac)
    seen = sorted({p for ex in corpus if (p := ex.pair) is not None})
    holdout = [p for p in nc.closed_pairs(levels) if held(*p)]
    opens = nc.open_pairs(levels)
    rng = np.random.default_rng(CORPUS_SEED + 1)
    pick = lambda pairs, r: [pairs[i] for i in r.permutation(len(pairs))[:N_EVAL]]  # noqa: E731
    sets = {
        "named_seen": [nc.make_example(a, b, names, rng) for a, b in pick(seen, rng)],
        "named_holdout": [nc.make_example(a, b, names, rng) for a, b in pick(holdout, rng)],
        "open": [nc.make_example(a, b, names, rng) for a, b in pick(opens, rng)],
    }

    # The alignment probe set: every color as op1, against each of its closed
    # partners. Exhaustive, so there is no sampling seed and every cell is
    # measured on identical lines.
    colors = list(palette.values())
    on_grid = set(levels)
    probe_lines, partners = [], []
    for c in colors:
        closed = [b for b in colors if all(v in on_grid for v in mix(c, b))]
        partners.append(len(closed))
        for b in closed:
            probe_lines.append([names[c], "+", names[b], "=", names[mix(c, b)], "\n"])
    assert len(set(partners)) == 1, f"probe set is ragged: {sorted(set(partners))}"
    probe_tokens = np.array([tokenizer.encode_words(line) for line in probe_lines], dtype=np.int32)

    # P(labeled) by token id: the label draw reads the first operand alone. The
    # affinity is recomputed here from the palette, and checked against the
    # design constants the report draws its figures from.
    rgb = np.array(list(palette.values()), dtype=float) / (N_LEVELS - 1)
    affinity = colorcube_redness(rgb) ** 8 * red_rate
    np.testing.assert_allclose(rgb, GRID_RGB, rtol=0, atol=1e-12)
    np.testing.assert_allclose(affinity, LABEL_P, rtol=1e-12, atol=0)
    label_p = np.zeros(tokenizer.vocab_size, dtype=np.float64)  # the tokenizer's, which counts the pad token
    for name, p in zip(palette, affinity, strict=True):
        label_p[tokenizer.stoi[name]] = p
    weights = affinity / affinity.sum()

    # Model-free references for the distance rows (they never read the model).
    vocab_rgb = np.array(colors)
    train_dists = baselines.distances(vocab_rgb, [ex.result for ex in sets["named_seen"]])
    blind = baselines.blind_index(train_dists)
    nulls = {}
    for set_name, exs in sets.items():
        dists = baselines.distances(vocab_rgb, [ex.result for ex in exs])
        shell = baselines.shell_mask(levels, vocab_rgb, [ex.result for ex in exs])
        nulls[set_name] = {
            "blind": baselines.blind_stats(dists, blind),
            "k2": baselines.k_nearest_stats(dists, 2),
            "chance_dist": float(dists.mean()),
            "floor_dist": float(dists.min(axis=1).mean()),
            "neighborhood_exact": baselines.neighborhood_exact_null(shell),
        }

    stats = {
        "n_colors": len(palette),
        "n_seen_distinct": len(seen),
        "n_holdout": len(holdout),
        "n_open": len(opens),
        "n_partners": partners[0],
        "total_tokens": int(len(tokens)),
        "eval_n": {k: len(v) for k, v in sets.items()},
        "label_rate": float(affinity.mean()),
        "nulls": nulls,
        "blind_index": blind,
    }
    return {
        "meta": meta,
        "stats": stats,
        "evals": put(dump_example_sets(sets), name=f"ex-2.1.6-{grid}-evals.json"),
        "probes": put(
            _npz(probe_tokens=probe_tokens, label_p=label_p, weights=weights, redness=colorcube_redness(rgb)),
            name=f"ex-2.1.6-{grid}-probes.npz",
        ),
    }


def _npz(**arrays) -> bytes:
    import io

    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _make_config(vocab_size: int, seed: int):
    """The d64-L4 config from ex-2.1.3, unchanged; only λ differs between cells."""
    from sca.config import (
        DataConfig,
        ModelConfig,
        OptimizerConfig,
        TokenizerConfig,
        TrainingConfig,
    )

    return TrainingConfig(
        model=ModelConfig(
            vocab_size=vocab_size,
            block_size=BLOCK,
            n_embd=64,
            n_head=8,
            n_head_dim=8,
            n_ff=256,
            n_layer=4,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(batch_size=BATCH, oversample=16, train_split=0.9, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=PEAK_LR, betas=(0.9, 0.95)),
        scheduler=SCHEDULER,
        seed=seed,
    )


def build_sweep(prep: dict) -> list[tuple]:
    """(config, anchor, condition, label) cells; cheap and deterministic per wake.

    The anchor arrives as a plain dict of the whole schedule rather than a peak
    plus module constants, so every parameter of the pull is part of the cell's
    memo key.
    """
    from sca.utils import align

    tc = prep["meta"].tokenizer_config
    return [
        (
            _with_tokenizer(_make_config(align(tc.vocab_size, 64), seed), tc),
            {
                "peak": c["lam"],
                "warmup_epochs": SCHEDULER.warmup_epochs,
                "anneal_start": c["anneal_start"],
                "anneal_end": ANNEAL_END,
                "floor": ANNEAL_FLOOR,
            },
            c["name"],
            f"{c['name']}-s{seed}",
        )
        for c in CONDITIONS
        for seed in SEEDS
    ]


def _with_tokenizer(config, tokenizer_config):
    config.tokenizer = tokenizer_config.model_copy()
    return config


def train_one(config, anchor: dict, grid: str, traj_stride: int, probes, label: str) -> dict:
    """Train one cell with its anchor weight, recording the alignment trajectory."""
    import numpy as np

    from sca.anchoring import AnchorSpec
    from sca.compute.training import train_anchored
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, label_p, weights = z["probe_tokens"], z["label_p"], z["weights"]

    # One line per color for the trajectory: at op1 the partner cannot matter,
    # so the cheap probe reads the same number as the exhaustive one.
    first_of_color = probe_tokens[:: len(probe_tokens) // len(weights)]

    _, metrics, traj = train_anchored(
        config,
        get_data_dir() / "corpora" / grid,
        anchor=AnchorSpec(**anchor),
        label_p=label_p,
        probe_tokens=first_of_color,
        probe_weights=weights,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
    )
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {k: v.tolist() for k, v in traj.items()},
        "checkpoint": put(workdir / "model", name=f"ex-2.1.6-{label}-ckpt"),
    }


def eval_one(trained: dict, evals, probes, grid: str, condition: str, lam: float, label: str) -> dict:
    """Behavior, the alignment map, and off-axis leakage for one cell."""
    import numpy as np

    from sca.anchoring import ANCHOR_AXIS, alignment, margin
    from sca.compute.model import load_checkpoint
    from sca.data.colors import N_LEVELS, load_example_sets
    from sca.data.named_colors import GRIDS, WordTokenizer, grid_palette
    from mini.store import get, put

    workdir = get_data_dir() / "eval" / label
    get(trained["checkpoint"], workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    tokenizer = WordTokenizer(config.tokenizer)

    palette = grid_palette(GRIDS[grid])
    names = list(palette)
    name_idx = {n: i for i, n in enumerate(names)}
    color_ids = np.array([tokenizer.stoi[n] for n in names])
    vocab_rgb = np.array(list(palette.values()), dtype=np.float32) / (N_LEVELS - 1)

    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, weights, red = z["probe_tokens"], z["weights"], z["redness"]

    # --- Behavior: one teacher-forced pass per eval set, read at the pre-answer position.
    sets, arrays = {}, {}
    for set_name, exs in load_example_sets(get(evals, workdir / "evals.json").read_bytes()).items():
        logp = _answer_logprobs(model, tokenizer, exs)[:, color_ids]
        result = np.array([ex.result for ex in exs], dtype=np.float32) / (N_LEVELS - 1)
        dists = np.linalg.norm(vocab_rgb[None] - result[:, None], axis=2)  # (N, V)
        guess = logp.argmax(axis=1)
        guess_dist = dists[np.arange(len(exs)), guess]
        p = np.exp(logp)
        floor = dists.min(axis=1)
        stats: dict = {
            "n": len(exs),
            "guess_dist": float(guess_dist.mean()),
            "exp_dist": float((p / p.sum(1, keepdims=True) * dists).sum(1).mean()),
            "floor_dist": float(floor.mean()),
            "nearest_acc": float((guess_dist <= floor + 1e-9).mean()),
            "p_offvocab": float((1 - p.sum(1)).mean()),
        }
        if exs[0].answer:  # closed sets: the true answer is a vocabulary name
            true_idx = np.array([name_idx[ex.answer] for ex in exs])
            stats["accuracy"] = float((guess == true_idx).mean())
            stats["nll"] = float(-logp[np.arange(len(exs)), true_idx].mean())
            stats["failures"] = [
                (ex.prompt, ex.answer, names[g]) for ex, g, t in zip(exs, guess, true_idx, strict=True) if g != t
            ][:8]
        sets[set_name] = stats
        arrays[f"logp/{set_name}"] = logp.astype(np.float32)

    # --- Alignment: cos(h, e₀) per layer × position, averaged over each color's partners.
    n_partners = len(probe_tokens) // len(weights)
    cos = alignment(model, probe_tokens)  # (L1, C*P, T)
    alpha = cos.reshape(cos.shape[0], len(weights), n_partners, cos.shape[2]).mean(axis=2)  # (L1, C, T)
    m = margin(alpha, weights)  # (L1, T)
    arrays["alpha"] = alpha.astype(np.float32)
    arrays["margin"] = m.astype(np.float32)

    # --- Leakage: redness from the residual stream at op1, with the anchor axis removed.
    #     At op1 a color's state is context-free, so the 216 colors are the whole sample.
    states = _op1_states(model, probe_tokens[::n_partners])  # (L1, C, D)
    off_axis = np.delete(states, ANCHOR_AXIS, axis=2)
    leak = [_cv_ridge_r2(off_axis[d], red) for d in range(off_axis.shape[0])]
    on_axis = [float(np.corrcoef(states[d, :, ANCHOR_AXIS], red)[0, 1] ** 2) for d in range(states.shape[0])]

    return {
        "label": label,
        "condition": condition,
        "lam": lam,
        "val_loss": trained["val_loss"],
        "train_loss": trained["train_loss"],
        "traj": trained["traj"],
        "sets": sets,
        "m_op1": float(m[:, 0].mean()),
        "m_op1_by_layer": m[:, 0].tolist(),
        "m_by_position": m.mean(axis=0).tolist(),
        "alpha_mean_op1": float(alpha[:, :, 0].mean()),
        "leak_r2": leak,
        "axis_r2": on_axis,
        "arrays": put(_npz(**arrays), name=f"ex-2.1.6-{label}-arrays.npz"),
    }


def _answer_logprobs(model, tokenizer, exs) -> "np.ndarray":
    """Full-vocabulary log-probabilities at the pre-answer position, per example."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np

    forward = eqx.filter_jit(model.__call__)
    seq = np.array([tokenizer.encode_words(ex.prompt.split()) for ex in exs])
    out = [
        np.asarray(jax.nn.log_softmax(forward(jnp.asarray(seq[i : i + 256]))[:, -1], axis=-1))
        for i in range(0, len(seq), 256)
    ]
    return np.concatenate(out)


def _op1_states(model, lines) -> "np.ndarray":
    """The residual stream at op1 for one line per color: (L1, C, D)."""
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np

    stream = eqx.filter_jit(model.residual_stream)
    return np.asarray(stream(jnp.asarray(lines)))[:, :, 0, :]


def _cv_ridge_r2(x, y, n_folds: int = 5, penalties=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)) -> float:
    """Held-out R² for a ridge probe, with the penalty chosen inside each training fold.

    No color is scored by a probe that saw it, and no penalty is picked by
    looking at the number reported. The split is a fixed interleave rather than
    a draw, so the estimate has no seed.
    """
    import numpy as np

    from sca.compute.evaluation import ridge_probe

    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64).reshape(-1, 1)
    fold = np.arange(len(x)) % n_folds
    preds = np.empty_like(y)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        inner = np.arange(tr.sum()) % n_folds
        best = max(
            penalties,
            key=lambda l2: float(
                np.mean([ridge_probe(x[tr][inner != j], y[tr][inner != j], x[tr][inner == j], y[tr][inner == j], l2)[2]
                         for j in range(n_folds)])
            ),
        )  # fmt: skip
        w, b, _ = ridge_probe(x[tr], y[tr], x[te], y[te], best)
        preds[te] = x[te] @ w + b
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def explore_geometry(checkpoints: dict, probes, grid: str) -> dict:
    """Post-hoc (added after the results): the shape of the color cloud, and what is
    special about redness off the anchor axis.

    Two questions the preregistered measurements left open.

    *Did the cube compress?* M1 needed a `separate` term to stop the latent
    collapsing into a small region of the sphere, and this experiment left that
    term out too. Here `|mean|` is how concentrated the 216 op1 states are
    (1 = all identical) and `spread` is what extent they keep; a cloud whose
    centroid swung onto the anchor keeps its spread, a collapsing one does not.

    *Is the off-axis leakage about red?* The report's number is a redness probe
    with the anchor direction removed. Redness is a function of color, and the
    task needs the color cube, so a high score may say only that the cube
    survived. Running the same probe for greenness, blueness and the raw
    channels is what tells those apart.
    """
    import numpy as np

    from sca.anchoring import ANCHOR_AXIS
    from sca.colorcube import redness as colorcube_redness
    from sca.compute.model import load_checkpoint
    from sca.data.colors import N_LEVELS
    from sca.data.named_colors import GRIDS, grid_palette
    from mini.store import get

    palette = grid_palette(GRIDS[grid])
    rgb = np.array(list(palette.values()), dtype=np.float64) / (N_LEVELS - 1)
    roll = lambda k: colorcube_redness(np.roll(rgb, -k, axis=1))  # noqa: E731 — same formula, other corner
    targets = {
        "redness": roll(0),
        "greenness": roll(1),  # [r,g,b] -> [g,b,r], so the same formula reads green
        "blueness": roll(2),
        "R": rgb[:, 0],
        "G": rgb[:, 1],
        "B": rgb[:, 2],
        "sim^1.5": SIM_TARGET.astype(np.float64),
    }

    workdir = get_data_dir() / "explore"
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["probe_tokens"]
    per_color = probe_tokens[:: len(probe_tokens) // len(rgb)]

    out: dict[str, dict] = {}
    for label, ckpt in checkpoints.items():
        get(ckpt, workdir / label / "model")
        model, _, _ = load_checkpoint(workdir / label)
        h = _op1_states(model, per_color).astype(np.float64)  # (L1, C, D)
        off = np.delete(h, ANCHOR_AXIS, axis=2)
        on = h[:, :, ANCHOR_AXIS]
        centre = h.mean(axis=1)  # (L1, D)
        centred = h - centre[:, None]
        out[label] = {
            "mean_norm": np.linalg.norm(centre, axis=1).tolist(),
            "spread": (np.einsum("lcd,lcd->l", centred, centred) / h.shape[1]).tolist(),
            "centre_dot_anchor": (centre[:, ANCHOR_AXIS] / np.linalg.norm(centre, axis=1)).tolist(),
            "off_axis_r2": {t: [_cv_ridge_r2(off[d], y) for d in range(off.shape[0])] for t, y in targets.items()},
            "on_axis_r2": {
                t: [float(np.corrcoef(on[d], y)[0, 1] ** 2) for d in range(on.shape[0])] for t, y in targets.items()
            },
        }
    return out


def publish_geometry(geometry: dict) -> dict:
    """Publish the post-hoc geometry under its own ref, beside the preregistered metrics."""
    import json

    from mini.store import put, set_ref

    set_ref(GEOMETRY_REF, put(json.dumps(geometry, indent=2).encode(), name="ex-2.1.6-geometry.json"))
    return {"n_cells": len(geometry)}


def publish_results(results: list[dict], stats: dict, checkpoints: dict, evals, probes) -> dict:
    """Publish metrics (JSON), stacked per-cell arrays (npz), the eval and probe sets."""
    import io
    import json

    import numpy as np

    from mini.store import get, put, set_ref

    cells = [{k: v for k, v in r.items() if k != "arrays"} for r in results]
    metrics = {
        "cells": cells,
        "corpus_stats": stats,
        "conditions": CONDITIONS,
        "design": {
            "lambdas": LAMBDAS,
            "seeds": SEEDS,
            "scoring_lambda": SCORING_LAMBDA,
            "h4_floor": H4_FLOOR,
            "anneal": {"start": ANNEAL_START, "end": ANNEAL_END, "floor": ANNEAL_FLOOR, "star": STAR_ANNEAL_START},
            "traj_stride": TRAJ_STRIDE,
            "anchor_span": list(ANCHOR_SPAN),
        },
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.1.6-metrics.json"))
    set_ref(EVALS_REF, evals)
    set_ref(PROBE_REF, probes)
    for label, ckpt in checkpoints.items():
        set_ref(f"{CKPT_REF}/{label}", ckpt)

    arrays = {}
    for r in results:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{name}": z[name] for name in z.files}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="ex-2.1.6-arrays.npz"))
    return {"n_cells": len(results)}


def main(ctx: Ctx) -> dict:
    prep = ctx.run(prepare_corpus, GRID, RED_RATE, N_EXAMPLES, HOLDOUT_FRAC, role="prep")
    configs, anchors, conditions, labels = zip(*build_sweep(prep), strict=True)
    n = len(labels)
    trained = ctx.map(
        train_one, configs, anchors, [GRID] * n, [TRAJ_STRIDE] * n, [prep["probes"]] * n, labels, role="train"
    )
    evaled = ctx.map(
        eval_one,
        trained,
        [prep["evals"]] * n,
        [prep["probes"]] * n,
        [GRID] * n,
        conditions,
        [a["peak"] for a in anchors],
        labels,
        role="eval",
    )
    ckpts = dict(zip(labels, [t["checkpoint"] for t in trained], strict=True))
    summary = ctx.run(publish_results, evaled, prep["stats"], ckpts, prep["evals"], prep["probes"], role="prep")
    # Post-hoc, added after the results came in; see the report's exploratory section.
    geometry = ctx.run(explore_geometry, ckpts, prep["probes"], GRID, role="eval")
    ctx.run(publish_geometry, geometry, role="prep")

    def by_condition(fn) -> dict:
        return {
            c["name"]: round(float(np.mean([fn(r) for r in evaled if r["condition"] == c["name"]])), 3)
            for c in CONDITIONS
        }

    return {
        **summary,
        "holdout_accuracy": by_condition(lambda r: r["sets"]["named_holdout"]["accuracy"]),
        "m_op1": by_condition(lambda r: r["m_op1"]),
        "alpha_mean_op1": by_condition(lambda r: r["alpha_mean_op1"]),
    }


experiment = Experiment(
    name="ex-2.1.6",
    main=main,
    roles={
        # Corpus sampling is a plain-numpy loop over 100k lines; the probe set is trivial beside it.
        "prep": dict(cpu=2, timeout=900),
        # ~3.3k steps of 64×64 tokens. The anchor term reads every residual slice,
        # so a step costs more than ex-2.1.3's, and the trajectory adds a probe pass every 50.
        "train": dict(gpu="L4", timeout=3600, watchdog=300, watchdog_grace=900),
        # Three teacher-forced eval sets, one probe pass, and 5 × 5-fold ridge probes on 216 rows.
        "eval": dict(gpu="L4", timeout=900),
    },
)
