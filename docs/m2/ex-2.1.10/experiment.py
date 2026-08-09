"""Experiment 2.1.10: a label that doesn't point.

Every anchored experiment so far keyed its labels on the first operand, so the
label itself told the pull which position carries the concept — ex-2.1.9's
pooled term found op1 unaided, but only op1 ever drew the label. A document-
level label on natural text says "red occurs in here" and nothing else. So the
labeller widens by one step: a line is labeled when *either* operand draws,
with the per-slot rate halved to keep the label budget where it was, and the
pooled pull has to find the red operand line by line.

The testbed, measurements, anchor schedule, anti-subspace operating point and
mellowmax pooling are ex-2.1.9's, unchanged; only the labelling rule moves
(plus two soft-τ arms, reported but not gated).

    bin/mini run docs/m2/ex-2.1.10/experiment.py --app modal --max-containers 8
    bin/mini status ex-2.1.10
"""

from __future__ import annotations

import numpy as np

from sca.anchoring import (
    PROMPT_SPAN,
    anchor_weight as _anchor_weight,
    anti_subspace_weight as _anti_subspace_weight,
    softmin_weights,
)
from sca.colorcube import redness, sim_to_red
from sca.config import SchedulerConfig
from sca.data.colors import N_LEVELS
from sca.data.named_colors import GRIDS
from mini import Ctx, Experiment, get_data_dir

__all__ = ["experiment", "softmin_weights"]  # the latter re-exported for the report's calibration cells

# --- Testbed, carried from ex-2.1.9 unchanged --------------------------------

GRID = "v216"  # the ex-2.1.3 cell: 216 colors, one word each, d64-L4
SEEDS = [0, 1, 2]
"""The base seeds, shared with ex-2.1.9. Each condition carries its own seed
count (see CONDITIONS); a condition's seeds are range(count), so every
condition includes these three."""
PEAK_LR = 1e-2

CORPUS_SEED = 0
N_EXAMPLES = 100_000
HOLDOUT_FRAC = 0.2  # of distinct closed pairs
N_EVAL = 256  # cap per eval set
N_PROBE = 27  # probe lines per color: all closed partners, exhaustive

SCHEDULER = SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01)
ANNEAL_START = 90.0  # epoch at which the anchor weight starts coming down
ANNEAL_END = 100.0  # and reaches the floor — the end of training
ANNEAL_FLOOR = 0.1  # fraction of peak it settles at; never zero (M1/ex-2.9.3)
TRAJ_STRIDE = 50  # steps between alignment measurements during training
BLOCK, BATCH = 64, 64  # ex-2.1.3's DataConfig, unchanged

LEVELS = np.asarray(GRIDS[GRID], dtype=float) / (N_LEVELS - 1)
GRID_RGB = np.stack(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), axis=-1).reshape(-1, 3)
REDNESS = redness(GRID_RGB)

SIM_TARGET = sim_to_red(GRID_RGB, power=1.5)
"""M1's ablation response shape mapped to alignment; see ex-2.1.6 for the derivation."""

SCORING_LAMBDA = 0.1
"""Anchor peak for every cell — ex-2.1.6's scoring rung, held fixed throughout."""

SPAN = PROMPT_SPAN
"""The four span roles the pooled pull chooses among: op1, `+`, op2, `=`."""

ROLES = ["op1", "+", "op2", "="]
"""Display names for the span roles, in position order."""

# --- The operating point, from ex-2.1.8 (via ex-2.1.9) -----------------------

ANTI_PEAK_RATIO = 2.5
"""μ/λ at epoch 0 — M1/ex-2.9.1's dopesheet, unchanged."""

ANTI_ANNEAL_END = 90.0
ANTI_HOLD_RATIO = 0.30
"""Ex-2.1.8's `end90-hold30`, fixed in every anchored cell, as in ex-2.1.9."""

# --- The labelling rule -------------------------------------------------------

RED_RATE = 0.08
"""P(labeled) for a pure-red first operand under the op1-keyed labeller — the
rate every experiment since ex-2.1.6 used, kept for the reference arm."""

PER_SLOT_RATE = RED_RATE / 2
"""The either-slot labeller's per-operand rate. Each operand draws
independently at redness⁸ · PER_SLOT_RATE and the line is labeled when either
draws. Halving keeps the expected labeled-line rate at the op1 labeller's, up
to the (tiny) both-draw overlap: E[labeled] = 2·E[p] − E[p]² per line against
the reference's 2·E[p], where p is the per-slot probability. The report's
label-budget cell computes both exactly over the corpus distribution."""


def p_slot(r: np.ndarray | float, rate: float = PER_SLOT_RATE) -> np.ndarray | float:
    """P(this operand draws a label), as a function of its color's redness."""
    return r**8 * rate


def line_label_p(r1: np.ndarray | float, r2: np.ndarray | float) -> np.ndarray | float:
    """P(line labeled) under the either-slot labeller: either operand draws."""
    p1, p2 = p_slot(r1), p_slot(r2)
    return p1 + p2 - p1 * p2


LABEL_P = REDNESS**8 * RED_RATE
"""The op1-keyed labeller's per-color rate — the reference arm's labels, and
the affinity the op1-keyed continuity statistics weight with."""

LABEL_W = LABEL_P / LABEL_P.sum()
"""The same distribution normalized: the weight $u_c$ the per-color statistics
(m_span, the grading response) score with, unchanged from ex-2.1.6 on."""

# --- The pooling --------------------------------------------------------------

TAUS = [0.100, 0.500, 2.500]
"""The mellowmax temperature ladder, geometric at ×5 from ex-2.1.9's τ = 0.1.

τ is a temperature: the weight ratio between two span positions is
exp(Δx/τ), so behavior changes multiplicatively and a ×2 step would give
near-duplicate arms. 0.1 is the only ex-2.1.9 rung whose op1 response stayed
graded in every seed (latch rate 0/3), so it is the primary here; 0.5 and 2.5
walk toward the mean, and at 2.5 the weight ratios on realistic alignment
spreads are ~1.2 — a deliberate "tiny bias toward the best-aligned position"
arm, nearly the span mean that ex-2.1.9's τ = ∞ arm pinned. The calibration
cell in `report.py` recomputes the leading-weight curves from ex-2.1.9's
published arrays to place the ladder before the freeze."""

LEAD_BAND = (0.6, 0.8)
"""Ex-2.1.9's target range for the leading softmin weight, kept for the
calibration figure's reference band."""


def mellowmax_min(x: np.ndarray, tau: float, axis: int = -1) -> np.ndarray:
    """The soft minimum −τ·log(mean exp(−x/τ)) along *axis*, as in ex-2.1.9."""
    if np.isinf(tau):
        return x.mean(axis=axis)
    z = -x / tau
    m = z.max(axis=axis, keepdims=True)
    return -tau * (np.log(np.mean(np.exp(z - m), axis=axis)) + np.squeeze(m, axis=axis))


def anchor_weight(epoch, *, peak: float = SCORING_LAMBDA) -> np.ndarray:
    """The anchor weight λ at (fractional) *epoch*. Unchanged from ex-2.1.6."""
    return _anchor_weight(
        epoch,
        peak=peak,
        warmup_epochs=SCHEDULER.warmup_epochs,
        anneal_start=ANNEAL_START,
        anneal_end=ANNEAL_END,
        floor=ANNEAL_FLOOR,
    )


def anti_subspace_weight(
    epoch,
    *,
    lam: float = SCORING_LAMBDA,
    anneal_end: float = ANTI_ANNEAL_END,
    hold_ratio: float = ANTI_HOLD_RATIO,
    peak_ratio: float = ANTI_PEAK_RATIO,
) -> np.ndarray:
    """The anti-subspace weight μ at (fractional) *epoch*, at the operating point."""
    return _anti_subspace_weight(
        epoch,
        lam=lam,
        peak_ratio=peak_ratio,
        hold_ratio=hold_ratio,
        anneal_end=anneal_end,
        anchor_anneal_start=ANNEAL_START,
        anchor_anneal_end=ANNEAL_END,
        floor=ANNEAL_FLOOR,
    )


# --- Conditions ----------------------------------------------------------------


N_SEEDS_PRIMARY = 9
"""Seeds for the primary condition. Ex-2.1.9's deep-slice choice was one-hot
per run, so the failure of interest is a rate across runs — three seeds can't
distinguish "latched once in three" from bad luck, and τ = 0.1's clean 0/3
there may itself have been luck. Nine on the one gated condition gives the
latch rate a usable denominator; the references and unscored arms keep three."""


def _either(tau: float, seeds: int = 3) -> dict:
    return {
        "name": f"either-t{tau * 1000:03.0f}",
        "lam": SCORING_LAMBDA,
        "tau": tau,
        "pull": "span",
        "labels": "either",
        "seeds": seeds,
    }


CONDITIONS: list[dict] = [
    {"name": "lam0", "lam": 0.0, "tau": np.inf, "pull": "span", "labels": "either", "seeds": 3},
    {"name": "op1-labels", "lam": SCORING_LAMBDA, "tau": 0.100, "pull": "span", "labels": "op1", "seeds": 3},
    _either(0.100, seeds=N_SEEDS_PRIMARY),
    _either(0.500),
    _either(2.500),
    {"name": "slot-oracle", "lam": SCORING_LAMBDA, "tau": np.inf, "pull": "slot", "labels": "either", "seeds": 3},
]
"""Six conditions; three seeds each except the primary's nine (24 runs). A
condition's seeds are range(seeds). Every anchored cell runs the anti-subspace
term at the operating point and (where the pull is pooled) the mellowmax over
the prompt span; only the labelling rule and τ vary.

`op1-labels` re-runs ex-2.1.9's `pool-t100` verbatim — op1-keyed labels at
RED_RATE, same seeds, same draws — so it is both the reproduction check for
the new labelling code path and the baseline that says what widening the
labels costs. `slot-oracle` runs the either-slot labeller but pulls only the
operand(s) that drew the label (`pull: "slot"`, no pooling needed): what
knowing which slot triggered is worth, measured through identical code — a
reference rather than a strict ceiling, for the same reason ex-2.1.9's
op1-oracle was.

Label draws are consumed identically by every condition sharing a labeller,
so conditions with the same `labels` value see identical batches and masks at
a given seed; across labellers the batches differ, and cross-labeller
comparisons carry that corpus-draw noise on top of seed noise."""

EITHER_NAMES = [_either(t)["name"] for t in TAUS]
REFERENCE_ARM = "op1-labels"
ORACLE_ARM = "slot-oracle"

PRIMARY = "either-t100"
"""The primary — the condition H2–H4 score, fixed before the run: the either-slot arm at
τ = 0.1, the only rung ex-2.1.9 found graded in every seed — a criterion from
the previous experiment's published result, independent of every statistic
gated here. The other two either-slot arms are the soft-τ sweep: reported
beside the primary, not gated."""

# --- Statistics shared with the report -------------------------------------------


def pooled_margin(alpha: np.ndarray, weights: np.ndarray) -> float:
    """m_span: mean over slices of the max-over-span margin, as in ex-2.1.9.

    *alpha* is (slices, colors, positions) keyed by op1 on the probe set;
    kept for continuity with ex-2.1.9's scored statistic.
    """
    m = np.einsum("c,lct->lt", weights, alpha) - alpha.mean(axis=1)  # (L1, T)
    return float(m[:, :SPAN].max(axis=1).mean())


def line_margin(alpha_lines: np.ndarray, line_w: np.ndarray) -> float:
    """m_line: the per-line generalization of m_span, the selectivity this
    experiment scores.

    *alpha_lines* is (slices, lines, positions) on the probe set — one row per
    probe line rather than per color — and *line_w* is the labeller's own
    P(labeled) per line, normalized. Per slice: the line-weighted mean
    alignment at each span role minus the unweighted mean, keeping the largest
    role; then the mean over slices. Reduces to m_span when the weights key on
    op1 alone and lines are averaged over partners first. The max over roles
    is applied to every condition and the control alike, so its
    maximum-of-noise inflation cancels in the control-subtracted comparisons.
    """
    m = np.einsum("n,lnt->lt", line_w, alpha_lines) - alpha_lines.mean(axis=1)  # (L1, T)
    return float(m[:, :SPAN].max(axis=1).mean())


def alpha_span(alpha: np.ndarray) -> float:
    """ᾱ_span: the layer-mean of the max-over-span mean alignment, as in ex-2.1.9."""
    return float(alpha[:, :, :SPAN].mean(axis=1).max(axis=1).mean())


def r2_sim(alpha_op1: np.ndarray) -> float:
    """Squared Pearson correlation (r²) between a per-color op1 response and
    SIM_TARGET — the grading statistic, unchanged from ex-2.1.9.
    """
    return float(np.corrcoef(alpha_op1, SIM_TARGET)[0, 1] ** 2)


def group_weights(r1: np.ndarray, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-line weights for the two localization groups H2 scores.

    G1 weights each probe line by P(op1 drew and op2 didn't); G2 by the
    reverse. The groups are the labeller's own one-slot-only probabilities, so
    no redness threshold has to be invented, and lines where both operands are
    red (where either answer is defensible) get next to no say in either
    group. Each returned vector is normalized to sum to 1.
    """
    p1, p2 = np.asarray(p_slot(r1)), np.asarray(p_slot(r2))
    g1 = p1 * (1.0 - p2)
    g2 = (1.0 - p1) * p2
    return g1 / g1.sum(), g2 / g2.sum()


# --- Gates -------------------------------------------------------------------------

TASK_GATE = 0.02
"""H1: |named_holdout EM − control| within this, per condition (seed means). As ex-2.1.9."""

LEAD_GATE = 0.4
"""H2(a): the leading softmin weight that counts as concentration, per group,
at the embedding slice. Uniform over four roles is 0.25; carried from
ex-2.1.9's H4(a), which `pool-t100` passed at 0.77."""

CONTRAST_GATE = 0.2
CONTRAST_PARTIAL = 0.1
"""H2(b): required between-group contrast in op2 weight over the four
post-attention slices, mean over slices of π̄_G2(l, op2) − π̄_G1(l, op2).
A contrast in [CONTRAST_PARTIAL, CONTRAST_GATE) scores H2 as partial when (a)
also holds.

The discriminator between per-line localization and a global winner. Ex-2.1.9
found the deep-slice weights one-hot per run: if a run picks one role
globally, both groups see the same profile and the contrast is ~0; if the
pool follows the label line by line, G2's op2 weight should exceed G1's by
most of the one-hot amplitude. 0.2 sits well above the ~0 of the global
account and well below the ≳0.5 a per-line account predicts, on a statistic
bounded in [−1, 1] whose ex-2.1.9 seed spreads were a few hundredths."""

ORACLE_FRAC_GATE = 0.75
ORACLE_FRAC_PARTIAL = 0.50
"""H4: control-subtracted m_line(primary) as a fraction of control-subtracted
m_line(slot-oracle).

Scale-free because the labeller itself is new, so no earlier experiment
supplies an absolute target. The global-latch failure predicts roughly the
op1-triggered share of the label mass (~half, since the slots draw
symmetrically) plus whatever the shared token embedding gives away for free,
so a pass needs to clear that: 0.75 of the oracle is comfortably above the
latch account, 0.50 is its upper edge and scores as partial only. For scale,
ex-2.1.9's pooled arms recovered 91–98% of their oracle's m_span."""

MEAN_ALIGN_GATE = 0.1
"""H3(a): ceiling on ᾱ, the mean alignment over all 216 colors at op1. Carried
from ex-2.1.8/2.1.9; it is also the contamination detector — a pull that
latches op1 globally drags the non-red op1 states of op2-triggered lines onto
the axis, and that lands here."""

RETENTION_FLOOR = 0.2
RETENTION_GATE = 0.8
RETENTION_STAT = "min"
"""H3(b): retention on the m_line trajectory — for every run whose running
maximum reaches `RETENTION_FLOOR`, the final value is at least
`RETENTION_GATE`× that maximum, quoted as the minimum across seeds (settled in
ex-2.1.8). With nine seeds on the primary the minimum runs over nine, a
stricter read than ex-2.1.9's three; intentional."""

GRADE_R2_DROP = 0.10
"""H3(c): the largest grading loss tolerated, r²(op1-labels) − r²(primary), on the
seed-mean op1 response against SIM_TARGET. Relative, as in ex-2.1.9; the
reference arm reproduces `pool-t100`, whose r² was 0.83."""

NOISE_PER_SEED = 0.056
NOISE_SEED_MEAN = 0.032
"""Margin noise floors, carried from ex-2.1.6: same architecture, same measurement."""

# --- Published references ---------------------------------------------------------

EX219_REFERENCE = {
    "lam0": {"m_span": 0.023, "alpha_span": 0.018, "m_op1": -0.029, "alpha_op1": -0.012},
    "span-mean": {"m_span": 0.648, "alpha_span": 0.355, "m_op1": 0.619, "alpha_op1": 0.100, "r2": 0.73},
    "pool-t100": {
        "m_span": 0.704,
        "alpha_span": 0.242,
        "m_op1": 0.704,
        "alpha_op1": 0.075,
        "r2": 0.83,
        "lead_emb": 0.77,
    },
    "op1-oracle": {"m_span": 0.715, "alpha_span": 0.147, "m_op1": 0.715, "alpha_op1": 0.063, "r2": 0.87},
}
"""Ex-2.1.9's published results, recomputed from its arrays (2026-08-07) as
means of per-run statistics — the convention results are scored under.
`pool-t100` is the reproduction target for the `op1-labels` arm; the
calibration cells in `report.py` re-derive these from the store rather than
trusting this dict."""

# --- Result refs --------------------------------------------------------------------

METRICS_REF = "reports/m2/ex-2.1.10/metrics"
ARRAYS_REF = "reports/m2/ex-2.1.10/arrays"
EVALS_REF = "reports/m2/ex-2.1.10/evals"
PROBE_REF = "reports/m2/ex-2.1.10/probes"
CKPT_REF = "reports/m2/ex-2.1.10/checkpoints"  # + f"/{label}"
GEOMETRY_REF = "reports/m2/ex-2.1.10/geometry"

EX219_METRICS_REF = "reports/m2/ex-2.1.9/metrics"
EX219_ARRAYS_REF = "reports/m2/ex-2.1.9/arrays"
"""Ex-2.1.9's published results, read by the calibration cells and the carried
baseline figures."""


# --- The DAG ----------------------------------------------------------------


def prepare_corpus(grid: str, red_rate: float, per_slot_rate: float, n_examples: int, holdout_frac: float) -> dict:
    """Build the corpus, the eval sets, the alignment probe set, and both label maps.

    Ex-2.1.9's prep step with the either-slot labeller's tables added: the
    per-operand rate by token id (`slot_p`), and the per-probe-line quantities
    the line-keyed statistics need — P(line labeled) and each operand's redness.
    The corpus itself is unchanged down to the seed, so every condition here
    trains on the same lines as ex-2.1.9's cells did. The design constants
    arrive as arguments rather than being read off the module, so changing one
    re-runs this step instead of silently reusing a corpus built under the old
    value.
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
    # measured on identical lines. The operands' colors ride along per line for
    # the line-keyed statistics and the H2 group weighting.
    colors = list(palette.values())
    on_grid = set(levels)
    probe_lines, partners, op1_rgb, op2_rgb = [], [], [], []
    for c in colors:
        closed = [b for b in colors if all(v in on_grid for v in mix(c, b))]
        partners.append(len(closed))
        for b in closed:
            probe_lines.append([names[c], "+", names[b], "=", names[mix(c, b)], "\n"])
            op1_rgb.append(c)
            op2_rgb.append(b)
    assert len(set(partners)) == 1, f"probe set is ragged: {sorted(set(partners))}"
    assert partners[0] == N_PROBE, f"expected {N_PROBE} partners per color, got {partners[0]}"
    probe_tokens = np.array([tokenizer.encode_words(line) for line in probe_lines], dtype=np.int32)

    # The labellers' tables, recomputed from the palette and checked against the
    # design constants the report draws its figures from. `label_p` is the
    # op1-keyed labeller (P(line labeled) by op1's token id, the reference arm);
    # `slot_p` is the either-slot labeller's per-operand rate by token id.
    rgb = np.array(list(palette.values()), dtype=float) / (N_LEVELS - 1)
    affinity = colorcube_redness(rgb) ** 8 * red_rate
    slot_affinity = colorcube_redness(rgb) ** 8 * per_slot_rate
    np.testing.assert_allclose(rgb, GRID_RGB, rtol=0, atol=1e-12)
    np.testing.assert_allclose(affinity, LABEL_P, rtol=1e-12, atol=0)
    np.testing.assert_allclose(slot_affinity, p_slot(REDNESS), rtol=1e-12, atol=0)
    label_p = np.zeros(tokenizer.vocab_size, dtype=np.float64)  # the tokenizer's, which counts the pad token
    slot_p = np.zeros(tokenizer.vocab_size, dtype=np.float64)
    for name, p, sp in zip(palette, affinity, slot_affinity, strict=True):
        label_p[tokenizer.stoi[name]] = p
        slot_p[tokenizer.stoi[name]] = sp
    weights = affinity / affinity.sum()

    # Per probe line: each operand's redness, and P(labeled) under the
    # either-slot labeller — the raw rate, so consumers normalize over whatever
    # subset of lines they weight.
    r1 = colorcube_redness(np.asarray(op1_rgb, dtype=float) / (N_LEVELS - 1))
    r2 = colorcube_redness(np.asarray(op2_rgb, dtype=float) / (N_LEVELS - 1))
    p1, p2 = r1**8 * per_slot_rate, r2**8 * per_slot_rate
    line_p = p1 + p2 - p1 * p2
    np.testing.assert_allclose(line_p, line_label_p(r1, r2), rtol=1e-12, atol=0)

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
        "line_label_rate": float(line_p.mean()),
        "nulls": nulls,
        "blind_index": blind,
    }
    return {
        "meta": meta,
        "stats": stats,
        "evals": put(dump_example_sets(sets), name=f"ex-2.1.10-{grid}-evals.json"),
        "probes": put(
            _npz(
                probe_tokens=probe_tokens,
                label_p=label_p,
                slot_p=slot_p,
                weights=weights,
                line_p=line_p,
                r1=r1,
                r2=r2,
                redness=colorcube_redness(rgb),
            ),
            name=f"ex-2.1.10-{grid}-probes.npz",
        ),
    }


def _npz(**arrays) -> bytes:
    import io

    import numpy as np

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _make_config(vocab_size: int, seed: int):
    """The d64-L4 config from ex-2.1.3, unchanged; only the labelling rule differs between cells."""
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
    """(config, anchor, anti, labeller, tau, condition, label) cells; cheap and deterministic per wake.

    As in ex-2.1.9, both schedules arrive as plain dicts of every keyframe, so
    each cell's memo key carries the whole shape of its pull, its repulsion and
    its pooling. New here, the labeller rides beside them: keying, pull, and
    the per-draw rate, so a change to the labelling rule re-runs a cell the
    same way a schedule change would. A condition's seeds are range(seeds) —
    nine on the primary, three elsewhere.
    """
    from sca.utils import align

    tc = prep["meta"].tokenizer_config
    cells = []
    for c in CONDITIONS:
        anchor = {
            "peak": c["lam"],
            "warmup_epochs": SCHEDULER.warmup_epochs,
            "anneal_start": ANNEAL_START,
            "anneal_end": ANNEAL_END,
            "floor": ANNEAL_FLOOR,
            "span": SPAN,
            "tau": float(c["tau"]),
        }
        anti = (
            {
                "lam": c["lam"],
                "peak_ratio": ANTI_PEAK_RATIO,
                "hold_ratio": ANTI_HOLD_RATIO,
                "anneal_end": ANTI_ANNEAL_END,
                "anchor_anneal_start": ANNEAL_START,
                "anchor_anneal_end": ANNEAL_END,
                "floor": ANNEAL_FLOOR,
            }
            if c["lam"] > 0
            else None
        )
        labeller = {
            "keying": c["labels"],
            "pull": c["pull"],
            "rate": RED_RATE if c["labels"] == "op1" else PER_SLOT_RATE,
        }
        for seed in range(c["seeds"]):
            config = _with_tokenizer(_make_config(align(tc.vocab_size, 64), seed), tc)
            cells.append((config, anchor, anti, labeller, float(c["tau"]), c["name"], f"{c['name']}-s{seed}"))
    return cells


def _with_tokenizer(config, tokenizer_config):
    config.tokenizer = tokenizer_config.model_copy()
    return config


def train_one(
    config, anchor: dict, anti: dict | None, labeller: dict, grid: str, traj_stride: int, probes, label: str
) -> dict:
    """Train one cell under its labeller, recording the trajectory.

    Ex-2.1.9's step with the labelling rule made a variable: the mask comes
    from a `LabelSpec` built off the labeller dict, so the op1-keyed reference
    arm consumes the rng exactly as ex-2.1.9's cells did while the either-slot
    arms draw per operand. The trajectory adds m_line — weighted by the
    either-slot labeller's own P(labeled) for every condition alike, so
    retention is read off one instrument.
    """
    import numpy as np

    from sca.anchoring import AnchorSpec, AntiSpec, LabelSpec
    from sca.compute.training import train_anchored
    from mini.store import get, put

    workdir = get_data_dir() / "cells" / label
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens, label_p, slot_p = z["probe_tokens"], z["label_p"], z["slot_p"]
        weights, line_p = z["weights"], z["line_p"]

    spec = LabelSpec(
        p=label_p if labeller["keying"] == "op1" else slot_p,
        keying=labeller["keying"],
        pull=labeller["pull"],
    )

    # One line per color for the trajectory. At op1 the partner cannot matter;
    # at the other span roles it can, so the trajectory's span statistics read
    # one fixed partner per color where the endpoint eval averages all 27 —
    # cheap and consistent within a run, which is what retention compares.
    stride = len(probe_tokens) // len(weights)
    first_of_color = probe_tokens[::stride]
    line_w = line_p[::stride] / line_p[::stride].sum()

    _, metrics, traj = train_anchored(
        config,
        get_data_dir() / "corpora" / grid,
        anchor=AnchorSpec(**anchor),
        anti=AntiSpec(**anti) if anti is not None else None,
        label_p=spec,
        probe_tokens=first_of_color,
        probe_weights=weights,
        probe_line_w=line_w,
        checkpoint_dir=workdir,
        traj_stride=traj_stride,
    )
    return {
        "label": label,
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "traj": {k: v.tolist() for k, v in traj.items()},
        "checkpoint": put(workdir / "model", name=f"ex-2.1.10-{label}-ckpt"),
    }


def eval_one(trained: dict, evals, probes, grid: str, tau: float, condition: str, label: str) -> dict:
    """Behavior, the alignment map, leakage, and the weight profiles for one cell.

    Ex-2.1.9's eval with the measurement widened to lines: the alignment map
    and the softmin weights are stored per probe line (slices, lines,
    positions) rather than only averaged into per-color rows, which is what
    the per-group profiles (H2) and m_line (H4) read. The per-color
    contractions are kept beside them for continuity.
    """
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
        line_p = z["line_p"]
    line_w = line_p / line_p.sum()

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

    # --- Alignment: cos(h, e₀) per layer × position, kept per line and also
    #     averaged over each color's partners for the per-color statistics.
    n_partners = len(probe_tokens) // len(weights)
    cos = alignment(model, probe_tokens)  # (L1, C*P, T)
    alpha = cos.reshape(cos.shape[0], len(weights), n_partners, cos.shape[2]).mean(axis=2)  # (L1, C, T)
    m = margin(alpha, weights)  # (L1, T)
    arrays["alpha_lines"] = cos.astype(np.float32)
    arrays["alpha"] = alpha.astype(np.float32)
    arrays["margin"] = m.astype(np.float32)

    # --- The weight profiles: what the pull chose, at this cell's τ. Weights are
    #     computed per probe line (pooling is per line in the loss too) and stored;
    #     the per-color profile is kept beside them, as in ex-2.1.9.
    x = 1.0 - cos[:, :, :SPAN]
    w_line = softmin_weights(x, tau)  # (L1, C*P, SPAN)
    w_color = w_line.reshape(cos.shape[0], len(weights), n_partners, SPAN).mean(axis=2)
    pi = np.einsum("lct,c->lt", w_color, weights)
    np.testing.assert_allclose(pi.sum(axis=-1), 1.0, rtol=0, atol=1e-5)  # a mean of simplex weights stays on it
    arrays["w_line"] = w_line.astype(np.float32)
    arrays["pi"] = pi.astype(np.float32)

    # --- Leakage: redness from the residual stream at op1, with the anchor axis removed.
    #     At op1 a color's state is context-free, so the 216 colors are the whole sample.
    states = _op1_states(model, probe_tokens[::n_partners])  # (L1, C, D)
    off_axis = np.delete(states, ANCHOR_AXIS, axis=2)
    leak = [_cv_ridge_r2(off_axis[d], red) for d in range(off_axis.shape[0])]
    on_axis = [float(np.corrcoef(states[d, :, ANCHOR_AXIS], red)[0, 1] ** 2) for d in range(states.shape[0])]

    return {
        "label": label,
        "condition": condition,
        "tau": tau,
        "val_loss": trained["val_loss"],
        "train_loss": trained["train_loss"],
        "traj": trained["traj"],
        "sets": sets,
        "m_op1": float(m[:, 0].mean()),
        "m_op1_by_layer": m[:, 0].tolist(),
        "m_by_position": m.mean(axis=0).tolist(),
        "m_span": pooled_margin(alpha, weights),
        "m_line": line_margin(cos, line_w),
        "alpha_mean_op1": float(alpha[:, :, 0].mean()),
        "alpha_span": alpha_span(alpha),
        "pi": pi.tolist(),
        "leak_r2": leak,
        "axis_r2": on_axis,
        "arrays": put(_npz(**arrays), name=f"ex-2.1.10-{label}-arrays.npz"),
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


def _span_states(model, lines, batch_size: int = 1024) -> "np.ndarray":
    """The residual stream at every span role, batched: (L1, N, SPAN, D)."""
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np

    stream = eqx.filter_jit(model.residual_stream)
    chunks = [
        np.asarray(stream(jnp.asarray(lines[i : i + batch_size])))[:, :, :SPAN, :]
        for i in range(0, len(lines), batch_size)
    ]
    return np.concatenate(chunks, axis=1)


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


def _loo_color_r2(x, y_color, n_partners: int, penalties=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)) -> float:
    """Held-out R² for a ridge probe over per-line states, leaving one color out at a time.

    Rows arrive grouped by color, *n_partners* lines each, and every line of a
    color shares its target (the redness of op1), so the held-out unit has to
    be the color: at op1 the partner lines of a color are identical states, and
    a per-row split would test a probe on rows it trained on. The penalty is
    chosen one level up — once per profile entry, by an interleaved 5-group CV
    over the same colors — so the held-out fits share a penalty that every
    color had a say in, while no fit is tested on a color it trained on.
    """
    import numpy as np

    from sca.compute.evaluation import ridge_probe

    x = np.asarray(x, dtype=np.float64)
    y = np.repeat(np.asarray(y_color, dtype=np.float64), n_partners).reshape(-1, 1)
    n_colors = len(y_color)
    color = np.repeat(np.arange(n_colors), n_partners)
    group = color % 5  # fixed interleave over colors: no seed
    best = max(
        penalties,
        key=lambda l2: float(
            np.mean([ridge_probe(x[group != g], y[group != g], x[group == g], y[group == g], l2)[2] for g in range(5)])
        ),
    )
    preds = np.empty_like(y)
    for c in range(n_colors):
        tr, te = color != c, color == c
        w, b, _ = ridge_probe(x[tr], y[tr], x[te], y[te], best)
        preds[te] = x[te] @ w + b
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def readability_one(ckpt, probes, grid: str, label: str) -> list:
    """R²(l, t): how well each (slice, role) of one checkpoint reads op1's redness.

    Ex-2.1.9's readability profile, computed per cell rather than for all
    checkpoints in one container: the pass was the serial tail there, and this
    experiment has 24 cells, so it fans out with the map. The control rows are
    the reference profile the H2 figure draws dashed behind the weight
    profiles; the anchored rows are exploratory.
    """
    import numpy as np

    from sca.colorcube import redness as colorcube_redness
    from sca.compute.model import load_checkpoint
    from sca.data.colors import N_LEVELS
    from sca.data.named_colors import GRIDS, grid_palette
    from mini.store import get

    palette = grid_palette(GRIDS[grid])
    rgb = np.array(list(palette.values()), dtype=np.float64) / (N_LEVELS - 1)
    red = colorcube_redness(rgb)

    workdir = get_data_dir() / "readability" / label
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["probe_tokens"]
    n_partners = len(probe_tokens) // len(rgb)

    get(ckpt, workdir / "model")
    model, _, _ = load_checkpoint(workdir)
    h = _span_states(model, probe_tokens)  # (L1, C*P, SPAN, D)
    return [[_loo_color_r2(h[d, :, t], red, n_partners) for t in range(SPAN)] for d in range(h.shape[0])]


def geometry_one(ckpt, probes, grid: str, label: str) -> dict:
    """The shape of the color cloud, and what is special about redness off the anchor axis.

    Carried from ex-2.1.9 (where it is the mechanism picture behind
    containment), computed per cell so it fans out like the readability pass.
    The leakage probe is repeated for the other color statistics because
    redness is a function of color and the task needs the color cube;
    greenness, blueness and the raw channels beside it tell "red survived"
    apart from "color survived".
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

    workdir = get_data_dir() / "explore" / label
    with np.load(get(probes, workdir / "probes.npz")) as z:
        probe_tokens = z["probe_tokens"]
    per_color = probe_tokens[:: len(probe_tokens) // len(rgb)]

    get(ckpt, workdir / "model")
    model, _, _ = load_checkpoint(workdir)
    h = _op1_states(model, per_color).astype(np.float64)  # (L1, C, D)
    off = np.delete(h, ANCHOR_AXIS, axis=2)
    on = h[:, :, ANCHOR_AXIS]
    centre = h.mean(axis=1)  # (L1, D)
    centred = h - centre[:, None]
    return {
        "mean_norm": np.linalg.norm(centre, axis=1).tolist(),
        "spread": (np.einsum("lcd,lcd->l", centred, centred) / h.shape[1]).tolist(),
        "centre_dot_anchor": (centre[:, ANCHOR_AXIS] / np.linalg.norm(centre, axis=1)).tolist(),
        "off_axis_r2": {t: [_cv_ridge_r2(off[d], y) for d in range(off.shape[0])] for t, y in targets.items()},
        "on_axis_r2": {
            t: [float(np.corrcoef(on[d], y)[0, 1] ** 2) for d in range(on.shape[0])] for t, y in targets.items()
        },
    }


def publish_results(results: list[dict], stats: dict, readability: dict, checkpoints: dict, evals, probes) -> dict:
    """Publish metrics (JSON), stacked per-cell arrays (npz), the eval and probe sets."""
    import io
    import json

    import numpy as np

    from mini.store import get, put, set_ref

    cells = [{k: (None if k == "tau" and np.isinf(v) else v) for k, v in r.items() if k != "arrays"} for r in results]
    metrics = {
        "cells": cells,
        "corpus_stats": stats,
        "readability": readability,
        # JSON has no ∞, so the τ = ∞ arms carry null here; the report reads the
        # grid off the experiment module, where the value is np.inf.
        "conditions": [{**c, "tau": None if np.isinf(c["tau"]) else c["tau"]} for c in CONDITIONS],
        "design": {
            "seeds": SEEDS,
            "n_seeds_primary": N_SEEDS_PRIMARY,
            "scoring_lambda": SCORING_LAMBDA,
            "span": SPAN,
            "taus": TAUS,
            "lead_band": LEAD_BAND,
            "primary": PRIMARY,
            "either": EITHER_NAMES,
            "reference_arm": REFERENCE_ARM,
            "oracle_arm": ORACLE_ARM,
            "retention_stat": RETENTION_STAT,
            "labelling": {"red_rate": RED_RATE, "per_slot_rate": PER_SLOT_RATE},
            "anti": {
                "peak_ratio": ANTI_PEAK_RATIO,
                "anneal_end": ANTI_ANNEAL_END,
                "hold_ratio": ANTI_HOLD_RATIO,
            },
            "anneal": {"start": ANNEAL_START, "end": ANNEAL_END, "floor": ANNEAL_FLOOR},
            "traj_stride": TRAJ_STRIDE,
            "noise": {"per_seed": NOISE_PER_SEED, "seed_mean": NOISE_SEED_MEAN},
            "ex219_reference": EX219_REFERENCE,
            "gates": {
                "task": TASK_GATE,
                "lead": LEAD_GATE,
                "contrast": CONTRAST_GATE,
                "contrast_partial": CONTRAST_PARTIAL,
                "mean_align": MEAN_ALIGN_GATE,
                "retention_floor": RETENTION_FLOOR,
                "retention": RETENTION_GATE,
                "grade_r2_drop": GRADE_R2_DROP,
                "oracle_frac": ORACLE_FRAC_GATE,
                "oracle_frac_partial": ORACLE_FRAC_PARTIAL,
            },
        },
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.1.10-metrics.json"))
    set_ref(EVALS_REF, evals)
    set_ref(PROBE_REF, probes)
    for label, ckpt in checkpoints.items():
        set_ref(f"{CKPT_REF}/{label}", ckpt)

    arrays = {}
    for r in results:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{name}": z[name] for name in z.files}
    for label, profile in readability.items():
        arrays[f"{label}/readability"] = np.asarray(profile, dtype=np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="ex-2.1.10-arrays.npz"))
    return {"n_cells": len(results)}


def publish_geometry(geometry: dict) -> dict:
    """Publish the geometry pass under its own ref, beside the metrics."""
    import json

    from mini.store import put, set_ref

    set_ref(GEOMETRY_REF, put(json.dumps(geometry, indent=2).encode(), name="ex-2.1.10-geometry.json"))
    return {"n_cells": len(geometry)}


def main(ctx: Ctx) -> dict:
    prep = ctx.run(prepare_corpus, GRID, RED_RATE, PER_SLOT_RATE, N_EXAMPLES, HOLDOUT_FRAC, role="prep")
    configs, anchors, antis, labellers, taus, conditions, labels = zip(*build_sweep(prep), strict=True)
    n = len(labels)
    trained = ctx.map(
        train_one,
        configs,
        anchors,
        antis,
        labellers,
        [GRID] * n,
        [TRAJ_STRIDE] * n,
        [prep["probes"]] * n,
        labels,
        role="train",
    )
    evaled = ctx.map(
        eval_one,
        trained,
        [prep["evals"]] * n,
        [prep["probes"]] * n,
        [GRID] * n,
        taus,
        conditions,
        labels,
        role="eval",
    )
    ckpt_list = [t["checkpoint"] for t in trained]
    ckpts = dict(zip(labels, ckpt_list, strict=True))
    profiles = ctx.map(readability_one, ckpt_list, [prep["probes"]] * n, [GRID] * n, labels, role="probe")
    readability = dict(zip(labels, profiles, strict=True))
    summary = ctx.run(
        publish_results, evaled, prep["stats"], readability, ckpts, prep["evals"], prep["probes"], role="prep"
    )
    geoms = ctx.map(geometry_one, ckpt_list, [prep["probes"]] * n, [GRID] * n, labels, role="probe")
    ctx.run(publish_geometry, dict(zip(labels, geoms, strict=True)), role="prep")

    def by_condition(fn) -> dict:
        return {
            c["name"]: round(float(np.mean([fn(r) for r in evaled if r["condition"] == c["name"]])), 3)
            for c in CONDITIONS
        }

    return {
        **summary,
        "holdout_accuracy": by_condition(lambda r: r["sets"]["named_holdout"]["accuracy"]),
        "m_line": by_condition(lambda r: r["m_line"]),
        "m_span": by_condition(lambda r: r["m_span"]),
        "alpha_mean_op1": by_condition(lambda r: r["alpha_mean_op1"]),
    }


experiment = Experiment(
    name="ex-2.1.10",
    main=main,
    roles={
        # Corpus sampling is a plain-numpy loop over 100k lines; the probe set is trivial beside it.
        "prep": dict(cpu=2, timeout=900),
        # ~3.3k steps of 64×64 tokens, as ex-2.1.9. The watchdog is sized for the
        # gap *after* the last step: the checkpoint upload emits no step progress
        # and took over 300s in ex-2.1.7 when eight containers pushed to the
        # store at once (todo-eng).
        "train": dict(gpu="L4", timeout=3600, watchdog=900, watchdog_grace=900),
        # Three teacher-forced eval sets, one probe pass over 5832 lines, and
        # 5 × 5-fold ridge probes on 216 rows.
        "eval": dict(gpu="L4", timeout=2700),
        # The per-cell readability/geometry passes: ridge probes over per-line
        # states. CPU-bound linear algebra; the forward pass doesn't need a GPU,
        # and per-cell fan-out keeps any one container's share small.
        "probe": dict(cpu=4, timeout=1800),
    },
)
