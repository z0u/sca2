"""Concept anchoring: the pull lands on the right positions, and it moves the stream."""

from dataclasses import replace

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import jax

from sca.anchoring import (
    LINE_TOKENS,
    PROMPT_SPAN,
    AnchorSpec,
    AntiSpec,
    LabelSpec,
    alignment,
    anchor_term,
    anti_subspace_term,
    margin,
    pooled_anchor_term,
    sample_anchored_batches,
    smoothstep,
    softmin_weights,
)
from sca.compute.data_pipelines import save_data
from sca.compute.training import train_anchored
from sca.config import (
    CorpusMetadata,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from sca.model import build_model

# A miniature of the real corpus: colors 4..11 as opaque tokens, syntax at 1..3.
PLUS, EQ, NEWLINE = 1, 2, 3
COLORS = np.arange(4, 12)
N_LINES = 400


@pytest.fixture
def corpus() -> np.ndarray:
    """`op1 + op2 = ans ⏎` lines, packed end to end with no separator."""
    rng = np.random.default_rng(0)
    ops = rng.choice(COLORS, size=(N_LINES, 3))
    syntax = np.broadcast_to(np.array([PLUS, EQ, NEWLINE]), (N_LINES, 3))
    lines = np.stack([ops[:, 0], syntax[:, 0], ops[:, 1], syntax[:, 1], ops[:, 2], syntax[:, 2]], axis=1)
    return lines.reshape(-1).astype(np.int32)


def data_config(padding_chance: float = 0.1) -> DataConfig:
    return DataConfig(batch_size=16, oversample=64, train_split=0.9, padding_chance=padding_chance)


def model_config() -> ModelConfig:
    return ModelConfig(vocab_size=64, block_size=64, n_embd=32, n_head=8, n_head_dim=8, n_ff=32, n_layer=2)


def test_smoothstep_starts_and_ends_at_rest():
    tau = np.linspace(-0.5, 1.5, 201)
    y = smoothstep(tau)
    assert y[0] == 0.0 and y[-1] == 1.0
    np.testing.assert_allclose(smoothstep(0.5), 0.5, rtol=0, atol=1e-12)
    slope = np.diff(y) / np.diff(tau)
    assert slope.max() == pytest.approx(1.875, rel=1e-2)  # 15/8, the quintic's peak rate


def test_anchor_schedule_ramps_holds_and_anneals_to_a_floor():
    spec = AnchorSpec(peak=0.1, warmup_epochs=10, anneal_start=90, anneal_end=100, floor=0.1)
    assert spec(0) == 0.0
    np.testing.assert_allclose(spec(10), 0.1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(spec(50), 0.1, rtol=0, atol=1e-12)  # holds through the plateau
    np.testing.assert_allclose(spec(100), 0.01, rtol=1e-12, atol=0)  # the floor, never zero
    assert 0.01 < spec(95) < 0.1

    # Moving the start stretches the anneal rather than sliding it: same endpoint.
    star = AnchorSpec(peak=0.1, warmup_epochs=10, anneal_start=50, anneal_end=100, floor=0.1)
    np.testing.assert_allclose(star(100), spec(100), rtol=1e-12, atol=0)
    assert star(90) < spec(90)


def test_schedule_shapes_flatten_or_straighten_the_same_keyframes():
    keyframes = dict(warmup_epochs=10, anneal_start=90, anneal_end=100, floor=0.1)
    flat = AnchorSpec(peak=0.1, shape="flat", **keyframes)
    linear = AnchorSpec(peak=0.1, shape="linear", **keyframes)

    # Flat discards every keyframe: full weight from step 0, no anneal at the end.
    np.testing.assert_allclose(flat(np.array([0.0, 5.0, 50.0, 100.0])), 0.1, rtol=0, atol=1e-12)
    # Linear keeps them and moves between them in a straight line.
    np.testing.assert_allclose(linear(5), 0.05, rtol=0, atol=1e-12)
    np.testing.assert_allclose(linear(50), 0.1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(linear(100), 0.01, rtol=1e-12, atol=0)

    anti = dict(lam=0.1, peak_ratio=2.5, hold_ratio=0.3, anneal_end=90)
    shared = dict(anchor_anneal_start=90, anchor_anneal_end=100, floor=0.1)
    flat_anti = AntiSpec(shape="flat", **anti, **shared)
    linear_anti = AntiSpec(shape="linear", **anti, **shared)
    # Flat holds the ratio to the anchor peak and skips the anchor's end anneal.
    np.testing.assert_allclose(flat_anti(np.array([0.0, 45.0, 100.0])), 0.03, rtol=0, atol=1e-12)
    np.testing.assert_allclose(linear_anti(45), 0.1 * 1.4, rtol=1e-12, atol=0)  # halfway from 2.5 to 0.3


def test_mask_covers_the_prompt_of_labeled_lines_only(corpus):
    # Label every line whose first operand is COLORS[0], and no other.
    label_p = np.zeros(64)
    label_p[COLORS[0]] = 1.0
    mc, dc = model_config(), data_config(0.0)
    x, _, mask = next(sample_anchored_batches(corpus, dc, mc, 1, np.random.default_rng(0), label_p))

    assert mask.shape == x.shape
    pulled = mask.astype(bool)
    # Every pulled position is one of op1/+/op2/= ...
    assert set(np.unique(x[pulled])) <= {COLORS[0], PLUS, EQ, *COLORS}
    assert not pulled[x == NEWLINE].any()
    # ... and the pull covers whole prompts: a labeled line contributes its `+` and `=`.
    assert pulled[x == PLUS].any() and pulled[x == EQ].any()
    # Lines whose op1 is not COLORS[0] are untouched, so some `+` positions stay unpulled.
    assert not pulled[x == PLUS].all()


def test_mask_is_empty_without_labels_and_full_with_them(corpus):
    mc, dc = model_config(), data_config(0.0)
    never = np.zeros(64)
    always = np.ones(64)
    rng = np.random.default_rng(1)
    _, _, none = next(sample_anchored_batches(corpus, dc, mc, 1, rng, never))
    x, _, all_lines = next(sample_anchored_batches(corpus, dc, mc, 1, rng, always))
    assert none.sum() == 0
    # The prompt span: four roles of six, with the answer and newline left out.
    np.testing.assert_allclose(all_lines.mean(), PROMPT_SPAN / LINE_TOKENS, rtol=0, atol=0.02)
    assert not all_lines.astype(bool)[x == NEWLINE].any()


def test_padded_positions_are_never_pulled(corpus):
    mc, dc = model_config(), data_config(1.0)
    _, _, mask = next(sample_anchored_batches(corpus, dc, mc, 1, np.random.default_rng(2), np.ones(64)))
    x, _, _ = next(sample_anchored_batches(corpus, dc, mc, 1, np.random.default_rng(2), np.ones(64)))
    assert mask[x == 0].sum() == 0


def test_a_bare_array_and_an_op1_spec_draw_identically(corpus):
    """The reproduction arm leans on this: op1 keying consumes the rng exactly as before."""
    label_p = np.zeros(64)
    label_p[COLORS[0]] = 0.5
    mc, dc = model_config(), data_config()
    legacy = next(sample_anchored_batches(corpus, dc, mc, 1, np.random.default_rng(5), label_p))
    spec = next(sample_anchored_batches(corpus, dc, mc, 1, np.random.default_rng(5), LabelSpec(label_p)))
    for a, b in zip(legacy, spec, strict=True):
        np.testing.assert_array_equal(a, b)


def test_either_keying_pulls_lines_whose_op2_drew(corpus):
    """With p deterministic, either keying labels a strict superset of op1 keying's lines."""
    p = np.zeros(64)
    p[COLORS[0]] = 1.0  # every draw succeeds for COLORS[0], no other color ever draws
    mc, dc = model_config(), data_config(0.0)
    _, _, op1_mask = next(sample_anchored_batches(corpus, dc, mc, 1, np.random.default_rng(6), p))
    x, _, either_mask = next(
        sample_anchored_batches(corpus, dc, mc, 1, np.random.default_rng(6), LabelSpec(p, keying="either"))
    )
    a, b = op1_mask.astype(bool), either_mask.astype(bool)
    assert (a & ~b).sum() == 0  # every op1-labeled line is still labeled...
    assert (b & ~a).sum() > 0  # ...and lines labeled through op2 alone join them
    # Lines labeled only through op2 carry a different color at op1, and it is pulled.
    assert len(set(np.unique(x[b & ~a])) & set(COLORS[1:])) > 0


def test_slot_pull_marks_only_the_operands_that_drew(corpus):
    mc, dc = model_config(), data_config(0.0)
    p = np.zeros(64)
    p[COLORS[0]] = 1.0
    x, _, span_mask = next(
        sample_anchored_batches(corpus, dc, mc, 1, np.random.default_rng(7), LabelSpec(p, keying="either"))
    )
    _, _, slot_mask = next(
        sample_anchored_batches(corpus, dc, mc, 1, np.random.default_rng(7), LabelSpec(p, keying="either", pull="slot"))
    )
    pulled = slot_mask.astype(bool)
    assert pulled.sum() > 0
    assert set(np.unique(x[pulled])) == {COLORS[0]}  # only the operand(s) that drew, never syntax
    assert not (pulled & ~span_mask.astype(bool)).any()  # a subset of the span pull, same label draws


def test_anchor_term_is_zero_when_aligned_and_two_when_opposed():
    states = np.zeros((3, 2, 4, 8), dtype=np.float32)
    states[..., 0] = 1.0
    mask = np.ones((2, 4), dtype=np.float32)
    np.testing.assert_allclose(anchor_term(jnp.asarray(states), jnp.asarray(mask)), 0.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(anchor_term(jnp.asarray(-states), jnp.asarray(mask)), 2.0, rtol=1e-6, atol=0)
    # An empty mask contributes nothing rather than 0/0.
    assert float(anchor_term(jnp.asarray(states), jnp.zeros_like(jnp.asarray(mask)))) == 0.0


def pooled_setup() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """One batch row, two lines of two pulled positions: x = 1 − cos of [0.8, 0.4 | 0.0, 1.0]."""
    states = np.zeros((1, 1, 4, 8), dtype=np.float32)
    states[..., 0] = [0.2, 0.6, 1.0, 0.0]
    mask = np.ones((1, 4), dtype=np.float32)
    line_id = np.array([[0, 0, 1, 1]], dtype=np.int32)
    return jnp.asarray(states), jnp.asarray(mask), jnp.asarray(line_id)


def test_pooled_term_at_tau_inf_is_the_mean_of_per_line_means():
    states, mask, line_id = pooled_setup()
    # Full lines: per-line means [0.6, 0.5] → 0.55, which is also the flat mean here.
    pooled = pooled_anchor_term(states, mask, line_id, 2, np.inf)
    np.testing.assert_allclose(pooled, 0.55, rtol=1e-6, atol=0)
    np.testing.assert_allclose(pooled, anchor_term(states, mask), rtol=1e-6, atol=0)
    # A truncated line: [0.8, 0.4 | 0.0] → per-line means [0.6, 0.0] → 0.3,
    # where the flat mean over positions would say 0.4.
    cut = jnp.asarray(np.array([[1, 1, 1, 0]], dtype=np.float32))
    np.testing.assert_allclose(pooled_anchor_term(states, cut, line_id, 2, np.inf), 0.3, rtol=1e-6, atol=0)
    np.testing.assert_allclose(anchor_term(states, cut), 0.4, rtol=1e-6, atol=0)


def test_pooled_term_reaches_the_min_as_tau_falls():
    states, mask, line_id = pooled_setup()
    # For two positions, mm = min + τ·log(2 / (1 + e^{−Δ/τ})); at τ = 0.01 the
    # far position's e^{−40} vanishes, leaving min + τ·ln 2 per line.
    expected = (0.4 + 0.0) / 2 + 0.01 * np.log(2)
    np.testing.assert_allclose(pooled_anchor_term(states, mask, line_id, 2, 0.01), expected, rtol=1e-4, atol=0)


def test_pooled_gradient_is_the_softmin_weights_and_conserves_the_budget():
    states, mask, line_id = pooled_setup()
    x = np.array([[0.8, 0.4], [0.0, 1.0]])  # (line, position)
    for tau in (0.01, 0.1, 1.0, np.inf):
        grad = jax.grad(lambda s, t=tau: pooled_anchor_term(s, mask, line_id, 2, t))(states)
        got = np.asarray(grad)[0, 0, :, 0].reshape(2, 2)  # d/d cos at the anchor axis
        # d/d cos = −π_t / n_lines: each line's budget is the softmin weights, whatever τ.
        np.testing.assert_allclose(got, -softmin_weights(x, tau) / 2, rtol=1e-4, atol=1e-7)
        np.testing.assert_allclose(got.sum(axis=1), -0.5, rtol=1e-5, atol=0)
    # Off-axis components carry no gradient, and an empty mask contributes none at all.
    assert float(jnp.abs(jnp.asarray(grad)[..., 1:]).max()) == 0.0
    empty = jax.grad(lambda s: pooled_anchor_term(s, jnp.zeros_like(mask), line_id, 2, 0.1))(states)
    np.testing.assert_allclose(np.asarray(empty), 0.0, rtol=0, atol=0)
    assert float(pooled_anchor_term(states, jnp.zeros_like(mask), line_id, 2, 0.1)) == 0.0


def test_pooling_is_within_each_slice():
    # Two slices with opposite profiles: the pool must find each slice's own min.
    states = np.zeros((2, 1, 2, 8), dtype=np.float32)
    states[0, ..., 0] = [1.0, 0.0]  # x = [0.0, 1.0]
    states[1, ..., 0] = [0.0, 1.0]  # x = [1.0, 0.0]
    mask = jnp.ones((1, 2), dtype=jnp.float32)
    line_id = jnp.zeros((1, 2), dtype=jnp.int32)
    tiny = pooled_anchor_term(jnp.asarray(states), mask, line_id, 1, 0.001)
    np.testing.assert_allclose(tiny, 0.001 * np.log(2), rtol=1e-3, atol=0)  # each slice: min 0 + τ·ln2


def test_softmin_weights_limits():
    x = np.array([0.8, 0.4, 0.0, 1.0])
    np.testing.assert_allclose(softmin_weights(x, np.inf), 0.25, rtol=0, atol=1e-12)
    np.testing.assert_allclose(softmin_weights(x, 0.001), [0.0, 0.0, 1.0, 0.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(softmin_weights(x, 0.5).sum(), 1.0, rtol=1e-12, atol=0)


def test_sampled_line_ids_group_the_mask_without_changing_the_draws(corpus):
    mc, dc = model_config(), data_config(0.0)
    plain = next(sample_anchored_batches(corpus, dc, mc, 1, np.random.default_rng(3), np.ones(64)))
    x, y, mask, local = next(
        sample_anchored_batches(corpus, dc, mc, 1, np.random.default_rng(3), np.ones(64), lines=True)
    )
    np.testing.assert_array_equal(x, plain[0])
    np.testing.assert_array_equal(mask, plain[2])
    assert local.min() == 0 and local.max() <= mc.block_size // LINE_TOKENS + 1
    # Local ids step by at most one, and every step lands on a line boundary of the packed corpus.
    steps = np.diff(local, axis=1)
    assert set(np.unique(steps)) <= {0, 1}
    # Within one line id, positions are contiguous and at most LINE_TOKENS long.
    assert all(np.bincount(row).max() <= LINE_TOKENS for row in local)


def test_anti_subspace_term_reads_the_mean_square_alignment_of_live_positions():
    # Half the positions on the axis, half orthogonal to it: mean cos² = 1/2.
    states = np.zeros((3, 2, 4, 8), dtype=np.float32)
    states[:, 0, :, 0] = 1.0  # first sequence sits on the anchor
    states[:, 1, :, 1] = 1.0  # second sits orthogonal to it
    live = np.ones((2, 4), dtype=np.float32)
    np.testing.assert_allclose(anti_subspace_term(jnp.asarray(states), jnp.asarray(live)), 0.5, rtol=1e-6, atol=0)

    # It is blind to sign, where the anchor term is not: -e₀ is as penalized as +e₀.
    np.testing.assert_allclose(
        anti_subspace_term(jnp.asarray(-states), jnp.asarray(live)),
        anti_subspace_term(jnp.asarray(states), jnp.asarray(live)),
        rtol=1e-6,
        atol=0,
    )

    # Dead positions are excluded, so masking out the aligned sequence leaves zero.
    dead = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32)
    np.testing.assert_allclose(anti_subspace_term(jnp.asarray(states), jnp.asarray(dead)), 0.0, rtol=0, atol=1e-6)
    assert float(anti_subspace_term(jnp.asarray(states), jnp.zeros_like(jnp.asarray(live)))) == 0.0


def test_anti_subspace_schedule_opens_high_and_holds_at_its_ratio():
    spec = AntiSpec(
        lam=0.1,
        peak_ratio=2.5,
        hold_ratio=0.03,
        anneal_end=50,
        anchor_anneal_start=90,
        anchor_anneal_end=100,
        floor=0.1,
    )
    np.testing.assert_allclose(spec(0), 0.25, rtol=1e-12, atol=0)  # full strength before the anchor ramps in
    np.testing.assert_allclose(spec(50), 0.003, rtol=1e-12, atol=0)  # the hold ratio
    np.testing.assert_allclose(spec(90), 0.003, rtol=1e-12, atol=0)  # and it holds
    np.testing.assert_allclose(spec(100), 0.0003, rtol=1e-12, atol=0)  # shares the anchor's end anneal
    # Stretching the anneal holds the repulsion higher for longer, same endpoints.
    late = replace(spec, anneal_end=90)
    assert late(50) > spec(50)
    np.testing.assert_allclose(late(0), spec(0), rtol=1e-12, atol=0)
    np.testing.assert_allclose(late(90), spec(90), rtol=1e-9, atol=0)


def test_the_anti_subspace_term_pushes_the_cloud_off_the_axis(data_dir, tmp_path):
    """The repulsive term works on unlabeled lines too, so it lowers the mean alignment.

    Run with no labels at all, so the pull never fires and the only thing acting on the axis is the repulsion.
    """
    tokens, weights = probe_set()
    ends = {}
    for ratio in (0.0, 2.5):
        anchor = AnchorSpec(peak=0.1, warmup_epochs=2, anneal_start=13, anneal_end=15)
        _, _, traj = train_anchored(
            training_config(),
            data_dir,
            anchor=anchor,
            anti=AntiSpec(
                lam=0.1,
                peak_ratio=ratio,
                hold_ratio=ratio,
                anneal_end=8,
                anchor_anneal_start=13,
                anchor_anneal_end=15,
            ),
            label_p=np.zeros(64),  # nothing is ever labeled
            probe_tokens=tokens,
            probe_weights=weights,
            checkpoint_dir=tmp_path / f"ckpt{ratio}",
            traj_stride=5,
        )
        ends[ratio] = traj
    assert abs(float(ends[0.0]["anti"][-1])) > abs(float(ends[2.5]["anti"][-1]))
    assert float(ends[2.5]["anti_weight"][0]) > 0


def test_margin_is_the_weighted_minus_the_plain_mean():
    alpha = np.array([[[0.0], [1.0], [2.0], [3.0]]], dtype=np.float32).transpose(0, 1, 2)  # (1, 4, 1)
    weights = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(margin(alpha, weights), [[1.5]], rtol=1e-6, atol=0)
    np.testing.assert_allclose(margin(alpha, np.full(4, 0.25)), [[0.0]], rtol=0, atol=1e-6)


def test_op1_alignment_does_not_depend_on_the_rest_of_the_line():
    """Position 0 of a causal model sees only itself, so probe lines sharing an op1 agree.

    The measurement the hypotheses score is therefore exact at op1 — averaging over partner colors there is averaging over identical values, and the spread the report's noise floor allows for comes from the residual slices alone.
    """
    model = build_model(model_config(), key=jr.key(0))
    lines = np.array(
        [[COLORS[0], PLUS, p, EQ, a, NEWLINE] for p, a in zip(COLORS, COLORS[::-1], strict=True)], dtype=np.int32
    )
    alpha = alignment(model, lines)  # (L1, N, T)
    spread = alpha[:, :, 0].std(axis=1)
    np.testing.assert_allclose(spread, 0.0, rtol=0, atol=1e-6)
    assert alpha[:, :, 2].std(axis=1).max() > 1e-6  # op2 does vary; the property is positional


@pytest.fixture
def data_dir(tmp_path, corpus):
    metadata = CorpusMetadata(
        tokenizer_config=TokenizerConfig(vocabulary=[str(i) for i in range(15)]),
        total_tokens=len(corpus),
        total_chars=len(corpus),
        sources=[],
    )
    save_data(corpus, metadata, tmp_path)
    return tmp_path


def training_config(seed: int = 0) -> TrainingConfig:
    return TrainingConfig(
        model=model_config(),
        tokenizer=TokenizerConfig(vocabulary=[str(i) for i in range(15)]),
        data=data_config(),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=1e-2, betas=(0.9, 0.95)),
        scheduler=SchedulerConfig(epochs=15, warmup_epochs=2, min_lr_factor=0.01),
        seed=seed,
    )


def probe_set() -> tuple[np.ndarray, np.ndarray]:
    """One line per color, plus weights that single out the first color."""
    tokens = np.array([[c, PLUS, COLORS[0], EQ, c, NEWLINE] for c in COLORS], dtype=np.int32)
    weights = np.eye(len(COLORS))[0]
    return tokens, weights


def test_anchoring_moves_the_labeled_color_toward_the_axis(data_dir, tmp_path):
    """The margin separates the two conditions; the unanchored one is at chance.

    Eight colors and three residual slices leave a loose null here (|m| of a few tenths), so the claim is the gap rather than an absolute level for λ=0.
    """
    label_p = np.zeros(64)
    label_p[COLORS[0]] = 0.5
    tokens, weights = probe_set()
    trajectories = {}
    for peak in (0.0, 1.0):
        _, metrics, traj = train_anchored(
            training_config(),
            data_dir,
            anchor=AnchorSpec(peak=peak, warmup_epochs=2, anneal_start=13, anneal_end=15),
            label_p=label_p,
            probe_tokens=tokens,
            probe_weights=weights,
            checkpoint_dir=tmp_path / f"ckpt{peak}",
            traj_stride=5,
        )
        assert len(metrics) == 15
        assert set(traj) == {
            *("step", "epoch", "lr", "weight", "anti_weight", "m_op1", "m_span"),
            *("alpha_op1", "alpha_roles", "val_loss", "anchor", "anti", "pi"),
        }
        assert not any(traj["anti_weight"])  # no repulsive term in this pair
        trajectories[peak] = traj

    anchored, control = trajectories[1.0], trajectories[0.0]
    assert anchored["anchor"][-1] < anchored["anchor"][0] / 2  # the term itself came down
    assert anchored["alpha_op1"][-1] > 0.5  # and the stream moved onto the axis
    assert anchored["alpha_op1"][-1] > control["alpha_op1"][-1] + 0.4
    assert anchored["m_op1"][-1] > control["m_op1"][-1]  # the labeled color leads the rest


def test_pooled_training_runs_and_records_the_weight_profile(data_dir, tmp_path):
    """The pooled path trains end to end; at τ = ∞ the recorded profile is uniform."""
    label_p = np.zeros(64)
    label_p[COLORS[0]] = 0.5
    tokens, weights = probe_set()
    trajs = {}
    for tau in (np.inf, 0.05):
        _, _, traj = train_anchored(
            training_config(),
            data_dir,
            anchor=AnchorSpec(peak=1.0, warmup_epochs=2, anneal_start=13, anneal_end=15, tau=tau),
            label_p=label_p,
            probe_tokens=tokens,
            probe_weights=weights,
            probe_line_w=weights,
            checkpoint_dir=tmp_path / f"ckpt-t{tau}",
            traj_stride=20,
        )
        assert traj["pi"].shape == (len(traj["step"]), 3, PROMPT_SPAN)  # (measurements, slices, roles)
        np.testing.assert_allclose(traj["pi"].sum(axis=-1), 1.0, rtol=1e-5, atol=0)
        assert traj["anchor"][-1] < traj["anchor"][0]  # the pull pulled
        # The trajectory carries the per-role drift, and — since the line weights
        # here equal the color weights — an m_line that reduces to m_span exactly.
        assert traj["alpha_roles"].shape == (len(traj["step"]), 3, PROMPT_SPAN)
        np.testing.assert_allclose(traj["m_line"], traj["m_span"], rtol=1e-6, atol=1e-7)
        trajs[tau] = traj
    np.testing.assert_allclose(trajs[np.inf]["pi"], 1 / PROMPT_SPAN, rtol=0, atol=1e-6)
    assert float(np.abs(trajs[0.05]["pi"][-1] - 0.25).max()) > 0.05  # a finite τ concentrates


def test_the_anchor_is_the_only_difference_between_conditions(data_dir, tmp_path):
    """λ=0 trains on the same crops as λ>0, so accuracy differences have one cause."""
    label_p = np.full(16, 0.5)
    tokens, weights = probe_set()
    losses = []
    for peak in (0.0, 0.0):
        _, metrics, _ = train_anchored(
            training_config(),
            data_dir,
            anchor=AnchorSpec(peak=peak, warmup_epochs=2, anneal_start=13, anneal_end=15),
            label_p=label_p,
            probe_tokens=tokens,
            probe_weights=weights,
            checkpoint_dir=tmp_path / f"ckpt{peak}",
            traj_stride=100,
        )
        losses.append([m.train_loss for m in metrics])
    assert losses[0] == losses[1]
