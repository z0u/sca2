"""Concept anchoring: the pull lands on the right positions, and it moves the stream."""

from dataclasses import replace

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from sca.anchoring import (
    LINE_TOKENS,
    PROMPT_SPAN,
    AnchorSpec,
    AntiSpec,
    alignment,
    anchor_term,
    anti_subspace_term,
    margin,
    sample_anchored_batches,
    smoothstep,
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


def test_anchor_term_is_zero_when_aligned_and_two_when_opposed():
    states = np.zeros((3, 2, 4, 8), dtype=np.float32)
    states[..., 0] = 1.0
    mask = np.ones((2, 4), dtype=np.float32)
    np.testing.assert_allclose(anchor_term(jnp.asarray(states), jnp.asarray(mask)), 0.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(anchor_term(jnp.asarray(-states), jnp.asarray(mask)), 2.0, rtol=1e-6, atol=0)
    # An empty mask contributes nothing rather than 0/0.
    assert float(anchor_term(jnp.asarray(states), jnp.zeros_like(jnp.asarray(mask)))) == 0.0


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

    Run with no labels at all, so the pull never fires and the only thing acting
    on the axis is the repulsion.
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

    The measurement the hypotheses score is therefore exact at op1 — averaging
    over partner colors there is averaging over identical values, and the spread
    the report's noise floor allows for comes from the residual slices alone.
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

    Eight colors and three residual slices leave a loose null here (|m| of a few
    tenths), so the claim is the gap rather than an absolute level for λ=0.
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
            *("step", "epoch", "lr", "weight", "anti_weight", "m_op1"),
            *("alpha_op1", "val_loss", "anchor", "anti"),
        }
        assert not any(traj["anti_weight"])  # no repulsive term in this pair
        trajectories[peak] = traj

    anchored, control = trajectories[1.0], trajectories[0.0]
    assert anchored["anchor"][-1] < anchored["anchor"][0] / 2  # the term itself came down
    assert anchored["alpha_op1"][-1] > 0.5  # and the stream moved onto the axis
    assert anchored["alpha_op1"][-1] > control["alpha_op1"][-1] + 0.4
    assert anchored["m_op1"][-1] > control["m_op1"][-1]  # the labeled color leads the rest


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
