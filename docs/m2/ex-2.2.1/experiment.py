"""
Experiment 2.2.1: suppress *red* on the D2.1 checkpoints.

The first intervention on an anchored transformer. No training: every run is a
published checkpoint of ex-2.1.10 — the nine-seed primary with *red* anchored
on e₁, and its three-seed un-anchored control — scored through the eval
contract (`sca.intervention`) on the 5,832-line probe set the D2.1 experiments
share. The operator is axis projection: remove the e₁ component at every slice
and position, re-project onto the sphere.

One task per checkpoint: it scores the clean pass and every intervention the
run takes part in, and returns per-run statistics plus an npz of per-line
arrays. A publish step stacks the twelve into one npz and one metrics JSON
under the refs below, which the report reads. The design constants sit above
the DAG so the report can import them and the prose and the gates cannot
drift apart.

    bin/mini run docs/m2/ex-2.2.1/experiment.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mini import Ctx, Experiment, get_data_dir

#: Result refs the report will read.
METRICS_REF = "reports/m2/ex-2.2.1/metrics"
ARRAYS_REF = "reports/m2/ex-2.2.1/arrays"


@dataclass(frozen=True)
class Condition:
    """A set of published runs, and where to find them."""

    exp: str
    name: str
    seeds: int
    title: str

    @property
    def key(self) -> str:
        return f"{self.exp}/{self.name}"


PRIMARY = Condition("ex-2.1.10", "either-t100", 9, "anchored")
"""The D2.1 recipe: pooled either-operand labels, anchor + anti-subspace at the ex-2.1.8 operating point."""

CONTROL = Condition("ex-2.1.10", "lam0", 3, "un-anchored")
"""Same corpus, seeds, and code path as the primary, with the anchor weight at zero."""

# --- The lines -----------------------------------------------------------------

#: Token positions of a probe line, in sequence order.
POSITIONS = ("op1", "+", "op2", "=", "ans", "\n")

#: The answer token's position; the response is read from the logits one position earlier.
ANSWER_POS = 4

#: Positions whose state can reach the answer: the prompt. Writes at the answer and newline positions
#: are measured for the bound but cannot move the response.
PROMPT_POSITIONS = (0, 1, 2, 3)

#: The operand positions alone — the `operands` arm's mask.
OPERAND_POSITIONS = (0, 2)

#: Slices of the residual stream: the embedding, then the state after each of the four blocks.
SLICES = (0, 1, 2, 3, 4)

#: A line's dose: the larger of its two operands' redness (M1's label affinity, r·(1 − g/2 − b/2)).
#: Either operand can carry the concept, as either could draw the label in ex-2.1.10.
RED_DOSE = 0.8
"""*Red lines* have dose ≥ 0.8: an operand among the five reddest colors of the grid (365 lines)."""

NONRED_DOSE = 0.2
"""*Non-red lines* have dose ≤ 0.2: neither operand carries more than a fifth of full redness (1,689 lines)."""

DOSE_EDGES = (0.2, 0.4, 0.6, 0.8)
"""Bin edges for the grading read: five bins, the first being the non-red lines and the last the red lines."""

# --- The interventions ---------------------------------------------------------

#: The primary intervention, and the one every hypothesis scores: full-strength axis projection at
#: every slice and every position.
PRIMARY_GAMMA = 1.0

#: Arms on the primary condition, reported at the same statistics without gates. Each changes one
#: thing from the primary intervention.
ARMS = {
    "operands": "positions restricted to op1 and op2",
    "embedding": "slice 0 only",
    "last": "slice 4 only",
    "lobe": "the M1 lobe in place of the plain projection",
    "ablate": "the weights that read and write e₁ zeroed, then re-normalized",
}
# REVIEW: added the `ablate` row. The report's arm table listed five arms and H5 scores `ablate`,
# so the module was one arm short of the frozen prose. Verify: `ARMS` should have a row for every
# row of the arm table in the report's Method, and H5 needs this one.

LOBE = dict(a=0.5, b=1.0, p=1.0)
"""The lobe arm: no suppression below α = 0.5, ramping linearly to full removal at α = 1. The threshold sits
above the constant component the syntax embeddings carry (α ≈ 0.3–0.4 in the published map) and below the
red operand's (α ≈ 0.9), so it is the operator that leaves the bulk alone by design."""

GAMMAS = (0.25, 0.5, 0.75, 1.0)
"""The strength sweep (E1): the fraction of the e₁ component removed."""

#: The post-hoc tier, fitted per control run and applied to that run: a direction per slice, from the
#: op1-position states of one line per color, applied at that slice at every position. The `axis` row
#: is e₁ itself on the control — the operator with nothing on the axis.
POSTHOC = ("axis", "diff-in-means", "probe", "leace")

#: The diff-in-means split: colors at or above RED_DOSE against the rest.
#: The probe and LEACE targets: the color's redness.

# --- Gates ---------------------------------------------------------------------

RED_ACC_GATE = 0.5
"""H1: "seed-mean exact-match accuracy on red lines under the primary intervention falls to 0.5 or below"."""

RED_ACC_PARTIAL = 0.8
"""H1 partial: "between 0.5 and 0.8"."""

TASK_GATE = 0.02
"""H2 and H5: "seed-mean accuracy on non-red lines stays within 0.02 of the clean accuracy". The width
every D2.1 task gate used, kept for continuity; the probe set is over six times the size of the holdout
set those gates were read on, so the binomial share of the noise is smaller here."""

TASK_PARTIAL = 0.05
"""H2 and H5 partial: "within 0.05"."""

GRADE_DIP = 0.02
"""H3: "seed-mean damage is non-decreasing across the five dose bins, allowing a dip between adjacent
bins of at most 0.02"."""

GRADE_SPAN = 0.5
"""H3: "and the top bin's damage exceeds the bottom bin's by at least 0.5"."""

WRITE_QUANTILE = 99
"""H4: the quantile, over non-red lines, that the write bound is read at on both sides."""

WRITE_TOLERANCE = 1.25
"""H4: "at every slice and position, the seed-mean 99th-percentile write on non-red lines is at most
1.25 times arcsin of the seed-mean 99th-percentile clean alignment at that site"."""

RESOLUTION_SD = 2.0
"""Differences between arms or tiers smaller than this many pooled between-seed standard deviations of
the statistic are reported as not resolved."""

#: The seeds that decide agreement in the undesigned-response read: a red line "agrees" when at least
#: this many of the nine primary seeds decode the same answer.
AGREE_SEEDS = 5

#: Ridge strength of the decoding probes (E6), matching ex-2.1.12 and the post-hoc probe direction.
DECODE_L2 = 1e-2

#: The probe fitted per run for the decoded write: the RGB of each operand, read at every site.
DECODE_TARGETS = ("op1", "op2")

#: Groups the per-line statistics are contracted over, in the order the arrays carry them.
GROUPS = ("all", "red", "nonred")

#: What a red line decodes to under intervention, in the order the composition is stored.
COMPOSITION = ("true", "red_operand", "visible_operand", "neighbor", "other")

#: Tolerance on the recomputed clean alignment map against the one ex-2.1.10 published, per line.
#: D2.1's re-key check used the same figure for the same array (a GPU pass against a CPU pass).
ALPHA_ATOL = 2e-3

# --- Interventions as data -------------------------------------------------------


@dataclass(frozen=True)
class Intervention:
    """One row of the results: an operator, where it acts, and which gate or arm it serves."""

    name: str
    kind: str
    """`projection`, `lobe`, `ablate`, or `posthoc`."""
    slices: tuple[int, ...] = SLICES
    positions: tuple[int, ...] | None = None
    gamma: float = PRIMARY_GAMMA
    fit: str | None = None
    """For `posthoc`: which direction to fit, from `POSTHOC`."""


PRIMARY_INTERVENTION = Intervention("primary", "projection")

ARM_INTERVENTIONS = (
    Intervention("operands", "projection", positions=OPERAND_POSITIONS),
    Intervention("embedding", "projection", slices=(0,)),
    Intervention("last", "projection", slices=(SLICES[-1],)),
    Intervention("lobe", "lobe"),
    Intervention("ablate", "ablate", slices=()),
)
assert tuple(a.name for a in ARM_INTERVENTIONS) == tuple(ARMS), "one arm per row of the arm table"

STRENGTH_INTERVENTIONS = tuple(Intervention(f"gamma-{g}", "projection", gamma=g) for g in GAMMAS if g != PRIMARY_GAMMA)

POSTHOC_INTERVENTIONS = tuple(
    Intervention(f"{fit}@{s}", "posthoc", slices=(s,), fit=fit) for fit in POSTHOC for s in SLICES
)


def interventions_for(cond: Condition) -> tuple[Intervention, ...]:
    """The primary run takes the arms and the strength sweep; the control takes the post-hoc tier.

    Both take the primary intervention: on the control it is the calibration row (the operator with
    nothing placed on the axis) that H1 and H2 read beside the anchored result.
    """
    if cond == PRIMARY:
        return (PRIMARY_INTERVENTION, *ARM_INTERVENTIONS, *STRENGTH_INTERVENTIONS)
    return (PRIMARY_INTERVENTION, *POSTHOC_INTERVENTIONS)


# --- The DAG ---------------------------------------------------------------------


def top_quantile(a, q: float = WRITE_QUANTILE, axis: int = -1):
    """The *q*-th percentile as an order statistic: the ⌈n(1 − q/100)⌉-th largest value along *axis*.

    H4 reads "the 17th-largest of 1,689 values within a run", so no interpolation: a monotone map of the
    values (arcsin of an alignment) then commutes with the quantile, which is what makes the embedding
    row of the bound an identity the pipeline can assert.
    """
    a = np.asarray(a)
    n = a.shape[axis]
    k = int(np.ceil(n * (1 - q / 100)))
    return np.take(np.sort(a, axis=axis), n - k, axis=axis)


def ridge_readout(x, y, l2: float = DECODE_L2):
    """A ridge probe fitted on *x* → *y* as a closure that decodes new states, centered as the fit was."""
    x64, y64 = np.asarray(x, np.float64), np.asarray(y, np.float64)
    mx, my = x64.mean(0), y64.mean(0)
    xc = x64 - mx
    w = np.linalg.solve(xc.T @ xc + l2 * np.eye(x64.shape[1]), xc.T @ (y64 - my))
    return lambda h: (np.asarray(h, np.float64) - mx) @ w + my


@dataclass(frozen=True)
class Lines:
    """The probe set as the statistics see it: groups, dose bins, and the color of every token."""

    tokens: np.ndarray
    """(N, T) token ids."""
    red: np.ndarray
    nonred: np.ndarray
    bins: np.ndarray
    """Per line, the dose bin: 0 is the non-red group, the last is the red group."""
    color_ids: np.ndarray
    """Token id of each palette color, in palette order."""
    tok2color: np.ndarray
    """Token id → palette index, −1 off the color vocabulary."""
    red_operand: np.ndarray
    """Per line, the position of the operand carrying the dose (op1 on a tie)."""
    n_partners: int

    @property
    def n(self) -> int:
        return len(self.tokens)

    @property
    def n_pos(self) -> int:
        return self.tokens.shape[1]

    @property
    def groups(self) -> dict[str, np.ndarray]:
        return {"all": np.ones(self.n, bool), "red": self.red, "nonred": self.nonred}

    @property
    def line_colors(self) -> np.ndarray:
        """(N, T) palette index per token; the syntax positions read −1."""
        return self.tok2color[self.tokens]

    @property
    def answer(self) -> np.ndarray:
        return self.tokens[:, ANSWER_POS]

    def by_group(self, v: np.ndarray) -> dict[str, float]:
        return {g: float(np.nanmean(v[m])) for g, m in self.groups.items()}


def read_lines(tokens: np.ndarray, r1: np.ndarray, r2: np.ndarray, color_redness: np.ndarray, tokenizer) -> Lines:
    """Group the probe lines by dose, and check the set is the one the design describes."""
    from sca.colorcube import redness as colorcube_redness
    from sca.data.named_colors import GRIDS, grid_palette
    from sca.vis_grading import GRID_RGB

    n_colors = len(color_redness)
    dose = np.maximum(r1, r2)
    eps = 1e-9
    red, nonred = dose >= RED_DOSE - eps, dose <= NONRED_DOSE + eps
    bins = np.digitize(dose, [e + eps for e in DOSE_EDGES[:-1]] + [DOSE_EDGES[-1] - eps])
    np.testing.assert_array_equal(bins == 0, nonred)
    np.testing.assert_array_equal(bins == len(DOSE_EDGES), red)
    color_ids = np.array([tokenizer.stoi[n] for n in grid_palette(GRIDS["v216"])])
    np.testing.assert_allclose(colorcube_redness(GRID_RGB), color_redness, rtol=0, atol=1e-12)
    tok2color = np.full(tokenizer.vocab_size, -1)
    tok2color[color_ids] = np.arange(n_colors)
    colors = tok2color[tokens]
    assert (colors[:, [0, 2, ANSWER_POS]] >= 0).all() and (colors[:, [1, 3, 5]] < 0).all()
    # The op1 column walks the palette in order, one color per partner group: what the post-hoc fits rely on.
    n_partners = len(tokens) // n_colors
    np.testing.assert_array_equal(colors[:, 0], np.repeat(np.arange(n_colors), n_partners))
    return Lines(tokens, red, nonred, bins, color_ids, tok2color, np.where(r1 >= r2, 0, 2), n_partners)


def composition(lines: Lines, guess: np.ndarray) -> np.ndarray:
    """Per line, what the decoded answer is, as an index into `COMPOSITION`.

    A one-step neighbor is a color adjacent to the true mix in the 6³ grid: one level away in at least
    one channel and no more than one in any.
    """
    g = lines.tok2color[guess]
    rows = np.arange(lines.n)
    cube = np.stack(np.unravel_index(np.arange(len(lines.color_ids)), (6, 6, 6)), axis=1)
    step = np.abs(cube[np.maximum(g, 0)] - cube[lines.line_colors[:, ANSWER_POS]]).max(1)
    tests = [
        guess == lines.answer,
        guess == lines.tokens[rows, lines.red_operand],
        guess == lines.tokens[rows, 2 - lines.red_operand],
        (g >= 0) & (step == 1),
    ]
    return np.select(tests, [0, 1, 2, 3], default=4)


def guess_distance(lines: Lines, guess: np.ndarray) -> np.ndarray:
    """Unit-cube distance from the decoded answer to the true mix; NaN off the color vocabulary."""
    from sca.vis_grading import GRID_RGB

    g = lines.tok2color[guess]
    d = np.linalg.norm(GRID_RGB[np.maximum(g, 0)] - GRID_RGB[lines.line_colors[:, ANSWER_POS]], axis=1)
    return np.where(g >= 0, d, np.nan)


def fit_decoders(clean_pre: np.ndarray, lines: Lines) -> dict[str, list[list]]:
    """One ridge readout per (target, slice, position), fitted on the clean states at that site (E6)."""
    from sca.vis_grading import GRID_RGB

    op_rgb = {t: GRID_RGB[lines.line_colors[:, p]] for t, p in zip(DECODE_TARGETS, (0, 2), strict=True)}
    return {
        t: [[ridge_readout(clean_pre[s, :, p], op_rgb[t]) for p in range(lines.n_pos)] for s in SLICES]
        for t in DECODE_TARGETS
    }


def decode_groups(decoders: dict[str, list[list]], states: np.ndarray, lines: Lines) -> np.ndarray:
    """Group-mean decoded RGB per site: (L1, T, target, group, channel)."""
    out = np.zeros((len(SLICES), lines.n_pos, len(DECODE_TARGETS), len(GROUPS), 3), np.float32)
    for ti, t in enumerate(DECODE_TARGETS):
        for s in SLICES:
            for p in range(lines.n_pos):
                z = decoders[t][s][p](states[s, :, p])
                out[s, p, ti] = [z[m].mean(0) for m in lines.groups.values()]
    return out


def fitted_direction(fit: str, x: np.ndarray, color_redness: np.ndarray):
    """The post-hoc tier's direction from one run's clean op1 states at a slice, one state per color."""
    from sca.intervention import Subspace, diff_in_means, leace, probe_direction

    assert x.shape == (len(color_redness), x.shape[1])
    match fit:
        case "axis":
            return Subspace.axis(x.shape[1])
        case "diff-in-means":
            return diff_in_means(x, color_redness >= RED_DOSE - 1e-9)
        case "probe":
            return probe_direction(x, color_redness, l2=DECODE_L2)
        case "leace":
            return leace(x, color_redness)
    raise ValueError(fit)


def build_triple(iv: Intervention, model, clean_pre: np.ndarray, lines: Lines, color_redness: np.ndarray):
    """The (model, subspace, operator) triple for one intervention, from the run's clean stream."""
    from sca.anchoring import ANCHOR_AXIS
    from sca.intervention import Subspace, ablate_weights, lobe, projection

    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    match iv.kind:
        case "projection":
            return model, sub, projection(sub, gamma=iv.gamma)
        case "lobe":
            return model, sub, lobe(sub, a=LOBE["a"], b=LOBE["b"], p=LOBE["p"])
        case "ablate":
            return ablate_weights(model, sub), sub, projection(sub)
        case "posthoc":
            assert iv.fit is not None and len(iv.slices) == 1
            edited = fitted_direction(iv.fit, clean_pre[iv.slices[0], :: lines.n_partners, 0], color_redness)
            return model, edited, projection(edited)
    raise ValueError(iv.kind)


def check_contract(iv: Intervention, out, alpha_clean: np.ndarray, theta: np.ndarray, positions) -> None:
    """The contract's promises, checked on every run.

    The embedding sees the clean stream whatever the operator, unless the weights themselves were edited;
    nothing moves at the sites an intervention skips; a projection's write is the closed form of the
    alignment it arrived with; and an ablated model never carries the axis.
    """
    from sca.anchoring import ANCHOR_AXIS
    from sca.intervention import write_angle

    alpha_pre = out.pre[..., ANCHOR_AXIS]
    if iv.kind != "ablate":
        np.testing.assert_allclose(alpha_pre[0], alpha_clean[0], rtol=0, atol=0)
    skipped = np.ones((len(SLICES), theta.shape[-1]), bool)
    for s in iv.slices:
        skipped[s] = False if positions is None else positions == 0
    off = skipped.nonzero()
    np.testing.assert_allclose(theta[off[0], :, off[1]], 0.0, rtol=0, atol=1e-6)
    if iv.kind == "projection":
        on = (~skipped).nonzero()
        expected = write_angle(alpha_pre[on[0], :, on[1]], iv.gamma)
        np.testing.assert_allclose(theta[on[0], :, on[1]], expected, rtol=0, atol=2e-3)
    if iv.kind == "ablate":
        assert np.abs(alpha_pre).max() < 1e-6, "an ablated model never carries the axis"


def score_run(exp: str, cond: str, seed: int, interventions: tuple[Intervention, ...]) -> dict:
    """Score one checkpoint: the clean pass, then each intervention through the eval contract.

    Every quantity the report reads comes out of here, from one code path per run: the response and
    its accuracy per group, damage per dose bin, the write and the arriving alignment at every site,
    the final-state displacement, the decoded write, and what red lines decode to. Per-run statistics
    return as the result; per-line arrays go to the store as one npz.
    """
    from mini.store import get_many, get_ref, put
    from sca.anchoring import ANCHOR_AXIS
    from sca.compute.model import load_checkpoint
    from sca.data.named_colors import WordTokenizer
    from sca.intervention import Subspace, angle_between, answer_logprobs, apply, projection
    from sca.model import NGPT

    label = f"{cond}-s{seed}"
    workdir = get_data_dir() / "score" / f"{exp}-{label}"
    probes_ref = get_ref(f"reports/m2/{exp}/probes")
    ckpt_ref = get_ref(f"reports/m2/{exp}/checkpoints/{label}")
    assert probes_ref is not None and ckpt_ref is not None, f"{exp} has not published {label} and its probe set"
    probe_path, _ = get_many([(probes_ref, workdir / "probes.npz"), (ckpt_ref, workdir / "model")])
    with np.load(probe_path) as z:
        tokens, r1, r2, color_redness = z["probe_tokens"], z["r1"], z["r2"], z["redness"]
    model, config, _ = load_checkpoint(workdir)
    assert isinstance(model, NGPT), "the contract's operators act on nGPT's between-block stream"
    lines = read_lines(tokens, r1, r2, color_redness, WordTokenizer(config.tokenizer))
    rows = np.arange(lines.n)

    # --- The clean pass: the scorer's own control, and the reference every intervention is read against.
    sub = Subspace.axis(model.transformer.wte.shape[1], ANCHOR_AXIS)
    clean = apply(model, tokens, projection(sub), slices=())
    alpha_clean = clean.pre[..., ANCHOR_AXIS]  # (L1, N, T)
    lp_clean = answer_logprobs(clean.logits, ANSWER_POS)
    p_clean = np.exp(lp_clean[rows, lines.answer])
    guess_clean = lp_clean.argmax(1)
    alpha_q99 = top_quantile(np.abs(alpha_clean[:, lines.nonred]), axis=1)  # (L1, T)
    bound = np.arcsin(alpha_q99)
    decoders = fit_decoders(clean.pre, lines)

    arrays: dict[str, Any] = {
        "clean/alpha": alpha_clean.astype(np.float32),  # full precision: checked against the published map
        "clean/p_ans": p_clean.astype(np.float32),
        "clean/guess": guess_clean.astype(np.int16),
        "clean/decode": decode_groups(decoders, clean.pre, lines),
    }
    stats: dict[str, Any] = {
        "acc": lines.by_group(guess_clean == lines.answer),
        "p_ans": lines.by_group(p_clean),
        "bound": bound.tolist(),
        "alpha_q99_nonred": alpha_q99.tolist(),
    }
    scored: dict[str, Any] = {}

    # --- Each intervention: build the triple, score it, contract the per-line quantities.
    for iv in interventions:
        positions = None if iv.positions is None else np.isin(np.arange(lines.n_pos), iv.positions).astype(np.float32)
        target, edited, op = build_triple(iv, model, clean.pre, lines, color_redness)
        out = apply(target, tokens, op, slices=iv.slices, positions=positions)

        lp = answer_logprobs(out.logits, ANSWER_POS)
        p_ans = np.exp(lp[rows, lines.answer])
        guess = lp.argmax(1)
        damage = p_clean - p_ans
        offvocab = 1.0 - np.exp(lp[:, lines.color_ids]).sum(1)
        theta = angle_between(out.pre, out.post)  # (L1, N, T): the write at every site
        alpha_pre = out.pre[..., ANCHOR_AXIS]  # the alignment each operator saw arrive
        disp = angle_between(out.post[-1, :, ANSWER_POS - 1], clean.post[-1, :, ANSWER_POS - 1])
        comp = composition(lines, guess)
        check_contract(iv, out, alpha_clean, theta, positions)
        q99_write = top_quantile(theta[:, lines.nonred], axis=1)  # (L1, T)
        if iv.name == PRIMARY_INTERVENTION.name:  # H4's embedding identity, to float32 roundoff
            np.testing.assert_allclose(q99_write[0], bound[0], rtol=0, atol=1e-3)

        scored[iv.name] = {
            "kind": iv.kind,
            "slices": list(iv.slices),
            "positions": None if iv.positions is None else list(iv.positions),
            "gamma": iv.gamma,
            "fit": iv.fit,
            "acc": lines.by_group(guess == lines.answer),
            "p_ans": lines.by_group(p_ans),
            "damage": lines.by_group(damage),
            "damage_bins": [float(damage[lines.bins == b].mean()) for b in range(len(DOSE_EDGES) + 1)],
            "offvocab": lines.by_group(offvocab),
            "disp": lines.by_group(disp),
            "write_prompt_sum": lines.by_group(theta[:, :, list(PROMPT_POSITIONS)].sum(axis=(0, 2))),
            "q99_write_nonred": q99_write.tolist(),
            "q99_alpha_nonred": top_quantile(np.abs(alpha_pre[:, lines.nonred]), axis=1).tolist(),
            "composition_red": np.bincount(comp[lines.red], minlength=len(COMPOSITION)).tolist(),
            "guess_dist_red": float(np.nanmean(guess_distance(lines, guess)[lines.red])),
            "cos_to_axis": float(abs(edited.basis[0] @ sub.basis[0])),
        }
        # The post-hoc tier acts at one slice, so only that slice's write is kept; the arriving alignment
        # is kept at every slice for every row, since it is the map H4 and its cloud are drawn from.
        arrays |= {
            f"{iv.name}/p_ans": p_ans.astype(np.float32),
            f"{iv.name}/guess": guess.astype(np.int16),
            f"{iv.name}/offvocab": offvocab.astype(np.float32),
            f"{iv.name}/disp": disp.astype(np.float32),
            f"{iv.name}/composition": comp.astype(np.int8),
            f"{iv.name}/decode": decode_groups(decoders, out.post, lines) - decode_groups(decoders, out.pre, lines),
            f"{iv.name}/alpha_pre": alpha_pre.astype(np.float16),
            f"{iv.name}/theta": (theta if iv.kind != "posthoc" else theta[iv.slices[0]]).astype(np.float16),
        }

    return {
        "label": label,
        "condition": cond,
        "seed": seed,
        "n": {g: int(m.sum()) for g, m in lines.groups.items()},
        "bin_sizes": np.bincount(lines.bins, minlength=len(DOSE_EDGES) + 1).tolist(),
        "clean": stats,
        "interventions": scored,
        "arrays": put(_npz(**arrays), name=f"ex-2.2.1-{label}-arrays.npz"),
    }


def _npz(**arrays) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def publish_results(results: list[dict]) -> dict:
    """Stack every run's arrays into one npz and its statistics into one JSON, under the refs above.

    Two things happen here that need every run at once: the recomputed clean alignment maps are checked
    against the ones ex-2.1.10 published, and the seed-agreement fraction of the undesigned response is
    read across the primary seeds.
    """
    import io
    import json

    from mini.store import get, get_ref, put, set_ref

    workdir = get_data_dir() / "publish"
    arrays: dict[str, Any] = {}
    for r in results:
        with np.load(get(r["arrays"], workdir / f"{r['label']}.npz")) as z:
            arrays |= {f"{r['label']}/{k}": z[k] for k in z.files}

    published_ref = get_ref(f"reports/m2/{PRIMARY.exp}/arrays")
    assert published_ref is not None
    alpha_diff = {}
    with np.load(get(published_ref, workdir / "published.npz")) as z:
        for r in results:
            ours = arrays[f"{r['label']}/clean/alpha"].astype(np.float32)
            alpha_diff[r["label"]] = float(np.abs(ours - z[f"{r['label']}/alpha_lines"]).max())
    assert max(alpha_diff.values()) < ALPHA_ATOL, alpha_diff

    probes_ref = get_ref(f"reports/m2/{PRIMARY.exp}/probes")
    assert probes_ref is not None
    with np.load(get(probes_ref, workdir / "probes.npz")) as z:
        tokens, r1, r2 = z["probe_tokens"], z["r1"], z["r2"]
    dose = np.maximum(r1, r2)
    arrays |= {"probe/tokens": tokens, "probe/dose": dose, "probe/r1": r1, "probe/r2": r2}
    red = dose >= RED_DOSE - 1e-9

    primary_runs = [r for r in results if r["condition"] == PRIMARY.name]
    agreement = {}
    for iv in interventions_for(PRIMARY):
        guesses = np.stack([arrays[f"{r['label']}/{iv.name}/guess"][red] for r in primary_runs])  # (seeds, red)
        modal = np.array([np.bincount(col).max() for col in guesses.T])
        agreement[iv.name] = float((modal >= AGREE_SEEDS).mean())

    metrics = {
        "runs": [{k: v for k, v in r.items() if k != "arrays"} for r in results],
        "agreement_red": agreement,
        "alpha_max_diff": alpha_diff,
        "design": {
            "primary": PRIMARY.__dict__,
            "control": CONTROL.__dict__,
            "positions": POSITIONS,
            "slices": SLICES,
            "arms": ARMS,
            "lobe": LOBE,
            "gammas": GAMMAS,
            "posthoc": POSTHOC,
            "groups": GROUPS,
            "composition": COMPOSITION,
            "decode_targets": DECODE_TARGETS,
            "dose": {"red": RED_DOSE, "nonred": NONRED_DOSE, "edges": DOSE_EDGES},
            "gates": {
                "red_acc": RED_ACC_GATE,
                "red_acc_partial": RED_ACC_PARTIAL,
                "task": TASK_GATE,
                "task_partial": TASK_PARTIAL,
                "grade_dip": GRADE_DIP,
                "grade_span": GRADE_SPAN,
                "write_quantile": WRITE_QUANTILE,
                "write_tolerance": WRITE_TOLERANCE,
                "resolution_sd": RESOLUTION_SD,
                "agree_seeds": AGREE_SEEDS,
            },
        },
    }
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.2.1-metrics.json"))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="ex-2.2.1-arrays.npz"))
    return {"n_runs": len(results), "n_arrays": len(arrays), "agreement_red": agreement}


def main(ctx: Ctx) -> dict:
    runs = [(c, s) for c in (PRIMARY, CONTROL) for s in range(c.seeds)]
    scored = ctx.map(
        score_run,
        [c.exp for c, _ in runs],
        [c.name for c, _ in runs],
        [s for _, s in runs],
        [interventions_for(c) for c, _ in runs],
        role="cpu",
    )
    summary = ctx.run(publish_results, scored, role="cpu")

    def seed_mean(cond: Condition, stat: str, group: str) -> float:
        values = [r["interventions"]["primary"][stat][group] for r in scored if r["condition"] == cond.name]
        return round(sum(values) / len(values), 3)

    return {
        **summary,
        "acc_red": {c.title: seed_mean(c, "acc", "red") for c in (PRIMARY, CONTROL)},
        "acc_nonred": {c.title: seed_mean(c, "acc", "nonred") for c in (PRIMARY, CONTROL)},
    }


experiment = Experiment(
    name="ex-2.2.1",
    main=main,
    # One checkpoint per task: a dozen forward passes over 5,832 six-token lines through a 4-layer,
    # 64-wide model, plus 60 ridge fits on 5,832 × 64 for the decoders. CPU work; the control runs
    # score 21 interventions each, and are the largest cells.
    roles={"cpu": dict(cpu=4, timeout=3600, watchdog_grace=1800)},
    deps=[PRIMARY.exp],
)
