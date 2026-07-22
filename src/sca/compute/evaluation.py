"""Task evaluation for the color-mixing experiments.

Two measurements, both against the domain's exact ground truth:

- **Completion accuracy**: greedy-decode the answer after ``... = `` and
  exact-match it. Greedy is the right decode here because the language gives
  every prompt a single correct completion.
- **Residual-stream probes**: closed-form ridge regression from the residual
  stream to the ground-truth colors, fit per layer at two positions — the last
  character of the first operand (is the operand's value represented?) and the
  space after ``=`` (is the *result* computed before the answer is emitted?).
  Probes are fit on one half of the probe set and scored (R²) on the other.
  The fitted directions are returned too: comparing them across seeds is the
  baseline for "SCA puts the concept where we choose" — without anchoring they
  should land somewhere different every run.
"""

from typing import Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from sca.data.colors import N_LEVELS, Example, redness
from sca.data.tokenizer import CharTokenizer
from sca.model import LanguageModel


def greedy_completions(
    model: LanguageModel,
    tokenizer: CharTokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    batch_size: int = 256,
) -> list[str]:
    """Greedy-decode each prompt up to (excluding) the first newline."""
    model = eqx.nn.inference_mode(model)
    forward = eqx.filter_jit(model.__call__)
    newline = tokenizer.stoi["\n"]

    out: list[str] = []
    for start in range(0, len(prompts), batch_size):
        seq = np.asarray(tokenizer.encode(list(prompts[start : start + batch_size])))
        _, P = seq.shape
        for _ in range(max_new_tokens):
            # Fixed (B, block_size) window so the jitted forward compiles once:
            # right-pad the (≤ block_size) context and read the last real position.
            window = seq[:, -model.block_size :]
            t = window.shape[1]
            if t < model.block_size:
                window = np.pad(window, ((0, 0), (0, model.block_size - t)))
            logits = forward(jnp.asarray(window))
            nxt = np.asarray(jnp.argmax(logits[:, t - 1], axis=-1))
            seq = np.concatenate([seq, nxt[:, None]], axis=1)
            if bool((seq[:, P:] == newline).any(axis=1).all()):
                break
        out += [text.split("\n")[0] for text in tokenizer.decode(seq[:, P:].tolist())]
    return out


def completion_accuracy(
    model: LanguageModel,
    tokenizer: CharTokenizer,
    examples: Sequence[Example],
    max_new_tokens: int = 12,
) -> dict:
    """Exact-match accuracy over *examples*, with a few failures for inspection."""
    got = greedy_completions(model, tokenizer, [ex.prompt for ex in examples], max_new_tokens)
    hits = [g == ex.answer for g, ex in zip(got, examples, strict=True)]
    failures = [(ex.prompt, ex.answer, g) for g, ex, hit in zip(got, examples, hits, strict=True) if not hit]
    return {"accuracy": float(np.mean(hits)), "n": len(examples), "failures": failures[:8]}


def _r2_cols(y_pred: Float[np.ndarray, "N K"], y_true: Float[np.ndarray, "N K"]) -> Float[np.ndarray, " K"]:
    """Coefficient of determination, per target column."""
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-12)


def _r2(y_pred: Float[np.ndarray, "N K"], y_true: Float[np.ndarray, "N K"]) -> float:
    """Coefficient of determination, averaged over target columns."""
    return float(_r2_cols(y_pred, y_true).mean())


def ridge_probe(
    x: Float[np.ndarray, "N C"],
    y: Float[np.ndarray, "N K"],
    x_test: Float[np.ndarray, "M C"],
    y_test: Float[np.ndarray, "M K"],
    l2: float = 1e-2,
) -> tuple[Float[np.ndarray, "C K"], Float[np.ndarray, " K"], float]:
    """Closed-form ridge regression; returns (weights, bias, test R²)."""
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    w = np.linalg.solve(xc.T @ xc + l2 * np.eye(x.shape[1]), xc.T @ yc)
    b = my - mx @ w
    return w, b, _r2(x_test @ w + b, y_test)


def ridge_probe_loo(
    x: Float[np.ndarray, "N C"],
    y: Float[np.ndarray, "N K"],
    l2: float = 1e-2,
) -> Float[np.ndarray, "N K"]:
    """Leave-one-out predictions: row ``i`` comes from a probe fit on every other row.

    The estimator to reach for when N is small, as it is for a color vocabulary.
    A k-fold split both trains on fewer rows and makes the answer depend on
    which split was drawn — at N = 27 that choice moves R² by a few hundredths —
    whereas this has no split to choose and so no seed.

    Exact, and roughly the cost of one fit rather than N: each fold's centered
    Gram matrices come from a rank-1 downdate of the full ones, since dropping
    row ``i`` subtracts its outer product and shifts the mean by a known amount.
    """
    x64, y64 = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n, c = x64.shape
    sum_x, sum_y = x64.sum(0), y64.sum(0)
    gram_xx, gram_xy = x64.T @ x64, x64.T @ y64
    penalty = l2 * np.eye(c)
    pred = np.empty_like(y64)
    for i in range(n):
        mx, my = (sum_x - x64[i]) / (n - 1), (sum_y - y64[i]) / (n - 1)
        xx = gram_xx - np.outer(x64[i], x64[i]) - (n - 1) * np.outer(mx, mx)
        xy = gram_xy - np.outer(x64[i], y64[i]) - (n - 1) * np.outer(mx, my)
        pred[i] = (x64[i] - mx) @ np.linalg.solve(xx + penalty, xy) + my
    return pred.astype(np.asarray(y).dtype)


def _forced_stats(
    model: LanguageModel,
    tokenizer: CharTokenizer,
    texts: Sequence[str],
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Teacher-forced per-position surprisal and entropy over left-padded *texts*.

    Returns ``(seq, nll, entropy)``: column ``j`` of the (N, T−1) stat arrays
    describes the prediction of sequence column ``j + 1``.
    """
    model = eqx.nn.inference_mode(model)
    forward = eqx.filter_jit(model.__call__)
    seq = np.asarray(tokenizer.encode(list(texts)))  # (N, T), left-padded together
    nll, ent = [], []
    for i in range(0, len(seq), batch_size):
        s = jnp.asarray(seq[i : i + batch_size])
        logp = jax.nn.log_softmax(forward(s), axis=-1)[:, :-1]
        nll.append(np.asarray(-jnp.take_along_axis(logp, s[:, 1:, None], axis=2)[..., 0]))
        ent.append(np.asarray(-(jnp.exp(logp) * logp).sum(axis=-1)))
    return seq, np.concatenate(nll), np.concatenate(ent)


def _answer_mask(stats_cols: int, n_answer: np.ndarray) -> np.ndarray:
    """Mask over stat columns selecting each row's last *n_answer* characters."""
    cols = np.arange(stats_cols)
    return cols[None, :] >= stats_cols - n_answer[:, None]


def candidate_logprobs(
    model: LanguageModel,
    tokenizer: CharTokenizer,
    prompts: Sequence[str],
    candidates: Sequence[str],
    batch_size: int = 256,
) -> Float[np.ndarray, "N K"]:
    """Teacher-forced log-probability of each candidate answer after each prompt.

    Scores the *complete* answer (candidate plus terminating newline), so
    candidates of different lengths are comparable. The margin between the true
    answer and the best competitor is the compute-vs-lookup measure from the
    ex-2.1.1 garden-path diagnosis.
    """
    texts = [p + c + "\n" for p in prompts for c in candidates]
    n_ans = np.array([len(c) + 1 for _ in prompts for c in candidates])
    _, nll, _ = _forced_stats(model, tokenizer, texts, batch_size)
    mask = _answer_mask(nll.shape[1], n_ans)
    return -(nll * mask).sum(axis=1).reshape(len(prompts), len(candidates))


def answer_calibration(
    model: LanguageModel,
    tokenizer: CharTokenizer,
    examples: Sequence[Example],
    batch_size: int = 256,
) -> dict:
    """Mean surprisal, entropy, and s₂ = (i − h) / log |V| over answer characters.

    s₂ is a graded companion to exact-match accuracy: ≈ 0 when the model knows
    its own uncertainty, ≫ 0 when it is confidently wrong. It measures
    calibration, not competence (a uniformly ignorant model also scores ≈ 0),
    so read it alongside accuracy or raw surprisal.
    """
    texts = [ex.prompt + ex.answer for ex in examples]
    n_ans = np.array([len(ex.answer) for ex in examples])
    _, nll, ent = _forced_stats(model, tokenizer, texts, batch_size)
    mask = _answer_mask(nll.shape[1], n_ans)
    log_v = float(np.log(sum(1 for c in tokenizer.vocabulary if c)))  # excludes the pad token
    i, h = (float((a * mask).sum() / mask.sum()) for a in (nll, ent))
    return {"nll": i, "entropy": h, "s2": (i - h) / log_v}


def probe_answer_schedule(
    model: LanguageModel,
    tokenizer: CharTokenizer,
    examples: Sequence[Example],
    offsets: Sequence[int] = tuple(range(-4, 4)),
    batch_size: int = 256,
) -> dict:
    """Per-channel decodability of the result at each position around the answer.

    Teacher-forces ``prompt + answer`` (every example must share both lengths —
    use a single-form set, e.g. hex) and fits a ridge probe per (offset from
    the answer start, residual depth, RGB channel). Digit ``k`` of a hex answer
    sits at offset ``k + 1`` (offset 0 is ``#``) and is emitted *from* offset
    ``k``, so decodability of channel ``k`` at offsets ≤ ``k`` is computation;
    from ``k + 1`` on, the digit is in the context and decoding it is trivial.

    Returns ``offsets`` and ``r2`` with shape (len(offsets), depth + 1, 3).
    """
    assert len({(len(ex.prompt), len(ex.answer)) for ex in examples}) == 1, "use a single-form example set"
    start = len(examples[0].prompt)
    seq = np.asarray(tokenizer.encode([ex.prompt + ex.answer for ex in examples]))
    stream = eqx.filter_jit(eqx.nn.inference_mode(model).residual_stream)
    acts = np.concatenate(
        [np.asarray(stream(jnp.asarray(seq[i : i + batch_size]))) for i in range(0, len(seq), batch_size)],
        axis=1,
    )  # (L+1, N, T, C)
    y = np.array([ex.result for ex in examples], dtype=np.float32) / (N_LEVELS - 1)
    half = len(examples) // 2
    r2 = np.empty((len(offsets), acts.shape[0], y.shape[1]))
    for i, o in enumerate(offsets):
        x = acts[:, :, start + o]
        for d in range(acts.shape[0]):
            w, b, _ = ridge_probe(x[d, :half], y[:half], x[d, half:], y[half:])
            r2[i, d] = _r2_cols(x[d, half:] @ w + b, y[half:])
    return {"offsets": list(offsets), "r2": r2}


def probe_transfer(
    model: LanguageModel,
    tokenizer: CharTokenizer,
    fit: Sequence[Example],
    eval_sets: Mapping[str, Sequence[Example]],
    batch_size: int = 256,
) -> dict[str, list[float]]:
    """Result-color probes fit on one prompt set, scored on others.

    Fits per-layer ridge probes from the pre-answer position to the result RGB
    on half of *fit*, then reports R² on the held-back half (key ``"fit"``, the
    same-distribution ceiling) and on each eval set. Transfer at the same
    position separates "the mix was never computed" from "the probe just
    doesn't carry across prompt sets".
    """
    stream = eqx.filter_jit(eqx.nn.inference_mode(model).residual_stream)

    def at_end(exs: Sequence[Example]) -> tuple[np.ndarray, np.ndarray]:
        seq = np.asarray(tokenizer.encode([ex.prompt for ex in exs]))  # left-padded: prompt ends last
        acts = np.concatenate(
            [np.asarray(stream(jnp.asarray(seq[i : i + batch_size]))) for i in range(0, len(seq), batch_size)],
            axis=1,
        )
        return acts[:, :, -1], np.array([ex.result for ex in exs], dtype=np.float32) / (N_LEVELS - 1)

    x_fit, y_fit = at_end(fit)
    half = len(fit) // 2
    fitted = [ridge_probe(x_fit[d, :half], y_fit[:half], x_fit[d, half:], y_fit[half:]) for d in range(len(x_fit))]
    out = {"fit": [r2 for *_, r2 in fitted]}
    for name, exs in eval_sets.items():
        x, y = at_end(exs)
        out[name] = [_r2(x[d] @ w + b, y) for d, (w, b, _) in enumerate(fitted)]
    return out


def probe_residual_stream(
    model: LanguageModel,
    tokenizer: CharTokenizer,
    examples: Sequence[Example],
    batch_size: int = 256,
) -> dict:
    """Per-layer linear decodability of operand and result colors.

    Returns R² lists indexed by residual-stream depth (0 = embedding, L =
    final) and the fitted probe weights, keyed by probe name.
    """
    prompts = [ex.prompt for ex in examples]
    seq = np.asarray(tokenizer.encode(prompts))  # (N, P), left-padded to a common length
    offset = seq.shape[1] - np.array([len(p) for p in prompts])
    pos_operand = offset + np.array([len(p.split(" ")[0]) for p in prompts]) - 1
    # Every prompt ends with the space after '=', so with left-padding the
    # pre-answer position is simply the last column.

    stream = eqx.filter_jit(eqx.nn.inference_mode(model).residual_stream)
    acts = np.concatenate(
        [np.asarray(stream(jnp.asarray(seq[i : i + batch_size]))) for i in range(0, len(seq), batch_size)],
        axis=1,
    )  # (L+1, N, P, C)
    at_operand = acts[:, np.arange(len(seq)), pos_operand]  # (L+1, N, C)
    at_result = acts[:, :, -1]

    rgb = lambda cs: np.array(cs, dtype=np.float32) / (N_LEVELS - 1)  # noqa: E731
    targets = {
        "operand_rgb": (at_operand, rgb([ex.lhs for ex in examples])),
        "operand_redness": (at_operand, np.array([[redness(ex.lhs)] for ex in examples], dtype=np.float32)),
        "result_rgb": (at_result, rgb([ex.result for ex in examples])),
        "result_redness": (at_result, np.array([[redness(ex.result)] for ex in examples], dtype=np.float32)),
    }

    half = len(examples) // 2
    r2s: dict[str, list[float]] = {}
    weights: dict[str, np.ndarray] = {}
    for name, (x, y) in targets.items():
        fitted = [ridge_probe(x[d, :half], y[:half], x[d, half:], y[half:]) for d in range(len(x))]
        r2s[name] = [r2 for _, _, r2 in fitted]
        weights[name] = np.stack([w for w, _, _ in fitted])  # (L+1, C, K)
    return {"r2": r2s, "weights": weights}
