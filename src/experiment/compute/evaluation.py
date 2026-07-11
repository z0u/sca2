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

from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from experiment.data.colors import N_LEVELS, Example, redness
from experiment.data.tokenizer import CharTokenizer
from experiment.model import LanguageModel


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


def _r2(y_pred: Float[np.ndarray, "N K"], y_true: Float[np.ndarray, "N K"]) -> float:
    """Coefficient of determination, averaged over target columns."""
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12)))


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
