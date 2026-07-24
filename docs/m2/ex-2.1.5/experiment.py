"""Experiment 2.1.5: disjoint vocabularies and more named colors.

Do two surface languages for the same domain converge on one internal
geometry? The corpus (`sca.data.mixed_vocab`) holds two sublanguages that are
never seen together — mixing equations over 140 real xkcd color names, and the
same arithmetic over 3-digit hex codes — with no alias lines and no mixed-form
equations. If the model places both vocabularies in the same latent geometry,
the cause is pressure internal to the network.

A star design around one center cell (d64-L4, 140 names, 216 hex operands,
no bridge), three seeds per cell: depth (L8), width (d32, d16, d16-L8),
density (hex-dense, palette-250), and a bridge arm that adds cross-form
equations. Per cell:

- **Exact match** per form on seen and held-out operand pairs, graded against
  two prompt-free references (the prompt-blind centroid and a guesser handed
  the answer's neighborhood; both computed at prep time, since neither reads
  the model).
- **Probe maps**: leave-one-out ridge R² for operand and mix values at every
  layer × grammar landmark, separately per form (`sca.compute.geometry`).
- **Cross-form alignment**: zero-shot transfer R² in both directions (the
  transfer ratio ρ), and principal angles between the two forms' probe
  subspaces — the H3-H5 measurements.

    bin/mini run docs/m2/ex-2.1.5/experiment.py --app modal --max-containers 5
    bin/mini status ex-2.1.5
"""

from __future__ import annotations

from mini import Ctx, Experiment, get_data_dir

SEEDS = [0, 1, 2]
PEAK_LR = 1e-2  # carried from the siblings; nGPT's normalization makes the LR forgiving
BLOCK_SIZE = 128  # named lines reach ~85 chars, so the siblings' 64 can't hold a line

CORPUS_SEED = 0
HEX_SUBSET_SEED = 0
N_EXAMPLES = 100_000
HOLDOUT_FRAC = 0.2  # of distinct unordered operand pairs, per form
N_EVAL = 256  # examples per eval set
N_PROBE = 768  # lines per single-form probe set
MAX_ANSWER = 28  # longest palette name (26) + newline + slack

# The star design. Every cell sees both forms; only the bridge arm adds cross
# lines. Attention is held at 8 heads × 8 dims; the residual stream and MLP
# scale with width.
ARMS: dict[str, dict] = {
    "center":      dict(names=140, hex=216,  bridge=False, width=64, depth=4),
    "L8":          dict(names=140, hex=216,  bridge=False, width=64, depth=8),
    "d32":         dict(names=140, hex=216,  bridge=False, width=32, depth=4),
    "d16":         dict(names=140, hex=216,  bridge=False, width=16, depth=4),
    "d16-L8":      dict(names=140, hex=216,  bridge=False, width=16, depth=8),
    "hex-dense":   dict(names=140, hex=2048, bridge=False, width=64, depth=4),
    "palette-250": dict(names=250, hex=216,  bridge=False, width=64, depth=4),
    "bridge":      dict(names=140, hex=216,  bridge=True,  width=64, depth=4),
}  # fmt: skip

METRICS_REF = "reports/m2/ex-2.1.5/metrics"
ARRAYS_REF = "reports/m2/ex-2.1.5/arrays"
EVALS_REF = "reports/m2/ex-2.1.5/evals"  # + f"/{corpus}"
CKPT_REF = "reports/m2/ex-2.1.5/checkpoints"  # + f"/{label}"


def corpus_key(arm: dict) -> str:
    """Cells that share (names, hex, bridge) share a corpus."""
    return f"n{arm['names']}-h{arm['hex']}" + ("-bridge" if arm["bridge"] else "")


def _hex_grid_lifted():
    """All 4096 hex-grid points in full-cube coordinates, as an (V, 3) array."""
    import numpy as np

    from sca.data.mixed_vocab import STEP

    g = np.arange(0, 256, STEP)
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)


def _reference_guessers(exs, candidates, train_answers, ks=(2, 5)) -> dict:
    """Model-free references for one eval set: prompt-blind centroid + k-NN.

    `candidates` is the (V, 3) answer vocabulary (palette or full hex grid);
    `train_answers` the (T, 3) snapped answer values seen in training. Exact
    match and distances are graded against the set's true answers, which are
    by construction the candidates nearest each exact mix.
    """
    import numpy as np

    cand = np.asarray(candidates, dtype=np.float32) / 255
    results = np.array([ex.result for ex in exs], dtype=np.float32) / 255
    dists = np.linalg.norm(cand[None] - results[:, None], axis=2)  # (N, V)
    answers = np.array([_answer_value(ex) for ex in exs], dtype=np.float32) / 255
    true_idx = np.linalg.norm(cand[None] - answers[:, None], axis=2).argmin(axis=1)

    train = np.asarray(train_answers, dtype=np.float32) / 255
    uniq, counts = np.unique(train, axis=0, return_counts=True)
    blind = int((np.linalg.norm(cand[:, None] - uniq[None], axis=2) * counts).sum(axis=1).argmin())

    order = np.argsort(dists, axis=1)
    knn = {}
    for k in ks:
        member = (order[:, :k] == true_idx[:, None]).any(axis=1)
        knn[f"k{k}"] = {
            "acc": float((member / k).mean()),
            "dist": float(np.take_along_axis(dists, order[:, :k], axis=1).mean()),
        }
    return {
        "blind": {"acc": float((true_idx == blind).mean()), "dist": float(dists[:, blind].mean())},
        "knn": knn,
        "floor_dist": float(np.linalg.norm(answers - results, axis=1).mean()),
        "chance_dist": float(dists.mean()),
    }


def _answer_value(ex) -> tuple[int, int, int]:
    """The snapped answer's full-cube value (the exact mix is `ex.result`)."""
    from sca.data import mixed_vocab as mv

    if ex.answer.startswith("#"):
        r, g, b = (int(c, 16) for c in ex.answer[1:])
        return mv.lift((r, g, b))
    return mv.xkcd_survey()[ex.answer]


def prepare_corpus(key: str, n_names: int, n_hex: int, bridge: bool) -> dict:
    """Generate one corpus, tokenize it onto the volume, and stash eval/probe sets."""
    import numpy as np

    from sca.compute.data_pipelines import save_data
    from sca.config import CorpusMetadata, DatasetMetadata, TokenizerConfig
    from sca.data import mixed_vocab as mv
    from sca.data.colors import dump_example_sets
    from sca.data.tokenizer import CharTokenizer
    from mini.store import put

    palette = mv.xkcd_palette(n_names)
    hex_ops = mv.hex_operands(n_hex, HEX_SUBSET_SEED)
    named_train, named_held = mv.holdout_split(mv.distinct_pairs(palette.values()), CORPUS_SEED, HOLDOUT_FRAC)
    hex_train, hex_held = mv.holdout_split(mv.distinct_pairs(hex_ops), CORPUS_SEED, HOLDOUT_FRAC)
    held_n, held_h = set(named_held), set(hex_held)
    weights = mv.BRIDGE_WEIGHTS if bridge else mv.FORM_WEIGHTS
    corpus = mv.sample_corpus(
        N_EXAMPLES, CORPUS_SEED, palette, hex_ops,
        lambda a, b: (min(a, b), max(a, b)) in held_n,
        lambda a, b: (min(a, b), max(a, b)) in held_h,
        weights,
    )  # fmt: skip

    text = "".join(ex.text for ex in corpus)
    tokenizer_config = TokenizerConfig(vocabulary=mv.alphabet())  # fixed a priori
    tokens = np.asarray(CharTokenizer(tokenizer_config).encode([text])[0], dtype=np.int32)
    meta = CorpusMetadata(
        tokenizer_config=tokenizer_config,
        total_tokens=len(tokens),
        total_chars=len(text),
        sources=[DatasetMetadata(title=f"disjoint-vocabulary corpus ({key})", fixes=[], total_chars=len(text))],
    )
    save_data(tokens, meta, get_data_dir() / "corpora" / key)

    # Split the corpus lines by form, for stats and the "seen" eval pairs.
    by_form: dict[str, list] = {"named": [], "hex": [], "cross": []}
    for ex in corpus:
        n_hash = ex.prompt.count("#")
        by_form["cross" if n_hash == 1 else "hex" if n_hash else "named"].append(ex)

    rng = np.random.default_rng(CORPUS_SEED + 1)
    pick = lambda pairs, n: [pairs[i] for i in rng.permutation(len(pairs))[:n]]  # noqa: E731
    seen = {f: sorted({p for ex in exs if (p := ex.pair)}) for f, exs in by_form.items()}
    sets = {
        "named_seen": mv.as_form(pick(seen["named"], N_EVAL), "named", palette, CORPUS_SEED + 2),
        "named_holdout": mv.as_form(pick(named_held, N_EVAL), "named", palette, CORPUS_SEED + 3),
        "hex_seen": mv.as_form(pick(seen["hex"], N_EVAL), "hex", palette, CORPUS_SEED + 4),
        "hex_holdout": mv.as_form(pick(hex_held, N_EVAL), "hex", palette, CORPUS_SEED + 5),
        # Probe suites: training-distribution lines, one form each.
        "named_probe": mv.as_form(pick(seen["named"], N_PROBE), "named", palette, CORPUS_SEED + 6),
        "hex_probe": mv.as_form(pick(seen["hex"], N_PROBE), "hex", palette, CORPUS_SEED + 7),
    }

    # Model-free references per eval set: neither guesser reads the prompt, so
    # they belong to the corpus, not the cell.
    pal_rgb = np.array(list(palette.values()))
    grid = _hex_grid_lifted()
    answers = {f: [_answer_value(ex) for ex in exs] for f, exs in by_form.items() if exs}
    nulls = {
        name: _reference_guessers(exs, pal_rgb if name.startswith("named") else grid, answers[name.split("_")[0]])
        for name, exs in sets.items()
        if not name.endswith("probe")
    }

    lengths = {f: [len(ex.text) for ex in exs] for f, exs in by_form.items() if exs}
    counts = np.zeros(len(palette), dtype=int)
    name_idx = {n: i for i, n in enumerate(palette)}
    for ex in by_form["named"]:
        counts[name_idx[ex.answer]] += 1
    p = counts[counts > 0] / counts.sum()
    stats = {
        "n_lines": {f: len(exs) for f, exs in by_form.items()},
        "n_seen_distinct": {f: len(ps) for f, ps in seen.items()},
        "n_holdout": {"named": len(named_held), "hex": len(hex_held)},
        "line_length": {f: {"mean": float(np.mean(ls)), "max": int(np.max(ls))} for f, ls in lengths.items()},
        "total_tokens": int(len(tokens)),
        "answer_counts": counts.tolist(),
        "answer_perplexity": float(np.exp(-(p * np.log(p)).sum())),
        "answers_reachable": int((counts > 0).sum()),
        "palette": {n: list(v) for n, v in palette.items()},
        "hex_ops": [mv.to_hex3(c) for c in hex_ops],
        "nulls": nulls,
    }
    return {
        "key": key,
        "meta": meta,
        "stats": stats,
        "evals": put(dump_example_sets(sets), name=f"ex-2.1.5-{key}.json"),
    }


def _make_config(width: int, depth: int, vocab_size: int, seed: int):
    """One cell's training config; schedule and data pipeline fixed across the sweep."""
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
            block_size=BLOCK_SIZE,
            n_embd=width,
            n_head=8,
            n_head_dim=8,
            n_ff=4 * width,
            n_layer=depth,
        ),
        tokenizer=TokenizerConfig(vocabulary=[]),
        data=DataConfig(batch_size=64, oversample=16, train_split=0.9, padding_chance=0.1),
        optimizer=OptimizerConfig(weight_decay=0, learning_rate=PEAK_LR, betas=(0.9, 0.95)),
        scheduler=SchedulerConfig(epochs=100, warmup_epochs=10, min_lr_factor=0.01),
        seed=seed,
    )


def build_sweep(preps: list[dict]) -> list[tuple]:
    """(config, corpus_key, arm, label) cells; cheap and deterministic per wake."""
    from sca.utils import align

    tc = {p["key"]: p["meta"].tokenizer_config for p in preps}
    cells = []
    for arm_name, arm in ARMS.items():
        key = corpus_key(arm)
        for seed in SEEDS:
            config = _make_config(arm["width"], arm["depth"], align(tc[key].vocab_size, 64), seed)
            config.tokenizer = tc[key].model_copy()
            cells.append((config, key, arm_name, f"{arm_name}-s{seed}"))
    return cells


def train_one(config, key: str, label: str) -> dict:
    """Train one cell on its corpus; checkpoint into the store."""
    import equinox as eqx
    import jax

    from sca.compute.training import train_model
    from mini.store import put

    ckpt_dir = get_data_dir() / "cells" / label
    model, metrics = train_model(config, get_data_dir() / "corpora" / key, checkpoint_dir=ckpt_dir)
    return {
        "label": label,
        "n_params": sum(int(x.size) for x in jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))),
        "val_loss": [m.val_loss for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "checkpoint": put(ckpt_dir / "model", name=f"ex-2.1.5-{label}-ckpt"),
    }


def eval_one(trained: dict, evals, n_names: int, arm: str, label: str) -> dict:
    """Exact match with miss geometry, probe maps, and cross-form alignment for one cell."""
    import io

    import numpy as np

    from sca.compute.evaluation import greedy_completions
    from sca.compute.geometry import (
        collect_activations,
        principal_angle_maps,
        probe_maps,
        rho,
        transfer_maps,
    )
    from sca.compute.model import load_checkpoint
    from sca.data import mixed_vocab as mv
    from sca.data.colors import load_example_sets
    from sca.data.tokenizer import CharTokenizer
    from mini.store import get, put

    workdir = get_data_dir() / "eval" / label
    get(trained["checkpoint"], workdir / "model")
    model, config, _ = load_checkpoint(workdir)
    tokenizer = CharTokenizer(config.tokenizer)

    palette = mv.xkcd_palette(n_names)
    pal_rgb = np.array(list(palette.values()), dtype=np.float32)
    grid = _hex_grid_lifted().astype(np.float32)
    all_sets = load_example_sets(get(evals, workdir / "evals.json").read_bytes())
    probe_sets = {f: all_sets.pop(f"{f}_probe") for f in ("named", "hex")}

    # --- Behavior: greedy completion per eval set, with miss geometry.
    sets, arrays = {}, {}
    for set_name, exs in all_sets.items():
        got = greedy_completions(model, tokenizer, [ex.prompt for ex in exs], MAX_ANSWER)
        named = set_name.startswith("named")
        cand = (pal_rgb if named else grid) / 255

        def guess_value(g: str, named: bool = named) -> tuple[int, int, int]:
            if not _well_formed(g, named, palette):
                return (0, 0, 0)
            if named:
                return palette[g]
            r, gr, b = (int(c, 16) for c in g[1:])
            return mv.lift((r, gr, b))

        values = np.array([guess_value(g) for g in got], dtype=np.float32) / 255
        well = np.array([_well_formed(g, named, palette) for g in got])
        results = np.array([ex.result for ex in exs], dtype=np.float32) / 255
        dists = np.linalg.norm(cand[None] - results[:, None], axis=2)  # (N, V)
        guess_dist = np.linalg.norm(values - results, axis=1)
        floor = np.linalg.norm(np.array([_answer_value(ex) for ex in exs], dtype=np.float32) / 255 - results, axis=1)
        # Rank of the guess among the candidate vocabulary, by distance to the true mix.
        rank = (dists < guess_dist[:, None] - 1e-9).sum(axis=1)
        hits = np.array([g == ex.answer for g, ex in zip(got, exs, strict=True)])
        sets[set_name] = {
            "n": len(exs),
            "accuracy": float(hits.mean()),
            "p_malformed": float(1 - well.mean()),
            "guess_dist": float(guess_dist[well].mean()) if well.any() else None,
            "floor_dist": float(floor.mean()),
            "nearest_rate": float((well & (guess_dist <= floor + 1e-9)).mean()),
            "failures": [(ex.prompt, ex.answer, g) for g, ex, h in zip(got, exs, hits, strict=True) if not h][:8],
        }
        arrays[f"evals/{set_name}/rank"] = np.where(well, rank, -1).astype(np.int32)
        arrays[f"evals/{set_name}/guess_dist"] = np.where(well, guess_dist, np.nan).astype(np.float32)
        arrays[f"evals/{set_name}/guess_value"] = np.where(well[:, None], values * 255, -1).astype(np.int16)
        arrays[f"evals/{set_name}/true_value"] = np.array([_answer_value(ex) for ex in exs], dtype=np.int16)
        arrays[f"evals/{set_name}/mix_value"] = np.array([ex.result for ex in exs], dtype=np.int16)

    # --- Geometry: probe maps per form, cross-form transfer, subspace angles.
    fitted, acts_lm, targets = {}, {}, {}
    for form, exs in probe_sets.items():
        form_acts, form_lm = collect_activations(model, tokenizer, exs)
        targets[form] = {
            "op1": np.array([ex.lhs for ex in exs], dtype=np.float32) / 255,
            "op2": np.array([ex.rhs for ex in exs], dtype=np.float32) / 255,
            "mix": np.array([ex.result for ex in exs], dtype=np.float32) / 255,
        }
        fitted[form] = probe_maps(form_acts, form_lm, targets[form])
        acts_lm[form] = (form_acts, form_lm)
        for t, maps in fitted[form].items():
            arrays[f"probes/{form}/{t}/r2"] = maps["r2"].astype(np.float32)
            arrays[f"probes/{form}/{t}/r2_ch"] = maps["r2_ch"].astype(np.float32)
            arrays[f"probes/{form}/{t}/weights"] = maps["weights"].astype(np.float32)

    acts_n, lm_n = acts_lm["named"]
    acts_h, lm_h = acts_lm["hex"]
    cross = {
        "hex2name": transfer_maps(fitted["hex"], acts_n, lm_n, targets["named"]),
        "name2hex": transfer_maps(fitted["named"], acts_h, lm_h, targets["hex"]),
    }
    rho_maps = {
        "hex2name": {t: rho(cross["hex2name"][t], fitted["named"][t]["r2"]) for t in targets["named"]},
        "name2hex": {t: rho(cross["name2hex"][t], fitted["hex"][t]["r2"]) for t in targets["hex"]},
    }
    angles = {
        t: principal_angle_maps(fitted["named"][t]["weights"], fitted["hex"][t]["weights"]) for t in targets["named"]
    }
    for d, ts in cross.items():
        for t, m in ts.items():
            arrays[f"cross/{d}/{t}/r2"] = m.astype(np.float32)
            arrays[f"cross/{d}/{t}/rho"] = rho_maps[d][t].astype(np.float32)
    for t, m in angles.items():
        arrays[f"angles/{t}"] = m.astype(np.float32)

    # Site summaries for the mix target. The primary site is chosen by
    # within-form strength alone (independent of ρ, per the methodology note);
    # the max-ρ site is reported beside it as an upper bound on sharing.
    w_named, w_hex = fitted["named"]["mix"]["r2"], fitted["hex"]["mix"]["r2"]
    primary = np.unravel_index(np.minimum(w_named, w_hex).argmax(), w_named.shape)
    summary = {
        "primary_site": _site(primary),
        "within_at_primary": {"named": float(w_named[primary]), "hex": float(w_hex[primary])},
        "rho_at_primary": {d: _float(rho_maps[d]["mix"][primary]) for d in rho_maps},
        "angles_at_primary": [float(a) for a in angles["mix"][primary]],
        "within_best": {"named": _site_of_max(w_named), "hex": _site_of_max(w_hex)},
        "rho_best": {d: _site_of_max(rho_maps[d]["mix"]) for d in rho_maps},
    }

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return {
        "label": label,
        "arm": arm,
        "n_params": trained["n_params"],
        "val_loss": trained["val_loss"],
        "train_loss": trained["train_loss"],
        "sets": sets,
        "alignment": summary,
        "arrays": put(buf.getvalue(), name=f"ex-2.1.5-{label}-arrays.npz"),
    }


def _well_formed(guess: str, named: bool, palette) -> bool:
    if named:
        return guess in palette
    return len(guess) == 4 and guess[0] == "#" and all(c in "0123456789abcdef" for c in guess[1:])


def _float(v) -> float | None:
    import math

    f = float(v)
    return None if math.isnan(f) else f


def _site(idx) -> dict:
    from sca.data.mixed_vocab import LANDMARKS

    return {"depth": int(idx[0]), "landmark": LANDMARKS[int(idx[1])]}


def _site_of_max(m) -> dict | None:
    import numpy as np

    if np.isnan(m).all():
        return None
    idx = np.unravel_index(np.nanargmax(m), m.shape)
    return _site(idx) | {"value": float(m[idx])}


def publish_results(results: list[dict], stats: dict, checkpoints: dict, evals: dict) -> dict:
    """Publish metrics (JSON), stacked per-cell arrays (npz), eval sets, and checkpoints."""
    import io
    import json

    import numpy as np

    from mini.store import get, put, set_ref

    cells = [{k: v for k, v in r.items() if k != "arrays"} for r in results]
    metrics = {"cells": cells, "corpus_stats": stats, "arms": {a: dict(v) for a, v in ARMS.items()}}
    set_ref(METRICS_REF, put(json.dumps(metrics, indent=2).encode(), name="ex-2.1.5-metrics.json"))
    for key, art in evals.items():
        set_ref(f"{EVALS_REF}/{key}", art)
    for label, ckpt in checkpoints.items():
        set_ref(f"{CKPT_REF}/{label}", ckpt)

    arrays = {}
    for r in results:
        path = get(r["arrays"], get_data_dir() / "publish" / f"{r['label']}.npz")
        with np.load(path) as z:
            arrays |= {f"{r['label']}/{name}": z[name] for name in z.files}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    set_ref(ARRAYS_REF, put(buf.getvalue(), name="ex-2.1.5-arrays.npz"))
    return {"n_cells": len(results)}


def main(ctx: Ctx) -> dict:
    # eval_one imports the geometry helpers locally, so the evidence fingerprint can't
    # reach LANDMARKS through them. Tag the eval map with the landmark scheme, so probes
    # (which index activations by landmark) re-run when it changes. See todo-eng.
    import hashlib

    from sca.data.mixed_vocab import LANDMARKS

    lm_tag = "lm-" + hashlib.sha1(repr(LANDMARKS).encode()).hexdigest()[:8]

    corpora = sorted({corpus_key(arm) for arm in ARMS.values()})
    specs = {corpus_key(arm): arm for arm in ARMS.values()}
    preps = ctx.map(
        prepare_corpus,
        corpora,
        [specs[k]["names"] for k in corpora],
        [specs[k]["hex"] for k in corpora],
        [specs[k]["bridge"] for k in corpora],
        role="prep",
    )
    prep_by_key = {p["key"]: p for p in preps}

    configs, keys, arms, labels = zip(*build_sweep(preps), strict=True)
    trained = ctx.map(train_one, configs, keys, labels, role="train")
    evaled = ctx.map(
        eval_one,
        trained,
        [prep_by_key[k]["evals"] for k in keys],
        [ARMS[a]["names"] for a in arms],
        arms,
        labels,
        role="eval",
        version=lm_tag,
    )
    stats = {p["key"]: p["stats"] for p in preps}
    ckpts = dict(zip(labels, [t["checkpoint"] for t in trained], strict=True))
    evals_by_key = {p["key"]: p["evals"] for p in preps}
    summary = ctx.run(publish_results, evaled, stats, ckpts, evals_by_key, role="prep")

    def by_arm(set_name: str, key: str) -> dict:
        out = {}
        for arm in ARMS:
            vals = [r["sets"][set_name][key] for r in evaled if r["arm"] == arm]
            out[arm] = round(sum(vals) / len(vals), 3)
        return out

    rho_center = [r["alignment"]["rho_at_primary"] for r in evaled if r["arm"] in ("center", "bridge")]
    return {
        **summary,
        "named_holdout_acc": by_arm("named_holdout", "accuracy"),
        "hex_holdout_acc": by_arm("hex_holdout", "accuracy"),
        "rho_at_primary_center_and_bridge": rho_center,
    }


experiment = Experiment(
    name="ex-2.1.5",
    main=main,
    roles={
        # Corpus generation (100k lines, ~3M chars) + model-free references.
        "prep": dict(cpu=2, timeout=1200),
        # ~8.5k steps of 64×128 tokens; L8 cells roughly double the L4 time.
        "train": dict(gpu="L4", timeout=5400, watchdog=300, watchdog_grace=1500),
        # Greedy decode (4 × 256 prompts) + LOO probe maps (2 forms × 3 targets).
        "eval": dict(gpu="L4", timeout=1800),
    },
)
