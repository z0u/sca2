from io import StringIO

import jax.numpy as jnp
import numpy as np
import optax
from pydantic import NonNegativeInt

from sca.config import SchedulerConfig


def configure_schedule(config: SchedulerConfig, peak_lr: float, epoch_length: NonNegativeInt) -> optax.Schedule:
    """Linear warmup to the peak LR, then cosine anneal down to `min_lr_factor * peak`; or, when the config
    carries a dopesheet (`lr_sheet`), that sheet stretched over the run.

    The schedule is evaluated once per batch (see the training loop), so all durations are expressed in steps rather than epochs.
    """
    total_steps = config.epochs * epoch_length
    if config.lr_sheet is not None:
        return sheet_schedule(config.lr_sheet, peak_lr, total_steps)

    warmup_steps = int(config.warmup_epochs * epoch_length)
    min_lr = config.min_lr_factor * peak_lr

    return optax.warmup_cosine_decay_schedule(
        init_value=min_lr,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=min_lr,
    )


def realize_lr_sheet(csv: str, total_steps: int) -> np.ndarray:
    """The learning-rate multiple at each of the *total_steps* + 1 steps of a run, from a dopesheet's `lr` column.

    The sheet is realized over its own STEP range (so a sheet keyed 0 to 10000 has a resolution of 0.01% of the
    run), then stretched linearly onto the run. Keyframes, interpolation space, and timing functions follow
    `mini.temporal.Dopesheet`.
    """
    from mini.temporal import Dopesheet, Timeline, realize_timeline

    sheet = Dopesheet.from_csv(StringIO(csv))
    last = len(sheet) - 1  # the last keyframe; the realized table runs a step past it
    df = realize_timeline(Timeline(sheet))
    df = df[df["STEP"] <= last]
    frac = df["STEP"].to_numpy(np.float64) / last
    return np.interp(np.arange(total_steps + 1) / max(total_steps, 1), frac, df["lr"].to_numpy(np.float64))


def sheet_schedule(csv: str, peak_lr: float, total_steps: int) -> optax.Schedule:
    """An optax schedule that looks up the dopesheet *csv*, realized over *total_steps* and scaled by *peak_lr*."""
    table = jnp.asarray(realize_lr_sheet(csv, total_steps) * peak_lr, dtype=jnp.float32)

    def schedule(count):
        return table[jnp.clip(count, 0, total_steps)]

    return schedule
