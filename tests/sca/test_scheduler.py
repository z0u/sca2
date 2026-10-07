import numpy as np
import pytest

from sca.config import SchedulerConfig
from sca.training.scheduler import configure_schedule

SHEET = "STEP,PHASE,ACTION,lr\n0,Warmup,,0.01\n100,Hold,,1\n800,Anneal,,0.5\n1000,,,0.01\n"


def test_lr_sheet_left_out_of_dump_when_unset():
    config = SchedulerConfig(epochs=4, warmup_epochs=1, min_lr_factor=0.01)
    assert "lr_sheet" not in config.model_dump(mode="json")
    assert "lr_sheet" in config.model_copy(update={"lr_sheet": SHEET}).model_dump(mode="json")


def test_lr_sheet_stretches_over_the_run():
    config = SchedulerConfig(epochs=4, warmup_epochs=1, min_lr_factor=0.01, lr_sheet=SHEET)
    peak, epoch_length = 0.002, 250  # 1000 steps, so sheet steps map one to one
    schedule = configure_schedule(config, peak, epoch_length)
    lr = np.array([float(np.asarray(schedule(t))) for t in (0, 100, 450, 800, 1000, 5000)])
    assert lr == pytest.approx(peak * np.array([0.01, 1.0, 0.75, 0.5, 0.01, 0.01]), rel=1e-4)

    # Minimum jerk eases symmetrically, so halfway between keyframes is halfway between their values.
    assert float(np.asarray(schedule(50))) == pytest.approx(peak * 0.505, rel=1e-4)

    # Twice the run, same shape: the keyframes land at twice the step.
    longer = configure_schedule(config, peak, 2 * epoch_length)
    assert float(np.asarray(longer(200))) == pytest.approx(peak, rel=1e-3)
    assert float(np.asarray(longer(1600))) == pytest.approx(0.5 * peak, rel=1e-3)


def test_lincos_sheet_matches_warmup_cosine():
    """A two-keyframe-per-cycle `lincos` sheet is the built-in warmup-then-cosine schedule, step for step."""
    sheet = "STEP,PHASE,ACTION,lr::lincos\n0,Warmup,,0.01\n250,Anneal,,1\n1000,,,0.01\n"
    peak, epoch_length = 0.002, 250
    from_sheet = configure_schedule(
        SchedulerConfig(epochs=4, warmup_epochs=1, min_lr_factor=0.01, lr_sheet=sheet), peak, epoch_length
    )
    built_in = configure_schedule(SchedulerConfig(epochs=4, warmup_epochs=1, min_lr_factor=0.01), peak, epoch_length)
    steps = np.arange(0, 1001)
    assert np.asarray(from_sheet(steps)) == pytest.approx(np.asarray(built_in(steps)), rel=1e-5, abs=1e-9)


def test_lincos_restarts_from_a_lower_floor():
    """A second cycle warms up again from where the first annealed to, and anneals along its own cosine."""
    sheet = "STEP,PHASE,ACTION,lr::lincos\n0,,,0.01\n100,,,2\n400,,,0.01\n500,,,1\n1000,,,0.01\n"
    schedule = configure_schedule(
        SchedulerConfig(epochs=4, warmup_epochs=1, min_lr_factor=0.01, lr_sheet=sheet), 1.0, 250
    )
    lr = np.asarray(schedule(np.array([50, 250, 400, 450, 750])))
    assert lr == pytest.approx([1.005, 1.005, 0.01, 0.505, 0.505], rel=1e-5)


def test_unknown_interpolator_is_refused():
    sheet = "STEP,PHASE,ACTION,lr::lincosine\n0,,,0.01\n1000,,,1\n"
    with pytest.raises(ValueError, match="unknown interpolator"):
        configure_schedule(SchedulerConfig(epochs=4, warmup_epochs=1, min_lr_factor=0.01, lr_sheet=sheet), 1.0, 250)
