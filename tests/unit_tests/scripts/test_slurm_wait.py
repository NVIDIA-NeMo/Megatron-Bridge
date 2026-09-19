import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest


pytestmark = pytest.mark.unit


@pytest.fixture
def waiter(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "scripts/common/slurm_wait.py"
    spec = importlib.util.spec_from_file_location("test_slurm_wait_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "sleep", Mock())
    runner = MagicMock()
    runner.__enter__.return_value = runner
    factory = Mock(return_value=runner)
    backend = ModuleType("nemo_run.run.torchx_backend.runner")
    backend.get_runner = factory
    api = ModuleType("torchx.specs.api")
    api.AppState = SimpleNamespace(SUCCEEDED="SUCCEEDED")
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.setitem(sys.modules, api.__name__, api)
    job = SimpleNamespace(launched=True, handle="slurm://nemo_run/123", state="SUBMITTED", cleanup=Mock())
    experiment = SimpleNamespace(jobs=[job])
    return module, runner, factory, experiment


@pytest.mark.parametrize("interval", [60, 120])
def test_wait_uses_requested_interval_and_preserves_terminal_state(waiter, interval):
    module, runner, factory, experiment = waiter
    runner.wait.return_value = SimpleNamespace(state="SUCCEEDED")

    module.wait_for_slurm_job(experiment, poll_interval=interval)

    factory.assert_called_once_with(experiment=experiment)
    runner.wait.assert_called_once_with("slurm://nemo_run/123", wait_interval=interval)
    module.sleep.assert_called_once_with(interval)
    assert experiment.jobs[0].state == "SUCCEEDED"
    experiment.jobs[0].cleanup.assert_called_once_with()
    runner.status.assert_not_called()
    runner.cancel.assert_not_called()
    runner.__exit__.assert_called_once()


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "UNKNOWN"])
def test_unsuccessful_status_is_not_a_success(waiter, state):
    module, runner, _, experiment = waiter
    runner.wait.return_value = SimpleNamespace(state=state)
    with pytest.raises(RuntimeError, match="did not succeed"):
        module.wait_for_slurm_job(experiment, poll_interval=60)
    assert experiment.jobs[0].state == state
    runner.cancel.assert_not_called()


def test_unavailable_status_does_not_cancel_or_clean_up_job(waiter):
    module, runner, _, experiment = waiter
    runner.wait.return_value = None
    with pytest.raises(RuntimeError, match="status is unavailable"):
        module.wait_for_slurm_job(experiment, poll_interval=60)
    runner.cancel.assert_not_called()
    experiment.jobs[0].cleanup.assert_not_called()
    assert runner.wait.call_count == module.sleep.call_count == 3


def test_accounting_registration_lag_retries_at_the_same_interval(waiter):
    module, runner, _, experiment = waiter
    events = []
    module.sleep.side_effect = lambda seconds: events.append(("sleep", seconds))
    statuses = iter([None, SimpleNamespace(state="SUCCEEDED")])

    def wait(_handle, *, wait_interval):
        events.append(("wait", wait_interval))
        return next(statuses)

    runner.wait.side_effect = wait
    module.wait_for_slurm_job(experiment, poll_interval=120)
    assert events == [("sleep", 120), ("wait", 120), ("sleep", 120), ("wait", 120)]
    assert experiment.jobs[0].state == "SUCCEEDED"


@pytest.mark.parametrize("error", [RuntimeError("scheduler unavailable"), KeyboardInterrupt()])
def test_monitor_failure_or_interrupt_preserves_submitted_job(waiter, error):
    module, runner, _, experiment = waiter
    runner.wait.side_effect = error
    with pytest.raises(type(error)):
        module.wait_for_slurm_job(experiment, poll_interval=60)
    runner.cancel.assert_not_called()
    experiment.jobs[0].cleanup.assert_not_called()
    runner.__exit__.assert_called_once()


@pytest.mark.parametrize("value", ["-1", "0", "2", "59", "nan", "1.5"])
def test_parser_rejects_unsafe_polling_interval(waiter, value):
    module, _, _, _ = waiter
    with pytest.raises(argparse.ArgumentTypeError):
        module.slurm_poll_interval(value)


@pytest.mark.parametrize("value", ["60", "120"])
def test_parser_accepts_safe_polling_interval(waiter, value):
    module, _, _, _ = waiter
    assert module.slurm_poll_interval(value) == int(value)


def test_wait_rejects_invalid_interval_before_query(waiter):
    module, runner, _, experiment = waiter
    with pytest.raises(ValueError, match="at least 60"):
        module.wait_for_slurm_job(experiment, poll_interval=2)
    runner.wait.assert_not_called()


@pytest.mark.parametrize("count", [0, 2])
def test_wait_rejects_missing_or_multiple_jobs(waiter, count):
    module, runner, _, experiment = waiter
    experiment.jobs *= count
    with pytest.raises(ValueError, match="exactly one job"):
        module.wait_for_slurm_job(experiment, poll_interval=60)
    runner.wait.assert_not_called()


def test_wait_rejects_unsubmitted_job(waiter):
    module, runner, _, experiment = waiter
    experiment.jobs[0].launched = False
    with pytest.raises(ValueError, match="not been submitted"):
        module.wait_for_slurm_job(experiment, poll_interval=60)
    runner.wait.assert_not_called()
