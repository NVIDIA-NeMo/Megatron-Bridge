import inspect
from types import SimpleNamespace

import pytest


def test_train_step_accepts_pg_collection_argument():
    # Import locally to avoid import-time side effects in unrelated modules
    from megatron.bridge.training import train as train_module

    sig = inspect.signature(train_module.train_step)
    assert "pg_collection" in sig.parameters, "train_step must accept pg_collection param"


def test_should_skip_iteration_uses_passed_pg_collection(monkeypatch):
    # Arrange minimal GlobalState with only the fields that are used
    from megatron.bridge.training.state import GlobalState
    from megatron.bridge.training import train as train_module

    state = GlobalState()

    # Set up a minimal config needed by _should_skip_and_handle_iteration
    # We skip step 0 so that the function executes the skip path.
    state.cfg = SimpleNamespace(
        train=SimpleNamespace(
            iterations_to_skip={0},
            micro_batch_size=4,
        )
    )

    # Fake pg_collection with a DP size
    class _DP:
        def size(self):
            return 3

    class _PG:
        def __init__(self):
            self.dp = _DP()

    fake_pg = _PG()

    # Ensure deterministic microbatch count without touching global calculators
    monkeypatch.setattr(train_module, "get_num_microbatches", lambda: 2)

    # Avoid any distributed or pipeline logic inside the dummy step
    monkeypatch.setattr(train_module, "_dummy_train_step", lambda *args, **kwargs: None)

    # Pre-check counters
    assert state.train_state.step == 0
    assert state.train_state.consumed_train_samples == 0
    assert state.train_state.skipped_train_samples == 0

    # Act
    did_skip = train_module._should_skip_and_handle_iteration(state, None, fake_pg)

    # Assert
    assert did_skip is True
    # One iteration skipped
    assert state.train_state.step == 1
    # Batch size = dp.size * micro_batch_size * num_microbatches = 3 * 4 * 2 = 24
    expected_batch = 3 * 4 * 2
    assert state.train_state.consumed_train_samples == expected_batch
    assert state.train_state.skipped_train_samples == expected_batch


