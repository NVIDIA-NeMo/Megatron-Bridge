# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import signal
from types import SimpleNamespace
from unittest.mock import Mock


def test_train_step_accepts_pg_collection_argument():
    # Import locally to avoid import-time side effects in unrelated modules
    from megatron.bridge.training import train as train_module

    sig = inspect.signature(train_module.train_step)
    assert "pg_collection" in sig.parameters, "train_step must accept pg_collection param"


def test_should_skip_iteration_uses_passed_pg_collection(monkeypatch):
    # Arrange minimal GlobalState with only the fields that are used
    from megatron.bridge.training import train as train_module
    from megatron.bridge.training.state import GlobalState

    state = GlobalState()

    # Set up a minimal config needed by _should_skip_and_handle_iteration
    # iterations_to_skip uses 1-based iteration numbers (matching MLM convention).
    # {1} means "skip the 1st iteration", which fires when step=0 (step+1==1).
    state.cfg = SimpleNamespace(
        train=SimpleNamespace(
            iterations_to_skip={1},
            micro_batch_size=4,
            exit_signal_handler=False,
            exit_signal=signal.SIGTERM,
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


def test_train_stops_nsys_profiler_when_skipped_iteration_reaches_profile_end(monkeypatch):
    """Skipping the stop iteration must still close an active Nsys capture."""
    from megatron.bridge.training import profiling as profiling_module
    from megatron.bridge.training import train as train_module

    profiling = SimpleNamespace(
        use_pytorch_profiler=False,
        use_nsys_profiler=True,
        profile_step_start=0,
        profile_step_end=1,
        profile_ranks=[0],
        record_shapes=False,
    )
    config = SimpleNamespace(
        train=SimpleNamespace(
            manual_gc=False,
            check_weight_hash_across_dp_replicas_interval=None,
            train_iters=1,
            micro_batch_size=1,
            iterations_to_skip={1},
        ),
        validation=SimpleNamespace(eval_interval=0, start_eval_at_iter=None),
        profiling=profiling,
        straggler=None,
        ddp=SimpleNamespace(use_megatron_fsdp=False, overlap_param_gather=False),
        optimizer=SimpleNamespace(
            use_distributed_optimizer=False,
            optimizer_cuda_graph=False,
        ),
        model=SimpleNamespace(
            virtual_pipeline_model_parallel_size=None,
            cuda_graph_warmup_steps=0,
            cuda_graph_use_single_mempool=False,
            moe_expert_rank_capacity_factor=None,
        ),
        logger=SimpleNamespace(log_throughput_to_tensorboard=False),
        checkpoint=SimpleNamespace(save=None, save_interval=None),
        tensor_inspect=None,
    )
    state = SimpleNamespace(
        cfg=config,
        train_state=SimpleNamespace(
            step=0,
            consumed_train_samples=0,
            skipped_train_samples=0,
            floating_point_operations_so_far=0,
        ),
        timers=Mock(),
        straggler_timer=Mock(),
        energy_monitor=None,
        nvrx_straggler_manager=None,
        tensorboard_logger=None,
        wandb_logger=None,
        _comet_logger=None,
    )
    model_config = SimpleNamespace(cuda_graph_impl=None)
    rerun_state_machine = SimpleNamespace(current_iteration=0)
    pg_collection = SimpleNamespace(
        dp=SimpleNamespace(size=lambda: 1),
        pp=SimpleNamespace(size=lambda: 1),
    )
    checkpoint_manager = Mock()
    start_nsys_profiler = Mock(return_value=Mock())
    stop_nsys_profiler = Mock()

    monkeypatch.setattr(train_module, "get_model_config", lambda _model: model_config)
    monkeypatch.setattr(train_module, "get_rerun_state_machine", lambda: rerun_state_machine)
    monkeypatch.setattr(train_module, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(train_module, "update_num_microbatches", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "should_disable_forward_pre_hook", lambda *args: False)
    monkeypatch.setattr(train_module, "get_forward_backward_func", lambda **kwargs: Mock())
    monkeypatch.setattr(train_module, "is_full_iteration_cuda_graph", lambda _config: False)
    monkeypatch.setattr(train_module, "P2PCommunicator", lambda **kwargs: Mock())
    monkeypatch.setattr(train_module, "_dummy_train_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "_delete_cuda_graphs", lambda _helper: None)
    monkeypatch.setattr(train_module, "safe_shutdown_nvrx_straggler_manager", lambda _manager: None)
    monkeypatch.setattr(train_module, "tensor_inspect_end_if_enabled", lambda _config: None)
    monkeypatch.setattr(train_module, "should_fire", lambda *args: False)
    monkeypatch.setattr(train_module, "nvtx_range_push", lambda **kwargs: None)
    monkeypatch.setattr(train_module, "nvtx_range_pop", lambda **kwargs: None)
    monkeypatch.setattr(train_module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(profiling_module, "start_nsys_profiler", start_nsys_profiler)
    monkeypatch.setattr(profiling_module, "stop_nsys_profiler", stop_nsys_profiler)
    monkeypatch.setattr(train_module.fault_tolerance, "on_checkpointing_start", lambda _state: None)
    monkeypatch.setattr(
        train_module.fault_tolerance,
        "on_checkpointing_end",
        lambda **kwargs: None,
    )

    train_module.train(
        forward_step_func=Mock(),
        model=[Mock()],
        optimizer=Mock(),
        scheduler=Mock(),
        train_data_iterator=None,
        valid_data_iterator=None,
        global_state=state,
        checkpoint_manager=checkpoint_manager,
        pg_collection=pg_collection,
    )

    start_nsys_profiler.assert_called_once_with(profiling)
    stop_nsys_profiler.assert_called_once()
