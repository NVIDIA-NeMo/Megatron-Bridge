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

"""Functional tests for the training callback system."""

import pytest
import torch

from megatron.bridge.models.llama import Llama32ModelProvider1B
from megatron.bridge.training.callbacks import Callback, CallbackContext, CallbackManager
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import initialize_distributed


class TrackingCallback(Callback):
    """Tracks event ordering and context field availability."""

    def __init__(self):
        self.events: list[str] = []
        self.context_snapshots: list[dict] = []

    def _record(self, event_name: str, context: CallbackContext) -> None:
        self.events.append(event_name)
        self.context_snapshots.append(
            {
                "event": event_name,
                "has_state": context.state is not None,
                "has_model": context.model is not None and len(context.model) > 0,
                "has_user_state": context.user_state is not None,
                "has_optimizer": context.optimizer is not None,
                "has_scheduler": context.scheduler is not None,
                "has_loss_dict": context.loss_dict is not None,
                "has_grad_norm": context.grad_norm is not None,
                "has_skipped_iter": context.skipped_iter is not None,
                "has_total_loss_dict": context.total_loss_dict is not None,
            }
        )

    def on_train_start(self, context: CallbackContext) -> None:
        self._record("on_train_start", context)

    def on_train_step_start(self, context: CallbackContext) -> None:
        self._record("on_train_step_start", context)

    def on_train_step_end(self, context: CallbackContext) -> None:
        self._record("on_train_step_end", context)

    def on_train_end(self, context: CallbackContext) -> None:
        self._record("on_train_end", context)

    def on_eval_start(self, context: CallbackContext) -> None:
        self._record("on_eval_start", context)

    def on_eval_step_start(self, context: CallbackContext) -> None:
        self._record("on_eval_step_start", context)

    def on_eval_step_end(self, context: CallbackContext) -> None:
        self._record("on_eval_step_end", context)

    def on_eval_end(self, context: CallbackContext) -> None:
        self._record("on_eval_end", context)

    def on_test_start(self, context: CallbackContext) -> None:
        self._record("on_test_start", context)

    def on_test_step_start(self, context: CallbackContext) -> None:
        self._record("on_test_step_start", context)

    def on_test_step_end(self, context: CallbackContext) -> None:
        self._record("on_test_step_end", context)

    def on_test_end(self, context: CallbackContext) -> None:
        self._record("on_test_end", context)

    def get_event_count(self, event_name: str) -> int:
        return sum(1 for e in self.events if e == event_name)

    def get_snapshots_for_event(self, event_name: str) -> list[dict]:
        return [s for s in self.context_snapshots if s["event"] == event_name]


class UserStateCallback(Callback):
    """Tests user_state persistence across events."""

    def __init__(self):
        self.step_values: list[int] = []
        self.eval_read_values: list[int] = []
        self.test_read_values: list[int] = []
        self.final_count: int | None = None

    def on_train_start(self, context: CallbackContext) -> None:
        context.user_state["counter"] = 0

    def on_train_step_end(self, context: CallbackContext) -> None:
        context.user_state["counter"] += 1
        self.step_values.append(context.user_state["counter"])

    def on_eval_start(self, context: CallbackContext) -> None:
        self.eval_read_values.append(context.user_state.get("counter", -1))

    def on_test_start(self, context: CallbackContext) -> None:
        self.test_read_values.append(context.user_state.get("counter", -1))

    def on_train_end(self, context: CallbackContext) -> None:
        self.final_count = context.user_state.get("counter", -1)


class TestCallbacksEndToEnd:
    """Functional tests for callbacks in the training loop."""

    @pytest.mark.run_only_on("GPU")
    def test_callbacks(self):
        """Comprehensive test of callback system with both registration patterns.

        Tests in a single training run:
        1. Class-based callbacks (TrackingCallback, UserStateCallback)
        2. Functional callbacks (via register())
        3. Event firing counts and ordering
        4. Context field availability at each event
        5. user_state persistence across callback invocations
        """
        initialize_distributed()

        # Training configuration
        # eval_interval doesn't evenly divide train_iters to avoid eval at last step
        # This ensures in-training eval only runs once (at step 5), not at step 8
        train_iters = 8
        eval_interval = 5  # Eval only at step 5 during training
        eval_iters = 2

        model_cfg = Llama32ModelProvider1B(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=False,
            attention_softmax_in_fp32=True,
            pipeline_dtype=torch.bfloat16,
            bf16=True,
            seq_length=512,
            make_vocab_size_divisible_by=128,
            vocab_size=None,
            num_layers=1,
        )

        cfg = ConfigContainer(
            model=model_cfg,
            train=TrainingConfig(
                train_iters=train_iters,
                eval_interval=eval_interval,
                eval_iters=eval_iters,
                global_batch_size=8,
                micro_batch_size=1,
                exit_signal_handler=True,
            ),
            optimizer=OptimizerConfig(
                optimizer="adam",
                bf16=True,
                fp16=False,
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_eps=1e-8,
                use_distributed_optimizer=True,
                clip_grad=1.0,
                lr=3e-3,
                weight_decay=0.01,
                min_lr=1e-6,
            ),
            scheduler=SchedulerConfig(
                start_weight_decay=0.033,
                end_weight_decay=0.033,
                weight_decay_incr_style="constant",
                lr_decay_style="cosine",
                lr_warmup_iters=2,
                lr_warmup_init=0.0,
                lr_decay_iters=train_iters,
                override_opt_param_scheduler=True,
            ),
            ddp=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                average_in_collective=True,
                use_distributed_optimizer=True,
            ),
            dataset=MockGPTDatasetConfig(
                random_seed=1234,
                reset_attention_mask=False,
                reset_position_ids=False,
                eod_mask_loss=False,
                seq_length=512,
                num_dataset_builder_threads=1,
                data_sharding=True,
                dataloader_type="single",
                num_workers=1,
            ),
            logger=LoggerConfig(log_interval=5),
            tokenizer=TokenizerConfig(
                tokenizer_type="NullTokenizer",
                vocab_size=10000,
            ),
            checkpoint=CheckpointConfig(save=None),
            rng=RNGConfig(seed=1234),
        )

        # Create callbacks
        tracking_callback = TrackingCallback()
        user_state_callback = UserStateCallback()

        # Track functional callback invocations
        functional_log: list[str] = []

        # Create manager with both class-based and functional callbacks
        manager = CallbackManager()
        manager.add([tracking_callback, user_state_callback])
        manager.register("on_train_start", lambda ctx: functional_log.append("fn_start"))
        manager.register("on_train_step_end", lambda ctx: functional_log.append("fn_step"))
        manager.register("on_train_end", lambda ctx: functional_log.append("fn_end"))

        # Run training
        pretrain(cfg, forward_step, callbacks=manager)

        # Verify event firing counts
        assert tracking_callback.get_event_count("on_train_start") == 1
        assert tracking_callback.get_event_count("on_train_end") == 1
        assert tracking_callback.get_event_count("on_train_step_start") == train_iters
        assert tracking_callback.get_event_count("on_train_step_end") == train_iters

        # Eval runs: 1 during training (step 5 only) + 1 post-training validation
        in_training_eval_runs = train_iters // eval_interval  # 8 // 5 = 1
        post_training_eval_runs = 1  # validation only (test uses on_test_* events)
        expected_eval_runs = in_training_eval_runs + post_training_eval_runs
        assert tracking_callback.get_event_count("on_eval_start") == expected_eval_runs
        assert tracking_callback.get_event_count("on_eval_end") == expected_eval_runs

        expected_eval_steps = expected_eval_runs * eval_iters
        assert tracking_callback.get_event_count("on_eval_step_start") == expected_eval_steps
        assert tracking_callback.get_event_count("on_eval_step_end") == expected_eval_steps

        # Test runs: 1 post-training test
        expected_test_runs = 1
        assert tracking_callback.get_event_count("on_test_start") == expected_test_runs
        assert tracking_callback.get_event_count("on_test_end") == expected_test_runs

        expected_test_steps = expected_test_runs * eval_iters
        assert tracking_callback.get_event_count("on_test_step_start") == expected_test_steps
        assert tracking_callback.get_event_count("on_test_step_end") == expected_test_steps

        # Verify event order
        events = tracking_callback.events
        assert events[0] == "on_train_start", "First event should be on_train_start"
        # Post-training test is the final phase, so on_test_end is last
        assert events[-1] == "on_test_end", "Last event should be on_test_end"
        # on_train_end should come before post-training test
        train_end_idx = events.index("on_train_end")
        test_start_idx = events.index("on_test_start")
        assert train_end_idx < test_start_idx, "on_train_end should precede post-training test"

        # Verify step events come in pairs (step_end before next step_start)
        for i, event in enumerate(events):
            if event == "on_train_step_start":
                remaining = events[i + 1 :]
                next_step_start = (
                    remaining.index("on_train_step_start") if "on_train_step_start" in remaining else len(remaining)
                )
                next_step_end = (
                    remaining.index("on_train_step_end") if "on_train_step_end" in remaining else len(remaining)
                )
                assert next_step_end < next_step_start or next_step_start == len(remaining)

        # Verify context data availability
        for snapshot in tracking_callback.context_snapshots:
            assert snapshot["has_state"], f"{snapshot['event']} missing state"
            assert snapshot["has_model"], f"{snapshot['event']} missing model"
            assert snapshot["has_user_state"], f"{snapshot['event']} missing user_state"

        training_events = ["on_train_start", "on_train_step_start", "on_train_step_end", "on_train_end"]
        for snapshot in tracking_callback.context_snapshots:
            if snapshot["event"] in training_events:
                assert snapshot["has_optimizer"], f"{snapshot['event']} missing optimizer"
                assert snapshot["has_scheduler"], f"{snapshot['event']} missing scheduler"

        for snapshot in tracking_callback.get_snapshots_for_event("on_train_step_end"):
            assert snapshot["has_loss_dict"], "on_train_step_end missing loss_dict"
            assert snapshot["has_grad_norm"], "on_train_step_end missing grad_norm"
            assert snapshot["has_skipped_iter"], "on_train_step_end missing skipped_iter"

        for snapshot in tracking_callback.get_snapshots_for_event("on_eval_end"):
            assert snapshot["has_total_loss_dict"], "on_eval_end missing total_loss_dict"

        for snapshot in tracking_callback.get_snapshots_for_event("on_test_end"):
            assert snapshot["has_total_loss_dict"], "on_test_end missing total_loss_dict"

        # Verify user_state persistence (UserStateCallback)
        assert user_state_callback.final_count == train_iters, (
            f"Final counter should be {train_iters}, got {user_state_callback.final_count}"
        )
        assert user_state_callback.step_values == list(range(1, train_iters + 1)), (
            f"Step values should be [1..{train_iters}], got {user_state_callback.step_values}"
        )
        # In-training eval happens after step 5, counter should be 5
        # Post-training validation reads counter=8 (final train_iters)
        assert user_state_callback.eval_read_values[0] == eval_interval, (
            f"First eval should read counter={eval_interval}, got {user_state_callback.eval_read_values[0]}"
        )
        assert user_state_callback.eval_read_values[-1] == train_iters, (
            f"Post-training eval should read counter={train_iters}, got {user_state_callback.eval_read_values[-1]}"
        )
        assert len(user_state_callback.eval_read_values) == expected_eval_runs, (
            f"Should have {expected_eval_runs} eval reads, got {len(user_state_callback.eval_read_values)}"
        )
        # Post-training test runs after training, reads counter=8
        assert len(user_state_callback.test_read_values) == expected_test_runs, (
            f"Should have {expected_test_runs} test reads, got {len(user_state_callback.test_read_values)}"
        )
        assert user_state_callback.test_read_values[0] == train_iters, (
            f"Test should read counter={train_iters}, got {user_state_callback.test_read_values[0]}"
        )

        # Verify functional callbacks fired
        assert functional_log[0] == "fn_start", "Functional on_train_start should fire"
        assert functional_log[-1] == "fn_end", "Functional on_train_end should fire"
        assert functional_log.count("fn_step") == train_iters, (
            f"Functional on_train_step_end should fire {train_iters} times"
        )
