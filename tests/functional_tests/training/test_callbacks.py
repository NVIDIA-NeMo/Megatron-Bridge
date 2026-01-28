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


class TrackingCallback(Callback):
    """Tracks event ordering and context field availability."""

    def __init__(self):
        """
        Initialize the callback's tracking state.
        
        Attributes:
            events (list[str]): Ordered list of event names recorded during the run.
            context_snapshots (list[dict]): Parallel list of dictionaries capturing availability flags for context fields at each recorded event.
        """
        self.events: list[str] = []
        self.context_snapshots: list[dict] = []

    def _record(self, event_name: str, context: CallbackContext) -> None:
        """
        Record an event and a snapshot of which fields on the provided CallbackContext are present.
        
        Parameters:
            event_name (str): The lifecycle event name to record (e.g., "on_train_start").
            context (CallbackContext): The callback context whose field presence will be inspected and stored.
        
        Notes:
            Appends `event_name` to `self.events` and a dictionary describing presence of selected
            context fields to `self.context_snapshots`.
        """
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
        """
        Record the training-start event and capture a snapshot of the callback context.
        
        Parameters:
            context (CallbackContext): The current callback context containing training state, model, optimizer, scheduler, and other runtime fields.
        """
        self._record("on_train_start", context)

    def on_train_step_start(self, context: CallbackContext) -> None:
        """
        Record the occurrence of a training step start and capture the current callback context snapshot.
        
        Parameters:
            context (CallbackContext): The context for the current callback invocation, whose fields are recorded for later inspection.
        """
        self._record("on_train_step_start", context)

    def on_train_step_end(self, context: CallbackContext) -> None:
        """
        Record that a training step has completed and capture a snapshot of the callback context.
        
        Appends the "on_train_step_end" event to the callback's event log and stores a snapshot of which
        CallbackContext fields (state, model, user_state, optimizer, scheduler, loss_dict, grad_norm,
        skipped_iter, total_loss_dict) are present.
        
        Parameters:
            context (CallbackContext): The current callback context whose field availability is recorded.
        """
        self._record("on_train_step_end", context)

    def on_train_end(self, context: CallbackContext) -> None:
        """
        Record that training has finished and capture a snapshot of the provided callback context.
        
        Parameters:
            context (CallbackContext): The callback context available at training end.
        """
        self._record("on_train_end", context)

    def on_eval_start(self, context: CallbackContext) -> None:
        """
        Record the start of an evaluation run and capture a snapshot of the callback context.
        
        Parameters:
            context (CallbackContext): The callback context for the evaluation start event; used to determine which context fields are available and to record their presence.
        """
        self._record("on_eval_start", context)

    def on_eval_step_start(self, context: CallbackContext) -> None:
        """
        Record that an evaluation step has started and capture a snapshot of the callback context.
        
        Parameters:
            context (CallbackContext): The current callback context whose field availability is captured.
        """
        self._record("on_eval_step_start", context)

    def on_eval_step_end(self, context: CallbackContext) -> None:
        """
        Record the occurrence of an evaluation-step end event and capture the current callback context snapshot.
        
        Parameters:
            context (CallbackContext): The callback context provided at the end of an evaluation step.
        """
        self._record("on_eval_step_end", context)

    def on_eval_end(self, context: CallbackContext) -> None:
        """
        Record that an evaluation phase has completed and capture the event's callback context for later inspection.
        
        Parameters:
            context (CallbackContext): Runtime callback context containing state, model, user_state, optimizer, scheduler, loss and related fields available at evaluation end.
        """
        self._record("on_eval_end", context)

    def on_test_start(self, context: CallbackContext) -> None:
        """
        Record the start of the test phase and capture a snapshot of the provided callback context.
        
        Parameters:
            context (CallbackContext): The callback context containing runtime fields (e.g., state, model, user_state, optimizer, scheduler, losses) to be recorded.
        """
        self._record("on_test_start", context)

    def on_test_step_start(self, context: CallbackContext) -> None:
        """
        Record the 'on_test_step_start' event and capture a snapshot of which CallbackContext fields are present.
        
        Parameters:
            context (CallbackContext): The callback context for the test step whose fields (e.g., state, model, user_state, optimizer, scheduler, loss/grad info) will be inspected and recorded.
        """
        self._record("on_test_step_start", context)

    def on_test_step_end(self, context: CallbackContext) -> None:
        """
        Record that a test step has finished and capture a snapshot of the provided callback context.
        
        Parameters:
            context (CallbackContext): The callback context at the end of the test step whose available fields are inspected and recorded.
        """
        self._record("on_test_step_end", context)

    def on_test_end(self, context: CallbackContext) -> None:
        """
        Record that the test phase has completed by capturing the event name and a snapshot of the callback context.
        
        Parameters:
            context (CallbackContext): The callback context at test end containing available runtime fields (state, model, user_state, optimizer, scheduler, loss information, etc.).
        """
        self._record("on_test_end", context)

    def get_event_count(self, event_name: str) -> int:
        """
        Return the number of times the specified event was recorded.
        
        Parameters:
            event_name (str): Event name to count in the recorded events.
        
        Returns:
            count (int): Number of occurrences of the specified event in self.events.
        """
        return sum(1 for e in self.events if e == event_name)

    def get_snapshots_for_event(self, event_name: str) -> list[dict]:
        """
        Retrieve all context snapshots recorded for a specific callback event.
        
        Parameters:
            event_name (str): The event name to filter snapshots by (e.g., "on_train_step_end").
        
        Returns:
            list[dict]: A list of snapshot dictionaries for the specified event. Each snapshot records presence flags for context fields such as `state`, `model`, `user_state`, `optimizer`, `scheduler`, `loss_dict`, `grad_norm`, `skipped_iter`, and `total_loss_dict`.
        """
        return [s for s in self.context_snapshots if s["event"] == event_name]


class UserStateCallback(Callback):
    """Tests user_state persistence across events."""

    def __init__(self):
        """
        Initialize the UserStateCallback's tracking fields.
        
        Attributes:
            step_values (list[int]): Recorded user_state "counter" values after each training step.
            eval_read_values (list[int]): User_state "counter" values read at the start of each evaluation run.
            test_read_values (list[int]): User_state "counter" values read at the start of each test run.
            final_count (int | None): Final user_state "counter" value observed at training end, or None if not set.
        """
        self.step_values: list[int] = []
        self.eval_read_values: list[int] = []
        self.test_read_values: list[int] = []
        self.final_count: int | None = None

    def on_train_start(self, context: CallbackContext) -> None:
        """
        Initialize the callback context's user_state "counter" to 0.
        
        Parameters:
            context (CallbackContext): The callback context whose `user_state` dictionary will receive the `"counter"` key.
        """
        context.user_state["counter"] = 0

    def on_train_step_end(self, context: CallbackContext) -> None:
        """
        Increment the per-run user_state counter and record its new value.
        
        Parameters:
            context (CallbackContext): Callback context whose `user_state["counter"]` will be incremented and whose new value will be appended to this callback's `step_values`.
        """
        context.user_state["counter"] += 1
        self.step_values.append(context.user_state["counter"])

    def on_eval_start(self, context: CallbackContext) -> None:
        """
        Record the current user_state "counter" value when evaluation starts.
        
        Parameters:
            context (CallbackContext): Callback context whose `user_state` mapping is read; appends the value of `user_state["counter"]` or -1 if absent to `eval_read_values`.
        """
        self.eval_read_values.append(context.user_state.get("counter", -1))

    def on_test_start(self, context: CallbackContext) -> None:
        """
        Record the current user_state "counter" value when testing starts.
        
        Parameters:
            context (CallbackContext): Callback context whose `user_state` is read; appends `user_state.get("counter", -1)` to `self.test_read_values`.
        """
        self.test_read_values.append(context.user_state.get("counter", -1))

    def on_train_end(self, context: CallbackContext) -> None:
        """
        Record the final value of the user_state "counter" into self.final_count.
        
        Parameters:
            context (CallbackContext): Callback context from which the `user_state["counter"]` value is read; stores -1 if the key is absent.
        """
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
                adam_eps=1e-5,
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