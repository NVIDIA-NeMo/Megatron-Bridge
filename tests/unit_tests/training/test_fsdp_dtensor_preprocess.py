# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""fsdp_dtensor preprocessing order, and validation of a partial model load.

The handlers themselves live in megatron-core; what is checked here is that Bridge runs
them, runs them in an order they depend on, degrades cleanly on a megatron-core that does
not have them yet, and reports model weights a partial load cannot supply.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch.nn.functional as F

from megatron.bridge.training.checkpointing import (
    CheckpointType,
    load_fsdp_dtensor_checkpoint,
    preprocess_fsdp_dtensor_state_dict,
)
from megatron.bridge.training.config import CheckpointConfig


_CKPT_MOD = "megatron.bridge.training.checkpointing"

# Handlers Bridge picks up from megatron-core, in the order preprocessing must run them.
_MLA_HANDLER = "handle_mla_down_proj_in_state_dict"
_MTP_HANDLER = "handle_mtp_in_state_dict"


@pytest.fixture
def preprocess_harness():
    """Run preprocess_fsdp_dtensor_state_dict with every handler replaced by a recorder.

    Yields the call-order list; handlers are enabled/disabled per test via ``enable``.
    """
    calls = []

    def recorder(name, splits_optimizer=True):
        def handler(model, model_state_dict, optimizer_state_dict):
            calls.append(name)
            return model_state_dict, optimizer_state_dict

        def expert_handler(model_state_dict, num_experts):
            calls.append(name)
            return model_state_dict

        return expert_handler if not splits_optimizer else handler

    def run(*, num_experts=None, gated_linear_unit=True, enable=(_MLA_HANDLER, _MTP_HANDLER), state_dict=None):
        model_config = SimpleNamespace(
            gated_linear_unit=gated_linear_unit,
            activation_func=F.silu,
            num_moe_experts=num_experts,
        )
        patches = {
            "handle_fp8_extra_state_case": MagicMock(),
            "preprocess_state_dict_for_uneven_dtensor": MagicMock(),
            "handle_swiglu_in_state_dict": recorder("swiglu"),
            "handle_gdn_in_state_dict": recorder("gdn"),
            "handle_experts_in_state_dict": recorder("experts", splits_optimizer=False),
            _MLA_HANDLER: recorder("mla") if _MLA_HANDLER in enable else None,
            _MTP_HANDLER: recorder("mtp") if _MTP_HANDLER in enable else None,
        }
        with patch("megatron.core.utils.get_model_config", return_value=model_config):
            with _patch_many(patches):
                return preprocess_fsdp_dtensor_state_dict(
                    None,
                    state_dict if state_dict is not None else {"model": {}, "optimizer": {"state": {}}},
                    MagicMock(),
                )

    yield SimpleNamespace(run=run, calls=calls)


def _patch_many(attrs):
    """Context manager applying several ``patch.object``-style overrides on the module."""
    import contextlib

    stack = contextlib.ExitStack()
    for name, value in attrs.items():
        stack.enter_context(patch(f"{_CKPT_MOD}.{name}", value))
    return stack


class TestPreprocessHandlerOrder:
    def test_all_handlers_run(self, preprocess_harness):
        preprocess_harness.run(num_experts=8)
        assert set(preprocess_harness.calls) == {"swiglu", "gdn", "mla", "experts", "mtp"}

    def test_mla_split_runs_before_expert_reindexing(self, preprocess_harness):
        preprocess_harness.run(num_experts=8)
        calls = preprocess_harness.calls
        assert calls.index("mla") < calls.index("experts")

    def test_mtp_rename_runs_last(self, preprocess_harness):
        """The other handlers resolve keys against live module paths, which still use
        mtp_model_layer, so the rename to the on-disk name has to come after them."""
        preprocess_harness.run(num_experts=8)
        assert preprocess_harness.calls[-1] == "mtp"

    def test_handlers_run_without_experts(self, preprocess_harness):
        preprocess_harness.run(num_experts=None)
        assert preprocess_harness.calls == ["swiglu", "gdn", "mla", "mtp"]

    def test_mla_and_mtp_run_for_non_swiglu_model(self, preprocess_harness):
        """MLA down-proj fusion and MTP are independent of the activation function."""
        preprocess_harness.run(gated_linear_unit=False)
        assert preprocess_harness.calls == ["gdn", "mla", "mtp"]

    def test_optimizer_state_passed_through(self, preprocess_harness):
        state_dict = {"model": {"a": 1}, "optimizer": {"state": {"b": 2}}}
        result = preprocess_harness.run(state_dict=state_dict)
        assert result["model"] == {"a": 1}
        assert result["optimizer"] == {"state": {"b": 2}}

    def test_runs_without_optimizer_section(self, preprocess_harness):
        result = preprocess_harness.run(state_dict={"model": {"a": 1}})
        assert "optimizer" not in result
        assert preprocess_harness.calls == ["swiglu", "gdn", "mla", "mtp"]


class TestPreprocessOlderMegatronCore:
    """Bridge tracks megatron-core through a submodule, so both handlers may be absent."""

    def test_missing_mla_handler_skipped(self, preprocess_harness):
        preprocess_harness.run(enable=(_MTP_HANDLER,))
        assert preprocess_harness.calls == ["swiglu", "gdn", "mtp"]

    def test_missing_mtp_handler_skipped(self, preprocess_harness):
        preprocess_harness.run(enable=(_MLA_HANDLER,))
        assert preprocess_harness.calls == ["swiglu", "gdn", "mla"]

    def test_both_handlers_missing(self, preprocess_harness):
        preprocess_harness.run(enable=())
        assert preprocess_harness.calls == ["swiglu", "gdn"]


@pytest.fixture
def load_harness():
    """Drive load_fsdp_dtensor_checkpoint far enough to reach the validation step."""
    metadata = {"model": {"present.weight": object()}}

    def run(*, strict_load=False, strictness="assume_ok_unexpected", finetune=False, validator=MagicMock()):
        ckpt_cfg = SimpleNamespace(
            strict_fsdp_dtensor_load=strict_load,
            dist_ckpt_strictness=strictness,
            finetune=finetune,
        )
        reader = MagicMock()
        reader.read_metadata.return_value = SimpleNamespace(state_dict_metadata=metadata)
        state_dict = {"model": {"present.weight": 1}, "_model": [MagicMock()]}

        patches = {
            "HAVE_MEGATRON_FSDP": True,
            "_get_filesystem_reader": MagicMock(return_value=reader),
            "get_checkpoint_name": MagicMock(return_value="/ckpt/iter_0000010"),
            "preprocess_fsdp_dtensor_state_dict": lambda cfg, sd, model: sd,
            "print_diff_in_state_dicts": MagicMock(),
            "validate_fsdp_dtensor_model_load": validator,
        }
        with _patch_many(patches):
            with patch("torch.distributed.get_rank", return_value=0):
                with patch("torch.distributed.checkpoint.load_state_dict"):
                    result = load_fsdp_dtensor_checkpoint(
                        "/ckpt", ckpt_cfg, rank0=False, sharded_state_dict=state_dict, iteration=10
                    )
        return result, validator

    return SimpleNamespace(run=run, metadata=metadata)


class TestLoadValidatesModelKeys:
    def test_validator_called_on_partial_load(self, load_harness):
        _, validator = load_harness.run(strict_load=False)
        validator.assert_called_once()

    def test_validator_receives_metadata_and_checkpoint_name(self, load_harness):
        _, validator = load_harness.run(strict_load=False)
        args, kwargs = validator.call_args
        assert args[0] is load_harness.metadata
        assert args[2] == "/ckpt/iter_0000010"

    def test_strictness_forwarded(self, load_harness):
        _, validator = load_harness.run(strictness="log_unexpected")
        assert validator.call_args.kwargs["strict"] == "log_unexpected"

    def test_reuses_dist_ckpt_strictness(self, load_harness):
        """The existing checkpoint strictness knob drives this check too, rather than a
        format-specific flag. megatron-core escalates its assume_ok_unexpected default to a
        raise for fsdp_dtensor, since a partial load never raises on its own."""
        _, validator = load_harness.run()
        assert validator.call_args.kwargs["strict"] == "assume_ok_unexpected"

    def test_skipped_when_finetuning(self, load_harness):
        """A finetune loads only part of the model on purpose."""
        _, validator = load_harness.run(finetune=True)
        validator.assert_not_called()

    def test_skipped_on_strict_load(self, load_harness):
        """With a fully strict load the underlying DCP load already raises."""
        _, validator = load_harness.run(strict_load=True)
        validator.assert_not_called()

    def test_skipped_on_older_megatron_core(self, load_harness):
        """With the symbol unavailable the load must still complete rather than fail."""
        (_, checkpoint_name, _, ckpt_type), _ = load_harness.run(validator=None)
        assert checkpoint_name == "/ckpt/iter_0000010"
        assert ckpt_type == CheckpointType.FSDP_DTENSOR

    def test_load_still_returns_fsdp_dtensor_type(self, load_harness):
        (_, _, _, ckpt_type), _ = load_harness.run()
        assert ckpt_type == CheckpointType.FSDP_DTENSOR


class TestFsdpDtensorLoadConfig:
    def test_strictness_knob_is_inherited_not_redeclared(self):
        """No bespoke config field: the check rides on the existing megatron-core knob."""
        assert CheckpointConfig().dist_ckpt_strictness == "assume_ok_unexpected"
        assert "strict_fsdp_dtensor_model_load" not in CheckpointConfig().__dataclass_fields__

    def test_partial_load_remains_the_default(self):
        """The new check is what makes a partial load safe, so partial stays the default."""
        assert CheckpointConfig().strict_fsdp_dtensor_load is False
