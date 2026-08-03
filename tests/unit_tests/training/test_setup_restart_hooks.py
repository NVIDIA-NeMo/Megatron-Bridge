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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch import nn

import megatron.bridge.training.setup as training_setup
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft.base import PEFT


class _OneTimeAdapterPEFT(PEFT):
    """Minimal PEFT transform that exposes a duplicate-application freeze."""

    def transform(self, module: nn.Module, name: str | None = None, prefix: str | None = None) -> nn.Module:
        if not isinstance(module, nn.Linear) or getattr(module, "_adapter_applied", False):
            return module

        module._adapter_applied = True
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad = True
        return module


def test_peft_setup_hook_does_not_accumulate_across_restart_attempts():
    """A rebuilt model must receive one PEFT transform while user hooks survive."""

    class StopAfterHooksRegistered(Exception):
        pass

    model_provider = GPTModelProvider(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=1,
        vocab_size=64,
    )
    user_hook = Mock(side_effect=lambda model: model)
    model_provider.register_pre_wrap_hook(user_hook)
    peft = _OneTimeAdapterPEFT()
    cfg = SimpleNamespace(
        checkpoint=SimpleNamespace(
            load=None,
            pretrained_checkpoint="/pretrained",
            save=None,
        ),
        dataset=SimpleNamespace(),
        dist=SimpleNamespace(enable_megatron_core_experimental=False, disable_jit_fuser=False),
        ft=None,
        logger=SimpleNamespace(
            filter_warnings=False,
            log_progress=False,
            logging_level="INFO",
            modules_to_filter=[],
            set_level_for_all_loggers=False,
        ),
        model=model_provider,
        peft=peft,
        profiling=SimpleNamespace(),
        tensor_inspect=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        train=SimpleNamespace(micro_batch_size=1, num_epochs=None),
    )
    timer = MagicMock()
    state = SimpleNamespace(
        cfg=cfg,
        initialize_async_checkpoint_worker=Mock(),
        start_time=0.0,
        timers=Mock(return_value=timer),
    )
    start_time_tensor = Mock()
    start_time_tensor.item.return_value = 0.0

    with (
        patch.multiple(
            training_setup,
            barrier_and_log=Mock(),
            build_tokenizer=Mock(return_value=SimpleNamespace(vocab_size=32)),
            checkpoint_exists=Mock(side_effect=lambda path: path == "/pretrained"),
            create_checkpoint_manager=Mock(),
            initialize_megatron=Mock(return_value=object()),
            initialize_tensor_inspect_pre_model_initialization=Mock(),
            maybe_log_and_save_config=Mock(),
            print_rank_0=Mock(),
            set_experimental_flag=Mock(),
            set_jit_fusion_options=Mock(),
            setup_logging=Mock(),
            start_memory_history_recording=Mock(),
            _build_distributed_model=Mock(side_effect=StopAfterHooksRegistered),
            _load_checkpoint_from_path=Mock(),
        ),
        patch.object(torch, "tensor", return_value=start_time_tensor),
        patch.object(torch.distributed, "all_reduce"),
    ):
        for _attempt in range(2):
            with pytest.raises(StopAfterHooksRegistered):
                training_setup.setup(state, Mock())

        model = [nn.Sequential(nn.Linear(4, 4))]
        transformed_model = model_provider.pre_wrap_hook(model)

        user_hook.assert_called_once_with(model)
        assert any(parameter.requires_grad for parameter in transformed_model[0].parameters())
        assert peft.params_to_save
