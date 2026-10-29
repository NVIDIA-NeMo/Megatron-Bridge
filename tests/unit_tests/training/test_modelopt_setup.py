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

import megatron.bridge.training.setup as training_setup


@pytest.mark.parametrize("resume_has_modelopt_state", [True, False])
def test_modelopt_restore_prefers_native_resume_checkpoint(resume_has_modelopt_state):
    class StopAfterHooksRegistered(Exception):
        pass

    cfg = SimpleNamespace(
        checkpoint=SimpleNamespace(
            load="/resume",
            pretrained_checkpoint="/pretrained",
            ckpt_step=17,
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
        model=SimpleNamespace(
            fine_grained_activation_offloading=False,
            restore_modelopt_state=True,
            vocab_size=64,
        ),
        peft=None,
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
    hooks = []
    start_time_tensor = Mock()
    start_time_tensor.item.return_value = 0.0

    with (
        patch.multiple(
            training_setup,
            barrier_and_log=Mock(),
            build_tokenizer=Mock(return_value=SimpleNamespace(vocab_size=32)),
            create_checkpoint_manager=Mock(),
            checkpoint_exists=Mock(return_value=True),
            initialize_megatron=Mock(return_value=object()),
            initialize_tensor_inspect_pre_model_initialization=Mock(),
            maybe_log_and_save_config=Mock(),
            print_rank_0=Mock(),
            set_experimental_flag=Mock(),
            set_jit_fusion_options=Mock(),
            setup_logging=Mock(),
            start_memory_history_recording=Mock(),
            _build_distributed_model=Mock(side_effect=StopAfterHooksRegistered),
            _register_pre_wrap_hook=Mock(side_effect=lambda _model, hook: hooks.append(hook)),
        ),
        patch.object(torch, "tensor", return_value=start_time_tensor),
        patch.object(torch.distributed, "all_reduce"),
        patch(
            "megatron.bridge.training.post_training.checkpointing.has_modelopt_state",
            side_effect=lambda checkpoint_path, **_kwargs: (
                resume_has_modelopt_state if checkpoint_path == "/resume" else True
            ),
        ),
        patch("megatron.bridge.training.post_training.checkpointing.load_modelopt_state") as mock_load_modelopt,
    ):
        with pytest.raises(StopAfterHooksRegistered):
            training_setup.setup(state, Mock())

        assert cfg.model.vocab_size == 64
        assert cfg.model.should_pad_vocab is False
        assert len(hooks) == 1
        if resume_has_modelopt_state:
            hooks[0]([])
        else:
            with pytest.raises(RuntimeError, match="No modelopt_state found in selected checkpoint=/resume"):
                hooks[0]([])

    if resume_has_modelopt_state:
        mock_load_modelopt.assert_called_once_with([], "/resume", ckpt_step=17)
    else:
        mock_load_modelopt.assert_not_called()
