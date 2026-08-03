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

import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


_SCRIPT = Path(__file__).parents[3] / "examples" / "conversion" / "hf_to_megatron_generate_nemotron_vlm.py"
_parallel_state = MagicMock()
_import_stubs = {
    "megatron": types.ModuleType("megatron"),
    "megatron.core": types.ModuleType("megatron.core"),
    "megatron.core.pipeline_parallel": types.ModuleType("megatron.core.pipeline_parallel"),
    "megatron.core.pipeline_parallel.schedules": types.ModuleType("megatron.core.pipeline_parallel.schedules"),
    "megatron.bridge": types.ModuleType("megatron.bridge"),
    "megatron.bridge.models": types.ModuleType("megatron.bridge.models"),
    "megatron.bridge.models.nemotron_vl": types.ModuleType("megatron.bridge.models.nemotron_vl"),
    "megatron.bridge.models.nemotron_vl.nemotron_vl_utils": types.ModuleType(
        "megatron.bridge.models.nemotron_vl.nemotron_vl_utils"
    ),
    "megatron.bridge.utils": types.ModuleType("megatron.bridge.utils"),
    "megatron.bridge.utils.common_utils": types.ModuleType("megatron.bridge.utils.common_utils"),
    "megatron.bridge.utils.safe_url": types.ModuleType("megatron.bridge.utils.safe_url"),
    "qwen_vl_utils": types.ModuleType("qwen_vl_utils"),
    "transformers": types.ModuleType("transformers"),
}
_import_stubs["megatron.core"].parallel_state = _parallel_state
_import_stubs["megatron.core.pipeline_parallel.schedules"].get_forward_backward_func = MagicMock()
_import_stubs["megatron.bridge"].AutoBridge = MagicMock()
_import_stubs["megatron.bridge.models.nemotron_vl.nemotron_vl_utils"].adjust_image_tokens = MagicMock()
_import_stubs["megatron.bridge.utils.common_utils"].get_last_rank = MagicMock()
_import_stubs["megatron.bridge.utils.common_utils"].print_rank_0 = MagicMock()
_import_stubs["megatron.bridge.utils.safe_url"].is_safe_public_http_url = MagicMock()
_import_stubs["megatron.bridge.utils.safe_url"].safe_url_open = MagicMock()
_import_stubs["qwen_vl_utils"].process_vision_info = MagicMock()
_import_stubs["transformers"].AutoProcessor = MagicMock()
_import_stubs["transformers"].AutoTokenizer = MagicMock()

with patch.dict(sys.modules, _import_stubs):
    _SCRIPT_GLOBALS = runpy.run_path(_SCRIPT)
_main = _SCRIPT_GLOBALS["main"]


@pytest.mark.unit
def test_text_only_generation_passes_empty_images_to_llava() -> None:
    """Prompt-only generation must preserve MCore's zero-tile image contract."""
    args = SimpleNamespace(
        ep=1,
        etp=1,
        hf_model_path="org/model",
        image_path=None,
        max_new_tokens=1,
        megatron_model_path=None,
        pp=1,
        prompt="hello",
        system_prompt=None,
        tp=1,
        video_path=None,
    )

    model = MagicMock(return_value=torch.zeros(1, 2, 8))
    model.cuda.return_value = model
    provider = MagicMock()
    provider.provide_distributed_model.return_value = [model]
    bridge = MagicMock()
    bridge.to_megatron_provider.return_value = provider

    tokenizer = MagicMock()
    tokenizer.convert_tokens_to_ids.side_effect = [10, 11]
    tokenizer.decode.return_value = "hello world"
    tokenizer.eos_token_id = 7
    tokenizer.pad_token = "<pad>"

    adjust_image_tokens = MagicMock(side_effect=lambda input_ids, *_args: input_ids)
    forward = MagicMock()

    def run_forward(**kwargs):
        output, _ = kwargs["forward_step_func"](kwargs["data_iterator"], kwargs["model"][0])
        output[0, -1, 1] = 1
        return [output]

    forward.side_effect = run_forward

    def all_gather(outputs, tensor, group):
        outputs[0].copy_(tensor)

    script_globals = {
        "AutoBridge": SimpleNamespace(from_hf_pretrained=MagicMock(return_value=bridge)),
        "AutoProcessor": SimpleNamespace(from_pretrained=MagicMock(return_value=MagicMock())),
        "AutoTokenizer": SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer)),
        "adjust_image_tokens": adjust_image_tokens,
        "get_forward_backward_func": MagicMock(return_value=forward),
        "get_last_rank": MagicMock(return_value=0),
        "print_rank_0": MagicMock(),
        "process_image_inputs": MagicMock(return_value=(torch.tensor([[3, 4]]), None, 0)),
    }

    with (
        patch.dict(_main.__globals__, script_globals),
        patch.object(torch.Tensor, "cuda", lambda self: self),
        patch.object(torch.distributed, "all_gather", side_effect=all_gather),
        patch.object(torch.distributed, "broadcast"),
        patch.object(_main.__globals__["parallel_state"], "get_tensor_model_parallel_group", return_value=None),
        patch.object(_main.__globals__["parallel_state"], "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(_main.__globals__["parallel_state"], "is_pipeline_last_stage", return_value=True),
    ):
        _main(args)

    images = model.call_args.kwargs["images"]
    assert images.dtype == torch.bfloat16
    assert images.numel() == 0
    adjust_image_tokens.assert_not_called()
    assert forward.call_count == 1
