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


_SCRIPT = Path(__file__).parents[3] / "examples" / "conversion" / "hf_to_megatron_generate_vlm.py"
_parallel_state = MagicMock()
_import_stubs = {
    "megatron": types.ModuleType("megatron"),
    "megatron.core": types.ModuleType("megatron.core"),
    "megatron.core.inference": types.ModuleType("megatron.core.inference"),
    "megatron.core.inference.contexts": types.ModuleType("megatron.core.inference.contexts"),
    "megatron.core.pipeline_parallel": types.ModuleType("megatron.core.pipeline_parallel"),
    "megatron.core.pipeline_parallel.schedules": types.ModuleType("megatron.core.pipeline_parallel.schedules"),
    "megatron.bridge": types.ModuleType("megatron.bridge"),
    "megatron.bridge.models": types.ModuleType("megatron.bridge.models"),
    "megatron.bridge.models.hf_pretrained": types.ModuleType("megatron.bridge.models.hf_pretrained"),
    "megatron.bridge.models.hf_pretrained.utils": types.ModuleType("megatron.bridge.models.hf_pretrained.utils"),
    "megatron.bridge.utils": types.ModuleType("megatron.bridge.utils"),
    "megatron.bridge.utils.common_utils": types.ModuleType("megatron.bridge.utils.common_utils"),
    "transformers": types.ModuleType("transformers"),
    "vlm_generate_utils": types.ModuleType("vlm_generate_utils"),
}
_import_stubs["megatron.core"].parallel_state = _parallel_state
_import_stubs["megatron.core.inference.contexts"].StaticInferenceContext = MagicMock()
_import_stubs["megatron.core.pipeline_parallel.schedules"].get_forward_backward_func = MagicMock()
_import_stubs["megatron.bridge"].AutoBridge = MagicMock()
_import_stubs["megatron.bridge.models.hf_pretrained.utils"].is_safe_repo = MagicMock()
_import_stubs["megatron.bridge.utils.common_utils"].get_last_rank = MagicMock()
_import_stubs["megatron.bridge.utils.common_utils"].maybe_initialize_distributed = MagicMock()
_import_stubs["megatron.bridge.utils.common_utils"].print_rank_0 = MagicMock()
_import_stubs["megatron.bridge.utils.common_utils"].print_rank_last = MagicMock()
for name in ("AutoConfig", "AutoProcessor", "AutoTokenizer", "GenerationConfig"):
    setattr(_import_stubs["transformers"], name, MagicMock())
for name in (
    "pad_input_ids_to_tp_multiple",
    "patch_kimi_vision_processor",
    "process_image_inputs",
    "process_multi_image_inputs",
    "process_video_inputs",
    "to_cuda",
):
    setattr(_import_stubs["vlm_generate_utils"], name, MagicMock())

with patch.dict(sys.modules, _import_stubs):
    _SCRIPT_GLOBALS = runpy.run_path(_SCRIPT)
_build_inference_context = _SCRIPT_GLOBALS["_build_inference_context"]
_checkpoint_load_overrides = _SCRIPT_GLOBALS["_checkpoint_load_overrides"]
_last_real_token_logits = _SCRIPT_GLOBALS["_last_real_token_logits"]
_main = _SCRIPT_GLOBALS["main"]
_vlm_forward_step = _SCRIPT_GLOBALS["vlm_forward_step"]


@pytest.mark.unit
def test_generation_stops_on_any_configured_eos_token() -> None:
    """An alternate EOS from generation_config must end the greedy loop."""
    args = SimpleNamespace(
        ep=1,
        etp=1,
        hf_model_path="org/model",
        hf_revision="revision",
        image_path=None,
        image_paths=None,
        max_new_tokens=2,
        megatron_model_path=None,
        pp=1,
        pp_layout=None,
        prompt="prompt",
        tp=1,
        trust_remote_code=False,
        video_fps=2.0,
        video_path=None,
    )

    model = MagicMock()
    model.cuda.return_value = model
    provider = MagicMock()
    provider.provide_distributed_model.return_value = [model]
    bridge = MagicMock()
    bridge.to_megatron_provider.return_value = provider

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 1
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = 0
    tokenizer.decode.return_value = "decoded"

    generation_config = SimpleNamespace(eos_token_id=[1, 2])
    generation_config_cls = MagicMock()
    generation_config_cls.from_pretrained.return_value = generation_config

    forward = MagicMock()

    def run_forward(**kwargs):
        seq_length = kwargs["seq_length"]
        logits = torch.zeros(1, seq_length, 8)
        next_token = 2 if forward.call_count == 1 else 1
        logits[0, seq_length - 1, next_token] = 1
        return [logits]

    forward.side_effect = run_forward

    def all_gather(outputs, tensor, group):
        outputs[0].copy_(tensor)

    script_globals = {
        "AutoBridge": SimpleNamespace(from_hf_pretrained=MagicMock(return_value=bridge)),
        "AutoConfig": SimpleNamespace(
            from_pretrained=MagicMock(return_value=SimpleNamespace(model_type="qwen2_5_vl"))
        ),
        "AutoProcessor": SimpleNamespace(from_pretrained=MagicMock(return_value=MagicMock())),
        "AutoTokenizer": SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer)),
        "GenerationConfig": generation_config_cls,
        "get_forward_backward_func": MagicMock(return_value=forward),
        "get_last_rank": MagicMock(return_value=0),
        "is_safe_repo": MagicMock(return_value=False),
        "pad_input_ids_to_tp_multiple": lambda input_ids, tp_size, pad_token_id=0: input_ids,
        "print_rank_0": MagicMock(),
        "print_rank_last": MagicMock(),
        "process_image_inputs": MagicMock(return_value=(torch.tensor([[3, 4]]), None, None, None, None, None)),
        "to_cuda": lambda value: value,
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

    assert forward.call_count == 1


@pytest.mark.unit
def test_kimi_builds_memory_bounded_prefill_context() -> None:
    input_ids = torch.ones((2, 4), dtype=torch.long)
    inference_context = MagicMock()
    context_factory = MagicMock(return_value=inference_context)

    with patch.dict(_build_inference_context.__globals__, {"StaticInferenceContext": context_factory}):
        result = _build_inference_context(input_ids, is_kimi=True)

    assert result is inference_context
    context_factory.assert_called_once_with(max_batch_size=2, max_sequence_length=4)
    assert inference_context.config.materialize_only_last_token_logits is False


@pytest.mark.unit
def test_non_kimi_vlm_does_not_receive_inference_context() -> None:
    assert _build_inference_context(torch.ones((1, 2), dtype=torch.long), is_kimi=False) is None


@pytest.mark.unit
def test_kimi_checkpoint_load_preserves_memory_bounded_prefill_config() -> None:
    provider = SimpleNamespace(mlp_chunks_for_prefill=4)

    overrides = _checkpoint_load_overrides(
        provider,
        tp=2,
        pp=1,
        ep=48,
        etp=1,
        pp_layout=None,
        is_kimi=True,
    )

    assert overrides["mlp_chunks_for_prefill"] == 4
    assert overrides["tensor_model_parallel_size"] == 2
    assert overrides["expert_model_parallel_size"] == 48
    assert "mlp_chunks_for_prefill" not in _checkpoint_load_overrides(
        provider,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        pp_layout=None,
        is_kimi=False,
    )


@pytest.mark.unit
def test_vlm_forward_step_propagates_inference_context_when_present() -> None:
    inference_context = object()
    batch = {
        "tokens": torch.tensor([[1, 2, 3]]),
        "position_ids": torch.arange(3).unsqueeze(0),
        "attention_mask": None,
        "inference_context": inference_context,
    }
    model = MagicMock(return_value=torch.randn(1, 3, 16))

    _vlm_forward_step(iter([batch]), model)

    assert model.call_args.kwargs["inference_context"] is inference_context


@pytest.mark.unit
def test_last_real_token_logits_ignore_tp_padding() -> None:
    logits = torch.arange(4 * 8, dtype=torch.float32).view(1, 4, 8)

    result = _last_real_token_logits(logits, real_sequence_length=3)

    assert torch.equal(result, logits[:, 2])
