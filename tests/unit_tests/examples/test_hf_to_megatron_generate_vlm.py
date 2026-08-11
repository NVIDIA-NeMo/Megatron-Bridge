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
import weakref
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
_gather_last_token_logits = _SCRIPT_GLOBALS["_gather_last_token_logits"]
_main = _SCRIPT_GLOBALS["main"]


@pytest.mark.unit
def test_checkpoint_inference_aligns_parameter_dtype_with_pipeline_dtype() -> None:
    """Checkpoint inference must not mix bf16 pipeline inputs with fp32 parameters."""
    args = SimpleNamespace(
        ep=1,
        etp=1,
        hf_model_path="org/gemma4-vl",
        hf_revision="revision",
        megatron_model_path="/checkpoint",
        pp=2,
        pp_layout=None,
        tp=1,
        trust_remote_code=False,
    )

    class StopAfterCheckpointLoad(Exception):
        pass

    provider = MagicMock()
    bridge = MagicMock()
    bridge.to_megatron_provider.return_value = provider
    bridge.load_megatron_model.side_effect = StopAfterCheckpointLoad

    script_globals = {
        "AutoBridge": SimpleNamespace(from_hf_pretrained=MagicMock(return_value=bridge)),
        "AutoConfig": SimpleNamespace(from_pretrained=MagicMock(return_value=SimpleNamespace(model_type="gemma4"))),
        "is_safe_repo": MagicMock(return_value=False),
        "maybe_initialize_distributed": MagicMock(),
        "print_rank_0": MagicMock(),
    }

    with patch.dict(_main.__globals__, script_globals), pytest.raises(StopAfterCheckpointLoad):
        _main(args)

    mp_overrides = bridge.load_megatron_model.call_args.kwargs["mp_overrides"]
    assert mp_overrides["pipeline_dtype"] == torch.bfloat16
    assert mp_overrides["params_dtype"] == mp_overrides["pipeline_dtype"]
    assert mp_overrides["bf16"] is True
    assert mp_overrides["fp16"] is False


@pytest.mark.unit
def test_generation_releases_logits_and_stops_on_any_configured_eos_token() -> None:
    """Generation must release prior logits and stop on an alternate EOS."""
    args = SimpleNamespace(
        ep=1,
        etp=1,
        hf_model_path="org/model",
        hf_revision="revision",
        image_path=None,
        image_paths=None,
        max_new_tokens=3,
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
    gathered_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def run_forward(**kwargs):
        if gathered_refs:
            assert gathered_refs[-1]() is None
        seq_length = kwargs["seq_length"]
        logits = torch.zeros(1, seq_length, 8)
        next_token = 3 if forward.call_count == 1 else 2
        logits[0, seq_length - 1, next_token] = 1
        return [logits]

    forward.side_effect = run_forward

    def gather_last_token_logits(output, real_seq_len):
        last_token_logits = output[:, real_seq_len - 1].clone()
        gathered_refs.append(weakref.ref(last_token_logits))
        return last_token_logits

    script_globals = {
        "AutoBridge": SimpleNamespace(from_hf_pretrained=MagicMock(return_value=bridge)),
        "AutoConfig": SimpleNamespace(
            from_pretrained=MagicMock(return_value=SimpleNamespace(model_type="qwen2_5_vl"))
        ),
        "AutoProcessor": SimpleNamespace(from_pretrained=MagicMock(return_value=MagicMock())),
        "AutoTokenizer": SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer)),
        "GenerationConfig": generation_config_cls,
        "_gather_last_token_logits": gather_last_token_logits,
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
        patch.object(torch.distributed, "broadcast"),
        patch.object(_main.__globals__["parallel_state"], "is_pipeline_last_stage", return_value=True),
    ):
        _main(args)

    assert forward.call_count == 2
    assert all(ref() is None for ref in gathered_refs)


@pytest.mark.unit
def test_generation_gathers_only_last_token_logits() -> None:
    """TP gathering must not retain or replicate full-prefix logits between steps."""
    output = torch.arange(1 * 3 * 4, dtype=torch.float32).view(1, 3, 4)
    output_ref = weakref.ref(output)
    gathered_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    gathered_shapes: list[torch.Size] = []

    def fake_all_gather(gathered, local_logits, group):
        gathered_shapes.append(local_logits.shape)
        gathered_refs.extend(weakref.ref(tensor) for tensor in gathered)
        gathered[0].copy_(local_logits)
        gathered[1].copy_(local_logits + 100)

    with (
        patch.object(torch.distributed, "all_gather", new=fake_all_gather),
        patch.object(
            _gather_last_token_logits.__globals__["parallel_state"],
            "get_tensor_model_parallel_group",
            return_value=None,
        ),
        patch.object(
            _gather_last_token_logits.__globals__["parallel_state"],
            "get_tensor_model_parallel_world_size",
            return_value=2,
        ),
    ):
        last_token_logits = _gather_last_token_logits(output, real_seq_len=3)

    assert gathered_shapes == [torch.Size([1, 4])]
    assert torch.equal(last_token_logits, torch.tensor([[8, 9, 10, 11, 108, 109, 110, 111]]))
    assert all(ref() is None for ref in gathered_refs)

    del output
    assert output_ref() is None
    assert torch.equal(last_token_logits[:, :4], torch.tensor([[8, 9, 10, 11]]))
