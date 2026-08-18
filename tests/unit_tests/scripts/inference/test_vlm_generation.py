# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Focused tests for the canonical VLM generation entry point."""

import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


_SCRIPT = Path(__file__).parents[4] / "scripts" / "inference" / "vlm_generation.py"
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
    "vlm_generation_utils": types.ModuleType("vlm_generation_utils"),
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
    "decode_generated_tokens",
    "pad_input_ids_to_tp_multiple",
    "patch_kimi_vision_processor",
    "process_image_inputs",
    "process_multi_image_inputs",
    "process_video_inputs",
    "to_cuda",
):
    setattr(_import_stubs["vlm_generation_utils"], name, MagicMock())

with patch.dict(sys.modules, _import_stubs):
    _SCRIPT_GLOBALS = runpy.run_path(_SCRIPT)
_main = _SCRIPT_GLOBALS["main"]


@pytest.mark.unit
def test_generation_stops_on_any_configured_eos_token() -> None:
    """Generation must stop when an alternate checkpoint EOS is selected."""
    args = SimpleNamespace(
        ep=1,
        etp=1,
        hf_model_path="org/qwen-vl",
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

    generation_config_cls = MagicMock()
    generation_config_cls.from_pretrained.return_value = SimpleNamespace(eos_token_id=[1, 2])

    forward = MagicMock()

    def run_forward(**kwargs):
        logits = torch.zeros(1, kwargs["seq_length"], 8)
        next_token = 2 if forward.call_count == 1 else 1
        logits[0, kwargs["seq_length"] - 1, next_token] = 1
        return [logits]

    forward.side_effect = run_forward

    def fake_all_gather(gathered, output, group):
        gathered[0].copy_(output)

    script_globals = {
        "AutoBridge": SimpleNamespace(from_hf_pretrained=MagicMock(return_value=bridge)),
        "AutoConfig": SimpleNamespace(
            from_pretrained=MagicMock(return_value=SimpleNamespace(model_type="qwen2_5_vl"))
        ),
        "AutoProcessor": SimpleNamespace(from_pretrained=MagicMock(return_value=MagicMock())),
        "AutoTokenizer": SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer)),
        "GenerationConfig": generation_config_cls,
        "decode_generated_tokens": MagicMock(return_value="decoded"),
        "get_forward_backward_func": MagicMock(return_value=forward),
        "get_last_rank": MagicMock(return_value=0),
        "is_safe_repo": MagicMock(return_value=False),
        "maybe_initialize_distributed": MagicMock(),
        "pad_input_ids_to_tp_multiple": lambda input_ids, tp_size, pad_token_id=0: input_ids,
        "print_rank_0": MagicMock(),
        "print_rank_last": MagicMock(),
        "process_image_inputs": MagicMock(return_value=(torch.tensor([[3, 4]]), None, None, None, None, None)),
        "to_cuda": lambda value: value,
    }

    with (
        patch.dict(_main.__globals__, script_globals),
        patch.object(torch.Tensor, "cuda", lambda self: self),
        patch.object(torch.distributed, "all_gather", new=fake_all_gather),
        patch.object(torch.distributed, "broadcast"),
        patch.object(_main.__globals__["parallel_state"], "get_tensor_model_parallel_group", return_value=None),
        patch.object(_main.__globals__["parallel_state"], "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(_main.__globals__["parallel_state"], "is_pipeline_last_stage", return_value=True),
    ):
        _main(args)

    assert forward.call_count == 1
