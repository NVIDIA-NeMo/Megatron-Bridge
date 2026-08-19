import gc
import runpy
import sys
import types
import weakref
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
for name in ("AutoConfig", "AutoProcessor", "AutoTokenizer"):
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
    _script_globals = runpy.run_path(_SCRIPT)
_gather_last_token_logits = _script_globals.get("_gather_last_token_logits")
_main = _script_globals["main"]


@pytest.mark.unit
def test_generation_gathers_only_last_real_token_logits() -> None:
    """TP gathering must reconstruct vocabulary logits only for the consumed token."""
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
        tp=2,
        trust_remote_code=False,
        video_fps=2.0,
        video_path=None,
    )

    model = MagicMock()
    model.cuda.return_value = model
    model.config = SimpleNamespace(mtp_num_layers=None, grad_scale_func=None)
    provider = MagicMock()
    provider.provide_distributed_model.return_value = [model]
    bridge = MagicMock()
    bridge.to_megatron_provider.return_value = provider

    tokenizer = MagicMock(eos_token_id=99, eos_token="<eos>", pad_token_id=0, pad_token="<pad>")
    tokenizer.decode.return_value = "decoded"

    output_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    last_token_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    collective_result_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def run_forward(**kwargs):
        gc.collect()
        assert all(ref() is None for ref in output_refs)
        assert all(ref() is None for ref in last_token_refs)
        assert all(ref() is None for ref in collective_result_refs)
        seq_length = kwargs["seq_length"]
        logits = torch.arange(seq_length * 4, dtype=torch.float32).view(1, seq_length, 4)
        output_refs.append(weakref.ref(logits))
        return [logits]

    gathered_shapes: list[torch.Size] = []
    gathered_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def fake_all_gather(gathered, local_logits, group):
        gathered_shapes.append(local_logits.shape)
        gathered_refs.extend(weakref.ref(tensor) for tensor in gathered)
        gathered[0].copy_(local_logits)
        gathered[1].copy_(local_logits + 100)

    def pad_to_tp_multiple(input_ids, tp_size, pad_token_id=0):
        remainder = input_ids.shape[1] % tp_size
        if remainder == 0:
            return input_ids
        padding = torch.full((1, tp_size - remainder), pad_token_id, dtype=input_ids.dtype)
        return torch.cat([input_ids, padding], dim=1)

    original_cat = torch.cat

    def recording_cat(tensors, *args, **kwargs):
        result = original_cat(tensors, *args, **kwargs)
        if tensors and tensors[0].dtype.is_floating_point:
            collective_result_refs.append(weakref.ref(result))
        return result

    def gather_last_token_logits(output, real_seq_len):
        assert _gather_last_token_logits is not None
        last_token_logits = _gather_last_token_logits(output, real_seq_len)
        last_token_refs.append(weakref.ref(last_token_logits))
        return last_token_logits

    selected_tokens: list[int] = []

    def capture_broadcast(tensor, src):
        selected_tokens.append(tensor.item())

    script_globals = {
        "AutoBridge": SimpleNamespace(from_hf_pretrained=MagicMock(return_value=bridge)),
        "AutoConfig": SimpleNamespace(
            from_pretrained=MagicMock(return_value=SimpleNamespace(model_type="qwen2_5_vl"))
        ),
        "AutoProcessor": SimpleNamespace(from_pretrained=MagicMock(return_value=MagicMock())),
        "AutoTokenizer": SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer)),
        "_gather_last_token_logits": gather_last_token_logits,
        "decode_generated_tokens": MagicMock(return_value="decoded"),
        "get_forward_backward_func": MagicMock(return_value=run_forward),
        "get_last_rank": MagicMock(return_value=0),
        "is_safe_repo": MagicMock(return_value=False),
        "maybe_initialize_distributed": MagicMock(),
        "pad_input_ids_to_tp_multiple": pad_to_tp_multiple,
        "print_rank_0": MagicMock(),
        "print_rank_last": MagicMock(),
        "process_image_inputs": MagicMock(return_value=(torch.tensor([[7, 8, 9]]), None, None, None, None, None)),
        "to_cuda": lambda value: value,
    }

    with (
        patch.dict(_main.__globals__, script_globals),
        patch.object(torch.Tensor, "cuda", lambda self: self),
        patch.object(torch, "cat", new=recording_cat),
        patch.object(torch.distributed, "all_gather", new=fake_all_gather),
        patch.object(torch.distributed, "broadcast", new=capture_broadcast),
        patch.object(_main.__globals__["parallel_state"], "is_pipeline_last_stage", return_value=True),
        patch.object(_main.__globals__["parallel_state"], "get_tensor_model_parallel_group", return_value=None),
        patch.object(_main.__globals__["parallel_state"], "get_tensor_model_parallel_world_size", return_value=2),
    ):
        _main(args)

    assert gathered_shapes == [torch.Size([1, 4]), torch.Size([1, 4])]
    assert selected_tokens == [7, 7]
    assert all(ref() is None for ref in gathered_refs)
    assert all(ref() is None for ref in output_refs)
    assert all(ref() is None for ref in last_token_refs)
    assert all(ref() is None for ref in collective_result_refs)
