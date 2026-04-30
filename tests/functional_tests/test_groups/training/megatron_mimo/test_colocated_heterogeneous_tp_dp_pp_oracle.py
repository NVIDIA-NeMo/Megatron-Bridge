# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Colocated MegatronMIMO heterogeneous TP/DP/PP oracle.

This 8-GPU functional oracle checks the language-PP three-phase schedule
against a colocated PP=1 reference:

* Dist: vision(tp=2, dp=4, pp=1) x language(tp=2, dp=2, pp=2)
* Ref:  vision(tp=2, dp=4, pp=1) x language(tp=2, dp=4, pp=1)

The dist side uses the real Bridge setup path, DDP wrapping, the colocated
language-PP adapter, and ``finalize_model_grads_multimodule``. The ref side uses
the same Bridge setup path with matching TP/DP layouts but no language PP, then
runs the standard no-pipeline schedule. Both sides use force-all-reduce so the
rank-local gradient tensors are directly comparable.

The assertions compare finalized encoder gradients and the rank-local language
PP shard's gradients. Encoder gradients are the most direct end-to-end signal
for this schedule: the loss flows through language PP, broadcasts back to the
detached encoder embeddings, runs encoder backward, then finalizes the encoder
DDP grads with the same global token scale as the reference.

The file also contains a focused 2-GPU force-all-reduce regression for the same
colocated language-PP adapter. That test protects the path where
``overlap_grad_reduce=False`` requires finalization to perform a synchronous
all-reduce.

Run:
    torchrun --nproc_per_node=8 -m pytest -v -s -x \\
        tests/functional_tests/test_groups/training/megatron_mimo/test_colocated_heterogeneous_tp_dp_pp_oracle.py
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterator


os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")

import pytest
import torch
import torch.distributed as dist
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_local_spec
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_model_config

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    megatron_mimo_runtime_config_update,
)
from megatron.bridge.training.megatron_mimo_parallel_utils import (
    get_active_module_pg,
    unwrap_megatron_mimo_model,
    zero_grad_buffer_for_multimodule,
)
from megatron.bridge.training.megatron_mimo_step import (
    forward_backward_colocated_mimo_with_pp,
)
from megatron.bridge.training.megatron_mimo_step import (
    forward_step as megatron_mimo_forward_step,
)
from megatron.bridge.training.setup_megatron_mimo import setup_megatron_mimo
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.utils.train_utils import prepare_forward_step_func
from tests.functional_tests.utils import initialize_distributed


_HIDDEN_SIZE = 64
_FFN_HIDDEN_SIZE = 256
_NUM_HEADS = 4
_NUM_LAYERS = 2
_VOCAB_SIZE = 1000
_SEQ_LENGTH = 32
_IMG_SIZE = 32
_PATCH_DIM = 16
_ENCODER_SEQ_LEN = (_IMG_SIZE // _PATCH_DIM) ** 2 + 1
_SPECIAL_TOKEN_ID = 999
_MICRO_BATCH_SIZE = 4
_NUM_MICROBATCHES = 2
_GLOBAL_BATCH_SIZE = _MICRO_BATCH_SIZE * _NUM_MICROBATCHES
_LANGUAGE_PP_SIZE = 2

_LANGUAGE = "language"
_VISION = "vision"
_VISION_ENCODER = "clip"
_LANGUAGE_LAYER_RE = re.compile(r"^(language_model\.decoder\.layers\.)(\d+)(\..*)$")


def _make_vision_config() -> TransformerConfig:
    cfg = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        pipeline_dtype=torch.float32,
        bf16=False,
        fp16=False,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        calculate_per_token_loss=True,
    )
    cfg.add_bias_linear = False
    cfg.add_qkv_bias = False
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.gated_linear_unit = False
    cfg.layernorm_zero_centered_gamma = False
    cfg.apply_query_key_layer_scaling = False
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.normalization = "LayerNorm"
    cfg.apply_rope_fusion = False
    return cfg


def _make_language_config() -> TransformerConfig:
    cfg = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        pipeline_dtype=torch.float32,
        bf16=False,
        fp16=False,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        calculate_per_token_loss=True,
    )
    cfg.hidden_dropout = 0.0
    cfg.attention_dropout = 0.0
    cfg.bias_activation_fusion = False
    cfg.bias_dropout_fusion = False
    cfg.attention_softmax_in_fp32 = True
    cfg.apply_rope_fusion = False
    return cfg


def _build_model_specs() -> tuple[ModuleSpec, dict[str, ModuleSpec], dict[str, int]]:
    vision_encoder = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": _make_vision_config(),
            "transformer_layer_spec": get_vit_layer_with_local_spec(),
            "patch_dim": _PATCH_DIM,
            "img_h": _IMG_SIZE,
            "img_w": _IMG_SIZE,
        },
    )
    vision_submodule = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={"encoders": {_VISION_ENCODER: vision_encoder}},
    )
    language_model = ModuleSpec(
        module=GPTModel,
        params={
            "config": _make_language_config(),
            "transformer_layer_spec": get_gpt_layer_local_spec(),
            "vocab_size": _VOCAB_SIZE,
            "max_sequence_length": _SEQ_LENGTH,
        },
    )
    return language_model, {_VISION: vision_submodule}, {_VISION: _SPECIAL_TOKEN_ID}


def _build_provider(par_cfg: MegatronMIMOParallelismConfig) -> MegatronMIMOProvider:
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs()
    return MegatronMIMOProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        megatron_mimo_parallelism_config=par_cfg,
        topology={_VISION: [_LANGUAGE], _LANGUAGE: []},
        bf16=False,
    )


def _build_dist_config(par_cfg: MegatronMIMOParallelismConfig) -> ConfigContainer:
    train_cfg = TrainingConfig(
        micro_batch_size=_MICRO_BATCH_SIZE,
        global_batch_size=_GLOBAL_BATCH_SIZE,
        train_iters=1,
    )
    train_cfg.num_microbatches = _NUM_MICROBATCHES

    return ConfigContainer(
        train=train_cfg,
        model=_build_provider(par_cfg),
        optimizer=OptimizerConfig(
            bf16=False,
            fp16=False,
            use_distributed_optimizer=True,
            lr=1e-4,
            min_lr=0.0,
        ),
        scheduler=SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0),
        dataset=_UnusedDataProvider(),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(),
        checkpoint=CheckpointConfig(save_rng=False),
    )


class _UnusedDataProvider:
    """Placeholder dataset config; the oracle injects its own batch iterator."""


def _build_data_iterators_fn(
    _cfg: ConfigContainer, _infra: Any, *, train_state: Any | None = None
) -> tuple[Iterator, None]:
    del train_state
    return iter([]), None


def _generate_global_microbatch(seed: int, *, microbatch_index: int) -> dict[str, Any]:
    if dist.get_rank() == 0:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        text_len = _SEQ_LENGTH - _ENCODER_SEQ_LEN
        image_tokens = torch.full((_MICRO_BATCH_SIZE, _ENCODER_SEQ_LEN), _SPECIAL_TOKEN_ID, dtype=torch.long)
        text_tokens = torch.randint(1, _SPECIAL_TOKEN_ID, (_MICRO_BATCH_SIZE, text_len), generator=gen)
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)

        labels = input_ids.clone()
        labels[input_ids == _SPECIAL_TOKEN_ID] = -100

        loss_mask = torch.zeros(_MICRO_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.float32)
        mask_starts = [0, text_len // 3, 2 * text_len // 3, text_len // 2]
        for sample_idx in range(_MICRO_BATCH_SIZE):
            start = mask_starts[(sample_idx + microbatch_index) % len(mask_starts)]
            loss_mask[sample_idx, _ENCODER_SEQ_LEN + start :] = 1.0

        position_ids = torch.arange(_SEQ_LENGTH).unsqueeze(0).expand(_MICRO_BATCH_SIZE, -1).contiguous()
        pixel_values = torch.randn(_MICRO_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE, generator=gen)
    else:
        input_ids = torch.empty(_MICRO_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.long)
        labels = torch.empty(_MICRO_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.long)
        loss_mask = torch.empty(_MICRO_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.float32)
        position_ids = torch.empty(_MICRO_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.long)
        pixel_values = torch.empty(_MICRO_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE, dtype=torch.float32)

    for tensor in (input_ids, labels, loss_mask, position_ids, pixel_values):
        cuda_tensor = tensor.cuda(non_blocking=False)
        dist.broadcast(cuda_tensor, src=0)
        tensor.copy_(cuda_tensor.cpu())

    device = torch.cuda.current_device()
    return {
        "input_ids": input_ids.to(device, non_blocking=False),
        "labels": labels.to(device, non_blocking=False),
        "loss_mask": loss_mask.to(device, non_blocking=False),
        "position_ids": position_ids.to(device, non_blocking=False),
        "modality_inputs": {
            _VISION: {_VISION_ENCODER: {"x": pixel_values.to(device, non_blocking=False)}},
        },
        "attention_mask": None,
    }


def _generate_global_microbatches(seed: int) -> list[dict[str, Any]]:
    return [
        _generate_global_microbatch(seed + microbatch_index, microbatch_index=microbatch_index)
        for microbatch_index in range(_NUM_MICROBATCHES)
    ]


def _batch_iterator(batches: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for batch in batches:
        yield {
            **batch,
            "modality_inputs": {
                _VISION: {
                    _VISION_ENCODER: {
                        **batch["modality_inputs"][_VISION][_VISION_ENCODER],
                    },
                },
            },
        }


def _pg_rank(pg: dist.ProcessGroup) -> int:
    return pg.rank() if hasattr(pg, "rank") else dist.get_rank(pg)


def _pg_size(pg: dist.ProcessGroup) -> int:
    return pg.size() if hasattr(pg, "size") else dist.get_world_size(pg)


def _install_global_parallel_state_for_mimo(infra: Any) -> None:
    """Mirror setup's canonical MIMO PG bridge before running a schedule.

    The oracle builds a dist model and a reference model in one process. Each
    setup installs its own canonical PGs into Megatron's global parallel_state,
    so reinstall the matching canonical PGs before running each schedule. The
    model/finalizer still use injected per-module pg_collections; this prevents
    accidental global-parallel_state reads from seeing the other setup's layout.
    This is private MCore state surgery because MCore has no public setter for
    this bridge; keep it in sync if ``parallel_state`` is refactored.
    """
    from megatron.core import parallel_state as mpu

    _, local_pg_collection = get_active_module_pg(infra)
    data_parallel_with_cp_group = getattr(local_pg_collection, "dp_cp", local_pg_collection.dp)
    context_parallel_group = getattr(local_pg_collection, "cp", None)

    mpu._TENSOR_MODEL_PARALLEL_GROUP = local_pg_collection.tp
    mpu._DATA_PARALLEL_GROUP = local_pg_collection.dp
    mpu._DATA_PARALLEL_GROUP_WITH_CP = data_parallel_with_cp_group
    mpu._PIPELINE_MODEL_PARALLEL_GROUP = local_pg_collection.pp
    mpu._CONTEXT_PARALLEL_GROUP = context_parallel_group

    # This oracle does not cover EP, VPP, or tensor+data composite globals.
    # Clear them so stale state from the other setup fails loudly if used.
    for name in (
        "_TENSOR_AND_DATA_PARALLEL_GROUP",
        "_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP",
        "_EXPERT_MODEL_PARALLEL_GROUP",
        "_EXPERT_TENSOR_PARALLEL_GROUP",
        "_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP",
        "_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP",
        "_EXPERT_DATA_PARALLEL_GROUP",
        "_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK",
        "_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE",
    ):
        setattr(mpu, name, None)

    assert mpu.get_tensor_model_parallel_group() is local_pg_collection.tp
    assert mpu.get_data_parallel_group() is local_pg_collection.dp
    assert mpu.get_data_parallel_group(with_context_parallel=True) is data_parallel_with_cp_group
    assert mpu.get_pipeline_model_parallel_group() is local_pg_collection.pp
    assert mpu.get_context_parallel_group(check_initialized=False) is context_parallel_group
    assert mpu.get_tensor_and_data_parallel_group(check_initialized=False) is None
    assert mpu.get_tensor_and_data_parallel_group(check_initialized=False, with_context_parallel=True) is None
    assert mpu.get_expert_model_parallel_group(check_initialized=False) is None
    assert mpu.get_virtual_pipeline_model_parallel_rank() is None


def _param_module_kind(name: str) -> str:
    if name.startswith("language_model."):
        return _LANGUAGE
    if name.startswith("modality_submodules."):
        return _VISION
    raise AssertionError(f"Cannot classify parameter name {name!r}")


def _named_params_unwrapped(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    inner = unwrap_megatron_mimo_model(model)
    params: dict[str, torch.nn.Parameter] = {}
    for name, param in inner.named_parameters():
        normalized = name
        while ".module." in normalized:
            normalized = normalized.replace(".module.", ".")
        params[normalized] = param
    return params


def _language_ref_name_for_dist_name(name: str, *, language_pp_rank: int) -> str:
    match = _LANGUAGE_LAYER_RE.match(name)
    if match is None:
        return name
    layers_per_pp_stage = _NUM_LAYERS // _LANGUAGE_PP_SIZE
    ref_layer = language_pp_rank * layers_per_pp_stage + int(match.group(2))
    return f"{match.group(1)}{ref_layer}{match.group(3)}"


def _copy_ref_weights_to_dist(
    ref_model: torch.nn.Module,
    dist_model: torch.nn.Module,
    *,
    language_pp_rank: int,
) -> None:
    ref_params = _named_params_unwrapped(ref_model)
    dist_params = _named_params_unwrapped(dist_model)

    with torch.no_grad():
        for dist_name, dist_param in dist_params.items():
            ref_name = (
                _language_ref_name_for_dist_name(dist_name, language_pp_rank=language_pp_rank)
                if _param_module_kind(dist_name) == _LANGUAGE
                else dist_name
            )
            assert ref_name in ref_params, f"Missing reference parameter for dist param {dist_name!r}: {ref_name!r}"
            ref_param = ref_params[ref_name]
            assert ref_param.shape == dist_param.shape, (
                f"Shape mismatch copying {ref_name} -> {dist_name}: "
                f"ref={tuple(ref_param.shape)}, dist={tuple(dist_param.shape)}"
            )
            dist_param.data.copy_(ref_param.data.to(device=dist_param.device, dtype=dist_param.dtype))


def _dist_grad(param: torch.nn.Parameter) -> torch.Tensor | None:
    grad = getattr(param, "main_grad", None)
    return grad if grad is not None else param.grad


def _assert_encoder_grads_match(
    ref_model: torch.nn.Module,
    dist_model: torch.nn.Module,
    *,
    rtol: float = 1e-3,
    atol: float = 3e-3,
) -> None:
    ref_params = _named_params_unwrapped(ref_model)
    dist_params = _named_params_unwrapped(dist_model)

    mismatches: list[str] = []
    one_sided: list[str] = []
    compared = 0
    local_ref_sq = torch.tensor(0.0, device=torch.cuda.current_device())
    local_diff_sq = torch.tensor(0.0, device=torch.cuda.current_device())
    local_ref_max = torch.tensor(0.0, device=torch.cuda.current_device())
    local_diff_max = torch.tensor(0.0, device=torch.cuda.current_device())

    for name, ref_param in ref_params.items():
        if _param_module_kind(name) != _VISION:
            continue
        if name not in dist_params:
            continue

        ref_grad = _dist_grad(ref_param)
        dist_grad = _dist_grad(dist_params[name])
        if ref_grad is None and dist_grad is None:
            continue
        if (ref_grad is None) != (dist_grad is None):
            one_sided.append(name)
            continue

        assert ref_grad is not None
        assert dist_grad is not None
        ref_compare = ref_grad.detach().float().to(dist_grad.device)
        dist_compare = dist_grad.detach().float()
        assert ref_compare.shape == dist_compare.shape, (
            f"Encoder grad shape mismatch for {name}: ref={tuple(ref_compare.shape)}, dist={tuple(dist_compare.shape)}"
        )

        diff = dist_compare - ref_compare
        local_ref_sq += torch.sum(ref_compare.square())
        local_diff_sq += torch.sum(diff.square())
        local_ref_max = torch.maximum(local_ref_max, ref_compare.abs().max())
        local_diff_max = torch.maximum(local_diff_max, diff.abs().max())
        compared += 1

        if not torch.allclose(dist_compare, ref_compare, rtol=rtol, atol=atol):
            mismatches.append(
                f"{name}: max_abs_diff={diff.abs().max().item():.3e}, "
                f"mean_abs_diff={diff.abs().mean().item():.3e}, "
                f"ref_max={ref_compare.abs().max().item():.3e}, "
                f"dist_max={dist_compare.abs().max().item():.3e}"
            )

    assert compared > 0, "No encoder gradients were compared"
    assert not one_sided, "One-sided encoder gradients: " + ", ".join(one_sided[:10])

    for tensor, op in (
        (local_ref_sq, dist.ReduceOp.SUM),
        (local_diff_sq, dist.ReduceOp.SUM),
        (local_ref_max, dist.ReduceOp.MAX),
        (local_diff_max, dist.ReduceOp.MAX),
    ):
        dist.all_reduce(tensor, op=op, group=dist.group.WORLD)

    ref_l2 = torch.sqrt(local_ref_sq).item()
    diff_l2 = torch.sqrt(local_diff_sq).item()
    rel_l2 = diff_l2 / max(ref_l2, 1e-12)
    assert ref_l2 > 1e-8, (
        f"Encoder gradient signal is too small for a schedule oracle: "
        f"ref_l2={ref_l2:.3e}, ref_max={local_ref_max.item():.3e}"
    )
    assert not mismatches, (
        f"Encoder gradient mismatch in {len(mismatches)} of {compared} params "
        f"(global rel_l2={rel_l2:.3e}, diff_l2={diff_l2:.3e}, "
        f"max_abs_diff={local_diff_max.item():.3e}, ref_max={local_ref_max.item():.3e}):\n  "
        + "\n  ".join(mismatches[:10])
    )


def _assert_language_grads_match(
    ref_model: torch.nn.Module,
    dist_model: torch.nn.Module,
    *,
    language_pp_rank: int,
    rtol: float = 1e-3,
    atol: float = 3e-3,
) -> None:
    ref_params = _named_params_unwrapped(ref_model)
    dist_params = _named_params_unwrapped(dist_model)

    mismatches: list[str] = []
    one_sided: list[str] = []
    compared = 0
    local_ref_sq = torch.tensor(0.0, device=torch.cuda.current_device())
    local_diff_sq = torch.tensor(0.0, device=torch.cuda.current_device())
    local_ref_max = torch.tensor(0.0, device=torch.cuda.current_device())
    local_diff_max = torch.tensor(0.0, device=torch.cuda.current_device())

    for dist_name, dist_param in dist_params.items():
        if _param_module_kind(dist_name) != _LANGUAGE:
            continue

        ref_name = _language_ref_name_for_dist_name(dist_name, language_pp_rank=language_pp_rank)
        assert ref_name in ref_params, f"Missing reference parameter for dist param {dist_name!r}: {ref_name!r}"
        ref_grad = _dist_grad(ref_params[ref_name])
        dist_grad = _dist_grad(dist_param)
        if ref_grad is None and dist_grad is None:
            continue
        if (ref_grad is None) != (dist_grad is None):
            one_sided.append(f"{dist_name} -> {ref_name}")
            continue

        assert ref_grad is not None
        assert dist_grad is not None
        ref_compare = ref_grad.detach().float().to(dist_grad.device)
        dist_compare = dist_grad.detach().float()
        assert ref_compare.shape == dist_compare.shape, (
            f"Language grad shape mismatch for {dist_name} -> {ref_name}: "
            f"ref={tuple(ref_compare.shape)}, dist={tuple(dist_compare.shape)}"
        )

        diff = dist_compare - ref_compare
        local_ref_sq += torch.sum(ref_compare.square())
        local_diff_sq += torch.sum(diff.square())
        local_ref_max = torch.maximum(local_ref_max, ref_compare.abs().max())
        local_diff_max = torch.maximum(local_diff_max, diff.abs().max())
        compared += 1

        if not torch.allclose(dist_compare, ref_compare, rtol=rtol, atol=atol):
            mismatches.append(
                f"{dist_name} -> {ref_name}: max_abs_diff={diff.abs().max().item():.3e}, "
                f"mean_abs_diff={diff.abs().mean().item():.3e}, "
                f"ref_max={ref_compare.abs().max().item():.3e}, "
                f"dist_max={dist_compare.abs().max().item():.3e}"
            )

    assert compared > 0, "No language gradients were compared"
    assert not one_sided, "One-sided language gradients: " + ", ".join(one_sided[:10])

    for tensor, op in (
        (local_ref_sq, dist.ReduceOp.SUM),
        (local_diff_sq, dist.ReduceOp.SUM),
        (local_ref_max, dist.ReduceOp.MAX),
        (local_diff_max, dist.ReduceOp.MAX),
    ):
        dist.all_reduce(tensor, op=op, group=dist.group.WORLD)

    ref_l2 = torch.sqrt(local_ref_sq).item()
    diff_l2 = torch.sqrt(local_diff_sq).item()
    rel_l2 = diff_l2 / max(ref_l2, 1e-12)
    assert ref_l2 > 1e-8, (
        f"Language gradient signal is too small for a schedule oracle: "
        f"ref_l2={ref_l2:.3e}, ref_max={local_ref_max.item():.3e}"
    )
    assert not mismatches, (
        f"Language gradient mismatch in {len(mismatches)} of {compared} params "
        f"(global rel_l2={rel_l2:.3e}, diff_l2={diff_l2:.3e}, "
        f"max_abs_diff={local_diff_max.item():.3e}, ref_max={local_ref_max.item():.3e}):\n  "
        + "\n  ".join(mismatches[:10])
    )


def _flatten_module_grads(model: torch.nn.Module, module_kind: str) -> torch.Tensor:
    grads: list[torch.Tensor] = []
    for name, param in _named_params_unwrapped(model).items():
        if _param_module_kind(name) != module_kind:
            continue
        grad = _dist_grad(param)
        if grad is not None:
            grads.append(grad.detach().float().reshape(-1))
    assert grads, f"No gradients found for module kind {module_kind!r}"
    return torch.cat(grads)


def _assert_module_dp_grads_synchronized(
    model: torch.nn.Module,
    *,
    module_kind: str,
    dp_group: dist.ProcessGroup,
    atol: float = 1e-6,
) -> None:
    local_grads = _flatten_module_grads(model, module_kind)
    local_norm = torch.linalg.vector_norm(local_grads)
    assert local_norm.item() > 1e-8, f"{module_kind} gradient signal is too small for force-all-reduce check"

    gathered = [torch.empty_like(local_grads) for _ in range(_pg_size(dp_group))]
    dist.all_gather(gathered, local_grads, group=dp_group)
    reference = gathered[0]
    max_diff = torch.tensor(0.0, device=local_grads.device)
    for other in gathered[1:]:
        max_diff = torch.maximum(max_diff, torch.max(torch.abs(other - reference)))

    assert max_diff.item() <= atol, (
        f"{module_kind} gradients are not synchronized across DP group after force_all_reduce=True: "
        f"max_abs_diff={max_diff.item():.3e}, local_norm={local_norm.item():.3e}"
    )


class TestColocatedHeterogeneousTpDpPpOracle:
    """Functional oracle for colocated heterogeneous TP/DP/PP on exactly 8 GPUs."""

    @pytest.mark.run_only_on("GPU")
    def test_language_pp_dist_matches_pp1_reference_grads(self) -> None:
        initialize_distributed()
        if dist.get_world_size() != 8:
            pytest.skip(f"Requires exactly 8 GPUs, got {dist.get_world_size()}")

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)

        import megatron.bridge.training.utils.train_utils as train_utils

        train_utils.report_theoretical_memory = lambda *args, **kwargs: None

        from megatron.core import parallel_state

        if parallel_state._GLOBAL_MEMORY_BUFFER is None:
            parallel_state._set_global_memory_buffer()

        dist_par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                _LANGUAGE: ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
                _VISION: ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=4,
                    rank_offset=0,
                ),
            },
        )
        ref_par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                _LANGUAGE: ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=4,
                    rank_offset=0,
                ),
                _VISION: ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=4,
                    rank_offset=0,
                ),
            },
        )

        dist_cfg = _build_dist_config(dist_par_cfg)
        megatron_mimo_runtime_config_update(dist_cfg)
        dist_state = GlobalState()
        dist_state.cfg = dist_cfg
        dist_setup = setup_megatron_mimo(
            state=dist_state,
            build_data_iterators_fn=_build_data_iterators_fn,
        )

        ref_cfg = _build_dist_config(ref_par_cfg)
        megatron_mimo_runtime_config_update(ref_cfg)
        ref_state = GlobalState()
        ref_state.cfg = ref_cfg
        ref_setup = setup_megatron_mimo(
            state=ref_state,
            build_data_iterators_fn=_build_data_iterators_fn,
        )

        ref_infra = ref_setup.megatron_mimo_infra
        dist_infra = dist_setup.megatron_mimo_infra
        language_pg = dist_infra.pg_collections[_LANGUAGE]
        assert language_pg is not None
        assert _pg_size(language_pg.tp) == 2
        assert _pg_size(language_pg.pp) == _LANGUAGE_PP_SIZE
        assert _pg_size(language_pg.dp) == 2

        language_pp_rank = _pg_rank(language_pg.pp)
        _copy_ref_weights_to_dist(
            ref_setup.model,
            dist_setup.model,
            language_pp_rank=language_pp_rank,
        )

        batches = _generate_global_microbatches(seed=67890)

        _install_global_parallel_state_for_mimo(dist_infra)
        zero_grad_buffer_for_multimodule(dist_setup.module_to_grid_tuple)
        p2p_communicator = P2PCommunicator(pp_group=language_pg.pp, config=get_model_config(dist_setup.model))
        forward_backward_colocated_mimo_with_pp(
            model=dist_setup.model,
            data_iterator=_batch_iterator(batches),
            infra=dist_infra,
            encoder_module_name=_VISION,
            num_microbatches=_NUM_MICROBATCHES,
            seq_length=_SEQ_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
            forward_only=False,
            p2p_communicator=p2p_communicator,
            force_all_reduce=True,
        )

        _install_global_parallel_state_for_mimo(ref_infra)
        zero_grad_buffer_for_multimodule(ref_setup.module_to_grid_tuple)
        ref_forward_step = prepare_forward_step_func(megatron_mimo_forward_step, ref_state)
        forward_backward_no_pipelining(
            forward_step_func=ref_forward_step,
            data_iterator=_batch_iterator(batches),
            model=[ref_setup.model],
            num_microbatches=_NUM_MICROBATCHES,
            seq_length=_SEQ_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
            forward_only=False,
            pg_collection=ref_infra.pg_collections[_LANGUAGE],
            force_all_reduce=True,
        )

        _assert_encoder_grads_match(ref_setup.model, dist_setup.model)
        _assert_language_grads_match(
            ref_setup.model,
            dist_setup.model,
            language_pp_rank=language_pp_rank,
        )


class TestColocatedPpForceAllReduceRegression:
    """Functional regression for force_all_reduce in colocated language PP."""

    @pytest.mark.run_only_on("GPU")
    def test_force_all_reduce_syncs_encoder_dp_grads_when_overlap_disabled(self) -> None:
        initialize_distributed()
        if dist.get_world_size() != 2:
            pytest.skip(f"Requires exactly 2 GPUs, got {dist.get_world_size()}")

        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(23456)
        torch.cuda.manual_seed_all(23456)

        import megatron.bridge.training.utils.train_utils as train_utils

        train_utils.report_theoretical_memory = lambda *args, **kwargs: None

        from megatron.core import parallel_state

        if parallel_state._GLOBAL_MEMORY_BUFFER is None:
            parallel_state._set_global_memory_buffer()

        par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                _LANGUAGE: ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=2,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                _VISION: ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
            },
        )
        cfg = _build_dist_config(par_cfg)
        cfg.ddp.use_distributed_optimizer = True
        cfg.ddp.overlap_grad_reduce = False
        megatron_mimo_runtime_config_update(cfg)
        state = GlobalState()
        state.cfg = cfg
        setup = setup_megatron_mimo(
            state=state,
            build_data_iterators_fn=_build_data_iterators_fn,
        )

        infra = setup.megatron_mimo_infra
        language_pg = infra.pg_collections[_LANGUAGE]
        vision_pg = infra.pg_collections[_VISION]
        assert language_pg is not None
        assert vision_pg is not None
        assert _pg_size(language_pg.pp) == 2
        assert _pg_size(vision_pg.dp) == 2

        batches = _generate_global_microbatches(seed=98765)

        _install_global_parallel_state_for_mimo(infra)
        zero_grad_buffer_for_multimodule(setup.module_to_grid_tuple)
        p2p_communicator = P2PCommunicator(pp_group=language_pg.pp, config=get_model_config(setup.model))
        forward_backward_colocated_mimo_with_pp(
            model=setup.model,
            data_iterator=_batch_iterator(batches),
            infra=infra,
            encoder_module_name=_VISION,
            num_microbatches=_NUM_MICROBATCHES,
            seq_length=_SEQ_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
            forward_only=False,
            p2p_communicator=p2p_communicator,
            force_all_reduce=True,
        )

        _assert_module_dp_grads_synchronized(
            setup.model,
            module_kind=_VISION,
            dp_group=vision_pg.dp,
        )
