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
"""Colocated MegatronMIMO heterogeneous parallelism oracles.

This file contains two small 2-GPU functional oracles:

* Heterogeneous TP/DP:
  ``vision(tp=1, dp=2) x language(tp=2, dp=1)`` compared to
  ``vision(tp=1, dp=2) x language(tp=1, dp=2)`` at encoder-gradient level.
* Language CP:
  ``vision(tp=1, dp=2) x language(cp=2, dp=1)`` compared to
  ``vision(tp=1, dp=2) x language(cp=1, dp=2)`` at post-step encoder weights.

Both references keep the same encoder TP/DP layout as dist, so encoder shards
and per-rank image batches line up 1:1. The dist side uses the real Megatron
Bridge setup path, DDP wrapping, colocated forward step, and
``finalize_model_grads_multimodule``.

Run:
    torchrun --nproc_per_node=2 -m pytest -v -s -x \\
        tests/functional_tests/test_groups/training/megatron_mimo/test_colocated_heterogeneous_tp_dp_oracle.py
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Iterator


os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
os.environ.pop("NVTE_FLASH_ATTN", None)
os.environ.pop("NVTE_FUSED_ATTN", None)
os.environ.pop("NVTE_UNFUSED_ATTN", None)

import pytest
import torch
import torch.distributed as dist
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_local_spec
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_model_config

from megatron.bridge.data.megatron_mimo.dp_utils import slice_batch_for_megatron_mimo_modules
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
    unwrap_megatron_mimo_model,
    zero_grad_buffer_for_multimodule,
)
from megatron.bridge.training.megatron_mimo_step import forward_step as megatron_mimo_forward_step
from megatron.bridge.training.setup_megatron_mimo import _update_megatron_mimo_model_config_funcs, setup_megatron_mimo
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
_GLOBAL_BATCH_SIZE = 2
_MICRO_BATCH_SIZE = 2

_LANGUAGE = "language"
_VISION = "vision"
_VISION_ENCODER = "clip"


def _set_torch_deterministic() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)


def _make_vision_config(*, bf16: bool = False) -> TransformerConfig:
    cfg = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
        fp16=False,
        variable_seq_lengths=True,
        moe_token_dispatcher_type="alltoall",
        calculate_per_token_loss=True,
    )
    # Avoid bias paths to keep TP-layout drift smaller than the
    # gradient-scaling signal this oracle checks.
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


def _make_language_config(*, bf16: bool = False) -> TransformerConfig:
    cfg = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_HIDDEN_SIZE,
        ffn_hidden_size=_FFN_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        pipeline_dtype=torch.bfloat16 if bf16 else torch.float32,
        bf16=bf16,
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


def _build_model_specs(
    *,
    bf16: bool = False,
    use_te_language: bool = False,
) -> tuple[ModuleSpec, dict[str, ModuleSpec], dict[str, int]]:
    vision_encoder = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": _make_vision_config(bf16=bf16),
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
            "config": _make_language_config(bf16=bf16),
            "transformer_layer_spec": (
                get_gpt_layer_with_transformer_engine_spec()
                if use_te_language
                else get_gpt_layer_local_spec()
            ),
            "vocab_size": _VOCAB_SIZE,
            "max_sequence_length": _SEQ_LENGTH,
        },
    )
    return language_model, {_VISION: vision_submodule}, {_VISION: _SPECIAL_TOKEN_ID}


def _build_provider(
    par_cfg: MegatronMIMOParallelismConfig,
    *,
    bf16: bool = False,
    use_te_language: bool = False,
) -> MegatronMIMOProvider:
    language_model_spec, modality_submodules_spec, special_token_ids = _build_model_specs(
        bf16=bf16,
        use_te_language=use_te_language,
    )
    provider = MegatronMIMOProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        megatron_mimo_parallelism_config=par_cfg,
        topology={_VISION: [_LANGUAGE], _LANGUAGE: []},
        bf16=bf16,
    )
    return provider


def _build_dist_config(
    par_cfg: MegatronMIMOParallelismConfig,
    *,
    bf16: bool = False,
    overlap_grad_reduce: bool = False,
    use_te_language: bool = False,
) -> ConfigContainer:
    train_cfg = TrainingConfig(
        micro_batch_size=_MICRO_BATCH_SIZE,
        global_batch_size=_GLOBAL_BATCH_SIZE,
        train_iters=1,
    )
    train_cfg.num_microbatches = 1

    cfg = ConfigContainer(
        train=train_cfg,
        model=_build_provider(par_cfg, bf16=bf16, use_te_language=use_te_language),
        optimizer=OptimizerConfig(
            bf16=bf16,
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
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = overlap_grad_reduce
    cfg.ddp.average_in_collective = False
    if hasattr(cfg.ddp, "gradient_reduce_div_factor"):
        cfg.ddp.gradient_reduce_div_factor = 1
    return cfg


class _UnusedDataProvider:
    """Placeholder dataset config; the oracle injects its own batch iterator."""


def _build_ref_model_and_infra(
    par_cfg: MegatronMIMOParallelismConfig, *, bf16: bool = False
) -> tuple[torch.nn.Module, Any]:
    provider = _build_provider(par_cfg, bf16=bf16)
    provider.initialize_model_parallel(seed=1234)
    infra = provider.build_infra()
    model = provider.provide()
    model = model.cuda(torch.cuda.current_device())
    if bf16:
        model = model.bfloat16()
    model.train()
    return model, infra


def _build_data_iterators_fn(
    _cfg: ConfigContainer, _infra: Any, *, train_state: Any | None = None
) -> tuple[Iterator, None]:
    del train_state
    return iter([]), None


def _generate_global_batch(seed: int, *, image_dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    if dist.get_rank() == 0:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        text_len = _SEQ_LENGTH - _ENCODER_SEQ_LEN
        image_tokens = torch.full((_GLOBAL_BATCH_SIZE, _ENCODER_SEQ_LEN), _SPECIAL_TOKEN_ID, dtype=torch.long)
        text_tokens = torch.randint(1, _SPECIAL_TOKEN_ID, (_GLOBAL_BATCH_SIZE, text_len), generator=gen)
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)

        labels = input_ids.clone()
        labels[input_ids == _SPECIAL_TOKEN_ID] = -100

        loss_mask = torch.zeros(_GLOBAL_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.float32)
        loss_mask[0, _ENCODER_SEQ_LEN:] = 1.0
        loss_mask[1, _ENCODER_SEQ_LEN + text_len // 2 :] = 1.0

        position_ids = torch.arange(_SEQ_LENGTH).unsqueeze(0).expand(_GLOBAL_BATCH_SIZE, -1).contiguous()
        pixel_values = torch.randn(_GLOBAL_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE, generator=gen)
    else:
        input_ids = torch.empty(_GLOBAL_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.long)
        labels = torch.empty(_GLOBAL_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.long)
        loss_mask = torch.empty(_GLOBAL_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.float32)
        position_ids = torch.empty(_GLOBAL_BATCH_SIZE, _SEQ_LENGTH, dtype=torch.long)
        pixel_values = torch.empty(_GLOBAL_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE, dtype=torch.float32)

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
            _VISION: {_VISION_ENCODER: {"x": pixel_values.to(device, dtype=image_dtype, non_blocking=False)}},
        },
        "attention_mask": None,
    }


def _batch_iterator(batch: dict[str, Any]) -> Iterator[dict[str, Any]]:
    while True:
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


def _ref_forward_backward_local(ref_model: torch.nn.Module, local_batch: dict[str, Any]) -> None:
    output = ref_model(
        input_ids=local_batch["input_ids"],
        labels=local_batch["labels"],
        loss_mask=local_batch["loss_mask"],
        position_ids=local_batch["position_ids"],
        modality_inputs=local_batch["modality_inputs"],
        attention_mask=None,
    )
    per_token_losses = output[0] if isinstance(output, tuple) else output
    loss_mask = local_batch["loss_mask"].contiguous().view(-1).float()
    local_loss_sum = torch.sum(per_token_losses.float().view(-1) * loss_mask)
    local_loss_sum.backward()


def _pg_rank(pg: dist.ProcessGroup) -> int:
    return pg.rank() if hasattr(pg, "rank") else dist.get_rank(pg)


def _pg_size(pg: dist.ProcessGroup) -> int:
    return pg.size() if hasattr(pg, "size") else dist.get_world_size(pg)


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


def _infer_shard_axis(ref_param: torch.nn.Parameter, dist_param: torch.nn.Parameter, tp_size: int) -> int | None:
    if ref_param.shape == dist_param.shape:
        return None

    partition_dim = getattr(dist_param, "partition_dim", -1)
    if partition_dim is not None and partition_dim >= 0:
        axis = int(partition_dim)
    else:
        diff_axes = [
            idx
            for idx, (ref_dim, dist_dim) in enumerate(zip(ref_param.shape, dist_param.shape))
            if ref_dim != dist_dim
        ]
        if len(diff_axes) != 1:
            raise AssertionError(
                f"Cannot infer TP shard axis: ref={tuple(ref_param.shape)}, dist={tuple(dist_param.shape)}"
            )
        axis = diff_axes[0]

    if len(ref_param.shape) != len(dist_param.shape):
        raise AssertionError(f"Rank mismatch: ref={tuple(ref_param.shape)}, dist={tuple(dist_param.shape)}")
    if ref_param.shape[axis] != dist_param.shape[axis] * tp_size:
        raise AssertionError(
            f"Bad TP shard ratio on axis {axis}: ref={tuple(ref_param.shape)}, "
            f"dist={tuple(dist_param.shape)}, tp_size={tp_size}"
        )
    return axis


def _copy_ref_weights_to_dist(
    ref_model: torch.nn.Module,
    dist_model: torch.nn.Module,
    *,
    language_tp_rank: int,
    language_tp_size: int,
) -> None:
    ref_params = _named_params_unwrapped(ref_model)
    dist_params = _named_params_unwrapped(dist_model)
    assert set(ref_params) == set(dist_params), (
        "Ref/dist parameter names differ:\n"
        f"  ref_only={sorted(set(ref_params) - set(dist_params))[:10]}\n"
        f"  dist_only={sorted(set(dist_params) - set(ref_params))[:10]}"
    )

    with torch.no_grad():
        for name, ref_param in ref_params.items():
            dist_param = dist_params[name]
            if _param_module_kind(name) == _VISION:
                assert ref_param.shape == dist_param.shape, (
                    f"Encoder parameter {name} should have identical shape, "
                    f"got ref={tuple(ref_param.shape)} dist={tuple(dist_param.shape)}"
                )
                dist_param.data.copy_(ref_param.data.to(device=dist_param.device, dtype=dist_param.dtype))
                continue

            shard_axis = _infer_shard_axis(ref_param, dist_param, language_tp_size)
            if shard_axis is None:
                dist_param.data.copy_(ref_param.data.to(device=dist_param.device, dtype=dist_param.dtype))
                continue

            shard_size = ref_param.shape[shard_axis] // language_tp_size
            shard = ref_param.data.narrow(shard_axis, language_tp_rank * shard_size, shard_size).contiguous()
            assert shard.shape == dist_param.shape, (
                f"Language shard shape mismatch for {name}: shard={tuple(shard.shape)}, dist={tuple(dist_param.shape)}"
            )
            dist_param.data.copy_(shard.to(device=dist_param.device, dtype=dist_param.dtype))


def _allreduce_and_scale_ref_grads(
    ref_model: torch.nn.Module,
    *,
    language_dp_group: dist.ProcessGroup,
    vision_dp_group: dist.ProcessGroup,
    global_num_tokens: int,
) -> None:
    scale = 1.0 / max(global_num_tokens, 1)
    for name, param in _named_params_unwrapped(ref_model).items():
        if param.grad is None:
            continue
        group = language_dp_group if _param_module_kind(name) == _LANGUAGE else vision_dp_group
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)
        param.grad.mul_(scale)


def _dist_grad(param: torch.nn.Parameter) -> torch.Tensor | None:
    grad = getattr(param, "main_grad", None)
    return grad if grad is not None else param.grad


def _assert_encoder_grads_match(
    ref_model: torch.nn.Module,
    dist_model: torch.nn.Module,
    *,
    rtol: float = 1e-3,
    atol: float = 3e-3,
    scale_tol: float = 0.15,
) -> None:
    ref_params = _named_params_unwrapped(ref_model)
    dist_params = _named_params_unwrapped(dist_model)
    assert set(ref_params) == set(dist_params), "Ref/dist parameter names changed after setup"

    mismatches: list[str] = []
    one_sided: list[str] = []
    compared = 0
    local_ref_sq = torch.tensor(0.0, device=torch.cuda.current_device())
    local_diff_sq = torch.tensor(0.0, device=torch.cuda.current_device())
    local_dot = torch.tensor(0.0, device=torch.cuda.current_device())
    local_ref_max = torch.tensor(0.0, device=torch.cuda.current_device())
    local_diff_max = torch.tensor(0.0, device=torch.cuda.current_device())

    for name, ref_param in ref_params.items():
        if _param_module_kind(name) != _VISION:
            continue

        ref_grad = ref_param.grad
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
        local_dot += torch.sum(dist_compare * ref_compare)
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
        (local_dot, dist.ReduceOp.SUM),
        (local_ref_max, dist.ReduceOp.MAX),
        (local_diff_max, dist.ReduceOp.MAX),
    ):
        dist.all_reduce(tensor, op=op, group=dist.group.WORLD)

    ref_l2 = torch.sqrt(local_ref_sq).item()
    diff_l2 = torch.sqrt(local_diff_sq).item()
    rel_l2 = diff_l2 / max(ref_l2, 1e-12)
    # TP=1-vs-TP=2 accumulation noise can be mostly orthogonal to the reference
    # gradient in this tiny 2-GPU model, making relative L2 too pessimistic.
    # The scale projection is the load-bearing check for this bug class: pure
    # scaling mistakes move it toward 0.5, 2.0, etc., while orthogonal TP drift
    # has much smaller effect on the projected coefficient.
    projected_scale = (local_dot / torch.clamp(local_ref_sq, min=1e-24)).item()
    assert ref_l2 > 1e-8, (
        f"Encoder gradient signal is too small for a scaling oracle: "
        f"ref_l2={ref_l2:.3e}, ref_max={local_ref_max.item():.3e}"
    )
    assert abs(projected_scale - 1.0) <= scale_tol, (
        f"Encoder gradients have projected scale {projected_scale:.3e} "
        f"(tol={scale_tol:.3e}, rel_l2={rel_l2:.3e}, ref_l2={ref_l2:.3e}, diff_l2={diff_l2:.3e}, "
        f"max_abs_diff={local_diff_max.item():.3e}, ref_max={local_ref_max.item():.3e})"
    )
    assert not mismatches, (
        f"Encoder gradient elementwise mismatch in {len(mismatches)} of {compared} params "
        f"(global rel_l2={rel_l2:.3e}, projected_scale={projected_scale:.3e}):\n  " + "\n  ".join(mismatches[:10])
    )


def _set_global_pg_from_setup(setup_output: Any) -> None:
    from megatron.core import parallel_state as mpu

    pg = setup_output.local_pg_collection
    mpu._TENSOR_MODEL_PARALLEL_GROUP = pg.tp
    mpu._DATA_PARALLEL_GROUP = pg.dp
    mpu._DATA_PARALLEL_GROUP_WITH_CP = getattr(pg, "dp_cp", pg.dp)
    if hasattr(pg, "pp"):
        mpu._PIPELINE_MODEL_PARALLEL_GROUP = pg.pp
    if getattr(pg, "cp", None) is not None:
        mpu._CONTEXT_PARALLEL_GROUP = pg.cp


def _rebuild_optimizer_after_weight_copy(setup_output: Any, cfg: ConfigContainer) -> None:
    optimizer = get_mimo_optimizer(unwrap_megatron_mimo_model(setup_output.model), cfg.optimizer)
    setup_output.optimizer = optimizer
    _update_megatron_mimo_model_config_funcs(
        setup_output.model,
        optimizer,
        setup_output.megatron_mimo_infra,
        setup_output.module_to_grid_tuple,
    )


def _assert_same_encoder_layout(
    dist_par_cfg: MegatronMIMOParallelismConfig,
    ref_par_cfg: MegatronMIMOParallelismConfig,
) -> None:
    dist_encoder = dist_par_cfg.module_parallelisms[_VISION]
    ref_encoder = ref_par_cfg.module_parallelisms[_VISION]
    assert dist_encoder.tensor_model_parallel_size == ref_encoder.tensor_model_parallel_size, (
        "CP oracle requires identical encoder TP between dist and ref so encoder shards compare 1:1."
    )
    assert dist_encoder.data_parallel_size == ref_encoder.data_parallel_size, (
        "CP oracle requires identical encoder DP between dist and ref so per-rank encoder batches match."
    )
    assert dist_encoder.context_parallel_size == ref_encoder.context_parallel_size == 1, (
        "CP oracle keeps encoder CP=1 on both sides; only the language module may vary CP."
    )


def _loss_position_count(output_tensor: torch.Tensor) -> int:
    if output_tensor.dim() <= 2:
        return output_tensor.numel()
    return output_tensor.flatten(start_dim=0, end_dim=-2).shape[0]


class _CPForwardCapture:
    def __init__(self) -> None:
        self.output: torch.Tensor | None = None
        self.loss_mask: torch.Tensor | None = None
        self.num_tokens: int | None = None


def _capturing_forward_step(
    capture: _CPForwardCapture,
    state: GlobalState,
    data_iterator: Iterator,
    model: torch.nn.Module,
):
    output_tensor, loss_fn = megatron_mimo_forward_step(state, data_iterator, model)
    if loss_fn is not None:
        assert isinstance(loss_fn, partial), "Expected MegatronMIMO loss function to be functools.partial"
        loss_mask = loss_fn.args[0]
        assert loss_mask.numel() == _loss_position_count(output_tensor), (
            f"CP-sharded loss shape mismatch: loss_mask={tuple(loss_mask.shape)} "
            f"numel={loss_mask.numel()}, output={tuple(output_tensor.shape)} "
            f"positions={_loss_position_count(output_tensor)}"
        )
        capture.output = output_tensor.detach()
        capture.loss_mask = loss_mask.detach()
        capture.num_tokens = int(loss_mask.sum().item())
    return output_tensor, loss_fn


class _FinalizeTokenCapture:
    def __init__(self) -> None:
        self.num_tokens: torch.Tensor | int | None = None


def _install_finalize_token_spy(model: torch.nn.Module) -> tuple[_FinalizeTokenCapture, object]:
    model_config = get_model_config(model)
    original_finalize = model_config.finalize_model_grads_func
    capture = _FinalizeTokenCapture()

    def wrapped_finalize(model_list, num_tokens=None, pg_collection=None, force_all_reduce=None):
        if isinstance(num_tokens, torch.Tensor):
            capture.num_tokens = num_tokens.detach().clone()
        else:
            capture.num_tokens = num_tokens
        return original_finalize(
            model_list,
            num_tokens,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    model_config.finalize_model_grads_func = wrapped_finalize
    return capture, original_finalize


def _assert_finalizer_saw_local_cp_tokens(
    forward_capture: _CPForwardCapture,
    finalize_capture: _FinalizeTokenCapture,
) -> None:
    assert forward_capture.num_tokens is not None, "CP oracle did not capture a sharded dist loss mask"
    assert isinstance(finalize_capture.num_tokens, torch.Tensor), (
        f"Expected finalizer num_tokens tensor, got {finalize_capture.num_tokens!r}"
    )
    observed = int(finalize_capture.num_tokens.detach().item())
    assert observed == forward_capture.num_tokens, (
        f"Finalizer received num_tokens={observed}, expected local CP-sharded "
        f"loss_mask.sum()={forward_capture.num_tokens}."
    )


def _assert_encoder_weights_match(
    ref_model: torch.nn.Module,
    dist_model: torch.nn.Module,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    ref_params = _named_params_unwrapped(ref_model)
    dist_params = _named_params_unwrapped(dist_model)
    assert set(ref_params) == set(dist_params), "Ref/dist parameter names changed after setup"

    mismatches: list[str] = []
    compared = 0
    for name, ref_param in ref_params.items():
        if _param_module_kind(name) != _VISION:
            continue

        dist_param = dist_params[name]
        assert ref_param.shape == dist_param.shape, (
            f"Encoder weight shape mismatch for {name}: "
            f"ref={tuple(ref_param.shape)}, dist={tuple(dist_param.shape)}"
        )
        ref_value = ref_param.detach().float().to(dist_param.device)
        dist_value = dist_param.detach().float()
        diff = dist_value - ref_value
        compared += 1
        if not torch.allclose(dist_value, ref_value, rtol=rtol, atol=atol):
            mismatches.append(
                f"{name}: max_abs_diff={diff.abs().max().item():.3e}, "
                f"mean_abs_diff={diff.abs().mean().item():.3e}, "
                f"ref_max={ref_value.abs().max().item():.3e}, "
                f"dist_max={dist_value.abs().max().item():.3e}"
            )

    assert compared > 0, "No encoder weights were compared"
    assert not mismatches, (
        f"Post-step encoder weight mismatch in {len(mismatches)} of {compared} params:\n  "
        + "\n  ".join(mismatches[:10])
    )


def _run_one_forward_backward(
    setup_output: Any,
    global_state: GlobalState,
    batch: dict[str, Any],
    *,
    capture: _CPForwardCapture | None = None,
) -> None:
    forward_step = (
        partial(_capturing_forward_step, capture)
        if capture is not None
        else megatron_mimo_forward_step
    )
    wrapped_forward_step = prepare_forward_step_func(forward_step, global_state)
    forward_backward_no_pipelining(
        forward_step_func=wrapped_forward_step,
        data_iterator=_batch_iterator(batch),
        model=[setup_output.model],
        num_microbatches=1,
        seq_length=_SEQ_LENGTH,
        micro_batch_size=_MICRO_BATCH_SIZE,
        forward_only=False,
        pg_collection=setup_output.megatron_mimo_infra.pg_collections[_LANGUAGE],
    )


def _step_and_assert_nonzero_grad_norm(optimizer: Any, label: str) -> None:
    success, grad_norm, _ = optimizer.step()
    assert success, f"{label} optimizer step failed"
    assert grad_norm is not None and grad_norm > 0, (
        f"{label} grad_norm={grad_norm}; encoder grads may have been silently zeroed "
        "by wrong CP scaling or a broken bridge-backward path."
    )


class TestColocatedHeterogeneousTpDpOracle:
    """Functional oracles for colocated heterogeneous parallelism on exactly 2 GPUs."""

    @pytest.mark.run_only_on("GPU")
    def test_heterogeneous_dist_matches_equal_dp_reference_encoder_grads(self) -> None:
        initialize_distributed()
        if dist.get_world_size() != 2:
            pytest.skip(f"Requires exactly 2 GPUs, got {dist.get_world_size()}")

        _set_torch_deterministic()

        import megatron.bridge.training.utils.train_utils as train_utils

        train_utils.report_theoretical_memory = lambda *args, **kwargs: None

        from megatron.core import parallel_state

        if parallel_state._GLOBAL_MEMORY_BUFFER is None:
            parallel_state._set_global_memory_buffer()

        dist_par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                _LANGUAGE: ModuleParallelismConfig(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=1,
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
        ref_par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                _LANGUAGE: ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    data_parallel_size=2,
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

        dist_cfg = _build_dist_config(dist_par_cfg)
        megatron_mimo_runtime_config_update(dist_cfg)
        global_state = GlobalState()
        global_state.cfg = dist_cfg
        setup_output = setup_megatron_mimo(
            state=global_state,
            build_data_iterators_fn=_build_data_iterators_fn,
        )

        ref_model, ref_infra = _build_ref_model_and_infra(ref_par_cfg)
        dist_infra = setup_output.megatron_mimo_infra
        language_tp_group = dist_infra.pg_collections[_LANGUAGE].tp
        language_tp_rank = _pg_rank(language_tp_group)
        language_tp_size = _pg_size(language_tp_group)
        assert language_tp_size == 2

        _copy_ref_weights_to_dist(
            ref_model,
            setup_output.model,
            language_tp_rank=language_tp_rank,
            language_tp_size=language_tp_size,
        )

        batch = _generate_global_batch(seed=67890)

        zero_grad_buffer_for_multimodule(setup_output.module_to_grid_tuple)
        wrapped_forward_step = prepare_forward_step_func(megatron_mimo_forward_step, global_state)
        forward_backward_no_pipelining(
            forward_step_func=wrapped_forward_step,
            data_iterator=_batch_iterator(batch),
            model=[setup_output.model],
            num_microbatches=1,
            seq_length=_SEQ_LENGTH,
            micro_batch_size=_MICRO_BATCH_SIZE,
            forward_only=False,
            pg_collection=dist_infra.pg_collections[_LANGUAGE],
            force_all_reduce=True,
        )

        ref_local_batch = slice_batch_for_megatron_mimo_modules(batch, grids=ref_infra.module_to_grid_map)
        ref_model.zero_grad(set_to_none=True)
        _ref_forward_backward_local(ref_model, ref_local_batch)
        _allreduce_and_scale_ref_grads(
            ref_model,
            language_dp_group=ref_infra.pg_collections[_LANGUAGE].dp,
            vision_dp_group=ref_infra.pg_collections[_VISION].dp,
            global_num_tokens=int(batch["loss_mask"].sum().item()),
        )

        _assert_encoder_grads_match(ref_model, setup_output.model)

    @pytest.mark.run_only_on("GPU")
    def test_language_cp2_dist_matches_cp1_reference_post_step_encoder_weights(self) -> None:
        initialize_distributed()
        if dist.get_world_size() != 2:
            pytest.skip(f"Requires exactly 2 GPUs, got {dist.get_world_size()}")

        _set_torch_deterministic()

        import megatron.bridge.training.utils.train_utils as train_utils

        train_utils.report_theoretical_memory = lambda *args, **kwargs: None

        from megatron.core import parallel_state

        if parallel_state._GLOBAL_MEMORY_BUFFER is None:
            parallel_state._set_global_memory_buffer()

        dist_par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                _LANGUAGE: ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    context_parallel_size=2,
                    data_parallel_size=1,
                    rank_offset=0,
                ),
                _VISION: ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    context_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
            },
        )
        ref_par_cfg = MegatronMIMOParallelismConfig(
            module_parallelisms={
                _LANGUAGE: ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    context_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
                _VISION: ModuleParallelismConfig(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    context_parallel_size=1,
                    data_parallel_size=2,
                    rank_offset=0,
                ),
            },
        )
        _assert_same_encoder_layout(dist_par_cfg, ref_par_cfg)

        dist_cfg = _build_dist_config(
            dist_par_cfg,
            bf16=True,
            overlap_grad_reduce=True,
            use_te_language=True,
        )
        ref_cfg = _build_dist_config(
            ref_par_cfg,
            bf16=True,
            overlap_grad_reduce=True,
            use_te_language=True,
        )
        megatron_mimo_runtime_config_update(dist_cfg)
        megatron_mimo_runtime_config_update(ref_cfg)

        dist_state = GlobalState()
        dist_state.cfg = dist_cfg
        dist_setup = setup_megatron_mimo(
            state=dist_state,
            build_data_iterators_fn=_build_data_iterators_fn,
        )

        ref_state = GlobalState()
        ref_state.cfg = ref_cfg
        ref_setup = setup_megatron_mimo(
            state=ref_state,
            build_data_iterators_fn=_build_data_iterators_fn,
        )

        language_tp_group = dist_setup.megatron_mimo_infra.pg_collections[_LANGUAGE].tp
        language_tp_rank = _pg_rank(language_tp_group)
        language_tp_size = _pg_size(language_tp_group)
        assert language_tp_size == 1, "CP oracle keeps language TP identical so weight shards compare directly."

        _copy_ref_weights_to_dist(
            ref_setup.model,
            dist_setup.model,
            language_tp_rank=language_tp_rank,
            language_tp_size=language_tp_size,
        )
        _rebuild_optimizer_after_weight_copy(ref_setup, ref_cfg)
        _rebuild_optimizer_after_weight_copy(dist_setup, dist_cfg)

        batch = _generate_global_batch(seed=67890, image_dtype=torch.bfloat16)
        dist_local_batch = slice_batch_for_megatron_mimo_modules(
            batch,
            grids=dist_setup.megatron_mimo_infra.module_to_grid_map,
        )
        ref_local_batch = slice_batch_for_megatron_mimo_modules(
            batch,
            grids=ref_setup.megatron_mimo_infra.module_to_grid_map,
        )
        dist_encoder_batch = dist_local_batch["modality_inputs"][_VISION][_VISION_ENCODER]["x"]
        ref_encoder_batch = ref_local_batch["modality_inputs"][_VISION][_VISION_ENCODER]["x"]
        assert dist_encoder_batch.shape == ref_encoder_batch.shape, (
            f"CP oracle requires matching per-rank encoder batch shapes, got "
            f"dist={tuple(dist_encoder_batch.shape)} ref={tuple(ref_encoder_batch.shape)}"
        )
        assert torch.equal(dist_encoder_batch, ref_encoder_batch), (
            "CP oracle requires identical per-rank encoder batches between dist and ref."
        )

        zero_grad_buffer_for_multimodule(dist_setup.module_to_grid_tuple)
        dist_forward_capture = _CPForwardCapture()
        finalize_capture, original_finalize = _install_finalize_token_spy(dist_setup.model)
        dist_model_config = get_model_config(dist_setup.model)
        try:
            _set_global_pg_from_setup(dist_setup)
            _run_one_forward_backward(
                dist_setup,
                dist_state,
                batch,
                capture=dist_forward_capture,
            )
        finally:
            dist_model_config.finalize_model_grads_func = original_finalize
        _assert_finalizer_saw_local_cp_tokens(dist_forward_capture, finalize_capture)
        _step_and_assert_nonzero_grad_norm(dist_setup.optimizer, "Dist")

        zero_grad_buffer_for_multimodule(ref_setup.module_to_grid_tuple)
        _set_global_pg_from_setup(ref_setup)
        _run_one_forward_backward(ref_setup, ref_state, batch)
        _step_and_assert_nonzero_grad_norm(ref_setup.optimizer, "Ref")

        _assert_encoder_weights_match(ref_setup.model, dist_setup.model, rtol=1e-3, atol=1e-3)
