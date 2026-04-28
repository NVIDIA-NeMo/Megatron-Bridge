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
"""Colocated MegatronMIMO heterogeneous TP/DP oracle.

This is a small 2-GPU functional oracle for the Stage-2 colocated shape:

* Dist: ``vision(tp=1, dp=2) x language(tp=2, dp=1)``
* Ref: ``vision(tp=1, dp=2) x language(tp=1, dp=2)``

The reference uses the same encoder TP/DP layout as dist, but an equal-DP
language layout. That makes the ref bridge an identity-style path while the
dist side exercises fan-in from encoder DP=2 into language DP=1.

The assertion is deliberately at gradient level. Adam's first update is almost
scale-invariant for positive gradient rescaling, which is exactly the bug class
this test is meant to catch. The dist side uses the real Megatron Bridge setup
path, DDP wrapping, colocated forward step, and ``finalize_model_grads_multimodule``.
The ref side is a bare MimoModel that runs the same local forward/backward math,
then manually does per-module DP SUM and a single ``1 / global_num_tokens`` scale.

Run:
    torchrun --nproc_per_node=2 -m pytest -v -s -x \\
        tests/functional_tests/test_groups/training/megatron_mimo/test_colocated_heterogeneous_tp_dp_oracle.py
"""

from __future__ import annotations

import os
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
from megatron.core.pipeline_parallel.schedules import forward_backward_no_pipelining
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

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
_GLOBAL_BATCH_SIZE = 2
_MICRO_BATCH_SIZE = 2

_LANGUAGE = "language"
_VISION = "vision"
_VISION_ENCODER = "clip"


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
    provider = MegatronMIMOProvider(
        language_model_spec=language_model_spec,
        modality_submodules_spec=modality_submodules_spec,
        special_token_ids=special_token_ids,
        megatron_mimo_parallelism_config=par_cfg,
        topology={_VISION: [_LANGUAGE], _LANGUAGE: []},
        bf16=False,
    )
    return provider


def _build_dist_config(par_cfg: MegatronMIMOParallelismConfig) -> ConfigContainer:
    train_cfg = TrainingConfig(
        micro_batch_size=_MICRO_BATCH_SIZE,
        global_batch_size=_GLOBAL_BATCH_SIZE,
        train_iters=1,
    )
    train_cfg.num_microbatches = 1

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


def _build_ref_model_and_infra(par_cfg: MegatronMIMOParallelismConfig) -> tuple[torch.nn.Module, Any]:
    provider = _build_provider(par_cfg)
    provider.initialize_model_parallel(seed=1234)
    infra = provider.build_infra()
    model = provider.provide()
    model = model.cuda(torch.cuda.current_device())
    model.train()
    return model, infra


def _build_data_iterators_fn(
    _cfg: ConfigContainer, _infra: Any, *, train_state: Any | None = None
) -> tuple[Iterator, None]:
    del train_state
    return iter([]), None


def _generate_global_batch(seed: int) -> dict[str, Any]:
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
            _VISION: {_VISION_ENCODER: {"x": pixel_values.to(device, non_blocking=False)}},
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


class TestColocatedHeterogeneousTpDpOracle:
    """Functional oracle for colocated heterogeneous TP/DP on exactly 2 GPUs."""

    @pytest.mark.run_only_on("GPU")
    def test_heterogeneous_dist_matches_equal_dp_reference_encoder_grads(self) -> None:
        initialize_distributed()
        if dist.get_world_size() != 2:
            pytest.skip(f"Requires exactly 2 GPUs, got {dist.get_world_size()}")

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
