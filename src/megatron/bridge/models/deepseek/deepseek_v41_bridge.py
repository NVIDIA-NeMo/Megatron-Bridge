# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Megatron Bridge support for the released DeepSeek-V4.1-Flash checkpoint.

DeepSeek-V4.1 is represented by Megatron-Core's native ``DeepSeekV41Model``.
Every released language block is split into an attention physical layer and an
expert physical layer. The bridge also maps the released Engram, vision, and
DSpark namespaces and decodes the checkpoint's 32-wide FP8 / packed FP4 storage.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.distributed
import torch.nn as nn
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.module import MegatronModule
from transformers import AutoConfig, PretrainedConfig

from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    HCAlphaMapping,
    InitOnlyMapping,
    MegatronParamMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.deepseek_v41_provider import DeepSeekV41ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource


class DeepSeekV41HFConfig(PretrainedConfig):
    """Lightweight Transformers config for checkpoints without ``auto_map``."""

    model_type = "deepseek_v41"
    is_composition = True

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        self._text_config_dict = deepcopy(text_config or {})
        self._vision_config_dict = deepcopy(vision_config)
        self.text_config = SimpleNamespace(**self._text_config_dict)
        self.vision_config = SimpleNamespace(**vision_config) if vision_config is not None else None
        for name in ("vocab_size", "tie_word_embeddings", "max_position_embeddings"):
            if hasattr(self.text_config, name):
                setattr(self, name, getattr(self.text_config, name))

    def to_dict(self):
        output = super().to_dict()
        output.pop("_text_config_dict", None)
        output.pop("_vision_config_dict", None)
        output["text_config"] = deepcopy(self._text_config_dict)
        output["vision_config"] = deepcopy(self._vision_config_dict)
        return output


AutoConfig.register("deepseek_v41", DeepSeekV41HFConfig, exist_ok=True)


def _value(obj: Any, name: str, default=None):
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _text_config(config: Any):
    return _value(config, "text_config", config)


def _load_deepseek_v41_config_class():
    try:
        from megatron.core.models.deepseek_v41.config import DeepSeekV41Config
    except ImportError as exc:  # pragma: no cover - depends on the selected MCore checkout
        raise RuntimeError(
            "DeepSeek-V4.1 requires the native Megatron-Core implementation from "
            "Megatron-LM-dev commit 0d4c86d17 (origin/pull-request/7503) or a descendant."
        ) from exc
    return DeepSeekV41Config


class _ReplicatedOptional(ReplicatedMapping):
    """Replicated mapping whose source key is conditional on the layer role."""

    def __init__(self, megatron_param: str, hf_param: str) -> None:
        super().__init__(megatron_param, hf_param)
        self.allow_hf_name_mismatch = True


class _ReplicatedBufferMapping(ReplicatedMapping):
    """Replicated mapping for a persistent buffer on a parameter-free module."""

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        target_name = self.megatron_param.rsplit(".", 1)[-1]
        target = getattr(megatron_module, target_name)
        hf_weights = hf_weights.to(device=target.device, dtype=target.dtype)
        if self.tp_size == 1:
            return hf_weights
        if target.device.type == "cuda" and target.device.index != torch.cuda.current_device():
            hf_weights = hf_weights.to(torch.cuda.current_device())
        if self.tp_rank > 0:
            hf_weights = torch.empty_like(hf_weights)
        return self.broadcast_tensor_to_tp_ranks(hf_weights, src_rank=0)


class _EngramEmbeddingMapping(MegatronParamMapping[torch.Tensor]):
    """Map one released fused Engram table to its uneven EP row shard."""

    def local_hf_param_specs(self, global_param_name: Optional[str] = None):
        return ()

    def hf_to_megatron(self, hf_weights: Optional[torch.Tensor], megatron_module: nn.Module) -> torch.Tensor:
        target = megatron_module.weight
        global_rows = int(megatron_module.global_num_embeddings)
        if self.tp_rank == 0:
            if hf_weights is None:
                raise ValueError(f"Engram table {self.hf_param} was not loaded on TP rank 0")
            shape = tuple(hf_weights.shape)
            local_shape = tuple(target.shape)
            if shape == local_shape:
                local = hf_weights
            elif shape == (global_rows, target.shape[1]):
                local = hf_weights[megatron_module.row_start : megatron_module.row_end]
            else:
                raise ValueError(
                    f"Engram table {self.hf_param} has shape {shape}; "
                    f"expected local {local_shape} or global {(global_rows, target.shape[1])}"
                )
            local = local.to(device=target.device, dtype=target.dtype).contiguous()
        else:
            local = torch.empty_like(target)
        return self.broadcast_tensor_to_tp_ranks(local, src_rank=0)

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[nn.Module],
    ) -> Dict[str, torch.Tensor]:
        local = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        if local is None or megatron_module is None:
            return {}
        local = self.maybe_dequantize(local)
        if self.ep_size == 1:
            return {str(self.hf_param): local}

        global_rows = int(megatron_module.global_num_embeddings)
        base, remainder = divmod(global_rows, self.ep_size)
        row_counts = [base + int(rank < remainder) for rank in range(self.ep_size)]
        max_rows = max(row_counts)
        padded = local.new_zeros((max_rows, local.shape[1]))
        padded[: local.shape[0]].copy_(local)
        gathered = [torch.empty_like(padded) for _ in range(self.ep_size)]
        torch.distributed.all_gather(gathered, padded, group=self.ep_group)
        full = torch.cat([shard[:rows] for shard, rows in zip(gathered, row_counts)], dim=0)
        return {str(self.hf_param): full}


def _hc_mappings(megatron_prefix: str, hf_prefix: str) -> list[MegatronParamMapping]:
    return [
        ReplicatedMapping(f"{megatron_prefix}.mapping_proj.weight", f"{hf_prefix}_fn"),
        ReplicatedMapping(f"{megatron_prefix}.bias", f"{hf_prefix}_base"),
        HCAlphaMapping(f"{megatron_prefix}.alpha_pre", f"{hf_prefix}_scale", 0),
        HCAlphaMapping(f"{megatron_prefix}.alpha_post", f"{hf_prefix}_scale", 1),
        HCAlphaMapping(f"{megatron_prefix}.alpha_res", f"{hf_prefix}_scale", 2),
    ]


def _attention_mappings(megatron_prefix: str, hf_prefix: str) -> list[MegatronParamMapping]:
    inner = f"{megatron_prefix}.inner_layer"
    core = f"{inner}.self_attention.core_attention"
    mappings: list[MegatronParamMapping] = [
        AutoMapping(f"{inner}.input_layernorm.weight", f"{hf_prefix}.attn_norm.weight"),
        AutoMapping(f"{inner}.self_attention.linear_q_down_proj.weight", f"{hf_prefix}.attn.wq_a.weight"),
        AutoMapping(f"{inner}.self_attention.q_layernorm.weight", f"{hf_prefix}.attn.q_norm.weight"),
        AutoMapping(f"{inner}.self_attention.linear_q_up_proj.weight", f"{hf_prefix}.attn.wq_b.weight"),
        AutoMapping(f"{inner}.self_attention.linear_kv_proj.weight", f"{hf_prefix}.attn.wkv.weight"),
        AutoMapping(f"{inner}.self_attention.kv_layernorm.weight", f"{hf_prefix}.attn.kv_norm.weight"),
        ReplicatedMapping(f"{inner}.self_attention.linear_o_group_proj", f"{hf_prefix}.attn.wo_a.weight"),
        AutoMapping(f"{inner}.self_attention.linear_proj.weight", f"{hf_prefix}.attn.wo_b.weight"),
        ColumnParallelMapping(f"{core}.attn_sink", f"{hf_prefix}.attn.attn_sink"),
        _ReplicatedOptional(f"{core}.compressor.linear_wkv.weight", f"{hf_prefix}.attn.compressor.wkv.weight"),
        _ReplicatedOptional(f"{core}.compressor.linear_wgate.weight", f"{hf_prefix}.attn.compressor.wgate.weight"),
        _ReplicatedOptional(f"{core}.compressor.norm.weight", f"{hf_prefix}.attn.compressor.norm.weight"),
        _ReplicatedOptional(f"{core}.indexer.linear_wq_b.weight", f"{hf_prefix}.attn.indexer.wq_b.weight"),
        _ReplicatedOptional(
            f"{core}.indexer.linear_weights_proj.weight",
            f"{hf_prefix}.attn.indexer.weights_proj.weight",
        ),
        _ReplicatedOptional(f"{core}.indexer.linear_wk.weight", f"{hf_prefix}.attn.indexer.wk.weight"),
        _ReplicatedOptional(f"{core}.indexer.k_norm.weight", f"{hf_prefix}.attn.indexer.k_norm.weight"),
    ]
    mappings.extend(_hc_mappings(f"{megatron_prefix}.hyper_connection", f"{hf_prefix}.hc_attn"))
    return mappings


def _moe_mappings(
    megatron_prefix: str,
    hf_prefix: str,
    *,
    multimodal_router: bool,
) -> list[MegatronParamMapping]:
    inner = f"{megatron_prefix}.inner_layer"
    router = f"{inner}.mlp.router"
    mappings: list[MegatronParamMapping] = [
        AutoMapping(f"{inner}.pre_mlp_layernorm.weight", f"{hf_prefix}.ffn_norm.weight"),
        ColumnParallelMapping(f"{router}.weight", f"{hf_prefix}.ffn.gate.weight"),
        GatedMLPMapping(
            megatron_param=f"{inner}.mlp.experts.linear_fc1.weight*",
            gate=f"{hf_prefix}.ffn.experts.*.w1.weight",
            up=f"{hf_prefix}.ffn.experts.*.w3.weight",
        ),
        AutoMapping(f"{inner}.mlp.experts.linear_fc2.weight*", f"{hf_prefix}.ffn.experts.*.w2.weight"),
        GatedMLPMapping(
            megatron_param=f"{inner}.mlp.experts.local_experts.*.linear_fc1.weight",
            gate=f"{hf_prefix}.ffn.experts.*.w1.weight",
            up=f"{hf_prefix}.ffn.experts.*.w3.weight",
        ),
        AutoMapping(
            f"{inner}.mlp.experts.local_experts.*.linear_fc2.weight",
            f"{hf_prefix}.ffn.experts.*.w2.weight",
        ),
        GatedMLPMapping(
            megatron_param=f"{inner}.mlp.shared_experts.linear_fc1.weight",
            gate=f"{hf_prefix}.ffn.shared_experts.w1.weight",
            up=f"{hf_prefix}.ffn.shared_experts.w3.weight",
        ),
        AutoMapping(
            f"{inner}.mlp.shared_experts.linear_fc2.weight",
            f"{hf_prefix}.ffn.shared_experts.w2.weight",
        ),
    ]
    if multimodal_router:
        mappings.extend(
            [
                _ReplicatedBufferMapping(f"{router}.text_balance.expert_bias", f"{hf_prefix}.ffn.gate.bias"),
                _ReplicatedBufferMapping(f"{router}.image_balance.expert_bias", f"{hf_prefix}.ffn.gate.bias_vl"),
            ]
        )
    else:
        mappings.append(ReplicatedMapping(f"{router}.expert_bias", f"{hf_prefix}.ffn.gate.bias"))
    mappings.extend(_hc_mappings(f"{megatron_prefix}.hyper_connection", f"{hf_prefix}.hc_ffn"))
    return mappings


@MegatronModelBridge.register_bridge(
    source="DeepseekV41ForCausalLM",
    target=HybridModel,
    provider=DeepSeekV41ModelProvider,
    model_type="deepseek_v41",
)
class DeepSeekV41Bridge(MegatronModelBridge):
    """Convert the released V4.1 checkpoint to the native MCore model."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekV41ModelProvider:
        hf_dict = hf_pretrained.config.to_dict()
        config_class = _load_deepseek_v41_config_class()
        native_config = config_class.from_hf(hf_dict)

        provider_fields = DeepSeekV41ModelProvider.__dataclass_fields__
        kwargs = {
            name: getattr(native_config, name)
            for name, field_info in provider_fields.items()
            if field_info.init and hasattr(native_config, name)
        }
        provider = DeepSeekV41ModelProvider(**kwargs)
        text = hf_dict.get("text_config", hf_dict)
        provider.vocab_size = int(text["vocab_size"])
        provider.seq_length = int(text.get("max_position_embeddings", 4096))
        provider.position_embedding_type = "none"
        provider.share_embeddings_and_output_weights = False
        provider.should_pad_vocab = False
        provider.mtp_num_layers = 0
        provider.hf_model_id = str(hf_pretrained.model_name_or_path)
        provider.hf_model_revision = hf_pretrained.init_kwargs.get("revision")
        provider._deepseek_v41_hf_config = deepcopy(hf_dict)
        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: DeepSeekV41ModelProvider) -> dict:
        stored = getattr(provider, "_deepseek_v41_hf_config", None)
        if stored is not None:
            return deepcopy(stored)
        return {
            "architectures": ["DeepseekV41ForCausalLM"],
            "model_type": "deepseek_v41",
            "text_config": {
                "num_hidden_layers": int(provider.num_layers) // 2,
                "hidden_size": provider.hidden_size,
                "num_attention_heads": provider.num_attention_heads,
                "vocab_size": provider.vocab_size,
                "tie_word_embeddings": False,
            },
        }

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param: str | dict[str, str],
        hf_state_dict: Mapping[str, Any],
    ):
        """Decode V4.1's FP8/FP4 tensors before tensor-parallel distribution."""
        if isinstance(hf_param, dict):
            return {key: self.maybe_modify_loaded_hf_weight(value, hf_state_dict) for key, value in hf_param.items()}
        if hf_param.endswith(".__init_only__"):
            return None

        engram_range = getattr(self, "_engram_import_ranges", {}).get(hf_param)
        if engram_range is not None:
            row_start, row_end, load_on_this_rank = engram_range
            if not load_on_this_rank:
                return None
            weight = self._load_state_rows(hf_state_dict, hf_param, row_start, row_end)
            if weight.dtype != torch.float8_e4m3fn:
                return weight
            scale_key = hf_param.removesuffix(".weight") + ".scale"
            if scale_key not in hf_state_dict:
                raise ValueError(f"Quantized Engram table {hf_param} is missing {scale_key}")
            scale = self._load_state_rows(hf_state_dict, scale_key, row_start, row_end)
            return quantization_utils.dequantize_fp8_e4m3fn_with_scale(
                weight,
                scale,
                name=hf_param,
                block_size=32,
                dtype=torch.bfloat16,
            )

        weight = hf_state_dict[hf_param]
        if weight.dtype == torch.int8:
            scale_key = hf_param.removesuffix(".weight") + ".scale"
            if not hf_param.endswith(".weight") or scale_key not in hf_state_dict:
                raise ValueError(f"Packed FP4 weight {hf_param} is missing {scale_key}")
            return quantization_utils.dequantize_mxfp4_e2m1_packed(
                weight,
                hf_state_dict[scale_key],
                dtype=torch.bfloat16,
            )
        if weight.dtype == torch.float8_e4m3fn and hf_param.endswith(".weight"):
            scale_key = hf_param.removesuffix(".weight") + ".scale"
            if scale_key not in hf_state_dict:
                raise ValueError(f"FP8 weight {hf_param} is missing {scale_key}")
            return quantization_utils.dequantize_fp8_e4m3fn_with_scale(
                weight,
                hf_state_dict[scale_key],
                name=hf_param,
                block_size=32,
                dtype=torch.bfloat16,
            )
        return weight

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained,
        megatron_model: Union[MegatronModule, list[MegatronModule]],
        allowed_mismatched_params: Optional[list[str]] = None,
    ) -> list[MegatronModule]:
        """Load regular weights normally and stream giant Engram tables in bounded chunks."""
        self._defer_engram_import = True
        self._deferred_engram_tasks = []
        try:
            loaded_models = super().load_weights_hf_to_megatron(
                hf_pretrained,
                megatron_model,
                allowed_mismatched_params=allowed_mismatched_params,
            )
            hf_state_dict = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}
            self._load_engram_tasks_streaming(hf_state_dict, self._deferred_engram_tasks)
            super().finalize_hf_import(loaded_models)
            return loaded_models
        finally:
            self._defer_engram_import = False
            self._deferred_engram_tasks = []

    def finalize_hf_import(self, megatron_model):
        """Delay parameter-derived caches until the streamed Engram copy is complete."""
        if getattr(self, "_defer_engram_import", False):
            return
        return super().finalize_hf_import(megatron_model)

    @torch.no_grad()
    def _load_engram_tasks_streaming(
        self,
        hf_state_dict: Mapping[str, Any],
        tasks: list[WeightConversionTask],
        *,
        chunk_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        """Copy owner-local Engram rows without materializing a full table or EP shard."""
        if chunk_bytes <= 0:
            raise ValueError("Engram import chunk_bytes must be positive")

        for task in tasks:
            if task.megatron_module is None or task.param_weight is None:
                continue
            if not isinstance(task.mapping, _EngramEmbeddingMapping):
                raise TypeError(f"Expected an Engram task, got {type(task.mapping).__name__}")

            module = task.megatron_module
            target = task.param_weight
            if not isinstance(target, torch.Tensor):
                raise TypeError(f"Engram target {task.param_name} must be a torch.Tensor")

            # The V4.1 RL path uses ordinary MCore EP-local parameters.  Keep
            # this importer row-streamed and copy directly into that local
            # storage; sharded parameter layouts are outside this integration.
            local_flat = target.reshape(-1)

            name = str(task.mapping.hf_param)
            row_start = int(module.row_start)
            row_end = int(module.row_end)
            if target.ndim != 2:
                raise ValueError(f"Engram target {task.param_name} must be rank 2, got {target.ndim}")
            width = int(target.shape[1])
            expected_shape = (row_end - row_start, width)
            if tuple(target.shape) != expected_shape:
                raise ValueError(
                    f"Engram target {task.param_name} has shape {tuple(target.shape)}; expected {expected_shape}"
                )

            scale_name = name.removesuffix(".weight") + ".scale"
            chunk_rows = max(1, chunk_bytes // (width * max(target.element_size(), 2)))
            if task.mapping.tp_rank == 0:
                source_names = [name]
                if scale_name in hf_state_dict:
                    source_names.append(scale_name)
                reader_context = self._state_row_reader(hf_state_dict, source_names)
            else:
                reader_context = nullcontext(None)

            with reader_context as read_rows:
                for global_start in range(row_start, row_end, chunk_rows):
                    global_end = min(global_start + chunk_rows, row_end)
                    rows = global_end - global_start
                    local_start = global_start - row_start
                    local_end = global_end - row_start

                    if task.mapping.tp_rank == 0:
                        assert read_rows is not None
                        chunk = read_rows(name, global_start, global_end)
                        if tuple(chunk.shape) != (rows, width):
                            raise ValueError(
                                f"Engram source {name}[{global_start}:{global_end}] has shape {tuple(chunk.shape)}; "
                                f"expected {(rows, width)}"
                            )
                        if chunk.dtype == torch.float8_e4m3fn:
                            if scale_name not in hf_state_dict:
                                raise ValueError(f"Quantized Engram table {name} is missing {scale_name}")
                            scale = read_rows(scale_name, global_start, global_end)
                            chunk = quantization_utils.dequantize_fp8_e4m3fn_with_scale(
                                chunk,
                                scale,
                                name=name,
                                block_size=32,
                                dtype=target.dtype,
                            )
                        else:
                            chunk = chunk.to(dtype=target.dtype)
                    else:
                        chunk = None

                    if task.mapping.tp_size > 1:
                        backend = str(torch.distributed.get_backend(task.mapping.tp_group)).lower()
                        if "nccl" in backend:
                            if not torch.cuda.is_available():
                                raise RuntimeError("NCCL Engram broadcast requires a CUDA device")
                            communication_device = (
                                target.device
                                if target.device.type == "cuda"
                                else torch.device("cuda", torch.cuda.current_device())
                            )
                        else:
                            communication_device = torch.device("cpu")
                        if chunk is None:
                            chunk = torch.empty((rows, width), dtype=target.dtype, device=communication_device)
                        else:
                            chunk = chunk.to(
                                device=communication_device,
                                dtype=target.dtype,
                                non_blocking=True,
                            )
                        task.mapping.broadcast_tensor_to_tp_ranks(chunk, src_rank=0)
                    else:
                        assert chunk is not None

                    destination_start = local_start * width
                    destination_end = local_end * width
                    local_flat[destination_start:destination_end].copy_(
                        chunk.to(device=local_flat.device, dtype=local_flat.dtype).reshape(-1)
                    )

    @contextmanager
    def _state_row_reader(self, hf_state_dict: Mapping[str, Any], names: list[str]):
        """Keep safetensors shards open while reading many row intervals."""
        source = getattr(hf_state_dict, "source", None)
        if not isinstance(source, SafeTensorsStateSource):
            yield lambda name, row_start, row_end: self._load_state_rows(hf_state_dict, name, row_start, row_end)
            return

        from safetensors import safe_open

        with ExitStack() as stack:
            checkpoints = {}
            slices = {}
            for name in names:
                filename = source.key_to_filename_map[name]
                if filename not in checkpoints:
                    checkpoints[filename] = stack.enter_context(
                        safe_open(source.path / filename, framework="pt", device="cpu")
                    )
                slices[name] = checkpoints[filename].get_slice(name)
            yield lambda name, row_start, row_end: slices[name][row_start:row_end]

    @staticmethod
    def _load_state_rows(
        hf_state_dict: Mapping[str, Any],
        name: str,
        row_start: int,
        row_end: int,
    ) -> torch.Tensor:
        """Read a row interval without materializing a giant safetensors value."""
        source = getattr(hf_state_dict, "source", None)
        if isinstance(source, SafeTensorsStateSource):
            from safetensors import safe_open

            filename = source.key_to_filename_map[name]
            with safe_open(source.path / filename, framework="pt", device="cpu") as checkpoint:
                return checkpoint.get_slice(name)[row_start:row_end]
        return hf_state_dict[name][row_start:row_end]

    def build_conversion_tasks(self, hf_pretrained, megatron_model, weight_dtype=None):
        """Record owner-local Engram rows for bounded checkpoint reads."""
        tasks = super().build_conversion_tasks(hf_pretrained, megatron_model, weight_dtype=weight_dtype)
        ranges = {}
        for task in tasks:
            if not isinstance(task.mapping, _EngramEmbeddingMapping) or task.megatron_module is None:
                continue
            module = task.megatron_module
            ranges[str(task.mapping.hf_param)] = (
                int(module.row_start),
                int(module.row_end),
                task.mapping.tp_rank == 0,
            )
        self._engram_import_ranges = ranges
        if getattr(self, "_defer_engram_import", False):
            self._deferred_engram_tasks = [task for task in tasks if isinstance(task.mapping, _EngramEmbeddingMapping)]
            return [task for task in tasks if not isinstance(task.mapping, _EngramEmbeddingMapping)]
        return tasks

    def mapping_registry(self) -> MegatronMappingRegistry:  # noqa: C901
        config = self.hf_config
        text = _text_config(config)
        depth = int(_value(text, "num_hidden_layers"))
        mappings: list[MegatronParamMapping] = [
            AutoMapping("embedding.word_embeddings.weight", "embed.weight"),
            AutoMapping("output_layer.weight", "head.weight"),
            AutoMapping("decoder.final_norm.weight", "norm.weight"),
        ]

        for layer in range(depth):
            hf_prefix = f"layers.{layer}"
            mappings.extend(_attention_mappings(f"decoder.layers.{2 * layer}", hf_prefix))
            mappings.extend(
                _moe_mappings(
                    f"decoder.layers.{2 * layer + 1}",
                    hf_prefix,
                    multimodal_router=_value(config, "vision_config") is not None,
                )
            )

        engram_layers = list(_value(text, "engram_layer_ids", []) or [])
        for layer in engram_layers:
            prefix = f"decoder.layers.{2 * int(layer)}.engram"
            hf_prefix = f"layers.{int(layer)}.engram"
            mappings.extend(
                [
                    _EngramEmbeddingMapping(f"{prefix}.embed.tables.0.weight", f"{hf_prefix}.embed.weight"),
                    ReplicatedMapping(f"{prefix}.wkv.weight", f"{hf_prefix}.wkv.weight"),
                    ReplicatedMapping(f"{prefix}.q_weight", f"{hf_prefix}.q_weight"),
                    ReplicatedMapping(f"{prefix}.k_weight", f"{hf_prefix}.k_weight"),
                ]
            )
        mappings.extend(
            InitOnlyMapping(name)
            for name in (
                "engram_hash.token_map",
                "engram_hash.primes",
                "engram_hash.offsets",
                "engram_hash.multipliers",
            )
        )

        vision = _value(config, "vision_config")
        if vision is not None:
            mappings.extend(
                ReplicatedMapping(name, name)
                for name in (
                    "image_start",
                    "image_end",
                    "image_newline",
                    "vision.patch_embed.proj.weight",
                    "vision.patch_embed.proj.bias",
                    "vision.norm.weight",
                    "aligner.w1.weight",
                    "aligner.w1.bias",
                    "aligner.w2.weight",
                    "aligner.w2.bias",
                )
            )
            for layer in range(int(_value(vision, "num_hidden_layers"))):
                mappings.extend(
                    ReplicatedMapping(f"vision.blocks.{layer}.{suffix}", f"vision.blocks.{layer}.{suffix}")
                    for suffix in (
                        "norm1.weight",
                        "attn.wqkv.weight",
                        "attn.wqkv.bias",
                        "attn.wo.weight",
                        "attn.wo.bias",
                        "norm2.weight",
                        "mlp.w1.weight",
                        "mlp.w2.weight",
                    )
                )

        num_draft_layers = int(_value(text, "num_nextn_predict_layers", 0) or 0)
        for layer in range(num_draft_layers):
            hf_prefix = f"mtp.{layer}"
            mappings.extend(_attention_mappings(f"dspark.decoder.layers.{2 * layer}", hf_prefix))
            mappings.extend(
                _moe_mappings(
                    f"dspark.decoder.layers.{2 * layer + 1}",
                    hf_prefix,
                    multimodal_router=False,
                )
            )
        if num_draft_layers:
            last = num_draft_layers - 1
            mappings.extend(
                [
                    ReplicatedMapping("dspark.main_proj.weight", "mtp.0.main_proj.weight"),
                    ReplicatedMapping("dspark.main_norm.weight", "mtp.0.main_norm.weight"),
                    ReplicatedMapping("dspark.norm.weight", f"mtp.{last}.norm.weight"),
                    ReplicatedMapping("dspark.markov_embed.weight", f"mtp.{last}.markov_head.embed.weight"),
                    ReplicatedMapping("dspark.markov_head.weight", f"mtp.{last}.markov_head.head.weight"),
                    ReplicatedMapping(
                        "dspark.confidence_head.weight",
                        f"mtp.{last}.confidence_head.proj.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mappings)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Restore the released 32-wide FP8 and packed-FP4 storage layouts."""
        if task.weight_dtype is not None:
            return converted_weights_dict

        result: dict[str, torch.Tensor] = {}
        for hf_param, weight in converted_weights_dict.items():
            if not hf_param.endswith(".weight"):
                paired_weight = hf_param.removesuffix(".scale") + ".weight"
                if hf_param.endswith(".scale") and paired_weight in converted_weights_dict:
                    continue
                result[hf_param] = weight
                continue
            scale_key = hf_param.removesuffix(".weight") + ".scale"
            if scale_key not in hf_state_dict:
                result[hf_param] = weight
                continue
            source_scale = hf_state_dict[scale_key]
            if ".ffn.experts." in hf_param and ".shared_experts." not in hf_param:
                q_weight, q_scale = quantization_utils.quantize_mxfp4_e2m1_like_scale(
                    weight,
                    source_scale,
                    name=hf_param,
                )
            else:
                q_weight, q_scale = quantization_utils.quantize_fp8_e4m3fn_like_scale(
                    weight,
                    source_scale,
                    name=hf_param,
                    block_size=32,
                )
            result[hf_param] = q_weight
            result[scale_key] = q_scale
        return result


__all__ = ["DeepSeekV41Bridge"]
