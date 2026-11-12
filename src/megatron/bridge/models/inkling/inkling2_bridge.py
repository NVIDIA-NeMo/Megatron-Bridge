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

"""Hugging Face <-> Megatron conversion for Inkling2."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.inkling import configuration_inkling2  # noqa: F401  (registers the HF config types)
from megatron.bridge.models.inkling.inkling2_config import Inkling2ModelConfig
from megatron.bridge.models.inkling.inkling2_quantization import (
    LazyQuantizedExperts,
    align_precision_config,
    companion_keys,
    expert_format,
    quantize_mxfp8_like,
    quantize_nvfp4_like,
)
from megatron.bridge.models.inkling.modeling_inkling2.model import Inkling2Model
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


_LLM = "model.llm"
# The release's vision_config keys, which the image encoder reads.
_VISION_KEYS = (
    "temporal_patch_size",
    "native_patch_size",
    "adapter_merge_size",
    "projection_hidden_dim",
    "projector_bias",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "mlp_ratio",
    "posemb_grid_size",
    "layer_norm_eps",
)
_EXPERT_SUFFIXES = (".mlp.experts.w13_weight", ".mlp.experts.w2_weight")
_SCALE_SUFFIXES = ("_scale", "_amax", "_outer_scale_denominator")
# Safetensors header dtypes of the tensors an Inkling2 checkpoint stores.
_SAFETENSORS_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "U8": "uint8",
    "F8_E4M3": "float8_e4m3fn",
}


def _vision_config_dict(vision_config: Mapping[str, Any] | PretrainedConfig | None) -> dict[str, Any]:
    """Return the image encoder settings under the release's keys.

    Bridge's configuration class keeps ``vision_config`` as the release's dict; Transformers' own Inkling2 classes
    hold it as a config object that reads the same keys as attributes.
    """
    if vision_config is None:
        return {}
    if isinstance(vision_config, Mapping):
        return dict(vision_config)
    return {key: getattr(vision_config, key) for key in _VISION_KEYS}


class _InterleavedGatedExpertMapping(FusedGatedExpertMapping):
    """Fused ``[E, 2I, H]`` experts whose gate and up rows alternate: ``g0, u0, g1, u1, ...``."""

    def local_hf_param_specs(self, global_param_name: str | None = None) -> tuple:
        """Interleaved rows have no contiguous per-expert gate/up views."""
        return ()

    def local_mxfp8_params(self, *args, **kwargs):
        """Native MXFP8 transport cannot express the interleaved row layout."""
        raise ValueError(f"{self.megatron_param}: native MXFP8 export does not support interleaved gate/up rows")

    def hf_to_megatron(self, hf_weights, megatron_module: torch.nn.Module) -> torch.Tensor:
        """Split one expert's interleaved rows into gate and up before TP distribution."""
        expert_weight = hf_weights[extract_expert_number_from_param(self.megatron_param)]
        return self._gated_mapping.hf_to_megatron(
            {"gate": expert_weight[0::2].contiguous(), "up": expert_weight[1::2].contiguous()},
            megatron_module,
        )

    def megatron_to_hf(self, megatron_weights, megatron_module) -> dict[str, torch.Tensor]:
        """Interleave gate and up rows back into the HF expert layout."""
        converted = self._gated_mapping.megatron_to_hf(megatron_weights, megatron_module)
        fused = {}
        for name, gate in converted.items():
            if not name.endswith(".gate"):
                continue
            base_name = name[: -len(".gate")]
            up = converted.get(f"{base_name}.up")
            if up is not None:
                fused[base_name] = torch.stack((gate, up), dim=-2).flatten(-3, -2)
        return fused


@MegatronModelBridge.register_bridge(
    source="Inkling2ForConditionalGeneration",
    target=Inkling2Model,
    model_type="inkling2",
)
class Inkling2Bridge(MegatronModelBridge):
    """Megatron Bridge for Inkling2 checkpoints.

    Routed experts stored as NVFP4 or MXFP8 are imported in BF16 and quantized
    back against their source scales on export, so an unchanged model exports
    the source bytes. Other tensors keep their source dtype, including the FP32
    AttnRes queries and router biases.

    A plain export (``weight_dtype=torch.bfloat16``, ``--export-weight-dtype
    bfloat16``) writes the experts in BF16 without their scale tensors, keeps
    the FP32 tensors in FP32 as ``hf_quant_config.json`` requires, and lists
    the experts there as BF16. That is the model as Bridge trains it, so a
    fine-tuned model exports without quantizing onto the source scales.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("thinkingmachines/Inkling-2-Small")
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = Inkling2ModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True
    ADDITIONAL_FILE_PATTERNS = [
        "hf_quant_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "tiktoken/*",
        "mtp.safetensors",
    ]
    PLAIN_EXPORT_IGNORED_SOURCE_KEY_SUFFIXES = MegatronModelBridge.PLAIN_EXPORT_IGNORED_SOURCE_KEY_SUFFIXES + tuple(
        f"{expert}{scale}" for expert in _EXPERT_SUFFIXES for scale in _SCALE_SUFFIXES
    )
    PLAIN_EXPORT_KEEPS_FLOAT32 = True

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Map the nested Inkling2 text configuration onto ``Inkling2ModelConfig`` kwargs."""
        text = hf_config.text_config
        unsupported = {
            "n_shared_experts": text.n_shared_experts != 0,
            "pre_norm_pre_affine": text.pre_norm_pre_affine,
            "final_norm_pre_affine": text.final_norm_pre_affine,
            "q_bias": text.q_bias,
            "o_bias": text.o_bias,
            "distinct local attention shapes": (
                text.swa_head_dim != text.head_dim
                or text.swa_num_attention_heads != text.num_attention_heads
                or text.swa_num_key_value_heads != text.num_key_value_heads
            ),
        }
        if any(unsupported.values()):
            raise ValueError(
                f"Unsupported Inkling2 configuration: {[name for name, value in unsupported.items() if value]}"
            )
        config_kwargs = super().hf_config_to_model_config_kwargs(text)
        vision_config = _vision_config_dict(hf_config.vision_config)
        config_kwargs.update(
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=False,
            position_embedding_type="none",
            share_embeddings_and_output_weights=False,
            seq_length=4096,
            ffn_hidden_size=text.intermediate_size,
            moe_ffn_hidden_size=text.intermediate_size,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=text.use_gate_bias,
            moe_router_bias_update_rate=0.0,
            moe_router_topk_scaling_factor=text.route_scale,
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            moe_permute_fusion=True,
            expert_tensor_parallel_size=1,
            use_te_activation_func=True,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            should_pad_vocab=False,
            # Each decoder block is a Hybrid attention layer followed by a MoE layer.
            num_layers=2 * text.num_hidden_layers,
            hybrid_layer_pattern="*E" * text.num_hidden_layers,
            # Layer outputs pack the AttnRes bank with the partial block; keep stage outputs allocated, as Kimi K3 does.
            deallocate_pipeline_outputs=False,
            logit_dtype=torch.float32,
            inkling2_local_layers=tuple(text.local_layer_ids),
            inkling2_sliding_window=text.sliding_window_size,
            inkling2_rel_dim=text.d_rel,
            inkling2_rel_extent=text.rel_extent,
            inkling2_qk_clip_threshold=text.qk_l2_clip_threshold,
            inkling2_rel_clip_threshold=text.rel_l2_clip_threshold,
            inkling2_value_norm=text.use_value_norm_no_gamma,
            inkling2_log_scaling_n_floor=float(text.log_scaling_n_floor),
            inkling2_log_scaling_alpha=text.log_scaling_alpha,
            inkling2_sconv_kernel_size=text.sconv_kernel_size,
            inkling2_attn_res_block_size=text.n_layers_per_attn_res_block,
            inkling2_attn_res_normalize_values=text.attn_res_normalize_values,
            inkling2_mlp_gate_softcap=text.mlp_gate_softcap,
            inkling2_mlp_up_softcap=text.mlp_up_softcap,
            inkling2_final_logit_softcap=text.final_logit_softcapping,
            inkling2_unpadded_vocab_size=text.unpadded_vocab_size,
            vision_config=vision_config,
            image_token_id=hf_config.image_token_id if vision_config else None,
        )
        return config_kwargs

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map each Inkling2 block to its Hybrid attention and MoE layers, the AttnRes queries, and the image encoder.

        Decoder block ``i`` is Hybrid layer ``2 * i`` (attention) and ``2 * i + 1`` (MoE).
        """
        if self.hf_config is None:
            raise ValueError("Inkling2 mapping construction requires the Hugging Face config")
        text = self.hf_config.text_config
        local_layers = set(text.local_layer_ids)
        mappings: list = [
            AutoMapping("embedding.word_embeddings.weight", f"{_LLM}.embed.weight"),
            AutoMapping("output_layer.weight", f"{_LLM}.unembed.weight"),
            ReplicatedMapping("embed_norm.weight", f"{_LLM}.embed_norm.weight"),
            ReplicatedMapping("decoder.final_norm.weight", f"{_LLM}.norm.weight"),
            ReplicatedMapping("vision.**", "model.visual.**"),
        ]
        for index in range(text.num_hidden_layers):
            hf_layer = f"{_LLM}.layers.{index}"
            hf_attention = f"{hf_layer}.attn"
            attention_layer = f"decoder.layers.{2 * index}"
            attention = f"{attention_layer}.self_attention"
            moe_layer = f"decoder.layers.{2 * index + 1}"
            for megatron_name, hf_name in (
                ("linear_q.weight", "wq_du.weight"),
                ("linear_k.weight", "wk_dv.weight"),
                ("linear_v.weight", "wv_dv.weight"),
                ("linear_r.weight", "wr_du.weight"),
                ("linear_gate.weight", "gate_attn.weight"),
                ("linear_sink.weight", "attn_sink_proj.weight"),
                ("q_clip_scale", "q_l2_clip.scale"),
                ("r_clip_scale", "rel_l2_clip.scale"),
            ):
                mappings.append(ColumnParallelMapping(f"{attention}.{megatron_name}", f"{hf_attention}.{hf_name}"))
            mappings += [
                RowParallelMapping(f"{attention}.linear_proj.weight", f"{hf_attention}.wo_ud.weight"),
                ReplicatedMapping(f"{attention}.rel_proj", f"{hf_attention}.rel_logits_proj.proj"),
                ReplicatedMapping(f"{attention}.out_conv_weight", f"{hf_layer}.attn_sconv.weight"),
                ReplicatedMapping(f"{moe_layer}.pre_mlp_layernorm.weight", f"{hf_layer}.mlp_norm.weight"),
                ReplicatedMapping(f"{moe_layer}.mlp_conv_weight", f"{hf_layer}.mlp_sconv.weight"),
                ReplicatedMapping(f"{moe_layer}.mlp.router.weight", f"{hf_layer}.mlp.gate.weight"),
                ReplicatedMapping(f"{moe_layer}.mlp.router.expert_bias", f"{hf_layer}.mlp.gate.bias"),
                _InterleavedGatedExpertMapping(
                    f"{moe_layer}.mlp.experts.linear_fc1.weight*", f"{hf_layer}.mlp.experts.w13_weight"
                ),
                FusedExpertMapping(f"{moe_layer}.mlp.experts.linear_fc2.weight*", f"{hf_layer}.mlp.experts.w2_weight"),
                _InterleavedGatedExpertMapping(
                    f"{moe_layer}.mlp.experts.local_experts.*.linear_fc1.weight", f"{hf_layer}.mlp.experts.w13_weight"
                ),
                FusedExpertMapping(
                    f"{moe_layer}.mlp.experts.local_experts.*.linear_fc2.weight", f"{hf_layer}.mlp.experts.w2_weight"
                ),
                ReplicatedMapping(f"{moe_layer}.mlp_res_query", f"{_LLM}.attn_res.mlp_queries.{index}"),
            ]
            # Local blocks have static K/V convolutions and one input norm; full-attention
            # blocks have dynamic K/V convolutions and separate Q/K/V input norms.
            if index in local_layers:
                mappings += [
                    ReplicatedMapping(f"{attention_layer}.input_layernorm.weight", f"{hf_layer}.attn_norm.weight"),
                    ColumnParallelMapping(f"{attention}.k_conv_weight", f"{hf_attention}.k_sconv.weight"),
                    ColumnParallelMapping(f"{attention}.v_conv_weight", f"{hf_attention}.v_sconv.weight"),
                ]
            else:
                mappings += [
                    ReplicatedMapping(f"{attention_layer}.input_layernorm.weight", f"{hf_layer}.q_input_norm.weight"),
                    ReplicatedMapping(f"{attention_layer}.k_input_norm.weight", f"{hf_layer}.k_input_norm.weight"),
                    ReplicatedMapping(f"{attention_layer}.v_input_norm.weight", f"{hf_layer}.v_input_norm.weight"),
                ]
                for prefix in ("k", "v"):
                    mappings += [
                        RowParallelMapping(
                            f"{attention}.{prefix}_conv_static", f"{hf_attention}.{prefix}_sconv.static_w"
                        ),
                        ColumnParallelMapping(
                            f"{attention}.{prefix}_conv_proj", f"{hf_attention}.{prefix}_sconv.weight_proj"
                        ),
                    ]
            if index > 0:
                mappings.append(
                    ReplicatedMapping(f"{attention_layer}.attn_res_query", f"{_LLM}.attn_res.attn_queries.{index - 1}")
                )
        last_moe_layer = 2 * text.num_hidden_layers - 1
        mappings.append(
            ReplicatedMapping(f"decoder.layers.{last_moe_layer}.output_res_query", f"{_LLM}.attn_res.output_query")
        )
        return MegatronMappingRegistry(*mappings)

    @staticmethod
    def get_hf_import_param_names(
        hf_param: str | dict[str, str],
        available_hf_param_names: set[str] | None = None,
    ) -> tuple[str, ...]:
        """Declare the scale tensors consumed with quantized expert weights."""
        names = MegatronModelBridge.get_hf_import_param_names(hf_param, available_hf_param_names)
        if available_hf_param_names is None:
            return names
        extra = []
        for name in names:
            if name.endswith(_EXPERT_SUFFIXES):
                extra.extend(companion_keys(name, expert_format(name, available_hf_param_names)))
        return tuple(dict.fromkeys((*names, *extra)))

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param: str | dict[str, str],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor] | LazyQuantizedExperts:
        """Read fused quantized experts lazily, one dequantized expert per access."""
        if isinstance(hf_param, str) and hf_param.endswith(_EXPERT_SUFFIXES):
            source = getattr(hf_state_dict, "source", None)
            keys = set(hf_state_dict.keys())
            fmt = expert_format(hf_param, keys)
            if fmt is not None and source is not None and hasattr(source, "key_to_filename_map"):
                return LazyQuantizedExperts(source, hf_param, fmt)
            if fmt is not None:
                raise ValueError(f"Quantized Inkling2 experts {hf_param} require a safetensors checkpoint source")
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

    def _accumulate_grouped_export(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        model_config,
        grouped_buffers: dict[str, dict[int, torch.Tensor]],
        hf_state_dict: Mapping[str, torch.Tensor],
        grouped_sources: dict[str, list[str]] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Quantize each completed fused expert tensor back to its source format and scales."""
        merged = super()._accumulate_grouped_export(
            task, converted_weights_dict, model_config, grouped_buffers, hf_state_dict, grouped_sources
        )
        if merged is None or task.weight_dtype is not None:
            return merged
        keys = set(hf_state_dict.keys())
        result: dict[str, torch.Tensor] = {}
        for name, weight in merged.items():
            fmt = expert_format(name, keys) if name.endswith(_EXPERT_SUFFIXES) else None
            if fmt is None:
                result[name] = weight
                continue
            scales = {key: hf_state_dict[key] for key in companion_keys(name, fmt)}
            scale = scales[f"{name}_scale"]
            quantized = []
            for expert in range(weight.shape[0]):
                if fmt == "nvfp4":
                    quantized.append(
                        quantize_nvfp4_like(
                            weight[expert],
                            scale[expert].to(weight.device),
                            scales[f"{name}_amax"][expert],
                            scales[f"{name}_outer_scale_denominator"],
                        )
                    )
                else:
                    quantized.append(quantize_mxfp8_like(weight[expert], scale[expert].to(weight.device)))
            result[name] = torch.stack(quantized)
            result.update(scales)
        return result

    def postprocess_hf_export_weights(self, path: Path) -> None:
        """List the tensors of a plain export under their written dtype in ``hf_quant_config.json``.

        The precision config copied from the source still names the experts
        NVFP4 or MXFP8 after a plain export writes them in BF16. Runtimes that
        read it, such as the reference SGLang runtime, check each tensor's dtype
        against it, so the experts move to the BF16 group. A default export
        writes the source dtypes and leaves the file unchanged.
        """
        path = Path(path)
        config_path = path / "hf_quant_config.json"
        if not config_path.is_file():
            return
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        else:
            shards = sorted(shard.name for shard in path.glob("*.safetensors"))
        written_dtypes = {}
        for shard in shards:
            with safe_open(path / shard, framework="pt") as handle:
                for name in handle.keys():
                    dtype = handle.get_slice(name).get_dtype()
                    written_dtypes[name] = _SAFETENSORS_DTYPES.get(dtype, dtype.lower())
        config = json.loads(config_path.read_text())
        if align_precision_config(config, written_dtypes):
            staged = config_path.with_name(f".{config_path.name}.tmp")
            staged.write_text(json.dumps(config, indent=2) + "\n")
            staged.replace(config_path)
