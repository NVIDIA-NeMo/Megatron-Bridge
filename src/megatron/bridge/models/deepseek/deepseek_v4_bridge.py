# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Bridge for DeepSeek-V4 (and V4-Flash).

Checkpoint format notes
-----------------------
DeepSeek-V4 uses a custom serialisation format that differs from standard
HuggingFace Transformers naming conventions:

  - embed.weight            (not model.embed_tokens.weight)
  - head.weight             (not lm_head.weight)
  - norm.weight             (not model.norm.weight)
  - layers.N.attn_norm.weight / layers.N.ffn_norm.weight
  - layers.N.attn.wq_a / wq_b / wkv / wo_a / wo_b …
  - layers.N.ffn.gate / experts / shared_experts …
  - layers.N.hc_attn_fn / hc_attn_base / hc_attn_scale  (Hyper-Connections)
  - layers.N.hc_ffn_fn  / hc_ffn_base  / hc_ffn_scale
  - hc_head_fn / hc_head_base / hc_head_scale            (global HC head)
  - mtp.N.*                                               (MTP layers)

All linear weights are FP8 (float8_e4m3fn) with per-128×128-block scales
stored as float8_e8m0fnu alongside each weight tensor.  The bridge
dequantises them to bfloat16 during import.

MoE router note
---------------
Hash-routing layers (layer_number <= moe_n_hash_layers) contain a
`tid2eid` buffer (int32 vocab→expert lookup table).  Buffers are not
parameters, so Megatron does not expose them via `named_parameters()`.
The bridge handles `tid2eid` via `maybe_modify_loaded_hf_weight()` and
a dedicated `_Tid2EidMapping` that writes it into `state_dict` directly.

Megatron-LM prerequisites (on weijiac/dsv4-bridge branch)
----------------------------------------------------------
  - HyperConnectionModule   (megatron.core.transformer.hyper_connection)
  - DSv4HybridSelfAttention / CompressedSparseAttention / CSAIndexer / Compressor
    (megatron.core.transformer.experimental_attention_variant.{deepseek_v4_hybrid_attention,csa})
  - Hash-routing tid2eid + SwiGLU clamp (PRs #4458 and #4481, merged into branch)
"""

from typing import Dict, Mapping

import torch
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec as _get_exp_attn_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


# ---------------------------------------------------------------------------
# Config class — registers "deepseek_v4" with transformers AutoConfig
# ---------------------------------------------------------------------------
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    """Minimal config stub for DeepSeek-V4 / V4-Flash.

    Registers the ``deepseek_v4`` model_type with HuggingFace AutoConfig so
    that :func:`AutoConfig.from_pretrained` can load the checkpoint config.
    """

    model_type = "deepseek_v4"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# Register at import time so AutoConfig.from_pretrained works before any bridge
# method is called.
try:
    AutoConfig.register("deepseek_v4", DeepseekV4Config)
except ValueError:
    pass  # already registered


# ---------------------------------------------------------------------------
# Custom mapping helpers
# ---------------------------------------------------------------------------


class _HCAlphaMapping(MegatronParamMapping):
    """Map Megatron's three scalar HC alpha parameters to/from the V4 checkpoint's
    3-element hc_*_scale tensor.

    V4 checkpoint  :  layers.N.hc_attn_scale  shape [3]  = [alpha_pre, alpha_post, alpha_res]
    Megatron       :  three separate nn.Parameter([1]) tensors
    """

    def __init__(self, megatron_pre: str, megatron_post: str, megatron_res: str, hf_param: str):
        # We register under the alpha_pre path; the others are handled inside hf_to_megatron.
        super().__init__(megatron_param=megatron_pre, hf_param=hf_param)
        self._megatron_post = megatron_post
        self._megatron_res = megatron_res

    @staticmethod
    def _resolve_single(pattern: str, captures) -> str:
        result = pattern
        ci = 0
        while "**" in result and ci < len(captures):
            result = result.replace("**", captures[ci], 1)
            ci += 1
        ci = 0
        while "*" in result and ci < len(captures):
            result = result.replace("*", captures[ci], 1)
            ci += 1
        return result

    def resolve(self, captures):
        resolved_mg, resolved_hf = self._resolve_names(captures)
        resolved_post = self._resolve_single(self._megatron_post, captures)
        resolved_res = self._resolve_single(self._megatron_res, captures)
        return _HCAlphaMapping(
            megatron_pre=resolved_mg,
            megatron_post=resolved_post,
            megatron_res=resolved_res,
            hf_param=resolved_hf,
        )

    def hf_to_megatron(self, hf_weights, megatron_module):
        # hf_weights is hc_*_scale [3]; we write alpha_pre here (index 0).
        # alpha_post and alpha_res are handled by their own mappings when registered.
        target = hf_weights.to(megatron_module.alpha_pre.device)
        return target[0:1]

    def megatron_to_hf(self, megatron_weights, megatron_module):
        # megatron_weights is alpha_pre [1]; gather all 3 from the same module.
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        if megatron_weights is None:
            return {}
        megatron_weights = self.maybe_dequantize(megatron_weights)
        post = megatron_module.alpha_post.detach()
        res = megatron_module.alpha_res.detach()
        return {self.hf_param: torch.cat([megatron_weights.float(), post.float(), res.float()])}


class _HCAlphaSecondaryMapping(MegatronParamMapping):
    """Secondary mapping for alpha_post (index=1) or alpha_res (index=2).

    Import: extracts element [index] from the 3-element hc_*_scale tensor.
    Export: returns {} because the primary _HCAlphaMapping (alpha_pre) already
    exports all three alpha values together. This mapping just suppresses the
    "No mapping found" warning for the secondary Megatron params during export.
    """

    def __init__(self, megatron_param: str, hf_scale_param: str, index: int):
        super().__init__(megatron_param=megatron_param, hf_param=hf_scale_param)
        self._index = index

    def hf_to_megatron(self, hf_weights, megatron_module):
        attr = "alpha_post" if self._index == 1 else "alpha_res"
        target = hf_weights.to(getattr(megatron_module, attr).device)
        return target[self._index : self._index + 1]

    def resolve(self, captures):
        resolved_mg, resolved_hf = self._resolve_names(captures)
        return _HCAlphaSecondaryMapping(resolved_mg, resolved_hf, self._index)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        # Already handled by the primary alpha_pre _HCAlphaMapping
        return {}


# MXFP4 E2M1 lookup table: nibble value 0-15 → float32
# FP4 E2M1 (sign=1, exponent=2, mantissa=1, exp_bias=1):
#   positive: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
#   negative: [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
_FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


class _MTPEHProjMapping(ColumnParallelMapping):
    """Map Megatron's single concatenated eh_proj (ColumnParallelLinear) to
    V4's separate e_proj + h_proj.

    V4 MTP:      out = e_proj(enorm(embed)) + h_proj(hnorm(hidden))
    Megatron MTP: out = eh_proj(cat(enorm(embed), hnorm(hidden), dim=-1))

    These are mathematically equivalent when:
        eh_proj.weight = cat(e_proj.weight, h_proj.weight, dim=1)

    eh_proj is a ColumnParallelLinear:
      - full weight:       (hidden, 2*hidden)
      - each TP rank gets: (hidden // TP, 2*hidden)  (split along rows)

    Import: cat(e, h, dim=1) -> (hidden, 2*hidden), ColumnParallel scatter along dim=0
    Export: ColumnParallel gather along dim=0 -> (hidden, 2*hidden), split on dim=1
    """

    def __init__(self, megatron_eh: str, hf_e: str, hf_h: str):
        # dict hf_param causes the load loop to fetch both tensors
        super().__init__(megatron_param=megatron_eh, hf_param={"e": hf_e, "h": hf_h})
        self._hf_e = hf_e
        self._hf_h = hf_h

    def hf_to_megatron(
        self,
        hf_weights: Dict[str, torch.Tensor],
        megatron_module,
    ) -> torch.Tensor:
        e = hf_weights["e"]  # (hidden, hidden)
        h = hf_weights["h"]  # (hidden, hidden)
        eh = torch.cat([e, h], dim=1)  # (hidden, 2*hidden)
        # ColumnParallelMapping splits eh along dim=0 and scatters to TP ranks
        return ColumnParallelMapping.hf_to_megatron(self, eh, megatron_module)

    def megatron_to_hf(
        self,
        megatron_weights,
        megatron_module,
    ) -> Dict[str, torch.Tensor]:
        # Broadcast from owning PP rank
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=self._hf_e)
        if megatron_weights is None:
            return {}
        megatron_weights = self.maybe_dequantize(megatron_weights)
        # Gather TP shards along dim=0 -> full (hidden, 2*hidden)
        if self.tp_size == 1:
            full_weights = megatron_weights
        else:
            gathered = self.gather_from_tp_ranks(megatron_weights)
            full_weights = torch.cat(gathered, dim=0)
        # Split back into e and h along dim=1
        e, h = full_weights.chunk(2, dim=1)
        return {self._hf_e: e, self._hf_h: h}


class _ReplicatedOptional(ReplicatedMapping):
    """ReplicatedMapping for CSA-optional weights (compressor / indexer).

    Sets allow_hf_name_mismatch=True so the export path does not validate
    the HF key against the real checkpoint's key set.  Compressor and indexer
    weights only exist on non-hash layers; when we build a tiny smoke-test
    model whose layer indices don't match the production compress_ratios, a
    strict hf_keys check would wrongly skip those weights.

    resolve_wildcards() uses type(self)(...) which preserves this subclass,
    so allow_hf_name_mismatch stays True after wildcard expansion.
    """

    def __init__(self, megatron_param: str, hf_param: str) -> None:
        super().__init__(megatron_param, hf_param)
        self.allow_hf_name_mismatch = True


# ---------------------------------------------------------------------------
# Bridge registration
# ---------------------------------------------------------------------------


@MegatronModelBridge.register_bridge(
    source="DeepseekV4ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v4",
)
class DeepSeekV4Bridge(MegatronModelBridge):
    """Megatron Bridge for DeepSeek-V4 / V4-Flash.

    Requires the weijiac/dsv4-bridge branch of Megatron-LM which contains:
      - Hyper-Connections (mHC, PR #3430)
      - DSv4 hybrid attention / CSA / Compressor / Indexer (PR #4458)
      - Hash MoE + SwiGLU clamp + new mHC contract (PR #4481)
    """

    # ------------------------------------------------------------------
    # Provider configuration
    # ------------------------------------------------------------------

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # ---- Attention ----
        provider.experimental_attention_variant = "dsv4_hybrid"
        provider.multi_latent_attention = True
        # V4 uses a heterogeneous per-layer spec (hash vs MLA layers differ);
        # override the default transformer_engine_layer_spec with the experimental
        # attention variant block spec builder.
        # GPTModelProvider IS the TransformerConfig (not cfg.transformer)
        provider.transformer_layer_spec = lambda cfg, vp_stage=None: _get_exp_attn_spec(config=cfg, vp_stage=vp_stage)
        provider.qk_layernorm = True
        provider.normalization = "RMSNorm"
        provider.add_bias_linear = False

        # V4 MLA geometry
        # head_dim = 512 (nope_dim + rope_dim = 448 + 64)
        provider.v_head_dim = hf_config.head_dim  # 512
        provider.qk_pos_emb_head_dim = hf_config.qk_rope_head_dim  # 64
        # qk_head_dim and kv_lora_rank derived automatically in DSv4HybridConfig
        provider.q_lora_rank = hf_config.q_lora_rank  # 1024
        provider.o_groups = hf_config.o_groups  # 8
        provider.o_lora_rank = hf_config.o_lora_rank  # 1024

        # ---- Rotary embeddings (YaRN) ----
        # Two separate RoPE bases in V4:
        #   - compress_rope_theta for compressed-KV layers
        #   - rope_theta for pure sliding-window layers (layers 0,1)
        # Megatron uses a single rotary_base; we set it to the YaRN base.
        provider.apply_rope_fusion = False
        provider.rope_type = "yarn"
        provider.rotary_base = float(hf_config.rope_theta)  # 10000
        provider.rotary_scaling_factor = float(hf_config.rope_scaling["factor"])  # 16
        provider.original_max_position_embeddings = int(
            hf_config.rope_scaling["original_max_position_embeddings"]
        )  # 65536
        provider.beta_fast = float(hf_config.rope_scaling.get("beta_fast", 32))
        provider.beta_slow = float(hf_config.rope_scaling.get("beta_slow", 1))

        # ---- CSA (Compressed Sparse Attention) ----
        # compress_ratios has num_hidden_layers + num_nextn_predict_layers entries;
        # csa_compress_ratios covers only the main transformer layers.
        num_main_layers = hf_config.num_hidden_layers
        provider.csa_compress_ratios = list(hf_config.compress_ratios[:num_main_layers])
        provider.csa_window_size = hf_config.sliding_window  # 128

        # DSA indexer geometry (matches index_n_heads / index_head_dim / index_topk in config)
        provider.dsa_indexer_n_heads = hf_config.index_n_heads  # 64
        provider.dsa_indexer_head_dim = hf_config.index_head_dim  # 128
        provider.dsa_indexer_topk = hf_config.index_topk  # 512

        # ---- Hyper-Connections (mHC) ----
        provider.enable_hyper_connections = True
        provider.num_residual_streams = hf_config.hc_mult  # 4
        provider.mhc_sinkhorn_iterations = hf_config.hc_sinkhorn_iters  # 20

        # ---- MoE ----
        provider.gated_linear_unit = True
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = False  # V4 uses post-topk normalisation
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "noaux_tc"
        provider.moe_shared_expert_overlap = True
        provider.moe_router_score_function = hf_config.scoring_func  # "sqrtsoftplus"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_router_topk = hf_config.num_experts_per_tok  # 6
        provider.norm_topk_prob = hf_config.norm_topk_prob
        provider.moe_router_topk_scaling_factor = hf_config.routed_scaling_factor  # 1.5

        # Hash routing
        provider.moe_n_hash_layers = hf_config.num_hash_layers  # 3
        provider.actual_vocab_size = hf_config.vocab_size  # 129280

        # SwiGLU activation clamp
        provider.activation_func_clamp_value = hf_config.swiglu_limit  # 10.0

        # All 43 layers are MoE (no dense prefix unlike V3)
        provider.moe_layer_freq = [1] * hf_config.num_hidden_layers
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * hf_config.n_shared_experts

        # ---- MTP ----
        provider.mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or None

        # ---- Misc ----
        provider.share_embeddings_and_output_weights = bool(hf_config.tie_word_embeddings)
        provider.gradient_accumulation_fusion = True
        provider.bias_dropout_fusion = True
        provider.cross_entropy_fusion_impl = "te"
        provider.cross_entropy_loss_fusion = True
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = False
        provider.make_vocab_size_divisible_by = 1280
        provider.seq_length = 4096

        return provider

    # ------------------------------------------------------------------
    # Export: HF config reconstruction
    # ------------------------------------------------------------------

    @classmethod
    def megatron_to_hf_config(cls, provider: MLAModelProvider) -> dict:
        hf_cfg = super(DeepSeekV4Bridge, cls).megatron_to_hf_config(provider)

        hf_cfg["num_nextn_predict_layers"] = getattr(provider, "mtp_num_layers", None) or 0
        hf_cfg["num_hash_layers"] = getattr(provider, "moe_n_hash_layers", 0)
        hf_cfg["swiglu_limit"] = getattr(provider, "activation_func_clamp_value", 0.0)

        compress_ratios = getattr(provider, "csa_compress_ratios", None)
        if compress_ratios is not None:
            num_mtp = hf_cfg.get("num_nextn_predict_layers", 0)
            hf_cfg["compress_ratios"] = list(compress_ratios) + [0] * num_mtp

        hf_cfg["sliding_window"] = getattr(provider, "csa_window_size", 128)
        hf_cfg["hc_mult"] = getattr(provider, "num_residual_streams", 4)
        hf_cfg["hc_sinkhorn_iters"] = getattr(provider, "mhc_sinkhorn_iterations", 20)
        hf_cfg["n_shared_experts"] = getattr(provider, "moe_shared_expert_intermediate_size", 0) // hf_cfg.get(
            "moe_intermediate_size", 1
        )

        return hf_cfg

    # ------------------------------------------------------------------
    # FP8 / MXFP4 dequantisation on import
    # ------------------------------------------------------------------

    def _dequant_mxfp4(
        self,
        hf_param: str,
        weight_i8: torch.Tensor,
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Dequantize MXFP4-packed expert weights (I8 body + F8_E8M0 block scale).

        Each I8 byte stores 2 FP4 E2M1 nibbles (low nibble = first element,
        high nibble = second element, i.e. [lo0, hi0, lo1, hi1, ...]).
        One E8M0 scale covers 32 consecutive FP4 elements along K.

        Args:
            hf_param: Weight tensor name (used to derive scale key).
            weight_i8: Raw I8 tensor of shape (M, K//2).
            hf_state_dict: Full HF state dict for scale lookup.

        Returns:
            Dequantised weight as bfloat16 of shape (M, K).
        """
        scale_key = hf_param[: -len(".weight")] + ".scale" if hf_param.endswith(".weight") else None
        if scale_key is None or scale_key not in hf_state_dict:
            return weight_i8.to(torch.bfloat16)

        scale = hf_state_dict[scale_key]  # F8_E8M0, shape (M, K_scale)

        # Reinterpret int8 as uint8 so bitwise ops give correct 0-15 nibbles
        w_u8 = weight_i8.view(torch.uint8)  # (M, K_packed)
        lo = (w_u8 & 0xF).to(torch.int64)  # (M, K_packed)
        hi = (w_u8 >> 4).to(torch.int64)  # (M, K_packed)

        table = _FP4_E2M1_TABLE.to(weight_i8.device)
        # stack [lo_row, hi_row] along last dim then flatten → interleaved (M, K_logical)
        # This creates a contiguous output avoiding slow strided writes.
        logical = torch.stack([table[lo], table[hi]], dim=-1).reshape(
            weight_i8.shape[0], -1
        )  # (M, K_logical), float32

        # E8M0 scale: value = 2^(e - 127), block_size = K_logical / K_scale = 32
        scale_f32 = 2.0 ** (scale.to(torch.float32) - 127.0)  # (M, K_scale)
        block_size = logical.shape[1] // scale_f32.shape[1]
        # repeat_interleave along dim=1 expands (M, K_scale) → (M, K_logical)
        scale_exp = scale_f32.repeat_interleave(block_size, dim=1)

        return (logical * scale_exp).to(torch.bfloat16)

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param,
        hf_state_dict: Mapping[str, torch.Tensor],
    ):
        """Dequantise quantized weights using their accompanying block-scale tensor.

        V4 stores attention/embedding weights as float8_e4m3fn with 128x128-block
        scales, and expert FFN weights as MXFP4 packed (I8, 2 nibbles/byte) with
        F8_E8M0 per-32-element scales.  For dict hf_param (GatedMLPMapping etc.),
        dequantizes each key individually so expert gate/up weights are also handled.
        """
        if isinstance(hf_param, dict):
            # Recurse for each key so each string key gets dequantized individually
            return {k: self.maybe_modify_loaded_hf_weight(v, hf_state_dict) for k, v in hf_param.items()}

        weight = hf_state_dict[hf_param]

        # MXFP4 packed (I8): expert FFN gate/up/down weights
        if weight.dtype == torch.int8:
            return self._dequant_mxfp4(hf_param, weight, hf_state_dict)

        if weight.dtype != torch.float8_e4m3fn:
            return weight

        if not hf_param.endswith(".weight"):
            return weight.to(torch.bfloat16)

        scale_key = hf_param[: -len(".weight")] + ".scale"
        if scale_key not in hf_state_dict:
            return weight.to(torch.bfloat16)

        scale = hf_state_dict[scale_key]  # [ceil(out/128), ceil(in/128)], float8_e8m0fnu
        weight_bf16 = weight.to(torch.bfloat16)

        # Detect scale format: 128-tile (attn) vs per-row/16-tile (expert FFN)
        # Attention: scale = (ceil(M/128), ceil(K/128))
        # Expert FFN: scale = (M, ceil(K/16))  [per-row, 16-element K-tiles]
        scale_f32 = scale.to(torch.float32)
        if scale_f32.dim() == 1:
            # 1-D scale: single value per row (broadcast over K)
            if scale_f32.shape[0] == weight_bf16.shape[0]:
                scale_exp = scale_f32.unsqueeze(1)
            else:
                scale_exp = scale_f32.repeat_interleave(128)[: weight_bf16.shape[0]].unsqueeze(1)
        elif scale_f32.shape[0] == weight_bf16.shape[0]:
            # Per-row format: scale[i, j] covers weight[i, j*16:(j+1)*16]
            tile_k = weight_bf16.shape[1] // scale_f32.shape[1]
            scale_exp = scale_f32.repeat_interleave(tile_k, dim=1)[:, : weight_bf16.shape[1]]
        else:
            # 128-tile format: scale[i, j] covers weight[i*128:(i+1)*128, j*128:(j+1)*128]
            scale_exp = scale_f32.repeat_interleave(128, dim=0)[: weight_bf16.shape[0]]
            scale_exp = scale_exp.repeat_interleave(128, dim=1)[:, : weight_bf16.shape[1]]
        if scale_exp.shape != weight_bf16.shape:
            raise RuntimeError(
                f"FP8 dequant shape mismatch for {hf_param!r}: "
                f"weight={tuple(weight_bf16.shape)} scale={tuple(scale.shape)} "
                f"scale_exp={tuple(scale_exp.shape)}"
            )
        return (weight_bf16.to(torch.float32) * scale_exp).to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Weight mapping registry
    # ------------------------------------------------------------------

    def mapping_registry(self) -> MegatronMappingRegistry:  # noqa: C901
        hf_config = self.hf_config
        num_mtp = getattr(hf_config, "num_nextn_predict_layers", 0)  # 1

        mappings = []

        # ------ Embeddings / LM head / final norm ------
        mappings += [
            AutoMapping("embedding.word_embeddings.weight", "embed.weight"),
            AutoMapping("output_layer.weight", "head.weight"),
            AutoMapping("decoder.final_layernorm.weight", "norm.weight"),
            # Global HC head (lives on TransformerBlock, not a parallel module → replicated)
            ReplicatedMapping("decoder.hc_head_fn.weight", "hc_head_fn"),
            ReplicatedMapping("decoder.hc_head_base", "hc_head_base"),
            ReplicatedMapping("decoder.hc_head_scale", "hc_head_scale"),
        ]

        # ------ Per-layer mappings ------
        mappings += [
            # Layer norms
            AutoMapping(
                "decoder.layers.*.input_layernorm.weight",
                "layers.*.attn_norm.weight",
            ),
            AutoMapping(
                "decoder.layers.*.pre_mlp_layernorm.weight",
                "layers.*.ffn_norm.weight",
            ),
            # Q down / Q norm / Q up (MLA)
            AutoMapping(
                "decoder.layers.*.self_attention.linear_q_down_proj.weight",
                "layers.*.attn.wq_a.weight",
            ),
            AutoMapping(
                "decoder.layers.*.self_attention.q_layernorm.weight",
                "layers.*.attn.q_norm.weight",
            ),
            AutoMapping(
                "decoder.layers.*.self_attention.linear_q_up_proj.weight",
                "layers.*.attn.wq_b.weight",
            ),
            # KV (single projection) / KV norm
            AutoMapping(
                "decoder.layers.*.self_attention.linear_kv_up_proj.weight",
                "layers.*.attn.wkv.weight",
            ),
            AutoMapping(
                "decoder.layers.*.self_attention.kv_layernorm.weight",
                "layers.*.attn.kv_norm.weight",
            ),
            # Factored output projection: wo_a (group param) + wo_b (row-parallel linear)
            # linear_o_group_proj is a plain nn.Parameter (all o_groups on every TP rank)
            ReplicatedMapping(
                "decoder.layers.*.self_attention.linear_o_group_proj",
                "layers.*.attn.wo_a.weight",
            ),
            AutoMapping(
                "decoder.layers.*.self_attention.linear_proj.weight",
                "layers.*.attn.wo_b.weight",
            ),
            # Attention sink: split by TP (size = num_heads // TP on each rank)
            ColumnParallelMapping(
                "decoder.layers.*.self_attention.core_attention.attn_sink",
                "layers.*.attn.attn_sink",
            ),
            # Compressor (compress_ratio > 1 layers: 128x and 4x)
            # All compressor linears use parallel_mode="duplicated" -> ReplicatedMapping
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.compressor.linear_wkv.weight",
                "layers.*.attn.compressor.wkv.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.compressor.linear_wgate.weight",
                "layers.*.attn.compressor.wgate.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.compressor.ape",
                "layers.*.attn.compressor.ape",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.compressor.norm.weight",
                "layers.*.attn.compressor.norm.weight",
            ),
            # Indexer (compress_ratio == 4 layers only)
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.linear_wq_b.weight",
                "layers.*.attn.indexer.wq_b.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.linear_weights_proj.weight",
                "layers.*.attn.indexer.weights_proj.weight",
            ),
            # Indexer sub-compressor (each indexer has its own compressor)
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.compressor.linear_wkv.weight",
                "layers.*.attn.indexer.compressor.wkv.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.compressor.linear_wgate.weight",
                "layers.*.attn.indexer.compressor.wgate.weight",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.compressor.ape",
                "layers.*.attn.indexer.compressor.ape",
            ),
            _ReplicatedOptional(
                "decoder.layers.*.self_attention.core_attention.indexer.compressor.norm.weight",
                "layers.*.attn.indexer.compressor.norm.weight",
            ),
            # MoE router weight and expert bias
            AutoMapping(
                "decoder.layers.*.mlp.router.weight",
                "layers.*.ffn.gate.weight",
            ),
            AutoMapping(
                "decoder.layers.*.mlp.router.expert_bias",
                "layers.*.ffn.gate.bias",
            ),
            # Hash-routing lookup table (buffer, not a parameter)
            AutoMapping(
                "decoder.layers.*.mlp.router.tid2eid",
                "layers.*.ffn.gate.tid2eid",
            ),
            # Routed expert MLP (w1=gate, w3=up, w2=down in V4 naming)
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="layers.*.ffn.experts.*.w1.weight",
                up="layers.*.ffn.experts.*.w3.weight",
            ),
            AutoMapping(
                "decoder.layers.*.mlp.experts.linear_fc2.weight*",
                "layers.*.ffn.experts.*.w2.weight",
            ),
            # Shared expert MLP
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                gate="layers.*.ffn.shared_experts.w1.weight",
                up="layers.*.ffn.shared_experts.w3.weight",
            ),
            AutoMapping(
                "decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                "layers.*.ffn.shared_experts.w2.weight",
            ),
            # Hyper-Connections: attn HC (HyperConnectionModule not in AutoMapping registry → replicated)
            ReplicatedMapping(
                "decoder.layers.*.self_attention_hyper_connection.mapping_proj.weight",
                "layers.*.hc_attn_fn",
            ),
            ReplicatedMapping(
                "decoder.layers.*.self_attention_hyper_connection.bias",
                "layers.*.hc_attn_base",
            ),
            # Hyper-Connections: FFN HC
            ReplicatedMapping(
                "decoder.layers.*.mlp_hyper_connection.mapping_proj.weight",
                "layers.*.hc_ffn_fn",
            ),
            ReplicatedMapping(
                "decoder.layers.*.mlp_hyper_connection.bias",
                "layers.*.hc_ffn_base",
            ),
        ]

        # HC alpha scalars need custom concatenation mapping (per-layer, both attn and ffn)
        # These are wildcarded across all layers.
        mappings += [
            _HCAlphaMapping(
                megatron_pre="decoder.layers.*.self_attention_hyper_connection.alpha_pre",
                megatron_post="decoder.layers.*.self_attention_hyper_connection.alpha_post",
                megatron_res="decoder.layers.*.self_attention_hyper_connection.alpha_res",
                hf_param="layers.*.hc_attn_scale",
            ),
            _HCAlphaMapping(
                megatron_pre="decoder.layers.*.mlp_hyper_connection.alpha_pre",
                megatron_post="decoder.layers.*.mlp_hyper_connection.alpha_post",
                megatron_res="decoder.layers.*.mlp_hyper_connection.alpha_res",
                hf_param="layers.*.hc_ffn_scale",
            ),
        ]

        # HC alpha secondary: register alpha_post and alpha_res to suppress export warnings
        mappings += [
            _HCAlphaSecondaryMapping(
                "decoder.layers.*.self_attention_hyper_connection.alpha_post",
                "layers.*.hc_attn_scale",
                1,
            ),
            _HCAlphaSecondaryMapping(
                "decoder.layers.*.self_attention_hyper_connection.alpha_res",
                "layers.*.hc_attn_scale",
                2,
            ),
            _HCAlphaSecondaryMapping(
                "decoder.layers.*.mlp_hyper_connection.alpha_post",
                "layers.*.hc_ffn_scale",
                1,
            ),
            _HCAlphaSecondaryMapping(
                "decoder.layers.*.mlp_hyper_connection.alpha_res",
                "layers.*.hc_ffn_scale",
                2,
            ),
        ]

        # ------ MTP layer mappings ------
        # MTP layers mirror the main layer structure under mtp.layers.N.*
        for mtp_idx in range(num_mtp):
            ck_pfx = f"mtp.{mtp_idx}"  # checkpoint prefix
            mg_pfx = f"mtp.layers.{mtp_idx}"  # Megatron prefix

            # Standard transformer weights (shared pattern with main layers)
            _mtp_plain = [
                (f"{mg_pfx}.mtp_model_layer.input_layernorm.weight", f"{ck_pfx}.attn_norm.weight"),
                (f"{mg_pfx}.mtp_model_layer.pre_mlp_layernorm.weight", f"{ck_pfx}.ffn_norm.weight"),
                (f"{mg_pfx}.mtp_model_layer.self_attention.linear_q_down_proj.weight", f"{ck_pfx}.attn.wq_a.weight"),
                (f"{mg_pfx}.mtp_model_layer.self_attention.q_layernorm.weight", f"{ck_pfx}.attn.q_norm.weight"),
                (f"{mg_pfx}.mtp_model_layer.self_attention.linear_q_up_proj.weight", f"{ck_pfx}.attn.wq_b.weight"),
                (f"{mg_pfx}.mtp_model_layer.self_attention.linear_kv_up_proj.weight", f"{ck_pfx}.attn.wkv.weight"),
                (f"{mg_pfx}.mtp_model_layer.self_attention.kv_layernorm.weight", f"{ck_pfx}.attn.kv_norm.weight"),
                (f"{mg_pfx}.mtp_model_layer.self_attention.linear_proj.weight", f"{ck_pfx}.attn.wo_b.weight"),
                (f"{mg_pfx}.mtp_model_layer.mlp.router.weight", f"{ck_pfx}.ffn.gate.weight"),
                (f"{mg_pfx}.mtp_model_layer.mlp.router.expert_bias", f"{ck_pfx}.ffn.gate.bias"),
                (f"{mg_pfx}.mtp_model_layer.mlp.router.tid2eid", f"{ck_pfx}.ffn.gate.tid2eid"),
                (
                    f"{mg_pfx}.mtp_model_layer.mlp.shared_experts.linear_fc2.weight",
                    f"{ck_pfx}.ffn.shared_experts.w2.weight",
                ),
                # MTP-specific norms / projections
                (f"{mg_pfx}.enorm.weight", f"{ck_pfx}.enorm.weight"),
                (f"{mg_pfx}.hnorm.weight", f"{ck_pfx}.hnorm.weight"),
                (f"{mg_pfx}.final_layernorm.weight", f"{ck_pfx}.norm.weight"),
            ]
            # MTP HC params use ReplicatedMapping (HyperConnectionModule not in AutoMapping registry)
            _mtp_hc_plain = [
                (
                    f"{mg_pfx}.mtp_model_layer.self_attention_hyper_connection.mapping_proj.weight",
                    f"{ck_pfx}.hc_attn_fn",
                ),
                (f"{mg_pfx}.mtp_model_layer.self_attention_hyper_connection.bias", f"{ck_pfx}.hc_attn_base"),
                (f"{mg_pfx}.mtp_model_layer.mlp_hyper_connection.mapping_proj.weight", f"{ck_pfx}.hc_ffn_fn"),
                (f"{mg_pfx}.mtp_model_layer.mlp_hyper_connection.bias", f"{ck_pfx}.hc_ffn_base"),
                # MTP HC head
                (f"{mg_pfx}.hc_head_fn.weight", f"{ck_pfx}.hc_head_fn"),
                (f"{mg_pfx}.hc_head_base", f"{ck_pfx}.hc_head_base"),
                (f"{mg_pfx}.hc_head_scale", f"{ck_pfx}.hc_head_scale"),
            ]
            for mg, hf in _mtp_plain:
                mappings.append(AutoMapping(mg, hf))
            for mg, hf in _mtp_hc_plain:
                mappings.append(ReplicatedMapping(mg, hf))
            # MTP attn_sink: TP-split like the main model attn_sink
            mappings.append(
                ColumnParallelMapping(
                    f"{mg_pfx}.mtp_model_layer.self_attention.core_attention.attn_sink",
                    f"{ck_pfx}.attn.attn_sink",
                )
            )
            # linear_o_group_proj is a plain nn.Parameter (all o_groups on every TP rank)
            mappings.append(
                ReplicatedMapping(
                    f"{mg_pfx}.mtp_model_layer.self_attention.linear_o_group_proj",
                    f"{ck_pfx}.attn.wo_a.weight",
                )
            )

            # MTP e_proj + h_proj → eh_proj (mathematical equivalence via concatenation)
            mappings.append(
                _MTPEHProjMapping(
                    megatron_eh=f"{mg_pfx}.eh_proj.weight",
                    hf_e=f"{ck_pfx}.e_proj.weight",
                    hf_h=f"{ck_pfx}.h_proj.weight",
                )
            )

            # MTP gated MLP (routed experts + shared expert)
            mappings += [
                GatedMLPMapping(
                    megatron_param=f"{mg_pfx}.mtp_model_layer.mlp.experts.linear_fc1.weight*",
                    gate=f"{ck_pfx}.ffn.experts.*.w1.weight",
                    up=f"{ck_pfx}.ffn.experts.*.w3.weight",
                ),
                AutoMapping(
                    f"{mg_pfx}.mtp_model_layer.mlp.experts.linear_fc2.weight*",
                    f"{ck_pfx}.ffn.experts.*.w2.weight",
                ),
                GatedMLPMapping(
                    megatron_param=f"{mg_pfx}.mtp_model_layer.mlp.shared_experts.linear_fc1.weight",
                    gate=f"{ck_pfx}.ffn.shared_experts.w1.weight",
                    up=f"{ck_pfx}.ffn.shared_experts.w3.weight",
                ),
            ]

            # MTP HC alpha scalars
            mappings += [
                _HCAlphaMapping(
                    megatron_pre=f"{mg_pfx}.mtp_model_layer.self_attention_hyper_connection.alpha_pre",
                    megatron_post=f"{mg_pfx}.mtp_model_layer.self_attention_hyper_connection.alpha_post",
                    megatron_res=f"{mg_pfx}.mtp_model_layer.self_attention_hyper_connection.alpha_res",
                    hf_param=f"{ck_pfx}.hc_attn_scale",
                ),
                _HCAlphaMapping(
                    megatron_pre=f"{mg_pfx}.mtp_model_layer.mlp_hyper_connection.alpha_pre",
                    megatron_post=f"{mg_pfx}.mtp_model_layer.mlp_hyper_connection.alpha_post",
                    megatron_res=f"{mg_pfx}.mtp_model_layer.mlp_hyper_connection.alpha_res",
                    hf_param=f"{ck_pfx}.hc_ffn_scale",
                ),
            ]

            # MTP HC alpha secondary: suppress export warnings for post/res
            for _hc_mg_sub, _hc_hf_key in [
                ("self_attention_hyper_connection", "hc_attn_scale"),
                ("mlp_hyper_connection", "hc_ffn_scale"),
            ]:
                mappings += [
                    _HCAlphaSecondaryMapping(
                        f"{mg_pfx}.mtp_model_layer.{_hc_mg_sub}.alpha_post",
                        f"{ck_pfx}.{_hc_hf_key}",
                        1,
                    ),
                    _HCAlphaSecondaryMapping(
                        f"{mg_pfx}.mtp_model_layer.{_hc_mg_sub}.alpha_res",
                        f"{ck_pfx}.{_hc_hf_key}",
                        2,
                    ),
                ]

        return MegatronMappingRegistry(*mappings)

    # ------------------------------------------------------------------
    # Export: synthesise inv_freq (keeps roundtrip HF compat)
    # ------------------------------------------------------------------

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """No-op for V4: the checkpoint does not contain inv_freq tensors."""
        return converted_weights_dict
