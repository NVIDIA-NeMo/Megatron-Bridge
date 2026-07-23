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

"""Bridge for the DeepSeek-V4 model family.

The bridge covers DeepSeek-V4 variants that share the ``deepseek_v4`` HF config
schema. It derives dimension- and layer-dependent fields from the HF config and
dispatches checkpoint import by tensor dtype so FP8 and FP8+MXFP4 formats can
share the same conversion path.

Checkpoint format notes: DeepSeek-V4 uses a custom serialisation format that
differs from standard HuggingFace Transformers naming conventions:

  - embed.weight            (not model.embed_tokens.weight)
  - head.weight             (not lm_head.weight)
  - norm.weight             (not model.norm.weight)
  - layers.N.attn_norm.weight / layers.N.ffn_norm.weight
  - layers.N.attn.wq_a / wq_b / wkv / wo_a / wo_b …
  - layers.N.ffn.gate / experts / shared_experts …
  - layers.N.hc_attn_fn / hc_attn_base / hc_attn_scale  (Hyper-Connections)
  - layers.N.hc_ffn_fn  / hc_ffn_base  / hc_ffn_scale
  - hc_head_fn / hc_head_base / hc_head_scale            (global HC head, learned output contraction)
  - mtp.N.*                                               (MTP layers)

Quantisation schemes: Two on-disk formats coexist in this family. The bridge
dispatches purely on tensor dtype, so the same code path handles both:

  Released variant     Attn / shared experts     Routed experts
  -------------------  ------------------------  ----------------------------
  Flash (post-trained) FP8_E4M3 + F8_E8M0 (...)  MXFP4 packed I8 + F8_E8M0
  Flash-Base / Pro /   FP8_E4M3 + F32  (...)     FP8_E4M3 + F32 (...)
  Pro-Base (raw)

All scale tensors are 128x128 block-tile geometry (scale.shape[i] == ceil(weight.shape[i]/128))
except the MXFP4 expert path, where scale is per-row over 32-element K-tiles.
``maybe_modify_loaded_hf_weight`` flattens both F8_E8M0 and F32 scales to
F32 via ``.to(torch.float32)`` and selects the tile expansion automatically.
All weights are dequantised to bfloat16 during import.

MoE router note: Hash-routing layers (layer_number <= moe_n_hash_layers)
contain a `tid2eid` buffer (int32 vocab→expert lookup table).  Buffers are not
parameters, so Megatron does not expose them via `named_parameters()`.
The bridge handles `tid2eid` via `maybe_modify_loaded_hf_weight()` and
a dedicated `_Tid2EidMapping` that writes it into `state_dict` directly.

Megatron-Core prerequisites:
  - HyperConnectionModule
  - DSv4HybridSelfAttention / CompressedSparseAttention / CSAIndexer / Compressor
  - Hash-routing tid2eid support and SwiGLU clamp
  - Separate MTP e_proj / h_proj modules with hyper-connections
"""

from typing import Dict, Mapping

import torch
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec as _get_exp_attn_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion import quantization_utils
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


_DSV4_LAYER_TYPE_TO_COMPRESS_RATIO = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}

_DSV4_COMPRESS_RATIO_TO_LAYER_TYPE = {
    ratio: layer_type for layer_type, ratio in _DSV4_LAYER_TYPE_TO_COMPRESS_RATIO.items()
}


def deepseek_v4_supports_blackwell_fused_kernels() -> bool:
    """Return whether DSv4 Blackwell-only fused kernels should default on."""
    if not torch.cuda.is_available():
        return False

    major, _minor = torch.cuda.get_device_capability()
    return major >= 10


def deepseek_v4_supports_fused_dsa_kernels() -> bool:
    """Return whether DSv4 fused DSA kernels can be enabled."""
    try:
        from cudnn import DSA  # noqa: F401
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except ImportError:
        return False

    return True


def set_deepseek_v4_pipeline_model_parallel_layout(model_cfg: MLAModelProvider) -> None:
    """Set an even DSv4 pipeline layout with MTP and loss on the last stage.

    DeepSeek-V4 uses hash-routed MoE layers that must co-locate with the
    embedding on the first pipeline stage, so an explicit
    ``pipeline_model_parallel_layout`` is required whenever
    ``pipeline_model_parallel_size > 1``. This builds an even decoder split with
    the embedding on the first stage and the MTP/loss layers on the last stage.

    Args:
        model_cfg: The DeepSeek-V4 model provider to configure in place.
    """
    pp_size = model_cfg.pipeline_model_parallel_size or 1
    if pp_size <= 1:
        model_cfg.pipeline_model_parallel_layout = None
        return

    num_layers = int(getattr(model_cfg, "num_layers", 0) or 0)
    if num_layers <= 0:
        model_cfg.pipeline_model_parallel_layout = None
        return

    mtp_layers = int(getattr(model_cfg, "mtp_num_layers", 0) or 0)
    base_layers, extra_layers = divmod(num_layers, pp_size)
    layout: list[list[str]] = []
    for pp_rank in range(pp_size):
        stage: list[str] = []
        if pp_rank == 0:
            stage.append("embedding")

        decoder_layers = base_layers + int(pp_rank < extra_layers)
        stage.extend(["decoder"] * decoder_layers)

        if pp_rank == pp_size - 1:
            stage.extend(["mtp"] * mtp_layers)
            stage.append("loss")
        layout.append(stage)

    model_cfg.pipeline_model_parallel_layout = layout


def _dsv4_num_hash_layers(hf_config) -> int:
    num_hash_layers = getattr(hf_config, "num_hash_layers", None)
    if num_hash_layers is not None:
        return int(num_hash_layers)

    mlp_layer_types = getattr(hf_config, "mlp_layer_types", None)
    if mlp_layer_types is None:
        return 0

    n_hash = 0
    for layer_type in mlp_layer_types:
        if layer_type != "hash_moe":
            break
        n_hash += 1

    if any(layer_type == "hash_moe" for layer_type in mlp_layer_types[n_hash:]):
        raise ValueError("DeepSeek-V4 hash MoE layers must be a contiguous prefix.")

    return n_hash


def _dsv4_compress_ratios(hf_config) -> list[int]:
    num_hidden_layers = int(hf_config.num_hidden_layers)
    num_mtp_layers = int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0)
    expected_len = num_hidden_layers + num_mtp_layers

    compress_ratios = getattr(hf_config, "compress_ratios", None)
    if compress_ratios is not None:
        ratios = [int(ratio) for ratio in compress_ratios]
    else:
        layer_types = getattr(hf_config, "layer_types", None)
        compress_rates = getattr(hf_config, "compress_rates", None)
        if layer_types is None or compress_rates is None:
            raise ValueError(
                "HF config missing 'compress_ratios' and native 'layer_types'/'compress_rates'. "
                "DeepSeek-V4 requires per-layer compression ratios."
            )

        ratios = []
        for layer_type in layer_types:
            if layer_type == "sliding_attention":
                ratios.append(0)
            elif layer_type in compress_rates:
                ratios.append(int(compress_rates[layer_type]))
            elif layer_type in _DSV4_LAYER_TYPE_TO_COMPRESS_RATIO:
                ratios.append(_DSV4_LAYER_TYPE_TO_COMPRESS_RATIO[layer_type])
            else:
                raise ValueError(f"Unsupported DeepSeek-V4 attention layer type: {layer_type!r}")

    if len(ratios) == num_hidden_layers and num_mtp_layers:
        ratios.extend([0] * num_mtp_layers)

    if len(ratios) < expected_len:
        raise ValueError(
            f"DeepSeek-V4 compression ratios length ({len(ratios)}) is shorter than "
            f"num_hidden_layers + num_nextn_predict_layers ({expected_len})."
        )

    return ratios[:expected_len]


def _dsv4_use_mxfp4_export(hf_param: str, weight: torch.Tensor, source_scale: torch.Tensor) -> bool:
    """Routed DSv4 experts use packed MXFP4; all other scaled weights export as FP8."""
    if ".ffn.experts." not in hf_param or ".shared_experts." in hf_param:
        return False
    return quantization_utils.is_mxfp4_e2m1_scale_geometry(weight, source_scale)


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
        # With PP > 1, megatron_module may be None on non-owning ranks,
        # so we broadcast alpha_post and alpha_res alongside alpha_pre.
        post_tensor = megatron_module.alpha_post.detach() if megatron_module is not None else None
        res_tensor = megatron_module.alpha_res.detach() if megatron_module is not None else None
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        post = self.broadcast_from_pp_rank(post_tensor, cache_key=str(self.hf_param) + "_post")
        res = self.broadcast_from_pp_rank(res_tensor, cache_key=str(self.hf_param) + "_res")
        if megatron_weights is None:
            return {}
        megatron_weights = self.maybe_dequantize(megatron_weights)
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
        self.allow_hf_name_mismatch = True  # export is no-op; skip hf_keys check

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
    """Megatron Bridge implementation for DeepSeek-V4 causal language models."""

    # ------------------------------------------------------------------
    # Provider configuration
    # ------------------------------------------------------------------

    @staticmethod
    def generate_pipeline_layout(num_layers: int, pp: int, mtp_layers: int = 1) -> list[list[str]]:
        """Generate a pipeline-parallel layout for DSv4 models.

        DSv4 with hash MoE routing requires an explicit pipeline layout when PP > 1.
        The layout distributes decoder layers across PP stages, placing the embedding
        on the first stage and MTP + loss on the last stage.

        Args:
            num_layers: Number of decoder layers (e.g. 43 for Flash, 61 for Pro).
            pp: Pipeline parallel size.
            mtp_layers: Number of MTP layers (default 1).

        Returns:
            List of lists, where each inner list describes one pipeline stage.
        """
        base, rem = num_layers // pp, num_layers % pp
        layout = []
        for i in range(pp):
            n = base + (1 if i < rem else 0)
            stage = ["decoder"] * n
            if i == 0:
                stage = ["embedding"] + stage
            if i == pp - 1:
                stage = stage + ["mtp"] * mtp_layers + ["loss"]
            layout.append(stage)
        return layout

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config
        use_blackwell_fused_kernels = deepseek_v4_supports_blackwell_fused_kernels()
        use_dsa_kernel_fusion = use_blackwell_fused_kernels and deepseek_v4_supports_fused_dsa_kernels()

        # ---- Attention ----
        provider.experimental_attention_variant = "dsv4_hybrid"
        provider.multi_latent_attention = True
        # V4 uses a heterogeneous per-layer spec (hash vs MLA layers differ);
        # override the default transformer_engine_layer_spec with the experimental
        # attention variant block spec builder.
        # GPTModelProvider IS the TransformerConfig (not cfg.transformer)
        provider.transformer_layer_spec = _get_exp_attn_spec
        provider.qk_layernorm = True
        provider.normalization = "RMSNorm"
        provider.add_bias_linear = False

        # V4 MLA geometry
        # head_dim = 512 (nope_dim + rope_dim = 448 + 64)
        provider.v_head_dim = hf_config.head_dim  # 512
        provider.qk_pos_emb_head_dim = hf_config.qk_rope_head_dim  # 64
        # HF's partial_rotary_factor (0.125) is relative to head_dim (512); the rope split is
        # already fully encoded by qk_pos_emb_head_dim (64). The generic partial_rotary_factor
        # -> rotary_percent mapping would shrink the rope cache to 64*0.125 = 8 dims: the
        # unfused path then silently rotates only 8 of 64 rope dims, and the fused MLA rope
        # kernel reads cos/sin out of bounds (garbage values -> the SFT loss NaN).
        provider.rotary_percent = 1.0
        # qk_head_dim and kv_lora_rank derived automatically in DSv4HybridConfig
        provider.q_lora_rank = hf_config.q_lora_rank  # 1024
        provider.o_groups = hf_config.o_groups  # 8
        provider.o_lora_rank = hf_config.o_lora_rank  # 1024

        # ---- Rotary embeddings (YaRN) ----
        # Two separate RoPE bases in V4:
        #   - compress_rope_theta for compressed-KV layers
        #   - rope_theta for pure sliding-window layers (layers 0,1)
        # Megatron keeps the regular and compressed CSA RoPE bases separately.
        provider.apply_rope_fusion = True
        provider.rope_type = "yarn"
        rope_params = getattr(hf_config, "rope_scaling", None) or getattr(hf_config, "rope_parameters", None) or {}
        if "compress" in rope_params:
            main_rope_params = rope_params.get("main", {})
            compress_rope_params = rope_params["compress"]
        else:
            main_rope_params = rope_params
            compress_rope_params = rope_params
        provider.rotary_base = float(main_rope_params.get("rope_theta", hf_config.rope_theta))  # 10000
        provider.csa_compress_rotary_base = float(
            getattr(hf_config, "compress_rope_theta", compress_rope_params.get("rope_theta", provider.rotary_base))
        )  # 160000
        provider.rotary_scaling_factor = float(compress_rope_params["factor"])  # 16
        provider.original_max_position_embeddings = int(
            compress_rope_params["original_max_position_embeddings"]
        )  # 65536
        provider.beta_fast = float(compress_rope_params.get("beta_fast", 32))
        provider.beta_slow = float(compress_rope_params.get("beta_slow", 1))
        # DSv4 has no mscale in HF config; Set both equal to cancel out (like DSv3).
        provider.mscale = 1.0
        provider.mscale_all_dim = 1.0

        # ---- CSA (Compressed Sparse Attention) ----
        # Legacy configs ship compress_ratios, while native Transformers configs
        # expose layer_types + compress_rates. MCore consumes the flattened list.
        _cr = _dsv4_compress_ratios(hf_config)
        _mtp = getattr(hf_config, "num_nextn_predict_layers", None)
        if _mtp is None:
            import logging

            logging.warning(
                "HF config missing 'num_nextn_predict_layers'; defaulting to 0. "
                "DeepSeek-V4-Flash uses num_nextn_predict_layers=1."
            )
            _mtp = 0
        _expected = hf_config.num_hidden_layers + _mtp
        provider.csa_compress_ratios = _cr[:_expected]
        provider.csa_window_size = hf_config.sliding_window  # 128

        # DSA indexer geometry (matches index_n_heads / index_head_dim / index_topk in config)
        provider.dsa_indexer_n_heads = hf_config.index_n_heads  # 64
        provider.dsa_indexer_head_dim = hf_config.index_head_dim  # 128
        provider.dsa_indexer_topk = hf_config.index_topk  # 512
        provider.apply_dsa_kernel_fusion = use_dsa_kernel_fusion

        # ---- Hyper-Connections (mHC) ----
        provider.enable_hyper_connections = True
        provider.use_fused_mhc = use_blackwell_fused_kernels
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
        provider.moe_n_hash_layers = _dsv4_num_hash_layers(hf_config)  # 3 for DSv4 Flash
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
        num_hidden_layers = hf_cfg.get("num_hidden_layers", getattr(provider, "num_layers", 0))
        num_hash_layers = getattr(provider, "moe_n_hash_layers", 0)
        hf_cfg["num_hash_layers"] = num_hash_layers
        hf_cfg["mlp_layer_types"] = ["hash_moe"] * min(num_hidden_layers, num_hash_layers) + ["moe"] * max(
            0, num_hidden_layers - num_hash_layers
        )
        hf_cfg["swiglu_limit"] = getattr(provider, "activation_func_clamp_value", 0.0)

        compress_ratios = getattr(provider, "csa_compress_ratios", None)
        if compress_ratios is not None:
            num_mtp = hf_cfg.get("num_nextn_predict_layers", 0)
            expected_len = num_hidden_layers + num_mtp
            compress_ratios = list(compress_ratios)
            if len(compress_ratios) == num_hidden_layers and num_mtp:
                compress_ratios = compress_ratios + [0] * num_mtp
            hf_cfg["compress_ratios"] = compress_ratios[:expected_len]
            hf_cfg["layer_types"] = [
                _DSV4_COMPRESS_RATIO_TO_LAYER_TYPE[ratio] for ratio in hf_cfg["compress_ratios"][:num_hidden_layers]
            ]
            hf_cfg["compress_rates"] = {
                "compressed_sparse_attention": _DSV4_LAYER_TYPE_TO_COMPRESS_RATIO["compressed_sparse_attention"],
                "heavily_compressed_attention": _DSV4_LAYER_TYPE_TO_COMPRESS_RATIO["heavily_compressed_attention"],
            }

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

    def maybe_modify_loaded_hf_weight(
        self,
        hf_param,
        hf_state_dict: Mapping[str, torch.Tensor],
    ):
        """Dequantise quantized weights using their accompanying block-scale tensor.

        V4 stores attention/embedding weights as float8_e4m3fn with 128x128-block
        scales, and expert FFN weights as MXFP4 packed (I8, 2 nibbles/byte) with
        F8_E8M0 per-32-element scales.  For dict hf_param (GatedMLPMapping etc.),
        dequantizes each key independently so expert gate/up weights are handled.
        Optional CSA indexer weights may use the legacy flat name or the native
        Transformers scorer submodule name.
        """
        if isinstance(hf_param, str) and ".indexer.weights_proj." in hf_param:
            scorer_param = hf_param.replace(".indexer.weights_proj.", ".indexer.scorer.weights_proj.")
            if scorer_param in hf_state_dict:
                hf_param = scorer_param
        return quantization_utils.maybe_dequantize_hf_quantized_weight(hf_param, hf_state_dict)

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
            ReplicatedMapping("decoder.hc_head_fn", "hc_head_fn"),
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
                "decoder.layers.*.self_attention.linear_kv_proj.weight",
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
            # Sequential (non-grouped) experts <-> per-expert HF (e.g. ModelOpt pruning).
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                gate="layers.*.ffn.experts.*.w1.weight",
                up="layers.*.ffn.experts.*.w3.weight",
            ),
            AutoMapping(
                "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
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
                (f"{mg_pfx}.mtp_model_layer.self_attention.linear_kv_proj.weight", f"{ck_pfx}.attn.wkv.weight"),
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
                # Per-MTP-layer HC head (output contraction); mirrors decoder.hc_head_* mappings.
                (f"{mg_pfx}.hc_head_fn", f"{ck_pfx}.hc_head_fn"),
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

            # MTP e_proj + h_proj are separate ColumnParallelLinear projections
            # when the MTP layer uses hyper-connections.
            # AutoMapping auto-detects ColumnParallelLinear and shards along dim 0.
            mappings += [
                AutoMapping(f"{mg_pfx}.e_proj.weight", f"{ck_pfx}.e_proj.weight"),
                AutoMapping(f"{mg_pfx}.h_proj.weight", f"{ck_pfx}.h_proj.weight"),
            ]

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
    # Export: restore HF quantized weight/scale pairs
    # ------------------------------------------------------------------

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Recreate DSv4 quantized weight/scale pairs expected by the source shard index.

        When ``task.weight_dtype`` is set, skip requantization and return the weights
        unchanged — the generic export path casts the dtype.
        """
        if task.weight_dtype is not None:
            return converted_weights_dict
        return quantization_utils.requantize_hf_weight_scale_pairs(
            converted_weights_dict,
            hf_state_dict,
            use_mxfp4=_dsv4_use_mxfp4_export,
        )
