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

"""Megatron Bridge for Qwen4-Exp (Qwen3.8-Flash-Next).

Qwen4-Exp is the Qwen3.5-MoE hybrid (3 x Gated DeltaNet + 1 x gated attention per block, 512-expert
MoE with a gated shared expert) extended with

* **Gated Residual** hyper connections: 4 residual streams, every sub-layer reads a sigmoid-gated
  mean of the per-stream RMS-normalized streams and writes back with per-stream scalar gates; the
  block output is a learned gated mix of the streams (``hyper_connection_mixer``);
* **Qwen Sparse Attention** on the full-attention layers: an MQA indexer scores mean-pooled key
  blocks and every query attends to its top ``indexer_budget`` tokens plus its own block;
* a **Per-Layer (hashed n-gram) Embedding** injected before one decoder layer, whose 51B-parameter
  table is stored as ``split_ngram_parts`` checkpoint shards;
* a 1-layer MTP head (not bridged: MTP is disabled for the Megatron model).

The checkpoint is a VL model (``Qwen4ExpForConditionalGeneration``); only the language model is
bridged, to :class:`~megatron.core.models.gpt.gpt_model.GPTModel`. The vision tower is not built.
"""

import re
from typing import Callable, Dict, Mapping, Optional

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    GDNConv1dMapping,
    GDNLinearMappingSeparate,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
)
from megatron.bridge.models.conversion.utils import moe_experts_stored_packed
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen.qwen35_bridge import _moe_routed_expert_mappings


_LAYER_TYPE_TO_LINEAR = {"linear_attention": 1, "full_attention": 0, "qwen_sparse_attention": 0}
_PLE_SHARD_RE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$")


def linear_attention_pattern_from_hf(text_config) -> list[int]:
    """Per-layer pattern (1 = Gated DeltaNet, 0 = (sparse) softmax attention).

    ``Qwen4ExpTextConfig.__post_init__`` rewrites the checkpoint's ``full_attention`` entries to
    ``qwen_sparse_attention``, which the generic Qwen3-Next helper does not know about.
    """
    layer_types = getattr(text_config, "layer_types", None)
    if isinstance(layer_types, list) and layer_types:
        try:
            return [_LAYER_TYPE_TO_LINEAR[layer_type] for layer_type in layer_types]
        except KeyError as error:
            raise ValueError(f"Unsupported Qwen4-Exp layer type: {error.args[0]}") from error
    interval = getattr(text_config, "full_attention_interval", 4)
    return [0 if (i + 1) % interval == 0 else 1 for i in range(text_config.num_hidden_layers)]


def get_qwen4_exp_text_config(hf_config):
    """The language-model config of a unified (VL) or text-only Qwen4-Exp config."""
    return hf_config.text_config if getattr(hf_config, "text_config", None) is not None else hf_config


def get_qwen4_exp_hf_lm_prefix(hf_config) -> str:
    """``model.language_model.`` for the VL checkpoint layout, ``model.`` for text-only ones."""
    return "model.language_model." if getattr(hf_config, "text_config", None) is not None else "model."


def ple_local_rows_from_shards(
    load_shard: Callable[[int], torch.Tensor],
    num_shards: int,
    num_rows_total: int,
    shard_rows: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Assemble one tensor-parallel rank's rows of a row-sharded HF table.

    Rank ``r`` owns rows ``[r * rows_per_rank, (r + 1) * rows_per_rank)`` (the
    ``VocabParallelEmbedding`` layout); only the HF shards overlapping that range are loaded.

    Args:
        load_shard: ``callable(shard_index) -> Tensor`` loading one HF shard.
        num_shards: Number of HF shards.
        num_rows_total: Total (padded) row count of the table.
        shard_rows: Rows per HF shard (all but possibly the last shard).
        tp_rank / tp_size: Tensor-parallel coordinates.
    """
    if num_rows_total % tp_size != 0:
        raise ValueError(f"PLE table rows ({num_rows_total}) are not divisible by tp_size={tp_size}")
    rows_per_rank = num_rows_total // tp_size
    start, end = tp_rank * rows_per_rank, (tp_rank + 1) * rows_per_rank
    pieces = []
    for shard_index in range(start // shard_rows, min(num_shards, (end - 1) // shard_rows + 1)):
        shard = load_shard(shard_index)
        shard_start = shard_index * shard_rows
        lo = max(start, shard_start) - shard_start
        hi = min(end, shard_start + shard.shape[0]) - shard_start
        pieces.append(shard[lo:hi])
    rows = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
    if rows.shape[0] != rows_per_rank:
        raise ValueError(f"PLE shards cover {rows.shape[0]} rows for TP rank {tp_rank}, expected {rows_per_rank}")
    return rows


class PLENGramEmbeddingMapping(MegatronParamMapping[Dict[str, torch.Tensor]]):
    """Vocab-parallel PLE n-gram table <-> ``split_ngram_parts`` HF shards.

    The HF checkpoint stores the table row-split into ``ngram_embedding.shard_<i>.weight`` tensors
    (concatenated along dim 0 they form the padded table). Megatron keeps the table as a
    :class:`~megatron.core.tensor_parallel.layers.VocabParallelEmbedding`, i.e. row-sharded across
    the tensor-parallel group.

    Import is rank-local: :meth:`Qwen4ExpBridge.maybe_modify_loaded_hf_weight` loads only the shards
    that overlap this rank's row range (:func:`ple_local_rows_from_shards`), so the 51B-parameter
    table is never materialized in full on any rank. Export streams the table back out one HF shard
    at a time (:class:`_PLEShardExport`): each shard is assembled from the TP ranks' rows when the
    export loop reaches it, so peak memory is one shard (~0.8 GB for Qwen3.8-Flash-Next), not the
    ~100 GB table.
    """

    def __init__(self, megatron_param: str, hf_param: str | Dict[str, str], num_shards: Optional[int] = None):
        """
        Args:
            megatron_param: Megatron parameter name pattern.
            hf_param: Either the HF shard-name pattern with a ``{}`` placeholder for the shard index
                (then ``num_shards`` is required), or the already expanded ``{"shard_<i>": name}``
                dict (the form :meth:`resolve` re-instantiates the mapping with).
            num_shards: Number of HF shards when ``hf_param`` is a pattern.
        """
        if isinstance(hf_param, str):
            if num_shards is None:
                raise ValueError("num_shards is required when hf_param is a shard-name pattern")
            hf_param = {f"shard_{i}": hf_param.format(i) for i in range(num_shards)}
        super().__init__(megatron_param, hf_param)
        self.num_shards = len(hf_param)

    def hf_to_megatron(self, hf_weights: Dict[str, torch.Tensor], megatron_module: nn.Module) -> torch.Tensor:
        """Return this rank's rows.

        ``hf_weights`` is either the rank-local ``{"local_rows": Tensor}`` produced by
        :meth:`Qwen4ExpBridge.maybe_modify_loaded_hf_weight`, or (fallback) the full shard dict.
        """
        if "local_rows" in hf_weights:
            rows = hf_weights["local_rows"]
        else:
            shards = [hf_weights[f"shard_{i}"] for i in range(self.num_shards)]
            total = sum(s.shape[0] for s in shards)
            rows = ple_local_rows_from_shards(
                lambda i: shards[i], self.num_shards, total, shards[0].shape[0], self.tp_rank, self.tp_size
            )
        target = megatron_module.weight
        return rows.to(device=target.device, dtype=target.dtype)

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Mapping[str, torch.Tensor]:
        """Re-split the TP row shards into the checkpoint's shard layout, one shard at a time.

        Returns a lazy mapping: shard ``i`` is assembled (a tensor-parallel collective) when the
        export loop iterates to it, so all TP ranks must consume the shards in the same order --
        which the export does, and which a ``dict(...)`` / ``.items()`` traversal preserves.
        """
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.megatron_param))
        if megatron_weights is None:
            return {}
        return _PLEShardExport(self, megatron_weights)

    def gather_shard(self, local_rows: torch.Tensor, shard_index: int) -> torch.Tensor:
        """Assemble HF shard ``shard_index`` from the tensor-parallel row shards.

        Rank ``r`` owns rows ``[r * rows_per_rank, (r + 1) * rows_per_rank)``; each rank writes
        its overlap with the shard's row range into a zero buffer and the buffers are summed over
        the TP group (exact: every row has exactly one owner).
        """
        rows_per_rank = local_rows.shape[0]
        total_rows = rows_per_rank * self.tp_size
        shard_rows = -(-total_rows // self.num_shards)
        start = shard_index * shard_rows
        end = min(total_rows, start + shard_rows)
        if self.tp_size == 1:
            return local_rows[start:end]
        rank_start = self.tp_rank * rows_per_rank
        lo = max(start, rank_start)
        hi = min(end, rank_start + rows_per_rank)
        shard = torch.zeros(end - start, local_rows.shape[1], dtype=local_rows.dtype, device=local_rows.device)
        if hi > lo:
            shard[lo - start : hi - start] = local_rows[lo - rank_start : hi - rank_start]
        torch.distributed.all_reduce(shard, group=self.tp_group)
        return shard


class _PLEShardExport(Mapping[str, torch.Tensor]):
    """Lazy ``{hf_shard_name: tensor}`` view over a TP-sharded PLE table (see ``gather_shard``)."""

    def __init__(self, mapping: PLENGramEmbeddingMapping, local_rows: torch.Tensor):
        self._mapping = mapping
        self._local_rows = local_rows
        self._names = [mapping.hf_param[f"shard_{i}"] for i in range(mapping.num_shards)]
        self._index = {name: i for i, name in enumerate(self._names)}

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._mapping.gather_shard(self._local_rows, self._index[name])

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


@MegatronModelBridge.register_bridge(
    source="Qwen4ExpForConditionalGeneration", target=GPTModel, model_type="qwen4_exp"
)
class Qwen4ExpBridge(MegatronModelBridge):
    """Megatron Bridge for the Qwen4-Exp language model (Qwen3.8-Flash-Next).

    Bridges ``Qwen4ExpForConditionalGeneration`` (vision + language) and
    ``Qwen4ExpForCausalLM`` (text-only) checkpoints to ``GPTModel``. Only the language model is
    converted; callers that load the VL checkpoint must not need the vision tower
    (``language_model_only``).

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3.8-Flash-Next")
        >>> provider = bridge.to_megatron_provider()
    """

    # Legacy provider construction path only (see MegatronModelBridge.MODEL_CONFIG_CLASS).
    MODEL_CONFIG_CLASS = None

    def provider_bridge(self, hf_pretrained) -> GPTModelProvider:
        """Convert the HuggingFace Qwen4-Exp config into a GPTModelProvider."""
        hf_config = hf_pretrained.config
        text_config = get_qwen4_exp_text_config(hf_config)
        rope_parameters = getattr(text_config, "rope_parameters", None) or {}
        tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False) or getattr(
            text_config, "tie_word_embeddings", False
        )

        provider = GPTModelProvider(
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.moe_intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=text_config.head_dim,
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            layernorm_epsilon=text_config.rms_norm_eps,
            init_method_std=text_config.initializer_range,
            attention_dropout=text_config.attention_dropout,
            hidden_dropout=0.0,
            rotary_base=rope_parameters.get("rope_theta", 10_000_000),
            share_embeddings_and_output_weights=tie_word_embeddings,
        )

        # --- Qwen3.5-MoE base: zero-centered RMSNorm, gated attention with QK norm, GDN hybrid ---
        provider.activation_func = self.hf_to_megatron_activation(text_config.hidden_act)
        provider.normalization = "RMSNorm"
        provider.layernorm_zero_centered_gamma = True
        provider.gated_linear_unit = True
        provider.add_qkv_bias = getattr(text_config, "attention_bias", False)
        provider.add_bias_linear = False
        provider.qk_layernorm = True
        provider.attention_output_gate = True
        provider.experimental_attention_variant = "gdn"
        provider.linear_attention_freq = linear_attention_pattern_from_hf(text_config)
        provider.linear_conv_kernel_dim = text_config.linear_conv_kernel_dim
        provider.linear_key_head_dim = text_config.linear_key_head_dim
        provider.linear_value_head_dim = text_config.linear_value_head_dim
        provider.linear_num_key_heads = text_config.linear_num_key_heads
        provider.linear_num_value_heads = text_config.linear_num_value_heads
        # Qwen4-Exp gates the GDN state output with a sigmoid instead of SiLU.
        provider.linear_attention_output_gate_activation = getattr(text_config, "output_gate_type", None) or (
            "silu" if text_config.hidden_act in ("silu", "swish") else text_config.hidden_act
        )
        provider.rotary_percent = rope_parameters.get("partial_rotary_factor", 0.25)

        # --- MoE: 512 routed experts (top-k, softmax-then-topk), gated shared expert ---
        provider.moe_ffn_hidden_size = text_config.moe_intermediate_size
        provider.num_moe_experts = text_config.num_experts
        provider.moe_router_topk = text_config.num_experts_per_tok
        provider.moe_shared_expert_intermediate_size = text_config.shared_expert_intermediate_size
        provider.moe_shared_expert_gate = True
        provider.moe_grouped_gemm = True
        provider.moe_router_load_balancing_type = "global_aux_loss"
        provider.moe_aux_loss_coeff = getattr(text_config, "router_aux_loss_coef", 0.001)
        provider.moe_router_pre_softmax = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.moe_router_dtype = "fp32"

        # --- Gated Residual hyper connections ---
        provider.enable_mhc_connections = True
        provider.mhc_variant = "gated_residual"
        provider.mhc_num_residual_streams = text_config.hc_count
        provider.mhc_gated_residual_rank = text_config.hc_lowrank

        # --- Qwen Sparse Attention on the softmax-attention layers ---
        if getattr(text_config, "indexer_n_heads", None) is not None:
            provider.qsa_indexer_n_heads = text_config.indexer_n_heads
            provider.qsa_indexer_kv_heads = text_config.indexer_kv_heads
            provider.qsa_indexer_head_dim = text_config.indexer_head_dim
            provider.qsa_indexer_budget = text_config.indexer_budget
            provider.qsa_indexer_compress_ratio = text_config.indexer_compress_ratio

        # --- Per-layer n-gram embedding ---
        ple_layer_ids = list(getattr(text_config, "ple_layer_ids", None) or [])
        if ple_layer_ids:
            eos_token_id = text_config.eos_token_id
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            provider.ple_layer_ids = ple_layer_ids
            provider.ple_embed_dim = getattr(text_config, "ple_embed_dim", None) or text_config.hidden_size
            provider.ple_conv_kernel_size = text_config.ple_conv_kernel_size
            provider.ple_ngram_size = text_config.ngram_size
            provider.ple_heads_per_ngram = text_config.heads_per_ngram
            provider.ple_ngram_vocab_size_base = text_config.ngram_vocab_size_base
            provider.ple_ngram_vocab_divisible_by = text_config.make_ngram_vocab_size_divisible_by
            provider.ple_seed = text_config.seed
            provider.ple_eos_token_id = eos_token_id
            provider.ple_unigram_vocab_size = text_config.vocab_size

        # MTP: the checkpoint carries a 1-layer MTP head; it is not built for the Megatron model.
        provider.mtp_num_layers = None

        provider.position_embedding_type = "rope"
        provider.autocast_dtype = torch.bfloat16
        provider.bos_token_id = getattr(text_config, "bos_token_id", None)
        eos = getattr(text_config, "eos_token_id", None)
        provider.eos_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos
        provider.transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec
        provider.hetereogenous_dist_checkpoint = True
        return provider

    # ------------------------------------------------------------------ mappings
    @staticmethod
    def _hyper_connection_mappings(megatron_prefix: str, hf_prefix: str, with_inject: bool) -> list:
        names = ["hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"]
        if with_inject:
            names.append("block_inject_weight.weight")
        return [ReplicatedMapping(f"{megatron_prefix}{n}", f"{hf_prefix}{n}") for n in names]

    @staticmethod
    def get_lm_mappings(hf_prefix: str, experts_packed: bool, text_config, num_ple_shards: int) -> list:
        """Parameter mappings of the language model (``megatron_prefix`` is empty for GPTModel)."""
        mp = ""
        simple = {
            f"{mp}embedding.word_embeddings.weight": f"{hf_prefix}embed_tokens.weight",
            f"{mp}output_layer.weight": "lm_head.weight",
            # MoE router
            f"{mp}decoder.layers.*.mlp.router.weight": f"{hf_prefix}layers.*.mlp.gate.weight",
            # (Sparse) attention layers
            f"{mp}decoder.layers.*.self_attention.q_layernorm.weight": f"{hf_prefix}layers.*.self_attn.q_norm.weight",
            f"{mp}decoder.layers.*.self_attention.k_layernorm.weight": f"{hf_prefix}layers.*.self_attn.k_norm.weight",
            f"{mp}decoder.layers.*.self_attention.linear_proj.weight": f"{hf_prefix}layers.*.self_attn.o_proj.weight",
            # Gated DeltaNet layers
            f"{mp}decoder.layers.*.self_attention.out_proj.weight": f"{hf_prefix}layers.*.linear_attn.out_proj.weight",
            f"{mp}decoder.layers.*.self_attention.A_log": f"{hf_prefix}layers.*.linear_attn.A_log",
            f"{mp}decoder.layers.*.self_attention.dt_bias": f"{hf_prefix}layers.*.linear_attn.dt_bias",
        }
        mappings = [AutoMapping(megatron_param=m, hf_param=h) for m, h in simple.items()]
        AutoMapping.register_module_type("SharedExpertMLP", "column")
        AutoMapping.register_module_type("GatedDeltaNet", "column")

        mappings.extend(
            [
                QKVMapping(
                    megatron_param=f"{mp}decoder.layers.*.self_attention.linear_qkv.weight",
                    q=f"{hf_prefix}layers.*.self_attn.q_proj.weight",
                    k=f"{hf_prefix}layers.*.self_attn.k_proj.weight",
                    v=f"{hf_prefix}layers.*.self_attn.v_proj.weight",
                ),
                GDNConv1dMapping(
                    megatron_param=f"{mp}decoder.layers.*.self_attention.conv1d.weight",
                    hf_param=f"{hf_prefix}layers.*.linear_attn.conv1d.weight",
                ),
                GDNLinearMappingSeparate(
                    megatron_param=f"{mp}decoder.layers.*.self_attention.in_proj.weight",
                    qkv=f"{hf_prefix}layers.*.linear_attn.in_proj_qkv.weight",
                    z=f"{hf_prefix}layers.*.linear_attn.in_proj_z.weight",
                    b=f"{hf_prefix}layers.*.linear_attn.in_proj_b.weight",
                    a=f"{hf_prefix}layers.*.linear_attn.in_proj_a.weight",
                ),
                # GDN output norm: HF stores a plain RMSNorm gain, Megatron a zero-centered one.
                RMSNorm2ZeroCenteredRMSNormMapping(
                    f"{mp}decoder.layers.*.self_attention.out_norm.weight",
                    f"{hf_prefix}layers.*.linear_attn.norm.weight",
                ),
                *_moe_routed_expert_mappings(hf_prefix, mp, experts_packed),
                GatedMLPMapping(
                    megatron_param=f"{mp}decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate=f"{hf_prefix}layers.*.mlp.shared_expert.gate_proj.weight",
                    up=f"{hf_prefix}layers.*.mlp.shared_expert.up_proj.weight",
                ),
                AutoMapping(
                    megatron_param=f"{mp}decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                    hf_param=f"{hf_prefix}layers.*.mlp.shared_expert.down_proj.weight",
                ),
                ReplicatedMapping(
                    megatron_param=f"{mp}decoder.layers.*.mlp.shared_experts.gate_weight",
                    hf_param=f"{hf_prefix}layers.*.mlp.shared_expert_gate.weight",
                ),
            ]
        )

        # QSA indexer (duplicated across TP ranks).
        if getattr(text_config, "indexer_n_heads", None) is not None:
            for name in ("index_qk_proj.weight", "q_layernorm.weight", "k_layernorm.weight"):
                mappings.append(
                    ReplicatedMapping(
                        f"{mp}decoder.layers.*.self_attention.indexer.{name}",
                        f"{hf_prefix}layers.*.self_attn.indexer.{name}",
                    )
                )

        # Gated Residual hyper connections (per sub-layer) and the final learned stream mixer.
        mappings.extend(
            Qwen4ExpBridge._hyper_connection_mappings(
                f"{mp}decoder.layers.*.self_attention_hyper_connection.",
                f"{hf_prefix}layers.*.attn_hyper_connection.",
                with_inject=True,
            )
        )
        mappings.extend(
            Qwen4ExpBridge._hyper_connection_mappings(
                f"{mp}decoder.layers.*.mlp_hyper_connection.",
                f"{hf_prefix}layers.*.mlp_hyper_connection.",
                with_inject=True,
            )
        )
        mappings.extend(
            Qwen4ExpBridge._hyper_connection_mappings(
                f"{mp}decoder.final_layernorm.", f"{hf_prefix}hyper_connection_mixer.", with_inject=False
            )
        )

        # Per-layer n-gram embedding.
        if getattr(text_config, "ple_layer_ids", None):
            ple_m = f"{mp}decoder.layers.*.per_layer_embedding."
            ple_h = f"{hf_prefix}layers.*.ple."
            for name in (
                "key_proj.weight",
                "value_proj.weight",
                "norm_key.weight",
                "norm_query.weight",
                "norm_conv.weight",
                "conv1d.weight",
                # Hash layout buffers: derived from the config on both sides, stored in both
                # checkpoint formats.
                "ple_embedding.layer_multipliers",
                "ple_embedding.ngram_heads_vocab_sizes",
                "ple_embedding.ngram_heads_offsets",
            ):
                mappings.append(ReplicatedMapping(f"{ple_m}{name}", f"{ple_h}{name}"))
            mappings.append(
                PLENGramEmbeddingMapping(
                    megatron_param=f"{ple_m}ple_embedding.ngram_embedding.weight",
                    hf_param=f"{ple_h}ple_embedding.ngram_embedding.shard_{{}}.weight",
                    num_shards=num_ple_shards,
                )
            )
        return mappings

    def _num_ple_shards(self, text_config) -> int:
        """Number of ``shard_<i>`` tensors per PLE table, from the checkpoint keys when available."""
        source = getattr(getattr(getattr(self, "hf_pretrained", None), "state", None), "source", None)
        if source is not None and hasattr(source, "get_all_keys"):
            found = {int(m.group(1)) for k in source.get_all_keys() if (m := _PLE_SHARD_RE.search(k))}
            if found:
                return max(found) + 1
        return int(getattr(text_config, "split_ngram_parts", 512))

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Parameter mappings of the Qwen4-Exp language model."""
        # `hf_pretrained` is a PreTrainedCausalLM in the AutoBridge flow, but adapter-export
        # callers may hand the bridge the bare HF config instead.
        hf_pretrained = self.hf_pretrained
        hf_config = hf_pretrained.config if hasattr(hf_pretrained, "config") else hf_pretrained
        text_config = get_qwen4_exp_text_config(hf_config)
        hf_prefix = get_qwen4_exp_hf_lm_prefix(hf_config)
        experts_packed = moe_experts_stored_packed(self.hf_pretrained, f"{hf_prefix}layers.", default=True)
        return MegatronMappingRegistry(
            *self.get_lm_mappings(hf_prefix, experts_packed, text_config, self._num_ple_shards(text_config))
        )

    # ------------------------------------------------------------------ rank-local PLE import
    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Load only the PLE table shards overlapping this tensor-parallel rank's row range."""
        if isinstance(hf_param, dict) and hf_param and all(k.startswith("shard_") for k in hf_param):
            num_shards = len(hf_param)
            shard_names = [hf_param[f"shard_{i}"] for i in range(num_shards)]
            if parallel_state.is_initialized():
                tp_rank = parallel_state.get_tensor_model_parallel_rank()
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
            else:
                tp_rank, tp_size = 0, 1
            cache: dict[int, torch.Tensor] = {0: hf_state_dict[shard_names[0]]}
            shard_rows = cache[0].shape[0]
            if num_shards > 1:
                cache[num_shards - 1] = hf_state_dict[shard_names[-1]]
                total = shard_rows * (num_shards - 1) + cache[num_shards - 1].shape[0]
            else:
                total = shard_rows

            def load_shard(i: int) -> torch.Tensor:
                return cache[i] if i in cache else hf_state_dict[shard_names[i]]

            rows = ple_local_rows_from_shards(load_shard, num_shards, total, shard_rows, tp_rank, tp_size)
            return {"local_rows": rows}
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)


@MegatronModelBridge.register_bridge(source="Qwen4ExpForCausalLM", target=GPTModel, model_type="qwen4_exp_text")
class Qwen4ExpTextBridge(Qwen4ExpBridge):
    """Bridge for text-only Qwen4-Exp checkpoints (``Qwen4ExpForCausalLM``, ``model.`` prefix)."""


__all__ = [
    "PLENGramEmbeddingMapping",
    "Qwen4ExpBridge",
    "Qwen4ExpTextBridge",
    "get_qwen4_exp_hf_lm_prefix",
    "get_qwen4_exp_text_config",
    "linear_attention_pattern_from_hf",
    "ple_local_rows_from_shards",
]
