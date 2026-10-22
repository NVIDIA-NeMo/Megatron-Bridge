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

"""NemotronLabsDiffusionAttention for sbd_block_diff diffusion LM training with YARN RoPE."""

import inspect
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers import ROPE_INIT_FUNCTIONS, PretrainedConfig

from megatron.bridge.diffusion.common.dllm import asymmetric_semi_ar_mask_mod, build_block_mask, compute_block_mask


# ---------------------------------------------------------------------------
# Compiled flex_attention kernel
# ---------------------------------------------------------------------------


_FLEX_SUPPORTS_GQA = "enable_gqa" in inspect.signature(flex_attention).parameters


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs", dynamic=False)
def fused_flex_attention(q, k, v, score_mod=None, block_mask=None, return_lse=False):
    """Run compiled FlexAttention with native GQA when supported by PyTorch."""
    if _FLEX_SUPPORTS_GQA:
        return flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            return_lse=return_lse,
            enable_gqa=q.shape[1] != k.shape[1],
        )
    n_rep = q.shape[1] // k.shape[1]
    return flex_attention(
        q,
        repeat_kv(k, n_rep),
        repeat_kv(v, n_rep),
        score_mod=score_mod,
        block_mask=block_mask,
        return_lse=return_lse,
    )


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def rotate_half(x):
    """Rotate the last half of the hidden dimension for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embeddings to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match query heads for GQA."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _get_llama_4_attn_scale(position_ids: torch.Tensor, beta: float, max_position_embeddings: int) -> torch.Tensor:
    scaling = 1 + beta * torch.log(1 + torch.floor(position_ids / max_position_embeddings))
    return scaling.unsqueeze(-1)


# ---------------------------------------------------------------------------
# YARN-aware Rotary Embedding (supports default + yarn rope_type)
# ---------------------------------------------------------------------------


class Ministral3RotaryEmbedding(nn.Module):
    """RoPE with YARN support; resolved parameters are used without retaining the config."""

    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.rope_type = config.rope_parameters["rope_type"]
        rope_init_fn = self._compute_default_rope_parameters
        if self.rope_type != "default":
            rp = getattr(config, "rope_parameters", {})
            if not hasattr(config, "rope_theta") and "rope_theta" in rp:
                config.rope_theta = rp["rope_theta"]
            if not hasattr(config, "rope_scaling"):
                config.rope_scaling = {k: v for k, v in rp.items() if k != "rope_type"}
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Keep a non-buffer fp32 copy: model dtype casts convert buffers to bf16.
        self.original_inv_freq = inv_freq.detach().clone().to(dtype=torch.float32)

    @classmethod
    def from_megatron_config(cls, config: TransformerConfig) -> "Ministral3RotaryEmbedding":
        """Build RoPE from resolved native fields, without reading or mutating HF metadata."""
        if getattr(config, "hf_config", None) is not None:
            raise ValueError(
                "Legacy NemotronLabsDiffusion provider contains hf_config. Reconvert HF initializers; "
                "existing training checkpoints need their provider config migrated before resuming."
            )
        if config.rope_type not in ("default", "yarn"):
            raise ValueError(f"Unsupported NemotronLabsDiffusion RoPE type: {config.rope_type}")
        rope = {"rope_type": config.rope_type, "rope_theta": config.rotary_base}
        if config.rope_type == "yarn":
            if config.yarn_rotary_scaling_factor <= 0 or config.yarn_original_max_position_embeddings <= 0:
                raise ValueError("YaRN factor and original context length must be positive")
            rope.update(
                factor=config.yarn_rotary_scaling_factor,
                original_max_position_embeddings=config.yarn_original_max_position_embeddings,
                beta_fast=config.yarn_beta_fast,
                beta_slow=config.yarn_beta_slow,
                mscale=config.yarn_mscale,
                mscale_all_dim=config.yarn_mscale_all_dim,
                attention_factor=config.yarn_attention_factor,
                truncate=config.yarn_correction_range_round_to_int,
            )
        # Transformers implements the frequency calculation. This private adapter
        # is derived from native values; it is not the checkpoint's HF config.
        rotary_config = PretrainedConfig(
            rope_parameters=rope,
            rope_theta=config.rotary_base,
            max_position_embeddings=config.seq_length,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            head_dim=config.kv_channels,
        )
        return cls(rotary_config)

    @staticmethod
    def _compute_default_rope_parameters(config=None, device=None, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = getattr(self, "original_inv_freq", self.inv_freq)
        inv_freq_expanded = (
            inv_freq[None, :, None].to(device=x.device, dtype=torch.float32).expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# NemotronLabsDiffusionAttention  (sbd_block_diff only)
# ---------------------------------------------------------------------------


@dataclass
class AsymmetricARMetadata:
    """Per-microbatch layout shared by all attention layers.

    Call ``build_asymmetric_ar_metadata`` for validation. Treat input tensors as
    immutable until every forward and recomputed backward using them completes.
    ``block_mask`` is built lazily once, then reused across layers.
    """

    noisy_length: int
    clean_length: int
    noisy_response_offset: int
    prompt_lengths: Tensor
    response_lengths: Tensor
    noisy_valid_lengths: Tensor
    clean_lengths: Tensor
    block_size: int
    block_mask: BlockMask | None = None


class NemotronLabsDiffusionAttention(MegatronModule):
    """NemotronLabsDiffusionAttention for semi-block-diffusion (sbd_block_diff) training.

    The sequence is doubled to ``[xt | x0]`` where xt are noised tokens and x0
    are clean tokens.  RoPE is applied independently to each half.  Llama-4
    style query-key layer scaling is applied when configured.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
        rope_module: Ministral3RotaryEmbedding | None = None,
    ):
        super().__init__(config=config)
        self.config = config

        self.cp_size = config.context_parallel_size

        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        else:
            assert hasattr(pg_collection, "tp"), (
                "NemotronLabsDiffusionAttention pg_collection must have tp process group"
            )

        world_size = pg_collection.tp.size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(config.num_query_groups, world_size)

        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        self.attention_dropout = torch.nn.Dropout(
            config.attention_dropout if attention_dropout is None else attention_dropout
        )

        self.rope_embedding_module = (
            rope_module if rope_module is not None else Ministral3RotaryEmbedding.from_megatron_config(config)
        )

        self.beta = None
        self.max_position_embeddings = None
        if config.apply_llama4_style_query_key_layer_scaling:
            if config.llama4_scaling_beta is None or config.yarn_original_max_position_embeddings <= 0:
                raise ValueError("Llama-4 query scaling requires beta and a positive original context length")
            self.beta = config.llama4_scaling_beta
            self.max_position_embeddings = config.yarn_original_max_position_embeddings

        # CP-capable causal attention has no parameters; construct it on first use.
        # Keep the CP=1 SDPA path unchanged.
        self._ar_core_attention = None
        self._ar_core_attention_kwargs = dict(
            config=config,
            layer_number=self.layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type=attention_type or "self",
            attention_dropout=0.0,
            softmax_scale=self.softmax_scale,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        self.block_size = getattr(config, "block_size", 16)
        self._asymmetric_ar_metadata: AsymmetricARMetadata | None = None

        # AR attention does not use the doubled-sequence diffusion mask.
        # Build it only when the diffusion forward path is first used.
        self.mask = None

        import torch._dynamo.config as dcfg

        dcfg.cache_size_limit = 512

        # Inference state
        self._inference_mode = False
        self._inference_causal = True
        self._cache_enabled = False
        self._kv_cache_k = None
        self._kv_cache_v = None
        self._kv_cache_seq_len = 0

    def build_asymmetric_ar_metadata(
        self,
        *,
        noisy_length: int,
        clean_length: int,
        noisy_response_offset: int,
        prompt_lengths: Tensor,
        response_lengths: Tensor,
        noisy_valid_lengths: Tensor,
        clean_lengths: Tensor,
    ) -> AsymmetricARMetadata:
        """Validate a compact ``[noisy response | clean prompt + response]`` layout.

        The noisy block grid starts at ``noisy_response_offset``; its logical
        positions start at each sample's ``prompt_lengths``. ``response_lengths``
        excludes the masked tail of the final block, while ``noisy_valid_lengths``
        includes it. Clean lengths exclude right padding. This path supports
        unpacked CP=1 sequences, like the existing symmetric attention path.
        """
        tensors = (prompt_lengths, response_lengths, noisy_valid_lengths, clean_lengths)
        if any(t.ndim != 1 or t.shape != prompt_lengths.shape for t in tensors):
            raise ValueError("Asymmetric metadata must contain matching 1D tensors")
        if any(t.dtype != torch.long or t.device != prompt_lengths.device for t in tensors):
            raise ValueError("Asymmetric metadata must use int64 tensors on one device")
        if noisy_length <= 0 or clean_length <= 0 or not 0 <= noisy_response_offset < noisy_length:
            raise ValueError("Invalid asymmetric sequence lengths or noisy response offset")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if any(bool((t < 0).any()) for t in tensors):
            raise ValueError("Asymmetric lengths must be nonnegative")
        if bool((clean_lengths > clean_length).any()) or bool(
            (prompt_lengths + response_lengths > clean_lengths).any()
        ):
            raise ValueError("Prompt and response must fit the valid clean sequence")
        expected = (response_lengths + self.block_size - 1) // self.block_size * self.block_size
        if not torch.equal(noisy_valid_lengths, expected):
            raise ValueError("noisy_valid_lengths must round responses up to complete blocks")
        if bool((noisy_response_offset + noisy_valid_lengths > noisy_length).any()):
            raise ValueError("Noisy response canvas exceeds noisy_length")
        return AsymmetricARMetadata(
            noisy_length,
            clean_length,
            noisy_response_offset,
            prompt_lengths,
            response_lengths,
            noisy_valid_lengths,
            clean_lengths,
            self.block_size,
        )

    def set_asymmetric_ar_metadata(self, metadata: AsymmetricARMetadata) -> None:
        """Enable response-relative attention for the next microbatch.

        Share the object returned by ``build_asymmetric_ar_metadata`` across all
        layers. Replace it for every microbatch; clear it before symmetric SFT.
        """
        if metadata.block_size != self.block_size:
            raise ValueError("Asymmetric metadata block size must match the model")
        self._asymmetric_ar_metadata = metadata

    def clear_asymmetric_ar_metadata(self) -> None:
        """Restore the default symmetric diffusion training layout."""
        self._asymmetric_ar_metadata = None

    def _asymmetric_semi_ar_forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        metadata = self._asymmetric_ar_metadata
        if metadata is None:
            raise RuntimeError("Asymmetric metadata has not been set")
        n, c = metadata.noisy_length, metadata.clean_length
        if query.shape[:2] != (n + c, metadata.prompt_lengths.numel()):
            raise ValueError("Query sequence and batch dimensions must match asymmetric metadata")
        if query.device != metadata.prompt_lengths.device:
            raise ValueError("Asymmetric metadata must be on the query device")
        query, key, value = (t.permute(1, 2, 0, 3) for t in (query, key, value))
        relative = torch.arange(n, device=query.device)[None] - metadata.noisy_response_offset
        valid = (relative >= 0) & (relative < metadata.noisy_valid_lengths[:, None])
        noisy_positions = torch.where(valid, metadata.prompt_lengths[:, None] + relative, 0)
        clean_positions = torch.arange(c, device=query.device)[None].expand(query.shape[0], -1)
        positions = torch.cat((noisy_positions, clean_positions), dim=1)
        cos, sin = self.rope_embedding_module(query, positions)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if self.beta is not None:
            scale = _get_llama_4_attn_scale(positions, self.beta, self.max_position_embeddings)
            query = query * scale.to(query.dtype).unsqueeze(1)
        if metadata.block_mask is None:
            mask_mod = asymmetric_semi_ar_mask_mod(
                block_size=self.block_size,
                noisy_length=n,
                noisy_response_offset=metadata.noisy_response_offset,
                prompt_lengths=metadata.prompt_lengths,
                noisy_valid_lengths=metadata.noisy_valid_lengths,
                clean_lengths=metadata.clean_lengths,
            )
            metadata.block_mask = build_block_mask(
                mask_mod,
                batch_size=query.shape[0],
                query_length=n + c,
                key_value_length=n + c,
                device=query.device,
            )
        context = fused_flex_attention(query, key, value, block_mask=metadata.block_mask)
        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context = self.attention_dropout(context)
        else:
            context = self.attention_dropout(context)
        return context.permute(2, 0, 1, 3).contiguous().view(n + c, query.shape[0], self.hidden_size_per_partition)

    def set_inference_mode(self, enabled: bool):
        """Enable or disable inference mode. Clears cache on disable."""
        self._inference_mode = enabled
        if not enabled:
            self.clear_kv_cache()

    def set_inference_params(self, causal: bool, cache_enabled: bool):
        self._inference_causal = causal
        self._cache_enabled = cache_enabled

    def clear_kv_cache(self):
        self._kv_cache_k = None
        self._kv_cache_v = None
        self._kv_cache_seq_len = 0

    def _get_ar_core_attention(self) -> TEDotProductAttention:
        """Lazily build the CP-capable causal TE core on first use (AR/causal path only).

        Kept out of ``__init__`` so the pure diffusion-training path never constructs it.
        ``TEDotProductAttention`` holds no learnable parameters, so late construction is
        safe (no optimizer/checkpoint/.to() implications).
        """
        if self._ar_core_attention is None:
            self._ar_core_attention = TEDotProductAttention(**self._ar_core_attention_kwargs)
        return self._ar_core_attention

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        if self._inference_mode:
            return self._inference_forward(query, key, value, packed_seq_params)

        if self.cp_size > 1:
            raise ValueError("Context parallelism requires the packed causal AR attention path")
        assert packed_seq_params is None, "Packed sequence is not supported in diffusion attention mode"

        if self._asymmetric_ar_metadata is not None:
            return self._asymmetric_semi_ar_forward(query, key, value)

        # Position ids for each half of the doubled sequence
        half_seq_len = query.shape[0] // 2
        position_ids = torch.arange(half_seq_len, device=query.device).unsqueeze(0)
        cos, sin = self.rope_embedding_module(query, position_ids)

        # [sq, b, np, hn] -> [b, np, sq, hn]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        # Apply RoPE independently to each half (xt and x0)
        q1, q2 = query.chunk(2, dim=2)
        k1, k2 = key.chunk(2, dim=2)
        q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
        q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)
        query = torch.cat([q1, q2], dim=2)
        key = torch.cat([k1, k2], dim=2)

        # Llama-4 attention scaling
        if self.beta is not None:
            cache_position = torch.arange(query.shape[2], device=query.device)
            query = query * _get_llama_4_attn_scale(cache_position, self.beta, self.max_position_embeddings).to(
                query.dtype
            )

        if self.mask is None:
            self.mask = compute_block_mask(
                block_size=self.block_size,
                max_seq_length=self.config.seq_length,
            )
        context = fused_flex_attention(query, key, value, block_mask=self.mask)

        # Dropout
        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context = self.attention_dropout(context)
        else:
            context = self.attention_dropout(context)

        # [b, np, sq, hn] -> [sq, b, hp]
        context = context.transpose(1, 2).transpose(0, 1)
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.contiguous().view(*new_context_shape)

        return context

    def _inference_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> Tensor:
        """SDPA-based forward for inference with KV cache support.

        Args:
            query, key, value: [seq_len, batch, num_heads, head_dim]  (Megatron layout)

        The method:
          1. Computes position IDs accounting for cached tokens
          2. Applies RoPE (same module as training)
          3. Applies Llama-4 attention scaling
          4. Concatenates new K/V with cached K/V
          5. Applies GQA repeat_kv
          6. Runs SDPA with causal or bidirectional mask
          7. Optionally stores the new K/V in cache
        """
        if packed_seq_params is not None:
            if not self._inference_causal or self._cache_enabled:
                raise ValueError("Packed AR attention requires causal=True and cache_enabled=False")
            if packed_seq_params.qkv_format != "thd":
                raise ValueError("Packed AR attention requires qkv_format='thd'")
            return self._packed_causal_forward(query, key, value, packed_seq_params)
        if self.cp_size > 1 and (not self._inference_causal or self._cache_enabled):
            raise ValueError("Unpacked CP attention requires causal=True and cache_enabled=False")
        sq = query.shape[0]

        # Transpose to [b, np, s, hn]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        # Position IDs: new tokens start after the cached tokens
        offset = self._kv_cache_seq_len
        if self.cp_size > 1:
            if sq % 2:
                raise ValueError("CP-local sequence length must be even for zigzag sharding")
            cp_group = self._ar_core_attention_kwargs["pg_collection"].cp
            cp_rank = cp_group.rank()
            segment = sq // 2
            positions = torch.arange(segment, device=query.device)
            # Each rank owns one early and one late segment of every sample.
            q_position_ids = torch.cat(
                (
                    positions + cp_rank * segment,
                    positions + (2 * self.cp_size - cp_rank - 1) * segment,
                )
            ).unsqueeze(0)
            k_position_ids = q_position_ids
        else:
            q_position_ids = torch.arange(offset, offset + sq, device=query.device).unsqueeze(0)
            k_position_ids = torch.arange(offset, offset + sq, device=key.device).unsqueeze(0)

        cos, sin = self.rope_embedding_module(query, q_position_ids)
        cos_k, sin_k = self.rope_embedding_module(key, k_position_ids)

        # Apply RoPE to new Q and K
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        cos_k = cos_k.unsqueeze(1)
        sin_k = sin_k.unsqueeze(1)
        query = (query * cos_q) + (rotate_half(query) * sin_q)
        key = (key * cos_k) + (rotate_half(key) * sin_k)

        # Llama-4 attention scaling on query
        if self.beta is not None:
            scale = _get_llama_4_attn_scale(q_position_ids.squeeze(0), self.beta, self.max_position_embeddings).to(
                query.dtype
            )
            query = query * scale  # broadcast [sq, 1] -> [b, np, sq, hn]

        if self.cp_size > 1:
            # Preserve SBHD batches: TE exchanges K/V across the CP group and
            # applies global causal masking without any packed-sequence metadata.
            return self._get_ar_core_attention()(
                query.permute(2, 0, 1, 3).contiguous(),
                key.permute(2, 0, 1, 3).contiguous(),
                value.permute(2, 0, 1, 3).contiguous(),
                attention_mask=None,
                attn_mask_type=AttnMaskType.causal,
            )

        # Concatenate with KV cache
        if self._kv_cache_k is not None:
            full_key = torch.cat([self._kv_cache_k, key], dim=2)
            full_value = torch.cat([self._kv_cache_v, value], dim=2)
        else:
            full_key = key
            full_value = value

        # Update cache if enabled
        if self._cache_enabled:
            self._kv_cache_k = full_key.detach()
            self._kv_cache_v = full_value.detach()
            self._kv_cache_seq_len = full_key.shape[2]

        # GQA: repeat KV heads to match query heads
        n_rep = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        full_key_expanded = repeat_kv(full_key, n_rep)
        full_value_expanded = repeat_kv(full_value, n_rep)

        sk = full_key_expanded.shape[2]

        # Build attention mask for SDPA
        if not self._inference_causal:
            # Bidirectional: no mask needed
            attn_mask = None
            is_causal = False
        elif sq == sk:
            # Full prefill: use SDPA's built-in causal
            attn_mask = None
            is_causal = True
        else:
            # Decode with KV cache: build explicit causal mask
            q_pos = torch.arange(offset, offset + sq, device=query.device)
            k_pos = torch.arange(sk, device=query.device)
            mask = q_pos[:, None] >= k_pos[None, :]  # [sq, sk]
            attn_mask = torch.zeros(sq, sk, dtype=query.dtype, device=query.device)
            attn_mask.masked_fill_(~mask, float("-inf"))
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, sq, sk]
            is_causal = False

        context = F.scaled_dot_product_attention(
            query,
            full_key_expanded,
            full_value_expanded,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.softmax_scale,
        )

        # Reshape back to Megatron layout: [sq, b, hp]
        context = context.transpose(1, 2).transpose(0, 1)  # [sq, b, np, hn]
        new_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.contiguous().view(*new_shape)
        return context

    @staticmethod
    def _packed_position_ids(
        packed_seq_params: PackedSeqParams,
        total_tokens: int,
        device: torch.device,
        cp_group: "torch.distributed.ProcessGroup | None" = None,
    ) -> Tensor:
        """Per-pack RoPE position ids for THD packing, correct under context parallelism.

        Positions reset to 0 at each packed sub-sequence boundary (so RoPE / Llama-4
        scaling are per-sequence). Under CP, Megatron load-balanced ("zigzag") shards
        each padded pack of length ``Lp`` into ``2*cp`` segments of ``cp_seg = Lp//(2*cp)``;
        this rank holds segments ``[cp_rank]`` (global positions ``cp_rank*cp_seg ...``)
        and ``[2*cp - cp_rank - 1]``. We reproduce that mapping (matching Megatron's
        ``_get_thd_freqs_on_this_cp_rank``) so the local tokens get their true global
        within-pack positions.

        Returns [1, local_total_tokens].
        """
        # Megatron's thd RoPE uses the padded cu_seqlens (full, un-divided by CP).
        cu = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        cp_size = 1 if cp_group is None else cp_group.size()

        if cp_size == 1:
            t = torch.arange(total_tokens, device=device, dtype=cu.dtype)
            pack = (torch.searchsorted(cu, t, right=True) - 1).clamp(min=0)
            return (t - cu[pack]).to(torch.long).unsqueeze(0)

        cp_rank = cp_group.rank()
        chunks = []
        for p in range(cu.numel() - 1):
            lp = int((cu[p + 1] - cu[p]).item())  # full (padded) pack length
            if lp <= 0:
                continue
            if lp % (2 * cp_size) != 0:
                raise ValueError("Packed sequence lengths must be divisible by 2 * context_parallel_size")
            cp_seg = lp // (2 * cp_size)
            fwd = cp_rank * cp_seg
            bwd = (2 * cp_size - cp_rank - 1) * cp_seg
            chunks.append(torch.arange(fwd, fwd + cp_seg, device=device))
            chunks.append(torch.arange(bwd, bwd + cp_seg, device=device))
        pos = (
            torch.cat(chunks).to(torch.long).unsqueeze(0)
            if chunks
            else torch.zeros(1, total_tokens, dtype=torch.long, device=device)
        )
        if pos.numel() != total_tokens:
            raise ValueError("Packed boundaries do not match the CP-local token count")
        return pos

    def _packed_causal_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        packed_seq_params: PackedSeqParams,
    ) -> Tensor:
        """AR / causal forward for THD-packed (and optionally CP-sharded) sequences.

        Causal attention is `no_bias`, which TE supports with qkv_format="thd" and
        context parallelism, so we apply the model's RoPE + Llama-4 scaling with
        per-pack positions and delegate the attention to the CP-capable causal TE
        core. The CP group (if any) is carried in packed_seq_params.cp_group.

        query, key, value: [total, np, hn] (Megatron squeezes the dummy batch dim
        for thd before calling core_attention). A 4D [total, 1, np, hn] caller (e.g.
        unit tests) is also accepted and squeezed.
        """
        # Normalize to 3D THD layout [total, heads, hn].
        if query.dim() == 4:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        total = query.shape[0]

        # Per-pack RoPE (positions reset at each cu_seqlens boundary). Under CP the
        # local tokens are a zigzag slice of the full sequence, so positions must be
        # the true global within-pack positions (see _packed_position_ids).
        cp_group = packed_seq_params.cp_group
        if cp_group is None and self.cp_size > 1:
            cp_group = parallel_state.get_context_parallel_group()
        position_ids = self._packed_position_ids(packed_seq_params, total, query.device, cp_group)  # [1, total]
        cos, sin = self.rope_embedding_module(query, position_ids)
        cos = cos.squeeze(0).unsqueeze(1)
        sin = sin.squeeze(0).unsqueeze(1)
        q = (query * cos) + (rotate_half(query) * sin)  # [total, np, hn]
        k = (key * cos) + (rotate_half(key) * sin)  # [total, n_kv, hn]
        v = value

        # Llama-4 query scaling (per-pack positions); scale [total, 1] -> [total, 1, 1]
        # to broadcast over [total, np, hn].
        if self.beta is not None:
            scale = _get_llama_4_attn_scale(position_ids.squeeze(0), self.beta, self.max_position_embeddings).to(
                q.dtype
            )
            q = q * scale.unsqueeze(-1)

        # q/k/v are already 3D [total, heads, hn] == TE's thd layout. GQA is handled
        # inside TE (num_gqa_groups). TE converts AttnMaskType.causal -> padding_causal
        # for thd and applies per-pack causal masking via cu_seqlens in packed_seq_params.
        context = self._get_ar_core_attention()(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attention_mask=None,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=packed_seq_params,
        )
        # Restore [total, b=1, hidden] (Megatron reshapes thd output to this anyway).
        if context.dim() == 2:
            context = context.unsqueeze(1)
        return context
