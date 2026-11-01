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

"""NemotronLabsDiffusionAttention for sbd_block_diff diffusion LM training with YARN RoPE.

Context-parallelism design note
===============================
This is the *core* attention submodule (it receives projected Q/K/V and returns
``[s, b, hp]``). The diffusion sequence is doubled to ``[xt | x0]`` (length 2L)
and uses the arbitrary ``sbd_block_diff`` attention pattern (bidirectional within
each xt block, block-causal xt->x0, fully-causal x0).

Why not TE's native context parallelism?
  TEDotProductAttention has built-in CP, but only for causal/padding-family
  masks. The only way to feed an arbitrary mask to TE is as an additive
  ``post_scale_bias`` -- and on this stack (TE 2.14 / cuDNN 9.10 / sm90) the
  ``post_scale_bias + context_parallel`` combination has NO available backend:
    - UnfusedDotProductAttention (the only arbitrary-mask backend) is disabled under CP,
    - FlashAttention never supports post_scale_bias,
    - cuDNN FusedAttention returns NoBackend for bias+CP (verified for p2p/all_gather/a2a,
      and for b1ss/bhss/1hss bias shapes -- it is not a shape issue).
  cuDNN *does* support post_scale_bias WITHOUT CP. (Latest TE main permits
  bias+CP with cp_comm_type="p2p" at the Python layer, but it still depends on a
  cuDNN kernel that 9.10 lacks -- revisit with a newer cuDNN.)

Approach used here ("cuDNN core in cp=1 mode + manual CP collectives"):
  Under CP we do the sequence communication ourselves --
    all-gather Q/K/V across the CP group -> full 2L  (cp_utils.all_gather_seq_cp)
    -> RoPE (per-half) + Llama-4 scale + GQA
    -> TEDotProductAttention built with a cp=1 config copy + dense sbd_block_diff
       post_scale_bias  (cuDNN fused, non-CP path: bias IS supported)
    -> scatter output back to this rank's zigzag slice  (cp_utils.scatter_seq_cp)
  The model input is zigzag-sharded and logits are re-gathered in DGPTStep; the
  loss is restricted to each rank's owned positions so Megatron's standard CP
  loss/grad reduction stays valid.

Trade-offs: each rank runs the full-2L attention (cp_size x attention FLOPs; no
comm/compute overlap), and the bias is a dense ``[1,1,2L,2L]`` tensor (O((2L)^2)
memory -- fine at short context, costly at 128k). Verified bit-exact forward and
numerically-equivalent gradients (cp=1 vs cp=2 vs cp=4) via the CP-sharding round-trip
tests in ``tests/unit_tests/diffusion/common/test_cp_utils.py``.
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
from torch import Tensor
from transformers import ROPE_INIT_FUNCTIONS

from megatron.bridge.diffusion.common.cp_utils import all_gather_seq_cp, scatter_seq_cp
from megatron.bridge.diffusion.common.dllm import compute_block_bias


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
    """RoPE with YARN support, driven by HF ``rope_parameters`` config."""

    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

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
        self.original_inv_freq = inv_freq

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
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
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
    ):
        super().__init__(config=config)
        self.config = config

        # Context parallelism: cuDNN's CP-integrated (ring) attention has no kernel
        # for an arbitrary post_scale_bias on this stack (TE 2.14 / cuDNN 9.10).
        # But cuDNN DOES support post_scale_bias WITHOUT CP. So we do the CP
        # collectives ourselves -- all-gather Q/K/V to the full 2L sequence, run
        # TEDotProductAttention in cp=1 mode with the dense sbd_block_diff bias,
        # then scatter the output back to this rank's zigzag slice (see forward).
        # cuDNN-backed; costs cp_size x attention compute (each rank does full 2L).
        self.cp_size = config.context_parallel_size
        assert not config.apply_query_key_layer_scaling, (
            "softmax_scale is passed to the TE core directly; apply_query_key_layer_scaling "
            "must be False (the model uses Llama-4 style query scaling instead)."
        )

        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
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

        # RoPE setup (always required)
        hf_text_config = getattr(config.hf_config, "text_config", config.hf_config)
        hf_text_config.max_position_embeddings = config.seq_length
        self.rope_embedding_module = Ministral3RotaryEmbedding(hf_text_config)

        # Llama-4 style query scaling (optional)
        self.beta = None
        self.max_position_embeddings = None
        if getattr(config, "apply_llama4_style_query_key_layer_scaling", False):
            self.beta = hf_text_config.rope_parameters["llama_4_scaling_beta"]
            self.max_position_embeddings = hf_text_config.rope_parameters["original_max_position_embeddings"]
            if (
                hasattr(config, "yarn_rotary_scaling_factor")
                and config.yarn_rotary_scaling_factor != hf_text_config.rope_parameters["factor"]
            ):
                hf_text_config.rope_parameters["factor"] = config.yarn_rotary_scaling_factor

        self.block_size = getattr(config, "block_size", 16)

        # TE core attention run WITHOUT CP (cp=1 config copy): cuDNN supports
        # post_scale_bias in the non-CP path. We feed it the full gathered 2L
        # sequence and the dense sbd_block_diff bias; CP comms are done by us.
        core_cfg = copy.copy(config)
        core_cfg.context_parallel_size = 1
        self.core_attention = TEDotProductAttention(
            config=core_cfg,
            layer_number=self.layer_number,
            attn_mask_type=AttnMaskType.no_mask,
            attention_type=attention_type or "self",
            attention_dropout=config.attention_dropout if attention_dropout is None else attention_dropout,
            softmax_scale=self.softmax_scale,
            pg_collection=None,
        )
        # Lazily-built additive sbd_block_diff bias [1, 1, 2L, 2L].
        self._sbd_bias = None

        # AR / causal core: a CP-capable TE attention for the autoregressive
        # (inference_causal) path. Causal attention is `no_bias`, which TE supports
        # with qkv_format="thd" AND context parallelism -- unlike the diffusion
        # dense-bias path. The CP group is supplied per-forward via
        # packed_seq_params.cp_group, so this single instance serves cp=1 and cp>1.
        # Built from the original `config` (keeps its context_parallel_size), not
        # the cp=1 copy used for the bias path.
        # pg_collection=None matches self.core_attention: TE derives tp/cp from the
        # global mpu parallel-state at runtime, and the CP group is overridden per
        # forward by packed_seq_params.cp_group. (Avoids depending on a CP-populated
        # pg_collection at construction.)
        # Constructed LAZILY on first use (see `_get_ar_core_attention`): only the AR /
        # causal inference path needs it; the pure diffusion-training path never does.
        # TEDotProductAttention has no learnable parameters, so late construction has no
        # optimizer / checkpoint / .to() implications.
        self._ar_core_attention = None
        self._ar_core_attention_kwargs = dict(
            config=config,
            layer_number=self.layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type=attention_type or "self",
            attention_dropout=config.attention_dropout if attention_dropout is None else attention_dropout,
            softmax_scale=self.softmax_scale,
            pg_collection=None,
        )

        import torch._dynamo.config as dcfg

        dcfg.cache_size_limit = 512

        # Inference state
        self._inference_mode = False
        self._inference_causal = True
        self._cache_enabled = False
        self._kv_cache_k = None
        self._kv_cache_v = None
        self._kv_cache_seq_len = 0

        # DiffuGRPO / BlockJustGRPO asymmetric [noisy | clean] replay metadata
        # (set per microbatch by the policy worker; None = normal paths).
        self._asym_meta = None

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

    def set_asymmetric_ar_metadata(
        self,
        *,
        noisy_length: int,
        clean_length: int,
        noisy_response_offset: int,
        prompt_lengths: Tensor,
        response_lengths: Tensor,
        noisy_valid_lengths: Tensor,
        clean_lengths: Tensor,
    ) -> None:
        """Enable the asymmetric ``[noisy | clean]`` completion-replay attention.

        The physical microbatch layout is ``[noisy(0..noisy_length) | clean(0..clean_length)]``
        per row: the noisy segment holds the (partially revealed) masked response canvas and
        the clean segment holds the real ``prompt + response`` tokens. This generalizes the
        pretraining ``[xt | x0]`` sbd_block_diff geometry (equal halves, absolute blocks) to
        unequal halves, response-aligned blocks, and per-row prompt offsets:

        - noisy query in response block ``b``: attends bidirectionally within its own noisy
          block (M_BD) and to clean tokens strictly before the block, i.e. clean positions
          ``< prompt_len + b * block_size`` (M_OBC; the prompt acts as block ``-1``);
        - clean query at position ``j``: fully causal over clean (M_FC);
        - clean never attends noisy; RoPE positions are the logical ones (noisy position
          ``i`` maps to ``prompt_len + i - noisy_response_offset``; clean position ``j``
          maps to ``j``).

        Used by DiffuGRPO / BlockJustGRPO workers; cleared per microbatch via
        :meth:`clear_asymmetric_ar_metadata`.
        """
        self._asym_meta = {
            "noisy_length": int(noisy_length),
            "clean_length": int(clean_length),
            "noisy_response_offset": int(noisy_response_offset),
            "prompt_lengths": prompt_lengths,
            "response_lengths": response_lengths,
            "noisy_valid_lengths": noisy_valid_lengths,
            "clean_lengths": clean_lengths,
        }

    def clear_asymmetric_ar_metadata(self) -> None:
        """Disable the asymmetric replay attention (return to normal paths)."""
        self._asym_meta = None

    def _get_ar_core_attention(self):
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
            # AR / causal path: supports packed (thd) sequences + context
            # parallelism through TE (causal is no_bias, which TE handles for thd+CP).
            return self._inference_forward(query, key, value, packed_seq_params)

        if self._asym_meta is not None:
            assert packed_seq_params is None, (
                "Packed sequence is not supported by the asymmetric [noisy | clean] replay path."
            )
            return self._asymmetric_ar_forward(query, key, value)

        # Diffusion (sbd_block_diff) path: the mask is a dense post_scale_bias, for
        # which TE has no THD attention kernel, so packed sequences are unsupported
        # here. CP for diffusion is done unpacked via the gather/scatter below.
        assert packed_seq_params is None, (
            "Packed sequence is not supported by NemotronLabsDiffusionAttention in "
            "diffusion mode (dense sbd_block_diff bias has no THD attention kernel)."
        )

        cp_size = self.cp_size
        cp_group = parallel_state.get_context_parallel_group() if cp_size > 1 else None

        # [local_seq, b, np, hn] -> [b, np, local_seq, hn]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        # Under CP, all-gather Q/K/V to the full doubled 2L sequence (undoing the
        # zigzag) so the cp=1 TE core sees the global sbd_block_diff structure. The
        # output is scattered back to this rank's slice after attention.
        if cp_size > 1:
            query = all_gather_seq_cp(query, cp_group, seq_dim=2)
            key = all_gather_seq_cp(key, cp_group, seq_dim=2)
            value = all_gather_seq_cp(value, cp_group, seq_dim=2)

        # Position ids for each half of the (now full) doubled sequence
        half_seq_len = query.shape[2] // 2
        position_ids = torch.arange(half_seq_len, device=query.device).unsqueeze(0)
        cos, sin = self.rope_embedding_module(query, position_ids)

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

        # GQA is handled inside TEDotProductAttention (num_gqa_groups); kv keeps
        # num_query_groups heads (no repeat_kv).

        # [b, np, seq, hn] -> [seq, b, np, hn] (TE sbhd layout)
        query = query.transpose(1, 2).transpose(0, 1).contiguous()
        key = key.transpose(1, 2).transpose(0, 1).contiguous()
        value = value.transpose(1, 2).transpose(0, 1).contiguous()

        # Dense sbd_block_diff bias [1, 1, 2L, 2L] (post_scale_bias); cuDNN applies
        # it in the non-CP fused path.
        full_2l = query.shape[0]
        if (
            self._sbd_bias is None
            or self._sbd_bias.device != query.device
            or self._sbd_bias.dtype != query.dtype
            or self._sbd_bias.shape[-1] != full_2l
        ):
            self._sbd_bias = compute_block_bias(
                block_size=self.block_size,
                max_seq_length=full_2l // 2,
                dtype=query.dtype,
                device=query.device,
            )

        # cp=1 TE core attention -> [seq, b, hp].
        context = self.core_attention(
            query,
            key,
            value,
            attention_mask=None,
            attn_mask_type=AttnMaskType.no_mask,
            attention_bias=self._sbd_bias,
        )

        # Scatter the full-sequence output back to this rank's zigzag slice.
        if cp_size > 1:
            context = scatter_seq_cp(context, cp_group, seq_dim=0)

        return context

    def _asymmetric_ar_forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """SDPA forward for the asymmetric ``[noisy | clean]`` replay layout.

        See :meth:`set_asymmetric_ar_metadata` for the layout and visibility rules.
        Context parallelism is not supported on this path (run replay with cp=1).
        """
        meta = self._asym_meta
        assert self.cp_size <= 1, (
            "The asymmetric [noisy | clean] replay path does not support context parallelism; "
            "run DiffuGRPO/BlockJustGRPO with context_parallel_size=1."
        )

        # [s, b, np, hn] -> [b, np, s, hn]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        device = query.device
        batch = query.shape[0]
        seq = query.shape[2]
        noisy_len = meta["noisy_length"]
        clean_len_padded = meta["clean_length"]
        offset = meta["noisy_response_offset"]
        assert seq == noisy_len + clean_len_padded, (
            f"asymmetric replay expects seq == noisy_length + clean_length "
            f"({noisy_len} + {clean_len_padded}), got {seq}"
        )
        block = int(self.block_size)
        prompt_lengths = meta["prompt_lengths"].to(device=device, dtype=torch.long).view(batch, 1)
        noisy_valid = meta["noisy_valid_lengths"].to(device=device, dtype=torch.long).view(batch, 1)
        clean_valid = meta["clean_lengths"].to(device=device, dtype=torch.long).view(batch, 1)

        idx = torch.arange(seq, device=device)
        is_noisy = idx < noisy_len
        resp_off = torch.where(is_noisy, (idx - offset).clamp_min(0), idx.new_zeros(())).view(1, seq)
        clean_pos = torch.where(~is_noisy, idx - noisy_len, idx.new_zeros(())).view(1, seq)

        # Logical RoPE positions: noisy i -> prompt_len + (i - offset); clean j -> j.
        positions = torch.where(
            is_noisy.view(1, seq),
            prompt_lengths + resp_off,
            clean_pos,
        )  # [b, seq]

        cos, sin = self.rope_embedding_module(query, positions)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query = (query * cos) + (rotate_half(query) * sin)
        key = (key * cos) + (rotate_half(key) * sin)

        if self.beta is not None:
            scale = _get_llama_4_attn_scale(positions, self.beta, self.max_position_embeddings)
            query = query * scale.unsqueeze(1).to(query.dtype)  # [b, 1, seq, 1]

        # Visibility mask [b, 1, seq, seq] (True = attend), generalizing
        # compute_block_bias's {M_BD, M_OBC, M_FC} to the asymmetric layout.
        q_noisy = is_noisy.view(1, seq, 1)
        k_noisy = is_noisy.view(1, 1, seq)
        q_block = (resp_off // block).view(1, seq, 1)
        k_block = (resp_off // block).view(1, 1, seq)
        k_resp_off = resp_off.view(1, 1, seq)
        q_clean_pos = clean_pos.view(1, seq, 1)
        k_clean_pos = clean_pos.view(1, 1, seq)

        k_noisy_valid = k_resp_off < noisy_valid.view(batch, 1, 1)
        k_clean_valid = k_clean_pos < clean_valid.view(batch, 1, 1)

        # M_BD: noisy -> noisy, same response block, valid keys only.
        block_diag = q_noisy & k_noisy & (q_block == k_block) & k_noisy_valid
        # M_OBC: noisy block b -> clean positions < prompt_len + b*block (prompt = block -1).
        obc = (
            q_noisy
            & ~k_noisy
            & (k_clean_pos < (prompt_lengths.view(batch, 1, 1) + q_block * block))
            & k_clean_valid
        )
        # M_FC: clean -> clean, fully causal.
        fc = ~q_noisy & ~k_noisy & (k_clean_pos <= q_clean_pos)
        allowed = block_diag | obc | fc
        # Padding guard: every query keeps at least its own position (avoids
        # fully-masked rows -> NaN for padded/invalid queries; outputs unused).
        diag = torch.eye(seq, device=device, dtype=torch.bool).view(1, seq, seq)
        allowed = (allowed | diag).unsqueeze(1)  # [b, 1, seq, seq]

        n_rep = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        key = repeat_kv(key, n_rep)
        value = repeat_kv(value, n_rep)

        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
            scale=self.softmax_scale,
        )

        # [b, np, s, hn] -> [s, b, hp]
        context = context.transpose(1, 2).transpose(0, 1)
        new_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        return context.contiguous().view(*new_shape)

    def _inference_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """SDPA-based forward for inference with KV cache support.

        Args:
            query, key, value: [seq_len, batch, num_heads, head_dim]  (Megatron layout)
            packed_seq_params: when not None, the inputs are THD-packed (and possibly
                CP-sharded). Routed to the TE-based packed causal path, which supports
                packing + context parallelism (the SDPA path below is for the
                unpacked / KV-cache decode case only).

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
            assert self._inference_causal, (
                "Packed-sequence AR path requires causal attention (set_inference_params(causal=True))."
            )
            assert not self._cache_enabled, "KV cache is not supported on the packed-sequence path."
            return self._packed_causal_forward(query, key, value, packed_seq_params)

        # Transpose to [b, np, s, hn]
        query = query.transpose(0, 1).transpose(1, 2)
        key = key.transpose(0, 1).transpose(1, 2)
        value = value.transpose(0, 1).transpose(1, 2)

        # Context parallelism (unpacked / full-prefill case, e.g. diffusion
        # leftmost-reveal bidirectional scoring). The input is zigzag-sharded
        # along the sequence; all-gather Q/K/V to the full sequence (natural
        # order) so SDPA sees the global (bidirectional or full-prefill causal)
        # structure, then scatter the output back to this rank's zigzag slice.
        # KV-cache decode is incompatible with this (the gathered sequence is the
        # whole prompt), so it is rejected below.
        cp_size = self.cp_size
        cp_group = parallel_state.get_context_parallel_group() if cp_size > 1 else None
        if cp_size > 1:
            assert not self._cache_enabled, (
                "KV cache is not supported with context parallelism on the "
                "diffusion inference path (set cache_enabled=False)."
            )
            query = all_gather_seq_cp(query, cp_group, seq_dim=2)
            key = all_gather_seq_cp(key, cp_group, seq_dim=2)
            value = all_gather_seq_cp(value, cp_group, seq_dim=2)

        sq = query.shape[2]

        # Position IDs: new tokens start after the cached tokens (offset=0 under CP)
        offset = self._kv_cache_seq_len
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

        # Scatter the full-sequence output back to this rank's zigzag slice.
        if cp_size > 1:
            context = scatter_seq_cp(context, cp_group, seq_dim=0)

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
        cp_group = parallel_state.get_context_parallel_group() if self.cp_size > 1 else None
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
