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

import functools
import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.activations import fast_gelu
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer import (
    MegatronModule,
    ModuleSpec,
    TransformerConfig,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from torch import Tensor

from megatron.bridge.models.common.te_layers import TERowParallelLinearLayerNorm
from megatron.bridge.models.gemma.modules import EmbeddingScalingMixin, extend_instance
from megatron.bridge.models.gpt_provider import GPTModelProvider


logger = logging.getLogger(__name__)

_HAVE_FLEX_ATTN = False
_flex_attn_func = None
_create_flex_block_mask = None

try:
    from torch.nn.attention.flex_attention import create_block_mask as _flex_mask_candidate
    from torch.nn.attention.flex_attention import flex_attention as _flex_candidate

    _flex_attn_func = torch.compile(_flex_candidate)
    _create_flex_block_mask = _flex_mask_candidate
    _HAVE_FLEX_ATTN = True
    logger.info("Gemma2: PyTorch FlexAttention available — softcap+SWA fused via Triton score_mod.")
    del _flex_candidate, _flex_mask_candidate
except ImportError:
    pass

if not _HAVE_FLEX_ATTN:
    logger.warning("Gemma2: FlexAttention not available — using unfused attention fallback.")


@functools.lru_cache(maxsize=None)
def _get_softcap_score_mod(softcap: float):
    """Return a score_mod closure for the given softcap, cached so all layers share one object.

    torch.compile guards on score_mod identity (id(fn)), so sharing one object across the
    N attention layers avoids N redundant Triton kernel recompilations at startup.
    """

    def _score_mod(score, b, h, q_idx, kv_idx):
        if softcap == 0.0:
            return score
        return softcap * torch.tanh(score / softcap)

    _score_mod.__qualname__ = f"softcap_score_mod_{softcap}"
    return _score_mod


class Gemma2DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models:
    https://arxiv.org/abs/2205.05198 for more details.
    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        **kwargs,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        if self.config.context_parallel_size != 1:
            raise ValueError("Context parallelism is only supported by TEDotProductAttention!")

        self.layer_number = max(1, layer_number)

        self.window_size = None
        if self.layer_number % 2 == 1:
            self.window_size = config.window_size

        self.attention_type = attention_type  # unused for now
        # SWA layers generate an external mask via get_swa() in forward(). With
        # AttnMaskType.causal, FusedScaleMaskSoftmax always takes the fused upper-
        # triangular causal kernel (ScaledUpperTriangMaskedSoftmax) which never reads
        # the mask argument, silently dropping the SWA mask. Switching to arbitrary
        # for SWA layers routes through ScaledMaskedSoftmax, which applies the mask.
        # Even-numbered layers remain causal and keep the fast fused causal path.
        self.attn_mask_type = AttnMaskType.arbitrary if self.window_size is not None else attn_mask_type

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        world_size = self.config.tensor_model_parallel_size
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        self.norm_factor = math.sqrt(config.query_pre_attn_scalar)

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ):
        """Forward.
        Modified from mcore.transformer.dot_product_attention to support Gemma2-specific
        final_logit_softcapping.
        """
        if packed_seq_params is not None:
            raise ValueError(
                "Packed sequence is not supported by DotProductAttention. Use TEDotProductAttention instead."
            )

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # [b, np, sq, sk]
        output_size = (
            query.size(1),
            query.size(2),
            query.size(0),
            key.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query.dtype,
            "mpu",
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )
        # Gemma 2 specific:
        matmul_result = logit_softcapping(matmul_result, self.config.attn_logit_softcapping)

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # sliding window attention: combine SWA mask with any incoming padding mask.
        # Both use True=masked-out; logical OR gives the union of masked positions.
        # get_swa() returns [sq, sk]; the fused CUDA softmax kernel requires a 4D
        # mask [b, np, sq, sk], so we unsqueeze to [1, 1, sq, sk] when there is no
        # padding mask. When a padding mask [b, 1, sq, sk] is present, the | already
        # produces a 4D result via broadcasting.
        # The mask is always generated for SWA layers: attn_mask_type=arbitrary means
        # FusedScaleMaskSoftmax routes through ScaledSoftmax (no causal masking) when
        # mask=None, so omitting the mask for short sequences would drop causal masking
        # entirely. get_swa() encodes causal structure via triu/tril and degenerates to
        # a pure causal mask when the window fully covers the sequence.
        if self.window_size is not None:
            swa_mask = get_swa(query.size(0), key.size(0), self.window_size)
            if attention_mask is None:
                attention_mask = swa_mask.unsqueeze(0).unsqueeze(0)
            else:
                attention_mask = swa_mask | attention_mask

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value.size(1),
            value.size(2),
            query.size(0),
            value.size(3),
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)
        return context


class Gemma2FlexDotProductAttention(Gemma2DotProductAttention):
    """Gemma2 fused attention with native softcap and sliding window support.

    Uses PyTorch FlexAttention (built-in, PyTorch 2.5+) to fuse softcap and SWA into
    a single Triton kernel. Falls back to the unfused parent when a padding
    attention_mask is present (fine-tuning / variable-length batches) or when
    dropout is active. Pretraining always uses the fused path.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        **kwargs,
    ):
        super().__init__(config, layer_number, attn_mask_type, attention_type, attention_dropout, **kwargs)
        # softcap passed directly to the fused kernel; avoids post-hoc tanh rescaling
        self.softcap = float(getattr(config, "attn_logit_softcapping", 0.0) or 0.0)
        # Gemma2 uses 1/sqrt(query_pre_attn_scalar=224), not 1/sqrt(head_dim) — must override
        self.softmax_scale = 1.0 / self.norm_factor
        self.dropout_p = config.attention_dropout if attention_dropout is None else attention_dropout
        # window_size for FlexAttention block_mask: (-1, -1) = full causal; (left, right) = SWA
        self._flex_window_size = (-1, -1) if self.window_size is None else (self.window_size[0], self.window_size[1])

        if _HAVE_FLEX_ATTN:
            self._flex_score_mod = _get_softcap_score_mod(self.softcap)
            self._flex_block_mask_cache: dict = {}

    def _build_flex_block_mask(self, sq: int, sk: int, device: torch.device):
        """Build a FlexAttention block_mask encoding causal + optional SWA."""
        window_left = self._flex_window_size[0]
        if window_left < 0:

            def _mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

        else:
            w = window_left

            def _mask(b, h, q_idx, kv_idx, _w=w):
                return (q_idx >= kv_idx) & (q_idx - kv_idx <= _w)

        return _create_flex_block_mask(_mask, B=None, H=None, Q_LEN=sq, KV_LEN=sk, device=device)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ):
        """Forward: FlexAttention fused path when possible, unfused fallback otherwise."""
        if packed_seq_params is not None:
            raise ValueError(
                "Packed sequence is not supported by DotProductAttention. Use TEDotProductAttention instead."
            )

        dropout_p = self.dropout_p if self.training else 0.0
        fused_eligible = attention_mask is None and dropout_p == 0.0

        if _HAVE_FLEX_ATTN and fused_eligible:
            # FlexAttention path — expects [b, np, sq, hn]
            sq, b, np_heads, hn = query.shape
            q = query.permute(1, 2, 0, 3)
            k = key.permute(1, 2, 0, 3)
            v = value.permute(1, 2, 0, 3)
            cache_key = (sq, key.size(0))
            if cache_key not in self._flex_block_mask_cache:
                self._flex_block_mask_cache[cache_key] = self._build_flex_block_mask(*cache_key, query.device)
            out = _flex_attn_func(
                q,
                k,
                v,
                score_mod=self._flex_score_mod,
                block_mask=self._flex_block_mask_cache[cache_key],
                scale=self.softmax_scale,
                enable_gqa=(k.size(1) != q.size(1)),
            )
            return out.permute(2, 0, 1, 3).contiguous().view(sq, b, np_heads * hn)

        return super().forward(
            query, key, value, attention_mask, attn_mask_type=attn_mask_type, packed_seq_params=None, **kwargs
        )


class Gemma2OutputLayer(ColumnParallelLinear):
    """Extends from ColumnParallelLinear with logit soft capping."""

    def forward(self, *args, **kwargs):
        """Forward with logit soft capping."""
        output, bias = super().forward(*args, **kwargs)
        output = logit_softcapping(output, self.config.final_logit_softcapping)
        return output, bias


def logit_softcapping(logits: torch.Tensor, scale: Optional[float]) -> torch.Tensor:
    """Prevents logits from growing excessively by scaling them to a fixed range"""
    if not scale:
        return logits

    return scale * torch.tanh(logits / scale)


def get_swa(seq_q: int, seq_kv: int, window_size: tuple[int, int]) -> torch.Tensor:
    """Create the equivalent attention mask for SWA in [seq_q, seq_kv] shape"""
    m = torch.ones(seq_q, seq_kv, dtype=torch.bool, device="cuda")
    mu = torch.triu(m, diagonal=seq_kv - seq_q - window_size[0])
    ml = torch.tril(mu, diagonal=seq_kv - seq_q + window_size[1])
    ml = ~ml

    return ml


def gemma2_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Gemma2-specific layer specification."""

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=Gemma2FlexDotProductAttention,  # FlexAttention fast path; falls back to unfused when unavailable
                    linear_proj=TERowParallelLinearLayerNorm,  # post attn RMSNorm
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinearLayerNorm,  # post mlp RMSNorm
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


@dataclass
class Gemma2ModelProvider(GPTModelProvider):
    """Configuration class for Gemma2 models.
    Extends GPTModelProvider with specific settings optimized for Gemma2 architectures.
    Includes configurations for normalization, activation functions, and various
    Gemma2-specific options like attention logit softcapping and sliding window attention.
    """

    # configs that are common across model sizes
    normalization: str = "RMSNorm"
    activation_func: Callable = fast_gelu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 8192
    kv_channels: int = 256
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = True
    # Note: different behavior compared to NeMo 1.0
    # NeMo 1.0 does not set layernorm_zero_centered_gamma and instead adds 1 in the HF -> NeMo conversion script
    # The present implementation is more in line with the official implementation
    layernorm_zero_centered_gamma: bool = True
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 10000

    window_size: tuple[int, int] = (4095, 0)
    vocab_size: int = 256000

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = gemma2_layer_spec

    query_pre_attn_scalar: int = 224
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core Gemma2 model.
        Extends the base configuration with Gemma2-specific embedding scaling and output layer modifications.
        Args:
            pre_process: Whether to include pre-processing in the model
            post_process: Whether to include post-processing in the model
            vp_stage: Virtual pipeline stage
            tokenizer: Tokenizer used with the model
        Returns:
            MCoreGPTModel: Configured Megatron Core GPT model instance
        """
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        # Apply Embedding Scaling for Gemma2: sqrt(hidden_size)
        if is_vp_first_stage(
            vp_stage=vp_stage, vp_size=self.virtual_pipeline_model_parallel_size
        ) and is_pp_first_stage(self._pg_collection.pp):
            extend_instance(model.embedding, EmbeddingScalingMixin)

        # Prevents final logits from growing excessively by scaling them to a fixed range
        if is_vp_last_stage(vp_stage=vp_stage, vp_size=self.virtual_pipeline_model_parallel_size) and is_pp_last_stage(
            self._pg_collection.pp
        ):
            extend_instance(model.output_layer, Gemma2OutputLayer)

        return model
