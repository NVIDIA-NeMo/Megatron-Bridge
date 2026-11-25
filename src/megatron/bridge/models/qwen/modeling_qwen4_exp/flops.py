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

"""Logical matrix-work estimates shared by Qwen4-Exp LM and VLM models."""

import warnings
from typing import TYPE_CHECKING

from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


if TYPE_CHECKING:
    from .model_config import Qwen4ExpModelConfig


def estimate_language_flops(
    config: "Qwen4ExpModelConfig",
    *,
    batch_size: int,
    seqlen_sum: float | None = None,
    seqlen_squared_sum: float | None = None,
    include_backward: bool = True,
) -> float:
    """Estimate logical language FLOPs, independent of parallel sharding.

    As in Core's model-owned estimators, trainable matrix work costs three
    forward passes when backward is included. Detached indexer projections
    run only forward. Standard blocks follow the pinned Core conventions:
    GDN recurrence is approximate, QSA uses a dense attention
    surrogate, and indexer scoring, norms, lookups, PLE convolution, router
    scoring, communication and optimizer work are excluded. Freezing the PLE
    lookup table does not remove its trainable projections.
    """
    c = config
    tokens = batch_size * c.seq_length if seqlen_sum is None else seqlen_sum
    squares = tokens * (tokens / batch_size if batch_size else 0) if seqlen_squared_sum is None else seqlen_squared_sum
    if min(batch_size, tokens, squares) < 0:
        raise ValueError("FLOPs require nonnegative batch and sequence statistics.")
    if tokens == 0:
        return 0.0
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.moe.moe_layer import MoELayer

    from .gated_residual import GatedResidualHyperConnection, GatedResidualOutputMixer
    from .per_layer_embedding import PerLayerEmbedding
    from .qsa import QSAIndexer, QwenSparseSelfAttention

    stats = dict(total_real_tokens_in_batch=tokens, seqlen_squared_sum_in_batch=squares)
    transformer = c.transformer
    frequency = c.linear_attention_freq
    gdn_layers = c.num_layers - c.num_layers // frequency if isinstance(frequency, int) else sum(frequency)
    attention_layers = c.num_layers - gdn_layers
    moe_frequency = c.moe_layer_freq
    moe_layers = (
        0
        if c.num_moe_experts is None
        else (len(range(0, c.num_layers, moe_frequency)) if isinstance(moe_frequency, int) else sum(moe_frequency))
    )
    forward = 0.0
    for count, module in (
        (gdn_layers, GatedDeltaNet),
        (attention_layers, QwenSparseSelfAttention),
        (moe_layers, MoELayer),
        (c.num_layers - moe_layers, MLP),
    ):
        if count:
            forward += count * module.estimate_flops(transformer, **stats)
    forward += 2 * c.num_layers * GatedResidualHyperConnection.estimate_flops(transformer, **stats)
    forward += GatedResidualOutputMixer.estimate_flops(transformer, **stats)
    forward += len(c.ple_layer_ids or ()) * PerLayerEmbedding.estimate_flops(transformer, **stats)
    vocab = c.vocab_size
    if c.should_pad_vocab:
        vocab = calculate_padded_vocab_size(vocab, c.make_vocab_size_divisible_by, c.tensor_model_parallel_size)
    forward += 2 * tokens * c.hidden_size * vocab

    indexer = 0
    if c.qsa_indexer_n_heads is not None and (
        c.qsa_force_sparse or c.seq_length > c.qsa_indexer_budget + c.qsa_indexer_compress_ratio - 1
    ):
        warnings.warn(
            "Qwen4 FLOPs use a dense-attention surrogate outside the QSA selection shortcut; "
            "sparse attention and indexer score work are not accurately accounted for.",
            UserWarning,
            stacklevel=2,
        )
        indexer = attention_layers * QSAIndexer.estimate_flops(transformer, **stats)
    return float(forward * (3 if include_backward else 1) + indexer)
