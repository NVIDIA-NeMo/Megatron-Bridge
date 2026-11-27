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

"""ERNIE 5.5 modules built on Megatron-Core.

* D layer: latent MQA + DSA with window and sink (``latent_mqa_attention``, ``latent_mqa_core``).
* H layer: Megatron-Core's DSv4 hybrid attention at compress ratio 128 with the ERNIE output gate,
  VHA head mix, plain RoPE or YaRN and sink seeding (``hca_attention``).
* MoE layer: Megatron-Core's ``MoELayer`` with the split-feature router and fp32 latent projections
  (``moe``).
* Context parallelism: D layers run zigzag inside the contiguous DSv4 stack
  (``context_parallel``).
* Hyper-connections with fp32 decoder streams: the eager fp32 mapping and mixing
  (``hyper_connection``), downcasting pre-norms (``norm``) and the fp32 decoder stack
  (``hybrid_stack``). ``norm`` also holds the reference implementation's RMSNorm arithmetic,
  behind ``ernie_reference_rmsnorm``.
* Multimax output head with its own gradient-norm group (``output_layer``).
* The reference implementation's Triton kernels for SiTU-GLU, multimax and the context-parallel
  local RoPE (``fused_ops``), dispatched behind ``ernie_fused_ops`` (``fused_dispatch``).
* SD-MTP, the MTP body sharing the last decoder layers' parameters (``mtp``), and the per-row MTP loss
  (``mtp_loss``).
* The model, which applies all of the above after Megatron-Core builds it (``model``).

FlashMLA and the cuDNN DSA kernels are imported lazily by Megatron-Core, and Triton by
``fused_ops`` on its first kernel launch, so importing this package needs none of them.
"""

from megatron.bridge.models.ernie.modeling_ernie55.context_parallel import validate_ernie55_context_parallel_config
from megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch import install_ernie55_fused_ops
from megatron.bridge.models.ernie.modeling_ernie55.hca_attention import (
    ERNIE55_ATTN_SINK_INIT_STD,
    ERNIE55_HCA_COMPRESS_RATIO,
    ERNIE55_HCA_ROPE_TYPES,
    Ernie55HCACoreAttention,
    Ernie55HCASelfAttention,
    Ernie55HCASelfAttentionSubmodules,
    check_ernie55_attention_layers,
    rotary_embedding_references,
    validate_ernie55_hca_config,
)
from megatron.bridge.models.ernie.modeling_ernie55.hybrid_stack import Ernie55HybridStack
from megatron.bridge.models.ernie.modeling_ernie55.hyper_connection import (
    Ernie55HyperConnectionModule,
    use_ernie55_hyper_connections,
    validate_ernie55_mhc_config,
)
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_attention import (
    Ernie55LatentMQASelfAttention,
    ernie55_d_layer_config,
    validate_ernie55_config,
)
from megatron.bridge.models.ernie.modeling_ernie55.latent_mqa_core import (
    INDEXER_SCORING_PACKED_THD_GENERAL,
    INDEXER_SCORING_REFERENCE_ORDER,
    INDEXER_SCORING_UNFUSED_BOUNDS,
    REFERENCE_FALLBACK_MAX_TOKENS,
    Ernie55LatentMQACoreAttention,
    Ernie55LatentMQACoreSubmodules,
    indexer_loss_layer_number,
)
from megatron.bridge.models.ernie.modeling_ernie55.model import Ernie55HybridModel
from megatron.bridge.models.ernie.modeling_ernie55.moe import (
    Ernie55LatentProjection,
    Ernie55MoELayer,
    Ernie55SplitFeatureRouter,
    validate_ernie55_moe_config,
)
from megatron.bridge.models.ernie.modeling_ernie55.mtp import (
    check_ernie55_mtp_aliases,
    share_mtp_body_with_last_layer,
    validate_ernie55_mtp_sharing,
)
from megatron.bridge.models.ernie.modeling_ernie55.mtp_loss import PER_ROW_MTP_LOSS_SUPPORTED, ernie55_process_mtp_loss
from megatron.bridge.models.ernie.modeling_ernie55.muon import ernie55_muon_split_sizes, tag_ernie55_muon_params
from megatron.bridge.models.ernie.modeling_ernie55.norm import (
    Ernie55RMSNorm,
    Ernie55RMSNormSelector,
    downcasting_norm,
    ernie55_rmsnorm,
    ernie55_rmsnorm_spec,
    is_downcasting_norm,
)
from megatron.bridge.models.ernie.modeling_ernie55.output_layer import (
    MULTIMAX_GRAD_NORM_GROUP,
    MULTIMAX_PARAMETER_NAMES,
    ernie55_multimax,
    install_ernie55_multimax,
    validate_ernie55_multimax_config,
)
from megatron.bridge.models.ernie.modeling_ernie55.stack_spec import ernie55_hybrid_stack_spec


__all__ = [
    "ERNIE55_ATTN_SINK_INIT_STD",
    "ERNIE55_HCA_COMPRESS_RATIO",
    "ERNIE55_HCA_ROPE_TYPES",
    "INDEXER_SCORING_PACKED_THD_GENERAL",
    "INDEXER_SCORING_REFERENCE_ORDER",
    "INDEXER_SCORING_UNFUSED_BOUNDS",
    "MULTIMAX_GRAD_NORM_GROUP",
    "MULTIMAX_PARAMETER_NAMES",
    "PER_ROW_MTP_LOSS_SUPPORTED",
    "REFERENCE_FALLBACK_MAX_TOKENS",
    "Ernie55HCACoreAttention",
    "Ernie55HCASelfAttention",
    "Ernie55HCASelfAttentionSubmodules",
    "Ernie55HybridModel",
    "Ernie55HybridStack",
    "Ernie55HyperConnectionModule",
    "Ernie55LatentMQACoreAttention",
    "Ernie55LatentMQACoreSubmodules",
    "Ernie55LatentMQASelfAttention",
    "Ernie55LatentProjection",
    "Ernie55MoELayer",
    "Ernie55RMSNorm",
    "Ernie55RMSNormSelector",
    "Ernie55SplitFeatureRouter",
    "check_ernie55_attention_layers",
    "check_ernie55_mtp_aliases",
    "downcasting_norm",
    "ernie55_d_layer_config",
    "ernie55_hybrid_stack_spec",
    "ernie55_muon_split_sizes",
    "ernie55_multimax",
    "ernie55_process_mtp_loss",
    "ernie55_rmsnorm",
    "ernie55_rmsnorm_spec",
    "indexer_loss_layer_number",
    "install_ernie55_fused_ops",
    "install_ernie55_multimax",
    "is_downcasting_norm",
    "rotary_embedding_references",
    "share_mtp_body_with_last_layer",
    "tag_ernie55_muon_params",
    "use_ernie55_hyper_connections",
    "validate_ernie55_config",
    "validate_ernie55_context_parallel_config",
    "validate_ernie55_hca_config",
    "validate_ernie55_mhc_config",
    "validate_ernie55_moe_config",
    "validate_ernie55_mtp_sharing",
    "validate_ernie55_multimax_config",
]
