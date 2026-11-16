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

"""H100 execution choices for Qwen3-30B-A3B, independent of benchmark policy."""

from typing import cast

from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


def qwen3_30b_a3b_bf16(cfg: ConfigContainer) -> None:
    """Add permutation/EP overlap tuning to the library's 16-H100 BF16 layout."""
    # These named library recipes return GPT configs, not the other ConfigContainer model families.
    model = cast(GPTModelConfig, cfg.model)
    model.moe_permute_fusion_into_hybridep = True
    if cfg.comm_overlap is None:
        raise ValueError("Select the Qwen 16-H100 BF16 library layout before applying its overlap settings.")
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 64,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 10,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 10,
    }


def qwen3_30b_a3b_fp8(cfg: ConfigContainer) -> None:
    """Select the existing EP16 HybridEP/TE-graph layout for explicit FP8."""
    # These named library recipes return GPT configs, not the other ConfigContainer model families.
    model = cast(GPTModelConfig, cfg.model)
    if not getattr(cfg.mixed_precision, "fp8", None):
        raise ValueError("The Qwen EP16 FP8 execution layout requires an explicit FP8 precision config.")
    model.recompute_granularity = None
    model.recompute_method = None
    model.recompute_num_layers = None
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.context_parallel_size = 1
    model.virtual_pipeline_model_parallel_size = None
    model.expert_model_parallel_size = 16
    model.sequence_parallel = False
    model.moe_flex_dispatcher_backend = "hybridep"
    model.moe_token_dispatcher_type = "flex"
    model.moe_hybridep_num_sms = 32
    model.cuda_graph_impl = "transformer_engine"
    model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
