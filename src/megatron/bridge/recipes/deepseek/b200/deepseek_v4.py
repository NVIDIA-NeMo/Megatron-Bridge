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
"""B200 NVL8 recipes for DeepSeek V4 Flash."""

from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_library_config,
)
from megatron.bridge.training.config import ConfigContainer


_DSV4_FLASH_PP8_VP2_LAYOUT = "Et*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*2|t*2|t*2|t*2|t*2mL"
_FLEX_DISPATCHER_ENV_VARS = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API",
    "NVLINK_DOMAIN_SIZE",
    "USE_MNNVL",
}


def deepseek_v4_flash_pretrain_64gpu_b200_fp8mx_library_config() -> ConfigContainer:
    """Return real-training DeepSeek V4 Flash for 64 B200 GPUs.

    PP8/VPP2 keeps each eight-rank expert group within one NVL8 system. The
    recipe uses the portable all-to-all dispatcher and preserves natural,
    unlimited-capacity routing without paged stash or CUDA graphs.
    """
    cfg = deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_library_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.virtual_pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.pipeline_model_parallel_layout = _DSV4_FLASH_PP8_VP2_LAYOUT

    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_flex_dispatcher_num_sms = None
    cfg.model.moe_deepep_num_sms = None
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_hybridep_num_sms_preprocessing = None
    cfg.model.moe_shared_expert_overlap = False

    cfg.model.recompute_modules = ["moe", "mhc", "mla_up_proj", "layernorm"]
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["core_attn", "attn_proj"]
    cfg.model.fine_grained_offloading_max_inflight_offloads = 2
    cfg.model.moe_pad_experts_for_cuda_graph_inference = False
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_modules = []
    cfg.model.cuda_graph_scope = None
    cfg.model.use_te_rng_tracker = False
    cfg.rng.te_rng_tracker = False

    cfg.env_vars = {key: value for key, value in cfg.env_vars.items() if key not in _FLEX_DISPATCHER_ENV_VARS}
    return cfg
