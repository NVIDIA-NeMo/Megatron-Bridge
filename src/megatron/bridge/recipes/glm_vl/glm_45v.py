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

"""GLM-4.5V finetuning recipes with parameterless API.

This module provides SFT and PEFT configurations for GLM-4.5V (106B MoE).
"""

from typing import List, Optional, Union

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common_vlm, _sft_common_vlm
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


def set_glm_45v_pipeline_model_parallel_layout(
    model_cfg: GPTModelProvider, layout: Optional[Union[str, List[List[str]]]] = None, is_peft: bool = False
) -> None:
    """Set the GLM-4.5V pipeline model parallel layout.

    GLM-4.5V (based on GLM-4.5 Air) has 46 decoder layers and no MTP layers.
    This function sets up predefined layouts for common PP/VP combinations.

    Args:
        model_cfg: The model provider configuration to modify.
        layout: Optional custom layout. If None, uses predefined layouts based on PP/VP sizes.
        is_peft: Whether the model is trained with PEFT.
    """
    # GLM-4.5V has no MTP layers
    last_layer = ["loss"]
    pp_size = model_cfg.pipeline_model_parallel_size or 1
    vp_size = model_cfg.virtual_pipeline_model_parallel_size or 1

    # GLM-4.5 Air has 46 decoder layers
    # GLM-4.5 Vision Encoder is huge, we need to balance the first stage with the least number of layers
    # Layout maps for common PP/VP combinations
    # We use different layouts for PEFT and full SFT.
    if is_peft:
        layout_map = {
            (4, 1): [
                ["embedding"] + ["decoder"] * 11,
                ["decoder"] * 12,
                ["decoder"] * 12,
                ["decoder"] * 11 + last_layer,
            ],
            (8, 1): [["embedding"] + ["decoder"] * 5] + [["decoder"] * 6] * 6 + [["decoder"] * 5 + last_layer],
            (16, 1): [["embedding"] + ["decoder"] * 2] + [["decoder"] * 3] * 14 + [["decoder"] * 2 + last_layer],
        }
    else:
        layout_map = {
            (4, 1): [
                ["embedding"] + ["decoder"] * 11,
                ["decoder"] * 12,
                ["decoder"] * 12,
                ["decoder"] * 11 + last_layer,
            ],
            (8, 1): [["embedding"] + ["decoder"]] + [["decoder"] * 7] * 6 + [["decoder"] * 3 + last_layer],
            (16, 1): [["embedding"]] + [["decoder"] * 3] * 14 + [["decoder"] * 3 + last_layer],
        }

    if layout is not None:
        model_cfg.pipeline_model_parallel_layout = layout
    elif (pp_size, vp_size) in layout_map:
        model_cfg.pipeline_model_parallel_layout = layout_map[(pp_size, vp_size)]


# =============================================================================
# GLM-4.5V SFT Configuration
# =============================================================================
def glm_45v_sft_config() -> ConfigContainer:
    """Return a full SFT config for GLM-4.5V (106B MoE).

    Default configuration: 64 nodes, 512 GPUs
    - TP=1, PP=8, EP=16
    - LR=5e-6 (full SFT)
    - Sequence length: 8192
    """
    cfg = _sft_common_vlm()

    # Model configuration
    hf_path = "zai-org/GLM-4.5V"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 8192

    # Parallel settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 16
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Set pipeline model parallel layout for asymmetric stages
    set_glm_45v_pipeline_model_parallel_layout(cfg.model, layout=None, is_peft=False)

    # Pipeline split for asymmetric stages are specified with the layout above
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # Token dispatcher settings (MoE)
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE kernel selections
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # MoE overlap
    cfg.model.moe_shared_expert_overlap = True

    # MoE force balance
    cfg.model.moe_router_force_load_balancing = False

    # MoE FP8 padding
    cfg.model.moe_router_padding_for_fp8 = False

    # Training config
    cfg.train.train_iters = 50
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation config
    cfg.validation.eval_interval = 5
    cfg.validation.eval_iters = 10

    # Optimizer - lower LR for full SFT
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        lr_decay_iters=50,
        max_lr=5e-6,
        min_lr=5e-6 * 0.1,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Optimizer precision settings (disabled by default for full precision)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 8192
    cfg.dataset.hf_processor_path = hf_path

    # DDP settings - GLM-4.5V specific settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Comm overlap settings (MoE)
    cfg.comm_overlap = None
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # FP8 and MXFP8 settings (disabled by default)
    cfg.mixed_precision = "bf16_mixed"
    # FP8 settings (uncomment to use FP8)
    # cfg.mixed_precision.fp8_recipe = None
    # cfg.mixed_precision.fp8 = False
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    # Checkpoint config
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below to use a pretrained checkpoint
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    return cfg


# =============================================================================
# GLM-4.5V PEFT Configuration
# =============================================================================
def glm_45v_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for GLM-4.5V (106B MoE).

    Default configuration: 8 nodes, 64 GPUs
    - TP=1, PP=8, EP=4
    - LR=1e-4 (PEFT)
    - Sequence length: 8192

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
    """
    cfg = _peft_common_vlm()

    # PEFT scheme
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme)
    else:
        cfg.peft = peft_scheme

    # Model configuration
    hf_path = "zai-org/GLM-4.5V"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 8192

    # Parallel settings - lower EP for PEFT
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Set pipeline model parallel layout for asymmetric stages
    set_glm_45v_pipeline_model_parallel_layout(cfg.model, layout=None, is_peft=True)

    # Pipeline split for asymmetric stages are specified with the layout above
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # Token dispatcher settings (MoE)
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE kernel selections
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # MoE overlap
    cfg.model.moe_shared_expert_overlap = True

    # MoE force balance
    cfg.model.moe_router_force_load_balancing = False

    # MoE FP8 padding
    cfg.model.moe_router_padding_for_fp8 = False

    # Training config
    cfg.train.train_iters = 50
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation config
    cfg.validation.eval_interval = 5
    cfg.validation.eval_iters = 10

    # Optimizer - higher LR for PEFT
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        lr_decay_iters=50,
        max_lr=1e-4,
        min_lr=1e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Optimizer precision settings (disabled by default for full precision)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 8192
    cfg.dataset.hf_processor_path = hf_path

    # DDP settings - GLM-4.5V specific settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Comm overlap settings (MoE)
    cfg.comm_overlap = None
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # FP8 and MXFP8 settings (disabled by default)
    cfg.mixed_precision = "bf16_mixed"
    # FP8 settings (uncomment to use FP8)
    # cfg.mixed_precision.fp8_recipe = None
    # cfg.mixed_precision.fp8 = False
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    # Checkpoint config
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below to use a pretrained checkpoint
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    return cfg
