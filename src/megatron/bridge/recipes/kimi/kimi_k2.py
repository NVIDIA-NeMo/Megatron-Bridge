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

import torch

from megatron.bridge.models.kimi import KimiK2Provider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.optimizer_utils import distributed_muon_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def _get_kimi_k2_pipeline_layout(pp_size: int, vp_size: int):
    """Get pipeline layout for Kimi-K2 based on PP and VP size."""
    map_pp_vp_to_layout = {
        (1, 1): None,
        (4, 1): [["embedding"] + ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 13 + ["loss"]],
        (8, 1): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + ["loss"]],
        (4, 2): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + ["loss"]],
        (16, 1): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
        (8, 2): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
        (4, 4): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
    }

    vp_size = 1 if vp_size is None else vp_size
    if (pp_size, vp_size) not in map_pp_vp_to_layout:
        raise ValueError(
            f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout "
            f"for Kimi-K2. Known PP and VP combinations: {map_pp_vp_to_layout.keys()}"
        )
    layout = map_pp_vp_to_layout[(pp_size, vp_size)]
    if layout is not None:
        layout = list([list(x) for x in layout])
    return layout


def kimi_k2_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Kimi-K2 (1T).

    Recommended parallelism: TP=2, PP=16, EP=32
    Uses Muon optimizer by default.
    """
    cfg = _pretrain_common()

    # Model config - uses KimiK2Provider instead of AutoBridge
    cfg.model = KimiK2Provider(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=16,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=32,
        sequence_parallel=True,
        expert_tensor_parallel_size=1,
        recompute_granularity="selective",
        recompute_modules=None,
        recompute_method=None,
        recompute_num_layers=None,
    )

    # Pipeline split settings (asymmetric stages)
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # Set pipeline layout
    cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(16, 1)

    # Tokenizer - uses NullTokenizer with model vocab_size
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.sequence_length = 4096
    cfg.dataset.num_workers = 8

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training config
    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1
    cfg.train.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_muon_with_cosine_annealing(
        lr_warmup_iters=2000,
        lr_decay_iters=cfg.train.train_iters,
        max_lr=3e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading) - already set in KimiK2Provider
    # cfg.model.recompute_granularity = "selective"
    # cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - Kimi-K2 uses custom MixedPrecisionConfig (NOT "bf16_mixed" string)
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False  # Pad router for FP8 alignment

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = True

    # Checkpoint config
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config (DIFFERENT for Muon optimizer)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False  # Muon needs this to be False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False  # Muon needs this to be False
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    if cfg.model.apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    return cfg
