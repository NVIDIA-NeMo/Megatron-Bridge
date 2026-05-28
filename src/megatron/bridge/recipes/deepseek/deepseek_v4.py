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

from typing import Literal

import torch
from megatron.core.quantization.quant_config import RecipeConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models import GPTModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
    distributed_muon_with_cosine_annealing,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed, bf16_with_mxfp8_mixed


DSV4_CSA_BACKEND = Literal["unfused", "cudnn_dsa", "tilelang_official", "flashmla_official"]
DSV4_OPTIMIZER = Literal["adam", "muon"]


def set_deepseek_v4_pipeline_model_parallel_layout(model_cfg: GPTModelProvider) -> None:
    """Set an even DSv4 pipeline layout with MTP and loss on the last stage."""
    pp_size = model_cfg.pipeline_model_parallel_size or 1
    if pp_size <= 1:
        model_cfg.pipeline_model_parallel_layout = None
        return

    num_layers = int(getattr(model_cfg, "num_layers", 0) or 0)
    if num_layers <= 0:
        model_cfg.pipeline_model_parallel_layout = None
        return

    mtp_layers = int(getattr(model_cfg, "mtp_num_layers", 0) or 0)
    base_layers, extra_layers = divmod(num_layers, pp_size)
    layout: list[list[str]] = []
    for pp_rank in range(pp_size):
        stage: list[str] = []
        if pp_rank == 0:
            stage.append("embedding")

        decoder_layers = base_layers + int(pp_rank < extra_layers)
        stage.extend(["decoder"] * decoder_layers)

        if pp_rank == pp_size - 1:
            stage.extend(["mtp"] * mtp_layers)
            stage.append("loss")
        layout.append(stage)

    model_cfg.pipeline_model_parallel_layout = layout


def _set_deepseek_v4_common_model_config(
    cfg: ConfigContainer,
    *,
    csa_backend: DSV4_CSA_BACKEND,
    use_fused_kernels: bool,
) -> None:
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.params_dtype = torch.bfloat16

    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = None
    cfg.model.apply_rope_fusion = use_fused_kernels
    cfg.model.use_fused_mhc = use_fused_kernels
    cfg.model.csa_backend = csa_backend
    # Keep indexer loss disabled until the DSv4 fused indexer-loss path is supported.
    cfg.model.dsa_indexer_loss_coeff = 0.0
    cfg.model.dsa_indexer_use_sparse_loss = False

    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act", "mhc"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3


def _set_deepseek_v4_common_training_config(cfg: ConfigContainer) -> None:
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size

    cfg.dataset.blend = None
    cfg.dataset.blend_per_split = None
    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8
    cfg.dataset.skip_getting_attention_mask_from_dataset = True
    cfg.dataset.dataloader_type = "single"

    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5
    cfg.validation.eval_interval = 2000
    cfg.validation.eval_iters = 32

    cfg.logger.log_interval = 10
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False
    cfg.dist.enable_megatron_core_experimental = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False


def _mxfp8_train_bf16_eval_quant_recipe() -> RecipeConfig:
    """Train TE linear modules in MXFP8 while evaluating them in BF16."""
    return RecipeConfig.from_config_dict(
        {
            "configs": {
                "mxfp8_evaluate_bf16": {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {"fp8_quantization_recipe": "mxfp8"},
                    "evaluation_recipe": {},
                },
            },
            "matchers": {
                "all_te_linears": {
                    "config": "mxfp8_evaluate_bf16",
                    "type": "glob",
                    "pattern": "*",
                    "enabled": True,
                },
            },
        }
    )


def _set_deepseek_v4_optimizer_and_precision(
    cfg: ConfigContainer,
    *,
    optimizer_type: DSV4_OPTIMIZER,
    mxfp8: bool,
) -> None:
    if optimizer_type == "adam":
        opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
            lr_warmup_iters=2000,
            lr_decay_iters=cfg.train.train_iters,
            max_lr=2.7e-4,
            min_lr=2.7e-5,
            weight_decay=0.1,
            clip_grad=1.0,
        )
        opt_cfg.use_precision_aware_optimizer = True
        opt_cfg.main_grads_dtype = torch.float32
        opt_cfg.main_params_dtype = torch.float32
        opt_cfg.exp_avg_dtype = torch.bfloat16
        opt_cfg.exp_avg_sq_dtype = torch.bfloat16
        opt_cfg.adam_beta1 = 0.9
        opt_cfg.adam_beta2 = 0.95
        opt_cfg.adam_eps = 1e-20

        scheduler_cfg.start_weight_decay = 0.1
        scheduler_cfg.end_weight_decay = 0.1
        scheduler_cfg.weight_decay_incr_style = "constant"

        cfg.ddp.use_distributed_optimizer = True
        cfg.ddp.overlap_param_gather = True
        cfg.ddp.overlap_grad_reduce = True
        cfg.ddp.grad_reduce_in_fp32 = True
        cfg.ddp.average_in_collective = True
        cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
        cfg.mixed_precision = bf16_with_mxfp8_mixed() if mxfp8 else bf16_mixed()
        if mxfp8:
            # Keep eval and param-gather-adjacent paths out of MXFP8 for DSv4 stability.
            # The training path still uses MXFP8, while validation/MTP eval remains BF16.
            cfg.mixed_precision.fp8_param_gather = False
            cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
            cfg.model.moe_router_padding_for_fp8 = True
            cfg.model.mtp_eval_in_bf16 = True
            cfg.model.quant_recipe = _mxfp8_train_bf16_eval_quant_recipe()
    elif optimizer_type == "muon":
        if mxfp8:
            raise ValueError("DeepSeek-V4 Muon + MXFP8 is not a supported recipe yet.")
        opt_cfg, scheduler_cfg = distributed_muon_with_cosine_annealing(
            muon_momentum=0.95,
            muon_use_nesterov=True,
            muon_scale_mode="unit_rms_norm",
            muon_fp32_matmul_prec="highest",
            muon_num_ns_steps=5,
            muon_extra_scale_factor=0.2,
            lr_warmup_iters=2000,
            lr_decay_iters=cfg.train.train_iters,
            max_lr=2.7e-4,
            min_lr=2.7e-5,
            weight_decay=0.1,
            clip_grad=1.0,
        )
        opt_cfg.optimizer = "muon"
        opt_cfg.adam_beta1 = 0.9
        opt_cfg.adam_beta2 = 0.95
        opt_cfg.adam_eps = 1e-20
        if hasattr(opt_cfg, "muon_coefficient_type"):
            opt_cfg.muon_coefficient_type = "quintic"

        scheduler_cfg.start_weight_decay = 0.1
        scheduler_cfg.end_weight_decay = 0.1
        scheduler_cfg.weight_decay_incr_style = "constant"

        cfg.ddp.use_distributed_optimizer = False
        cfg.ddp.overlap_param_gather = False
        cfg.ddp.overlap_grad_reduce = True
        cfg.ddp.grad_reduce_in_fp32 = True
        cfg.ddp.average_in_collective = True
        cfg.ddp.data_parallel_sharding_strategy = "no_shard"
        cfg.mixed_precision = bf16_mixed()
        cfg.mixed_precision.grad_reduce_in_fp32 = True
    else:
        raise ValueError(f"Invalid DeepSeek-V4 optimizer type: {optimizer_type}")

    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_megatron_fsdp = False


def _deepseek_v4_flash_pretrain_config(
    *,
    optimizer_type: DSV4_OPTIMIZER = "adam",
    mxfp8: bool = False,
    csa_backend: DSV4_CSA_BACKEND = "cudnn_dsa",
    use_fused_kernels: bool = True,
    hf_path: str = "deepseek-ai/DeepSeek-V4-Flash",
) -> ConfigContainer:
    cfg = _pretrain_common()
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(load_weights=False)

    _set_deepseek_v4_common_model_config(cfg, csa_backend=csa_backend, use_fused_kernels=use_fused_kernels)
    _set_deepseek_v4_common_training_config(cfg)
    _set_deepseek_v4_optimizer_and_precision(cfg, optimizer_type=optimizer_type, mxfp8=mxfp8)
    return cfg


def deepseek_v4_flash_pretrain_mxfp8_config(hf_path: str = "deepseek-ai/DeepSeek-V4-Flash") -> ConfigContainer:
    """Return the DeepSeek-V4-Flash Adam + MXFP8 pre-training config."""
    return _deepseek_v4_flash_pretrain_config(optimizer_type="adam", mxfp8=True, hf_path=hf_path)


def deepseek_v4_flash_pretrain_muon_config(hf_path: str = "deepseek-ai/DeepSeek-V4-Flash") -> ConfigContainer:
    """Return the DeepSeek-V4-Flash BF16 Muon pre-training config."""
    return _deepseek_v4_flash_pretrain_config(optimizer_type="muon", mxfp8=False, hf_path=hf_path)
