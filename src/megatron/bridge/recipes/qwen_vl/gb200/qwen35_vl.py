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

"""GB200 functional recipes shared by Qwen3.5/Qwen3.6-VL 35B-A3B."""

from __future__ import annotations

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import EnergonDatasetConfig, QwenVLEnergonTaskEncoderConfig
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_27b_pretrain_16gpu_h100_bf16_mock_config,
    qwen35_vl_35b_a3b_peft_4gpu_h100_bf16_config,
    qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed, bf16_with_mxfp8_mixed, get_mixed_precision_config
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


def qwen35_vl_27b_pretrain_16gpu_gb200_bf16_mock_config() -> ConfigContainer:
    """Return Qwen3.5-VL 27B language-and-projector pretraining for sixteen GB200 GPUs.

    This keeps the dense pretraining objective and trainable-parameter contract
    from the H100 recipe while using the measured GB200 execution layout. In
    particular, deterministic mode must remain opt-in: Qwen3.5's Gated DeltaNet
    replaces its fused FLA kernels with a torch-native reference path when
    deterministic mode is enabled.
    """
    cfg = qwen35_vl_27b_pretrain_16gpu_h100_bf16_mock_config()

    # TP2 leaves enough memory headroom for cold Gated DeltaNet autotuning
    # while PP1 removes pipeline bubbles. Full language-tower training requires
    # MBS2; MBS4 exceeds GB200 memory during the first language forward.
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.calculate_per_token_loss = True

    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 2

    # MBS2 fits full language-tower training without repeating the expensive
    # GDN recurrence or dense MLP.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None

    # Qwen-VL's multimodal inputs are not yet a safe full-iteration graph
    # target, and its mRoPE path is incompatible with fused RoPE.
    cfg.model.apply_rope_fusion = False
    cfg.model.cuda_graph_impl = "none"
    set_cuda_graph_modules(cfg.model, [])
    cfg.model.use_te_rng_tracker = False
    cfg.rng.te_rng_tracker = False

    cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False

    cfg.dataset.do_validation = False
    cfg.dataset.pad_to_max_length = True
    cfg.train.eval_interval = 0
    cfg.train.eval_iters = 0
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.checkpoint.load = None
    cfg.checkpoint.save = None
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.logger.tensorboard_dir = None
    cfg.dist.distributed_timeout_minutes = 30

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
    )

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def _apply_qwen35_vl_35b_a3b_gb200_functional_defaults(cfg: ConfigContainer) -> None:
    """Apply shared settings for bounded GB200 functional runs."""
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    cfg.train.global_batch_size = 32
    # Learned routing can concentrate enough real MedPix tokens on one rank
    # for MBS2 to exhaust that rank during full-vocabulary cross entropy.
    # MBS1 preserves learned-routing semantics; MBS4 is reserved for the
    # force-balanced performance-only recipe.
    cfg.train.micro_batch_size = 1
    # Real VLM batches compile multiple multimodal shapes. Release cached
    # blocks after each optimizer step so inactive compiled-shape allocations
    # cannot accumulate into a later fused-cross-entropy OOM.
    cfg.train.empty_unused_memory_level = 2

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.high_priority_a2a_comm_stream = True

    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None

    # Start with eager execution. Fixed-shape VLM CUDA graph capture must be
    # demonstrated on real multimodal data before it becomes a recipe default.
    cfg.model.cuda_graph_impl = "none"
    set_cuda_graph_modules(cfg.model, [])
    cfg.model.use_te_rng_tracker = False
    cfg.rng.te_rng_tracker = False

    cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    # Preserve headroom for the full-vocabulary loss and learned-routing
    # expert-shape imbalance instead of reserving DDP overlap buffers.
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False

    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    cfg.dataset.do_validation = False
    cfg.dataset.pad_to_max_length = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.checkpoint.load = None
    cfg.checkpoint.save = None
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.logger.tensorboard_dir = None
    cfg.dist.distributed_timeout_minutes = 30

    # Qwen-VL's multimodal schedule cannot yet use MoE A2A overlap without
    # bypassing vision preprocessing.
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
    )

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }


def qwen35_vl_35b_a3b_sft_8gpu_gb200_bf16_functional_config() -> ConfigContainer:
    """Return shared Qwen3.5/Qwen3.6-VL 35B-A3B SFT for eight GB200 GPUs."""
    cfg = qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config()
    _apply_qwen35_vl_35b_a3b_gb200_functional_defaults(cfg)
    return cfg


_QWEN35_35B_A3B_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"  # pragma: allowlist secret


def _qwen35_vl_35b_a3b_long_context_common() -> ConfigContainer:
    """Build the shared 32-GB200 long-context SFT configuration."""
    cfg = _sft_common_vlm()
    hf_path = "Qwen/Qwen3.5-35B-A3B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, revision=_QWEN35_35B_A3B_REVISION).to_megatron_provider(
        load_weights=False
    )
    cfg.model.calculate_per_token_loss = True
    cfg.ddp.average_in_collective = False

    cfg.model.mtp_num_layers = 1
    cfg.model.bias_activation_fusion = True
    cfg.model.cuda_graph_scope = []
    cfg.model.moe_router_fusion = True
    cfg.model.offload_modules = None
    cfg.model.seq_length = 131072
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.context_parallel_size = 8
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True

    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["gdn_norm_out", "moe"]
    cfg.model.vision_recompute_granularity = "full"
    cfg.model.vision_recompute_method = "uniform"
    cfg.model.vision_recompute_num_layers = 1

    cfg.train.micro_batch_size = 1
    # Nemotron Image Training v3, clevr_2 subset, prepared as Energon shards.
    # Native packing combines samples into one physical 128K microbatch.
    cfg.dataset = EnergonDatasetConfig(
        seq_length=cfg.model.seq_length,
        micro_batch_size=cfg.train.micro_batch_size,
        num_workers=1,
        shuffle_buffer_size=2,
        packing_buffer_size=8,
        dataset_kwargs={"image_decode_spec": "pilrgb"},
        task_encoder=QwenVLEnergonTaskEncoderConfig(
            hf_processor_path=hf_path,
            hf_processor_revision=_QWEN35_35B_A3B_REVISION,
            max_pixels=401408,
            max_num_images=4,
            max_visual_tokens=2048,
        ),
    )

    cfg.optimizer.lr = 2e-5
    cfg.optimizer.min_lr = 2e-6
    cfg.scheduler.lr_warmup_iters = 200
    cfg.scheduler.lr_decay_iters = 300000
    cfg.scheduler.lr_wsd_decay_iters = 300000
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
    )
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Both language and vision towers use automatic TE backend selection.
        "NVTE_FUSED_ATTN": 1,
        "NVTE_FLASH_ATTN": 1,
        "NVTE_UNFUSED_ATTN": 1,
        "NCCL_GRAPH_REGISTER": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 16,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "USE_MNNVL": 1,
    }
    return cfg


def qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_bf16_config() -> ConfigContainer:
    """Return 128K BF16 Qwen3.5-VL SFT with TP2/CP8/EP16 on 32 GB200 GPUs.

    Override with ``"dataset.path=$CLEVR2_ENERGON"`` to select the prepared
    Nemotron Image Training v3 ``clevr_2`` Energon shards, and supply a
    pretrained checkpoint.

    Prepare the shards with `prepare_nemotron_image_v3.py
    <https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/energon/prepare_nemotron_image_v3.py>`_
    using ``--subsets clevr_2 --skip-source-integrity-check`` after verifying
    the source revision; the built-in integrity manifest covers only ``turing``.
    Native Energon packing uses one physical microbatch per 128K pack.
    """
    cfg = _qwen35_vl_35b_a3b_long_context_common()
    cfg.mixed_precision = bf16_mixed()
    return cfg


def qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Return packed CLEVR2 128K MXFP8 SFT with BF16 parameter communication.

    Override with ``"dataset.path=$CLEVR2_ENERGON"`` to select the prepared
    Nemotron Image Training v3 ``clevr_2`` Energon shards, and supply a
    pretrained checkpoint.

    Prepare the shards with `prepare_nemotron_image_v3.py
    <https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/data/energon/prepare_nemotron_image_v3.py>`_
    using ``--subsets clevr_2 --skip-source-integrity-check`` after verifying
    the source revision; the built-in integrity manifest covers only ``turing``.
    Native Energon packing uses one physical microbatch per 128K pack.
    """
    cfg = _qwen35_vl_35b_a3b_long_context_common()
    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    return cfg


def qwen35_vl_35b_a3b_peft_8gpu_gb200_bf16_functional_config() -> ConfigContainer:
    """Return shared Qwen3.5/Qwen3.6-VL 35B-A3B LoRA for eight GB200 GPUs."""
    cfg = qwen35_vl_35b_a3b_peft_4gpu_h100_bf16_config()
    _apply_qwen35_vl_35b_a3b_gb200_functional_defaults(cfg)
    return cfg


__all__ = [
    "qwen35_vl_27b_pretrain_16gpu_gb200_bf16_mock_config",
    "qwen35_vl_35b_a3b_peft_8gpu_gb200_bf16_functional_config",
    "qwen35_vl_35b_a3b_sft_8gpu_gb200_bf16_functional_config",
    "qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_bf16_config",
    "qwen35_vl_35b_a3b_sft_long_context_32gpu_gb200_fp8mx_config",
]
