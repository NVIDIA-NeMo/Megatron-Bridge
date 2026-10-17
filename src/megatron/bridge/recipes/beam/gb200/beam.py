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

"""Beam BF16 library recipes for GB200 NVLink domains."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config, default_tulu3_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


BEAM_HF_MODEL_ID = "reflection/Beam-1.0-501B-A23B"
BEAM_HF_MODEL_REVISION = "06641076793dad51b8907f7eb2a14f4e08fd3ec8"  # pragma: allowlist secret
TULU3_REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"  # pragma: allowlist secret


def beam_pretrain_64gpu_gb200_bf16_config(hf_path: str | None = None) -> ConfigContainer:
    """Return a Beam pretrain config for 64 GB200 GPUs in one NVLink domain.

    Uses TP=4, PP=1, CP=1, EP=64, ETP=1 and HybridEP. BF16 model
    computation retains FP32 gradients, master parameters and Adam moments.
    Expert routing follows the released model, including its fixed expert biases.

    Args:
        hf_path: Optional local HF checkpoint instead of the pinned Hub revision.

    Returns:
        ConfigContainer with the model, dataset and training settings.
    """
    cfg = _pretrain_common()
    if hf_path is None:
        bridge = AutoBridge.from_hf_pretrained(BEAM_HF_MODEL_ID, revision=BEAM_HF_MODEL_REVISION)
        cfg.tokenizer.tokenizer_model = BEAM_HF_MODEL_ID
        cfg.tokenizer.hf_tokenizer_kwargs = {"revision": BEAM_HF_MODEL_REVISION}
    else:
        bridge = AutoBridge.from_hf_pretrained(hf_path)
        cfg.tokenizer.tokenizer_model = hf_path
    cfg.model = bridge.get_model_config()

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    # CP would change the per-example loss normalization of the finetuning recipes.
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "a2a"
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    # Beam's sandwich norms and depth-scaled residuals require the ordinary layer schedule.
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False, overlap_moe_expert_parallel_comm=False)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True

    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.lr_decay_style = "cosine"
    # Preserve the effective scheduler weight decay of the H200 recipes.
    cfg.scheduler.start_weight_decay = 0.033
    cfg.scheduler.end_weight_decay = 0.033
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.min_lr = 1e-5
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_grad = 1.0
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def beam_sft_64gpu_gb200_bf16_config(hf_path: str | None = None) -> ConfigContainer:
    """Return a Beam sft config for 64 GB200 GPUs in one NVLink domain.

    Uses TP=8, PP=1, CP=1, EP=64, ETP=1 and HybridEP. BF16 model
    computation retains FP32 gradients, master parameters and Adam moments.
    Expert routing follows the released model, including its fixed expert biases.
    Periodic garbage collection limits host pauses; activation fusion stays disabled.

    Args:
        hf_path: Optional local HF checkpoint instead of the pinned Hub revision.

    Returns:
        ConfigContainer with the model, dataset and training settings.
    """
    cfg = _sft_common()
    if hf_path is None:
        bridge = AutoBridge.from_hf_pretrained(BEAM_HF_MODEL_ID, revision=BEAM_HF_MODEL_REVISION)
        cfg.tokenizer.tokenizer_model = BEAM_HF_MODEL_ID
        cfg.tokenizer.hf_tokenizer_kwargs = {"revision": BEAM_HF_MODEL_REVISION}
    else:
        bridge = AutoBridge.from_hf_pretrained(hf_path)
        cfg.tokenizer.tokenizer_model = hf_path
    cfg.model = bridge.get_model_config()

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    # CP would change the per-example loss normalization of the finetuning recipes.
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "a2a"
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 8192
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.bias_activation_fusion = False
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    # Beam's sandwich norms and depth-scaled residuals require the ordinary layer schedule.
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False, overlap_moe_expert_parallel_comm=False)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True

    cfg.dataset = default_tulu3_config(seq_length=8192, enable_offline_packing=True, pad_seq_to_mult=8)
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": TULU3_REVISION}
    cfg.dataset.do_validation = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.seed = 1234
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.lr_decay_style = "cosine"
    # Preserve the effective scheduler weight decay of the H200 recipes.
    cfg.scheduler.start_weight_decay = 0.033
    cfg.scheduler.end_weight_decay = 0.033
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.optimizer.lr = 5e-6
    cfg.optimizer.min_lr = 0.0
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_grad = 1.0
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = 100
    cfg.logger.log_interval = 1

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def beam_peft_32gpu_gb200_bf16_config(peft_scheme: str | PEFT = "lora", hf_path: str | None = None) -> ConfigContainer:
    """Return a Beam peft config for 32 GB200 GPUs in one NVLink domain.

    Uses TP=4, PP=1, CP=1, EP=32, ETP=1 and HybridEP. BF16 model
    computation retains FP32 gradients, master parameters and Adam moments.
    Expert routing follows the released model, including its fixed expert biases.
    Periodic garbage collection limits host pauses; activation fusion stays disabled.

    Offline packing pads each sample to a multiple of four for TP4. This differs
    from H200 TP2 packing and can change pack membership and data order.

    Args:
        peft_scheme: LoRA, DoRA, or a custom PEFT instance.
        hf_path: Optional local HF checkpoint instead of the pinned Hub revision.

    Returns:
        ConfigContainer with the model, dataset and training settings.
    """
    cfg = _peft_common()
    if hf_path is None:
        bridge = AutoBridge.from_hf_pretrained(BEAM_HF_MODEL_ID, revision=BEAM_HF_MODEL_REVISION)
        cfg.tokenizer.tokenizer_model = BEAM_HF_MODEL_ID
        cfg.tokenizer.hf_tokenizer_kwargs = {"revision": BEAM_HF_MODEL_REVISION}
    else:
        bridge = AutoBridge.from_hf_pretrained(hf_path)
        cfg.tokenizer.tokenizer_model = hf_path
    cfg.model = bridge.get_model_config()

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    # CP would change the per-example loss normalization of the finetuning recipes.
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "a2a"
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 8192
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.bias_activation_fusion = False
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    # Beam's sandwich norms and depth-scaled residuals require the ordinary layer schedule.
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False, overlap_moe_expert_parallel_comm=False)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True

    cfg.dataset = default_tulu3_config(seq_length=8192, enable_offline_packing=True, pad_seq_to_mult=4)
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": TULU3_REVISION}
    cfg.dataset.do_validation = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.seed = 1234
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.lr_decay_style = "cosine"
    # Preserve the effective scheduler weight decay of the H200 recipes.
    cfg.scheduler.start_weight_decay = 0.033
    cfg.scheduler.end_weight_decay = 0.033
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.min_lr = 0.0
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_grad = 1.0
    cfg.optimizer.use_distributed_optimizer = True
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.check_for_large_grads = True
    cfg.rerun_state_machine.check_for_nan_in_loss = True
    cfg.checkpoint.load = None
    cfg.checkpoint.save_interval = 100
    cfg.logger.log_interval = 1

    peft_cfg = default_peft_config(peft_scheme)
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
        peft_cfg.dropout = 0.0
        peft_cfg.target_modules = ["linear_qkv", "linear_proj"]
    cfg.peft = peft_cfg

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg
