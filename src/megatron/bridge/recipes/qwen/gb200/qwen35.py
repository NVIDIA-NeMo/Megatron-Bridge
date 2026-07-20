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

"""GB200 text-only pretraining recipes for Qwen3.5 dense and MoE models."""

from __future__ import annotations

import torch
from transformers import AutoConfig

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_QWEN35_9B_BASE = "Qwen/Qwen3.5-9B-Base"
_QWEN35_35B_A3B_BASE = "Qwen/Qwen3.5-35B-A3B-Base"


def qwen35_text_9b_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a text-only Qwen3.5-9B pretraining config for eight GB200 GPUs."""
    cfg = _pretrain_common()

    text_config = AutoConfig.from_pretrained(_QWEN35_9B_BASE).text_config
    # The nested text config intentionally omits ``architectures``. AutoBridge
    # needs it to select the registered causal-LM bridge instead of the VLM.
    text_config.architectures = ["Qwen3_5ForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = _QWEN35_9B_BASE
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Follow the Llama 3 8B GB200 topology: keep model parallelism at one and
    # use all eight GPUs for data parallelism.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 2

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.bias_activation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.apply_rope_fusion = True

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Capture the dense attention and MLP modules. Keep cross entropy on the
    # native fused path validated by the 64-GPU GB200 performance run.
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = ["attn", "mlp"]
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.rerun_state_machine.check_for_nan_in_loss = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    return cfg


def qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a text-only Qwen3.5-35B-A3B pretraining config for eight GB200 GPUs."""
    cfg = _pretrain_common()

    text_config = AutoConfig.from_pretrained(_QWEN35_35B_A3B_BASE).text_config
    # The nested text config intentionally omits ``architectures``. AutoBridge
    # needs it to select the registered causal-LM bridge instead of the VLM.
    text_config.architectures = ["Qwen3_5MoeForCausalLM"]
    cfg.model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = _QWEN35_35B_A3B_BASE
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Match the Qwen3.5-VL GB200 topology while training only the text model.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02
    cfg.train.global_batch_size = 512
    # MBS4 is suitable for force-balanced throughput benchmarking, but OOMs
    # with learned routing. MBS1 was validated with real RP2 data on GB200.
    cfg.train.micro_batch_size = 1

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.bias_activation_fusion = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    # Keep the library-safe native implementation instead of the performance
    # harness's TE cross-entropy path, which currently warns about stability.
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.apply_rope_fusion = True

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Fixed-length text batches can use the scopes that the VLM recipe must
    # disable for variable-length multimodal inputs.
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )
    return cfg
