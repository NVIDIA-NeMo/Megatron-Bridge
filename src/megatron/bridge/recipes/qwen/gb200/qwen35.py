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
from megatron.bridge.data.builders.synthetic_varlen import SyntheticVarlenDatasetConfig
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


def qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_dynamic_cp_config() -> ConfigContainer:
    """Return a Qwen3.5-35B-A3B long-context config with online sequence packing and dynamic CP.

    Starts from :func:`qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config` and switches
    the data path to Megatron-Core's online sequence-packing scheduler: samples with
    variable lengths are packed into THD bins once per step, and every bin runs on
    a context-parallel group sized for its longest sequence (``default_dynamic_cp``)
    instead of the static CP4 that a 32k cap would otherwise force on every sample.

    Topology: TP1 PP1 CP4 EP8 on eight GB200 GPUs (a DPxCP pool of 8 ranks). A sample
    of at most ``max_seqlen_per_dp_cp_rank`` tokens runs on one GPU, up to twice that
    on CP2, and so on up to the full pool. The synthetic dataset has a lognormal
    length distribution so all CP classes occur; replace ``cfg.dataset`` with a real
    variable-length provider for actual training.

    Requires the Megatron-Core ``dev`` submodule pin (``./scripts/switch_mcore.sh dev``);
    ``ConfigContainer.validate`` reports a clear error on the ``main`` pin. Qwen3.5 attention
    uses 256-wide heads, which Transformer Engine only trains with FlashAttention; until
    Megatron-Core derives ``pad_between_seqs`` from the actual ``cu_seqlens``
    (https://github.com/NVIDIA/Megatron-LM/pull/7417), FlashAttention also needs bins without padding between sequences, which the
    dataset's ``fold_padding_into_sequence`` provides. See
    ``docs/training/dynamic-context-parallel.md`` for the backend and memory notes.
    """
    cfg = qwen35_text_35b_a3b_pretrain_8gpu_gb200_bf16_config()

    seq_length = 32768
    context_parallel_size = 4
    cfg.model.seq_length = seq_length
    cfg.dataset = SyntheticVarlenDatasetConfig(
        seq_length=seq_length,
        min_seq_length=64,
        median_seq_length=4096,
        lognormal_sigma=1.0,
        # Qwen3.5 attention heads are 256 wide, which Transformer Engine only trains with
        # FlashAttention; report the CP alignment padding as part of each sequence so the
        # packed bins carry no padding between sequences (padding stays loss-masked).
        fold_padding_into_sequence=True,
        random_seed=1234,
        num_workers=2,
    )

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = context_parallel_size
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False

    # Online packing + dynamic CP.
    cfg.model.sequence_packing_scheduler = "default_dynamic_cp"
    cfg.model.dynamic_context_parallel = True
    cfg.model.min_dynamic_context_parallel_size = 1
    cfg.model.max_seqlen_per_dp_cp_rank = seq_length // context_parallel_size
    cfg.model.calculate_per_token_loss = True
    cfg.ddp.average_in_collective = False
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 256

    # Packed THD microbatches have data-dependent shapes: no CUDA graphs, and the
    # all-to-all MoE dispatcher (the scheduler supports alltoall and flex).
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = None
    cfg.model.cuda_graph_modules = None
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_flex_dispatcher_num_sms = None
    cfg.model.moe_hybridep_num_sms = None

    # Long packed bins: recompute every layer and keep optimizer moments in bf16.
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    return cfg
