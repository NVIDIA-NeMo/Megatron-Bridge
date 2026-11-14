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

"""Bounded GB200 training recipes for Inkling2 Small (594B total, 17B active parameters).

Every recipe keeps each expert-parallel group of 16 or 32 GPUs inside one NVL72
NVLink domain and dispatches the 384 routed experts with HybridEP. Inkling2 is an
MCore HybridModel whose decoder blocks are an attention and a MoE layer (``*E``);
the Hybrid pattern's ``|`` separators set the pipeline stages. The image encoder
stays frozen on text data.

Pretraining holds the optimizer state of every expert slice on 128 GPUs: four
pipeline stages of 13, 13, 12, and 10 decoder blocks with EP32 and full activation recompute. The
packed SFT and LoRA recipes train GBS 8 on eight data-parallel replicas, one pack
each, so a step is a single pass through the 48 decoder blocks and more pipeline
stages only add GPUs. Full SFT therefore also runs on 96 GPUs, the smallest layout
that holds its optimizer state, and both full-SFT recipes spend their spare memory
on recomputing fewer layers, none at 128 GPUs; LoRA runs the frozen base in a single
stage on 32 or 16 GPUs. The long-context recipe splits every 32K pack over four context-parallel
ranks. The image-text recipes train on CORD-v2 receipts through the checkpoint's
Inkling2 processor; their full SFT also trains the image encoder.

Each recipe is self-contained: it sets every model, data, and training option it
relies on, even where that repeats another recipe.
"""

from __future__ import annotations

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.inkling.data.tokenizer import inkling2_tokenizer_metadata
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import (
    _peft_common,
    _peft_common_vlm,
    _pretrain_common,
    _sft_common,
    _sft_common_vlm,
)
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config, default_tulu3_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_MODEL_ID = "thinkingmachines/Inkling-2-Small"
_TULU3_REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"  # pragma: allowlist secret
_CORD_V2_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"  # pragma: allowlist secret


def inkling2_small_pretrain_128gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a 100-step bounded text pretraining config on 128 GB200 GPUs.

    TP1/PP4/EP32: each pipeline stage holds one EP32 group of 32 contiguous ranks
    (8 nodes) inside one NVL72 domain, which dispatches the experts with HybridEP;
    full activation recompute keeps the AttnRes residual blocks of in-flight
    microbatches small. The stages hold 13, 13, 12, and 10 decoder blocks: a
    block's cost grows with depth, as the AttnRes state it reads grows, and the
    last stage also computes the output layer and loss. Twelve blocks per stage
    makes the step about 11% slower; the first stage, with four microbatches in
    flight, peaks near 114 GB. Pass the imported checkpoint with
    ``--pretrained_checkpoint`` for natural expert routing; the recipe leaves it
    unset. The peak learning rate is 3e-5 because the run continues a trained
    model rather than starting from random weights.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    # Inkling2MegatronTokenizer assigns the BOS and EOS roles that the checkpoint's tokenizer leaves unset.
    cfg.tokenizer.metadata_path = inkling2_tokenizer_metadata()
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    # Later blocks cost more and the last stage also computes the loss, so the earlier stages hold more blocks.
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (13, 13, 12, 10))
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    # Text data never exercises the image encoder.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    # Run the AttnRes mixtures, short convolutions, clips, and SiTU activations as compiled, fused kernels.
    cfg.model.inkling2_compile_ops = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    # HybridEP: each EP32 group is one pipeline stage's 32 contiguous ranks (8 nodes) in one NVL72 domain.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_permute_fusion_into_hybridep = True

    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8
    cfg.dataset.random_seed = 1234

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False
    # Overlapping the data-parallel gradient reduce and parameter gather makes the step about 4% slower.
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 1234
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    # A full garbage collection stalls the step it runs in by seconds; collect every 100 steps.
    cfg.train.manual_gc_interval = 100
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=40,
        max_lr=3e-5,
        min_lr=3e-6,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def inkling2_small_sft_128gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a 100-step full SFT config on packed 8K Tulu 3 with 128 GB200 GPUs.

    TP4/PP4/EP32 with sequence parallelism leaves eight data-parallel replicas,
    one 8K pack each per step (GBS 8), so a step is a single pass through the 48
    decoder blocks. That pass's activations fit without recompute: the last stage
    peaks near 145 GB, and recomputing the first 6 of each stage's 24 Hybrid layers
    makes the step about 13% slower. The 96-GPU recipe trains the same run on fewer GPUs.
    Conversations are rendered with the checkpoint's chat template and packs are aligned
    to the TP4 sequence-parallel width. The final step saves the full model weights
    for HF export.
    """
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    # Inkling2MegatronTokenizer assigns the BOS and EOS roles that the checkpoint's tokenizer leaves unset.
    cfg.tokenizer.metadata_path = inkling2_tokenizer_metadata()
    cfg.model.seq_length = 8192
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (12, 12, 12, 12))
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    # Text data never exercises the image encoder.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    # Run the AttnRes mixtures, short convolutions, clips, and SiTU activations as compiled, fused kernels.
    cfg.model.inkling2_compile_ops = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    # HybridEP: each EP32 group is one pipeline stage's 32 contiguous ranks (8 nodes) in one NVL72 domain.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_permute_fusion_into_hybridep = True
    # Packed sequences can give EP ranks different token counts; HybridEP needs equal ones, so pad to the group max.
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True

    # Sequence parallelism over TP4 aligns each packed sequence to a multiple of 4.
    cfg.dataset = default_tulu3_config(seq_length=8192, enable_offline_packing=True, pad_seq_to_mult=4)
    # Fixed-width packs keep one compiled FlexAttention shape; packs of varying width would recompile it
    # for every new length.
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = "work/data/inkling2/tulu3-10k-8k"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 10000
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.max_single_sequence_length = 8192 - 4

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    # A full garbage collection stalls the step it runs in by seconds; collect every 100 steps.
    cfg.train.manual_gc_interval = 100
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=5e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def inkling2_small_sft_96gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a 100-step full SFT config on packed 8K Tulu 3 with 96 GB200 GPUs.

    Trains the same data, batch, and schedule as the 128-GPU recipe on three pipeline
    stages instead of four. TP4/PP3/EP32 with sequence parallelism leaves eight
    data-parallel replicas, one 8K pack each per step (GBS 8), so a step is a single
    pass through the 48 decoder blocks and a fourth stage only adds GPUs. Each stage
    holds 16 decoder blocks and one EP32 group of 32 contiguous ranks (8 nodes) inside
    one NVL72 domain. Only the first 20 of each stage's 32 Hybrid layers recompute their
    activations, which keeps the last stage near 151 GB; full recompute makes the step
    about 8% slower. The final step saves the full model weights for HF export.
    """
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    # Inkling2MegatronTokenizer assigns the BOS and EOS roles that the checkpoint's tokenizer leaves unset.
    cfg.tokenizer.metadata_path = inkling2_tokenizer_metadata()
    cfg.model.seq_length = 8192
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 3
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (16, 16, 16))
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    # Text data never exercises the image encoder.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    # Run the AttnRes mixtures, short convolutions, clips, and SiTU activations as compiled, fused kernels.
    cfg.model.inkling2_compile_ops = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "block"
    cfg.model.recompute_num_layers = 20
    # HybridEP: each EP32 group is one pipeline stage's 32 contiguous ranks (8 nodes) in one NVL72 domain.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_permute_fusion_into_hybridep = True
    # Packed sequences can give EP ranks different token counts; HybridEP needs equal ones, so pad to the group max.
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True

    # Sequence parallelism over TP4 aligns each packed sequence to a multiple of 4.
    cfg.dataset = default_tulu3_config(seq_length=8192, enable_offline_packing=True, pad_seq_to_mult=4)
    # Fixed-width packs keep one compiled FlexAttention shape; packs of varying width would recompile it
    # for every new length.
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = "work/data/inkling2/tulu3-10k-8k"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 10000
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.max_single_sequence_length = 8192 - 4

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    # A full garbage collection stalls the step it runs in by seconds; collect every 100 steps.
    cfg.train.manual_gc_interval = 100
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=5e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def inkling2_small_sft_128gpu_gb200_bf16_long_context_config() -> ConfigContainer:
    """Return a 20-step full SFT config on packed 32K Tulu 3 with 128 GB200 GPUs.

    TP4/PP4/EP32/CP4 with sequence parallelism. Context parallelism leaves two
    data-parallel replicas, so each replica runs four packs per step and keeps the
    four pipeline stages busy, where CP 1 would give each replica one pack and run the
    stages one after another. FlexAttention keeps attention memory linear in the
    sequence length. Only the first 20 of each stage's 24 Hybrid layers recompute
    their activations: the first stage, with four packs in flight, peaks near 120 GB,
    and full recompute makes the step about 7% slower.
    """
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    # Inkling2MegatronTokenizer assigns the BOS and EOS roles that the checkpoint's tokenizer leaves unset.
    cfg.tokenizer.metadata_path = inkling2_tokenizer_metadata()
    cfg.model.seq_length = 32768
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (12, 12, 12, 12))
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 4
    cfg.model.sequence_parallel = True
    # Context-parallel SFT weights the loss by token across ranks.
    cfg.model.calculate_per_token_loss = True
    # Text data never exercises the image encoder.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    # Run the AttnRes mixtures, short convolutions, clips, and SiTU activations as compiled, fused kernels.
    cfg.model.inkling2_compile_ops = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "block"
    cfg.model.recompute_num_layers = 20
    # HybridEP: each EP32 group is one pipeline stage's 32 contiguous ranks (8 nodes) in one NVL72 domain.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_permute_fusion_into_hybridep = True
    # Packed sequences can give EP ranks different token counts; HybridEP needs equal ones, so pad to the group max.
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True

    # CP4 cuts each packed sequence into 8 chunks and sequence parallelism splits every CP rank's share over TP4,
    # so each sequence is aligned to lcm(2 x 4, 4 x 4) = 16 tokens.
    cfg.dataset = default_tulu3_config(seq_length=32768, enable_offline_packing=True, pad_seq_to_mult=16)
    # Fixed-width packs keep one compiled FlexAttention shape; packs of varying width would recompile it
    # for every new length.
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = "work/data/inkling2/tulu3-10k-32k-pad16"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 10000
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.max_single_sequence_length = 32768 - 16

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 20
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    # A full garbage collection stalls the step it runs in by seconds; collect every 100 steps.
    cfg.train.manual_gc_interval = 100
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=2,
        max_lr=1e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 20
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = None
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def inkling2_small_peft_32gpu_gb200_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a 100-step LoRA config on packed 8K Tulu 3 with 32 GB200 GPUs.

    LoRA rank 8 / alpha 16 adapts the attention Q, K, V, and output projections of
    the frozen base. The base runs in one pipeline stage at TP4/EP32 with sequence
    parallelism. The EP32 group is all 32 GPUs (8 nodes) in one NVL72 domain and
    dispatches the experts with HybridEP. Eight data-parallel replicas take one 8K pack
    each per step (GBS 8), so a step is a single pass through the 48 decoder blocks.
    The frozen base keeps activations small enough to skip activation recompute: peak
    memory stays near 121 GB.
    """
    cfg = _peft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    # Inkling2MegatronTokenizer assigns the BOS and EOS roles that the checkpoint's tokenizer leaves unset.
    cfg.tokenizer.metadata_path = inkling2_tokenizer_metadata()
    cfg.model.seq_length = 8192
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.hybrid_layer_pattern = "*E" * 48
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    # Text data never exercises the image encoder.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    # Run the AttnRes mixtures, short convolutions, clips, and SiTU activations as compiled, fused kernels.
    cfg.model.inkling2_compile_ops = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    # HybridEP: the EP32 group is all 32 ranks (8 nodes) in one NVL72 domain.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_permute_fusion_into_hybridep = True
    # Packed sequences can give EP ranks different token counts; HybridEP needs equal ones, so pad to the group max.
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True

    # Sequence parallelism over TP4 aligns each packed sequence to a multiple of 4.
    cfg.dataset = default_tulu3_config(seq_length=8192, enable_offline_packing=True, pad_seq_to_mult=4)
    # Fixed-width packs keep one compiled FlexAttention shape; packs of varying width would recompile it
    # for every new length.
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = "work/data/inkling2/tulu3-10k-8k"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 10000
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.max_single_sequence_length = 8192 - 4

    peft_cfg = default_peft_config(peft_scheme)
    peft_cfg.target_modules = ["linear_q", "linear_k", "linear_v", "linear_proj"]
    peft_cfg.dim = 8
    peft_cfg.alpha = 16
    peft_cfg.dropout = 0.0
    cfg.peft = peft_cfg

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    # A full garbage collection stalls the step it runs in by seconds; collect every 100 steps.
    cfg.train.manual_gc_interval = 100
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=1e-4,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def inkling2_small_peft_16gpu_gb200_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a 100-step LoRA config on packed 8K Tulu 3 with 16 GB200 GPUs.

    Trains the same adapters on the same data and schedule as the 32-GPU recipe. The
    frozen base runs in one pipeline stage at TP2/EP16 with sequence parallelism: the
    EP16 group is all 16 GPUs in one NVL72 domain, and eight data-parallel replicas take
    one 8K pack each per step (GBS 8), so a step is a single pass through the 48 decoder
    blocks with full activation recompute.
    """
    cfg = _peft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    # Inkling2MegatronTokenizer assigns the BOS and EOS roles that the checkpoint's tokenizer leaves unset.
    cfg.tokenizer.metadata_path = inkling2_tokenizer_metadata()
    cfg.model.seq_length = 8192
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.hybrid_layer_pattern = "*E" * 48
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    # Text data never exercises the image encoder.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    # Run the AttnRes mixtures, short convolutions, clips, and SiTU activations as compiled, fused kernels.
    cfg.model.inkling2_compile_ops = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    # HybridEP: the EP16 group is all 16 ranks (4 nodes) in one NVL72 domain.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_permute_fusion_into_hybridep = True
    # Packed sequences can give EP ranks different token counts; HybridEP needs equal ones, so pad to the group max.
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True

    # Sequence parallelism over TP2 needs packs aligned to 2; the 32-GPU recipe's alignment of 4 keeps the same packs.
    cfg.dataset = default_tulu3_config(seq_length=8192, enable_offline_packing=True, pad_seq_to_mult=4)
    # Fixed-width packs keep one compiled FlexAttention shape; packs of varying width would recompile it
    # for every new length.
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = "work/data/inkling2/tulu3-10k-8k"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 10000
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.max_single_sequence_length = 8192 - 4

    peft_cfg = default_peft_config(peft_scheme)
    peft_cfg.target_modules = ["linear_q", "linear_k", "linear_v", "linear_proj"]
    peft_cfg.dim = 8
    peft_cfg.alpha = 16
    peft_cfg.dropout = 0.0
    cfg.peft = peft_cfg

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    # A full garbage collection stalls the step it runs in by seconds; collect every 100 steps.
    cfg.train.manual_gc_interval = 100
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=1e-4,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 16,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def inkling2_small_sft_128gpu_gb200_bf16_multimodal_config() -> ConfigContainer:
    """Return a 100-step full SFT config on image-text CORD-v2 with 128 GB200 GPUs.

    TP4/PP4/EP32 with sequence parallelism leaves eight data-parallel replicas, eight
    receipts each per step (GBS 64). Images take one token per 40x40-pixel patch, and a
    receipt over 3,072 image tokens is encoded at the largest same-aspect size within that
    budget; the image encoder and its projection train with the language model. The final
    step saves the full model weights for HF export.
    """
    cfg = _sft_common_vlm()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (12, 12, 12, 12))
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    # Run the AttnRes mixtures, short convolutions, clips, and SiTU activations as compiled, fused kernels.
    cfg.model.inkling2_compile_ops = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    # HybridEP: each EP32 group is one pipeline stage's 32 contiguous ranks (8 nodes) in one NVL72 domain.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_permute_fusion_into_hybridep = True
    # Every sample is padded to the full sequence length, so every rank dispatches the same number of tokens.
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = False

    cfg.dataset.seq_length = 4096
    # The checkpoint holds the tokenizer and processor_config.json. Name Bridge's Inkling2Processor, which marks the
    # assistant turns for the loss and applies the image budget: from Transformers 5.23, AutoProcessor resolves
    # Transformers' own processor of the same name.
    cfg.dataset.hf_processor_path = _MODEL_ID
    cfg.dataset.hf_processor_class = "megatron.bridge.models.inkling.data.processor.Inkling2Processor"
    # CORD-v2 phone photos run to 7,676 image tokens at native size. Larger receipts are encoded at the largest
    # same-aspect size within 3,072 tokens (the reference Lanczos3 resize), which leaves room for the answer.
    cfg.dataset.hf_processor_kwargs = {"max_image_tokens": 3072}
    cfg.dataset.source.split = "train"
    cfg.dataset.source.load_kwargs = {"revision": _CORD_V2_REVISION}
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.num_workers = 4
    cfg.dataset.dataloader_type = "cyclic"
    # One padded width keeps one compiled FlexAttention shape.
    cfg.dataset.pad_to_max_length = True
    cfg.dataset.enable_in_batch_packing = False

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    # A full garbage collection stalls the step it runs in by seconds; collect every 100 steps.
    cfg.train.manual_gc_interval = 100
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=5e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def inkling2_small_peft_32gpu_gb200_bf16_multimodal_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a 100-step LoRA config on image-text CORD-v2 with 32 GB200 GPUs.

    LoRA rank 8 / alpha 16 adapts the language model's attention Q, K, V, and output
    projections while the base model, image encoder included, stays frozen. TP4/PP2/EP16
    with sequence parallelism leaves four data-parallel replicas, two receipts each per
    step (GBS 8), with the same 3,072-token image budget as full SFT. The frozen base
    needs no activation recompute: peak memory stays near 71 GB.
    """
    cfg = _peft_common_vlm()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (24, 24))
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.gradient_accumulation_fusion = True
    # Run the AttnRes mixtures, short convolutions, clips, and SiTU activations as compiled, fused kernels.
    cfg.model.inkling2_compile_ops = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    # HybridEP: each EP16 group is one pipeline stage's 16 contiguous ranks (4 nodes) in one NVL72 domain.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_permute_fusion_into_hybridep = True
    # Every sample is padded to the full sequence length, so every rank dispatches the same number of tokens.
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = False

    cfg.dataset.seq_length = 4096
    # The checkpoint holds the tokenizer and processor_config.json. Name Bridge's Inkling2Processor, which marks the
    # assistant turns for the loss and applies the image budget: from Transformers 5.23, AutoProcessor resolves
    # Transformers' own processor of the same name.
    cfg.dataset.hf_processor_path = _MODEL_ID
    cfg.dataset.hf_processor_class = "megatron.bridge.models.inkling.data.processor.Inkling2Processor"
    # CORD-v2 phone photos run to 7,676 image tokens at native size. Larger receipts are encoded at the largest
    # same-aspect size within 3,072 tokens (the reference Lanczos3 resize), which leaves room for the answer.
    cfg.dataset.hf_processor_kwargs = {"max_image_tokens": 3072}
    cfg.dataset.source.split = "train"
    cfg.dataset.source.load_kwargs = {"revision": _CORD_V2_REVISION}
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.num_workers = 4
    cfg.dataset.dataloader_type = "cyclic"
    # One padded width keeps one compiled FlexAttention shape.
    cfg.dataset.pad_to_max_length = True
    cfg.dataset.enable_in_batch_packing = False

    peft_cfg = default_peft_config(peft_scheme)
    peft_cfg.target_modules = ["linear_q", "linear_k", "linear_v", "linear_proj"]
    peft_cfg.dim = 8
    peft_cfg.alpha = 16
    peft_cfg.dropout = 0.0
    cfg.peft = peft_cfg

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    # A full garbage collection stalls the step it runs in by seconds; collect every 100 steps.
    cfg.train.manual_gc_interval = 100
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=1e-4,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 16,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


__all__ = [
    "inkling2_small_peft_16gpu_gb200_bf16_config",
    "inkling2_small_peft_32gpu_gb200_bf16_config",
    "inkling2_small_peft_32gpu_gb200_bf16_multimodal_config",
    "inkling2_small_pretrain_128gpu_gb200_bf16_config",
    "inkling2_small_sft_128gpu_gb200_bf16_config",
    "inkling2_small_sft_96gpu_gb200_bf16_config",
    "inkling2_small_sft_128gpu_gb200_bf16_long_context_config",
    "inkling2_small_sft_128gpu_gb200_bf16_multimodal_config",
]
