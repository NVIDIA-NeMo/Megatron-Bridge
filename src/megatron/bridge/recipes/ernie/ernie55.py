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

"""ERNIE 5.5 recipe on GB300: supervised fine-tuning of a Hugging Face checkpoint.

The recipe trains with the reference implementation's settings: the per-row SFT loss
(``ernie_per_row_loss``), MTP loss scale 0.1, the sparse DSA indexer loss at 0.01, no auxiliary or z loss
(the z loss is still tracked), a router expert-bias update rate of 1e-4, full uniform recompute of every
layer, no fused kernels besides the grouped expert GEMM and the reference's own Triton kernels for SiTU-GLU,
multimax and the context-parallel RoPE (``ernie_fused_ops``, which ``model.ernie_fused_ops=false`` turns off),
the cuDNN DSA backend, and ``dist_muon`` (Muon with
the layer-wise distributed optimizer; AdamW for the embedding, the output layer, the router matrices, the
compressors' APE and every 1-D parameter; the reference's per-head and gate/up Muon row blocks are per-parameter
``qkv_layout`` blocks that Muon orthogonalizes one by one under ``muon_split_qkv``,
see :mod:`~megatron.bridge.models.ernie.modeling_ernie55.muon`) with
weight decay 0.1 on every matrix, a 2e-5 to 2e-6 cosine schedule and gradient clipping at 1.0 (the multimax
head is clipped by its own norm). Train it with the ERNIE 5.5 step, which builds the packed THD rows and
selects the per-row loss: ``scripts/training/run_recipe.py --recipe <name> --step_func ernie55_step``.

:func:`ernie55_sft_64gpu_gb300_bf16_config` fine-tunes a Hugging Face ERNIE 5.5 checkpoint (``config.json``,
safetensors and tokenizer files) at the reference's production shape, CP64 EP64 at 256K tokens per row on 16 nodes
of 4 GPUs. It runs Bridge's ``finetune`` loop, which ``run_recipe.py`` infers from its name; it loads the checkpoint
of ``hf_path`` (or ``$ERNIE55_HF_PATH``) as ``checkpoint.pretrained_checkpoint`` and reads its architecture. Like
every Bridge recipe it builds without arguments; without ``hf_path`` and ``$ERNIE55_HF_PATH`` it uses the placeholder
:data:`ERNIE55_FULL_SIZE_HF_CONFIG`, which is not ERNIE 5.5's layout, and finetuning then stops in setup.

The tokenizer is ``NullTokenizer`` with the model's unpadded vocabulary; the data is SFT JSONL (``dataset.path``,
text rows with ``dataset.hf_tokenizer`` or pre-tokenized rows) or synthetic rows
(:class:`~megatron.bridge.models.ernie.ernie55_data.Ernie55SFTDatasetConfig`).

ERNIE 5.5 needs Megatron-Core dev at or after f78809f6d, which has the
``DSv4HybridAttention._post_o_group_proj`` hook.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.ernie.configuration_ernie55 import Ernie55Config
from megatron.bridge.models.ernie.ernie55_data import Ernie55SFTDatasetConfig
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_muon_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


__all__ = [
    "ERNIE55_FULL_SIZE_HF_CONFIG",
    "ERNIE55_HF_PATH_ENV",
    "ernie55_sft_64gpu_gb300_bf16_config",
]

ERNIE55_HF_PATH_ENV = "ERNIE55_HF_PATH"
"""Environment variable naming the ``config.json`` source when a recipe gets no ``hf_path``."""

ERNIE55_FULL_SIZE_HF_CONFIG: dict[str, Any] = {
    "architectures": ["Ernie5_5_MoeForCausalLM"],
    "dtype": "bfloat16",
    "vocab_size": 200027,
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "intermediate_size": 12288,
    "first_k_dense_replace": 0,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "v_head_dim": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 448,
    "q_lora_rank": 1024,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "rope_theta": 8000000.0,
    "compress_rope_theta": 160000.0,
    "sliding_window": 128,
    "compress_ratios": [128, -2] * 21 + [-2, -2],
    "csa_dense_mode": True,
    "hca_rope_type": "rope",
    "rotary_scaling_factor": 1.0,
    "original_max_position_embeddings": 4096,
    "hybrid_mla_attention": "mqa_dsa",
    "hybrid_mla_num_attention_heads": 64,
    "hybrid_mla_q_lora_rank": 1024,
    "hybrid_mla_kv_lora_rank": 512,
    "hybrid_mla_qk_nope_head_dim": 448,
    "hybrid_mla_qk_rope_head_dim": 64,
    "hybrid_mla_v_head_dim": 256,
    "mqa_latent_rope_adjacent_pairing": True,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 2048,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 2048,
    "n_shared_experts": 1,
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "latent_moe_use_norm": False,
    "num_nextn_predict_layers": 1,
    "mtp_shared_last_layer": True,
    "sigmoid_gate_fusion": False,
    "use_qk_norm": True,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "pad_token_id": 151643,
}
"""A placeholder full-size ``config.json``, not taken from a released checkpoint, that the recipe builds when it gets
no ``hf_path``: 21 alternating H and D layers, one more D layer and one shared-body D-layer MTP depth, all MoE with
256 routed experts (DeepSeek-V4-Flash backbone sizes), with the reference's vocabulary and attention geometry. Pass
``hf_path`` to build the model from a real checkpoint's ``config.json``."""

_PRODUCTION_SEQ_LENGTH = 262144
# Synthetic tokens stay below the reference tokenizer's special tokens, which start at 151643.
_SYNTHETIC_TOKEN_RANGE = (1, 151643)


def _ernie55_hf_config(hf_path: str | None) -> tuple[PretrainedConfig, str | None]:
    """The ERNIE 5.5 ``config.json`` of ``hf_path``, of ``$ERNIE55_HF_PATH``, or :data:`ERNIE55_FULL_SIZE_HF_CONFIG`.

    Returns:
        The Hugging Face config and the path it was read from (``None`` for the built-in config).
    """
    path = hf_path or os.environ.get(ERNIE55_HF_PATH_ENV) or None
    if path is not None:
        return AutoConfig.from_pretrained(path), path
    return Ernie55Config(**ERNIE55_FULL_SIZE_HF_CONFIG), None


def _ernie55_common_config(
    hf_config: PretrainedConfig,
    *,
    seq_length: int,
    context_parallel_size: int,
    expert_model_parallel_size: int,
    train_iters: int,
    lr_warmup_iters: int,
    global_batch_size: int,
) -> ConfigContainer:
    """Model, loss, optimizer, schedule, DDP, precision and tokenizer settings of the ERNIE 5.5 recipe.

    Data, checkpointing and the process environment are left to the recipe.
    """
    cfg = _sft_common()
    cfg.model = AutoBridge.from_hf_config(hf_config).get_model_config()
    model = cfg.model

    # Parallelism: no tensor or pipeline parallelism; experts over EP, the sequence over CP.
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.virtual_pipeline_model_parallel_size = None
    model.context_parallel_size = context_parallel_size
    model.expert_model_parallel_size = expert_model_parallel_size
    model.expert_tensor_parallel_size = 1
    model.sequence_parallel = False
    model.seq_length = seq_length

    # Kernels: the cuDNN DSA backend, no fused kernel besides the grouped expert GEMM and the reference's Triton
    # kernels (SiTU-GLU, multimax, context-parallel RoPE).
    model.ernie_fused_ops = True
    # Transformer Engine's RMSNorm; model.ernie_reference_rmsnorm=true switches to the reference's eager arithmetic.
    model.ernie_reference_rmsnorm = False
    model.attention_backend = AttnBackend.local
    model.dsa_kernel_backend = "cudnn"
    model.apply_rope_fusion = False
    model.masked_softmax_fusion = False
    model.bias_activation_fusion = False
    model.bias_dropout_fusion = False
    model.gradient_accumulation_fusion = False
    model.persist_layer_norm = False
    model.cross_entropy_loss_fusion = False
    model.use_fused_mhc = False
    model.cuda_graph_impl = "none"

    # MoE: all-to-all dispatch, fp32 router, no capacity, no auxiliary loss; the z loss is tracked at 0.
    model.moe_token_dispatcher_type = "alltoall"
    model.moe_grouped_gemm = True
    model.moe_permute_fusion = False
    model.moe_router_fusion = False
    model.moe_router_aux_loss_fusion = False
    model.moe_router_dtype = "fp32"
    model.moe_expert_capacity_factor = None
    model.moe_pad_expert_input_to_capacity = False
    model.moe_aux_loss_coeff = 0.0
    model.moe_z_loss_coeff = 0.0
    model.moe_router_bias_update_rate = 1e-4
    model.moe_router_skip_muon = True
    model.moe_shared_expert_overlap = False

    # Losses: per-row LM and MTP losses, MTP at 0.1, the sparse indexer KL at 0.01.
    model.ernie_per_row_loss = True
    model.calculate_per_token_loss = False
    model.mtp_loss_scaling_factor = 0.1
    model.dsa_indexer_loss_coeff = 0.01
    model.dsa_indexer_use_sparse_loss = True

    # Activations: recompute every layer.
    model.recompute_granularity = "full"
    model.recompute_method = "uniform"
    model.recompute_num_layers = 1

    # Muon (layer-wise distributed) for the matrices; AdamW for the embedding, the output layer, the router matrices,
    # the compressors' APE and 1-D tensors.
    cfg.optimizer, cfg.scheduler = distributed_muon_with_cosine_annealing(
        precision="bf16-mixed",
        muon_momentum=0.95,
        muon_use_nesterov=True,
        muon_scale_mode="spectral",
        muon_fp32_matmul_prec="medium",
        muon_num_ns_steps=5,
        muon_tp_mode="blockwise",
        muon_extra_scale_factor=0.2,
        muon_scalar_optimizer="adam",
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
        weight_decay=0.1,
        max_lr=2e-5,
        min_lr=2e-6,
        clip_grad=1.0,
    )
    cfg.optimizer.muon_coefficient_type = "quintic"
    # The reference's per-head and gate/up row blocks: under muon_split_qkv every 2-D weight with a qkv_layout (the
    # ERNIE 5.5 model sets one per row-blocked weight, see modeling_ernie55.muon) is orthogonalized block by block;
    # muon_split_qkv=False would silently orthogonalize them whole. The one-group layouts split the same rows without
    # muon_split_qkv_per_head, which stays off: its batched Newton-Schulz for equal blocks fails on blocks with more
    # rows than columns (k_b_proj, the dense fc1 halves), see modeling_ernie55.muon. Megatron-Core's own MLA layouts
    # stay whole matrices (muon_split_mla_qkv off).
    cfg.optimizer.muon_split_qkv = True
    cfg.optimizer.muon_split_qkv_per_head = False
    cfg.optimizer.muon_split_mla_qkv = False
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.decoupled_weight_decay = True
    # Weight decay skips biases and 1-D tensors (norms, sinks, multimax terms, hyper-connection scalars).
    cfg.optimizer.apply_wd_to_qk_layernorm = False
    cfg.optimizer.use_distributed_optimizer = False
    # The helper's scheduler decays weights by 0.033; the reference uses the optimizer's 0.1 throughout.
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.scheduler.lr_warmup_init = 0.0

    # bf16 parameters with fp32 gradient accumulation and reduction; the averaging is pre-scaled, not in NCCL.
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        average_in_collective=False,
        use_distributed_optimizer=False,
    )
    cfg.comm_overlap = None
    cfg.dist.enable_megatron_core_experimental = False

    # The vocabulary is the model's, unpadded: the multimax head and the loss see exactly vocab_size logits.
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = model.vocab_size
    cfg.tokenizer.use_tokenizer_vocab_size = False
    cfg.tokenizer.make_vocab_size_divisible_by = 1

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.rng.seed = 42
    return cfg


def _ernie55_production_config(hf_config: PretrainedConfig) -> ConfigContainer:
    """The reference's production shape on 16 nodes: CP64 EP64, 256K-token rows, the reference's schedule."""
    cfg = _ernie55_common_config(
        hf_config,
        seq_length=_PRODUCTION_SEQ_LENGTH,
        context_parallel_size=64,
        expert_model_parallel_size=64,
        train_iters=2065,
        lr_warmup_iters=103,
        global_batch_size=2,
    )
    # cp_partition_mode is a Megatron-Core dev field. Without it (Megatron-Core main) the model config's finalize
    # rejects CP > 1 with a clear message instead of an AttributeError here.
    if "cp_partition_mode" in {field.name for field in dataclasses.fields(TransformerConfig)}:
        cfg.model.cp_partition_mode = "contiguous"
    cfg.dataset = Ernie55SFTDatasetConfig(
        seq_length=_PRODUCTION_SEQ_LENGTH,
        pad_token_id=hf_config.pad_token_id,
        vocab_size=hf_config.vocab_size,
        path=None,
        truncation="split",
        pack=False,
        pad_to_multiple_of=2 * 64,
        mock_document_lengths=(_PRODUCTION_SEQ_LENGTH,),
        mock_prompt_fraction=0.25,
        mock_token_range=_SYNTHETIC_TOKEN_RANGE,
        seed=42,
    )
    cfg.checkpoint.save_interval = cfg.train.train_iters
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    return cfg


def ernie55_sft_64gpu_gb300_bf16_config(*, hf_path: str | None = None) -> ConfigContainer:
    """Fine-tune a Hugging Face ERNIE 5.5 checkpoint at the reference's production shape on 16 nodes of 4 GB300 GPUs.

    TP1 PP1 CP64 EP64 (DP1), micro-batch 1, global batch 2, 262144-token rows split contiguously over the 64
    context-parallel ranks (``cp_partition_mode="contiguous"``, the DSv4 layout), 2065 steps with 103 warmup steps,
    the reference's model, loss and optimizer settings (see the module docstring), with the model built from the
    checkpoint's ``config.json`` and starting from its weights (``checkpoint.pretrained_checkpoint``). The data are
    SFT JSONL (``dataset.path``; text rows need ``dataset.hf_tokenizer``), one sample per row, samples longer than
    262144 tokens split into rows (the reference's ``--packing false --truncation_strategy split``); without a path,
    every row is one synthetic 262144-token document. Each document is padded to a multiple of 128 tokens (2 x CP).
    The final checkpoint holds the weights only, as the reference's does.

    Needs Megatron-Core dev with the ``_post_o_group_proj`` hook: Megatron-Core main has no CP partition modes.

    Args:
        hf_path: Directory of the Hugging Face ERNIE 5.5 checkpoint (``config.json`` and safetensors); default
            ``$ERNIE55_HF_PATH``. Required to train.

    Returns:
        The training config; run it with ``--step_func ernie55_step`` (``run_recipe.py`` infers ``finetune``).
        Without ``hf_path`` and ``$ERNIE55_HF_PATH`` it builds the placeholder :data:`ERNIE55_FULL_SIZE_HF_CONFIG`
        with no checkpoint, as Bridge's recipe matrix requires; finetuning then stops in setup. Pass ``hf_path`` or
        set ``$ERNIE55_HF_PATH`` so the model is built from the checkpoint's own ``config.json``.
    """
    hf_config, path = _ernie55_hf_config(hf_path)
    cfg = _ernie55_production_config(hf_config)
    cfg.checkpoint.pretrained_checkpoint = path
    return cfg
