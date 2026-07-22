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
"""GB300 performance recipes for NemotronH and Nemotron 3."""

from megatron.bridge.perf_recipes.nemotronh.common import (
    _TE_QUANT_CFG_PATH,
    ConfigContainer,
    _apply_nemotron_3_super_perf_defaults,
    _apply_nemotron_3_ultra_fsdp_hsdp,
    _apply_nemotron_3_ultra_perf_defaults,
    _benchmark_common,
    _nemotron_3_super_nvfp4_precision,
    _perf_precision,
    _with_global_batch_size,
    load_quantization_recipe,
    nemotron_3_nano_pretrain_config,
    nemotron_3_super_pretrain_config,
    nemotron_3_ultra_pretrain_config,
    nemotronh_56b_pretrain_config,
)


def nemotronh_56b_pretrain_64gpu_gb300_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 64× GB300, FP8 current-scaling."""
    cfg = nemotronh_56b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 192
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba", "attn"]

    _benchmark_common(cfg)
    return cfg


def nemotron_3_super_pretrain_64gpu_gb300_bf16_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB300, BF16."""
    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    _apply_nemotron_3_super_perf_defaults(cfg)
    return cfg


def nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB300, MXFP8."""
    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_padding_for_quantization = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    _apply_nemotron_3_super_perf_defaults(cfg)
    return cfg


def nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB300, NVFP4."""
    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = _nemotron_3_super_nvfp4_precision()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.quant_recipe = load_quantization_recipe(str(_TE_QUANT_CFG_PATH))

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    _apply_nemotron_3_super_perf_defaults(cfg)
    return cfg


def _nemotron_3_ultra_gb300_fp8mx_config(
    *, num_gpus: int, expert_model_parallel_size: int, global_batch_size: int
) -> ConfigContainer:
    """Shared builder for Nemotron 3 Ultra GB300 MXFP8 Megatron-FSDP perf recipes."""
    cfg = nemotron_3_ultra_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    """
    Uses TP1 / PP1 / CP1 / EP64 / ETP1, GBS 256 / MBS 1,
    seq 8192, HybridEP flex dispatcher, CuteDSL fused grouped MLP, selective
    recompute + activation offload of the expert MLP, MTP=2. The MoE architecture
    (512 experts, latent MoE, MTP, squared-relu, hybrid Mamba/attention pattern,
    ...) is inherited from the base recipe via ``AutoBridge``.
    """
    # Parallelism
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 8192

    # Only tensors larger than 500MB are offloaded, which
    # approximates offloading the moe_act input for seq 8192 / MBS 1.
    cfg.model.min_offloaded_tensor_size = 500_000_000

    # MXFP8 requires router padding for quantization.
    cfg.model.moe_router_padding_for_quantization = True

    # GPU-count specific overrides of the canonical (256-GPU / EP64) defaults.
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.train.global_batch_size = global_batch_size

    # Fine-grained activation offloading. Requires NVTE_CPU_OFFLOAD_V1=1 in the
    # launch environment (set by perf_plugins.py).
    # NOTE: also requires setting the min_offloaded_tensor_size to avoid CPU OOM issues
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["fused_group_mlp"]

    # Selective recompute of the MoE activation
    # recomputes the activation output of the MoE expert MLP, while FC1 output (activation input) is saved and offloaded to cpu
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act"]

    _apply_nemotron_3_ultra_perf_defaults(cfg)

    # Apply HSDP / FSDP dtype overrides last so they win over the generic defaults.
    _apply_nemotron_3_ultra_fsdp_hsdp(cfg, num_gpus=num_gpus)

    return cfg


def nemotron_3_ultra_pretrain_256gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Ultra (550B-A55B LatentMoE) pretrain: 256× GB300, MXFP8, Megatron-FSDP (HSDP).

    TP1 / PP1 / CP1 / EP64 / ETP1, GBS 256 / MBS 1, seq 8192, BF16 + MXFP8 mixed
    precision, HybridEP flex dispatcher, CuteDSL fused grouped MLP, selective
    recompute + fine-grained activation offload of the expert MLP, MTP=2.
    """
    return _nemotron_3_ultra_gb300_fp8mx_config(num_gpus=256, expert_model_parallel_size=64, global_batch_size=256)


def nemotron_3_nano_pretrain_8gpu_gb300_bf16_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB300, BF16."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    cfg.model.moe_hybridep_num_sms = 16
    return cfg


def nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB300, MXFP8."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    cfg.model.moe_hybridep_num_sms = 16
    return cfg


def nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB300, NVFP4."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    cfg.model.moe_hybridep_num_sms = 16
    return cfg


def nemotronh_56b_pretrain_256gpu_gb300_bf16_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 256× GB300, BF16 (same layout as FP8-CS)."""
    cfg = nemotronh_56b_pretrain_64gpu_gb300_fp8cs_config()
    cfg.mixed_precision = _perf_precision("bf16")
    return cfg


def nemotronh_56b_pretrain_256gpu_gb300_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 256× GB300, FP8 current-scaling, legacy-scaled GBS."""
    return _with_global_batch_size(nemotronh_56b_pretrain_64gpu_gb300_fp8cs_config(), 768)
