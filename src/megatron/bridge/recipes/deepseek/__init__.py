"""DeepSeek recipe exports.

This module re-exports AutoBridge-based pretrain config helpers for DeepSeek
models (V2, V2-Lite, V3).
"""

# DeepSeek V2/V2-Lite
from .deepseek_v2 import (
    deepseek_v2_lite_pretrain_config,
    deepseek_v2_pretrain_config,
)

# DeepSeek V3
from .deepseek_v3 import (
    deepseek_v3_pretrain_config,
    deepseek_v3_pretrain_config_32nodes,
)

# DeepSeek V3 perf recipes
from .deepseek_v3_perf import (
    # 64 GPU H100 aliases
    deepseek_v3_pretrain_64gpu_h100_bf16_config,
    deepseek_v3_pretrain_64gpu_h100_fp8cs_config,
    # V1 — 256 GPU
    deepseek_v3_pretrain_256gpu_b200_bf16_config,
    deepseek_v3_pretrain_256gpu_b200_fp8cs_config,
    deepseek_v3_pretrain_256gpu_b200_fp8mx_config,
    # V1 — NVFP4 / FP8-SC aliases
    deepseek_v3_pretrain_256gpu_b200_nvfp4_config,
    deepseek_v3_pretrain_256gpu_b300_bf16_config,
    deepseek_v3_pretrain_256gpu_b300_fp8cs_config,
    deepseek_v3_pretrain_256gpu_b300_fp8mx_config,
    deepseek_v3_pretrain_256gpu_b300_nvfp4_config,
    deepseek_v3_pretrain_256gpu_gb200_bf16_config,
    deepseek_v3_pretrain_256gpu_gb200_fp8cs_config,
    deepseek_v3_pretrain_256gpu_gb200_fp8mx_config,
    deepseek_v3_pretrain_256gpu_gb200_nvfp4_config,
    deepseek_v3_pretrain_256gpu_gb300_bf16_config,
    deepseek_v3_pretrain_256gpu_gb300_fp8cs_config,
    deepseek_v3_pretrain_256gpu_gb300_fp8mx_config,
    deepseek_v3_pretrain_256gpu_gb300_nvfp4_config,
    deepseek_v3_pretrain_1024gpu_h100_bf16_config,
    deepseek_v3_pretrain_1024gpu_h100_fp8cs_config,
    deepseek_v3_pretrain_1024gpu_h100_fp8sc_config,
    # V2 — 256 GPU
    deepseek_v3_pretrain_v2_256gpu_b200_bf16_config,
    deepseek_v3_pretrain_v2_256gpu_b200_fp8cs_config,
    deepseek_v3_pretrain_v2_256gpu_b200_fp8mx_config,
    # V2 — NVFP4 / FP8-SC / VR200 aliases
    deepseek_v3_pretrain_v2_256gpu_b200_nvfp4_config,
    deepseek_v3_pretrain_v2_256gpu_b300_bf16_config,
    deepseek_v3_pretrain_v2_256gpu_b300_fp8cs_config,
    deepseek_v3_pretrain_v2_256gpu_b300_fp8mx_config,
    deepseek_v3_pretrain_v2_256gpu_b300_nvfp4_config,
    deepseek_v3_pretrain_v2_256gpu_gb200_bf16_config,
    deepseek_v3_pretrain_v2_256gpu_gb200_fp8cs_config,
    deepseek_v3_pretrain_v2_256gpu_gb200_fp8mx_config,
    deepseek_v3_pretrain_v2_256gpu_gb200_nvfp4_config,
    deepseek_v3_pretrain_v2_256gpu_gb300_bf16_config,
    deepseek_v3_pretrain_v2_256gpu_gb300_fp8cs_config,
    deepseek_v3_pretrain_v2_256gpu_gb300_fp8mx_config,
    deepseek_v3_pretrain_v2_256gpu_gb300_nvfp4_config,
    deepseek_v3_pretrain_v2_256gpu_vr200_bf16_config,
    deepseek_v3_pretrain_v2_256gpu_vr200_fp8cs_config,
    deepseek_v3_pretrain_v2_256gpu_vr200_fp8mx_config,
    deepseek_v3_pretrain_v2_256gpu_vr200_nvfp4_config,
    deepseek_v3_pretrain_v2_1024gpu_h100_bf16_config,
    deepseek_v3_pretrain_v2_1024gpu_h100_fp8cs_config,
    deepseek_v3_pretrain_v2_1024gpu_h100_fp8sc_config,
)


__all__ = [
    # DeepSeek V2/V2-Lite
    "deepseek_v2_pretrain_config",
    "deepseek_v2_lite_pretrain_config",
    # DeepSeek V3
    "deepseek_v3_pretrain_config",
    "deepseek_v3_pretrain_config_32nodes",
    # DeepSeek V3 perf recipes — V1 (256 GPU)
    "deepseek_v3_pretrain_256gpu_gb300_bf16_config",
    "deepseek_v3_pretrain_256gpu_gb300_fp8cs_config",
    "deepseek_v3_pretrain_256gpu_gb300_fp8mx_config",
    "deepseek_v3_pretrain_256gpu_gb300_nvfp4_config",
    "deepseek_v3_pretrain_256gpu_gb200_bf16_config",
    "deepseek_v3_pretrain_256gpu_gb200_fp8cs_config",
    "deepseek_v3_pretrain_256gpu_gb200_fp8mx_config",
    "deepseek_v3_pretrain_256gpu_b300_bf16_config",
    "deepseek_v3_pretrain_256gpu_b300_fp8cs_config",
    "deepseek_v3_pretrain_256gpu_b300_fp8mx_config",
    "deepseek_v3_pretrain_256gpu_b200_bf16_config",
    "deepseek_v3_pretrain_256gpu_b200_fp8cs_config",
    "deepseek_v3_pretrain_256gpu_b200_fp8mx_config",
    "deepseek_v3_pretrain_1024gpu_h100_bf16_config",
    "deepseek_v3_pretrain_1024gpu_h100_fp8cs_config",
    # DeepSeek V3 perf recipes — V2 (256 GPU)
    "deepseek_v3_pretrain_v2_256gpu_gb300_bf16_config",
    "deepseek_v3_pretrain_v2_256gpu_gb300_fp8cs_config",
    "deepseek_v3_pretrain_v2_256gpu_gb300_fp8mx_config",
    "deepseek_v3_pretrain_v2_256gpu_gb300_nvfp4_config",
    "deepseek_v3_pretrain_v2_256gpu_gb200_bf16_config",
    "deepseek_v3_pretrain_v2_256gpu_gb200_fp8cs_config",
    "deepseek_v3_pretrain_v2_256gpu_gb200_fp8mx_config",
    "deepseek_v3_pretrain_v2_256gpu_b300_bf16_config",
    "deepseek_v3_pretrain_v2_256gpu_b300_fp8cs_config",
    "deepseek_v3_pretrain_v2_256gpu_b300_fp8mx_config",
    "deepseek_v3_pretrain_v2_256gpu_b200_bf16_config",
    "deepseek_v3_pretrain_v2_256gpu_b200_fp8cs_config",
    "deepseek_v3_pretrain_v2_256gpu_b200_fp8mx_config",
    "deepseek_v3_pretrain_v2_1024gpu_h100_bf16_config",
    "deepseek_v3_pretrain_v2_1024gpu_h100_fp8cs_config",
    # DeepSeek V3 perf recipes — V1 NVFP4 / FP8-SC aliases
    "deepseek_v3_pretrain_256gpu_b200_nvfp4_config",
    "deepseek_v3_pretrain_256gpu_b300_nvfp4_config",
    "deepseek_v3_pretrain_256gpu_gb200_nvfp4_config",
    "deepseek_v3_pretrain_1024gpu_h100_fp8sc_config",
    # DeepSeek V3 perf recipes — V2 NVFP4 / FP8-SC / VR200 aliases
    "deepseek_v3_pretrain_v2_256gpu_b200_nvfp4_config",
    "deepseek_v3_pretrain_v2_256gpu_b300_nvfp4_config",
    "deepseek_v3_pretrain_v2_256gpu_gb200_nvfp4_config",
    "deepseek_v3_pretrain_v2_1024gpu_h100_fp8sc_config",
    "deepseek_v3_pretrain_v2_256gpu_vr200_bf16_config",
    "deepseek_v3_pretrain_v2_256gpu_vr200_fp8cs_config",
    "deepseek_v3_pretrain_v2_256gpu_vr200_fp8mx_config",
    "deepseek_v3_pretrain_v2_256gpu_vr200_nvfp4_config",
    # 64 GPU H100 aliases
    "deepseek_v3_pretrain_64gpu_h100_bf16_config",
    "deepseek_v3_pretrain_64gpu_h100_fp8cs_config",
]
