try:
    import megatron.bridge  # noqa: F401

    HAVE_MEGATRON_BRIDGE = True
except ModuleNotFoundError:
    HAVE_MEGATRON_BRIDGE = False

if HAVE_MEGATRON_BRIDGE:
    from .kimi_llm_pretrain import (
        kimi_k2_pretrain_config_b200,
        kimi_k2_pretrain_config_gb200,
        kimi_k2_pretrain_config_gb300,
        kimi_k2_pretrain_config_h100,
    )

from .kimi_workload_base_configs import (
    KIMI_K2_PRETRAIN_CONFIG_B200_BF16,
    KIMI_K2_PRETRAIN_CONFIG_B200_FP8_CS,
    KIMI_K2_PRETRAIN_CONFIG_B200_FP8_MX,
    KIMI_K2_PRETRAIN_CONFIG_GB200_BF16,
    KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_CS,
    KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX,
    KIMI_K2_PRETRAIN_CONFIG_GB300_BF16,
    KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS,
    KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX,
    KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4,
    KIMI_K2_PRETRAIN_CONFIG_H100_BF16,
    KIMI_K2_PRETRAIN_CONFIG_H100_FP8_CS,
    KIMI_K2_PRETRAIN_CONFIG_H100_FP8_SC,
)


__all__ = [
    "KIMI_K2_PRETRAIN_CONFIG_B200_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_B200_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_B200_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4",
    "KIMI_K2_PRETRAIN_CONFIG_H100_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_H100_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_H100_FP8_SC",
]

if HAVE_MEGATRON_BRIDGE:
    __all__.extend(
        [
            "kimi_k2_pretrain_config_gb300",
            "kimi_k2_pretrain_config_gb200",
            "kimi_k2_pretrain_config_b200",
            "kimi_k2_pretrain_config_h100",
        ]
    )
