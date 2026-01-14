try:
    import megatron.bridge  # noqa: F401

    HAVE_MEGATRON_BRIDGE = True
except ModuleNotFoundError:
    HAVE_MEGATRON_BRIDGE = False

if HAVE_MEGATRON_BRIDGE:
    from .gpt_oss_llm_pretrain import (
        gpt_oss_120b_pretrain_config_b200,
        gpt_oss_120b_pretrain_config_b300,
        gpt_oss_120b_pretrain_config_gb200,
        gpt_oss_120b_pretrain_config_gb300,
        gpt_oss_120b_pretrain_config_h100,
    )

from .gpt_oss_workload_base_configs import (
    # V2 (GBS=1280)
    GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16_V1,
    # V1 (GBS=512)
    GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16_V2,
    GPT_OSS_120B_PRETRAIN_CONFIG_B300_BF16_V1,
    GPT_OSS_120B_PRETRAIN_CONFIG_B300_BF16_V2,
    GPT_OSS_120B_PRETRAIN_CONFIG_GB200_BF16_V1,
    GPT_OSS_120B_PRETRAIN_CONFIG_GB200_BF16_V2,
    GPT_OSS_120B_PRETRAIN_CONFIG_GB300_BF16_V1,
    GPT_OSS_120B_PRETRAIN_CONFIG_GB300_BF16_V2,
    GPT_OSS_120B_PRETRAIN_CONFIG_H100_BF16_V1,
    GPT_OSS_120B_PRETRAIN_CONFIG_H100_BF16_V2,
)


__all__ = [
    # V1 (GBS=512)
    "GPT_OSS_120B_PRETRAIN_CONFIG_B300_BF16_V2",
    "GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16_V2",
    "GPT_OSS_120B_PRETRAIN_CONFIG_GB200_BF16_V2",
    "GPT_OSS_120B_PRETRAIN_CONFIG_GB300_BF16_V2",
    "GPT_OSS_120B_PRETRAIN_CONFIG_H100_BF16_V2",
    # V2 (GBS=1280)
    "GPT_OSS_120B_PRETRAIN_CONFIG_B300_BF16_V1",
    "GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16_V1",
    "GPT_OSS_120B_PRETRAIN_CONFIG_GB200_BF16_V1",
    "GPT_OSS_120B_PRETRAIN_CONFIG_GB300_BF16_V1",
    "GPT_OSS_120B_PRETRAIN_CONFIG_H100_BF16_V1",
]

if HAVE_MEGATRON_BRIDGE:
    __all__.extend(
        [
            "gpt_oss_120b_pretrain_config_gb300",
            "gpt_oss_120b_pretrain_config_gb200",
            "gpt_oss_120b_pretrain_config_b300",
            "gpt_oss_120b_pretrain_config_b200",
            "gpt_oss_120b_pretrain_config_h100",
        ]
    )
