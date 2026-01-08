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
    GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16,
    GPT_OSS_120B_PRETRAIN_CONFIG_B300_BF16,
    GPT_OSS_120B_PRETRAIN_CONFIG_GB200_BF16,
    GPT_OSS_120B_PRETRAIN_CONFIG_GB300_BF16,
    GPT_OSS_120B_PRETRAIN_CONFIG_H100_BF16,
)


__all__ = [
    "GPT_OSS_120B_PRETRAIN_CONFIG_B300_BF16",
    "GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16",
    "GPT_OSS_120B_PRETRAIN_CONFIG_GB200_BF16",
    "GPT_OSS_120B_PRETRAIN_CONFIG_GB300_BF16",
    "GPT_OSS_120B_PRETRAIN_CONFIG_H100_BF16",
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
