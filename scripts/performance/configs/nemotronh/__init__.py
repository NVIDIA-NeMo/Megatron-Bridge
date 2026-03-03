try:
    import megatron.bridge  # noqa: F401

    HAVE_MEGATRON_BRIDGE = True
except ModuleNotFoundError:
    HAVE_MEGATRON_BRIDGE = False

if HAVE_MEGATRON_BRIDGE:
    from .nemotron_3_nano_llm_pretrain import (
        nemotron_3_nano_pretrain_config_b200,
        nemotron_3_nano_pretrain_config_b300,
        nemotron_3_nano_pretrain_config_gb200,
        nemotron_3_nano_pretrain_config_gb300,
        nemotron_3_nano_pretrain_config_h100,
    )
    from .nemotronh_llm_pretrain import (
        nemotronh_56b_pretrain_config_b200,
        nemotronh_56b_pretrain_config_b300,
        nemotronh_56b_pretrain_config_gb200,
        nemotronh_56b_pretrain_config_gb300,
        nemotronh_56b_pretrain_config_h100,
    )

from .nemotron_3_nano_workload_base_configs import (
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_BF16_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_FP8_MX_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_NVFP4_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_BF16_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_FP8_MX_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_NVFP4_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_BF16_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_FP8_MX_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_NVFP4_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_BF16_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_FP8_MX_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_NVFP4_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_BF16_V1,
    NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_FP8_CS_V1,
)
from .nemotronh_workload_base_configs import (
    NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS_V1,
    NEMOTRONH_56B_PRETRAIN_CONFIG_B300_FP8_CS_V1,
    NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS_V1,
    NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS_V1,
    NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS_V1,
)


__all__ = [
    "NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS_V1",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS_V1",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_B300_FP8_CS_V1",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS_V1",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB300_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_GB200_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B300_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_FP8_MX_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_B200_NVFP4_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_BF16_V1",
    "NEMOTRON_3_NANO_PRETRAIN_CONFIG_H100_FP8_CS_V1",
]

if HAVE_MEGATRON_BRIDGE:
    __all__.extend(
        [
            "nemotronh_56b_pretrain_config_gb300",
            "nemotronh_56b_pretrain_config_gb200",
            "nemotronh_56b_pretrain_config_b300",
            "nemotronh_56b_pretrain_config_b200",
            "nemotronh_56b_pretrain_config_h100",
            "nemotron_3_nano_pretrain_config_gb300",
            "nemotron_3_nano_pretrain_config_gb200",
            "nemotron_3_nano_pretrain_config_b300",
            "nemotron_3_nano_pretrain_config_b200",
            "nemotron_3_nano_pretrain_config_h100",
        ]
    )
