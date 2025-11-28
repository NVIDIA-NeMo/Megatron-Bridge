try:
    import megatron.bridge  # noqa: F401

    HAVE_MEGATRON_BRIDGE = True
except ModuleNotFoundError:
    HAVE_MEGATRON_BRIDGE = False

if HAVE_MEGATRON_BRIDGE:
    from .nemotronh_llm_pretrain import (
        nemotronh_56b_pretrain_config_b200_config,
        nemotronh_56b_pretrain_config_gb200_config,
        nemotronh_56b_pretrain_config_gb300_config,
        nemotronh_56b_pretrain_config_h100_config,
    )

from .nemotronh_workload_base_configs import (
    NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS_BASE_CONFIG,
    NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS_BASE_CONFIG,
    NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS_BASE_CONFIG,
    NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS_BASE_CONFIG,
)


__all__ = [
    "NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS_BASE_CONFIG",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS_BASE_CONFIG",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS_BASE_CONFIG",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS_BASE_CONFIG",
]

if HAVE_MEGATRON_BRIDGE:
    __all__.extend(
        [
            "nemotronh_56b_pretrain_config_gb300_config",
            "nemotronh_56b_pretrain_config_gb200_config",
            "nemotronh_56b_pretrain_config_b200_config",
            "nemotronh_56b_pretrain_config_h100_config",
        ]
    )
