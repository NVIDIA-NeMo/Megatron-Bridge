try:
    import megatron.bridge  # noqa: F401

    HAVE_MEGATRON_BRIDGE = True
except ModuleNotFoundError:
    HAVE_MEGATRON_BRIDGE = False

if HAVE_MEGATRON_BRIDGE:
    from .nemotronh_llm_pretrain import (
        nemotronh_56b_pretrain_config_b200,
        nemotronh_56b_pretrain_config_b300,
        nemotronh_56b_pretrain_config_gb200,
        nemotronh_56b_pretrain_config_gb300,
        nemotronh_56b_pretrain_config_h100,
    )

from .nemotronh_workload_base_configs import (
    NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS,
    NEMOTRONH_56B_PRETRAIN_CONFIG_B300_FP8_CS,
    NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS,
    NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS,
    NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS,
)


__all__ = [
    "NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_B300_FP8_CS",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS",
]

if HAVE_MEGATRON_BRIDGE:
    __all__.extend(
        [
            "nemotronh_56b_pretrain_config_gb300",
            "nemotronh_56b_pretrain_config_gb200",
            "nemotronh_56b_pretrain_config_b300",
            "nemotronh_56b_pretrain_config_b200",
            "nemotronh_56b_pretrain_config_h100",
        ]
    )
