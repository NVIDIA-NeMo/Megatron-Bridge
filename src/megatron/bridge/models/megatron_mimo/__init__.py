# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.bridge.models.megatron_mimo.build_model import build_megatron_mimo_model
from megatron.bridge.models.megatron_mimo.infra import MegatronMIMOInfra
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.model_config import (
    MegatronMIMOModelBuilder,
    MegatronMIMOModelConfig,
)


def __getattr__(name: str):
    if name == "MegatronMIMOBridge":
        from megatron.bridge.models.megatron_mimo.conversion import MegatronMIMOBridge

        return MegatronMIMOBridge
    if name == "LlavaMegatronMIMOProvider":
        from megatron.bridge.models.megatron_mimo.llava_provider import LlavaMegatronMIMOProvider

        return LlavaMegatronMIMOProvider
    if name == "MegatronMIMOProvider":
        from megatron.bridge.models.megatron_mimo import megatron_mimo_provider

        return getattr(megatron_mimo_provider, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LlavaMegatronMIMOProvider",
    "MegatronMIMOBridge",
    "MegatronMIMOInfra",
    "MegatronMIMOProvider",
    "MegatronMIMOParallelismConfig",
    "MegatronMIMOModelBuilder",
    "MegatronMIMOModelConfig",
    "ModuleParallelismConfig",
    "build_megatron_mimo_model",
]
