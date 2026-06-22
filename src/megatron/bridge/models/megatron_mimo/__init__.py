# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.bridge.models.megatron_mimo.build_model import build_megatron_mimo_model
from megatron.bridge.models.megatron_mimo.llava_provider import LlavaMegatronMIMOProvider
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import (
    MegatronMIMOInfra,
    MegatronMIMOProvider,
)


def __getattr__(name: str):
    if name == "MegatronMIMOBridge":
        from megatron.bridge.models.megatron_mimo.conversion import MegatronMIMOBridge

        return MegatronMIMOBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LlavaMegatronMIMOProvider",
    "MegatronMIMOBridge",
    "MegatronMIMOInfra",
    "MegatronMIMOProvider",
    "MegatronMIMOParallelismConfig",
    "ModuleParallelismConfig",
    "build_megatron_mimo_model",
]
