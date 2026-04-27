"""Nemotron Omni model family (Vision-Language + Audio) for Megatron Bridge."""

from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import NemotronOmniBridge
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NemotronOmniModelProvider,
    NemotronVLModelProvider,
)
from megatron.bridge.models.nemotron_omni.nemotron_omni_sound import BridgeSoundEncoder


__all__ = [
    "NemotronOmniModel",
    "NemotronOmniBridge",
    "NemotronOmniModelProvider",
    "NemotronVLModelProvider",
    "BridgeSoundEncoder",
]
