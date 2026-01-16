from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from megatron.core.transformer.spec_utils import ModuleSpec


@dataclass
class EncoderTransformerConfig:
    """Lightweight base config for encoder providers."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    seq_length: int


class EncoderProvider(ABC):
    """Interface for encoder providers used in MIMO setups."""

    @abstractmethod
    def provide_model(self, pg_collection) -> object:
        """Create the encoder module (unwrapped)."""

    @abstractmethod
    def get_transformer_layer_spec(self) -> ModuleSpec:
        """Return the ModuleSpec for the encoder stack."""

    @abstractmethod
    def get_projection_spec(self) -> Optional[ModuleSpec]:
        """Optional projection ModuleSpec for encoder outputs."""


class GenericVisionEncoderProvider(EncoderProvider):
    """Minimal stub encoder provider for Phase 1 wiring."""

    def __init__(self, config: EncoderTransformerConfig) -> None:
        self.config = config

    def provide_model(self, pg_collection) -> object:
        # Stub: actual encoder creation will be implemented in Phase 2.
        raise NotImplementedError("GenericVisionEncoderProvider.provide_model not implemented.")

    def get_transformer_layer_spec(self) -> ModuleSpec:
        raise NotImplementedError("GenericVisionEncoderProvider.get_transformer_layer_spec not implemented.")

    def get_projection_spec(self) -> Optional[ModuleSpec]:
        return None
