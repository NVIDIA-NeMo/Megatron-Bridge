# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
from pathlib import Path
from typing import Generic, List, Optional, TypeVar, Union

from peft import PeftConfig

from megatron.bridge.models.hf_pretrained.base import PreTrainedBase
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict


# Python 3.12+ supports PEP 692 (TypedDict Unpack)
if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


AdapterConfigType = TypeVar("AdapterConfigType", bound=PeftConfig)


class PreTrainedAdapters(PreTrainedBase, Generic[AdapterConfigType]):
    """
    A generic class for pretrained PEFT adapters with lazy loading.

    Allows type-safe access to specific adapter implementations like LoraConfig.

    Examples:
        Basic usage with lazy loading:
        >>> from megatron.bridge.peft.conversion.pretrained_adapters import PreTrainedAdapters
        >>> # Create instance - no adapter loading happens yet
        >>> adapters = PreTrainedAdapters.from_pretrained("username/llama-lora-adapters")
        >>> # Components are loaded on first access
        >>> config = adapters.config  # Loads adapter config
        >>> state = adapters.state    # Loads adapter weights

        Using specific adapter types with type hints:
        >>> from peft import LoraConfig
        >>> # Type-safe access to LoRA-specific features
        >>> lora_adapters: PreTrainedAdapters[LoraConfig] = PreTrainedAdapters.from_pretrained(
        ...     "username/llama-lora-adapters",
        ...     trust_remote_code=False
        ... )
        >>> # Access LoRA-specific attributes
        >>> config = lora_adapters.config  # Type is LoraConfig
        >>> rank = config.r  # LoRA rank
        >>> alpha = config.lora_alpha  # LoRA alpha

        Loading with validation:
        >>> adapters = PreTrainedAdapters.from_pretrained(
        ...     "path/to/local/adapters",
        ...     strict=True  # Validate all expected keys exist
        ... )
        >>> # Check adapter properties
        >>> print(f"PEFT type: {adapters.get_peft_type()}")
        >>> print(f"Target modules: {adapters.get_target_modules()}")
        >>> print(f"Rank: {adapters.get_rank()}")
    """

    ARTIFACTS = ["config"]
    OPTIONAL_ARTIFACTS = []

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        trust_remote_code: bool = False,
        strict: bool = True,
        **kwargs,
    ):
        """
        Initialize PreTrainedAdapters with lazy loading.

        Args:
            model_name_or_path: HuggingFace adapter identifier or local path
            trust_remote_code: Whether to trust remote code when loading
            strict: Whether to validate all expected keys exist
            **kwargs: Additional arguments passed to loading methods
        """
        self._model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.strict = strict
        super().__init__(**kwargs)

    def _load_model(self) -> None:
        """Adapters don't have a model, only config and state."""
        raise NotImplementedError("PreTrainedAdapters doesn't load a model")

    def _load_config(self) -> AdapterConfigType:
        """Load the adapter config."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load adapter config")

        path = self._resolve_path(self.model_name_or_path)
        config_file = path / "adapter_config.json"

        if not config_file.exists():
            raise FileNotFoundError(
                f"No adapter_config.json found in {path}. This does not appear to be a valid PEFT adapter directory."
            )

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        # Validate required fields
        if "peft_type" not in config_dict:
            raise ValueError("adapter_config.json must contain 'peft_type' field")

        # Convert to PeftConfig object
        return PeftConfig.from_dict(config_dict)

    @property
    def model_name_or_path(self) -> Optional[Union[str, Path]]:
        """Return the adapter model name or path."""
        return self._model_name_or_path

    @property
    def config(self) -> AdapterConfigType:
        """Lazy load and return the adapter config."""
        return super().config

    @config.setter
    def config(self, value: AdapterConfigType):
        """Set the adapter config manually."""
        self._config = value

    @property
    def state(self) -> StateDict:
        """
        Get the state dict accessor for adapter weights.

        This accessor provides pandas-like querying for adapter weights,
        backed by safetensors files for efficient loading.

        Examples:
            adapters.state()  # Get full adapter state dict
            adapters.state["layers.0.self_attn.q_proj.lora_A.weight"]  # Get single parameter
            adapters.state.regex(r".*lora_A.*")  # Get all LoRA A matrices
        """
        if self._state_dict_accessor is None:
            if self.model_name_or_path is None:
                raise ValueError("model_name_or_path must be provided to load adapter state")

            path = self._resolve_path(self.model_name_or_path)
            source = SafeTensorsStateSource(path)

            # Validate at least one expected key exists
            if self.strict:
                keys = source.get_all_keys()
                if not keys:
                    raise ValueError(f"No adapter weights found in {path}")

                # Basic validation for LoRA-style keys
                lora_keys = [k for k in keys if any(x in k for x in ["lora_A", "lora_B", "adapters"])]
                if not lora_keys:
                    raise ValueError(f"No recognizable PEFT adapter keys found in {path}")

            self._state_dict_accessor = StateDict(source)

        return self._state_dict_accessor

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, Path], trust_remote_code: bool = False, strict: bool = True, **kwargs
    ) -> "PreTrainedAdapters[AdapterConfigType]":
        """
        Create a PreTrainedAdapters instance for lazy loading.

        Args:
            model_name_or_path: HuggingFace adapter identifier or local path
            trust_remote_code: Whether to trust remote code when loading
            strict: Whether to validate all expected keys exist
            **kwargs: Additional arguments for loading methods

        Returns:
            PreTrainedAdapters instance configured for lazy loading
        """
        return cls(model_name_or_path=model_name_or_path, trust_remote_code=trust_remote_code, strict=strict, **kwargs)

    def get_target_modules(self) -> List[str]:
        """Get the list of target modules from the configuration."""
        return getattr(self.config, "target_modules", [])

    def get_peft_type(self) -> str:
        """Get the PEFT type from the configuration."""
        return getattr(self.config, "peft_type", "LORA")

    def get_rank(self) -> int:
        """Get the adapter rank/dimension from the configuration."""
        return getattr(self.config, "r", 8)

    def get_alpha(self) -> Union[int, float]:
        """Get the adapter alpha scaling parameter."""
        return getattr(self.config, "lora_alpha", 16)

    def get_dropout(self) -> float:
        """Get the adapter dropout rate."""
        return getattr(self.config, "lora_dropout", 0.0)

    def supports_layout(self, layout: str) -> bool:
        """Check if the adapter supports a particular layout.

        Args:
            layout: Layout type ("canonical" or "fused")

        Returns:
            True if the layout is supported
        """
        target_modules = self.get_target_modules()

        canonical_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        fused_modules = {"linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"}

        has_canonical = any(t in canonical_modules for t in target_modules)
        has_fused = any(t in fused_modules for t in target_modules)

        if layout == "canonical":
            return has_canonical and not has_fused
        elif layout == "fused":
            return has_fused and not has_canonical
        else:
            return False

    @staticmethod
    def _resolve_path(model_name_or_path: Union[str, Path]) -> Path:
        """Resolve adapter path, handling both local paths and Hub identifiers."""
        path = Path(model_name_or_path)

        if path.is_dir():
            return path
        else:
            # Try to download from HuggingFace Hub
            try:
                from huggingface_hub import snapshot_download

                cache_dir = snapshot_download(
                    repo_id=str(model_name_or_path), allow_patterns=["*.json", "*.safetensors", "*.bin"]
                )
                return Path(cache_dir)
            except Exception as e:
                raise ValueError(
                    f"Could not resolve path '{model_name_or_path}'. "
                    f"Not a local directory and failed to download from Hub: {e}"
                )

    def __repr__(self) -> str:
        """Return a string representation of the PreTrainedAdapters instance."""
        try:
            # Access config to trigger lazy loading for a richer repr
            _ = self.config
        except Exception:
            # If loading fails, repr shouldn't crash
            pass

        lines = [f"{self.__class__.__name__}("]

        # Add config info
        if hasattr(self, "_config") and self._config is not None:
            config = self._config
            peft_type = getattr(config, "peft_type", "N/A")
            rank = getattr(config, "r", "N/A")
            alpha = getattr(config, "lora_alpha", "N/A")
            lines.append(f"  (config): {config.__class__.__name__} [peft_type={peft_type}, r={rank}, alpha={alpha}]")
        else:
            lines.append("  (config): PeftConfig [not loaded]")

        # Add state info
        if hasattr(self, "_state_dict_accessor") and self._state_dict_accessor is not None:
            try:
                num_keys = len(self._state_dict_accessor.keys())
                lines.append(f"  (state): StateDict [loaded, {num_keys} parameters]")
            except Exception:
                lines.append("  (state): StateDict [loaded]")
        else:
            lines.append("  (state): StateDict [not loaded]")

        lines.append(f"  (path): {self.model_name_or_path}")
        lines.append(")")

        return "\n".join(lines)


# TypedDict definitions for method parameters
class LoadAdaptersKwargs(TypedDict, total=False):
    """TypedDict for adapter loading parameters."""

    trust_remote_code: bool
    strict: bool
    force_download: bool
    cache_dir: Optional[str]
