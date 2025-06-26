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

from pathlib import Path
from typing import Any, List, Protocol, Type, Union

from transformers import AutoConfig

from megatron.hub.bridge.causal_bridge import CausalLMBridge


_BRIDGES: List[Type["BridgeProtocol"]] = [
    CausalLMBridge,
]


class AutoBridge:
    """
    Automatically select and instantiate the appropriate bridge for a model.

    This class examines the model configuration and selects the first bridge
    that supports it. No dynamic discovery or decorators are used - all bridges
    must be explicitly imported and added to the _BRIDGES list.

    The AutoBridge provides a user-friendly interface similar to HuggingFace's
    Auto classes, allowing seamless model loading without needing to know the
    specific bridge implementation required for each model architecture.

    Example:
        >>> from megatron.hub import AutoBridge
        >>> # Load a Llama model without knowing it needs CausalLMBridge
        >>> bridge = AutoBridge.from_pretrained("meta-llama/Llama-3-8B")
        >>> # Automatically returns a CausalLMBridge instance

        >>> # Works with local paths too
        >>> bridge = AutoBridge.from_pretrained("/path/to/model")

        >>> # Check if a model is supported before loading
        >>> if AutoBridge.can_handle("microsoft/phi-2"):
        ...     bridge = AutoBridge.from_pretrained("microsoft/phi-2")

        >>> # Pass additional arguments to the underlying bridge
        >>> bridge = AutoBridge.from_pretrained(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     torch_dtype=torch.float16,
        ...     device_map="auto",
        ...     trust_remote_code=True
        ... )

    Note:
        Currently supports CausalLM models. Additional model types can be
        supported by implementing the appropriate bridge and adding it to
        the _BRIDGES list.
    """

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], **kwargs) -> "BridgeProtocol":
        """
        Load a pretrained model, automatically selecting the appropriate bridge.

        This method:
        1. Loads only the model configuration (no weights)
        2. Iterates through registered bridges to find one that supports it
        3. Uses that bridge to load the full model

        Args:
            path: Path to model directory or HuggingFace model ID
            **kwargs: Additional arguments passed to the bridge's from_pretrained
                method (e.g., trust_remote_code, device_map, etc.)

        Returns:
            An instance of the appropriate bridge with the model loaded

        Raises:
            ValueError: If no registered bridge supports the model
        """
        # Load only the configuration - this is fast and doesn't load weights
        try:
            config = AutoConfig.from_pretrained(path, trust_remote_code=kwargs.get("trust_remote_code", False))
        except Exception as e:
            raise ValueError(
                f"Failed to load configuration from {path}. "
                f"Ensure the path is valid and contains a config.json file. "
                f"Error: {e}"
            )

        bridge_cls = cls._find_bridge_for_config(config, "from_pretrained", path)

        try:
            return bridge_cls.from_pretrained(path, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load model with {bridge_cls.__name__}: {e}") from e

    @classmethod
    def from_config(cls, config) -> "BridgeProtocol":
        """
        Create a bridge from a HuggingFace configuration.

        This method creates a bridge instance from just a model configuration,
        without loading any weights. This is useful for:
        - Creating Megatron models with random initialization
        - Working with model architectures without downloading weights
        - Testing and development scenarios

        Args:
            config: HuggingFace PretrainedConfig instance containing model
                architecture information

        Returns:
            BridgeProtocol: Bridge instance configured for the architecture

        Raises:
            ValueError: If no registered bridge supports the configuration

        Example:
            >>> from transformers import AutoConfig
            >>>
            >>> # Load just the configuration
            >>> config = AutoConfig.from_pretrained("meta-llama/Llama-3-8B")
            >>>
            >>> # Create bridge from config (no weights)
            >>> bridge = AutoBridge.from_config(config)
            >>>
            >>> # Create Megatron model with random initialization
            >>> provider = bridge.to_provider(load_weights=False)
            >>> model = provider(wrap_with_ddp=False)
        """
        bridge_cls = cls._find_bridge_for_config(config, "from_config")

        try:
            return bridge_cls.from_config(config)
        except Exception as e:
            raise ValueError(f"Failed to create {bridge_cls.__name__} from config: {e}") from e

    @classmethod
    def get_supported_bridges(cls) -> List[str]:
        """
        Get list of all registered bridge class names.

        This method is useful for debugging and for understanding which model
        types are currently supported by the AutoBridge system.

        Returns:
            List[str]: Bridge class names in priority order. The order matters
                as bridges are tried sequentially during model loading.

        Example:
            >>> bridges = AutoBridge.get_supported_bridges()
            >>> print(f"Supported bridges: {bridges}")
            ['CausalLMBridge']
        """
        return [bridge.__name__ for bridge in _BRIDGES]

    @classmethod
    def can_handle(cls, path: Union[str, Path], trust_remote_code: bool = False) -> bool:
        """
        Check if any registered bridge can handle the model at the given path.

        This method allows you to verify model compatibility before attempting
        to load it, which can be useful for validation or UI feedback.

        Args:
            path: Path to model directory or HuggingFace model ID
                Examples: "meta-llama/Llama-3-8B", "/models/my_model"
            trust_remote_code: Whether to trust remote code when loading config.
                Set to True for models that use custom modeling code.

        Returns:
            bool: True if at least one bridge supports the model, False otherwise

        Example:
            >>> # Check if a model is supported
            >>> if AutoBridge.can_handle("meta-llama/Llama-3-8B"):
            ...     print("Model is supported!")
            ... else:
            ...     print("Model requires a custom bridge implementation")
        """
        try:
            config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
            return any(hasattr(bridge_cls, "supports") and bridge_cls.supports(config) for bridge_cls in _BRIDGES)
        except Exception:
            return False

    @classmethod
    def _find_bridge_for_config(cls, config, method_name: str, path: Union[str, Path, None] = None):
        """
        Find a bridge that supports the given configuration and has the specified method.

        Args:
            config: HuggingFace PretrainedConfig instance
            method_name: Name of the method to check for ('from_config' or 'from_pretrained')
            path: Optional path for better error messages

        Returns:
            Bridge class that supports the config

        Raises:
            ValueError: If no suitable bridge is found
        """
        # Try each bridge in order
        for bridge_cls in _BRIDGES:
            if hasattr(bridge_cls, "supports") and bridge_cls.supports(config):
                # Found a supporting bridge - check if it has the required method
                if hasattr(bridge_cls, method_name):
                    return bridge_cls
                else:
                    # Bridge doesn't have the required method
                    raise ValueError(
                        f"{bridge_cls.__name__} supports the configuration but does not "
                        f"implement {method_name} method. This is likely a bug in the bridge "
                        f"implementation."
                    )

        # No bridge found - create detailed error message
        architectures = getattr(config, "architectures", ["unknown"])
        model_type = getattr(config, "model_type", "unknown")

        error_msg = (
            f"No bridge found for "
            f"{'configuration' if path is None else f'model at {path}'}. "
            f"Model type: {model_type}, architectures: {architectures}. "
            f"Available bridges: {[b.__name__ for b in _BRIDGES]}. "
            f"Please use a specific bridge directly or implement a new bridge "
            f"for this model type."
        )

        raise ValueError(error_msg)


class BridgeProtocol(Protocol):
    """
    Protocol defining the interface for model bridges.

    All bridges that want to participate in automatic selection must implement
    these methods. This is a typing-only protocol with no runtime checks.
    """

    @classmethod
    def supports(cls, config: Any) -> bool:
        """
        Check if this bridge supports the given model configuration.

        Args:
            config: HuggingFace model config object

        Returns:
            True if this bridge can handle the model, False otherwise
        """
        ...

    @classmethod
    def from_config(cls, config: Any) -> "BridgeProtocol":
        """
        Create a bridge from a HuggingFace configuration.

        Args:
            config: HuggingFace PretrainedConfig instance containing model
                architecture information

        Returns:
            Instance of the bridge configured for the architecture
        """
        ...

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], **kwargs) -> "BridgeProtocol":
        """
        Load a pretrained model using this bridge.

        Args:
            path: Path to the model (local directory or HuggingFace model ID)
            **kwargs: Additional arguments passed to the underlying loader

        Returns:
            Instance of the bridge with loaded model
        """
        ...
