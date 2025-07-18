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

import dataclasses
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Iterable, Literal, Optional, Type, TypeVar, Union, overload

import torch.distributed
import transformers
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from typing_extensions import Unpack

from megatron.bridge.models import model_bridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.model_bridge import MegatronModelBridge, WeightDistributionMode
from megatron.bridge.models.model_provider_mixin import GetModelKwargs, ModelProviderMixin
from megatron.bridge.models.state import PyTorchStateSource, SafeTensorsStateSource


if TYPE_CHECKING:
    from megatron.bridge.models.model_bridge import HFWeightTuple


MegatronModelT = TypeVar("ModelT", bound=MegatronModule)
DataclassT = TypeVar("DataclassT")


class CausalLMBridge(Generic[MegatronModelT]):
    """
    Bridge for converting Causal Language Models between HuggingFace and Megatron formats.

    This bridge handles the conversion of causal language models (e.g., GPT, Llama, Phi)
    between HuggingFace's transformers library format and Megatron-Core's distributed
    training format. It manages weight mapping, tensor parallelism distribution, and
    configuration translation.

    The bridge supports both directions of conversion:
    - HuggingFace → Megatron: For training or inference with Megatron
    - Megatron → HuggingFace: For saving trained models in HF format

    Args:
        hf_pretrained: Either a PreTrainedCausalLM instance with loaded model,
            or a PretrainedConfig for configuration-only operations

    Example:
        >>> # Load and convert a model to Megatron format
        >>> bridge = CausalLMBridge.from_hf_pretrained("meta-llama/Llama-3-8B")
        >>> provider = bridge.to_megatron_provider()
        >>> megatron_model = provider(wrap_with_ddp=False)

        >>> # Export a Megatron model back to HuggingFace format
        >>> bridge.save_hf_pretrained(megatron_model, "./exported_model")

        >>> # Convert weights with custom settings
        >>> for name, weight in bridge.export_hf_weights(
        ...     megatron_model,
        ...     order="safetensors",
        ...     mode="consolidate"
        ... ):
        ...     print(f"Exported {name}: {weight.shape}")

    Note:
        The bridge automatically detects the model architecture and applies
        the appropriate weight mappings. Custom architectures require implementing
        a MegatronModelBridge subclass.
    """

    def __init__(
        self,
        hf_pretrained: PreTrainedCausalLM | PretrainedConfig,
        bridge_class: Optional[Type[MegatronModelBridge]] = None,
    ):
        if not isinstance(hf_pretrained, (PreTrainedCausalLM, PretrainedConfig)):
            raise ValueError("hf_pretrained must be a PreTrainedCausalLM or PretrainedConfig instance")

        # Validate bridge_class if provided
        if bridge_class is not None:
            if not (isinstance(bridge_class, type) and issubclass(bridge_class, MegatronModelBridge)):
                raise ValueError(
                    f"bridge_class must be a subclass of MegatronModelBridge, got {type(bridge_class).__name__}"
                )

        self.hf_pretrained: PreTrainedCausalLM | PretrainedConfig = hf_pretrained
        self._manual_bridge_class: Optional[Type[MegatronModelBridge]] = bridge_class

    @classmethod
    def list_supported_models(cls) -> list[str]:
        """
        List all model architectures currently supported by the bridge system.

        Returns:
            List of supported HuggingFace model architecture names
        """
        # Get all registered implementations from the dispatch system
        supported = []

        # Access the dispatch registry to find all registered types

        if hasattr(model_bridge.get_model_bridge, "_exact_types"):
            for arch_type in model_bridge.get_model_bridge._exact_types.keys():
                if hasattr(arch_type, "__name__"):
                    supported.append(arch_type.__name__)

        return sorted(supported)

    @classmethod
    def supports(cls, config: Any) -> bool:
        """
        Check if this bridge supports the given model configuration.

        A model is supported if it has at least one architecture ending with 'ForCausalLM'.

        Args:
            config: HuggingFace model config object

        Returns:
            True if this bridge can handle the model, False otherwise
        """
        architectures = getattr(config, "architectures", [])
        if not architectures:
            return False

        return any(arch.endswith("ForCausalLM") for arch in architectures)

    @classmethod
    def from_hf_config(
        cls, config: PretrainedConfig, bridge_class: Optional[Type[MegatronModelBridge]] = None
    ) -> "CausalLMBridge":
        """
        Create a CausalLMBridge from a HuggingFace configuration.

        This method creates a bridge instance from just a model configuration,
        without loading any weights. This is useful for:
        - Creating Megatron models with random initialization
        - Working with model architectures without downloading weights
        - Testing and development scenarios

        Args:
            config: HuggingFace PretrainedConfig instance containing model
                architecture information
            bridge_class: Optional MegatronModelBridge subclass to use for conversion.
                If not provided, will attempt to auto-detect based on architecture.

        Returns:
            CausalLMBridge: Bridge instance configured for the architecture

        Raises:
            ValueError: If the configuration is not for a supported CausalLM model
                and no bridge_class is provided

        Example:
            >>> from transformers import AutoConfig
            >>>
            >>> # Load just the configuration
            >>> config = AutoConfig.from_pretrained("meta-llama/Llama-3-8B")
            >>>
            >>> # Create bridge from config (no weights)
            >>> bridge = CausalLMBridge.from_hf_config(config)
            >>>
            >>> # Create Megatron model with random initialization
            >>> provider = bridge.to_megatron_provider(load_weights=False)
            >>> model = provider(wrap_with_ddp=False)

            >>> # Or use for architecture exploration
            >>> transformer_config = bridge.transformer_config
            >>> print(f"Hidden size: {transformer_config.hidden_size}")
            >>> print(f"Num layers: {transformer_config.num_layers}")

            >>> # For models requiring manual bridge specification
            >>> from megatron.bridge.models.baichuan import BaichuanCausalBridge
            >>> bridge = CausalLMBridge.from_hf_config(config, bridge_class=BaichuanCausalBridge)

        See Also:
            from_hf_pretrained: Create bridge with loaded weights
            transformer_config: Access the Megatron TransformerConfig
        """
        if bridge_class is None:
            cls._validate_config(config)
        return cls(config, bridge_class=bridge_class)

    @classmethod
    def from_hf_pretrained(
        cls, path: str | Path, bridge_class: Optional[Type[MegatronModelBridge]] = None, **kwargs
    ) -> "CausalLMBridge":
        """
        Load a CausalLMBridge from a pretrained model.

        This method loads a model from HuggingFace Hub or a local directory and
        creates a bridge instance ready for conversion operations. The model
        architecture is validated to ensure compatibility.

        Args:
            path: HuggingFace model ID or path to model directory
                Examples: "meta-llama/Llama-3-8B", "./my_model"
            bridge_class: Optional MegatronModelBridge subclass to use for conversion.
                If not provided, will attempt to auto-detect based on architecture.
                Use this for models with custom code (trust_remote_code=True).
            **kwargs: Additional arguments passed to HuggingFace from_pretrained
                Common options include:
                - torch_dtype: Model precision (torch.float16, torch.bfloat16)
                - device_map: Device placement strategy ("auto", "cuda:0", etc.)
                - trust_remote_code: Allow custom model code execution
                - attn_implementation: Attention implementation ("flash_attention_2", etc.)

        Returns:
            CausalLMBridge: Bridge instance with loaded model

        Raises:
            ValueError: If the model architecture is not supported and no
                bridge_class is provided

        Example:
            >>> # Basic loading
            >>> bridge = CausalLMBridge.from_hf_pretrained("gpt2")

            >>> # Load with specific settings
            >>> bridge = CausalLMBridge.from_hf_pretrained(
            ...     "meta-llama/Llama-3-8B",
            ...     torch_dtype=torch.float16,
            ...     device_map="auto"
            ... )

            >>> # For models requiring trust_remote_code
            >>> from megatron.bridge.models.baichuan import BaichuanCausalBridge
            >>> bridge = CausalLMBridge.from_hf_pretrained(
            ...     "baichuan-inc/Baichuan2-7B-Base",
            ...     bridge_class=BaichuanCausalBridge,
            ...     trust_remote_code=True
            ... )
        """
        # First load just the config to check architecture support
        trust_remote = kwargs.get("trust_remote_code", False)
        config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote)

        if bridge_class is None:
            cls._validate_config(config, path)

        return cls(PreTrainedCausalLM.from_pretrained(path, **kwargs), bridge_class=bridge_class)

    @overload
    def __call__(
        self,
        model: list[MegatronModelT],
        order: Literal["megatron", "hf", "safetensors"] = "megatron",
        cpu: bool = False,
        show_progress: bool = True,
        mode: Union[str, WeightDistributionMode] = WeightDistributionMode.CONSOLIDATE,
    ) -> Iterable["HFWeightTuple"]: ...

    def __call__(
        self,
        model,
        order: Literal["megatron", "hf", "safetensors"] = "megatron",
        cpu: bool = False,
        show_progress: bool = True,
        mode: Union[str, WeightDistributionMode] = WeightDistributionMode.CONSOLIDATE,
    ) -> Iterable["HFWeightTuple"]:
        return self.export_hf_weights(model=model, order=order, cpu=cpu, show_progress=show_progress, mode=mode)

    def load_hf_weights(self, model: list[MegatronModelT], hf_path: str | Path | None = None) -> None:
        """
        Load HuggingFace weights into a Megatron model.

        This method handles the conversion and distribution of weights from
        HuggingFace format to Megatron's distributed format, including proper
        tensor parallel and pipeline parallel distribution.

        Args:
            model: List of Megatron model instances (one per virtual pipeline stage)
            hf_path: Optional path to load weights from. If None, uses weights
                from the bridge's hf_pretrained instance

        Returns:
            The input model with loaded weights

        Raises:
            ValueError: If hf_path is None and bridge was created without weights

        Example:
            >>> # Load weights from bridge's pretrained model
            >>> bridge = CausalLMBridge.from_hf_pretrained("gpt2")
            >>> megatron_model = create_megatron_model()  # Your model creation
            >>> bridge.load_hf_weights(megatron_model)

            >>> # Load weights from a different checkpoint
            >>> bridge.load_hf_weights(megatron_model, "./finetuned_model")
        """
        if hf_path is None:
            if not isinstance(self.hf_pretrained, PreTrainedCausalLM):
                raise ValueError("hf_path is required when hf_pretrained is not a PreTrainedCausalLM instance")
            pre_trained = self.hf_pretrained
        else:
            pre_trained = PreTrainedCausalLM.from_pretrained(hf_path)
        self._model_bridge.load_weights_hf_to_megatron(model, pre_trained)

        return model

    @overload
    def export_hf_weights(
        self,
        model: list[MegatronModelT],
        order: Literal["megatron", "hf", "safetensors"] = "megatron",
        cpu: bool = False,
        show_progress: bool = True,
        mode: Union[str, WeightDistributionMode] = WeightDistributionMode.CONSOLIDATE,
    ) -> Iterable["HFWeightTuple"]: ...

    def export_hf_weights(
        self,
        model,
        order: Literal["megatron", "hf", "safetensors"] = "megatron",
        cpu: bool = False,
        show_progress: bool = True,
        mode: Union[str, WeightDistributionMode] = WeightDistributionMode.CONSOLIDATE,
    ) -> Iterable["HFWeightTuple"]:
        """
        Export Megatron model weights to HuggingFace format.

        This method yields weight tensors in HuggingFace format, handling the
        gathering of distributed tensors and format conversion. It's useful for
        streaming weight export or custom processing.

        Args:
            model: Megatron model instance or list of instances
            order: Export order for weights
                - "megatron": Follow Megatron's parameter order
                - "hf": Follow HuggingFace state dict order
                - "safetensors": Group by safetensors file, then by key
            cpu: Whether to move tensors to CPU before yielding
            show_progress: Display progress bar during export
            mode: Weight distribution mode
                - "consolidate": Gather to rank 0
                - "replicate": All ranks get full tensors
                - "distribute": Each rank keeps its shard

        Yields:
            HFWeightTuple: Named tuples of (param_name, weight_tensor)

        Example:
            >>> # Export and process weights
            >>> for name, weight in bridge.export_hf_weights(model):
            ...     print(f"{name}: {weight.shape}")

            >>> # Export with specific settings
            >>> weights = list(bridge.export_hf_weights(
            ...     model,
            ...     order="safetensors",
            ...     cpu=True,
            ...     mode="replicate"  # All ranks get full weights
            ... ))
        """
        # When using a manual bridge, call the bridge directly
        if self._manual_bridge_class is not None:
            bridge = self._model_bridge
            return bridge.stream_weights_megatron_to_hf(
                model, self.hf_pretrained, order=order, cpu=cpu, show_progress=show_progress, mode=mode
            )

        # Otherwise use the dispatch system
        dispatch_instance = (self._get_causal_lm_architecture(), self._get_model_instance(model))
        return model_bridge.stream_weights_megatron_to_hf(
            dispatch_instance, model, self.hf_pretrained, order=order, cpu=cpu, show_progress=show_progress, mode=mode
        )

    @overload
    def save_hf_pretrained(self, model: list[MegatronModelT], path: str | Path) -> None: ...

    def save_hf_pretrained(self, model, path: str | Path, show_progress: bool = True) -> None:
        """
        Save a Megatron model in HuggingFace format.

        This method exports the complete model including configuration, tokenizer,
        and weights to a directory that can be loaded with HuggingFace's
        from_pretrained methods.

        Args:
            model: Megatron model instance or list of instances
            path: Directory path to save the model
            show_progress: Display progress bar during weight export

        Example:
            >>> # Save model after training
            >>> bridge.save_hf_pretrained(megatron_model, "./my_finetuned_model")

            >>> # Load the saved model with HuggingFace
            >>> from transformers import AutoModelForCausalLM
            >>> hf_model = AutoModelForCausalLM.from_pretrained("./my_finetuned_model")

        Note:
            This method is collective - all ranks must call it. Only rank 0
            saves the configuration files, while weight saving is coordinated
            across all ranks.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Distributed training, only rank 0 saves artifacts
            if torch.distributed.get_rank() == 0:
                self.hf_pretrained.save_artifacts(path)
        else:
            # No distributed training, save artifacts
            self.hf_pretrained.save_artifacts(path)

        self.save_hf_weights(model, path, show_progress)

    @overload
    def save_hf_weights(self, model: list[MegatronModelT], path: str | Path, show_progress: bool = True) -> None: ...

    def save_hf_weights(self, model, path: str | Path, show_progress: bool = True) -> None:
        """
        Save Megatron model weights in HuggingFace safetensors format.

        This method exports only the model weights (not configuration or tokenizer)
        to safetensors files compatible with HuggingFace. It uses streaming save
        to handle large models efficiently without requiring all weights in memory
        at once.

        The weights are gathered from distributed ranks and saved in the standard
        HuggingFace sharded format when the model is large.

        Args:
            model: Megatron model instance or list of instances
            path: Directory path where weight files will be saved
            show_progress: Display progress bar during export

        Raises:
            ValueError: If the state source doesn't support streaming save

        Example:
            >>> # Save just the weights
            >>> bridge.save_hf_weights(megatron_model, "./model_weights")

            >>> # Save without progress bar (useful in scripts)
            >>> bridge.save_hf_weights(megatron_model, "./weights", show_progress=False)

        Note:
            - This method is collective and must be called by all ranks
            - Uses safetensors format for efficient loading and security
            - Automatically handles model sharding for large models
            - The saved weights can be loaded with HuggingFace's from_pretrained
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # When using a manual bridge, call the bridge directly
        if self._manual_bridge_class is not None:
            bridge = self._model_bridge
            generator = bridge.stream_weights_megatron_to_hf(
                model, self.hf_pretrained, order="safetensors", cpu=True, show_progress=show_progress
            )
        else:
            # Otherwise use the dispatch system
            dispatch_instance = (self._get_causal_lm_architecture(), self._get_model_instance(model))
            generator = model_bridge.stream_weights_megatron_to_hf(
                dispatch_instance,
                model,
                self.hf_pretrained,
                order="safetensors",
                cpu=True,
                show_progress=show_progress,
            )

        # Check if the state source supports streaming save
        if hasattr(self.hf_pretrained, "state") and hasattr(self.hf_pretrained.state, "source"):
            source = self.hf_pretrained.state.source
            if isinstance(source, (SafeTensorsStateSource, PyTorchStateSource)):
                source.save_generator(generator, path)
            else:
                raise ValueError(
                    f"The state source {type(source).__name__} does not support streaming save. "
                    "Only SafeTensorsStateSource and PyTorchStateSource are supported."
                )
        else:
            raise ValueError("No state source available for streaming save.")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def push_to_hub(self, path: str | Path) -> None: ...

    def to_megatron_model(
        self,
        load_weights: bool = True,
        hf_path: str | Path | None = None,
        **kwargs: Unpack[GetModelKwargs],
    ) -> list[MegatronModelT]:
        provider = self.to_megatron_provider(load_weights, hf_path)
        return provider(**kwargs)

    def to_megatron_provider(self, load_weights: bool = True, hf_path: str | Path | None = None) -> GPTModelProvider:
        """
        Convert to a Megatron model provider.

        This method creates a GPTModelProvider configured to match the HuggingFace
        model's architecture. The provider can then be used to instantiate
        Megatron models for training or inference.

        Args:
            load_weights: Whether to configure the provider to load weights
                from HuggingFace format. If False, creates model with random
                initialization.
            hf_path: Optional path to load weights from. If None, uses weights
                from the bridge's hf_pretrained instance. Useful for loading
                weights from a different checkpoint.

        Returns:
            GPTModelProvider: A configured model provider ready to create
                Megatron models

        Example:
            >>> # Create provider and model with loaded weights
            >>> bridge = CausalLMBridge.from_hf_pretrained("meta-llama/Llama-3-8B")
            >>> provider = bridge.to_megatron_provider()
            >>> model = provider.get_model()

            >>> # Create provider without loading weights (for training from scratch)
            >>> provider = bridge.to_megatron_provider(load_weights=False)
            >>> model = provider.get_model()  # Random initialization

            >>> # Load weights from a different checkpoint
            >>> bridge = CausalLMBridge.from_hf_config(config)  # Config only
            >>> provider = bridge.to_megatron_provider(hf_path="./finetuned_model")
            >>> model = provider.get_model()  # Loads finetuned weights

        See Also:
            GPTModelProvider: The provider class for creating models
            load_weights: Method to load weights into existing models
        """

        provider: ModelProviderMixin = self._model_bridge.provider_bridge(self.hf_pretrained)

        if load_weights:
            if hf_path is None:
                provider.register_pre_wrap_hook(
                    partial(self._model_bridge.load_weights_hf_to_megatron, self.hf_pretrained)
                )
            else:
                # Load from specified path
                pre_trained = PreTrainedCausalLM.from_pretrained(hf_path)
                provider.register_pre_wrap_hook(partial(self._model_bridge.load_weights_hf_to_megatron, pre_trained))

        return provider

    @property
    def transformer_config(self) -> TransformerConfig:
        _model_provider = self.to_megatron_provider(load_weights=False)
        return self._create_config_from_provider(_model_provider, TransformerConfig)

    @property
    def mla_transformer_config(self) -> MLATransformerConfig:
        _model_provider = self.to_megatron_provider(load_weights=False)
        return self._create_config_from_provider(_model_provider, MLATransformerConfig)

    @property
    def _model_bridge(self) -> MegatronModelBridge:
        # Use manually specified bridge if provided
        if self._manual_bridge_class is not None:
            return self._manual_bridge_class()

        # Otherwise, auto-detect based on architecture
        return model_bridge.get_model_bridge(self._get_causal_lm_architecture())

    def _get_causal_lm_architecture(self):
        """
        Get the CausalLM architecture class from the HuggingFace model.

        Returns:
            The transformers class for the CausalLM architecture

        Raises:
            ValueError: If no CausalLM architecture is found or if the class cannot be imported
        """
        if isinstance(self.hf_pretrained, PreTrainedCausalLM):
            architectures = getattr(self.hf_pretrained.config, "architectures", [])
        else:
            architectures = getattr(self.hf_pretrained, "architectures", [])

        if not architectures:
            raise ValueError(
                "\n✗ No architectures found in model config\n\n"
                "The model configuration does not specify any architectures.\n"
                "This is required for determining the model type."
            )

        causal_lm_arch = None
        for architecture_name in architectures:
            # TODO: Can we improve this?
            if architecture_name.endswith("ForCausalLM"):
                causal_lm_arch = architecture_name
                break

        if not causal_lm_arch:
            raise ValueError(
                f"\n✗ No CausalLM architecture found\n\n"
                f"Model architectures: {architectures}\n\n"
                f"None of the architectures end with 'ForCausalLM'.\n"
                f"This bridge only supports causal language models.\n"
                f"For other model types, use a different bridge class."
            )

        try:
            return getattr(transformers, causal_lm_arch)
        except AttributeError:
            # For models with trust_remote_code, try to get the class from the loaded model
            if isinstance(self.hf_pretrained, PreTrainedCausalLM):
                # Force model loading to get the actual class
                try:
                    model = self.hf_pretrained.model
                    model_class = type(model)
                    if model_class.__name__ == causal_lm_arch:
                        return model_class
                except Exception:
                    pass

            # If we have a manual bridge class specified, we can use a placeholder
            # since the dispatch won't actually be used
            if self._manual_bridge_class is not None:
                # Return AutoModelForCausalLM as a placeholder
                return transformers.AutoModelForCausalLM

            raise ValueError(
                f"\n✗ Architecture class '{causal_lm_arch}' not found in transformers\n\n"
                f"This could mean:\n"
                f"1. The model requires a newer version of transformers\n"
                f"2. The model uses a custom modeling file not in the standard library\n"
                f"3. There's a typo in the architecture name\n\n"
                f"For models with custom code (trust_remote_code=True), you can manually\n"
                f"specify the bridge to use:\n\n"
                f"    from megatron.bridge.models.your_model import YourModelBridge\n"
                f"    bridge = CausalLMBridge.from_hf_pretrained(\n"
                f"        'model-name',\n"
                f"        bridge_class=YourModelBridge,\n"
                f"        trust_remote_code=True\n"
                f"    )\n\n"
                f"Please verify your transformers installation and the model requirements."
            )

    @classmethod
    def _validate_config(cls, config: PretrainedConfig, path: str | None = None) -> None:
        # Check if this is a causal LM model
        if not cls.supports(config):
            architectures = getattr(config, "architectures", [])
            raise ValueError(
                f"\n✗ Model architecture not supported by CausalLMBridge\n\n"
                f"Model: {path}\n"
                f"Architectures: {architectures}\n\n"
                f"CausalLMBridge only supports models with architectures ending in 'ForCausalLM'.\n"
                f"Found architectures that don't match this pattern.\n\n"
                f"If this is a different model type (e.g., Vision, Sequence-to-Sequence),\n"
                f"you may need to use a different bridge class."
            )

        # Check if we have an implementation for this specific architecture
        architecture = None
        for arch_name in config.architectures:
            if arch_name.endswith("ForCausalLM"):
                architecture = arch_name
                break

        if architecture:
            # Try to get the transformers class to check dispatch registration
            try:
                arch_class = getattr(transformers, architecture)
                # Test if we have a registered implementation
                # Check if this architecture is registered in the dispatch system
                has_implementation = False
                if hasattr(model_bridge.get_model_bridge, "_exact_types"):
                    has_implementation = arch_class in model_bridge.get_model_bridge._exact_types

                if not has_implementation:
                    # Get list of supported models
                    supported_models = cls.list_supported_models()

                    raise ValueError(
                        f"\n✗ Model architecture '{architecture}' is not yet supported\n\n"
                        f"Model: {path}\n"
                        f"Architecture: {architecture}\n\n"
                        f"Currently supported architectures:\n"
                        + "\n".join(f"  • {model}" for model in supported_models)
                        + f"\n\nYou have two options:\n\n"
                        f"Option 1: If you have a bridge implementation, specify it manually:\n"
                        f"    from megatron.bridge.models.your_model import YourModelBridge\n"
                        f"    bridge = CausalLMBridge.from_hf_pretrained(\n"
                        f"        '{path or 'model-name'}',\n"
                        f"        bridge_class=YourModelBridge\n"
                        f"    )\n\n"
                        f"Option 2: Create and register a new bridge:\n"
                        f"1. Create a new bridge class that inherits from MegatronModelBridge\n"
                        f"2. Implement the required methods (provider_bridge, mapping_registry)\n"
                        f"3. Register it with @MegatronModelBridge.register_bridge decorator\n\n"
                        f"Example implementation:\n"
                        f"  from megatron.bridge.models.model_bridge import MegatronModelBridge\n"
                        f"  from transformers import {architecture}\n"
                        f"  from megatron.core.models.gpt import GPTModel\n\n"
                        f"  @MegatronModelBridge.register_bridge(source={architecture}, target=GPTModel)\n"
                        f"  class Megatron{architecture.replace('ForCausalLM', '')}Bridge(MegatronModelBridge):\n"
                        f"      def provider_bridge(self, hf_pretrained):\n"
                        f"          # Return a ModelProvider instance\n"
                        f"          ...\n\n"
                        f"      def mapping_registry(self):\n"
                        f"          # Return a MegatronMappingRegistry with weight mappings\n"
                        f"          ...\n\n"
                        f"For reference implementations, see:\n"
                        f"  • src/megatron/hub/models/llama/llama_causal_bridge.py\n"
                        f"  • src/megatron/hub/models/qwen/qwen_2_causal_bridge.py"
                    ) from None
            except AttributeError:
                raise ValueError(
                    f"\n✗ Could not find architecture class '{architecture}' in transformers\n\n"
                    f"This might be because:\n"
                    f"1. The transformers library version is too old\n"
                    f"2. The model requires a custom modeling file (trust_remote_code=True)\n"
                    f"3. The architecture name is incorrect\n\n"
                    f"For models with custom code, specify the bridge manually:\n"
                    f"    from megatron.bridge.models.your_model import YourModelBridge\n"
                    f"    bridge = CausalLMBridge.from_hf_pretrained(\n"
                    f"        '{path or 'model-name'}',\n"
                    f"        bridge_class=YourModelBridge,\n"
                    f"        trust_remote_code=True\n"
                    f"    )\n\n"
                    f"Please check your transformers installation and model requirements."
                )

    def _get_model_instance(self, model: list[MegatronModelT]) -> MegatronModelT:
        model_instance = model[0]
        while hasattr(model_instance, "module"):
            model_instance = model_instance.module
        return model_instance

    def _create_config_from_provider(self, source_obj: Any, target_dataclass: Type[DataclassT]) -> DataclassT:
        kwargs = {}
        for field in dataclasses.fields(target_dataclass):
            if hasattr(source_obj, field.name):
                kwargs[field.name] = getattr(source_obj, field.name)
        return target_dataclass(**kwargs)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__

        lines_for_build = []

        # Format hf_pretrained
        hf_repr_actual_lines = repr(self.hf_pretrained).splitlines()
        if hf_repr_actual_lines:
            # First line of hf_pretrained part
            lines_for_build.append(f"  (hf_pretrained): {hf_repr_actual_lines[0]}")
            # Subsequent lines of hf_pretrained part, indented
            for line in hf_repr_actual_lines[1:]:
                lines_for_build.append(f"  {line}")
        else:
            lines_for_build.append("  (hf_pretrained): ")  # Fallback for empty repr

        # Format to_megatron dispatcher
        tm_repr_actual_lines = repr(model_bridge.to_megatron).splitlines()
        if tm_repr_actual_lines:
            # First line of to_megatron part
            lines_for_build.append(f"  (to_megatron): {tm_repr_actual_lines[0]}")
            # Subsequent lines of to_megatron part, indented
            for line in tm_repr_actual_lines[1:]:
                lines_for_build.append(f"  {line}")
        else:
            lines_for_build.append("  (to_megatron): ")  # Fallback for empty repr

        return f"{class_name}(\n" + "\n".join(lines_for_build) + "\n)"
