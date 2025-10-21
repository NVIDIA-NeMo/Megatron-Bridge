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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch


# Make peft import optional
try:
    from peft import PeftConfig

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

    # Create a dummy PeftConfig class that raises an error when used
    class PeftConfig:  # type: ignore
        """Dummy PeftConfig that raises an error when peft is not installed."""

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            raise ImportError(
                "HuggingFace PEFT library is required to load adapters. "
                "Please install it with: pip install megatron-bridge[peft]"
            )


from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, WeightConversionTask
from megatron.bridge.peft.api import MegatronPEFTModel
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.conversion import peft_bridge
from megatron.bridge.peft.conversion.pretrained_adapters import PreTrainedAdapters


class AutoPEFTBridge:
    """
    Automatically select and instantiate the appropriate PEFT bridge for adapters.

    This unified PEFT bridge class combines automatic adapter detection with full bridge
    functionality for converting adapters between HuggingFace and Megatron formats.
    It handles the conversion of PEFT adapters (LoRA, DoRA, etc.) between HuggingFace's
    PEFT library format and Megatron-Core's distributed training format.

    The bridge supports both directions of conversion:
    - HuggingFace → Megatron: For applying pretrained adapters to Megatron training
    - Megatron → HuggingFace: For saving trained adapters in HF PEFT format

    Args:
        adapters: PreTrainedAdapters instance with loaded adapter config and weights
        adapter_name: Name of the adapter for future multi-adapter support

    Example:
        >>> # Manual base model specification
        >>> base_bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3-8B")
        >>> peft_bridge = AutoPEFTBridge.from_hf_pretrained("username/llama-lora", base_bridge)
        >>> peft_model = peft_bridge.to_megatron_model()
        >>>
        >>> # Automatic base model detection
        >>> peft_bridge = AutoPEFTBridge.from_hf_pretrained("codelion/Llama-3.2-1B-Instruct-tool-calling-lora")
        >>> peft_model = peft_bridge.to_megatron_model()  # Auto-detects base model
    """

    def __init__(self, adapters: PreTrainedAdapters, adapter_name: str = "default"):
        """Initialize AutoPEFTBridge with pretrained adapters.

        Args:
            adapters: Loaded adapters with config and state
            adapter_name: Name for this adapter (for multi-adapter support)
        """
        if not isinstance(adapters, PreTrainedAdapters):
            raise ValueError("adapters must be a PreTrainedAdapters instance")
        self.adapters: PreTrainedAdapters = adapters
        self.adapter_name = adapter_name
        self._peft_transform: Optional[PEFT] = None
        self._base_bridge: Optional[AutoBridge] = None
        self._peft_bridge = None

    @classmethod
    def from_hf_pretrained(
        cls,
        path: Union[str, Path],
        base_bridge: Optional[AutoBridge] = None,
        *,
        adapter_name: str = "default",
        **kwargs,
    ) -> "AutoPEFTBridge":
        """Load an AutoPEFTBridge from pretrained adapters.

        Args:
            path: HuggingFace adapter model ID or path to adapter directory
            base_bridge: AutoBridge for the base model. If None, will attempt
                to auto-detect from adapter config's 'base_model_name_or_path' field.
            adapter_name: Name of the adapter for multi-adapter support
            **kwargs: Additional arguments passed to PreTrainedAdapters.from_pretrained

        Returns:
            AutoPEFTBridge instance with loaded adapters and base bridge

        Example:
            >>> # Manual base model specification
            >>> base_bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")
            >>> peft_bridge = AutoPEFTBridge.from_hf_pretrained("username/lora", base_bridge)
            >>> peft_model = peft_bridge.to_megatron_model()
            >>>
            >>> # Automatic base model detection
            >>> peft_bridge = AutoPEFTBridge.from_hf_pretrained("codelion/Llama-3.2-1B-Instruct-tool-calling-lora")
            >>> peft_model = peft_bridge.to_megatron_model()  # Auto-detects base model
        """
        # Load adapters from the specified path
        adapters = PreTrainedAdapters.from_pretrained(path, **kwargs)

        # Auto-detect base model if not provided
        if base_bridge is None:
            base_bridge = cls._auto_detect_base_bridge(adapters)

        # Create the bridge instance
        bridge_instance = cls(adapters=adapters, adapter_name=adapter_name)
        bridge_instance._base_bridge = base_bridge

        return bridge_instance

    def to_megatron_model(
        self,
        *,
        training: bool = True,
        wrap_with_ddp: bool = True,
        use_cpu_initialization: bool = False,
    ) -> MegatronPEFTModel:
        """Convert adapters to Megatron PEFT model.

        Args:
            training: Whether the model will be used for training
            wrap_with_ddp: Whether to wrap with DDP for distributed training
            use_cpu_initialization: Initialize model on CPU to save memory

        Returns:
            PEFTModel: A PEFT-enabled Megatron model ready for training/inference

        Example:
            >>> peft_bridge = AutoPEFTBridge.from_hf_pretrained("adapter", base_bridge)
            >>> peft_model = peft_bridge.to_megatron_model(training=True)
        """
        # Use the base bridge that was provided at load time
        if self._base_bridge is None:
            raise RuntimeError("Base bridge not available. This should have been set during from_hf_pretrained().")

        # Get the PEFT bridge implementation
        bridge = self._peft_bridge_impl

        # Initialize the base bridge with the base model bridge
        bridge.base_bridge = self._base_bridge

        # Create the PEFT transform
        peft = bridge.peft_bridge(self.adapters)
        self._peft_transform = peft

        # Create base model provider and apply PEFT
        provider = self._base_bridge.to_megatron_provider()

        def _apply_peft_hook(model_or_stages):
            adapted = peft(model_or_stages, training=training)
            # Load adapter weights after structure is applied
            bridge.load_adapters_hf_to_megatron(self.adapters, adapted)
            return adapted

        provider.register_pre_wrap_hook(_apply_peft_hook)
        stages = provider.provide_distributed_model(
            wrap_with_ddp=wrap_with_ddp, use_cpu_initialization=use_cpu_initialization
        )

        return MegatronPEFTModel(stages, peft)

    @property
    def peft_config(self) -> PeftConfig:
        """Get the PEFT configuration."""
        return self.adapters.config

    @property
    def base_bridge(self) -> AutoBridge:
        """Get the PEFT bridge."""
        return self._base_bridge

    @property
    def _peft_bridge_impl(self):
        """Get the underlying PEFT bridge implementation."""
        if self._peft_bridge is None:
            config_class = type(self.adapters.config)
            self._peft_bridge = peft_bridge.get_peft_bridge(config_class)
        return self._peft_bridge

    @staticmethod
    def _auto_detect_base_bridge(adapters: PreTrainedAdapters) -> AutoBridge:
        """Auto-detect base model from adapter config."""
        config = adapters.config
        base_model_path = getattr(config, "base_model_name_or_path", None)

        if base_model_path is None:
            raise ValueError(
                "\n✗ Base bridge not provided and cannot be auto-detected\n\n"
                "Please provide a base_bridge argument to from_hf_pretrained(), "
                "or ensure the adapter configuration includes 'base_model_name_or_path'.\n\n"
                "Example:\n"
                "  # Option 1: Provide base_bridge explicitly\n"
                "  base_bridge = AutoBridge.from_hf_pretrained('meta-llama/Llama-3.2-1B')\n"
                "  peft_bridge = AutoPEFTBridge.from_hf_pretrained('adapter', base_bridge)\n\n"
                "  # Option 2: Use adapter with base_model_name_or_path in config\n"
                "  peft_bridge = AutoPEFTBridge.from_hf_pretrained('adapter')  # Auto-detects"
            )

        print(f"🔍 Auto-detected base model: {base_model_path}")
        return AutoBridge.from_hf_pretrained(base_model_path)

    def export_adapter_weights(
        self,
        model: MegatronPEFTModel,
        cpu: bool = False,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable["HFWeightTuple"]:
        """
        Export PEFT model adapter weights to HuggingFace format.

        This method yields adapter weight tensors in HuggingFace format, handling the
        gathering of distributed tensors and format conversion. It's useful for
        streaming adapter export or custom processing. All ranks get full tensors.

        Args:
            model: PEFT model instance
            cpu: Whether to move tensors to CPU before yielding
            show_progress: Display progress bar during export
            conversion_tasks: Pre-built conversion tasks (advanced usage)

        Yields:
            HFWeightTuple: Named tuples of (param_name, weight_tensor)

        Example:
            >>> # Export and process adapter weights
            >>> for name, weight in peft_bridge.export_adapter_weights(peft_model):
            ...     print(f"Exported {name}: {weight.shape}")

            >>> # Export with specific settings
            >>> weights = list(peft_bridge.export_adapter_weights(
            ...     peft_model,
            ...     cpu=True
            ... ))
        """
        if self._base_bridge is None:
            raise RuntimeError("Base bridge not available. Call from_hf_pretrained() first.")

        # Get PEFT bridge for streaming conversion
        bridge = self._peft_bridge_impl
        bridge.base_bridge = self._base_bridge

        return bridge.stream_adapters_megatron_to_hf(
            model.stages,
            adapters=self.adapters,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
        )

    def save_adapter_weights(
        self, model: MegatronPEFTModel, path: Union[str, Path], show_progress: bool = True
    ) -> None:
        """
        Save PEFT model adapter weights in HuggingFace safetensors format.

        This method exports only the adapter weights (not configuration) to
        safetensors files compatible with HuggingFace PEFT. It uses streaming save
        to handle large adapters efficiently.

        Args:
            model: PEFT model instance
            path: Directory path where weight files will be saved
            show_progress: Display progress bar during export

        Example:
            >>> # Save just the adapter weights
            >>> peft_bridge.save_adapter_weights(peft_model, "./adapter_weights")

            >>> # Save without progress bar
            >>> peft_bridge.save_adapter_weights(peft_model, "./weights", show_progress=False)
        """
        self._save_adapter_weights(model, path, show_progress)

    @classmethod
    def can_handle(cls, path: Union[str, Path], trust_remote_code: bool = False) -> bool:
        """
        Check if the bridge can handle the adapters at the given path.

        This method allows you to verify adapter compatibility before attempting
        to load them, which can be useful for validation or UI feedback.

        Args:
            path: Path to adapter directory or HuggingFace adapter ID
            trust_remote_code: Whether to trust remote code when loading config

        Returns:
            bool: True if the bridge supports the adapters, False otherwise

        Example:
            >>> # Check if adapters are supported
            >>> if AutoPEFTBridge.can_handle("username/llama-lora-adapters"):
            ...     print("Adapters are supported!")
            ... else:
            ...     print("Adapters require custom bridge implementation")
        """
        try:
            adapters = PreTrainedAdapters.from_pretrained(path, trust_remote_code=trust_remote_code)
            return cls.supports(adapters.config)
        except Exception:
            return False

    @classmethod
    def list_supported_adapters(cls) -> List[str]:
        """List all adapter types currently supported by the PEFT bridge system."""
        from megatron.bridge.peft.conversion.peft_bridge import list_registered_bridges

        supported = []
        bridges = list_registered_bridges()
        for source_type in bridges.keys():
            type_name = source_type.__name__.replace("Config", "").upper()
            if type_name not in supported:
                supported.append(type_name)

        return sorted(supported)

    @classmethod
    def supports(cls, adapter_config: Dict) -> bool:
        """Check if this bridge supports the given adapter configuration."""
        try:
            from peft import get_peft_config
        except ImportError:
            raise ImportError(
                "HuggingFace PEFT library is required to check adapter support. "
                "Please install it with: pip install megatron-bridge[peft]"
            )

        from megatron.bridge.peft.conversion.peft_bridge import list_registered_bridges

        try:
            config_obj = get_peft_config(adapter_config)
            config_class = type(config_obj)
            bridges = list_registered_bridges()
            return config_class in bridges
        except Exception:
            return False

    def save_hf_pretrained(
        self, peft_model: MegatronPEFTModel, path: Union[str, Path], show_progress: bool = True
    ) -> None:
        """Save a PEFT model adapters in HuggingFace format."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self._save_adapter_config(path)
        else:
            self._save_adapter_config(path)

        self._save_adapter_weights(peft_model, path, show_progress)

    def _save_adapter_config(self, path: Union[str, Path]) -> None:
        """Save adapter configuration to directory."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # Convert config to dict and make it JSON serializable
        if hasattr(self.adapters.config, "to_dict"):
            config_dict = self.adapters.config.to_dict()
        else:
            config_dict = dict(self.adapters.config)

        # Convert any sets to lists for JSON serialization
        def make_json_serializable(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj

        config_dict = make_json_serializable(config_dict)

        with open(out / "adapter_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def _save_adapter_weights(
        self, peft_model: MegatronPEFTModel, path: Union[str, Path], show_progress: bool = True
    ) -> None:
        """Save adapter weights in HuggingFace safetensors format."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Get PEFT bridge for streaming conversion
        bridge = self._peft_bridge_impl
        bridge.base_bridge = self._base_bridge

        # Stream weights and collect on rank 0
        gathered_weights = {}
        for hf_weight_tuple in bridge.stream_adapters_megatron_to_hf(
            peft_model.stages, adapters=self.adapters, show_progress=show_progress
        ):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                gathered_weights[hf_weight_tuple.param_name] = hf_weight_tuple.weight.cpu()

        # Only rank 0 writes the files
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"💾 Saving {len(gathered_weights)} adapter weights to {Path(path) / 'adapter_model.safetensors'}")
            import safetensors.torch

            safetensors.torch.save_file(gathered_weights, Path(path) / "adapter_model.safetensors")
            print("✅ Adapter weights saved successfully")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    @classmethod
    def import_ckpt(
        cls,
        hf_adapter_path: Union[str, Path],
        megatron_path: Union[str, Path],
        base_bridge: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Import HuggingFace PEFT adapters and save as a Megatron checkpoint.

        This is a convenience method that combines loading HuggingFace PEFT adapters,
        converting them to Megatron format, and saving them as a native Megatron
        checkpoint. This is useful for preparing adapters for Megatron training.

        Args:
            hf_adapter_path: HuggingFace adapter ID or path to adapter directory
                Examples: "username/llama-lora", "./my_adapters"
            megatron_path: Directory path where the Megatron checkpoint will be saved
            base_bridge: Optional AutoBridge instance for the base model. If not provided,
                the base model will be loaded from the adapter's base_model_name_or_path.
            **kwargs: Additional arguments passed to from_hf_pretrained
                Common options include:
                - trust_remote_code: Allow custom model code execution

        Example:
            >>> from megatron.bridge import AutoBridge
            >>> from megatron.bridge.peft import AutoPEFTBridge
            >>>
            >>> # Basic import (loads base model automatically)
            >>> AutoPEFTBridge.import_ckpt(
            ...     "username/llama-lora",
            ...     "./megatron_checkpoints/llama_lora"
            ... )
            >>>
            >>> # Import with explicit base model
            >>> base = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B")
            >>> AutoPEFTBridge.import_ckpt(
            ...     "username/llama-lora",
            ...     "./megatron_checkpoints/llama_lora",
            ...     base_bridge=base
            ... )

        Note:
            This saves the full PEFT model state (base + adapters) for training continuation.
            For adapter-only exports, use save_adapter_weights() or save_hf_pretrained().
        """
        from megatron.bridge.training.model_load_save import save_megatron_model

        # Load the HuggingFace PEFT adapters
        peft_bridge = cls.from_hf_pretrained(hf_adapter_path, base_bridge=base_bridge, **kwargs)

        # Convert to Megatron PEFT model
        peft_model = peft_bridge.to_megatron_model(wrap_with_ddp=False, use_cpu_initialization=True)

        # Save as Megatron checkpoint (full model with adapters)
        save_megatron_model(
            peft_model,
            megatron_path,
            hf_tokenizer_path=peft_bridge._base_bridge.hf_pretrained.model_name_or_path,
        )

    def export_ckpt(
        self,
        megatron_path: Union[str, Path],
        hf_path: Union[str, Path],
        show_progress: bool = True,
    ) -> None:
        """
        Export a Megatron PEFT checkpoint to HuggingFace adapter format.

        This is a convenience method that loads a Megatron PEFT checkpoint and
        exports the adapters to HuggingFace format. This is useful for sharing
        trained adapters or deploying them with HuggingFace inference tools.

        Args:
            megatron_path: Directory path where the Megatron PEFT checkpoint is stored
            hf_path: Directory path where the HuggingFace adapter will be saved
            show_progress: Display progress bar during adapter export

        Example:
            >>> from megatron.bridge.peft import AutoPEFTBridge
            >>>
            >>> # Load bridge with base model info
            >>> peft_bridge = AutoPEFTBridge.from_hf_pretrained("username/llama-lora")
            >>>
            >>> # Export checkpoint to HuggingFace format
            >>> peft_bridge.export_ckpt(
            ...     "./megatron_checkpoints/llama_lora",
            ...     "./hf_exports/llama_lora"
            ... )
            >>>
            >>> # Load the exported adapter with HuggingFace
            >>> from peft import PeftModel, AutoModel
            >>> base_model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B")
            >>> model = PeftModel.from_pretrained(base_model, "./hf_exports/llama_lora")

        Note:
            This exports only the adapter weights and config, not the full base model.
            The base model must be loaded separately when using the exported adapter.
        """
        try:
            from megatron.bridge.training.model_load_save import temporary_distributed_context
        except ImportError:
            raise ImportError("megatron.bridge.training is not available.")

        # Export ckpt performs on CPU
        with temporary_distributed_context(backend="gloo"):
            # Load the Megatron PEFT model
            peft_model = self.load_megatron_model(megatron_path, wrap_with_ddp=False)

            # Save in HuggingFace adapter format
            self.save_hf_pretrained(peft_model, hf_path, show_progress=show_progress)

    def load_megatron_model(
        self,
        megatron_path: Union[str, Path],
        wrap_with_ddp: bool = True,
    ) -> MegatronPEFTModel:
        """
        Load a Megatron PEFT checkpoint.

        This method loads a checkpoint that was saved with import_ckpt() and recreates
        the PEFT model structure with loaded weights.

        Args:
            megatron_path: Directory path where the Megatron PEFT checkpoint is stored
            wrap_with_ddp: Whether to wrap the model with DistributedDataParallel

        Returns:
            MegatronPEFTModel: The loaded PEFT model with weights from checkpoint

        Note:
            This requires that the base bridge was initialized with the correct base model
            configuration. The checkpoint must have been created with a compatible PEFT config.

        Warning:
            This is a simplified implementation that recreates the PEFT structure but may not
            properly restore all adapter weights from the checkpoint. Full checkpoint restoration
            requires loading the saved adapter state dict and applying it to the PEFT model.
            For now, this method provides the interface for checkpoint loading workflows.
        """
        # TODO: Implement proper checkpoint loading
        # Current limitation: This creates a fresh PEFT model structure but doesn't load
        # the adapter weights from the checkpoint. Proper implementation would:
        # 1. Load the base model from checkpoint
        # 2. Extract adapter weights from checkpoint
        # 3. Re-create PEFT model structure
        # 4. Load adapter weights into the PEFT structure

        # For now, just create a fresh PEFT model with the expected structure
        peft_model = self.to_megatron_model(wrap_with_ddp=wrap_with_ddp)
        return peft_model
