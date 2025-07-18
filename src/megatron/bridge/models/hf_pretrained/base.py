#!/usr/bin/env python3
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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, PreTrainedModel

from megatron.bridge.models.state import PyTorchStateSource, SafeTensorsStateSource, StateDict, StateSource


class PreTrainedBase(ABC):
    """
    Abstract base class for all pretrained models.

    This class provides a generic mechanism for managing model artifacts
    (e.g., config, tokenizer) with lazy loading. Subclasses that are
    decorated with `@dataclass` can define artifacts as fields with metadata
    specifying a loader method. The `model` itself is handled via a
    dedicated property that relies on the abstract `_load_model` method.

    Example:
        @dataclass
        class MyModel(PreTrainedBase):
            config: AutoConfig = field(
                init=False,
                metadata=artifact(loader="_load_config")
            )

            def _load_model(self) -> "PreTrainedModel":
                # Implementation for the loading logic
                ...
    """

    model_name_or_path: Union[str, Path]
    ARTIFACTS: ClassVar[List[str]] = []
    OPTIONAL_ARTIFACTS: ClassVar[List[str]] = []

    def __init__(self, **kwargs):
        self._state_dict_accessor: Optional[StateDict] = None
        self.init_kwargs = kwargs

    def get_artifacts(self) -> Dict[str, str]:
        """Get the artifacts dictionary mapping artifact names to their attribute names."""
        return {artifact: f"_{artifact}" for artifact in self.ARTIFACTS}

    def save_artifacts(self, save_directory: Union[str, Path]):
        """
        Saves all loaded, generic artifacts that have a `save_pretrained` method
        to the specified directory. Note: This does not save the `model` attribute.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the config if it's already loaded
        if hasattr(self, "_config") and self._config is not None:
            self._config.save_pretrained(save_path)

        # Iterate over required artifacts to save them in a predictable order
        for name in self.ARTIFACTS:
            # Check if the artifact is already loaded (has the private attribute)
            attr_name = f"_{name}"
            if hasattr(self, attr_name):
                artifact = getattr(self, attr_name)
                if artifact is not None and hasattr(artifact, "save_pretrained"):
                    artifact.save_pretrained(save_path)

        # Iterate over optional artifacts - only save if they exist and have save_pretrained
        for name in self.OPTIONAL_ARTIFACTS:
            artifact = getattr(self, name, None)
            if artifact is not None and hasattr(artifact, "save_pretrained"):
                artifact.save_pretrained(save_path)

    @abstractmethod
    def _load_model(self) -> PreTrainedModel:
        """Subclasses must implement this to load the main model."""
        pass

    @abstractmethod
    def _load_config(self) -> AutoConfig:
        """Subclasses must implement this to load the model config."""
        pass

    @property
    def model(self) -> PreTrainedModel:
        """Lazily loads and returns the underlying model."""
        if not hasattr(self, "_model"):
            self._model = self._load_model()
        return self._model

    @model.setter
    def model(self, value: PreTrainedModel):
        """Manually set the model."""
        self._model = value

    @property
    def config(self) -> AutoConfig:
        """Lazy load and return the model config."""
        if not hasattr(self, "_config"):
            self._config = self._load_config()
        return self._config

    @config.setter
    def config(self, value: AutoConfig):
        """Set the config manually."""
        self._config = value

    @property
    def state(self) -> StateDict:
        """
        Get the state dict accessor for pandas-like querying.

        This accessor can be backed by either a fully loaded model in memory
        or a ".safetensors" checkpoint on disk, enabling lazy loading of tensors.

        Examples:
            model.state()  # Get full state dict
            model.state["key"]  # Get single entry
            model.state[["key1", "key2"]]  # Get multiple entries
            model.state["*.weight"]  # Glob pattern
            model.state.regex(r".*\\.bias$")  # Regex pattern
        """
        if self._state_dict_accessor is None:
            source: Optional[Union[Dict[str, torch.Tensor], StateSource]] = None
            # Prioritize the loaded model's state_dict if available
            if hasattr(self, "_model") and self._model is not None:
                source = self.model.state_dict()
            elif hasattr(self, "model_name_or_path") and self.model_name_or_path:
                # Auto-detect checkpoint format
                source = self._create_state_source(self.model_name_or_path)

            if source is None:
                raise ValueError(
                    "Cannot create StateDict accessor: model is not loaded and model_name_or_path is not set."
                )
            self._state_dict_accessor = StateDict(source)
        return self._state_dict_accessor

    def _create_state_source(self, model_name_or_path: Union[str, Path]) -> StateSource:
        """
        Auto-detect checkpoint format and create appropriate StateSource.

        Checks for both local files and handles HuggingFace Hub models.
        Prefers SafeTensors format when available.
        """
        from glob import glob as file_glob

        model_path = Path(model_name_or_path)

        # For local directories, check what files exist
        if model_path.is_dir():
            # Check for safetensors
            safetensor_files = file_glob(str(model_path / "*.safetensors"))
            safetensor_index = model_path / "model.safetensors.index.json"

            # Check for pytorch files
            pytorch_files = file_glob(str(model_path / "*.bin")) + file_glob(str(model_path / "*.pt"))
            pytorch_index = model_path / "pytorch_model.bin.index.json"

            # Prefer safetensors if available
            if safetensor_files or safetensor_index.exists():
                return SafeTensorsStateSource(model_name_or_path)
            elif pytorch_files or pytorch_index.exists():
                return PyTorchStateSource(model_name_or_path)
            else:
                # No checkpoint files found
                raise FileNotFoundError(
                    f"No checkpoint files found in {model_name_or_path}. Expected .safetensors or .bin/.pt files."
                )
        else:
            # For HuggingFace Hub models, we need to probe what format is available
            # First try to get config to understand the model better
            try:
                from huggingface_hub import HfFileSystem

                # Use HfFileSystem to list files without downloading
                fs = HfFileSystem()
                try:
                    files = fs.ls(model_name_or_path, detail=False)
                    # Extract just the filenames
                    filenames = [f.split("/")[-1] for f in files]

                    # Check what's available
                    has_safetensors = any(f.endswith(".safetensors") for f in filenames)
                    has_safetensors_index = "model.safetensors.index.json" in filenames
                    has_pytorch = any(f.endswith(".bin") or f.endswith(".pt") for f in filenames)
                    has_pytorch_index = "pytorch_model.bin.index.json" in filenames

                    if has_safetensors or has_safetensors_index:
                        return SafeTensorsStateSource(model_name_or_path)
                    elif has_pytorch or has_pytorch_index:
                        return PyTorchStateSource(model_name_or_path)
                    else:
                        raise FileNotFoundError(f"No checkpoint files found in HuggingFace model {model_name_or_path}")
                except Exception:
                    # If we can't list files, fall back to trying both
                    pass
            except ImportError:
                # huggingface_hub not available
                pass

            # Fall back to trying SafeTensors first (preferred format)
            # The sources will handle downloading if the model exists
            try:
                # Try to create SafeTensorsStateSource
                # If it fails during initialization, fall back to PyTorch
                source = SafeTensorsStateSource(model_name_or_path)
                # Do a quick check if it can access keys
                try:
                    source.get_all_keys()
                    return source
                except FileNotFoundError:
                    # SafeTensors not available, try PyTorch format
                    return PyTorchStateSource(model_name_or_path)
            except Exception:
                # If SafeTensors fails, try PyTorch format
                return PyTorchStateSource(model_name_or_path)
