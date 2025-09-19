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

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, PreTrainedModel

from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict, StateSource


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
        # Store the original source path for custom modeling file preservation
        self._original_source_path: Optional[Union[str, Path]] = None

    def get_artifacts(self) -> Dict[str, str]:
        """Get the artifacts dictionary mapping artifact names to their attribute names."""
        return {artifact: f"_{artifact}" for artifact in self.ARTIFACTS}

    def _copy_custom_modeling_files(self, source_path: Union[str, Path], target_path: Union[str, Path]) -> None:
        """Copy custom modeling files from source to target directory.
        
        This preserves custom modeling files that were used during model loading
        with trust_remote_code=True, ensuring the saved model can be loaded properly.
        
        Args:
            source_path: Source directory containing custom modeling files
            target_path: Target directory to copy files to
        """
        source_path = Path(source_path)
        target_path = Path(target_path)
        
        # Common custom modeling file patterns
        custom_file_patterns = [
            "modeling_*.py",
            "configuration_*.py", 
            "tokenization_*.py",
            "processing_*.py",
            "feature_extraction_*.py"
        ]
        
        copied_files = []
        
        # First, try to copy from local directory if it exists
        if source_path.exists() and source_path.is_dir():
            for pattern in custom_file_patterns:
                for file_path in source_path.glob(pattern):
                    if file_path.is_file():
                        target_file = target_path / file_path.name
                        try:
                            shutil.copy2(file_path, target_file)
                            copied_files.append(file_path.name)
                        except (OSError, IOError):
                            # Silently skip files that can't be copied
                            pass
        
        # If no files were copied and source_path looks like a HuggingFace Hub ID,
        # try to download the custom modeling files directly from the Hub
        if not copied_files and "/" in str(source_path) and not source_path.exists():
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
                
                # Get list of Python files in the repository
                repo_files = list_repo_files(str(source_path))
                python_files = [f for f in repo_files if f.endswith('.py')]
                
                for py_file in python_files:
                    # Check if it matches our custom file patterns
                    if any(py_file.startswith(pattern.replace('*.py', '').replace('*', '')) 
                           for pattern in custom_file_patterns):
                        try:
                            downloaded_file = hf_hub_download(
                                repo_id=str(source_path),
                                filename=py_file,
                                local_dir=target_path,
                                local_dir_use_symlinks=False
                            )
                            copied_files.append(py_file)
                        except Exception:
                            # Silently skip files that can't be downloaded
                            pass
                            
            except Exception:
                # If HuggingFace Hub operations fail, silently continue
                pass
        
        return copied_files

    def save_artifacts(self, save_directory: Union[str, Path]):
        """
        Saves all loaded, generic artifacts that have a `save_pretrained` method
        to the specified directory. Note: This does not save the `model` attribute.
        
        If the model was loaded with trust_remote_code=True, this method will also
        attempt to preserve any custom modeling files to ensure the saved model
        can be loaded properly.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        _ = getattr(self, "config")  # trigger lazy loading of config
        if hasattr(self, "_config") and self._config is not None:
            self._config.save_pretrained(save_path)

        # Iterate over required artifacts to save them in a predictable order
        for name in self.ARTIFACTS:
            # Access the public property to trigger lazy loading if needed
            artifact = getattr(self, name)
            attr_name = f"_{name}"
            if hasattr(self, attr_name):
                if artifact is not None and hasattr(artifact, "save_pretrained"):
                    artifact.save_pretrained(save_path)

        # Iterate over optional artifacts - only save if they exist and have save_pretrained
        for name in self.OPTIONAL_ARTIFACTS:
            artifact = getattr(self, name, None)
            if artifact is not None and hasattr(artifact, "save_pretrained"):
                artifact.save_pretrained(save_path)
                
        # Preserve custom modeling files if trust_remote_code was used
        if (hasattr(self, 'trust_remote_code') and self.trust_remote_code):
            # Try original source path first, then fallback to model_name_or_path
            source_paths = []
            if hasattr(self, '_original_source_path') and self._original_source_path:
                source_paths.append(self._original_source_path)
            if hasattr(self, 'model_name_or_path') and self.model_name_or_path:
                source_paths.append(self.model_name_or_path)
                
            for source_path in source_paths:
                copied_files = self._copy_custom_modeling_files(source_path, save_path)
                if copied_files:
                    # Successfully copied files, no need to try other paths
                    break

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
                source = SafeTensorsStateSource(self.model_name_or_path)

            if source is None:
                raise ValueError(
                    "Cannot create StateDict accessor: model is not loaded and model_name_or_path is not set."
                )
            self._state_dict_accessor = StateDict(source)
        return self._state_dict_accessor
