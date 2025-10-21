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
from typing import ClassVar

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

    model_name_or_path: str | Path
    ARTIFACTS: ClassVar[list[str]] = []
    OPTIONAL_ARTIFACTS: ClassVar[list[str]] = []

    def __init__(self, **kwargs):
        self._state_dict_accessor: StateDict | None = None
        self.init_kwargs = kwargs

    def get_artifacts(self) -> dict[str, str]:
        """Get the artifacts dictionary mapping artifact names to their attribute names."""
        return {artifact: f"_{artifact}" for artifact in self.ARTIFACTS}

    def save_artifacts(self, save_directory: str | Path):
        """
        Saves all loaded, generic artifacts that have a `save_pretrained` method
        to the specified directory. Note: This does not save the `model` attribute.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        _ = getattr(self, "config")  # trigger lazy loading of config
        if hasattr(self, "_config") and self._config is not None:
            if hasattr(self._config, "quantization_config"):
                # quantized export is not supported currently
                del self._config.quantization_config
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
            source: dict[str, torch.Tensor] | StateSource | None = None
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
