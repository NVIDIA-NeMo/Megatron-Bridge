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
import shutil
from pathlib import Path
from typing import Union

from diffusers import FluxTransformer2DModel
from megatron.bridge.models.hf_pretrained.base import PreTrainedBase
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict, StateSource
from transformers import AutoConfig


class FluxSafeTensorsStateSource(SafeTensorsStateSource):
    """
    FLUX-specific state source that writes exported HF shards under 'transformer/'.
    """

    def save_generator(self, generator, output_path, strict: bool = True):
        # Ensure shards are written under transformer/
        target_dir = Path(output_path) / "transformer"
        return super().save_generator(generator, target_dir, strict=strict)


class PreTrainedFlux(PreTrainedBase):
    """
    Lightweight pretrained wrapper for Diffusers FLUX models.

    Provides access to FLUX config and state through the common PreTrainedBase API
    so bridges can consume `.config` and `.state` uniformly.

    NOTE: Due to FLUX uses HF's Diffusers library, which has different checkpoint directory structure to HF's Transformer library,
          we need a wrapper to load the model weights and config from the correct directory (e.g., ./transformer).
          The diffusers's structure includes all components in the diffusion pipeline (VAE, text encoders, etc.).
          The actual transformer weights are stored in the ./transformer directory. Hence, we adjust the input and output
          path directory accordingly. We also need to override the save_artifacts method to save relevant correct configs
          files to the corresponding directory.
    """

    def __init__(self, model_name_or_path: Union[str, Path], **kwargs):
        self._model_name_or_path = str(model_name_or_path)
        super().__init__(**kwargs)

    @property
    def model_name_or_path(self) -> str:
        return self._model_name_or_path

    # Model loading is optional for conversion; implemented for completeness
    def _load_model(self) -> FluxTransformer2DModel:
        return FluxTransformer2DModel.from_pretrained(self.model_name_or_path)

    # Config is required by the FLUX bridge
    def _load_config(self) -> AutoConfig:
        # FluxTransformer2DModel returns a config-like object with required fields

        print(f"Loading config from {self.model_name_or_path}")

        return FluxTransformer2DModel.from_pretrained(self.model_name_or_path, subfolder="transformer").config

    @property
    def state(self) -> StateDict:
        """
        FLUX-specific StateDict that reads safetensors from the fixed 'transformer/' subfolder.
        """
        if getattr(self, "_state_dict_accessor", None) is None:
            source: StateSource | None = None
            if hasattr(self, "_model") and self._model is not None:
                # If model is loaded, use its in-memory state_dict
                source = self.model.state_dict()
            else:
                # Always load from 'transformer/' subfolder for FLUX
                source = FluxSafeTensorsStateSource(Path(self.model_name_or_path) / "transformer")
            self._state_dict_accessor = StateDict(source)
        return self._state_dict_accessor

    def save_artifacts(self, save_directory: Union[str, Path]):
        """
        Save FLUX artifacts (currently config) alongside exported weights.
        Writes transformer/config.json into the destination.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        # Ensure transformer subdir exists at destination
        dest_transformer = save_path / "transformer"
        dest_transformer.mkdir(parents=True, exist_ok=True)

        # 1) If source has a config.json under transformer/, copy it
        src_config = Path(self.model_name_or_path) / "transformer" / "config.json"
        src_index = Path(self.model_name_or_path) / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
        if src_config.exists():
            shutil.copyfile(src_config, dest_transformer / "config.json")
            if src_index.exists():
                shutil.copyfile(src_index, dest_transformer / "diffusion_pytorch_model.safetensors.index.json")
            return

        # 2) Otherwise, try to export config from the HF model instance
        try:
            model = FluxTransformer2DModel.from_pretrained(self.model_name_or_path, subfolder="transformer")
            cfg = getattr(model, "config", None)
            if cfg is not None:
                # Prefer to_dict if available
                cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
                with open(dest_transformer / "config.json", "w") as f:
                    json.dump(cfg_dict, f, indent=2)
        except Exception:
            # Best-effort: if config cannot be produced, leave only weights
            pass
