# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Lazy standalone language-checkpoint views of nested multimodal weights."""

from pathlib import Path

import torch
from transformers import PretrainedConfig

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict


class _LanguageModelStateSource(SafeTensorsStateSource):
    """Rename a single language subtree while retaining streaming HF export."""

    def __init__(self, path: str | Path, *, prefix: str, revision: str | None, hub_kwargs: dict[str, object]) -> None:
        super().__init__(path)
        if not prefix or not prefix.endswith("."):
            raise ValueError("A language-model prefix must be nonempty and end with '.'.")
        self.prefix = prefix
        self.revision = revision
        self.hub_kwargs = hub_kwargs

    @property
    def path(self) -> Path:
        """Resolve only the index, never download a multimodal snapshot eagerly."""
        if self._resolved_path_cache is None:
            if Path(self.model_name_or_path).is_dir():
                self._resolved_path_cache = Path(self.model_name_or_path)
            else:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.errors import EntryNotFoundError

                try:
                    index = hf_hub_download(
                        str(self.model_name_or_path),
                        "model.safetensors.index.json",
                        revision=self.revision,
                        **self.hub_kwargs,
                    )
                except EntryNotFoundError:
                    index = hf_hub_download(
                        str(self.model_name_or_path),
                        "model.safetensors",
                        revision=self.revision,
                        **self.hub_kwargs,
                    )
                self._resolved_path_cache = Path(index).parent
        return self._resolved_path_cache

    @property
    def key_to_filename_map(self) -> dict[str, str]:
        """Expose only language keys, with the enclosing model prefix removed."""
        if self._key_to_filename_map_cache is None:
            original = super().key_to_filename_map
            selected = {
                key.removeprefix(self.prefix): filename
                for key, filename in original.items()
                if key.startswith(self.prefix)
            }
            # The base property caches the unfiltered map; never retain it.
            self._key_to_filename_map_cache = None
            if not selected:
                raise ValueError(f"Checkpoint contains no language weights under {self.prefix!r}.")
            self._key_to_filename_map_cache = selected
        return self._key_to_filename_map_cache

    def load_tensors(self, keys_to_load: list[str]) -> dict[str, torch.Tensor]:
        """Read exactly the requested language tensors from their source shards."""
        from safetensors import safe_open

        files: dict[str, list[str]] = {}
        for key in keys_to_load:
            # Fail before opening files if a language weight is missing.
            files.setdefault(self.key_to_filename_map[key], []).append(key)
        result = {}
        for filename, keys in files.items():
            shard = self.path / filename
            if not Path(self.model_name_or_path).is_dir():
                from huggingface_hub import hf_hub_download

                shard = Path(
                    hf_hub_download(
                        str(self.model_name_or_path),
                        filename,
                        revision=self.revision,
                        **self.hub_kwargs,
                    )
                )
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in keys:
                    result[key] = handle.get_tensor(self.prefix + key)
        return result


def create_text_only_pretrained(
    source: PreTrainedCausalLM, *, config: PretrainedConfig, prefix: str
) -> PreTrainedCausalLM:
    """Select language weights for conversion through an existing text bridge.

    No HF model is instantiated. The ordinary pretrained wrapper receives the
    model-specific text config and a lazy, prefix-filtered checkpoint source.

    Args:
        source: Original multimodal checkpoint wrapper.
        config: Standalone language config validated by the model's bridge.
        prefix: Language subtree prefix to remove from checkpoint keys.

    Returns:
        A standard pretrained wrapper containing only the language checkpoint.
    """
    kwargs = dict(source.init_kwargs)
    if kwargs.get("subfolder"):
        raise ValueError("text_only=True does not yet support HF subfolder checkpoints; use a local model directory.")
    revision = getattr(source.config, "_commit_hash", None) or kwargs.get("revision")
    if revision is not None:
        kwargs["revision"] = revision
    pretrained = PreTrainedCausalLM(
        source.model_name_or_path,
        device=source.device,
        torch_dtype=source.torch_dtype,
        trust_remote_code=source.trust_remote_code,
        **kwargs,
    )
    pretrained.config = config
    pretrained._text_only = True
    # Mark media artifacts absent without changing the wrapper's shared defaults
    # or triggering AutoProcessor against the original multimodal repository.
    pretrained._processor = None
    pretrained._image_processor = None
    pretrained.custom_file_patterns = []
    pretrained._state_dict_accessor = StateDict(
        _LanguageModelStateSource(
            source.model_name_or_path,
            prefix=prefix,
            revision=revision,
            hub_kwargs={
                key: kwargs[key]
                for key in ("token", "cache_dir", "local_files_only", "force_download")
                if key in kwargs
            },
        )
    )
    return pretrained
