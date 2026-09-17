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
from typing import NoReturn

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


class TextOnlyPreTrainedCausalLM(PreTrainedCausalLM):
    """Present nested language weights as a standalone causal-LM checkpoint.

    The config must describe the language model, not the original multimodal
    wrapper. Conversion and export use the existing text model bridge unchanged.
    Model families opt in explicitly after validating their config semantics.
    """

    OPTIONAL_ARTIFACTS = ["generation_config"]

    def __init__(self, source: PreTrainedCausalLM, *, config: PretrainedConfig, prefix: str) -> None:
        kwargs = dict(source.init_kwargs)
        if kwargs.get("subfolder"):
            raise ValueError(
                "text_only=True does not yet support HF subfolder checkpoints; use a local model directory."
            )
        revision = getattr(source.config, "_commit_hash", None) or kwargs.get("revision")
        if revision is not None:
            kwargs["revision"] = revision
        super().__init__(
            source.model_name_or_path,
            device=source.device,
            torch_dtype=source.torch_dtype,
            trust_remote_code=source.trust_remote_code,
            **kwargs,
        )
        self.config = config
        self.custom_file_patterns = []
        self._language_prefix = prefix
        self._hub_kwargs = {
            key: kwargs[key] for key in ("token", "cache_dir", "local_files_only", "force_download") if key in kwargs
        }

    @property
    def state(self) -> StateDict:
        """Return the lazy language-only state, including serialized MTP weights."""
        if self._state_dict_accessor is None:
            self._state_dict_accessor = StateDict(
                _LanguageModelStateSource(
                    self.model_name_or_path,
                    prefix=self._language_prefix,
                    revision=self.init_kwargs.get("revision"),
                    hub_kwargs=self._hub_kwargs,
                )
            )
        return self._state_dict_accessor

    def _load_model(self) -> NoReturn:
        # Transformers' prefix heuristics are not a safe substitute for the
        # explicit conversion view. Inference is supported on the exported
        # standalone text checkpoint, not this lazy source wrapper.
        raise NotImplementedError("Use the Megatron text model or load its standalone HF export for inference.")
