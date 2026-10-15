# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Language-only checkpoint views and standalone export artifacts for Muse Spark Open."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import torch

from megatron.bridge.models.hf_pretrained import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict


class MuseSparkTextStateSource(SafeTensorsStateSource):
    """Expose the VL decoder and separate LM head through standalone text keys.

    Prefix sources retain index-first loading and the existing streaming writer.
    No perception tensor is loaded, including when shards mix modalities.
    """

    def __init__(self, path: str | Path, *, revision: str | None, hub_kwargs: dict) -> None:
        super().__init__(path, revision=revision, hub_kwargs=hub_kwargs)
        self._parts = {
            "model.": SafeTensorsStateSource(
                path, revision=revision, key_prefix="model.language_model.", hub_kwargs=hub_kwargs
            ),
            "lm_head.": SafeTensorsStateSource(path, revision=revision, key_prefix="lm_head.", hub_kwargs=hub_kwargs),
        }

    @property
    def key_to_filename_map(self) -> dict[str, str]:
        """Retain source shard grouping while excluding perception keys."""
        if self._key_to_filename_map_cache is None:
            self._key_to_filename_map_cache = {
                prefix + key: filename
                for prefix, source in self._parts.items()
                for key, filename in source.key_to_filename_map.items()
            }
        return self._key_to_filename_map_cache

    def load_tensors(self, keys_to_load: list[str]) -> dict[str, torch.Tensor]:
        """Read only the selected language tensors under their original names."""
        missing = set(keys_to_load) - self.key_to_filename_map.keys()
        if missing:
            raise KeyError(f"Not a selected language tensor: {sorted(missing)}")
        result = {}
        for prefix, source in self._parts.items():
            keys = [key.removeprefix(prefix) for key in keys_to_load if key.startswith(prefix)]
            if keys:
                result.update({prefix + key: value for key, value in source.load_tensors(keys).items()})
        return result


def select_text_pretrained(hf_pretrained: PreTrainedCausalLM) -> PreTrainedCausalLM:
    """Project a VL source without mutating its config, index, or weights."""
    if not hasattr(hf_pretrained.config, "text_config"):
        return hf_pretrained
    config = copy.deepcopy(hf_pretrained.config.text_config)
    config.architectures = ["MuseSparkForCausalLM"]
    config.tie_word_embeddings = getattr(hf_pretrained.config, "tie_word_embeddings", config.tie_word_embeddings)
    if hasattr(config, "auto_map"):
        del config.auto_map
    dtype = getattr(hf_pretrained.config, "dtype", None)
    if dtype is not None:
        config.dtype = dtype
    kwargs = dict(hf_pretrained.init_kwargs)
    if kwargs.get("subfolder"):
        raise ValueError("text_only=True requires a local model directory instead of an HF subfolder.")
    revision = getattr(hf_pretrained.config, "_commit_hash", None) or kwargs.get("revision")
    if revision is not None:
        kwargs["revision"] = revision
    text = PreTrainedCausalLM(
        hf_pretrained.model_name_or_path,
        device=hf_pretrained.device,
        torch_dtype=hf_pretrained.torch_dtype,
        trust_remote_code=hf_pretrained.trust_remote_code,
        **kwargs,
    )
    text.config = config
    text._text_only = True
    text._state_dict_accessor = StateDict(
        MuseSparkTextStateSource(
            hf_pretrained.model_name_or_path,
            revision=revision,
            hub_kwargs={
                key: kwargs[key]
                for key in ("token", "cache_dir", "local_files_only", "force_download")
                if key in kwargs
            },
        )
    )
    for attr in ("_tokenizer", "_generation_config"):
        if hasattr(hf_pretrained, attr):
            setattr(text, attr, getattr(hf_pretrained, attr))
    text._processor = None
    text._image_processor = None
    text.custom_file_patterns = []
    return text


def export_text_artifacts(path: Path) -> None:
    """Install the small HF head adapter that the Transformers text-only class needs."""
    filename = "muse_spark_open_transformers_patch.py"
    source = Path(__file__).parent / "modeling_muse_spark_open" / filename
    shutil.copyfile(source, path / filename)
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = ["MuseSparkForCausalLM"]
    config["auto_map"] = {"AutoModelForCausalLM": "muse_spark_open_transformers_patch.MuseSparkForCausalLM"}
    config_path.write_text(json.dumps(config, indent=2) + "\n")
