# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

from collections.abc import Mapping, MutableMapping


def get_hf_model_id_from_model_config(model_config: object) -> str | None:
    """Return the Hugging Face model identifier stored on a model config.

    Legacy provider configs expose ``hf_model_id`` directly. Builder-backed
    configs keep it in ``extra_checkpoint_metadata`` so it survives
    ``ModelConfig.as_dict()`` without adding a model-specific phantom field.

    Args:
        model_config: A model config object or its serialized mapping.

    Returns:
        The configured Hugging Face model identifier, or ``None`` when absent.
    """
    if isinstance(model_config, Mapping):
        hf_model_id = model_config.get("hf_model_id")
        metadata = model_config.get("extra_checkpoint_metadata")
    else:
        hf_model_id = getattr(model_config, "hf_model_id", None)
        metadata = getattr(model_config, "extra_checkpoint_metadata", None)

    if hf_model_id:
        return str(hf_model_id)
    if isinstance(metadata, Mapping):
        hf_model_id = metadata.get("hf_model_id")
        if hf_model_id:
            return str(hf_model_id)
    return None


def set_hf_model_id_on_model_config(model_config: object, hf_model_id: str) -> None:
    """Store a Hugging Face model identifier on a model config.

    Legacy provider configs own an ``hf_model_id`` field. Builder-backed
    configs store the value in serializable checkpoint metadata instead of
    introducing a model-specific field.

    Args:
        model_config: A mutable model config object or serialized mapping.
        hf_model_id: Hugging Face model identifier or local source path.

    Raises:
        TypeError: If a serialized config mapping is immutable or existing
            checkpoint metadata is not a mapping.
    """
    if isinstance(model_config, Mapping):
        if not isinstance(model_config, MutableMapping):
            raise TypeError("Cannot set hf_model_id on an immutable model-config mapping.")
        if "hf_model_id" in model_config:
            model_config["hf_model_id"] = hf_model_id
            return
        metadata = model_config.get("extra_checkpoint_metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("extra_checkpoint_metadata must be a mapping when present.")
        updated_metadata = dict(metadata or {})
        updated_metadata["hf_model_id"] = hf_model_id
        model_config["extra_checkpoint_metadata"] = updated_metadata
        return

    if hasattr(model_config, "hf_model_id"):
        setattr(model_config, "hf_model_id", hf_model_id)
        return

    metadata = getattr(model_config, "extra_checkpoint_metadata", None)
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("extra_checkpoint_metadata must be a mapping when present.")
    updated_metadata = dict(metadata or {})
    updated_metadata["hf_model_id"] = hf_model_id
    setattr(model_config, "extra_checkpoint_metadata", updated_metadata)
