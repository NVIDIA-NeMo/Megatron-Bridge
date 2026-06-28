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

from collections.abc import Mapping


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
