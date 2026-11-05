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
"""Per-token scalar value head for critic engines (``from_recipe(output_head="value")``).

Follows Bridge's own value-model convention: the language-model output layer is replaced by the
replicated ``LinearForLastLayer`` head (``create_value_head_hook``), so a value engine's forward is the
plain no-labels call and the head output is what the model returns as "logits", ``[B, T, 1]``. Because
the replacement happens in a pre-wrap hook, the pretrained checkpoint is loaded into the bare model
first, while ``output_layer`` still has the vocabulary shape the checkpoint holds; the optimizer and
the DDP buffers are then built around the scalar head. As in Bridge's token-classification provider,
the recipe copy gets MTP and embedding tying turned off, since both treat ``output_layer`` as the
vocabulary projection.

TODO(value-export): ``export`` of a value-head engine is rejected; the HF layout has no slot for the head.
"""

from __future__ import annotations

import logging
from typing import Any

from megatron.core.transformer import MegatronModule

from megatron.bridge.engine.checkpoint import load_pretrained_weights
from megatron.bridge.models.common.heads import create_value_head_hook


logger = logging.getLogger(__name__)

VALUE_HEAD_MODEL_SETTINGS = {
    "mtp_num_layers": None,
    "mtp_enabled": False,
    "share_embeddings_and_output_weights": False,
}


def apply_value_head_model_settings(model_config: Any) -> dict[str, Any]:
    """Turn MTP and embedding tying off on the engine's recipe copy; returns the fields that changed."""
    changed = {}
    for name, value in VALUE_HEAD_MODEL_SETTINGS.items():
        if hasattr(model_config, name) and getattr(model_config, name) != value:
            changed[name] = getattr(model_config, name)
            setattr(model_config, name, value)
    if changed:
        logger.info("output_head='value': overriding %s on the recipe copy", changed)
    return changed


def make_value_head_hook(recipe: Any, checkpoint_path: str | None):
    """Pre-wrap hook: load the pretrained weights into the bare model, then replace the LM head."""
    provider = recipe.model
    replace = create_value_head_hook(
        hidden_size=provider.hidden_size, sequence_parallel=bool(getattr(provider, "sequence_parallel", False))
    )

    def hook(model: list[MegatronModule]) -> list[MegatronModule]:
        if checkpoint_path is not None:
            load_pretrained_weights(recipe, model, None, checkpoint_path)
        return replace(model) or model

    return hook
