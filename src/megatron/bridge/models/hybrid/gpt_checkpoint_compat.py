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

"""Load GPTModel distributed checkpoints into HybridModels that pair each mixer with a feed-forward layer.

A GPTModel layer ``i`` holds a token mixer and its MLP/MoE. A HybridModel
migrated from it splits that layer into positions ``2*i`` (mixer) and
``2*i+1`` (feed-forward), so only the checkpoint keys differ. Reading the old
keys keeps checkpoints written before such a migration usable.
"""

import re
from collections.abc import Collection, Iterable

import torch
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedBase, ShardedStateDict
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, parse_hybrid_pattern
from megatron.core.models.hybrid.hybrid_model import HybridModel


_DECODER_LAYER = re.compile(r"(^|\.)decoder\.layers\.(\d+)\.")
_MTP_INNER_LAYER = re.compile(r"\.(mtp_model_layer|transformer_layer)\.layers\.\d+\.")
_FEED_FORWARD_SYMBOLS = {Symbols.MLP, Symbols.MOE}


def _has_paired_layers(pattern: str | None) -> bool:
    if not pattern:
        return False
    main = (parse_hybrid_pattern(pattern).main_pattern or "").replace(Symbols.PIPE, "")
    return (
        bool(main)
        and len(main) % 2 == 0
        and all(symbol not in _FEED_FORWARD_SYMBOLS for symbol in main[::2])
        and all(symbol in _FEED_FORWARD_SYMBOLS for symbol in main[1::2])
    )


def has_paired_hybrid_layers(models: Iterable[torch.nn.Module]) -> bool:
    """Whether every HybridModel in ``models`` pairs each mixer with a following feed-forward layer."""
    patterns = [
        module.hybrid_layer_pattern
        for model in models
        if isinstance(model, torch.nn.Module)
        for module in model.modules()
        if isinstance(module, HybridModel)
    ]
    return bool(patterns) and all(_has_paired_layers(pattern) for pattern in patterns)


def gpt_checkpoint_key(key: str) -> str:
    """Return the GPTModel checkpoint key of a paired HybridModel sharded key."""
    key = key.replace("decoder.final_norm.", "decoder.final_layernorm.")
    # GPT MTP layers save their inner layer as ``transformer_layer``; Hybrid MTP splits it into two layers.
    key = _MTP_INNER_LAYER.sub(".transformer_layer.", key)
    return _DECODER_LAYER.sub(lambda match: f"{match.group(1)}decoder.layers.{int(match.group(2)) // 2}.", key)


def is_gpt_layout_checkpoint(checkpoint_keys: Collection[str]) -> bool:
    """Whether a checkpoint stores the decoder with GPTModel keys rather than HybridModel keys."""
    keys = list(checkpoint_keys)
    return any("decoder.final_layernorm." in key for key in keys) and not any(
        "decoder.final_norm." in key for key in keys
    )


def read_gpt_layout_keys(sharded_state_dict: ShardedStateDict, models: list[torch.nn.Module]) -> None:
    """Point paired Hybrid submodules at their GPTModel checkpoint keys.

    Only the keys used to locate data in the checkpoint change; the state dict
    still loads into the HybridModel's own parameters. Optimizer state built
    from this state dict follows the same keys. GPTModel never saves the output
    layer's (empty) extra state, so that entry is dropped. Prefixes come from
    the unwrapped models, leaving other decoders (such as vision) unchanged.
    """
    for index, model in enumerate(models):
        prefixes = tuple(
            name + "." if name else ""
            for name, module in model.named_modules()
            if isinstance(module, HybridModel) and _has_paired_layers(module.hybrid_layer_pattern)
        )
        model_state = sharded_state_dict["model" if len(models) == 1 else f"model{index}"]
        for value in nested_values(model_state):
            if isinstance(value, ShardedBase) and value.key.startswith(prefixes):
                value.key = gpt_checkpoint_key(value.key)
        _drop_output_layer_extra_state(model_state, {prefix + "output_layer._extra_state" for prefix in prefixes})


def _drop_output_layer_extra_state(node: dict | list, keys: set[str]) -> None:
    items = node.items() if isinstance(node, dict) else enumerate(node)
    for name, value in list(items):
        if isinstance(value, (dict, list)):
            _drop_output_layer_extra_state(value, keys)
        elif name in keys:
            del node[name]
