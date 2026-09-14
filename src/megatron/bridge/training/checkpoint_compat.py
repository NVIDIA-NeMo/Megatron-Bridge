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

"""Model-specific compatibility hooks for distributed checkpoint loading."""

from typing import Any

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedBase
from megatron.core.transformer import MegatronModule


def _retarget_model_sharded_state_dict_for_load(
    model: list[MegatronModule], sharded_state_dict: dict[str, Any], checkpoint_path: str
) -> None:
    """Apply model-specific sharded-key compatibility mappings before checkpoint load."""
    model_state_dicts = []
    if len(model) == 1:
        model_state_dicts.append((model[0], sharded_state_dict["model"]))
    else:
        for index, module in enumerate(model):
            model_key = f"model{index}"
            if model_key in sharded_state_dict:
                model_state_dicts.append((module, sharded_state_dict[model_key]))

    load_hooks = [
        (module, state_dict, getattr(type(module), "_retarget_sharded_state_dict_for_load", None))
        for module, state_dict in model_state_dicts
    ]
    load_hooks = [(module, state_dict, hook) for module, state_dict, hook in load_hooks if callable(hook)]
    if not load_hooks:
        return

    checkpoint_metadata = dist_checkpointing.load_tensors_metadata(checkpoint_path)
    checkpoint_keys = {
        item.key for item in checkpoint_metadata.values() if isinstance(getattr(item, "key", None), str)
    }

    model_key_snapshots = [
        (value, value.key)
        for _, model_state_dict, _ in load_hooks
        for value in nested_values(model_state_dict)
        if isinstance(value, ShardedBase)
    ]
    for module, model_state_dict, hook in load_hooks:
        hook(module, model_state_dict, checkpoint_keys)

    key_rewrites = sorted(
        {(old_key, value.key) for value, old_key in model_key_snapshots if value.key != old_key},
        key=lambda rewrite: len(rewrite[0]),
        reverse=True,
    )
    # Optimizer shard keys end with the model shard key they were derived from.
    # Apply each hook's verified model-key rewrite to those dependent entries as
    # well, while leaving unrelated or mixed-format checkpoint entries intact.
    for value in nested_values(sharded_state_dict):
        if not isinstance(value, ShardedBase) or value.key in checkpoint_keys:
            continue
        for current_model_key, checkpoint_model_key in key_rewrites:
            if value.key == current_model_key or value.key.endswith(f".{current_model_key}"):
                candidate_key = f"{value.key[: -len(current_model_key)]}{checkpoint_model_key}"
                if candidate_key in checkpoint_keys:
                    value.key = candidate_key
                    break
