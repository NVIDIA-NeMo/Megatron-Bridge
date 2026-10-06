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

"""Feature-gated checkpoint helpers for MCore revisions without GTP support."""

from contextlib import AbstractContextManager, nullcontext
from importlib import import_module
from typing import Any


def apply_gtp_checkpoint_padding(
    sharded_state_dict: dict[str, Any], checkpoint_name: str, model_config: Any, *, gtp_enabled: bool
) -> None:
    """Apply MCore padding metadata, retaining ordinary loads on older revisions."""
    utils = import_module("megatron.core.utils")
    resolve_alignment = getattr(utils, "resolve_gtp_pad_for_alignment", None)
    grant_shape_mismatch = getattr(utils, "grant_shape_mismatch_for_gtp_padding", None)
    if resolve_alignment is None or grant_shape_mismatch is None:
        if gtp_enabled:
            raise RuntimeError("GTP checkpoint loading requires MCore's GTP padding helpers.")
        return
    alignment = resolve_alignment(
        fp4=bool(getattr(model_config, "fp4", None)),
        fp8_recipe=getattr(model_config, "fp8_recipe", None),
        fp8=bool(getattr(model_config, "fp8", None)),
    )
    grant_shape_mismatch(sharded_state_dict, checkpoint_name, alignment)


def gtp_checkpoint_load_context(module: Any, *, gtp_enabled: bool) -> AbstractContextManager:
    """Return the native GTP load context when supported, or an ordinary context."""
    module_name = "megatron.core.tensor_parallel.gtp_api"
    try:
        gtp_api = import_module(module_name)
    except ModuleNotFoundError as error:
        # An import failure inside an existing GTP module is a runtime problem,
        # not evidence that this MCore revision predates the feature.
        if error.name != module_name:
            raise
        gtp_api = None
    if gtp_api is not None and gtp_api.HAVE_GTP:
        return gtp_api.gtp_native_fp8_load_context(module)
    if gtp_enabled:
        raise RuntimeError("GTP checkpoint loading requires MCore and Transformer Engine GTP support.")
    return nullcontext()


def save_tokenizer_assets(tokenizer: Any, tokenizer_config: Any, checkpoint_name: str) -> None:
    """Delegate tokenizer asset saving, failing explicitly on unsupported MCore."""
    checkpointing = import_module("megatron.training.checkpointing")
    save_assets = getattr(checkpointing, "save_tokenizer_assets", None)
    if save_assets is None:
        raise RuntimeError(
            "This MCore revision does not support saving tokenizer assets. "
            "Set checkpoint.save_tokenizer_assets=False or use MCore with save_tokenizer_assets support."
        )
    save_assets(tokenizer, tokenizer_config, checkpoint_name)
