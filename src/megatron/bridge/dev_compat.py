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

"""Compatibility shims for APIs missing from the pinned Megatron Core dev revision.

Keep guarded MCore imports and temporary Bridge backports in this module so the
compatibility surface has one removal point when the dev APIs reach main.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from megatron.core.msc_utils import MultiStorageClientFeature


if TYPE_CHECKING:
    from megatron.bridge.training.tokenizers.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer

logger = getLogger(__name__)

try:
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
        FullyShardedDataParallelV1,
        FullyShardedDataParallelV2,
    )
except ImportError:
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel

    MEGATRON_FSDP_TYPES = (FullyShardedDataParallel,)
    MEGATRON_FSDP_V2_TYPES = ()
    MCORE_HAS_MEGATRON_FSDP_V2 = False
else:
    MEGATRON_FSDP_TYPES = (FullyShardedDataParallelV1, FullyShardedDataParallelV2)
    MEGATRON_FSDP_V2_TYPES = (FullyShardedDataParallelV2,)
    MCORE_HAS_MEGATRON_FSDP_V2 = True

try:
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec as MCORE_HYBRID_DSV4_STACK_SPEC
except ImportError:
    MCORE_HYBRID_DSV4_STACK_SPEC = None


try:
    from megatron.core._rank_utils import set_default_log_ranks as _set_default_log_ranks
except ImportError:
    _set_default_log_ranks = None

try:
    from megatron.core.utils import (
        grant_shape_mismatch_for_gtp_padding,
        resolve_gtp_pad_for_alignment,
    )
except ImportError:

    def resolve_gtp_pad_for_alignment(*, fp4: bool = False, fp8_recipe: str | None = None, fp8: bool = False) -> int:
        """Map quantized GTP recipes to their dim-0 alignment without requiring a newer MCore."""
        if fp4:
            return 16
        if fp8_recipe == "mxfp8":
            return 32
        if fp8:
            return 16
        return 1

    def grant_shape_mismatch_for_gtp_padding(
        sharded_state_dict: dict[str, Any], checkpoint_dir: str, pad_for_alignment: int
    ) -> None:
        """Allow only checkpoint shape differences explained by GTP dim-0 padding."""
        from megatron.core.dist_checkpointing.dict_utils import nested_values
        from megatron.core.dist_checkpointing.mapping import ShardedTensor
        from megatron.core.dist_checkpointing.serialization import load_tensors_metadata

        sharded_tensors = [value for value in nested_values(sharded_state_dict) if isinstance(value, ShardedTensor)]
        if not sharded_tensors:
            return
        try:
            checkpoint_metadata = load_tensors_metadata(str(checkpoint_dir))
        except Exception as exc:  # noqa: BLE001
            getLogger(__name__).warning("Could not read checkpoint metadata for GTP padding: %s", exc)
            return
        for sharded_tensor in sharded_tensors:
            if sharded_tensor.allow_shape_mismatch or sharded_tensor.key not in checkpoint_metadata:
                continue
            checkpoint_shape = checkpoint_metadata[sharded_tensor.key].global_shape
            axis = sharded_tensor.prepend_axis_num
            if axis >= len(checkpoint_shape):
                continue
            checkpoint_dim = int(checkpoint_shape[axis])
            required_dim = int(sharded_tensor.global_shape[axis])
            if checkpoint_dim == required_dim:
                continue
            unpadded_dim = required_dim - int(getattr(sharded_tensor, "gtp_pad_length", 0))
            is_valid_padding = checkpoint_dim == unpadded_dim or (
                pad_for_alignment > 1 and checkpoint_dim % pad_for_alignment == 0
            )
            sharded_tensor.allow_shape_mismatch = checkpoint_dim >= unpadded_dim and is_valid_padding


try:
    from megatron.training.checkpointing import save_tokenizer_assets
except ImportError:

    def save_tokenizer_assets(
        tokenizer: MegatronTokenizer,
        tokenizer_config: TokenizerConfig,
        checkpoint_path: str,
        *,
        raise_on_error: bool = False,
    ) -> None:
        """Save tokenizer files to the checkpoint directory.

        Always saves tokenizer files to ensure checkpoints are self-contained
        and portable. Handles both HuggingFace tokenizers and file-based tokenizers.
        Compatible with MultiStorageClient for cloud storage support.

        Args:
            tokenizer: The tokenizer instance to save.
            tokenizer_config: The tokenizer configuration (used for file-based tokenizers).
            checkpoint_path: The checkpoint directory path.
            raise_on_error: Propagate tokenizer persistence errors to the caller.
        """
        if tokenizer is None:
            return

        # Only rank 0 saves tokenizer files
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank != 0:
            return

        def resolve_path(path_str: str) -> str:
            """Resolve relative paths to absolute paths."""
            if not path_str:
                return path_str
            path_obj = Path(path_str)
            if path_obj.is_absolute():
                return path_str
            # Resolve relative to current working directory
            return str(path_obj.resolve())

        try:
            # Check if MultiStorageClient is enabled
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                checkpoint_path_obj = msc.Path(checkpoint_path)
                tokenizer_dir = checkpoint_path_obj / "tokenizer"
                tokenizer_dir.mkdir(parents=True, exist_ok=True)
                use_msc = True
            else:
                tokenizer_dir = os.path.join(checkpoint_path, "tokenizer")
                os.makedirs(tokenizer_dir, exist_ok=True)
                use_msc = False

            tokenizer_type = tokenizer_config.tokenizer_type

            # Handle HuggingFace and Multimodal tokenizers
            if tokenizer_type in ("HuggingFaceTokenizer", "MultimodalTokenizer"):
                if use_msc:
                    import tempfile

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        if hasattr(tokenizer, "save_pretrained"):
                            tokenizer.save_pretrained(tmp_dir)
                        elif hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "save_pretrained"):
                            tokenizer._tokenizer.save_pretrained(tmp_dir)
                        else:
                            logger.debug(
                                f"{tokenizer_type} does not support save_pretrained(), skipping tokenizer save"
                            )
                            return

                        logger.debug(f"Saving {tokenizer_type} files to {tokenizer_dir}")
                        for filename in os.listdir(tmp_dir):
                            src_path = os.path.join(tmp_dir, filename)
                            if os.path.isfile(src_path):
                                dest_path = tokenizer_dir / filename
                                with open(src_path, "rb") as src_f:
                                    with msc.open(str(dest_path), "wb") as dest_f:
                                        dest_f.write(src_f.read())
                else:
                    logger.debug(f"Saving {tokenizer_type} files to {tokenizer_dir}")
                    if hasattr(tokenizer, "save_pretrained"):
                        tokenizer.save_pretrained(tokenizer_dir)
                    elif hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "save_pretrained"):
                        tokenizer._tokenizer.save_pretrained(tokenizer_dir)
                return

            # Handle file-based tokenizers - resolve all paths
            files_to_copy = []

            if tokenizer_type in ("BertWordPieceLowerCase", "BertWordPieceCase"):
                if tokenizer_config.vocab_file:
                    resolved_path = resolve_path(tokenizer_config.vocab_file)
                    files_to_copy.append(("vocab_file", resolved_path, "vocab.txt"))

            elif tokenizer_type == "GPT2BPETokenizer":
                if tokenizer_config.vocab_file:
                    resolved_path = resolve_path(tokenizer_config.vocab_file)
                    files_to_copy.append(("vocab_file", resolved_path, "vocab.json"))
                if tokenizer_config.merge_file:
                    resolved_path = resolve_path(tokenizer_config.merge_file)
                    files_to_copy.append(("merge_file", resolved_path, "merges.txt"))

            elif tokenizer_type in ("SentencePieceTokenizer", "GPTSentencePieceTokenizer", "Llama2Tokenizer"):
                if tokenizer_config.tokenizer_model:
                    resolved_path = resolve_path(tokenizer_config.tokenizer_model)
                    files_to_copy.append(("tokenizer_model", resolved_path, "tokenizer.model"))

            elif tokenizer_type == "TikTokenizer":
                if tokenizer_config.tokenizer_model:
                    resolved_path = resolve_path(tokenizer_config.tokenizer_model)
                    files_to_copy.append(("tokenizer_model", resolved_path, "tokenizer.json"))

            elif tokenizer_type == "NullTokenizer":
                logger.debug(f"{tokenizer_type} requires no file artifacts")
                return

            # Copy the files
            if files_to_copy:
                logger.debug(f"Saving {tokenizer_type} files to {tokenizer_dir}")
                for config_attr, source_path, dest_filename in files_to_copy:
                    if source_path and os.path.exists(source_path):
                        if use_msc:
                            dest_path = tokenizer_dir / dest_filename
                            with open(source_path, "rb") as src_f:
                                with msc.open(str(dest_path), "wb") as dest_f:
                                    dest_f.write(src_f.read())
                            logger.debug(f"Copied {config_attr}: {source_path} -> {dest_path}")
                        else:
                            dest_path = os.path.join(tokenizer_dir, dest_filename)
                            shutil.copy2(source_path, dest_path)
                            logger.debug(f"Copied {config_attr}: {source_path} -> {dest_path}")
                    else:
                        logger.debug(f"{config_attr} not found at resolved path: {source_path}")

        except Exception as e:
            if rank == 0:
                logger.error(f"Failed to save tokenizer files: {e}")
                import traceback

                logger.error(traceback.format_exc())
            if raise_on_error:
                raise


def get_gtp_api() -> Any | None:
    """Return MCore's optional GTP API when the selected core provides it."""
    try:
        from megatron.core.tensor_parallel import gtp_api
    except ImportError:
        return None
    return gtp_api


def get_gtp_native_fp8_load_context(module: torch.nn.Module):
    """Return the native-FP8 load context, or a no-op on cores without GTP."""
    gtp_api = get_gtp_api()
    if gtp_api is None or not gtp_api.HAVE_GTP:
        from contextlib import nullcontext

        return nullcontext()
    return gtp_api.gtp_native_fp8_load_context(module)


def set_default_log_ranks(ranks: Iterable[int]) -> None:
    """Set MCore's default logging ranks when the pinned revision supports it."""
    if _set_default_log_ranks is not None:
        _set_default_log_ranks(ranks)
