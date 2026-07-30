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

"""Prepare a legacy Nemotron-Omni MCore checkpoint for current Megatron-Bridge.

Legacy checkpoints saved before Bridge v0.6 have three incompatibilities with
the current codebase:

1. ``common.pt`` stores ``ModelType.encoder_and_decoder`` (integer value 2),
   which was removed from ``megatron.core.enums.ModelType``.  Loading the file
   raises ``ValueError: 2 is not a valid ModelType``.

2. DCP weight keys use the old module prefixes (``language_model.*``,
   ``vision_model.*``, ``vision_projection.*``, ``sound_model.*``,
   ``sound_projection.*``) instead of the current ``llava_model.*`` namespace.
   The ``.metadata`` pickle must be rewritten to match.

3. No ``run_config.yaml`` is present.  Bridge's ``convert.sh`` (and
   ``AutoBridge.from_auto_config``) expect this file alongside the checkpoint.
   It is generated from the checkpoint's own ``common.pt`` args plus the HF
   reference config.

This script creates a lightweight *compatibility view* in ``--output-dir``:

- Symlinks to the original shard files (zero data copy).
- A patched ``common.pt`` with the corrected ``ModelType``.
- A patched ``.metadata`` pickle with renamed DCP keys.
- A generated ``run_config.yaml`` reflecting the checkpoint's actual settings.
- ``latest_checkpointed_iteration.txt`` pointing at the patched iteration.

The original checkpoint directory is **never modified**.

Usage
-----
::

    python scripts/conversion/prepare_nemotron_omni_legacy_ckpt.py \\
        --source-ckpt /path/to/source/iter_0000020 \\
        --hf-ref     nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \\
        --output-dir /path/to/output/compat-ckpt

Then pass ``--megatron-path /path/to/output/compat-ckpt`` to ``convert.sh``.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import shutil
from dataclasses import replace
from pathlib import Path

import torch


logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# DCP key prefixes used by old checkpoints → current llava_model.* namespace.
_LEGACY_ROOTS = (
    "language_model.",
    "vision_model.",
    "vision_projection.",
    "sound_model.",
    "sound_projection.",
)


def _patch_model_type(common: dict) -> dict:
    """Replace the legacy ModelType(2) with the current encoder_or_decoder."""
    from megatron.core.enums import ModelType

    # Temporarily register the missing value so torch.load can deserialize it.
    ModelType._value2member_map_[2] = ModelType.encoder_or_decoder

    model_type = common.get("model_type")
    if isinstance(model_type, int) and model_type == 2:
        common["model_type"] = ModelType.encoder_or_decoder
    elif hasattr(model_type, "value") and model_type.value == 2:
        common["model_type"] = ModelType.encoder_or_decoder
    else:
        log.info("model_type already up-to-date: %s", model_type)

    args = common.get("args")
    if args is not None and hasattr(args, "model_type"):
        if isinstance(args.model_type, int) and args.model_type == 2:
            args.model_type = ModelType.encoder_or_decoder
        elif hasattr(args.model_type, "value") and args.model_type.value == 2:
            args.model_type = ModelType.encoder_or_decoder

    return common


def _rename_key(fqn: str) -> str:
    if any(fqn.startswith(root) for root in _LEGACY_ROOTS):
        return f"llava_model.{fqn}"
    return fqn


def _patch_metadata(src_metadata_path: Path, dst_metadata_path: Path) -> int:
    """Rewrite .metadata with renamed DCP keys; return number of renamed keys."""
    with src_metadata_path.open("rb") as fh:
        metadata = pickle.load(fh)

    renamed_state = {_rename_key(k): v for k, v in metadata.state_dict_metadata.items()}
    renamed_storage = {replace(idx, fqn=_rename_key(idx.fqn)): info for idx, info in metadata.storage_data.items()}
    metadata.state_dict_metadata = renamed_state
    metadata.storage_data = renamed_storage

    with dst_metadata_path.open("wb") as fh:
        pickle.dump(metadata, fh)

    n_renamed = sum(1 for k in renamed_state if k.startswith("llava_model."))
    return n_renamed


def _load_common_pt(src_path: Path) -> dict:
    """Load common.pt, temporarily registering the missing ModelType value."""
    from megatron.core.enums import ModelType

    ModelType._value2member_map_[2] = ModelType.encoder_or_decoder
    common = torch.load(src_path, map_location="cpu", weights_only=False)
    return common


def _generate_run_config(common: dict, hf_ref: str, output_iter_dir: Path, *, trust_remote_code: bool) -> None:
    """Generate run_config.yaml from the HF reference + common.pt overrides."""
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.training.config import ConfigContainer
    from megatron.bridge.utils.yaml_utils import dump_dataclass_to_yaml

    log.info("Loading HF reference config from %s ...", hf_ref)
    bridge = AutoBridge.from_hf_pretrained(hf_ref, trust_remote_code=trust_remote_code)
    provider = bridge.to_megatron_provider(load_weights=False)

    # Override settings that differ between the HF reference and this checkpoint.
    args = common.get("args")
    if args is not None:
        max_pos = getattr(args, "max_position_embeddings", None)
        if max_pos is not None:
            provider.max_position_embeddings = max_pos
            log.info("  max_position_embeddings: %d", max_pos)

        has_sound = getattr(args, "has_sound", False)
        provider.has_sound = bool(has_sound)
        log.info("  has_sound: %s", provider.has_sound)

        # Temporal video embedder: nano-3.5-omni was trained without it.
        sep_video = getattr(args, "separate_video_embedder", False)
        provider.separate_video_embedder = bool(sep_video)
        if not sep_video:
            provider.temporal_ckpt_compat = False
            provider.temporal_patch_dim = 1
        log.info("  separate_video_embedder: %s", provider.separate_video_embedder)

        # RADIO class tokens: read from checkpoint args when present.
        # The Bridge fix (PR #5170) reads this from the HF vision_config on
        # the import path; here we propagate it into run_config.yaml so the
        # export path also sees the correct value.
        radio_cls = getattr(args, "vision_class_token_len", None)
        if radio_cls is not None:
            provider.vision_class_token_len = int(radio_cls)
            log.info("  vision_class_token_len: %d", provider.vision_class_token_len)

    run_config_path = output_iter_dir / "run_config.yaml"
    provider.finalize()

    # Bridge has no provider-level "save run config" API: ``run_config.yaml`` is
    # normally written by ``ConfigContainer.to_yaml()`` during training, and read
    # back by ``load_model_config()``, which only consumes the top-level ``model``
    # key and rebuilds the provider via ``instantiate()``. Emitting just that key
    # is therefore sufficient, and avoids fabricating optimizer/dataset/train
    # sections we do not have.
    #
    # The conversion MUST go through ``ConfigContainer._convert_value_to_dict``.
    # Handing the provider straight to the YAML representers looks like it works
    # but silently emits only ``_target_``/``_call_`` with no field values, so
    # ``instantiate()`` returns a default-constructed provider (num_layers=None,
    # hidden_size=0) and every checkpoint-specific override is lost.
    model_dict = ConfigContainer._convert_value_to_dict(provider)
    dump_dataclass_to_yaml({"model": model_dict}, str(run_config_path))
    log.info("Wrote run_config.yaml → %s", run_config_path)


def _symlink_shards(src_iter_dir: Path, dst_iter_dir: Path) -> int:
    """Create relative symlinks for every .distcp shard."""
    n = 0
    rel_src = os.path.relpath(src_iter_dir, dst_iter_dir)
    for shard in src_iter_dir.glob("__*.distcp"):
        dst = dst_iter_dir / shard.name
        if not dst.exists():
            dst.symlink_to(Path(rel_src) / shard.name)
            n += 1
    return n


def prepare(
    source_iter_dir: Path,
    hf_ref: str,
    output_dir: Path,
    *,
    trust_remote_code: bool,
) -> None:
    """Build the compatibility view."""
    if not source_iter_dir.is_dir():
        raise ValueError(f"source checkpoint iteration dir not found: {source_iter_dir}")

    common_src = source_iter_dir / "common.pt"
    if not common_src.exists():
        raise ValueError(f"common.pt not found under {source_iter_dir}")

    iteration_name = source_iter_dir.name  # e.g. iter_0000020
    dst_iter_dir = output_dir / iteration_name
    dst_iter_dir.mkdir(parents=True, exist_ok=True)

    # latest_checkpointed_iteration.txt
    latest_src = source_iter_dir.parent / "latest_checkpointed_iteration.txt"
    latest_dst = output_dir / "latest_checkpointed_iteration.txt"
    if latest_src.exists():
        shutil.copy2(latest_src, latest_dst)
    else:
        latest_dst.write_text(iteration_name.lstrip("iter_0").lstrip("0") + "\n")

    # metadata.json (Bridge reads this for shard topology)
    meta_json_src = source_iter_dir / "metadata.json"
    if meta_json_src.exists():
        shutil.copy2(meta_json_src, dst_iter_dir / "metadata.json")

    # .metadata pickle — rename DCP weight keys
    dot_meta_src = source_iter_dir / ".metadata"
    n_renamed = 0
    if dot_meta_src.exists():
        n_renamed = _patch_metadata(dot_meta_src, dst_iter_dir / ".metadata")
        log.info("Renamed %d DCP keys to llava_model.* namespace", n_renamed)
    else:
        log.warning(".metadata not found — skipping key renaming (may fail on export)")

    # Symlink .distcp shards
    n_shards = _symlink_shards(source_iter_dir, dst_iter_dir)
    log.info("Created %d shard symlinks", n_shards)

    # common.pt — patch ModelType
    log.info("Loading and patching common.pt ...")
    common = _load_common_pt(common_src)
    common = _patch_model_type(common)
    torch.save(common, dst_iter_dir / "common.pt")
    log.info("Wrote patched common.pt")

    # run_config.yaml
    if hf_ref:
        _generate_run_config(common, hf_ref, dst_iter_dir, trust_remote_code=trust_remote_code)
    else:
        log.warning(
            "--hf-ref not provided; skipping run_config.yaml generation.\n"
            "You must supply this file manually before running convert.sh."
        )

    log.info(
        "\nCompatibility view written to: %s\n"
        "Pass this directory to convert.sh as --megatron-path:\n\n"
        "  ./scripts/conversion/convert.sh export \\\n"
        "      --hf-model <HF_REF> \\\n"
        "      --megatron-path %s \\\n"
        "      --hf-path <OUTPUT_HF_DIR> \\\n"
        "      --tp 2 --pp 1 --ep 2 --etp 1 \\\n"
        "      --not-strict\n",
        output_dir,
        output_dir,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-ckpt",
        required=True,
        type=Path,
        help=("Path to the iteration directory inside the source MCore checkpoint (e.g. /path/to/ckpt/iter_0000020)."),
    )
    parser.add_argument(
        "--hf-ref",
        default="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        help=(
            "HF model ID or local path used as the architectural reference for "
            "generating run_config.yaml.  Weights are not loaded.  "
            "Defaults to the public Nano-Omni-30B model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write the compatibility view into.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True when loading the HF reference config (default: True).",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"--output-dir already exists and is non-empty: {args.output_dir}")

    prepare(
        source_iter_dir=args.source_ckpt,
        hf_ref=args.hf_ref,
        output_dir=args.output_dir,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
