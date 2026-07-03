# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from torch import int_repr

from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule
from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider


# Kaiko gold-layer webdatasets sit under ``<asset_name>/delta<N>_v<hex>/``. Using the
# terminal segment as the val-loss name yields ``delta0_v72030cff`` — the asset name is
# in the parent directory.
_DELTA_STEM = re.compile(r"^delta\d+_v[0-9a-f]+$")


def _name_from_path(raw_path: str) -> str:
    p = Path(raw_path)
    if _DELTA_STEM.fullmatch(p.stem):
        return p.parent.name
    return p.stem


def _parse_val_blend_entries(metadataset_path: str) -> list[tuple[str, str]]:
    """Extract validation sub-blend ``(name, absolute_path)`` pairs from a MetadatasetV2 YAML.

    Returns one entry per ``splits.val.blend[]`` item. Names are the sub-blend path stems
    (e.g. ``qa_blend`` from ``./qa_blend.yaml``); relative paths are resolved against the
    metadataset's directory, absolute paths kept as-is. Paths whose terminal segment matches
    ``delta<N>_v<hex>`` (Kaiko's delta-versioned gold-layer layout) use the parent directory
    name instead, so ``.../pmc_full_publications_wd/delta0_v72030cff`` becomes
    ``pmc_full_publications_wd``.

    Raises:
        ValueError: if the metadataset has no ``splits.val.blend`` entries.
    """
    base_dir = Path(metadataset_path).parent
    with open(metadataset_path) as f:
        meta = yaml.safe_load(f)

    val_blend = meta.get("splits", {}).get("val", {}).get("blend", [])
    if not val_blend:
        raise ValueError(f"No splits.val.blend entries found in {metadataset_path}")

    entries: list[tuple[str, str]] = []
    for entry in val_blend:
        raw_path = entry["path"]
        resolved = str(base_dir / raw_path) if not Path(raw_path).is_absolute() else raw_path
        entries.append((_name_from_path(raw_path), resolved))
    return entries


@dataclass(kw_only=True)
class EnergonProvider(DatasetProvider):
    """Energon Provider."""

    path: str
    image_processor: Optional[Any] = None
    seq_length: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int_repr
    dataloader_type: str = "external"
    task_encoder: Optional[Any] = None
    # Enable batch-level online sequence packing
    pack_sequences_in_batch: bool = False
    # Size of Energon's packing buffer. Required to enable Energon's sample-packing path: when
    # None, Energon never calls the task encoder's select_samples_to_pack / pack_selected_samples
    # hooks, so any packing_method set on the encoder is a silent no-op.
    packing_buffer_size: Optional[int] = None
    # Evaluate each metadataset val sub-blend separately. Pairs with ValidationConfig.multiple_validation_sets.
    multiple_validation_sets: bool = False

    def _make_datamodule(self, path: str, context: DatasetBuildContext) -> EnergonMultiModalDataModule:
        if (
            self.pack_sequences_in_batch
            and self.task_encoder is not None
            and hasattr(self.task_encoder, "pack_sequences")
        ):
            self.task_encoder.pack_sequences = True
        return EnergonMultiModalDataModule(
            path=path,
            tokenizer=context.tokenizer if context.tokenizer is not None else self.tokenizer,
            image_processor=self.image_processor,
            seq_length=self.seq_length,
            task_encoder=self.task_encoder,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
            packing_buffer_size=self.packing_buffer_size,
            pg_collection=context.pg_collection,
        )

    def build_datasets(self, context: DatasetBuildContext):
        assert self.path, "EnergonProvider.path must be set. Use CLI override: dataset.path=<path>"
        # Energon sample packing (packing_buffer_size) and megatron-bridge batch-level online packing
        # should not be called simultaneously.
        if self.pack_sequences_in_batch and (self.packing_buffer_size or 0) > 0:
            raise ValueError(
                "pack_sequences_in_batch and packing_buffer_size are mutually exclusive: the former "
                "enables megatron-bridge in-batch packing and the latter enables Energon sample "
                "packing, so setting both packs the data twice. Disable one."
            )
        dataset = self._make_datamodule(self.path, context)
        valid = iter(dataset.val_dataloader())
        if self.multiple_validation_sets:
            # Blended loader plus one val-only loader per sub-blend (sub-blends have no train split).
            valid = (
                valid,
                [
                    (name, iter(self._make_datamodule(blend_path, context).val_dataloader()))
                    for name, blend_path in _parse_val_blend_entries(self.path)
                ],
            )
        # Train un-wrapped (not iter()) so RerunDataIterator keeps save_state/restore_state for resume.
        return dataset.train_dataloader(), valid, iter(dataset.val_dataloader())
