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

"""Declarative BAGEL WDS/Energon training dataset."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from megatron.energon import WorkerConfig, get_train_dataset

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.models.bagel.data.energon import (
    BagelEditingTaskEncoder,
    BagelT2ITaskEncoder,
    BagelVLMTaskEncoder,
)
from megatron.bridge.models.bagel.data.external import BagelExternalLoader, BagelRNGIterator
from megatron.bridge.models.bagel.data.order import BagelPlannedLoader, plan_manifest_indices
from megatron.bridge.models.bagel.data.packing import BagelPacker


@dataclass(kw_only=True)
class BagelDatasetConfig(DatasetProvider):
    """Build the validated raw-data→WDS→Energon→BAGEL packing chain."""

    dataset_root: str | None = None
    bagel_repo: str | None = None
    tokenizer_model: str | None = None
    seed: int = 42
    data_seed: int = 42
    t2i_num_used_data: int = 10
    editing_num_used_data: int = 10
    vlm_num_used_data: int = 1000
    expected_num_tokens: int = 32768
    max_num_tokens_per_sample: int = 16384
    max_num_tokens: int = 36864
    prefer_buffer_before: int = 16384
    max_buffer_size: int = 50
    max_latent_size: int = 64
    text_cond_dropout_prob: float = 0.1
    vit_cond_dropout_prob: float = 0.4
    vae_cond_dropout_prob: float = 0.1
    dataloader_type: Literal["external"] = "external"
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False

    @staticmethod
    def _raw_dataset(path: Path, task_encoder: object, worker_config: WorkerConfig) -> object:
        """Build an Energon dataset used through deterministic restore keys."""
        return get_train_dataset(
            path,
            split_part="train",
            worker_config=worker_config,
            batch_size=None,
            shuffle_buffer_size=1,
            max_samples_per_sequence=1024,
            task_encoder=task_encoder,
        )

    def build_datasets(
        self,
        context: DatasetBuildContext,
    ) -> tuple[BagelExternalLoader | None, None, None]:
        """Build one packed external loader for this DP rank."""
        if context.train_samples <= 0:
            return None, None, None
        if self.dataset_root is None or self.bagel_repo is None or self.tokenizer_model is None:
            raise ValueError("BAGEL dataset requires dataset_root, bagel_repo, and tokenizer_model")
        if context.pg_collection is None:
            raise RuntimeError("BAGEL dataset requires initialized process groups")

        root = Path(self.dataset_root)
        repo = str(Path(self.bagel_repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from modeling.qwen2 import Qwen2Tokenizer

        tokenizer = Qwen2Tokenizer.from_pretrained(self.tokenizer_model, local_files_only=True)
        tokenizer, special_tokens, _ = add_special_tokens(tokenizer)
        vae_transform = ImageTransform(image_stride=16, max_image_size=1024, min_image_size=512)
        editing_vit_transform = ImageTransform(image_stride=14, max_image_size=518, min_image_size=224)
        vlm_transform = ImageTransform(
            image_stride=14,
            max_image_size=980,
            min_image_size=378,
            max_pixels=2_007_040,
        )
        encoders = (
            BagelT2ITaskEncoder(tokenizer, vae_transform, 16),
            BagelEditingTaskEncoder(tokenizer, vae_transform, editing_vit_transform, 16, 14),
            BagelVLMTaskEncoder(tokenizer, vlm_transform, 14),
        )
        groups = ("t2i", "editing", "vlm")
        num_used_data = (
            self.t2i_num_used_data,
            self.editing_num_used_data,
            self.vlm_num_used_data,
        )
        worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)
        datasets = [
            self._raw_dataset(root / group, encoder, worker_config) for group, encoder in zip(groups, encoders)
        ]
        dp_rank = context.pg_collection.dp.rank()
        dp_size = context.pg_collection.dp.size()
        source_loaders = [
            BagelPlannedLoader(
                dataset,
                plan_manifest_indices(
                    root / group / "manifest.json",
                    seed=self.data_seed,
                    rank=dp_rank,
                    world_size=dp_size,
                    worker_id=0,
                    num_workers=1,
                    num_used_data=used,
                ),
                worker_config,
            )
            for group, used, dataset in zip(groups, num_used_data, datasets)
        ]
        packer = BagelPacker(
            source_loaders,
            [1.0, 1.0, 1.0],
            [True, False, True],
            special_tokens,
            expected_num_tokens=self.expected_num_tokens,
            max_num_tokens_per_sample=self.max_num_tokens_per_sample,
            max_num_tokens=self.max_num_tokens,
            prefer_buffer_before=self.prefer_buffer_before,
            max_buffer_size=self.max_buffer_size,
            max_latent_size=self.max_latent_size,
            text_cond_dropout_prob=self.text_cond_dropout_prob,
            vit_cond_dropout_prob=self.vit_cond_dropout_prob,
            vae_cond_dropout_prob=self.vae_cond_dropout_prob,
        )
        isolated_packer = BagelRNGIterator(packer, self.seed * dp_size + dp_rank)
        return (
            BagelExternalLoader(
                isolated_packer,
                length=context.train_samples,
                stateful_loaders=source_loaders,
            ),
            None,
            None,
        )
