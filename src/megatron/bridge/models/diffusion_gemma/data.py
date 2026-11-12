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

"""Batch collation and synthetic data for DiffusionGemma block-diffusion SFT."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider


def _as_long(value: Any) -> torch.Tensor:
    return value.long().flatten() if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)


def collate_diffusion_gemma(
    examples: Sequence[dict[str, Any]],
    *,
    pad_token_id: int,
    canvas_length: int,
    pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """Collate prompt/canvas examples into the DiffusionGemma step contract.

    Each example provides ``input_ids`` (encoder prompt, including image
    placeholder tokens) and ``canvas_ids`` (EOS-terminated target, at most
    ``canvas_length`` tokens). Optional image tensors are stacked per example.
    Prompts are left-padded to a common length that is a multiple of
    ``pad_to_multiple_of`` (use the TP size with sequence parallelism).
    """
    if not examples:
        raise ValueError("Cannot collate an empty DiffusionGemma batch")
    if pad_to_multiple_of < 1:
        raise ValueError("pad_to_multiple_of must be positive")
    prompts = [_as_long(example["input_ids"]) for example in examples]
    canvases = [_as_long(example["canvas_ids"]) for example in examples]
    if any(len(canvas) == 0 or len(canvas) > canvas_length for canvas in canvases):
        raise ValueError(f"Each canvas must contain 1..{canvas_length} tokens")
    width = max(len(prompt) for prompt in prompts)
    width += (-width) % pad_to_multiple_of
    batch_size = len(examples)
    input_ids = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
    mm_token_type_ids = torch.zeros((batch_size, width), dtype=torch.long)
    canvas_ids = torch.full((batch_size, canvas_length), pad_token_id, dtype=torch.long)
    canvas_mask = torch.zeros((batch_size, canvas_length), dtype=torch.float32)
    for row, (example, prompt, canvas) in enumerate(zip(examples, prompts, canvases, strict=True)):
        input_ids[row, width - len(prompt) :] = prompt
        attention_mask[row, width - len(prompt) :] = 1
        if example.get("mm_token_type_ids") is not None:
            mm_token_type_ids[row, width - len(prompt) :] = _as_long(example["mm_token_type_ids"])
        canvas_ids[row, : len(canvas)] = canvas
        canvas_mask[row, : len(canvas)] = 1
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": torch.arange(width)[None].expand(batch_size, -1).clone(),
        "mm_token_type_ids": mm_token_type_ids,
        "canvas_ids": canvas_ids,
        "canvas_mask": canvas_mask,
    }
    has_images = [example.get("pixel_values") is not None for example in examples]
    if any(has_images):
        if not all(has_images):
            raise ValueError("Mixed image/text micro-batches are not supported; group examples by modality")
        pixel_values = [torch.as_tensor(example["pixel_values"]) for example in examples]
        positions = [torch.as_tensor(example["image_position_ids"]) for example in examples]
        if len({tuple(value.shape) for value in pixel_values}) != 1:
            raise ValueError("Image micro-batches must share one patch grid; use micro_batch_size=1 or bucket images")
        batch["pixel_values"] = torch.cat([value.reshape(-1, *value.shape[-2:]) for value in pixel_values])
        batch["image_position_ids"] = torch.cat([value.reshape(-1, *value.shape[-2:]) for value in positions]).long()
    return batch


class _MockDiffusionGemmaDataset(Dataset):
    def __init__(self, config: "MockDiffusionGemmaDatasetConfig", size: int, offset: int):
        self.config = config
        self.size = size
        self.offset = offset

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        cfg = self.config
        generator = torch.Generator().manual_seed(cfg.seed + self.offset + int(index))
        low, high = cfg.token_range
        text = torch.randint(low, high, (cfg.prompt_length,), generator=generator)
        prompt = [cfg.bos_token_id, *text[: cfg.prompt_length // 2].tolist()]
        example: dict[str, torch.Tensor] = {}
        if cfg.image_patches_per_side:
            side = cfg.image_patches_per_side
            if side % cfg.pooling_kernel_size:
                raise ValueError("image_patches_per_side must be divisible by pooling_kernel_size")
            soft_tokens = (side // cfg.pooling_kernel_size) ** 2
            prompt += [cfg.boi_token_id, *([cfg.image_token_id] * soft_tokens), cfg.eoi_token_id]
            patch_dim = 3 * cfg.patch_size * cfg.patch_size
            example["pixel_values"] = torch.rand((1, side * side, patch_dim), generator=generator) * 2 - 1
            grid_x, grid_y = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="xy")
            example["image_position_ids"] = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)
        prompt += text[cfg.prompt_length // 2 :].tolist()
        prompt_tensor = torch.tensor(prompt, dtype=torch.long)
        example["input_ids"] = prompt_tensor
        if cfg.image_token_id is None:
            example["mm_token_type_ids"] = torch.zeros_like(prompt_tensor)
        else:
            example["mm_token_type_ids"] = (prompt_tensor == cfg.image_token_id).long()
        # A tiny fixed vocabulary of answer "programs" makes overfitting measurable.
        answer_id = int(index) % cfg.num_distinct_answers
        answer_generator = torch.Generator().manual_seed(cfg.seed + 7_919 * answer_id)
        answer = torch.randint(low, high, (cfg.completion_length - 1,), generator=answer_generator)
        example["canvas_ids"] = torch.cat((answer, torch.tensor([cfg.eos_token_id])))
        return example

    def collate_fn(self, examples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return collate_diffusion_gemma(
            examples,
            pad_token_id=self.config.pad_token_id,
            canvas_length=self.config.canvas_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )


@dataclass(kw_only=True)
class MockDiffusionGemmaDatasetConfig(DatasetProvider):
    """Synthetic, deterministic DiffusionGemma SFT data for smoke tests."""

    canvas_length: int = 256
    prompt_length: int = 32
    completion_length: int = 64
    num_distinct_answers: int = 4
    token_range: tuple[int, int] = (1_000, 20_000)
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    image_token_id: int | None = None
    boi_token_id: int | None = None
    eoi_token_id: int | None = None
    image_patches_per_side: int = 0
    patch_size: int = 16
    pooling_kernel_size: int = 3
    pad_to_multiple_of: int = 1
    seed: int = 1234
    dataloader_type: str | None = "single"
    num_workers: int = 0
    persistent_workers: bool = False

    def finalize(self) -> None:
        super().finalize()
        if not 1 <= self.completion_length <= self.canvas_length:
            raise ValueError("completion_length must be in [1, canvas_length]")
        if self.image_patches_per_side and None in (self.image_token_id, self.boi_token_id, self.eoi_token_id):
            raise ValueError("image token ids are required for synthetic image examples")

    def build_datasets(self, context: DatasetBuildContext):
        train = _MockDiffusionGemmaDataset(self, max(context.train_samples, 1), offset=0)
        valid = _MockDiffusionGemmaDataset(self, max(context.valid_samples, 1), offset=10_000_000)
        return train, valid, None


class DiffusionGemmaJSONLDataset(Dataset):
    """Load message/image prompts and completions without fitting them all on a GPU.

    Rows are ``{"messages": [...], "completion": "..."}``. Image items contain
    local ``image`` paths relative to the JSONL file. Each completion is split
    into fixed-length canvases; previous *clean* blocks are appended to the
    encoder context. The current block and future blocks never enter that context.
    """

    def __init__(self, path: str, *, processor: Any, config: "DiffusionGemmaDatasetConfig") -> None:
        self.path = Path(path)
        self.processor = processor
        self.config = config
        self.rows: list[dict[str, Any]] = []
        self.targets: list[torch.Tensor] = []
        self.blocks: list[tuple[int, int]] = []
        with self.path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("messages"), list) or not isinstance(row.get("completion"), str):
                    raise ValueError(f"{self.path}:{line_number}: expected messages list and completion string")
                if any(message.get("role") == "assistant" for message in row["messages"][-1:]):
                    raise ValueError("The prompt must end before the assistant completion")
                ids = processor.tokenizer.encode(row["completion"], add_special_tokens=False)
                eos = processor.tokenizer.eos_token_id
                if eos is None:
                    raise ValueError("DiffusionGemma completion tokenizer must have an EOS token")
                if not ids or ids[-1] != eos:
                    ids.append(eos)
                row_index = len(self.rows)
                self.rows.append(row)
                self.targets.append(torch.tensor(ids, dtype=torch.long))
                self.blocks.extend((row_index, offset) for offset in range(0, len(ids), config.canvas_length))
        if not self.blocks:
            raise ValueError(f"No training examples in {self.path}")

    def __len__(self) -> int:
        """Return the number of target blocks, not JSONL rows."""
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Process a single CPU example and expose only its prior target blocks."""
        from PIL import Image

        row_index, offset = self.blocks[index]
        messages = deepcopy(self.rows[row_index]["messages"])
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == "image":
                    image_path = item.get("image")
                    if not isinstance(image_path, str) or "://" in image_path:
                        raise ValueError("Dataset image payloads must be local file paths, not remote URLs")
                    path = Path(image_path)
                    if not path.is_absolute():
                        path = self.path.parent / path
                    with Image.open(path) as image:
                        item["image"] = image.convert("RGB")
        processed = dict(
            self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
        )
        prompt = _as_long(processed["input_ids"])
        prior = self.targets[row_index][:offset]
        prompt = torch.cat((prompt, prior))
        if prompt.numel() > self.config.max_encoder_length:
            raise ValueError(
                f"Encoder context has {prompt.numel()} tokens, exceeding max_encoder_length="
                f"{self.config.max_encoder_length}. Filter or bucket the dataset; targets are never silently truncated."
            )
        mm = processed.get("mm_token_type_ids", torch.zeros_like(processed["input_ids"]))
        result = {
            "input_ids": prompt,
            "mm_token_type_ids": torch.cat((_as_long(mm), torch.zeros_like(prior))),
            "canvas_ids": self.targets[row_index][offset : offset + self.config.canvas_length],
        }
        for key in ("pixel_values", "image_position_ids"):
            if processed.get(key) is not None:
                result[key] = processed[key]
        return result

    def collate_fn(self, examples: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Pad one CPU microbatch to the configured model-parallel multiple."""
        return collate_diffusion_gemma(
            examples,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            canvas_length=self.config.canvas_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )


@dataclass(kw_only=True)
class DiffusionGemmaDatasetConfig(DatasetProvider):
    """Serializable real-data adapter preserving explicit train/validation/test files."""

    train_path: str
    hf_processor_path: str
    validation_path: str | None = None
    test_path: str | None = None
    canvas_length: int = 256
    max_encoder_length: int = 8192
    pad_to_multiple_of: int = 1
    dataloader_type: str | None = "cyclic"
    num_workers: int = 0
    persistent_workers: bool = False

    def finalize(self) -> None:
        """Validate lengths and ensure the split files are explicitly different."""
        super().finalize()
        if self.canvas_length <= 0 or self.max_encoder_length <= 0 or self.pad_to_multiple_of <= 0:
            raise ValueError("Dataset canvas/context/padding lengths must be positive")
        if self.canvas_length % self.pad_to_multiple_of:
            raise ValueError("canvas_length must be divisible by pad_to_multiple_of")
        paths = [Path(path).resolve() for path in (self.train_path, self.validation_path, self.test_path) if path]
        if len(set(paths)) != len(paths):
            raise ValueError("Train, validation, and test must use distinct split files")

    def build_datasets(self, context: DatasetBuildContext) -> tuple[Dataset, Dataset | None, Dataset | None]:
        """Build CPU datasets from the provided split files without repartitioning."""
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.hf_processor_path)
        train = DiffusionGemmaJSONLDataset(self.train_path, processor=processor, config=self)
        validation = (
            DiffusionGemmaJSONLDataset(self.validation_path, processor=processor, config=self)
            if self.validation_path
            else None
        )
        test = DiffusionGemmaJSONLDataset(self.test_path, processor=processor, config=self) if self.test_path else None
        return train, validation, test
