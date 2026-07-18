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

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    """Parse oracle arguments."""
    parser = argparse.ArgumentParser(description="Dump BAGEL official dataloader order")
    parser.add_argument("--bagel-repo", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def to_jsonable(value: object) -> object:
    """Recursively convert supported values to JSON-compatible objects."""
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def configure_official_data(data_root: Path, dataset_meta: dict, dataset_info: dict) -> None:
    """Point BAGEL's in-memory metadata at the downloaded official sample."""
    editing_dir = data_root / "editing" / "seedxedit_multi"
    parquet_info_path = data_root / "editing" / "parquet_info" / "seedxedit_multi.json"

    dataset_info["t2i_pretrain"]["t2i"]["data_dir"] = str(data_root / "t2i")
    dataset_info["unified_edit"]["seedxedit_multi"]["data_dir"] = str(editing_dir)
    dataset_info["unified_edit"]["seedxedit_multi"]["parquet_info_path"] = str(parquet_info_path)
    dataset_info["vlm_sft"]["llava_ov"]["data_dir"] = str(data_root / "vlm" / "images")
    dataset_info["vlm_sft"]["llava_ov"]["jsonl_path"] = str(data_root / "vlm" / "llava_ov_si.jsonl")

    with parquet_info_path.open(encoding="utf-8") as stream:
        parquet_info = json.load(stream)
    dataset_meta["unified_edit"]["parquet_info"] = {
        str(editing_dir / Path(key).name): value for key, value in parquet_info.items()
    }


def main() -> None:
    """Run the official loader and write its first batches as JSONL."""
    args = parse_args()
    sys.path.insert(0, str(args.bagel_repo))

    from data.data_utils import add_special_tokens
    from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
    from data.dataset_info import DATASET_INFO
    from modeling.qwen2 import Qwen2Tokenizer
    from transformers import set_seed

    set_seed(args.seed)
    with args.dataset_config.open(encoding="utf-8") as stream:
        dataset_meta = yaml.safe_load(stream)
    configure_official_data(args.data_root, dataset_meta, DATASET_INFO)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer_model, local_files_only=True)
    tokenizer, special_tokens, _ = add_special_tokens(tokenizer)
    dataset = PackedDataset(
        DataConfig(grouped_datasets=dataset_meta),
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        local_rank=0,
        world_size=1,
        num_workers=1,
    )
    dataset.set_epoch(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_wrapper(),
        num_workers=0,
    )

    with args.output.open("w", encoding="utf-8") as stream:
        for step, batch in zip(range(args.num_batches), loader):
            record = {
                "step": step,
                "batch_data_indexes": batch.batch_data_indexes,
                "sequence_length": batch.sequence_length,
                "sample_lens": batch.sample_lens,
            }
            stream.write(json.dumps(to_jsonable(record), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
