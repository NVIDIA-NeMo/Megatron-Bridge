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
import hashlib
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
from megatron.energon import SavableDataLoader, WorkerConfig, get_savable_loader, get_train_dataset

from megatron.bridge.models.bagel.data.energon import (
    BagelT2IRawSample,
    BagelT2IRawTaskEncoder,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Verify BAGEL T2I Energon order and restore")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--num-rows", type=int, default=10)
    parser.add_argument("--save-after", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_loader(dataset_dir: Path) -> SavableDataLoader[BagelT2IRawSample]:
    """Build the explicit single-rank BAGEL Energon loader."""
    worker_config = WorkerConfig(rank=0, world_size=1, num_workers=1)
    dataset = get_train_dataset(
        dataset_dir,
        split_part="train",
        worker_config=worker_config,
        batch_size=None,
        shuffle_buffer_size=1,
        max_samples_per_sequence=10,
        task_encoder=BagelT2IRawTaskEncoder(),
    )
    return get_savable_loader(dataset)


def validate_sample(
    sample: BagelT2IRawSample,
    row: dict[str, object],
    *,
    parquet: Path,
    shard_name: str,
    row_group: int,
    row_index: int,
) -> dict[str, object]:
    """Validate one Energon sample and return its normalized trace entry."""
    expected_key = f"{shard_name}/t2i-{parquet.stem}-rg{row_group}-row{row_index}"
    expected_metadata = {
        "dataset_group": "t2i_pretrain",
        "dataset_name": "t2i",
        "source": {
            "parquet": parquet.name,
            "row_group": row_group,
            "row": row_index,
        },
        "captions": row["captions"],
    }
    if sample.__key__ != expected_key:
        raise ValueError(f"expected key {expected_key}, found {sample.__key__}")
    if sample.metadata != expected_metadata:
        raise ValueError(f"metadata differs for source row {row_index}")
    if sample.image != row["image"]:
        raise ValueError(f"image bytes differ for source row {row_index}")
    return {
        "key": sample.__key__,
        "restore_key": sample.__restore_key__,
        "image_sha256": hashlib.sha256(sample.image).hexdigest(),
        "metadata": sample.metadata,
    }


def main() -> None:
    """Validate physical order and exact save/restore suffix reproduction."""
    args = parse_args()
    if not 0 < args.save_after < args.num_rows:
        raise ValueError("save-after must be between zero and num-rows")

    table = pq.ParquetFile(args.parquet).read_row_group(args.row_group, columns=["image", "captions"])
    rows = table.slice(args.start_row, args.num_rows).to_pylist()
    if len(rows) != args.num_rows:
        raise ValueError(f"requested {args.num_rows} rows, found {len(rows)}")
    shard_name = next(args.dataset_dir.glob("*.tar")).name

    loader = build_loader(args.dataset_dir)
    iterator = iter(loader)
    full_trace = []
    state = None
    for offset, row in enumerate(rows):
        full_trace.append(
            validate_sample(
                next(iterator),
                row,
                parquet=args.parquet,
                shard_name=shard_name,
                row_group=args.row_group,
                row_index=args.start_row + offset,
            )
        )
        if offset + 1 == args.save_after:
            state = loader.save_state_rank()
    if state is None:
        raise RuntimeError("Energon did not return loader state")
    suffix = full_trace[args.save_after :]
    del iterator
    del loader

    restored_loader = build_loader(args.dataset_dir)
    restored_loader.restore_state_rank(state)
    restored_iterator = iter(restored_loader)
    restored_suffix = [
        validate_sample(
            next(restored_iterator),
            rows[offset],
            parquet=args.parquet,
            shard_name=shard_name,
            row_group=args.row_group,
            row_index=args.start_row + offset,
        )
        for offset in range(args.save_after, args.num_rows)
    ]
    if restored_suffix != suffix:
        raise ValueError("restored Energon suffix differs from the original suffix")

    trace = {
        "save_after": args.save_after,
        "full": full_trace,
        "suffix": suffix,
        "restored_suffix": restored_suffix,
    }
    args.output.write_text(
        json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    logger.info("Verified %d samples and %d restored samples", len(full_trace), len(restored_suffix))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
