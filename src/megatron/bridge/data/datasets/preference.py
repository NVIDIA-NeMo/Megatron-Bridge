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

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from megatron.core.msc_utils import MultiStorageClientFeature
from torch.utils.data import DataLoader, Dataset

from megatron.bridge.data.collators.sequence_padding import _ceil_to_multiple
from megatron.bridge.data.samplers import build_pretraining_data_loader


SCORING_METADATA_FILENAME = "scoring_metadata.json"

REF_LOGPROBS_FILENAME = "ref_logprobs.jsonl"


@dataclass(frozen=True)
class ScoringFingerprint:
    """Scoring-run inputs a training run must reproduce; written to and checked against ``scoring_metadata.json``."""

    dataset: str
    split: str | None
    tokenizer: str
    max_seq_length: int
    prompt_key: str | None
    tensor_model_parallel_size: int
    sequence_parallel: bool


def _open_path(path: str, mode: str):
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.open(path, mode)
    if "w" in mode:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, mode)


REF_LOGPROB_COLUMNS: tuple[str, ...] = (
    "ref_chosen_logprob_sum",
    "ref_chosen_num_tokens",
    "ref_rejected_logprob_sum",
    "ref_rejected_num_tokens",
)


def ref_logprobs_from_rows(rows: Iterable[Mapping[str, Any]], expected_num_pairs: int) -> dict[int, dict[str, Any]]:
    """Validate offline-scorer output rows and key them by ``pair_id``."""
    mapping: dict[int, dict[str, Any]] = {}
    for row in rows:
        missing = [col for col in ("pair_id",) + REF_LOGPROB_COLUMNS if col not in row]
        if missing:
            raise ValueError(f"Ref-logprob row is missing columns {missing}; present keys: {sorted(row)}.")
        pair_id = int(row["pair_id"])
        if pair_id in mapping:
            raise ValueError(f"Ref-logprob artifact has duplicate pair_id={pair_id}.")
        mapping[pair_id] = {col: row[col] for col in REF_LOGPROB_COLUMNS}

    expected_ids = set(range(expected_num_pairs))
    if mapping.keys() != expected_ids:
        missing_ids = sorted(expected_ids - mapping.keys())
        unexpected_ids = sorted(mapping.keys() - expected_ids)
        raise ValueError(
            f"Ref-logprob artifact must cover pair_id 0..{expected_num_pairs - 1} exactly; "
            f"missing {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}, "
            f"unexpected {unexpected_ids[:10]}{'...' if len(unexpected_ids) > 10 else ''}. "
            "Was the artifact scored against this dataset?"
        )
    return mapping


def _artifact_path(artifact_dir: str, filename: str) -> str:
    return f"{artifact_dir.rstrip('/')}/{filename}"


def write_ref_logprobs(records: Sequence[Mapping[str, Any]], artifact_dir: str) -> None:
    """Write the ref-logprob rows as JSONL to ``artifact_dir`` (local or msc://)."""
    path = _artifact_path(artifact_dir, REF_LOGPROBS_FILENAME)
    with _open_path(path, "w") as f:
        for record in records:
            f.write(json.dumps(dict(record)) + "\n")


def load_ref_logprobs(path: str, expected_num_pairs: int) -> dict[int, dict[str, Any]]:
    """Load and validate a ref-scorer artifact from ``path`` (local or msc://)."""
    with _open_path(_artifact_path(path, REF_LOGPROBS_FILENAME), "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return ref_logprobs_from_rows(rows, expected_num_pairs)


def write_scoring_metadata(metadata: Mapping[str, Any], artifact_dir: str) -> None:
    """Write ``scoring_metadata.json`` to ``artifact_dir`` (local or msc://)."""
    path = _artifact_path(artifact_dir, SCORING_METADATA_FILENAME)
    with _open_path(path, "w") as f:
        json.dump(dict(metadata), f, indent=2)


def read_scoring_metadata(artifact_dir: str) -> dict[str, Any]:
    """Load the artifact's ``scoring_metadata.json`` (local or msc://)."""
    path = _artifact_path(artifact_dir, SCORING_METADATA_FILENAME)
    try:
        with _open_path(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(
            f"No {SCORING_METADATA_FILENAME} in {artifact_dir} — is this a scorer artifact "
            "(written by score_reference_logprobs.py)?"
        ) from None


def validate_scoring_metadata(artifact_dir: str, expected: ScoringFingerprint) -> dict[str, Any]:
    """Raise if the artifact was scored under different inputs than ``expected``; return its metadata."""
    metadata = read_scoring_metadata(artifact_dir)
    mismatched = {
        key: (metadata.get(key), value) for key, value in asdict(expected).items() if metadata.get(key) != value
    }
    if mismatched:
        raise ValueError(
            f"scoring_metadata.json mismatch, artifact vs this run: {mismatched}. "
            "Re-score the reference logprobs or fix the training config — training against a "
            "differently-scored artifact silently corrupts every margin."
        )
    return metadata


def pair_token_lengths(dataset: Dataset) -> list[int]:
    """Per-pair padded row length (max of the two sides), one full pass over ``dataset``."""
    return [
        max(len(item["chosen_input_ids"]), len(item["rejected_input_ids"]))
        for item in (dataset[i] for i in range(len(dataset)))
    ]


def pack_pairs_by_token_budget(
    lengths: Sequence[int],
    *,
    budget_tokens: int,
    max_pairs_per_batch: int = 64,
) -> list[list[int]]:
    """Deterministic longest-first batches under ``budget_tokens``; an over-budget pair gets a singleton."""
    if budget_tokens < 1:
        raise ValueError(f"budget_tokens must be positive, got {budget_tokens}.")
    if max_pairs_per_batch < 1:
        raise ValueError(f"max_pairs_per_batch must be positive, got {max_pairs_per_batch}.")

    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches: list[list[int]] = []
    batch: list[int] = []
    for idx in order:
        # Descending order makes the first item the batch max.
        batch_max = lengths[batch[0]] if batch else lengths[idx]
        if batch and (len(batch) == max_pairs_per_batch or 2 * (len(batch) + 1) * batch_max > budget_tokens):
            batches.append(batch)
            batch = []
        batch.append(idx)
    if batch:
        batches.append(batch)
    return batches


def preference_collate_fn(
    batch: list[Mapping[str, Any]],
    pad_token_id: int,
    pad_seq_length_to_mult: int = 1,
    require_ref_logprobs: bool = True,
) -> dict[str, Any]:
    """Collate pair records into an interleaved row batch (chosen@even, rejected@odd)."""
    input_ids: list[Sequence[int]] = []
    context_lens: list[int] = []
    pair_ids: list[int] = []
    loss_multipliers: list[float] = []
    ref_sums: list[float] = []
    ref_counts: list[int] = []

    for record in batch:
        input_ids.append(record["chosen_input_ids"])
        input_ids.append(record["rejected_input_ids"])
        context_lens.append(int(record["chosen_context_len"]))
        context_lens.append(int(record["rejected_context_len"]))
        pair_ids.extend([int(record["pair_id"])] * 2)
        loss_multipliers.extend([float(record.get("loss_multiplier", 1.0))] * 2)
        if require_ref_logprobs:
            ref_sums.append(float(record["ref_chosen_logprob_sum"]))
            ref_sums.append(float(record["ref_rejected_logprob_sum"]))
            ref_counts.append(int(record["ref_chosen_num_tokens"]))
            ref_counts.append(int(record["ref_rejected_num_tokens"]))

    for ids, ctx_len, pair_id in zip(input_ids, context_lens, pair_ids):
        if not 1 <= ctx_len < len(ids):
            raise ValueError(
                f"Pair {pair_id}: context_len={ctx_len} must leave at least one completion "
                f"token in a sequence of length {len(ids)}. The prep script should have "
                "dropped this pair."
            )

    num_rows = len(input_ids)
    # Row length after the tokens/labels shift.
    seq_lens = torch.tensor([len(ids) - 1 for ids in input_ids], dtype=torch.long)
    max_length = _ceil_to_multiple(int(seq_lens.max()), pad_seq_length_to_mult)

    # One padded id tensor; tokens and labels are its two shifted views.
    full = torch.full((num_rows, max_length + 1), pad_token_id, dtype=torch.long)
    for i, ids in enumerate(input_ids):
        full[i, : len(ids)] = torch.as_tensor(ids, dtype=torch.long)

    positions = torch.arange(max_length, dtype=torch.long)
    valid = positions < seq_lens.unsqueeze(1)
    # Label position j predicts original token j+1, so completion labels start at ctx_len - 1.
    completion_start = (torch.tensor(context_lens, dtype=torch.long) - 1).unsqueeze(1)

    collated: dict[str, Any] = {
        "tokens": full[:, :-1].masked_fill(~valid, pad_token_id),
        "labels": full[:, 1:],
        "loss_mask": (valid & (positions >= completion_start)).long(),
        "position_ids": positions.unsqueeze(0).expand(num_rows, -1).contiguous(),
        "attention_mask": None,
        "pair_id": torch.tensor(pair_ids, dtype=torch.long),
        "loss_multiplier": torch.tensor(loss_multipliers, dtype=torch.float32),
    }
    if require_ref_logprobs:
        collated["ref_logprob_sum"] = torch.tensor(ref_sums, dtype=torch.float32)
        collated["ref_num_tokens"] = torch.tensor(ref_counts, dtype=torch.long)
    return collated


def build_preference_data_loader(
    dataset: Dataset,
    micro_batch_size: int,
    global_batch_size: int,
    data_parallel_rank: int,
    data_parallel_size: int,
    consumed_samples: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    worker_init_fn: Callable[[int], None] | None = None,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader:
    """Build a DPO loader from row-denominated sizes (one pair == two rows), halved here for the stock loader."""
    if micro_batch_size % 2 != 0 or global_batch_size % 2 != 0:
        raise ValueError(
            f"DPO batch sizes are row-denominated and must be even (one pair == two rows); "
            f"got micro_batch_size={micro_batch_size}, global_batch_size={global_batch_size}."
        )
    if consumed_samples % 2 != 0:
        raise ValueError(f"consumed_samples must be even for DPO, got {consumed_samples}.")
    # The sampler checks this too, but reports the halved pair counts, which do not
    # match the row-denominated numbers the user configured.
    if global_batch_size % (micro_batch_size * data_parallel_size) != 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} must be divisible by "
            f"micro_batch_size*data_parallel_size={micro_batch_size * data_parallel_size}."
        )

    return build_pretraining_data_loader(
        dataset,
        consumed_samples=consumed_samples // 2,
        dataloader_type="batch",
        micro_batch_size=micro_batch_size // 2,
        global_batch_size=global_batch_size // 2,
        num_workers=num_workers,
        data_sharding=False,  # unused by the 'batch' sampler
        worker_init_fn=worker_init_fn,
        collate_fn=dataset.collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        drop_last=True,
        shuffle=shuffle,
        seed=seed,
    )
