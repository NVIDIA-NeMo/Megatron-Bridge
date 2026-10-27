#!/usr/bin/env python3
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

"""Select real trajectories by effective training length.

Natural retains every trainable row unchanged. Mixed selects requested counts from three length
buckets, using complete trajectories first and real prefixes ending at assistant-message boundaries
when short or medium rows are scarce. Each source contributes at most one row and stays in its
original split.

Adapted from the Megatron-LM agentic SFT example by Zhongbo Zhu.
"""

import argparse
import hashlib
import json
import logging
import math
import multiprocessing
import os
import statistics
import sys
import tempfile
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))

from agentic_sft_data import ChatEncoder, JsonlRows, atomic_json, sha256_file, truncate_chat  # noqa: E402


DEFAULT_WORKSPACE = Path("agentic_sft_workspace")
LOGGER = logging.getLogger(__name__)
SPLITS = ("training", "validation")
MIXED_STRATEGY = "assistant_prefix_v2"
DEFAULT_BUCKET_RATIOS = (0.5, 0.3, 0.2)
MAX_PREFIX_ENCODINGS = 16
_ROWS = None
_ENCODER = None
_TOKENIZER_PATH = None
_SEQUENCE_LENGTH = None
_LOSS_MODE = None


def length_bucket(length, sequence_length):
    """Minimum CP1/2/4 capacity bucket for a post-shift, unpadded length."""
    if sequence_length < 32 or sequence_length % 32:
        raise ValueError("Sequence length must be a positive multiple of 32")
    if not 1 <= length <= sequence_length:
        raise ValueError("Runtime length must be between 1 and sequence length")
    return 0 if length <= sequence_length // 4 else 1 if length <= sequence_length // 2 else 2


def stable_priority(seed, identifier, purpose="selection"):
    """Return a seeded, reproducible sort key for an identifier and purpose."""
    payload = json.dumps([seed, identifier, purpose], ensure_ascii=False).encode()
    return hashlib.sha256(payload).digest(), identifier


def bucket_quotas(sample_count, ratios):
    """Allocate an exact total with largest remainders; bucket order breaks ties."""
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("Mixed sample counts must be positive integers")
    if (
        len(ratios) != 3
        or any(not math.isfinite(value) or value < 0 for value in ratios)
        or not math.isclose(sum(ratios), 1.0, abs_tol=1e-8)
    ):
        raise ValueError("Length bucket ratios must be three nonnegative finite values summing to 1")
    raw = [sample_count * value / sum(ratios) for value in ratios]
    counts = [math.floor(value) for value in raw]
    for bucket in sorted(range(3), key=lambda i: (-(raw[i] - counts[i]), i))[: sample_count - sum(counts)]:
        counts[bucket] += 1
    return counts


def select_indices(trajectory_ids, lengths, distribution, seed, sequence_length):
    """Natural selection preserves the source order; mixed requires prefix search."""
    if len(trajectory_ids) != len(lengths) or len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("IDs and lengths must have equal sizes and unique trajectory IDs")
    for length in lengths:
        length_bucket(length, sequence_length)
    if distribution != "natural":
        raise ValueError("Use select_mixed_records with explicit quotas for mixed selection")
    return list(range(len(lengths)))


def prefix_target(seed, identifier, bucket, sequence_length):
    """Pick a reproducible target length inside a short or medium bucket."""
    upper = (sequence_length // 4, sequence_length // 2)[bucket]
    lower = 1 if bucket == 0 else sequence_length // 4 + 1
    digest, _ = stable_priority(seed, identifier, f"prefix-target-{bucket}")
    return lower + int.from_bytes(digest[:8], "big") % (upper - lower + 1)


def select_mixed_records(records, quotas, seed, sequence_length, resolve_prefixes, batch_size=128):
    """Choose unique parents, preferring whole rows; resolve prefixes in bounded batches."""
    eligible = [record for record in records if "rejected_reason" not in record]
    if len(eligible) < sum(quotas):
        raise ValueError(
            f"Mixed needs {sum(quotas)} unique trainable source trajectories; found {len(eligible)}. "
            "Increase --percentage or lower the requested sample count; no samples are repeated."
        )
    ordered = sorted(eligible, key=lambda record: stable_priority(seed, record["trajectory_id"]))
    selected, used = [], set()
    counts = [0, 0, 0]
    for record in ordered:
        bucket = length_bucket(record["runtime_tokens"], sequence_length)
        if counts[bucket] < quotas[bucket]:
            chosen = dict(record, is_prefix=False, source_id=record["trajectory_id"])
            chosen["source_original_runtime_tokens"] = record["original_runtime_tokens"]
            selected.append(chosen)
            used.add(record["trajectory_id"])
            counts[bucket] += 1
    if counts[2] != quotas[2]:
        raise ValueError(f"Mixed long bucket shortage: requested {quotas}, found {counts}")
    for bucket in range(3):
        if counts[bucket] == quotas[bucket]:
            continue
        if bucket == 2:
            raise ValueError(f"Mixed long bucket shortage: requested {quotas}, found {counts}")
        upper = (sequence_length // 4, sequence_length // 2)[bucket]
        candidates = [
            record
            for record in ordered
            if record["trajectory_id"] not in used and record["original_runtime_tokens"] > upper
        ]
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            tasks = [
                (
                    record["row_index"],
                    bucket,
                    prefix_target(seed, record["trajectory_id"], bucket, sequence_length),
                )
                for record in batch
            ]
            results = resolve_prefixes(tasks)
            if len(results) != len(batch):
                raise ValueError("Prefix worker returned an incomplete batch")
            for source, result in zip(batch, results):
                if result is None or counts[bucket] == quotas[bucket]:
                    continue
                if (
                    result["trajectory_id"] != source["trajectory_id"]
                    or result["row_index"] != source["row_index"]
                    or length_bucket(result["runtime_tokens"], sequence_length) != bucket
                    or result["runtime_tokens"] != result["original_runtime_tokens"]
                    or result["supervised_tokens"] < 1
                    or not result["is_prefix"]
                ):
                    raise ValueError("Invalid verified prefix measurement")
                result = dict(result, source_id=source["trajectory_id"])
                result["source_original_runtime_tokens"] = source["original_runtime_tokens"]
                selected.append(result)
                used.add(source["trajectory_id"])
                counts[bucket] += 1
            LOGGER.info("Mixed prefix selection: requested %s, selected %s", quotas, counts)
            if counts[bucket] == quotas[bucket]:
                break
        if counts[bucket] != quotas[bucket]:
            raise ValueError(
                f"Mixed bounded prefix search shortage: requested {quotas}, found {counts}. "
                "Increase --percentage or lower the requested counts; no samples are repeated."
            )
    if len(used) != sum(quotas):
        raise ValueError("Mixed selection reused a source trajectory")
    return sorted(selected, key=lambda record: stable_priority(seed, record["trajectory_id"], "order"))


def validate_unique_ids(splits):
    """Reject duplicate IDs within a split and accidental train/validation overlap."""
    seen = set()
    for split, records in splits.items():
        for record in records:
            identifier = record["trajectory_id"]
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError(f"Missing or duplicate trajectory ID in {split}")
            seen.add(identifier)


def summarize_lengths(records, sequence_length):
    """Summarize runtime and supervised token counts, overall and per CP-capacity bucket."""
    lengths = [record["runtime_tokens"] for record in records]
    buckets = [
        [record for record in records if length_bucket(record["runtime_tokens"], sequence_length) == bucket]
        for bucket in range(3)
    ]
    total = sum(lengths)
    return {
        "samples": len(records),
        "runtime_tokens": total,
        "supervised_tokens": sum(record["supervised_tokens"] for record in records),
        "original_runtime_tokens": sum(record["original_runtime_tokens"] for record in records),
        "truncated_samples": sum(record["original_runtime_tokens"] > sequence_length for record in records),
        "min": min(lengths),
        "median": statistics.median(lengths),
        "mean": statistics.mean(lengths),
        "max": max(lengths),
        "coefficient_of_variation": statistics.pstdev(lengths) / statistics.mean(lengths),
        "buckets": [
            {
                "minimum_cp_size": cp,
                "samples": len(bucket),
                "sample_fraction": len(bucket) / len(records),
                "runtime_tokens": sum(record["runtime_tokens"] for record in bucket),
                "token_fraction": sum(record["runtime_tokens"] for record in bucket) / total,
            }
            for cp, bucket in zip((1, 2, 4), buckets)
        ],
    }


def _initialize_worker(rows, tokenizer_path, sequence_length, loss_mode="assistant"):
    global _ROWS, _ENCODER, _TOKENIZER_PATH, _SEQUENCE_LENGTH, _LOSS_MODE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    _ROWS, _TOKENIZER_PATH, _SEQUENCE_LENGTH = rows, tokenizer_path, sequence_length
    _ENCODER = None
    _LOSS_MODE = loss_mode


def rejection_reason(error, sequence_length):
    """Recognize data-contract failures; never turn unexpected errors into skips."""
    if not isinstance(error, ValueError):
        return None
    message = str(error)
    if message == "No next-token assistant targets in this conversation":
        return "no_assistant_targets"
    if message == (
        f"No assistant targets remain after truncation to {sequence_length}; "
        "increase --seq-length or filter the source trajectory"
    ):
        return "no_assistant_targets_after_truncation"
    return None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = ChatEncoder(_TOKENIZER_PATH, _LOSS_MODE)
    return _ENCODER


def _measure_row(index):
    row = _ROWS[index]
    try:
        tokens, targets = _encoder().encode(row)
        original_length = len(tokens) - 1
        tokens, targets = truncate_chat(tokens, targets, _SEQUENCE_LENGTH, _LOSS_MODE)
    except Exception as error:
        reason = rejection_reason(error, _SEQUENCE_LENGTH)
        if reason is not None:
            return {
                "trajectory_id": row["trajectory_id"],
                "row_index": index,
                "source_line": int(_ROWS.line_numbers[index]),
                "rejected_reason": reason,
            }
        raise ValueError(f"Length measurement failed at {_ROWS.location(index)}") from error
    return {
        "trajectory_id": row["trajectory_id"],
        "row_index": index,
        "original_runtime_tokens": original_length,
        "runtime_tokens": len(tokens) - 1,
        "supervised_tokens": int((targets[1:] != -100).sum()),
    }


def find_prefix(row, bucket, target, sequence_length, encode):
    """Search assistant boundaries, then verify a prefix by actual template encoding.

    Binary search is a candidate heuristic, not an assumption about all templates.
    At most 16 distinct prefixes are encoded per source/bucket. Every accepted result
    is independently checked against its bucket; failure means this bounded search
    found no usable prefix, not a claim that no possible boundary exists.
    """
    upper = (sequence_length // 4, sequence_length // 2)[bucket]
    lower = 1 if bucket == 0 else sequence_length // 4 + 1
    messages = row["messages"]
    endings = [index for index, message in enumerate(messages[:-1]) if message["role"] == "assistant"]
    if not endings:
        return None
    measured = {}

    def evaluate(position):
        if position in measured:
            return measured[position]
        if len(measured) >= MAX_PREFIX_ENCODINGS:
            return None
        end = endings[position]
        try:
            tokens, targets = encode(messages[: end + 1])
        except ValueError as error:
            if rejection_reason(error, sequence_length) is not None:
                measured[position] = None
                return None
            raise
        length = len(tokens) - 1
        supervised = sum(int(value != -100) for value in targets[1:])
        result = {
            "runtime_tokens": length,
            "original_runtime_tokens": length,
            "supervised_tokens": supervised,
            "end_message_index": end,
            "is_prefix": True,
            "target_runtime_tokens": target,
        }
        measured[position] = result
        return result

    # Try the earliest complete response and then bracket the dispersed target.
    evaluate(0)
    left, right = 0, len(endings) - 1
    while left <= right and len(measured) < MAX_PREFIX_ENCODINGS - 4:
        middle = (left + right) // 2
        result = evaluate(middle)
        if result is None or result["runtime_tokens"] < target:
            left = middle + 1
        else:
            right = middle - 1
    for position in (left, right, left + 1, right - 1, len(endings) - 1):
        if 0 <= position < len(endings):
            evaluate(position)
    valid = [
        result
        for result in measured.values()
        if result is not None and lower <= result["runtime_tokens"] <= upper and result["supervised_tokens"] > 0
    ]
    if not valid:
        return None
    return min(
        valid,
        key=lambda result: (abs(result["runtime_tokens"] - target), result["end_message_index"]),
    )


def _measure_prefix(task):
    index, bucket, target = task
    row = _ROWS[index]
    encoder = _encoder()

    def encode(messages):
        return encoder.encode(row, messages)

    try:
        result = find_prefix(row, bucket, target, _SEQUENCE_LENGTH, encode)
    except Exception as error:
        raise ValueError(f"Prefix measurement failed at {_ROWS.location(index)}") from error
    if result is not None:
        result.update(trajectory_id=row["trajectory_id"], row_index=index)
    return result


def build_mixed_selection(
    data_dir,
    tokenizer_path,
    index,
    sequence_length,
    num_workers,
    loss_mode,
    seed,
    split_counts,
    ratios,
):
    """Cache verified prefix measurements with all selection and encoding inputs."""
    policy = {
        "strategy": MIXED_STRATEGY,
        "seed": seed,
        "split_counts": split_counts,
        "ratios": list(ratios),
        "max_prefix_encodings": MAX_PREFIX_ENCODINGS,
    }
    policy_hash = hashlib.sha256(
        json.dumps({"policy": policy, "fingerprints": index["fingerprints"]}, sort_keys=True).encode()
    ).hexdigest()[:16]
    cache_path = data_dir / f"prefixes_{loss_mode}_{sequence_length}_{policy_hash}.json"
    cache = {"fingerprints": index["fingerprints"], "policy": policy, "splits": {}}
    if cache_path.is_file():
        previous = json.loads(cache_path.read_text())
        if previous.get("fingerprints") == cache["fingerprints"] and previous.get("policy") == policy:
            cache = previous
    selections = {}
    for split in SPLITS:
        rows = JsonlRows(data_dir / f"{split}.jsonl")
        worker_args = (rows, tokenizer_path, sequence_length, loss_mode)
        context = multiprocessing.get_context("spawn")
        pool = None
        if num_workers == 1:
            _initialize_worker(*worker_args)
        else:
            pool = context.Pool(min(num_workers, len(rows)), initializer=_initialize_worker, initargs=worker_args)
        measurements = cache["splits"].setdefault(split, {})

        def resolve(tasks):
            missing = [task for task in tasks if ":".join(map(str, task)) not in measurements]
            if missing:
                results = (
                    pool.imap(_measure_prefix, missing, chunksize=1)
                    if pool is not None
                    else map(_measure_prefix, missing)
                )
                for task, result in zip(missing, results):
                    measurements[":".join(map(str, task))] = result
                atomic_json(cache_path, cache)
            return [measurements[":".join(map(str, task))] for task in tasks]

        try:
            selections[split] = select_mixed_records(
                index["splits"][split],
                bucket_quotas(split_counts[split], ratios),
                seed,
                sequence_length,
                resolve,
                batch_size=max(32, num_workers * 4),
            )
        except BaseException as error:
            if pool is not None:
                pool.terminate()
            if isinstance(error, ValueError):
                raise ValueError(f"{split}: {error}") from error
            raise
        else:
            if pool is not None:
                pool.close()
        finally:
            if pool is not None:
                pool.join()
            rows.close()
    if input_fingerprints(data_dir, tokenizer_path, sequence_length, loss_mode) != index["fingerprints"]:
        raise RuntimeError("Inputs or tokenization code changed during prefix measurement")
    atomic_json(cache_path, cache)
    return selections, cache_path


def input_fingerprints(data_dir, tokenizer_path, sequence_length, loss_mode="assistant"):
    """Hash every input that determines a measurement: data, tokenizer, code, library versions."""
    paths = ("02_5_filter_coderforge.py", "agentic_sft_data.py")
    versions = {}
    for package in ("transformers", "tokenizers", "jinja2", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "inputs": {split: sha256_file(data_dir / f"{split}.jsonl") for split in SPLITS},
        "tokenizer_files": {
            str(path.relative_to(tokenizer_path)): sha256_file(path)
            for path in sorted(tokenizer_path.rglob("*"))
            if path.is_file()
        },
        "source_files": {path: sha256_file(EXAMPLE_DIR / path) for path in paths},
        "sequence_length": sequence_length,
        "mask_source": "chat_template_generation",
        "loss_mode": loss_mode,
        "runtime_versions": versions,
    }


def load_length_index(data_dir, tokenizer_path, sequence_length, num_workers, loss_mode="assistant"):
    """Reuse measurements only when data, tokenizer, length policy and code match."""
    fingerprints = input_fingerprints(data_dir, tokenizer_path, sequence_length, loss_mode)
    fingerprint_hash = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = data_dir / f"lengths_{loss_mode}_{sequence_length}_{fingerprint_hash}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("fingerprints") == fingerprints:
            validate_unique_ids(cached["splits"])
            LOGGER.info("Reusing length index: %s", cache_path)
            return cached, cache_path
    measured = {}
    for split in SPLITS:
        rows = JsonlRows(data_dir / f"{split}.jsonl")
        LOGGER.info("Measuring %s: %d trajectories, %d workers", split, len(rows), num_workers)
        worker_args = (rows, tokenizer_path, sequence_length, loss_mode)
        if num_workers == 1:
            _initialize_worker(*worker_args)
            measured[split] = list(map(_measure_row, range(len(rows))))
        else:
            context = multiprocessing.get_context("spawn")
            with context.Pool(
                min(num_workers, len(rows)), initializer=_initialize_worker, initargs=worker_args
            ) as pool:
                measured[split] = []
                for record in pool.imap(_measure_row, range(len(rows)), chunksize=4):
                    measured[split].append(record)
                    if len(measured[split]) % 1000 == 0:
                        LOGGER.info("Measured %s %d/%d", split, len(measured[split]), len(rows))
        rows.close()
    validate_unique_ids(measured)
    if input_fingerprints(data_dir, tokenizer_path, sequence_length, loss_mode) != fingerprints:
        raise RuntimeError("Inputs or tokenization code changed during length measurement")
    cached = {"fingerprints": fingerprints, "splits": measured}
    atomic_json(cache_path, cached)
    return cached, cache_path


def write_selection(data_dir, output_dir, index, distribution, seed, sequence_length, mixed_selections=None):
    """Validate both splits before publishing; keep natural rows byte-identical."""
    validate_unique_ids(index["splits"])
    selections = {}
    for split, records in index["splits"].items():
        valid = [record for record in records if "rejected_reason" not in record]
        if not valid:
            raise ValueError(f"{split}: no trainable trajectories remain")
        if distribution == "natural":
            selections[split] = [
                dict(
                    record,
                    is_prefix=False,
                    source_id=record["trajectory_id"],
                    source_original_runtime_tokens=record["original_runtime_tokens"],
                )
                for record in valid
            ]
        elif distribution == "mixed" and mixed_selections is not None:
            selections[split] = [dict(record) for record in mixed_selections[split]]
        else:
            raise ValueError("Mixed requires verified prefix selections")
    validate_unique_ids(selections)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with tempfile.TemporaryDirectory(prefix=".filter_", dir=output_dir) as temporary:
        for split in SPLITS:
            selected = {record["row_index"]: record for record in selections[split]}
            if len(selected) != len(selections[split]):
                raise ValueError(f"{split}: duplicate source row in selection")
            records = index["splits"][split]
            destination = Path(temporary) / f"{split}.jsonl"
            # Keep only byte offsets in memory; natural may contain many GB of text.
            selected_offsets = {}
            row_index = 0
            with (
                (data_dir / f"{split}.jsonl").open("rb") as source,
                destination.open("wb") as stream,
            ):
                while True:
                    offset = source.tell()
                    line = source.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    if row_index in selected:
                        selected_offsets[row_index] = offset
                    row_index += 1
                if row_index != len(records) or set(selected_offsets) != set(selected):
                    raise ValueError(f"{split}: source row count changed after measurement")
                for measurement in selections[split]:
                    source.seek(selected_offsets[measurement["row_index"]])
                    line = source.readline()
                    row = json.loads(line)
                    if row["trajectory_id"] != measurement["source_id"]:
                        raise ValueError(f"{split}: selected source identity mismatch")
                    end = measurement.get("end_message_index", len(row["messages"]) - 1)
                    if measurement["is_prefix"]:
                        if not 0 <= end < len(row["messages"]) - 1 or row["messages"][end]["role"] != "assistant":
                            raise ValueError("Prefix must end at a complete assistant message")
                        row["messages"] = row["messages"][: end + 1]
                    measurement["end_message_index"] = end
                    if distribution == "mixed":
                        if "selection_provenance" in row:
                            raise ValueError("Input already contains selection_provenance")
                        row["selection_provenance"] = {
                            "strategy": MIXED_STRATEGY,
                            **{
                                key: measurement[key]
                                for key in (
                                    "source_id",
                                    "is_prefix",
                                    "end_message_index",
                                    "source_original_runtime_tokens",
                                    "original_runtime_tokens",
                                    "runtime_tokens",
                                    "supervised_tokens",
                                )
                            },
                        }
                        line = (json.dumps(row, ensure_ascii=False) + "\n").encode()
                    stream.write(line if line.endswith(b"\n") else line + b"\n")
            outputs[split] = {
                "file": destination.name,
                "sha256": sha256_file(destination),
                "rows": len(selected),
                "candidate_rows": len(records),
                "prefix_rows": sum(record["is_prefix"] for record in selections[split]),
                "rejected_rows": [
                    {
                        "trajectory_id": record["trajectory_id"],
                        "source_line": record.get("source_line", i + 1),
                        "reason": record["rejected_reason"],
                    }
                    for i, record in enumerate(records)
                    if "rejected_reason" in record
                ],
                "rejected_rows_by_reason": dict(
                    Counter(record["rejected_reason"] for record in records if "rejected_reason" in record)
                ),
                "candidate_lengths": summarize_lengths(
                    [record for record in records if "rejected_reason" not in record],
                    sequence_length,
                ),
                "selected_lengths": summarize_lengths(selections[split], sequence_length),
            }
        # A cache describes immutable inputs. Check again before publishing either split.
        for split in SPLITS:
            if sha256_file(data_dir / f"{split}.jsonl") != index["fingerprints"]["inputs"][split]:
                raise ValueError(f"{split}: source changed after length measurement")
        atomic_json(
            Path(temporary) / "selection.json",
            {
                "format_version": 2,
                "strategy": MIXED_STRATEGY if distribution == "mixed" else "natural",
                "splits": selections,
            },
        )
        for name in ("training.jsonl", "validation.jsonl", "selection.json"):
            os.replace(Path(temporary) / name, output_dir / name)
    return outputs


def main():
    """Measure trajectory lengths, select the requested distribution and write the filtered splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distribution", choices=("natural", "mixed"), default="natural")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_WORKSPACE / "models/qwen35_35b_a3b/tokenizer")
    parser.add_argument("--seq-length", type=int, default=65536)
    parser.add_argument("--loss-mode", choices=("assistant", "full"), default="assistant")
    parser.add_argument("--num-tokenizer-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=None,
        help="Mixed only; default 5120 unique source trajectories",
    )
    parser.add_argument(
        "--num-validation-samples",
        type=int,
        default=None,
        help="Mixed only; default 256 unique source trajectories",
    )
    parser.add_argument(
        "--length-bucket-ratios",
        type=float,
        nargs=3,
        default=None,
        metavar=("SHORT", "MEDIUM", "LONG"),
        help="Mixed only; sample count ratios, default 0.5 0.3 0.2",
    )
    args = parser.parse_args()
    mixed_args = (args.num_train_samples, args.num_validation_samples, args.length_bucket_ratios)
    if args.distribution == "natural" and any(value is not None for value in mixed_args):
        parser.error("Sample counts and bucket ratios apply only to --distribution mixed")
    split_counts = {
        "training": 5120 if args.num_train_samples is None else args.num_train_samples,
        "validation": 256 if args.num_validation_samples is None else args.num_validation_samples,
    }
    ratios = DEFAULT_BUCKET_RATIOS if args.length_bucket_ratios is None else args.length_bucket_ratios
    try:
        quotas = {split: bucket_quotas(count, ratios) for split, count in split_counts.items()}
    except ValueError as error:
        parser.error(str(error))
    if args.seq_length < 32 or args.seq_length % 32 or args.num_tokenizer_workers < 1:
        parser.error("Sequence length must be a positive multiple of 32; workers must be positive")
    if not args.tokenizer.is_dir():
        parser.error("The pinned tokenizer is missing; run script 01 first")
    if args.output_dir.resolve() == args.data_dir.resolve():
        parser.error("Output must differ from input")
    if any(
        (args.output_dir / name).exists()
        for name in ("training.jsonl", "validation.jsonl", "selection.json", "manifest.json")
    ):
        parser.error("Output already exists; choose a fresh --output-dir")
    if any(not (args.data_dir / f"{split}.jsonl").is_file() for split in SPLITS):
        parser.error("Input must contain training.jsonl and validation.jsonl from script 02")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    index, cache_path = load_length_index(
        args.data_dir, args.tokenizer, args.seq_length, args.num_tokenizer_workers, args.loss_mode
    )
    source_manifest = args.data_dir / "manifest.json"
    if source_manifest.is_file():
        prepared = json.loads(source_manifest.read_text())
        for split in SPLITS:
            expected = prepared.get("outputs", {}).get(split, {}).get("sha256")
            if expected and expected != index["fingerprints"]["inputs"][split]:
                raise ValueError(f"{split}: input does not match its preparation manifest")
    selections, prefix_cache = None, None
    if args.distribution == "mixed":
        selections, prefix_cache = build_mixed_selection(
            args.data_dir,
            args.tokenizer,
            index,
            args.seq_length,
            args.num_tokenizer_workers,
            args.loss_mode,
            args.seed,
            split_counts,
            ratios,
        )
    if input_fingerprints(args.data_dir, args.tokenizer, args.seq_length, args.loss_mode) != index["fingerprints"]:
        raise RuntimeError("Inputs changed before publication")
    outputs = write_selection(
        args.data_dir,
        args.output_dir,
        index,
        args.distribution,
        args.seed,
        args.seq_length,
        mixed_selections=selections,
    )
    manifest = {
        "format_version": 2,
        "source_dir": str(args.data_dir.resolve()),
        "distribution": args.distribution,
        "seed": args.seed,
        "selection": (
            "all trainable rows; no length downsampling"
            if args.distribution == "natural"
            else "requested bucket counts; whole rows first, verified assistant-message prefixes next; unique sources"
        ),
        "bucket_upper_bounds": [args.seq_length // 4, args.seq_length // 2, args.seq_length],
        "length_policy": "Render with the prepared tokenizer; measure after one shift before padding. Whole rows use the runtime cap; prefixes end at complete assistant messages and are never token-truncated.",
        "split_policy": "Preserve source train/validation membership; no new task-disjointness guarantee",
        "rejection_policy": "Exclude and report missing assistant targets; fail on unexpected errors",
        "row_order": ("source order" if args.distribution == "natural" else "seeded source-ID hash shuffle"),
        "selection_file": "selection.json",
        "selection_sha256": sha256_file(args.output_dir / "selection.json"),
        "length_cache": str(cache_path.resolve()),
        "length_cache_sha256": sha256_file(cache_path),
        "fingerprints": index["fingerprints"],
        "outputs": outputs,
    }
    if args.distribution == "mixed":
        manifest["mixed_strategy"] = MIXED_STRATEGY
        manifest["requested_samples"] = split_counts
        manifest["length_bucket_ratios"] = list(ratios)
        manifest["requested_bucket_counts"] = quotas
        manifest["prefix_search"] = {
            "maximum_encodings_per_source_bucket": MAX_PREFIX_ENCODINGS,
            "target": "uniform integer within bucket from seed and source ID hash",
            "cache": str(prefix_cache.resolve()),
            "cache_sha256": sha256_file(prefix_cache),
        }
    if source_manifest.is_file():
        manifest["source_manifest"] = str(source_manifest.resolve())
        manifest["source_manifest_sha256"] = sha256_file(source_manifest)
    atomic_json(args.output_dir / "manifest.json", manifest)
    for split, output in outputs.items():
        if output["rejected_rows"]:
            LOGGER.warning("Rejected %s: %s", split, output["rejected_rows_by_reason"])
        LOGGER.info("%s: %s", split, json.dumps(output["selected_lengths"]))


if __name__ == "__main__":
    main()
