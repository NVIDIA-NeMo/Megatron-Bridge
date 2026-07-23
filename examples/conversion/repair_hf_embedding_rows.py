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

"""Repair near-zero input embedding rows in a local Hugging Face safetensors checkpoint.

This is an offline preprocessing utility for diagnosed checkpoint defects where
rare token IDs have near-zero input embedding rows and produce extreme gradients
during continued pretraining. The script writes a repaired Hugging Face checkpoint
directory that can be used as the source for Megatron Bridge import/CPT.

Example:
    uv run python examples/conversion/repair_hf_embedding_rows.py \
        --input-hf-path /models/NVIDIA-Nemotron-3-Nano-4B-BF16 \
        --output-hf-path /models/NVIDIA-Nemotron-3-Nano-4B-BF16-repaired \
        --min-norm 1.0e-4 \
        --max-rows 256
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


logger = logging.getLogger(__name__)

G_SAFETENSORS_INDEX = "model.safetensors.index.json"
G_SINGLE_SAFETENSORS_FILE = "model.safetensors"
G_MANIFEST_NAME = "embedding_row_repair_manifest.json"

G_INPUT_EMBEDDING_CANDIDATES = (
    "backbone.embeddings.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "transformer.wte.weight",
    "gpt_neox.embed_in.weight",
)
G_OUTPUT_EMBEDDING_CANDIDATES = (
    "lm_head.weight",
    "embed_out.weight",
    "model.output.weight",
    "output_layer.weight",
)


@dataclass(frozen=True)
class TensorLocation:
    """Location of a tensor inside a local safetensors checkpoint."""

    name: str
    filename: str
    path: Path


@dataclass(frozen=True)
class EmbeddingRowRepairReport:
    """Summary of an embedding row repair operation."""

    input_embedding_name: str
    output_embedding_name: str
    repaired_row_ids: tuple[int, ...]
    min_norm_before: float
    target_norm: float | None
    input_embedding_shape: tuple[int, int]
    input_embedding_dtype: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(f"expected a finite non-negative float, got {value}")
    return parsed


def _load_weight_map(checkpoint_dir: Path) -> dict[str, str]:
    index_path = checkpoint_dir / G_SAFETENSORS_INDEX
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as index_file:
            index = json.load(index_file)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"{index_path} does not contain a valid weight_map")
        return {str(tensor_name): str(filename) for tensor_name, filename in weight_map.items()}

    safetensors_path = checkpoint_dir / G_SINGLE_SAFETENSORS_FILE
    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"Expected {G_SAFETENSORS_INDEX} or {G_SINGLE_SAFETENSORS_FILE} under {checkpoint_dir}"
        )

    with safe_open(str(safetensors_path), framework="pt", device="cpu") as reader:
        return {tensor_name: G_SINGLE_SAFETENSORS_FILE for tensor_name in reader.keys()}


def _resolve_tensor_name(
    weight_map: dict[str, str],
    *,
    requested_name: str | None,
    candidates: Sequence[str],
    role: str,
) -> str:
    if requested_name is not None:
        if requested_name not in weight_map:
            raise ValueError(f"Could not find requested {role} tensor {requested_name!r} in checkpoint")
        return requested_name

    for candidate in candidates:
        if candidate in weight_map:
            return candidate

    matching_keys = sorted(
        tensor_name for tensor_name in weight_map if "embed" in tensor_name.lower() or "lm_head" in tensor_name.lower()
    )
    raise ValueError(
        f"Could not infer the {role} tensor name. "
        f"Tried: {list(candidates)}. "
        f"Matching checkpoint keys: {matching_keys[:32]}"
    )


def _tensor_location(checkpoint_dir: Path, weight_map: dict[str, str], tensor_name: str) -> TensorLocation:
    filename = weight_map[tensor_name]
    tensor_path = checkpoint_dir / filename
    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor {tensor_name!r} points to missing shard {tensor_path}")
    return TensorLocation(name=tensor_name, filename=filename, path=tensor_path)


def _load_tensor(location: TensorLocation) -> torch.Tensor:
    with safe_open(str(location.path), framework="pt", device="cpu") as reader:
        return reader.get_tensor(location.name)


def _row_squared_norms(weight: torch.Tensor, *, chunk_rows: int) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape={tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise ValueError(f"Expected a floating-point embedding matrix, got dtype={weight.dtype}")

    row_squared_norms = torch.empty(weight.shape[0], dtype=torch.float64)
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        chunk = weight[start:end].to(torch.float32)
        row_squared_norms[start:end] = chunk.square().sum(dim=1, dtype=torch.float64)
    return row_squared_norms


def repair_embedding_rows(
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    input_embedding_name: str,
    output_embedding_name: str,
    min_norm: float,
    max_rows: int,
    chunk_rows: int,
) -> tuple[torch.Tensor, EmbeddingRowRepairReport]:
    """Repair near-zero input embedding rows using matching output embedding rows.

    Args:
        input_weight: Input embedding tensor with shape ``[vocab, hidden]``.
        output_weight: Output embedding tensor with shape ``[vocab, hidden]``.
        input_embedding_name: Input embedding tensor name for reporting.
        output_embedding_name: Output embedding tensor name for reporting.
        min_norm: Inclusive L2 norm threshold for damaged input rows.
        max_rows: Safety limit for the number of rows that may be rewritten.
        chunk_rows: Number of embedding rows to process at once when computing norms.

    Returns:
        A repaired input embedding tensor and a report describing the operation.

    Raises:
        ValueError: If shapes mismatch, too many rows are damaged, or replacement
            output rows are also unusable.
    """
    if input_weight.shape != output_weight.shape:
        raise ValueError(
            "Input and output embedding shapes must match for row repair, got "
            f"{tuple(input_weight.shape)} and {tuple(output_weight.shape)}"
        )

    input_squared_norms = _row_squared_norms(input_weight, chunk_rows=chunk_rows)
    input_norms = input_squared_norms.sqrt()
    damaged_mask = ~torch.isfinite(input_norms) | (input_norms <= min_norm)
    repaired_row_ids = tuple(int(row_id) for row_id in torch.nonzero(damaged_mask, as_tuple=False).flatten().tolist())
    min_norm_before = float(input_norms.min().item())

    if len(repaired_row_ids) > max_rows:
        raise ValueError(
            f"Refusing to repair {len(repaired_row_ids)} input embedding rows; configured max_rows={max_rows}. "
            "This likely indicates a broadly damaged or mismatched checkpoint."
        )

    if not repaired_row_ids:
        report = EmbeddingRowRepairReport(
            input_embedding_name=input_embedding_name,
            output_embedding_name=output_embedding_name,
            repaired_row_ids=(),
            min_norm_before=min_norm_before,
            target_norm=None,
            input_embedding_shape=(int(input_weight.shape[0]), int(input_weight.shape[1])),
            input_embedding_dtype=str(input_weight.dtype),
        )
        return input_weight.clone(), report

    healthy_mask = torch.isfinite(input_norms) & (input_norms > min_norm)
    healthy_count = int(healthy_mask.sum().item())
    if healthy_count == 0:
        raise ValueError("Cannot repair input embeddings because the checkpoint has no healthy input rows")

    target_norm = math.sqrt(float(input_squared_norms[healthy_mask].mean().item()))
    if not math.isfinite(target_norm) or target_norm <= min_norm:
        raise ValueError(f"Computed invalid healthy input embedding RMS norm: {target_norm}")

    row_ids = torch.tensor(repaired_row_ids, dtype=torch.long)
    source_rows = output_weight.index_select(0, row_ids).to(torch.float32)
    source_norms = torch.linalg.vector_norm(source_rows, dim=1).to(torch.float64)
    unusable_source_mask = ~torch.isfinite(source_norms) | (source_norms <= min_norm)
    if bool(unusable_source_mask.any()):
        unusable_ids = [
            repaired_row_ids[index] for index in torch.nonzero(unusable_source_mask, as_tuple=False).flatten().tolist()
        ]
        raise ValueError(
            "Cannot repair input embedding rows because the corresponding output rows are also damaged: "
            f"{unusable_ids}"
        )

    replacement_scale = torch.tensor(target_norm, dtype=torch.float32).div(source_norms.to(torch.float32))
    replacement_rows = source_rows.mul(replacement_scale.unsqueeze(1)).to(input_weight.dtype)
    repaired_weight = input_weight.clone()
    repaired_weight.index_copy_(0, row_ids, replacement_rows)

    repaired_norms = torch.linalg.vector_norm(repaired_weight.index_select(0, row_ids).to(torch.float32), dim=1)
    failed_mask = ~torch.isfinite(repaired_norms) | (repaired_norms <= min_norm)
    if bool(failed_mask.any()):
        failed_ids = [
            repaired_row_ids[index] for index in torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
        ]
        raise RuntimeError(f"Input embedding row repair did not produce healthy rows: {failed_ids}")

    report = EmbeddingRowRepairReport(
        input_embedding_name=input_embedding_name,
        output_embedding_name=output_embedding_name,
        repaired_row_ids=repaired_row_ids,
        min_norm_before=min_norm_before,
        target_norm=target_norm,
        input_embedding_shape=(int(input_weight.shape[0]), int(input_weight.shape[1])),
        input_embedding_dtype=str(input_weight.dtype),
    )
    return repaired_weight, report


def _load_safetensors_file(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        tensors = {tensor_name: reader.get_tensor(tensor_name) for tensor_name in reader.keys()}
    return tensors, metadata


def _copy_checkpoint_tree(source_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve(strict=False)
    if output_resolved == source_resolved:
        raise ValueError("Output checkpoint path must be different from input checkpoint path")
    if source_resolved in output_resolved.parents:
        raise ValueError("Output checkpoint path must not be inside the input checkpoint directory")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output checkpoint path already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, output_dir, ignore=shutil.ignore_patterns(".git", "__pycache__"))


def _rewrite_safetensors_tensor(location: TensorLocation, repaired_tensor: torch.Tensor) -> None:
    tensors, metadata = _load_safetensors_file(location.path)
    tensors[location.name] = repaired_tensor
    save_file(tensors, str(location.path), metadata=metadata)


def _write_manifest(
    output_dir: Path,
    report: EmbeddingRowRepairReport,
    *,
    input_hf_path: Path,
    min_norm: float,
    max_rows: int,
    input_shard_filename: str,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source_checkpoint": str(input_hf_path),
        "input_embedding_name": report.input_embedding_name,
        "output_embedding_name": report.output_embedding_name,
        "input_embedding_shape": list(report.input_embedding_shape),
        "input_embedding_dtype": report.input_embedding_dtype,
        "input_shard_filename": input_shard_filename,
        "min_norm": min_norm,
        "max_rows": max_rows,
        "min_norm_before": report.min_norm_before,
        "target_norm": report.target_norm,
        "num_repaired_rows": len(report.repaired_row_ids),
        "repaired_row_ids": list(report.repaired_row_ids),
    }
    with (output_dir / G_MANIFEST_NAME).open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")


def repair_hf_checkpoint(
    *,
    input_hf_path: Path,
    output_hf_path: Path | None,
    input_embedding_name: str | None,
    output_embedding_name: str | None,
    min_norm: float,
    max_rows: int,
    chunk_rows: int,
    overwrite: bool,
    dry_run: bool,
) -> EmbeddingRowRepairReport:
    """Repair a local Hugging Face safetensors checkpoint and optionally write a new checkpoint.

    Args:
        input_hf_path: Local source Hugging Face checkpoint directory.
        output_hf_path: Destination directory for the repaired checkpoint, or ``None`` for dry-run mode.
        input_embedding_name: Optional explicit input embedding tensor name.
        output_embedding_name: Optional explicit output embedding tensor name.
        min_norm: Inclusive L2 norm threshold for damaged input rows.
        max_rows: Safety limit for the number of rows that may be rewritten.
        chunk_rows: Number of embedding rows to process at once when computing norms.
        overwrite: Replace ``output_hf_path`` if it already exists.
        dry_run: Diagnose and report rows without writing output files.

    Returns:
        Report describing the detected and repaired rows.
    """
    if not input_hf_path.is_dir():
        raise FileNotFoundError(f"Input HF checkpoint path is not a directory: {input_hf_path}")
    if output_hf_path is None and not dry_run:
        raise ValueError("output_hf_path is required unless dry_run=True")

    weight_map = _load_weight_map(input_hf_path)
    resolved_input_name = _resolve_tensor_name(
        weight_map,
        requested_name=input_embedding_name,
        candidates=G_INPUT_EMBEDDING_CANDIDATES,
        role="input embedding",
    )
    resolved_output_name = _resolve_tensor_name(
        weight_map,
        requested_name=output_embedding_name,
        candidates=G_OUTPUT_EMBEDDING_CANDIDATES,
        role="output embedding",
    )
    input_location = _tensor_location(input_hf_path, weight_map, resolved_input_name)
    output_location = _tensor_location(input_hf_path, weight_map, resolved_output_name)

    logger.info("Loading input embedding %s from %s", input_location.name, input_location.filename)
    input_weight = _load_tensor(input_location)
    logger.info("Loading output embedding %s from %s", output_location.name, output_location.filename)
    output_weight = _load_tensor(output_location)

    repaired_weight, report = repair_embedding_rows(
        input_weight,
        output_weight,
        input_embedding_name=input_location.name,
        output_embedding_name=output_location.name,
        min_norm=min_norm,
        max_rows=max_rows,
        chunk_rows=chunk_rows,
    )

    if report.repaired_row_ids:
        logger.warning(
            "Detected %d damaged input embedding row(s): ids=%s, minimum_before=%.3e, target_norm=%.3e",
            len(report.repaired_row_ids),
            list(report.repaired_row_ids),
            report.min_norm_before,
            report.target_norm,
        )
    else:
        logger.info(
            "No input embedding rows at or below norm %.3e; minimum row norm is %.3e",
            min_norm,
            report.min_norm_before,
        )

    if dry_run:
        logger.info("Dry run complete; no checkpoint files were written")
        return report

    if output_hf_path is None:
        raise ValueError("output_hf_path is required when dry_run=False")

    _copy_checkpoint_tree(input_hf_path, output_hf_path, overwrite=overwrite)
    if report.repaired_row_ids:
        output_input_location = TensorLocation(
            name=input_location.name,
            filename=input_location.filename,
            path=output_hf_path / input_location.filename,
        )
        logger.info("Rewriting repaired input embedding shard %s", output_input_location.path)
        _rewrite_safetensors_tensor(output_input_location, repaired_weight)

    _write_manifest(
        output_hf_path,
        report,
        input_hf_path=input_hf_path,
        min_norm=min_norm,
        max_rows=max_rows,
        input_shard_filename=input_location.filename,
    )
    logger.info("Wrote repaired checkpoint to %s", output_hf_path)
    logger.info("Wrote repair manifest to %s", output_hf_path / G_MANIFEST_NAME)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-hf-path",
        required=True,
        type=Path,
        help="Local Hugging Face checkpoint directory to inspect and repair.",
    )
    parser.add_argument(
        "--output-hf-path",
        type=Path,
        help="Destination Hugging Face checkpoint directory. Required unless --dry-run is set.",
    )
    parser.add_argument(
        "--input-embedding-name",
        help="Input embedding tensor name. Defaults to known HF names such as backbone.embeddings.weight.",
    )
    parser.add_argument(
        "--output-embedding-name",
        help="Output embedding tensor name. Defaults to known HF names such as lm_head.weight.",
    )
    parser.add_argument(
        "--min-norm",
        type=_nonnegative_float,
        default=1.0e-4,
        help="Inclusive L2 norm threshold for damaged input rows.",
    )
    parser.add_argument(
        "--max-rows",
        type=_positive_int,
        default=256,
        help="Abort if more than this many input embedding rows need repair.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=_positive_int,
        default=4096,
        help="Number of embedding rows to process at once when computing norms.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-hf-path if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only diagnose damaged rows; do not write an output checkpoint.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the embedding row repair CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args(argv)
    repair_hf_checkpoint(
        input_hf_path=args.input_hf_path,
        output_hf_path=args.output_hf_path,
        input_embedding_name=args.input_embedding_name,
        output_embedding_name=args.output_embedding_name,
        min_norm=args.min_norm,
        max_rows=args.max_rows,
        chunk_rows=args.chunk_rows,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
