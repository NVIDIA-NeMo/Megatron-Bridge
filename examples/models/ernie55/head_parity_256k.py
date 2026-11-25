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

"""Observe and compare the first, pre-update microbatch's main and MTP head values."""

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np


def install(dest: Path, *, model_parts=(), observe_ce_gradient=None) -> Callable[[], None]:
    """Record real CE, sampled logits and head-input gradients at global packed coordinates."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state

    from megatron.bridge.models.ernie import ernie55_step
    from megatron.bridge.models.ernie.modeling_ernie55 import chunked_head, cuda_head, mtp_loss

    rank = int(os.environ["RANK"])
    original_batch = ernie55_step.get_batch
    original_cuda = cuda_head.forward
    original_head = chunked_head.chunked_training_loss
    original_roll = mtp_loss.roll_tensor
    mb, start, tag = 0, 0, None
    data = {}
    pieces = []
    local = raw_local = None

    def cpu(t):
        return t.detach().cpu().numpy().copy()

    def batch(iterator, config, *, cp_group, device):
        nonlocal mb, local, raw_local
        mb += 1
        result = original_batch(iterator, config, cp_group=cp_group, device=device)
        if mb == 1:
            cp, size = dist.get_rank(cp_group), dist.get_world_size(cp_group)
            data["dp"] = np.asarray(parallel_state.get_data_parallel_rank(with_context_parallel=False))
            bounds = cpu(result.packed_seq_params.cu_seqlens_q_padded).astype("int64")
            data["bounds"] = bounds
            if config.cp_partition_mode != "zigzag" or size != 16:
                raise ValueError("The example head probe requires CP16 zigzag.")
            blocks = []
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                width, rem = divmod(hi - lo, 2 * size)
                if rem:
                    raise ValueError("Expected each document to be aligned to twice CP.")
                for part in (cp, 2 * size - 1 - cp):
                    blocks.append(np.arange(lo + part * width, lo + (part + 1) * width))
            pos = np.concatenate(blocks)
            data["pos"] = pos
            if not np.array_equal(
                cpu(result.position_ids).ravel(), pos - bounds[np.searchsorted(bounds[1:], pos, "right")]
            ):
                raise ValueError("Global mapping disagrees with the batch's per-document position IDs.")
            data["tokens"] = cpu(result.tokens).ravel()
            data["main_labels"] = cpu(result.labels).ravel()

            def sample_rows(positions):
                return np.flatnonzero(np.isin(positions % 2048, [15, 1023]) | np.isin(positions, bounds[1:] - 2))

            chosen = sample_rows(pos)
            data["sample"] = pos[chosen]
            local = torch.as_tensor(chosen, device=device)
            raw_local = local
            if getattr(config, "ernie_head_reference_wgrad", False):
                # Raw logits are now inside the contiguous-owner projection. The public
                # loss and the incoming hidden gradient still use the stack's zigzag rows.
                raw_pos = cp * len(pos) + np.arange(len(pos))
                raw_rows = sample_rows(raw_pos)
                data["logit_sample"] = raw_pos[raw_rows]
                raw_local = torch.as_tensor(raw_rows, device=device)
        return result

    def roll(*args, **kwargs):
        result = original_roll(*args, **kwargs)
        if mb == 1 and kwargs.get("sequence_fields") == ["labels", "loss_mask"]:
            labels, mask = (cpu(t).ravel() for t in result[:2])
            data["mtp_labels"] = np.where(mask != 0, labels, -100)
        return result

    def raw(logits, rt, labels, maxima, denom, loss, preds):
        nonlocal start
        if mb == 1 and tag is not None:
            rows = raw_local[(raw_local >= start) & (raw_local < start + len(logits))] - start
            sample = logits.index_select(0, rows)
            pieces.append(cpu(sample[:, ::1237].float()))
            start += len(logits)
        return original_cuda(logits, rt, labels, maxima, denom, loss, preds)

    def head(layer, hidden, labels, weight, **kw):
        nonlocal start, tag, pieces
        if mb != 1:
            return original_head(layer, hidden, labels, weight, **kw)
        kind = kw.get("head_kind", "main")
        start, tag, pieces = 0, kind, []
        try:
            result = original_head(layer, hidden, labels, weight, **kw)
        finally:
            tag = None
        if result is None or start != len(hidden):
            raise RuntimeError("The CUDA head did not produce one logit row per hidden row.")
        data[kind + "_logits"] = np.concatenate(pieces)
        data[kind + "_ce"] = cpu(result[0].float()).ravel()
        if observe_ce_gradient is not None:
            observe_ce_gradient(kind, result[0], data[kind + "_labels"])

        def capture_grad(grad):
            data[kind + "_dh"] = cpu(grad.index_select(0, local).float().flatten(1))

        hidden.register_hook(capture_grad)
        return result

    ernie55_step.get_batch = batch
    mtp_loss.roll_tensor = roll
    cuda_head.forward = raw
    chunked_head.chunked_training_loss = head

    if model_parts:
        # Model construction binds the chunked loss to each output-layer instance.
        # A recorder attached after warmup must also instrument these existing bound methods.
        from types import MethodType

        bound = 0
        for part in model_parts:
            for module in part.modules():
                method = getattr(module, "chunked_training_loss", None)
                if getattr(method, "__func__", None) is original_head:
                    module.chunked_training_loss = MethodType(head, module)
                    bound += 1
        if not bound:
            raise RuntimeError("No initialized ERNIE CUDA output layer uses the chunked head.")

    def save():
        required = {"dp", "bounds", "pos", "tokens", "sample"}
        required.update(f"{kind}_{field}" for kind in ("main", "mtp") for field in ("labels", "ce", "logits", "dh"))
        if missing := required.difference(data):
            raise RuntimeError(f"Incomplete head capture on rank {rank}: {sorted(missing)}")
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"rank{rank:02}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **data)

    return save


def compare(a: Path, b: Path) -> dict:
    """Require identical global coordinates, masks, tokens and targets; compare all valid rows."""
    totals = {k: np.zeros(14, dtype="float64") for k in ("main", "mtp")}

    def globally_ordered(path: Path):
        # Keep four replicas separate even when an example repeats the same row on all four.
        shards = {}
        for rank in range(64):
            with np.load(path / f"rank{rank:02}.npz") as record:
                dp = int(record["dp"])
                shards.setdefault(dp, []).append({k: record[k] for k in record.files if k != "dp"})
        if len(shards) != 4:
            raise ValueError(f"{path}: expected four dense replicas.")
        for dp, ranks in shards.items():
            if any(not np.array_equal(r["bounds"], ranks[0]["bounds"]) for r in ranks):
                raise ValueError(f"{path}: inconsistent document boundaries in DP{dp}.")
            combined = {"bounds": ranks[0]["bounds"]}
            for coord, fields in (
                ("pos", ["tokens", "main_labels", "mtp_labels", "main_ce", "mtp_ce"]),
                ("sample", ["main_dh", "mtp_dh"]),
                ("logit_sample", ["main_logits", "mtp_logits"]),
            ):
                # Older Bridge and independent reference captures share one sample
                # mapping for logits and dHidden. New head owners may differ per rank.
                rank_coords = [
                    r.get("logit_sample", r["sample"]) if coord == "logit_sample" else r[coord] for r in ranks
                ]
                coords = np.concatenate(rank_coords)
                if len(np.unique(coords)) != len(coords):
                    raise ValueError(f"{path}: duplicate {coord} in DP{dp}.")
                order = np.argsort(coords)
                combined[coord] = coords[order]
                for key in fields:
                    if any(len(r[key]) != len(rows) for r, rows in zip(ranks, rank_coords)):
                        raise ValueError(f"{path}: {key} disagrees with {coord} in DP{dp}.")
                    combined[key] = np.concatenate([r[key] for r in ranks])[order]
            if not np.array_equal(combined["sample"], combined["logit_sample"]):
                raise ValueError(f"{path}: logits and dHidden cover different global sample coordinates in DP{dp}.")
            if not np.array_equal(combined["pos"], np.arange(262144)):
                raise ValueError(f"{path}: incomplete global coordinates in DP{dp}.")
            yield dp, combined

    first = dict(globally_ordered(a))
    for dp, y in globally_ordered(b):
        x = first.pop(dp)
        for key in ("bounds", "tokens", "pos", "main_labels", "mtp_labels"):
            if not np.array_equal(x[key], y[key]):
                raise ValueError(f"DP{dp} differs in {key}.")
        common, ix, iy = np.intersect1d(x["sample"], y["sample"], return_indices=True)
        if not len(common):
            raise ValueError(f"No common sampled global positions in DP{dp}.")
        for name, acc in totals.items():
            valid = x[name + "_labels"] >= 0
            acc[0] += x[name + "_ce"][valid].sum(dtype=np.float64)
            acc[1] += y[name + "_ce"][valid].sum(dtype=np.float64)
            keep = valid[common]
            if not keep.any():
                raise ValueError(f"No common valid sampled positions for {name} in DP{dp}.")
            for i, (field, left, right) in enumerate(
                (("logits", ix[keep], iy[keep]), ("dh", ix[keep], iy[keep]), ("ce", valid, valid))
            ):
                ref = x[f"{name}_{field}"][left].astype("float64")
                cur = y[f"{name}_{field}"][right].astype("float64")
                if ref.shape != cur.shape:
                    raise ValueError(f"Different {field} shapes for {name} in DP{dp}.")
                diff = cur - ref
                acc[2 + 4 * i] += np.square(diff).sum()
                acc[3 + 4 * i] += np.square(ref).sum()
                acc[4 + 4 * i] = max(acc[4 + 4 * i], np.abs(diff).max(initial=0))
                acc[5 + 4 * i] = max(acc[5 + 4 * i], np.abs(ref).max(initial=0))
    if first:
        raise ValueError("Missing data replicas on the candidate.")
    results = {}
    for name, z in totals.items():
        if min(z[0], z[3], z[5], z[7], z[9], z[11], z[13]) <= 0 or not np.isfinite(z).all():
            raise ValueError(f"Missing or nonfinite reference norm for {name}.")
        results[name] = dict(
            relative_mean_ce=abs(z[1] - z[0]) / z[0],
            valid_ce_relative_l2=np.sqrt(z[10] / z[11]),
            valid_ce_peak_relative=z[12] / z[13],
            sample_raw_logit_relative_l2=np.sqrt(z[2] / z[3]),
            sample_raw_logit_peak_relative=z[4] / z[5],
            sample_head_input_grad_relative_l2=np.sqrt(z[6] / z[7]),
            sample_head_input_grad_peak_relative=z[8] / z[9],
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--check", action="store_true", help="Exit with a failure if a 256K BF16 head gate fails.")
    files = parser.parse_args()
    results = compare(files.reference, files.candidate)
    if files.check:
        for metrics in results.values():
            metrics["passed"] = bool(
                metrics["relative_mean_ce"] <= 1e-5
                and all(value <= 0.005 for key, value in metrics.items() if key.endswith("relative_l2"))
                and all(value <= 0.01 for key, value in metrics.items() if key.endswith("peak_relative"))
            )
    sys.stdout.write(json.dumps(results, indent=2) + "\n")
    if files.check and not all(metrics["passed"] for metrics in results.values()):
        sys.exit(1)
