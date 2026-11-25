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


def install(dest: Path) -> Callable[[], None]:
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
    local = None

    def cpu(t):
        return t.detach().cpu().numpy().copy()

    def batch(iterator, config, *, cp_group, device):
        nonlocal mb, local
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
            chosen = np.flatnonzero(np.isin(pos % 2048, [15, 1023]) | np.isin(pos, bounds[1:] - 2))
            data["sample"] = pos[chosen]
            local = torch.as_tensor(chosen, device=device)
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
            rows = local[(local >= start) & (local < start + len(logits))] - start
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

        def capture_grad(grad):
            data[kind + "_dh"] = cpu(grad.index_select(0, local).float().flatten(1))

        hidden.register_hook(capture_grad)
        return result

    ernie55_step.get_batch = batch
    mtp_loss.roll_tensor = roll
    cuda_head.forward = raw
    chunked_head.chunked_training_loss = head

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
    totals = {k: np.zeros(10, dtype="float64") for k in ("main", "mtp")}

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
                ("sample", ["main_logits", "mtp_logits", "main_dh", "mtp_dh"]),
            ):
                coords = np.concatenate([r[coord] for r in ranks])
                if len(np.unique(coords)) != len(coords):
                    raise ValueError(f"{path}: duplicate {coord} in DP{dp}.")
                order = np.argsort(coords)
                combined[coord] = coords[order]
                for key in fields:
                    if any(len(r[key]) != len(r[coord]) for r in ranks):
                        raise ValueError(f"{path}: {key} disagrees with {coord} in DP{dp}.")
                    combined[key] = np.concatenate([r[key] for r in ranks])[order]
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
            for i, field in enumerate(("logits", "dh")):
                ref = x[name + "_" + field][ix[keep]].astype("float64")
                cur = y[name + "_" + field][iy[keep]].astype("float64")
                if ref.shape != cur.shape:
                    raise ValueError(f"Different sampled {field} shapes for {name} in DP{dp}.")
                diff = cur - ref
                acc[2 + 4 * i] += np.square(diff).sum()
                acc[3 + 4 * i] += np.square(ref).sum()
                acc[4 + 4 * i] = max(acc[4 + 4 * i], np.abs(diff).max(initial=0))
                acc[5 + 4 * i] = max(acc[5 + 4 * i], np.abs(ref).max(initial=0))
    if first:
        raise ValueError("Missing data replicas on the candidate.")
    results = {}
    for name, z in totals.items():
        if min(z[0], z[3], z[5], z[7], z[9]) <= 0 or not np.isfinite(z).all():
            raise ValueError(f"Missing or nonfinite reference norm for {name}.")
        results[name] = dict(
            relative_mean_ce=abs(z[1] - z[0]) / z[0],
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
    files = parser.parse_args()
    sys.stdout.write(json.dumps(compare(files.reference, files.candidate), indent=2) + "\n")
