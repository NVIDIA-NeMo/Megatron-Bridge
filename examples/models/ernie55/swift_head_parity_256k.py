#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Export the first ms-swift-int ERNIE 5.5 training microbatch in head_parity_256k.py's schema."""

import argparse
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def prepare(jsonl: Path, directory: Path) -> None:
    """Build the original Swift trainer's cached, padding-free 256K row from the Bridge fixture."""
    from datasets import Dataset

    if directory.exists():
        raise FileExistsError(directory)
    tokens, labels, positions = [], [], []
    with jsonl.open() as stream:
        for line in stream:
            doc = json.loads(line)
            ids, targets = doc["input_ids"], doc["labels"]
            if not ids or len(ids) != len(targets):
                raise ValueError("Each document needs equal nonzero lengths for input_ids and labels.")
            padded = (len(ids) + 31) // 32 * 32
            tokens.extend([*ids, *([0] * (padded - len(ids)))])
            labels.extend([*targets, *([-100] * (padded - len(ids)))])
            positions.extend(range(padded))
    if len(tokens) != 262144:
        raise ValueError("Expected exactly 262144 positions after aligning each document to 2 * CP.")

    # A prepared row preserves document order and resets positions at each document's first token.
    def rows():
        for _ in range(32):
            yield {
                "input_ids": tokens,
                "labels": labels,
                "position_ids": positions,
                "length": len(tokens),
                "lengths": [len(tokens)],
            }

    Dataset.from_generator(rows).save_to_disk(str(directory))


def physical_rows(bounds: np.ndarray, positions: np.ndarray, cp: int) -> np.ndarray:
    """Join the active contiguous CP partition to padded global token positions."""
    if bounds[0] != 0 or bounds[-1] != 262144 or np.any(np.diff(bounds) <= 0):
        raise ValueError("Invalid global padded document boundaries.")
    if positions.size != 16384 or not 0 <= cp < 16:
        raise ValueError("Expected 16384 local positions on a CP16 shard.")
    whole = np.arange(cp * positions.size, (cp + 1) * positions.size)
    spans = np.diff(bounds)
    choices = [whole]
    if np.all(spans % 16 == 0):
        choices.append(
            np.concatenate([np.arange(lo + cp * (n // 16), lo + (cp + 1) * (n // 16)) for lo, n in zip(bounds, spans)])
        )
    matches = [g for g in choices if np.array_equal(positions, g - bounds[np.searchsorted(bounds[1:], g, "right")])]
    if not matches or any(not np.array_equal(g, matches[0]) for g in matches[1:]):
        raise ValueError(f"Observed position_ids do not uniquely identify the contiguous partition on CP{cp}.")
    return matches[0]


class CaptureComplete(BaseException):
    """End a diagnostic before the next microbatch and the first parameter update."""


class HeadCapture:
    """Observe main and MTP values on the original Swift autograd graph."""

    def __init__(self, directory):
        self.directory = directory
        self.data = {}
        self.heads = []
        self.rolls = []
        self.active = False
        self.saved = False
        self.restore = []

    def install(self):
        import torch
        import torch.distributed as dist
        from mcore_bridge.model import gpt_model
        from mcore_bridge.model.gpts.ernie5_5 import Ernie5_5GPTModel as Ernie
        from megatron.core import parallel_state as ps

        def wrap(owner, attr, observe):
            original = getattr(owner, attr)
            self.restore.append((owner, attr, owner.__dict__.get(attr), attr in owner.__dict__))
            setattr(owner, attr, observe(original))

        def cpu(t):
            return t.detach().cpu().numpy().copy()

        def forward(original):
            signature = inspect.signature(original)

            def record(model, *args, **kwargs):
                if self.active:
                    self.save()
                    raise CaptureComplete()
                if not self.saved:
                    params = signature.bind(model, *args, **kwargs).arguments
                    packed = params["packed_seq_params"]
                    if dist.get_world_size() != 64 or ps.get_context_parallel_world_size() != 16:
                        raise ValueError("Expected 64 ranks with CP16.")
                    if (
                        ps.get_tensor_model_parallel_world_size(),
                        ps.get_pipeline_model_parallel_world_size(),
                        ps.get_expert_model_parallel_world_size(),
                    ) != (1, 1, 64):
                        raise ValueError("Expected TP1, PP1 and EP64.")
                    if packed.cp_partition_mode != "contiguous" or model.config.mtp_num_layers != 1:
                        raise ValueError("Expected contiguous CP and one active MTP head.")
                    boundaries = getattr(packed, "cu_seqlens_q_padded", None)
                    if boundaries is None:
                        boundaries = packed.cu_seqlens_q
                    bounds = cpu(boundaries).ravel().astype(np.int64)
                    pos = physical_rows(
                        bounds,
                        cpu(params["position_ids"]).ravel(),
                        ps.get_context_parallel_rank(),
                    )
                    select = np.flatnonzero(np.isin(pos % 2048, (15, 1023)) | np.isin(pos, bounds[1:] - 2))
                    ids = params["input_ids"]
                    self.index = torch.as_tensor(select, device=ids.device)
                    self.main_mask = params.get("loss_mask")
                    self.data.update(
                        dp=np.asarray(ps.get_data_parallel_rank(with_context_parallel=False)),
                        bounds=bounds,
                        pos=pos,
                        sample=pos[select],
                        tokens=cpu(ids).ravel(),
                    )
                    self.active = True
                return original(model, *args, **kwargs)

            return record

        def raw(original):
            def record(model, hidden, *args, **kwargs):
                logits = original(model, hidden, *args, **kwargs)
                if self.active:
                    if len(self.heads) >= 2:
                        raise ValueError("Unexpected extra output projection during the first microbatch.")
                    tag = ("mtp", "main")[len(self.heads)]
                    self.heads.append(tag)
                    if logits.shape != (16384, 1, 201216) or hidden.shape != (
                        16384,
                        1,
                        4096,
                    ):
                        raise ValueError(f"Unexpected Swift head shapes {logits.shape}, {hidden.shape}")
                    self.data[tag + "_logits"] = cpu(logits[:, 0, ::1237].index_select(0, self.index).float())

                    def gradient(g):
                        if tag + "_dh" in self.data:
                            raise ValueError(f"Duplicate {tag} head-input gradient.")
                        self.data[tag + "_dh"] = cpu(g.index_select(0, self.index).flatten(1).float())

                    hidden.register_hook(gradient)
                return logits

            return record

        def rolling(original):
            def record(*args, **kwargs):
                out = original(*args, **kwargs)
                if self.active and self.heads == ["mtp"] and "mtp_ce" not in self.data:
                    self.rolls.append(out[0])
                return out

            return record

        def loss(original):
            def record(model, labels, logits, *args, **kwargs):
                values = original(model, labels, logits, *args, **kwargs)
                if self.active:
                    tag = self.heads[-1]
                    if tag + "_ce" in self.data:
                        raise ValueError(f"Duplicate {tag} loss.")
                    actual = cpu(labels).ravel()
                    mask = self.main_mask
                    if tag == "mtp":
                        if len(self.rolls) != 2 or not torch.equal(self.rolls[0], labels):
                            raise ValueError("Expected the MTP label roll and mask roll before its CE.")
                        mask = self.rolls[1]
                    if mask is not None:
                        m = cpu(mask).ravel()
                        if m.shape != actual.shape or not np.isin(m, (0, 1)).all():
                            raise ValueError("Swift loss mask shape/values disagree with the head targets.")
                        actual = np.where(m != 0, actual, -100)
                    self.data[tag + "_labels"] = actual
                    self.data[tag + "_ce"] = cpu(values.float()).ravel()
                    self.rolls.clear()
                return values

            return record

        wrap(Ernie, "forward", forward)
        wrap(gpt_model.GPTModel, "_forward_output_layer", raw)
        wrap(gpt_model, "roll_tensor", rolling)
        wrap(Ernie, "compute_language_model_loss", loss)

    def save(self):
        if self.saved:
            return
        required = {"dp", "bounds", "pos", "sample", "tokens"}
        required.update(f"{h}_{k}" for h in ("main", "mtp") for k in ("labels", "ce", "logits", "dh"))
        if required != self.data.keys() or self.heads != ["mtp", "main"]:
            raise RuntimeError(f"Incomplete first-microbatch capture. Missing {sorted(required - self.data.keys())}.")
        rank = int(os.environ["RANK"])
        self.directory.mkdir(parents=True, exist_ok=True)
        final = self.directory / f"rank{rank:02}.npz"
        with tempfile.NamedTemporaryFile(dir=self.directory, prefix=f".rank{rank:02}-", delete=False) as stream:
            tmp = Path(stream.name)
            try:
                np.savez_compressed(stream, **self.data)
                stream.flush()
                os.link(tmp, final)  # Atomic publish; an existing capture must never be overwritten.
            finally:
                tmp.unlink(missing_ok=True)
        self.saved = True
        self.active = False
        self.rolls.clear()
        self.data.clear()
        self.main_mask = None

    def uninstall(self):
        for owner, attr, value, existed in reversed(self.restore):
            if existed:
                setattr(owner, attr, value)
            else:
                delattr(owner, attr)


def main():
    """Prepare the reference input or run its trainer with the capture hooks."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    cache = commands.add_parser("prepare", help="Prepare one public JSONL fixture for Swift.")
    cache.add_argument("jsonl", type=Path)
    cache.add_argument("directory", type=Path)
    run = commands.add_parser("record", help="Pass ordinary megatron_sft_main arguments after --.")
    run.add_argument("directory", type=Path)
    run.add_argument("--deterministic", action="store_true")
    own, sep, swift_args = sys.argv[1:], False, []
    if "--" in own:
        i = own.index("--")
        own, sep, swift_args = own[:i], True, own[i + 1 :]
    args = parser.parse_args(own)
    if args.action == "prepare":
        if sep:
            parser.error("prepare does not take trainer options.")
        prepare(args.jsonl, args.directory)
        return
    if not sep or not swift_args:
        parser.error("Pass the original Swift training arguments after --.")
    import torch

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from swift.megatron import megatron_sft_main

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    cap = HeadCapture(args.directory)
    cap.install()
    try:
        sys.argv[1:] = swift_args
        megatron_sft_main()
        raise RuntimeError("Trainer returned before the first complete pre-update capture.")
    except CaptureComplete:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    finally:
        cap.uninstall()


if __name__ == "__main__":
    main()
