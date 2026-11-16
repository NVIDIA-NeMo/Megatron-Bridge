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

"""Import an Inkling2 checkpoint at one parallel layout, export it, and compare every tensor's bytes.

Quantized experts must come back in their stored NVFP4 or MXFP8 encoding, so the
comparison is byte-exact for every dtype, FP8 included. Rank 0 writes a JSON report.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist

from megatron.bridge import AutoBridge


def main() -> None:
    """Run the round trip and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    bridge = AutoBridge.from_hf_pretrained(args.hf_model, torch_dtype=torch.bfloat16)
    config = bridge.get_model_config()
    config.tensor_model_parallel_size = args.tp
    config.pipeline_model_parallel_size = args.pp
    config.expert_model_parallel_size = args.ep
    config.expert_tensor_parallel_size = 1
    config.sequence_parallel = args.tp > 1
    config.pipeline_dtype = torch.bfloat16
    model = bridge.get_model(config, load_weights=True, wrap_with_ddp=False)
    rank = dist.get_rank()

    source = bridge.hf_pretrained.state
    expected = set(source.keys())
    seen: set[str] = set()
    mismatches: list[dict] = []
    extra: list[str] = []
    for name, exported in bridge.export_hf_weights(model, cpu=True, show_progress=False):
        if rank != 0:
            continue
        if name not in expected:
            extra.append(name)
            continue
        seen.add(name)
        original = source[name]
        same = (
            exported.dtype == original.dtype
            and tuple(exported.shape) == tuple(original.shape)
            and torch.equal(exported.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8))
        )
        if not same:
            mismatches.append(
                {
                    "name": name,
                    "exported": [str(exported.dtype), list(exported.shape)],
                    "source": [str(original.dtype), list(original.shape)],
                }
            )
    if rank == 0:
        missing = sorted(expected - seen)
        report = {
            "layout": {"tp": args.tp, "pp": args.pp, "ep": args.ep},
            "source_tensors": len(expected),
            "exported_matching": len(seen) - len(mismatches),
            "mismatches": mismatches,
            "missing": missing,
            "extra": extra,
        }
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
