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

"""Convert a public Ling 3.0 Tiny or Flash Hugging Face checkpoint to native DCP."""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch.distributed as dist

from megatron.bridge import AutoBridge


def main() -> None:
    """Run the model-only Ling 3.0 HF-to-DCP conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True, help="Local Ling 3.0 Tiny or Flash Hugging Face directory.")
    parser.add_argument("--output", required=True, help="Output Megatron torch_dist checkpoint directory.")
    parser.add_argument("--revision", default=None, help="Optional source revision recorded in provider config.")
    parser.add_argument(
        "--low-memory-save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the model-destroying low-memory DCP save path (default: enabled).",
    )
    args = parser.parse_args()

    bridge_kwargs = {"trust_remote_code": True}
    if args.revision is not None:
        bridge_kwargs["revision"] = args.revision
    bridge = AutoBridge.from_hf_pretrained(args.hf_path, **bridge_kwargs)

    from megatron.bridge.training.model_load_save import temporary_distributed_context

    context = nullcontext() if dist.is_initialized() else temporary_distributed_context(backend="gloo")
    with context:
        model = bridge.to_megatron_model(
            load_weights=True,
            wrap_with_ddp=False,
            use_cpu_initialization=True,
            mixed_precision_wrapper=None,
        )
        bridge.save_megatron_model(model, args.output, low_memory_save=args.low_memory_save)


if __name__ == "__main__":
    main()
