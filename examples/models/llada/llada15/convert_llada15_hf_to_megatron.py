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

"""One-shot HF → Megatron checkpoint conversion for LLaDA1.5.

Loads the HuggingFace ``GSAI-ML/LLaDA-1.5`` model via :class:`AutoBridge`
(which dispatches to :class:`LLaDA15Bridge`), converts the weights in-memory,
and writes a Megatron distributed checkpoint to disk. The saved checkpoint
can then be reloaded for inference (see ``run_llada15_chat.py``) or training
without re-doing the HF→Megatron conversion every time.

Usage::

    PYTHONPATH=src:/opt/Megatron-Bridge/src python3 \\
        examples/models/llada/llada15/convert_llada15_hf_to_megatron.py \\
        --hf-path /path/to/huggingface/hub/models--GSAI-ML--LLaDA-1.5/snapshots/<commit-hash> \\
        --out-path /path/to/llada15_megatron_ckpt

Result on disk::

    /path/to/llada15_megatron_ckpt/
        iter_0000000/
            ...torch_dist shards...
        latest_checkpointed_iteration.txt
        tokenizer/
            tokenizer.json
            tokenizer_config.json
            ...
"""

import argparse
import os

import torch
import torch.distributed as dist

from megatron.bridge import AutoBridge

# Side effect: registers the LLaDA15Bridge with AutoBridge.
from megatron.bridge.diffusion.conversion.llada15 import llada15_bridge  # noqa: F401


def setup_distributed_single_gpu():
    """Initialize a 1-rank process group; Megatron's checkpoint save requires it."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", world_size=1, rank=0)


def main():
    """Convert a HuggingFace LLaDA-1.5 snapshot into a Megatron distributed checkpoint on disk."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", required=True, help="Local HF snapshot dir for LLaDA-1.5.")
    parser.add_argument("--out-path", required=True, help="Where to save the Megatron checkpoint.")
    args = parser.parse_args()

    setup_distributed_single_gpu()

    print(f"Loading HF model via AutoBridge: {args.hf_path}")
    bridge = AutoBridge.from_hf_pretrained(args.hf_path, trust_remote_code=True)

    print("Building Megatron GPTModel and loading weights via the bridge...")
    megatron_model = bridge.to_megatron_model(wrap_with_ddp=False)
    if not isinstance(megatron_model, list):
        megatron_model = [megatron_model]

    print(f"Saving Megatron checkpoint to {args.out_path}")
    # hf_tokenizer_path embeds tokenizer metadata into the checkpoint so the
    # later chat script doesn't need to know the original HF path.
    bridge.save_megatron_model(
        megatron_model,
        args.out_path,
        hf_tokenizer_path=args.hf_path,
        hf_tokenizer_kwargs={"trust_remote_code": True},
    )
    print("Done.")


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
