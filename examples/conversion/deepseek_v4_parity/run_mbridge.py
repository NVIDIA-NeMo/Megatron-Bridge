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

"""Run a Megatron-Bridge forward pass on a fixed prompt set and save the last-position logits.

The output is a `.pt` file consumed by ``compare_logits.py``. Format is shared with
``run_hf_reference.py``.

Forward-pass logic mirrors ``examples/conversion/compare_hf_and_megatron/compare.py`` (TP
all-gather of the vocab-sharded logits, MTP disabled for inference, attention_mask=None so
Megatron auto-builds the causal mask). We do not load an HF model on the same ranks.

DSv4 currently requires TP=1; scale via expert parallelism. Typical config for the
Flash variant on a single 4xB200 node is TP=1, EP=4, PP=1.

Run examples:

    # DSv4-Flash on 4xB200 (single node, TP=1, EP=4)
    uv run python -m torch.distributed.run --nproc_per_node=4 \
        examples/conversion/deepseek_v4_parity/run_mbridge.py \
        --hf_model_path deepseek-ai/DeepSeek-V4-Flash \
        --prompts_file examples/conversion/deepseek_v4_parity/prompts.json \
        --output /chcui/parity/dsv4/logits_mb.pt \
        --tp 1 --ep 4
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import disable_mtp_for_inference, print_rank_0


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Megatron-Bridge forward pass and save logits")
    parser.add_argument("--hf_model_path", type=str, required=True, help="HuggingFace model id or local BF16 path")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output .pt file path (written from rank 0)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallel size")
    parser.add_argument(
        "--sequence_parallel",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Enable sequence parallel. 'auto' (default): on when tp>1 and ep>1.",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to HF loaders")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Truncate prompts longer than this")
    parser.add_argument("--top_k", type=int, default=10, help="Save top-K token ids/logits for sanity inspection")
    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=0,
        help="Token id used for right-padding. Must match the HF reference run.",
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help="Override the seq-length padding multiple. Defaults to --tp; useful at TP=1 where "
        "we still want explicit padding to mirror the HF reference's --pad_to_multiple_of value.",
    )
    return parser.parse_args()


def load_prompts(prompts_file: str) -> list[dict]:
    """Load the prompts JSON file and return its ``prompts`` list."""
    with open(prompts_file) as f:
        data = json.load(f)
    return data["prompts"]


def pad_input_ids_to_multiple(input_ids: torch.Tensor, multiple: int, pad_token_id: int) -> tuple[torch.Tensor, int]:
    """Pad ``input_ids`` so the sequence length is divisible by ``multiple``.

    Required for sequence-parallel + EP (multiple = TP size). Also useful at TP=1 to mirror
    the HF reference's explicit padding. Returns the padded tensor and the original
    (unpadded) sequence length so we can later index into the unpadded last position.
    """
    if multiple <= 1:
        return input_ids, int(input_ids.shape[1])
    seq_len = input_ids.shape[1]
    remainder = seq_len % multiple
    if remainder == 0:
        return input_ids, seq_len
    pad_len = multiple - remainder
    padding = torch.full((input_ids.shape[0], pad_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([input_ids, padding], dim=1), seq_len


class SingleBatchIterator:
    """Iterator yielding exactly one batch, then ``StopIteration``.

    Required by ``forward_backward_func`` which expects an iterable.
    """

    def __init__(self, input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        self.batch = {"tokens": input_ids, "position_ids": position_ids, "attention_mask": attention_mask}
        self._yielded = False

    def __iter__(self) -> "SingleBatchIterator":
        return self

    def __next__(self) -> dict:
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def forward_step(data_iterator, model, **kwargs) -> tuple[torch.Tensor, callable]:
    """Forward step for ``forward_backward_func``: run the model and return its output."""
    batch = next(data_iterator)
    out = model(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        attention_mask=batch.get("attention_mask", None),
    )
    if isinstance(out, tuple):
        out = out[0]

    def loss_func(x, **_):
        return x

    return out, loss_func


def load_megatron_model(args: argparse.Namespace) -> tuple[list, AutoBridge]:
    """Load a Megatron model from HF weights via AutoBridge.

    Returns the list of model components (one per PP stage held by this rank) and the bridge
    instance used to perform the conversion.
    """
    print_rank_0(f"Loading bridge from {args.hf_model_path}")
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model_path,
        trust_remote_code=is_safe_repo(trust_remote_code=args.trust_remote_code, hf_path=args.hf_model_path),
    )
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = args.tp
    provider.pipeline_model_parallel_size = args.pp
    provider.expert_model_parallel_size = args.ep
    provider.expert_tensor_parallel_size = args.etp
    provider.pipeline_dtype = torch.bfloat16
    # Required by Megatron when TP>1 + EP>1; otherwise opt-in via flag.
    if args.sequence_parallel == "auto":
        provider.sequence_parallel = args.tp > 1 and args.ep > 1
    else:
        provider.sequence_parallel = args.sequence_parallel == "true"
    print_rank_0(f"sequence_parallel={provider.sequence_parallel}")

    # DSv4 default parity config is PP=1, so no layout helper is required. If a future caller
    # raises PP > 1 they must add an entry to the recipe layout map (see HANDOFF.md guidance).
    if args.pp > 1:
        raise NotImplementedError(
            "DSv4 parity harness does not currently configure pipeline_model_parallel_layout; "
            "either keep --pp 1 or extend run_mbridge.py with a (pp, vp) layout map."
        )

    provider.finalize()
    megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)

    for m in megatron_model:
        disable_mtp_for_inference(m)

    return [m.eval() for m in megatron_model], bridge


def run_one_prompt(
    *,
    megatron_model: list,
    tokenizer,
    prompt_text: str,
    pad_multiple: int,
    max_seq_len: int,
    top_k: int,
    pad_token_id: int,
) -> dict | None:
    """Run a single forward pass on ``prompt_text`` and return the per-prompt result dict on rank 0.

    Returns ``None`` on non-rank-0 (or non-final-PP-stage non-rank-0) so callers can ignore them.
    """
    enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_seq_len, add_special_tokens=True)
    input_ids = enc.input_ids.cuda()
    input_ids_padded, orig_seq_len = pad_input_ids_to_multiple(input_ids, pad_multiple, pad_token_id)
    padded_seq_len = int(input_ids_padded.shape[1])

    position_ids = (
        torch.arange(padded_seq_len, dtype=torch.long, device=input_ids_padded.device)
        .unsqueeze(0)
        .expand_as(input_ids_padded)
    )
    # Pass None: Megatron auto-builds the correct causal mask. Megatron uses the inverted
    # convention vs HF (True = mask OUT) so handing the HF attention mask through would be wrong.
    attention_mask = None

    fwd_bwd = get_forward_backward_func()
    iterator = SingleBatchIterator(input_ids_padded, position_ids, attention_mask)
    with torch.no_grad():
        out = fwd_bwd(
            forward_step_func=forward_step,
            data_iterator=iterator,
            model=megatron_model,
            num_microbatches=1,
            forward_only=True,
            seq_length=padded_seq_len,
            micro_batch_size=1,
            collect_non_loss_data=True,
        )

    if isinstance(out, list) and len(out) > 0:
        out = out[0]

    is_last_stage = not dist.is_initialized() or parallel_state.is_pipeline_last_stage()
    if not is_last_stage:
        return None

    # All-gather vocab-sharded logits along the last dim.
    if dist.is_initialized() and parallel_state.get_tensor_model_parallel_world_size() > 1:
        world = parallel_state.get_tensor_model_parallel_world_size()
        gathered = [torch.zeros_like(out) for _ in range(world)]
        dist.all_gather(gathered, out, group=parallel_state.get_tensor_model_parallel_group())
        out = torch.cat(gathered, dim=2)

    # Save logits at both the last real (semantically meaningful) and last padded position.
    # compare.py samples at the last padded position; that's a useful sanity check for parity
    # against an HF reference that processes the same padded input.
    last_real_idx = orig_seq_len - 1
    last_real_logits = out[0, last_real_idx, :].float().cpu()
    last_padded_logits = out[0, -1, :].float().cpu()
    # `last_pos_logits` retained as the primary field for back-compat with the comparator.
    last_pos_logits = last_real_logits
    vocab_size = int(last_pos_logits.shape[0])

    top_vals, top_ids = torch.topk(last_real_logits, k=top_k)
    top_token_strs = [tokenizer.decode([tid.item()]) for tid in top_ids]
    top_padded_vals, top_padded_ids = torch.topk(last_padded_logits, k=top_k)
    top_padded_strs = [tokenizer.decode([tid.item()]) for tid in top_padded_ids]

    # Only the global rank-0 of the last PP stage writes results back to the caller; other
    # ranks in the same stage have the same data but we only need one copy.
    if dist.is_initialized() and (
        parallel_state.get_tensor_model_parallel_rank() != 0 or parallel_state.get_expert_model_parallel_rank() != 0
    ):
        return None

    return {
        "prompt": prompt_text,
        "input_ids": input_ids[0].cpu(),
        "seq_len": int(orig_seq_len),
        "padded_seq_len": padded_seq_len,
        "last_pos_logits": last_pos_logits,
        "last_real_logits": last_real_logits,
        "last_padded_logits": last_padded_logits,
        "vocab_size": vocab_size,
        "top_k": top_k,
        "top_ids": top_ids.cpu(),
        "top_logits": top_vals.float().cpu(),
        "top_strs": top_token_strs,
        "top_padded_ids": top_padded_ids.cpu(),
        "top_padded_strs": top_padded_strs,
    }


def _is_writer_rank() -> bool:
    """Return True if this rank should write the output artifact.

    With PP>1 the logits live on the last PP stage, not on global rank 0. We pick the rank
    that holds the data: last PP stage, TP rank 0, EP rank 0. Falls back to global rank 0
    for single-process runs.
    """
    if not dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0)) == 0
    return (
        parallel_state.is_pipeline_last_stage()
        and parallel_state.get_tensor_model_parallel_rank() == 0
        and parallel_state.get_expert_model_parallel_rank() == 0
    )


def main() -> None:
    """Drive the Megatron-Bridge forward pass over the prompt set and save artifacts."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s rank=%(name)s] %(message)s")
    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    prompts = load_prompts(args.prompts_file)
    print_rank_0(f"Loaded {len(prompts)} prompts from {args.prompts_file}")

    print_rank_0(f"Loading tokenizer from {args.hf_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    megatron_model, _bridge = load_megatron_model(args)
    print_rank_0("Megatron model loaded; starting forward passes")

    pad_multiple = args.pad_to_multiple_of if args.pad_to_multiple_of is not None else args.tp
    print_rank_0(f"pad_to_multiple_of={pad_multiple}")

    results: list[dict] = []
    for i, item in enumerate(prompts):
        prompt_id = item.get("id", str(i))
        prompt_text = item["text"]
        result = run_one_prompt(
            megatron_model=megatron_model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            pad_multiple=pad_multiple,
            max_seq_len=args.max_seq_len,
            top_k=args.top_k,
            pad_token_id=args.pad_token_id,
        )
        if result is not None:
            result["id"] = prompt_id
            results.append(result)
            top1 = int(result["top_ids"][0].item())
            top1_str = result["top_strs"][0]
            top1_pad = int(result["top_padded_ids"][0].item())
            top1_pad_str = result["top_padded_strs"][0]
            # Print directly (not via print_rank_0) — under PP>1 the writer rank is not global rank 0.
            print(
                f"[{i + 1}/{len(prompts)}] id={prompt_id} seq_len={result['seq_len']} "
                f"real_top1={top1} ({top1_str!r}) padded_top1={top1_pad} ({top1_pad_str!r}) "
                f"real_mean={result['last_real_logits'].mean():.4f} "
                f"padded_mean={result['last_padded_logits'].mean():.4f}",
                flush=True,
            )

        # Keep all ranks lock-step so collective ops in the next iteration don't deadlock.
        if dist.is_initialized():
            dist.barrier()

    if _is_writer_rank():
        artifact = {
            "source": "megatron_bridge",
            "model_path": args.hf_model_path,
            "dtype": "bfloat16",
            "tp": args.tp,
            "pp": args.pp,
            "ep": args.ep,
            "etp": args.etp,
            "results": results,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, out_path)
        print_rank_0(f"Saved {len(results)} results to {out_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
