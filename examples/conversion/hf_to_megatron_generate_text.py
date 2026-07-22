# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Example:
  # Load from HuggingFace model:
  uv run python examples/conversion/hf_to_megatron_generate_text.py --hf_model_path="meta-llama/Llama-3.2-1B" --prompt="Hello, how are you?"

  # Load from Megatron checkpoint:
  uv run python examples/conversion/hf_to_megatron_generate_text.py --hf_model_path="meta-llama/Llama-3.2-1B" --megatron_model_path="/path/to/megatron/checkpoint" --prompt="Hello, how are you?"
"""

import argparse

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import disable_mtp_for_inference, get_last_rank, print_rank_0


class SingleBatchIterator:
    """Iterator that yields a single batch of data for text generation.
    Required by the forward_backward_func function.

    This class creates an iterator that yields exactly one batch containing
    input tokens, position IDs, and attention mask, then raises StopIteration.
    Used for single-step inference in the forward pass.
    """

    def __init__(self, input_ids, position_ids, inference_context=None):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
            inference_context=inference_context,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def _hf_revision_kwargs(revision: str | None) -> dict[str, str]:
    """Return an optional immutable Hugging Face revision argument."""
    return {"revision": revision} if revision is not None else {}


def text_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step function for text generation.
    Required by the forward_backward_func function.

    Extracts a batch from the data iterator and runs the model forward pass
    with the provided input tokens, position IDs, and attention mask.

    Args:
        data_iterator: Iterator providing batches of input data
        model: The Megatron model to run forward pass on
        **kwargs: Additional keyword arguments (unused)

    Returns:
        Tuple of (model_output, loss_function)
    """
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
        "inference_context": batch["inference_context"],
        "runtime_gather_output": True,
    }

    def loss_func(x, **kwargs):
        return x

    model_output = model(**forward_args)
    if isinstance(model_output, tuple):
        model_output = model_output[0]
    return model_output, loss_func


def _run_megatron_forward(fwd_bwd_function, **kwargs):
    """Run a Megatron forward pass with inference execution paths active."""
    with InferenceMode.active():
        return fwd_bwd_function(**kwargs)


def _maybe_gather_tensor_parallel_logits(output, full_vocab_size: int, world_size: int, group):
    """Gather sharded TP logits without gathering an already complete vocabulary."""
    if output.size(-1) >= full_vocab_size:
        return output

    gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, output, group=group)
    gathered_output = torch.cat(gathered_tensors, dim=2)
    if gathered_output.size(-1) < full_vocab_size:
        raise ValueError(
            f"Gathered Megatron vocabulary ({gathered_output.size(-1)}) is smaller than "
            f"the configured vocabulary ({full_vocab_size})."
        )
    return gathered_output


def _tokenize_prompt(tokenizer, prompt: str, *, apply_chat_template: bool, thinking_mode: str) -> torch.Tensor:
    """Tokenize a raw prompt, optionally formatting it as a user chat turn."""
    if not apply_chat_template:
        return tokenizer.encode(prompt, return_tensors="pt")

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        thinking_mode=thinking_mode,
    )
    return encoded["input_ids"]


def _decode_completion(tokenizer, generated_ids: torch.Tensor, prompt_length: int) -> str:
    """Decode generated tokens without echoing the prompt or special tokens."""
    return tokenizer.decode(
        generated_ids[0, prompt_length:].tolist(),
        skip_special_tokens=True,
    )


def main(args) -> None:
    """Main function for text generation from HuggingFace or Megatron models.

    Loads a model either from HuggingFace (with optional conversion to Megatron)
    or directly from a Megatron checkpoint, then performs greedy text generation
    using the provided prompt.

    Args:
        args: Parsed command line arguments containing model paths, prompt,
              parallelism settings, and generation parameters
    """
    # pylint: disable=C0115,C0116
    tp = args.tp
    pp = args.pp
    ep = args.ep
    etp = args.etp

    # Choose loading method based on arguments
    if args.megatron_model_path:
        # Load from Megatron checkpoint
        print_rank_0(f"Loading Megatron model from: {args.megatron_model_path}")

        # We still need HF config for tokenizer, but we'll load the model from Megatron checkpoint
        # Create bridge from HF config only (no weights)
        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=args.trust_remote_code,
                hf_path=args.hf_model_path,
            ),
            **_hf_revision_kwargs(args.hf_revision),
        )

        # Initialize model parallel before loading
        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16

        # Read pipeline layout from checkpoint for PP > 1
        if pp > 1:
            from pathlib import Path

            import yaml

            ckpt_path = Path(args.megatron_model_path)
            for candidate in [ckpt_path, *ckpt_path.glob("iter_*")]:
                rc = candidate / "run_config.yaml"
                if rc.exists():
                    with open(rc) as f:
                        cfg = yaml.safe_load(f)
                    saved_layout = cfg.get("model", {}).get("pipeline_model_parallel_layout")
                    if isinstance(saved_layout, list):
                        model_provider.pipeline_model_parallel_layout = saved_layout
                        break

        # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)

        # Load the Megatron model directly
        model = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides={
                "tensor_model_parallel_size": tp,
                "pipeline_model_parallel_size": pp,
                "expert_model_parallel_size": ep,
                "expert_tensor_parallel_size": etp,
                "pipeline_dtype": torch.bfloat16,
            },
            wrap_with_ddp=False,
        )

    else:
        # Load from HuggingFace and convert to Megatron
        print_rank_0(f"Loading HuggingFace model from: {args.hf_model_path}")
        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=args.trust_remote_code,
                hf_path=args.hf_model_path,
            ),
            **_hf_revision_kwargs(args.hf_revision),
        )
        model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16

        # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()
        disable_mtp_for_inference(m)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path,
        trust_remote_code=is_safe_repo(
            trust_remote_code=args.trust_remote_code,
            hf_path=args.hf_model_path,
        ),
        **_hf_revision_kwargs(args.hf_revision),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the input prompt
    prompt = args.prompt
    input_ids = _tokenize_prompt(
        tokenizer,
        prompt,
        apply_chat_template=args.apply_chat_template,
        thinking_mode=args.thinking_mode,
    ).cuda()
    prompt_length = input_ids.size(1)
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    generated_ids = input_ids.clone()

    stop_tokens = [tokenizer.eos_token_id]

    # Greedy generation loop
    for step in range(args.max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step}")

            fwd_bwd_function = get_forward_backward_func()
            inference_context = StaticInferenceContext(
                max_batch_size=input_ids.size(0),
                max_sequence_length=input_ids.size(1),
            )
            iterator = SingleBatchIterator(input_ids, position_ids, inference_context)

            output = _run_megatron_forward(
                fwd_bwd_function,
                forward_step_func=text_forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                output = _maybe_gather_tensor_parallel_logits(
                    output,
                    model_provider.vocab_size,
                    world_size,
                    parallel_state.get_tensor_model_parallel_group(),
                )
                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

                # Debug: print token information
                if step < 5:  # Only for first few iterations
                    print_rank_0(f"Step {step}: output shape={output.shape}, var={output.var():.4f}")
                    logits = output[0, -1, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
                    print_rank_0(f"Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")
                    print_rank_0(
                        f"Selected: '{tokenizer.decode([next_token_ids.item()])}' (id={next_token_ids.item()})"
                    )
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() in stop_tokens:
                break

    # Decode only the completion. Passing CUDA tensor objects directly to the
    # tokenizer can produce corrupt text with some remote-code tokenizers.
    generated_text = _decode_completion(tokenizer, generated_ids, prompt_length)
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    print_rank_0(f"Prompt: {prompt}")
    print_rank_0(f"Generated: {generated_text}")
    print_rank_0("=======================================")


def build_parser() -> argparse.ArgumentParser:
    """Build the text-generation CLI parser."""
    parser = argparse.ArgumentParser(description="Text Generation from HuggingFace Models")
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace model.",
    )
    parser.add_argument(
        "--hf-revision",
        help="Immutable Hugging Face Hub revision used for config and tokenizer loading.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Input prompt for text generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Format the prompt as a user turn using the tokenizer's chat template.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("enabled", "adaptive", "disabled"),
        default="adaptive",
        help="Thinking mode passed to the chat template when --apply-chat-template is set.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument("--megatron_model_path", type=str, default=None, help="Path to the Megatron model checkpoint")
    parser.add_argument("--trust-remote-code", action="store_true", help="if trust_remote_code")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
