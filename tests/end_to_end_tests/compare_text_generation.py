#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
End to end test that compares text generated from a HuggingFace model
using the following methods:
    - Generate with Transformers
    - Load a Megatron model directly from the HuggingFace weights and
        generate text from it (AutoBridge.from_hf_pretrained method)
    - Convert the HuggingFace weights to a Megatron checkpoint,
        load that checkpoint, and generate text from it (AutoBridge.import_ckpt method)
    - Export the Megatron checkpoint (that was previously imported from HF) back
        to HuggingFace format and generate using Transformers again, as a sanity test.

Supports loading the Megatron model with different parallelisms.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer import MegatronModule
from transformers import AutoModelForCausalLM, AutoTokenizer

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.training.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer, build_tokenizer
from megatron.bridge.utils.common_utils import get_last_rank, get_local_rank_preinit, print_rank_0


def _safe_init_distributed():
    if not torch.distributed.is_initialized():
        torch.cuda.set_device(get_local_rank_preinit())
        torch.distributed.init_process_group("nccl")


def _safe_destroy_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def validate_path(path: str, must_exist: bool = False) -> Path:
    """Validate and convert string path to Path object."""
    path_obj = Path(path)
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    return path_obj


def transformers_generate(hf_model: str, prompt: str, max_new_tokens: int = 20) -> tuple(
    list[str], tuple(torch.Tensor)
):
    """
    Generate text from a HuggingFace model using transformers.

    This serves as the baseline for this test.

    Args:
        model_id: HuggingFace model ID or path to model directory
        prompt: Input text for the model
        max_new_tokens: Upper bound on how many tokens to generate.
           May generate fewer tokens than this limit. (default: 20)
    """
    print_rank_0(f"Loading model and tokenizer: {hf_model}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True).cuda()

    print_rank_0("Generating text using HF weights...")
    in_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **in_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_text = tokenizer.decode(output.sequences[0])

    print_rank_0("====== HF GENERATED TEXT OUTPUT ======")
    print_rank_0(f"Prompt: {prompt}")
    print_rank_0(f"Generated: {generated_text}")
    print_rank_0("======================================")
    return generated_text, output.scores


class SingleBatchIterator:
    """Iterator that yields a single batch of data for text generation.
    Required by the forward_backward_func function.

    This class creates an iterator that yields exactly one batch containing
    input tokens, position IDs, and attention mask, then raises StopIteration.
    Used for single-step inference in the forward pass.
    """

    def __init__(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def text_forward_step(data_iterator: SingleBatchIterator, model: MegatronModule, **kwargs) -> torch.Tensor:
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
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def megatron_generate(
    megatron_model: list[MegatronModule], tokenizer: MegatronTokenizer, prompt: str, max_new_tokens: int = 20
) -> tuple(list[str], list[torch.Tensor]):
    """
    Generate text from a Megatron model using MCore.

    For the purpose of this test, the model may be created either from AutoBridge
    or by loading an HF->Megatron converted checkpoint.

    Args:
        megatron_model: The loaded Megatron model to generate from
        tokenizer: Tokenizer to use on the prompt with the Megatron model
        prompt: Input text for the model
        max_new_tokens: Upper bound on how many tokens to generate.
           May generate fewer tokens than this limit. (default: 20)
    """
    print_rank_0("Moving model to GPU and setting to eval mode.")
    megatron_model = [m.cuda() for m in megatron_model]
    for m in megatron_model:
        m.eval()

    print_rank_0("Tokenizing input prompt.")
    input_ids = torch.tensor(tokenizer.tokenize(prompt)).unsqueeze(0).cuda()
    generated_ids = input_ids.clone()
    num_input_tokens = input_ids.shape[1]

    generated_logits = []

    print_rank_0("Generating text from Megatron model...")
    for step in range(max_new_tokens):
        print_rank_0(f"Generation step: {step}")
        with torch.no_grad():
            # recreate iterator after each generation
            position_ids = (
                torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )
            iterator = SingleBatchIterator(input_ids, position_ids)

            fwd_bwd_function = get_forward_backward_func()
            output = fwd_bwd_function(
                forward_step_func=text_forward_step,
                data_iterator=iterator,
                model=megatron_model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(-1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            # only generating for 1 batch at a time
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            # gather and concat output tensor across tp ranks.
            # then identify most likely next token
            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                torch.distributed.all_gather(
                    gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group()
                )
                output = torch.cat(gathered_tensors, dim=2)
                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

                # save for comparison
                generated_logits.append(output[0, -1].cpu())
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() in [tokenizer.eos_id, tokenizer.eod_id]:
                break

    num_generated_tokens = generated_ids.shape[1] - num_input_tokens
    if not parallel_state.is_pipeline_last_stage():
        generated_logits = [None for _ in range(num_generated_tokens)]
    torch.distributed.broadcast_object_list(
        generated_logits,
        group=parallel_state.get_pipeline_model_parallel_group(),
        group_src=parallel_state.get_pipeline_model_parallel_last_rank(),
    )

    generated_text = tokenizer.detokenize(generated_ids[0])

    print_rank_0("====== MEGATRON GENERATED TEXT OUTPUT ======")
    print_rank_0(f"Prompt: {prompt}")
    print_rank_0(f"Generated: {generated_text}")
    print_rank_0("============================================")
    return generated_text, generated_logits


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def import_hf_to_megatron(
    hf_model: str,
    megatron_path: str,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
) -> None:
    """
    Import a HuggingFace model and save it as a Megatron checkpoint.

    Args:
        hf_model: HuggingFace model ID or path to model directory
        megatron_path: Directory path where the Megatron checkpoint will be saved
        torch_dtype: Model precision ("float32", "float16", "bfloat16")
        device_map: Device placement strategy ("auto", "cuda:0", etc.)
        trust_remote_code: Allow custom model code execution
    """
    print_rank_0(f"🔄 Starting import: {hf_model} -> {megatron_path}")

    # Prepare kwargs
    kwargs = {}
    if torch_dtype:
        kwargs["torch_dtype"] = get_torch_dtype(torch_dtype)
        print_rank_0(f"   Using torch_dtype: {torch_dtype}")

    if device_map:
        kwargs["device_map"] = device_map
        print_rank_0(f"   Using device_map: {device_map}")

    # Import using the convenience method
    print_rank_0(f"📥 Loading HuggingFace model: {hf_model}")
    AutoBridge.import_ckpt(
        hf_model_id=hf_model,
        megatron_path=megatron_path,
        trust_remote_code=True,
        **kwargs,
    )

    print_rank_0(f"✅ Successfully imported model to: {megatron_path}")

    # Destroy model parallel process groups created by import ckpt
    parallel_state.destroy_model_parallel()


def export_megatron_to_hf(
    hf_model: str,
    megatron_path: str,
    hf_path: str,
) -> None:
    """
    Export a Megatron checkpoint to HuggingFace format.

    Args:
        hf_model: HuggingFace model ID or path to model directory
        megatron_path: Directory path where the Megatron checkpoint is stored
        hf_path: Directory path where the HuggingFace model will be saved
    """
    print_rank_0(f"🔄 Starting export: {megatron_path} -> {hf_path}")

    # Validate megatron checkpoint exists
    checkpoint_path = validate_path(megatron_path, must_exist=True)
    print_rank_0(f"📂 Found Megatron checkpoint: {checkpoint_path}")

    bridge = AutoBridge.from_hf_pretrained(hf_model)
    print_rank_0("📤 Exporting to HuggingFace format...")
    bridge.export_ckpt(
        megatron_path=megatron_path,
        hf_path=hf_path,
    )

    print_rank_0(f"✅ Successfully exported model to: {hf_path}")

    # Destroy model parallel process groups created by export ckpt
    parallel_state.destroy_model_parallel()


def megatron_generate_from_checkpoint(
    megatron_path: str, prompt: str, max_new_tokens: int = 20, tp: int = 1, pp: int = 1, ep: int = 1, etp: int = 1
) -> tuple(list[str], list[torch.Tensor]):
    """
    Generate text from a Megatron checkpoint.

    For the purpose of this test, megatron_path should be a checkpoint
    imported from HuggingFace using AutoBridge.
    This function is just a wrapper around megatron_generate(), with some setup.

    Args:
        megatron_path: Path to the Megatron checkpoint directory. Should contain
            at least one 'iter_#######' directory which is a dist ckpt.
        prompt: Input prompt for the model
        max_new_tokens: Upper bound on how many tokens to generate.
           May generate fewer tokens than this limit. (default: 20)
        tp: Tensor parallelism size override. (default: 1)
        pp: Pipeline parallelism size override. (default: 1)
        ep: Expert parallelism size override. (default: 1)
        etp: Expert tensor parallelism size override. (default: 1)
    """
    from megatron.bridge.training.model_load_save import build_and_load_model, load_model_config, load_tokenizer

    checkpoint_path = Path(megatron_path)
    # Check for iter_* folders
    iter_folders = [f for f in checkpoint_path.iterdir() if f.is_dir() and f.name.startswith("iter_")]
    if iter_folders:
        # Find the folder with the largest iteration number
        def get_iter_number(folder_name):
            try:
                return int(folder_name.replace("iter_", ""))
            except ValueError:
                return -1  # Invalid format, put at the end

        latest_iter = max(iter_folders, key=lambda f: get_iter_number(f.name))
        checkpoint_path = checkpoint_path / latest_iter.name
    # else: checkpoint_path remains as the input path (no iter folders found), and we assume it is a dist ckpt

    print_rank_0(f"Loading Megatron model from checkpoint at {checkpoint_path}")

    # Override parallelisms in model config
    model_provider, _ = load_model_config(str(checkpoint_path))
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp

    # Initialize parallel state
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    # Initialize and load the model and tokenizer
    megatron_model = build_and_load_model(str(checkpoint_path), model_provider)
    tokenizer = load_tokenizer(str(checkpoint_path))

    generated = megatron_generate(megatron_model, tokenizer, prompt, max_new_tokens)

    # each tested generation method should recreate this, since
    # user will likely use these methods in isolation
    parallel_state.destroy_model_parallel()

    return generated


def megatron_generate_from_hf(
    hf_model: str, prompt: str, max_new_tokens: int = 20, tp: int = 1, pp: int = 1, ep: int = 1, etp: int = 1
) -> tuple(list[str], list[torch.Tensor]):
    """
    Generate text from a Megatron model with weights loaded from HuggingFace using AutoBridge.

    This function is just a wrapper around megatron_generate(), with some setup.

    Args:
        hf_model: HuggingFace model ID or path to model directory
        prompt: Input prompt for the model
        max_new_tokens: Upper bound on how many tokens to generate.
           May generate fewer tokens than this limit. (default: 20)
        tp: Tensor parallelism size override. (default: 1)
        pp: Pipeline parallelism size override. (default: 1)
        ep: Expert parallelism size override. (default: 1)
        etp: Expert tensor parallelism size override. (default: 1)
    """

    print_rank_0(f"Loading HuggingFace model from: {hf_model}")

    # Get model config from HF
    bridge = AutoBridge.from_hf_pretrained(hf_model, trust_remote_code=True)
    model_provider = bridge.to_megatron_provider(load_weights=True)

    # Override parallelisms
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp

    # Initialize parallel state
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    # Initialize and load the model and tokenizer
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)
    tokenizer_cfg = TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=hf_model)
    tokenizer = build_tokenizer(tokenizer_cfg, trust_remote_code=True)

    generated = megatron_generate(megatron_model, tokenizer, prompt, max_new_tokens)

    # each tested generation method should recreate this, since
    # user will likely use these methods in isolation
    parallel_state.destroy_model_parallel()

    return generated


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Compare text generated through various Megatron-Bridge conversion methods against HuggingFace direct"
    )

    # Common arguments for generation
    parser.add_argument("--hf-model-id", type=str, required=True, help="Repo or path to the HuggingFace model.")
    parser.add_argument("--prompt", type=str, default="What is a GPU?", help="Input prompt for text generation.")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Maximum number of new tokens to generate.")

    # Import arguments
    parser.add_argument(
        "--megatron-path", required=True, help="Directory path where the Megatron checkpoint will be saved."
    )
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], help="Model precision")
    parser.add_argument("--device-map", help='Device placement strategy (e.g., "auto", "cuda:0")')

    # Export arguments
    parser.add_argument(
        "--hf-save-path", required=True, help="Directory path where the export HuggingFace checkpoint will be saved."
    )
    # Parallelism settings
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")

    return parser.parse_args()


def compare_text(
    hf_text_preconvert: list[str],
    megatron_text_fromhf: list[str],
    megatron_text_fromckpt: list[str],
    hf_text_postconvert: list[str],
):
    # TODO: implement
    pass


def compare_logits(
    hf_logits_preconvert: list[torch.Tensor],
    megatron_logits_fromhf: list[torch.Tensor],
    megatron_logits_fromckpt: list[torch.Tensor],
    hf_logits_postconvert: list[torch.Tensor],
):
    # TODO: implement
    pass


def main():
    _safe_init_distributed()

    args = parse_cli_args()

    hf_text_preconvert, hf_logits_preconvert = transformers_generate(
        args.hf_model_id, args.prompt, args.max_new_tokens
    )

    megatron_text_fromhf, megatron_logits_fromhf = megatron_generate_from_hf(
        args.hf_model_id, args.prompt, args.max_new_tokens, args.tp, args.pp, args.ep, args.etp
    )

    # Generate from imported checkpoint
    import_hf_to_megatron(args.hf_model_id, args.megatron_path, args.torch_dtype, args.device_map)
    megatron_text_fromckpt, megatron_logits_fromckpt = megatron_generate_from_checkpoint(
        args.megatron_path, args.prompt, args.max_new_tokens, args.tp, args.pp, args.ep, args.etp
    )

    # Generate from exported checkpoint
    _safe_destroy_distributed()  # Export happens on cpu-only and creates new gloo groups
    export_megatron_to_hf(args.hf_model_id, args.megatron_path, args.hf_save_path)
    hf_text_postconvert, hf_logits_postconvert = transformers_generate(
        args.hf_save_path, args.prompt, args.max_new_tokens
    )

    # Compare results
    compare_text(hf_text_preconvert, megatron_text_fromhf, megatron_text_fromckpt, hf_text_postconvert)
    compare_logits(hf_logits_preconvert, megatron_logits_fromhf, megatron_logits_fromckpt, hf_logits_postconvert)


if __name__ == "__main__":
    main()
