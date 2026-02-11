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
Audio-Language Model Generation Script for Qwen2-Audio.

This script demonstrates how to use Qwen2-Audio models with Megatron-Bridge
for audio understanding tasks.

Example:
  # Audio-Language generation with audio from URL:
  uv run python examples/conversion/hf_to_megatron_generate_alm.py \
    --hf_model_path="Qwen/Qwen2-Audio-7B-Instruct" \
    --audio_url="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3" \
    --prompt="What's that sound?"

  # Audio-Language generation with local audio file:
  uv run python examples/conversion/hf_to_megatron_generate_alm.py \
    --hf_model_path="Qwen/Qwen2-Audio-7B-Instruct" \
    --audio_path="/path/to/audio.wav" \
    --prompt="Describe what you hear in this audio."

  # Text-only generation (no audio):
  uv run python examples/conversion/hf_to_megatron_generate_alm.py \
    --hf_model_path="Qwen/Qwen2-Audio-7B-Instruct" \
    --prompt="Hello, how are you?"

  # Load from Megatron checkpoint:
  uv run python examples/conversion/hf_to_megatron_generate_alm.py \
    --hf_model_path="Qwen/Qwen2-Audio-7B-Instruct" \
    --megatron_model_path="/path/to/megatron/checkpoint" \
    --audio_url="https://example.com/audio.mp3" \
    --prompt="What's in this audio?"
"""

import argparse
from io import BytesIO
from typing import Optional
from urllib.request import urlopen

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoProcessor, AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


# Try to import librosa for audio loading
try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    librosa = None
    HAS_LIBROSA = False


class SingleBatchIterator:
    """Iterator that yields a single batch of data for audio-language generation.
    Required by the forward_backward_func function.

    This class creates an iterator that yields exactly one batch containing
    input tokens, position IDs, attention mask, and optional audio inputs,
    then raises StopIteration. Used for single-step inference in the forward pass.
    """

    def __init__(
        self,
        input_ids,
        position_ids,
        attention_mask,
        input_features=None,
        feature_attention_mask=None,
    ):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        # Add audio inputs if provided
        if input_features is not None:
            self.batch["input_features"] = input_features
        if feature_attention_mask is not None:
            self.batch["feature_attention_mask"] = feature_attention_mask

        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def alm_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step function for audio-language generation.
    Required by the forward_backward_func function.

    Extracts a batch from the data iterator and runs the model forward pass
    with the provided input tokens, position IDs, attention mask, and audio inputs.

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

    # Add audio inputs if present
    if "input_features" in batch:
        forward_args["input_features"] = batch["input_features"]
    if "feature_attention_mask" in batch:
        forward_args["feature_attention_mask"] = batch["feature_attention_mask"]

    def loss_func(x, **kwargs):
        return x

    model_output = model(**forward_args)
    if isinstance(model_output, tuple):
        output_tensor, _ = model_output
    else:
        output_tensor = model_output

    return output_tensor, loss_func


def load_audio(audio_path: str, sampling_rate: int = 16000):
    """Load an audio file from URL or file path.

    Args:
        audio_path: URL or local file path to the audio file
        sampling_rate: Target sampling rate for the audio

    Returns:
        Audio data as numpy array
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa is required for audio loading. Please install it: pip install librosa")

    if audio_path.startswith(("http://", "https://")):
        audio_data, _ = librosa.load(BytesIO(urlopen(audio_path).read()), sr=sampling_rate)
    else:
        audio_data, _ = librosa.load(audio_path, sr=sampling_rate)

    return audio_data


def process_audio_inputs(processor, audio_path: Optional[str], prompt: str):
    """Process audio inputs for audio-language model.

    Args:
        processor: AutoProcessor for the audio-language model
        audio_path: Path or URL to the audio file (optional)
        prompt: Text prompt

    Returns:
        Tuple of (input_ids, input_features, feature_attention_mask, messages)
    """
    if audio_path:
        # Get sampling rate from processor
        sampling_rate = processor.feature_extractor.sampling_rate

        # Load audio
        audio_data = load_audio(audio_path, sampling_rate)

        # Create messages with audio and text for Qwen2-Audio format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Process inputs with audio
        inputs = processor(text=text, audio=[audio_data], return_tensors="pt", padding=True)

        return (
            inputs.input_ids,
            inputs.input_features,
            getattr(inputs, "feature_attention_mask", None),
            messages,
        )
    else:
        # Text-only processing
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, return_tensors="pt")
        return inputs.input_ids, None, None, messages


def main(args) -> None:
    """Main function for audio-language generation from HuggingFace models.

    Loads an audio-language model either from HuggingFace (with optional conversion to Megatron)
    or directly from a Megatron checkpoint, then performs greedy generation
    using the provided prompt and optional audio input.

    Args:
        args: Parsed command line arguments containing model paths, prompt,
              audio path, parallelism settings, and generation parameters
    """
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
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path)

        # Initialize model parallel before loading
        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
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
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path)
        model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()

    # Set grad_scale_func to None on the model's config for inference
    for m in model:
        if hasattr(m, "config"):
            m.config.grad_scale_func = None

    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path,
        trust_remote_code=is_safe_repo(
            trust_remote_code=args.trust_remote_code,
            hf_path=args.hf_model_path,
        ),
    )
    processor = AutoProcessor.from_pretrained(
        args.hf_model_path,
        trust_remote_code=is_safe_repo(
            trust_remote_code=args.trust_remote_code,
            hf_path=args.hf_model_path,
        ),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine audio path (URL or file)
    audio_path = args.audio_url or args.audio_path

    # Process inputs (text and audio if provided)
    prompt = args.prompt
    input_ids, input_features, feature_attention_mask, messages = process_audio_inputs(processor, audio_path, prompt)

    # Move to GPU
    input_ids = input_ids.cuda()
    if input_features is not None:
        input_features = input_features.cuda()
    if feature_attention_mask is not None:
        feature_attention_mask = feature_attention_mask.cuda()

    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    generated_ids = input_ids.clone()

    stop_tokens = [tokenizer.eos_token_id]

    # Greedy generation loop
    for step in range(args.max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step}")

            fwd_bwd_function = get_forward_backward_func()

            # Keep passing audio inputs for all steps to ensure audio features are available
            iterator = SingleBatchIterator(
                input_ids, position_ids, attention_mask, input_features, feature_attention_mask
            )

            output = fwd_bwd_function(
                forward_step_func=alm_forward_step,
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
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                # All-gather operation
                dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
                # Concatenate along last dimension (dim=2)
                output = torch.cat(gathered_tensors, dim=2)
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
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() in stop_tokens:
                break

    # Decode the generated sequence
    generated_text = tokenizer.decode(list(generated_ids[0]), skip_special_tokens=True)
    print_rank_0("======== GENERATED TEXT OUTPUT ========")
    if audio_path:
        print_rank_0(f"Audio: {audio_path}")
    print_rank_0(f"Prompt: {prompt}")
    print_rank_0(f"Generated: {generated_text}")
    print_rank_0("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-Language Generation from HuggingFace Audio-Language Models")
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace audio-language model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What's that sound?",
        help="Input prompt for audio-language generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument("--megatron_model_path", type=str, default=None, help="Path to the Megatron model checkpoint")
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Local path to the audio file for audio-language generation (optional).",
    )
    parser.add_argument(
        "--audio_url",
        type=str,
        default=None,
        help="URL to the audio file for audio-language generation (optional).",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="if trust_remote_code")
    args = parser.parse_args()

    main(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
