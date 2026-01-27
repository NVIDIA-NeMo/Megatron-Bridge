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
Model comparison utilities for comparing 1-step generation between HuggingFace and Megatron models.

This module provides utilities to compare the forward pass outputs between HuggingFace models
and their Megatron equivalents, supporting both text-only and vision-language models.

Run Script Examples:
    # Regular LLM comparison between HF and Megatron models:
    uv run python examples/conversion/compare_hf_and_megatron/compare.py \
        --hf_model_path "Qwen/Qwen3-1.7B" \
        --prompt "Hello, how are you?"


    # Vision-language comparison with image from URL:
    uv run python examples/conversion/compare_hf_and_megatron/compare.py \
        --hf_model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
        --model_class "Qwen2_5_VLForConditionalGeneration" \
        --image_path "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
        --prompt "Describe this image."

    # Vision-language comparison with local image:
    uv run python examples/conversion/compare_hf_and_megatron/compare.py \
        --hf_model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
        --model_class "Qwen2_5_VLForConditionalGeneration" \
        --image_path "/path/to/local/image.jpg" \
        --prompt "What do you see in this image?"

    # Multi-GPU comparison with tensor parallelism (regular LLM):
    uv run python -m torch.distributed.run --nproc_per_node=2 examples/conversion/compare_hf_and_megatron/compare.py \
        --hf_model_path "Qwen/Qwen3-1.7B" \
        --prompt "Hello world" \
        --tp 2

    # Pipeline parallel comparison (VL model):
    uv run python -m torch.distributed.run --nproc_per_node=2 examples/conversion/compare_hf_and_megatron/compare.py \
        --hf_model_path "Qwen/Qwen2.5-VL-3B-Instruct" \
        --model_class "Qwen2_5_VLForConditionalGeneration" \
        --prompt "Hello world" \
        --pp 2

    # Compare with pre-converted Megatron checkpoint:
    uv run python examples/conversion/compare_hf_and_megatron/compare.py \
        --hf_model_path "Qwen/Qwen3-1.7B" \
        --megatron_model_path "/path/to/megatron/checkpoint" \
        --prompt "Hello world"

    # Enable debug hooks to inspect forward pass intermediate results:
    uv run python examples/conversion/compare_hf_and_megatron/compare.py \
        --hf_model_path "Qwen/Qwen3-1.7B" \
        --prompt "Hello world" \
        --enable_debug_hooks

Available Arguments:
    --hf_model_path: HuggingFace model path/name (required)
    --prompt: Text prompt for generation (required)
    --image_path: Path or URL to image for VL models (optional)
    --model_class: Specific HuggingFace model class (e.g., 'Qwen2_5_VLForConditionalGeneration' for VL models, optional)
    --megatron_model_path: Path to Megatron checkpoint (optional, converts from HF if not provided)
    --tp: Tensor parallelism size (default: 1)
    --pp: Pipeline parallelism size (default: 1)
    --ep: Expert parallelism size (default: 1)
    --etp: Expert tensor parallelism size (default: 1)
    --enable_debug_hooks: Enable debug hooks to log forward pass information for both models to JSONL files (optional)

Output:
    The script outputs detailed comparison metrics including:
    - Token predictions from both models
    - Logits statistics (mean, std)
    - Top-5 token predictions
    - Cosine similarity between logits
    - Absolute differences in logits

    When --enable_debug_hooks is used, additional JSONL files are generated:
    - hf_debug_fwd_log_<world_size>_rank_<rank>.jsonl: HuggingFace model forward pass logs
    - megatron_debug_component_<i>_fwd_log_<world_size>_rank_<rank>.jsonl: Megatron model component forward pass logs
    These files contain detailed information about inputs, outputs, and weights for each module during forward passes.
"""

import argparse
import importlib
import os
import sys
from typing import Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer


try:
    from qwen_vl_utils import process_vision_info

    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    process_vision_info = None
import os

# Import debugger module from same directory
import sys

import requests
from PIL import Image

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import get_last_rank, print_rank_0


# Cosine similarity threshold: require at least 98% similarity (2% tolerance)
SIMILARITY_THRESHOLD = 0.98


sys.path.append(os.path.dirname(__file__))
import debugger


def _is_rank_0() -> bool:
    """Check if current process is rank 0 for both distributed and non-distributed setups.

    Returns:
        True if current process is rank 0 or in single-process mode.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    elif "LOCAL_RANK" in os.environ:
        return int(os.environ.get("LOCAL_RANK", 0)) == 0
    return True


def load_model_class(model_class_name: str):
    """Dynamically import and return a model class from transformers.

    Args:
        model_class_name: Name of the model class (e.g., 'Qwen2_5_VLForConditionalGeneration')

    Returns:
        The model class

    Raises:
        ImportError: If the model class cannot be imported
    """
    try:
        # Try importing from transformers
        transformers_module = importlib.import_module("transformers")
        model_class = getattr(transformers_module, model_class_name)
        return model_class
    except AttributeError:
        # If not found in main transformers, try specific model modules
        try:
            # Try common model-specific modules
            if "qwen" in model_class_name.lower():
                module_name = "transformers.models.qwen2_vl"
            elif "llava" in model_class_name.lower():
                module_name = "transformers.models.llava"
            elif "blip" in model_class_name.lower():
                module_name = "transformers.models.blip_2"
            else:
                # Generic fallback - try to guess module name
                base_name = model_class_name.replace("ForConditionalGeneration", "").replace("ForCausalLM", "").lower()
                module_name = f"transformers.models.{base_name}"

            specific_module = importlib.import_module(module_name)
            model_class = getattr(specific_module, model_class_name)
            return model_class
        except (ImportError, AttributeError):
            raise ImportError(f"Could not import model class '{model_class_name}' from transformers")


def get_model_class(model_class_name: str = None, is_vl_model: bool = False):
    """Get the appropriate model class for loading.

    Args:
        model_class_name: Optional specific model class name
        is_vl_model: Whether this is a vision-language model

    Returns:
        Model class to use for loading
    """
    if model_class_name:
        print_rank_0(f"Using specified model class: {model_class_name}")
        return load_model_class(model_class_name)
    else:
        # Default behavior
        if is_vl_model:
            print_rank_0(
                "Warning: VL model detected but no model class specified. Using AutoModelForCausalLM which may not work."
            )
            print_rank_0(
                "Consider using --model_class argument (e.g., --model_class Qwen2_5_VLForConditionalGeneration)"
            )
        return AutoModelForCausalLM


def is_vision_language_model(model_path: str, trust_remote_code: bool | None = None) -> bool:
    """Check if the model is a vision-language model.

    Args:
        model_path: Path to the HuggingFace model

    Returns:
        True if the model supports vision inputs, False otherwise
    """
    try:
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=trust_remote_code,
                hf_path=model_path,
            ),
        )

        # Check for VL model indicators in config
        model_type = getattr(config, "model_type", "").lower()
        arch = getattr(config, "architectures", [])
        arch_str = " ".join(arch).lower() if arch else ""

        # Common patterns for VL models
        vl_indicators = [
            "vl",
            "vision",
            "multimodal",
            "clip",
            "blip",
            "flamingo",
            "llava",
            "qwen2_vl",
            "qwen_vl",
            "minicpm",
        ]

        return any(indicator in model_type or indicator in arch_str for indicator in vl_indicators)

    except Exception as e:
        print_rank_0(f"Warning: Could not determine model type from config: {e}")
        # Fallback: check if qwen_vl_utils is available and model name contains vl indicators
        return any(indicator in model_path.lower() for indicator in ["vl", "vision"])


class SingleBatchIterator:
    """Iterator that yields a single batch of data for text generation.
    Required by the forward_backward_func function.

    This class creates an iterator that yields exactly one batch containing
    input tokens, position IDs, attention mask, and optional vision inputs,
    then raises StopIteration. Used for single-step inference in the forward pass.
    """

    def __init__(self, input_ids, position_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        # Add vision inputs if provided
        if pixel_values is not None:
            self.batch["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            self.batch["image_grid_thw"] = image_grid_thw

        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def vlm_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step function for vision-language generation.
    Required by the forward_backward_func function.

    Extracts a batch from the data iterator and runs the model forward pass
    with the provided input tokens, position IDs, attention mask, and vision inputs.

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

    # Add vision inputs if present
    if "pixel_values" in batch:
        forward_args["pixel_values"] = batch["pixel_values"]
    if "image_grid_thw" in batch:
        forward_args["image_grid_thw"] = batch["image_grid_thw"]

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def load_image(image_path: str) -> Image.Image:
    """Load an image from URL or file path.

    Args:
        image_path: URL or local file path to the image

    Returns:
        PIL Image object
    """
    if image_path.startswith(("http://", "https://")):
        response = requests.get(image_path)
        response.raise_for_status()
        return Image.open(requests.get(image_path, stream=True).raw)
    else:
        return Image.open(image_path)


def pad_input_ids_to_tp_multiple(input_ids, tp_size: int, pad_token_id: int = 0):
    """Pad input_ids so sequence length is divisible by tp_size.

    this is needed for sequence parallel, which is required for moe models
    when using tensor parallel and expert parallel together.

    Args:
        input_ids: Input token IDs tensor
        tp_size: Tensor parallel size
        pad_token_id: Token ID to use for padding

    Returns:
        Padded input_ids tensor
    """
    seq_len = input_ids.shape[1]
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        padding = torch.full(
            (input_ids.shape[0], pad_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, padding], dim=1)
    return input_ids


def process_inputs(tokenizer, processor, image_path: Optional[str], prompt: str, is_vl_model: bool, tp_size: int = 1):
    """Process inputs for both vision-language and regular LLM models.

    Args:
        tokenizer: AutoTokenizer for the model
        processor: AutoProcessor for VL models (None for regular LLMs)
        image_path: Path or URL to the image (optional)
        prompt: Text prompt
        is_vl_model: Whether the model is a vision-language model
        tp_size: Tensor parallel size for padding sequence length

    Returns:
        Tuple of (input_ids, pixel_values, image_grid_thw, messages)
    """
    if is_vl_model and image_path:
        if not QWEN_VL_UTILS_AVAILABLE:
            raise ImportError("qwen_vl_utils is required for vision-language models but not installed")

        # Create messages with image and text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_ids = pad_input_ids_to_tp_multiple(inputs.input_ids, tp_size, tokenizer.pad_token_id or 0)
        return input_ids, inputs.pixel_values, inputs.image_grid_thw, messages
    else:
        # Text-only processing for both VL models without images and regular LLMs
        if is_vl_model and processor:
            # Use processor for VL models even in text-only mode
            inputs = processor(text=[prompt], return_tensors="pt")
        else:
            # Use tokenizer for regular LLMs
            inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = pad_input_ids_to_tp_multiple(inputs.input_ids, tp_size, tokenizer.pad_token_id or 0)
        return input_ids, None, None, None


def _load_hf_model(args, is_vl_model: bool):
    """Load HuggingFace model on rank 0.

    Args:
        args: Command line arguments.
        is_vl_model: Whether this is a vision-language model.

    Returns:
        Loaded HF model or None if not on rank 0.
    """
    if not _is_rank_0():
        return None

    print_rank_0("Loading HuggingFace model...")
    model_class = get_model_class(args.model_class, is_vl_model)
    hf_model = model_class.from_pretrained(
        args.hf_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=is_safe_repo(
            trust_remote_code=args.trust_remote_code,
            hf_path=args.hf_model_path,
        ),
    )
    hf_model = hf_model.eval()
    print_rank_0(f"Loaded with {model_class.__name__}")

    # Register debug hooks if enabled
    if args.enable_debug_hooks:
        print_rank_0("Registering debug hooks for HuggingFace model...")
        debugger.register_hooks(hf_model, file_prefix="hf_debug_")
        print_rank_0("HuggingFace debug hooks registered.")

    return hf_model


def _export_and_load_roundtrip_hf_model(args, is_vl_model: bool, megatron_model, bridge):
    """Export HF weights from Megatron model, save, and load exported HF model for comparison.

    Returns:
        Exported HF model loaded from disk (rank 0) or None (non-rank-0).
    """
    print_rank_0("Performing HF round-trip export from Megatron model...")
    model_name = args.hf_model_path.split("/")[-1]
    parent_dir = args.exported_hf_dir if args.exported_hf_dir else "."
    save_path = os.path.join(parent_dir, f"{model_name}_roundtrip")

    if _is_rank_0():
        print_rank_0(f"Exporting HF checkpoint to: {save_path}")
    # Quick verification of exported weights against original HF weights (rank 0 summary)
    matches = 0
    mismatches = 0
    for name, param in bridge.export_hf_weights(megatron_model, show_progress=False):
        if _is_rank_0():
            original_param = bridge.hf_pretrained.state[name]
            if torch.allclose(param, original_param.to(param.device), atol=1e-1):
                matches += 1
            else:
                mismatches += 1
    if _is_rank_0():
        print_rank_0(f"Export verification - matches: {matches}, mismatches: {mismatches}")

    # Save exported HF checkpoint
    bridge.save_hf_pretrained(megatron_model, save_path)

    # Load exported HF model only on rank 0
    if _is_rank_0():
        print_rank_0("Loading exported HF model for comparison...")
        model_class = get_model_class(args.model_class, is_vl_model)
        hf_model = model_class.from_pretrained(
            save_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        ).eval()
        if args.enable_debug_hooks:
            print_rank_0("Registering debug hooks for exported HF model...")
            debugger.register_hooks(hf_model, file_prefix="hf_debug_")
            print_rank_0("Exported HF debug hooks registered.")
        return hf_model
    return None


def _run_hf_inference(hf_model, input_ids, pixel_values, image_grid_thw, tokenizer):
    """Run HuggingFace model inference and return results.

    Args:
        hf_model: The HuggingFace model (may be None for non-rank-0).
        input_ids: Input token IDs.
        pixel_values: Pixel values for vision models (optional).
        image_grid_thw: Image grid dimensions (optional).
        tokenizer: Tokenizer for decoding.

    Returns:
        Tuple of (hf_logits, hf_next_token, hf_logits_stats, hf_top5_info, logits_shape).
    """
    print_rank_0("=== RUNNING HF MODEL (1-STEP) ===")

    if not _is_rank_0() or hf_model is None:
        return None, None, None, None, None

    with torch.no_grad():
        hf_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        }
        if pixel_values is not None:
            hf_inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            hf_inputs["image_grid_thw"] = image_grid_thw

        hf_output = hf_model(**hf_inputs)

        # Debug: Check output type
        print_rank_0(f"HF output type: {type(hf_output)}")

        # Extract logits from AutoModelForCausalLM output
        if hasattr(hf_output, "logits"):
            hf_logits = hf_output.logits[0, -1, :]  # Last token logits
            logits_shape = hf_output.logits.shape
        else:
            print_rank_0("Error: AutoModelForCausalLM output doesn't have logits attribute")
            return None, None, None, None, None

        hf_next_token = torch.argmax(hf_logits, dim=-1)

        hf_logits_stats = f"mean: {hf_logits.mean():.4f}, std: {hf_logits.std():.4f}"
        # Show top 5 tokens
        top5_vals, top5_ids = torch.topk(hf_logits, 5)
        top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
        hf_top5_info = list(zip(top5_tokens, top5_vals.tolist()))

        print_rank_0(f"HF output shape: {logits_shape}")
        print_rank_0(f"HF logits stats - {hf_logits_stats}")
        print_rank_0(f"HF next token: {hf_next_token.item()} ('{tokenizer.decode([hf_next_token.item()])}')")
        print_rank_0(f"HF Top 5: {hf_top5_info}")

        return hf_logits, hf_next_token, hf_logits_stats, hf_top5_info, logits_shape


def _load_megatron_model(args):
    """Load Megatron model from checkpoint or convert from HF.

    Args:
        args: Command line arguments.

    Returns:
        Tuple of (List of Megatron model components, bridge).
    """
    print_rank_0("Loading Megatron model...")
    tp, pp, ep, etp = args.tp, args.pp, args.ep, args.etp

    if args.megatron_model_path:
        # Load from Megatron checkpoint
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path)
        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        megatron_model = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides={
                "tensor_model_parallel_size": tp,
                "pipeline_model_parallel_size": pp,
                "expert_model_parallel_size": ep,
                "expert_tensor_parallel_size": etp,
            },
            wrap_with_ddp=False,
        )
    else:
        # Convert from HF to Megatron
        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model_path,
            trust_remote_code=is_safe_repo(
                trust_remote_code=args.trust_remote_code,
                hf_path=args.hf_model_path,
            ),
        )
        model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size = ep
        model_provider.expert_tensor_parallel_size = etp
        model_provider.pipeline_dtype = torch.bfloat16
        model_provider.finalize()
        megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    model_components = [m.eval() for m in megatron_model]

    # Register debug hooks if enabled
    if args.enable_debug_hooks:
        print_rank_0("Registering debug hooks for Megatron model...")
        for i, model_component in enumerate(model_components):
            debugger.register_hooks(model_component, file_prefix=f"megatron_debug_component_{i}_")
        print_rank_0("Megatron debug hooks registered.")

    return model_components, bridge


def _setup_tokenizer_and_processor(args, is_vl_model: bool):
    """Setup tokenizer and processor for the model.

    Args:
        args: Command line arguments.
        is_vl_model: Whether this is a vision-language model.

    Returns:
        Tuple of (tokenizer, processor).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path,
        trust_remote_code=is_safe_repo(
            trust_remote_code=args.trust_remote_code,
            hf_path=args.hf_model_path,
        ),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = None
    if is_vl_model:
        try:
            processor = AutoProcessor.from_pretrained(
                args.hf_model_path,
                trust_remote_code=is_safe_repo(
                    trust_remote_code=args.trust_remote_code,
                    hf_path=args.hf_model_path,
                ),
            )
        except Exception as e:
            print_rank_0(f"Warning: Could not load processor for VL model: {e}")
            print_rank_0("Falling back to tokenizer-only mode")

    return tokenizer, processor


def compare_models_one_step(args) -> None:
    """Compare 1-step generation between HF and Megatron models with debugging.

    Args:
        args: Parsed command line arguments
    """
    print_rank_0("=== STARTING MODEL COMPARISON (1-STEP) ===")

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        print_rank_0(f"Set CUDA device to: {torch.cuda.current_device()}")

    # Detect model type
    is_vl_model = is_vision_language_model(args.hf_model_path, args.trust_remote_code)
    print_rank_0(f"Detected model type: {'Vision-Language' if is_vl_model else 'Text-only LLM'}")

    # Validate vision requirements
    if args.image_path and not is_vl_model:
        print_rank_0("Warning: Image provided but model is not a vision-language model. Ignoring image.")
        args.image_path = None

    # Load Megatron model (and bridge)
    megatron_model, bridge = _load_megatron_model(args)

    # Optionally perform HF round-trip export and use exported HF model for comparison
    if getattr(args, "roundtrip_hf", False):
        hf_model = _export_and_load_roundtrip_hf_model(args, is_vl_model, megatron_model, bridge)
    else:
        # Load HF model directly from the hub/path
        hf_model = _load_hf_model(args, is_vl_model)

    # Setup tokenizer and processor
    tokenizer, processor = _setup_tokenizer_and_processor(args, is_vl_model)

    # Process inputs
    print_rank_0(f"Processing inputs - Prompt: '{args.prompt}', Image: {args.image_path}")
    input_ids, pixel_values, image_grid_thw, messages = process_inputs(
        tokenizer, processor, args.image_path, args.prompt, is_vl_model, args.tp
    )

    # Move to GPU
    input_ids = input_ids.cuda()
    if pixel_values is not None:
        pixel_values = pixel_values.cuda()
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.cuda()

    print_rank_0(f"Input shape: {input_ids.shape}")
    print_rank_0(f"Pixel values shape: {pixel_values.shape if pixel_values is not None else 'None'}")

    # Run HF model forward pass
    hf_logits, hf_next_token, hf_logits_stats, hf_top5_info, logits_shape = _run_hf_inference(
        hf_model, input_ids, pixel_values, image_grid_thw, tokenizer
    )

    del hf_model
    # Reload Megatron model to ensure a fresh instance before comparison
    megatron_model, _ = _load_megatron_model(args)

    # Broadcast HF results to all ranks after Megatron initialization
    # (following the pattern from generate_from_hf.py)
    if torch.distributed.is_initialized():
        # Create tensors for broadcasting if they don't exist on non-rank-0
        if hf_next_token is None:
            hf_next_token = torch.zeros(1, device=input_ids.device, dtype=torch.long)
        if hf_logits is None:
            # Get vocab size from tokenizer for proper tensor size
            vocab_size = getattr(
                tokenizer, "vocab_size", len(tokenizer.vocab) if hasattr(tokenizer, "vocab") else 32000
            )
            hf_logits = torch.zeros(vocab_size, device=input_ids.device, dtype=torch.float32)

        # Broadcast from rank 0 to all ranks
        torch.distributed.broadcast(hf_next_token, 0)
        torch.distributed.broadcast(hf_logits, 0)

    # Run Megatron model forward pass
    print_rank_0("=== RUNNING MEGATRON MODEL (1-STEP) ===")
    with torch.no_grad():
        position_ids = (
            torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .expand_as(input_ids)
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        fwd_bwd_function = get_forward_backward_func()
        iterator = SingleBatchIterator(input_ids, position_ids, attention_mask, pixel_values, image_grid_thw)

        megatron_output = fwd_bwd_function(
            forward_step_func=vlm_forward_step,
            data_iterator=iterator,
            model=megatron_model,
            num_microbatches=1,
            forward_only=True,
            seq_length=input_ids.size(1),
            micro_batch_size=1,
            collect_non_loss_data=True,
        )

        if isinstance(megatron_output, list) and len(megatron_output) > 0:
            megatron_output = megatron_output[0]

        # Handle both single GPU and multi-GPU cases
        is_last_stage = not torch.distributed.is_initialized() or parallel_state.is_pipeline_last_stage()

        if is_last_stage:
            # Gather tensor parallel results if using TP
            if torch.distributed.is_initialized() and parallel_state.get_tensor_model_parallel_world_size() > 1:
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(megatron_output) for _ in range(world_size)]
                dist.all_gather(
                    gathered_tensors, megatron_output, group=parallel_state.get_tensor_model_parallel_group()
                )
                megatron_output = torch.cat(gathered_tensors, dim=2)

            megatron_logits = megatron_output[0, -1, :]
            megatron_next_token = torch.argmax(megatron_logits, dim=-1)

            if not torch.distributed.is_initialized() or parallel_state.get_tensor_model_parallel_rank() == 0:
                print(f"Megatron output shape: {megatron_output.shape}")
                print(f"Megatron logits stats - mean: {megatron_logits.mean():.4f}, std: {megatron_logits.std():.4f}")
                print(
                    f"Megatron next token: {megatron_next_token.item()} ('{tokenizer.decode([megatron_next_token.item()])}')"
                )

                # Show top 5 tokens
                top5_vals, top5_ids = torch.topk(megatron_logits, 5)
                top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
                print(f"Megatron Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")

                # Compare outputs (only where we have valid Megatron results)
                print("=== COMPARISON ===")
                token_match = hf_next_token.item() == megatron_next_token.item()
                token_status_emoji = "✅" if token_match else "❌"
                print(f"Token match: {token_match} {token_status_emoji}")

                # Compare logits if shapes match
                if hf_logits.shape == megatron_logits.shape:
                    diff = (hf_logits - megatron_logits).abs()
                    print(f"Logits diff - max: {diff.max():.6f}, mean: {diff.mean():.6f}")
                    cosine_sim = torch.cosine_similarity(hf_logits.unsqueeze(0), megatron_logits.unsqueeze(0))
                    cos_val = cosine_sim.item()
                    percent = cos_val * 100.0
                    status_emoji = "✅" if cos_val >= SIMILARITY_THRESHOLD else "❌"
                    tolerance_text = "within ±2%" if cos_val >= SIMILARITY_THRESHOLD else "outside ±2%"
                    print(
                        f"Cosine similarity: {cos_val:.6f} ({percent:.2f}%) {status_emoji} ({tolerance_text} tolerance)"
                    )
                else:
                    print(f"Shape mismatch: HF {hf_logits.shape} vs Megatron {megatron_logits.shape}")
                    print("Cannot compare logits directly due to shape mismatch")

                print("=== COMPARISON COMPLETE ===")
        else:
            # Non-last pipeline stages: create dummy tensor for broadcasting
            megatron_next_token = torch.zeros(1, device=input_ids.device, dtype=torch.long)

        # Broadcast Megatron results from last rank to all ranks (following generate_from_hf.py pattern)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(megatron_next_token, get_last_rank())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare HuggingFace and Megatron models")
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt for generation.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path or URL to the image for vision-language generation (optional).",
    )
    parser.add_argument("--megatron_model_path", type=str, default=None, help="Path to the Megatron model checkpoint")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    parser.add_argument(
        "--model_class",
        type=str,
        default=None,
        help="Specific HuggingFace model class to use (e.g., 'Qwen2_5_VLForConditionalGeneration' for VL models). If not specified, uses AutoModelForCausalLM.",
    )
    parser.add_argument(
        "--enable_debug_hooks",
        action="store_true",
        help="Enable debug hooks to log forward pass information for both HF and Megatron models to JSONL files",
    )
    parser.add_argument(
        "--roundtrip_hf",
        action="store_true",
        help="Export HF weights from the Megatron model and compare using the exported HF model instead of the original.",
    )
    parser.add_argument(
        "--exported_hf_dir",
        type=str,
        default=None,
        help="Directory where the exported HF model will be saved during round-trip. Defaults to current directory.",
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="if trust_remote_code")

    args = parser.parse_args()

    compare_models_one_step(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
