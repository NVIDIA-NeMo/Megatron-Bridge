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

from typing import List, Optional, Union

import torch
import torch.distributed
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from PIL.Image import Image
from transformers import AutoProcessor

from megatron.bridge import AutoBridge
from megatron.bridge.models.qwen_vl import Qwen3VLModelProvider, Qwen25VLModelProvider
from megatron.bridge.training.utils.checkpoint_utils import get_hf_model_id_from_checkpoint
from megatron.bridge.utils.common_utils import print_rank_0

from .qwenvl_inference_wrapper import QwenVLInferenceWrapper
from .vlm_engine import VLMEngine
from .vlm_inference_controller import QwenVLTextGenerationController, VLMTextGenerationController


def setup_model_and_tokenizer(
    megatron_model_path: str,
    tp: int = 1,
    pp: int = 1,
    params_dtype: torch.dtype = torch.bfloat16,
    inference_batch_times_seqlen_threshold: int = 1000,
    inference_max_seq_length: int = 8192,
    inference_max_batch_size: int = 4,
):
    """Set up model and tokenizer from a Megatron checkpoint.

    Args:
        megatron_model_path: Path to the Megatron checkpoint.
        tp: Tensor model parallel size.
        pp: Pipeline model parallel size.
        params_dtype: Data type for model parameters.
        inference_batch_times_seqlen_threshold: Threshold for inference batching.
        inference_max_seq_length: Maximum sequence length for inference (prompt + generated tokens).
        inference_max_batch_size: Maximum batch size for inference.
    Returns:
        A tuple of (inference_wrapped_model, processor).
    """
    # Load from Megatron checkpoint
    print_rank_0(f"Loading Megatron model from: {megatron_model_path}")

    # Get HF model path from checkpoint metadata
    hf_model_path = get_hf_model_id_from_checkpoint(megatron_model_path)

    # We still need HF config for tokenizer, but we'll load the model from Megatron checkpoint
    # Create bridge from HF config only (no weights)
    bridge = AutoBridge.from_hf_pretrained(hf_model_path)

    # Initialize model parallel before loading
    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.parallel_output = False
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    # Load the Megatron model directly
    model = bridge.load_megatron_model(
        megatron_model_path,
        mp_overrides={
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "pipeline_dtype": torch.bfloat16,
        },
        wrap_with_ddp=False,
    )

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()

    # Set grad_scale_func to None on the model's config for inference
    for m in model:
        if hasattr(m, "config"):
            m.config.grad_scale_func = None

    # Initialize tokenizer and processor
    processor = AutoProcessor.from_pretrained(
        hf_model_path,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Setup inference wrapper
    inference_wrapped_model = setup_inference_wrapper(
        model[0],
        processor.tokenizer,
        params_dtype=torch.bfloat16,
        inference_batch_times_seqlen_threshold=1000,
        inference_max_seq_length=inference_max_seq_length,
        inference_max_batch_size=inference_max_batch_size,
    )

    return inference_wrapped_model, processor


def _expose_decoder_from_language_model(model):
    """Recursively get language_model from model and expose decoder, handling wrapped modules."""
    current = model
    while hasattr(current, "module"):
        current = current.module

    if hasattr(current, "language_model"):
        language_model = current.language_model
        current.decoder = language_model.decoder
        current.vocab_size = language_model.vocab_size


def setup_inference_wrapper(
    model,
    tokenizer,
    params_dtype: torch.dtype = torch.bfloat16,
    inference_batch_times_seqlen_threshold: int = 1000,
    inference_max_seq_length: int = 8192,
    inference_max_batch_size: int = 4,
):
    """Set up inference wrapper for the model"""
    config = model.config

    mcore_model = model.cuda()
    mcore_model = mcore_model.to(params_dtype)
    mcore_model.eval()

    # if isinstance(config, vlm.Qwen2VLConfig):
    if isinstance(config, Qwen25VLModelProvider) or isinstance(config, Qwen3VLModelProvider):
        wrapper_cls = QwenVLInferenceWrapper
        if isinstance(config, Qwen25VLModelProvider):
            _ = config.hidden_size
            # Expose decoder for MCore Infernce Engine compatibility (used by get_mamba_inference_state_config_from_model)
            _expose_decoder_from_language_model(mcore_model)
        else:
            _ = config.language_transformer_config.hidden_size
    else:
        raise ValueError(f"Unknown model config: {config}")

    inference_wrapped_model = wrapper_cls(
        mcore_model,
        inference_context=StaticInferenceContext(
            max_batch_size=inference_max_batch_size,
            max_sequence_length=inference_max_seq_length,
        ),
    )

    return inference_wrapped_model


def generate(
    wrapped_model: AbstractModelInferenceWrapper,
    tokenizer,
    image_processor,
    prompts: List[str],
    images: List[Union[Image, List[Image]]],
    processor=None,
    random_seed: Optional[int] = None,
    sampling_params: Optional[SamplingParams] = None,
) -> dict:
    """
    Generates text using a NeMo VLM model.
    Args:
        wrapped_model (AbstractModelInferenceWrapper): The model inference wrapper.
        tokenizer: tokenizer for the input text,
        image_processor: image processor for the input image,
        prompts (list[str]): The list of prompts to generate text for.
        images (list): The list of images to generate text for.
        random_seed (Optional[int], optional): The random seed. Defaults to None.
        sampling_params (Optional["SamplingParams"], optional): The sampling parameters defined in
            Mcore's SamplingParams. Defaults to None.

    Returns:
        list[Union["InferenceRequest", str]]: A list of generated text,
            either as a string or as an InferenceRequest object.
    """

    if isinstance(wrapped_model, QwenVLInferenceWrapper):
        text_generation_controller = QwenVLTextGenerationController(
            inference_wrapped_model=wrapped_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            processor=processor,
        )
    else:
        text_generation_controller = VLMTextGenerationController(
            inference_wrapped_model=wrapped_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
    mcore_engine = VLMEngine(text_generation_controller=text_generation_controller, random_seed=random_seed)

    if sampling_params is None:
        sampling_params = SamplingParams(num_tokens_to_generate=50)

    results = mcore_engine.generate(
        prompts=prompts,
        images=images,
        sampling_params=sampling_params,
    )

    return results
