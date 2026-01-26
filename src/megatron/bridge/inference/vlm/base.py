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
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from PIL.Image import Image

from megatron.bridge.models.qwen_vl import Qwen3VLModelProvider, Qwen25VLModelProvider

from .qwenvl_inference_wrapper import QwenVLInferenceWrapper
from .vlm_engine import VLMEngine
from .vlm_inference_controller import QwenVLTextGenerationController, VLMTextGenerationController


def setup_inference_wrapper(
    model,
    tokenizer,
    params_dtype: torch.dtype = torch.bfloat16,
    inference_batch_times_seqlen_threshold: int = 1000,
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
            hidden_size = config.hidden_size
        else:
            hidden_size = config.language_transformer_config.hidden_size
    else:
        raise ValueError(f"Unknown model config: {config}")

    inference_wrapped_model = wrapper_cls(
        mcore_model,
        InferenceWrapperConfig(
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            padded_vocab_size=tokenizer.vocab_size,
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
    max_batch_size: int = 4,
    random_seed: Optional[int] = None,
    inference_params: Optional[CommonInferenceParams] = None,
) -> dict:
    """
    Generates text using a NeMo VLM model.
    Args:
        wrapped_model (AbstractModelInferenceWrapper): The model inference wrapper.
        tokenizer: tokenizer for the input text,
        image_processor: image processor for the input image,
        prompts (list[str]): The list of prompts to generate text for.
        images (list): The list of images to generate text for.
        max_batch_size (int, optional): The maximum batch size. Defaults to 4.
        random_seed (Optional[int], optional): The random seed. Defaults to None.
        inference_params (Optional["CommonInferenceParams"], optional): The inference parameters defined in
            Mcore's CommonInferenceParams. Defaults to None.

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
    mcore_engine = VLMEngine(
        text_generation_controller=text_generation_controller, max_batch_size=max_batch_size, random_seed=random_seed
    )

    common_inference_params = inference_params or CommonInferenceParams(num_tokens_to_generate=50)

    results = mcore_engine.generate(
        prompts=prompts,
        images=images,
        common_inference_params=common_inference_params,
    )

    return results
