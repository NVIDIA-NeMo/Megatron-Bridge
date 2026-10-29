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
from megatron.core.inference.engines.static_engine import StaticInferenceEngine
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import TextGenerationController
from PIL.Image import Image

from ._mcore_compat import InferenceMode


class VLMEngine(StaticInferenceEngine):
    """VLM inference engine extending MCoreEngine with image support."""

    def __init__(
        self,
        text_generation_controller: TextGenerationController,
        max_batch_size: int | None = None,
        random_seed: int | None = None,
        legacy: bool = False,
        buffer_size_gb: float | None = 40,
    ) -> None:
        super().__init__(
            text_generation_controller=text_generation_controller,
            max_batch_size=max_batch_size,
            random_seed=random_seed,
            legacy=legacy,
            buffer_size_gb=buffer_size_gb,
        )
        self._sampling_random_seed = random_seed

    # pylint: disable=C0115,C0116
    def generate(
        self,
        prompts: List[str],
        images: Optional[List[Union[Image, List[Image]]]] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[InferenceRequest]:
        # pylint: disable=C0115,C0116
        InferenceMode.set_active()
        try:
            request_ids: List[str] = []
            prepared_requests = []

            if self.random_seed:
                torch.random.manual_seed(self.random_seed)
            if self._sampling_random_seed is not None:
                self.controller.sampling_rng.manual_seed(self._sampling_random_seed)

            if images is not None and len(images) != len(prompts):
                raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")

            for i in range(len(prompts)):
                prompt = prompts[i]
                image = images[i] if images is not None else None
                prompt_tokens, image_dict = self.controller.tokenize_prompt(prompt, image)
                prepared_requests.append((prompt, prompt_tokens, image_dict))

            prompt_lengths = {len(prompt_tokens) for _, prompt_tokens, _ in prepared_requests}
            serialize_requests = len(prompt_lengths) > 1 and any(
                image_dict is not None for _, _, image_dict in prepared_requests
            )

            for prompt, prompt_tokens, image_dict in prepared_requests:
                # Reuse encoder_prompt from scheduler to pass image
                request_id = self.scheduler.add_request(
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    encoder_prompt=image_dict,
                    sampling_params=sampling_params,
                )
                request_ids.append(request_id)
                if serialize_requests:
                    self.run_engine()

            if not serialize_requests:
                self.run_engine()

            result: List[InferenceRequest] = [
                self.scheduler.completed_request_pool[request_id] for request_id in request_ids
            ]
            return result
        finally:
            InferenceMode.unset_active()
