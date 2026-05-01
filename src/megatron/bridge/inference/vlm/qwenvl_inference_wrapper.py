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

from typing import Any, Dict, List, Optional

import torch
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference_params import InferenceParams


def _to_cuda_optional(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.cuda(non_blocking=True)


class QwenVLInferenceWrapper(AbstractModelInferenceWrapper):
    """Constructor for the model inference wrapper

    The wrapper prepares the model for inference, provides the required input
    data, and runs the forward pass

    Args:
        model (Qwen2VLModel): The Qwen2VL model
    """

    def __init__(self, model, inference_context=None):
        super().__init__(model, inference_context=inference_context)

    def prep_inference_input(
        self,
        prompts_tokens: torch.Tensor,
        image_dict: List[Dict] | None = None,
    ):
        # pylint: disable=C0115,C0116
        batch_size = prompts_tokens.size(0)
        seq_length = prompts_tokens.size(1)

        self.inference_params = InferenceParams(batch_size, seq_length)

        pixel_values = None
        image_grid_thw = None
        mm_token_type_ids = None

        if image_dict and image_dict[0] is not None:
            image_dict = image_dict[0]
            pixel_values = _to_cuda_optional(image_dict.get("pixel_values"))
            image_grid_thw = _to_cuda_optional(image_dict.get("image_grid_thw"))
            mm_token_type_ids = _to_cuda_optional(image_dict.get("mm_token_type_ids"))

        out: Dict[str, Any] = {
            "input_ids": prompts_tokens,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        if mm_token_type_ids is not None:
            if mm_token_type_ids.size(1) < seq_length:
                pad = torch.zeros(
                    mm_token_type_ids.size(0),
                    seq_length - mm_token_type_ids.size(1),
                    dtype=mm_token_type_ids.dtype,
                    device=mm_token_type_ids.device,
                )
                mm_token_type_ids = torch.cat([mm_token_type_ids, pad], dim=-1)
            out["mm_token_type_ids"] = mm_token_type_ids

        out["position_ids"] = (
            torch.arange(prompts_tokens.size(1), dtype=torch.long, device=prompts_tokens.device)
            .unsqueeze(0)
            .expand_as(prompts_tokens)
        )
        out["attention_mask"] = torch.ones_like(prompts_tokens, dtype=torch.bool)
        return out

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        # pylint: disable=C0115,C0116
        tokens2use = inference_input["input_ids"][:, :context_end_position]

        out: Dict[str, Any] = {
            "input_ids": tokens2use,
            "position_ids": inference_input["position_ids"][:, :context_end_position],
            "attention_mask": inference_input["attention_mask"][:, :context_end_position],
            "pixel_values": inference_input.get("pixel_values"),
            "image_grid_thw": inference_input.get("image_grid_thw"),
            "mm_token_type_ids": inference_input.get("mm_token_type_ids"),
        }
        if out["mm_token_type_ids"] is not None:
            out["mm_token_type_ids"] = out["mm_token_type_ids"][:, :context_end_position]
        return out

    def forward_pass_without_pipeline_parallel(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """Utility to carry out simple forward pass for TP or no model parallel models

        Runs a very simple forward pass for model. Used in the case of models without
        any parallelism or only tensor parallelism.

        Args:
            inference_input (Dict): A dictionary containing the inputs for the qwen
                model [input_ids, position_ids, attention_mask, pixel_values, image_grid_thw]

        Returns:
            torch.Tensor: The output logits of shape [batch_size, seq_len, padded_vocab_size]
        """
        logits = self.model(**inference_input)

        return logits
