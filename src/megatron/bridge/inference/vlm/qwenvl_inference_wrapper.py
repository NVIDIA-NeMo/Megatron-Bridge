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

    def run_one_forward_step(self, inference_input: Dict[str, Any], recv_buffer_seq_len: Optional[int] = None):
        """Run a single forward pass, rejecting pipeline parallelism.

        This wrapper only implements the tensor-parallel / no-pipeline path
        (``forward_pass_without_pipeline_parallel``), whose inputs are keyed on
        ``input_ids`` and include the multimodal tensors (``pixel_values`` /
        ``image_grid_thw``). The base-class pipeline path instead reads
        ``inference_input["tokens"]`` and drops those multimodal inputs, so it cannot
        run this VLM. Fail fast with a clear error instead of raising a cryptic
        ``KeyError: 'tokens'`` deep inside the pipeline path.

        Args:
            inference_input: The prepared model inputs for the current context window.
            recv_buffer_seq_len: Optional pipeline recv-buffer sequence length (unused here).

        Returns:
            The output logits from the tensor-parallel forward pass.
        """
        if getattr(self.config, "pipeline_model_parallel_size", 1) > 1:
            raise NotImplementedError(
                "QwenVLInferenceWrapper does not support pipeline parallelism "
                "(pipeline_model_parallel_size > 1); run VLM inference with "
                "pipeline_model_parallel_size=1."
            )
        return super().run_one_forward_step(inference_input, recv_buffer_seq_len)

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

        if image_dict:
            pixel_values_per_request = [
                _to_cuda_optional(request.get("pixel_values")) for request in image_dict if request is not None
            ]
            pixel_values_per_request = [value for value in pixel_values_per_request if value is not None]
            if pixel_values_per_request:
                pixel_values = torch.cat(pixel_values_per_request, dim=0)

            image_grids_per_request = [
                _to_cuda_optional(request.get("image_grid_thw")) for request in image_dict if request is not None
            ]
            image_grids_per_request = [value for value in image_grids_per_request if value is not None]
            if image_grids_per_request:
                image_grid_thw = torch.cat(image_grids_per_request, dim=0)

            mm_token_type_ids_per_request = [
                _to_cuda_optional(request.get("mm_token_type_ids")) if request is not None else None
                for request in image_dict
            ]
            mm_token_type_ids_template = next(
                (value for value in mm_token_type_ids_per_request if value is not None), None
            )
            if mm_token_type_ids_template is not None:
                padded_mm_token_type_ids = []
                for value in mm_token_type_ids_per_request:
                    if value is None:
                        value = torch.zeros(
                            1,
                            seq_length,
                            dtype=mm_token_type_ids_template.dtype,
                            device=mm_token_type_ids_template.device,
                        )
                    elif value.size(1) < seq_length:
                        pad = torch.zeros(
                            value.size(0),
                            seq_length - value.size(1),
                            dtype=value.dtype,
                            device=value.device,
                        )
                        value = torch.cat([value, pad], dim=-1)
                    padded_mm_token_type_ids.append(value)
                mm_token_type_ids = torch.cat(padded_mm_token_type_ids, dim=0)

        out: Dict[str, Any] = {
            "input_ids": prompts_tokens,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        if mm_token_type_ids is not None:
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
        logits = self.model(
            **inference_input,
            inference_context=self.inference_context,
            runtime_gather_output=True,
        )

        return logits
