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

from typing import TYPE_CHECKING, Iterable, List, Mapping, Optional, Tuple, TypeVar, Union

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model

if TYPE_CHECKING:
    from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, WeightConversionTask


MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)
HFPreTrained = TypeVar("HFPreTrained")


class MegatronQuantizationBridge:
    """Mixin providing quantization-aware utilities for Megatron model bridges."""

    def stream_weights_megatron_to_hf_quant(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        should_quantize: callable,
        quant_fn: callable,
        quant_block_size: Optional[Tuple[int, int]] = None,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List["WeightConversionTask"]] = None,
        merge_adapter_weights: bool = True,
    ) -> Iterable["HFWeightTuple"]:
        """Export Megatron weights to HuggingFace format with quantization.
        """
        from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

        assert not merge_adapter_weights, "Adapter merging is not supported for quantized weights. Use merge_adapter_weights=False instead."

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # Use provided conversion tasks or build them
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)

        megatron_to_hf_tasks = conversion_tasks
        unwrapped_model = unwrap_model(megatron_model)[0]
        model_config = unwrapped_model.config
        embeddings_are_tied = self._share_embeddings_and_output_weights(model_config, unwrapped_model)

        hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}

        for task in self._with_progress_tracking(megatron_to_hf_tasks, "Converting to HuggingFace (Quantized)", show_progress):
            converted_weights_dict = task.mapping.megatron_to_hf_quant(
                task.param_weight, task.megatron_module, should_quantize, quant_fn, quant_block_size
            )
            converted_weights_dict = self.maybe_modify_converted_hf_weight(
                task,
                converted_weights_dict,
                hf_state_dict,
            )

            for hf_name, tensor in converted_weights_dict.items():
                final_tensor = tensor.cpu() if cpu else tensor

                if not merge_adapter_weights and "to_wrap.weight" in task.global_param_name:
                    hf_name = hf_name[: -len("weight")] + "base_layer.weight"

                if embeddings_are_tied and hf_name == "model.embed_tokens.weight":
                    yield HFWeightTuple(hf_name, final_tensor)
                    if hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source"):
                        expected_keys = hf_pretrained.state.source.get_all_keys()
                        if "lm_head.weight" in expected_keys:
                            yield HFWeightTuple("lm_head.weight", final_tensor.clone().detach())
                elif embeddings_are_tied and hf_name == "lm_head.weight":
                    raise ValueError("Encountered lm_head.weight when embeddings are tied. This indicates a mapping error.")
                else:
                    yield HFWeightTuple(hf_name, final_tensor)
