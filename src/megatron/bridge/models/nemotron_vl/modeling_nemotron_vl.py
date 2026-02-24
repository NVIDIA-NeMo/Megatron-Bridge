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


import torch
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import NemotronNano12Bv2VLModelProvider


class NemotronVLModel(MegatronModule):
    """A *stub* Megatron implementation of a Nemotron Vision-Language model.

    At the moment the class only supports language-only forward passes.  Vision
    inputs will raise ``NotImplementedError`` until a reference vision encoder
    is open-sourced.
    """

    def __init__(
        self,
        config: "NemotronNano12Bv2VLModelProvider" | None = None,
        *,
        llava_model: LLaVAModel | None = None,
        pre_process: bool | None = True,
        post_process: bool | None = True,
        vp_stage: int | None = None,
    ) -> None:
        """Create a wrapper that exposes an existing :class:`LLaVAModel` via the
        Bridge API.

        Parameters:
        llava_model:
            A fully-assembled instance of :class:`~megatron.core.models.multimodal.llava_model.LLaVAModel`.
        config:
            (Optional) The provider used to generate the model.  If omitted we
            fall back to ``llava_model.config``.
        """

        if llava_model is not None:
            super().__init__(config=config or llava_model.config)
            self.llava_model = llava_model
            return

        # ------------------------------------------------------------------
        # Legacy path – build language-only stub the old way so existing tests
        # continue to work until provider is upgraded to construct LLaVAModel.
        # ------------------------------------------------------------------
        assert config is not None, "config must be provided when llava_model is None"

        super().__init__(config=config)

        self.pre_process = bool(pre_process)
        self.post_process = bool(post_process)
        self.vp_stage = vp_stage

        # Fallback: build just the language model using provider API
        self.language_model = config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        self.llava_model = None  # type: ignore

    # ---------------------------------------------------------------------
    # Megatron pipeline helpers (delegated to wrapped model)
    # ---------------------------------------------------------------------

    def set_input_tensor(self, input_tensor):  # type: ignore[override]
        if hasattr(self.llava_model, "set_input_tensor"):
            self.llava_model.set_input_tensor(input_tensor)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(self, *args, **kwargs):  # type: ignore[override]
        """Delegate the forward pass to the wrapped :class:`LLaVAModel`."""

        return self.llava_model(*args, **kwargs)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def freeze(
        self,
        *,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = False,
        freeze_vision_projection: bool = False,
    ) -> None:
        """Freeze selected sub-modules by turning off ``requires_grad``."""

        modules: list[torch.nn.Module] = []
        module_names: list[str] = []

        if freeze_language_model:
            modules.append(self.llava_model.language_model)
            module_names.append("language_model")

        if freeze_vision_model:
            modules.append(self.llava_model.vision_model)
            module_names.append("vision_model")

        if freeze_vision_projection:
            modules.append(self.llava_model.vision_projection)
            module_names.append("vision_projection")

        for module_name, module in zip(module_names, modules):
            for name, param in module.named_parameters():
                print(f"Freezing {module_name}.{name}")
                param.requires_grad = False
