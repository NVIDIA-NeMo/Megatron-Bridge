# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Builder for ERNIE 5.5: Megatron-Core's Hybrid builder constructing :class:`Ernie55HybridModel` stages."""

import inspect
from typing import Any

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import ModuleSpec
from megatron.training.vocab_utils import calculate_padded_vocab_size

from megatron.bridge.models.ernie.ernie55_config import Ernie55ModelConfig
from megatron.bridge.models.ernie.modeling_ernie55 import (
    Ernie55HybridModel,
    Ernie55HybridStack,
    check_ernie55_mtp_aliases,
    ernie55_hybrid_stack_spec,
    validate_ernie55_mtp_sharing,
)
from megatron.bridge.models.hybrid.hybrid_builder import BridgeHybridModelBuilder
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg


__all__ = ["Ernie55ModelBuilder"]


class Ernie55ModelBuilder(BridgeHybridModelBuilder):
    """Build every pipeline (and virtual pipeline) stage of ERNIE 5.5 as :class:`Ernie55HybridModel`.

    Each stage is the ERNIE model class, whether or not it holds the output layer: its constructor applies
    the ERNIE contract (eager fp32 hyper-connections, the multimax head and its gradient-norm group, which
    every stage must register, SD-MTP sharing and the H/D attention check) before DDP, the optimizer or a
    checkpoint load see the parameters.

    :meth:`build_distributed_models` additionally rejects the wrappers that would split the SD-MTP shared
    parameters (Megatron-FSDP, torch FSDP2, meta-device initialization) before building, and checks after
    wrapping that every MTP alias is still the decoder's parameter.
    """

    def __init__(self, model_config: Ernie55ModelConfig) -> None:
        if not isinstance(model_config, Ernie55ModelConfig):
            raise TypeError(f"Ernie55ModelBuilder needs an Ernie55ModelConfig, got {type(model_config).__name__}.")
        super().__init__(model_config)

    def _resolve_hybrid_stack_spec(self) -> ModuleSpec:
        """The ERNIE 5.5 stack spec: :func:`ernie55_hybrid_stack_spec`, or the configured ERNIE spec.

        Raises:
            TypeError: If ``hybrid_stack_spec`` resolves to a stack other than :class:`Ernie55HybridStack`.
        """
        configured = self._model_config.hybrid_stack_spec
        if configured is None:
            spec = ernie55_hybrid_stack_spec(self._model_config)
        elif isinstance(configured, ModuleSpec):
            spec = configured
        elif inspect.signature(configured).parameters:
            spec = configured(self._model_config)
        else:
            spec = configured()
        if not (
            isinstance(spec, ModuleSpec)
            and isinstance(spec.module, type)
            and issubclass(spec.module, Ernie55HybridStack)
        ):
            raise TypeError(
                "ERNIE 5.5 needs an Ernie55HybridStack spec (ernie55_hybrid_stack_spec()); hybrid_stack_spec resolved to "
                f"{getattr(spec, 'module', spec)!r}."
            )
        return spec

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> HybridModel:
        """Build one ERNIE 5.5 stage.

        Args:
            pg_collection: Process groups used to construct the model.
            pre_process: Whether this stage holds the embedding (default: first pipeline stage).
            post_process: Whether this stage holds the output layer (default: last pipeline stage).
            vp_stage: Virtual pipeline stage to construct.

        Returns:
            An :class:`Ernie55HybridModel` stage.

        Raises:
            ValueError: If ``vocab_size`` is unset or the configuration would pad the vocabulary.
        """
        config = self._model_config
        if config.vocab_size is None:
            raise ValueError("vocab_size must be set before building ERNIE 5.5.")
        if config.should_pad_vocab:
            padded = calculate_padded_vocab_size(
                config.vocab_size,
                config.make_vocab_size_divisible_by,
                config.transformer.tensor_model_parallel_size,
                logging_enabled=False,
            )
            if padded != config.vocab_size:
                raise ValueError(
                    f"ERNIE 5.5 must not pad its vocabulary ({config.vocab_size} -> {padded}): the multimax head and "
                    "the loss would see the padded logits. Set should_pad_vocab=False (the default for a model "
                    "vocab_size at least the tokenizer's)."
                )
        stack_spec = self._resolve_hybrid_stack_spec()
        kwargs: dict[str, Any] = dict(
            config=config.transformer,
            hybrid_stack_spec=stack_spec,
            vocab_size=config.vocab_size,
            max_sequence_length=config.seq_length,
            hybrid_layer_pattern=config.hybrid_layer_pattern,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy,
            parallel_output=config.parallel_output,
            share_embeddings_and_output_weights=config.share_embeddings_and_output_weights,
            position_embedding_type=config.position_embedding_type,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            seq_len_interpolation_factor=config.seq_len_interpolation_factor,
            pre_process=is_pp_first_stage(pg_collection.pp) if pre_process is None else pre_process,
            post_process=is_pp_last_stage(pg_collection.pp) if post_process is None else post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        kwargs.update(logit_dtype_kwarg(HybridModel, config.logit_dtype))
        return Ernie55HybridModel(**kwargs)

    def build_distributed_models(
        self, pg_collection: ProcessGroupCollection, ddp_config: Any = None, *args: Any, **kwargs: Any
    ) -> list[HybridModel]:
        """Build and wrap every stage; keep the SD-MTP parameters shared.

        Arguments are Megatron-Core's ``HybridModelBuilder.build_distributed_models`` arguments.

        Raises:
            ValueError: If SD-MTP is combined with Megatron-FSDP, torch FSDP2 or meta-device initialization.
            RuntimeError: If wrapping replaced a shared MTP parameter.
        """
        bound = inspect.signature(super().build_distributed_models).bind_partial(
            pg_collection, ddp_config, *args, **kwargs
        )
        use_megatron_fsdp = bool(bound.arguments.get("use_megatron_fsdp", False)) or bool(
            getattr(ddp_config, "use_megatron_fsdp", False)
        )
        validate_ernie55_mtp_sharing(
            self._model_config.transformer,
            use_megatron_fsdp=use_megatron_fsdp,
            use_torch_fsdp2=bool(bound.arguments.get("use_torch_fsdp2", False)),
        )
        models = super().build_distributed_models(pg_collection, ddp_config, *args, **kwargs)
        check_ernie55_mtp_aliases(models)
        return models
