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

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.optimizer import OptimizerConfig, ParamKey, get_standard_config_overrides
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.transformer.moe.router import Router, TopKRouter
from megatron.core.utils import unwrap_model

from megatron.bridge.peft.base import PEFT, ModelType
from megatron.bridge.peft.lora_layers import (
    LinearAdapter,
    LoRALinear,
    LoRATopKRouter,
    TEFusedLoRALinear,
    TEFusedLoRAMergeLinear,
)
from megatron.bridge.peft.module_matcher import ModuleMatcher
from megatron.bridge.peft.utils import (
    GroupedExpertLinearAdapter,
    ParallelLinearAdapter,
    SharedOuterGroupedExpertAdapter,
    align_expert_dim_for_tp,
    get_adapter_attributes_from_linear,
    get_effective_lora_dim,
    is_expert_linear,
    is_grouped_expert_linear,
    is_modelopt_linear,
)


logger = logging.getLogger(__name__)


def _model_chunks(model: ModelType) -> list[nn.Module]:
    """The modules ``PEFT._walk_model`` walks: a list's chunks, a DDP model's module, or the model itself."""
    if isinstance(model, list):
        return list(model)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return [model.module]
    return [model]


def _is_frozen(module: nn.Module) -> bool:
    """Whether ``module`` has parameters and trains none of them."""
    parameters = list(module.parameters())
    return bool(parameters) and not any(parameter.requires_grad for parameter in parameters)


def get_lora_plus_config_overrides(
    config: OptimizerConfig,
    lora_plus_ratio: float,
) -> Dict[ParamKey, ParamGroupOverride]:
    """Build Megatron-Core optimizer ``config_overrides`` for LoRA+.

    LoRA+ (https://arxiv.org/abs/2402.12354) trains ``B`` (``linear_out``) at
    ``lora_plus_ratio`` times the LR of ``A`` (``linear_in``); the ratio is fixed
    and only the base LR is tuned. Standard overrides (bias/1-D ``wd_mult=0.0``,
    decoupled LR) are preserved as the starting point.

    Args:
        config: base optimizer config; ``lr`` / ``min_lr`` are A's learning rates.
        lora_plus_ratio: ``lr_B / lr_A`` (e.g. 16.0). Must be > 0.

    Returns:
        Per-group overrides ready for ``get_megatron_optimizer(config_overrides=...)``.
    """
    if lora_plus_ratio <= 0:
        raise ValueError(f"lora_plus_ratio must be > 0, got {lora_plus_ratio}")

    config_overrides = get_standard_config_overrides(config)

    base_lr = config.lr
    base_min_lr = config.min_lr if config.min_lr is not None else 0.0

    config_overrides[ParamKey(name="*.linear_in.weight")] = ParamGroupOverride(max_lr=base_lr, min_lr=base_min_lr)
    config_overrides[ParamKey(name="*.linear_out.weight")] = ParamGroupOverride(
        max_lr=base_lr * lora_plus_ratio, min_lr=base_min_lr * lora_plus_ratio
    )

    logger.info(
        "LoRA+ enabled: lora_plus_ratio=%s (linear_in/A max_lr=%s, linear_out/B max_lr=%s)",
        lora_plus_ratio,
        base_lr,
        base_lr * lora_plus_ratio,
    )
    return config_overrides


@dataclass
class LoRA(PEFT, ModuleMatcher):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
        exclude_modules (List[str], optional): A list of module names not to apply LoRa to. It will
            match all nn.Linear & nn.Linear-adjacent modules whose name does not match any string in
            exclude_modules. If used, will require target_modules to be empty list or None.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.
        sequence_parallel_input_regather (bool): Reduce retained activation memory for eligible
            column-parallel LoRA-A projections. The full LayerNorm output consumed by LoRA-A is
            gathered temporarily in forward, released after the LoRA-A computation, and gathered
            again during backward for the LoRA-A weight gradient. MCore overlaps the backward
            all-gather with dgrad computation when possible. This has no effect when sequence
            parallelism is disabled and falls back for unsupported adapters or overlapping
            activation recompute. Defaults to False.
        a2a_experimental (bool): Enables the experimental All-to-All (A2A) communication strategy. Defaults to False.
        lora_A_init_method (str): Initialization method for the low-rank matrix A. Defaults to "xavier".
        lora_B_init_method (str): Initialization method for the low-rank matrix B. Defaults to "zero".
        lora_dtype (torch.dtype): Parameter data type for LoRA weights. Default None (will use model's dtype).
        normalize_moe_lora (bool): When True, expert linear layers use dim // moe_router_topk as the LoRA rank
            while non-expert layers keep the full dim. This normalizes the total adapter capacity for MoE models
            so it is comparable to a dense model. Defaults to False.
        share_expert_adapters (bool): When True, grouped MoE expert linears share one adapter across all local
            experts on the EP rank. Set to False to create one adapter per local expert instead. Defaults to True.
        experts_shared_outer_loras (bool): When True, grouped-expert LoRA
            (``TE*ParallelGroupedLinear`` base modules) uses
            :class:`SharedOuterGroupedExpertAdapter` — ``gate_up`` lora_A and
            ``down`` lora_B are shared across experts (expert_dim=1), matching
            SGLang's ``experts_shared_outer_loras=True`` serving contract (PR
            #21466). Default False preserves the adapter layout selected by
            ``share_expert_adapters``.
        use_transformer_engine_op_fuser (bool): Enable Transformer Engine fusion only for LoRA wrappers.
        freeze_routing (bool): Keep MoE routing exactly the base model's while the adapters train. LoRA then
            raises if a target resolves to a router, or if a router keeps a quantile-balancing ``qb_beta``
            (Megatron-Core updates it with no switch to stop it), and sets every router's
            ``frozen_expert_bias``, so ``finalize_model_grads`` stops moving the expert bias (aux-loss-free
            or global-batch quantile balancing). Disabling the adapters then gives back the base model
            exactly, and the adapters hold everything training changed. Defaults to False: routers can be
            targets and the expert bias keeps updating.
        skip_frozen_modules (bool): Leave out the target modules whose parameters were all frozen before LoRA
            was applied, such as the parts a vision-language model's ``freeze_language_model``,
            ``freeze_vision_model`` and ``freeze_vision_projection`` freeze when it is built, so adapters
            go only where the recipe trains. Raises if every matched module was frozen; with pipeline
            parallelism, a stage left without adapters only logs a warning, as other stages may hold them.
            Defaults to False: every matched module gets an adapter.
    """

    target_modules: List[str] = field(
        default_factory=lambda: ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    sequence_parallel_input_regather: bool = False
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: torch.dtype = None
    normalize_moe_lora: bool = False
    share_expert_adapters: bool = True
    experts_shared_outer_loras: bool = False
    use_transformer_engine_op_fuser: bool = False
    freeze_routing: bool = False
    skip_frozen_modules: bool = False

    # Runtime state of one ``__call__`` for ``skip_frozen_modules``; not serialized.
    _frozen_before_peft: set[int] = field(default_factory=set, init=False, repr=False)
    _skipped_frozen: list[str] = field(default_factory=list, init=False, repr=False)

    def __call__(self, model: ModelType, training: bool = True) -> ModelType:
        """Apply LoRA to the model, honoring ``freeze_routing`` and ``skip_frozen_modules``.

        The routing checks run before any module is wrapped; which modules are frozen is read before
        ``PEFT.__call__`` freezes the base, so it reflects what the model trained before LoRA.

        Args:
            model: The model or list of model chunks to adapt.
            training: Whether the model will be trained.

        Returns:
            The model with LoRA applied.

        Raises:
            ValueError: With ``freeze_routing``, if a target resolves to an MoE router; with
                ``skip_frozen_modules`` and no pipeline parallelism, if every module the targets matched
                was frozen.
            NotImplementedError: With ``freeze_routing``, if a router keeps a quantile-balancing ``qb_beta``.
        """
        if self.freeze_routing:
            self._check_routing_can_stay_base(model)
        if self.skip_frozen_modules:
            self._frozen_before_peft = {
                id(module) for chunk in _model_chunks(model) for module in chunk.modules() if _is_frozen(module)
            }
        try:
            model = super().__call__(model, training=training)
            skipped = list(self._skipped_frozen)
        finally:
            self._frozen_before_peft = set()
            self._skipped_frozen = []
        if skipped and not any(
            isinstance(module, (LoRALinear, LoRATopKRouter))
            for chunk in _model_chunks(model)
            for module in chunk.modules()
        ):
            message = (
                f"skip_frozen_modules left out all {len(skipped)} module(s) target_modules={list(self.target_modules)} "
                f"matched, because they were frozen before LoRA (e.g. {skipped[0]})"
            )
            if (
                parallel_state.model_parallel_is_initialized()
                and parallel_state.get_pipeline_model_parallel_world_size() > 1
            ):
                logger.warning("%s: this pipeline stage holds no adapters.", message)
            else:
                raise ValueError(f"{message}: nothing would be trained.")
        if self.freeze_routing:
            for chunk in _model_chunks(model):
                for module in chunk.modules():
                    if hasattr(module, "frozen_expert_bias"):
                        module.frozen_expert_bias = True
        return model

    def _check_routing_can_stay_base(self, model: ModelType) -> None:
        """Raise if a target resolves to an MoE router or a router keeps a quantile-balancing ``qb_beta``.

        Matching goes through ``match``, so names and wildcards resolve as they will when LoRA wraps the
        model (``PEFT.__call__`` then resets the match bookkeeping this walk records).
        """
        self._init_target_match_state()
        matched: list[str] = []
        quantile_balanced: list[str] = []

        def check(module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
            if isinstance(module, Router):
                full_name = f"{prefix}.{name}" if prefix else str(name)
                if self.match(module, name, prefix) is not None:
                    matched.append(full_name)
                if getattr(module, "qb_beta", None) is not None:
                    quantile_balanced.append(full_name)
            return module

        self._walk_model(model, check)
        if matched:
            raise ValueError(
                f"freeze_routing keeps MoE routing the base model's, but target_modules={list(self.target_modules)} "
                f"matches {len(matched)} router(s), e.g. {matched[0]}."
            )
        if quantile_balanced:
            raise NotImplementedError(
                "freeze_routing does not support routers with a quantile-balancing qb_beta: finalize_model_grads "
                f"updates it during training (e.g. {quantile_balanced[0]}), so routing would drift from the base "
                "model."
            )

    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        # Skip already transformed modules
        adapter_types = (LoRALinear, LoRATopKRouter)
        if isinstance(module, adapter_types):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            _, full_name = ans
            if id(module) in self._frozen_before_peft:
                self._skipped_frozen.append(full_name)
                return module
            if isinstance(module, nn.Linear) and not is_modelopt_linear(module):
                adapter = LinearAdapter(
                    module,
                    dim=self.dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    lora_A_init_method=self.lora_A_init_method,
                    lora_dtype=self.lora_dtype,
                )
                return LoRALinear(module, adapter)

            is_expert = is_expert_linear(full_name)
            attrs = get_adapter_attributes_from_linear(
                module,
                is_expert=is_expert,
                sequence_parallel_input_regather=self.sequence_parallel_input_regather and not is_expert,
            )

            dim = get_effective_lora_dim(
                module, dim=self.dim, normalize_moe_lora=self.normalize_moe_lora, is_expert=is_expert
            )
            dim = align_expert_dim_for_tp(
                module,
                dim,
                normalize_moe_lora=self.normalize_moe_lora,
                is_expert=is_expert,
                input_is_parallel=attrs.input_is_parallel,
            )
            is_grouped_expert_name = is_grouped_expert_linear(full_name)
            use_shared_outer_adapter = self.experts_shared_outer_loras and is_grouped_expert_name
            use_per_expert_adapter = (
                is_grouped_expert_name and not self.share_expert_adapters and not use_shared_outer_adapter
            )
            use_grouped_expert_adapter = use_shared_outer_adapter or use_per_expert_adapter

            enable_op_fuser = (
                not use_grouped_expert_adapter
                and (
                    self.use_transformer_engine_op_fuser
                    or getattr(module.config, "use_transformer_engine_op_fuser", False)
                )
                # TP not yet supported
                and parallel_state.get_tensor_model_parallel_world_size() == 1
            )

            logger.info(f"Adding lora to: {full_name}")
            if use_shared_outer_adapter:
                adapter_cls = SharedOuterGroupedExpertAdapter
            elif use_per_expert_adapter:
                adapter_cls = GroupedExpertLinearAdapter
            else:
                adapter_cls = ParallelLinearAdapter
            adapter_kwargs = dict(
                base_linear_name=full_name,
                activation="identity",
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                input_is_parallel=attrs.input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=module.config,
                alpha=self.alpha,
                base_linear_is_parallel=attrs.base_linear_is_parallel,
            )
            if use_grouped_expert_adapter:
                first_param = next(module.parameters())
                adapter_kwargs.update(
                    num_local_experts=module.num_gemms,
                    params_device=first_param.device,
                    params_dtype=first_param.dtype,
                )
            else:
                adapter_kwargs.update(
                    is_expert=is_expert,
                    a2a_experimental=self.a2a_experimental,
                    sequence_parallel_input_regather=self.sequence_parallel_input_regather,
                    disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
                    disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                    replicate_adapter=attrs.replicate_adapter,
                )
            adapter = adapter_cls(attrs.in_features, attrs.out_features, dim, **adapter_kwargs)
            if isinstance(module, TopKRouter):
                return LoRATopKRouter(module, adapter)
            if enable_op_fuser:
                if is_expert:
                    return TEFusedLoRAMergeLinear(module, adapter)
                return TEFusedLoRALinear(module, adapter)
            else:
                return LoRALinear(module, adapter)
        return module


@dataclass
class VLMLoRA(LoRA):
    """
    Implements the LoRA for Vision-Language Models.
    VLMLoRA additionally allows the user to specify whether the language or vision
    models should be frozen.
    For example, a common finetuning workload for multimodal models is to apply adapters to language model and fully
    finetune the vision model.

    """

    freeze_vision_model: bool = True
    freeze_vision_projection: bool = True
    freeze_language_model: bool = True

    def freeze_model(self, model: nn.Module, training: bool = True) -> None:
        modules_to_freeze = []

        model = unwrap_model(model)[0]
        if hasattr(model, "llava_model"):
            model = model.llava_model

        if self.freeze_vision_model and model.vision_model is not None:
            modules_to_freeze.append(model.vision_model)
        if self.freeze_vision_projection and model.vision_projection is not None:
            modules_to_freeze.append(model.vision_projection)
        if self.freeze_language_model and model.language_model is not None:
            modules_to_freeze.append(model.language_model)

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        if training:
            if isinstance(model, list):
                for model_chunk in model:
                    model_chunk.train(mode=True)
            elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.train(mode=True)
            else:
                model.train(mode=True)
