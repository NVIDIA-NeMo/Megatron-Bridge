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

"""Pure MegatronMIMO configuration and heterogeneous model builder."""

from __future__ import annotations

import copy
import importlib
from dataclasses import asdict, dataclass, field, replace
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import get_model_config
from megatron.training.models.base import ModelBuilder, ModelConfig

from megatron.bridge.models.megatron_mimo.infra import MegatronMIMOInfra
from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import (
    build_hypercomm_grids,
    is_pp_first_stage,
    is_pp_last_stage,
    populate_embedding_and_position_groups,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_ddp import wrap_megatron_mimo_model_distributed


@dataclass(kw_only=True)
class MegatronMIMOModelConfig(ModelConfig):
    """Serializable inputs for heterogeneous MegatronMIMO construction."""

    builder: ClassVar[str] = "megatron.bridge.models.megatron_mimo.model_config.MegatronMIMOModelBuilder"
    source_model_config: ModelConfig
    megatron_mimo_parallelism_config: MegatronMIMOParallelismConfig
    language_spec_builder: str
    modality_spec_builder: str
    modality_keys: dict[str, str]
    special_token_ids: dict[str, int]
    topology: dict[str, list[str]] | None = None
    module_output_ndim: dict[str, int] | None = None
    freeze_language_model: bool = False
    freeze_modality_encoders: dict[str, bool] = field(default_factory=dict)
    freeze_modality_projections: dict[str, bool] = field(default_factory=dict)
    fp16: bool = False
    bf16: bool = True
    use_cpu_initialization: bool = False
    init_model_with_meta_device: bool = False

    def __post_init__(self) -> None:
        """Restore parallelism dataclasses from YAML-compatible dictionaries."""
        parallelism = self.megatron_mimo_parallelism_config
        if isinstance(parallelism, dict):
            module_parallelisms = parallelism.get("module_parallelisms")
            if not isinstance(module_parallelisms, dict):
                raise TypeError("MegatronMIMO parallelism config must define a module_parallelisms mapping.")
            self.megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
                module_parallelisms={
                    name: _restore_module_parallelism_config(module_config)
                    for name, module_config in module_parallelisms.items()
                },
                special_token_ids=dict(parallelism.get("special_token_ids", {})),
            )
            return

        parallelism.module_parallelisms = {
            name: _restore_module_parallelism_config(module_config)
            for name, module_config in parallelism.module_parallelisms.items()
        }

    def as_dict(self) -> dict[str, Any]:
        """Serialize heterogeneous parallelism as YAML-safe primitive mappings."""
        data = super().as_dict()
        parallelism = self.megatron_mimo_parallelism_config
        data["megatron_mimo_parallelism_config"] = {
            "module_parallelisms": {
                name: asdict(module_config) for name, module_config in parallelism.module_parallelisms.items()
            },
            "special_token_ids": dict(parallelism.special_token_ids),
        }
        return data


def _restore_module_parallelism_config(
    config: ModuleParallelismConfig | dict[str, Any],
) -> ModuleParallelismConfig:
    if isinstance(config, ModuleParallelismConfig):
        return config
    if not isinstance(config, dict):
        raise TypeError(
            "MegatronMIMO module parallelism entries must be ModuleParallelismConfig instances or mappings."
        )
    return ModuleParallelismConfig(**config)


def _resolve_builder(path: str):
    module_name, function_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), function_name)


class MegatronMIMOModelBuilder(ModelBuilder[MimoModel, MegatronMIMOModelConfig]):
    """Build MegatronMIMO models with per-module HyperCommGrid and DDP state."""

    def __init__(self, model_config: MegatronMIMOModelConfig):
        super().__init__(model_config)
        self.infra: MegatronMIMOInfra | None = None

    def finalize(self) -> None:
        """Validate heterogeneous rank allocation against the distributed world."""
        if not dist.is_initialized():
            raise RuntimeError("MegatronMIMO requires torch.distributed to be initialized before construction.")
        self._model_config.megatron_mimo_parallelism_config.finalize(dist.get_world_size())

    def build_infra(self) -> MegatronMIMOInfra:
        """Create per-module grids and process-group collections."""
        config = self._model_config
        grids = build_hypercomm_grids(config.megatron_mimo_parallelism_config)
        pg_collections: dict[str, ProcessGroupCollection | None] = {}
        for module_name, grid in grids.items():
            pp_group = grid.get_pg(["pp"])
            pos_embd_pg, embd_pg = populate_embedding_and_position_groups(pp_group)
            if grid.is_current_rank_in_grid():
                first_stage = is_pp_first_stage(pp_group)
                last_stage = is_pp_last_stage(pp_group)
                pg_collections[module_name] = ProcessGroupCollection(
                    tp=grid.get_pg(["tp"]),
                    dp=grid.get_pg(["dp"]),
                    pp=pp_group,
                    cp=grid.get_pg(["cp"]),
                    ep=grid.get_pg(["ep"]),
                    dp_cp=grid.get_pg(["dp", "cp"]),
                    mp=grid.get_pg(["tp", "pp"]),
                    tp_ep_pp=grid.get_pg(["tp", "ep", "pp"]),
                    pos_embd=pos_embd_pg if first_stage else None,
                    embd=embd_pg if first_stage or last_stage else None,
                )
            else:
                pg_collections[module_name] = None

        topology = config.topology or {
            **{name: [MIMO_LANGUAGE_MODULE_KEY] for name in config.modality_keys},
            MIMO_LANGUAGE_MODULE_KEY: [],
        }
        output_ndim = config.module_output_ndim or {
            name: 3 if name == MIMO_LANGUAGE_MODULE_KEY else 2 for name in grids
        }
        self.infra = MegatronMIMOInfra(
            module_to_grid_map=grids,
            topology=topology,
            pg_collections=pg_collections,
            participating_modules=[name for name, pg in pg_collections.items() if pg is not None],
            module_output_ndim=output_ndim,
        )
        return self.infra

    @staticmethod
    def _inject_language_pg(
        spec: ModuleSpec,
        pg_collection: ProcessGroupCollection,
        *,
        pre_process: bool,
        post_process: bool,
    ) -> ModuleSpec:
        spec = copy.deepcopy(spec)
        spec.params = dict(spec.params or {})
        spec.params.update(
            pg_collection=pg_collection,
            pre_process=pre_process,
            post_process=post_process,
        )
        return spec

    @staticmethod
    def _inject_modality_pg(spec: ModuleSpec, pg_collection: ProcessGroupCollection) -> ModuleSpec:
        spec = copy.deepcopy(spec)
        if spec.submodules and "encoders" in spec.submodules:
            for encoder_spec in spec.submodules["encoders"].values():
                encoder_spec.params = dict(encoder_spec.params or {})
                encoder_spec.params["pg_collection"] = pg_collection
                transformer = encoder_spec.params.get("transformer_config")
                if transformer is not None and pg_collection.tp is not None:
                    transformer.tensor_model_parallel_size = pg_collection.tp.size()
        if spec.submodules and "input_projections" in spec.submodules:
            for projection_spec in spec.submodules["input_projections"]:
                if isinstance(projection_spec, ModuleSpec):
                    projection_spec.params = dict(projection_spec.params or {})
                    projection_spec.params.setdefault("tp_group", pg_collection.tp)
        return spec

    def build_model(
        self,
        pg_collection: ProcessGroupCollection | None = None,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
        *,
        infra: MegatronMIMOInfra | None = None,
    ) -> MimoModel:
        """Build the rank-local MIMO module graph."""
        del pg_collection, pre_process, post_process, vp_stage
        config = self._model_config
        infra = infra or self.infra or self.build_infra()
        language_pg = infra.pg_collections.get(MIMO_LANGUAGE_MODULE_KEY)
        language_pp_rank = 0
        if language_pg is not None and language_pg.pp is not None:
            language_pp_rank = dist.get_rank(group=language_pg.pp)
        source_config = config.source_model_config
        language_parallelism = config.megatron_mimo_parallelism_config.module_parallelisms[MIMO_LANGUAGE_MODULE_KEY]
        source_transformer = replace(
            source_config.transformer,
            tensor_model_parallel_size=language_parallelism.tensor_model_parallel_size,
            pipeline_model_parallel_size=language_parallelism.pipeline_model_parallel_size,
            context_parallel_size=language_parallelism.context_parallel_size,
            expert_tensor_parallel_size=language_parallelism.expert_tensor_parallel_size,
            mtp_num_layers=None,
        )
        source_config = replace(source_config, transformer=source_transformer)
        language_spec = _resolve_builder(config.language_spec_builder)(
            source_config,
            pp_rank=language_pp_rank,
        )
        if language_pg is not None:
            language_spec = self._inject_language_pg(
                language_spec,
                language_pg,
                pre_process=is_pp_first_stage(language_pg.pp),
                post_process=is_pp_last_stage(language_pg.pp),
            )

        modality_specs = _resolve_builder(config.modality_spec_builder)(source_config)
        for module_name, spec in tuple(modality_specs.items()):
            module_pg = infra.pg_collections.get(module_name)
            if module_pg is not None:
                modality_specs[module_name] = self._inject_modality_pg(spec, module_pg)
        _finalize_transformer_configs_in_specs(language_spec, modality_specs)
        model = MimoModel(
            MimoModelConfig(
                language_model_spec=language_spec,
                modality_submodules_spec=modality_specs,
                special_token_ids=config.special_token_ids,
                module_to_grid_map=infra.module_to_grid_map,
            )
        )
        self._apply_freezing(model)
        return model

    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection | None = None,
        ddp_config: DistributedDataParallelConfig | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = True,
        mixed_precision_wrapper: Any = None,
        *,
        fp16: bool | None = None,
        bf16: bool | None = None,
        infra: MegatronMIMOInfra | None = None,
    ) -> list[MegatronModule]:
        """Build, hook, cast, and independently DDP-wrap MIMO submodules."""
        del pg_collection, overlap_param_gather_with_optimizer_step, data_parallel_random_init, mixed_precision_wrapper
        if wrap_with_ddp and ddp_config is None:
            raise ValueError("ddp_config is required when wrap_with_ddp is True")
        if use_megatron_fsdp or use_torch_fsdp2:
            raise NotImplementedError("MegatronMIMO supports heterogeneous DDP only; FSDP is not supported.")
        config = self._model_config
        infra = infra or self.infra or self.build_infra()
        model_list: list[MegatronModule] = [self.build_model(infra=infra)]
        for hook in config.pre_wrap_hooks:
            result = hook(model_list)
            if result is not None:
                model_list = result
        if not config.use_cpu_initialization and not config.init_model_with_meta_device:
            model_list = [model.cuda(torch.cuda.current_device()) for model in model_list]
        for model in model_list:
            get_model_config(model).variable_seq_lengths = True
        if fp16 if fp16 is not None else config.fp16:
            model_list = [model.half() for model in model_list]
        elif bf16 if bf16 is not None else config.bf16:
            model_list = [model.bfloat16() for model in model_list]
        for model in model_list:
            self._move_frozen_params_to_device(model)
        if wrap_with_ddp:
            assert ddp_config is not None
            model_list = [
                wrap_megatron_mimo_model_distributed(
                    megatron_mimo_model=model,
                    ddp_config=ddp_config,
                    megatron_mimo_parallelism_config=config.megatron_mimo_parallelism_config,
                    grids=infra.module_to_grid_map,
                    pg_collections=infra.pg_collections,
                )
                for model in model_list
            ]
        for hook in config.post_wrap_hooks:
            result = hook(model_list)
            if result is not None:
                model_list = result
        return model_list

    def _apply_freezing(self, model: MimoModel) -> None:
        config = self._model_config
        if config.freeze_language_model and getattr(model, "language_model", None) is not None:
            for parameter in model.language_model.parameters():
                parameter.requires_grad = False
        for modality, should_freeze in config.freeze_modality_encoders.items():
            if should_freeze and modality in model.modality_submodules:
                for parameter in model.modality_submodules[modality].encoders.parameters():
                    parameter.requires_grad = False
        for modality, should_freeze in config.freeze_modality_projections.items():
            if should_freeze and modality in model.modality_submodules:
                for parameter in model.modality_submodules[modality].input_projections.parameters():
                    parameter.requires_grad = False

    @staticmethod
    def _move_frozen_params_to_device(model: torch.nn.Module) -> None:
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        for parameter in model.parameters():
            if not parameter.requires_grad and parameter.device.type == "cpu":
                parameter.data = parameter.data.to(device)
        for buffer in model.buffers():
            if buffer.device.type == "cpu":
                buffer.data = buffer.data.to(device)


def _finalize_transformer_configs_in_specs(
    language_spec: ModuleSpec,
    modality_specs: dict[str, ModuleSpec],
) -> None:
    """Finalize nested transformer configs embedded in module specifications."""
    from megatron.core.transformer import TransformerConfig

    def _finalize(spec: ModuleSpec | None) -> None:
        if spec is None:
            return
        for value in (spec.params or {}).values():
            if isinstance(value, TransformerConfig) and hasattr(value, "finalize"):
                value.finalize()

    _finalize(language_spec)
    for modality_spec in modality_specs.values():
        submodules = modality_spec.submodules or {}
        for encoder_spec in (submodules.get("encoders") or {}).values():
            _finalize(encoder_spec)
        for projection_spec in submodules.get("input_projections") or []:
            _finalize(projection_spec)
        for projection_spec in submodules.get("output_projections") or []:
            _finalize(projection_spec)
        for decoder_spec in (submodules.get("decoders") or {}).values():
            _finalize(decoder_spec)


__all__ = ["MegatronMIMOInfra", "MegatronMIMOModelBuilder", "MegatronMIMOModelConfig"]
