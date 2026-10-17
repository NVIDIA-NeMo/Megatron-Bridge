# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""DP2 MIMO optimizer layout regression with real DDP and optimizer updates.

Run using ``uv run python -m torch.distributed.run --standalone
--nproc_per_node=2 tests/functional_tests/training/mimo_optimizer_layout.py
--output /tmp/mimo-layout``. Component forwards intentionally isolate the
optimizer contract from the MIMO pipeline schedule and tokenizer/model assets.
"""

import argparse
import copy
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config import MimoModelConfig
from megatron.core.models.mimo.optimizer import MimoOptimizer, get_mimo_optimizer
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import unwrap_model

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_ddp import wrap_megatron_mimo_model_distributed
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import _cast_model_dtype_except_fp32_marked
from megatron.bridge.training.checkpointing import _build_sharded_state_dict_metadata, _uses_distributed_optimizer
from megatron.bridge.training.config import (
    CheckpointConfig,
    OptimizerConfig,
    _validate_and_sync_distributed_optimizer_settings,
)
from megatron.bridge.training.setup_megatron_mimo import _to_mcore_optimizer_config
from megatron.bridge.training.train import force_param_sync


logger = logging.getLogger(__name__)


class TinyComponent(MegatronModule):
    """Matrices, an Adam-managed bias, a tied weight, and a frozen matrix."""

    def __init__(
        self, config: TransformerConfig, pg_collection: ProcessGroupCollection, matrices_only: bool = False
    ) -> None:
        super().__init__(config)
        self.pg_collection = pg_collection
        self.linear = torch.nn.Linear(16, 16, bias=not matrices_only)
        self.second = torch.nn.Linear(16, 32, bias=False)
        self.tied = torch.nn.Linear(16, 16, bias=False)
        self.tied.weight = self.linear.weight
        self.excluded = torch.nn.Linear(16, 16, bias=False).requires_grad_(not matrices_only)
        self.excluded.weight.use_muon = False
        self.output = torch.nn.Linear(16, 16, bias=False).requires_grad_(not matrices_only)
        self.output.weight.is_embedding_or_output_parameter = True
        self.frozen = torch.nn.Linear(16, 16, bias=False).requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.linear(x)
            + self.second(x).reshape(-1, 2, 16).sum(dim=1)
            + self.tied(x)
            + self.frozen(x)
            + self.excluded(x)
            + self.output(x)
        )


def components(model: MimoModel) -> list[DistributedDataParallel]:
    return [model.language_model, model.modality_submodules["vision"]]


def build(
    optimizer_name: str, mode: str, *, matrices_only: bool = False, overlap: bool = False
) -> tuple[MimoModel, MimoOptimizer]:
    torch.manual_seed(123)
    config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=2, bf16=True)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    grid = HyperCommGrid([1, 1, 1, dist.get_world_size()], ["tp", "cp", "pp", "dp"])
    grids = {"language": grid, "vision": grid}
    spec = ModuleSpec(
        module=TinyComponent, params={"config": config, "pg_collection": pg, "matrices_only": matrices_only}
    )
    model = MimoModel(
        MimoModelConfig(
            language_model_spec=spec,
            modality_submodules_spec={
                "vision": ModuleSpec(
                    module=VisionModalitySubmodules,
                    params={"pg_collection": pg},
                    submodules={"encoders": {"tiny": spec}},
                )
            },
            module_to_grid_map=grids,
        ),
        cp_group=pg.cp,
        tp_group=pg.tp,
    ).cuda()
    original = {id(p): name for name, p in model.named_parameters()}
    model = _cast_model_dtype_except_fp32_marked([model], torch.bfloat16)[0]
    optimizer_config = OptimizerConfig(
        optimizer=optimizer_name,
        use_distributed_optimizer=mode != "replicated",
        bf16=True,
        lr=0.001,
        min_lr=0.001,
        weight_decay=0.0,
        clip_grad=0.0,
        muon_split_qkv=False,
    )
    ddp = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True, overlap_grad_reduce=overlap, overlap_param_gather=overlap
    )
    _validate_and_sync_distributed_optimizer_settings(SimpleNamespace(optimizer=optimizer_config, ddp=ddp))
    model = wrap_megatron_mimo_model_distributed(
        model,
        ddp,
        MegatronMIMOParallelismConfig(
            module_parallelisms={name: ModuleParallelismConfig(data_parallel_size=2) for name in grids}
        ),
        grids,
        {"language": pg, "vision": pg},
        use_layer_wise_distributed_optimizer=optimizer_config.use_layer_wise_distributed_optimizer,
        use_layer_wise_param_layout=mode == "full",
    )
    # Dtype casting and DDP must preserve parameter identity, including tied aliases.
    assert set(original) == {id(p) for p in model.parameters()}
    for component in components(model):
        raw = unwrap_model(component)
        tiny = raw if isinstance(raw, TinyComponent) else raw.encoders["tiny"]
        assert tiny.linear.weight is tiny.tied.weight
        assert raw.pg_collection.dp_cp.size() == dist.get_world_size() == 2
        layout = component.full_param_layout
        if layout is not None:
            entries = {
                id(p): key.is_managed_by_layer_wise_optimizer
                for key, buffer in layout.layouts.items()
                for p in buffer.param_index_map
            }
            assert set(entries) == {id(p) for p in component.parameters() if p.requires_grad}
            if mode == "full":
                for identity, layer_wise in entries.items():
                    expected = optimizer_name == "muon" and original[identity].endswith(
                        (".linear.weight", ".second.weight")
                    )
                    assert layer_wise == expected, original[identity]
            logger.info("%s %s layout: %s", optimizer_name, mode, {original[k]: v for k, v in entries.items()})
        else:
            logger.info("%s %s: no full layout", optimizer_name, mode)
    # Deliberately construct before asserting the expected layout mode: the old
    # normalization must fail in Core's real _shard_params_from_layout assertion.
    optimizer = get_mimo_optimizer(model, _to_mcore_optimizer_config(optimizer_config))
    for component in components(model):
        if optimizer_name != "adam" and mode == "compact":
            assert component.full_param_layout is None
            assert not component.ddp_config.use_distributed_optimizer
    for module_name, info in optimizer.module_infos.items():
        opt = info.optimizer
        if optimizer_name == "adam" and mode != "replicated":
            children = getattr(opt, "chained_optimizers", [opt])
            assert children and all(isinstance(child, DistributedOptimizer) for child in children)
        elif optimizer_name == "muon" and mode == "compact":
            assert isinstance(opt, LayerWiseDistributedOptimizer)
            assert not opt.use_buffer_param_sync
        elif optimizer_name == "muon" and mode == "full":
            assert isinstance(opt.chained_optimizers[0], LayerWiseDistributedOptimizer)
            assert opt.chained_optimizers[0].use_buffer_param_sync
            assert any(isinstance(child, DistributedOptimizer) for child in opt.chained_optimizers)
        if optimizer_name == "muon" and mode != "replicated":
            layer_wise = opt if mode == "compact" else opt.chained_optimizers[0]
            owned = [{id(p) for p in shard} for shard in layer_wise.dp_cp_params_list]
            component = model.language_model if module_name == "language" else model.modality_submodules[module_name]
            trainable = {id(p) for p in component.parameters() if p.requires_grad}
            matrices = {
                identity for identity in trainable if original[identity].endswith((".linear.weight", ".second.weight"))
            }
            assert set.union(*owned) == (trainable if mode == "compact" else matrices)
            assert not owned[0].intersection(owned[1])
            assert all(shard.intersection(matrices) for shard in owned)
    logger.info("%s %s optimizers: %s", optimizer_name, mode, [type(o).__name__ for o in optimizer.chained_optimizers])
    return model, optimizer


def step(
    model: MimoModel, optimizer: MimoOptimizer, index: int
) -> tuple[float, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    optimizer.zero_grad()
    for component in components(model):
        component.zero_grad_buffer()
    x = torch.arange(64, device="cuda", dtype=torch.float32).reshape(4, 16) / 64
    x = (x + 0.125 * (index + dist.get_rank())).bfloat16()
    # Compute the gradient oracle with plain modules, outside DDP's accumulation
    # and collective paths. Explicit averaging defines the global-batch gradient.
    names = {id(p): name for name, p in model.named_parameters()}
    expected_grads = {}
    for component in components(model):
        raw = unwrap_model(component)
        tiny = raw if isinstance(raw, TinyComponent) else raw.encoders["tiny"]
        plain = TinyComponent(tiny.config, tiny.pg_collection).cuda().bfloat16()
        plain.load_state_dict(tiny.state_dict())
        plain(x).float().square().mean().backward()
        for name, param in tiny.named_parameters():
            if param.requires_grad:
                gradient = dict(plain.named_parameters())[name].grad.float()
                dist.all_reduce(gradient)
                expected_grads[names[id(param)]] = gradient / dist.get_world_size()
    losses = [model.language_model(x).float().square().mean()]
    losses.append(model.modality_submodules["vision"](encoder_inputs={"tiny": {"x": x}}).float().square().mean())
    sum(losses).backward()
    expected_norm = torch.stack([g.square().sum() for g in expected_grads.values()]).sum().sqrt()
    for component in components(model):
        component.finish_grad_sync()
    success, norm, _ = optimizer.step()
    assert success
    assert norm is not None
    torch.testing.assert_close(torch.tensor(norm, device="cuda"), expected_norm, rtol=1e-5, atol=1e-6)
    force_param_sync([model], optimizer)
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    for weight in weights.values():
        reference = weight.clone()
        dist.broadcast(reference, 0)
        torch.testing.assert_close(weight, reference, rtol=0, atol=0)
        assert torch.isfinite(weight).all()
    return norm, weights, expected_grads


def checkpoint(model: MimoModel, optimizer: MimoOptimizer, path: Path, *, loading: bool) -> None:
    metadata = _build_sharded_state_dict_metadata(
        _uses_distributed_optimizer(optimizer.config, optimizer), CheckpointConfig(ckpt_format="torch_dist")
    )
    model_sd = model.sharded_state_dict(metadata=metadata)
    state = {
        "model": model_sd,
        "optimizer": optimizer.sharded_state_dict(model_sd, is_loading=loading, metadata=metadata),
    }
    if loading:
        state = dist_checkpointing.load(state, str(path))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        force_param_sync([model], optimizer)
    else:
        if dist.get_rank() == 0:
            path.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        dist_checkpointing.save(state, str(path), content_metadata=metadata)


def optimizer_snapshot(optimizer: MegatronOptimizer) -> object:
    """Capture local FP32 masters and moments, including nested optimizer leaves."""
    children = getattr(optimizer, "chained_optimizers", None)
    if children is not None:
        return [optimizer_snapshot(child) for child in children]
    state = optimizer.optimizer.state_dict()
    return {
        "parameters": [p.detach().clone() for p in optimizer.get_parameters()],
        "state": copy.deepcopy(state["state"]),
        # Restore materializes step metadata on empty groups as well; only
        # nonempty groups have an update counter to compare.
        "steps": [copy.deepcopy(g.get("step", 0)) for g in state["param_groups"] if g["params"]],
    }


def run(output: Path, optimizer_name: str, mode: str, *, overlap: bool) -> None:
    model, optimizer = build(optimizer_name, mode, overlap=overlap)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    first = step(model, optimizer, 0)
    for name, value in first[1].items():
        if "frozen" in name:
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
        else:
            assert not torch.equal(value, initial[name]), name
    checkpoint(model, optimizer, output / f"{optimizer_name}-{mode}-{overlap}", loading=False)
    continued = step(model, optimizer, 1)
    continued_optimizer = optimizer_snapshot(optimizer)
    resumed_model, resumed_optimizer = build(optimizer_name, mode, overlap=overlap)
    checkpoint(resumed_model, resumed_optimizer, output / f"{optimizer_name}-{mode}-{overlap}", loading=True)
    resumed = step(resumed_model, resumed_optimizer, 1)
    torch.testing.assert_close(resumed, continued, rtol=0, atol=0)
    torch.testing.assert_close(optimizer_snapshot(resumed_optimizer), continued_optimizer, rtol=0, atol=0)
    # Replicated optimizers provide a numerical oracle with no distributed
    # parameter ownership; compare every unique trainable/frozen parameter.
    reference_model, reference_optimizer = build(optimizer_name, "replicated")
    reference_first = step(reference_model, reference_optimizer, 0)
    reference_second = step(reference_model, reference_optimizer, 1)
    torch.testing.assert_close(first, reference_first, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(continued, reference_second, rtol=1e-5, atol=1e-6)
    logger.info(
        "PASS %s %s overlap=%s: DP2 updates, replicated oracle, frozen/tied identity, checkpoint continuation",
        optimizer_name,
        mode,
        overlap,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=["construct", "compact", "full", "adam", "all"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel()
    try:
        if args.case in ("construct", "all"):
            build("muon", "compact", matrices_only=True)
        for name, mode in [("muon", "compact"), ("muon", "full"), ("adam", "full")]:
            if args.case in ("all", "adam" if name == "adam" else mode):
                for overlap in (False, True):
                    run(args.output, name, mode, overlap=overlap)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
