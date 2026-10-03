# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Distributed functional check for pipeline-stage-local HF export."""

import argparse
import hashlib
import os

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import _get_pp_rank
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.slurm_utils import resolve_slurm_master_addr, resolve_slurm_master_port


def _configure_slurm_distributed_environment() -> None:
    if os.environ.get("WORLD_SIZE") is not None or os.environ.get("SLURM_NTASKS") is None:
        return
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    master_addr = resolve_slurm_master_addr()
    master_port = resolve_slurm_master_port()
    if master_addr is not None:
        os.environ["MASTER_ADDR"] = master_addr
    if master_port is not None:
        os.environ["MASTER_PORT"] = str(master_port)


def _signature(tensor: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    return tuple(tensor.shape), str(tensor.dtype), digest


def _verify(bridge: AutoBridge, megatron_model, baseline: dict) -> None:
    local = {}
    for name, weight in bridge.export_hf_weights(
        megatron_model, cpu=False, show_progress=False, current_pp_stage_only=True
    ):
        if name in local:
            raise AssertionError(f"PP-local export yielded duplicate tensor {name}")
        local[name] = _signature(weight)

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    baselines = [None] * world_size
    records = [None] * world_size
    torch.distributed.all_gather_object(baselines, baseline)
    torch.distributed.all_gather_object(records, (_get_pp_rank(megatron_model), local))

    errors = []
    if rank == 0:
        reference = baselines[0]
        errors.extend(
            f"full export differs on rank {idx}"
            for idx, candidate in enumerate(baselines)
            if candidate != reference
        )
        stage_replicas = {}
        for world_rank, (pp_rank, signatures) in enumerate(records):
            stage_replicas.setdefault(pp_rank, []).append((world_rank, signatures))
            for name, signature in signatures.items():
                if reference.get(name) != signature:
                    errors.append(f"rank {world_rank} PP-local tensor differs from baseline: {name}")

        stage_streams = {}
        for pp_rank, replicas in stage_replicas.items():
            representative_rank, representative = replicas[0]
            stage_streams[pp_rank] = representative
            for replica_rank, replica in replicas[1:]:
                if replica != representative:
                    errors.append(f"PP stage {pp_rank} differs between ranks {representative_rank} and {replica_rank}")

        owners = {}
        for pp_rank, signatures in stage_streams.items():
            for name in signatures:
                owners.setdefault(name, []).append(pp_rank)
        overlaps = {name: stages for name, stages in owners.items() if len(stages) > 1}
        local_names = set(owners)
        baseline_names = set(reference)
        if overlaps:
            errors.append(f"PP-local rank outputs overlap: {overlaps}")
        if local_names != baseline_names:
            errors.append(
                f"PP-local union differs from baseline: missing={sorted(baseline_names - local_names)}, "
                f"extra={sorted(local_names - baseline_names)}"
            )
        if not errors:
            counts = {stage: len(weights) for stage, weights in stage_streams.items()}
            print(f"PP-local verification passed: stage_counts={counts}, intersection=0, union={len(local_names)}")

    payload = [errors]
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0]:
        raise AssertionError("PP-local export mismatch: " + "; ".join(payload[0]))


@torchrun_main
def main(hf_model_id: str, tp: int, pp: int, ep: int, etp: int) -> None:
    _configure_slurm_distributed_environment()
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        trust_remote_code=is_safe_repo(trust_remote_code=False, hf_path=hf_model_id),
        torch_dtype=torch.bfloat16,
    )
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = tp
    provider.pipeline_model_parallel_size = pp
    provider.pipeline_dtype = torch.bfloat16
    provider.params_dtype = torch.bfloat16
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = etp
    provider.finalize()
    provider.initialize_model_parallel(seed=0)
    megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)
    baseline = {
        name: _signature(weight)
        for name, weight in bridge.export_hf_weights(megatron_model, show_progress=False)
    }
    _verify(bridge, megatron_model, baseline)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-id", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--etp", type=int, default=1)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(args.hf_model_id, args.tp, args.pp, args.ep, args.etp)
