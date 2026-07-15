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
"""Distributed GPU checkpoint conversion backend."""

import datetime
import os
from pathlib import Path

import torch
import yaml
from utils import parse_dtype, prepare_output_directory

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import print_rank_0
from megatron.bridge.utils.slurm_utils import resolve_slurm_master_addr, resolve_slurm_master_port


def _ensure_distributed_initialized(timeout_minutes: int | None) -> None:
    """Initialize NCCL from NeMo Run's torchrun or Slurm task environment."""
    if torch.distributed.is_initialized():
        return
    if os.environ.get("WORLD_SIZE") is None and os.environ.get("SLURM_NTASKS") is not None:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        master_addr = resolve_slurm_master_addr()
        master_port = resolve_slurm_master_port()
        if master_addr is not None:
            os.environ["MASTER_ADDR"] = master_addr
        if master_port is not None:
            os.environ["MASTER_PORT"] = str(master_port)
    if os.environ.get("WORLD_SIZE") is None:
        raise RuntimeError("GPU conversion must be launched through NeMo Run's local or Slurm executor.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    kwargs: dict[str, object] = {"backend": "nccl"}
    if timeout_minutes is not None:
        kwargs["timeout"] = datetime.timedelta(minutes=timeout_minutes)
    torch.distributed.init_process_group(**kwargs)


def _prepare_distributed_output(path: str, *, overwrite: bool) -> None:
    """Prepare an output directory once and synchronize all ranks."""
    if torch.distributed.get_rank() == 0:
        prepare_output_directory(path, overwrite=overwrite)
    torch.distributed.barrier()


def _maybe_generate_pipeline_layout(bridge: AutoBridge, model_provider: GPTModelProvider, pp: int) -> None:
    """Generate a bridge-specific pipeline layout when the model requires one."""
    if pp <= 1 or not hasattr(bridge._model_bridge, "generate_pipeline_layout"):
        return
    hf_config = bridge.hf_pretrained.config
    num_layers = hf_config.num_hidden_layers
    mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or 0
    model_provider.pipeline_model_parallel_layout = bridge._model_bridge.generate_pipeline_layout(
        num_layers, pp, mtp_layers
    )
    print_rank_0(f"Auto-generated pipeline layout for PP={pp} ({num_layers} layers, {mtp_layers} MTP)")


def _maybe_restore_pipeline_layout(
    bridge: AutoBridge, model_provider: GPTModelProvider, megatron_path: str, pp: int
) -> None:
    """Restore a serialized pipeline layout or regenerate it for export."""
    if pp <= 1:
        return
    checkpoint_path = Path(megatron_path)
    for candidate in [checkpoint_path, *checkpoint_path.glob("iter_*")]:
        config_path = candidate / "run_config.yaml"
        if not config_path.exists():
            continue
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file)
        saved_layout = config.get("model", {}).get("pipeline_model_parallel_layout")
        if isinstance(saved_layout, list):
            model_provider.pipeline_model_parallel_layout = saved_layout
            print_rank_0(f"Read pipeline layout from checkpoint ({len(saved_layout)} stages)")
            return
    _maybe_generate_pipeline_layout(bridge, model_provider, pp)


@torchrun_main
def import_checkpoint(
    *,
    hf_model: str,
    megatron_path: str,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    torch_dtype: str,
    trust_remote_code: bool,
    distributed_timeout_minutes: int | None,
    overwrite: bool,
) -> None:
    """Import a Hugging Face model into a distributed Megatron checkpoint.

    Args:
        hf_model: Hugging Face model ID or local path.
        megatron_path: Destination Megatron checkpoint path.
        tp: Tensor parallelism size.
        pp: Pipeline parallelism size.
        ep: Expert parallelism size.
        etp: Expert tensor parallelism size.
        torch_dtype: Weight dtype name.
        trust_remote_code: Allow custom Hugging Face repository code.
        distributed_timeout_minutes: Process-group timeout in minutes.
        overwrite: Delete a non-empty destination before conversion.
    """
    _ensure_distributed_initialized(distributed_timeout_minutes)
    _prepare_distributed_output(megatron_path, overwrite=overwrite)
    dtype = parse_dtype(torch_dtype)

    print_rank_0(f"GPU import: {hf_model} -> {megatron_path}")
    print_rank_0(f"Parallelism: TP={tp} PP={pp} EP={ep} ETP={etp}; dtype={torch_dtype}")
    bridge = AutoBridge.from_hf_pretrained(
        hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model),
        torch_dtype=dtype,
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = dtype
    model_provider.params_dtype = dtype
    _maybe_generate_pipeline_layout(bridge, model_provider, pp)
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    tokenizer_kwargs = {}
    if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
        tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs() or {}
    if trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    bridge.save_megatron_model(
        megatron_model,
        megatron_path,
        hf_tokenizer_path=hf_model,
        hf_tokenizer_kwargs=tokenizer_kwargs,
    )
    print_rank_0(f"GPU import complete: {megatron_path}")


@torchrun_main
def export_checkpoint(
    *,
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    torch_dtype: str,
    trust_remote_code: bool,
    strict: bool,
    show_progress: bool,
    distributed_save: bool,
    save_every_n_ranks: int,
    distributed_timeout_minutes: int | None,
    export_weight_dtype: str | None,
    overwrite: bool,
) -> None:
    """Export a distributed Megatron checkpoint to Hugging Face format.

    Args:
        hf_model: Hugging Face model ID or local config reference.
        megatron_path: Source Megatron checkpoint path.
        hf_path: Destination Hugging Face checkpoint path.
        tp: Tensor parallelism size.
        pp: Pipeline parallelism size.
        ep: Expert parallelism size.
        etp: Expert tensor parallelism size.
        torch_dtype: Model dtype name.
        trust_remote_code: Allow custom Hugging Face repository code.
        strict: Require source and destination parameter keys to match.
        show_progress: Display export progress.
        distributed_save: Let ranks save assigned Hugging Face shards independently.
        save_every_n_ranks: Write files from every Nth rank.
        distributed_timeout_minutes: Process-group timeout in minutes.
        export_weight_dtype: Optional dtype for exported weights.
        overwrite: Delete a non-empty destination before conversion.
    """
    _ensure_distributed_initialized(distributed_timeout_minutes)
    if not Path(megatron_path).exists():
        raise FileNotFoundError(f"Megatron checkpoint does not exist: {megatron_path}")
    _prepare_distributed_output(hf_path, overwrite=overwrite)
    dtype = parse_dtype(torch_dtype)

    print_rank_0(f"GPU export: {megatron_path} -> {hf_path}")
    print_rank_0(f"Parallelism: TP={tp} PP={pp} EP={ep} ETP={etp}; dtype={torch_dtype}")
    bridge = AutoBridge.from_hf_pretrained(
        hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model),
        torch_dtype=dtype,
    )
    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = dtype
    model_provider.params_dtype = dtype
    _maybe_restore_pipeline_layout(bridge, model_provider, megatron_path, pp)
    resolved_pipeline_layout = model_provider.pipeline_model_parallel_layout if pp > 1 else None
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    model_parallel_overrides: dict[str, object] = {
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "expert_model_parallel_size": ep,
        "expert_tensor_parallel_size": etp,
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
    }
    if isinstance(resolved_pipeline_layout, list):
        model_parallel_overrides["pipeline_model_parallel_layout"] = resolved_pipeline_layout

    megatron_model = bridge.load_megatron_model(
        megatron_path,
        mp_overrides=model_parallel_overrides,
        wrap_with_ddp=False,
    )
    bridge.save_hf_pretrained(
        megatron_model,
        hf_path,
        show_progress=show_progress,
        strict=strict,
        distributed_save=distributed_save,
        save_every_n_ranks=save_every_n_ranks,
        weight_dtype=parse_dtype(export_weight_dtype) if export_weight_dtype else None,
    )
    print_rank_0(f"GPU export complete: {hf_path}")
