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
from collections.abc import Iterable
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.table import Table
from utils import parse_dtype, prepare_output_directory, validate_output_path

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import print_rank_0
from megatron.bridge.utils.slurm_utils import resolve_slurm_master_addr, resolve_slurm_master_port


_IGNORE_PRECISION_PARAMS = (
    "e_score_correction_bias",
    "A_log",
    "linear_attn.norm.weight",
    "dt_bias",
    "expert_bias",
    "q_norm.weight",
    "k_norm.weight",
    "block_sparse_moe.gate.weight",
    "mlp.gate.weight",
    "moe.gate.weight",
)
_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
_ROUNDTRIP_ATOL = 1e-1
_ROUNDTRIP_RTOL = 1e-5
_CONSOLE = Console()


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


def _prepare_distributed_output(path: str, *, overwrite: bool, source_paths: Iterable[str] = ()) -> None:
    """Prepare an output directory once and synchronize all ranks."""
    source_paths = tuple(source_paths)
    validate_output_path(path, source_paths=source_paths)
    if torch.distributed.get_rank() == 0:
        prepare_output_directory(path, overwrite=overwrite, source_paths=source_paths)
    torch.distributed.barrier()


def _maybe_generate_pipeline_layout(bridge: AutoBridge, model_provider: GPTModelProvider, pp: int) -> bool:
    """Generate a bridge-specific pipeline layout when the model requires one."""
    if pp <= 1 or not hasattr(bridge._model_bridge, "generate_pipeline_layout"):
        return False
    hf_config = bridge.hf_pretrained.config
    num_layers = hf_config.num_hidden_layers
    mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0) or 0
    model_provider.pipeline_model_parallel_layout = bridge._model_bridge.generate_pipeline_layout(
        num_layers, pp, mtp_layers
    )
    print_rank_0(f"Auto-generated pipeline layout for PP={pp} ({num_layers} layers, {mtp_layers} MTP)")
    return True


def _rebalance_pipeline_layout(saved_layout: list[list[str]], pp: int) -> list[list[str]]:
    """Redistribute a saved flexible pipeline layout across a new PP size."""
    layers = [layer for stage in saved_layout for layer in stage]
    base, remainder = divmod(len(layers), pp)
    rebalanced_layout = []
    offset = 0
    for stage_index in range(pp):
        stage_size = base + (1 if stage_index < remainder else 0)
        rebalanced_layout.append(layers[offset : offset + stage_size])
        offset += stage_size
    return rebalanced_layout


def _maybe_restore_pipeline_layout(
    bridge: AutoBridge, model_provider: GPTModelProvider, megatron_path: str, pp: int
) -> None:
    """Restore a serialized pipeline layout or regenerate it for export."""
    checkpoint_path = Path(megatron_path)
    iteration_paths = [path for path in checkpoint_path.glob("iter_*") if path.is_dir()]

    def _iteration_number(path: Path) -> int:
        try:
            return int(path.name.removeprefix("iter_"))
        except ValueError:
            return -1

    config_root = max(iteration_paths, key=_iteration_number) if iteration_paths else checkpoint_path
    config_path = config_root / "run_config.yaml"
    root_config_path = checkpoint_path / "run_config.yaml"
    if not config_path.exists() and root_config_path.exists():
        config_path = root_config_path
    if config_path.exists():
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file) or {}
        model_config = config.get("model", {})
        saved_layout = model_config.get("pipeline_model_parallel_layout")
        saved_pp = model_config.get("pipeline_model_parallel_size")
        is_valid_layout = isinstance(saved_layout, list) and all(isinstance(stage, list) for stage in saved_layout)
        is_same_topology = saved_pp == pp or (saved_pp is None and len(saved_layout or []) % pp == 0)
        if is_valid_layout and is_same_topology and len(saved_layout) % pp == 0:
            model_provider.pipeline_model_parallel_layout = saved_layout
            print_rank_0(f"Read pipeline layout from checkpoint ({len(saved_layout)} PP/VPP stages)")
            return
        if is_valid_layout:
            print_rank_0(
                f"Rebalancing checkpoint pipeline layout from PP={saved_pp or 'unknown'} to requested PP={pp}"
            )
            if not _maybe_generate_pipeline_layout(bridge, model_provider, pp):
                model_provider.pipeline_model_parallel_layout = _rebalance_pipeline_layout(saved_layout, pp)
            return
    _maybe_generate_pipeline_layout(bridge, model_provider, pp)


def _configure_model_provider(
    model_provider: GPTModelProvider,
    *,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    dtype: torch.dtype,
) -> None:
    """Apply distributed parallelism and dtype settings to a model provider."""
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = dtype
    model_provider.params_dtype = dtype


def _hf_tokenizer_kwargs(bridge: AutoBridge, *, trust_remote_code: bool) -> dict[str, object]:
    """Build tokenizer metadata for a saved Megatron checkpoint."""
    tokenizer_kwargs: dict[str, object] = {}
    if hasattr(bridge._model_bridge, "get_hf_tokenizer_kwargs"):
        tokenizer_kwargs = bridge._model_bridge.get_hf_tokenizer_kwargs() or {}
    if trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    if bridge.hf_model_revision is not None:
        tokenizer_kwargs["revision"] = bridge.hf_model_revision
    return tokenizer_kwargs


def _roundtrip_weights_match(name: str, exported: torch.Tensor, original: torch.Tensor) -> tuple[bool, bool]:
    """Compare one exported parameter with its original Hugging Face value.

    Returns:
        A pair containing whether the values match and whether comparison was
        skipped because either value uses a lossy FP8 dtype.
    """
    if original.dtype in _FP8_DTYPES or exported.dtype in _FP8_DTYPES:
        return True, True
    if exported.dtype != original.dtype or any(part in name for part in _IGNORE_PRECISION_PARAMS):
        exported = exported.float()
        original = original.float()
    return (
        torch.allclose(
            exported,
            original.to(exported.device),
            atol=_ROUNDTRIP_ATOL,
            rtol=_ROUNDTRIP_RTOL,
        ),
        False,
    )


def _verify_roundtrip_weights(bridge: AutoBridge, megatron_model: list[torch.nn.Module]) -> None:
    """Verify exported Megatron weights against the original Hugging Face state."""
    is_rank_0 = torch.distributed.get_rank() == 0
    all_match = True
    fp8_skip_count = 0
    fp8_skip_samples: list[str] = []
    table = None
    if is_rank_0:
        table = Table(title="Hugging Face Weights Verification")
        table.add_column("Weight Name", style="cyan")
        table.add_column("Shape")
        table.add_column("DType")
        table.add_column("Device")
        table.add_column("Matches Original", justify="center")

    for name, exported in bridge.export_hf_weights(megatron_model, show_progress=False):
        if not is_rank_0:
            continue
        assert table is not None
        original = bridge.hf_pretrained.state[name]
        match, skipped_fp8 = _roundtrip_weights_match(name, exported, original)
        if skipped_fp8:
            fp8_skip_count += 1
            if len(fp8_skip_samples) < 20:
                fp8_skip_samples.append(f"{name}: exported {exported.dtype} vs original {original.dtype}")
        all_match = all_match and match
        table.add_row(
            name,
            str(tuple(exported.shape)),
            str(exported.dtype).replace("torch.", ""),
            str(exported.device),
            "✅" if match else "❌",
        )

    if is_rank_0:
        assert table is not None
        if fp8_skip_count:
            _CONSOLE.print(
                f"[yellow]WARNING: {fp8_skip_count} FP8 params skipped allclose (dequantisation is lossy):[/yellow]"
            )
            for entry in fp8_skip_samples:
                _CONSOLE.print(f"  [yellow]{entry}[/yellow]")
            if fp8_skip_count > len(fp8_skip_samples):
                _CONSOLE.print(f"  [yellow]... and {fp8_skip_count - len(fp8_skip_samples)} more[/yellow]")
        _CONSOLE.print(table)

    mismatch = torch.tensor(not all_match, dtype=torch.int32, device=torch.cuda.current_device())
    torch.distributed.broadcast(mismatch, src=0)
    if mismatch.item():
        raise ValueError("Weight mismatch detected")


@torchrun_main
def import_checkpoint(
    *,
    hf_model: str,
    hf_revision: str | None,
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
        hf_revision: Hugging Face Hub revision to load.
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
    _prepare_distributed_output(megatron_path, overwrite=overwrite, source_paths=[hf_model])
    dtype = parse_dtype(torch_dtype)

    print_rank_0(f"GPU import: {hf_model} -> {megatron_path}")
    print_rank_0(f"Parallelism: TP={tp} PP={pp} EP={ep} ETP={etp}; dtype={torch_dtype}")
    revision_kwargs = {"revision": hf_revision} if hf_revision is not None else {}
    bridge = AutoBridge.from_hf_pretrained(
        hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model),
        torch_dtype=dtype,
        **revision_kwargs,
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)
    _configure_model_provider(model_provider, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype)
    _maybe_generate_pipeline_layout(bridge, model_provider, pp)
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    bridge.save_megatron_model(
        megatron_model,
        megatron_path,
        hf_tokenizer_path=hf_model,
        hf_tokenizer_kwargs=_hf_tokenizer_kwargs(bridge, trust_remote_code=trust_remote_code),
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
    _prepare_distributed_output(hf_path, overwrite=overwrite, source_paths=[megatron_path, hf_model])
    dtype = parse_dtype(torch_dtype)

    print_rank_0(f"GPU export: {megatron_path} -> {hf_path}")
    print_rank_0(f"Parallelism: TP={tp} PP={pp} EP={ep} ETP={etp}; dtype={torch_dtype}")
    trusted = is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model)
    bridge = AutoBridge.from_hf_pretrained(
        hf_model,
        trust_remote_code=trusted,
        torch_dtype=dtype,
    )
    checkpoint_config_bridge = AutoBridge.from_auto_config(
        megatron_path,
        hf_model,
        trust_remote_code=trusted,
    )
    # Preserve the reference wrapper's streaming state source and shard map while
    # exporting the checkpoint-derived architecture and vocabulary configuration.
    bridge.hf_pretrained.config = checkpoint_config_bridge.hf_pretrained
    model_provider = bridge.to_megatron_provider(load_weights=False)
    _configure_model_provider(model_provider, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype)
    _maybe_restore_pipeline_layout(bridge, model_provider, megatron_path, pp)
    resolved_pipeline_layout = model_provider.pipeline_model_parallel_layout
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


@torchrun_main
def roundtrip_checkpoint(
    *,
    hf_model: str,
    tp: int,
    pp: int,
    ep: int,
    etp: int,
    trust_remote_code: bool,
    distributed_timeout_minutes: int | None,
) -> None:
    """Validate a Hugging Face to Megatron to Hugging Face round trip.

    Args:
        hf_model: Hugging Face model ID or local path.
        tp: Tensor parallelism size.
        pp: Pipeline parallelism size.
        ep: Expert parallelism size.
        etp: Expert tensor parallelism size.
        trust_remote_code: Allow custom Hugging Face repository code.
        distributed_timeout_minutes: Process-group timeout in minutes.
    """
    _ensure_distributed_initialized(distributed_timeout_minutes)
    dtype = torch.bfloat16
    print_rank_0(f"GPU round trip: {hf_model}")
    print_rank_0(f"Parallelism: TP={tp} PP={pp} EP={ep} ETP={etp}; dtype=bfloat16")
    bridge = AutoBridge.from_hf_pretrained(
        hf_model,
        trust_remote_code=is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model),
        torch_dtype=dtype,
    )

    model_provider = bridge.to_megatron_provider(load_weights=True)
    _configure_model_provider(model_provider, tp=tp, pp=pp, ep=ep, etp=etp, dtype=dtype)
    _maybe_generate_pipeline_layout(bridge, model_provider, pp)
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)
    _verify_roundtrip_weights(bridge, megatron_model)
    print_rank_0("GPU round-trip validation complete")
