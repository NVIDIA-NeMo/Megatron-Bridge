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
"""Single-process CPU checkpoint conversion backend."""

import logging
import shutil
from pathlib import Path

from utils import parse_dtype, prepare_output_directory

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo


logger = logging.getLogger(__name__)


def _find_run_config(checkpoint_path: Path) -> Path:
    """Find the run config used to synthesize an exported HF config."""
    config_files = list(checkpoint_path.glob("**/run_config.yaml"))
    if config_files:
        return config_files[0]

    iteration_dirs = [path for path in checkpoint_path.iterdir() if path.is_dir() and path.name.startswith("iter_")]
    if iteration_dirs:
        latest_iteration = max(iteration_dirs, key=lambda path: int(path.name.removeprefix("iter_")))
        config_path = latest_iteration / "run_config.yaml"
        if config_path.exists():
            return config_path
    raise FileNotFoundError(
        f"Could not find run_config.yaml in {checkpoint_path}. Ensure this is a valid Megatron checkpoint."
    )


def import_checkpoint(
    *,
    hf_model: str,
    hf_revision: str | None,
    megatron_path: str,
    torch_dtype: str,
    trust_remote_code: bool,
    overwrite: bool,
) -> None:
    """Import a Hugging Face model into a CPU-initialized Megatron checkpoint.

    Args:
        hf_model: Hugging Face model ID or local path.
        hf_revision: Hugging Face Hub revision to load.
        megatron_path: Destination Megatron checkpoint path.
        torch_dtype: Weight dtype name.
        trust_remote_code: Allow custom Hugging Face repository code.
        overwrite: Delete a non-empty destination before conversion.
    """
    prepare_output_directory(megatron_path, overwrite=overwrite, source_paths=[hf_model])
    trusted = is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model)
    logger.info("CPU import: %s -> %s", hf_model, megatron_path)
    revision_kwargs = {"revision": hf_revision} if hf_revision is not None else {}
    AutoBridge.import_ckpt(
        hf_model_id=hf_model,
        megatron_path=megatron_path,
        torch_dtype=parse_dtype(torch_dtype),
        device_map="cpu",
        trust_remote_code=trusted,
        **revision_kwargs,
    )
    logger.info("CPU import complete: %s", megatron_path)


def export_checkpoint(
    *,
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    show_progress: bool,
    strict: bool,
    trust_remote_code: bool,
    overwrite: bool,
) -> None:
    """Export a Megatron checkpoint to Hugging Face format on CPU.

    Args:
        hf_model: Hugging Face model ID or local config reference.
        megatron_path: Source Megatron checkpoint path.
        hf_path: Destination Hugging Face checkpoint path.
        show_progress: Display conversion progress.
        strict: Require source and destination parameter keys to match.
        trust_remote_code: Allow custom Hugging Face repository code.
        overwrite: Delete a non-empty destination before conversion.
    """
    checkpoint_path = Path(megatron_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Megatron checkpoint does not exist: {checkpoint_path}")
    config_path = _find_run_config(checkpoint_path)

    trusted = is_safe_repo(trust_remote_code=trust_remote_code, hf_path=hf_model)
    logger.info("CPU export: %s -> %s", megatron_path, hf_path)
    logger.info("Using Megatron run config: %s", config_path)
    bridge = AutoBridge.from_auto_config(megatron_path, hf_model, trust_remote_code=trusted)
    prepare_output_directory(hf_path, overwrite=overwrite, source_paths=[megatron_path, hf_model])
    try:
        bridge.export_ckpt(
            megatron_path=megatron_path,
            hf_path=hf_path,
            show_progress=show_progress,
            strict=strict,
        )
    except Exception as error:
        if strict:
            shutil.rmtree(Path(hf_path), ignore_errors=True)
            raise RuntimeError(
                f"Strict Megatron-to-HF export failed: {error}. Partial output at {hf_path} was deleted. "
                "Re-run with --not-strict only when unmatched keys are expected."
            ) from error
        raise
    logger.info("CPU export complete: %s", hf_path)
