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


from typing import Any, Optional

from megatron.core.transformer import MegatronModule

from megatron.bridge.utils.common_utils import print_rank_0
from megatron.bridge.training.config import TensorInspectConfig

MISSING_NVINSPECT_MSG = (
    "nvdlfw_inspect is not available. Please install it with `pip install nvdlfw-inspect`."
)

try:
    import nvdlfw_inspect.api as nvinspect_api
    from nvdlfw_inspect.logging import (
        BaseLogger,
        MetricLogger,
        wrap_tensorboard_writer,
    )
    HAVE_NVINSPECT = True
except (ImportError, ModuleNotFoundError):
    HAVE_NVINSPECT = False


def initialize_tensor_inspect_pre_model_initialization(tensor_inspect_config: TensorInspectConfig | None) -> None:
    """Initialize NVIDIA-DL-Framework-Inspect before model construction.

    When enabled and the API is unavailable or fails, raise to stop training.
    """

    if tensor_inspect_config is None or not tensor_inspect_config.enabled:
        return

    if not HAVE_NVINSPECT:
        print_rank_0(MISSING_NVINSPECT_MSG)
        raise ImportError(MISSING_NVINSPECT_MSG)

    try:
        nvinspect_api.initialize(
            config_file=tensor_inspect_config.features or "",
            feature_dirs=tensor_inspect_config.feature_dirs,
            log_dir=tensor_inspect_config.log_dir,
            statistics_logger=None,
            init_training_step=tensor_inspect_config.init_training_step,
            default_logging_enabled=True,
        )
        print_rank_0("Initialized NVIDIA-DL-Framework-Inspect (pre-model).")
    except Exception as e:
        # Treat initialization failures as fatal when enabled so training exits
        print_rank_0(f"NVIDIA DLFw Inspect pre-init failed: {e}")
        raise


def _maybe_attach_metric_loggers(tensorboard_logger: Optional[Any], wandb_logger: Optional[Any]) -> None:
    """Attach supported metric loggers (TensorBoard, W&B raw module)."""

    try:
        # TensorBoard
        if tensorboard_logger is not None:
            tb_logger = wrap_tensorboard_writer(tensorboard_logger)
            MetricLogger.add_logger(tb_logger)

        # Raw wandb module (with .log)
        if wandb_logger is not None and hasattr(wandb_logger, "log"):
            if BaseLogger is None:
                return

            class _WandbModuleLogger(BaseLogger):  # type: ignore
                def __init__(self, wandb_module):
                    super().__init__()
                    self._wandb = wandb_module

                def log_scalar(self, name: str, value: float | int, iteration: int, **kwargs):  # type: ignore[override]
                    self._wandb.log({name: value}, step=iteration)

            MetricLogger.add_logger(_WandbModuleLogger(wandb_logger))
    except Exception as e:  
        print_rank_0(f"Skipping NVIDIA DLFw Inspect metric logger attach due to error: {e}")


def finalize_tensor_inspect_post_model_initialization(
    tensor_inspect_config: TensorInspectConfig | None,
    model: list[MegatronModule],
    tensorboard_logger: Optional[Any],
    wandb_logger: Optional[Any],
    current_training_step: Optional[int] = None,
) -> None:
    """Finalize setup after model creation: attach loggers, set names and groups."""

    if tensor_inspect_config is None or not tensor_inspect_config.enabled:
        return

    if not HAVE_NVINSPECT:
        print_rank_0(MISSING_NVINSPECT_MSG)
        raise ImportError(MISSING_NVINSPECT_MSG)

    try:
        from megatron.core.parallel_state import get_tensor_and_data_parallel_group

        _maybe_attach_metric_loggers(tensorboard_logger, wandb_logger)

        # Align nvinspect_api's step count with the resumed training step after checkpoint load
        if current_training_step is not None:
            nvinspect_api.initialize_training_step(int(current_training_step))

        nvinspect_api.infer_and_assign_layer_names(model)
        nvinspect_api.set_tensor_reduction_group(get_tensor_and_data_parallel_group())
        print_rank_0("Finalized NVIDIA DLFw Inspect (post-model).")
    except Exception as e:
        print_rank_0(f"NVIDIA DLFw Inspect post-init failed: {e}")
        raise


def tensor_inspect_step_if_enabled(tensor_inspect_config: TensorInspectConfig | None) -> None:
    """Advance DLFw Inspect step if enabled."""

    if tensor_inspect_config is None or not tensor_inspect_config.enabled:
        return
    if not HAVE_NVINSPECT:
        print_rank_0(MISSING_NVINSPECT_MSG)
        raise ImportError(MISSING_NVINSPECT_MSG)
    try:
        nvinspect_api.step()
    except Exception as e:
        print_rank_0(f"NVIDIA DLFw Inspect step failed: {e}")
        raise


def tensor_inspect_end_if_enabled(tensor_inspect_config: TensorInspectConfig | None) -> None:
    """Shutdown DLFw Inspect if enabled."""

    if tensor_inspect_config is None or not tensor_inspect_config.enabled:
        return
    if not HAVE_NVINSPECT:
        return
    try:
        nvinspect_api.end_debug()
    except Exception as e:
        print_rank_0(f"NVIDIA DLFw Inspect end failed: {e}")


