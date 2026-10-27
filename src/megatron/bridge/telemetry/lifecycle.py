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

"""Install and flush the OpenTelemetry providers of a process that Megatron Bridge owns.

Only :func:`~megatron.bridge.training.pretrain.pretrain` calls :func:`init_telemetry`, so the
providers belong to the training job. Functions that other applications call directly, such as the
checkpoint functions or ``AutoBridge``, never install providers: in NeMo-RL worker processes, for
example, the application owns telemetry, and those functions record under its providers.
"""

import atexit
import importlib.util
import logging
import os
import signal
import threading
from collections.abc import Callable
from types import FrameType
from typing import TYPE_CHECKING, Any

import torch

from megatron.bridge import telemetry
from megatron.bridge.package_info import __version__ as _bridge_version
from megatron.bridge.training.utils.checkpoint_utils import get_hf_source_from_model_config
from megatron.bridge.utils.common_utils import get_local_rank_preinit, get_rank_safe, get_world_size_safe


if TYPE_CHECKING:
    from nemo.lens import TelemetryHandle

    from megatron.bridge.training.config import ConfigContainer

logger = logging.getLogger(__name__)

#: Environment variable prefixes, shared with Megatron-LM: ``MEGATRON_OTEL_<KEY>``, then ``NEMO_LENS_<KEY>``.
ENV_PREFIX = "MEGATRON_OTEL"
ENV_FALLBACK_PREFIX = "NEMO_LENS"
#: ``service.name`` when neither ``OTEL_SERVICE_NAME`` nor ``logger.otel_service_name`` is set.
DEFAULT_SERVICE_NAME = "megatron-bridge"

# Resource attributes that nemo-lens semconv has no constant for.
CONTEXT_PARALLEL_SIZE = "dl.context_parallel.size"
EXPERT_PARALLEL_SIZE = "dl.expert_parallel.size"
MODEL_CONFIG = "megatron_bridge.model.config"
MODEL_HF_ID = "megatron_bridge.model.hf_model_id"

_INSTALL_HINT = "Install Megatron Bridge with its `otel` extra: `uv sync --extra otel`."
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
# Bounds each flush, so that an unreachable collector delays a terminating process by at most this long.
_FLUSH_TIMEOUT_MILLIS = 5000

_HANDLE: "TelemetryHandle | None" = None
_HANDLE_LOCK = threading.Lock()
_RECORD_TRAINING_METRICS: Callable[..., None] | None = None


def init_telemetry(config: "ConfigContainer") -> "TelemetryHandle | None":
    """Install the OpenTelemetry providers of this process when telemetry is enabled.

    Settings come from the ``MEGATRON_OTEL_<KEY>`` environment variables, falling back to
    ``NEMO_LENS_<KEY>``, as in Megatron-LM. The ``logger.otel_*`` config fields override them when they
    are set; ``logger.otel_enabled=False`` leaves the decision to the environment. The OTLP endpoint
    and headers come from the standard ``OTEL_EXPORTER_OTLP_*`` variables.

    Call it on every rank, after the config is final and before setup, so that setup is traced. A
    later call returns the first handle. When telemetry is disabled nothing is installed:
    OpenTelemetry providers can be set only once per process, so they stay free for another library.
    When another library already installed a tracer provider, spans go to it, and this returns
    ``None``.

    Args:
        config: The run configuration.

    Returns:
        The handle, or ``None`` when Megatron Bridge does not own telemetry in this process.

    Raises:
        ModuleNotFoundError: ``logger.otel_enabled`` is set, but nemo-lens with ``SpanRegistry`` or the
            OpenTelemetry SDK is not installed. When only the environment enables telemetry, a warning
            is logged instead.
    """
    global _HANDLE, _RECORD_TRAINING_METRICS

    with _HANDLE_LOCK:
        if _HANDLE is not None:
            return _HANDLE
        logger_config = config.logger
        # getattr: the otel_* fields come from megatron-core's LoggerConfig, which megatron-core dev lacks.
        config_enabled = getattr(logger_config, "otel_enabled", False) is True
        service_name = getattr(logger_config, "otel_service_name", None)
        span_groups = getattr(logger_config, "otel_span_groups", None)
        if not telemetry.HAVE_NEMO_LENS:
            if config_enabled or _env_enabled():
                _unavailable(
                    "nemo-lens is not installed or is too old to have SpanRegistry", raise_error=config_enabled
                )
            return None

        lens_config = telemetry._lens.NemoLensConfig.from_env(prefix=ENV_PREFIX, fallback_prefix=ENV_FALLBACK_PREFIX)
        if not os.environ.get("OTEL_SERVICE_NAME", "").strip():
            lens_config.service_name = DEFAULT_SERVICE_NAME
        if config_enabled:
            lens_config.enabled = True
        if isinstance(service_name, str) and service_name:
            lens_config.service_name = service_name
        if isinstance(span_groups, str) and span_groups:
            lens_config.span_groups = span_groups
        if not lens_config.enabled:
            return None

        if importlib.util.find_spec("opentelemetry.sdk") is None:
            _unavailable("the OpenTelemetry SDK is not installed", raise_error=config_enabled)
            return None
        if not isinstance(telemetry._otel_trace.get_tracer_provider(), telemetry._otel_trace.ProxyTracerProvider):
            logger.warning(
                "An OpenTelemetry tracer provider is already installed in this process, so Megatron Bridge "
                "does not install its own. Its spans go to the installed provider."
            )
            return None

        world_size = get_world_size_safe()
        if not lens_config.run_id:
            lens_config.run_id = _job_run_id(world_size)
        handle = telemetry._lens.setup_telemetry(
            lens_config,
            resource_attributes=_resource_attributes(config, rank=get_rank_safe(), world_size=world_size),
        )
        if lens_config.logs_enabled:
            from nemo.lens.logging_bridge import setup_logging_bridge

            setup_logging_bridge()
        atexit.register(handle.shutdown)
        _flush_on_sigterm()
        try:
            from megatron.core.telemetry.training_metrics import record_training_metrics
        except ImportError:
            # megatron-core without telemetry, such as its dev branch: spans only.
            logger.info("megatron-core has no telemetry module, so training metrics are not exported.")
        else:
            _RECORD_TRAINING_METRICS = record_training_metrics
        _HANDLE = handle
        logger.info(
            "Telemetry enabled: exporter=%s span_groups=%s run_id=%s",
            lens_config.exporter,
            lens_config.span_groups,
            lens_config.run_id,
        )
        return handle


def flush_telemetry() -> None:
    """Export the spans and metrics recorded so far, without shutting the providers down.

    The providers shut down at process exit, so a later ``pretrain`` call in the same process still
    records. Does nothing unless :func:`init_telemetry` installed providers.
    """
    if _HANDLE is None:
        return
    from opentelemetry import metrics

    for provider in (telemetry._otel_trace.get_tracer_provider(), metrics.get_meter_provider()):
        force_flush = getattr(provider, "force_flush", None)
        if force_flush is None:
            continue
        try:
            force_flush(timeout_millis=_FLUSH_TIMEOUT_MILLIS)
        except Exception:
            # Telemetry must not mask the exception or exit code of the training job.
            logger.warning("Failed to flush telemetry", exc_info=True)


def record_training_metrics(*, step_duration_ms: float, throughput_tflops: float | None, skipped_iters: int) -> None:
    """Record megatron-core's ``megatron.training.*`` metrics, the names Megatron-LM exports too.

    Also records the peak GPU memory this rank has allocated. Records nothing unless
    :func:`init_telemetry` installed providers in this process and megatron-core has its telemetry
    module.

    Args:
        step_duration_ms: Average duration of an iteration over the logging interval.
        throughput_tflops: Model TFLOP/s per GPU over the logging interval, or ``None`` when it is not
            computed.
        skipped_iters: Iterations skipped in the logging interval.
    """
    if _HANDLE is None or _RECORD_TRAINING_METRICS is None:
        return
    _RECORD_TRAINING_METRICS(
        _HANDLE.meter,
        step_duration_ms=step_duration_ms,
        throughput_tflops=throughput_tflops,
        memory_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None,
        skipped_iters=skipped_iters,
    )


def _unavailable(reason: str, *, raise_error: bool) -> None:
    message = f"Telemetry is enabled, but {reason}. {_INSTALL_HINT}"
    if raise_error:
        raise ModuleNotFoundError(message)
    logger.warning("%s Telemetry is off.", message)


def _env_enabled() -> bool:
    """Return whether the environment enables telemetry, with the precedence of ``NemoLensConfig.from_env``."""
    for prefix in (ENV_PREFIX, ENV_FALLBACK_PREFIX):
        value = os.environ.get(f"{prefix}_ENABLED", "").strip().lower()
        if value:
            return value in _TRUE_VALUES
    return False


def _flush_on_sigterm() -> None:
    """Flush telemetry on SIGTERM, then run the handler that was installed before.

    The default action of SIGTERM ends the process without running ``atexit``, so a job that its
    scheduler terminates would lose what it recorded since the last export. When Megatron Bridge's
    ``train.exit_signal_handler`` is on, its handler runs after the flush and training stops
    gracefully.
    """
    previous = signal.getsignal(signal.SIGTERM)
    if previous is None:
        # A handler that was not installed from Python cannot be chained.
        return

    def handler(signum: int, frame: FrameType | None) -> None:
        flush_telemetry()
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_DFL:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)

    try:
        signal.signal(signal.SIGTERM, handler)
    except ValueError:
        # Not the main thread: SIGTERM keeps its handler, and atexit still flushes on a normal exit.
        logger.debug("Telemetry is not flushed on SIGTERM: init_telemetry ran outside the main thread.")


def _job_run_id(world_size: int) -> str:
    """Return an id that every rank derives alike, or an empty string to let nemo-lens choose.

    nemo-lens uses the Slurm job id, and otherwise a random id per process. Under torchrun the elastic
    run id is shared by every rank; it is ``"none"`` unless the job uses ``--standalone`` or
    ``--rdzv-id``. The process group does not exist yet, so the id cannot be broadcast.
    """
    if os.environ.get("SLURM_JOB_ID"):
        return ""
    elastic_run_id = os.environ.get("TORCHELASTIC_RUN_ID", "")
    if elastic_run_id and elastic_run_id != "none":
        return elastic_run_id
    if world_size > 1:
        logger.warning(
            "Each rank reports a different nemo.run.id. For a job-wide id, set %s_RUN_ID, or launch with "
            "Slurm or `torchrun --standalone`.",
            ENV_PREFIX,
        )
    return ""


def _resource_attributes(config: "ConfigContainer", *, rank: int, world_size: int) -> dict[str, Any]:
    """Return the attributes fixed for the life of the process: identity, parallel layout, versions."""
    semconv = telemetry.semconv
    model = config.model
    attributes: dict[str, Any] = {
        "service.version": _bridge_version,
        semconv.NV_DL_RANK: rank,
        semconv.NV_DL_WORLD_SIZE: world_size,
        semconv.NV_DL_LOCAL_RANK: get_local_rank_preinit(),
        MODEL_CONFIG: type(model).__name__,
    }
    # Builder-backed model configs share a few classes, such as BridgeGPTModelConfig, so the Hugging Face
    # source is what names the model. A Hub ID names it; a local path would only expose the directory layout.
    hf_model_id, _ = get_hf_source_from_model_config(model)
    if isinstance(hf_model_id, str) and not os.path.isabs(hf_model_id) and not os.path.exists(hf_model_id):
        attributes[MODEL_HF_ID] = hf_model_id
    # getattr: not every model provider, such as the MIMO ones, carries every parallel size.
    sizes = {
        semconv.DL_TENSOR_PARALLEL_SIZE: getattr(model, "tensor_model_parallel_size", None),
        semconv.DL_PIPELINE_PARALLEL_SIZE: getattr(model, "pipeline_model_parallel_size", None),
        semconv.DL_DATA_PARALLEL_SIZE: getattr(config, "data_parallel_size", None),
        CONTEXT_PARALLEL_SIZE: getattr(model, "context_parallel_size", None),
        EXPERT_PARALLEL_SIZE: getattr(model, "expert_model_parallel_size", None),
    }
    attributes.update({key: value for key, value in sizes.items() if isinstance(value, int)})
    return attributes
