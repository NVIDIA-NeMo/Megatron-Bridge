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

"""OpenTelemetry spans for Megatron Bridge through nemo-lens.

This module holds the call-site helpers (:func:`span`, :func:`traced`, :func:`enabled`,
:func:`set_span_attributes`) and the span groups. Library code imports only this module and never
installs OpenTelemetry providers: NeMo-RL calls Megatron Bridge functions inside worker processes whose
telemetry it already owns. :mod:`megatron.bridge.telemetry.lifecycle` installs providers for processes
that Megatron Bridge owns, and only :func:`~megatron.bridge.training.pretrain.pretrain` calls it.

nemo-lens is resolved once, at import. When it is not installed, or is too old to have ``SpanRegistry``
(v0.2.0 and older), or a site's span group is not enabled, the site creates no span and costs one
function call and one set lookup.

Megatron Bridge registers its span groups under the ``megatron_bridge`` namespace when this module is
imported. The names are shared on purpose with Megatron-LM and NeMo-RL, so one span-group spec selects
the same phase in every library of a job. The registration also covers the groups that megatron-core's
own spans use (``microbatch``, ``layer``, ``communication``, ``activation_offload``). megatron-core
declares them through ``SpanGroup``, which nemo-lens replaced with ``SpanRegistry`` after v0.2.0, so
without this registration they could not be enabled.
"""

import logging
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, TypeVar, cast


if TYPE_CHECKING:
    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

try:
    import nemo.lens as _lens
    from nemo.lens import SpanRegistry as _SpanRegistry
    from nemo.lens import semconv
    from opentelemetry import trace as _otel_trace
except ImportError:
    # nemo-lens is not installed, or it predates SpanRegistry (v0.2.0 and older): every site stays inert.
    HAVE_NEMO_LENS = False
else:
    HAVE_NEMO_LENS = True

#: Namespace that Megatron Bridge registers its span groups under.
NAMESPACE = "megatron_bridge"

JOB = "job"
"""Job-level phases such as startup."""
MODEL_INIT = "model_init"
"""Model construction, wrapping, and weight initialization. Once per model."""
LOAD_CHECKPOINT = "load_checkpoint"
"""Restoring a checkpoint. Once per load."""
CHECKPOINT = "checkpoint"
"""Saving a checkpoint, and finalizing an asynchronous save. Once per checkpoint."""
EVALUATE = "evaluate"
"""Validation and test passes."""
FIRST_ITERATION = "first_iteration"
"""The first training iteration of the process, which pays one-off costs such as compilation."""
STEP = "step"
"""One training iteration."""
FORWARD_BACKWARD = "forward_backward"
"""The forward and backward passes of one iteration, over all its microbatches."""
OPTIMIZER = "optimizer"
"""The optimizer step of one iteration."""
CONVERSION = "conversion"
"""Hugging Face to Megatron conversion and export."""
DATA_LOADING = "data_loading"
"""Building the datasets and data iterators."""
# Groups whose spans live in megatron-core; Megatron Bridge only registers them.
MICROBATCH = "microbatch"
LAYER = "layer"
COMMUNICATION = "communication"
ACTIVATION_OFFLOAD = "activation_offload"

#: Span attribute for the training iteration a span belongs to, numbered as the training log numbers it.
#: Equal to ``nemo.lens.semconv.DL_ITERATION``; defined here so that call sites need no nemo-lens.
ITERATION = "dl.iteration"

_DEFAULT_GROUPS = frozenset({JOB, CHECKPOINT, EVALUATE, FIRST_ITERATION, CONVERSION})
# Follows megatron-core's MegatronSpanGroup presets for the groups Megatron Bridge records, and adds conversion
# to `default`.
_SPAN_GROUP_PRESETS: dict[str, frozenset[str]] = {
    # Spans that occur a bounded number of times per job.
    "default": _DEFAULT_GROUPS,
    # Adds spans that occur on every iteration, and the one-off model and load phases.
    "per_step": _DEFAULT_GROUPS
    | {MODEL_INIT, LOAD_CHECKPOINT, STEP, FORWARD_BACKWARD, OPTIMIZER, COMMUNICATION, DATA_LOADING},
}
_SPAN_GROUPS = _SPAN_GROUP_PRESETS["per_step"] | {MICROBATCH, LAYER, ACTIVATION_OFFLOAD}

if HAVE_NEMO_LENS and NAMESPACE not in _SpanRegistry.namespaces():
    # Without allow_override, nemo-lens warns when another library has already claimed one of these group
    # names, so an overlap stays visible. The namespace check keeps a reloaded module from raising.
    _SpanRegistry.register(NAMESPACE, _SPAN_GROUPS, presets=_SPAN_GROUP_PRESETS)

_NO_SPAN: AbstractContextManager[None] = nullcontext()
_F = TypeVar("_F", bound=Callable[..., Any])


def enabled(group: str) -> bool:
    """Return whether spans of ``group`` are recorded in this process.

    Guard work done only for telemetry with it, such as building the attributes for
    :func:`set_span_attributes`. That includes reading ``semconv``, which is undefined when nemo-lens
    is not installed.

    Args:
        group: Span group name, for example :data:`CHECKPOINT`.

    Returns:
        ``True`` when nemo-lens is installed and ``group`` is enabled.
    """
    return HAVE_NEMO_LENS and _lens.is_span_group_enabled(group)


def span(group: str, name: str) -> AbstractContextManager["Span | None"]:
    """Return a context manager that records its block as a span when ``group`` is enabled.

    The context manager yields the active span, or ``None`` when nemo-lens is not installed or ``group``
    is not enabled. The block runs either way. Compute attribute values only inside
    ``if span is not None:``. An exception raised in the block is recorded on the span, which ends with
    ERROR status, and propagates unchanged.

    Args:
        group: Span group that gates the site, for example :data:`STEP`.
        name: Span name, for example ``"megatron.train.iteration"``.

    Returns:
        A context manager that yields the span, or ``None``.
    """
    if HAVE_NEMO_LENS and _lens.is_span_group_enabled(group):
        # nemo-lens is untyped.
        return cast("AbstractContextManager[Span | None]", _lens.managed_span(group, name))
    return _NO_SPAN


def _undecorated(func: _F) -> _F:
    return func


def traced(group: str, name: str) -> Callable[[_F], _F]:
    """Decorate a function so that each call is recorded as a span when ``group`` is enabled.

    The group is checked on every call. Without nemo-lens the function is returned unchanged. Do not
    decorate generator functions: the span would cover only the creation of the generator.

    Args:
        group: Span group that gates the function, for example :data:`CHECKPOINT`.
        name: Span name, for example ``"megatron.checkpoint.save"``.

    Returns:
        The decorator.
    """
    if not HAVE_NEMO_LENS:
        return _undecorated
    # nemo-lens is untyped.
    return cast("Callable[[_F], _F]", _lens.trace_fn(group, name))


def set_span_attributes(group: str, attributes: Mapping[str, str | bool | int | float]) -> None:
    """Set attributes on the span of the enclosing :func:`traced` call when ``group`` is enabled.

    Call it from the decorated function, outside any span the function opens itself, and only under
    ``if enabled(group):`` so that the attributes are not built for a disabled site. When ``group`` is
    not enabled, the current span belongs to an enclosing site, so nothing is set.

    Args:
        group: The span group passed to :func:`traced`.
        attributes: Attribute names and scalar values.
    """
    if HAVE_NEMO_LENS and _lens.is_span_group_enabled(group):
        _lens.safe_set_span_attributes(_otel_trace.get_current_span(), dict(attributes))
