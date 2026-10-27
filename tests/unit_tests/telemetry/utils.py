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

"""Helpers for tests that observe the spans and metrics Megatron Bridge records through nemo-lens.

OpenTelemetry providers, the enabled nemo-lens span groups, the SIGTERM handler, and the handle of
:mod:`megatron.bridge.telemetry.lifecycle` are process-global. The fixtures here return all of them to
the state of a fresh process before and after each test, so telemetry never leaks into other tests.

Test modules elsewhere import the fixtures by name: ``from tests.unit_tests.telemetry.utils import
telemetry_capture  # noqa: F401``.
"""

import atexit
import functools
import signal
from dataclasses import dataclass
from typing import Any

import pytest

from megatron.bridge import telemetry


def reset_telemetry_state() -> None:
    """Return OpenTelemetry, nemo-lens, and Megatron Bridge telemetry state to that of a fresh process."""
    import nemo.lens.handle as lens_handle
    import opentelemetry._logs._internal as otel_logs
    import opentelemetry.metrics._internal as otel_metrics
    import opentelemetry.trace as otel_trace
    from nemo.lens import set_enabled_span_groups
    from opentelemetry.util._once import Once

    from megatron.bridge.telemetry import lifecycle

    if lifecycle._HANDLE is not None:
        atexit.unregister(lifecycle._HANDLE.shutdown)
    for provider in (
        otel_trace.get_tracer_provider(),
        otel_metrics.get_meter_provider(),
        otel_logs.get_logger_provider(),
    ):
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    otel_metrics._METER_PROVIDER = None
    otel_metrics._METER_PROVIDER_SET_ONCE = Once()
    otel_logs._LOGGER_PROVIDER = None
    otel_logs._LOGGER_PROVIDER_SET_ONCE = Once()
    lens_handle._INITIALIZED = False
    set_enabled_span_groups(frozenset())
    lifecycle._HANDLE = None
    lifecycle._RECORD_TRAINING_METRICS = None


@dataclass
class TelemetryCapture:
    """In-memory readers for the spans and metrics of one test."""

    span_exporter: Any
    metric_reader: Any
    metric_exporter: Any = None

    def spans(self) -> list:
        """Return every span finished so far."""
        import opentelemetry.trace as otel_trace

        otel_trace.get_tracer_provider().force_flush()
        return list(self.span_exporter.get_finished_spans())

    def span_named(self, name: str) -> Any:
        """Return the only finished span called ``name``."""
        spans = [span for span in self.spans() if span.name == name]
        assert len(spans) == 1, f"expected one {name!r} span, got {len(spans)}: {[s.name for s in self.spans()]}"
        return spans[0]

    def span_names(self) -> list[str]:
        """Return the names of the finished spans, in the order they ended."""
        return [span.name for span in self.spans()]

    def metrics(self) -> dict[str, Any]:
        """Return the collected metrics, keyed by name."""
        data = self.metric_reader.get_metrics_data()
        if data is None:
            return {}
        return {
            metric.name: metric
            for resource_metrics in data.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
        }


def recording_metric_exporter() -> Any:
    """Return a metric exporter that keeps every exported batch, as a collector would receive them.

    An ``InMemoryMetricReader`` reports only the latest collection, and collecting resets the value of a
    synchronous gauge, so a flush hides gauges from a later read. This exporter keeps what each flush
    exported. Pair it with a ``PeriodicExportingMetricReader`` whose interval outlasts the test, so that
    only flushes export.
    """
    from opentelemetry.sdk.metrics.export import MetricExporter, MetricExportResult

    class RecordingMetricExporter(MetricExporter):
        def __init__(self) -> None:
            super().__init__()
            self.batches: list = []

        def export(self, metrics_data, timeout_millis: float = 10_000, **kwargs) -> MetricExportResult:
            self.batches.append(metrics_data)
            return MetricExportResult.SUCCESS

        def force_flush(self, timeout_millis: float = 10_000) -> bool:
            return True

        def shutdown(self, timeout_millis: float = 30_000, **kwargs) -> None:
            pass

        def points(self, name: str) -> list:
            """Return the data points of metric ``name`` across every exported batch, in order."""
            return [
                point
                for batch in self.batches
                for resource_metrics in batch.resource_metrics
                for scope_metrics in resource_metrics.scope_metrics
                for metric in scope_metrics.metrics
                if metric.name == name
                for point in metric.data.data_points
            ]

    return RecordingMetricExporter()


def _require_nemo_lens() -> None:
    if not telemetry.HAVE_NEMO_LENS:
        pytest.skip("nemo-lens with SpanRegistry is not installed")
    pytest.importorskip("opentelemetry.sdk")


@pytest.fixture
def clean_telemetry(monkeypatch):
    """Reset telemetry state before and after the test, and restore the SIGTERM handler."""
    _require_nemo_lens()
    for name in ("OTEL_RESOURCE_ATTRIBUTES", "OTEL_SERVICE_NAME", "SLURM_JOB_ID", "TORCHELASTIC_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")):
        monkeypatch.setenv(name, value)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    reset_telemetry_state()
    yield
    reset_telemetry_state()
    signal.signal(signal.SIGTERM, previous_sigterm)


@pytest.fixture
def telemetry_capture(clean_telemetry):
    """Return a factory that installs telemetry with in-memory readers, for testing call sites.

    Call it with a span-group spec, for example ``telemetry_capture("per_step")``. The providers are
    installed directly, as an application that owns telemetry would, not through
    :func:`~megatron.bridge.telemetry.lifecycle.init_telemetry`.
    """
    from nemo.lens import NemoLensConfig, setup_telemetry
    from nemo.lens.semconv import NV_DL_RANK, NV_DL_WORLD_SIZE
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    def install(span_groups: str) -> TelemetryCapture:
        capture = TelemetryCapture(span_exporter=InMemorySpanExporter(), metric_reader=InMemoryMetricReader())
        setup_telemetry(
            NemoLensConfig(enabled=True, exporter="console", span_groups=span_groups, run_id="test-run"),
            resource_attributes={NV_DL_RANK: 0, NV_DL_WORLD_SIZE: 1},
            span_exporter=capture.span_exporter,
            metric_reader=capture.metric_reader,
        )
        return capture

    return install


@pytest.fixture
def lifecycle_capture(clean_telemetry, monkeypatch):
    """Route the providers that ``init_telemetry`` installs to in-memory readers; return the readers."""
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    capture = TelemetryCapture(span_exporter=InMemorySpanExporter(), metric_reader=InMemoryMetricReader())
    monkeypatch.setattr(
        telemetry._lens,
        "setup_telemetry",
        functools.partial(
            telemetry._lens.setup_telemetry,
            span_exporter=capture.span_exporter,
            metric_reader=capture.metric_reader,
        ),
    )
    return capture
