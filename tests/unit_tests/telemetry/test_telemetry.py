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

"""Tests for the call-site helpers and span groups of megatron.bridge.telemetry."""

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest

from megatron.bridge import telemetry
from tests.unit_tests.telemetry.utils import clean_telemetry, telemetry_capture  # noqa: F401


pytestmark = pytest.mark.unit


def _load_telemetry_copy(monkeypatch, nemo_lens_module):
    """Execute a fresh copy of the telemetry module while ``nemo.lens`` resolves to ``nemo_lens_module``."""
    monkeypatch.setitem(sys.modules, "nemo.lens", nemo_lens_module)
    spec = importlib.util.spec_from_file_location("_megatron_bridge_telemetry_copy", Path(telemetry.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestWithoutNemoLens:
    """Every helper is inert when nemo-lens is missing or has no SpanRegistry."""

    @pytest.mark.parametrize(
        "nemo_lens_module",
        [None, types.ModuleType("nemo.lens")],
        ids=["not-installed", "without-SpanRegistry"],
    )
    def test_helpers_are_inert(self, monkeypatch, nemo_lens_module):
        module = _load_telemetry_copy(monkeypatch, nemo_lens_module)

        assert module.HAVE_NEMO_LENS is False
        assert module.enabled(module.STEP) is False
        with module.span(module.STEP, "megatron.train.iteration") as span:
            assert span is None
        module.set_span_attributes(module.CHECKPOINT, {module.ITERATION: 1})

        def function(value):
            return value + 1

        assert module.traced(module.CHECKPOINT, "megatron.checkpoint.save")(function) is function


class TestSpanGroups:
    """Megatron Bridge registers its span groups, and megatron-core's, when the module is imported."""

    @pytest.fixture(autouse=True)
    def _require_nemo_lens(self):
        if not telemetry.HAVE_NEMO_LENS:
            pytest.skip("nemo-lens with SpanRegistry is not installed")

    def test_namespace_is_registered_on_import(self):
        from nemo.lens import SpanRegistry

        assert telemetry.NAMESPACE in SpanRegistry.namespaces()
        assert telemetry._SPAN_GROUPS <= SpanRegistry.groups()

    def test_default_preset_holds_only_bounded_groups(self):
        from nemo.lens import SpanRegistry

        enabled, pending = SpanRegistry.resolve("default")

        assert pending == frozenset()
        assert {"job", "checkpoint", "evaluate", "first_iteration", "conversion"} <= enabled
        assert enabled.isdisjoint({"step", "forward_backward", "optimizer", "microbatch", "layer"})

    def test_per_step_preset_adds_per_iteration_groups(self):
        from nemo.lens import SpanRegistry

        enabled, _ = SpanRegistry.resolve("per_step")

        assert {
            "model_init",
            "load_checkpoint",
            "step",
            "forward_backward",
            "optimizer",
            "communication",
            "data_loading",
        } <= enabled
        assert enabled.isdisjoint({"microbatch", "layer", "activation_offload"})

    def test_megatron_core_groups_can_be_enabled(self):
        from nemo.lens import SpanRegistry

        enabled, pending = SpanRegistry.resolve("microbatch,layer,communication,activation_offload")

        assert enabled == {"microbatch", "layer", "communication", "activation_offload"}
        assert pending == frozenset()
        assert enabled <= SpanRegistry.resolve("all")[0]

    def test_iteration_key_is_the_semconv_name(self):
        from nemo.lens import semconv

        assert telemetry.ITERATION == semconv.DL_ITERATION


class TestHelpers:
    """The helpers record spans only for enabled groups."""

    def test_disabled_group_records_nothing(self, telemetry_capture):
        capture = telemetry_capture("default")

        assert telemetry.enabled(telemetry.STEP) is False
        with telemetry.span(telemetry.STEP, "megatron.train.iteration") as span:
            assert span is None

        assert capture.spans() == []

    def test_enabled_group_records_span(self, telemetry_capture):
        capture = telemetry_capture("per_step")

        assert telemetry.enabled(telemetry.STEP) is True
        with telemetry.span(telemetry.STEP, "megatron.train.iteration") as span:
            span.set_attribute(telemetry.ITERATION, 3)

        recorded = capture.span_named("megatron.train.iteration")
        assert recorded.attributes[telemetry.ITERATION] == 3

    def test_span_records_exception_and_reraises(self, telemetry_capture):
        from opentelemetry.trace import StatusCode

        capture = telemetry_capture("default")

        with pytest.raises(ValueError, match="boom"):
            with telemetry.span(telemetry.JOB, "megatron.startup"):
                raise ValueError("boom")

        recorded = capture.span_named("megatron.startup")
        assert recorded.status.status_code is StatusCode.ERROR
        assert [event.name for event in recorded.events] == ["exception"]

    def test_traced_records_each_call_with_attributes(self, telemetry_capture):
        capture = telemetry_capture("default")

        @telemetry.traced(telemetry.CHECKPOINT, "megatron.checkpoint.save")
        def save(step, *, is_async=False):
            telemetry.set_span_attributes(telemetry.CHECKPOINT, {telemetry.ITERATION: step})
            return step * 2

        assert save(5) == 10
        assert list(inspect.signature(save).parameters) == ["step", "is_async"]
        recorded = capture.span_named("megatron.checkpoint.save")
        assert recorded.attributes[telemetry.ITERATION] == 5

    def test_traced_checks_the_group_on_every_call(self, telemetry_capture):
        @telemetry.traced(telemetry.STEP, "megatron.train.iteration")
        def step():
            return "ran"

        capture = telemetry_capture("default")
        assert step() == "ran"
        assert capture.spans() == []

    def test_disabled_site_leaves_enclosing_span_untouched(self, telemetry_capture):
        """``set_span_attributes`` for a disabled group must not write to an enclosing site's span."""
        capture = telemetry_capture("default")

        @telemetry.traced(telemetry.STEP, "megatron.train.iteration")
        def disabled_site():
            telemetry.set_span_attributes(telemetry.STEP, {telemetry.ITERATION: 99})

        with telemetry.span(telemetry.JOB, "megatron.startup"):
            disabled_site()

        assert telemetry.ITERATION not in capture.span_named("megatron.startup").attributes
