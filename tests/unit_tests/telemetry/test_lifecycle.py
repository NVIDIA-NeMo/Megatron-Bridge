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

"""Tests for megatron.bridge.telemetry.lifecycle: when providers are installed, and with which settings."""

import atexit
import importlib.util
import logging
import os
import signal
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.bridge import telemetry
from megatron.bridge.package_info import __version__ as bridge_version
from megatron.bridge.telemetry import lifecycle
from megatron.bridge.training.config import LoggerConfig
from tests.unit_tests.telemetry.utils import clean_telemetry, lifecycle_capture  # noqa: F401


pytestmark = pytest.mark.unit

requires_otel_config_fields = pytest.mark.skipif(
    "otel_enabled" not in LoggerConfig.__dataclass_fields__,
    reason="this megatron-core LoggerConfig has no otel_* fields",
)


def _config(**logger_fields):
    """Return a run config with a real LoggerConfig and a real builder-style GPT model config."""
    from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
    from megatron.bridge.models.transformer_config import TransformerConfig

    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    return SimpleNamespace(
        logger=LoggerConfig(**logger_fields),
        model=BridgeGPTModelConfig(transformer=transformer, vocab_size=128, seq_length=64),
        data_parallel_size=4,
    )


def _resource(capture):
    with telemetry.span(telemetry.JOB, "probe"):
        pass
    return dict(capture.span_named("probe").resource.attributes)


@pytest.fixture
def setup_spy(monkeypatch):
    """Count the calls to nemo-lens ``setup_telemetry`` without changing what it does."""
    spy = mock.Mock(wraps=telemetry._lens.setup_telemetry)
    monkeypatch.setattr(telemetry._lens, "setup_telemetry", spy)
    return spy


class TestInitTelemetry:
    """init_telemetry installs providers only when telemetry is enabled and nobody else owns them."""

    def test_disabled_installs_nothing(self, clean_telemetry, setup_spy):
        from opentelemetry import trace

        sigterm_handler = signal.getsignal(signal.SIGTERM)

        assert lifecycle.init_telemetry(_config()) is None

        setup_spy.assert_not_called()
        assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)
        assert telemetry.enabled(telemetry.JOB) is False
        assert signal.getsignal(signal.SIGTERM) is sigterm_handler

    def test_mock_config_does_not_enable(self, clean_telemetry, setup_spy):
        """Unit tests across the suite pass MagicMock configs to pretrain; they must not install telemetry."""
        assert lifecycle.init_telemetry(mock.MagicMock()) is None
        setup_spy.assert_not_called()

    @requires_otel_config_fields
    def test_config_enables_with_job_identity(self, lifecycle_capture, monkeypatch):
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "elastic-run-7")
        register = mock.Mock()
        monkeypatch.setattr(atexit, "register", register)

        handle = lifecycle.init_telemetry(_config(otel_enabled=True))

        assert handle is not None
        assert mock.call(handle.shutdown) in register.call_args_list
        assert telemetry.enabled(telemetry.JOB) is True
        assert telemetry.enabled(telemetry.STEP) is False
        resource = _resource(lifecycle_capture)
        assert resource["service.name"] == "megatron-bridge"
        assert resource["service.version"] == bridge_version
        assert resource["nemo.run.id"] == "elastic-run-7"
        assert resource["nv.dl.rank"] == 1
        assert resource["nv.dl.world_size"] == 8
        assert resource["nv.dl.local_rank"] == 1
        assert resource["dl.tensor_parallel.size"] == 2
        assert resource["dl.pipeline_parallel.size"] == 1
        assert resource["dl.data_parallel.size"] == 4
        assert resource["dl.context_parallel.size"] == 1
        assert resource["dl.expert_parallel.size"] == 1
        assert resource["megatron_bridge.model.config"] == "BridgeGPTModelConfig"
        assert "megatron_bridge.model.hf_model_id" not in resource

    @pytest.mark.parametrize(
        ("source", "exported"),
        [
            pytest.param("meta-llama/Llama-3.2-1B", True, id="hub-id"),
            pytest.param("/checkpoints/llama-3.2-1b", False, id="absolute-path"),
            pytest.param("checkpoints/llama-3.2-1b", False, id="existing-relative-path"),
        ],
    )
    def test_hf_model_id_names_hub_models_only(self, lifecycle_capture, monkeypatch, tmp_path, source, exported):
        """Builder-backed configs share a few classes, so the Hub ID names the model; a local path is never exported."""
        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints" / "llama-3.2-1b").mkdir(parents=True)
        config = _config()
        # Where AutoBridge.get_model_config records the source of a builder-backed config.
        config.model.extra_checkpoint_metadata = {"hf_model_id": source}

        lifecycle.init_telemetry(config)

        resource = _resource(lifecycle_capture)
        assert resource["megatron_bridge.model.config"] == "BridgeGPTModelConfig"
        assert resource.get("megatron_bridge.model.hf_model_id") == (source if exported else None)

    def test_hf_model_id_of_a_model_provider(self, lifecycle_capture, monkeypatch):
        from megatron.bridge.models.gpt_provider import GPTModelProvider

        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")
        config = _config()
        config.model = GPTModelProvider(
            num_layers=2, hidden_size=64, num_attention_heads=4, hf_model_id="Qwen/Qwen3-0.6B"
        )

        lifecycle.init_telemetry(config)

        resource = _resource(lifecycle_capture)
        assert resource["megatron_bridge.model.config"] == "GPTModelProvider"
        assert resource["megatron_bridge.model.hf_model_id"] == "Qwen/Qwen3-0.6B"

    @requires_otel_config_fields
    def test_config_fields_override_environment(self, lifecycle_capture, monkeypatch):
        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")
        monkeypatch.setenv("MEGATRON_OTEL_SPAN_GROUPS", "default")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "from-environment")

        lifecycle.init_telemetry(_config(otel_service_name="from-config", otel_span_groups="per_step"))

        assert telemetry.enabled(telemetry.STEP) is True
        assert _resource(lifecycle_capture)["service.name"] == "from-config"

    def test_otel_service_name_replaces_the_default(self, lifecycle_capture, monkeypatch):
        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "from-environment")

        lifecycle.init_telemetry(_config())

        assert _resource(lifecycle_capture)["service.name"] == "from-environment"

    def test_nemo_lens_prefix_is_the_fallback(self, lifecycle_capture, monkeypatch):
        monkeypatch.setenv("NEMO_LENS_ENABLED", "true")

        assert lifecycle.init_telemetry(_config()) is not None

    def test_megatron_otel_prefix_takes_precedence(self, clean_telemetry, monkeypatch, setup_spy):
        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "0")
        monkeypatch.setenv("NEMO_LENS_ENABLED", "1")

        assert lifecycle.init_telemetry(_config()) is None
        setup_spy.assert_not_called()

    def test_later_calls_return_the_first_handle(self, lifecycle_capture, monkeypatch, setup_spy):
        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")

        first = lifecycle.init_telemetry(_config())
        second = lifecycle.init_telemetry(_config())

        assert first is not None
        assert second is first
        setup_spy.assert_called_once()

    def test_existing_tracer_provider_is_kept(self, clean_telemetry, monkeypatch, setup_spy, caplog):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        host_provider = TracerProvider()
        trace.set_tracer_provider(host_provider)
        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")

        with caplog.at_level(logging.WARNING, logger=lifecycle.logger.name):
            assert lifecycle.init_telemetry(_config()) is None

        setup_spy.assert_not_called()
        assert trace.get_tracer_provider() is host_provider
        assert "already installed" in caplog.text

    def test_logs_enabled_bridges_python_logging(self, lifecycle_capture, monkeypatch):
        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")
        monkeypatch.setenv("MEGATRON_OTEL_LOGS_ENABLED", "1")
        with mock.patch("nemo.lens.logging_bridge.setup_logging_bridge") as setup_logging_bridge:
            lifecycle.init_telemetry(_config())

        setup_logging_bridge.assert_called_once_with()

    def test_logs_are_not_bridged_by_default(self, lifecycle_capture, monkeypatch):
        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")
        with mock.patch("nemo.lens.logging_bridge.setup_logging_bridge") as setup_logging_bridge:
            lifecycle.init_telemetry(_config())

        setup_logging_bridge.assert_not_called()


class TestUnavailable:
    """Explicit configuration fails fast without nemo-lens or the SDK; the environment alone only warns."""

    @pytest.fixture
    def without_nemo_lens(self, clean_telemetry, monkeypatch):
        monkeypatch.setattr(telemetry, "HAVE_NEMO_LENS", False)

    @pytest.fixture
    def without_sdk(self, clean_telemetry, monkeypatch):
        find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, package=None: None if name == "opentelemetry.sdk" else find_spec(name, package),
        )

    @requires_otel_config_fields
    @pytest.mark.parametrize("missing", ["without_nemo_lens", "without_sdk"])
    def test_config_raises(self, request, missing):
        request.getfixturevalue(missing)

        with pytest.raises(ModuleNotFoundError, match="uv sync --extra otel"):
            lifecycle.init_telemetry(_config(otel_enabled=True))

    @pytest.mark.parametrize("missing", ["without_nemo_lens", "without_sdk"])
    def test_environment_warns(self, request, missing, monkeypatch, caplog):
        request.getfixturevalue(missing)
        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")

        with caplog.at_level(logging.WARNING, logger=lifecycle.logger.name):
            assert lifecycle.init_telemetry(_config()) is None

        assert "uv sync --extra otel" in caplog.text
        assert lifecycle._HANDLE is None

    def test_disabled_is_silent(self, without_nemo_lens, caplog):
        with caplog.at_level(logging.WARNING, logger=lifecycle.logger.name):
            assert lifecycle.init_telemetry(_config()) is None

        assert caplog.text == ""


class TestFlush:
    """flush_telemetry exports buffered data and keeps the providers usable."""

    @pytest.fixture(autouse=True)
    def _slow_batch_export(self, monkeypatch):
        # Keep the batch span processor from exporting on its own, so only a flush can export.
        monkeypatch.setenv("OTEL_BSP_SCHEDULE_DELAY", "600000")
        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")

    def test_flush_exports_without_shutting_down(self, lifecycle_capture):
        lifecycle.init_telemetry(_config())
        with telemetry.span(telemetry.JOB, "first"):
            pass
        assert lifecycle_capture.span_exporter.get_finished_spans() == ()

        lifecycle.flush_telemetry()

        assert [span.name for span in lifecycle_capture.span_exporter.get_finished_spans()] == ["first"]
        with telemetry.span(telemetry.JOB, "second"):
            pass
        assert lifecycle_capture.span_names() == ["first", "second"]

    def test_flush_without_telemetry_does_nothing(self, clean_telemetry):
        lifecycle.flush_telemetry()

    def test_sigterm_flushes_then_runs_previous_handler(self, lifecycle_capture):
        previous_handler = mock.Mock()
        signal.signal(signal.SIGTERM, previous_handler)
        lifecycle.init_telemetry(_config())
        with telemetry.span(telemetry.JOB, "before.sigterm"):
            pass

        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        assert [span.name for span in lifecycle_capture.span_exporter.get_finished_spans()] == ["before.sigterm"]
        previous_handler.assert_called_once_with(signal.SIGTERM, None)

    @requires_otel_config_fields
    def test_sigterm_flushes_before_default_termination(self, clean_telemetry, tmp_path):
        """Under the default SIGTERM action the process still dies by SIGTERM, after exporting."""
        exported = tmp_path / "spans.txt"
        script = textwrap.dedent(
            """
            import os, signal, sys, time
            from functools import partial
            from types import SimpleNamespace

            from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

            from megatron.bridge import telemetry
            from megatron.bridge.telemetry import lifecycle
            from megatron.bridge.training.config import LoggerConfig


            class FileSpanExporter(SpanExporter):
                def export(self, spans):
                    with open(sys.argv[1], "a") as output:
                        output.writelines(f"{span.name}\\n" for span in spans)
                    return SpanExportResult.SUCCESS


            telemetry._lens.setup_telemetry = partial(telemetry._lens.setup_telemetry, span_exporter=FileSpanExporter())
            config = SimpleNamespace(logger=LoggerConfig(otel_enabled=True), model=SimpleNamespace(), data_parallel_size=1)
            assert lifecycle.init_telemetry(config) is not None
            with telemetry.span(telemetry.JOB, "before.sigterm"):
                pass
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(60)
            sys.exit("SIGTERM did not end the process")
            """
        )
        environment = {
            **{key: value for key, value in os.environ.items() if not key.startswith("MEGATRON_OTEL")},
            "OTEL_BSP_SCHEDULE_DELAY": "600000",
            "MEGATRON_OTEL_METRICS_ENABLED": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
        }

        result = subprocess.run(
            [sys.executable, "-c", script, str(exported)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == -signal.SIGTERM, result.stderr[-2000:]
        assert exported.read_text() == "before.sigterm\n"


class TestTrainingMetrics:
    """record_training_metrics exports megatron-core's metrics only when Megatron Bridge owns telemetry."""

    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch):
        monkeypatch.setenv("MEGATRON_OTEL_ENABLED", "1")

    def test_records_megatron_core_metrics(self, lifecycle_capture):
        lifecycle.init_telemetry(_config())

        lifecycle.record_training_metrics(step_duration_ms=250.0, throughput_tflops=123.5, skipped_iters=2)

        metrics = lifecycle_capture.metrics()
        step_duration = metrics["megatron.training.step_duration_ms"].data.data_points[0]
        assert (step_duration.count, step_duration.sum) == (1, 250.0)
        assert metrics["megatron.training.step_duration_ms"].unit == "ms"
        assert metrics["megatron.training.throughput_tflops"].data.data_points[0].value == 123.5
        assert metrics["megatron.training.skipped_iters"].data.data_points[0].value == 2
        assert ("megatron.training.memory_allocated_gb" in metrics) == torch.cuda.is_available()
        assert "megatron.training.loss" not in metrics

    def test_does_nothing_without_telemetry(self, clean_telemetry):
        lifecycle.record_training_metrics(step_duration_ms=1.0, throughput_tflops=None, skipped_iters=0)

    def test_megatron_core_without_telemetry_records_nothing(self, lifecycle_capture, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "megatron.core.telemetry.training_metrics", None)

        with caplog.at_level(logging.INFO, logger=lifecycle.logger.name):
            assert lifecycle.init_telemetry(_config()) is not None
        lifecycle.record_training_metrics(step_duration_ms=1.0, throughput_tflops=None, skipped_iters=0)

        assert "training metrics are not exported" in caplog.text
        assert lifecycle_capture.metrics() == {}


class TestJobRunId:
    """The run id is one that every rank derives alike."""

    def test_slurm_job_id_is_left_to_nemo_lens(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "elastic")

        assert lifecycle._job_run_id(world_size=8) == ""

    def test_torchelastic_run_id_is_shared(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "elastic")

        assert lifecycle._job_run_id(world_size=8) == "elastic"

    @pytest.mark.parametrize(("world_size", "warns"), [(2, True), (1, False)])
    def test_without_shared_id(self, monkeypatch, caplog, world_size, warns):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "none")

        with caplog.at_level(logging.WARNING, logger=lifecycle.logger.name):
            assert lifecycle._job_run_id(world_size=world_size) == ""

        assert ("MEGATRON_OTEL_RUN_ID" in caplog.text) == warns
