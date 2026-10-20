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

"""End-to-end telemetry: train and resume with telemetry enabled, and check what every rank exports."""

import functools
import os
import signal
from collections import Counter

import pytest
import torch

from megatron.bridge import telemetry
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import broadcast_path, clear_directories, initialize_distributed
from tests.unit_tests.telemetry.utils import TelemetryCapture, recording_metric_exporter, reset_telemetry_state


RUN_ID = "functional-telemetry"


def _config(checkpoint_dir: str, *, train_iters: int, load: str | None) -> ConfigContainer:
    seq_length = 64
    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )
    return ConfigContainer(
        model=BridgeGPTModelConfig(
            transformer=transformer,
            vocab_size=128,
            seq_length=seq_length,
            share_embeddings_and_output_weights=False,
        ),
        # Two ranks of data parallelism, so two microbatches per rank and iteration.
        train=TrainingConfig(train_iters=train_iters, global_batch_size=4, micro_batch_size=1),
        validation=ValidationConfig(eval_interval=2, eval_iters=1),
        optimizer=OptimizerConfig(
            optimizer="adam",
            bf16=True,
            fp16=False,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
            clip_grad=1.0,
            lr=3e-4,
            min_lr=3e-5,
        ),
        scheduler=SchedulerConfig(
            start_weight_decay=0.01,
            end_weight_decay=0.01,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=1,
            lr_decay_iters=train_iters,
            override_opt_param_scheduler=True,
        ),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=False,
            use_distributed_optimizer=False,
        ),
        dataset=MockGPTDatasetConfig(
            random_seed=1234,
            seq_length=seq_length,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            dataloader_type="single",
            num_workers=0,
        ),
        logger=LoggerConfig(log_interval=1, otel_enabled=True, otel_span_groups="all"),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=128),
        checkpoint=CheckpointConfig(save=checkpoint_dir, load=load, save_interval=2, ckpt_format="torch_dist"),
        rng=RNGConfig(seed=1234),
    )


def _by_name(spans: list) -> dict[str, list]:
    named: dict[str, list] = {}
    for span in spans:
        named.setdefault(span.name, []).append(span)
    return named


def _iterations(spans: list) -> list[int]:
    return sorted(span.attributes[telemetry.ITERATION] for span in spans)


def _ancestors(span, spans_by_id: dict) -> list[str]:
    names = []
    while span.parent is not None and span.parent.span_id in spans_by_id:
        span = spans_by_id[span.parent.span_id]
        names.append(span.name)
    return names


@pytest.fixture
def capture(monkeypatch):
    """Route the providers that pretrain installs to in-memory readers, and reset them afterwards."""
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    if not telemetry.HAVE_NEMO_LENS:
        pytest.skip("nemo-lens with SpanRegistry is not installed")
    metric_exporter = recording_metric_exporter()
    capture = TelemetryCapture(
        span_exporter=InMemorySpanExporter(),
        # Only the flushes that pretrain does export: the interval outlasts the test.
        metric_reader=PeriodicExportingMetricReader(metric_exporter, export_interval_millis=3_600_000),
        metric_exporter=metric_exporter,
    )
    monkeypatch.setenv("MEGATRON_OTEL_RUN_ID", RUN_ID)
    monkeypatch.setattr(
        telemetry._lens,
        "setup_telemetry",
        functools.partial(
            telemetry._lens.setup_telemetry,
            span_exporter=capture.span_exporter,
            metric_reader=capture.metric_reader,
        ),
    )
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    reset_telemetry_state()
    yield capture
    reset_telemetry_state()
    signal.signal(signal.SIGTERM, previous_sigterm)


class TestTelemetry:
    """Train four iterations, then resume for two, recording every span group on each rank."""

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_and_resume_export_spans_and_metrics(self, tmp_path, capture):
        initialize_distributed()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        base_dir = broadcast_path(tmp_path)
        checkpoint_dir = os.path.join(base_dir, "checkpoints")
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
        torch.distributed.barrier()

        try:
            pretrain(_config(checkpoint_dir, train_iters=4, load=None), forward_step)
            trained = capture.spans()
            capture.span_exporter.clear()
            pretrain(_config(checkpoint_dir, train_iters=6, load=checkpoint_dir), forward_step)
            resumed = capture.spans()
        finally:
            clear_directories(base_dir)

        # Every span carries this rank's identity and the job's run id.
        resource = trained[0].resource.attributes
        assert resource["service.name"] == "megatron-bridge"
        assert resource["nemo.run.id"] == RUN_ID
        assert resource["nv.dl.rank"] == rank
        assert resource["nv.dl.world_size"] == world_size
        assert resource["dl.data_parallel.size"] == world_size

        spans = _by_name(trained)
        spans_by_id = {span.context.span_id: span for span in trained}
        counts = Counter(span.name for span in trained)
        assert counts["megatron.startup"] == 1
        assert counts["megatron.train.first_iteration"] == 1
        for name in ("megatron.bridge.model.build", "megatron.startup.dataloader"):
            assert _ancestors(spans[name][0], spans_by_id) == ["megatron.startup"], name

        # One span per iteration, numbered as the training log numbers them; the first one is nested
        # in the first-iteration span.
        assert _iterations(spans["megatron.train.iteration"]) == [1, 2, 3, 4]
        assert _iterations(spans["megatron.train.first_iteration"]) == [1]
        first_iteration = min(spans["megatron.train.iteration"], key=lambda span: span.attributes[telemetry.ITERATION])
        assert _ancestors(first_iteration, spans_by_id) == ["megatron.train.first_iteration"]
        for name in ("megatron.train.iteration.forward_backward", "megatron.train.iteration.optimizer"):
            assert counts[name] == 4, name
            assert all(_ancestors(span, spans_by_id)[0] == "megatron.train.iteration" for span in spans[name])
        assert {
            span.attributes["megatron.train.num_microbatches"]
            for span in spans["megatron.train.iteration.forward_backward"]
        } == {2}

        # megatron-core's own spans light up under Megatron Bridge's providers, nested in the training
        # iterations (two microbatches each) or in the evaluation passes.
        microbatch_parents = Counter(
            "train" if "megatron.train.iteration.forward_backward" in ancestors else "evaluate"
            for ancestors in (_ancestors(span, spans_by_id) for span in spans["megatron.microbatch.forward"])
            if "megatron.train.iteration.forward_backward" in ancestors or "megatron.evaluate" in ancestors
        )
        assert microbatch_parents["train"] == 4 * 2
        assert sum(microbatch_parents.values()) == counts["megatron.microbatch.forward"]
        assert any(name.startswith("megatron.layer.") for name in counts)

        # Checkpoints at iterations 2 and 4, each with its phases.
        assert _iterations(spans["megatron.checkpoint.exposed_save"]) == [2, 4]
        assert _iterations(spans["megatron.checkpoint.save"]) == [2, 4]
        for save in spans["megatron.checkpoint.save"]:
            assert _ancestors(save, spans_by_id) == ["megatron.checkpoint.exposed_save"]
            assert save.attributes["megatron.checkpoint.format"] == "torch_dist"
            assert save.attributes["megatron.checkpoint.async"] is False
        for phase in ("megatron.checkpoint.save.state_dict", "megatron.checkpoint.save.io_write"):
            assert all(_ancestors(span, spans_by_id)[0] == "megatron.checkpoint.save" for span in spans[phase])
        assert set(_iterations(spans["megatron.evaluate"])) == {2, 4}
        assert "megatron.checkpoint.load" not in counts

        # The resumed run reuses the providers, loads the checkpoint, and continues the numbering.
        spans = _by_name(resumed)
        spans_by_id = {span.context.span_id: span for span in resumed}
        assert len(spans["megatron.checkpoint.load"]) == 1
        assert _ancestors(spans["megatron.checkpoint.load.io_read"][0], spans_by_id)[0] == "megatron.checkpoint.load"
        assert _iterations(spans["megatron.train.iteration"]) == [5, 6]
        assert _iterations(spans["megatron.train.first_iteration"]) == [5]

        # Training metrics, recorded at every log interval of both runs and exported by the flush that
        # ends each run.
        exported = capture.metric_exporter
        assert exported.points("megatron.training.step_duration_ms")[-1].count == 6
        assert exported.points("megatron.training.step_duration_ms")[-1].sum > 0
        assert len(exported.points("megatron.training.throughput_tflops")) == 2
        assert all(point.value > 0 for point in exported.points("megatron.training.memory_allocated_gb"))
        assert exported.points("megatron.training.loss") == []
