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

import time
from dataclasses import dataclass

import numpy as np
import torch

from megatron.bridge.training.utils.train_utils import (
    report_l2_norm_grad,
    report_memory,
    report_runtime,
    report_throughput,
)


@dataclass
class MockTrainState:
    step: int = None
    consumed_train_samples: int = None


@dataclass
class MockTrainConfig:
    global_batch_size: int = None
    micro_batch_size: int = None


@dataclass
class MockParam:
    requires_grad: bool = True
    main_grad: float = None


class MockModelChunk:
    def __init__(self, layer_name, param):
        self.layer_name = layer_name
        self.param = param

    def named_parameters(self):
        yield self.layer_name, self.param


class TestTrainingMetrics:
    """Test extra training metrics."""

    def test_report_memory(self):
        """Test memory metrics."""
        memory_report = report_memory(memory_keys=None)
        assert len(memory_report) == 10

        memory_keys = {
            "reserved_bytes.all.current": "mem-reserved-bytes",
            "reserved_bytes.all.peak": "mem-max-reserved-bytes",
        }
        excepted_keys = ["mem-reserved-gigabytes", "mem-max-reserved-gigabytes"]
        memory_report = report_memory(memory_keys=memory_keys)
        assert list(memory_report.keys()) == excepted_keys

    def test_report_runtime(self):
        """Test runtime metrics."""
        start_time = time.time()

        step = 100
        consumed_train_samples = 1000
        seq_length = 2048
        train_iters = 1000

        train_state = MockTrainState(step=step, consumed_train_samples=consumed_train_samples)
        runtime_report = report_runtime(
            train_state=train_state,
            start_time=start_time,
            seq_length=seq_length,
            train_iters=train_iters,
        )

        assert runtime_report["time/tokens"] == consumed_train_samples * seq_length
        assert runtime_report["time/samples"] == consumed_train_samples

    def test_report_throughput(self):
        """Test throughput metrics."""
        global_batch_size = 64
        micro_batch_size = 4
        iteration = 100
        seq_length = 4096
        history_wct = [0.9, 1.7, 2.9, 4.2, 5.9]
        window_size = len(history_wct)
        train_config = MockTrainConfig(global_batch_size=global_batch_size, micro_batch_size=micro_batch_size)

        throughput_report = report_throughput(
            train_config=train_config,
            iteration=iteration,
            seq_length=seq_length,
            history_wct=history_wct,
            window_size=window_size,
        )

        assert throughput_report["throughput/tokens_per_sec"] == 209715.2
        assert throughput_report["throughput/batches_per_sec"] == 0.8
        assert throughput_report["throughput/micro_batch_size"] == 4
        assert throughput_report["throughput/device/samples_per_sec"] == 51.2

    def test_l2_norm_grad(self):
        """Test l2 norm grad metrics."""
        num_chunks = 10
        layer_name = "layer"
        model = []
        # generate mock model
        for i in range(num_chunks):
            chunk_name = f"{layer_name}_{i}"
            main_grad = torch.tensor(i).float()
            param = MockParam(main_grad=main_grad)
            model_chunk = MockModelChunk(chunk_name, param)
            model.append(model_chunk)

        l2_norm_report = report_l2_norm_grad(model)
        assert np.round(l2_norm_report["l2_norm/grad/global"], 2) == 74.92
        assert l2_norm_report["l2_norm/grad/layer_2"] == 2.0
        assert l2_norm_report["l2_norm/grad/layer_9"] == 9.0
