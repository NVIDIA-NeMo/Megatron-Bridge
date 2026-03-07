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

from megatron.bridge.diffusion.data.wan import wan_energon_datamodule as wan_dm_mod
from megatron.bridge.diffusion.data.wan.wan_taskencoder import WanTaskEncoder


class _FakeDiffusionDataModule:
    def __init__(
        self,
        *,
        path: str,
        seq_length: int,
        packing_buffer_size: int,
        task_encoder,
        micro_batch_size: int,
        global_batch_size: int,
        num_workers: int,
    ):
        self.path = path
        self.seq_length = seq_length
        self.packing_buffer_size = packing_buffer_size
        self.task_encoder = task_encoder
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers

    # mimic API used by WanDataModuleConfig.build_datasets
    def train_dataloader(self):
        return "train"


def test_wan_datamodule_config_initialization(monkeypatch):
    # Patch the symbol used inside wan_energon_datamodule module
    monkeypatch.setattr(wan_dm_mod, "DiffusionDataModule", _FakeDiffusionDataModule)

    cfg = wan_dm_mod.WanDataModuleConfig(
        path="",
        seq_length=128,
        task_encoder_seq_length=128,
        packing_buffer_size=4,
        micro_batch_size=2,
        global_batch_size=8,
        num_workers=0,
    )

    # __post_init__ should construct a dataset with WanTaskEncoder and propagate seq_length
    assert isinstance(cfg.dataset, _FakeDiffusionDataModule)
    assert cfg.sequence_length == cfg.dataset.seq_length == 128
    assert isinstance(cfg.dataset.task_encoder, WanTaskEncoder)
    assert cfg.dataset.task_encoder.seq_length == 128
    assert cfg.dataset.task_encoder.packing_buffer_size == 4

    # build_datasets should return train loader thrice
    train, val, test = cfg.build_datasets(context=None)
    assert train == "train" and val == "train" and test == "train"
