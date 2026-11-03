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

"""Real Energon stream ownership and checkpoint continuation with GTP and CP."""

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.config import set_experimental_flag
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.energon import DefaultTaskEncoder, TextSample, stateless

from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule
from megatron.bridge.data.energon.prepare import prepare_webdataset
from megatron.bridge.training.checkpointing import maybe_load_dataloader_state, maybe_save_dataloader_state
from tests.functional_tests.utils import broadcast_path, initialize_distributed


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """Generate the dataset locally without downloading test assets."""
    yield


class _TextEncoder(DefaultTaskEncoder):
    @stateless
    def encode_sample(self, sample: TextSample) -> int:
        return int(sample.text)

    def batch(self, samples: list[int]) -> torch.Tensor:
        return torch.tensor(samples)


def _prepare_text_dataset(path: Path) -> None:
    path.mkdir(parents=True)
    for shard in range(8):
        with tarfile.open(path / f"train-{shard:03d}.tar", "w") as archive:
            for index in range(shard * 16, (shard + 1) * 16):
                content = str(index).encode()
                info = tarfile.TarInfo(f"{index:06d}.txt")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    prepare_webdataset(path, {"train": "train-.*"}, num_workers=1)
    (path / ".nv-meta" / "dataset.yaml").write_text(
        "sample_type:\n  __module__: megatron.energon\n  __class__: TextSample\nfield_map:\n  text: txt\n"
    )


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("topology", ["gtp", "cp"])
def test_energon_gtp_streams_and_checkpoint_resume(tmp_path, topology):
    if topology == "gtp" and not HAVE_GTP:
        pytest.skip("GTP requires TransformerEngine >= 2.19")
    initialize_distributed()
    if torch.distributed.get_world_size() != 2:
        pytest.skip("This test requires exactly two ranks")
    set_experimental_flag(True)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=2 if topology == "cp" else 1,
        gtp_remat_size=2 if topology == "gtp" else 1,
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        root = Path(broadcast_path(tmp_path))
        dataset_path = root / "dataset"
        if torch.distributed.get_rank() == 0:
            _prepare_text_dataset(dataset_path)
        torch.distributed.barrier()

        def build_loader():
            module = EnergonMultiModalDataModule(
                path=str(dataset_path),
                tokenizer=None,
                micro_batch_size=1,
                num_workers=0,
                shuffle_buffer_size=0,
                task_encoder=_TextEncoder(),
                pg_collection=pg,
            )
            return module.train_dataloader()

        loader = build_loader()
        first_samples = [int(next(loader).item()) for _ in range(8)]
        peer_samples = [None, None]
        torch.distributed.all_gather_object(peer_samples, first_samples)
        if topology == "gtp":
            assert set(peer_samples[0]).isdisjoint(peer_samples[1])
        else:
            assert peer_samples[0] == peer_samples[1]

        state_path = str(root / "state")
        maybe_save_dataloader_state([], SimpleNamespace(iterable=loader), 8, state_path, pg_collection=pg)
        torch.distributed.barrier()
        state_files = list((root / "state" / "iter_0000008").glob("train_dataloader_dprank*.pt"))
        assert len(state_files) == (2 if topology == "gtp" else 1)
        expected = [int(next(loader).item()) for _ in range(12)]

        restored = build_loader()
        maybe_load_dataloader_state(SimpleNamespace(iterable=restored), 8, state_path, pg_collection=pg)
        actual = [int(next(restored).item()) for _ in range(12)]
        assert actual == expected
        torch.distributed.barrier()
    finally:
        parallel_state.destroy_model_parallel()
