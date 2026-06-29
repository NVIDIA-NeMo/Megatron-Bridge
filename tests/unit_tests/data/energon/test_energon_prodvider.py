# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

from megatron.bridge.data.energon.energon_provider import EnergonProvider
from megatron.bridge.data.utils import DatasetBuildContext


class _FakeTaskEncoder:
    """Minimal task encoder carrying the fields build_datasets() re-applies."""

    def __init__(self):
        # Sentinels that differ from the provider values so we can confirm assignment.
        self.seq_length = -1


def _mock_datamodule(mock_datamodule_cls):
    inst = MagicMock()
    mock_datamodule_cls.return_value = inst
    inst.train_dataloader.return_value = iter([])
    inst.val_dataloader.side_effect = lambda: iter([])
    return inst


def _make_provider(task_encoder, **overrides):
    params = dict(
        path="test/path",
        seq_length=8192,
        micro_batch_size=1,
        global_batch_size=8,
        num_workers=0,
        task_encoder=task_encoder,
    )
    params.update(overrides)
    return EnergonProvider(**params)


class TestEnergonProvider:
    @patch("megatron.bridge.data.energon.energon_provider.EnergonMultiModalDataModule")
    def test_init_and_build_datasets(self, mock_datamodule_cls):
        # Setup mock instance
        mock_dataset_instance = MagicMock()
        mock_datamodule_cls.return_value = mock_dataset_instance

        # Setup mock return values for dataloaders
        # Making them iterable
        mock_dataset_instance.train_dataloader.return_value = iter([1, 2])
        # Since val_dataloader is called twice and returns an iterator, we need to be careful.
        # However, calling iter() on an iterator is fine.
        # But if the method returns a list, iter() works.
        # If it returns an iterator, and we iterate it once, the second time it will be empty if it's the SAME iterator.
        # The implementation calls `iter(self.dataset.val_dataloader())`.
        # So `val_dataloader()` is called twice.
        # We should make sure it returns a new iterable/iterator each time.
        mock_dataset_instance.val_dataloader.side_effect = lambda: iter([3, 4])

        mock_dataset_instance.seq_length = 2048

        # Define params
        params = {
            "path": "test/path",
            "image_processor": MagicMock(),
            "seq_length": 2048,
            "micro_batch_size": 1,
            "global_batch_size": 8,
            "num_workers": 4,
            "task_encoder": MagicMock(),
        }

        # Instantiate provider
        provider = EnergonProvider(**params)

        # Check sequence_length property
        assert provider.seq_length == 2048

        # Test build_datasets
        context = MagicMock(spec=DatasetBuildContext)
        train_iter, val_iter, test_iter = provider.build_datasets(context)

        # Check if EnergonMultiModalDataModule was initialized with correct args
        mock_datamodule_cls.assert_called_once_with(
            path=params["path"],
            tokenizer=context.tokenizer,
            image_processor=params["image_processor"],
            seq_length=params["seq_length"],
            task_encoder=params["task_encoder"],
            micro_batch_size=params["micro_batch_size"],
            global_batch_size=params["global_batch_size"],
            num_workers=params["num_workers"],
            pg_collection=context.pg_collection,
        )

        # Check dataloader calls
        mock_dataset_instance.train_dataloader.assert_called_once()
        assert mock_dataset_instance.val_dataloader.call_count == 2

        # Verify returned iterators
        assert list(train_iter) == [1, 2]
        assert list(val_iter) == [3, 4]
        assert list(test_iter) == [3, 4]

    # build_datasets() re-applies provider fields onto the eagerly-built task encoder so a
    # dataset.seq_length=... override reaches it before data loading.

    @patch("megatron.bridge.data.energon.energon_provider.EnergonMultiModalDataModule")
    def test_build_datasets_syncs_seq_length(self, mock_datamodule_cls):
        _mock_datamodule(mock_datamodule_cls)
        encoder = _FakeTaskEncoder()
        provider = _make_provider(encoder, seq_length=8192)

        provider.build_datasets(MagicMock(spec=DatasetBuildContext))

        assert encoder.seq_length == 8192

    @patch("megatron.bridge.data.energon.energon_provider.EnergonMultiModalDataModule")
    def test_build_datasets_no_op_when_task_encoder_is_none(self, mock_datamodule_cls):
        _mock_datamodule(mock_datamodule_cls)
        provider = _make_provider(task_encoder=None)
        # Should not raise even though task_encoder is None.
        provider.build_datasets(MagicMock(spec=DatasetBuildContext))

    @patch("megatron.bridge.data.energon.energon_provider.EnergonMultiModalDataModule")
    def test_build_datasets_passes_synced_encoder_for_validation(self, mock_datamodule_cls):
        # The provider passes a single task_encoder to the datamodule, which reuses it as
        # validation_task_encoder when no separate one is given, so the validation split also
        # sees the synced seq_length.
        _mock_datamodule(mock_datamodule_cls)
        encoder = _FakeTaskEncoder()
        provider = _make_provider(encoder, seq_length=8192)

        provider.build_datasets(MagicMock(spec=DatasetBuildContext))

        _, kwargs = mock_datamodule_cls.call_args
        assert kwargs["task_encoder"] is encoder
        assert kwargs["task_encoder"].seq_length == 8192
