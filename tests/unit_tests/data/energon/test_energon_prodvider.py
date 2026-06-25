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

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.data.energon.energon_provider import EnergonProvider, _parse_val_blend_entries
from megatron.bridge.data.utils import DatasetBuildContext


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
            packing_buffer_size=None,
            pg_collection=context.pg_collection,
        )

        # Check dataloader calls
        mock_dataset_instance.train_dataloader.assert_called_once()
        assert mock_dataset_instance.val_dataloader.call_count == 2

        # Verify returned iterators
        assert list(train_iter) == [1, 2]
        assert list(val_iter) == [3, 4]
        assert list(test_iter) == [3, 4]

    @patch("megatron.bridge.data.energon.energon_provider.EnergonMultiModalDataModule")
    def test_build_datasets_train_split_preserves_save_state(self, mock_datamodule_cls):
        """The train split must be returned un-wrapped (not iter(...)) so the downstream
        RerunDataIterator retains save_state/restore_state for checkpoint save and resume."""
        mock_dataset_instance = MagicMock()
        mock_datamodule_cls.return_value = mock_dataset_instance

        # Stands in for the EnergonDataloader wrapper (exposes save_state/restore_state).
        train_loader = MagicMock()
        mock_dataset_instance.train_dataloader.return_value = train_loader
        mock_dataset_instance.val_dataloader.side_effect = lambda: iter([])
        mock_dataset_instance.seq_length = 2048

        provider = EnergonProvider(
            path="test/path",
            image_processor=MagicMock(),
            seq_length=2048,
            micro_batch_size=1,
            global_batch_size=8,
            num_workers=1,
            task_encoder=MagicMock(),
        )

        train_iter, _, _ = provider.build_datasets(MagicMock(spec=DatasetBuildContext))

        # Returned object is the dataloader itself, not iter(...), so save_state survives wrapping.
        assert train_iter is train_loader
        assert hasattr(train_iter, "save_state")

    @patch("megatron.bridge.data.energon.energon_provider.EnergonMultiModalDataModule")
    def test_build_datasets_forwards_packing_buffer_size(self, mock_datamodule_cls):
        """packing_buffer_size must reach the datamodule, otherwise Energon's sample-packing path
        stays a silent no-op (the task encoder's packing hooks are never invoked)."""
        mock_dataset_instance = MagicMock()
        mock_datamodule_cls.return_value = mock_dataset_instance
        mock_dataset_instance.val_dataloader.side_effect = lambda: iter([])

        provider = EnergonProvider(
            path="test/path",
            image_processor=MagicMock(),
            seq_length=2048,
            micro_batch_size=1,
            global_batch_size=8,
            num_workers=1,
            task_encoder=MagicMock(),
            packing_buffer_size=256,
        )

        provider.build_datasets(MagicMock(spec=DatasetBuildContext))

        _, kwargs = mock_datamodule_cls.call_args
        assert kwargs["packing_buffer_size"] == 256

    @patch("megatron.bridge.data.energon.energon_provider.EnergonMultiModalDataModule")
    def test_build_datasets_rejects_simultaneous_packing(self, mock_datamodule_cls):
        """Energon sample packing and megatron-bridge in-batch packing both concatenate samples;
        enabling both would pack the data twice, so build_datasets must reject the combination."""
        provider = EnergonProvider(
            path="test/path",
            image_processor=MagicMock(),
            seq_length=2048,
            micro_batch_size=1,
            global_batch_size=8,
            num_workers=1,
            task_encoder=MagicMock(),
            packing_buffer_size=256,
            pack_sequences_in_batch=True,
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            provider.build_datasets(MagicMock(spec=DatasetBuildContext))

        # The datamodule must not be constructed when the config is rejected.
        mock_datamodule_cls.assert_not_called()

    @patch("megatron.bridge.data.energon.energon_provider._parse_val_blend_entries")
    @patch("megatron.bridge.data.energon.energon_provider.EnergonMultiModalDataModule")
    def test_build_datasets_multiple_validation_sets(self, mock_datamodule_cls, mock_parse):
        """With multiple_validation_sets, the validation slot is the
        (combined_loader, [(name, loader), ...]) tuple bridge's per-set eval consumes, the test slot
        is None, and one val-only datamodule is built per sub-blend (plus the blended one)."""
        instance = MagicMock()
        mock_datamodule_cls.return_value = instance
        train_loader = MagicMock(name="train_loader")
        instance.train_dataloader.return_value = train_loader
        instance.val_dataloader.side_effect = lambda: iter([])
        mock_parse.return_value = [("qa", "/data/qa"), ("conv", "/data/conv")]

        provider = EnergonProvider(
            path="meta.yaml",
            image_processor=MagicMock(),
            seq_length=2048,
            micro_batch_size=1,
            global_batch_size=8,
            num_workers=1,
            task_encoder=MagicMock(),
            multiple_validation_sets=True,
        )
        train, valid, test = provider.build_datasets(MagicMock(spec=DatasetBuildContext))

        # train returned un-wrapped (base contract for save_state/restore_state).
        assert train is train_loader
        # validation slot is the named-tuple shape.
        assert isinstance(valid, tuple)
        _combined, named = valid
        assert [name for name, _ in named] == ["qa", "conv"]
        # blended datamodule (self.path) + one per sub-blend, built from their paths.
        built_paths = [kwargs["path"] for _args, kwargs in mock_datamodule_cls.call_args_list]
        assert built_paths == ["meta.yaml", "/data/qa", "/data/conv"]
        # test slot stays the plain blended val iterator (return type untouched, not the tuple).
        assert test is not None
        assert not isinstance(test, tuple)


class TestParseValBlendEntries:
    @staticmethod
    def _write(tmp_path: Path, body: str) -> str:
        path = tmp_path / "metadataset.yaml"
        path.write_text(textwrap.dedent(body))
        return str(path)

    def test_relative_paths_resolved_and_named_by_stem(self, tmp_path: Path) -> None:
        meta = self._write(
            tmp_path,
            """
            splits:
              val:
                blend:
                  - path: ./qa_blend.yaml
                  - path: ./conversation_blend.yaml
            """,
        )
        entries = _parse_val_blend_entries(meta)
        assert [name for name, _ in entries] == ["qa_blend", "conversation_blend"]
        assert [path for _, path in entries] == [
            str(tmp_path / "qa_blend.yaml"),
            str(tmp_path / "conversation_blend.yaml"),
        ]

    def test_absolute_path_kept_as_is(self, tmp_path: Path) -> None:
        abs_blend = "/abs/data/pretrain_blend.yaml"
        meta = self._write(
            tmp_path,
            f"""
            splits:
              val:
                blend:
                  - path: {abs_blend}
            """,
        )
        assert _parse_val_blend_entries(meta) == [("pretrain_blend", abs_blend)]

    def test_no_val_blend_raises(self, tmp_path: Path) -> None:
        meta = self._write(
            tmp_path,
            """
            splits:
              train:
                blend:
                  - path: ./train_blend.yaml
            """,
        )
        with pytest.raises(ValueError, match="No splits.val.blend"):
            _parse_val_blend_entries(meta)
