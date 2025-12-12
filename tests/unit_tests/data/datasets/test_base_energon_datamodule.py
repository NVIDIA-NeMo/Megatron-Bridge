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

from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.data.datasets.base_energon_datamodule import (
    EnergonDataloader,
    EnergonMultiModalDataModule,
    cyclic_iter,
)


class TestEnergonMultiModalDataModule:
    @pytest.fixture
    def mock_dependencies(self):
        with (
            patch("megatron.bridge.data.datasets.base_energon_datamodule.parallel_state") as mock_parallel_state,
            patch("megatron.bridge.data.datasets.base_energon_datamodule.get_train_dataset") as mock_get_train_dataset,
            patch(
                "megatron.bridge.data.datasets.base_energon_datamodule.get_savable_loader"
            ) as mock_get_savable_loader,
            patch("megatron.bridge.data.datasets.base_energon_datamodule.WorkerConfig") as mock_worker_config,
        ):
            mock_parallel_state.is_initialized.return_value = True
            mock_parallel_state.get_data_parallel_rank.return_value = 0
            mock_parallel_state.get_data_parallel_world_size.return_value = 1
            mock_parallel_state.get_data_parallel_group.return_value = MagicMock()

            yield {
                "parallel_state": mock_parallel_state,
                "get_train_dataset": mock_get_train_dataset,
                "get_savable_loader": mock_get_savable_loader,
                "WorkerConfig": mock_worker_config,
            }

    @pytest.fixture
    def datamodule(self):
        return EnergonMultiModalDataModule(
            path="test_path",
            tokenizer=MagicMock(),
            image_processor=MagicMock(),
            seq_length=1024,
            micro_batch_size=2,
            global_batch_size=4,
            num_workers=2,
        )

    def test_init(self, datamodule):
        assert datamodule.path == "test_path"
        assert datamodule.seq_length == 1024
        assert datamodule.micro_batch_size == 2
        assert datamodule.global_batch_size == 4
        assert datamodule.num_workers == 2
        assert datamodule.pin_memory is True
        assert datamodule.init_global_step == 0
        assert datamodule.train_dataloader_object is None
        assert datamodule.val_dataloader_object is None

    def test_datasets_provider(self, datamodule, mock_dependencies):
        mock_worker_config = MagicMock()
        mock_dependencies["get_train_dataset"].return_value = "mock_dataset"

        # Test train split
        dataset = datamodule.datasets_provider(mock_worker_config, split="train")
        assert dataset == "mock_dataset"
        mock_dependencies["get_train_dataset"].assert_called_with(
            "test_path",
            batch_size=2,
            task_encoder=datamodule.task_encoder,
            worker_config=mock_worker_config,
            packing_buffer_size=None,
            split_part="train",
            shuffle_buffer_size=100,
            max_samples_per_sequence=None,
        )

        # Test val split
        mock_dependencies["get_train_dataset"].reset_mock()
        dataset = datamodule.datasets_provider(mock_worker_config, split="val")
        assert dataset == "mock_dataset"
        mock_dependencies["get_train_dataset"].assert_called_with(
            "test_path",
            batch_size=2,
            task_encoder=datamodule.validation_task_encoder,
            worker_config=mock_worker_config,
            packing_buffer_size=None,
            split_part="val",
            shuffle_buffer_size=100,
            max_samples_per_sequence=None,
        )

        # Test invalid split
        with pytest.raises(ValueError, match="Invalid value for split"):
            datamodule.datasets_provider(mock_worker_config, split="invalid")

    def test_train_dataloader(self, datamodule, mock_dependencies):
        mock_dependencies["get_savable_loader"].return_value = MagicMock()
        mock_dependencies["WorkerConfig"].return_value = MagicMock()

        dataloader = datamodule.train_dataloader()

        assert isinstance(dataloader, EnergonDataloader)
        assert datamodule.train_dataloader_object is not None
        mock_dependencies["WorkerConfig"].assert_called()
        mock_dependencies["get_savable_loader"].assert_called()

        # Test returning cached dataloader
        cached_dataloader = datamodule.train_dataloader()
        assert cached_dataloader == datamodule.train_dataloader_object

    def test_train_dataloader_uninitialized_parallel_state(self, datamodule, mock_dependencies):
        mock_dependencies["parallel_state"].is_initialized.return_value = False
        mock_dependencies["WorkerConfig"].default_worker_config.return_value = MagicMock()
        mock_dependencies["get_savable_loader"].return_value = MagicMock()

        datamodule.train_dataloader()

        mock_dependencies["WorkerConfig"].default_worker_config.assert_called_with(datamodule.num_workers)

    def test_val_dataloader(self, datamodule, mock_dependencies):
        mock_dependencies["get_savable_loader"].return_value = MagicMock()
        mock_dependencies["WorkerConfig"].return_value = MagicMock()

        dataloader = datamodule.val_dataloader()
        
        assert isinstance(dataloader, EnergonDataloader)
        assert datamodule.val_dataloader_object is not None
        mock_dependencies["WorkerConfig"].assert_called()
        mock_dependencies["get_savable_loader"].assert_called()

    def test_test_dataloader(self, datamodule):
        assert datamodule.test_dataloader() is None

    def test_state_dict(self, datamodule, mock_dependencies):
        mock_trainer = MagicMock()
        mock_trainer.train_dataloader = MagicMock()
        mock_trainer.global_step = 100
        datamodule.trainer = mock_trainer
        datamodule.data_sampler = MagicMock()
        datamodule.data_sampler.compute_consumed_samples.return_value = 500

        mock_dependencies["parallel_state"].get_context_parallel_rank.return_value = 0
        mock_dependencies["parallel_state"].get_pipeline_model_parallel_rank.return_value = 0
        mock_dependencies["parallel_state"].get_tensor_model_parallel_rank.return_value = 0
        mock_dependencies["parallel_state"].get_expert_model_parallel_rank.return_value = 0
        
        mock_trainer.train_dataloader.save_state_global.return_value = ["state"]

        state_dict = datamodule.state_dict()
        
        assert state_dict["consumed_samples"] == 500
        assert state_dict["dataloader_state"] == ["state"]

        # Test empty return when no trainer
        datamodule.trainer = None
        assert datamodule.state_dict() == {}

    def test_load_state_dict(self, datamodule, mock_dependencies):
        mock_trainer = MagicMock()
        mock_trainer.datamodule.train_dataloader.return_value = MagicMock()
        datamodule.trainer = mock_trainer
        datamodule.data_sampler = MagicMock()
        
        state_dict = {"dataloader_state": ["state"], "consumed_samples": 500}

        # Patch megatron.core.num_microbatches_calculator module so that when it's imported
        # inside load_state_dict, we get our mock.
        mock_module = MagicMock()
        mock_update = MagicMock()
        mock_module.update_num_microbatches = mock_update
        
        with patch.dict("sys.modules", {"megatron.core.num_microbatches_calculator": mock_module}):
            datamodule.load_state_dict(state_dict)
            
            mock_trainer.datamodule.train_dataloader().restore_state_global.assert_called_with(["state"])
            assert datamodule.data_sampler.init_consumed_samples == 500
            mock_update.assert_called_with(consumed_samples=500, consistency_check=False)

    def test_load_state_dict_missing_key(self, datamodule):
        datamodule.load_state_dict({})
        # Should log warning and return, no error raised
    
    def test_load_state_dict_no_trainer(self, datamodule):
        datamodule.trainer = None
        datamodule.data_sampler = MagicMock()
        state_dict = {"dataloader_state": ["state"], "consumed_samples": 500}
        
        # Patch megatron.core.num_microbatches_calculator module to avoid side effects
        mock_module = MagicMock()
        mock_update = MagicMock()
        mock_module.update_num_microbatches = mock_update

        with patch.dict("sys.modules", {"megatron.core.num_microbatches_calculator": mock_module}):
            # Should raise ValueError inside the try block, caught and logged as warning
            datamodule.load_state_dict(state_dict)
            
            # Verify update_num_microbatches was still called because the exception is caught
            mock_update.assert_called_with(consumed_samples=500, consistency_check=False)


class TestEnergonDataloader:
    def test_init(self):
        mock_loader = MagicMock()
        mock_loader.__iter__.side_effect = lambda: iter([1, 2, 3])
        
        dataloader = EnergonDataloader(mock_loader)
        
        assert dataloader._dataloader == mock_loader
        # EnergonDataloader makes it cyclic
        
        # Test iteration
        it = iter(dataloader)
        assert next(it) == 1
        assert next(it) == 2
        assert next(it) == 3
        assert next(it) == 1  # Cycling back

    def test_save_state(self):
        mock_loader = MagicMock()
        dataloader = EnergonDataloader(mock_loader)
        
        dataloader.save_state()
        mock_loader.save_state_rank.assert_called_once()


class TestCyclicIter:
    def test_cyclic_iter(self):
        data = [1, 2]
        it = cyclic_iter(data)
        
        assert next(it) == 1
        assert next(it) == 2
        assert next(it) == 1
        assert next(it) == 2
