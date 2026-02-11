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

import datetime
import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from megatron.bridge.data.energon.base_energon_datamodule import (
    EnergonDataloader,
    EnergonMultiModalDataModule,
)


class TestEnergonMultiModalDataModuleFunctional:
    @pytest.fixture(autouse=True)
    def setup_and_teardown_parallel_state(self):
        """Setup and teardown parallel state for Megatron tests."""

        # Initialize distributed backend if not already done
        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29501"  # Use a different port to avoid conflicts
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(0)

            init_process_group_kwargs = {
                "backend": "nccl" if device_count > 0 else "gloo",
                "world_size": 1,
                "rank": 0,
                "timeout": datetime.timedelta(minutes=30),
            }

            dist.init_process_group(**init_process_group_kwargs)

        assert dist.is_initialized(), "Distributed backend not initialized"

        # Initialize model parallel state
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                context_parallel_size=1,
            )

        assert parallel_state.model_parallel_is_initialized(), "Model parallel not initialized"

        # Seed
        from megatron.core.process_groups_config import ProcessGroupCollection

        from megatron.bridge.training.initialize import _set_random_seed

        # Create pg_collection from initialized mpu
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
            pg_collection=pg_collection,
        )

        yield

        # Teardown
        try:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
                # Clean up environment variables
                for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                    os.environ.pop(key, None)
        except (NameError, AttributeError, RuntimeError):
            pass

    @pytest.fixture
    def mock_energon_dependencies(self):
        """
        Mock the external Energon dependencies (dataset creation, loading)
        since we don't have a real Energon dataset available in this environment.
        """
        with (
            patch("megatron.bridge.data.energon.base_energon_datamodule.get_train_dataset") as mock_get_dataset,
            patch("megatron.bridge.data.energon.base_energon_datamodule.get_savable_loader") as mock_get_loader,
        ):
            # Setup dataset mock
            mock_dataset = MagicMock()
            mock_get_dataset.return_value = mock_dataset

            # Setup loader mock
            mock_loader_instance = MagicMock()
            # Infinite iterator of mock data
            mock_data = [{"id": i} for i in range(10)]
            mock_loader_instance.__iter__.side_effect = lambda: iter(mock_data)
            mock_loader_instance.save_state_rank.return_value = {"rank_state": 123}

            mock_get_loader.return_value = mock_loader_instance

            yield {
                "get_train_dataset": mock_get_dataset,
                "get_savable_loader": mock_get_loader,
                "loader_instance": mock_loader_instance,
            }

    def test_datamodule_distributed_initialization(self, mock_energon_dependencies):
        """
        Test that the DataModule correctly initializes in a distributed environment
        (using the real parallel_state).
        """

        # 1. Initialization
        datamodule = EnergonMultiModalDataModule(
            path="/tmp/mock_dataset",
            tokenizer=MagicMock(),
            image_processor=MagicMock(),
            seq_length=1024,
            micro_batch_size=2,
            global_batch_size=4,
            num_workers=2,
        )

        # 2. Build DataLoaders
        # This triggers worker_config creation using real parallel_state
        train_loader, val_loader = datamodule.build()

        assert isinstance(train_loader, EnergonDataloader)
        assert isinstance(val_loader, EnergonDataloader)

        # 3. Verify WorkerConfig was created correctly
        # We can inspect the calls to get_train_dataset to see what worker_config was passed
        args, kwargs = mock_energon_dependencies["get_train_dataset"].call_args_list[0]  # First call (train)
        worker_config = kwargs["worker_config"]

        # Verify worker config properties derived from parallel_state
        assert worker_config.rank == 0
        assert worker_config.world_size == 1
        assert worker_config.num_workers == 2

        # 4. Functional check of the wrapper (Data Iteration)
        train_iterator = iter(train_loader)
        samples = []
        for _ in range(3):
            samples.append(next(train_iterator))

        assert len(samples) == 3
        assert samples[0] == {"id": 0}

        # 5. State Saving
        state = train_loader.save_state()
        assert state == {"rank_state": 123}
