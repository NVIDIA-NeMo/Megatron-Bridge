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
import random
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


class TestEnergonDataModuleCPHandling:
    """
    Unit tests for Context Parallelism (CP) handling in the energon datamodule.

    These tests mock parallel_state to simulate various CP/DP configurations
    without requiring real distributed initialization.
    """

    MODULE_PATH = "megatron.bridge.data.energon.base_energon_datamodule"

    @pytest.fixture
    def mock_energon(self):
        """Mock energon dependencies (get_train_dataset, get_savable_loader)."""
        with (
            patch(f"{self.MODULE_PATH}.get_train_dataset") as mock_get_dataset,
            patch(f"{self.MODULE_PATH}.get_savable_loader") as mock_get_loader,
        ):
            mock_dataset = MagicMock()
            mock_get_dataset.return_value = mock_dataset

            mock_loader = MagicMock()
            mock_data = [{"id": i} for i in range(10)]
            mock_loader.__iter__.side_effect = lambda: iter(mock_data)
            mock_loader.save_state_rank.return_value = {"step": 0}
            mock_get_loader.return_value = mock_loader

            yield {
                "get_train_dataset": mock_get_dataset,
                "get_savable_loader": mock_get_loader,
                "loader": mock_loader,
            }

    @pytest.fixture
    def mock_parallel_state(self):
        """Mock parallel_state with configurable DP/CP ranks."""
        with patch(f"{self.MODULE_PATH}.parallel_state") as mock_ps:
            mock_ps.is_initialized.return_value = True
            mock_dp_group = MagicMock()
            mock_ps.get_data_parallel_group.return_value = mock_dp_group
            yield mock_ps

    def _make_datamodule(self, num_workers=2, num_val_workers=None, **kwargs):
        """Helper to create an EnergonMultiModalDataModule with sensible defaults."""
        return EnergonMultiModalDataModule(
            path="/tmp/mock_dataset",
            tokenizer=MagicMock(),
            image_processor=MagicMock(),
            seq_length=1024,
            micro_batch_size=2,
            global_batch_size=8,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            **kwargs,
        )

    def _get_worker_config(self, mocks, call_index=0):
        """Extract the worker_config passed to get_train_dataset."""
        return mocks["get_train_dataset"].call_args_list[call_index][1]["worker_config"]

    # ----------------------------------------------------------------
    # CP handling: train_dataloader
    # ----------------------------------------------------------------

    def test_train_dataloader_uses_pure_dp_rank_with_cp(self, mock_energon, mock_parallel_state):
        """
        With CP=2 and DP=4, the train dataloader should use the pure DP rank
        (not the combined DP-CP rank), so CP ranks within the same DP replica
        read the same data shard.
        """
        mock_parallel_state.get_data_parallel_rank.return_value = 1
        mock_parallel_state.get_data_parallel_world_size.return_value = 4
        mock_parallel_state.get_context_parallel_rank.return_value = 1
        mock_parallel_state.get_context_parallel_world_size.return_value = 2

        dm = self._make_datamodule()
        dm.train_dataloader()

        wc = self._get_worker_config(mock_energon)
        assert wc.rank == 1
        assert wc.world_size == 4
        # Verify pure DP functions were called (without with_context_parallel=True)
        mock_parallel_state.get_data_parallel_rank.assert_called_once_with()
        mock_parallel_state.get_data_parallel_world_size.assert_called_once_with()
        mock_parallel_state.get_data_parallel_group.assert_called_once_with()

    def test_train_cp_ranks_in_same_dp_replica_get_same_config(self, mock_energon, mock_parallel_state):
        """
        Two CP ranks (cp_rank=0 and cp_rank=1) within the same DP replica (dp_rank=0)
        should produce identical WorkerConfig (same rank, world_size).
        """
        configs = []
        for cp_rank in [0, 1]:
            # Reset mocks for each "rank"
            mock_energon["get_train_dataset"].reset_mock()
            mock_energon["get_savable_loader"].reset_mock()

            mock_parallel_state.get_data_parallel_rank.return_value = 0
            mock_parallel_state.get_data_parallel_world_size.return_value = 2
            mock_parallel_state.get_context_parallel_rank.return_value = cp_rank
            mock_parallel_state.get_context_parallel_world_size.return_value = 2

            dm = self._make_datamodule()
            dm.train_dataloader()
            configs.append(self._get_worker_config(mock_energon))

        assert configs[0].rank == configs[1].rank == 0
        assert configs[0].world_size == configs[1].world_size == 2

    def test_train_different_dp_ranks_get_different_config(self, mock_energon, mock_parallel_state):
        """Different DP ranks should receive different worker config ranks for data sharding."""
        configs = []
        for dp_rank in [0, 1]:
            mock_energon["get_train_dataset"].reset_mock()
            mock_energon["get_savable_loader"].reset_mock()

            mock_parallel_state.get_data_parallel_rank.return_value = dp_rank
            mock_parallel_state.get_data_parallel_world_size.return_value = 2
            mock_parallel_state.get_context_parallel_rank.return_value = 0
            mock_parallel_state.get_context_parallel_world_size.return_value = 2

            dm = self._make_datamodule()
            dm.train_dataloader()
            configs.append(self._get_worker_config(mock_energon))

        assert configs[0].rank == 0
        assert configs[1].rank == 1
        assert configs[0].world_size == configs[1].world_size == 2

    def test_train_dataloader_cp1_equivalent_to_no_cp(self, mock_energon, mock_parallel_state):
        """With CP=1, behavior should be identical to no context parallelism."""
        mock_parallel_state.get_data_parallel_rank.return_value = 3
        mock_parallel_state.get_data_parallel_world_size.return_value = 8
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_world_size.return_value = 1

        dm = self._make_datamodule()
        dm.train_dataloader()

        wc = self._get_worker_config(mock_energon)
        assert wc.rank == 3
        assert wc.world_size == 8

    # ----------------------------------------------------------------
    # CP handling: val_dataloader
    # ----------------------------------------------------------------

    def test_val_dataloader_uses_pure_dp_rank_with_cp(self, mock_energon, mock_parallel_state):
        """Val dataloader should also use the pure DP rank, not combined DP-CP."""
        mock_parallel_state.get_data_parallel_rank.return_value = 2
        mock_parallel_state.get_data_parallel_world_size.return_value = 4
        mock_parallel_state.get_context_parallel_rank.return_value = 1
        mock_parallel_state.get_context_parallel_world_size.return_value = 2

        dm = self._make_datamodule()
        dm.val_dataloader()

        wc = self._get_worker_config(mock_energon)
        assert wc.rank == 2
        assert wc.world_size == 4

    def test_val_dataloader_uses_num_val_workers(self, mock_energon, mock_parallel_state):
        """Val dataloader should use num_val_workers, not num_workers."""
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_data_parallel_world_size.return_value = 1
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_world_size.return_value = 1

        dm = self._make_datamodule(num_workers=4, num_val_workers=8)
        dm.val_dataloader()

        wc = self._get_worker_config(mock_energon)
        assert wc.num_workers == 8

    def test_val_dataloader_defaults_num_val_workers_to_num_workers(self, mock_energon, mock_parallel_state):
        """When num_val_workers is not set, it should default to num_workers."""
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_data_parallel_world_size.return_value = 1
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_world_size.return_value = 1

        dm = self._make_datamodule(num_workers=6)
        dm.val_dataloader()

        wc = self._get_worker_config(mock_energon)
        assert wc.num_workers == 6

    # ----------------------------------------------------------------
    # Uninitialized parallel state fallback
    # ----------------------------------------------------------------

    @pytest.mark.usefixtures("mock_energon")
    def test_train_dataloader_uninitialized_uses_default_worker_config(self):
        """When parallel_state is not initialized, should use WorkerConfig.default_worker_config."""
        with patch(f"{self.MODULE_PATH}.parallel_state") as mock_ps:
            mock_ps.is_initialized.return_value = False
            with patch(f"{self.MODULE_PATH}.WorkerConfig") as mock_wc_cls:
                mock_default_wc = MagicMock()
                mock_wc_cls.default_worker_config.return_value = mock_default_wc

                dm = self._make_datamodule(num_workers=3)
                dm.train_dataloader()

                mock_wc_cls.default_worker_config.assert_called_once_with(3)

    @pytest.mark.usefixtures("mock_energon")
    def test_val_dataloader_uninitialized_uses_default_worker_config(self):
        """Val path should use num_val_workers with default_worker_config when uninitialized."""
        with patch(f"{self.MODULE_PATH}.parallel_state") as mock_ps:
            mock_ps.is_initialized.return_value = False
            with patch(f"{self.MODULE_PATH}.WorkerConfig") as mock_wc_cls:
                mock_default_wc = MagicMock()
                mock_wc_cls.default_worker_config.return_value = mock_default_wc

                dm = self._make_datamodule(num_workers=3, num_val_workers=5)
                dm.val_dataloader()

                mock_wc_cls.default_worker_config.assert_called_once_with(5)

    # ----------------------------------------------------------------
    # Dataloader caching
    # ----------------------------------------------------------------

    def test_train_dataloader_is_cached(self, mock_energon, mock_parallel_state):
        """Second call to train_dataloader should not rebuild the underlying loader."""
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_data_parallel_world_size.return_value = 1
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_world_size.return_value = 1

        dm = self._make_datamodule()
        loader1 = dm.train_dataloader()
        assert isinstance(loader1, EnergonDataloader)

        dm.train_dataloader()

        # get_savable_loader should only be called once — the underlying
        # energon loader is cached and not recreated on subsequent calls.
        assert mock_energon["get_savable_loader"].call_count == 1

    def test_val_dataloader_is_cached(self, mock_energon, mock_parallel_state):
        """Second call to val_dataloader should not rebuild the underlying loader."""
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_data_parallel_world_size.return_value = 1
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_world_size.return_value = 1

        dm = self._make_datamodule()
        loader1 = dm.val_dataloader()
        assert isinstance(loader1, EnergonDataloader)

        dm.val_dataloader()

        assert mock_energon["get_savable_loader"].call_count == 1

    # ----------------------------------------------------------------
    # datasets_provider and other edge cases
    # ----------------------------------------------------------------

    def test_datasets_provider_invalid_split_raises(self):
        """datasets_provider should raise ValueError for invalid split names."""
        dm = self._make_datamodule()
        with pytest.raises(ValueError, match="Invalid value for split"):
            dm.datasets_provider(MagicMock(), split="test")

    def test_datasets_provider_uses_validation_task_encoder_for_val(self, mock_energon):
        """Val split should use validation_task_encoder, not the train task_encoder."""
        train_encoder = MagicMock(name="train_encoder")
        val_encoder = MagicMock(name="val_encoder")

        dm = self._make_datamodule(task_encoder=train_encoder, validation_task_encoder=val_encoder)
        dm.datasets_provider(MagicMock(), split="val")

        _, kwargs = mock_energon["get_train_dataset"].call_args
        assert kwargs["task_encoder"] is val_encoder

    def test_datasets_provider_uses_train_task_encoder_for_train(self, mock_energon):
        """Train split should use the train task_encoder."""
        train_encoder = MagicMock(name="train_encoder")
        val_encoder = MagicMock(name="val_encoder")

        dm = self._make_datamodule(task_encoder=train_encoder, validation_task_encoder=val_encoder)
        dm.datasets_provider(MagicMock(), split="train")

        _, kwargs = mock_energon["get_train_dataset"].call_args
        assert kwargs["task_encoder"] is train_encoder

    def test_test_dataloader_returns_none(self):
        """test_dataloader should return None."""
        dm = self._make_datamodule()
        assert dm.test_dataloader() is None

    @pytest.mark.usefixtures("mock_energon")
    def test_build_returns_train_and_val(self, mock_parallel_state):
        """build() should return (train_dataloader, val_dataloader)."""
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_data_parallel_world_size.return_value = 1
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_world_size.return_value = 1

        dm = self._make_datamodule()
        train_loader, val_loader = dm.build()

        assert isinstance(train_loader, EnergonDataloader)
        assert isinstance(val_loader, EnergonDataloader)

    @pytest.mark.usefixtures("mock_energon")
    def test_energon_dataloader_cyclic_iteration(self, mock_parallel_state):
        """EnergonDataloader should cycle through data indefinitely."""
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_data_parallel_world_size.return_value = 1
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_world_size.return_value = 1

        dm = self._make_datamodule()
        loader = dm.train_dataloader()

        # Iterate beyond the 10 mock items to verify cycling
        items = [next(loader) for _ in range(12)]
        assert items[0] == {"id": 0}
        assert items[9] == {"id": 9}
        # Item 10 should cycle back to the beginning
        assert items[10] == {"id": 0}

    def test_energon_dataloader_save_state(self, mock_energon, mock_parallel_state):
        """EnergonDataloader.save_state() should delegate to the underlying loader."""
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_data_parallel_world_size.return_value = 1
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_world_size.return_value = 1

        dm = self._make_datamodule()
        loader = dm.train_dataloader()
        state = loader.save_state()

        assert state == {"step": 0}
        mock_energon["loader"].save_state_rank.assert_called_once()


class TestEnergonDataShardingVerification:
    """
    Verification tests that simulate multi-rank data loading to confirm:
    - Different DP ranks receive different batches
    - CP ranks within the same DP group receive the same input_ids
    - Reproducibility with the same seed

    These tests use rank-aware mock loaders that produce deterministic,
    rank-specific data — simulating how energon shards by WorkerConfig.rank.
    """

    MODULE_PATH = "megatron.bridge.data.energon.base_energon_datamodule"
    SEQ_LENGTH = 16
    VOCAB_SIZE = 32000
    NUM_BATCHES = 5
    MICRO_BATCH_SIZE = 2

    @staticmethod
    def _generate_rank_batches(rank, seed, num_batches, seq_length, micro_batch_size, vocab_size):
        """
        Generate deterministic batches for a given (rank, seed) pair.

        Simulates energon's behavior: same (rank, seed) always produces the
        same sequence of batches; different ranks produce different sequences.
        """
        rng = random.Random(seed * 1000 + rank)
        batches = []
        for _ in range(num_batches):
            input_ids = torch.tensor(
                [[rng.randint(0, vocab_size - 1) for _ in range(seq_length)] for _ in range(micro_batch_size)]
            )
            batches.append({"input_ids": input_ids})
        return batches

    def _make_rank_aware_energon_mocks(self, seed):
        """
        Create mock get_savable_loader that returns a loader whose data
        is deterministic per (WorkerConfig.rank, seed).
        """

        def loader_factory(dataset, worker_config):
            rank = worker_config.rank
            batches = self._generate_rank_batches(
                rank=rank,
                seed=seed,
                num_batches=self.NUM_BATCHES,
                seq_length=self.SEQ_LENGTH,
                micro_batch_size=self.MICRO_BATCH_SIZE,
                vocab_size=self.VOCAB_SIZE,
            )
            mock_loader = MagicMock()
            mock_loader.__iter__.side_effect = lambda: iter(batches)
            mock_loader.save_state_rank.return_value = {}
            return mock_loader

        return loader_factory

    def _build_loader_for_rank(self, dp_rank, dp_world_size, cp_rank, cp_size, seed):
        """
        Construct a datamodule + dataloader for a simulated (dp_rank, cp_rank),
        using rank-aware mocks. Returns the list of batches drawn from the loader.
        """
        with (
            patch(f"{self.MODULE_PATH}.parallel_state") as mock_ps,
            patch(f"{self.MODULE_PATH}.get_train_dataset") as mock_get_dataset,
            patch(f"{self.MODULE_PATH}.get_savable_loader") as mock_get_loader,
        ):
            mock_ps.is_initialized.return_value = True
            mock_ps.get_data_parallel_rank.return_value = dp_rank
            mock_ps.get_data_parallel_world_size.return_value = dp_world_size
            mock_ps.get_data_parallel_group.return_value = MagicMock()
            mock_ps.get_context_parallel_rank.return_value = cp_rank
            mock_ps.get_context_parallel_world_size.return_value = cp_size

            mock_get_dataset.return_value = MagicMock()
            mock_get_loader.side_effect = self._make_rank_aware_energon_mocks(seed)

            dm = EnergonMultiModalDataModule(
                path="/tmp/mock_dataset",
                tokenizer=MagicMock(),
                image_processor=MagicMock(),
                seq_length=self.SEQ_LENGTH,
                micro_batch_size=self.MICRO_BATCH_SIZE,
                global_batch_size=self.MICRO_BATCH_SIZE * dp_world_size,
                num_workers=1,
            )
            loader = dm.train_dataloader()
            batches = [next(loader) for _ in range(self.NUM_BATCHES)]
            return batches

    # ----------------------------------------------------------------
    # Verification: Different DP ranks receive different batches
    # ----------------------------------------------------------------

    def test_different_dp_ranks_receive_different_batches(self):
        """
        Simulate DP=2, CP=2.

        DP rank 0 and DP rank 1 should receive DIFFERENT batches,
        since they load different shards of the dataset.
        """
        seed = 42
        dp0_batches = self._build_loader_for_rank(dp_rank=0, dp_world_size=2, cp_rank=0, cp_size=2, seed=seed)
        dp1_batches = self._build_loader_for_rank(dp_rank=1, dp_world_size=2, cp_rank=0, cp_size=2, seed=seed)

        # At least one batch must differ between the two DP ranks
        all_same = all(
            torch.equal(dp0_batches[i]["input_ids"], dp1_batches[i]["input_ids"]) for i in range(self.NUM_BATCHES)
        )
        assert not all_same, (
            "DP rank 0 and DP rank 1 produced identical batches — data is not being sharded across DP ranks"
        )

    # ----------------------------------------------------------------
    # Verification: CP ranks in same DP group receive same input_ids
    # ----------------------------------------------------------------

    def test_cp_ranks_in_same_dp_group_receive_same_input_ids(self):
        """
        Simulate DP=2, CP=2.

        CP rank 0 and CP rank 1 within the SAME DP group (dp_rank=0)
        should receive IDENTICAL input_ids, because they process
        different sequence portions of the same batch.
        """
        seed = 42
        cp0_batches = self._build_loader_for_rank(dp_rank=0, dp_world_size=2, cp_rank=0, cp_size=2, seed=seed)
        cp1_batches = self._build_loader_for_rank(dp_rank=0, dp_world_size=2, cp_rank=1, cp_size=2, seed=seed)

        for i in range(self.NUM_BATCHES):
            assert torch.equal(cp0_batches[i]["input_ids"], cp1_batches[i]["input_ids"]), (
                f"Batch {i}: CP rank 0 and CP rank 1 within DP group 0 "
                f"received different input_ids.\n"
                f"  cp0: {cp0_batches[i]['input_ids'][0, :8].tolist()}...\n"
                f"  cp1: {cp1_batches[i]['input_ids'][0, :8].tolist()}..."
            )

    def test_cp_ranks_in_second_dp_group_also_match(self):
        """
        Same verification for the second DP group (dp_rank=1):
        CP rank 0 and CP rank 1 should still receive identical input_ids.
        """
        seed = 42
        cp0_batches = self._build_loader_for_rank(dp_rank=1, dp_world_size=2, cp_rank=0, cp_size=2, seed=seed)
        cp1_batches = self._build_loader_for_rank(dp_rank=1, dp_world_size=2, cp_rank=1, cp_size=2, seed=seed)

        for i in range(self.NUM_BATCHES):
            assert torch.equal(cp0_batches[i]["input_ids"], cp1_batches[i]["input_ids"]), (
                f"Batch {i}: CP rank 0 and CP rank 1 within DP group 1 received different input_ids"
            )

    # ----------------------------------------------------------------
    # Verification: Reproducibility with same seed
    # ----------------------------------------------------------------

    def test_same_seed_produces_identical_batches(self):
        """
        Two independent runs with the same (dp_rank, cp_rank, seed)
        should produce byte-identical batches.
        """
        seed = 42
        run1 = self._build_loader_for_rank(dp_rank=0, dp_world_size=4, cp_rank=0, cp_size=2, seed=seed)
        run2 = self._build_loader_for_rank(dp_rank=0, dp_world_size=4, cp_rank=0, cp_size=2, seed=seed)

        for i in range(self.NUM_BATCHES):
            assert torch.equal(run1[i]["input_ids"], run2[i]["input_ids"]), (
                f"Batch {i}: Two runs with identical config produced different data — not reproducible"
            )

    def test_different_seed_produces_different_batches(self):
        """
        Same rank but different seeds should produce different batches,
        confirming the seed actually controls randomness.
        """
        batches_seed42 = self._build_loader_for_rank(dp_rank=0, dp_world_size=2, cp_rank=0, cp_size=1, seed=42)
        batches_seed99 = self._build_loader_for_rank(dp_rank=0, dp_world_size=2, cp_rank=0, cp_size=1, seed=99)

        all_same = all(
            torch.equal(batches_seed42[i]["input_ids"], batches_seed99[i]["input_ids"])
            for i in range(self.NUM_BATCHES)
        )
        assert not all_same, "Different seeds produced identical batches — seed has no effect"

    # ----------------------------------------------------------------
    # Verification: Full rank matrix (DP=2, CP=2 → 4 ranks)
    # ----------------------------------------------------------------

    def test_full_dp2_cp2_rank_matrix(self):
        """
        Exhaustive check for a DP=2, CP=2 setup (4 total data-loading ranks).

        Expected behavior:
          Global rank 0 (dp=0, cp=0) ─┐ same input_ids
          Global rank 1 (dp=0, cp=1) ─┘
          Global rank 2 (dp=1, cp=0) ─┐ same input_ids (but different from dp=0)
          Global rank 3 (dp=1, cp=1) ─┘
        """
        seed = 123
        batches = {}
        for dp_rank in [0, 1]:
            for cp_rank in [0, 1]:
                batches[(dp_rank, cp_rank)] = self._build_loader_for_rank(
                    dp_rank=dp_rank, dp_world_size=2, cp_rank=cp_rank, cp_size=2, seed=seed
                )

        # Within DP group 0: cp=0 and cp=1 must match
        for i in range(self.NUM_BATCHES):
            assert torch.equal(
                batches[(0, 0)][i]["input_ids"],
                batches[(0, 1)][i]["input_ids"],
            ), f"Batch {i}: DP group 0 — CP ranks diverged"

        # Within DP group 1: cp=0 and cp=1 must match
        for i in range(self.NUM_BATCHES):
            assert torch.equal(
                batches[(1, 0)][i]["input_ids"],
                batches[(1, 1)][i]["input_ids"],
            ), f"Batch {i}: DP group 1 — CP ranks diverged"

        # Across DP groups: dp=0 and dp=1 must differ
        all_same = all(
            torch.equal(batches[(0, 0)][i]["input_ids"], batches[(1, 0)][i]["input_ids"])
            for i in range(self.NUM_BATCHES)
        )
        assert not all_same, "DP group 0 and DP group 1 produced identical batches"
