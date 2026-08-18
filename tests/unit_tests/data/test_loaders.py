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

from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import RerunDataIterator

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.loaders import (
    build_train_valid_test_data_iterators,
    build_train_valid_test_data_loaders,
    get_train_valid_test_num_samples,
)
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.training.state import TrainState


@pytest.mark.unit
@mock.patch("torch.distributed.broadcast")
@mock.patch("torch.distributed.get_world_size", return_value=1)
@mock.patch("torch.distributed.get_rank", return_value=0)
def test_batch_loader_does_not_supervise_custom_dataset_padding(_mock_rank, _mock_world_size, _mock_broadcast):
    class OrdinaryDataset:
        def __init__(self, size):
            self.samples = [
                {"sample_id": torch.tensor(index), "loss_mask": torch.tensor(1.0)} for index in range(size)
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    @dataclass
    class OrdinaryDatasetProvider(DatasetProvider):
        def build_datasets(self, context: DatasetBuildContext):
            return OrdinaryDataset(context.train_samples), None, None

    for dataset_size in (3, 4):
        provider = OrdinaryDatasetProvider(
            dataloader_type="batch",
            drop_last=False,
            num_workers=0,
            persistent_workers=False,
        )
        provider.finalize()
        cfg = SimpleNamespace(
            model=object(),
            dataset=provider,
            train=SimpleNamespace(
                train_samples=dataset_size,
                train_iters=1,
                global_batch_size=4,
                micro_batch_size=1,
                num_epochs=None,
                exit_signal=None,
                exit_signal_handler_for_dataloader=False,
            ),
            validation=SimpleNamespace(
                eval_interval=0,
                eval_iters=0,
                eval_global_batch_size=None,
                eval_micro_batch_size=None,
                skip_train=False,
                eval_at_step_zero=False,
                multiple_validation_sets=False,
                validation_set_names=None,
            ),
        )
        real_torch_tensor = torch.tensor

        def tensor_on_cpu(*args, **kwargs):
            kwargs.pop("device", None)
            return real_torch_tensor(*args, **kwargs)

        try:
            with mock.patch(
                "megatron.bridge.data.loaders.torch.tensor",
                side_effect=tensor_on_cpu,
            ):
                train_dataloader, _, _ = build_train_valid_test_data_loaders(
                    cfg=cfg,
                    train_state=TrainState(),
                    build_train_valid_test_datasets_provider=get_dataset_provider(provider),
                    dp_group=object(),
                )
        except ValueError as error:
            assert dataset_size == 3
            assert "drop_last=False" in str(error)
            assert "padding" in str(error)
            continue

        batch = next(iter(train_dataloader))
        assert batch["loss_mask"].sum().item() == dataset_size, (
            "The padded batch must not supervise a duplicated real sample"
        )


@pytest.mark.unit
@mock.patch("torch.distributed.broadcast")
@mock.patch("torch.distributed.get_world_size", return_value=1)
@mock.patch("torch.distributed.get_rank", return_value=0)
def test_multiple_validation_sets_build_one_dataloader_per_set(_mock_rank, _mock_world_size, _mock_broadcast):
    class RangeDataset:
        def __init__(self, offset, size):
            self.samples = [{"sample_id": torch.tensor(offset + index)} for index in range(size)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    @dataclass
    class MultiValDatasetProvider(DatasetProvider):
        def build_datasets(self, context: DatasetBuildContext):
            return (
                RangeDataset(0, context.train_samples),
                [RangeDataset(100, 4), RangeDataset(200, 4)],
                None,
            )

    def make_cfg(provider, multiple_validation_sets, validation_set_names=None):
        return SimpleNamespace(
            model=object(),
            dataset=provider,
            train=SimpleNamespace(
                train_samples=4,
                train_iters=1,
                global_batch_size=2,
                micro_batch_size=1,
                num_epochs=None,
                exit_signal=None,
                exit_signal_handler_for_dataloader=False,
            ),
            validation=SimpleNamespace(
                eval_interval=1,
                eval_iters=1,
                eval_global_batch_size=None,
                eval_micro_batch_size=None,
                skip_train=False,
                eval_at_step_zero=False,
                multiple_validation_sets=multiple_validation_sets,
                validation_set_names=validation_set_names,
            ),
        )

    provider = MultiValDatasetProvider(
        dataloader_type="single",
        drop_last=True,
        num_workers=0,
        persistent_workers=False,
    )
    provider.finalize()

    real_torch_tensor = torch.tensor

    def tensor_on_cpu(*args, **kwargs):
        kwargs.pop("device", None)
        return real_torch_tensor(*args, **kwargs)

    with mock.patch("megatron.bridge.data.loaders.torch.tensor", side_effect=tensor_on_cpu):
        with pytest.raises(ValueError, match="multiple_validation_sets"):
            build_train_valid_test_data_loaders(
                cfg=make_cfg(provider, multiple_validation_sets=False),
                train_state=TrainState(),
                build_train_valid_test_datasets_provider=get_dataset_provider(provider),
                dp_group=object(),
            )

        _, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
            cfg=make_cfg(provider, multiple_validation_sets=True),
            train_state=TrainState(),
            build_train_valid_test_datasets_provider=get_dataset_provider(provider),
            dp_group=object(),
        )

    assert isinstance(valid_dataloader, list) and len(valid_dataloader) == 2
    assert test_dataloader is None
    assert next(iter(valid_dataloader[0]))["sample_id"].item() == 100
    assert next(iter(valid_dataloader[1]))["sample_id"].item() == 200

    # The iterator layer preserves the per-set structure: one RerunDataIterator
    # per set, in blend order, while train stays a single iterator.
    with mock.patch("megatron.bridge.data.loaders.torch.tensor", side_effect=tensor_on_cpu):
        train_iter, valid_iters, test_iter = build_train_valid_test_data_iterators(
            cfg=make_cfg(provider, multiple_validation_sets=True),
            train_state=TrainState(),
            build_train_valid_test_datasets_provider=get_dataset_provider(provider),
            dp_group=object(),
        )

    assert isinstance(train_iter, RerunDataIterator)
    assert test_iter is None
    assert isinstance(valid_iters, list) and len(valid_iters) == 2
    assert all(isinstance(it, RerunDataIterator) for it in valid_iters)
    assert next(valid_iters[0])["sample_id"].item() == 100
    assert next(valid_iters[1])["sample_id"].item() == 200


@pytest.mark.unit
@mock.patch("torch.distributed.broadcast")
@mock.patch("torch.distributed.get_world_size", return_value=1)
@mock.patch("torch.distributed.get_rank", return_value=0)
def test_multiple_validation_sets_build_time_guards(_mock_rank, _mock_world_size, _mock_broadcast):
    """The set-name length and cross-rank set-count guards fail at dataloader build time."""

    class RangeDataset:
        def __init__(self, offset, size):
            self.samples = [{"sample_id": torch.tensor(offset + index)} for index in range(size)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    @dataclass
    class MultiValDatasetProvider(DatasetProvider):
        def build_datasets(self, context: DatasetBuildContext):
            return (
                RangeDataset(0, context.train_samples),
                [RangeDataset(100, 4), RangeDataset(200, 4)],
                None,
            )

    def make_cfg(provider, validation_set_names=None):
        return SimpleNamespace(
            model=object(),
            dataset=provider,
            train=SimpleNamespace(
                train_samples=4,
                train_iters=1,
                global_batch_size=2,
                micro_batch_size=1,
                num_epochs=None,
                exit_signal=None,
                exit_signal_handler_for_dataloader=False,
            ),
            validation=SimpleNamespace(
                eval_interval=1,
                eval_iters=1,
                eval_global_batch_size=None,
                eval_micro_batch_size=None,
                skip_train=False,
                eval_at_step_zero=False,
                multiple_validation_sets=True,
                validation_set_names=validation_set_names,
            ),
        )

    provider = MultiValDatasetProvider(
        dataloader_type="single",
        drop_last=True,
        num_workers=0,
        persistent_workers=False,
    )
    provider.finalize()

    with pytest.raises(ValueError, match="validation_set_names"):
        build_train_valid_test_data_loaders(
            cfg=make_cfg(provider, validation_set_names=["only-one"]),
            train_state=TrainState(),
            build_train_valid_test_datasets_provider=get_dataset_provider(provider),
            dp_group=object(),
        )

    def fake_all_reduce(tensor, op=None):
        tensor[0] = 3
        tensor[1] = -2

    with (
        mock.patch("megatron.bridge.data.loaders.torch.distributed.is_initialized", return_value=True),
        mock.patch("megatron.bridge.data.loaders.torch.distributed.all_reduce", side_effect=fake_all_reduce),
    ):
        with pytest.raises(RuntimeError, match=r"min 2, max 3; this rank has 2"):
            build_train_valid_test_data_loaders(
                cfg=make_cfg(provider),
                train_state=TrainState(),
                build_train_valid_test_datasets_provider=get_dataset_provider(provider),
                dp_group=object(),
            )


@pytest.mark.unit
def test_eval_at_step_zero_reserves_pre_and_post_training_passes():
    """eval_at_step_zero budgets the step-zero pass, plus the post-training pass in _pretrain()
    when eval_interval is None and the base formula reserves nothing; a finite
    dataloader_type="single" loader must be sized for both."""

    def make_cfg(eval_interval):
        return SimpleNamespace(
            train=SimpleNamespace(train_samples=None, train_iters=10, global_batch_size=4),
            validation=SimpleNamespace(
                eval_interval=eval_interval,
                eval_iters=2,
                eval_global_batch_size=None,
                eval_at_step_zero=True,
            ),
        )

    # With a schedule, the base formula's "+1" already covers the post-training pass;
    # only the step-zero pass is extra.
    _, valid_samples, _ = get_train_valid_test_num_samples(make_cfg(eval_interval=5))
    assert valid_samples == ((10 // 5 + 1) * 2 + 2) * 4

    # Without a schedule, the run still evaluates at step zero and once after training.
    _, valid_samples, _ = get_train_valid_test_num_samples(make_cfg(eval_interval=None))
    assert valid_samples == 2 * 2 * 4


@pytest.mark.unit
@mock.patch("torch.distributed.broadcast")
@mock.patch("torch.distributed.get_world_size", return_value=1)
@mock.patch("torch.distributed.get_rank", return_value=0)
def test_disabling_multiple_validation_sets_resets_resume_offset(_mock_rank, _mock_world_size, _mock_broadcast):
    """Resuming with per-set counters after disabling multi-set validation restarts from offset 0.

    The aggregate counter sums samples consumed across the former sets; applied to a single
    dataset it would skip samples that were never drawn from it and can exhaust a finite loader.
    """

    class RangeDataset:
        def __init__(self, size):
            self.samples = [{"sample_id": torch.tensor(index)} for index in range(size)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    @dataclass
    class SingleValDatasetProvider(DatasetProvider):
        def build_datasets(self, context: DatasetBuildContext):
            return RangeDataset(context.train_samples), RangeDataset(6), None

    provider = SingleValDatasetProvider(
        dataloader_type="single",
        drop_last=True,
        num_workers=0,
        persistent_workers=False,
    )
    provider.finalize()

    cfg = SimpleNamespace(
        model=object(),
        dataset=provider,
        train=SimpleNamespace(
            train_samples=4,
            train_iters=1,
            global_batch_size=2,
            micro_batch_size=1,
            num_epochs=None,
            exit_signal=None,
            exit_signal_handler_for_dataloader=False,
        ),
        validation=SimpleNamespace(
            eval_interval=1,
            eval_iters=1,
            eval_global_batch_size=None,
            eval_micro_batch_size=None,
            skip_train=False,
            eval_at_step_zero=False,
            multiple_validation_sets=False,
            validation_set_names=None,
        ),
    )

    train_state = TrainState()
    train_state.consumed_valid_samples = 300
    train_state.consumed_valid_samples_per_set = [100, 200]

    real_torch_tensor = torch.tensor

    def tensor_on_cpu(*args, **kwargs):
        kwargs.pop("device", None)
        return real_torch_tensor(*args, **kwargs)

    with mock.patch("megatron.bridge.data.loaders.torch.tensor", side_effect=tensor_on_cpu):
        _, valid_dataloader, _ = build_train_valid_test_data_loaders(
            cfg=cfg,
            train_state=train_state,
            build_train_valid_test_datasets_provider=get_dataset_provider(provider),
            dp_group=object(),
        )

    assert next(iter(valid_dataloader))["sample_id"][0].item() == 0


@pytest.mark.unit
@mock.patch("torch.distributed.broadcast")
@mock.patch("torch.distributed.get_world_size", return_value=1)
@mock.patch("torch.distributed.get_rank", return_value=0)
def test_multiple_validation_sets_built_in_gpt_builder(_mock_rank, _mock_world_size, _mock_broadcast, tmp_path):
    """The built-in GPT blended-builder path splits the validation blend into one dataset per
    prefix when the dataset-side flag (mirrored from ValidationConfig by validate()) is set."""
    from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder
    from megatron.core.datasets.utils import compile_helpers

    from megatron.bridge.training.config import GPTDatasetConfig

    compile_helpers()

    class _Tokenizer:
        vocab_size = 128
        eod = 0
        pad = 1

    def make_prefix(name, token_offset):
        prefix = str(tmp_path / name)
        builder = IndexedDatasetBuilder(prefix + ".bin", dtype=DType.optimal_dtype(_Tokenizer.vocab_size))
        for doc in range(8):
            tokens = [(token_offset + doc + k) % _Tokenizer.vocab_size for k in range(32)]
            builder.add_document(tokens, [len(tokens)])
        builder.finalize(prefix + ".idx")
        return prefix

    train_prefix = make_prefix("train", 0)
    valid_prefix_a = make_prefix("valid_a", 1)
    valid_prefix_b = make_prefix("valid_b", 2)

    dataset_cfg = GPTDatasetConfig(
        seq_length=8,
        random_seed=1234,
        blend_per_split=[
            get_blend_from_list([train_prefix]),
            get_blend_from_list([valid_prefix_a, valid_prefix_b]),
            None,
        ],
        multiple_validation_sets=True,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
        tokenizer=_Tokenizer(),
        path_to_cache=str(tmp_path / "cache"),
        dataloader_type="single",
        drop_last=True,
        num_workers=0,
        persistent_workers=False,
    )
    dataset_cfg.finalize()

    cfg = SimpleNamespace(
        model=object(),
        dataset=dataset_cfg,
        train=SimpleNamespace(
            train_samples=8,
            train_iters=1,
            global_batch_size=2,
            micro_batch_size=1,
            num_epochs=None,
            exit_signal=None,
            exit_signal_handler_for_dataloader=False,
        ),
        validation=SimpleNamespace(
            eval_interval=1,
            eval_iters=1,
            eval_global_batch_size=None,
            eval_micro_batch_size=None,
            skip_train=False,
            eval_at_step_zero=False,
            multiple_validation_sets=True,
            validation_set_names=["a", "b"],
        ),
    )

    real_torch_tensor = torch.tensor

    def tensor_on_cpu(*args, **kwargs):
        kwargs.pop("device", None)
        return real_torch_tensor(*args, **kwargs)

    with mock.patch("megatron.bridge.data.loaders.torch.tensor", side_effect=tensor_on_cpu):
        _, valid_dataloader, _ = build_train_valid_test_data_loaders(
            cfg=cfg,
            train_state=TrainState(),
            build_train_valid_test_datasets_provider=get_dataset_provider(dataset_cfg),
            dp_group=object(),
        )

    assert isinstance(valid_dataloader, list) and len(valid_dataloader) == 2
    for per_set_loader in valid_dataloader:
        batch = next(iter(per_set_loader))
        assert batch["tokens"].shape[-1] == 8
