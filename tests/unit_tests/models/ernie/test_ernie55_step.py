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

"""CPU tests of the ERNIE 5.5 training step, its SFT data and its metric logging.

* The batch contract: documents from ``cu_seqlens``, labels masked by ``loss_mask``, padding rows dropped.
* The loss pairing: ``ernie_per_row_loss`` selects the per-row loss and requires
  ``calculate_per_token_loss=False``; at CP > 1 anything else must be per-token.
* The SFT data: Hugging Face labels shifted per sample (after splitting long samples), right truncation,
  first-fit packing, synthetic rows, and the collate contract the step consumes.
* ``forward_step`` is registered with the recipe runner; ``test_ernie55_step_cp.py`` runs it end to end at CP 2
  and 4 against CP 1.
* Logging: the indexer loss is divided by the ERNIE layers plus MTP depths, and the MoE z-loss by the MoE layers
  with the MTP's (both the reference's conventions).
"""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.models.ernie.ernie55_test_utils import allow_conversion_imports_without_te


# Bridge's conversion stack imports its LoRA layers, which need Transformer Engine; without TE, import it with a
# placeholder for them. The training stack imports ``datasets``; without it this module is skipped.
allow_conversion_imports_without_te()
pytest.importorskip("megatron.bridge.models.conversion.model_bridge", reason="needs Bridge's conversion stack")
pytest.importorskip("megatron.bridge.training.utils.train_utils", reason="needs Bridge's training stack")

from megatron.bridge.data.base import DatasetBuildContext
from megatron.bridge.models.ernie import ernie55_step
from megatron.bridge.models.ernie.ernie55_context_parallel import IGNORE_LABEL, ernie55_per_row_lm_loss
from megatron.bridge.models.ernie.ernie55_data import Ernie55SFTDatasetConfig, collate_ernie55_rows
from megatron.bridge.models.ernie.ernie55_step import (
    ernie55_documents_from_batch,
    ernie55_loss_function,
    get_batch,
)
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.utils import train_utils
from tests.unit_tests.models.ernie.test_ernie55_builder import _build, _tiny_config
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import (  # noqa: F401 - pytest fixtures
    _cpu_nvtx,
    _fresh_trackers,
    _randomize_,
    needs_hook,
    pg_collection,
)
from tests.unit_tests.models.ernie.test_ernie55_model_config import _config as _proxy_model_config


pytestmark = pytest.mark.unit


class _FakeGroup:
    def __init__(self, size: int, rank: int = 0):
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


@pytest.fixture
def fake_group_sizes(monkeypatch):
    """Let the step read the size of a :class:`_FakeGroup` without an initialized process group."""
    monkeypatch.setattr(ernie55_step, "get_pg_size", lambda group: 1 if group is None else group.size())


class TestBatchContract:
    def test_documents_from_cu_seqlens(self):
        batch = dict(
            tokens=torch.arange(10).unsqueeze(0),
            labels=torch.arange(100, 110).unsqueeze(0),
            cu_seqlens=torch.tensor([0, 3, 7]),
        )
        documents = ernie55_documents_from_batch(batch)
        assert [tokens.tolist() for tokens, _ in documents] == [[0, 1, 2], [3, 4, 5, 6]]
        assert [labels.tolist() for _, labels in documents] == [[100, 101, 102], [103, 104, 105, 106]]

    def test_rows_are_documents_without_cu_seqlens_and_loss_mask_drops_targets(self):
        batch = dict(
            tokens=torch.arange(6).view(2, 3),
            labels=torch.arange(10, 16).view(2, 3),
            loss_mask=torch.tensor([[1, 0, 1], [0, 1, 1]]),
        )
        documents = ernie55_documents_from_batch(batch)
        assert [labels.tolist() for _, labels in documents] == [[10, IGNORE_LABEL, 12], [IGNORE_LABEL, 14, 15]]

    @pytest.mark.parametrize(
        "batch, message",
        [
            (dict(tokens=torch.zeros(1, 4), labels=torch.zeros(1, 3)), "one \\[b, T\\] shape"),
            (dict(tokens=torch.zeros(2, 4), labels=torch.zeros(2, 4), cu_seqlens=torch.tensor([0, 4])), "one packed"),
            (dict(tokens=torch.zeros(1, 4), labels=torch.zeros(1, 4), cu_seqlens=torch.tensor([0, 5])), "increase"),
            (dict(tokens=torch.zeros(1, 4), labels=torch.zeros(1, 4), cu_seqlens=torch.tensor([0, 2, 2])), "increase"),
            (dict(tokens=torch.zeros(1, 4), labels=torch.zeros(1, 4), cu_seqlens=torch.tensor([1, 4])), "increase"),
        ],
    )
    def test_malformed_batches(self, batch, message):
        with pytest.raises(ValueError, match=message):
            ernie55_documents_from_batch(batch)

    # 6 tokens are a multiple of CP = 2 but not of 2 * CP: the zigzag halves still need padding.
    @pytest.mark.parametrize("length", [7, 6])
    def test_unaligned_documents_need_a_pad_token(self, fake_group_sizes, length):
        batch = dict(tokens=torch.zeros(1, length, dtype=torch.long), labels=torch.zeros(1, length, dtype=torch.long))
        with pytest.raises(ValueError, match="pad_token_id"):
            get_batch(
                iter([batch]), SimpleNamespace(cp_partition_mode="contiguous"), cp_group=_FakeGroup(2), device="cpu"
            )

    def test_aligned_pack_too_short_for_the_boundary_exchange_needs_a_pad_token(self, fake_group_sizes):
        # 8 aligned tokens at CP = 2 with an H layer (128 rows per rank): the pack must grow to 256 rows.
        batch = dict(tokens=torch.zeros(1, 8, dtype=torch.long), labels=torch.zeros(1, 8, dtype=torch.long))
        config = SimpleNamespace(cp_partition_mode="contiguous", csa_compress_ratios=[128, 0], csa_window_size=128)
        with pytest.raises(ValueError, match="pad_token_id"):
            get_batch(iter([batch]), config, cp_group=_FakeGroup(2), device="cpu")

    def test_cp1_batch(self):
        collated = collate_ernie55_rows(
            [[([5, 6, 7], [6, 7, IGNORE_LABEL]), ([8, 9], [9, IGNORE_LABEL])]], pad_token_id=0
        )
        batch = get_batch(iter([collated]), SimpleNamespace(cp_partition_mode="zigzag"), cp_group=None, device="cpu")
        assert batch.tokens.tolist() == [[5, 6, 7, 8, 9]]
        assert batch.position_ids.tolist() == [[0, 1, 2, 0, 1]]
        assert batch.loss_mask.tolist() == [[1, 1, 0, 1, 0]]
        assert batch.packed_seq_params.cu_seqlens_q.tolist() == [0, 3, 5]
        assert batch.packed_seq_params.cu_seqlens_kv is batch.packed_seq_params.cu_seqlens_q_padded


class TestLossPairing:
    def test_per_row(self):
        config = SimpleNamespace(ernie_per_row_loss=True, calculate_per_token_loss=False)
        loss_function = ernie55_loss_function(
            torch.ones(1, 4), config=config, cp_group=None, check_for_nan_in_loss=True, check_for_spiky_loss=False
        )
        assert loss_function.func is ernie55_per_row_lm_loss
        loss, num_tokens, report = loss_function(torch.tensor([[1.0, 2.0, 3.0, 6.0]]))
        assert float(loss) == pytest.approx(3.0) and int(num_tokens) == 1
        assert report["lm loss"].tolist() == pytest.approx([3.0, 1.0])

    def test_per_row_requires_micro_batch_normalization(self):
        config = SimpleNamespace(ernie_per_row_loss=True, calculate_per_token_loss=True)
        with pytest.raises(ValueError, match="calculate_per_token_loss=False"):
            ernie55_loss_function(
                torch.ones(1, 4), config=config, cp_group=None, check_for_nan_in_loss=True, check_for_spiky_loss=False
            )

    def test_context_parallelism_without_per_row_loss_must_be_per_token(self, fake_group_sizes):
        config = SimpleNamespace(ernie_per_row_loss=False, calculate_per_token_loss=False)
        with pytest.raises(ValueError, match="per-row loss"):
            ernie55_loss_function(
                torch.ones(1, 4),
                config=config,
                cp_group=_FakeGroup(2),
                check_for_nan_in_loss=True,
                check_for_spiky_loss=False,
            )
        config.calculate_per_token_loss = True
        loss_function = ernie55_loss_function(
            torch.ones(1, 4),
            config=config,
            cp_group=_FakeGroup(2),
            check_for_nan_in_loss=True,
            check_for_spiky_loss=False,
        )
        assert loss_function.func is masked_next_token_loss


def _write_jsonl(path: Path, samples: list[tuple[list[int], list[int]]]) -> str:
    path.write_text("".join(json.dumps({"input_ids": ids, "labels": labels}) + "\n" for ids, labels in samples))
    return str(path)


class TestSFTData:
    def test_hugging_face_labels_are_shifted_per_sample(self, tmp_path):
        path = _write_jsonl(tmp_path / "data.jsonl", [([1, 2, 3, 4], [-100, -100, 3, 4])])
        config = Ernie55SFTDatasetConfig(seq_length=16, pad_token_id=0, path=path)
        config.finalize()
        train, valid, test = config.build_datasets(
            DatasetBuildContext(train_samples=2, valid_samples=0, test_samples=0)
        )
        assert valid is None and test is None and len(train) == 2
        # The target of position t is the HF label of t + 1; the last token has none.
        assert train[0] == [([1, 2, 3, 4], [-100, 3, 4, -100])]
        assert train[1] == train[0]

    def test_split_and_right_truncation(self, tmp_path):
        sample = (list(range(1, 11)), list(range(1, 11)))
        path = _write_jsonl(tmp_path / "data.jsonl", [sample])
        split = Ernie55SFTDatasetConfig(seq_length=4, pad_token_id=0, path=path, truncation="split")
        rows = split.build_datasets(DatasetBuildContext(train_samples=3, valid_samples=0, test_samples=0))[0]
        # Split before the shift: every chunk's last token has no target, as in the reference.
        assert [rows[index] for index in range(3)] == [
            [([1, 2, 3, 4], [2, 3, 4, -100])],
            [([5, 6, 7, 8], [6, 7, 8, -100])],
            [([9, 10], [10, -100])],
        ]
        right = Ernie55SFTDatasetConfig(seq_length=4, pad_token_id=0, path=path, truncation="right")
        rows = right.build_datasets(DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0))[0]
        assert len(rows) == 1 and rows[0] == [([1, 2, 3, 4], [2, 3, 4, -100])]

    def test_aligned_labels_are_kept(self, tmp_path):
        path = _write_jsonl(tmp_path / "data.jsonl", [([1, 2, 3], [2, 3, -100])])
        config = Ernie55SFTDatasetConfig(seq_length=8, pad_token_id=0, path=path, labels_are_shifted=True)
        train = config.build_datasets(DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0))[0]
        assert train[0] == [([1, 2, 3], [2, 3, -100])]

    def test_packing_fills_rows_with_padded_lengths(self, tmp_path):
        samples = [([1] * length, [1] * length) for length in (6, 5, 3, 2, 2)]
        path = _write_jsonl(tmp_path / "data.jsonl", samples)
        config = Ernie55SFTDatasetConfig(seq_length=8, pad_token_id=0, path=path, pack=True, pad_to_multiple_of=2)
        train = config.build_datasets(DatasetBuildContext(train_samples=1, valid_samples=0, test_samples=0))[0]
        rows = [[len(ids) for ids, _ in train[index]] for index in range(len(train))]
        # Padded to 2: 6, 6, 4, 2, 2 -> first fit longest first into rows of 8.
        assert sorted(sorted(row) for row in rows) == [[2, 5], [2, 6], [3]]
        for row in rows:
            assert sum(math.ceil(length / 2) * 2 for length in row) <= 8

    def test_synthetic_rows(self):
        config = Ernie55SFTDatasetConfig(
            seq_length=32, pad_token_id=0, mock_document_lengths=(10, 7), mock_prompt_fraction=0.5, seed=3
        )
        config.finalize()
        train = config.build_datasets(DatasetBuildContext(train_samples=4, valid_samples=0, test_samples=0))[0]
        row = train[1]
        assert [len(ids) for ids, _ in row] == [10, 7]
        ids, labels = row[0]
        assert labels[:4] == [IGNORE_LABEL] * 4 and labels[4:9] == ids[5:10] and labels[9] == IGNORE_LABEL
        assert train[1] == row and train[2] != row

    def test_validation_and_test_splits_read_their_own_files(self, tmp_path):
        train_path = _write_jsonl(tmp_path / "train.jsonl", [([1, 2, 3, 4], [1, 2, 3, 4])])
        valid_path = _write_jsonl(tmp_path / "valid.jsonl", [([5, 6, 7], [5, 6, 7])])
        test_path = _write_jsonl(tmp_path / "test.jsonl", [([8, 9], [8, 9])])
        config = Ernie55SFTDatasetConfig(
            seq_length=16, pad_token_id=0, path=train_path, valid_path=valid_path, test_path=test_path
        )
        config.finalize()
        train, valid, test = config.build_datasets(
            DatasetBuildContext(train_samples=1, valid_samples=1, test_samples=1)
        )
        assert train[0] == [([1, 2, 3, 4], [2, 3, 4, -100])]
        assert valid[0] == [([5, 6, 7], [6, 7, -100])]
        assert test[0] == [([8, 9], [9, -100])]
        # The split configurations are copies that only differ in the file they read.
        assert valid.config.path == valid_path and valid.config.seq_length == 16
        assert config.path == train_path and config.valid_path == valid_path

    def test_splits_default_to_the_training_file(self, tmp_path):
        path = _write_jsonl(tmp_path / "train.jsonl", [([1, 2, 3, 4], [1, 2, 3, 4])])
        config = Ernie55SFTDatasetConfig(seq_length=16, pad_token_id=0, path=path)
        config.finalize()
        train, valid, test = config.build_datasets(
            DatasetBuildContext(train_samples=1, valid_samples=1, test_samples=1)
        )
        assert valid.config is config and test.config is config
        assert valid[0] == train[0] == test[0]

    def test_the_two_held_out_files_are_independent(self, tmp_path):
        train_path = _write_jsonl(tmp_path / "train.jsonl", [([1, 2, 3, 4], [1, 2, 3, 4])])
        test_path = _write_jsonl(tmp_path / "test.jsonl", [([8, 9], [8, 9])])
        config = Ernie55SFTDatasetConfig(seq_length=16, pad_token_id=0, path=train_path, test_path=test_path)
        config.finalize()
        train, valid, test = config.build_datasets(
            DatasetBuildContext(train_samples=1, valid_samples=1, test_samples=1)
        )
        # Only test_path is set: the validation split still reads the training file, the test split its own file.
        assert valid.config is config and valid[0] == train[0]
        assert test.config is not config and test[0] == [([8, 9], [9, -100])]
        with pytest.raises(ValueError, match="set dataset.path too"):
            Ernie55SFTDatasetConfig(
                seq_length=16, pad_token_id=0, mock_document_lengths=(4,), test_path=test_path
            ).finalize()

    def test_held_out_files_need_a_training_file(self, tmp_path):
        valid_path = _write_jsonl(tmp_path / "valid.jsonl", [([5, 6, 7], [5, 6, 7])])
        with pytest.raises(ValueError, match="set dataset.path too"):
            Ernie55SFTDatasetConfig(
                seq_length=16, pad_token_id=0, mock_document_lengths=(4,), valid_path=valid_path
            ).finalize()

    def test_synthetic_rows_must_fit(self):
        with pytest.raises(ValueError, match="more than seq_length"):
            Ernie55SFTDatasetConfig(
                seq_length=16, pad_token_id=0, mock_document_lengths=(9, 7), pad_to_multiple_of=4
            ).finalize()

    def test_collate_feeds_the_step(self, tmp_path):
        config = Ernie55SFTDatasetConfig(seq_length=64, pad_token_id=0, mock_document_lengths=(9, 5))
        config.finalize()
        train = config.build_datasets(DatasetBuildContext(train_samples=2, valid_samples=0, test_samples=0))[0]
        collated = train.collate_fn([train[0], train[1]])
        assert collated["tokens"].shape == (1, 28)
        assert collated["cu_seqlens"].tolist() == [0, 9, 14, 23, 28]
        assert collated["pad_token_id"] == 0
        documents = ernie55_documents_from_batch(collated)
        assert [len(tokens) for tokens, _ in documents] == [9, 5, 9, 5]


def test_recipe_runner_registers_the_step():
    script = Path(__file__).resolve().parents[4] / "scripts" / "training" / "recipe_runner.py"
    tree = ast.parse(script.read_text())
    registry = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "STEP_FUNCTIONS"
    )
    entries = ast.literal_eval(registry)
    module_name, attribute = entries["ernie55_step"]
    assert getattr(__import__(module_name, fromlist=[attribute]), attribute) is ernie55_step.forward_step


class TestMetricConventions:
    def test_indexer_loss_divisor(self, monkeypatch):
        from megatron.core.transformer.experimental_attention_variant import dsa

        calls = []
        monkeypatch.setattr(
            dsa.DSAIndexerLossLoggingHelper,
            "track_indexer_metrics",
            staticmethod(lambda **kwargs: calls.append(kwargs)),
        )
        model_config = _proxy_model_config(dsa_indexer_loss_coeff=0.01, dsa_indexer_use_sparse_loss=True)
        config = SimpleNamespace(model=model_config)
        state = SimpleNamespace(global_batch_packing_num_microbatches=4)
        train_utils._track_dsa_indexer_metrics(config, state, 7, None, None, {}, SimpleNamespace(pp=None, dp=None))
        assert len(calls) == 1
        # (L_D2 + L_D4 + L_MTP) / (4 layers + 1 MTP depth), averaged over the 4 micro-batches.
        assert calls[0]["num_indexer_layers"] == 5
        assert calls[0]["loss_scale"] == pytest.approx(0.25)
        assert calls[0]["iteration"] == 7

    def test_no_indexer_loss_no_logging(self, monkeypatch):
        from megatron.core.transformer.experimental_attention_variant import dsa

        calls = []
        monkeypatch.setattr(
            dsa.DSAIndexerLossLoggingHelper,
            "track_indexer_metrics",
            staticmethod(lambda **kwargs: calls.append(kwargs)),
        )
        config = SimpleNamespace(model=_proxy_model_config())
        train_utils._track_dsa_indexer_metrics(config, SimpleNamespace(), 1, None, None, {}, None)
        assert not calls

    def test_other_models_keep_megatron_cores_divisor(self):
        assert train_utils._dsa_indexer_metric_num_layers(SimpleNamespace()) is None

    def test_indexer_tracker_initialization_is_megatron_cores(self, monkeypatch):
        """Setup sizes the tracker through Megatron-Core's initializer, after clearing an earlier model's tracker."""
        from megatron.core.transformer.experimental_attention_variant import dsa

        dsa_logging = pytest.importorskip(
            "megatron.core.transformer.experimental_attention_variant.dsa_logging",
            reason="Megatron-Core main has no fixed-capacity DSA tracker",
        )
        calls = []
        monkeypatch.setattr(dsa_logging, "initialize_dsa_metric_tracker", lambda *args: calls.append(args) or 7)
        dsa.DSAIndexerLossLoggingHelper.tracker["values"] = torch.ones(3)
        model, pg_collection = [object()], object()
        config = _proxy_model_config(dsa_indexer_loss_coeff=0.01, dsa_indexer_use_sparse_loss=True)
        assert train_utils.initialize_dsa_indexer_metric_tracker(config, model, pg_collection) == 7
        assert calls == [(model, pg_collection)]
        assert "values" not in dsa.DSAIndexerLossLoggingHelper.tracker
        # Without an indexer loss nothing is initialized.
        assert train_utils.initialize_dsa_indexer_metric_tracker(_proxy_model_config(), model, pg_collection) == 0
        assert len(calls) == 1

    @needs_hook
    @pytest.mark.parametrize(
        "overrides",
        [
            dict(hybrid_layer_pattern="D-DE/DE", num_layers=4, csa_compress_ratios=None),
            # The proxy's layout, H layers included (they have no indexer).
            {},
        ],
        ids=["d-only", "proxy-layout"],
    )
    def test_every_ernie_indexer_write_fits_the_fixed_capacity(self, pg_collection, overrides):
        """The capacity Megatron-Core derives from the model holds every ERNIE D layer's write, the MTP depth's
        included, and the tracker tensor is never rebound (a CUDA graph may hold its address)."""
        from megatron.core.transformer.experimental_attention_variant import dsa

        config = _tiny_config(dsa_indexer_loss_coeff=0.01, dsa_indexer_use_sparse_loss=True, **overrides)
        model = _build(pg_collection, config)
        _randomize_(model, seed=5)
        capacity = train_utils.initialize_dsa_indexer_metric_tracker(config, [model], pg_collection)
        tracker = dsa.DSAIndexerLossLoggingHelper.tracker
        if capacity == 0:
            # Megatron-Core main: the tracker stays lazily sized.
            assert "values" not in tracker
            return
        values = tracker["values"]
        assert values.shape == (capacity,)
        # SBHD rows long enough for the H layers' 128-token compression blocks.
        seq = 300
        tokens = torch.randint(1, 97, (1, seq), generator=torch.Generator().manual_seed(6))
        losses = model(
            input_ids=tokens,
            position_ids=torch.arange(seq).unsqueeze(0),
            attention_mask=None,
            labels=tokens.roll(-1, dims=1),
            loss_mask=torch.ones(1, seq),
        )
        losses.sum().backward()
        assert tracker["values"] is values
        # Both layouts have two D layers in the decoder and one in the MTP depth, each with its own slot.
        assert int(values.count_nonzero()) == 3

    def test_moe_metrics_divide_by_the_moe_layers_and_mtp(self):
        # The reference logs z-loss with num_layers=4, moe_layer_freq=[0, 1, 1, 1] and one MTP layer: 4 MoE layers.
        assert train_utils._get_num_moe_layers(_proxy_model_config()) == 4
