"""Unit tests for DPODatasetConfig (the ConfigContainer.dataset entry for DPO)."""

import json

import pytest
import torch

from megatron.bridge.data.builders.dpo import (
    DPODatasetBuilder,
    DPODatasetConfig,
    build_preference_split,
    dpo_train_valid_test_datasets_provider,
)
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from megatron.bridge.data.sources.jsonl import JSONLSourceConfig
from tests.unit_tests.data.preference_fakes import FakeChatTokenizer


class FakeSource:
    """datasets.Dataset stand-in: sized, indexable, selectable."""

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def select(self, indices):
        return FakeSource([self.rows[i] for i in indices])


def chat_row(pair_id: int) -> dict:
    conversation = [
        {"role": "user", "content": f"prompt {pair_id}"},
        {"role": "assistant", "content": "an answer"},
    ]
    worse = conversation[:-1] + [{"role": "assistant", "content": "worse"}]
    return {"chosen": conversation, "rejected": worse}


def ref_mapping(num_pairs: int) -> dict:
    return {
        pair_id: {
            "ref_chosen_logprob_sum": -1.0,
            "ref_chosen_num_tokens": 3,
            "ref_rejected_logprob_sum": -2.0,
            "ref_rejected_num_tokens": 3,
        }
        for pair_id in range(num_pairs)
    }


def make_config(**overrides) -> DPODatasetConfig:
    fields = dict(
        tokenizer_name="org/some-model",
        seq_length=64,
        ref_artifact="/tmp/ref_artifact",
        source=HFDatasetSourceConfig(path_or_dataset="org/some-preference-set", split="train"),
    )
    fields.update(overrides)
    return DPODatasetConfig(**fields)


def make_validation_config(**overrides) -> DPODatasetConfig:
    fields = dict(
        validation_ref_artifact="/tmp/validation_ref_artifact",
        validation_source=HFDatasetSourceConfig(path_or_dataset="org/some-preference-set", split="validation"),
    )
    fields.update(overrides)
    return make_config(**fields)


def test_dataloader_type_is_pinned_to_batch():
    config = make_config()
    config.finalize()
    assert config.dataloader_type == "batch"
    with pytest.raises(ValueError, match="dataloader_type"):
        make_config(dataloader_type="single").finalize()


def test_source_must_be_a_supported_source_type():
    """The source type selects the mode, so anything else fails loudly."""
    with pytest.raises(TypeError, match="HFDatasetSourceConfig or JSONLSourceConfig"):
        make_config(source=None).finalize()
    with pytest.raises(TypeError, match="HFDatasetSourceConfig or JSONLSourceConfig"):
        make_config(source=["/tmp/pairs.jsonl"]).finalize()


def test_mapping_shaped_sources_are_rebuilt_after_an_override_round_trip():
    """OmegaConf overrides hand nested dataclasses back as plain mappings; validate must rebuild them."""
    config = make_config(
        source={"path_or_dataset": "org/some-preference-set", "split": "train"},
        validation_source={"paths": ["/tmp/v.jsonl"]},
        validation_ref_artifact="/tmp/validation_ref_artifact",
    )
    config.finalize()
    assert isinstance(config.source, HFDatasetSourceConfig)
    assert isinstance(config.validation_source, JSONLSourceConfig)
    assert config.source_identity(config.source) == ("org/some-preference-set", "train")
    assert config.source_identity(config.validation_source) == ("/tmp/v.jsonl", None)


def test_seq_length_must_be_positive():
    with pytest.raises(ValueError, match="seq_length"):
        make_config(seq_length=0).finalize()


def test_jsonl_sources_reject_formats_the_memmap_reader_cannot_parse():
    config = make_config(source=JSONLSourceConfig(paths=["/tmp/pairs.parquet"]))
    with pytest.raises(ValueError, match="accepts only"):
        config.finalize()


def test_remote_jsonl_requires_an_index_mapping_dir():
    """The .idx sidecars cannot be written beside data in a read-only bucket."""
    config = make_config(source=JSONLSourceConfig(paths=["msc://bucket/pairs.jsonl"]))
    with pytest.raises(ValueError, match="index_mapping_dir"):
        config.finalize()

    ok = make_config(source=JSONLSourceConfig(paths=["msc://bucket/pairs.jsonl"], index_mapping_dir="/tmp/idx"))
    ok.finalize()


def test_source_identity_covers_each_source_mode():
    """These land in scoring_metadata.json, so both modes need a stable identity."""
    hf = make_config()
    assert hf.source_identity(hf.source) == ("org/some-preference-set", "train")

    jsonl = make_config(source=JSONLSourceConfig(paths=["/tmp/a.jsonl", "/tmp/b.jsonl"]))
    assert jsonl.source_identity(jsonl.source) == ("/tmp/a.jsonl,/tmp/b.jsonl", None)


def test_validate_resolves_a_jsonl_path_on_the_hf_source_in_place(tmp_path):
    """After validation the config holds one source type, and its identity has no split."""
    config = make_config(
        source=HFDatasetSourceConfig(path_or_dataset="/tmp/pairs.jsonl", split="train"),
        index_mapping_dir=str(tmp_path / "index"),
    )
    config.finalize()
    assert isinstance(config.source, JSONLSourceConfig)
    assert config.source.index_mapping_dir == str(tmp_path / "index")
    assert config.source_identity(config.source) == ("/tmp/pairs.jsonl", None)


def test_load_tokenizer_routes_trust_remote_code_through_the_repo_guard(monkeypatch):
    import megatron.bridge.data.builders.dpo as dpo_module

    seen = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            seen.update(name=name, **kwargs)
            return FakeChatTokenizer()

    monkeypatch.setattr(dpo_module, "AutoTokenizer", FakeAutoTokenizer)
    make_config().load_tokenizer()
    assert seen == {"name": "org/some-model", "trust_remote_code": False}
    make_config(trust_remote_code=True).load_tokenizer()
    assert seen["trust_remote_code"] is True


def test_build_preference_split_without_an_artifact_is_the_scorer_shape(monkeypatch):
    """The scorer builds through the same helper as the trainer, so token streams match by construction."""
    monkeypatch.setattr(DPODatasetConfig, "load_source", lambda self, split="train": [chat_row(0), chat_row(1)])
    config = make_config(pad_seq_length_to_mult=4, prompt_key=None)
    dataset = build_preference_split(config, "train", FakeChatTokenizer())
    assert len(dataset) == 2
    assert dataset.require_ref_logprobs is False
    assert dataset.max_seq_length == config.seq_length
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert batch["tokens"].shape[1] % 4 == 0
    assert "ref_logprob_sum" not in batch


def test_validation_split_requires_its_own_artifact():
    config = make_validation_config(validation_ref_artifact=None)
    with pytest.raises(ValueError, match="validation_ref_artifact"):
        config.finalize()


def test_validation_artifact_without_a_validation_source_is_rejected():
    config = make_config(validation_ref_artifact="/tmp/validation_ref_artifact")
    with pytest.raises(ValueError, match="set validation_source"):
        config.finalize()


def test_load_source_validation_split_requires_a_validation_source():
    with pytest.raises(ValueError, match="No validation split"):
        make_config().load_source("validation")


def test_num_pairs_truncates_by_position_and_zero_or_oversized_is_a_no_op():
    rows = [{"i": i} for i in range(5)]
    import megatron.bridge.data.builders.dpo as dpo_module

    seen = []
    for num_pairs, expected in [(0, 5), (9, 5), (2, 2)]:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(dpo_module, "prepare_hf_dataset_sources", lambda sources: seen.append(sources))
            mp.setattr(dpo_module, "load_hf_dataset_source", lambda source: rows)
            out = DPODatasetConfig._load_rows(HFDatasetSourceConfig(path_or_dataset="org/x", split="train"), num_pairs)
        assert len(out) == expected
        assert [out[i]["i"] for i in range(len(out))] == list(range(expected))


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_a_jsonl_path_on_the_hf_source_routes_to_the_memmap_reader(tmp_path):
    """The launcher channel: run_recipe can only override existing scalar fields, so a
    JSONL path arrives as ``source.path_or_dataset`` and must still reach the memmap reader."""
    rows = [chat_row(0), chat_row(1)]
    write_jsonl(tmp_path / "pairs.jsonl", rows)

    config = make_config(
        source=HFDatasetSourceConfig(path_or_dataset=str(tmp_path / "pairs.jsonl"), split="train"),
        index_mapping_dir=str(tmp_path / "index"),
    )
    source = config.load_source()

    assert [dict(source[i]) for i in range(len(source))] == rows
    assert (tmp_path / "index").is_dir()


def test_a_hub_id_on_the_hf_source_still_loads_through_hf_datasets(monkeypatch, tmp_path):
    """Only path-shaped sources switch readers; a hub id must keep the HF loader."""
    seen = []
    monkeypatch.setattr(
        "megatron.bridge.data.builders.dpo.prepare_hf_dataset_sources",
        lambda sources: seen.append(sources),
    )
    monkeypatch.setattr(
        "megatron.bridge.data.builders.dpo.load_hf_dataset_source",
        lambda source: FakeSource([chat_row(0)]),
    )
    source = make_config().load_source()

    assert len(source) == 1
    assert seen and seen[0][0].path_or_dataset == "org/some-preference-set"


def test_load_source_concatenates_files_in_order_and_truncates_to_num_pairs(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_jsonl(data_dir / "a.jsonl", [{"x": 0}])
    write_jsonl(data_dir / "b.jsonl", [{"x": 1}, {"x": 2}])

    config = make_config(
        source=JSONLSourceConfig(
            paths=[str(data_dir / "a.jsonl"), str(data_dir / "b.jsonl")],
            index_mapping_dir=str(tmp_path / "index"),
        ),
        num_pairs=2,
    )
    source = config.load_source()

    assert [source[i]["x"] for i in range(len(source))] == [0, 1]


@pytest.fixture
def patched_build_deps(monkeypatch):
    """Route source loading, artifact loading, and the HF tokenizer to fakes."""
    source = FakeSource([chat_row(i) for i in range(6)])
    validation_source = FakeSource([chat_row(i) for i in range(4)])
    loaded_artifacts = []

    def fake_load_source(self, split="train"):
        rows, num_pairs = (
            (validation_source, self.validation_num_pairs) if split == "validation" else (source, self.num_pairs)
        )
        return rows.select(range(num_pairs)) if num_pairs else rows

    def fake_load_ref_logprobs(path, expected_num_pairs):
        loaded_artifacts.append((path, expected_num_pairs))
        return ref_mapping(expected_num_pairs)

    import megatron.bridge.data.builders.dpo as provider_module

    fake_auto_tokenizer = type(
        "AutoTokenizer", (), {"from_pretrained": staticmethod(lambda name, **kwargs: FakeChatTokenizer())}
    )
    monkeypatch.setattr(provider_module, "AutoTokenizer", fake_auto_tokenizer)
    monkeypatch.setattr(DPODatasetConfig, "load_source", fake_load_source)
    monkeypatch.setattr(provider_module, "load_ref_logprobs", fake_load_ref_logprobs)
    return loaded_artifacts


def build_datasets(config: DPODatasetConfig, valid_samples: int = 0):
    """Invoke the registry provider the way ``build_train_valid_test_datasets`` does."""
    return dpo_train_valid_test_datasets_provider([0, valid_samples, 0], config)


def test_provider_requires_a_ref_artifact():
    with pytest.raises(ValueError, match="ref_artifact"):
        build_datasets(make_config(ref_artifact=None))


def test_builder_validates_at_construction():
    """Bad configs fail when the builder is created, before any data is touched."""
    with pytest.raises(ValueError, match="ref_artifact"):
        DPODatasetBuilder(make_config(ref_artifact=None))
    with pytest.raises(TypeError, match="HFDatasetSourceConfig or JSONLSourceConfig"):
        DPODatasetBuilder(make_config(source=None))


def test_provider_returns_train_only_without_a_validation_split(patched_build_deps):
    train_ds, valid_ds, test_ds = build_datasets(make_config(), valid_samples=8)
    assert valid_ds is None and test_ds is None
    assert len(train_ds) == 6
    assert train_ds.require_ref_logprobs
    assert patched_build_deps == [("/tmp/ref_artifact", 6)]


def test_provider_builds_the_validation_dataset_from_its_own_artifact(patched_build_deps):
    train_ds, valid_ds, test_ds = build_datasets(make_validation_config(), valid_samples=8)
    assert test_ds is None
    assert len(train_ds) == 6
    assert len(valid_ds) == 4
    assert valid_ds.require_ref_logprobs
    assert patched_build_deps == [("/tmp/ref_artifact", 6), ("/tmp/validation_ref_artifact", 4)]


def test_provider_skips_the_validation_split_when_the_run_never_validates(patched_build_deps):
    """SFT-builder behavior: a configured validation split with valid_samples == 0 is not built."""
    _, valid_ds, _ = build_datasets(make_validation_config(), valid_samples=0)
    assert valid_ds is None
    assert patched_build_deps == [("/tmp/ref_artifact", 6)]


def test_provider_selects_validation_num_pairs(patched_build_deps):
    _, valid_ds, _ = build_datasets(make_validation_config(validation_num_pairs=2), valid_samples=8)
    assert len(valid_ds) == 2
    assert patched_build_deps == [("/tmp/ref_artifact", 6), ("/tmp/validation_ref_artifact", 2)]


def test_registry_resolves_dpo_config_to_the_provider():
    pytest.importorskip("megatron.core.datasets.blended_megatron_dataset_builder")
    from megatron.bridge.data.utils import get_dataset_provider

    assert get_dataset_provider(make_config()) is dpo_train_valid_test_datasets_provider


def test_built_dataset_collates_dpo_batch_keys(patched_build_deps):
    train_ds, _, _ = build_datasets(make_config())
    batch = train_ds.collate_fn([train_ds[0], train_ds[1]])
    for key in ("tokens", "labels", "loss_mask", "pair_id", "loss_multiplier", "ref_logprob_sum", "ref_num_tokens"):
        assert key in batch, key
    assert batch["tokens"].shape[0] == 4  # 2 pairs -> 4 interleaved rows


def explicit_chat_row(pair_id: int) -> dict:
    """``chat_row`` in TRL's explicit-prompt layout: shared prompt, completion-only sides."""
    return {
        "messages": [{"role": "user", "content": f"prompt {pair_id}"}],
        "chosen": [{"role": "assistant", "content": "an answer"}],
        "rejected": [{"role": "assistant", "content": "worse"}],
    }


def test_explicit_prompt_rows_build_the_same_batch_as_implicit_rows(monkeypatch, patched_build_deps):
    """Config plumbing proven end-to-end: the provider's batch is format-independent."""
    implicit_batch = _collated_first_batch(make_config())

    monkeypatch.setattr(
        DPODatasetConfig,
        "load_source",
        lambda self, split="train": FakeSource([explicit_chat_row(i) for i in range(6)]),
    )
    explicit_batch = _collated_first_batch(make_config(prompt_key="messages"))

    assert implicit_batch.keys() == explicit_batch.keys()
    for key, value in implicit_batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, explicit_batch[key]), key
        else:
            assert value == explicit_batch[key], key


def _collated_first_batch(config: DPODatasetConfig):
    train_ds, _, _ = build_datasets(config)
    return train_ds.collate_fn([train_ds[0], train_ds[1]])
