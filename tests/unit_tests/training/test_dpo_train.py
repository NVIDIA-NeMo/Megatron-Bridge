"""Unit tests for the DPO entry point's run-config guardrails and artifact validation."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import megatron.bridge.training.dpo_train as dpo_train_module
from megatron.bridge.data.builders.dpo import DPODatasetConfig
from megatron.bridge.data.datasets.preference import ScoringFingerprint
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from megatron.bridge.training.dpo import DPOLossConfig
from megatron.bridge.training.dpo_train import dpo_train, expected_scoring_metadata, validate_dpo_run_config


SCORED_FINGERPRINT = ScoringFingerprint(
    dataset="org/some-preference-set",
    split="train",
    tokenizer="org/some-model",
    max_seq_length=64,
    prompt_key=None,
    tensor_model_parallel_size=1,
    sequence_parallel=False,
)


def write_artifact_metadata(artifact_dir, **overrides) -> str:
    metadata = asdict(SCORED_FINGERPRINT)
    metadata.update(overrides)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "scoring_metadata.json").write_text(json.dumps(metadata))
    return str(artifact_dir)


def make_config(tmp_path) -> SimpleNamespace:
    dataset = DPODatasetConfig(
        tokenizer_name="org/some-model",
        seq_length=64,
        ref_artifact=write_artifact_metadata(tmp_path / "train_ref"),
        source=HFDatasetSourceConfig(path_or_dataset="org/some-preference-set", split="train"),
    )
    return SimpleNamespace(
        dpo=DPOLossConfig(),
        dataset=dataset,
        model=SimpleNamespace(
            calculate_per_token_loss=True,
            mtp_num_layers=None,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=None,
            context_parallel_size=1,
            sequence_parallel=False,
        ),
        ddp=SimpleNamespace(average_in_collective=False),
        train=SimpleNamespace(micro_batch_size=4, global_batch_size=8),
        validation=SimpleNamespace(
            eval_iters=0, eval_interval=None, eval_global_batch_size=None, eval_micro_batch_size=None
        ),
        dist=SimpleNamespace(eval_context_parallel_size=None),
    )


def add_validation_split(config, tmp_path, **metadata_overrides) -> SimpleNamespace:
    config.dataset.validation_source = HFDatasetSourceConfig(
        path_or_dataset="org/some-preference-set", split="validation"
    )
    config.dataset.validation_ref_artifact = write_artifact_metadata(
        tmp_path / "validation_ref", split="validation", **metadata_overrides
    )
    config.validation.eval_iters = 2
    config.validation.eval_interval = 10
    return config


def test_valid_configs_pass(tmp_path):
    validate_dpo_run_config(make_config(tmp_path))
    validate_dpo_run_config(add_validation_split(make_config(tmp_path), tmp_path))


def with_validation(mutate):
    def apply(config, tmp_path):
        add_validation_split(config, tmp_path)
        mutate(config)

    return apply


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda c, _: setattr(c, "dpo", None), "config.dpo"),
        (lambda c, _: setattr(c, "dataset", SimpleNamespace()), "DPODatasetConfig"),
        (lambda c, _: setattr(c.dataset, "ref_artifact", None), "ref_artifact"),
        (lambda c, _: setattr(c.model, "calculate_per_token_loss", False), "calculate_per_token_loss"),
        (lambda c, _: setattr(c.ddp, "average_in_collective", True), "average_in_collective"),
        (lambda c, _: setattr(c.model, "context_parallel_size", 2), "context"),
        (lambda c, _: setattr(c.model, "mtp_num_layers", 1), "mtp_num_layers"),
        (lambda c, _: setattr(c.train, "micro_batch_size", 3), "even"),
        (lambda c, _: setattr(c.train, "global_batch_size", 7), "even"),
        (lambda c, _: setattr(c.validation, "eval_iters", 2), "validation split"),
        (with_validation(lambda c: setattr(c.validation, "eval_interval", None)), "eval_interval"),
        (with_validation(lambda c: setattr(c.validation, "eval_global_batch_size", 7)), "eval_global_batch_size"),
        (with_validation(lambda c: setattr(c.validation, "eval_micro_batch_size", 7)), "eval_micro_batch_size"),
        (with_validation(lambda c: setattr(c.dist, "eval_context_parallel_size", 2)), "eval_context_parallel_size"),
    ],
)
def test_invalid_config_raises_naming_the_knob(tmp_path, mutate, match):
    config = make_config(tmp_path)
    mutate(config, tmp_path)
    with pytest.raises(ValueError, match=match):
        validate_dpo_run_config(config)


def test_all_violations_reported_at_once(tmp_path):
    config = make_config(tmp_path)
    config.model.calculate_per_token_loss = False
    config.ddp.average_in_collective = True
    with pytest.raises(ValueError) as exc_info:
        validate_dpo_run_config(config)
    assert "calculate_per_token_loss" in str(exc_info.value)
    assert "average_in_collective" in str(exc_info.value)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_artifact_metadata_mismatch_fails_validation(tmp_path, split):
    config = add_validation_split(make_config(tmp_path), tmp_path)
    write_artifact_metadata(tmp_path / f"{split}_ref", split=split, max_seq_length=4096)
    with pytest.raises(ValueError, match="max_seq_length"):
        validate_dpo_run_config(config)


def test_validation_artifact_is_not_checked_when_eval_is_disabled(tmp_path):
    """SFT behavior: eval_iters=0 turns validation off without touching the dataset config."""
    config = add_validation_split(make_config(tmp_path), tmp_path, max_seq_length=4096)
    config.validation.eval_iters = 0
    validate_dpo_run_config(config)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_expected_scoring_metadata_reads_the_split_source_and_model_config(tmp_path, split):
    config = add_validation_split(make_config(tmp_path), tmp_path)
    assert expected_scoring_metadata(config, split) == ScoringFingerprint(
        **{**asdict(SCORED_FINGERPRINT), "split": split}
    )


def test_expected_scoring_metadata_routes_jsonl_paths_like_the_loader(tmp_path):
    config = make_config(tmp_path)
    config.dataset.source = HFDatasetSourceConfig(path_or_dataset="/data/pairs.jsonl", split="train")
    fingerprint = expected_scoring_metadata(config, "train")
    assert (fingerprint.dataset, fingerprint.split) == ("/data/pairs.jsonl", None)


def test_expert_layout_mismatch_warns_but_passes(tmp_path, caplog):
    config = make_config(tmp_path)
    config.model.expert_model_parallel_size = 8
    config.model.expert_tensor_parallel_size = 1
    with caplog.at_level("WARNING", logger="megatron.bridge.training.dpo_train"):
        validate_dpo_run_config(config)
    assert "Expert layout" in caplog.text
    assert "(8, 1)" in caplog.text


def test_matching_expert_layout_does_not_warn(tmp_path, caplog):
    config = make_config(tmp_path)
    config.model.expert_model_parallel_size = 8
    config.model.expert_tensor_parallel_size = 1
    write_artifact_metadata(tmp_path / "train_ref", expert_model_parallel_size=8, expert_tensor_parallel_size=1)
    with caplog.at_level("WARNING", logger="megatron.bridge.training.dpo_train"):
        validate_dpo_run_config(config)
    assert "Expert layout" not in caplog.text


def test_artifact_without_expert_keys_implies_scorer_layout(tmp_path, caplog):
    """Pre-MoE artifacts carry no expert keys: EP defaults to 1 and ETP to the scorer's TP."""
    config = make_config(tmp_path)
    config.model.tensor_model_parallel_size = 2
    config.model.expert_tensor_parallel_size = 2
    write_artifact_metadata(tmp_path / "train_ref", tensor_model_parallel_size=2)
    with caplog.at_level("WARNING", logger="megatron.bridge.training.dpo_train"):
        validate_dpo_run_config(config)
    assert "Expert layout" not in caplog.text


def step_stub(state, data_iterator, model):
    raise AssertionError("the forward step is never called in these tests")


def test_dpo_train_validates_then_hands_off_to_finetune(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dpo_train_module, "finetune", lambda *args, **kwargs: calls.append((args, kwargs)))

    config = make_config(tmp_path)
    user_callback = object()
    dpo_train(config, step_stub, callbacks=[user_callback])
    assert calls == [((config, step_stub), {"callbacks": [user_callback]})]

    config.validation.eval_iters = 5  # invalid: no validation split
    with pytest.raises(ValueError, match="validation split"):
        dpo_train(config, step_stub)
    assert len(calls) == 1
