"""Unit tests for the DPO entry point's run-config guardrails and artifact validation."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from megatron.bridge.data.builders.dpo import DPODatasetConfig
from megatron.bridge.data.datasets.preference import ScoringFingerprint
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from megatron.bridge.training.dpo import DPOLossConfig
from megatron.bridge.training.dpo_train import validate_dpo_run_config


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
    metadata = {**asdict(SCORED_FINGERPRINT), **overrides}
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


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda c, _: None, None),
        (lambda c, _: setattr(c.dataset, "ref_artifact", None), "ref_artifact"),
        (lambda c, _: setattr(c.model, "calculate_per_token_loss", False), "calculate_per_token_loss"),
        (lambda c, _: setattr(c.ddp, "average_in_collective", True), "average_in_collective"),
        (lambda c, _: setattr(c.model, "context_parallel_size", 2), "context"),
        (lambda c, _: setattr(c.train, "micro_batch_size", 3), "even"),
        (lambda c, _: setattr(c.validation, "eval_iters", 2), "validation split"),
        # The artifact was scored with other inputs than this run tokenizes with.
        (lambda c, tmp: write_artifact_metadata(tmp / "train_ref", max_seq_length=4096), "max_seq_length"),
    ],
    ids=[
        "valid",
        "no_artifact",
        "per_token_loss",
        "average_in_collective",
        "cp",
        "odd_mbs",
        "no_val_split",
        "fingerprint",
    ],
)
def test_run_config_validation_names_the_offending_knob(tmp_path, mutate, match):
    config = make_config(tmp_path)
    mutate(config, tmp_path)
    if match is None:
        validate_dpo_run_config(config)
    else:
        with pytest.raises(ValueError, match=match):
            validate_dpo_run_config(config)
