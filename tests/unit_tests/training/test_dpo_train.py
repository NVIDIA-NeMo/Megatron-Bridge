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

"""Unit tests for the DPO entry point's run-config guardrails and artifact validation."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from megatron.bridge.data.builders.dpo import DPODatasetConfig
from megatron.bridge.data.datasets.preference import ScoringFingerprint
from megatron.bridge.data.sources.hf import HFDatasetSourceConfig
from megatron.bridge.training.dpo_train import validate_dpo_run_config


SCORED_FINGERPRINT = ScoringFingerprint(
    dataset="org/some-preference-set",
    split="train",
    tokenizer="org/some-model",
    max_seq_length=64,
    prompt_key=None,
    tools_key=None,
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
        seq_length=64,
        ref_artifact=write_artifact_metadata(tmp_path / "train_ref"),
        source=HFDatasetSourceConfig(path_or_dataset="org/some-preference-set", split="train"),
    )
    return SimpleNamespace(
        dataset=dataset,
        tokenizer=SimpleNamespace(tokenizer_model="org/some-model"),
        model=SimpleNamespace(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=None,
            sequence_parallel=False,
        ),
        validation=SimpleNamespace(eval_iters=0, eval_interval=None),
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda c, _: None, None),
        (lambda c, _: setattr(c.dataset, "ref_artifact", None), "ref_artifact"),
        (lambda c, _: setattr(c, "dataset", SimpleNamespace()), "DPODatasetConfig"),
        (lambda c, _: setattr(c.validation, "eval_iters", 2), "validation split"),
        # The artifact was scored with other inputs than this run tokenizes with.
        (lambda c, tmp: write_artifact_metadata(tmp / "train_ref", max_seq_length=4096), "max_seq_length"),
        (
            lambda c, tmp: write_artifact_metadata(tmp / "train_ref", tokenization_version=None),
            "tokenization_version",
        ),
    ],
    ids=["valid", "no_artifact", "not_a_dpo_dataset", "no_val_split", "fingerprint", "old_artifact"],
)
def test_run_config_validation_names_the_offending_knob(tmp_path, mutate, match):
    config = make_config(tmp_path)
    mutate(config, tmp_path)
    if match is None:
        validate_dpo_run_config(config)
    else:
        with pytest.raises(ValueError, match=match):
            validate_dpo_run_config(config)
