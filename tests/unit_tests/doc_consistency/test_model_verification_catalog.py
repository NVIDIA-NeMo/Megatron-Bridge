# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_PATH = REPO_ROOT / "scripts/docs/generate_model_verification_catalog.py"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("model_verification_catalog", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _load_generator()


@pytest.fixture(scope="module")
def catalog(generator: ModuleType) -> dict[str, object]:
    return generator.build_catalog(REPO_ROOT)


def _models(catalog: dict[str, object]) -> dict[str, dict[str, object]]:
    return {model["slug"]: model for model in catalog["models"]}


def test_catalog_discovers_every_card_without_a_model_list(catalog: dict[str, object]) -> None:
    cards = sorted(REPO_ROOT.glob("examples/model_verification_cards/*/card.yaml"))
    models = _models(catalog)

    assert catalog["schema_version"] == 1
    assert set(models) == {card.parent.name for card in cards}
    assert {model["source_card"] for model in models.values()} == {
        card.relative_to(REPO_ROOT).as_posix() for card in cards
    }


@pytest.mark.parametrize(
    ("slug", "hf_id", "command_fragment", "metric_fragment"),
    [
        (
            "glm5-2",
            "zai-org/GLM-5.2",
            "glm52_pretrain_416gpu_h100_bf16_config",
            "last_10_steps_step_time_ms_avg",
        ),
        (
            "qwen3.8-27b",
            "Qwen/Qwen3.8-27B",
            "qwen35_vl_27b_pretrain_16gpu_gb200_bf16_mock_config",
            "last_10_steps_model_tflops_per_gpu_avg",
        ),
    ],
)
def test_pilot_models_render_end_to_end(
    generator: ModuleType,
    catalog: dict[str, object],
    slug: str,
    hf_id: str,
    command_fragment: str,
    metric_fragment: str,
) -> None:
    model = _models(catalog)[slug]
    page = generator.render_model_page(model, REPO_ROOT, fern=False)

    assert f"# {hf_id}" in page
    assert model["source_card"] in page
    assert model["hf_revision"] in page
    assert command_fragment in page
    assert metric_fragment in page
    assert "#### Exact command" in page
    assert "#### Expected result" in page


def test_current_heterogeneous_shapes_are_concrete_entries(catalog: dict[str, object]) -> None:
    models = _models(catalog)
    fsdp_entries = [
        entry for entry in models["nemotron-3.5-lightning"]["entries"] if entry["workflow"] == "pretrain_fsdp"
    ]
    weak_scaling_entries = [
        entry for entry in models["qwen3-30b-a3b"]["entries"] if entry["workflow"] == "pretrain_weak_scaling"
    ]

    assert {(entry["hardware"], entry["variant"], entry["precision"]) for entry in fsdp_entries} == {
        ("GB200", "bf16", "bf16"),
        ("GB200", "fp8_mx", "fp8_mx"),
    }
    assert [entry["dimensions"]["num_gpus"] for entry in weak_scaling_entries] == [8, 32, 128, 256]
    assert all(len(entry["commands"]) == 1 for entry in weak_scaling_entries)


def test_normalizer_never_generates_a_cartesian_product(generator: ModuleType, tmp_path: Path) -> None:
    card_path = tmp_path / "examples/model_verification_cards/fixture/card.yaml"
    card_path.parent.mkdir(parents=True)
    card_path.write_text(
        """\
title: fixture
summary: Exact-pair fixture.
verification_index: {}
model:
  hf_id: example/fixture
  hf_revision: deadbeef
  architecture: FixtureForCausalLM
  min_transformers_version: '1.0'
verification_environment:
  base_container: example:latest
  bridge_commit: cafe0000
items:
  pretrain:
    H100:
      status: verified
      precision: bf16
      command: run --nodes 1 --gpus-per-node 8 --precision bf16
      last_verified: 2026-08-24
      metrics: {loss: 1.0}
      expected_result: H100 BF16 succeeds.
    GB200:
      status: unverified
      precision: fp8_mx
      command: run --nodes 2 --gpus-per-node 4 --precision fp8_mx
      expected_result: GB200 FP8 is pending.
  export:
    all:
      status: unsupported
      expected_result: Export is unsupported.
  score:
    all:
      status: not_applicable
      expected_result: Scoring does not apply.
""",
        encoding="utf-8",
    )

    fixture = generator.build_catalog(tmp_path)
    entries = fixture["models"][0]["entries"]
    combinations = {(entry["workflow"], entry["hardware"], entry["precision"]) for entry in entries}

    assert combinations == {
        ("pretrain", "H100", "bf16"),
        ("pretrain", "GB200", "fp8_mx"),
        ("export", "all", None),
        ("score", "all", None),
    }
    assert ("pretrain", "H100", "fp8_mx") not in combinations
    assert ("pretrain", "GB200", "bf16") not in combinations
    assert {entry["status"] for entry in entries} == {
        "verified",
        "unverified",
        "unsupported",
        "not_applicable",
    }
    assert len({entry["source_pointer"] for entry in entries}) == len(entries)


def test_catalog_table_has_one_row_per_concrete_entry(generator: ModuleType, catalog: dict[str, object]) -> None:
    expected_count = sum(len(model["entries"]) for model in catalog["models"])
    page = generator.render_catalog_page(catalog, fern=False)

    assert len([line for line in page.splitlines() if line.startswith("| [")]) == expected_count
    assert "never combines independent fields into a synthetic command" in page
    assert "verification-catalog-filters" in page


def test_generated_outputs_and_navigation_are_current(generator: ModuleType, catalog: dict[str, object]) -> None:
    nav = (REPO_ROOT / "docs/fern/versions/nightly.yml").read_text(encoding="utf-8")

    assert generator.generate(REPO_ROOT, check=True)
    for model in catalog["models"]:
        assert f"model-verification/models/{model['slug']}.mdx" in nav


def test_normalized_commands_equal_card_scalars(catalog: dict[str, object]) -> None:
    for model in catalog["models"]:
        with (REPO_ROOT / model["source_card"]).open(encoding="utf-8") as stream:
            card = yaml.safe_load(stream)
        for entry in model["entries"]:
            node = card
            for part in entry["source_pointer"].split("."):
                if "[" in part:
                    key, index = part[:-1].split("[")
                    node = node[key][int(index)]
                else:
                    node = node[part]
            source_commands = node.get("commands")
            if source_commands is None and node.get("command") is not None:
                source_commands = [node["command"]]
            if source_commands is None:
                source_commands = []
            assert entry["commands"] == [command.strip() for command in source_commands]
