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

import html
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
    ("slug", "command_fragment"),
    [
        (
            "glm5-2",
            "glm52_pretrain_416gpu_h100_bf16_config",
        ),
        (
            "qwen3.8-27b",
            "qwen35_vl_27b_pretrain_16gpu_gb200_bf16_mock_config",
        ),
    ],
)
def test_pilot_models_render_end_to_end(
    generator: ModuleType,
    catalog: dict[str, object],
    slug: str,
    command_fragment: str,
) -> None:
    model = _models(catalog)[slug]
    page = generator.render_model_section([model])

    assert f'id="verified-{slug}"' in page
    assert command_fragment in page
    assert 'class="verification-model-explorer"' in page
    assert "Exact command" in page
    assert "Expected result" in page
    assert "Full verification records" not in page


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


def test_catalog_is_a_simple_model_directory(generator: ModuleType, catalog: dict[str, object]) -> None:
    page = generator.render_supported_models_page(catalog, REPO_ROOT, fern=False)

    assert page.count('class="verification-model-link"') == len(catalog["models"])
    assert "never combined into synthetic commands" in page
    assert 'href="deepseek/deepseek-v3.html#verified-deepseek-v3"' in page
    assert "NVIDIA-Nemotron-3-Super-120B-A12B-BF16</strong> <!-- pragma: allowlist secret -->" in page
    assert "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16</strong> <!-- pragma: allowlist secret -->" in page
    assert "# Supported Models" in page
    assert "data-model-explorer" not in page
    assert "verification-catalog-filters" not in page


def test_sphinx_model_page_renders_focused_combinations(generator: ModuleType, catalog: dict[str, object]) -> None:
    model = _models(catalog)["qwen3-30b-a3b"]
    page = generator.render_model_section([model])
    capability_workflows = {
        "import-export": {
            "hf_to_megatron_cpu",
            "hf_to_megatron_gpu",
            "megatron_to_hf_cpu",
            "megatron_to_hf_gpu",
        },
        "pretrain": {"pretrain", "pretrain_fsdp", "pretrain_weak_scaling"},
        "benchmark": {"pretrain_performance"},
        "sft": {"sft"},
        "lora": {"peft"},
        "long-context": {"sft_long_context"},
    }
    expected_entries = [
        entry for entry in model["entries"] if entry["workflow"] in set().union(*capability_workflows.values())
    ]

    assert page.count('data-capability-tab="') == len(capability_workflows)
    assert page.count('class="verification-combination"') == len(expected_entries)
    assert page.count('class="verification-model-detail"') == len(expected_entries)
    assert page.count('class="verification-filter-row"') == 1
    assert model["summary"] not in page
    assert "source card" not in page
    assert "Verification:" not in page
    assert f"### {model['hf_id']}" not in page
    assert set(workflow for _, _, workflows in generator.CAPABILITY_TABS for workflow in workflows) == set().union(
        *capability_workflows.values()
    )
    for precision in ("bf16", "fp8_mx", "nvfp4"):
        assert f'data-precision="{precision}"' in page
    hardware_values = {
        str(entry["hardware"]) for entry in expected_entries if entry.get("hardware") not in (None, "all")
    }
    for hardware in hardware_values:
        assert f'<button type="button" data-hardware="{hardware}">{hardware}</button>' in page
    for entry in expected_entries:
        assert page.count(f'id="{entry["entry_id"]}"') == 1
        assert f'data-entry="{entry["entry_id"]}"' in page
        if entry["workflow"] == "pretrain_performance":
            button = next(line for line in page.splitlines() if f'data-entry="{entry["entry_id"]}"' in line)
            assert 'data-capability="benchmark"' in button
        expected_hardware = "" if entry.get("hardware") in (None, "all") else entry["hardware"]
        button = next(line for line in page.splitlines() if f'data-entry="{entry["entry_id"]}"' in line)
        assert f'data-hardware="{expected_hardware}"' in button
        for command in entry["commands"]:
            assert html.escape(command) in page
        assert html.escape(entry["expected_result"]) in page


def test_full_verification_records_are_removed_from_every_model_page(
    generator: ModuleType, catalog: dict[str, object]
) -> None:
    for model in catalog["models"]:
        page = generator.render_model_section([model])
        assert "Full verification records" not in page
        assert 'class="verification-model-explorer"' in page


def test_recorded_training_metrics_are_rendered(generator: ModuleType, catalog: dict[str, object]) -> None:
    model = _models(catalog)["gpt-oss-120b"]
    page = generator.render_model_section([model])

    assert "Recorded metrics" in page
    assert 'class="verification-metric-list"' in page
    assert "verification-metric-grid" not in page
    assert "Initial loss" in page
    assert "Final loss" in page
    assert "Step time · last 10 avg" in page
    assert "2,481.140 ms" in page
    assert "Model throughput · last 10 avg" in page
    assert "1,077.350 TFLOP/s/GPU" in page
    assert "Token throughput · last 10 avg" in page
    assert "33,017.081 tokens/s/GPU" in page


def test_models_map_to_canonical_guides(generator: ModuleType, catalog: dict[str, object]) -> None:
    models = _models(catalog)
    paths = {slug: generator._find_model_doc(REPO_ROOT, model) for slug, model in models.items()}

    assert all(path != "models/README.md" for path in paths.values())
    assert len(set(paths.values())) == len(models)
    assert paths["gpt-oss-20b"] == "models/gpt_oss/gpt-oss-20b.md"
    assert paths["gpt-oss-120b"] == "models/gpt_oss/gpt-oss-120b.md"
    assert paths["muse-glimmer-30b"] == "models/muse_glimmer/muse-glimmer.md"
    assert paths["nemotron-3-nano-4b"] == "models/nemotron/nemotron3-nano-4b.md"
    assert paths["nemotron-3-super-120b-a12b"] == "models/nemotron/nemotron3-super.md"
    assert paths["nemotron-3-ultra-550b-a55b"] == "models/nemotron/nemotron3-ultra.md"
    assert paths["qwen3.8-27b"] == "models/qwen/qwen3.8-27b.md"


def test_model_page_merge_preserves_intro_and_replaces_old_sections(generator: ModuleType) -> None:
    original = "# Example\n\nHandwritten summary.\n\n## Architecture\n\nDetails.\n"
    section = f"{generator.MODEL_SECTION_START}\n\n## Verified configurations\n\n{generator.MODEL_SECTION_END}\n"
    merged = generator._merge_model_page(original, section)

    assert merged.startswith("# Example\n\nHandwritten summary.\n\n" + generator.MODEL_SECTION_START)
    assert "## Architecture" not in merged
    assert "Details." not in merged
    assert generator._merge_model_page(merged, section) == merged
    assert generator._merge_model_page(original, section, title="New title").startswith("# New title\n\n")


def test_generated_outputs_and_navigation_are_current(generator: ModuleType, catalog: dict[str, object]) -> None:
    nav = (REPO_ROOT / "docs/fern/versions/nightly.yml").read_text(encoding="utf-8")
    supported_models = (REPO_ROOT / "docs/models/README.md").read_text(encoding="utf-8")

    assert generator.generate(REPO_ROOT, check=True)
    for model in catalog["models"]:
        model_doc = generator._find_model_doc(REPO_ROOT, model)
        page = (REPO_ROOT / "docs" / model_doc).read_text(encoding="utf-8")
        assert f'id="verified-{model["slug"]}"' in page
        assert page.count(generator.MODEL_SECTION_START) == 1
        assert model["summary"] not in page
        assert f"model-verification/models/{model['slug']}.mdx" not in nav
    deepseek_page = (REPO_ROOT / "docs/models/deepseek/deepseek-v3.md").read_text(encoding="utf-8")
    assert "is a large-scale Mixture-of-Experts" in deepseek_page
    assert "## Conversion with" not in deepseek_page
    assert "pretrain_performance.H100" not in deepseek_page
    assert 'href="deepseek/deepseek-v3.html#verified-deepseek-v3"' in supported_models
    assert 'href="gpt_oss/gpt-oss-20b.html#verified-gpt-oss-20b"' in supported_models
    assert 'href="gpt_oss/gpt-oss-120b.html#verified-gpt-oss-120b"' in supported_models
    assert "./nightly/pages/models/gpt_oss/gpt-oss-20b.mdx" in nav
    assert "./nightly/pages/models/gpt_oss/gpt-oss-120b.mdx" in nav
    assert "./nightly/pages/models/nemotron/nemotron3-ultra.mdx" in nav
    for retired_doc in generator.RETIRED_COMBINED_MODEL_DOCS:
        assert not (REPO_ROOT / "docs" / retired_doc).exists()
        assert not (REPO_ROOT / "docs/fern/versions/nightly/pages" / Path(retired_doc).with_suffix(".mdx")).exists()
    assert "Model Verification Catalog" not in nav
    assert not (REPO_ROOT / "docs/model-verification/index.md").exists()
    assert not (REPO_ROOT / "docs/fern/versions/nightly/pages/model-verification/index.mdx").exists()
    assert not list((REPO_ROOT / "docs/model-verification/models").glob("*.md"))
    assert not list((REPO_ROOT / "docs/fern/versions/nightly/pages/model-verification/models").glob("*.mdx"))


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
