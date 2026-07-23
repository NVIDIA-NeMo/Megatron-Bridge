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
"""Doc-consistency regression tests for README drift (NVBug 6366190).

Validates that documented examples in README/tutorial files match the actual
source tree: recipe names exist, referenced script paths exist, config field
names are correct, implemented CLI flags are documented, and the bundled
Megatron-LM tool path is correct.

Deliberately stdlib-only (no torch / megatron import) so it scans source files
directly and runs anywhere, including without the GPU stack.
"""

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPES_DIR = REPO_ROOT / "src" / "megatron" / "bridge" / "recipes"
PERF_RECIPES_DIR = REPO_ROOT / "src" / "megatron" / "bridge" / "perf_recipes"
TRAINING_README = REPO_ROOT / "scripts" / "training" / "README.md"
DATASET_UTILS = REPO_ROOT / "src" / "megatron" / "bridge" / "recipes" / "utils" / "dataset_utils.py"
LLAMA_README = REPO_ROOT / "tutorials" / "recipes" / "llama" / "README.md"
DCLM_README = REPO_ROOT / "tutorials" / "data" / "dclm" / "README.md"
GEMMA3_VL_README = REPO_ROOT / "examples" / "models" / "gemma" / "gemma3_vl" / "README.md"
GEMMA3_VL_RECIPES = RECIPES_DIR / "gemma3_vl" / "gemma3_vl.py"
RUN_RECIPE = REPO_ROOT / "scripts" / "training" / "run_recipe.py"
MODEL_EXAMPLES = REPO_ROOT / "examples" / "models"
TRAINING_CONFIG = REPO_ROOT / "src" / "megatron" / "bridge" / "training" / "config.py"
QWEN_OMNI_RECIPE = REPO_ROOT / "src" / "megatron" / "bridge" / "recipes" / "qwen_omni" / "h100" / "qwen3_omni.py"
QWEN_OMNI_TRAINING_SCRIPT = REPO_ROOT / "examples" / "models" / "qwen" / "qwen3_omni" / "local_train_thinker_full.sh"
LEGACY_VLM_DATASETS = REPO_ROOT / "src" / "megatron" / "bridge" / "data" / "vlm_datasets"
LOCAL_CONVERSATION_SOURCE = REPO_ROOT / "src" / "megatron" / "bridge" / "data" / "sources" / "local_conversation.py"
HF_MULTIMODAL_README = REPO_ROOT / "tutorials" / "data" / "hf-multimodal" / "README.md"
DATA_PREPARATION_DOCS = (
    REPO_ROOT / "docs" / "training" / "data-preparation.md",
    REPO_ROOT / "docs" / "fern" / "versions" / "nightly" / "pages" / "training" / "data-preparation.mdx",
)
QWEN3_VL_README = REPO_ROOT / "examples" / "models" / "qwen" / "qwen3_vl" / "README.md"
QWEN25_VL_DOCS = (
    REPO_ROOT / "docs" / "models" / "qwen" / "qwen2.5-vl.md",
    REPO_ROOT / "docs" / "fern" / "versions" / "nightly" / "pages" / "models" / "qwen" / "qwen2.5-vl.mdx",
)
VALOR_TUTORIAL = REPO_ROOT / "tutorials" / "data" / "valor32k-avqa" / "data-preparation.md"
SPHINX_TUTORIAL_LINK_DOCS = (
    REPO_ROOT / "docs" / "training" / "data-preparation.md",
    REPO_ROOT / "docs" / "models" / "qwen" / "qwen2.5-vl.md",
)
LEGACY_RUN_RECIPE_ARGUMENTS = (
    "--dry_run",
    "--dump_env",
    "--peft-scheme",
    "--peft_scheme",
    "--packed-sequence",
    "--packed_sequence",
    "--seq-length",
    "--seq_length",
    "--dataset llm-pretrain",
    "--dataset llm-pretrain-mock",
    "--dataset llm-finetune",
    "--dataset llm-finetune-preloaded",
    "--dataset vlm-energon",
    "--dataset vlm-hf",
    "--dataset vlm-preloaded",
)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _defined_recipe_names() -> set[str]:
    """All recipe-config functions and public aliases under src/.../recipes/** and perf_recipes/**."""
    names: set[str] = set()
    recipe_files = list(RECIPES_DIR.rglob("*.py"))
    if PERF_RECIPES_DIR.is_dir():
        recipe_files += list(PERF_RECIPES_DIR.rglob("*.py"))
    for py in recipe_files:
        tree = ast.parse(_read(py), filename=str(py))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.endswith("_config"):
                names.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    public_name = alias.asname or alias.name.rsplit(".", 1)[-1]
                    if public_name.endswith("_config"):
                        names.add(public_name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(
                    target.id for target in targets if isinstance(target, ast.Name) and target.id.endswith("_config")
                )
    return names


def test_training_readme_recipes_exist():
    """Every `--recipe NAME` in the training README is a real recipe function (bugs 1, 2)."""
    defined = _defined_recipe_names()
    assert defined, "no recipe configs discovered — path assumption wrong"
    referenced = set(re.findall(r"--recipe\s+(\w+)", _read(TRAINING_README)))
    missing = sorted(r for r in referenced if r not in defined)
    assert not missing, f"README references nonexistent recipes: {missing}"


def test_training_readme_dataset_table_matches_public_presets():
    """Every public dataset preset is documented once in the launcher README."""
    tree = ast.parse(_read(DATASET_UTILS), filename=str(DATASET_UTILS))
    registry = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "DATASET_PRESETS"
    )
    assert isinstance(registry, ast.Dict)
    registered = {key.value for key in registry.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    documented = re.findall(r"^\| `([^`]+)` \|", _read(TRAINING_README), re.MULTILINE)

    assert len(documented) == len(set(documented))
    assert set(documented) == registered
    assert "allenai/tulu-3-sft-mixture" in _read(TRAINING_README)


def test_gemma3_vl_readme_recipes_are_exported():
    """Every Gemma3-VL recipe advertised in its README is a launcher-visible alias."""
    documented = set(re.findall(r"`(gemma3_vl_\w+_config)`", _read(GEMMA3_VL_README)))
    assert documented, "no Gemma3-VL recipe configs documented"

    exported = set(re.findall(r"as\s+(gemma3_vl_\w+_config)", _read(GEMMA3_VL_RECIPES)))
    missing = sorted(documented - exported)
    assert not missing, f"Gemma3-VL README references unexported recipes: {missing}"


def test_llama_readme_conversion_path_exists():
    """The conversion script path in the llama tutorial resolves to a real file (bug 3)."""
    text = _read(LLAMA_README)
    assert "scripts/conversion/convert.sh" in text, "expected stable conversion CLI path"
    assert (REPO_ROOT / "scripts" / "conversion" / "convert.sh").is_file()
    assert "examples/conversion/convert_checkpoints.py" not in text, "stale example conversion path still present"


def test_shell_conversion_launcher_is_not_run_through_python():
    """The stable shell conversion launcher must be invoked as a shell command."""
    invalid_invocation = re.compile(
        r"(?:\bpython(?:\d+(?:\.\d+)?)?\b|\btorchrun\b|\$\{?PYTHON\}?)[^\n]*scripts/conversion/convert\.sh"
    )
    offenders: list[str] = []
    for root in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs",
        REPO_ROOT / "examples",
        REPO_ROOT / "scripts",
        REPO_ROOT / "skills",
        REPO_ROOT / "tutorials",
    ):
        paths = (root,) if root.is_file() else root.rglob("*")
        for path in paths:
            if path.suffix not in {".md", ".mdx", ".py", ".sh"}:
                continue
            logical_lines = re.sub(r"\\\s*\n\s*", " ", _read(path))
            if invalid_invocation.search(logical_lines):
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, f"convert.sh is invoked through Python or torchrun: {offenders}"


def test_llama_readme_gptdataset_field_name():
    """The GPTDatasetConfig YAML block uses the canonical field name (bug 4)."""
    # Bridge exposes seq_length and copies it to MCore's internal sequence_length during finalization.
    assert "seq_length: int" in _read(TRAINING_CONFIG), "seq_length not found in config source"
    text = _read(LLAMA_README)
    m = re.search(r"#\s*GPTDatasetConfig\b(.*?)(?:\n\s*\n)", text, re.DOTALL)
    assert m, "could not locate GPTDatasetConfig YAML block"
    block = m.group(1)
    assert "seq_length:" in block, "GPTDatasetConfig block should use seq_length"
    assert "sequence_length:" not in block, "GPTDatasetConfig block still uses internal sequence_length"


def test_hf_path_flag_documented_if_implemented():
    """If run_recipe.py implements --hf_path, the README documents it (bug 5)."""
    if '"--hf_path"' not in _read(RUN_RECIPE):
        return  # flag not implemented — nothing to document
    assert "--hf_path" in _read(TRAINING_README), "--hf_path is implemented but not documented in README"


def test_model_examples_use_current_run_recipe_arguments():
    """Model examples that call run_recipe.py must not advertise removed arguments."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(MODEL_EXAMPLES.rglob("*")):
        if path.suffix not in {".md", ".sh"}:
            continue
        text = _read(path)
        if "scripts/training/run_recipe.py" not in text:
            continue
        stale_arguments = [argument for argument in LEGACY_RUN_RECIPE_ARGUMENTS if argument in text]
        if stale_arguments:
            offenders[str(path.relative_to(REPO_ROOT))] = stale_arguments

    assert not offenders, f"Model examples use removed run_recipe.py arguments: {offenders}"


def test_nemotron_and_qwen2_audio_finetune_launchers_use_exported_recipes():
    """SFT and PEFT example recipes must match their launcher-visible aliases."""
    launcher_expectations = {
        MODEL_EXAMPLES / "nemotron" / "nemotron_3" / "nano" / "slurm_sft.sh": (
            "${MODEL_NAME}_sft_config",
            "nemotron_3_nano_finetune_config",
        ),
        MODEL_EXAMPLES / "nemotron" / "nemotron_3" / "nano" / "slurm_peft.sh": (
            "${MODEL_NAME}_peft_config",
            "nemotron_3_nano_finetune_config",
        ),
        MODEL_EXAMPLES / "qwen" / "qwen2_audio" / "sft.sh": (
            "qwen2_audio_7b_sft_config",
            "qwen2_audio_7b_finetune_config",
        ),
    }
    for path, (expected_recipe, stale_recipe) in launcher_expectations.items():
        text = _read(path)
        assert expected_recipe in text
        assert stale_recipe not in text

    defined_recipes = _defined_recipe_names()
    expected_recipes = {
        "nemotron_3_nano_sft_config",
        "nemotron_3_nano_peft_config",
        "qwen2_audio_7b_sft_config",
        "qwen2_audio_7b_peft_config",
    }
    assert expected_recipes <= defined_recipes


def test_dclm_readme_megatron_lm_tool_path():
    """The DCLM tutorial points at the bundled submodule tool path (bug 6)."""
    gitmodules = _read(REPO_ROOT / ".gitmodules")
    assert "3rdparty/Megatron-LM" in gitmodules, "submodule path assumption wrong"
    text = _read(DCLM_README)
    assert "3rdparty/Megatron-LM/tools/preprocess_data.py" in text, "expected 3rdparty submodule path"
    # the bare path (no 3rdparty/ prefix) is broken from /opt/Megatron-Bridge
    assert not re.search(r"(?<![\w/])Megatron-LM/tools/preprocess_data\.py", text), (
        "stale bare Megatron-LM path present"
    )


def test_vlm_json_script_uses_hf_loader_without_local_provider():
    """JSON VLM entrypoints should use Direct HF rather than a local provider."""
    assert not LEGACY_VLM_DATASETS.exists()
    assert not LOCAL_CONVERSATION_SOURCE.exists()
    for path in (RUN_RECIPE, QWEN_OMNI_RECIPE, QWEN_OMNI_TRAINING_SCRIPT):
        text = _read(path)
        assert "PreloadedVLMConversationProvider" not in text
        assert "LocalConversationDatasetSourceConfig" not in text
        assert "vlm-preloaded" not in text
        assert "vlm-local" not in text
    assert "DirectHFSFTDatasetConfig" in _read(QWEN_OMNI_RECIPE)
    assert "HFDatasetSourceConfig" in _read(QWEN_OMNI_RECIPE)
    assert 'path_or_dataset="json"' in _read(QWEN_OMNI_RECIPE)
    training_script = _read(QWEN_OMNI_TRAINING_SCRIPT)
    assert "dataset.source.load_kwargs={data_files:{train:" in training_script
    assert "dataset.validation_source.load_kwargs={data_files:{validation:" in training_script
    assert "dataset.test_source.load_kwargs={data_files:{test:" in training_script
    assert ".load_kwargs.data_files." not in training_script
    assert "OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-True}" in training_script
    assert "OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}" in training_script
    assert 'optimizer.overlap_cpu_optimizer_d2h_h2d="${OVERLAP_CPU_OPTIMIZER_D2H_H2D}"' in training_script


def test_hf_multimodal_local_media_docs_use_processor_native_content():
    """Local JSON examples must not promise removed preloaded media adaptation."""
    for path in (HF_MULTIMODAL_README, *DATA_PREPARATION_DOCS):
        text = _read(path)
        assert '"content": [{"type": "image", "image": "' in text
        assert '"content": "<image>Describe' not in text
        assert '"images": ["receipt' not in text
        assert "top-level media-list schema is not" in text


def test_multimodal_model_docs_use_current_launchers_and_conversion_flags():
    """Model examples should point at current recipes and canonical tutorials."""
    qwen3 = _read(QWEN3_VL_README)
    assert "qwen3_vl_8b_peft_config" in qwen3
    assert "qwen3_vl_8b_finetune_config" not in qwen3
    assert "--source-dir/path" not in qwen3
    assert "hf-multimodal/README.md" in qwen3
    assert "data/energon/README.md" in qwen3

    for path in QWEN25_VL_DOCS:
        text = _read(path)
        assert "scripts/training/run_recipe.py" in text
        assert "qwen25_vl_3b_sft_config" in text
        assert "qwen25_vl_3b_peft_config" in text
        assert "finetune_qwen25_vl.py" not in text
        assert "qwen25_vl_3b_finetune_config" not in text
        assert "--dataset-type" not in text

    valor = _read(VALOR_TUTORIAL)
    assert "--hf-model <HF_MODEL_PATH>" in valor
    assert "--megatron-path /checkpoints/nemotron_omni" in valor
    assert "--hf_path" not in valor
    assert "--output_dir /checkpoints/nemotron_omni" not in valor


def test_sphinx_docs_link_out_of_tree_tutorials_as_urls():
    """Sphinx must not treat repository-root tutorials as source documents."""
    for path in SPHINX_TUTORIAL_LINK_DOCS:
        relative_tutorial_links = re.findall(r"\]\((?:\.\./)+tutorials/[^)]+\)", _read(path))
        assert not relative_tutorial_links, f"{path} has out-of-tree Sphinx links: {relative_tutorial_links}"


if __name__ == "__main__":
    # Allow standalone RED-GREEN without pytest/torch:  python3 test_readme_consistency.py
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
