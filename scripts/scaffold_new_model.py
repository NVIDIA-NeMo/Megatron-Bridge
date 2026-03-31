#!/usr/bin/env python3
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

"""
Scaffold a new model bridge for Megatron Bridge.

Generates the boilerplate files needed to add a new model:
  - Provider (HF config → TransformerConfig)
  - Bridge (registration + parameter mappings)
  - Unit tests (provider + roundtrip)
  - __init__.py

Usage:
    python scripts/scaffold_new_model.py my_model
    python scripts/scaffold_new_model.py my_model --hf-class MyModelForCausalLM
    python scripts/scaffold_new_model.py my_model --hf-class MyModelForCausalLM --hf-model-id org/MyModel-7B
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "src" / "megatron" / "bridge" / "models"
TESTS_DIR = REPO_ROOT / "tests" / "unit_tests" / "models"

COPYRIGHT_HEADER = """\
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
"""


def to_class_name(snake: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in snake.split("_"))


def write_file(path: Path, content: str) -> None:
    """Write content to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  Created {path.relative_to(REPO_ROOT)}")


def scaffold(model_name: str, hf_class: str, hf_model_id: str) -> None:
    """Generate boilerplate files for a new model bridge."""
    class_name = to_class_name(model_name)
    model_dir = MODELS_DIR / model_name
    test_dir = TESTS_DIR / model_name

    if model_dir.exists():
        print(f"Error: {model_dir.relative_to(REPO_ROOT)} already exists.", file=sys.stderr)
        sys.exit(1)

    print(f"Scaffolding model: {model_name} (class: {class_name}, HF: {hf_class})")
    print()

    # __init__.py
    write_file(
        model_dir / "__init__.py",
        f"{COPYRIGHT_HEADER}\n"
        f"# Import bridge to trigger registration\n"
        f"from megatron.bridge.models.{model_name} import {model_name}_bridge  # noqa: F401\n",
    )

    # Provider
    write_file(
        model_dir / f"{model_name}_provider.py",
        f"{COPYRIGHT_HEADER}\n"
        f'"""Provider for {class_name} models."""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"from megatron.bridge.models.gpt_provider import GPTModelProvider\n"
        f"\n"
        f"\n"
        f"class {class_name}Provider(GPTModelProvider):\n"
        f'    """{class_name} model provider.\n'
        f"\n"
        f"    Maps HuggingFace {hf_class} config to Megatron-Core TransformerConfig.\n"
        f'    """\n'
        f"\n"
        f"    # TODO: Override methods to map HF config fields to TransformerConfig.\n"
        f"    #   See llama_provider.py or qwen_provider.py for reference.\n"
        f"    pass\n",
    )

    # Bridge
    write_file(
        model_dir / f"{model_name}_bridge.py",
        f"{COPYRIGHT_HEADER}\n"
        f'"""Bridge for {class_name} models."""\n'
        f"\n"
        f"from __future__ import annotations\n"
        f"\n"
        f"from megatron.core.models.gpt.gpt_model import GPTModel\n"
        f"from transformers import {hf_class}\n"
        f"\n"
        f"from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry\n"
        f"from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge\n"
        f"from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping\n"
        f"from megatron.bridge.models.{model_name}.{model_name}_provider import {class_name}Provider\n"
        f"\n"
        f"\n"
        f"@MegatronModelBridge.register_bridge(\n"
        f"    source={hf_class},\n"
        f"    target=GPTModel,\n"
        f'    model_type="{model_name}",\n'
        f")\n"
        f"class {class_name}Bridge(MegatronModelBridge):\n"
        f'    """{class_name} HF ↔ Megatron bridge."""\n'
        f"\n"
        f"    def provider_bridge(self, hf_model, **kwargs):\n"
        f'        """Map HF model/config to a Megatron provider."""\n'
        f"        # TODO: Create and return a {class_name}Provider from hf_model.config\n"
        f'        raise NotImplementedError("Implement provider_bridge for {class_name}")\n'
        f"\n"
        f"    def mapping_registry(self):\n"
        f'        """Define parameter name mappings between HF and Megatron."""\n'
        f"        # TODO: Fill in the actual parameter mappings.\n"
        f"        #   See llama_bridge.py or qwen3_bridge.py for reference.\n"
        f"        return MegatronMappingRegistry(\n"
        f"            [\n"
        f"                AutoMapping(\n"
        f'                    "embedding.word_embeddings.weight",\n'
        f'                    "model.embed_tokens.weight",\n'
        f"                ),\n"
        f"                # QKVMapping(...),\n"
        f"                # GatedMLPMapping(...),\n"
        f"                AutoMapping(\n"
        f'                    "decoder.final_layernorm.weight",\n'
        f'                    "model.norm.weight",\n'
        f"                ),\n"
        f"                AutoMapping(\n"
        f'                    "output_layer.weight",\n'
        f'                    "lm_head.weight",\n'
        f"                ),\n"
        f"            ]\n"
        f"        )\n",
    )

    # Tests
    write_file(
        test_dir / "__init__.py",
        f"{COPYRIGHT_HEADER}\n",
    )

    write_file(
        test_dir / f"test_{model_name}_bridge.py",
        f"{COPYRIGHT_HEADER}\n"
        f'"""Tests for {class_name} bridge."""\n'
        f"\n"
        f"import pytest\n"
        f"\n"
        f"from megatron.bridge import AutoBridge\n"
        f"\n"
        f"\n"
        f'HF_MODEL_ID = "{hf_model_id}"  # TODO: Use a small model for fast tests\n'
        f"\n"
        f"\n"
        f"@pytest.mark.unit\n"
        f"class Test{class_name}Bridge:\n"
        f'    """Unit tests for {class_name} bridge."""\n'
        f"\n"
        f"    def test_supports(self):\n"
        f'        """Verify the model is detected as supported."""\n'
        f"        assert AutoBridge.can_handle(HF_MODEL_ID)\n"
        f"\n"
        f"    def test_roundtrip(self, tmp_path):\n"
        f'        """Load HF → Megatron → export HF, verify weights match."""\n'
        f"        bridge = AutoBridge.from_hf_pretrained(HF_MODEL_ID)\n"
        f"        megatron_model = bridge.to_megatron_model(wrap_with_ddp=False)\n"
        f'        bridge.save_hf_pretrained(megatron_model, str(tmp_path / "export"))\n'
        f"        # TODO: Add weight comparison assertions\n",
    )

    print()
    print("Next steps:")
    print(f"  1. Implement {model_name}_provider.py — map HF config → TransformerConfig")
    print(f"  2. Implement {model_name}_bridge.py — fill in provider_bridge() and mapping_registry()")
    print(f"  3. Run tests: make test-k K=test_{model_name}")
    print()
    print("Reference implementations:")
    print("  - Llama (simple):    src/megatron/bridge/models/llama/")
    print("  - Qwen (with MoE):  src/megatron/bridge/models/qwen/")
    print("  - DeepSeek (MLA):   src/megatron/bridge/models/deepseek/")
    print()
    print("Guides:")
    print("  - Quick reference:   docs/contributing-a-model.md")
    print("  - Full guide:        docs/adding-new-models.md")
    print("  - AI agent skills:   skills/adding-model-support/  (for Cursor, Claude Code, etc.)")


def main():
    """Parse arguments and run the scaffold."""
    parser = argparse.ArgumentParser(
        description="Scaffold boilerplate files for a new model bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/scaffold_new_model.py phi\n"
            "  python scripts/scaffold_new_model.py phi --hf-class PhiForCausalLM\n"
            "  python scripts/scaffold_new_model.py phi --hf-class PhiForCausalLM --hf-model-id microsoft/phi-2\n"
        ),
    )
    parser.add_argument("model_name", help="Snake_case model name (e.g., phi, starcoder)")
    parser.add_argument(
        "--hf-class",
        default=None,
        help="HuggingFace model class name (default: <ModelName>ForCausalLM)",
    )
    parser.add_argument(
        "--hf-model-id",
        default=None,
        help="HuggingFace model ID for test fixtures (default: org/<ModelName>)",
    )
    args = parser.parse_args()

    model_name = args.model_name.lower().replace("-", "_")
    class_name = to_class_name(model_name)
    hf_class = args.hf_class or f"{class_name}ForCausalLM"
    hf_model_id = args.hf_model_id or f"org/{class_name}"

    scaffold(model_name, hf_class, hf_model_id)


if __name__ == "__main__":
    main()
