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

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import AutoConfig, dynamic_module_utils


# Reference checkpoint we derive the architecture + custom modeling code from. We only
# read its *config* (config.json is tiny) and reuse its trust_remote_code modules — we
# never load its ~30B weights.
_DEFAULT_HF_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

# Toy overrides — shrink only the param-dominant, constraint-free knobs so the roundtrip
# exercises the same conversion code paths in seconds instead of loading/converting/
# writing a ~30B (≈60 GB) model. All hidden/head/projector dims and the vision (RADIO)
# config are left intact so the vision→LLM and sound→LLM projectors stay consistent.
#
# ``hybrid_override_pattern`` must stay the same length as ``num_hidden_layers``; we take
# the leading 6 chars of the real 52-char pattern, which keeps a representative mamba (M),
# MoE-MLP (E) and attention (*) layer mix.
_LLM_OVERRIDES = {
    "num_hidden_layers": 6,
    "hybrid_override_pattern": "MEMEM*",
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 4096,
}
_SOUND_OVERRIDES = {
    "num_hidden_layers": 2,
}


def _apply_overrides(sub_config, overrides: dict) -> None:
    """Apply size overrides to a sub-config that may be either a config object or a dict."""
    for key, value in overrides.items():
        if isinstance(sub_config, dict):
            sub_config[key] = value
        else:
            setattr(sub_config, key, value)


def _fix_tied_weights_keys(model: nn.Module) -> None:
    """Convert _tied_weights_keys from list to dict for transformers 5.x compatibility."""
    for module in model.modules():
        tied = getattr(module, "_tied_weights_keys", None)
        if isinstance(tied, list):
            module._tied_weights_keys = {k: k for k in tied}


def _copy_custom_code_from_source(model_dir: Path, source_file: str | Path) -> None:
    """Copy custom modeling/configuration modules needed for local trust_remote_code loading."""
    copied_files: set[str] = set()

    source_file = Path(source_file)
    source_dir = source_file.parent
    for py_file in source_dir.glob("*.py"):
        target = model_dir / py_file.name
        shutil.copy2(py_file, target)
        copied_files.add(target.name)

    from transformers.dynamic_module_utils import get_relative_import_files

    for source in map(Path, get_relative_import_files(source_file)):
        target = model_dir / source.name
        if target.name not in copied_files:
            shutil.copy2(source, target)
            copied_files.add(target.name)


class TestNemotronOmniConversion:
    @pytest.fixture(scope="class")
    def nemotron_omni_toy_model_path(self, tmp_path_factory):
        """Build a tiny NemotronH-Nano-Omni HF checkpoint (random weights) for fast conversion testing.

        Reads only the reference config (no ~30B weight download), shrinks the LLM and sound
        encoder, instantiates a random-init model, and saves it with its trust_remote_code
        modules so the conversion subprocess can load it offline.
        """
        pytest.importorskip("transformer_engine_torch")

        temp_dir = tmp_path_factory.mktemp("nemotron_omni_toy_model")
        model_dir = temp_dir / "nemotron_omni_toy"

        # Config-only load (config.json is tiny); never pulls the 30B weights.
        config = AutoConfig.from_pretrained(_DEFAULT_HF_ID, trust_remote_code=True)

        _apply_overrides(config.llm_config, _LLM_OVERRIDES)
        if getattr(config, "sound_config", None) is not None:
            _apply_overrides(config.sound_config, _SOUND_OVERRIDES)
        config.torch_dtype = torch.bfloat16

        # Resolve the omni model class from the reference repo's trust_remote_code modules.
        model_class_ref = config.auto_map["AutoModel"]
        model_class = dynamic_module_utils.get_class_from_dynamic_module(
            class_reference=model_class_ref,
            pretrained_model_name_or_path=_DEFAULT_HF_ID,
            repo_id=_DEFAULT_HF_ID,
        )

        model = model_class(config)
        model = model.bfloat16() if hasattr(model, "bfloat16") else model
        _fix_tied_weights_keys(model)

        model.save_pretrained(model_dir, safe_serialization=True, save_original_format=False)

        # Copy the custom modeling/config modules so trust_remote_code loading works offline.
        modeling_filepath = os.path.abspath(sys.modules[model_class.__module__].__file__)
        _copy_custom_code_from_source(model_dir, modeling_filepath)

        # Persist the (overridden) config so the saved checkpoint reflects the toy sizes.
        with open(model_dir / "config.json", "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        # The conversion re-export (save_hf_pretrained) copies the *input* tokenizer to the
        # output, so the toy dir must carry a loadable tokenizer. Copy the reference repo's
        # fast tokenizer files verbatim (no instantiation) so AutoTokenizer.from_pretrained()
        # loads tokenizer.json directly — avoiding any slow→fast conversion backend
        # (sentencepiece/tiktoken), which is not a dependency. The tokenizer's vocab need not
        # match the toy model's reduced vocab: the roundtrip only copies the tokenizer through.
        from huggingface_hub import hf_hub_download

        for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            shutil.copy2(hf_hub_download(_DEFAULT_HF_ID, fname), model_dir / fname)

        return str(model_dir)

    @pytest.mark.run_only_on("GPU")
    def test_nemotron_omni_conversion_roundtrip(self, nemotron_omni_toy_model_path, tmp_path):
        # Allow an explicit external override (e.g. to run against the full HF model), but
        # default to the tiny locally-built toy checkpoint for a fast, deterministic run.
        hf_model_id = os.environ.get("NEMOTRON_OMNI_HF_MODEL") or nemotron_omni_toy_model_path

        output_dir = tmp_path / "nemotron_omni_roundtrip"
        output_dir.mkdir(exist_ok=True)

        tp = os.environ.get("NEMOTRON_OMNI_CONVERSION_TP", "2")
        pp = os.environ.get("NEMOTRON_OMNI_CONVERSION_PP", "1")
        nproc = os.environ.get("NEMOTRON_OMNI_CONVERSION_GPUS", tp)

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            nproc,
            "--nnodes",
            "1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            hf_model_id,
            "--output-dir",
            str(output_dir),
            "--tp",
            tp,
            "--pp",
            pp,
            "--trust-remote-code",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[5],
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Nemotron Omni conversion failed with return code {result.returncode}"

        model_name = Path(hf_model_id).name
        converted_model_dir = output_dir / model_name
        assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

        config_file = converted_model_dir / "config.json"
        assert config_file.exists(), f"config.json not found in converted model at {config_file}"
        assert list(converted_model_dir.glob("model*.safetensors")), (
            f"Model weights file not found in converted model at {converted_model_dir}"
        )

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["architectures"][0] == "NemotronH_Nano_Omni_Reasoning_V3"
        assert saved_config["model_type"] == "NemotronH_Nano_Omni_Reasoning_V3"
        assert "llm_config" in saved_config
        assert "vision_config" in saved_config
        assert "sound_config" in saved_config
