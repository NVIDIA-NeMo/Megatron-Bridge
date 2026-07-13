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

import inspect
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MIMO_V2_FLASH_HF_ID = "XiaomiMiMo/MiMo-V2-Flash"


def _compute_default_rope_inv_freq(config, device=None, seq_len=None, **kwargs):
    """Standard RoPE inv_freq computation (the ``"default"`` variant)."""
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def _patch_hf_mimo_for_transformers5() -> None:
    """Apply HF-side compatibility fixes needed to instantiate the toy model."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_inv_freq)

    try:
        hf_rotary_cls = get_class_from_dynamic_module(
            "modeling_mimo_v2_flash.MiMoV2FlashRotaryEmbedding",
            MIMO_V2_FLASH_HF_ID,
        )
        if not hasattr(hf_rotary_cls, "compute_default_rope_parameters"):
            hf_rotary_cls.compute_default_rope_parameters = staticmethod(_compute_default_rope_inv_freq)
    except Exception:
        pass

    try:
        hf_config_cls = get_class_from_dynamic_module(
            "configuration_mimo_v2_flash.MiMoV2FlashConfig",
            MIMO_V2_FLASH_HF_ID,
        )
        if not hf_config_cls.model_type:
            hf_config_cls.model_type = "mimo_v2_flash"
    except Exception:
        pass


class TestMiMoV2FlashConversion:
    """Test MiMo-V2-Flash model conversion with different parallelism configurations."""

    @pytest.fixture(scope="class")
    def mimo_v2_flash_toy_model_path(self, tmp_path_factory):
        """Create a toy MiMo-V2-Flash checkpoint with drastically reduced dimensions."""
        _patch_hf_mimo_for_transformers5()

        temp_dir = tmp_path_factory.mktemp("mimo_v2_flash_toy_model")
        model_dir = temp_dir / "mimo_v2_flash_toy"
        model_dir.mkdir(parents=True, exist_ok=True)

        config = AutoConfig.from_pretrained(MIMO_V2_FLASH_HF_ID, trust_remote_code=True)
        config.torch_dtype = torch.bfloat16

        # Core architecture
        config.num_hidden_layers = 4
        config.hidden_size = 256
        config.intermediate_size = 256
        config.num_attention_heads = 8
        config.swa_num_attention_heads = 8
        config.num_key_value_heads = 2
        config.swa_num_key_value_heads = 4
        config.head_dim = 64
        config.swa_head_dim = 64
        config.v_head_dim = 32
        config.swa_v_head_dim = 32
        config.partial_rotary_factor = 0.5
        config.vocab_size = 2048
        config.max_position_embeddings = 4096

        # Hybrid attention: alternating full + SWA; first dense, rest MoE
        config.hybrid_layer_pattern = [0, 1, 0, 1]
        config.moe_layer_freq = [0, 1, 0, 1]
        config.sliding_window_size = 64
        config.sliding_window = 64
        config.attention_chunk_size = 64

        # MoE: fine-grained, top-2 of 4 experts
        config.n_routed_experts = 4
        config.num_experts_per_tok = 2
        config.moe_intermediate_size = 128

        # MTP — disabled (HF init does not handle MTP correctly)
        if hasattr(config, "num_nextn_predict_layers"):
            config.num_nextn_predict_layers = 0

        # Drop any quantization config from the checkpoint metadata so the
        # toy bf16 weights load cleanly.
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")

        # Clamp out-of-range special tokens
        for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
            if getattr(config, attr, None) is not None and getattr(config, attr) >= config.vocab_size:
                setattr(config, attr, 0)

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to(dtype=torch.bfloat16)

        with torch.no_grad():
            for tensor in list(model.parameters()) + list(model.buffers()):
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    tensor.zero_()

        for m in model.modules():
            if hasattr(m, "_tied_weights_keys"):
                m._tied_weights_keys = []

        model.save_pretrained(model_dir, safe_serialization=True)

        # Copy custom modeling code
        source_file = inspect.getfile(type(model))
        source_dir = Path(source_file).parent
        for py_file in source_dir.glob("*.py"):
            target = model_dir / py_file.name
            if not target.exists():
                shutil.copy2(py_file, target)

        config.save_pretrained(model_dir)

        # In-place patch of the copied configuration file so that
        # ``MiMoV2FlashConfig.model_type`` defaults to ``"mimo_v2_flash"``
        config_py_file = model_dir / "configuration_mimo_v2_flash.py"
        if config_py_file.exists():
            src = config_py_file.read_text()
            marker = "# === Megatron-Bridge mimo_v2_flash model_type patch ==="
            if marker not in src:
                config_py_file.write_text(
                    src
                    + f"\n{marker}\n"
                    + "if not getattr(MiMoV2FlashConfig, 'model_type', ''):\n"
                    + "    MiMoV2FlashConfig.model_type = 'mimo_v2_flash'\n"
                )

        modeling_file = model_dir / "modeling_mimo_v2_flash.py"
        if modeling_file.exists():
            src = modeling_file.read_text()
            if "_mb_default_rope" not in src:
                shim = (
                    "\n# === transformers 5.x compatibility shim (added by Megatron-Bridge tests) ===\n"
                    "def _mb_default_rope(config, device=None, seq_len=None, **kwargs):\n"
                    "    import torch as _torch\n"
                    "    base = config.rope_theta\n"
                    "    prf = getattr(config, 'partial_rotary_factor', 1.0)\n"
                    "    hd = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads\n"
                    "    dim = int(hd * prf)\n"
                    "    inv_freq = 1.0 / (base ** (_torch.arange(0, dim, 2, dtype=_torch.int64).to(device=device, dtype=_torch.float) / dim))\n"
                    "    return inv_freq, 1.0\n"
                    "try:\n"
                    "    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _MB_ROPE_INIT_FUNCTIONS\n"
                    "    _MB_ROPE_INIT_FUNCTIONS.setdefault('default', _mb_default_rope)\n"
                    "except Exception:\n"
                    "    pass\n"
                    "if not hasattr(MiMoV2FlashRotaryEmbedding, 'compute_default_rope_parameters'):\n"
                    "    MiMoV2FlashRotaryEmbedding.compute_default_rope_parameters = staticmethod(_mb_default_rope)\n"
                    "if not getattr(MiMoV2FlashConfig, 'model_type', ''):\n"
                    "    MiMoV2FlashConfig.model_type = 'mimo_v2_flash'\n"
                    "_mb_orig_eager_attn = eager_attention_forward\n"
                    "def eager_attention_forward(*args, **kwargs):\n"
                    "    kwargs.pop('position_ids', None)\n"
                    "    return _mb_orig_eager_attn(*args, **kwargs)\n"
                    "import sys as _mb_sys\n"
                    "_mb_sys.modules[__name__].eager_attention_forward = eager_attention_forward\n"
                )
                modeling_file.write_text(src + shim)

        tokenizer = AutoTokenizer.from_pretrained(MIMO_V2_FLASH_HF_ID, trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)

        return str(model_dir)

    def test_toy_model_creation(self, mimo_v2_flash_toy_model_path):
        """Verify the toy checkpoint was created and has the expected fields."""
        model_path = Path(mimo_v2_flash_toy_model_path)
        assert model_path.exists()

        config_file = model_path / "config.json"
        assert config_file.exists()

        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            sharded = list(model_path.glob("model-*-of-*.safetensors"))
            assert len(sharded) > 0, "No model weight files found"

        with open(config_file) as f:
            cfg = json.load(f)

        assert cfg.get("model_type") == "mimo_v2_flash"
        assert cfg.get("num_hidden_layers") == 4
        assert cfg.get("v_head_dim") == 32
        assert cfg.get("hybrid_layer_pattern") == [0, 1, 0, 1]
        assert cfg.get("n_routed_experts") == 4
        assert cfg.get("num_experts_per_tok") == 2

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,etp,test_name",
        [
            (2, 1, 1, 1, "TP"),
            (1, 2, 1, 1, "PP"),
            (1, 1, 2, 1, "EP"),
            (1, 1, 1, 2, "ETP"),
        ],
    )
    def test_mimo_v2_flash_conversion_parallelism(
        self, mimo_v2_flash_toy_model_path, tmp_path, tp, pp, ep, etp, test_name
    ):
        """Round-trip conversion (HF → Megatron → HF) under TP / PP / EP / ETP."""
        test_output_dir = tmp_path / f"mimo_v2_flash_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        repo_root = Path(__file__).resolve().parents[5]
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            mimo_v2_flash_toy_model_path,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--ep",
            str(ep),
            "--etp",
            str(etp),
            "--trust-remote-code",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"MiMo-V2-Flash {test_name} conversion failed with return code {result.returncode}")

        assert f"Expert tensor parallel size: {etp}" in result.stdout

        model_name = Path(mimo_v2_flash_toy_model_path).name
        converted_dir = test_output_dir / model_name
        assert converted_dir.exists(), f"Converted model directory not found at {converted_dir}"

        config_file = converted_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["model_type"] == "mimo_v2_flash"
        assert saved_config["num_hidden_layers"] == 4
        assert saved_config["v_head_dim"] == 32
        assert saved_config["n_routed_experts"] == 4

    @pytest.mark.run_only_on("GPU")
    def test_mimo_v2_flash_autoconfig_roundtrip(self, mimo_v2_flash_toy_model_path, tmp_path):
        from unittest.mock import patch

        from transformers import AutoModelForCausalLM

        from tests.functional_tests.utils import autoconfig_roundtrip

        original_from_pretrained = AutoModelForCausalLM.from_pretrained

        def _pinned_from_pretrained(*args, **kwargs):
            if kwargs.get("device_map") == "auto":
                kwargs["device_map"] = {"": "cuda:0"}
            return original_from_pretrained(*args, **kwargs)

        with patch.object(AutoModelForCausalLM, "from_pretrained", _pinned_from_pretrained):
            autoconfig_roundtrip(mimo_v2_flash_toy_model_path, tmp_path, trust_remote_code=True)
