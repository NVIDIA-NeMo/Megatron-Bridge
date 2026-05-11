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

import json
import subprocess
from pathlib import Path

import pytest


# Minimal GPT-OSS config used for building tiny local HF directories to test conversion.
# hidden_size and intermediate_size are different (and both divisible by 32) so the per-expert
# down_proj/gate_up_proj are non-square and the bridge can detect orientation from shape.
GPT_OSS_TOY_OVERRIDES = {
    "architectures": ["GptOssForCausalLM"],
    "hidden_size": 512,
    "intermediate_size": 1536,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "num_hidden_layers": 2,
    "num_local_experts": 8,  # enable MoE to exercise EP handling
    "vocab_size": 32000,
    "torch_dtype": "bfloat16",
}


def _build_toy_models(bf16_dir: Path, mxfp4_dir: Path, seed: int = 0):
    """Build two faithful toy GPT-OSS checkpoints sharing the same underlying weights.

    BF16 toy:
        Stores per-expert ``gate_up_proj`` as ``[E, hidden, 2*intermediate]`` and
        ``down_proj`` as ``[E, intermediate, hidden]`` — matching ``unsloth/gpt-oss-20b-BF16``
        and what ``transformers.GptOssForCausalLM`` produces at init.

    MXFP4 toy:
        Stores ``*_blocks`` and ``*_scales`` whose dequantization (via
        ``_dequantize_mxfp4``) yields the BF16 toy's per-expert values *transposed*
        — i.e. ``[E, 2*intermediate, hidden]`` for gate_up_proj and ``[E, hidden,
        intermediate]`` for down_proj — matching how ``openai/gpt-oss-20b`` ships.

    The two toys are built so that BF16 == dequant(MXFP4).t(-1, -2) per expert,
    which means a Megatron checkpoint imported from either source must contain
    identical per-expert weights, and exporting that Megatron checkpoint back to
    HF format must match the BF16 toy on every tensor.
    """
    import torch
    from safetensors.torch import save_file
    from transformers import GptOssConfig, GptOssForCausalLM

    from megatron.bridge.models.gpt_oss.gpt_oss_bridge import _dequantize_mxfp4

    config = GptOssConfig(**GPT_OSS_TOY_OVERRIDES)
    model = GptOssForCausalLM(config).bfloat16()

    e = config.num_local_experts
    h = config.hidden_size
    i_ = config.intermediate_size
    num_layers = config.num_hidden_layers
    assert h % 32 == 0 and i_ % 32 == 0, "Both dims must be divisible by MXFP4 block size 32"

    gen = torch.Generator().manual_seed(seed)

    # Per-layer MXFP4 generation, then dequantize to obtain BF16 reference values.
    layer_data = []
    for _ in range(num_layers):
        gu_blocks = torch.randint(0, 256, (e, 2 * i_, h // 32, 16), dtype=torch.int32, generator=gen).to(torch.uint8)
        # UE8M0 scales clustered near 127 so dequantized magnitudes are O(1).
        gu_scales = torch.randint(124, 130, (e, 2 * i_, h // 32), dtype=torch.int32, generator=gen).to(torch.uint8)
        # Dequant returns (E, 2*intermediate, hidden) = (E, out, in).
        gu_dq = _dequantize_mxfp4(gu_blocks, gu_scales)
        # BF16 HF layout is (E, in, out) = (E, hidden, 2*intermediate).
        gu_bf16 = gu_dq.transpose(-1, -2).contiguous()

        dn_blocks = torch.randint(0, 256, (e, h, i_ // 32, 16), dtype=torch.int32, generator=gen).to(torch.uint8)
        dn_scales = torch.randint(124, 130, (e, h, i_ // 32), dtype=torch.int32, generator=gen).to(torch.uint8)
        # Dequant returns (E, hidden, intermediate) = (E, out, in).
        dn_dq = _dequantize_mxfp4(dn_blocks, dn_scales)
        # BF16 HF layout is (E, in, out) = (E, intermediate, hidden).
        dn_bf16 = dn_dq.transpose(-1, -2).contiguous()

        layer_data.append(
            {
                "gu_blocks": gu_blocks,
                "gu_scales": gu_scales,
                "gu_bf16": gu_bf16,
                "dn_blocks": dn_blocks,
                "dn_scales": dn_scales,
                "dn_bf16": dn_bf16,
            }
        )

    # Inject the dequantized values back into the BF16 model so its on-disk weights match
    # exactly what the MXFP4 path will produce after dequantization.
    sd = dict(model.state_dict())
    for li in range(num_layers):
        sd[f"model.layers.{li}.mlp.experts.gate_up_proj"] = layer_data[li]["gu_bf16"]
        sd[f"model.layers.{li}.mlp.experts.down_proj"] = layer_data[li]["dn_bf16"]
    model.load_state_dict(sd)

    # ---- BF16 toy ----
    bf16_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(bf16_dir, safe_serialization=True)
    with open(bf16_dir / "config.json", "w") as f:
        json.dump(model.config.to_dict(), f, indent=2)

    # ---- MXFP4 toy ----
    mxfp4_dir.mkdir(parents=True, exist_ok=True)
    mxfp4_sd = {}
    for n, p in model.state_dict().items():
        if n.endswith(".mlp.experts.gate_up_proj") or n.endswith(".mlp.experts.down_proj"):
            continue
        mxfp4_sd[n] = p.detach().clone().contiguous()
    for li in range(num_layers):
        prefix = f"model.layers.{li}.mlp.experts"
        mxfp4_sd[f"{prefix}.gate_up_proj_blocks"] = layer_data[li]["gu_blocks"]
        mxfp4_sd[f"{prefix}.gate_up_proj_scales"] = layer_data[li]["gu_scales"]
        mxfp4_sd[f"{prefix}.down_proj_blocks"] = layer_data[li]["dn_blocks"]
        mxfp4_sd[f"{prefix}.down_proj_scales"] = layer_data[li]["dn_scales"]
    save_file(mxfp4_sd, str(mxfp4_dir / "model.safetensors"))
    with open(mxfp4_dir / "config.json", "w") as f:
        json.dump(model.config.to_dict(), f, indent=2)

    # Tokenizer (best-effort; both toys share it)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.save_pretrained(bf16_dir)
        tok.save_pretrained(mxfp4_dir)
    except Exception:
        pass


class TestGptOssConversion:
    """Functional tests for GPT-OSS toy conversion paths.

    Two toys are built once per class:
      - ``bf16``: faithful BF16 layout (matches ``unsloth/gpt-oss-20b-BF16``).
      - ``mxfp4``: faithful MXFP4 layout (matches ``openai/gpt-oss-20b``).

    Each parallelism config is exercised against both sources. The MXFP4 path runs as a
    two-step convert→roundtrip because the roundtrip script's verification table cannot
    look up ``gate_up_proj``/``down_proj`` directly in a quantized state dict; instead we
    import MXFP4 → save Megatron → reload Megatron → export → compare against the BF16
    toy as the reference.
    """

    @pytest.fixture(scope="class")
    def gpt_oss_toy_paths(self, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("gptoss_toys")
        bf16_dir = tmp_dir / "gpt_oss_toy_bf16"
        mxfp4_dir = tmp_dir / "gpt_oss_toy_mxfp4"

        transformers = pytest.importorskip("transformers")
        if not all(hasattr(transformers, n) for n in ("GptOssForCausalLM", "GptOssConfig")):
            pytest.skip("transformers installation does not include GPT-OSS classes")

        _build_toy_models(bf16_dir, mxfp4_dir)
        return {"bf16": str(bf16_dir), "mxfp4": str(mxfp4_dir)}

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,ep,parallel_name",
        [
            (1, 2, 1, "PP"),
            (1, 1, 2, "EP"),
        ],
    )
    @pytest.mark.parametrize("source", ["bf16", "mxfp4"])
    def test_gpt_oss_conversion_parallelism(self, gpt_oss_toy_paths, tmp_path, tp, pp, ep, parallel_name, source):
        repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
        bf16_path = gpt_oss_toy_paths["bf16"]
        toy_path = gpt_oss_toy_paths[source]

        out_dir = tmp_path / f"gpt_oss_{source}_{parallel_name}"
        out_dir.mkdir(exist_ok=True)

        common_dist_args = [
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
        ]

        if source == "bf16":
            # Single-step roundtrip: import + export + compare against the source.
            cmd = common_dist_args + [
                "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
                "--hf-model-id",
                bf16_path,
                "--output-dir",
                str(out_dir),
                "--tp",
                str(tp),
                "--pp",
                str(pp),
                "--ep",
                str(ep),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            assert result.returncode == 0, f"GPT-OSS bf16 {parallel_name} roundtrip failed with rc={result.returncode}"
        else:
            # Two-step: (1) import MXFP4 -> save Megatron, (2) reload Megatron and export,
            # comparing exported HF tensors against the BF16 toy reference (which equals
            # dequant(MXFP4) by construction).
            mcore_dir = tmp_path / f"mcore_mxfp4_{parallel_name}"
            mcore_dir.mkdir(exist_ok=True)
            import_cmd = common_dist_args + [
                "examples/conversion/convert_checkpoints_multi_gpu.py",
                "import",
                "--hf-model",
                toy_path,
                "--megatron-path",
                str(mcore_dir),
                "--tp",
                str(tp),
                "--pp",
                str(pp),
                "--ep",
                str(ep),
            ]
            res = subprocess.run(import_cmd, capture_output=True, text=True, cwd=repo_root)
            if res.returncode != 0:
                print(f"STDOUT: {res.stdout}")
                print(f"STDERR: {res.stderr}")
            assert res.returncode == 0, f"GPT-OSS mxfp4 {parallel_name} import failed with rc={res.returncode}"

            roundtrip_cmd = common_dist_args + [
                "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
                "--hf-model-id",
                bf16_path,
                "--megatron-load-path",
                str(mcore_dir),
                "--output-dir",
                str(out_dir),
                "--tp",
                str(tp),
                "--pp",
                str(pp),
                "--ep",
                str(ep),
                "--skip-save",
            ]
            res = subprocess.run(roundtrip_cmd, capture_output=True, text=True, cwd=repo_root)
            if res.returncode != 0:
                print(f"STDOUT: {res.stdout}")
                print(f"STDERR: {res.stderr}")
            assert res.returncode == 0, (
                f"GPT-OSS mxfp4 {parallel_name} roundtrip-vs-bf16 failed with rc={res.returncode}"
            )
            # MXFP4 path uses --skip-save, so there's no exported HF directory to inspect; the
            # roundtrip script's internal verification table is the assertion of correctness.
            return

        # Verify output structure for the BF16 path.
        model_name = Path(bf16_path).name
        converted_dir = out_dir / model_name
        assert converted_dir.exists()

        config_file = converted_dir / "config.json"
        assert config_file.exists()

        weights_file_safetensors = converted_dir / "model.safetensors"
        weights_file_pytorch = converted_dir / "pytorch_model.bin"
        weights_found = weights_file_safetensors.exists() or weights_file_pytorch.exists()
        if not weights_found:
            shards_st = list(converted_dir.glob("model-*-of-*.safetensors"))
            shards_pt = list(converted_dir.glob("pytorch_model-*-of-*.bin"))
            weights_found = len(shards_st) > 0 or len(shards_pt) > 0
        assert weights_found

        with open(config_file) as f:
            saved = json.load(f)

        assert saved["num_hidden_layers"] == GPT_OSS_TOY_OVERRIDES["num_hidden_layers"]
        assert saved["num_attention_heads"] == GPT_OSS_TOY_OVERRIDES["num_attention_heads"]
        assert saved.get("num_local_experts", 0) == GPT_OSS_TOY_OVERRIDES["num_local_experts"]
        assert saved["vocab_size"] == GPT_OSS_TOY_OVERRIDES["vocab_size"]

        print(f"SUCCESS: GPT-OSS {source} {parallel_name} conversion test completed successfully")
        print(f"Converted model saved at: {converted_dir}")
