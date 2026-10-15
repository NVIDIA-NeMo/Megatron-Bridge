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

"""Exhaustive Hugging Face to Megatron to Hugging Face round trip of a toy Muse Spark Open checkpoint.

The toy keeps every component of the complete model: the sliding-window and full-attention
decoder schedule with per-layer NoPE, latent MoE experts with per-expert normalization, the
residual gates, the separate LM head, and the Hugging Face vision tower, vision adapter and
projection, and audio tower. Only the sizes are reduced.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open


# Skip each test rather than the module: a module-level skip leaves the launcher with no collected tests, and
# pytest then exits with code 5, which fails the L0 job.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers.models.muse_spark") is None,
    reason="The installed Transformers release does not include Muse Spark Open.",
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_TEXT_CONFIG = {
    "num_hidden_layers": 4,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1024,
    "max_position_embeddings": 4096,
    "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    # Full-attention layers are NoPE in the released checkpoint.
    "layer_rope_theta": [500000.0, 0, 500000.0, 0],
    "mlp_layer_types": ["sparse"] * 4,
    "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0},
    "sliding_window": 64,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "num_shared_experts": 0,
    "moe_hidden_size": 32,
    "moe_intermediate_size": 64,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
}
_VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "pos_emb_height": 8,
    "pos_emb_width": 8,
    "max_position_embeddings": 64,
}
_AUDIO_CONFIG = {
    "patch_sizes": [16, 6],
    "hidden_sizes": [16, 32],
    "depths": [1, 1],
    "num_hidden_layers": 2,
}


@pytest.fixture(scope="module")
def muse_spark_toy_model_path(tmp_path_factory) -> str:
    """Save a randomly initialized toy Muse Spark Open checkpoint in the released on-disk layout."""
    from transformers.models import muse_spark

    config = muse_spark.MuseSparkConfig(
        text_config=_TEXT_CONFIG,
        vision_config=_VISION_CONFIG,
        audio_config=_AUDIO_CONFIG,
        # The vision tower merges 2x2 patches, so its output width is four times its hidden size.
        out_hidden_size=4 * _VISION_CONFIG["hidden_size"],
        projector_hidden_size=48,
        image_token_id=1000,
        video_token_id=1001,
        audio_token_id=1002,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = muse_spark.MuseSparkForConditionalGeneration(config).to(torch.bfloat16)
    model_dir = tmp_path_factory.mktemp("muse_spark_toy") / "muse_spark_toy"
    model.save_pretrained(model_dir)
    return str(model_dir)


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    ("tp", "pp", "ep"),
    [(2, 1, 1), (1, 2, 1), (1, 1, 2)],
    ids=["tp2", "pp2", "ep2"],
)
def test_muse_spark_conversion_roundtrip(muse_spark_toy_model_path: str, tp: int, pp: int, ep: int) -> None:
    """Every exported tensor must match the source checkpoint after import at each layout."""
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
        "-m",
        "coverage",
        "run",
        f"--data-file={_REPO_ROOT / '.coverage'}",
        f"--source={_REPO_ROOT}",
        "--parallel-mode",
        "scripts/conversion/run_conversion.py",
        "roundtrip",
        "--device",
        "gpu",
        "--hf-model",
        muse_spark_toy_model_path,
        "--tp",
        str(tp),
        "--pp",
        str(pp),
        "--ep",
        str(ep),
        "--etp",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    assert result.returncode == 0, f"Muse Spark Open round trip failed at TP{tp}/PP{pp}/EP{ep}"
    output = result.stdout + result.stderr
    assert "GPU round-trip validation complete" in output

    # The comparison must cover every tensor in the source checkpoint, not only those that were exported.
    source_tensors = 0
    for shard in Path(muse_spark_toy_model_path).glob("*.safetensors"):
        with safe_open(shard, framework="pt") as handle:
            source_tensors += len(list(handle.keys()))
    verification = re.search(r"Weight verification: (\d+)/(\d+) compared weights matched", output)
    assert verification is not None, "Round trip did not report its weight verification"
    matched, compared = map(int, verification.groups())
    assert matched == compared == source_tensors
