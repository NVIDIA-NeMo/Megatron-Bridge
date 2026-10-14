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

"""One-rank toy VLM/text_only import, forward/backward and exact export smoke."""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist

from tests.functional_tests.test_groups.models.qwen.qwen4_exp_parity import (
    HF_QWEN4_EXP_TOY_MODEL_CONFIG,
    _load_safetensors,
)


def main() -> None:
    """Exercise changed construction/conversion paths without training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MCORE_QSA_SPARSE_BACKEND", "dense_masked")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

    from megatron.bridge import AutoBridge

    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    text_config = dict(HF_QWEN4_EXP_TOY_MODEL_CONFIG)
    text_config["rope_parameters"] = dict(
        text_config["rope_parameters"], mrope_section=[2, 1, 1], mrope_interleaved=True
    )
    hf_config = Qwen4ExpConfig(
        text_config=text_config,
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": 64,
            "hidden_act": "gelu_pytorch_tanh",
            "in_channels": 3,
        },
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
        tie_word_embeddings=False,
    )
    hf_model = Qwen4ExpForConditionalGeneration(hf_config)
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if param.ndim and "A_log" not in name and "dt_bias" not in name and param.abs().max() == 0:
                param.normal_(0, 0.1)
            if "ple.conv1d" in name:
                param.normal_(0, 0.3)
    source_dir = args.output_dir / "source"
    hf_model.save_pretrained(source_dir, safe_serialization=True)
    del hf_model
    source = _load_safetensors(str(source_dir))
    for text_only in (False, True):
        bridge = AutoBridge.from_hf_pretrained(str(source_dir), text_only=text_only)
        config = bridge.get_model_config()
        config.params_dtype = torch.float32
        config.bf16 = config.fp16 = False
        config.pipeline_dtype = config.autocast_dtype = torch.float32
        config.attention_backend = AttnBackend.unfused
        config.gradient_accumulation_fusion = False
        config.moe_router_dtype = "fp32"
        config.variable_seq_lengths = True
        config.qsa_force_sparse = True
        models = bridge.get_model(config, wrap_with_ddp=False)
        model = models[0].train()
        tokens = torch.randint(3, 240, (2, 40), device="cuda")
        kwargs = {}
        if not text_only:
            tokens[:, 2:5] = torch.tensor([252, 250, 253], device="cuda")
            kwargs = {
                "pixel_values": torch.randn(8, 24, device="cuda"),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]], device="cuda"),
            }
        positions = torch.arange(40, device="cuda").expand(2, -1)
        logits = model(tokens, positions, None, runtime_gather_output=True, **kwargs).float()
        assert torch.isfinite(logits).all(), "Nonfinite logits"
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.roll(-1, 1).flatten())
        assert torch.isfinite(loss), "Nonfinite loss"
        loss.backward()
        groups = {
            "embedding": "word_embeddings",
            "ple": ".per_layer_embedding.",
            "gdn": "A_log",
            "gates": "input_mix_weight",
        }
        if not text_only:
            groups["vision"] = "vision_model."
        for group, fragment in groups.items():
            gradients = [p.grad for name, p in model.named_parameters() if fragment in name and p.grad is not None]
            assert gradients, f"Missing {group} gradients ({fragment})"
            assert all(torch.isfinite(g).all() for g in gradients), f"Nonfinite {group} gradients"
            assert any(g.abs().max() > 0 for g in gradients), f"Zero {group} gradients"
        export_dir = args.output_dir / ("text_export" if text_only else "vlm_export")
        bridge.save_hf_weights(models, str(export_dir), show_progress=False)
        exported = _load_safetensors(str(export_dir))
        expected = source
        if text_only:
            expected = {
                name.replace("model.language_model.", "model.", 1): tensor
                for name, tensor in source.items()
                if name.startswith("model.language_model.") or name == "lm_head.weight"
            }
        assert exported.keys() == expected.keys(), (
            exported.keys() - expected.keys(),
            expected.keys() - exported.keys(),
        )
        assert all(torch.equal(exported[name], tensor) for name, tensor in expected.items()), "Export tensor mismatch"
        logging.info(
            "SMOKE OK text_only=%s model=%s loss=%.6f tensors=%s gradients=%s",
            text_only,
            type(model).__name__,
            loss.item(),
            len(exported),
            list(groups),
        )
        del model, models, bridge, logits, loss
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
