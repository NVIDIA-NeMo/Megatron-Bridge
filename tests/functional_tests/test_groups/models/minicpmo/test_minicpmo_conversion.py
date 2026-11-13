# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU checkpoint regression using native MiniCPM-o modules at toy depth.

Set MINICPMO_HF_SOURCE to a local copy of openbmb/MiniCPM-o-4_5 to run offline.
Only config, custom code, and tokenizer assets are needed to create the toy model.
The two worker phases run in fresh processes so loader topology is not inherited.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def _set_declared(config, overrides):
    for name, value in overrides.items():
        if not hasattr(config, name):
            raise ValueError(f"Missing model configuration field: {name}")
        setattr(config, name, value)


def _create_toy(source: str, destination: Path) -> None:
    config = AutoConfig.from_pretrained(source, trust_remote_code=True)
    _set_declared(
        config,
        dict(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            init_tts=False,
            use_cache=False,
            stream_input=False,
        ),
    )
    _set_declared(config, {"layer_types": config.layer_types[:2]})
    _set_declared(
        config.vision_config,
        dict(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4),
    )
    _set_declared(
        config.audio_config,
        dict(d_model=128, encoder_ffn_dim=512, encoder_layers=2, encoder_attention_heads=4),
    )
    torch.manual_seed(1234)
    model = AutoModel.from_config(config, trust_remote_code=True).to(torch.bfloat16)
    model.save_pretrained(destination, safe_serialization=True)
    AutoTokenizer.from_pretrained(source, trust_remote_code=True).save_pretrained(destination)


def _worker(phase: str, source: str, output: str) -> None:
    from megatron.core import parallel_state
    from megatron.core.transformer.enums import AttnBackend
    from safetensors import safe_open

    from megatron.bridge import AutoBridge

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    try:
        destination = Path(output)
        if phase == "save":
            bridge = AutoBridge.from_hf_pretrained(source, trust_remote_code=True)
            config = bridge.get_model_config()
            config.tensor_model_parallel_size = 2
            config.sequence_parallel = False
            config.bf16 = True
            config.params_dtype = torch.bfloat16
            config.gradient_accumulation_fusion = False
            config.attention_backend = AttnBackend.unfused
            models = bridge.get_model(config, load_weights=True, wrap_with_ddp=False)
            bridge.save_megatron_model(
                models, destination, hf_tokenizer_path=source, hf_tokenizer_kwargs={"trust_remote_code": True}
            )
        elif phase in ("reload_tp1", "reload_tp2"):
            reload_tp = int(phase[-1])
            bridge = AutoBridge.from_auto_config(output, source, trust_remote_code=True)
            models = bridge.load_megatron_model(
                destination, mp_overrides={"tensor_model_parallel_size": reload_tp}, wrap_with_ddp=False
            )
            expected = {}
            for shard in Path(source).glob("*.safetensors"):
                with safe_open(shard, framework="pt", device="cpu") as weights:
                    expected.update({name: weights.get_tensor(name) for name in weights.keys()})
            seen = set()
            for name, value in bridge.export_hf_weights(models, cpu=True):
                assert name in expected, name
                reference = expected[name]
                assert value.shape == reference.shape, name
                assert value.dtype == reference.dtype, name
                assert torch.equal(value, reference), name
                seen.add(name)
            if torch.distributed.get_rank() == 0:
                assert seen == expected.keys(), (seen - expected.keys(), expected.keys() - seen)
        else:
            raise ValueError(phase)
        assert parallel_state.get_tensor_model_parallel_world_size() == (2 if phase == "save" else reload_tp)
        assert parallel_state.get_pipeline_model_parallel_world_size() == 1
        torch.distributed.barrier()
    finally:
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_persisted_tp2_roundtrip(tmp_path):
    source = os.environ.get("MINICPMO_HF_SOURCE", "openbmb/MiniCPM-o-4_5")
    toy = tmp_path / "toy"
    output = tmp_path / "megatron"
    _create_toy(source, toy)
    for phase in ("save", "reload_tp1", "reload_tp2"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(Path(__file__).resolve()),
                phase,
                str(toy),
                str(output),
            ],
            check=True,
            timeout=300,
        )


if __name__ == "__main__":
    _worker(*sys.argv[1:])
