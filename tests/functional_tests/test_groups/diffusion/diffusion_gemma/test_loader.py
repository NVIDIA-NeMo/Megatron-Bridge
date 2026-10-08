"""Saved HF checkpoint loading, text-weight roundtrip, and pretrained SFT."""

import shutil
from pathlib import Path

import pytest
import torch

from tests.functional_tests.test_groups.diffusion.diffusion_gemma.test_sft import (
    _assert_hf_text_weights,
    _run_checkpoint_resume,
    _write_fixture,
)


pytestmark = pytest.mark.run_only_on("GPU")


def _write_hf_fixture(root: Path) -> Path:
    from transformers import (
        DiffusionGemmaConfig,
        DiffusionGemmaForBlockDiffusion,
        DiffusionGemmaTextConfig,
        Gemma4VisionConfig,
    )

    text = DiffusionGemmaTextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=16,
        num_global_key_value_heads=1,
        num_experts=2,
        top_k_experts=1,
        moe_intermediate_size=16,
        sliding_window=8,
        max_position_embeddings=32,
        pad_token_id=1,
        eos_token_id=3,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
        },
    )
    config = DiffusionGemmaConfig(
        text_config=text,
        canvas_length=4,
        eos_token_id=3,
        pad_token_id=1,
        vision_config=Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            position_embedding_size=16,
        ),
    )
    torch.manual_seed(42)
    model = DiffusionGemmaForBlockDiffusion(config).to(torch.bfloat16)
    model.model.encoder.language_model.layers[0].layer_scalar.fill_(0.75)
    model.model.decoder.layers[0].layer_scalar.fill_(1.25)
    destination = root / "hf_model"
    model.save_pretrained(destination)
    for artifact in (root / "tokenizer").iterdir():
        shutil.copyfile(artifact, destination / artifact.name)
    return destination


@pytest.mark.parametrize("grouped_experts", [False, True])
def test_saved_hf_text_weights_and_forward(tmp_path, grouped_experts):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.utils import unwrap_model
    from transformers import DiffusionGemmaForBlockDiffusion

    from megatron.bridge import AutoBridge

    _write_fixture(tmp_path)
    checkpoint = _write_hf_fixture(tmp_path)
    torch.distributed.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(7)
    try:
        bridge = AutoBridge.from_hf_pretrained(checkpoint)
        provider = bridge.to_megatron_provider(load_weights=True)
        provider.make_vocab_size_divisible_by = 1
        provider.moe_grouped_gemm = grouped_experts
        provider.moe_permute_fusion = False
        provider.gradient_accumulation_fusion = False
        provider.finalize()
        models = provider.provide_distributed_model(wrap_with_ddp=False)
        assert len(models) == 1
        model = unwrap_model(models[0]).eval()
        _assert_hf_text_weights(model, checkpoint, bridge=bridge)

        reference = DiffusionGemmaForBlockDiffusion.from_pretrained(checkpoint, dtype=torch.bfloat16).cuda().eval()
        prefix = torch.tensor([[4, 5]], device="cuda")
        noisy = torch.tensor([[6, 7, 8, 9]], device="cuda")
        with torch.no_grad():
            preview = reference(input_ids=prefix, decoder_input_ids=noisy)
            expected = reference(input_ids=prefix, decoder_input_ids=noisy, self_conditioning_logits=preview.logits)
            clean = torch.cat((prefix, noisy), dim=1)
            _, actual = model(
                clean,
                torch.arange(6, device="cuda")[None],
                noisy_tokens=noisy,
                canvas_positions=torch.arange(2, 6, device="cuda")[None],
                encoder_valid_mask=torch.ones_like(clean, dtype=torch.bool),
                prefix_lengths=torch.tensor([2], device="cuda"),
            )
        similarity = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.logits.float().flatten(), dim=0
        )
        assert similarity.item() >= 0.99
        assert torch.equal(actual.argmax(-1), expected.logits.argmax(-1))
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs for pretrained DP resume")
def test_hf_initialized_sft_resume(tmp_path):
    _write_fixture(tmp_path)
    _write_hf_fixture(tmp_path)
    _run_checkpoint_resume(tmp_path)
