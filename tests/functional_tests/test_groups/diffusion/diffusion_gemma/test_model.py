"""CUDA native stack gradients and distributed checkpoint roundtrip."""

import pytest
import torch

from megatron.bridge.diffusion.models.diffusion_gemma.modeling_diffusion_gemma import build_attention_masks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native Gemma4 TE/MoE requires CUDA")
def test_native_three_pass_gradient_and_sharded_checkpoint(tmp_path):
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing import load, save
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from megatron.bridge.diffusion.models.diffusion_gemma.provider import DiffusionGemmaModelProvider

    torch.distributed.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(7)

    def provider():
        return DiffusionGemmaModelProvider(
            num_layers=2,
            hidden_size=32,
            ffn_hidden_size=64,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=8,
            global_head_dim=8,
            num_global_key_value_heads=1,
            global_rotary_percent=0.5,
            seq_length=8,
            vocab_size=32,
            make_vocab_size_divisible_by=1,
            num_moe_experts=2,
            moe_router_topk=1,
            moe_ffn_hidden_size=16,
            moe_shared_expert_intermediate_size=32,
            moe_grouped_gemm=False,
            moe_permute_fusion=False,
            gradient_accumulation_fusion=False,
            window_size=4,
            interleaved_attn_pattern=["sliding_attention", "full_attention"],
        )

    try:
        model = provider().provide().cuda()
        model.decoder.layers[0].encoder_layer_scalar.fill_(0.75)
        model.decoder.layers[0].layer_scalar.fill_(1.25)
        ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
        pos = torch.arange(5, device="cuda")[None]
        canvas = torch.tensor([[2, 3]], device="cuda")
        valid = torch.ones_like(ids, dtype=torch.bool)
        prefix = torch.tensor([2], device="cuda")
        calls = []
        handle = model.self_conditioning.register_forward_hook(
            lambda module, args, output: calls.append(output.requires_grad)
        )
        encoder_logits, decoder_logits = model(
            ids,
            pos,
            noisy_tokens=ids[:, 2:4],
            canvas_positions=canvas,
            encoder_valid_mask=valid,
            prefix_lengths=prefix,
        )
        handle.remove()
        assert encoder_logits.shape == (1, 5, 32)
        assert decoder_logits.shape == (1, 2, 32)
        assert calls == [False, True]
        decoder_logits.float().square().mean().backward()
        assert model.self_conditioning.down_proj.weight.grad.abs().sum() > 0
        assert model.embedding.word_embeddings.weight.grad.abs().sum() > 0
        assert all(
            parameter.grad is None for layer in model.decoder.layers for parameter in layer.mlp.router.parameters()
        )

        # A decoder-only loss must differentiate through the original clean KV.
        model.zero_grad(set_to_none=True)
        encoder_masks, decoder_masks = build_attention_masks(pos, canvas, valid, prefix, 4)
        _, kvs = model._stack(model.embedding(ids, pos), pos, encoder_masks)
        for key, value in kvs:
            key.retain_grad()
            value.retain_grad()
        noisy = model.embedding(ids[:, 2:4], canvas)
        decoded, _ = model._stack(model.self_conditioning(noisy, torch.zeros_like(noisy)), canvas, decoder_masks, kvs)
        model._logits(decoded).float().square().mean().backward()
        assert all(key.grad[:, :, :, :].abs().sum() > 0 and value.grad.abs().sum() > 0 for key, value in kvs)

        # Save includes heterogeneous attention, both scalar buffers, and AnalogBits.
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        save(model.sharded_state_dict(), checkpoint, async_sharded_save=False)
        restored = provider().provide().cuda()
        restored.load_state_dict(load(restored.sharded_state_dict(), checkpoint))
        for name, tensor in model.state_dict().items():
            if torch.is_tensor(tensor):
                torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
