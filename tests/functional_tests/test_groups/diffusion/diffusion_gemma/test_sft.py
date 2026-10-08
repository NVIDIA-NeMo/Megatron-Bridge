"""Two-rank BF16 offline SFT and exact native checkpoint resume on a tiny model."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _write_fixture(root: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3}
    vocab.update({f"word{i}": i for i in range(4, 32)})
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>", eos_token="<eos>"
    )
    tokenizer.save_pretrained(root / "tokenizer")
    data = root / "data"
    data.mkdir()
    rows = [{"input": "word4 word5", "output": " ".join(f"word{6 + j}" for j in range(2 + i % 5))} for i in range(32)]
    for split in ("training", "validation", "test"):
        (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def _assert_hf_text_weights(model, checkpoint: Path, *, bridge=None) -> None:
    from safetensors.torch import load_file

    from megatron.bridge import AutoBridge

    if bridge is None:
        bridge = AutoBridge.from_hf_pretrained(checkpoint)
    source = load_file(checkpoint / "model.safetensors")
    exported = dict(bridge.export_hf_weights([model], cpu=True, show_progress=False))
    text_keys = {
        key for key in source if key.startswith("model.decoder.") or key.startswith("model.encoder.language_model.")
    }
    assert text_keys <= exported.keys()
    for key in text_keys:
        torch.testing.assert_close(exported[key], source[key], rtol=0, atol=0, msg=key)
    assert "model.decoder.layers.1.self_attn.v_proj.weight" not in exported
    assert model.decoder.layers[0].encoder_layer_scalar.item() == 0.75
    assert model.decoder.layers[0].layer_scalar.item() == 1.25


def _assert_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs for DP checkpoint coverage")
def test_sft_checkpoint_resume(tmp_path):
    """Match model, optimizer, scheduler, and sampler progress after a step-2 restart."""
    _write_fixture(tmp_path)
    _run_checkpoint_resume(tmp_path)


def _run_checkpoint_resume(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "WANDB_MODE": "disabled",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    }
    for run, steps in (("baseline", 4), ("resumed", 2), ("resumed", 4)):
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "tests.functional_tests.test_groups.diffusion.diffusion_gemma.test_sft",
            str(tmp_path),
            run,
            str(steps),
        ]
        result = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[5], env=env, capture_output=True, text=True, timeout=600
        )
        assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        baseline = torch.load(tmp_path / "baseline" / f"state_rank{rank}.pt", map_location="cpu", weights_only=True)
        resumed = torch.load(tmp_path / "resumed" / f"state_rank{rank}.pt", map_location="cpu", weights_only=True)
        _assert_equal(resumed, baseline)
        assert resumed["step"] == 4
        assert resumed["consumed_train_samples"] == 16


def _train_fixture(root: Path, run: str, steps: int) -> None:
    from megatron.core.activations import fast_gelu
    from megatron.core.utils import unwrap_model

    from megatron.bridge.diffusion.models.diffusion_gemma.provider import DiffusionGemmaModelProvider
    from megatron.bridge.diffusion.models.diffusion_gemma.step import DiffusionGemmaStep
    from megatron.bridge.diffusion.recipes.diffusion_gemma.sft import diffusion_gemma_sft_config
    from megatron.bridge.training.callbacks import Callback, CallbackContext
    from megatron.bridge.training.pretrain import pretrain
    from tests.functional_tests.utils import verify_checkpoint_files

    output = root / run
    hf_source = root / "hf_model"
    initialization = (
        {"hf_model": str(hf_source)}
        if hf_source.is_dir()
        else {
            "allow_random_init": True,
            "model": DiffusionGemmaModelProvider(
                num_layers=2,
                hidden_size=32,
                ffn_hidden_size=32,
                num_attention_heads=4,
                num_query_groups=2,
                kv_channels=8,
                global_head_dim=8,
                num_global_key_value_heads=1,
                global_rotary_percent=0.5,
                vocab_size=32,
                make_vocab_size_divisible_by=1,
                num_moe_experts=2,
                moe_router_topk=1,
                moe_ffn_hidden_size=16,
                moe_shared_expert_intermediate_size=32,
                moe_grouped_gemm=False,
                moe_permute_fusion=False,
                window_size=8,
                interleaved_attn_pattern=["sliding_attention", "full_attention"],
                gradient_accumulation_fusion=True,
                deterministic_mode=True,
            ),
        }
    )
    cfg = diffusion_gemma_sft_config(
        dataset_root=str(root / "data"),
        tokenizer_model=str(root / "tokenizer"),
        pretrained_checkpoint=None,
        experiment_dir=str(output),
        seq_length=32,
        global_batch_size=4,
        micro_batch_size=1,
        train_iters=steps,
        **initialization,
    )
    cfg.model.moe_grouped_gemm = False
    cfg.model.moe_permute_fusion = False
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.deterministic_mode = True
    # Pin the eager expression: compiled GELU specializes on expert token-count history.
    cfg.model.activation_func = fast_gelu.__wrapped__
    cfg.dataset.max_train_samples = 64
    cfg.dataset.num_workers = 0
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.dataset_kwargs["tokens_to_generate"] = 0
    cfg.validation.eval_iters = 0
    cfg.scheduler.lr_decay_iters = 4
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.override_opt_param_scheduler = True
    cfg.optimizer.lr = 1e-3
    cfg.optimizer.min_lr = 1e-3
    cfg.optimizer.use_distributed_optimizer = False
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.checkpoint.save_interval = 2
    cfg.checkpoint.async_save = False
    cfg.logger.tensorboard_dir = None

    class VerifyTraining(Callback):
        def on_train_start(self, context: CallbackContext) -> None:
            model = unwrap_model(context.model[0])
            if hf_source.is_dir() and context.state.train_state.step == 0:
                _assert_hf_text_weights(model, hf_source)
            self.initial_backbone = {
                name: value.detach().clone()
                for name, value in model.named_parameters()
                if name in ("embedding.word_embeddings.weight", "decoder.layers.0.self_attention.linear_qkv.weight")
            }
            assert len(self.initial_backbone) == 2
            self.initial_conditioning = model.self_conditioning.down_proj.weight.detach().clone()
            self.routers = {
                name: value.detach().clone() for name, value in model.named_parameters() if ".router." in name
            }

        def on_train_step_end(self, context: CallbackContext) -> None:
            assert not context.skipped_iter
            assert (
                context.grad_norm is not None
                and context.grad_norm > 0
                and torch.isfinite(torch.tensor(context.grad_norm))
            )
            assert {"diffusion loss", "encoder loss"} <= context.loss_dict.keys()
            assert all(torch.isfinite(value).all() for value in context.loss_dict.values())

        def on_train_end(self, context: CallbackContext) -> None:
            verify_checkpoint_files(str(output / "checkpoints"), steps)
            model = unwrap_model(context.model[0])
            assert not torch.equal(model.self_conditioning.down_proj.weight, self.initial_conditioning)
            parameters = dict(model.named_parameters())
            for name, before in self.initial_backbone.items():
                assert not torch.equal(parameters[name], before), name
            for name, before in self.routers.items():
                torch.testing.assert_close(parameters[name], before, rtol=0, atol=0)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": context.optimizer.state_dict(),
                    "scheduler": context.scheduler.state_dict(),
                    "step": context.state.train_state.step,
                    "consumed_train_samples": context.state.train_state.consumed_train_samples,
                },
                output / f"state_rank{torch.distributed.get_rank()}.pt",
            )

    pretrain(cfg, DiffusionGemmaStep(canvas_length=4), callbacks=[VerifyTraining()])


if __name__ == "__main__":
    _train_fixture(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
