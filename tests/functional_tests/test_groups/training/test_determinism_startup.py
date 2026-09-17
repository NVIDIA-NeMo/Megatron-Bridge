"""Fresh-process checks of Bridge's public startup path on the GPU stack."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("case", ["provider", "builder", "late"])
def test_determinism_startup_in_fresh_process(case: str, tmp_path: Path) -> None:
    # Imports and first CUDA use must happen in the child, independently of the
    # parent pytest process's imports, fixtures, and previous test kernels.
    script = textwrap.dedent("""
        import json
        import logging
        import os
        import sys
        from pathlib import Path
        from unittest.mock import patch

        import torch
        from megatron.core import parallel_state
        from megatron.core.determinism import configure_determinism
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear
        from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
        from megatron.bridge.models.gpt_provider import GPTModelProvider
        from megatron.bridge.models.transformer_config import TransformerConfig
        from megatron.bridge.recipes.utils.determinism_utils import apply_determinism_overrides
        from megatron.bridge.training.config import (
            CheckpointConfig, ConfigContainer, LoggerConfig, MockGPTDatasetConfig,
            OptimizerConfig, SchedulerConfig, TokenizerConfig, TrainingConfig,
            ValidationConfig, runtime_config_update,
        )
        from megatron.bridge.training.initialize import initialize_megatron

        logging.basicConfig(level=logging.INFO)
        case, report_path = sys.argv[1:]
        assert not torch.cuda.is_initialized(), 'Imports initialized CUDA before policy setup'
        assert not torch.distributed.is_initialized()
        options = dict(
            num_layers=1, hidden_size=128, ffn_hidden_size=256, num_attention_heads=4,
            use_cpu_initialization=True, gradient_accumulation_fusion=False,
        )
        if case == 'builder':
            model = BridgeGPTModelConfig(
                transformer=TransformerConfig(**options), vocab_size=128, seq_length=32,
            )
        else:
            model = GPTModelProvider(**options, vocab_size=128, seq_length=32)
        cfg = ConfigContainer(
            model=model,
            train=TrainingConfig(train_iters=2, micro_batch_size=1, global_batch_size=1),
            optimizer=OptimizerConfig(lr=1e-3, use_distributed_optimizer=False),
            scheduler=SchedulerConfig(lr_decay_iters=2),
            dataset=MockGPTDatasetConfig(
                random_seed=1234, seq_length=32, reset_position_ids=False,
                reset_attention_mask=False, eod_mask_loss=False, num_workers=0,
            ),
            logger=LoggerConfig(),
            tokenizer=TokenizerConfig(tokenizer_type='NullTokenizer', vocab_size=128),
            checkpoint=CheckpointConfig(save=None, load=None),
            validation=ValidationConfig(eval_iters=0),
        )
        apply_determinism_overrides(cfg)

        if case == 'late':
            torch.cuda.init()
            try:
                runtime_config_update(cfg)
            except RuntimeError as error:
                assert 'before CUDA' in str(error)
            else:
                raise AssertionError('Accepted the first policy setup after CUDA initialization')
            Path(report_path).write_text(json.dumps({'case': case, 'late_setup_rejected': True}))
            sys.exit(0)

        finalizations = []
        original_finalize = type(model).finalize
        def checked_finalize(self):
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert torch.backends.cudnn.deterministic
            assert not torch.backends.cudnn.benchmark
            assert os.environ['NCCL_ALGO'] == 'Ring'
            assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            finalizations.append(True)
            return original_finalize(self)

        with patch.object(type(model), 'finalize', checked_finalize):
            runtime_config_update(cfg)
        assert finalizations
        groups = initialize_megatron(cfg)
        try:
            assert torch.cuda.is_initialized()
            assert torch.distributed.is_initialized()
            transformer = model.transformer if case == 'builder' else model
            layer = ColumnParallelLinear(
                32, 16, config=transformer, init_method=torch.nn.init.normal_, bias=True,
                gather_output=False, tp_group=groups.tp, pg_collection=groups,
            ).cuda()
            inputs = torch.randn(8, 1, 32, device='cuda', requires_grad=True)
            def replay():
                layer.zero_grad(set_to_none=True)
                inputs.grad = None
                output, _ = layer(inputs)
                output.square().mean().backward()
                tensors = (output, inputs.grad, layer.weight.grad, layer.bias.grad)
                assert all(t is not None and t.numel() for t in tensors)
                return [t.detach().clone() for t in tensors]
            first, second = replay(), replay()
            for a, b in zip(first, second, strict=True):
                assert torch.equal(
                    a.contiguous().reshape(-1).view(torch.uint8),
                    b.contiguous().reshape(-1).view(torch.uint8),
                )
            policy = configure_determinism(model)  # Unchanged repeated setup is valid.
            model.tp_comm_overlap = True
            try:
                initialize_megatron(cfg)
            except AssertionError as error:
                assert 'tp_comm_overlap' in str(error)
            else:
                raise AssertionError('Accepted a conflicting option before repeated initialization')
            model.tp_comm_overlap = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            try:
                initialize_megatron(cfg)
            except RuntimeError as error:
                assert 'before CUDA' in str(error)
            else:
                raise AssertionError('Accepted changed workspace policy after initialization')
            Path(report_path).write_text(json.dumps({
                'case': case, 'policy': policy, 'compared_tensors': len(first),
                'config_drift_rejected': True, 'environment_drift_rejected': True,
            }))
        finally:
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()
    """)
    # Keep transport/device configuration, but exercise unset policy defaults.
    policy_keys = {
        "NCCL_ALGO",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
        "CUBLAS_WORKSPACE_CONFIG",
        "MAMBA_DETERMINISTIC",
        "CAUSAL_CONV1D_DETERMINISTIC",
        "TRITON_CACHE_AUTOTUNING",
    }
    env = {key: value for key, value in os.environ.items() if key not in policy_keys}
    report_path = tmp_path / f"{case}.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=1",
            "--no_python",
            sys.executable,
            "-c",
            script,
            case,
            str(report_path),
        ],
        cwd=Path(__file__).resolve().parents[4],
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["case"] == case
    if case == "late":
        assert report["late_setup_rejected"]
    else:
        assert report["compared_tensors"] == 4
        assert report["config_drift_rejected"] and report["environment_drift_rejected"]
        assert report["policy"]["torch"]["deterministic_algorithms"]
