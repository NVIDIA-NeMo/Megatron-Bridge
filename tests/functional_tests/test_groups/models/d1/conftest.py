"""Single-process native Megatron setup for the focused D1 block tests."""

from collections.abc import Generator

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


@pytest.fixture
def native_process_groups(tmp_path) -> Generator[None, None, None]:
    """Initialize and release the real single-GPU Megatron process groups."""
    if not torch.cuda.is_available():
        pytest.skip("Transformer Engine D1 composition requires CUDA")
    if torch.distributed.is_initialized():
        raise RuntimeError("Run these standalone block tests in their own process")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        backend="nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(7)
        yield
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
