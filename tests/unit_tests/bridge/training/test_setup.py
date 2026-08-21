import pytest
import torch
import torch.nn as nn

from megatron.bridge.training.setup import _apply_peft_transformation


class _DummyModelChunk(nn.Module):
    def __init__(self, total, trainable):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(total), requires_grad=trainable)


class _DummyPEFT:
    def __call__(self, base_model, training=True):
        return base_model

    def set_params_to_save(self, model):
        pass


def test_apply_peft_transformation_counts_all_chunks():
    """PEFT parameter stats must account for every model chunk, not only chunk 0."""
    chunk0 = _DummyModelChunk(10, False)
    chunk1 = _DummyModelChunk(5, True)
    peft = _DummyPEFT()

    logs = []

    def fake_print_rank_0(msg):
        logs.append(msg)

    import megatron.bridge.training.setup as setup_module

    orig_print_rank_0 = setup_module.print_rank_0
    setup_module.print_rank_0 = fake_print_rank_0
    try:
        _apply_peft_transformation(peft, [chunk0, chunk1])
    finally:
        setup_module.print_rank_0 = orig_print_rank_0

    assert any("Total parameters: 15" in msg for msg in logs)
    assert any("Trainable parameters: 5" in msg for msg in logs)
