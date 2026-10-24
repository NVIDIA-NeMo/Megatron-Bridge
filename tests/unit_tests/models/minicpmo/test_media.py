import pytest
import torch

from megatron.bridge.models.minicpmo.media import insert_media_embeddings


def test_inserts_every_sample_and_preserves_gradients():
    text = torch.zeros(2, 6, 3, dtype=torch.double, requires_grad=True)
    first = torch.full((2, 3), 2.0, dtype=torch.double, requires_grad=True)
    second = torch.full((1, 3), 7.0, dtype=torch.double, requires_grad=True)
    bounds = [torch.tensor([[1, 3]]), torch.tensor([[4, 5]])]
    output, occupied = insert_media_embeddings(text, [[first], [second]], bounds)
    assert torch.equal(output[0, 1:3], first)
    assert torch.equal(output[1, 4:5], second)
    assert not text.detach().any()
    output.sum().backward()
    assert torch.equal(text.grad, (~occupied).unsqueeze(-1).expand_as(text).double())
    assert torch.equal(first.grad, torch.ones_like(first))
    assert torch.equal(second.grad, torch.ones_like(second))
    assert torch.autograd.gradcheck(
        lambda x, y, z: insert_media_embeddings(x, [[y], [z]], bounds)[0], (text, first, second)
    )


def test_multiple_modalities_are_disjoint():
    text = torch.zeros(1, 6, 2)
    image, occupied = insert_media_embeddings(text, [torch.ones(2, 2)], [torch.tensor([[1, 3]])])
    audio, occupied2 = insert_media_embeddings(
        image, [torch.full((1, 2), 3.0)], [torch.tensor([[4, 5]])], occupied=occupied
    )
    assert audio[0, 1:3].eq(1).all() and audio[0, 4].eq(3).all()
    assert occupied.sum() == 2 and occupied2.sum() == 3
    with pytest.raises(ValueError, match="overwrite"):
        insert_media_embeddings(image, [torch.ones(1, 2)], [torch.tensor([[2, 3]])], occupied=occupied)


@pytest.mark.parametrize("spans", [[[-1, 1]], [[2, 2]], [[3, 7]], [[2, 4], [3, 5]], [[4, 5], [1, 2]]])
def test_reject_invalid_spans(spans):
    with pytest.raises(ValueError, match="spans"):
        insert_media_embeddings(torch.zeros(1, 6, 2), [[]], [torch.tensor(spans)])


@pytest.mark.parametrize("features", [torch.zeros(1, 2), torch.zeros(2, 3)])
def test_reject_feature_shape_mismatch(features):
    with pytest.raises(ValueError):
        insert_media_embeddings(torch.zeros(1, 6, 2), [features], [torch.tensor([[1, 3]])])


def test_text_only_and_empty_batch():
    text = torch.randn(2, 6, 3, requires_grad=True)
    output, occupied = insert_media_embeddings(text, [[], []], [torch.empty(0, 2, dtype=torch.long)] * 2)
    assert torch.equal(text, output) and not occupied.any()
    output.sum().backward()
    assert text.grad.eq(1).all()
    empty, mask = insert_media_embeddings(torch.empty(0, 6, 3), [], [])
    assert empty.shape == (0, 6, 3) and mask.shape == (0, 6)


def test_equal_total_tokens_cannot_hide_wrong_per_clip_lengths():
    with pytest.raises(ValueError, match="own token span"):
        insert_media_embeddings(
            torch.zeros(1, 8, 2), [[torch.ones(1, 2), torch.ones(3, 2)]], [torch.tensor([[1, 3], [5, 7]])]
        )
