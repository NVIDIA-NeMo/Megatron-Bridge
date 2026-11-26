import math

import torch

from megatron.bridge.models.clm.decision_head import CLMHeadConfig, CLMHeadPair


def test_clm_head_scores_and_checkpoint_compatibility(tmp_path):
    config = CLMHeadConfig(
        hidden_size=8,
        width=6,
        depth=3,
        projection_dim=4,
        activation="gelu",
        layer_norm=True,
        residual=True,
    )
    head = config.build(initial_temperature=0.07)
    states = torch.randn(3, 8)
    actions = torch.randn(5, 8)

    logits = head(states, actions)

    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()
    assert "state_head.inp.weight" in head.state_dict()
    assert "action_head.out.bias" in head.state_dict()

    checkpoint = {
        "state_head": head.state_head.state_dict(),
        "action_head": head.action_head.state_dict(),
        "logit_scale": torch.tensor(math.log(1.0 / 0.07)),
        "cfg": {
            "hidden_size": 8,
            "width": 6,
            "depth": 3,
            "projection_dim": 4,
            "activation": "gelu",
            "layernorm": True,
            "residual": True,
        },
    }
    path = tmp_path / "head.pt"
    torch.save(checkpoint, path)

    restored = CLMHeadPair.from_upstream_checkpoint(path)

    assert restored.config == config
    for expected, actual in zip(head.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)


def test_clm_head_rejects_wrong_embedding_width():
    head = CLMHeadConfig(
        hidden_size=8,
        width=6,
        depth=2,
        projection_dim=4,
        activation="relu",
        layer_norm=False,
        residual=False,
    ).build()

    try:
        head(torch.randn(2, 7), torch.randn(2, 8))
    except ValueError as error:
        assert "state_embeddings" in str(error)
    else:
        raise AssertionError("wrong embedding width should fail")
