import random
from collections.abc import Callable, Iterator

import numpy as np
import pytest
import torch

from megatron.bridge.models.bagel.data.energon import BagelT2ISample, BagelVLMSample
from megatron.bridge.models.bagel.data.packing import BagelPacker, BagelSample


pytestmark = pytest.mark.unit


def _t2i(row: int) -> BagelT2ISample:
    return BagelT2ISample(
        __key__=f"t2i-{row}",
        __restore_key__=("t2i", row),
        __subflavor__=None,
        __subflavors__={},
        image_tensor_list=[torch.full((3, 16, 16), row / 10)],
        text_ids_list=[[10, row]],
        num_tokens=3,
        sequence_plan=[
            {"type": "text", "enable_cfg": 0, "loss": 0, "special_token_loss": 0, "special_token_label": None},
            {
                "type": "vae_image",
                "enable_cfg": 0,
                "loss": 1,
                "special_token_loss": 0,
                "special_token_label": None,
            },
        ],
        metadata={"dataset_group": "t2i_pretrain", "source": {"parquet": "t2i.parquet", "row": row}},
    )


def _vlm(row: int) -> BagelVLMSample:
    return BagelVLMSample(
        __key__=f"vlm-{row}",
        __restore_key__=("vlm", row),
        __subflavor__=None,
        __subflavors__={},
        image_tensor_list=[torch.full((3, 14, 14), row / 10)],
        text_ids_list=[[20, row]],
        num_tokens=3,
        sequence_plan=[
            {
                "type": "vit_image",
                "enable_cfg": 0,
                "loss": 0,
                "special_token_loss": 0,
                "special_token_label": None,
            },
            {"type": "text", "enable_cfg": 0, "loss": 1, "special_token_loss": 0, "special_token_label": None},
        ],
        metadata={"dataset_group": "vlm_sft", "source": {"jsonl": "vlm.jsonl", "row": row}},
    )


def _repeat(factory: Callable[[int], BagelSample]) -> Iterator[BagelSample]:
    row = 0
    while True:
        yield factory(row)
        row += 1


class _Rows:
    def __init__(self, factory: Callable[[int], BagelSample], start: int = 0) -> None:
        self.factory = factory
        self.position = start

    def __iter__(self) -> "_Rows":
        return self

    def __next__(self) -> BagelSample:
        sample = self.factory(self.position)
        self.position += 1
        return sample


def _assert_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_packer_keeps_mandatory_sample_and_reuses_fifo_buffer() -> None:
    random.seed(42)
    np.random.seed(42)
    packer = BagelPacker(
        [_repeat(_t2i), _repeat(_vlm)],
        [0, 1],
        [True, False],
        {"bos_token_id": 1, "eos_token_id": 2, "start_of_image": 3, "end_of_image": 4},
        expected_num_tokens=100,
        max_num_tokens_per_sample=20,
        max_num_tokens=20,
        prefer_buffer_before=10,
        max_buffer_size=1,
        text_cond_dropout_prob=0,
        vit_cond_dropout_prob=0,
        vae_cond_dropout_prob=0,
    )

    batches = iter(packer)
    first = next(batches)
    second = next(batches)

    assert first["source_ids"] == [
        {"dataset_name": "t2i_pretrain", "source": {"parquet": "t2i.parquet", "row": 0}},
        {"dataset_name": "vlm_sft", "source": {"jsonl": "vlm.jsonl", "row": 0}},
    ]
    assert second["source_ids"] == [
        {"dataset_name": "t2i_pretrain", "source": {"parquet": "t2i.parquet", "row": 1}},
        {"dataset_name": "vlm_sft", "source": {"jsonl": "vlm.jsonl", "row": 1}},
    ]
    assert first["sample_lens"] == [7, 7]
    assert first["packed_timesteps"].item() == pytest.approx(0.4967141530112327)


def test_packer_state_restores_buffer_and_rng() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    t2i_rows = _Rows(_t2i)
    vlm_rows = _Rows(_vlm)
    kwargs = {
        "expected_num_tokens": 100,
        "max_num_tokens_per_sample": 20,
        "max_num_tokens": 20,
        "prefer_buffer_before": 10,
        "max_buffer_size": 1,
        "text_cond_dropout_prob": 0,
        "vit_cond_dropout_prob": 0,
        "vae_cond_dropout_prob": 0,
    }
    packer = BagelPacker(
        [t2i_rows, vlm_rows],
        [0, 1],
        [True, False],
        {"bos_token_id": 1, "eos_token_id": 2, "start_of_image": 3, "end_of_image": 4},
        **kwargs,
    )
    next(packer)
    state = packer.state_dict()
    positions = (t2i_rows.position, vlm_rows.position)
    expected = next(packer)

    restored = BagelPacker(
        [_Rows(_t2i, positions[0]), _Rows(_vlm, positions[1])],
        [0, 1],
        [True, False],
        {"bos_token_id": 1, "eos_token_id": 2, "start_of_image": 3, "end_of_image": 4},
        **kwargs,
    )
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    restored.load_state_dict(state)

    _assert_equal(next(restored), expected)
