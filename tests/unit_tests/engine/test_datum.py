import pytest
import torch

from megatron.bridge.engine import Datum, check_datums


def _datum(n: int, **extra) -> Datum:
    ids = torch.arange(n + 1, dtype=torch.int64)
    return Datum(input_ids=ids[:-1], target_ids=ids[1:], loss_mask=torch.ones(n), **extra)


def test_accepts_consistent_batch_and_counts_tokens():
    datums = [_datum(4), _datum(2)]
    assert check_datums(datums) is datums
    assert [d.num_tokens for d in datums] == [4, 2]


@pytest.mark.parametrize("datums", [None, []])
def test_rejects_none_and_empty(datums):
    with pytest.raises(ValueError):
        check_datums(datums)


def test_rejects_non_datum_and_bad_dtypes_and_shapes():
    with pytest.raises(TypeError):
        check_datums([torch.zeros(3, dtype=torch.int64)])
    bad = _datum(3)
    bad.input_ids = bad.input_ids.to(torch.int32)
    with pytest.raises(ValueError, match="int64"):
        check_datums([bad])
    bad = _datum(3)
    bad.target_ids = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError, match="shape \\[3\\]"):
        check_datums([bad])
    bad = _datum(3)
    bad.loss_mask = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="shape"):
        check_datums([bad])


def test_common_fields_all_or_none_and_token_aligned():
    ok = [_datum(3, advantages=torch.zeros(3)), _datum(2, advantages=torch.ones(2))]
    check_datums(ok)
    with pytest.raises(ValueError, match="advantages must be set for every Datum"):
        check_datums([_datum(3, advantages=torch.zeros(3)), _datum(2)])
    with pytest.raises(ValueError, match="old_logprobs must have shape"):
        check_datums([_datum(3, old_logprobs=torch.zeros(2))])


def test_routed_experts_all_or_none_integer_and_layer_aligned():
    routed = torch.zeros(3, 4, 2, dtype=torch.uint8)  # any integer dtype; [L, num_moe_layers, topk]
    check_datums([_datum(3, routed_experts=routed), _datum(2, routed_experts=routed[:2])])
    with pytest.raises(ValueError, match="routed_experts must be set for every Datum"):
        check_datums([_datum(3, routed_experts=routed), _datum(2)])
    with pytest.raises(ValueError, match="same trailing dims"):
        check_datums([_datum(3, routed_experts=routed), _datum(2, routed_experts=torch.zeros(2, 4, 3))])
    with pytest.raises(ValueError, match="num_moe_layers, topk"):
        check_datums([_datum(3, routed_experts=torch.zeros(3, 4))])
    with pytest.raises(ValueError, match="integer expert ids"):
        check_datums([_datum(3, routed_experts=torch.zeros(3, 4, 2))])
    with pytest.raises(ValueError, match="valid expert id at every position"):
        check_datums([_datum(3, routed_experts=torch.full((3, 4, 2), -1, dtype=torch.int16))])


def test_custom_loss_inputs_must_be_token_aligned_tensors():
    check_datums([_datum(3, loss_inputs={"teacher": torch.zeros(3)}, sample_inputs={"group": 7})])
    with pytest.raises(ValueError, match="loss_inputs\\['teacher'\\]"):
        check_datums([_datum(3, loss_inputs={"teacher": torch.zeros(4)})])
    with pytest.raises(TypeError):
        check_datums([_datum(3, loss_inputs={"teacher": [0.0, 0.0, 0.0]})])
    with pytest.raises(ValueError, match="loss_inputs keys must match"):
        check_datums([_datum(3, loss_inputs={"teacher": torch.zeros(3)}), _datum(2)])
