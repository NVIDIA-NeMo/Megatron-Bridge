import pytest
import torch

from megatron.bridge.engine import Datum, validate_datums


def _datum(n: int, sample_id: int | None = None, **loss_inputs: torch.Tensor) -> Datum:
    return Datum(input_ids=torch.arange(n, dtype=torch.int64), loss_inputs=dict(loss_inputs), sample_id=sample_id)


def test_assigns_missing_sample_ids_in_order_and_skips_used_ones():
    datums = [_datum(4), _datum(3, sample_id=1), _datum(5)]
    validate_datums(datums)
    assert [d.sample_id for d in datums] == [0, 1, 2]
    assert datums[0].num_tokens == 4


def test_rejects_duplicate_sample_ids():
    with pytest.raises(ValueError, match="unique"):
        validate_datums([_datum(2, sample_id=7), _datum(2, sample_id=7)])


@pytest.mark.parametrize("datums", [None, []])
def test_rejects_none_and_empty(datums):
    with pytest.raises(ValueError):
        validate_datums(datums)


def test_rejects_non_datum_and_bad_input_ids():
    with pytest.raises(TypeError):
        validate_datums([torch.zeros(3, dtype=torch.int64)])
    with pytest.raises(ValueError, match="int64"):
        validate_datums([Datum(input_ids=torch.zeros(3, dtype=torch.int32))])
    with pytest.raises(ValueError, match="1-D"):
        validate_datums([Datum(input_ids=torch.zeros(2, 3, dtype=torch.int64))])


def test_loss_inputs_must_be_token_aligned_tensors():
    ok = _datum(4, weights=torch.ones(4), old_logprobs=torch.zeros(4))
    validate_datums([ok])
    with pytest.raises(ValueError, match="first dimension"):
        validate_datums([_datum(4, weights=torch.ones(3))])
    with pytest.raises(TypeError):
        validate_datums([Datum(input_ids=torch.zeros(4, dtype=torch.int64), loss_inputs={"weights": [1.0] * 4})])
