import numpy as np

from megatron.bridge.training.setup import _get_token_dtype_code


def test_get_token_dtype_code_uint16():
    """Vocab sizes up to 65536 fit in uint16 (numpy code 4)."""
    assert _get_token_dtype_code(1) == np.dtype(np.uint16).num
    assert _get_token_dtype_code(65535) == np.dtype(np.uint16).num
    assert _get_token_dtype_code(65536) == np.dtype(np.uint16).num


def test_get_token_dtype_code_uint64():
    """Vocab sizes larger than 65536 require uint64 (numpy code 8)."""
    assert _get_token_dtype_code(65537) == np.dtype(np.uint64).num
    assert _get_token_dtype_code(100000) == np.dtype(np.uint64).num


def test_get_token_dtype_code_boundary():
    """The boundary between uint16 and uint64 is exactly 65536."""
    assert _get_token_dtype_code(65536) == 4
