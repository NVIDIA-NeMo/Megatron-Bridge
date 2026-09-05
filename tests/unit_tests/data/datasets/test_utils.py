import numpy as np
import pytest

from megatron.bridge.data.datasets import utils


def _reference_build_index(data: bytes, newline_int: int) -> np.ndarray:
    """Reproduce the previous build_index_from_memdata behavior in memory."""
    mdata = np.frombuffer(data, dtype=np.uint8)

    midx = np.where(mdata == newline_int)[0]
    midx_dtype = midx.dtype

    midx = midx.tolist()

    # Add an EOF sentinel when the file does not end with a newline.
    if (len(midx) == 0) or (midx[-1] + 1 != len(mdata)):
        midx = midx + [len(mdata) + 1]

    # Remove empty lines from the end of the file.
    while len(midx) > 1 and (midx[-1] - midx[-2]) < 2:
        midx.pop(-1)

    return np.asarray(midx, dtype=midx_dtype)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"abc", [4]),
        (b"abc\n", [3]),
        (b"abc\ndef\n", [3, 7]),
        (b"abc\ndef", [3, 8]),
        (b"\n", [0]),
        (b"abc\n\n", [3]),
        (b"abc\n\n\n", [3]),
        (b"\n\n", [0]),
    ],
)
def test_build_index_from_memdata(tmp_path, content, expected):
    """Test basic indexing and edge cases."""
    path = tmp_path / "data.jsonl"
    path.write_bytes(content)

    result = utils.build_index_from_memdata(str(path), ord("\n"))

    np.testing.assert_array_equal(
        result,
        np.asarray(expected, dtype=np.intp),
    )


def test_build_index_from_memdata_across_chunk_boundaries(tmp_path, monkeypatch):
    """Test newline indexing when records and delimiters cross chunk boundaries."""
    monkeypatch.setattr(utils, "_MEMMAP_INDEX_CHUNK_SIZE", 4)

    path = tmp_path / "data.jsonl"
    path.write_bytes(b"abc\ndef\nghi")

    result = utils.build_index_from_memdata(str(path), ord("\n"))

    np.testing.assert_array_equal(
        result,
        np.asarray([3, 7, 12], dtype=np.intp),
    )


def test_build_index_from_memdata_newline_at_chunk_boundary(tmp_path, monkeypatch):
    """Test a newline located exactly at the end of a chunk."""
    monkeypatch.setattr(utils, "_MEMMAP_INDEX_CHUNK_SIZE", 4)

    path = tmp_path / "data.jsonl"
    path.write_bytes(b"abc\ndef")

    result = utils.build_index_from_memdata(str(path), ord("\n"))

    np.testing.assert_array_equal(
        result,
        np.asarray([3, 8], dtype=np.intp),
    )


def test_build_index_from_memdata_multiple_chunks(tmp_path, monkeypatch):
    """Test indexing over several small chunks."""
    monkeypatch.setattr(utils, "_MEMMAP_INDEX_CHUNK_SIZE", 3)

    path = tmp_path / "data.jsonl"
    path.write_bytes(b"aa\nbbb\ncccc\nddddd")

    result = utils.build_index_from_memdata(str(path), ord("\n"))

    np.testing.assert_array_equal(
        result,
        np.asarray([2, 6, 11, 18], dtype=np.intp),
    )


def test_build_index_from_memdata_trailing_empty_lines_across_chunks(
    tmp_path,
    monkeypatch,
):
    """Test removal of trailing empty lines when they span chunk boundaries."""
    monkeypatch.setattr(utils, "_MEMMAP_INDEX_CHUNK_SIZE", 4)

    path = tmp_path / "data.jsonl"
    path.write_bytes(b"abc\ndef\n\n\n")

    result = utils.build_index_from_memdata(str(path), ord("\n"))

    np.testing.assert_array_equal(
        result,
        np.asarray([3, 7], dtype=np.intp),
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 16, 64])
def test_build_index_from_memdata_matches_previous_behavior(
    tmp_path,
    monkeypatch,
    chunk_size,
):
    """Ensure the chunked implementation matches the previous implementation."""
    monkeypatch.setattr(utils, "_MEMMAP_INDEX_CHUNK_SIZE", chunk_size)

    rng = np.random.default_rng(1234)

    for case_idx in range(50):
        size = int(rng.integers(1, 256))

        # Use arbitrary bytes so the test also covers multiple adjacent
        # delimiters, files without a final delimiter, and unusual contents.
        data = rng.integers(
            0,
            256,
            size=size,
            dtype=np.uint8,
        ).tobytes()

        path = tmp_path / f"data_{chunk_size}_{case_idx}.bin"
        path.write_bytes(data)

        expected = _reference_build_index(data, ord("\n"))
        actual = utils.build_index_from_memdata(
            str(path),
            ord("\n"),
        )

        np.testing.assert_array_equal(actual, expected)


def test_build_index_from_memdata_matches_previous_behavior_for_jsonl(
    tmp_path,
    monkeypatch,
):
    """Compare against the previous behavior on representative JSONL input."""
    monkeypatch.setattr(utils, "_MEMMAP_INDEX_CHUNK_SIZE", 7)

    data = b'{"text":"first"}\n{"text":"second"}\n{"text":"third"}\n{"text":"fourth"}'

    path = tmp_path / "training.jsonl"
    path.write_bytes(data)

    expected = _reference_build_index(data, ord("\n"))
    actual = utils.build_index_from_memdata(
        str(path),
        ord("\n"),
    )

    np.testing.assert_array_equal(actual, expected)
