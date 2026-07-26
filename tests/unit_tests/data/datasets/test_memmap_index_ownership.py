from unittest import mock

import pytest

from megatron.bridge.data.datasets.utils import _JSONLMemMapDataset


@pytest.mark.unit
def test_node_local_index_is_built_by_local_rank_zero(tmp_path, monkeypatch):
    source_path = tmp_path / "training.jsonl"
    source_path.write_text('{"input":"node-local","output":"index"}\n', encoding="utf-8")
    index_mapping_dir = tmp_path / "index"
    monkeypatch.setenv("LOCAL_RANK", "0")

    with (
        mock.patch("torch.distributed.is_available", return_value=True),
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_rank", return_value=1),
        mock.patch("torch.distributed.barrier"),
    ):
        dataset = _JSONLMemMapDataset(
            dataset_paths=[str(source_path)],
            workers=1,
            index_mapping_dir=str(index_mapping_dir),
        )

    assert len(dataset) == 1
    assert dataset[0] == {"input": "node-local", "output": "index"}
