import io
import json
import random
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from megatron.bridge.models.bagel.data.energon import (
    BagelT2IRawSample,
    BagelT2ISample,
    BagelT2ITaskEncoder,
    cook_bagel_t2i,
)


pytestmark = pytest.mark.unit


def test_cook_bagel_t2i_preserves_raw_data_and_sample_keys() -> None:
    captions = '{"caption": "keep this string unparsed"}'
    metadata = {
        "dataset_group": "t2i_pretrain",
        "dataset_name": "t2i",
        "source": {"parquet": "chunk_0.parquet", "row_group": 0, "row": 0},
        "captions": captions,
    }
    image = b"raw-image-bytes"
    restore_key = ("Webdataset", 0, 0)
    subflavors = {"source": "bagel"}
    crude_sample = {
        "__key__": "t2i-chunk_0-rg0-row0",
        "__restore_key__": restore_key,
        "__subflavor__": None,
        "__subflavors__": subflavors,
        "image": image,
        "json": json.dumps(metadata).encode("utf-8"),
    }

    sample = cook_bagel_t2i(crude_sample)

    assert isinstance(sample, BagelT2IRawSample)
    assert sample.image is image
    assert sample.metadata == metadata
    assert sample.metadata["captions"] == captions
    assert isinstance(sample.metadata["captions"], str)
    assert sample.__key__ == crude_sample["__key__"]
    assert sample.__restore_key__ == restore_key
    assert sample.__subflavor__ is None
    assert sample.__subflavors__ == subflavors


def test_t2i_task_encoder_processes_caption_image_and_sequence_plan() -> None:
    image_buffer = io.BytesIO()
    Image.new("RGB", (32, 16), (10, 20, 30)).save(image_buffer, format="PNG")
    metadata = {
        "dataset_group": "t2i_pretrain",
        "dataset_name": "t2i",
        "source": {"parquet": "chunk_0.parquet", "row_group": 0, "row": 0},
        "captions": json.dumps({"first": "caption one", "second": "caption two"}),
    }
    crude_sample = {
        "__key__": "t2i-chunk_0-rg0-row0",
        "__restore_key__": ("Webdataset", 0, 0),
        "__subflavor__": None,
        "__subflavors__": {},
        "image": image_buffer.getvalue(),
        "json": json.dumps(metadata).encode("utf-8"),
    }
    tokenizer = Mock()
    tokenizer.encode.side_effect = {"caption one": [1], "caption two": [2], " ": [0]}.get

    def transform(image: Image.Image) -> torch.Tensor:
        assert image.mode == "RGB"
        return torch.full((3, 16, 32), 0.5)

    random.seed(42)
    sample = BagelT2ITaskEncoder(tokenizer, transform, 16).cookers[0].cook(crude_sample)

    assert isinstance(sample, BagelT2ISample)
    assert torch.equal(sample.image_tensor_list[0], transform(Image.new("RGB", (32, 16))))
    assert sample.text_ids_list == [[1]]
    assert sample.num_tokens == 3
    assert sample.sequence_plan == [
        {"type": "text", "enable_cfg": 1, "loss": 0, "special_token_loss": 0, "special_token_label": None},
        {"type": "vae_image", "enable_cfg": 0, "loss": 1, "special_token_loss": 0, "special_token_label": None},
    ]
    assert sample.metadata == metadata
