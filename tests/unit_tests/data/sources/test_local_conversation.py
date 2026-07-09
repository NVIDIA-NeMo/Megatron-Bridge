# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import json

import pytest

from megatron.bridge.data.sources.local_conversation import (
    LocalConversationDatasetSourceConfig,
    load_local_conversation_examples,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_local_source_loads_json_and_jsonl_through_one_normalizer(tmp_path, suffix):
    source_path = tmp_path / f"conversations{suffix}"
    record = {
        "messages": [
            {"role": "user", "content": "<image>Describe this. <audio>"},
            {"role": "assistant", "content": "A narrated receipt."},
        ],
        "images": ["images/receipt.png"],
        "audio_path": "audio/receipt.wav",
        "sample_id": 7,
    }
    if suffix == ".jsonl":
        source_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    else:
        source_path.write_text(json.dumps({"records": [record]}), encoding="utf-8")

    examples = load_local_conversation_examples(
        LocalConversationDatasetSourceConfig(path=str(source_path), media_root=str(tmp_path))
    )

    assert examples == [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(tmp_path / "images/receipt.png")},
                        {"type": "audio", "audio": str(tmp_path / "audio/receipt.wav")},
                        {"type": "text", "text": "Describe this. "},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "A narrated receipt."}]},
            ],
            "sample_id": 7,
        }
    ]


def test_local_source_resolves_inline_media_and_preserves_remote_urls(tmp_path):
    source_path = tmp_path / "conversations.json"
    source_path.write_text(
        json.dumps(
            [
                {
                    "conversation": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": "images/local.png"},
                                {"type": "video", "video": "https://example.com/video.mp4"},
                                {"type": "text", "text": "Compare them."},
                            ],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )

    examples = load_local_conversation_examples(
        LocalConversationDatasetSourceConfig(path=str(source_path), media_root=str(tmp_path))
    )

    content = examples[0]["conversation"][0]["content"]
    assert content[0]["image"] == str(tmp_path / "images/local.png")
    assert content[1]["video"] == "https://example.com/video.mp4"


@pytest.mark.parametrize("path", [None, "", "conversations.txt"])
def test_local_source_rejects_missing_or_unsupported_path(path):
    with pytest.raises(ValueError, match="path"):
        LocalConversationDatasetSourceConfig(path=path).validate()


def test_local_source_selects_explicit_wrapped_records_key(tmp_path):
    source_path = tmp_path / "conversations.json"
    source_path.write_text(
        json.dumps(
            {
                "custom": [
                    {
                        "messages": [
                            {"role": "user", "content": "Question"},
                            {"role": "assistant", "content": "Answer"},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    examples = load_local_conversation_examples(
        LocalConversationDatasetSourceConfig(path=str(source_path), records_key="custom")
    )

    assert examples[0]["conversation"][1]["role"] == "assistant"
