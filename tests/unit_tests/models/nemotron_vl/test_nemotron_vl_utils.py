# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SSRF-guard regression tests for nemotron_vl_utils.

Covers the ``_is_safe_public_http_url`` helper and the integration
behavior of ``maybe_path_or_url_to_data_urls`` — notably that a
malicious video URL is never handed to ``urllib`` for retrieval.
"""

import random
import socket
from unittest import mock

import pytest
import torch

from megatron.bridge.models.nemotron_vl import nemotron_vl_utils as vlu


IMG_START_ID = 93
IMG_TOKEN_ID = 92
IMG_END_ID = 94
PAD_ID = 0


def _fake_getaddrinfo(ip: str):
    """Return a ``getaddrinfo`` stub that resolves any host to ``ip``."""

    def _stub(host, port, *args, **kwargs):
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 0, "", (ip, port or 0))]

    return _stub


class TestIsSafePublicHttpUrl:
    def test_rejects_non_http_scheme(self):
        ok, reason = vlu._is_safe_public_http_url("file:///etc/passwd")
        assert not ok
        assert "scheme" in reason

    def test_rejects_missing_hostname(self):
        ok, reason = vlu._is_safe_public_http_url("http:///x.mp4")
        assert not ok
        assert "hostname" in reason

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",  # loopback
            "10.0.0.1",  # RFC 1918
            "172.16.0.1",  # RFC 1918
            "192.168.1.1",  # RFC 1918
            "169.254.169.254",  # link-local (cloud metadata)
            "0.0.0.0",  # unspecified
            "::1",  # IPv6 loopback
            "fc00::1",  # IPv6 unique local
            "fe80::1",  # IPv6 link-local
        ],
    )
    def test_rejects_non_public_addresses(self, ip):
        with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo(ip)):
            ok, reason = vlu._is_safe_public_http_url("http://attacker.example.com/x.mp4")
        assert not ok
        assert "non-public" in reason

    def test_accepts_public_address(self):
        with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("93.184.216.34")):
            ok, reason = vlu._is_safe_public_http_url("https://example.com/x.mp4")
        assert ok
        assert reason == ""

    def test_rejects_when_any_resolved_ip_is_private(self):
        # Multiple records where one is private — must be rejected
        def stub(host, port, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
            ]

        with mock.patch("socket.getaddrinfo", side_effect=stub):
            ok, reason = vlu._is_safe_public_http_url("http://mixed.example.com/x.mp4")
        assert not ok
        assert "non-public" in reason

    def test_rejects_when_dns_fails(self):
        with mock.patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
            ok, reason = vlu._is_safe_public_http_url("http://does-not-resolve.invalid/x.mp4")
        assert not ok
        assert "DNS" in reason

    def test_opt_out_env_var_bypasses_check(self, monkeypatch):
        monkeypatch.setenv(vlu._ALLOW_PRIVATE_URL_FETCH_ENV, "1")
        # Would otherwise be rejected — env var bypasses the guard entirely.
        ok, reason = vlu._is_safe_public_http_url("http://127.0.0.1/x.mp4")
        assert ok
        assert reason == ""

    def test_opt_out_requires_exact_value(self, monkeypatch):
        monkeypatch.setenv(vlu._ALLOW_PRIVATE_URL_FETCH_ENV, "true")  # not "1"
        with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("127.0.0.1")):
            ok, _ = vlu._is_safe_public_http_url("http://localhost/x.mp4")
        assert not ok


class TestMaybePathOrUrlSsrfIntegration:
    def test_rejected_url_never_opens_socket(self, caplog):
        """A loopback URL must return unchanged without opening any connection."""
        with mock.patch.object(vlu, "_safe_url_open") as mocked_open:
            with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("127.0.0.1")):
                out, meta = vlu.maybe_path_or_url_to_data_urls("http://attacker.example.com/evil.mp4")

        mocked_open.assert_not_called()
        assert out == ["http://attacker.example.com/evil.mp4"]
        assert meta is None

    def test_metadata_endpoint_blocked(self):
        with mock.patch.object(vlu, "_safe_url_open") as mocked_open:
            out, _ = vlu.maybe_path_or_url_to_data_urls("http://169.254.169.254/latest/meta-data/evil.mp4")
        mocked_open.assert_not_called()
        assert out == ["http://169.254.169.254/latest/meta-data/evil.mp4"]

    def test_non_mp4_http_url_not_fetched(self):
        """Unchanged pre-existing behavior: non-.mp4 URLs are returned as-is."""
        with mock.patch.object(vlu, "_safe_url_open") as mocked_open:
            out, _ = vlu.maybe_path_or_url_to_data_urls("http://example.com/page.html")
        mocked_open.assert_not_called()
        assert out == ["http://example.com/page.html"]


class TestAdjustImageTokens:
    def test_randomized_rows_match_independent_row_local_reference(self):
        rng = random.Random(20260713)
        for _ in range(500):
            batch_size = rng.randint(1, 8)
            source_rows = []
            source_loss_rows = []
            target_rows = []
            target_loss_rows = []
            tile_counts = []
            for row_index in range(batch_size):
                source = [1000 + row_index]
                source_loss = [0]
                target = [1000 + row_index]
                target_loss = [0]
                for image_index in range(rng.randint(0, 3)):
                    old_count = rng.randint(1, 8)
                    new_count = rng.randint(0, 8)
                    tile_counts.append(new_count)
                    source.extend([IMG_START_ID, *([IMG_TOKEN_ID] * old_count), IMG_END_ID])
                    source_loss.extend([0] * (old_count + 2))
                    target.extend([IMG_START_ID, *([IMG_TOKEN_ID] * new_count), IMG_END_ID])
                    target_loss.extend([0] * (new_count + 2))
                    separator = 2000 + row_index * 10 + image_index
                    source.append(separator)
                    source_loss.append(0)
                    target.append(separator)
                    target_loss.append(0)
                answer = 3000 + row_index
                source.append(answer)
                source_loss.append(1)
                target.append(answer)
                target_loss.append(1)
                source_rows.append(source)
                source_loss_rows.append(source_loss)
                target_rows.append(target)
                target_loss_rows.append(target_loss)

            source_width = max(len(row) for row in source_rows) + rng.randint(0, 12)
            input_ids = torch.full((batch_size, source_width), PAD_ID, dtype=torch.long)
            loss_mask = torch.zeros((batch_size, source_width), dtype=torch.long)
            attention_mask = torch.zeros((batch_size, source_width), dtype=torch.long)
            for row_index, (source, source_loss) in enumerate(zip(source_rows, source_loss_rows, strict=True)):
                input_ids[row_index, : len(source)] = torch.tensor(source)
                loss_mask[row_index, : len(source)] = torch.tensor(source_loss)
                attention_mask[row_index, : len(source)] = 1

            adjusted = vlu.adjust_image_tokens(
                {"input_ids": input_ids, "loss_mask": loss_mask, "attention_mask": attention_mask},
                tile_counts,
                IMG_START_ID,
                IMG_END_ID,
                padding_values={"input_ids": PAD_ID, "loss_mask": 0, "attention_mask": 0},
            )

            target_width = max(len(row) for row in target_rows)
            expected_ids = torch.full((batch_size, target_width), PAD_ID, dtype=torch.long)
            expected_loss = torch.zeros((batch_size, target_width), dtype=torch.long)
            expected_attention = torch.zeros((batch_size, target_width), dtype=torch.long)
            for row_index, (target, target_loss) in enumerate(zip(target_rows, target_loss_rows, strict=True)):
                expected_ids[row_index, : len(target)] = torch.tensor(target)
                expected_loss[row_index, : len(target)] = torch.tensor(target_loss)
                expected_attention[row_index, : len(target)] = 1

            assert torch.equal(adjusted["input_ids"], expected_ids)
            assert torch.equal(adjusted["loss_mask"], expected_loss)
            assert torch.equal(adjusted["attention_mask"], expected_attention)

    def test_discards_stale_right_padding_before_repadding_adjusted_rows(self):
        input_ids = torch.tensor(
            [
                [10, 11, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
                [
                    20,
                    IMG_START_ID,
                    IMG_TOKEN_ID,
                    IMG_TOKEN_ID,
                    IMG_TOKEN_ID,
                    IMG_TOKEN_ID,
                    IMG_TOKEN_ID,
                    IMG_TOKEN_ID,
                    IMG_END_ID,
                    21,
                ],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        loss_mask = torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        adjusted = vlu.adjust_image_tokens(
            {"input_ids": input_ids, "loss_mask": loss_mask, "attention_mask": attention_mask},
            [1],
            IMG_START_ID,
            IMG_END_ID,
            padding_values={"input_ids": PAD_ID, "loss_mask": 0, "attention_mask": 0},
        )

        assert adjusted["input_ids"].shape == (2, 5)
        assert adjusted["input_ids"].tolist() == [
            [10, 11, PAD_ID, PAD_ID, PAD_ID],
            [20, IMG_START_ID, IMG_TOKEN_ID, IMG_END_ID, 21],
        ]
        assert adjusted["attention_mask"].tolist() == [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]
        assert adjusted["loss_mask"].tolist() == [[0, 1, 0, 0, 0], [0, 0, 0, 0, 1]]

    def test_preserves_batch_rows_when_removing_image_tokens(self):
        input_ids = torch.tensor(
            [
                [10, IMG_START_ID, IMG_TOKEN_ID, IMG_TOKEN_ID, IMG_TOKEN_ID, IMG_END_ID, 11, PAD_ID, PAD_ID],
                [20, IMG_START_ID, IMG_TOKEN_ID, IMG_END_ID, 21, 22, PAD_ID, PAD_ID, PAD_ID],
            ]
        )
        loss_mask = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0],
            ]
        )

        adjusted = vlu.adjust_image_tokens(
            {"input_ids": input_ids, "loss_mask": loss_mask, "attention_mask": attention_mask},
            torch.tensor([1, 1]),
            IMG_START_ID,
            IMG_END_ID,
            padding_values={"input_ids": PAD_ID, "loss_mask": 0, "attention_mask": 0},
        )

        assert adjusted["input_ids"].shape == (2, 6)
        assert adjusted["input_ids"].tolist() == [
            [10, IMG_START_ID, IMG_TOKEN_ID, IMG_END_ID, 11, PAD_ID],
            [20, IMG_START_ID, IMG_TOKEN_ID, IMG_END_ID, 21, 22],
        ]
        assert adjusted["loss_mask"].tolist() == [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ]
        assert adjusted["attention_mask"].tolist() == [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ]
        assert 20 not in adjusted["input_ids"][0]
        assert 10 not in adjusted["input_ids"][1]

    def test_maps_flat_tile_counts_across_zero_and_multiple_image_rows(self):
        input_ids = torch.tensor(
            [
                [10, 11, 12, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID],
                [
                    20,
                    IMG_START_ID,
                    IMG_TOKEN_ID,
                    IMG_TOKEN_ID,
                    IMG_TOKEN_ID,
                    IMG_TOKEN_ID,
                    IMG_END_ID,
                    21,
                    IMG_START_ID,
                    IMG_TOKEN_ID,
                    IMG_TOKEN_ID,
                    IMG_TOKEN_ID,
                    IMG_END_ID,
                ],
            ]
        )
        loss_mask = torch.tensor(
            [
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]
        )

        adjusted = vlu.adjust_image_tokens(
            {"input_ids": input_ids, "loss_mask": loss_mask},
            [2, 1],
            IMG_START_ID,
            IMG_END_ID,
            padding_values={"input_ids": PAD_ID, "loss_mask": 0},
        )

        assert adjusted["input_ids"].shape == (2, 13)
        assert adjusted["input_ids"][0].tolist() == input_ids[0].tolist()
        assert adjusted["input_ids"][1].tolist() == [
            20,
            IMG_START_ID,
            IMG_TOKEN_ID,
            IMG_TOKEN_ID,
            IMG_END_ID,
            21,
            IMG_START_ID,
            IMG_TOKEN_ID,
            IMG_END_ID,
            PAD_ID,
            PAD_ID,
            PAD_ID,
            PAD_ID,
        ]
        assert adjusted["loss_mask"][1].tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    def test_expands_media_tokens_and_repeats_aligned_values(self):
        adjusted = vlu.adjust_image_tokens(
            {
                "input_ids": torch.tensor([[10, IMG_START_ID, IMG_TOKEN_ID, IMG_END_ID, 11]]),
                "loss_mask": torch.tensor([[0, 0, 0, 0, 1]]),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            },
            3,
            IMG_START_ID,
            IMG_END_ID,
            padding_values={"input_ids": PAD_ID, "loss_mask": 0, "attention_mask": 0},
        )

        assert adjusted["input_ids"].tolist() == [
            [10, IMG_START_ID, IMG_TOKEN_ID, IMG_TOKEN_ID, IMG_TOKEN_ID, IMG_END_ID, 11]
        ]
        assert adjusted["loss_mask"].tolist() == [[0, 0, 0, 0, 0, 0, 1]]
        assert adjusted["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 1]]
