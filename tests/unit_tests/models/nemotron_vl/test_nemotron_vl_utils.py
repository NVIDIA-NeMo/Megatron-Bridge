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

import socket
from unittest import mock

import pytest

from megatron.bridge.models.nemotron_vl import nemotron_vl_utils as vlu


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
