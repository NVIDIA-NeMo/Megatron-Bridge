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

"""SSRF-guard tests for megatron.bridge.utils.safe_url.

Covers ``is_safe_public_http_url``, ``safe_url_open``, and integration
with ``load_image`` in ``vlm_generate_utils`` and ``load_audio`` in
``hf_to_megatron_generate_audio_lm``.
"""

import socket
from unittest import mock

import pytest

from megatron.bridge.utils.safe_url import (
    ALLOW_PRIVATE_URL_FETCH_ENV,
    is_safe_public_http_url,
    safe_url_open,
)


def _fake_getaddrinfo(ip: str):
    """Return a ``getaddrinfo`` stub that resolves any host to ``ip``."""

    def _stub(host, port, *args, **kwargs):
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 0, "", (ip, port or 0))]

    return _stub


class TestIsSafePublicHttpUrl:
    """Tests for the ``is_safe_public_http_url`` SSRF validation function."""

    def test_rejects_non_http_scheme(self):
        """Reject file:// scheme which could read local filesystem."""
        ok, reason = is_safe_public_http_url("file:///etc/passwd")
        assert not ok
        assert "scheme" in reason

    def test_rejects_ftp_scheme(self):
        """Reject ftp:// scheme — only http(s) is allowed."""
        ok, reason = is_safe_public_http_url("ftp://example.com/file.bin")
        assert not ok
        assert "scheme" in reason

    def test_rejects_missing_hostname(self):
        """Reject URLs with empty hostname (e.g. http:///path)."""
        ok, reason = is_safe_public_http_url("http:///image.png")
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
            "100.64.0.1",  # RFC 6598 CGNAT / shared address space
            "0.0.0.0",  # unspecified
            "::1",  # IPv6 loopback
            "fc00::1",  # IPv6 unique local
            "fe80::1",  # IPv6 link-local
        ],
    )
    def test_rejects_non_public_addresses(self, ip):
        """Reject URLs that resolve to any non-globally-routable IP address."""
        with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo(ip)):
            ok, reason = is_safe_public_http_url("http://attacker.example.com/image.png")
        assert not ok
        assert "non-public" in reason

    def test_accepts_public_address(self):
        """Allow URLs that resolve to a public, globally-routable IP address."""
        with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("93.184.216.34")):
            ok, reason = is_safe_public_http_url("https://example.com/image.png")
        assert ok
        assert reason == ""

    def test_rejects_when_any_resolved_ip_is_private(self):
        """Reject if even one DNS record points to a private address (split-horizon DNS)."""

        def stub(host, port, *args, **kwargs):
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0)),
            ]

        with mock.patch("socket.getaddrinfo", side_effect=stub):
            ok, reason = is_safe_public_http_url("http://mixed.example.com/image.png")
        assert not ok
        assert "non-public" in reason

    def test_rejects_when_dns_fails(self):
        """Reject URLs whose hostname cannot be resolved."""
        with mock.patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
            ok, reason = is_safe_public_http_url("http://does-not-resolve.invalid/image.png")
        assert not ok
        assert "DNS" in reason

    def test_opt_out_env_var_bypasses_check(self, monkeypatch):
        """Setting MEGATRON_BRIDGE_ALLOW_PRIVATE_URL_FETCH=1 disables the guard entirely."""
        monkeypatch.setenv(ALLOW_PRIVATE_URL_FETCH_ENV, "1")
        ok, reason = is_safe_public_http_url("http://127.0.0.1/image.png")
        assert ok
        assert reason == ""

    def test_opt_out_requires_exact_value(self, monkeypatch):
        """The opt-out env var must be exactly '1' — 'true' or other truthy values are ignored."""
        monkeypatch.setenv(ALLOW_PRIVATE_URL_FETCH_ENV, "true")  # not "1"
        with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("127.0.0.1")):
            ok, _ = is_safe_public_http_url("http://localhost/image.png")
        assert not ok


class TestSafeUrlOpen:
    """Tests for the ``safe_url_open`` redirect-validating URL opener."""

    def test_redirect_handler_rejects_private_redirect_target(self):
        """The custom redirect handler must block redirects to non-public IPs.

        Instantiates the redirect handler built by ``safe_url_open`` and
        verifies that ``redirect_request`` raises ``URLError`` when the
        redirect target resolves to a private address.
        """
        import urllib.error
        import urllib.request

        # Build the opener to get the redirect handler instance
        with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("93.184.216.34")):
            with mock.patch("urllib.request.OpenerDirector.open"):
                safe_url_open("https://example.com/image.png")

        # Extract the handler class from a fresh call and test it directly
        handlers = []
        original_build = urllib.request.build_opener

        def capture_build(*handler_classes):
            handlers.extend(handler_classes)
            return original_build(*handler_classes)

        with mock.patch("urllib.request.build_opener", side_effect=capture_build):
            with mock.patch("urllib.request.OpenerDirector.open"):
                safe_url_open("https://example.com/image.png")

        redirect_handler = handlers[0]
        req = urllib.request.Request("https://example.com/image.png")

        with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("127.0.0.1")):
            with pytest.raises(urllib.error.URLError, match="redirect blocked"):
                redirect_handler.redirect_request(req, None, 302, "Found", {}, "http://evil.com/redir.png")

    def test_redirect_handler_allows_public_redirect_target(self):
        """The redirect handler must allow redirects to public IPs."""
        import urllib.request

        handlers = []
        original_build = urllib.request.build_opener

        def capture_build(*handler_classes):
            handlers.extend(handler_classes)
            return original_build(*handler_classes)

        with mock.patch("urllib.request.build_opener", side_effect=capture_build):
            with mock.patch("urllib.request.OpenerDirector.open"):
                safe_url_open("https://example.com/image.png")

        redirect_handler = handlers[0]
        req = urllib.request.Request("https://example.com/image.png")

        with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("93.184.216.34")):
            result = redirect_handler.redirect_request(req, None, 302, "Found", {}, "https://cdn.example.com/img.png")
        assert result is not None


class TestLoadImageSsrf:
    """Integration tests verifying ``load_image`` in ``vlm_generate_utils`` rejects unsafe URLs."""

    def test_private_url_raises_value_error(self):
        """A URL resolving to a loopback address must raise ValueError."""
        from examples.conversion.vlm_generate_utils import load_image

        with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("127.0.0.1")):
            with pytest.raises(ValueError, match="Refusing to fetch image URL"):
                load_image("http://attacker.example.com/evil.png")

    def test_metadata_endpoint_blocked(self):
        """Cloud metadata endpoint (169.254.169.254) must be blocked."""
        from examples.conversion.vlm_generate_utils import load_image

        with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("169.254.169.254")):
            with pytest.raises(ValueError, match="non-public"):
                load_image("http://169.254.169.254/latest/meta-data/image.png")

    def test_safe_url_never_called_for_unsafe_url(self):
        """When URL validation fails, the fetch function must never be invoked."""
        from examples.conversion import vlm_generate_utils

        with mock.patch.object(vlm_generate_utils, "safe_url_open") as mocked_open:
            with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("10.0.0.1")):
                with pytest.raises(ValueError):
                    vlm_generate_utils.load_image("http://internal.example.com/secret.png")
        mocked_open.assert_not_called()

    def test_local_file_path_not_validated(self, tmp_path):
        """Local file paths must be opened directly without SSRF validation."""
        from examples.conversion import vlm_generate_utils
        from PIL import Image

        img = Image.new("RGB", (1, 1))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        # Patch on the module where the name is bound, not the source module
        with mock.patch.object(vlm_generate_utils, "is_safe_public_http_url") as mocked_check:
            result = vlm_generate_utils.load_image(str(img_path))
        mocked_check.assert_not_called()
        assert isinstance(result, Image.Image)


class TestLoadAudioSsrf:
    """Integration tests verifying ``load_audio`` in ``hf_to_megatron_generate_audio_lm`` rejects unsafe URLs."""

    @pytest.fixture(autouse=True)
    def _import_audio_module(self):
        """Import the audio module, skipping the entire class if dependencies are missing."""
        try:
            from examples.conversion import hf_to_megatron_generate_audio_lm

            self.audio_module = hf_to_megatron_generate_audio_lm
        except ImportError as exc:
            pytest.skip(f"audio_lm module not importable: {exc}")

    def test_private_url_raises_value_error(self):
        """A URL resolving to a loopback address must raise ValueError."""
        with mock.patch.object(self.audio_module, "HAS_LIBROSA", True):
            with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("127.0.0.1")):
                with pytest.raises(ValueError, match="Refusing to fetch audio URL"):
                    self.audio_module.load_audio("http://attacker.example.com/evil.mp3")

    def test_metadata_endpoint_blocked(self):
        """Cloud metadata endpoint (169.254.169.254) must be blocked."""
        with mock.patch.object(self.audio_module, "HAS_LIBROSA", True):
            with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("169.254.169.254")):
                with pytest.raises(ValueError, match="non-public"):
                    self.audio_module.load_audio("http://169.254.169.254/latest/meta-data/audio.mp3")

    def test_safe_url_never_called_for_unsafe_url(self):
        """When URL validation fails, the fetch function must never be invoked."""
        with mock.patch.object(self.audio_module, "HAS_LIBROSA", True):
            with mock.patch.object(self.audio_module, "safe_url_open") as mocked_open:
                with mock.patch("socket.getaddrinfo", side_effect=_fake_getaddrinfo("10.0.0.1")):
                    with pytest.raises(ValueError):
                        self.audio_module.load_audio("http://internal.example.com/secret.mp3")
            mocked_open.assert_not_called()

    def test_local_file_path_not_validated(self, tmp_path):
        """Local file paths must be loaded directly without SSRF validation."""
        dummy_audio_path = str(tmp_path / "test.wav")

        fake_audio_data = "fake_audio_array"
        with mock.patch.object(self.audio_module, "HAS_LIBROSA", True):
            with mock.patch.object(self.audio_module, "librosa") as mocked_librosa:
                mocked_librosa.load.return_value = (fake_audio_data, 16000)
                with mock.patch.object(self.audio_module, "is_safe_public_http_url") as mocked_check:
                    result = self.audio_module.load_audio(dummy_audio_path)
        mocked_check.assert_not_called()
        mocked_librosa.load.assert_called_once_with(dummy_audio_path, sr=16000)
        assert result == fake_audio_data
