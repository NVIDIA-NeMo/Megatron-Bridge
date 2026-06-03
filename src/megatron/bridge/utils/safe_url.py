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

"""SSRF-safe URL fetching utilities.

Provides helpers that validate URLs against non-public IP addresses before
fetching, mitigating Server-Side Request Forgery (SSRF) when loading remote
resources from untrusted inputs (e.g. dataset entries, user-supplied image
URLs).
"""

import ipaddress
import os
import socket
from urllib.parse import urlparse


# Opt-out env var for the SSRF guard in ``is_safe_public_http_url``. Set to
# ``"1"`` when training on a trusted network where dataset URLs may legitimately
# point at internal hosts. Defaults off — the guard blocks loopback / private /
# link-local destinations (including cloud metadata at 169.254.169.254).
ALLOW_PRIVATE_URL_FETCH_ENV = "MEGATRON_BRIDGE_ALLOW_PRIVATE_URL_FETCH"


def is_safe_public_http_url(url: str) -> tuple[bool, str]:
    """Check that ``url`` is a public http(s) URL safe to fetch.

    Rejects non-http schemes, missing hostnames, and any hostname that
    resolves to a loopback, private (RFC 1918), link-local, multicast,
    reserved, or unspecified address. Used to mitigate SSRF when fetching
    remote URLs from untrusted inputs.

    Set ``MEGATRON_BRIDGE_ALLOW_PRIVATE_URL_FETCH=1`` to bypass (trusted
    networks only).

    Returns:
        Tuple of ``(is_safe, reason)``. ``reason`` is empty when safe.
    """
    if os.environ.get(ALLOW_PRIVATE_URL_FETCH_ENV) == "1":
        return True, ""
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        return False, f"invalid URL: {exc}"
    if parsed.scheme not in ("http", "https"):
        return False, f"unsupported URL scheme: {parsed.scheme!r}"
    host = parsed.hostname
    if not host:
        return False, "URL missing hostname"
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return False, f"DNS resolution failed: {exc}"
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"unparseable resolved address: {ip_str}"
        if not ip.is_global:
            return False, f"URL resolves to non-public address: {ip}"
    return True, ""


def safe_url_open(url: str):
    """Open ``url`` via a urllib opener that re-validates redirect targets.

    Prevents SSRF via redirect: a public URL returning a 3xx to an internal
    address would otherwise bypass :func:`is_safe_public_http_url`. The
    initial URL must already have been validated by the caller.
    """
    import urllib.error
    import urllib.request

    class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            is_safe, reason = is_safe_public_http_url(newurl)
            if not is_safe:
                raise urllib.error.URLError(f"redirect blocked ({reason}): {newurl}")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(_SafeRedirectHandler())
    return opener.open(url)  # noqa: S310  -- URL validated by caller + redirect handler
