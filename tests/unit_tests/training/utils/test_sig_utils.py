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

import signal
import threading

import pytest

from megatron.bridge.training.utils.sig_utils import DistributedSignalHandler


@pytest.fixture
def restore_sigterm_handler():
    """Snapshot the SIGTERM handler and restore it after the test."""
    original = signal.getsignal(signal.SIGTERM)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, original)


@pytest.mark.unit
class TestDistributedSignalHandlerThreading:
    """Threading behavior of `DistributedSignalHandler`."""

    def test_context_manager_on_side_thread_does_not_raise(self, restore_sigterm_handler):
        """Entering/exiting the context off the main thread must not crash."""
        before = signal.getsignal(signal.SIGTERM)
        captured: dict = {}

        def worker() -> None:
            captured["is_main_thread"] = threading.current_thread() is threading.main_thread()
            try:
                with DistributedSignalHandler(signal.SIGTERM) as handler:
                    captured["installed"] = handler._installed
                captured["released"] = handler.released
            except Exception as exc:
                captured["error"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        after = signal.getsignal(signal.SIGTERM)

        assert captured.get("is_main_thread") is False
        assert "error" not in captured, f"side-thread use raised: {captured.get('error')!r}"
        assert captured["installed"] is False
        assert captured["released"] is True
        assert after is before

    def test_context_manager_on_main_thread_installs_and_restores_trap(self, restore_sigterm_handler):
        """The thread handler should work on the main thread."""
        before = signal.getsignal(signal.SIGTERM)
        handler = DistributedSignalHandler(signal.SIGTERM)

        with handler:
            installed_handler = signal.getsignal(signal.SIGTERM)
            assert handler._installed is True
            assert installed_handler is not before, "SIGTERM trap was not installed on the main thread"
            assert callable(installed_handler)

        assert handler.released is True
        assert signal.getsignal(signal.SIGTERM) is before

    def test_release_is_idempotent(self, restore_sigterm_handler):
        """`release()` returns True the first time and False on subsequent calls."""
        handler = DistributedSignalHandler(signal.SIGTERM)
        handler.__enter__()

        assert handler.release() is True
        assert handler.released is True
        assert handler.release() is False
