# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models.hf_pretrained.safe_config_loader import safe_load_config_with_retry


class TestSafeLoadConfigWithRetry:
    """Test suite for safe_load_config_with_retry function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_config = MagicMock(spec=PretrainedConfig)
        self.mock_config.model_type = "llama"
        self.mock_config.architectures = ["LlamaForCausalLM"]
        self.test_path = "meta-llama/Llama-2-7b-hf"

    def test_basic_successful_load(self):
        """Test basic successful configuration loading."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = self.mock_config

            result = safe_load_config_with_retry(self.test_path)

            assert result == self.mock_config
            mock_auto_config.from_pretrained.assert_called_once_with(self.test_path, trust_remote_code=False)

    def test_with_trust_remote_code(self):
        """Test configuration loading with trust_remote_code=True."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = self.mock_config

            result = safe_load_config_with_retry(self.test_path, trust_remote_code=True)

            assert result == self.mock_config
            mock_auto_config.from_pretrained.assert_called_once_with(self.test_path, trust_remote_code=True)

    def test_with_additional_kwargs(self):
        """Test configuration loading with additional kwargs."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = self.mock_config

            result = safe_load_config_with_retry(
                self.test_path, trust_remote_code=True, cache_dir="/custom/cache", local_files_only=True
            )

            assert result == self.mock_config
            mock_auto_config.from_pretrained.assert_called_once_with(
                self.test_path, trust_remote_code=True, cache_dir="/custom/cache", local_files_only=True
            )

    def test_with_file_locking(self):
        """Test configuration loading with file locking enabled."""
        mock_lock = MagicMock()

        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.filelock.FileLock") as mock_filelock:
            mock_filelock.return_value = mock_lock
            mock_lock.__enter__ = Mock(return_value=mock_lock)
            mock_lock.__exit__ = Mock(return_value=None)

            with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
                mock_auto_config.from_pretrained.return_value = self.mock_config

                result = safe_load_config_with_retry(self.test_path)

                # Verify file lock was used
                expected_hash = hashlib.md5(str(self.test_path).encode()).hexdigest()
                expected_lock_file = (
                    Path.home() / ".cache" / "huggingface" / f".megatron_config_lock_{expected_hash}.lock"
                )
                mock_filelock.assert_called_once_with(str(expected_lock_file), timeout=60)

                # Verify context manager was used
                mock_lock.__enter__.assert_called_once()
                mock_lock.__exit__.assert_called_once()

                assert result == self.mock_config

    def test_retry_on_transient_failure(self):
        """Test retry mechanism on transient network failures."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            # First two calls fail, third succeeds
            mock_auto_config.from_pretrained.side_effect = [
                Exception("Connection timeout"),
                Exception("Temporary network error"),
                self.mock_config,
            ]

            with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.time.sleep") as mock_sleep:
                result = safe_load_config_with_retry(self.test_path, max_retries=3, base_delay=0.1)

                assert result == self.mock_config
                assert mock_auto_config.from_pretrained.call_count == 3

                # Verify exponential backoff was used
                assert mock_sleep.call_count == 2  # Two retries
                sleep_calls = mock_sleep.call_args_list
                # First delay should be around 0.1, second around 0.2 (plus jitter)
                assert 0.1 <= sleep_calls[0][0][0] <= 0.3
                assert 0.2 <= sleep_calls[1][0][0] <= 0.5

    def test_no_retry_on_permanent_failure_config_not_found(self):
        """Test no retry on permanent failures (config.json not found)."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.side_effect = Exception(
                "does not appear to have a file named config.json"
            )

            with pytest.raises(ValueError, match="Failed to load configuration"):
                safe_load_config_with_retry(self.test_path, max_retries=3)

            # Should not retry on permanent failures
            mock_auto_config.from_pretrained.assert_called_once()

    def test_no_retry_on_permanent_failure_repo_not_found(self):
        """Test no retry on permanent failures (repository not found)."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.side_effect = Exception("Repository not found")

            with pytest.raises(ValueError, match="Failed to load configuration"):
                safe_load_config_with_retry(self.test_path, max_retries=3)

            # Should not retry on permanent failures
            mock_auto_config.from_pretrained.assert_called_once()

    def test_no_retry_on_access_denied(self):
        """Test no retry on access denied errors."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.side_effect = Exception("401 client error")

            with pytest.raises(ValueError, match="Failed to load configuration"):
                safe_load_config_with_retry(self.test_path, max_retries=3)

            # Should not retry on permanent failures
            mock_auto_config.from_pretrained.assert_called_once()

    def test_max_retries_exhausted(self):
        """Test behavior when all retries are exhausted."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.side_effect = Exception("Persistent network error")

            with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.time.sleep"):
                with pytest.raises(ValueError, match="Failed to load configuration.*after 3 attempts"):
                    safe_load_config_with_retry(self.test_path, max_retries=2)

                # Should try initial + 2 retries = 3 total attempts
                assert mock_auto_config.from_pretrained.call_count == 3

    def test_pathlib_path_input(self):
        """Test configuration loading with pathlib.Path input."""
        path_obj = Path(self.test_path)

        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = self.mock_config

            result = safe_load_config_with_retry(path_obj)

            assert result == self.mock_config
            mock_auto_config.from_pretrained.assert_called_once_with(path_obj, trust_remote_code=False)

    def test_concurrent_access_thread_safety(self):
        """Test thread safety with concurrent access."""
        call_count = 0
        call_times = []
        lock = threading.Lock()

        def mock_from_pretrained(*args, **kwargs):
            nonlocal call_count
            with lock:
                call_count += 1
                call_times.append(time.time())

            # Simulate some processing time
            time.sleep(0.02)
            return self.mock_config

        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained = mock_from_pretrained

            # Test with multiple threads
            num_threads = 5
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for i in range(num_threads):
                    future = executor.submit(safe_load_config_with_retry, self.test_path)
                    futures.append(future)

                # Wait for all to complete
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

            # All should succeed
            assert len(results) == num_threads
            assert all(r == self.mock_config for r in results)
            assert call_count == num_threads

    def test_exponential_backoff_with_jitter(self):
        """Test that exponential backoff includes jitter."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.side_effect = [
                Exception("Network error"),
                Exception("Network error"),
                self.mock_config,
            ]

            with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.time.sleep") as mock_sleep:
                # Mock time.time() to control jitter
                with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.time.time", return_value=0.5):
                    safe_load_config_with_retry(self.test_path, max_retries=2, base_delay=1.0)

                    # Verify sleep was called with exponential backoff + jitter
                    sleep_calls = mock_sleep.call_args_list
                    assert len(sleep_calls) == 2

                    # First delay: base_delay * (2^0) + jitter = 1.0 + 0.05 = 1.05
                    assert sleep_calls[0][0][0] == pytest.approx(1.05, abs=0.01)

                    # Second delay: base_delay * (2^1) + jitter = 2.0 + 0.05 = 2.05
                    assert sleep_calls[1][0][0] == pytest.approx(2.05, abs=0.01)

    def test_custom_retry_parameters(self):
        """Test configuration loading with custom retry parameters."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.side_effect = [Exception("Network error"), self.mock_config]

            with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.time.sleep") as mock_sleep:
                result = safe_load_config_with_retry(self.test_path, max_retries=5, base_delay=0.5)

                assert result == self.mock_config
                assert mock_auto_config.from_pretrained.call_count == 2

                # Should use custom base_delay
                sleep_calls = mock_sleep.call_args_list
                assert len(sleep_calls) == 1
                delay = sleep_calls[0][0][0]
                assert 0.5 <= delay <= 1.0  # base_delay + jitter

    def test_lock_file_creation(self):
        """Test that lock files are created in the correct location."""
        mock_lock = MagicMock()

        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.filelock.FileLock") as mock_filelock:
            mock_filelock.return_value = mock_lock
            mock_lock.__enter__ = Mock(return_value=mock_lock)
            mock_lock.__exit__ = Mock(return_value=None)

            with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
                mock_auto_config.from_pretrained.return_value = self.mock_config

                # Test with different paths to ensure unique lock files
                test_paths = ["model1", "model2", "path/to/model"]

                for path in test_paths:
                    safe_load_config_with_retry(path)

                    # Verify unique lock file for each path
                    expected_hash = hashlib.md5(str(path).encode()).hexdigest()
                    expected_lock_file = (
                        Path.home() / ".cache" / "huggingface" / f".megatron_config_lock_{expected_hash}.lock"
                    )

                    # Check that FileLock was called with the correct path
                    found_call = False
                    for call_args in mock_filelock.call_args_list:
                        if call_args[0][0] == str(expected_lock_file):
                            found_call = True
                            break
                    assert found_call, f"Expected lock file {expected_lock_file} not found in calls"

    def test_error_message_includes_details(self):
        """Test that error messages include helpful details."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            original_error = Exception("Original network error")
            mock_auto_config.from_pretrained.side_effect = original_error

            with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.time.sleep"):
                with pytest.raises(ValueError) as exc_info:
                    safe_load_config_with_retry(self.test_path, max_retries=1)

                error_msg = str(exc_info.value)
                assert self.test_path in error_msg
                assert "2 attempts" in error_msg  # 1 initial + 1 retry
                assert "concurrent access conflicts" in error_msg
                assert "Original network error" in error_msg

    def test_zero_retries(self):
        """Test behavior with zero retries configured."""
        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.side_effect = Exception("Network error")

            with pytest.raises(ValueError, match="after 1 attempts"):
                safe_load_config_with_retry(self.test_path, max_retries=0)

            # Should only try once (no retries)
            mock_auto_config.from_pretrained.assert_called_once()

    def test_path_type_handling(self):
        """Test that different path types are handled correctly."""
        test_cases = [
            "string/path",
            Path("pathlib/path"),
            "model-with-dashes",
            "org/model-name",
        ]

        with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
            mock_auto_config.from_pretrained.return_value = self.mock_config

            for path in test_cases:
                result = safe_load_config_with_retry(path)
                assert result == self.mock_config

                # Verify the path was passed correctly
                mock_auto_config.from_pretrained.assert_called_with(path, trust_remote_code=False)
                mock_auto_config.reset_mock()

    def test_custom_lock_directory_env_var(self):
        """Test that MEGATRON_CONFIG_LOCK_DIR environment variable overrides default lock directory."""
        mock_lock = MagicMock()
        custom_lock_dir = "/custom/locks"

        with patch.dict(os.environ, {"MEGATRON_CONFIG_LOCK_DIR": custom_lock_dir}):
            with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.filelock.FileLock") as mock_filelock:
                mock_filelock.return_value = mock_lock
                mock_lock.__enter__ = Mock(return_value=mock_lock)
                mock_lock.__exit__ = Mock(return_value=None)

                with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
                    mock_auto_config.from_pretrained.return_value = self.mock_config

                    result = safe_load_config_with_retry(self.test_path)

                    # Verify custom lock directory was used
                    expected_hash = hashlib.md5(str(self.test_path).encode()).hexdigest()
                    expected_lock_file = Path(custom_lock_dir) / f".megatron_config_lock_{expected_hash}.lock"
                    mock_filelock.assert_called_once_with(str(expected_lock_file), timeout=60)

                    assert result == self.mock_config

    def test_default_lock_directory_when_env_var_not_set(self):
        """Test that default lock directory is used when MEGATRON_CONFIG_LOCK_DIR is not set."""
        mock_lock = MagicMock()

        # Ensure the environment variable is not set
        with patch.dict(os.environ, {}, clear=True):
            with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.filelock.FileLock") as mock_filelock:
                mock_filelock.return_value = mock_lock
                mock_lock.__enter__ = Mock(return_value=mock_lock)
                mock_lock.__exit__ = Mock(return_value=None)

                with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
                    mock_auto_config.from_pretrained.return_value = self.mock_config

                    result = safe_load_config_with_retry(self.test_path)

                    # Verify default lock directory was used
                    expected_hash = hashlib.md5(str(self.test_path).encode()).hexdigest()
                    expected_lock_file = (
                        Path.home() / ".cache" / "huggingface" / f".megatron_config_lock_{expected_hash}.lock"
                    )
                    mock_filelock.assert_called_once_with(str(expected_lock_file), timeout=60)

                    assert result == self.mock_config

    def test_custom_lock_directory_with_pathlib_path(self):
        """Test that custom lock directory works with pathlib.Path inputs."""
        mock_lock = MagicMock()
        custom_lock_dir = "/shared/cluster/locks"
        path_obj = Path(self.test_path)

        with patch.dict(os.environ, {"MEGATRON_CONFIG_LOCK_DIR": custom_lock_dir}):
            with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.filelock.FileLock") as mock_filelock:
                mock_filelock.return_value = mock_lock
                mock_lock.__enter__ = Mock(return_value=mock_lock)
                mock_lock.__exit__ = Mock(return_value=None)

                with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
                    mock_auto_config.from_pretrained.return_value = self.mock_config

                    result = safe_load_config_with_retry(path_obj)

                    # Verify custom lock directory was used with pathlib input
                    expected_hash = hashlib.md5(str(path_obj).encode()).hexdigest()
                    expected_lock_file = Path(custom_lock_dir) / f".megatron_config_lock_{expected_hash}.lock"
                    mock_filelock.assert_called_once_with(str(expected_lock_file), timeout=60)

                    assert result == self.mock_config

    def test_lock_directory_creation_with_env_var(self):
        """Test that custom lock directory is created if it doesn't exist."""
        custom_lock_dir = "/tmp/test_megatron_locks"

        with patch.dict(os.environ, {"MEGATRON_CONFIG_LOCK_DIR": custom_lock_dir}):
            # Mock the Path.mkdir method to verify it's called
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                mock_lock = MagicMock()

                with patch(
                    "megatron.bridge.models.hf_pretrained.safe_config_loader.filelock.FileLock"
                ) as mock_filelock:
                    mock_filelock.return_value = mock_lock
                    mock_lock.__enter__ = Mock(return_value=mock_lock)
                    mock_lock.__exit__ = Mock(return_value=None)

                    with patch(
                        "megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig"
                    ) as mock_auto_config:
                        mock_auto_config.from_pretrained.return_value = self.mock_config

                        safe_load_config_with_retry(self.test_path)

                        # Verify directory creation was attempted
                        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_different_paths_different_locks_with_custom_dir(self):
        """Test that different model paths create different lock files in custom directory."""
        mock_lock = MagicMock()
        custom_lock_dir = "/cluster/shared/locks"
        test_paths = ["model1", "model2", "org/model-name"]

        with patch.dict(os.environ, {"MEGATRON_CONFIG_LOCK_DIR": custom_lock_dir}):
            with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.filelock.FileLock") as mock_filelock:
                mock_filelock.return_value = mock_lock
                mock_lock.__enter__ = Mock(return_value=mock_lock)
                mock_lock.__exit__ = Mock(return_value=None)

                with patch("megatron.bridge.models.hf_pretrained.safe_config_loader.AutoConfig") as mock_auto_config:
                    mock_auto_config.from_pretrained.return_value = self.mock_config

                    expected_lock_files = set()

                    for path in test_paths:
                        safe_load_config_with_retry(path)

                        # Calculate expected lock file for this path
                        expected_hash = hashlib.md5(str(path).encode()).hexdigest()
                        expected_lock_file = Path(custom_lock_dir) / f".megatron_config_lock_{expected_hash}.lock"
                        expected_lock_files.add(str(expected_lock_file))

                    # Verify all expected lock files were used
                    actual_lock_files = set()
                    for call_args in mock_filelock.call_args_list:
                        actual_lock_files.add(call_args[0][0])

                    assert actual_lock_files == expected_lock_files
                    assert len(expected_lock_files) == len(test_paths)  # All should be unique
