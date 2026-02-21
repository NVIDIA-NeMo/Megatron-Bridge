#!/usr/bin/env python3
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


import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

try:
    from megatron.bridge.models.hf_pretrained.base import PreTrainedBase
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the Megatron-Bridge directory")
    sys.exit(1)


class MockPreTrainedBase(PreTrainedBase):
    """Mock implementation for testing."""

    ARTIFACTS = ["tokenizer"]
    OPTIONAL_ARTIFACTS = ["generation_config"]

    def __init__(self, model_name_or_path=None, trust_remote_code=False, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        super().__init__(**kwargs)

        # Mock the artifacts that save_artifacts will try to access
        self._tokenizer = Mock()
        self._tokenizer.save_pretrained = Mock()
        self._generation_config = None  # Optional artifact

    def _load_model(self):
        return Mock()

    def _load_config(self):
        return Mock()

    @property
    def tokenizer(self):
        """Mock tokenizer property."""
        return self._tokenizer

    @property
    def generation_config(self):
        """Mock generation_config property."""
        return self._generation_config


def test_copy_custom_modeling_files_basic():
    """Test basic copying of custom modeling files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Setup source directory with custom files
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # Create some custom modeling files
        (source_dir / "modeling_nemotron_h.py").write_text("# Custom modeling code")
        (source_dir / "configuration_nemotron_h.py").write_text("# Custom config code")
        (source_dir / "tokenization_nemotron_h.py").write_text("# Custom tokenizer code")
        (source_dir / "regular_file.txt").write_text("# Should not be copied")

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        # Test the copy function
        base = MockPreTrainedBase()
        copied_files = base._copy_custom_modeling_files(source_dir, target_dir)

        # Verify custom files were copied
        assert (target_dir / "modeling_nemotron_h.py").exists()
        assert (target_dir / "configuration_nemotron_h.py").exists()
        assert (target_dir / "tokenization_nemotron_h.py").exists()

        # Verify non-custom files were not copied
        assert not (target_dir / "regular_file.txt").exists()

        # Verify return value
        assert "modeling_nemotron_h.py" in copied_files
        assert "configuration_nemotron_h.py" in copied_files
        assert "tokenization_nemotron_h.py" in copied_files

        print("✅ test_copy_custom_modeling_files_basic passed")


def test_copy_custom_modeling_files_nonexistent_source():
    """Test handling of nonexistent source directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        source_dir = tmp_path / "nonexistent"
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        base = MockPreTrainedBase()
        # Should not raise exception
        copied_files = base._copy_custom_modeling_files(source_dir, target_dir)
        assert copied_files == []

        print("✅ test_copy_custom_modeling_files_nonexistent_source passed")


def test_save_artifacts_with_trust_remote_code_true():
    """Test save_artifacts preserves custom files when trust_remote_code=True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Setup source directory with custom files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "modeling_custom.py").write_text("# Custom modeling")

        target_dir = tmp_path / "target"

        # Create base with trust_remote_code=True
        base = MockPreTrainedBase(model_name_or_path=str(source_dir), trust_remote_code=True)

        # Mock the config to avoid loading issues
        mock_config = Mock()
        mock_config.save_pretrained = Mock()
        base._config = mock_config

        # Call save_artifacts
        base.save_artifacts(target_dir)

        # Verify custom file was copied
        assert (target_dir / "modeling_custom.py").exists()

        print("✅ test_save_artifacts_with_trust_remote_code_true passed")


def test_save_artifacts_with_trust_remote_code_false():
    """Test save_artifacts does not copy custom files when trust_remote_code=False."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Setup source directory with custom files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "modeling_custom.py").write_text("# Custom modeling")

        target_dir = tmp_path / "target"

        # Create base with trust_remote_code=False
        base = MockPreTrainedBase(model_name_or_path=str(source_dir), trust_remote_code=False)

        # Mock the config to avoid loading issues
        mock_config = Mock()
        mock_config.save_pretrained = Mock()
        base._config = mock_config

        # Call save_artifacts
        base.save_artifacts(target_dir)

        # Verify custom file was NOT copied
        assert not (target_dir / "modeling_custom.py").exists()

        print("✅ test_save_artifacts_with_trust_remote_code_false passed")


def test_save_artifacts_without_model_name_or_path():
    """Test save_artifacts handles missing model_name_or_path gracefully."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        target_dir = tmp_path / "target"

        # Create base without model_name_or_path
        base = MockPreTrainedBase(trust_remote_code=True)

        # Mock the config to avoid loading issues
        mock_config = Mock()
        mock_config.save_pretrained = Mock()
        base._config = mock_config

        # Should not raise exception
        base.save_artifacts(target_dir)

        print("✅ test_save_artifacts_without_model_name_or_path passed")


def test_copy_handles_permission_errors():
    """Test that copy failures are handled gracefully."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "modeling_test.py").write_text("# Test content")

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        base = MockPreTrainedBase()

        # Mock shutil.copy2 to raise an exception
        with patch("shutil.copy2", side_effect=OSError("Permission denied")):
            # Should not raise exception
            copied_files = base._copy_custom_modeling_files(source_dir, target_dir)

        # File should not exist due to copy failure
        assert not (target_dir / "modeling_test.py").exists()
        assert copied_files == []

        print("✅ test_copy_handles_permission_errors passed")


def test_additional_files_exact_names():
    """Test copying additional files with exact file names."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Setup source directory with various files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "vocab.json").write_text('{"vocab": "data"}')
        (source_dir / "merges.txt").write_text("merge data")
        (source_dir / "special_tokens.json").write_text('{"special": "tokens"}')
        (source_dir / "ignore_me.json").write_text('{"ignore": "this"}')

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        # Test copying specific files
        base = MockPreTrainedBase(model_name_or_path=str(source_dir))
        copied_files = base._copy_custom_modeling_files(
            source_dir, target_dir, file_patterns=["vocab.json", "merges.txt"]
        )

        # Verify only specified files were copied
        assert (target_dir / "vocab.json").exists()
        assert (target_dir / "merges.txt").exists()
        assert not (target_dir / "special_tokens.json").exists()
        assert not (target_dir / "ignore_me.json").exists()

        # Verify return value
        assert "vocab.json" in copied_files
        assert "merges.txt" in copied_files
        assert len(copied_files) == 2

        print("✅ test_additional_files_exact_names passed")


def test_additional_files_glob_patterns():
    """Test copying additional files with glob patterns."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Setup source directory with various files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "config.json").write_text('{"config": "data"}')
        (source_dir / "vocab.json").write_text('{"vocab": "data"}')
        (source_dir / "merges.txt").write_text("merge data")
        (source_dir / "readme.md").write_text("# README")

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        # Test copying with glob patterns
        base = MockPreTrainedBase(model_name_or_path=str(source_dir))
        copied_files = base._copy_custom_modeling_files(
            source_dir, target_dir, file_patterns=["*.json", "*.md"]
        )

        # Verify files matching patterns were copied
        assert (target_dir / "config.json").exists()
        assert (target_dir / "vocab.json").exists()
        assert (target_dir / "readme.md").exists()
        assert not (target_dir / "merges.txt").exists()

        # Verify return value
        assert "config.json" in copied_files
        assert "vocab.json" in copied_files
        assert "readme.md" in copied_files

        print("✅ test_additional_files_glob_patterns passed")


def test_additional_files_mixed_patterns_and_names():
    """Test copying additional files with mixed patterns and exact names."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Setup source directory
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "config.json").write_text('{"config": "data"}')
        (source_dir / "vocab.json").write_text('{"vocab": "data"}')
        (source_dir / "special_file.txt").write_text("special content")
        (source_dir / "readme.md").write_text("# README")

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        # Test copying with mixed patterns and exact names
        base = MockPreTrainedBase(model_name_or_path=str(source_dir))
        copied_files = base._copy_custom_modeling_files(
            source_dir, target_dir, file_patterns=["*.json", "special_file.txt", "readme.md"]
        )

        # Verify all specified files were copied
        assert (target_dir / "config.json").exists()
        assert (target_dir / "vocab.json").exists()
        assert (target_dir / "special_file.txt").exists()
        assert (target_dir / "readme.md").exists()

        print("✅ test_additional_files_mixed_patterns_and_names passed")


def test_save_artifacts_with_additional_files():
    """Test save_artifacts copies additional files when specified."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Setup source directory with additional files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "vocab.json").write_text('{"vocab": "data"}')
        (source_dir / "custom_config.yaml").write_text("custom: config")
        (source_dir / "readme.md").write_text("# Model README")

        target_dir = tmp_path / "target"

        # Create base
        base = MockPreTrainedBase(model_name_or_path=str(source_dir))

        # Mock the config to avoid loading issues
        mock_config = Mock()
        mock_config.save_pretrained = Mock()
        base._config = mock_config

        # Call save_artifacts with additional_files
        base.save_artifacts(target_dir, additional_files=["vocab.json", "*.yaml", "readme.md"])

        # Verify additional files were copied
        assert (target_dir / "vocab.json").exists()
        assert (target_dir / "custom_config.yaml").exists()
        assert (target_dir / "readme.md").exists()

        print("✅ test_save_artifacts_with_additional_files passed")


def test_save_artifacts_with_additional_files_and_trust_remote_code():
    """Test save_artifacts copies both custom modeling files and additional files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Setup source directory with both custom modeling and additional files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "modeling_custom.py").write_text("# Custom modeling")
        (source_dir / "vocab.json").write_text('{"vocab": "data"}')
        (source_dir / "readme.md").write_text("# Model README")

        target_dir = tmp_path / "target"

        # Create base with trust_remote_code=True
        base = MockPreTrainedBase(model_name_or_path=str(source_dir), trust_remote_code=True)

        # Mock the config to avoid loading issues
        mock_config = Mock()
        mock_config.save_pretrained = Mock()
        base._config = mock_config

        # Call save_artifacts with additional_files
        base.save_artifacts(target_dir, additional_files=["vocab.json", "readme.md"])

        # Verify both custom modeling file and additional files were copied
        assert (target_dir / "modeling_custom.py").exists()  # From trust_remote_code
        assert (target_dir / "vocab.json").exists()  # From additional_files
        assert (target_dir / "readme.md").exists()  # From additional_files

        print("✅ test_save_artifacts_with_additional_files_and_trust_remote_code passed")


def test_additional_files_with_original_source_path():
    """Test additional_files uses original_source_path when provided."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Setup original source directory
        original_source_dir = tmp_path / "original_source"
        original_source_dir.mkdir()
        (original_source_dir / "vocab.json").write_text('{"vocab": "from_original"}')

        # Setup model directory (different from original source)
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "vocab.json").write_text('{"vocab": "from_model"}')

        target_dir = tmp_path / "target"

        # Create base with model_name_or_path pointing to model_dir
        base = MockPreTrainedBase(model_name_or_path=str(model_dir))

        # Mock the config
        mock_config = Mock()
        mock_config.save_pretrained = Mock()
        base._config = mock_config

        # Call save_artifacts with original_source_path
        base.save_artifacts(
            target_dir, original_source_path=str(original_source_dir), additional_files=["vocab.json"]
        )

        # Verify file from original_source_path was copied (not from model_dir)
        assert (target_dir / "vocab.json").exists()
        content = (target_dir / "vocab.json").read_text()
        assert "from_original" in content

        print("✅ test_additional_files_with_original_source_path passed")


def main():
    """Run all tests."""
    print("Running PreTrainedBase custom modeling file preservation tests...")

    try:
        test_copy_custom_modeling_files_basic()
        test_copy_custom_modeling_files_nonexistent_source()
        test_save_artifacts_with_trust_remote_code_true()
        test_save_artifacts_with_trust_remote_code_false()
        test_save_artifacts_without_model_name_or_path()
        test_copy_handles_permission_errors()

        # New tests for additional_files feature
        test_additional_files_exact_names()
        test_additional_files_glob_patterns()
        test_additional_files_mixed_patterns_and_names()
        test_save_artifacts_with_additional_files()
        test_save_artifacts_with_additional_files_and_trust_remote_code()
        test_additional_files_with_original_source_path()

        print("\n🎉 All tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
