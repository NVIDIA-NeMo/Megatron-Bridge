"""Tests for PreTrainedBase custom modeling file preservation functionality.

This is a standalone test file that doesn't depend on global conftest.py.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import sys

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
        base = MockPreTrainedBase(
            model_name_or_path=str(source_dir),
            trust_remote_code=True
        )
        
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
        base = MockPreTrainedBase(
            model_name_or_path=str(source_dir),
            trust_remote_code=False
        )
        
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
        with patch('shutil.copy2', side_effect=OSError("Permission denied")):
            # Should not raise exception
            copied_files = base._copy_custom_modeling_files(source_dir, target_dir)
            
        # File should not exist due to copy failure
        assert not (target_dir / "modeling_test.py").exists()
        assert copied_files == []
        
        print("✅ test_copy_handles_permission_errors passed")


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
        
        print("\n🎉 All tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 