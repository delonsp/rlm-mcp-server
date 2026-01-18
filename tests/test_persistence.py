"""
Tests for rlm_mcp.persistence module.
"""

import pytest

from rlm_mcp.persistence import PersistenceManager


class TestSaveAndLoadVariable:
    """Tests for save_variable and load_variable methods."""

    def test_roundtrip_string(self, temp_db):
        """Test saving and loading a string value."""
        pm = PersistenceManager(db_path=temp_db)

        original = "hello world"
        assert pm.save_variable("test_string", original) is True

        result = pm.load_variable("test_string")
        assert result == original

    def test_roundtrip_dict(self, temp_db):
        """Test saving and loading a dictionary value."""
        pm = PersistenceManager(db_path=temp_db)

        original = {
            "name": "test",
            "count": 42,
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [1, 2, 3],
        }
        assert pm.save_variable("test_dict", original) is True

        result = pm.load_variable("test_dict")
        assert result == original

    def test_roundtrip_list(self, temp_db):
        """Test saving and loading a list value."""
        pm = PersistenceManager(db_path=temp_db)

        original = [1, 2, 3, "four", {"five": 5}, [6, 7]]
        assert pm.save_variable("test_list", original) is True

        result = pm.load_variable("test_list")
        assert result == original

    def test_load_nonexistent_variable(self, temp_db):
        """Test loading a variable that doesn't exist returns None."""
        pm = PersistenceManager(db_path=temp_db)

        result = pm.load_variable("nonexistent")
        assert result is None

    def test_roundtrip_empty_string(self, temp_db):
        """Test saving and loading an empty string."""
        pm = PersistenceManager(db_path=temp_db)

        original = ""
        assert pm.save_variable("empty_string", original) is True

        result = pm.load_variable("empty_string")
        assert result == original

    def test_roundtrip_empty_dict(self, temp_db):
        """Test saving and loading an empty dictionary."""
        pm = PersistenceManager(db_path=temp_db)

        original = {}
        assert pm.save_variable("empty_dict", original) is True

        result = pm.load_variable("empty_dict")
        assert result == original

    def test_roundtrip_empty_list(self, temp_db):
        """Test saving and loading an empty list."""
        pm = PersistenceManager(db_path=temp_db)

        original = []
        assert pm.save_variable("empty_list", original) is True

        result = pm.load_variable("empty_list")
        assert result == original

    def test_roundtrip_with_metadata(self, temp_db):
        """Test saving and loading a variable with metadata."""
        pm = PersistenceManager(db_path=temp_db)

        original = "test value"
        metadata = {"source": "test", "version": 1}
        assert pm.save_variable("with_metadata", original, metadata=metadata) is True

        result = pm.load_variable("with_metadata")
        assert result == original

    def test_overwrite_existing_variable(self, temp_db):
        """Test that saving a variable with the same name overwrites it."""
        pm = PersistenceManager(db_path=temp_db)

        pm.save_variable("overwrite_test", "first value")
        pm.save_variable("overwrite_test", "second value")

        result = pm.load_variable("overwrite_test")
        assert result == "second value"

    def test_roundtrip_large_string(self, temp_db, sample_text):
        """Test saving and loading a large string (for compression testing)."""
        pm = PersistenceManager(db_path=temp_db)

        assert pm.save_variable("large_text", sample_text) is True

        result = pm.load_variable("large_text")
        assert result == sample_text
        assert len(result) == len(sample_text)
