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


class TestDeleteVariable:
    """Tests for delete_variable method."""

    def test_delete_existing_variable(self, temp_db):
        """Test deleting an existing variable removes it from the database."""
        pm = PersistenceManager(db_path=temp_db)

        # Save a variable
        pm.save_variable("to_delete", "some value")
        assert pm.load_variable("to_delete") == "some value"

        # Delete it
        result = pm.delete_variable("to_delete")
        assert result is True

        # Verify it's gone
        assert pm.load_variable("to_delete") is None

    def test_delete_nonexistent_variable(self, temp_db):
        """Test deleting a variable that doesn't exist still returns True."""
        pm = PersistenceManager(db_path=temp_db)

        # Delete something that doesn't exist (SQLite DELETE succeeds even if no rows)
        result = pm.delete_variable("nonexistent")
        assert result is True

    def test_delete_removes_associated_index(self, temp_db):
        """Test deleting a variable also removes its associated index."""
        pm = PersistenceManager(db_path=temp_db)

        # Save variable and its index
        pm.save_variable("indexed_var", "some text")
        pm.save_index("indexed_var", {"term1": [0, 10], "term2": [20]})

        # Verify index exists
        assert pm.load_index("indexed_var") is not None

        # Delete variable
        pm.delete_variable("indexed_var")

        # Both variable and index should be gone
        assert pm.load_variable("indexed_var") is None
        assert pm.load_index("indexed_var") is None

    def test_delete_does_not_affect_other_variables(self, temp_db):
        """Test deleting one variable doesn't affect others."""
        pm = PersistenceManager(db_path=temp_db)

        # Save multiple variables
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", "value2")
        pm.save_variable("var3", "value3")

        # Delete one
        pm.delete_variable("var2")

        # Others should still exist
        assert pm.load_variable("var1") == "value1"
        assert pm.load_variable("var2") is None
        assert pm.load_variable("var3") == "value3"
