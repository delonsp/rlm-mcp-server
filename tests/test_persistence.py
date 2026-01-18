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


class TestListVariables:
    """Tests for list_variables method."""

    def test_list_variables_empty_database(self, temp_db):
        """Test listing variables on an empty database returns empty list."""
        pm = PersistenceManager(db_path=temp_db)

        result = pm.list_variables()
        assert result == []

    def test_list_variables_returns_correct_metadata(self, temp_db):
        """Test that list_variables returns all expected metadata fields."""
        pm = PersistenceManager(db_path=temp_db)

        # Save a string variable
        pm.save_variable("test_var", "hello world")

        result = pm.list_variables()

        assert len(result) == 1
        var_info = result[0]

        # Check all required fields exist
        assert "name" in var_info
        assert "type" in var_info
        assert "size_bytes" in var_info
        assert "created_at" in var_info
        assert "updated_at" in var_info

        # Check values
        assert var_info["name"] == "test_var"
        assert var_info["type"] == "str"
        assert var_info["size_bytes"] > 0
        assert var_info["created_at"] is not None
        assert var_info["updated_at"] is not None

    def test_list_variables_correct_types(self, temp_db):
        """Test that type_name is correctly stored for different types."""
        pm = PersistenceManager(db_path=temp_db)

        # Save variables of different types
        pm.save_variable("string_var", "hello")
        pm.save_variable("dict_var", {"key": "value"})
        pm.save_variable("list_var", [1, 2, 3])
        pm.save_variable("int_var", 42)

        result = pm.list_variables()
        vars_by_name = {v["name"]: v for v in result}

        assert vars_by_name["string_var"]["type"] == "str"
        assert vars_by_name["dict_var"]["type"] == "dict"
        assert vars_by_name["list_var"]["type"] == "list"
        assert vars_by_name["int_var"]["type"] == "int"

    def test_list_variables_multiple(self, temp_db):
        """Test listing multiple variables."""
        pm = PersistenceManager(db_path=temp_db)

        # Save multiple variables
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", {"data": 2})
        pm.save_variable("var3", [1, 2, 3])

        result = pm.list_variables()

        assert len(result) == 3
        names = {v["name"] for v in result}
        assert names == {"var1", "var2", "var3"}

    def test_list_variables_ordered_by_updated_at_desc(self, temp_db):
        """Test that variables are ordered by updated_at descending."""
        import time

        pm = PersistenceManager(db_path=temp_db)

        # Save variables with small delays to ensure different timestamps
        pm.save_variable("first", "1")
        time.sleep(0.01)
        pm.save_variable("second", "2")
        time.sleep(0.01)
        pm.save_variable("third", "3")

        result = pm.list_variables()

        # Most recently updated should be first
        assert result[0]["name"] == "third"
        assert result[1]["name"] == "second"
        assert result[2]["name"] == "first"

    def test_list_variables_size_bytes_accurate(self, temp_db):
        """Test that size_bytes reflects the pickled size of the data."""
        import pickle

        pm = PersistenceManager(db_path=temp_db)

        # Save a known value
        test_value = "hello world"
        pm.save_variable("sized_var", test_value)

        result = pm.list_variables()
        var_info = result[0]

        # size_bytes should match the pickled size (before compression)
        expected_size = len(pickle.dumps(test_value))
        assert var_info["size_bytes"] == expected_size

    def test_list_variables_after_update(self, temp_db):
        """Test that updated_at changes when variable is overwritten."""
        import time

        pm = PersistenceManager(db_path=temp_db)

        # Save initial variable
        pm.save_variable("update_test", "initial")
        result1 = pm.list_variables()
        initial_updated_at = result1[0]["updated_at"]
        initial_created_at = result1[0]["created_at"]

        # Wait and update
        time.sleep(0.01)
        pm.save_variable("update_test", "updated")

        result2 = pm.list_variables()
        new_updated_at = result2[0]["updated_at"]
        new_created_at = result2[0]["created_at"]

        # created_at should stay the same, updated_at should change
        assert new_created_at == initial_created_at
        assert new_updated_at > initial_updated_at


class TestSaveAndLoadIndex:
    """Tests for save_index and load_index methods."""

    def test_roundtrip_simple_index(self, temp_db):
        """Test saving and loading a simple index with term -> positions mapping."""
        pm = PersistenceManager(db_path=temp_db)

        # Create a simple index: term -> list of positions
        index_data = {
            "medo": [0, 100, 250],
            "ansiedade": [50, 300],
            "trabalho": [150, 400, 550],
        }
        assert pm.save_index("test_var", index_data) is True

        result = pm.load_index("test_var")
        assert result == index_data

    def test_roundtrip_empty_index(self, temp_db):
        """Test saving and loading an empty index."""
        pm = PersistenceManager(db_path=temp_db)

        index_data = {}
        assert pm.save_index("empty_index_var", index_data) is True

        result = pm.load_index("empty_index_var")
        assert result == index_data

    def test_load_nonexistent_index(self, temp_db):
        """Test loading an index that doesn't exist returns None."""
        pm = PersistenceManager(db_path=temp_db)

        result = pm.load_index("nonexistent")
        assert result is None

    def test_roundtrip_large_index(self, temp_db):
        """Test saving and loading an index with many terms."""
        pm = PersistenceManager(db_path=temp_db)

        # Create a large index with 1000 terms
        index_data = {
            f"term_{i}": [j * 100 for j in range(10)]  # Each term has 10 positions
            for i in range(1000)
        }
        assert pm.save_index("large_index_var", index_data) is True

        result = pm.load_index("large_index_var")
        assert result == index_data
        assert len(result) == 1000

    def test_overwrite_existing_index(self, temp_db):
        """Test that saving an index with the same var_name overwrites it."""
        pm = PersistenceManager(db_path=temp_db)

        # Save initial index
        initial_index = {"term1": [0, 10], "term2": [20]}
        pm.save_index("overwrite_var", initial_index)

        # Overwrite with new index
        new_index = {"new_term": [100, 200, 300]}
        pm.save_index("overwrite_var", new_index)

        result = pm.load_index("overwrite_var")
        assert result == new_index
        assert "term1" not in result

    def test_index_without_associated_variable(self, temp_db):
        """Test that index can be saved without an associated variable."""
        pm = PersistenceManager(db_path=temp_db)

        # Save index without saving a variable first
        # (The foreign key is defined but SQLite doesn't enforce it by default)
        index_data = {"orphan_term": [0, 50]}
        assert pm.save_index("orphan_var", index_data) is True

        result = pm.load_index("orphan_var")
        assert result == index_data

    def test_index_terms_with_special_characters(self, temp_db):
        """Test index with terms containing special characters."""
        pm = PersistenceManager(db_path=temp_db)

        index_data = {
            "medo-pânico": [0, 100],
            "trabalho/estresse": [50],
            "família (nuclear)": [200, 300],
            "síndrome@burnout": [400],
        }
        assert pm.save_index("special_var", index_data) is True

        result = pm.load_index("special_var")
        assert result == index_data

    def test_index_preserves_position_order(self, temp_db):
        """Test that position lists preserve their order."""
        pm = PersistenceManager(db_path=temp_db)

        # Positions intentionally not sorted
        index_data = {
            "termo": [500, 100, 300, 50, 999],
        }
        assert pm.save_index("order_var", index_data) is True

        result = pm.load_index("order_var")
        assert result["termo"] == [500, 100, 300, 50, 999]

    def test_multiple_indexes_independent(self, temp_db):
        """Test that multiple indexes for different variables are independent."""
        pm = PersistenceManager(db_path=temp_db)

        index1 = {"term_a": [0, 10]}
        index2 = {"term_b": [100, 200]}
        index3 = {"term_c": [300]}

        pm.save_index("var1", index1)
        pm.save_index("var2", index2)
        pm.save_index("var3", index3)

        assert pm.load_index("var1") == index1
        assert pm.load_index("var2") == index2
        assert pm.load_index("var3") == index3


class TestClearAll:
    """Tests for clear_all method."""

    def test_clear_all_returns_count_of_removed_variables(self, temp_db):
        """Test that clear_all returns the number of variables removed."""
        pm = PersistenceManager(db_path=temp_db)

        # Save multiple variables
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", "value2")
        pm.save_variable("var3", "value3")

        result = pm.clear_all()
        assert result == 3

    def test_clear_all_removes_all_variables(self, temp_db):
        """Test that clear_all removes all variables from the database."""
        pm = PersistenceManager(db_path=temp_db)

        # Save multiple variables
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", {"key": "value"})
        pm.save_variable("var3", [1, 2, 3])

        # Clear all
        pm.clear_all()

        # All variables should be gone
        assert pm.load_variable("var1") is None
        assert pm.load_variable("var2") is None
        assert pm.load_variable("var3") is None

    def test_clear_all_removes_all_indices(self, temp_db):
        """Test that clear_all removes all indices from the database."""
        pm = PersistenceManager(db_path=temp_db)

        # Save variables with indices
        pm.save_variable("var1", "text1")
        pm.save_variable("var2", "text2")
        pm.save_index("var1", {"term1": [0, 10]})
        pm.save_index("var2", {"term2": [0, 20]})

        # Clear all
        pm.clear_all()

        # All indices should be gone
        assert pm.load_index("var1") is None
        assert pm.load_index("var2") is None

    def test_clear_all_empty_database_returns_zero(self, temp_db):
        """Test that clear_all on empty database returns 0."""
        pm = PersistenceManager(db_path=temp_db)

        result = pm.clear_all()
        assert result == 0

    def test_clear_all_list_variables_empty_after(self, temp_db):
        """Test that list_variables returns empty list after clear_all."""
        pm = PersistenceManager(db_path=temp_db)

        # Save some variables
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", "value2")

        # Verify they exist
        assert len(pm.list_variables()) == 2

        # Clear all
        pm.clear_all()

        # list_variables should return empty list
        assert pm.list_variables() == []

    def test_clear_all_can_add_variables_after(self, temp_db):
        """Test that variables can be added after clear_all."""
        pm = PersistenceManager(db_path=temp_db)

        # Save and clear
        pm.save_variable("old_var", "old value")
        pm.clear_all()

        # Should be able to add new variables
        assert pm.save_variable("new_var", "new value") is True
        assert pm.load_variable("new_var") == "new value"

    def test_clear_all_preserves_collections(self, temp_db):
        """Test that clear_all does not remove collections (only variables and indices)."""
        pm = PersistenceManager(db_path=temp_db)

        # Create a collection and add variables
        pm.create_collection("test_collection", "Test description")
        pm.save_variable("var1", "value1")
        pm.add_to_collection("test_collection", ["var1"])

        # Clear all variables
        pm.clear_all()

        # Collection should still exist (though empty)
        collections = pm.list_collections()
        assert len(collections) == 1
        assert collections[0]["name"] == "test_collection"


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats_empty_database(self, temp_db):
        """Test get_stats on an empty database returns zeros."""
        pm = PersistenceManager(db_path=temp_db)

        stats = pm.get_stats()

        assert stats["variables_count"] == 0
        assert stats["variables_total_size"] == 0
        assert stats["indices_count"] == 0
        assert stats["total_indexed_terms"] == 0
        assert stats["db_file_size"] > 0  # SQLite file exists with schema
        assert stats["db_path"] == temp_db

    def test_get_stats_returns_all_expected_keys(self, temp_db):
        """Test get_stats returns all expected dictionary keys."""
        pm = PersistenceManager(db_path=temp_db)

        stats = pm.get_stats()

        expected_keys = {
            "variables_count",
            "variables_total_size",
            "indices_count",
            "total_indexed_terms",
            "db_file_size",
            "db_path",
        }
        assert set(stats.keys()) == expected_keys

    def test_get_stats_variables_count(self, temp_db):
        """Test get_stats returns correct variables_count."""
        pm = PersistenceManager(db_path=temp_db)

        # Add variables
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", "value2")
        pm.save_variable("var3", "value3")

        stats = pm.get_stats()
        assert stats["variables_count"] == 3

    def test_get_stats_variables_total_size(self, temp_db):
        """Test get_stats returns correct variables_total_size (sum of size_bytes)."""
        import pickle

        pm = PersistenceManager(db_path=temp_db)

        # Add variables with known sizes
        val1 = "hello"
        val2 = {"key": "value"}
        val3 = [1, 2, 3, 4, 5]

        pm.save_variable("var1", val1)
        pm.save_variable("var2", val2)
        pm.save_variable("var3", val3)

        expected_total = (
            len(pickle.dumps(val1))
            + len(pickle.dumps(val2))
            + len(pickle.dumps(val3))
        )

        stats = pm.get_stats()
        assert stats["variables_total_size"] == expected_total

    def test_get_stats_indices_count(self, temp_db):
        """Test get_stats returns correct indices_count."""
        pm = PersistenceManager(db_path=temp_db)

        # Add indices
        pm.save_index("var1", {"term1": [0, 10]})
        pm.save_index("var2", {"term2": [20, 30]})

        stats = pm.get_stats()
        assert stats["indices_count"] == 2

    def test_get_stats_total_indexed_terms(self, temp_db):
        """Test get_stats returns correct total_indexed_terms (sum of terms_count)."""
        pm = PersistenceManager(db_path=temp_db)

        # Add indices with known term counts
        pm.save_index("var1", {"term1": [0], "term2": [10], "term3": [20]})  # 3 terms
        pm.save_index("var2", {"termA": [0], "termB": [10]})  # 2 terms
        pm.save_index("var3", {"only_term": [0]})  # 1 term

        stats = pm.get_stats()
        assert stats["total_indexed_terms"] == 6  # 3 + 2 + 1

    def test_get_stats_db_file_size_is_positive(self, temp_db):
        """Test that db_file_size is a positive integer representing actual file size."""
        import os

        pm = PersistenceManager(db_path=temp_db)

        # Add some data
        pm.save_variable("var1", "value1")
        pm.save_index("var1", {"term": [0, 10]})

        stats = pm.get_stats()

        # File size should match actual file size on disk
        actual_size = os.path.getsize(temp_db)
        assert stats["db_file_size"] == actual_size
        assert stats["db_file_size"] > 0

    def test_get_stats_db_path_correct(self, temp_db):
        """Test that db_path in stats matches the configured path."""
        pm = PersistenceManager(db_path=temp_db)

        stats = pm.get_stats()
        assert stats["db_path"] == temp_db

    def test_get_stats_after_clear_all(self, temp_db):
        """Test get_stats returns zeros after clear_all."""
        pm = PersistenceManager(db_path=temp_db)

        # Add data
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", "value2")
        pm.save_index("var1", {"term": [0, 10, 20]})

        # Verify data exists
        stats_before = pm.get_stats()
        assert stats_before["variables_count"] == 2
        assert stats_before["indices_count"] == 1

        # Clear all
        pm.clear_all()

        # Stats should show zeros
        stats_after = pm.get_stats()
        assert stats_after["variables_count"] == 0
        assert stats_after["variables_total_size"] == 0
        assert stats_after["indices_count"] == 0
        assert stats_after["total_indexed_terms"] == 0

    def test_get_stats_with_mixed_data(self, temp_db):
        """Test get_stats with variables of different types and indices."""
        import pickle

        pm = PersistenceManager(db_path=temp_db)

        # Add variables of different types
        string_val = "hello world"
        dict_val = {"nested": {"key": [1, 2, 3]}}
        list_val = list(range(100))
        int_val = 42

        pm.save_variable("str_var", string_val)
        pm.save_variable("dict_var", dict_val)
        pm.save_variable("list_var", list_val)
        pm.save_variable("int_var", int_val)

        # Add indices for some variables
        pm.save_index("str_var", {"hello": [0], "world": [6]})
        pm.save_index("dict_var", {"nested": [0, 100, 200]})

        expected_size = (
            len(pickle.dumps(string_val))
            + len(pickle.dumps(dict_val))
            + len(pickle.dumps(list_val))
            + len(pickle.dumps(int_val))
        )

        stats = pm.get_stats()

        assert stats["variables_count"] == 4
        assert stats["variables_total_size"] == expected_size
        assert stats["indices_count"] == 2
        assert stats["total_indexed_terms"] == 3  # 2 terms + 1 term (number of keys in each index)
