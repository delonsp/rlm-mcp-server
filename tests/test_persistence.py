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


class TestCreateCollectionAndListCollections:
    """Tests for create_collection and list_collections methods."""

    def test_create_collection_returns_true(self, temp_db):
        """Test that create_collection returns True on success."""
        pm = PersistenceManager(db_path=temp_db)

        result = pm.create_collection("test_collection")
        assert result is True

    def test_create_collection_with_description(self, temp_db):
        """Test creating a collection with a description."""
        pm = PersistenceManager(db_path=temp_db)

        result = pm.create_collection("my_collection", description="A test collection")
        assert result is True

        # Verify by listing
        collections = pm.list_collections()
        assert len(collections) == 1
        assert collections[0]["name"] == "my_collection"
        assert collections[0]["description"] == "A test collection"

    def test_create_collection_without_description(self, temp_db):
        """Test creating a collection without a description stores None."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("no_desc_collection")

        collections = pm.list_collections()
        assert len(collections) == 1
        assert collections[0]["name"] == "no_desc_collection"
        assert collections[0]["description"] is None

    def test_create_collection_sets_created_at(self, temp_db):
        """Test that create_collection sets created_at timestamp."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("timestamped_collection")

        collections = pm.list_collections()
        assert len(collections) == 1
        assert collections[0]["created_at"] is not None
        # ISO format timestamp should have T separator
        assert "T" in collections[0]["created_at"]

    def test_create_collection_overwrite_preserves_created_at(self, temp_db):
        """Test that creating a collection with same name preserves original created_at."""
        import time

        pm = PersistenceManager(db_path=temp_db)

        # Create initial collection
        pm.create_collection("overwrite_test", description="first")
        collections1 = pm.list_collections()
        original_created_at = collections1[0]["created_at"]

        # Wait and overwrite
        time.sleep(0.01)
        pm.create_collection("overwrite_test", description="second")

        collections2 = pm.list_collections()
        assert collections2[0]["created_at"] == original_created_at
        assert collections2[0]["description"] == "second"

    def test_list_collections_empty_database(self, temp_db):
        """Test list_collections returns empty list on empty database."""
        pm = PersistenceManager(db_path=temp_db)

        result = pm.list_collections()
        assert result == []

    def test_list_collections_returns_correct_fields(self, temp_db):
        """Test that list_collections returns all expected fields."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("test_collection", description="Test")

        collections = pm.list_collections()
        assert len(collections) == 1

        collection = collections[0]
        assert "name" in collection
        assert "description" in collection
        assert "created_at" in collection
        assert "var_count" in collection

    def test_list_collections_multiple(self, temp_db):
        """Test listing multiple collections."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("collection_a", description="First")
        pm.create_collection("collection_b", description="Second")
        pm.create_collection("collection_c", description="Third")

        collections = pm.list_collections()
        assert len(collections) == 3

        names = {c["name"] for c in collections}
        assert names == {"collection_a", "collection_b", "collection_c"}

    def test_list_collections_ordered_by_name(self, temp_db):
        """Test that collections are ordered alphabetically by name."""
        pm = PersistenceManager(db_path=temp_db)

        # Create in non-alphabetical order
        pm.create_collection("zebra")
        pm.create_collection("apple")
        pm.create_collection("mango")

        collections = pm.list_collections()

        assert collections[0]["name"] == "apple"
        assert collections[1]["name"] == "mango"
        assert collections[2]["name"] == "zebra"

    def test_list_collections_var_count_empty(self, temp_db):
        """Test that var_count is 0 for empty collection."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("empty_collection")

        collections = pm.list_collections()
        assert collections[0]["var_count"] == 0

    def test_list_collections_var_count_with_variables(self, temp_db):
        """Test that var_count reflects number of variables in collection."""
        pm = PersistenceManager(db_path=temp_db)

        # Create collection and add variables
        pm.create_collection("filled_collection")
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", "value2")
        pm.save_variable("var3", "value3")
        pm.add_to_collection("filled_collection", ["var1", "var2", "var3"])

        collections = pm.list_collections()
        assert collections[0]["var_count"] == 3

    def test_create_collection_with_special_characters(self, temp_db):
        """Test creating a collection with special characters in name and description."""
        pm = PersistenceManager(db_path=temp_db)

        result = pm.create_collection(
            "homeopatia-unicista_2024",
            description="Materiais de homeopatia (unicista) - Scholten & Kent"
        )
        assert result is True

        collections = pm.list_collections()
        assert len(collections) == 1
        assert collections[0]["name"] == "homeopatia-unicista_2024"
        assert "Scholten & Kent" in collections[0]["description"]

    def test_create_multiple_collections_independent(self, temp_db):
        """Test that multiple collections are independent."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("collection_1", description="First collection")
        pm.create_collection("collection_2", description="Second collection")

        # Add variables to first collection only
        pm.save_variable("var_a", "value_a")
        pm.add_to_collection("collection_1", ["var_a"])

        collections = pm.list_collections()
        coll_by_name = {c["name"]: c for c in collections}

        assert coll_by_name["collection_1"]["var_count"] == 1
        assert coll_by_name["collection_2"]["var_count"] == 0


class TestAddToCollectionAndGetCollectionVars:
    """Tests for add_to_collection and get_collection_vars methods."""

    def test_add_to_collection_returns_count(self, temp_db):
        """Test that add_to_collection returns the number of variables added."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("test_collection")
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", "value2")

        result = pm.add_to_collection("test_collection", ["var1", "var2"])
        assert result == 2

    def test_add_to_collection_creates_collection_if_not_exists(self, temp_db):
        """Test that add_to_collection creates the collection if it doesn't exist."""
        pm = PersistenceManager(db_path=temp_db)

        pm.save_variable("var1", "value1")

        # Collection doesn't exist yet
        assert pm.list_collections() == []

        # add_to_collection should create it
        result = pm.add_to_collection("auto_created", ["var1"])
        assert result == 1

        # Verify collection was created
        collections = pm.list_collections()
        assert len(collections) == 1
        assert collections[0]["name"] == "auto_created"

    def test_add_to_collection_ignores_duplicates(self, temp_db):
        """Test that adding the same variable twice doesn't duplicate it."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("test_collection")
        pm.save_variable("var1", "value1")

        # Add once
        result1 = pm.add_to_collection("test_collection", ["var1"])
        assert result1 == 1

        # Add again - should be ignored
        result2 = pm.add_to_collection("test_collection", ["var1"])
        assert result2 == 0

        # Collection should still have only one variable
        vars = pm.get_collection_vars("test_collection")
        assert vars == ["var1"]

    def test_add_to_collection_partial_duplicates(self, temp_db):
        """Test adding a mix of new and existing variables."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("test_collection")
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", "value2")
        pm.save_variable("var3", "value3")

        # Add var1
        pm.add_to_collection("test_collection", ["var1"])

        # Add var1 (duplicate) and var2, var3 (new)
        result = pm.add_to_collection("test_collection", ["var1", "var2", "var3"])
        assert result == 2  # Only var2 and var3 are new

        vars = pm.get_collection_vars("test_collection")
        assert set(vars) == {"var1", "var2", "var3"}

    def test_add_to_collection_empty_list(self, temp_db):
        """Test adding an empty list of variables."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("test_collection")

        result = pm.add_to_collection("test_collection", [])
        assert result == 0

    def test_add_to_collection_nonexistent_variables(self, temp_db):
        """Test adding variables that don't exist in the database."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("test_collection")

        # Variables don't exist, but add_to_collection still creates associations
        # (foreign key not enforced by SQLite by default)
        result = pm.add_to_collection("test_collection", ["nonexistent1", "nonexistent2"])
        assert result == 2

        vars = pm.get_collection_vars("test_collection")
        assert set(vars) == {"nonexistent1", "nonexistent2"}

    def test_add_to_collection_sets_added_at(self, temp_db):
        """Test that add_to_collection sets added_at timestamp."""
        import sqlite3

        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("test_collection")
        pm.save_variable("var1", "value1")
        pm.add_to_collection("test_collection", ["var1"])

        # Query database directly to check added_at
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT added_at FROM collection_vars WHERE collection_name = ? AND var_name = ?",
                ("test_collection", "var1")
            )
            row = cursor.fetchone()
            assert row is not None
            assert "T" in row[0]  # ISO format with T separator

    def test_get_collection_vars_returns_list(self, temp_db):
        """Test that get_collection_vars returns a list of variable names."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("test_collection")
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", "value2")
        pm.add_to_collection("test_collection", ["var1", "var2"])

        result = pm.get_collection_vars("test_collection")

        assert isinstance(result, list)
        assert set(result) == {"var1", "var2"}

    def test_get_collection_vars_empty_collection(self, temp_db):
        """Test get_collection_vars for an empty collection."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("empty_collection")

        result = pm.get_collection_vars("empty_collection")
        assert result == []

    def test_get_collection_vars_nonexistent_collection(self, temp_db):
        """Test get_collection_vars for a collection that doesn't exist."""
        pm = PersistenceManager(db_path=temp_db)

        result = pm.get_collection_vars("nonexistent")
        assert result == []

    def test_get_collection_vars_ordered_by_name(self, temp_db):
        """Test that get_collection_vars returns variables ordered by name."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("test_collection")
        pm.save_variable("zebra", "z")
        pm.save_variable("apple", "a")
        pm.save_variable("mango", "m")

        pm.add_to_collection("test_collection", ["zebra", "apple", "mango"])

        result = pm.get_collection_vars("test_collection")

        assert result == ["apple", "mango", "zebra"]

    def test_add_to_multiple_collections(self, temp_db):
        """Test adding the same variable to multiple collections."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("collection_a")
        pm.create_collection("collection_b")
        pm.save_variable("shared_var", "shared value")

        pm.add_to_collection("collection_a", ["shared_var"])
        pm.add_to_collection("collection_b", ["shared_var"])

        # Both collections should have the variable
        assert pm.get_collection_vars("collection_a") == ["shared_var"]
        assert pm.get_collection_vars("collection_b") == ["shared_var"]

    def test_add_to_collection_many_variables(self, temp_db):
        """Test adding many variables to a collection at once."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("large_collection")

        # Create 50 variables
        var_names = [f"var_{i:03d}" for i in range(50)]
        for name in var_names:
            pm.save_variable(name, f"value for {name}")

        # Add all at once
        result = pm.add_to_collection("large_collection", var_names)
        assert result == 50

        # Verify all were added
        vars = pm.get_collection_vars("large_collection")
        assert len(vars) == 50
        assert vars[0] == "var_000"  # Ordered alphabetically
        assert vars[-1] == "var_049"

    def test_add_to_collection_with_special_characters(self, temp_db):
        """Test adding variables with special characters in names."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("test_collection")
        pm.save_variable("var-with-dashes", "value1")
        pm.save_variable("var_with_underscores", "value2")
        pm.save_variable("var.with.dots", "value3")

        result = pm.add_to_collection(
            "test_collection",
            ["var-with-dashes", "var_with_underscores", "var.with.dots"]
        )
        assert result == 3

        vars = pm.get_collection_vars("test_collection")
        assert len(vars) == 3


class TestDeleteCollection:
    """Tests for delete_collection method."""

    def test_delete_collection_returns_true(self, temp_db):
        """Test that delete_collection returns True on success."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("test_collection", description="Test")

        result = pm.delete_collection("test_collection")
        assert result is True

    def test_delete_collection_removes_from_database(self, temp_db):
        """Test that delete_collection removes the collection from the database."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("to_delete", description="Will be deleted")

        # Verify it exists
        collections = pm.list_collections()
        assert len(collections) == 1
        assert collections[0]["name"] == "to_delete"

        # Delete it
        pm.delete_collection("to_delete")

        # Verify it's gone
        collections = pm.list_collections()
        assert len(collections) == 0

    def test_delete_collection_removes_associations(self, temp_db):
        """Test that delete_collection removes associations from collection_vars table."""
        import sqlite3

        pm = PersistenceManager(db_path=temp_db)

        # Create collection and add variables
        pm.create_collection("test_collection")
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", "value2")
        pm.add_to_collection("test_collection", ["var1", "var2"])

        # Verify associations exist
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM collection_vars WHERE collection_name = ?", ("test_collection",))
            count = cursor.fetchone()[0]
            assert count == 2

        # Delete collection
        pm.delete_collection("test_collection")

        # Verify associations are gone
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM collection_vars WHERE collection_name = ?", ("test_collection",))
            count = cursor.fetchone()[0]
            assert count == 0

    def test_delete_collection_does_not_delete_variables(self, temp_db):
        """Test that delete_collection does NOT delete the variables themselves."""
        pm = PersistenceManager(db_path=temp_db)

        # Create collection and add variables
        pm.create_collection("test_collection")
        pm.save_variable("var1", "value1")
        pm.save_variable("var2", "value2")
        pm.add_to_collection("test_collection", ["var1", "var2"])

        # Delete collection
        pm.delete_collection("test_collection")

        # Variables should still exist
        assert pm.load_variable("var1") == "value1"
        assert pm.load_variable("var2") == "value2"

        # Variables should still appear in list_variables
        vars_list = pm.list_variables()
        names = {v["name"] for v in vars_list}
        assert names == {"var1", "var2"}

    def test_delete_nonexistent_collection(self, temp_db):
        """Test that deleting a collection that doesn't exist returns True."""
        pm = PersistenceManager(db_path=temp_db)

        # SQLite DELETE succeeds even if no rows match
        result = pm.delete_collection("nonexistent")
        assert result is True

    def test_delete_empty_collection(self, temp_db):
        """Test deleting a collection that has no variables."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("empty_collection")

        result = pm.delete_collection("empty_collection")
        assert result is True

        collections = pm.list_collections()
        assert len(collections) == 0

    def test_delete_collection_does_not_affect_other_collections(self, temp_db):
        """Test that deleting one collection doesn't affect others."""
        pm = PersistenceManager(db_path=temp_db)

        # Create multiple collections
        pm.create_collection("collection_a", description="First")
        pm.create_collection("collection_b", description="Second")
        pm.create_collection("collection_c", description="Third")

        # Add variables to each
        pm.save_variable("var_a", "value_a")
        pm.save_variable("var_b", "value_b")
        pm.save_variable("var_c", "value_c")
        pm.add_to_collection("collection_a", ["var_a"])
        pm.add_to_collection("collection_b", ["var_b"])
        pm.add_to_collection("collection_c", ["var_c"])

        # Delete collection_b
        pm.delete_collection("collection_b")

        # Other collections should still exist
        collections = pm.list_collections()
        names = {c["name"] for c in collections}
        assert names == {"collection_a", "collection_c"}

        # Their variables should still be in the collections
        assert pm.get_collection_vars("collection_a") == ["var_a"]
        assert pm.get_collection_vars("collection_c") == ["var_c"]

    def test_delete_collection_variable_can_be_added_to_new_collection(self, temp_db):
        """Test that after deleting a collection, its variables can be added to a new collection."""
        pm = PersistenceManager(db_path=temp_db)

        # Create collection and add variable
        pm.create_collection("old_collection")
        pm.save_variable("var1", "value1")
        pm.add_to_collection("old_collection", ["var1"])

        # Delete collection
        pm.delete_collection("old_collection")

        # Create new collection and add the same variable
        pm.create_collection("new_collection")
        result = pm.add_to_collection("new_collection", ["var1"])
        assert result == 1

        # Variable should be in new collection
        vars = pm.get_collection_vars("new_collection")
        assert vars == ["var1"]

    def test_delete_collection_with_shared_variable(self, temp_db):
        """Test deleting a collection when a variable belongs to multiple collections."""
        pm = PersistenceManager(db_path=temp_db)

        # Create two collections
        pm.create_collection("collection_a")
        pm.create_collection("collection_b")

        # Add same variable to both
        pm.save_variable("shared_var", "shared value")
        pm.add_to_collection("collection_a", ["shared_var"])
        pm.add_to_collection("collection_b", ["shared_var"])

        # Delete collection_a
        pm.delete_collection("collection_a")

        # Variable should still be in collection_b
        assert pm.get_collection_vars("collection_b") == ["shared_var"]

        # Variable itself should still exist
        assert pm.load_variable("shared_var") == "shared value"

    def test_delete_collection_with_many_variables(self, temp_db):
        """Test deleting a collection with many variables."""
        pm = PersistenceManager(db_path=temp_db)

        pm.create_collection("large_collection")

        # Create and add 50 variables
        var_names = [f"var_{i:03d}" for i in range(50)]
        for name in var_names:
            pm.save_variable(name, f"value for {name}")
        pm.add_to_collection("large_collection", var_names)

        # Verify all variables are in collection
        assert len(pm.get_collection_vars("large_collection")) == 50

        # Delete collection
        result = pm.delete_collection("large_collection")
        assert result is True

        # Collection should be gone
        assert pm.list_collections() == []

        # All variables should still exist
        for name in var_names:
            assert pm.load_variable(name) is not None
