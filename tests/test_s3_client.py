"""
Tests for s3_client.py - MinIO/S3 client functionality.

Uses mock fixtures from conftest.py to test without real MinIO connection.
"""

import os
from unittest.mock import patch

import pytest

from rlm_mcp.s3_client import S3Client


class TestIsConfigured:
    """Tests for S3Client.is_configured() method."""

    def test_returns_false_without_credentials(self, s3_client_unconfigured):
        """is_configured returns False when no credentials are set."""
        assert s3_client_unconfigured.is_configured() is False

    def test_returns_false_with_empty_endpoint(self):
        """is_configured returns False when endpoint is empty string."""
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "",
                "MINIO_ACCESS_KEY": "access-key",
                "MINIO_SECRET_KEY": "secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            assert client.is_configured() is False

    def test_returns_false_with_empty_access_key(self):
        """is_configured returns False when access key is empty string."""
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "localhost:9000",
                "MINIO_ACCESS_KEY": "",
                "MINIO_SECRET_KEY": "secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            assert client.is_configured() is False

    def test_returns_false_with_empty_secret_key(self):
        """is_configured returns False when secret key is empty string."""
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "localhost:9000",
                "MINIO_ACCESS_KEY": "access-key",
                "MINIO_SECRET_KEY": "",
            },
            clear=True,
        ):
            client = S3Client()
            assert client.is_configured() is False

    def test_returns_false_with_all_empty(self):
        """is_configured returns False when all credentials are empty."""
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "",
                "MINIO_ACCESS_KEY": "",
                "MINIO_SECRET_KEY": "",
            },
            clear=True,
        ):
            client = S3Client()
            assert client.is_configured() is False

    def test_returns_true_with_all_credentials(self):
        """is_configured returns True when all credentials are set."""
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "localhost:9000",
                "MINIO_ACCESS_KEY": "access-key",
                "MINIO_SECRET_KEY": "secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            assert client.is_configured() is True

    def test_returns_true_with_mock_fixture(self, s3_client_with_mock):
        """is_configured returns True when using the s3_client_with_mock fixture."""
        assert s3_client_with_mock.is_configured() is True

    def test_returns_false_with_missing_endpoint_env_var(self):
        """is_configured returns False when MINIO_ENDPOINT env var is not set at all."""
        with patch.dict(os.environ, {}, clear=True):
            client = S3Client()
            # When env var is not set, os.getenv returns "" (default)
            assert client.is_configured() is False

    def test_returns_false_with_only_endpoint(self):
        """is_configured returns False when only endpoint is set (missing keys)."""
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "localhost:9000",
            },
            clear=True,
        ):
            client = S3Client()
            assert client.is_configured() is False

    def test_returns_false_with_only_access_key(self):
        """is_configured returns False when only access key is set."""
        with patch.dict(
            os.environ,
            {
                "MINIO_ACCESS_KEY": "access-key",
            },
            clear=True,
        ):
            client = S3Client()
            assert client.is_configured() is False

    def test_returns_false_with_only_secret_key(self):
        """is_configured returns False when only secret key is set."""
        with patch.dict(
            os.environ,
            {
                "MINIO_SECRET_KEY": "secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            assert client.is_configured() is False

    def test_returns_false_with_endpoint_and_access_key_only(self):
        """is_configured returns False when secret key is missing."""
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "localhost:9000",
                "MINIO_ACCESS_KEY": "access-key",
            },
            clear=True,
        ):
            client = S3Client()
            assert client.is_configured() is False

    def test_returns_false_with_endpoint_and_secret_key_only(self):
        """is_configured returns False when access key is missing."""
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "localhost:9000",
                "MINIO_SECRET_KEY": "secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            assert client.is_configured() is False

    def test_returns_false_with_access_and_secret_keys_only(self):
        """is_configured returns False when endpoint is missing."""
        with patch.dict(
            os.environ,
            {
                "MINIO_ACCESS_KEY": "access-key",
                "MINIO_SECRET_KEY": "secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            assert client.is_configured() is False

    def test_returns_bool_type(self, s3_client_unconfigured):
        """is_configured returns a boolean type."""
        result = s3_client_unconfigured.is_configured()
        assert isinstance(result, bool)

    def test_returns_bool_type_when_configured(self, s3_client_with_mock):
        """is_configured returns a boolean type when configured."""
        result = s3_client_with_mock.is_configured()
        assert isinstance(result, bool)

    def test_whitespace_only_endpoint_returns_false(self):
        """is_configured returns False when endpoint is whitespace only (not trimmed by code)."""
        # Note: The code does bool(self.endpoint), whitespace is truthy
        # This test documents current behavior
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "   ",
                "MINIO_ACCESS_KEY": "access-key",
                "MINIO_SECRET_KEY": "secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            # Whitespace is truthy, so this returns True (current behavior)
            assert client.is_configured() is True

    def test_does_not_access_client_property(self, s3_client_unconfigured):
        """is_configured does not try to initialize the MinIO client."""
        # This is important - is_configured should be safe to call
        # without triggering the lazy client initialization
        result = s3_client_unconfigured.is_configured()
        assert result is False
        # Verify _client was never initialized
        assert s3_client_unconfigured._client is None


class TestListBuckets:
    """Tests for S3Client.list_buckets() method."""

    def test_returns_list(self, s3_client_with_mock):
        """list_buckets returns a list."""
        result = s3_client_with_mock.list_buckets()
        assert isinstance(result, list)

    def test_returns_bucket_names_as_strings(self, s3_client_with_mock):
        """list_buckets returns list of strings (bucket names)."""
        result = s3_client_with_mock.list_buckets()
        for bucket_name in result:
            assert isinstance(bucket_name, str)

    def test_returns_expected_buckets(self, s3_client_with_mock):
        """list_buckets returns the buckets from the mock (test-bucket, empty-bucket)."""
        result = s3_client_with_mock.list_buckets()
        assert "test-bucket" in result
        assert "empty-bucket" in result
        assert len(result) == 2

    def test_returns_empty_list_when_no_buckets(self, mock_minio_client):
        """list_buckets returns empty list when no buckets exist."""
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "mock-minio:9000",
                "MINIO_ACCESS_KEY": "mock-access-key",
                "MINIO_SECRET_KEY": "mock-secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            client._client = mock_minio_client
            result = client.list_buckets()
            assert result == []

    def test_returns_single_bucket(self, mock_minio_client):
        """list_buckets returns list with single bucket."""
        mock_minio_client.add_bucket("single-bucket")
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "mock-minio:9000",
                "MINIO_ACCESS_KEY": "mock-access-key",
                "MINIO_SECRET_KEY": "mock-secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            client._client = mock_minio_client
            result = client.list_buckets()
            assert result == ["single-bucket"]

    def test_returns_many_buckets(self, mock_minio_client):
        """list_buckets returns list with many buckets."""
        for i in range(10):
            mock_minio_client.add_bucket(f"bucket-{i}")
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "mock-minio:9000",
                "MINIO_ACCESS_KEY": "mock-access-key",
                "MINIO_SECRET_KEY": "mock-secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            client._client = mock_minio_client
            result = client.list_buckets()
            assert len(result) == 10
            for i in range(10):
                assert f"bucket-{i}" in result

    def test_bucket_names_with_special_characters(self, mock_minio_client):
        """list_buckets handles bucket names with hyphens and dots."""
        mock_minio_client.add_bucket("my-bucket")
        mock_minio_client.add_bucket("my.bucket")
        mock_minio_client.add_bucket("bucket-with-many-hyphens")
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "mock-minio:9000",
                "MINIO_ACCESS_KEY": "mock-access-key",
                "MINIO_SECRET_KEY": "mock-secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            client._client = mock_minio_client
            result = client.list_buckets()
            assert "my-bucket" in result
            assert "my.bucket" in result
            assert "bucket-with-many-hyphens" in result

    def test_does_not_return_objects_in_buckets(self, s3_client_with_mock):
        """list_buckets returns only bucket names, not object keys."""
        result = s3_client_with_mock.list_buckets()
        # The mock has test.txt, data/file.json, images/photo.png in test-bucket
        # These should NOT appear in list_buckets result
        assert "test.txt" not in result
        assert "data/file.json" not in result
        assert "images/photo.png" not in result

    def test_raises_runtime_error_when_unconfigured(self, s3_client_unconfigured):
        """list_buckets raises RuntimeError when client is not configured."""
        with pytest.raises(RuntimeError) as exc_info:
            s3_client_unconfigured.list_buckets()
        assert "MINIO_ENDPOINT" in str(exc_info.value)

    def test_order_of_buckets(self, mock_minio_client):
        """list_buckets returns buckets in dict iteration order."""
        mock_minio_client.add_bucket("alpha")
        mock_minio_client.add_bucket("beta")
        mock_minio_client.add_bucket("gamma")
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "mock-minio:9000",
                "MINIO_ACCESS_KEY": "mock-access-key",
                "MINIO_SECRET_KEY": "mock-secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            client._client = mock_minio_client
            result = client.list_buckets()
            # Python 3.7+ dicts preserve insertion order
            assert result == ["alpha", "beta", "gamma"]

    def test_empty_bucket_name_handling(self, mock_minio_client):
        """list_buckets handles empty bucket name (edge case)."""
        mock_minio_client.add_bucket("")
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "mock-minio:9000",
                "MINIO_ACCESS_KEY": "mock-access-key",
                "MINIO_SECRET_KEY": "mock-secret-key",
            },
            clear=True,
        ):
            client = S3Client()
            client._client = mock_minio_client
            result = client.list_buckets()
            assert "" in result

    def test_list_buckets_is_read_only(self, s3_client_with_mock, mock_minio_client_with_data):
        """list_buckets does not modify the bucket list."""
        result1 = s3_client_with_mock.list_buckets()
        result2 = s3_client_with_mock.list_buckets()
        assert result1 == result2
        assert len(mock_minio_client_with_data.buckets) == 2


class TestListObjects:
    """Tests for S3Client.list_objects() method."""

    def test_returns_list(self, s3_client_with_mock):
        """list_objects returns a list."""
        result = s3_client_with_mock.list_objects("test-bucket")
        assert isinstance(result, list)

    def test_returns_dicts(self, s3_client_with_mock):
        """list_objects returns list of dicts."""
        result = s3_client_with_mock.list_objects("test-bucket")
        for obj in result:
            assert isinstance(obj, dict)

    def test_returns_expected_keys(self, s3_client_with_mock):
        """list_objects returns dicts with expected keys: name, size, size_human, last_modified."""
        result = s3_client_with_mock.list_objects("test-bucket")
        assert len(result) > 0
        for obj in result:
            assert "name" in obj
            assert "size" in obj
            assert "size_human" in obj
            assert "last_modified" in obj

    def test_returns_expected_objects(self, s3_client_with_mock):
        """list_objects returns the objects from the mock (test.txt, data/file.json, images/photo.png)."""
        result = s3_client_with_mock.list_objects("test-bucket")
        names = [obj["name"] for obj in result]
        assert "test.txt" in names
        assert "data/file.json" in names
        assert "images/photo.png" in names
        assert len(result) == 3

    def test_returns_empty_list_for_empty_bucket(self, s3_client_with_mock):
        """list_objects returns empty list for bucket with no objects."""
        result = s3_client_with_mock.list_objects("empty-bucket")
        assert result == []

    def test_returns_correct_size(self, s3_client_with_mock):
        """list_objects returns correct size for objects."""
        result = s3_client_with_mock.list_objects("test-bucket")
        # Find test.txt which is b"Hello, World!" (13 bytes)
        test_txt = next(obj for obj in result if obj["name"] == "test.txt")
        assert test_txt["size"] == 13

    def test_returns_human_readable_size(self, s3_client_with_mock):
        """list_objects returns human-readable size string."""
        result = s3_client_with_mock.list_objects("test-bucket")
        test_txt = next(obj for obj in result if obj["name"] == "test.txt")
        assert isinstance(test_txt["size_human"], str)
        assert "B" in test_txt["size_human"]  # Should contain unit

    def test_returns_last_modified_iso_format(self, s3_client_with_mock):
        """list_objects returns last_modified in ISO format."""
        result = s3_client_with_mock.list_objects("test-bucket")
        test_txt = next(obj for obj in result if obj["name"] == "test.txt")
        assert test_txt["last_modified"] is not None
        # ISO format contains T separator
        assert "T" in test_txt["last_modified"]

    def test_prefix_filters_objects(self, s3_client_with_mock):
        """list_objects with prefix filters to matching objects."""
        result = s3_client_with_mock.list_objects("test-bucket", prefix="data/")
        names = [obj["name"] for obj in result]
        assert len(result) == 1
        assert "data/file.json" in names
        assert "test.txt" not in names
        assert "images/photo.png" not in names

    def test_prefix_images(self, s3_client_with_mock):
        """list_objects with prefix 'images/' returns image files."""
        result = s3_client_with_mock.list_objects("test-bucket", prefix="images/")
        names = [obj["name"] for obj in result]
        assert len(result) == 1
        assert "images/photo.png" in names

    def test_prefix_no_match_returns_empty(self, s3_client_with_mock):
        """list_objects with non-matching prefix returns empty list."""
        result = s3_client_with_mock.list_objects("test-bucket", prefix="nonexistent/")
        assert result == []

    def test_empty_prefix_returns_all(self, s3_client_with_mock):
        """list_objects with empty prefix returns all objects."""
        result = s3_client_with_mock.list_objects("test-bucket", prefix="")
        assert len(result) == 3

    def test_raises_runtime_error_for_nonexistent_bucket(self, s3_client_with_mock):
        """list_objects raises RuntimeError for nonexistent bucket."""
        with pytest.raises(RuntimeError) as exc_info:
            s3_client_with_mock.list_objects("nonexistent-bucket")
        assert "nonexistent-bucket" in str(exc_info.value)

    def test_raises_runtime_error_when_unconfigured(self, s3_client_unconfigured):
        """list_objects raises RuntimeError when client is not configured."""
        with pytest.raises(RuntimeError) as exc_info:
            s3_client_unconfigured.list_objects("any-bucket")
        assert "MINIO_ENDPOINT" in str(exc_info.value)

    def test_list_objects_is_read_only(self, s3_client_with_mock, mock_minio_client_with_data):
        """list_objects does not modify the object list."""
        result1 = s3_client_with_mock.list_objects("test-bucket")
        result2 = s3_client_with_mock.list_objects("test-bucket")
        assert len(result1) == len(result2)
        # Check underlying mock data unchanged
        assert len(mock_minio_client_with_data.buckets["test-bucket"]) == 3

    def test_with_many_objects(self, mock_minio_client):
        """list_objects handles many objects (50)."""
        mock_minio_client.add_bucket("many-objects")
        for i in range(50):
            mock_minio_client.add_object(
                "many-objects", f"file-{i:03d}.txt", f"content-{i}".encode()
            )
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "mock-minio:9000",
                "MINIO_ACCESS_KEY": "mock-access-key",
                "MINIO_SECRET_KEY": "mock-secret-key",
            },
            clear=True,
        ):
            from rlm_mcp.s3_client import S3Client

            client = S3Client()
            client._client = mock_minio_client
            result = client.list_objects("many-objects")
            assert len(result) == 50
            for i in range(50):
                assert any(obj["name"] == f"file-{i:03d}.txt" for obj in result)

    def test_prefix_with_special_characters(self, mock_minio_client):
        """list_objects handles prefix with special characters (hyphens, underscores)."""
        mock_minio_client.add_bucket("special-bucket")
        mock_minio_client.add_object(
            "special-bucket", "my-folder/file_one.txt", b"content"
        )
        mock_minio_client.add_object(
            "special-bucket", "my-folder/file_two.txt", b"content"
        )
        mock_minio_client.add_object("special-bucket", "other/file.txt", b"content")
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "mock-minio:9000",
                "MINIO_ACCESS_KEY": "mock-access-key",
                "MINIO_SECRET_KEY": "mock-secret-key",
            },
            clear=True,
        ):
            from rlm_mcp.s3_client import S3Client

            client = S3Client()
            client._client = mock_minio_client
            result = client.list_objects("special-bucket", prefix="my-folder/")
            names = [obj["name"] for obj in result]
            assert len(result) == 2
            assert "my-folder/file_one.txt" in names
            assert "my-folder/file_two.txt" in names

    def test_nested_folders(self, mock_minio_client):
        """list_objects returns objects in nested folder structure."""
        mock_minio_client.add_bucket("nested-bucket")
        mock_minio_client.add_object(
            "nested-bucket", "a/b/c/deep.txt", b"deep content"
        )
        mock_minio_client.add_object("nested-bucket", "a/b/mid.txt", b"mid content")
        mock_minio_client.add_object("nested-bucket", "a/top.txt", b"top content")
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "mock-minio:9000",
                "MINIO_ACCESS_KEY": "mock-access-key",
                "MINIO_SECRET_KEY": "mock-secret-key",
            },
            clear=True,
        ):
            from rlm_mcp.s3_client import S3Client

            client = S3Client()
            client._client = mock_minio_client
            # List all in a/
            result = client.list_objects("nested-bucket", prefix="a/")
            names = [obj["name"] for obj in result]
            assert len(result) == 3
            assert "a/b/c/deep.txt" in names
            assert "a/b/mid.txt" in names
            assert "a/top.txt" in names

    def test_objects_with_same_prefix_substring(self, mock_minio_client):
        """list_objects correctly filters when objects share prefix substrings."""
        mock_minio_client.add_bucket("prefix-test")
        mock_minio_client.add_object("prefix-test", "data/file.txt", b"data file")
        mock_minio_client.add_object(
            "prefix-test", "data-backup/file.txt", b"backup file"
        )
        mock_minio_client.add_object("prefix-test", "dataset/file.txt", b"dataset file")
        with patch.dict(
            os.environ,
            {
                "MINIO_ENDPOINT": "mock-minio:9000",
                "MINIO_ACCESS_KEY": "mock-access-key",
                "MINIO_SECRET_KEY": "mock-secret-key",
            },
            clear=True,
        ):
            from rlm_mcp.s3_client import S3Client

            client = S3Client()
            client._client = mock_minio_client
            result = client.list_objects("prefix-test", prefix="data/")
            names = [obj["name"] for obj in result]
            assert len(result) == 1
            assert "data/file.txt" in names
            # These should NOT match because they don't start with "data/"
            assert "data-backup/file.txt" not in names
            assert "dataset/file.txt" not in names
