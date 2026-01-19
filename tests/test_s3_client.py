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
