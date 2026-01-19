"""
Tests to validate conftest.py fixtures work correctly.
"""

import os


def test_temp_db_fixture_creates_file(temp_db):
    """Verify temp_db fixture creates a valid file path."""
    assert temp_db.endswith(".db")
    assert os.path.exists(temp_db)


def test_sample_text_fixture_is_large_enough(sample_text):
    """Verify sample_text fixture generates ~200k+ characters."""
    assert len(sample_text) >= 200_000
    assert "medo" in sample_text
    assert "trabalho" in sample_text
    assert "ansiedade" in sample_text


# ============================================================================
# MinIO Mock Fixtures Tests
# ============================================================================


class TestMockMinioClient:
    """Tests for mock_minio_client fixture."""

    def test_mock_minio_client_is_empty_by_default(self, mock_minio_client):
        """Verify mock_minio_client starts with no buckets."""
        assert len(mock_minio_client.buckets) == 0
        assert mock_minio_client.list_buckets() == []

    def test_mock_minio_client_can_add_bucket(self, mock_minio_client):
        """Verify can add buckets to mock."""
        mock_minio_client.add_bucket("my-bucket")
        assert "my-bucket" in mock_minio_client.buckets
        buckets = mock_minio_client.list_buckets()
        assert len(buckets) == 1
        assert buckets[0].name == "my-bucket"

    def test_mock_minio_client_can_add_object(self, mock_minio_client):
        """Verify can add objects to mock."""
        mock_minio_client.add_object("my-bucket", "file.txt", b"Hello")
        # Bucket should be auto-created
        assert "my-bucket" in mock_minio_client.buckets
        assert "file.txt" in mock_minio_client.buckets["my-bucket"]
        assert mock_minio_client.buckets["my-bucket"]["file.txt"] == b"Hello"

    def test_mock_minio_client_get_object(self, mock_minio_client):
        """Verify can get objects from mock."""
        mock_minio_client.add_object("my-bucket", "file.txt", b"Hello, World!")
        response = mock_minio_client.get_object("my-bucket", "file.txt")
        assert response.read() == b"Hello, World!"

    def test_mock_minio_client_stat_object(self, mock_minio_client):
        """Verify can stat objects in mock."""
        mock_minio_client.add_object(
            "my-bucket", "file.txt", b"Hello", content_type="text/plain"
        )
        stat = mock_minio_client.stat_object("my-bucket", "file.txt")
        assert stat.size == 5
        assert stat.content_type == "text/plain"
        assert stat.etag == "etag-file.txt"

    def test_mock_minio_client_list_objects(self, mock_minio_client):
        """Verify can list objects in mock."""
        mock_minio_client.add_object("my-bucket", "file1.txt", b"A")
        mock_minio_client.add_object("my-bucket", "file2.txt", b"BB")
        objects = mock_minio_client.list_objects("my-bucket")
        assert len(objects) == 2
        names = [obj.object_name for obj in objects]
        assert "file1.txt" in names
        assert "file2.txt" in names


class TestMockMinioClientWithData:
    """Tests for mock_minio_client_with_data fixture."""

    def test_has_test_bucket(self, mock_minio_client_with_data):
        """Verify test-bucket exists."""
        assert "test-bucket" in mock_minio_client_with_data.buckets

    def test_has_empty_bucket(self, mock_minio_client_with_data):
        """Verify empty-bucket exists."""
        assert "empty-bucket" in mock_minio_client_with_data.buckets

    def test_test_bucket_has_test_txt(self, mock_minio_client_with_data):
        """Verify test.txt exists in test-bucket."""
        response = mock_minio_client_with_data.get_object("test-bucket", "test.txt")
        assert response.read() == b"Hello, World!"

    def test_test_bucket_has_json_file(self, mock_minio_client_with_data):
        """Verify data/file.json exists in test-bucket."""
        response = mock_minio_client_with_data.get_object(
            "test-bucket", "data/file.json"
        )
        assert b'"key": "value"' in response.read()

    def test_empty_bucket_has_no_objects(self, mock_minio_client_with_data):
        """Verify empty-bucket has no objects."""
        objects = mock_minio_client_with_data.list_objects("empty-bucket")
        assert len(objects) == 0


class TestS3ClientWithMock:
    """Tests for s3_client_with_mock fixture."""

    def test_s3_client_is_configured(self, s3_client_with_mock):
        """Verify S3Client reports as configured."""
        assert s3_client_with_mock.is_configured() is True

    def test_s3_client_can_list_buckets(self, s3_client_with_mock):
        """Verify S3Client can list buckets via mock."""
        buckets = s3_client_with_mock.list_buckets()
        assert "test-bucket" in buckets
        assert "empty-bucket" in buckets

    def test_s3_client_can_get_object(self, s3_client_with_mock):
        """Verify S3Client can get objects via mock."""
        data = s3_client_with_mock.get_object("test-bucket", "test.txt")
        assert data == b"Hello, World!"

    def test_s3_client_can_get_object_text(self, s3_client_with_mock):
        """Verify S3Client can get objects as text via mock."""
        text = s3_client_with_mock.get_object_text("test-bucket", "test.txt")
        assert text == "Hello, World!"


class TestS3ClientUnconfigured:
    """Tests for s3_client_unconfigured fixture."""

    def test_s3_client_is_not_configured(self, s3_client_unconfigured):
        """Verify S3Client reports as not configured."""
        assert s3_client_unconfigured.is_configured() is False
