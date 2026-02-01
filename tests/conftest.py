"""
Pytest fixtures for RLM MCP Server tests.

IMPORTANT: The RLM_PERSIST_DIR environment variable must be set at module load
time, BEFORE any rlm_mcp modules are imported. This is done in pytest_configure().
"""

import os
import tempfile
import shutil
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Global setup: RLM_PERSIST_DIR MUST be set BEFORE any module imports
# ============================================================================

# Global variable to track the temp directory
_TEST_PERSIST_DIR = None


def pytest_configure(config):
    """
    Called early in pytest startup, before any test modules are imported.

    This is the right place to set environment variables that affect module load.
    """
    global _TEST_PERSIST_DIR
    _TEST_PERSIST_DIR = tempfile.mkdtemp(prefix="rlm_test_persist_")
    os.environ["RLM_PERSIST_DIR"] = _TEST_PERSIST_DIR


def pytest_unconfigure(config):
    """
    Called when pytest is about to exit.

    Clean up the temp directory.
    """
    global _TEST_PERSIST_DIR
    if _TEST_PERSIST_DIR and os.path.exists(_TEST_PERSIST_DIR):
        shutil.rmtree(_TEST_PERSIST_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_persistence_singleton():
    """
    Reset the persistence singleton before each test.

    This ensures each test starts with a fresh PersistenceManager instance
    that uses the test persist directory.
    """
    # Reset the singleton so it re-initializes with the test directory
    import rlm_mcp.persistence as persistence_module
    persistence_module._persistence = None

    yield

    # Reset again after the test
    persistence_module._persistence = None


@pytest.fixture
def temp_db():
    """
    Creates a temporary SQLite database file for testing.

    Yields the path to the temporary database, then cleans up after the test.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_text():
    """
    Generates a large text (~200k characters) for testing indexation.

    Contains common terms used in semantic indexing: emotions, relationships,
    work, physical symptoms, etc.
    """
    # Base text with varied terms for realistic indexing tests
    base = "medo ansiedade trabalho família saúde dor cabeça estômago "
    # Repeat to get ~200k characters (base is ~60 chars, 25000 * 8 words)
    return base * 25000  # ~200k characters


# ============================================================================
# MinIO/S3 Mock Fixtures
# ============================================================================


class MockMinioObject:
    """Mock for MinIO object metadata returned by list_objects."""

    def __init__(self, object_name: str, size: int, last_modified: datetime = None):
        self.object_name = object_name
        self.size = size
        self.last_modified = last_modified or datetime.now(timezone.utc)


class MockMinioStat:
    """Mock for MinIO stat_object response."""

    def __init__(
        self,
        size: int,
        content_type: str = "application/octet-stream",
        last_modified: datetime = None,
        etag: str = "abc123",
    ):
        self.size = size
        self.content_type = content_type
        self.last_modified = last_modified or datetime.now(timezone.utc)
        self.etag = etag


class MockMinioBucket:
    """Mock for MinIO bucket returned by list_buckets."""

    def __init__(self, name: str):
        self.name = name


class MockMinioResponse:
    """Mock for MinIO get_object response."""

    def __init__(self, data: bytes):
        self._data = data
        self._stream = BytesIO(data)

    def read(self) -> bytes:
        return self._data

    def close(self):
        pass

    def release_conn(self):
        pass


class MockMinioClient:
    """
    Mock MinIO client for testing S3Client without real MinIO connection.

    Usage:
        mock_client = MockMinioClient()
        mock_client.add_bucket("test-bucket")
        mock_client.add_object("test-bucket", "file.txt", b"content")
    """

    def __init__(self):
        self.buckets: dict[str, dict[str, bytes]] = {}
        self.object_metadata: dict[str, dict[str, MockMinioStat]] = {}

    def add_bucket(self, name: str):
        """Add a bucket to the mock."""
        if name not in self.buckets:
            self.buckets[name] = {}
            self.object_metadata[name] = {}

    def add_object(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ):
        """Add an object to a bucket."""
        if bucket not in self.buckets:
            self.add_bucket(bucket)
        self.buckets[bucket][key] = data
        self.object_metadata[bucket][key] = MockMinioStat(
            size=len(data),
            content_type=content_type,
            last_modified=datetime.now(timezone.utc),
            etag=f"etag-{key}",
        )

    def list_buckets(self) -> list[MockMinioBucket]:
        """List all buckets."""
        return [MockMinioBucket(name) for name in self.buckets.keys()]

    def list_objects(
        self, bucket: str, prefix: str = "", recursive: bool = True
    ) -> list[MockMinioObject]:
        """List objects in a bucket."""
        if bucket not in self.buckets:
            raise Exception(f"Bucket '{bucket}' not found")

        objects = []
        for key, data in self.buckets[bucket].items():
            if key.startswith(prefix):
                stat = self.object_metadata[bucket].get(key)
                objects.append(
                    MockMinioObject(
                        object_name=key,
                        size=len(data),
                        last_modified=stat.last_modified if stat else None,
                    )
                )
        return objects

    def get_object(self, bucket: str, key: str) -> MockMinioResponse:
        """Get an object from a bucket."""
        if bucket not in self.buckets:
            raise Exception(f"Bucket '{bucket}' not found")
        if key not in self.buckets[bucket]:
            raise Exception(f"Object '{key}' not found in bucket '{bucket}'")
        return MockMinioResponse(self.buckets[bucket][key])

    def stat_object(self, bucket: str, key: str) -> MockMinioStat:
        """Get object metadata."""
        if bucket not in self.buckets:
            raise Exception(f"Bucket '{bucket}' not found")
        if key not in self.object_metadata[bucket]:
            raise Exception(f"Object '{key}' not found in bucket '{bucket}'")
        return self.object_metadata[bucket][key]

    def put_object(
        self,
        bucket: str,
        key: str,
        data: BytesIO,
        length: int,
        content_type: str = "application/octet-stream",
    ) -> MagicMock:
        """Put an object in a bucket."""
        if bucket not in self.buckets:
            self.add_bucket(bucket)
        content = data.read()
        self.buckets[bucket][key] = content
        self.object_metadata[bucket][key] = MockMinioStat(
            size=len(content),
            content_type=content_type,
            last_modified=datetime.now(timezone.utc),
            etag=f"etag-{key}",
        )
        result = MagicMock()
        result.etag = f"etag-{key}"
        return result

    def presigned_put_object(self, bucket: str, key: str, expires) -> str:
        """Generate presigned URL for upload."""
        return f"https://mock-minio/presigned-put/{bucket}/{key}"

    def presigned_get_object(self, bucket: str, key: str, expires) -> str:
        """Generate presigned URL for download."""
        return f"https://mock-minio/presigned-get/{bucket}/{key}"


@pytest.fixture
def mock_minio_client():
    """
    Provides a MockMinioClient instance for testing.

    The mock is pre-configured with no buckets. Use add_bucket() and add_object()
    to set up test data.
    """
    return MockMinioClient()


@pytest.fixture
def mock_minio_client_with_data():
    """
    Provides a MockMinioClient with sample test data.

    Pre-configured with:
    - Bucket 'test-bucket' with files: test.txt, data/file.json, images/photo.png
    - Bucket 'empty-bucket' with no files
    """
    client = MockMinioClient()

    # Add buckets
    client.add_bucket("test-bucket")
    client.add_bucket("empty-bucket")

    # Add objects to test-bucket
    client.add_object(
        "test-bucket", "test.txt", b"Hello, World!", content_type="text/plain"
    )
    client.add_object(
        "test-bucket",
        "data/file.json",
        b'{"key": "value", "number": 42}',
        content_type="application/json",
    )
    client.add_object(
        "test-bucket",
        "images/photo.png",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,  # Fake PNG header + padding
        content_type="image/png",
    )

    return client


@pytest.fixture
def s3_client_with_mock(mock_minio_client_with_data):
    """
    Provides an S3Client instance with a mocked MinIO client.

    The mock is injected directly, bypassing environment variable checks.
    Use this fixture for testing S3Client methods without real MinIO connection.
    """
    from rlm_mcp.s3_client import S3Client

    # Create S3Client with mocked environment
    with patch.dict(
        os.environ,
        {
            "MINIO_ENDPOINT": "mock-minio:9000",
            "MINIO_ACCESS_KEY": "mock-access-key",
            "MINIO_SECRET_KEY": "mock-secret-key",
            "MINIO_SECURE": "false",
        },
    ):
        client = S3Client()
        # Inject the mock client directly
        client._client = mock_minio_client_with_data
        yield client


@pytest.fixture
def s3_client_unconfigured():
    """
    Provides an S3Client instance without MinIO credentials configured.

    Useful for testing is_configured() returns False.
    """
    from rlm_mcp.s3_client import S3Client

    # Clear any environment variables
    with patch.dict(
        os.environ,
        {
            "MINIO_ENDPOINT": "",
            "MINIO_ACCESS_KEY": "",
            "MINIO_SECRET_KEY": "",
        },
        clear=True,
    ):
        # Need to reimport to get fresh instance
        client = S3Client()
        yield client
