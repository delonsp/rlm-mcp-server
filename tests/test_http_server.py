# tests/test_http_server.py
"""
Tests for HTTP integration (FastAPI TestClient).
"""

import pytest
from fastapi.testclient import TestClient

from rlm_mcp.http_server import app


@pytest.fixture
def client():
    """FastAPI TestClient for testing HTTP endpoints."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_returns_200_status_code(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_json(self, client):
        """Health endpoint should return JSON content."""
        response = client.get("/health")
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_status_healthy(self, client):
        """Health endpoint should return status='healthy'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_returns_timestamp(self, client):
        """Health endpoint should return a timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data
        assert data["timestamp"] is not None

    def test_timestamp_is_iso_format(self, client):
        """Health endpoint timestamp should be in ISO format."""
        from datetime import datetime
        response = client.get("/health")
        data = response.json()
        # Should not raise exception
        datetime.fromisoformat(data["timestamp"])

    def test_returns_memory_info(self, client):
        """Health endpoint should return memory info."""
        response = client.get("/health")
        data = response.json()
        assert "memory" in data
        memory = data["memory"]
        assert "total_bytes" in memory
        assert "total_human" in memory
        assert "variable_count" in memory
        assert "max_allowed_mb" in memory
        assert "usage_percent" in memory

    def test_returns_version(self, client):
        """Health endpoint should return version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_no_authentication_required(self, client):
        """Health endpoint should work without authentication."""
        # Even if RLM_API_KEY is set, health check should work
        response = client.get("/health")
        assert response.status_code == 200

    def test_memory_types_are_correct(self, client):
        """Memory values should have correct types."""
        response = client.get("/health")
        data = response.json()
        memory = data["memory"]
        assert isinstance(memory["total_bytes"], int)
        assert isinstance(memory["total_human"], str)
        assert isinstance(memory["variable_count"], int)
        assert isinstance(memory["max_allowed_mb"], int)
        assert isinstance(memory["usage_percent"], (int, float))

    def test_response_has_all_required_fields(self, client):
        """Health endpoint should return all required fields."""
        response = client.get("/health")
        data = response.json()
        required_fields = {"status", "timestamp", "memory", "version"}
        assert required_fields.issubset(data.keys())

    def test_multiple_requests_succeed(self, client):
        """Multiple health requests should all succeed."""
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200

    def test_response_is_dict(self, client):
        """Health endpoint should return a dictionary."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data, dict)
