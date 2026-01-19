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


class TestMcpInitialize:
    """Tests for MCP initialize method via POST /mcp endpoint."""

    def make_mcp_request(self, client, method: str, params: dict = None, request_id: int = 1):
        """Helper to make MCP JSON-RPC requests."""
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        return client.post("/mcp", json=payload)

    def test_returns_200_status_code(self, client):
        """MCP initialize should return 200 OK."""
        response = self.make_mcp_request(client, "initialize")
        assert response.status_code == 200

    def test_returns_json(self, client):
        """MCP initialize should return JSON content."""
        response = self.make_mcp_request(client, "initialize")
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """MCP initialize should return jsonrpc 2.0."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """MCP initialize should return the same request id."""
        response = self.make_mcp_request(client, "initialize", request_id=42)
        data = response.json()
        assert data["id"] == 42

    def test_returns_result_dict(self, client):
        """MCP initialize should return a result dict."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict)

    def test_returns_protocol_version(self, client):
        """MCP initialize should return protocolVersion."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        assert "protocolVersion" in data["result"]
        assert data["result"]["protocolVersion"] == "2024-11-05"

    def test_returns_capabilities(self, client):
        """MCP initialize should return capabilities dict."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        assert "capabilities" in data["result"]
        assert isinstance(data["result"]["capabilities"], dict)

    def test_capabilities_has_tools(self, client):
        """MCP capabilities should include tools."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        caps = data["result"]["capabilities"]
        assert "tools" in caps
        assert isinstance(caps["tools"], dict)

    def test_tools_list_changed_is_false(self, client):
        """MCP tools capability should have listChanged=False."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        tools_cap = data["result"]["capabilities"]["tools"]
        assert "listChanged" in tools_cap
        assert tools_cap["listChanged"] is False

    def test_returns_server_info(self, client):
        """MCP initialize should return serverInfo."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        assert "serverInfo" in data["result"]
        assert isinstance(data["result"]["serverInfo"], dict)

    def test_server_info_has_name(self, client):
        """MCP serverInfo should include name."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        server_info = data["result"]["serverInfo"]
        assert "name" in server_info
        assert server_info["name"] == "rlm-mcp-server"

    def test_server_info_has_version(self, client):
        """MCP serverInfo should include version."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        server_info = data["result"]["serverInfo"]
        assert "version" in server_info
        assert server_info["version"] == "0.1.0"

    def test_no_error_in_response(self, client):
        """MCP initialize should not return error."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        # error key should either be absent or None
        assert data.get("error") is None

    def test_with_string_id(self, client):
        """MCP initialize should work with string id."""
        payload = {
            "jsonrpc": "2.0",
            "id": "request-123",
            "method": "initialize"
        }
        response = client.post("/mcp", json=payload)
        data = response.json()
        assert data["id"] == "request-123"
        assert "result" in data

    def test_with_null_id(self, client):
        """MCP initialize should work with null id (id is excluded from response)."""
        payload = {
            "jsonrpc": "2.0",
            "id": None,
            "method": "initialize"
        }
        response = client.post("/mcp", json=payload)
        data = response.json()
        # When id is None, it's excluded from response due to exclude_none=True
        assert "id" not in data or data.get("id") is None
        assert "result" in data

    def test_with_params(self, client):
        """MCP initialize should work with params (ignored but valid)."""
        response = self.make_mcp_request(
            client, "initialize",
            params={"clientInfo": {"name": "test-client", "version": "1.0.0"}}
        )
        data = response.json()
        assert "result" in data
        assert data["result"]["protocolVersion"] == "2024-11-05"

    def test_multiple_requests(self, client):
        """Multiple MCP initialize requests should all succeed."""
        for i in range(3):
            response = self.make_mcp_request(client, "initialize", request_id=i)
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == i
            assert "result" in data

    def test_result_has_all_required_fields(self, client):
        """MCP initialize result should have all required fields."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        result = data["result"]
        required_fields = {"protocolVersion", "capabilities", "serverInfo"}
        assert required_fields.issubset(result.keys())

    def test_returns_dict_type(self, client):
        """MCP initialize should return a dictionary response."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        assert isinstance(data, dict)
