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


class TestMcpToolsList:
    """Tests for MCP tools/list method via POST /mcp endpoint."""

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
        """MCP tools/list should return 200 OK."""
        response = self.make_mcp_request(client, "tools/list")
        assert response.status_code == 200

    def test_returns_json(self, client):
        """MCP tools/list should return JSON content."""
        response = self.make_mcp_request(client, "tools/list")
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """MCP tools/list should return jsonrpc 2.0."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """MCP tools/list should return the same request id."""
        response = self.make_mcp_request(client, "tools/list", request_id=99)
        data = response.json()
        assert data["id"] == 99

    def test_returns_result_dict(self, client):
        """MCP tools/list should return a result dict."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict)

    def test_result_has_tools_key(self, client):
        """MCP tools/list result should have 'tools' key."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        assert "tools" in data["result"]

    def test_tools_is_list(self, client):
        """MCP tools/list should return tools as a list."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        assert isinstance(data["result"]["tools"], list)

    def test_tools_not_empty(self, client):
        """MCP tools/list should return non-empty tools list."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        assert len(data["result"]["tools"]) > 0

    def test_tools_count(self, client):
        """MCP tools/list should return expected number of tools."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]
        # Based on get_tools_list() in http_server.py, there are 19 tools
        assert len(tools) == 19

    def test_all_expected_tools_present(self, client):
        """MCP tools/list should return all expected tool names."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]
        tool_names = [t["name"] for t in tools]

        expected_tools = [
            "rlm_execute",
            "rlm_load_data",
            "rlm_load_file",
            "rlm_list_vars",
            "rlm_var_info",
            "rlm_clear",
            "rlm_memory",
            "rlm_load_s3",
            "rlm_list_buckets",
            "rlm_list_s3",
            "rlm_upload_url",
            "rlm_process_pdf",
            "rlm_search_index",
            "rlm_persistence_stats",
            "rlm_collection_create",
            "rlm_collection_add",
            "rlm_collection_list",
            "rlm_collection_info",
            "rlm_search_collection",
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_names, f"Expected tool '{tool_name}' not found"

    def test_each_tool_has_name(self, client):
        """Each tool should have a 'name' field."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        for tool in tools:
            assert "name" in tool
            assert isinstance(tool["name"], str)
            assert len(tool["name"]) > 0

    def test_each_tool_has_description(self, client):
        """Each tool should have a 'description' field."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        for tool in tools:
            assert "description" in tool
            assert isinstance(tool["description"], str)
            assert len(tool["description"]) > 0

    def test_each_tool_has_input_schema(self, client):
        """Each tool should have an 'inputSchema' field."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        for tool in tools:
            assert "inputSchema" in tool
            assert isinstance(tool["inputSchema"], dict)

    def test_input_schema_has_type_object(self, client):
        """Each tool's inputSchema should have type='object'."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        for tool in tools:
            schema = tool["inputSchema"]
            assert schema.get("type") == "object", f"Tool {tool['name']} inputSchema type is not 'object'"

    def test_input_schema_has_properties(self, client):
        """Each tool's inputSchema should have 'properties' field."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        for tool in tools:
            schema = tool["inputSchema"]
            assert "properties" in schema, f"Tool {tool['name']} inputSchema missing 'properties'"
            assert isinstance(schema["properties"], dict)

    def test_rlm_execute_has_code_property(self, client):
        """rlm_execute tool should have 'code' in inputSchema properties."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        execute_tool = next((t for t in tools if t["name"] == "rlm_execute"), None)
        assert execute_tool is not None
        assert "code" in execute_tool["inputSchema"]["properties"]

    def test_rlm_execute_code_is_required(self, client):
        """rlm_execute tool should require 'code' parameter."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        execute_tool = next((t for t in tools if t["name"] == "rlm_execute"), None)
        assert execute_tool is not None
        assert "required" in execute_tool["inputSchema"]
        assert "code" in execute_tool["inputSchema"]["required"]

    def test_rlm_load_data_has_required_properties(self, client):
        """rlm_load_data tool should have 'name' and 'data' as required."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        load_data_tool = next((t for t in tools if t["name"] == "rlm_load_data"), None)
        assert load_data_tool is not None
        assert "name" in load_data_tool["inputSchema"]["properties"]
        assert "data" in load_data_tool["inputSchema"]["properties"]
        assert "name" in load_data_tool["inputSchema"]["required"]
        assert "data" in load_data_tool["inputSchema"]["required"]

    def test_rlm_load_s3_has_required_properties(self, client):
        """rlm_load_s3 tool should have 'key' and 'name' as required."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        load_s3_tool = next((t for t in tools if t["name"] == "rlm_load_s3"), None)
        assert load_s3_tool is not None
        assert "key" in load_s3_tool["inputSchema"]["properties"]
        assert "name" in load_s3_tool["inputSchema"]["properties"]
        assert "key" in load_s3_tool["inputSchema"]["required"]
        assert "name" in load_s3_tool["inputSchema"]["required"]

    def test_rlm_search_index_has_required_properties(self, client):
        """rlm_search_index tool should have 'var_name' and 'terms' as required."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        search_tool = next((t for t in tools if t["name"] == "rlm_search_index"), None)
        assert search_tool is not None
        assert "var_name" in search_tool["inputSchema"]["properties"]
        assert "terms" in search_tool["inputSchema"]["properties"]
        assert "var_name" in search_tool["inputSchema"]["required"]
        assert "terms" in search_tool["inputSchema"]["required"]

    def test_no_error_in_response(self, client):
        """MCP tools/list should not return error."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        assert data.get("error") is None

    def test_with_string_id(self, client):
        """MCP tools/list should work with string id."""
        payload = {
            "jsonrpc": "2.0",
            "id": "tools-list-request",
            "method": "tools/list"
        }
        response = client.post("/mcp", json=payload)
        data = response.json()
        assert data["id"] == "tools-list-request"
        assert "result" in data
        assert "tools" in data["result"]

    def test_multiple_requests(self, client):
        """Multiple MCP tools/list requests should return same tools."""
        responses = []
        for i in range(3):
            response = self.make_mcp_request(client, "tools/list", request_id=i)
            data = response.json()
            responses.append(data)

        # All responses should have same tools
        first_tools = set(t["name"] for t in responses[0]["result"]["tools"])
        for resp in responses[1:]:
            tools = set(t["name"] for t in resp["result"]["tools"])
            assert tools == first_tools

    def test_tools_order_is_consistent(self, client):
        """MCP tools/list should return tools in consistent order."""
        response1 = self.make_mcp_request(client, "tools/list", request_id=1)
        response2 = self.make_mcp_request(client, "tools/list", request_id=2)

        tools1 = [t["name"] for t in response1.json()["result"]["tools"]]
        tools2 = [t["name"] for t in response2.json()["result"]["tools"]]

        assert tools1 == tools2

    def test_tools_with_optional_params(self, client):
        """Tools with optional params should not have them in 'required'."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        # rlm_clear has optional 'name' and 'all' params
        clear_tool = next((t for t in tools if t["name"] == "rlm_clear"), None)
        assert clear_tool is not None
        # required should be empty or not include 'name' and 'all'
        required = clear_tool["inputSchema"].get("required", [])
        assert "name" not in required or "all" not in required

    def test_tools_without_params_have_empty_properties(self, client):
        """Tools without params should have empty properties dict."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        # rlm_memory has no required params
        memory_tool = next((t for t in tools if t["name"] == "rlm_memory"), None)
        assert memory_tool is not None
        # Properties can be empty dict
        props = memory_tool["inputSchema"]["properties"]
        assert isinstance(props, dict)

    def test_tool_names_follow_naming_convention(self, client):
        """All tool names should start with 'rlm_'."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        for tool in tools:
            assert tool["name"].startswith("rlm_"), f"Tool {tool['name']} doesn't follow 'rlm_' naming convention"

    def test_response_is_dict(self, client):
        """MCP tools/list should return a dictionary response."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        assert isinstance(data, dict)
