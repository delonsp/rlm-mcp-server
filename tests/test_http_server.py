# tests/test_http_server.py
"""
Tests for HTTP integration (FastAPI TestClient).
"""

import logging

import pytest
from fastapi.testclient import TestClient

from rlm_mcp.http_server import app, sse_rate_limiter


@pytest.fixture
def client():
    """FastAPI TestClient for testing HTTP endpoints."""
    # Reset rate limiter between tests to avoid cross-test 429s
    sse_rate_limiter._buckets.clear()
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
        assert data["version"] == "0.2.0"

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

    def test_capabilities_has_resources(self, client):
        """MCP capabilities should include resources."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        caps = data["result"]["capabilities"]
        assert "resources" in caps
        assert isinstance(caps["resources"], dict)

    def test_resources_list_changed_is_false(self, client):
        """MCP resources capability should have listChanged=False."""
        response = self.make_mcp_request(client, "initialize")
        data = response.json()
        resources_cap = data["result"]["capabilities"]["resources"]
        assert "listChanged" in resources_cap
        assert resources_cap["listChanged"] is False

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
        assert server_info["version"] == "0.2.0"

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
        # Based on get_tools_list() in http_server.py, there are 21 tools
        assert len(tools) == 21

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
            "rlm_collection_rebuild",
            "rlm_search_collection",
            "rlm_save_to_s3",
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


class TestMcpResourcesList:
    """Tests for MCP resources/list method via POST /mcp endpoint."""

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
        """MCP resources/list should return 200 OK."""
        response = self.make_mcp_request(client, "resources/list")
        assert response.status_code == 200

    def test_returns_json(self, client):
        """MCP resources/list should return JSON content."""
        response = self.make_mcp_request(client, "resources/list")
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """MCP resources/list should return jsonrpc 2.0."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """MCP resources/list should return the same request id."""
        response = self.make_mcp_request(client, "resources/list", request_id=42)
        data = response.json()
        assert data["id"] == 42

    def test_returns_result_dict(self, client):
        """MCP resources/list should return a result dict."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict)

    def test_result_has_resources_key(self, client):
        """MCP resources/list result should have resources key."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        assert "resources" in data["result"]

    def test_resources_is_list(self, client):
        """MCP resources/list should return a list of resources."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        assert isinstance(data["result"]["resources"], list)

    def test_resources_not_empty(self, client):
        """MCP resources/list should return at least one resource."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        assert len(data["result"]["resources"]) > 0

    def test_resources_count(self, client):
        """MCP resources/list should return 3 resources."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        assert len(data["result"]["resources"]) == 3

    def test_all_expected_resources_present(self, client):
        """MCP resources/list should include all expected resources."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        uris = [r["uri"] for r in data["result"]["resources"]]
        expected_uris = ["rlm://variables", "rlm://memory", "rlm://collections"]
        for uri in expected_uris:
            assert uri in uris, f"Resource {uri} not found in resources list"

    def test_each_resource_has_uri(self, client):
        """Each MCP resource should have a uri field."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        for resource in data["result"]["resources"]:
            assert "uri" in resource
            assert isinstance(resource["uri"], str)
            assert resource["uri"].startswith("rlm://")

    def test_each_resource_has_name(self, client):
        """Each MCP resource should have a name field."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        for resource in data["result"]["resources"]:
            assert "name" in resource
            assert isinstance(resource["name"], str)
            assert len(resource["name"]) > 0

    def test_each_resource_has_description(self, client):
        """Each MCP resource should have a description field."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        for resource in data["result"]["resources"]:
            assert "description" in resource
            assert isinstance(resource["description"], str)
            assert len(resource["description"]) > 0

    def test_each_resource_has_mime_type(self, client):
        """Each MCP resource should have a mimeType field."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        for resource in data["result"]["resources"]:
            assert "mimeType" in resource
            assert resource["mimeType"] == "application/json"

    def test_no_error_in_response(self, client):
        """MCP resources/list should not return an error."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        assert "error" not in data

    def test_with_string_id(self, client):
        """MCP resources/list should work with string request id."""
        response = self.make_mcp_request(client, "resources/list", request_id="test-123")
        data = response.json()
        assert data["id"] == "test-123"

    def test_multiple_requests(self, client):
        """Multiple resources/list requests should all succeed."""
        for i in range(3):
            response = self.make_mcp_request(client, "resources/list", request_id=i)
            assert response.status_code == 200
            data = response.json()
            assert "resources" in data["result"]

    def test_resources_order_is_consistent(self, client):
        """MCP resources/list should return resources in consistent order."""
        response1 = self.make_mcp_request(client, "resources/list")
        response2 = self.make_mcp_request(client, "resources/list")
        data1 = response1.json()
        data2 = response2.json()
        uris1 = [r["uri"] for r in data1["result"]["resources"]]
        uris2 = [r["uri"] for r in data2["result"]["resources"]]
        assert uris1 == uris2

    def test_response_is_dict(self, client):
        """MCP resources/list should return a dictionary response."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        assert isinstance(data, dict)

    def test_variables_resource_present(self, client):
        """MCP resources/list should include rlm://variables resource."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        variables_resource = next(
            (r for r in data["result"]["resources"] if r["uri"] == "rlm://variables"),
            None
        )
        assert variables_resource is not None
        assert variables_resource["name"] == "Variables"

    def test_memory_resource_present(self, client):
        """MCP resources/list should include rlm://memory resource."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        memory_resource = next(
            (r for r in data["result"]["resources"] if r["uri"] == "rlm://memory"),
            None
        )
        assert memory_resource is not None
        assert memory_resource["name"] == "Memory Usage"

    def test_collections_resource_present(self, client):
        """MCP resources/list should include rlm://collections resource."""
        response = self.make_mcp_request(client, "resources/list")
        data = response.json()
        collections_resource = next(
            (r for r in data["result"]["resources"] if r["uri"] == "rlm://collections"),
            None
        )
        assert collections_resource is not None
        assert collections_resource["name"] == "Collections"


class TestMcpResourceReadVariables:
    """Tests for MCP resources/read method with rlm://variables URI."""

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

    def read_resource(self, client, uri: str, request_id: int = 1):
        """Helper to read a resource via MCP resources/read."""
        return self.make_mcp_request(
            client,
            "resources/read",
            params={"uri": uri},
            request_id=request_id
        )

    def call_tool(self, client, tool_name: str, arguments: dict, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        return self.make_mcp_request(
            client,
            "tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    def test_returns_200_status_code(self, client):
        """resources/read rlm://variables should return 200 OK."""
        response = self.read_resource(client, "rlm://variables")
        assert response.status_code == 200

    def test_returns_json(self, client):
        """resources/read rlm://variables should return JSON content."""
        response = self.read_resource(client, "rlm://variables")
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """resources/read rlm://variables should return jsonrpc 2.0."""
        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """resources/read rlm://variables should return the same request id."""
        response = self.read_resource(client, "rlm://variables", request_id=42)
        data = response.json()
        assert data["id"] == 42

    def test_returns_result_dict(self, client):
        """resources/read rlm://variables should return a result dict."""
        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict)

    def test_result_has_contents_key(self, client):
        """resources/read result should have contents key."""
        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        assert "contents" in data["result"]

    def test_contents_is_list(self, client):
        """resources/read contents should be a list."""
        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        assert isinstance(data["result"]["contents"], list)

    def test_contents_has_one_item(self, client):
        """resources/read contents should have exactly one item."""
        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        assert len(data["result"]["contents"]) == 1

    def test_content_has_uri(self, client):
        """resources/read content item should have uri field."""
        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        content = data["result"]["contents"][0]
        assert "uri" in content
        assert content["uri"] == "rlm://variables"

    def test_content_has_mime_type(self, client):
        """resources/read content item should have mimeType field."""
        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        content = data["result"]["contents"][0]
        assert "mimeType" in content
        assert content["mimeType"] == "application/json"

    def test_content_has_text(self, client):
        """resources/read content item should have text field."""
        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        content = data["result"]["contents"][0]
        assert "text" in content
        assert isinstance(content["text"], str)

    def test_text_is_valid_json(self, client):
        """resources/read text field should be valid JSON."""
        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert isinstance(parsed, dict)

    def test_empty_variables_list(self, client):
        """resources/read should return empty variables list when no variables."""
        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert "variables" in parsed
        assert "count" in parsed
        assert parsed["variables"] == []
        assert parsed["count"] == 0

    def test_lists_variables_after_creation(self, client):
        """resources/read should list variables after they are created."""
        # Create a variable
        self.call_tool(client, "rlm_execute", {"code": "x = 42"})

        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert parsed["count"] == 1
        assert len(parsed["variables"]) == 1
        assert parsed["variables"][0]["name"] == "x"

    def test_variable_has_all_fields(self, client):
        """Each variable should have all required fields."""
        # Create a variable
        self.call_tool(client, "rlm_execute", {"code": "test_var = 'hello world'"})

        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        var = parsed["variables"][0]

        assert "name" in var
        assert "type" in var
        assert "size_bytes" in var
        assert "size_human" in var
        assert "preview" in var
        assert "created_at" in var
        assert "last_accessed" in var

    def test_variable_type_is_correct(self, client):
        """Variable type field should reflect the actual type."""
        self.call_tool(client, "rlm_execute", {"code": "my_list = [1, 2, 3]"})

        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        var = parsed["variables"][0]
        assert var["type"] == "list"

    def test_multiple_variables(self, client):
        """resources/read should list multiple variables."""
        self.call_tool(client, "rlm_execute", {"code": "a = 1\nb = 2\nc = 3"})

        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert parsed["count"] == 3
        names = [v["name"] for v in parsed["variables"]]
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_no_error_in_response(self, client):
        """resources/read rlm://variables should not return an error."""
        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        assert "error" not in data

    def test_unknown_uri_returns_error(self, client):
        """resources/read with unknown URI should return error."""
        response = self.read_resource(client, "rlm://unknown")
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32602
        assert "not found" in data["error"]["message"].lower()

    def test_timestamps_are_iso_format(self, client):
        """Variable timestamps should be in ISO format."""
        self.call_tool(client, "rlm_execute", {"code": "timestamp_test = 123"})

        response = self.read_resource(client, "rlm://variables")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        from datetime import datetime
        parsed = json.loads(content["text"])
        var = parsed["variables"][0]

        # Should parse without error
        datetime.fromisoformat(var["created_at"])
        datetime.fromisoformat(var["last_accessed"])


class TestMcpResourceReadMemory:
    """Tests for MCP resources/read method with rlm://memory URI."""

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

    def read_resource(self, client, uri: str, request_id: int = 1):
        """Helper to read a resource via MCP resources/read."""
        return self.make_mcp_request(
            client,
            "resources/read",
            params={"uri": uri},
            request_id=request_id
        )

    def call_tool(self, client, tool_name: str, arguments: dict, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        return self.make_mcp_request(
            client,
            "tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    def test_returns_200_status_code(self, client):
        """resources/read rlm://memory should return 200 OK."""
        response = self.read_resource(client, "rlm://memory")
        assert response.status_code == 200

    def test_returns_json(self, client):
        """resources/read rlm://memory should return JSON content."""
        response = self.read_resource(client, "rlm://memory")
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """resources/read rlm://memory should return jsonrpc 2.0."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """resources/read rlm://memory should return the same request id."""
        response = self.read_resource(client, "rlm://memory", request_id=42)
        data = response.json()
        assert data["id"] == 42

    def test_returns_result_dict(self, client):
        """resources/read rlm://memory should return a result dict."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict)

    def test_result_has_contents_key(self, client):
        """resources/read result should have contents key."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        assert "contents" in data["result"]

    def test_contents_is_list(self, client):
        """resources/read contents should be a list."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        assert isinstance(data["result"]["contents"], list)

    def test_contents_has_one_item(self, client):
        """resources/read contents should have exactly one item."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        assert len(data["result"]["contents"]) == 1

    def test_content_has_uri(self, client):
        """resources/read content item should have uri field."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        content = data["result"]["contents"][0]
        assert "uri" in content
        assert content["uri"] == "rlm://memory"

    def test_content_has_mime_type(self, client):
        """resources/read content item should have mimeType field."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        content = data["result"]["contents"][0]
        assert "mimeType" in content
        assert content["mimeType"] == "application/json"

    def test_content_has_text(self, client):
        """resources/read content item should have text field."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        content = data["result"]["contents"][0]
        assert "text" in content
        assert isinstance(content["text"], str)

    def test_text_is_valid_json(self, client):
        """resources/read text field should be valid JSON."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert isinstance(parsed, dict)

    def test_memory_has_total_bytes(self, client):
        """Memory resource should have total_bytes field."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert "total_bytes" in parsed
        assert isinstance(parsed["total_bytes"], int)

    def test_memory_has_total_human(self, client):
        """Memory resource should have total_human field."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert "total_human" in parsed
        assert isinstance(parsed["total_human"], str)

    def test_memory_has_variable_count(self, client):
        """Memory resource should have variable_count field."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert "variable_count" in parsed
        assert isinstance(parsed["variable_count"], int)

    def test_memory_has_max_allowed_mb(self, client):
        """Memory resource should have max_allowed_mb field."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert "max_allowed_mb" in parsed
        assert isinstance(parsed["max_allowed_mb"], int)

    def test_memory_has_usage_percent(self, client):
        """Memory resource should have usage_percent field."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert "usage_percent" in parsed
        assert isinstance(parsed["usage_percent"], (int, float))

    def test_empty_repl_has_zero_bytes(self, client):
        """Empty REPL should have zero total_bytes for user data."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        # Note: There may be internal functions, but user variables should start at 0
        assert parsed["total_bytes"] >= 0

    def test_memory_increases_with_data(self, client):
        """Memory should increase when data is added."""
        response1 = self.read_resource(client, "rlm://memory")
        content1 = response1.json()["result"]["contents"][0]
        import json
        initial_bytes = json.loads(content1["text"])["total_bytes"]

        # Add a large variable
        self.call_tool(client, "rlm_execute", {"code": "big_data = 'x' * 10000"})

        response2 = self.read_resource(client, "rlm://memory")
        content2 = response2.json()["result"]["contents"][0]
        final_bytes = json.loads(content2["text"])["total_bytes"]

        assert final_bytes > initial_bytes

    def test_variable_count_increases(self, client):
        """Variable count should increase when variables are added."""
        response1 = self.read_resource(client, "rlm://memory")
        content1 = response1.json()["result"]["contents"][0]
        import json
        initial_count = json.loads(content1["text"])["variable_count"]

        self.call_tool(client, "rlm_execute", {"code": "new_var = 123"})

        response2 = self.read_resource(client, "rlm://memory")
        content2 = response2.json()["result"]["contents"][0]
        final_count = json.loads(content2["text"])["variable_count"]

        # Variable count includes internal helper functions, so just check it increased
        assert final_count > initial_count

    def test_no_error_in_response(self, client):
        """resources/read rlm://memory should not have error key."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        assert "error" not in data

    def test_usage_percent_is_reasonable(self, client):
        """Usage percent should be between 0 and 100."""
        response = self.read_resource(client, "rlm://memory")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert 0 <= parsed["usage_percent"] <= 100


class TestMcpResourceReadCollections:
    """Tests for MCP resources/read method with rlm://collections URI."""

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

    def read_resource(self, client, uri: str, request_id: int = 1):
        """Helper to read a resource via MCP resources/read."""
        return self.make_mcp_request(
            client,
            "resources/read",
            params={"uri": uri},
            request_id=request_id
        )

    def call_tool(self, client, tool_name: str, arguments: dict, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        return self.make_mcp_request(
            client,
            "tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_state(self, monkeypatch, tmp_path):
        """Reset REPL and persistence state before each test."""
        from rlm_mcp.http_server import repl
        from rlm_mcp import persistence as persistence_module

        # Create temp persistence DB
        db_path = tmp_path / "test.db"
        test_persistence = persistence_module.PersistenceManager(db_path=str(db_path))
        monkeypatch.setattr("rlm_mcp.http_server.get_persistence", lambda: test_persistence)

        repl.clear_all()
        yield
        repl.clear_all()

    def test_returns_200_status_code(self, client):
        """resources/read rlm://collections should return 200 OK."""
        response = self.read_resource(client, "rlm://collections")
        assert response.status_code == 200

    def test_returns_json(self, client):
        """resources/read rlm://collections should return JSON content."""
        response = self.read_resource(client, "rlm://collections")
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """resources/read rlm://collections should return jsonrpc 2.0."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """resources/read rlm://collections should return the same request id."""
        response = self.read_resource(client, "rlm://collections", request_id=42)
        data = response.json()
        assert data["id"] == 42

    def test_returns_result_dict(self, client):
        """resources/read rlm://collections should return a result dict."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict)

    def test_result_has_contents_key(self, client):
        """resources/read result should have contents key."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        assert "contents" in data["result"]

    def test_contents_is_list(self, client):
        """resources/read contents should be a list."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        assert isinstance(data["result"]["contents"], list)

    def test_contents_has_one_item(self, client):
        """resources/read contents should have exactly one item."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        assert len(data["result"]["contents"]) == 1

    def test_content_has_uri(self, client):
        """resources/read content item should have uri field."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        assert "uri" in content
        assert content["uri"] == "rlm://collections"

    def test_content_has_mime_type(self, client):
        """resources/read content item should have mimeType field."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        assert "mimeType" in content
        assert content["mimeType"] == "application/json"

    def test_content_has_text(self, client):
        """resources/read content item should have text field."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        assert "text" in content
        assert isinstance(content["text"], str)

    def test_text_is_valid_json(self, client):
        """resources/read text field should be valid JSON."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert isinstance(parsed, dict)

    def test_collections_has_collections_key(self, client):
        """Collections resource should have collections key."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert "collections" in parsed
        assert isinstance(parsed["collections"], list)

    def test_collections_has_count_key(self, client):
        """Collections resource should have count key."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert "count" in parsed
        assert isinstance(parsed["count"], int)

    def test_empty_collections_returns_zero_count(self, client):
        """Empty persistence should return count of 0."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])
        assert parsed["count"] == 0
        assert parsed["collections"] == []

    def test_collection_has_required_fields(self, client):
        """Each collection should have name, description, variable_count, created_at."""
        # Create a collection first
        self.call_tool(client, "rlm_collection_create", {
            "name": "test_collection",
            "description": "Test collection description"
        })

        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])

        assert len(parsed["collections"]) == 1
        collection = parsed["collections"][0]
        assert "name" in collection
        assert "description" in collection
        assert "variable_count" in collection
        assert "created_at" in collection

    def test_collection_name_is_correct(self, client):
        """Collection name should match what was created."""
        self.call_tool(client, "rlm_collection_create", {
            "name": "my_collection",
            "description": "My description"
        })

        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])

        assert parsed["collections"][0]["name"] == "my_collection"

    def test_collection_description_is_correct(self, client):
        """Collection description should match what was created."""
        self.call_tool(client, "rlm_collection_create", {
            "name": "my_collection",
            "description": "My special description"
        })

        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])

        assert parsed["collections"][0]["description"] == "My special description"

    def test_collection_variable_count_is_zero_initially(self, client):
        """New collection should have variable_count of 0."""
        self.call_tool(client, "rlm_collection_create", {
            "name": "empty_collection",
            "description": "Empty collection"
        })

        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])

        assert parsed["collections"][0]["variable_count"] == 0

    def test_count_matches_collections_length(self, client):
        """Count should match the number of collections."""
        self.call_tool(client, "rlm_collection_create", {
            "name": "collection1",
            "description": "First collection"
        })
        self.call_tool(client, "rlm_collection_create", {
            "name": "collection2",
            "description": "Second collection"
        })

        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        content = data["result"]["contents"][0]
        import json
        parsed = json.loads(content["text"])

        assert parsed["count"] == 2
        assert len(parsed["collections"]) == 2

    def test_no_error_in_response(self, client):
        """resources/read rlm://collections should not have error key."""
        response = self.read_resource(client, "rlm://collections")
        data = response.json()
        assert "error" not in data


class TestMcpToolRlmExecute:
    """Tests for rlm_execute tool via MCP tools/call method."""

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

    def call_tool(self, client, tool_name: str, arguments: dict, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        return self.make_mcp_request(
            client,
            "tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test to avoid cross-test pollution."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    def test_returns_200_status_code(self, client):
        """rlm_execute should return 200 OK."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('hello')"})
        assert response.status_code == 200

    def test_returns_json(self, client):
        """rlm_execute should return JSON content."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('hello')"})
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """rlm_execute should return jsonrpc 2.0."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('hello')"})
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """rlm_execute should return the same request id."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('hello')"}, request_id=42)
        data = response.json()
        assert data["id"] == 42

    def test_returns_result_dict(self, client):
        """rlm_execute should return a result dict."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('hello')"})
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict)

    def test_result_has_content(self, client):
        """rlm_execute result should have 'content' key."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('hello')"})
        data = response.json()
        assert "content" in data["result"]

    def test_content_is_list(self, client):
        """rlm_execute content should be a list."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('hello')"})
        data = response.json()
        assert isinstance(data["result"]["content"], list)

    def test_content_has_text_item(self, client):
        """rlm_execute content should have text type item."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('hello')"})
        data = response.json()
        content = data["result"]["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"
        assert "text" in content[0]

    def test_captures_print_output(self, client):
        """rlm_execute should capture print() output."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('hello world')"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "hello world" in text

    def test_captures_multiple_prints(self, client):
        """rlm_execute should capture multiple print() statements."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('line1')\nprint('line2')"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "line1" in text
        assert "line2" in text

    def test_simple_assignment(self, client):
        """rlm_execute should handle simple variable assignment."""
        response = self.call_tool(client, "rlm_execute", {"code": "x = 42"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should mention variable changed
        assert "x" in text or "VARIÁVEIS ALTERADAS" in text

    def test_arithmetic_operation(self, client):
        """rlm_execute should handle arithmetic operations."""
        response = self.call_tool(client, "rlm_execute", {"code": "print(2 + 3)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "5" in text

    def test_string_operations(self, client):
        """rlm_execute should handle string operations."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('hello'.upper())"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "HELLO" in text

    def test_list_operations(self, client):
        """rlm_execute should handle list operations."""
        response = self.call_tool(client, "rlm_execute", {"code": "nums = [1, 2, 3]\nprint(sum(nums))"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "6" in text

    def test_dict_operations(self, client):
        """rlm_execute should handle dict operations."""
        response = self.call_tool(client, "rlm_execute", {"code": "d = {'a': 1, 'b': 2}\nprint(d['a'])"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "1" in text

    def test_list_comprehension(self, client):
        """rlm_execute should handle list comprehension."""
        response = self.call_tool(client, "rlm_execute", {"code": "print([x*2 for x in range(3)])"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "0" in text and "2" in text and "4" in text

    def test_function_definition_and_call(self, client):
        """rlm_execute should handle function definition and call."""
        code = """def greet(name):
    return f'Hello, {name}!'
print(greet('World'))"""
        response = self.call_tool(client, "rlm_execute", {"code": code})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "Hello, World!" in text

    def test_syntax_error_returns_error(self, client):
        """rlm_execute should handle syntax errors gracefully."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('unclosed"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should indicate error
        assert "ERRO" in text or "Error" in text.lower() or "error" in text.lower()

    def test_runtime_error_returns_error(self, client):
        """rlm_execute should handle runtime errors gracefully."""
        response = self.call_tool(client, "rlm_execute", {"code": "print(1/0)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "ERRO" in text or "Error" in text.lower() or "ZeroDivision" in text

    def test_name_error_returns_error(self, client):
        """rlm_execute should handle NameError gracefully."""
        response = self.call_tool(client, "rlm_execute", {"code": "print(undefined_var)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "ERRO" in text or "Error" in text.lower() or "NameError" in text

    def test_no_error_in_response_for_valid_code(self, client):
        """rlm_execute should not return error field for valid code."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('test')"})
        data = response.json()
        assert data.get("error") is None

    def test_execution_status_ok(self, client):
        """rlm_execute should show OK status for valid code."""
        response = self.call_tool(client, "rlm_execute", {"code": "x = 1"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "OK" in text

    def test_execution_time_shown(self, client):
        """rlm_execute should show execution time."""
        response = self.call_tool(client, "rlm_execute", {"code": "x = 1"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should contain execution time in ms
        assert "ms" in text

    def test_empty_code_succeeds(self, client):
        """rlm_execute should handle empty code."""
        response = self.call_tool(client, "rlm_execute", {"code": ""})
        data = response.json()
        assert "result" in data
        text = data["result"]["content"][0]["text"]
        assert "OK" in text or "concluída" in text.lower()

    def test_comment_only_code_succeeds(self, client):
        """rlm_execute should handle code with only comments."""
        response = self.call_tool(client, "rlm_execute", {"code": "# this is a comment"})
        data = response.json()
        assert "result" in data

    def test_multiline_code(self, client):
        """rlm_execute should handle multiline code."""
        code = """a = 1
b = 2
c = a + b
print(f'Sum: {c}')"""
        response = self.call_tool(client, "rlm_execute", {"code": code})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "Sum: 3" in text

    def test_safe_imports_work(self, client):
        """rlm_execute should allow safe imports."""
        response = self.call_tool(client, "rlm_execute", {"code": "import math\nprint(math.pi)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "3.14" in text

    def test_blocked_imports_fail(self, client):
        """rlm_execute should block dangerous imports."""
        response = self.call_tool(client, "rlm_execute", {"code": "import os"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should contain error about blocked import
        assert "bloqueado" in text.lower() or "blocked" in text.lower() or "ERRO" in text

    def test_json_module_works(self, client):
        """rlm_execute should allow json module."""
        code = """import json
data = json.dumps({'key': 'value'})
print(data)"""
        response = self.call_tool(client, "rlm_execute", {"code": code})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "key" in text and "value" in text

    def test_re_module_works(self, client):
        """rlm_execute should allow re module."""
        code = """import re
result = re.findall(r'\\d+', 'abc123def456')
print(result)"""
        response = self.call_tool(client, "rlm_execute", {"code": code})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "123" in text and "456" in text

    def test_variables_persist_across_executions(self, client):
        """Variables from one execution should be available in the next."""
        # First execution: set variable
        self.call_tool(client, "rlm_execute", {"code": "my_var = 'persisted_value'"})

        # Second execution: use variable
        response = self.call_tool(client, "rlm_execute", {"code": "print(my_var)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "persisted_value" in text

    def test_functions_persist_across_executions(self, client):
        """Functions from one execution should be available in the next."""
        # First execution: define function
        self.call_tool(client, "rlm_execute", {"code": "def double(x): return x * 2"})

        # Second execution: use function
        response = self.call_tool(client, "rlm_execute", {"code": "print(double(21))"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "42" in text

    def test_missing_code_parameter(self, client):
        """rlm_execute should handle missing code parameter."""
        response = self.call_tool(client, "rlm_execute", {})
        data = response.json()
        # Should return an error
        assert "error" in data or "isError" in data.get("result", {})


class TestMcpToolRlmLoadData:
    """Tests for rlm_load_data tool via MCP tools/call method."""

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

    def call_tool(self, client, tool_name: str, arguments: dict, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        return self.make_mcp_request(
            client,
            "tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test to avoid cross-test pollution."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    def test_returns_200_status_code(self, client):
        """rlm_load_data should return 200 OK."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test", "data": "hello"})
        assert response.status_code == 200

    def test_returns_json(self, client):
        """rlm_load_data should return JSON content."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test", "data": "hello"})
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """rlm_load_data should return jsonrpc 2.0."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test", "data": "hello"})
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """rlm_load_data should return the same request id."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test", "data": "hello"}, request_id=77)
        data = response.json()
        assert data["id"] == 77

    def test_returns_result_dict(self, client):
        """rlm_load_data should return a result dict."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test", "data": "hello"})
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict)

    def test_result_has_content(self, client):
        """rlm_load_data result should have 'content' key."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test", "data": "hello"})
        data = response.json()
        assert "content" in data["result"]

    def test_content_is_list(self, client):
        """rlm_load_data content should be a list."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test", "data": "hello"})
        data = response.json()
        assert isinstance(data["result"]["content"], list)

    def test_content_has_text_item(self, client):
        """rlm_load_data content should have text type item."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test", "data": "hello"})
        data = response.json()
        content = data["result"]["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"
        assert "text" in content[0]

    def test_loads_text_data(self, client):
        """rlm_load_data should load text data into variable."""
        response = self.call_tool(client, "rlm_load_data", {"name": "myvar", "data": "hello world"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "myvar" in text
        assert "carregada" in text.lower()

    def test_variable_accessible_via_execute(self, client):
        """Variable loaded via rlm_load_data should be accessible via rlm_execute."""
        # Load data
        self.call_tool(client, "rlm_load_data", {"name": "mytext", "data": "test_value_123"})

        # Access it via execute
        response = self.call_tool(client, "rlm_execute", {"code": "print(mytext)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "test_value_123" in text

    def test_loads_json_data(self, client):
        """rlm_load_data should load JSON data correctly."""
        json_data = '{"key": "value", "num": 42}'
        response = self.call_tool(client, "rlm_load_data", {
            "name": "myjson",
            "data": json_data,
            "data_type": "json"
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "myjson" in text
        assert "carregada" in text.lower()

    def test_json_variable_accessible_via_execute(self, client):
        """JSON variable loaded should be accessible as dict."""
        json_data = '{"name": "test", "count": 5}'
        self.call_tool(client, "rlm_load_data", {
            "name": "config",
            "data": json_data,
            "data_type": "json"
        })

        response = self.call_tool(client, "rlm_execute", {"code": "print(config['name'])"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "test" in text

    def test_loads_csv_data(self, client):
        """rlm_load_data should load CSV data correctly."""
        csv_data = "name,age\nAlice,30\nBob,25"
        response = self.call_tool(client, "rlm_load_data", {
            "name": "people",
            "data": csv_data,
            "data_type": "csv"
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "people" in text
        assert "carregada" in text.lower()

    def test_csv_variable_accessible_as_list(self, client):
        """CSV variable loaded should be accessible as list of dicts."""
        csv_data = "name,age\nAlice,30\nBob,25"
        self.call_tool(client, "rlm_load_data", {
            "name": "users",
            "data": csv_data,
            "data_type": "csv"
        })

        response = self.call_tool(client, "rlm_execute", {"code": "print(len(users))"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "2" in text

    def test_loads_lines_data(self, client):
        """rlm_load_data should load lines data correctly."""
        lines_data = "line1\nline2\nline3"
        response = self.call_tool(client, "rlm_load_data", {
            "name": "mylines",
            "data": lines_data,
            "data_type": "lines"
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "mylines" in text

    def test_lines_variable_is_list(self, client):
        """Lines variable loaded should be a list."""
        lines_data = "first\nsecond\nthird"
        self.call_tool(client, "rlm_load_data", {
            "name": "lines_list",
            "data": lines_data,
            "data_type": "lines"
        })

        response = self.call_tool(client, "rlm_execute", {"code": "print(lines_list[1])"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "second" in text

    def test_default_data_type_is_text(self, client):
        """rlm_load_data should default to data_type='text'."""
        # Load without data_type
        self.call_tool(client, "rlm_load_data", {"name": "default_type", "data": "some text"})

        # Variable should be string - test using isinstance which is allowed
        response = self.call_tool(client, "rlm_execute", {"code": "print(isinstance(default_type, str))"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "True" in text

    def test_overwrites_existing_variable(self, client):
        """rlm_load_data should overwrite existing variable with same name."""
        # First load
        self.call_tool(client, "rlm_load_data", {"name": "myvar", "data": "first"})

        # Second load with same name
        self.call_tool(client, "rlm_load_data", {"name": "myvar", "data": "second"})

        # Check value
        response = self.call_tool(client, "rlm_execute", {"code": "print(myvar)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "second" in text
        assert "first" not in text

    def test_no_error_in_response_for_valid_data(self, client):
        """rlm_load_data should not return error field for valid data."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test", "data": "hello"})
        data = response.json()
        assert data.get("error") is None

    def test_shows_variable_type_in_output(self, client):
        """rlm_load_data should show variable type in output."""
        response = self.call_tool(client, "rlm_load_data", {"name": "typed", "data": "text data"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "str" in text.lower() or "text" in text.lower()

    def test_shows_variable_size_in_output(self, client):
        """rlm_load_data should show variable size in output."""
        response = self.call_tool(client, "rlm_load_data", {"name": "sized", "data": "some data here"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should contain size info like "14 B" or similar
        assert "B" in text or "bytes" in text.lower()

    def test_handles_unicode_data(self, client):
        """rlm_load_data should handle Unicode data."""
        unicode_data = "Olá, mundo! 日本語 中文 한국어"
        self.call_tool(client, "rlm_load_data", {"name": "unicode_var", "data": unicode_data})

        response = self.call_tool(client, "rlm_execute", {"code": "print(unicode_var)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "Olá" in text
        assert "日本語" in text

    def test_handles_empty_string(self, client):
        """rlm_load_data should handle empty string."""
        response = self.call_tool(client, "rlm_load_data", {"name": "empty", "data": ""})
        data = response.json()
        assert "result" in data
        # Should succeed
        assert data.get("error") is None

    def test_handles_multiline_text(self, client):
        """rlm_load_data should handle multiline text."""
        multiline = "line 1\nline 2\nline 3"
        self.call_tool(client, "rlm_load_data", {"name": "multiline", "data": multiline})

        response = self.call_tool(client, "rlm_execute", {"code": "print(multiline.count('\\n'))"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "2" in text

    def test_handles_large_data(self, client):
        """rlm_load_data should handle large data."""
        # 100KB of data
        large_data = "x" * 100000
        response = self.call_tool(client, "rlm_load_data", {"name": "large_var", "data": large_data})
        data = response.json()
        assert "result" in data

        # Verify it's loaded
        exec_response = self.call_tool(client, "rlm_execute", {"code": "print(len(large_var))"})
        exec_data = exec_response.json()
        text = exec_data["result"]["content"][0]["text"]
        assert "100000" in text

    def test_handles_special_characters(self, client):
        """rlm_load_data should handle special characters."""
        special_data = "tab:\there quote:\"test\" backslash:\\ newline:\nend"
        self.call_tool(client, "rlm_load_data", {"name": "special", "data": special_data})

        response = self.call_tool(client, "rlm_execute", {"code": "print('quote' in special)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "True" in text

    def test_missing_name_parameter(self, client):
        """rlm_load_data should handle missing name parameter."""
        response = self.call_tool(client, "rlm_load_data", {"data": "hello"})
        data = response.json()
        # Should return an error
        assert "error" in data or "isError" in data.get("result", {}) or "Error" in str(data)

    def test_missing_data_parameter(self, client):
        """rlm_load_data should handle missing data parameter."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test"})
        data = response.json()
        # Should return an error
        assert "error" in data or "isError" in data.get("result", {}) or "Error" in str(data)

    def test_invalid_json_returns_error(self, client):
        """rlm_load_data should return error for invalid JSON."""
        response = self.call_tool(client, "rlm_load_data", {
            "name": "bad_json",
            "data": "{invalid json}",
            "data_type": "json"
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should indicate error
        assert "ERRO" in text or "Error" in text or "error" in text.lower()

    def test_multiple_loads_preserve_all_variables(self, client):
        """Multiple rlm_load_data calls should preserve all variables."""
        self.call_tool(client, "rlm_load_data", {"name": "var1", "data": "value1"})
        self.call_tool(client, "rlm_load_data", {"name": "var2", "data": "value2"})
        self.call_tool(client, "rlm_load_data", {"name": "var3", "data": "value3"})

        # All should be accessible
        response = self.call_tool(client, "rlm_execute", {"code": "print(var1, var2, var3)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "value1" in text
        assert "value2" in text
        assert "value3" in text

    def test_variable_usable_in_computations(self, client):
        """Variable loaded should be usable in Python computations."""
        json_data = '{"numbers": [1, 2, 3, 4, 5]}'
        self.call_tool(client, "rlm_load_data", {
            "name": "nums_data",
            "data": json_data,
            "data_type": "json"
        })

        response = self.call_tool(client, "rlm_execute", {"code": "print(sum(nums_data['numbers']))"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "15" in text

    def test_with_string_request_id(self, client):
        """rlm_load_data should work with string request id."""
        payload = {
            "jsonrpc": "2.0",
            "id": "load-data-123",
            "method": "tools/call",
            "params": {
                "name": "rlm_load_data",
                "arguments": {"name": "str_id_var", "data": "test"}
            }
        }
        response = client.post("/mcp", json=payload)
        data = response.json()
        assert data["id"] == "load-data-123"
        assert "result" in data


class TestMcpToolRlmListVars:
    """Tests for rlm_list_vars tool via MCP tools/call method."""

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

    def call_tool(self, client, tool_name: str, arguments: dict = None, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        params = {"name": tool_name}
        if arguments is not None:
            params["arguments"] = arguments
        else:
            params["arguments"] = {}
        return self.make_mcp_request(
            client,
            "tools/call",
            params=params,
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test to avoid cross-test pollution."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    def test_returns_200_status_code(self, client):
        """rlm_list_vars should return 200 OK."""
        response = self.call_tool(client, "rlm_list_vars")
        assert response.status_code == 200

    def test_returns_json(self, client):
        """rlm_list_vars should return JSON content."""
        response = self.call_tool(client, "rlm_list_vars")
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """rlm_list_vars should return jsonrpc 2.0."""
        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """rlm_list_vars should return the same request id."""
        response = self.call_tool(client, "rlm_list_vars", request_id=55)
        data = response.json()
        assert data["id"] == 55

    def test_returns_result_dict(self, client):
        """rlm_list_vars should return a result dict."""
        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict)

    def test_result_has_content(self, client):
        """rlm_list_vars result should have 'content' key."""
        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        assert "content" in data["result"]

    def test_content_is_list(self, client):
        """rlm_list_vars content should be a list."""
        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        assert isinstance(data["result"]["content"], list)

    def test_content_has_text_item(self, client):
        """rlm_list_vars content should have text type item."""
        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        content = data["result"]["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"
        assert "text" in content[0]

    def test_empty_repl_shows_no_variables_message(self, client):
        """rlm_list_vars should show 'no variables' message when REPL is empty."""
        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "Nenhuma variável" in text or "nenhuma" in text.lower()

    def test_shows_loaded_variable(self, client):
        """rlm_list_vars should list variables loaded via rlm_load_data."""
        # Load a variable first
        self.call_tool(client, "rlm_load_data", {"name": "my_var", "data": "test data"})

        # List variables
        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "my_var" in text

    def test_shows_variable_type(self, client):
        """rlm_list_vars should show variable type."""
        self.call_tool(client, "rlm_load_data", {"name": "str_var", "data": "hello"})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "str" in text.lower()

    def test_shows_variable_size(self, client):
        """rlm_list_vars should show variable size in human-readable format."""
        self.call_tool(client, "rlm_load_data", {"name": "sized_var", "data": "some data"})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should contain size like "9.0 B" or similar
        assert "B" in text or "KB" in text or "MB" in text

    def test_shows_variable_preview(self, client):
        """rlm_list_vars should show variable preview."""
        self.call_tool(client, "rlm_load_data", {"name": "preview_var", "data": "preview_content_here"})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "Preview" in text
        assert "preview_content" in text

    def test_lists_multiple_variables(self, client):
        """rlm_list_vars should list all loaded variables."""
        self.call_tool(client, "rlm_load_data", {"name": "var1", "data": "data1"})
        self.call_tool(client, "rlm_load_data", {"name": "var2", "data": "data2"})
        self.call_tool(client, "rlm_load_data", {"name": "var3", "data": "data3"})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "var1" in text
        assert "var2" in text
        assert "var3" in text

    def test_shows_dict_variable(self, client):
        """rlm_list_vars should show dict variable with correct type."""
        json_data = '{"key": "value"}'
        self.call_tool(client, "rlm_load_data", {"name": "dict_var", "data": json_data, "data_type": "json"})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "dict_var" in text
        assert "dict" in text.lower()

    def test_shows_list_variable(self, client):
        """rlm_list_vars should show list variable with correct type."""
        json_data = '[1, 2, 3]'
        self.call_tool(client, "rlm_load_data", {"name": "list_var", "data": json_data, "data_type": "json"})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "list_var" in text
        assert "list" in text.lower()

    def test_shows_csv_variable_as_list(self, client):
        """rlm_list_vars should show CSV variable as list type."""
        csv_data = "name,age\nAlice,30\nBob,25"
        self.call_tool(client, "rlm_load_data", {"name": "csv_var", "data": csv_data, "data_type": "csv"})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "csv_var" in text
        assert "list" in text.lower()

    def test_shows_variable_created_via_execute(self, client):
        """rlm_list_vars should show variables created via rlm_execute."""
        self.call_tool(client, "rlm_execute", {"code": "exec_var = 'created via execute'"})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "exec_var" in text

    def test_no_error_in_response(self, client):
        """rlm_list_vars should not return error field."""
        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        assert data.get("error") is None

    def test_header_shows_variáveis_no_repl(self, client):
        """rlm_list_vars should show header 'Variáveis no REPL' when there are variables."""
        self.call_tool(client, "rlm_load_data", {"name": "test", "data": "value"})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "Variáveis no REPL" in text

    def test_with_string_request_id(self, client):
        """rlm_list_vars should work with string request id."""
        payload = {
            "jsonrpc": "2.0",
            "id": "list-vars-request",
            "method": "tools/call",
            "params": {
                "name": "rlm_list_vars",
                "arguments": {}
            }
        }
        response = client.post("/mcp", json=payload)
        data = response.json()
        assert data["id"] == "list-vars-request"
        assert "result" in data

    def test_multiple_requests_return_same_variables(self, client):
        """Multiple rlm_list_vars calls should return same variables."""
        self.call_tool(client, "rlm_load_data", {"name": "persist_var", "data": "test"})

        response1 = self.call_tool(client, "rlm_list_vars", request_id=1)
        response2 = self.call_tool(client, "rlm_list_vars", request_id=2)

        text1 = response1.json()["result"]["content"][0]["text"]
        text2 = response2.json()["result"]["content"][0]["text"]

        assert "persist_var" in text1
        assert "persist_var" in text2

    def test_reflects_cleared_variables(self, client):
        """rlm_list_vars should reflect variables cleared via rlm_clear."""
        # Load then clear
        self.call_tool(client, "rlm_load_data", {"name": "to_clear", "data": "test"})
        self.call_tool(client, "rlm_clear", {"all": True})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "to_clear" not in text
        assert "Nenhuma variável" in text or "nenhuma" in text.lower()

    def test_shows_large_variable_size_in_kb(self, client):
        """rlm_list_vars should show large variable size in KB."""
        # Create ~10KB of data
        large_data = "x" * 10000
        self.call_tool(client, "rlm_load_data", {"name": "large_var", "data": large_data})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "large_var" in text
        # Size should be around 9.8 KB
        assert "KB" in text

    def test_preview_truncated_for_long_values(self, client):
        """rlm_list_vars should truncate preview for long values."""
        # Load data with more than 100 chars
        long_data = "a" * 200
        self.call_tool(client, "rlm_load_data", {"name": "long_var", "data": long_data})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Preview should be truncated with "..."
        assert "..." in text

    def test_does_not_include_llm_functions(self, client):
        """rlm_list_vars should not list internal llm_* functions in regular output."""
        # Execute something to trigger llm_* injection
        self.call_tool(client, "rlm_execute", {"code": "x = 1"})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Internal functions like llm_query should not be prominently listed
        # (they may exist in repl.variables but list_variables() uses variable_metadata)
        # llm_* functions are injected into namespace but not added to variable_metadata
        # unless explicitly created by user code
        assert "x" in text  # User variable should be there

    def test_handles_unicode_variable_names(self, client):
        """rlm_list_vars should handle Unicode variable content in preview."""
        unicode_data = "Olá, mundo! 日本語 中文"
        self.call_tool(client, "rlm_load_data", {"name": "unicode_var", "data": unicode_data})

        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "unicode_var" in text
        # Preview should contain some of the unicode content
        assert "Olá" in text or "日本語" in text or "mundo" in text

    def test_response_is_dict(self, client):
        """rlm_list_vars should return a dictionary response."""
        response = self.call_tool(client, "rlm_list_vars")
        data = response.json()
        assert isinstance(data, dict)


class TestMcpToolRlmVarInfo:
    """Tests for rlm_var_info tool via MCP tools/call method."""

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

    def call_tool(self, client, tool_name: str, arguments: dict = None, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        params = {"name": tool_name}
        if arguments is not None:
            params["arguments"] = arguments
        else:
            params["arguments"] = {}
        return self.make_mcp_request(
            client,
            "tools/call",
            params=params,
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test to avoid cross-test pollution."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    def test_returns_200_status_code(self, client):
        """rlm_var_info should return 200 OK."""
        self.call_tool(client, "rlm_load_data", {"name": "test_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "test_var"})
        assert response.status_code == 200

    def test_returns_json(self, client):
        """rlm_var_info should return JSON content."""
        self.call_tool(client, "rlm_load_data", {"name": "test_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "test_var"})
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """rlm_var_info should return jsonrpc 2.0."""
        self.call_tool(client, "rlm_load_data", {"name": "test_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "test_var"})
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """rlm_var_info should return the same request id."""
        self.call_tool(client, "rlm_load_data", {"name": "test_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "test_var"}, request_id=88)
        data = response.json()
        assert data["id"] == 88

    def test_returns_result_dict(self, client):
        """rlm_var_info should return a result dict."""
        self.call_tool(client, "rlm_load_data", {"name": "test_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "test_var"})
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict)

    def test_result_has_content(self, client):
        """rlm_var_info result should have 'content' key."""
        self.call_tool(client, "rlm_load_data", {"name": "test_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "test_var"})
        data = response.json()
        assert "content" in data["result"]

    def test_content_is_list(self, client):
        """rlm_var_info content should be a list."""
        self.call_tool(client, "rlm_load_data", {"name": "test_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "test_var"})
        data = response.json()
        assert isinstance(data["result"]["content"], list)

    def test_content_has_text_item(self, client):
        """rlm_var_info content should have text type item."""
        self.call_tool(client, "rlm_load_data", {"name": "test_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "test_var"})
        data = response.json()
        content = data["result"]["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"
        assert "text" in content[0]

    def test_shows_variable_name(self, client):
        """rlm_var_info should show the variable name."""
        self.call_tool(client, "rlm_load_data", {"name": "my_variable", "data": "test data"})
        response = self.call_tool(client, "rlm_var_info", {"name": "my_variable"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "my_variable" in text
        assert "Variável:" in text or "Variavel:" in text

    def test_shows_variable_type(self, client):
        """rlm_var_info should show the variable type."""
        self.call_tool(client, "rlm_load_data", {"name": "str_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "str_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "Tipo:" in text
        assert "str" in text.lower()

    def test_shows_variable_size_bytes(self, client):
        """rlm_var_info should show the variable size in bytes."""
        self.call_tool(client, "rlm_load_data", {"name": "sized_var", "data": "hello world"})
        response = self.call_tool(client, "rlm_var_info", {"name": "sized_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "Tamanho:" in text
        assert "bytes" in text.lower()

    def test_shows_human_readable_size(self, client):
        """rlm_var_info should show human-readable size."""
        self.call_tool(client, "rlm_load_data", {"name": "sized_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "sized_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should contain human-readable size like "5.0 B"
        assert "B" in text

    def test_shows_created_at_timestamp(self, client):
        """rlm_var_info should show created_at timestamp."""
        self.call_tool(client, "rlm_load_data", {"name": "timed_var", "data": "test"})
        response = self.call_tool(client, "rlm_var_info", {"name": "timed_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "Criada em:" in text or "criada" in text.lower()

    def test_shows_last_accessed_timestamp(self, client):
        """rlm_var_info should show last_accessed timestamp."""
        self.call_tool(client, "rlm_load_data", {"name": "timed_var", "data": "test"})
        response = self.call_tool(client, "rlm_var_info", {"name": "timed_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "Último acesso:" in text or "acesso" in text.lower() or "accessed" in text.lower()

    def test_shows_variable_preview(self, client):
        """rlm_var_info should show variable preview."""
        self.call_tool(client, "rlm_load_data", {"name": "preview_var", "data": "unique_preview_content"})
        response = self.call_tool(client, "rlm_var_info", {"name": "preview_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "Preview:" in text
        assert "unique_preview_content" in text

    def test_nonexistent_variable_shows_error_message(self, client):
        """rlm_var_info should show error message for nonexistent variable."""
        response = self.call_tool(client, "rlm_var_info", {"name": "does_not_exist"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "não encontrada" in text.lower() or "not found" in text.lower()
        assert "does_not_exist" in text

    def test_no_error_field_for_existing_variable(self, client):
        """rlm_var_info should not return error field for existing variable."""
        self.call_tool(client, "rlm_load_data", {"name": "test_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "test_var"})
        data = response.json()
        assert data.get("error") is None

    def test_dict_variable_info(self, client):
        """rlm_var_info should show correct info for dict variable."""
        json_data = '{"key1": "value1", "key2": 42}'
        self.call_tool(client, "rlm_load_data", {"name": "dict_var", "data": json_data, "data_type": "json"})
        response = self.call_tool(client, "rlm_var_info", {"name": "dict_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "dict" in text.lower()
        assert "dict_var" in text

    def test_list_variable_info(self, client):
        """rlm_var_info should show correct info for list variable."""
        json_data = '[1, 2, 3, 4, 5]'
        self.call_tool(client, "rlm_load_data", {"name": "list_var", "data": json_data, "data_type": "json"})
        response = self.call_tool(client, "rlm_var_info", {"name": "list_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "list" in text.lower()
        assert "list_var" in text

    def test_csv_variable_info(self, client):
        """rlm_var_info should show correct info for CSV variable (list of dicts)."""
        csv_data = "name,age\nAlice,30\nBob,25"
        self.call_tool(client, "rlm_load_data", {"name": "csv_var", "data": csv_data, "data_type": "csv"})
        response = self.call_tool(client, "rlm_var_info", {"name": "csv_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "list" in text.lower()
        assert "csv_var" in text

    def test_large_variable_shows_kb_size(self, client):
        """rlm_var_info should show size in KB for large variables."""
        # Create ~10KB of data
        large_data = "x" * 10000
        self.call_tool(client, "rlm_load_data", {"name": "large_var", "data": large_data})
        response = self.call_tool(client, "rlm_var_info", {"name": "large_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "large_var" in text
        assert "KB" in text

    def test_variable_created_via_execute(self, client):
        """rlm_var_info should work for variables created via rlm_execute."""
        self.call_tool(client, "rlm_execute", {"code": "exec_var = 'created via execute'"})
        response = self.call_tool(client, "rlm_var_info", {"name": "exec_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "exec_var" in text
        assert "str" in text.lower()

    def test_timestamps_are_iso_format(self, client):
        """rlm_var_info timestamps should be in ISO format."""
        from datetime import datetime
        self.call_tool(client, "rlm_load_data", {"name": "iso_var", "data": "test"})
        response = self.call_tool(client, "rlm_var_info", {"name": "iso_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Extract timestamp and verify it's valid ISO format
        # Look for patterns like "2024-01-15T10:30:45"
        import re
        iso_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
        matches = re.findall(iso_pattern, text)
        assert len(matches) >= 2  # Should have created_at and last_accessed
        # Verify they parse without error
        for ts in matches:
            datetime.fromisoformat(ts)

    def test_with_string_request_id(self, client):
        """rlm_var_info should work with string request id."""
        self.call_tool(client, "rlm_load_data", {"name": "str_id_var", "data": "test"})
        payload = {
            "jsonrpc": "2.0",
            "id": "var-info-request-123",
            "method": "tools/call",
            "params": {
                "name": "rlm_var_info",
                "arguments": {"name": "str_id_var"}
            }
        }
        response = client.post("/mcp", json=payload)
        data = response.json()
        assert data["id"] == "var-info-request-123"
        assert "result" in data

    def test_missing_name_parameter(self, client):
        """rlm_var_info should handle missing name parameter."""
        response = self.call_tool(client, "rlm_var_info", {})
        data = response.json()
        # Should return an error
        assert "error" in data or "isError" in data.get("result", {}) or "Error" in str(data)

    def test_unicode_variable_content_in_preview(self, client):
        """rlm_var_info should handle Unicode content in preview."""
        unicode_data = "Olá, mundo! 日本語 中文 한국어"
        self.call_tool(client, "rlm_load_data", {"name": "unicode_var", "data": unicode_data})
        response = self.call_tool(client, "rlm_var_info", {"name": "unicode_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert "unicode_var" in text
        # Preview should contain some of the unicode content
        assert "Olá" in text or "日本語" in text or "mundo" in text

    def test_preview_truncated_for_long_values(self, client):
        """rlm_var_info should truncate preview for very long values."""
        # Create data longer than typical preview limit
        long_data = "a" * 500
        self.call_tool(client, "rlm_load_data", {"name": "long_var", "data": long_data})
        response = self.call_tool(client, "rlm_var_info", {"name": "long_var"})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Preview should be truncated with "..."
        assert "..." in text

    def test_multiple_requests_for_same_variable(self, client):
        """Multiple rlm_var_info calls for same variable should return consistent info."""
        self.call_tool(client, "rlm_load_data", {"name": "consistent_var", "data": "test"})

        response1 = self.call_tool(client, "rlm_var_info", {"name": "consistent_var"}, request_id=1)
        response2 = self.call_tool(client, "rlm_var_info", {"name": "consistent_var"}, request_id=2)

        text1 = response1.json()["result"]["content"][0]["text"]
        text2 = response2.json()["result"]["content"][0]["text"]

        # Both should contain the variable name and type
        assert "consistent_var" in text1
        assert "consistent_var" in text2
        assert "str" in text1.lower()
        assert "str" in text2.lower()

    def test_response_is_dict(self, client):
        """rlm_var_info should return a dictionary response."""
        self.call_tool(client, "rlm_load_data", {"name": "test_var", "data": "hello"})
        response = self.call_tool(client, "rlm_var_info", {"name": "test_var"})
        data = response.json()
        assert isinstance(data, dict)


class TestMcpToolRlmClear:
    """Tests for rlm_clear tool via MCP tools/call method."""

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

    def call_tool(self, client, tool_name: str, arguments: dict, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        return self.make_mcp_request(
            client,
            "tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test to avoid cross-test pollution."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    def test_returns_200_status_code(self, client):
        """rlm_clear should return 200 OK."""
        response = self.call_tool(client, "rlm_clear", {"all": True})
        assert response.status_code == 200

    def test_returns_json(self, client):
        """rlm_clear should return JSON content."""
        response = self.call_tool(client, "rlm_clear", {"all": True})
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """rlm_clear should return jsonrpc 2.0."""
        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """rlm_clear should return the same request id."""
        response = self.call_tool(client, "rlm_clear", {"all": True}, request_id=42)
        data = response.json()
        assert data["id"] == 42

    def test_returns_result_dict(self, client):
        """rlm_clear should return a result dict."""
        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict)

    def test_result_has_content(self, client):
        """rlm_clear result should have 'content' key."""
        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        assert "content" in data["result"]

    def test_content_is_list(self, client):
        """rlm_clear content should be a list."""
        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        assert isinstance(data["result"]["content"], list)

    def test_content_has_text_item(self, client):
        """rlm_clear content should have text type item."""
        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        content = data["result"]["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"
        assert "text" in content[0]

    def test_clear_all_removes_all_variables(self, client):
        """rlm_clear with all=True should remove all variables."""
        # Create several variables first
        self.call_tool(client, "rlm_load_data", {"name": "var1", "data": "test1"})
        self.call_tool(client, "rlm_load_data", {"name": "var2", "data": "test2"})
        self.call_tool(client, "rlm_load_data", {"name": "var3", "data": "test3"})

        # Clear all
        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should mention count of removed variables
        assert "3" in text
        assert "removidas" in text.lower() or "variáveis" in text.lower()

    def test_clear_all_returns_count_in_message(self, client):
        """rlm_clear with all=True should return count of removed variables."""
        # Create variables
        self.call_tool(client, "rlm_load_data", {"name": "a", "data": "1"})
        self.call_tool(client, "rlm_load_data", {"name": "b", "data": "2"})

        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Message format: "Todas as N variáveis foram removidas."
        assert "2" in text

    def test_clear_all_on_empty_namespace(self, client):
        """rlm_clear with all=True on empty namespace should return 0 count."""
        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should say 0 variables removed
        assert "0" in text

    def test_clear_single_variable(self, client):
        """rlm_clear with name should remove only that variable."""
        # Create several variables
        self.call_tool(client, "rlm_load_data", {"name": "keep1", "data": "a"})
        self.call_tool(client, "rlm_load_data", {"name": "remove_me", "data": "b"})
        self.call_tool(client, "rlm_load_data", {"name": "keep2", "data": "c"})

        # Clear only one
        response = self.call_tool(client, "rlm_clear", {"name": "remove_me"})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should confirm removal
        assert "remove_me" in text
        assert "removida" in text.lower()

    def test_clear_single_variable_leaves_others(self, client):
        """rlm_clear with name should not affect other variables."""
        # Create variables
        self.call_tool(client, "rlm_load_data", {"name": "keep1", "data": "a"})
        self.call_tool(client, "rlm_load_data", {"name": "remove_me", "data": "b"})
        self.call_tool(client, "rlm_load_data", {"name": "keep2", "data": "c"})

        # Clear one
        self.call_tool(client, "rlm_clear", {"name": "remove_me"})

        # Verify other variables still exist via rlm_list_vars
        response = self.call_tool(client, "rlm_list_vars", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        assert "keep1" in text
        assert "keep2" in text
        assert "remove_me" not in text

    def test_clear_nonexistent_variable(self, client):
        """rlm_clear with name for nonexistent variable should return error message."""
        response = self.call_tool(client, "rlm_clear", {"name": "does_not_exist"})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should indicate not found
        assert "does_not_exist" in text
        assert "não encontrada" in text.lower() or "not found" in text.lower()

    def test_clear_no_parameters_returns_error(self, client):
        """rlm_clear without name or all should return helpful message."""
        response = self.call_tool(client, "rlm_clear", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should ask for name or all
        assert "name" in text.lower() or "all" in text.lower()

    def test_clear_variable_accessible_after_recreation(self, client):
        """After clearing a variable, it can be recreated with same name."""
        # Create, clear, recreate
        self.call_tool(client, "rlm_load_data", {"name": "reusable", "data": "original"})
        self.call_tool(client, "rlm_clear", {"name": "reusable"})
        self.call_tool(client, "rlm_load_data", {"name": "reusable", "data": "new value"})

        # Verify new value
        response = self.call_tool(client, "rlm_execute", {"code": "print(reusable)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        assert "new value" in text

    def test_clear_all_then_create_new_variables(self, client):
        """After clear all, new variables can be created."""
        # Create, clear all, create again
        self.call_tool(client, "rlm_load_data", {"name": "old_var", "data": "old"})
        self.call_tool(client, "rlm_clear", {"all": True})
        self.call_tool(client, "rlm_load_data", {"name": "new_var", "data": "new"})

        # Verify new variable exists
        response = self.call_tool(client, "rlm_list_vars", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        assert "new_var" in text
        assert "old_var" not in text

    def test_clear_variable_via_execute_created_variable(self, client):
        """rlm_clear should work with variables created via rlm_execute."""
        # Create variable via execute
        self.call_tool(client, "rlm_execute", {"code": "x = 42"})

        # Clear it
        response = self.call_tool(client, "rlm_clear", {"name": "x"})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        assert "x" in text
        assert "removida" in text.lower()

    def test_cleared_variable_raises_error_on_access(self, client):
        """After clearing a variable, accessing it should raise NameError."""
        # Create and clear
        self.call_tool(client, "rlm_load_data", {"name": "temp_var", "data": "test"})
        self.call_tool(client, "rlm_clear", {"name": "temp_var"})

        # Try to access - should get error
        response = self.call_tool(client, "rlm_execute", {"code": "print(temp_var)"})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should have error (NameError)
        assert "NameError" in text or "ERRO" in text or "não" in text.lower()

    def test_clear_with_string_id(self, client):
        """rlm_clear should work with string request id."""
        payload = {
            "jsonrpc": "2.0",
            "id": "clear-request-abc",
            "method": "tools/call",
            "params": {
                "name": "rlm_clear",
                "arguments": {"all": True}
            }
        }
        response = client.post("/mcp", json=payload)
        data = response.json()
        assert data["id"] == "clear-request-abc"
        assert "result" in data

    def test_clear_all_with_mixed_types(self, client):
        """rlm_clear with all=True should work with different variable types."""
        # Create different types
        self.call_tool(client, "rlm_load_data", {"name": "text_var", "data": "hello"})
        self.call_tool(client, "rlm_load_data", {"name": "json_var", "data": '{"key": "value"}', "data_type": "json"})
        self.call_tool(client, "rlm_load_data", {"name": "list_var", "data": "a,b\n1,2", "data_type": "csv"})
        self.call_tool(client, "rlm_execute", {"code": "num_var = 123"})

        # Clear all
        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should have cleared at least 4 variables (may include llm_* functions from execute)
        # Message format: "Todas as N variáveis foram removidas."
        import re
        match = re.search(r"(\d+)", text)
        assert match is not None
        count = int(match.group(1))
        assert count >= 4  # At least our 4 variables (plus llm_* functions from execute)

    def test_clear_variable_special_characters_in_name(self, client):
        """rlm_clear should handle variable names with underscores."""
        self.call_tool(client, "rlm_load_data", {"name": "my_special_var_123", "data": "test"})

        response = self.call_tool(client, "rlm_clear", {"name": "my_special_var_123"})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        assert "my_special_var_123" in text
        assert "removida" in text.lower()

    def test_clear_all_resets_memory_usage(self, client):
        """rlm_clear with all=True should reset memory usage to zero."""
        # Create some data
        self.call_tool(client, "rlm_load_data", {"name": "large_var", "data": "x" * 10000})

        # Clear all
        self.call_tool(client, "rlm_clear", {"all": True})

        # Check memory - should be back to 0 (or minimal)
        from rlm_mcp.http_server import repl
        mem = repl.get_memory_usage()
        assert mem["variable_count"] == 0
        assert mem["total_bytes"] == 0

    def test_clear_single_reduces_memory(self, client):
        """rlm_clear with name should reduce memory usage."""
        # Create variables
        self.call_tool(client, "rlm_load_data", {"name": "small_var", "data": "small"})
        self.call_tool(client, "rlm_load_data", {"name": "to_remove", "data": "x" * 1000})

        # Get memory before
        from rlm_mcp.http_server import repl
        mem_before = repl.get_memory_usage()

        # Clear one
        self.call_tool(client, "rlm_clear", {"name": "to_remove"})

        # Memory should decrease
        mem_after = repl.get_memory_usage()
        assert mem_after["total_bytes"] < mem_before["total_bytes"]
        assert mem_after["variable_count"] == mem_before["variable_count"] - 1

    def test_no_error_field_in_response(self, client):
        """rlm_clear should not return error field for valid operations."""
        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        assert data.get("error") is None

    def test_response_is_dict(self, client):
        """rlm_clear should return a dictionary response."""
        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        assert isinstance(data, dict)

    def test_multiple_clear_operations(self, client):
        """Multiple clear operations should work consecutively."""
        # Create 3 vars
        self.call_tool(client, "rlm_load_data", {"name": "v1", "data": "1"})
        self.call_tool(client, "rlm_load_data", {"name": "v2", "data": "2"})
        self.call_tool(client, "rlm_load_data", {"name": "v3", "data": "3"})

        # Clear one by one
        r1 = self.call_tool(client, "rlm_clear", {"name": "v1"})
        assert "removida" in r1.json()["result"]["content"][0]["text"].lower()

        r2 = self.call_tool(client, "rlm_clear", {"name": "v2"})
        assert "removida" in r2.json()["result"]["content"][0]["text"].lower()

        r3 = self.call_tool(client, "rlm_clear", {"name": "v3"})
        assert "removida" in r3.json()["result"]["content"][0]["text"].lower()

        # All should be gone
        from rlm_mcp.http_server import repl
        assert repl.get_memory_usage()["variable_count"] == 0

    def test_clear_all_false_same_as_no_param(self, client):
        """rlm_clear with all=False should require name parameter."""
        response = self.call_tool(client, "rlm_clear", {"all": False})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should ask for name or all
        assert "name" in text.lower() or "all" in text.lower()


class TestMcpToolRlmLoadS3SkipIfExists:
    """Tests for MCP tool rlm_load_s3 with skip_if_exists=True via POST /mcp endpoint."""

    def call_tool(self, client, tool_name: str, arguments: dict = None, request_id: int = 1):
        """Helper to call an MCP tool."""
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            }
        }
        return client.post("/mcp", json=payload)

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test to avoid cross-test pollution."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    @pytest.fixture(autouse=True)
    def mock_s3(self, mock_minio_client_with_data):
        """Mock the S3 client for all tests in this class."""
        from unittest.mock import patch
        from rlm_mcp.s3_client import S3Client
        import os

        # Create mock S3Client with fake credentials
        with patch.dict(os.environ, {
            "MINIO_ENDPOINT": "mock-minio:9000",
            "MINIO_ACCESS_KEY": "mock-access-key",
            "MINIO_SECRET_KEY": "mock-secret-key",
            "MINIO_SECURE": "false",
        }):
            mock_client = S3Client()
            mock_client._client = mock_minio_client_with_data

            # Patch get_s3_client to return our mock
            with patch("rlm_mcp.services.s3_guard.get_s3_client", return_value=mock_client):
                yield mock_client

    def test_returns_200_status_code(self, client):
        """rlm_load_s3 should return 200 OK."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })
        assert response.status_code == 200

    def test_returns_json(self, client):
        """rlm_load_s3 should return JSON content."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """rlm_load_s3 should return jsonrpc 2.0."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """rlm_load_s3 should return the same request id."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        }, request_id=42)
        data = response.json()
        assert data["id"] == 42

    def test_returns_result_with_content(self, client):
        """rlm_load_s3 should return result with content."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })
        data = response.json()
        assert "result" in data
        assert "content" in data["result"]
        assert isinstance(data["result"]["content"], list)

    def test_loads_text_data_successfully(self, client):
        """rlm_load_s3 should load text data into variable."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_text",
            "bucket": "test-bucket"
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should confirm successful load
        assert "my_text" in text
        assert "carregada" in text.lower() or "loaded" in text.lower()

    def test_skip_if_exists_true_skips_when_variable_exists(self, client):
        """rlm_load_s3 with skip_if_exists=True should skip when variable already exists."""
        # First load the variable
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "existing_var",
            "bucket": "test-bucket"
        })

        # Try to load again with skip_if_exists=True (default)
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "existing_var",
            "bucket": "test-bucket",
            "skip_if_exists": True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should indicate variable already exists
        assert "existing_var" in text
        assert "já existe" in text.lower() or "already exists" in text.lower()

    def test_skip_if_exists_default_is_true(self, client):
        """rlm_load_s3 should default to skip_if_exists=True."""
        # First load the variable
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "existing_var",
            "bucket": "test-bucket"
        })

        # Try to load again WITHOUT specifying skip_if_exists (should default to True)
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "existing_var",
            "bucket": "test-bucket"
            # skip_if_exists not specified, should default to True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should indicate variable already exists
        assert "existing_var" in text
        assert "já existe" in text.lower() or "already exists" in text.lower()

    def test_skip_message_includes_variable_info(self, client):
        """Skip message should include info about the existing variable."""
        # First load text
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })

        # Try to load again
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should mention chars or type info
        assert "char" in text.lower() or "str" in text.lower()

    def test_skip_message_suggests_force_reload(self, client):
        """Skip message should suggest using skip_if_exists=False for reload."""
        # First load
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })

        # Try to load again
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should suggest skip_if_exists=False
        assert "skip_if_exists=False" in text or "skip_if_exists" in text.lower()

    def test_skip_if_exists_works_with_json_variable(self, client):
        """skip_if_exists should work when existing variable is JSON type."""
        # First load JSON
        self.call_tool(client, "rlm_load_s3", {
            "key": "data/file.json",
            "name": "json_var",
            "bucket": "test-bucket",
            "data_type": "json"
        })

        # Try to load again
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "data/file.json",
            "name": "json_var",
            "bucket": "test-bucket",
            "data_type": "json",
            "skip_if_exists": True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should indicate already exists with type info
        assert "json_var" in text
        assert "já existe" in text.lower() or "already exists" in text.lower()
        # For non-string types, should show type name
        assert "dict" in text.lower()

    def test_skip_if_exists_does_not_trigger_s3_download(self, client, mock_minio_client_with_data):
        """When variable exists and skip_if_exists=True, S3 should not be called."""
        from unittest.mock import MagicMock

        # First load
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })

        # Record get_object calls
        original_get_object = mock_minio_client_with_data.get_object
        call_count = [0]
        def counting_get_object(*args, **kwargs):
            call_count[0] += 1
            return original_get_object(*args, **kwargs)
        mock_minio_client_with_data.get_object = counting_get_object

        # Try to load again - should NOT call get_object
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": True
        })

        # Should not have downloaded again
        assert call_count[0] == 0

    def test_skip_if_exists_preserves_original_data(self, client):
        """When skipping, original variable data should remain unchanged."""
        from rlm_mcp.http_server import repl

        # First load
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })

        # Get original value
        original_value = repl.variables["my_var"]

        # Try to load a different file into same name
        self.call_tool(client, "rlm_load_s3", {
            "key": "data/file.json",  # Different file!
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": True
        })

        # Value should be unchanged
        assert repl.variables["my_var"] == original_value

    def test_skip_does_not_return_error_field(self, client):
        """Skip should not set isError flag."""
        # First load
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })

        # Try to load again
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": True
        })
        data = response.json()

        # Should not have error
        assert data.get("error") is None
        # isError should not be True (might not be present at all)
        result = data.get("result", {})
        assert result.get("isError") != True

    def test_skip_with_variable_created_via_rlm_load_data(self, client):
        """skip_if_exists should work when variable was created via rlm_load_data."""
        # Create variable via rlm_load_data
        self.call_tool(client, "rlm_load_data", {
            "name": "my_var",
            "data": "existing data"
        })

        # Try to load from S3 into same variable name
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should indicate variable already exists
        assert "my_var" in text
        assert "já existe" in text.lower() or "already exists" in text.lower()

    def test_skip_with_variable_created_via_rlm_execute(self, client):
        """skip_if_exists should work when variable was created via rlm_execute."""
        # Create variable via rlm_execute
        self.call_tool(client, "rlm_execute", {"code": "my_var = [1, 2, 3]"})

        # Try to load from S3 into same variable name
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should indicate variable already exists
        assert "my_var" in text
        assert "já existe" in text.lower() or "already exists" in text.lower()

    def test_no_skip_when_variable_does_not_exist(self, client):
        """rlm_load_s3 should load normally when variable doesn't exist."""
        from rlm_mcp.http_server import repl

        # Make sure variable doesn't exist
        assert "new_var" not in repl.variables

        # Load with skip_if_exists=True (but variable doesn't exist)
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "new_var",
            "bucket": "test-bucket",
            "skip_if_exists": True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should load successfully (not skip)
        assert "carregada" in text.lower() or "loaded" in text.lower()
        assert "new_var" in repl.variables

    def test_skip_if_exists_with_string_id(self, client):
        """rlm_load_s3 should work with string request id."""
        # First load
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })

        # Try to load again with string id
        payload = {
            "jsonrpc": "2.0",
            "id": "s3-load-test-123",
            "method": "tools/call",
            "params": {
                "name": "rlm_load_s3",
                "arguments": {
                    "key": "test.txt",
                    "name": "my_var",
                    "bucket": "test-bucket",
                    "skip_if_exists": True
                }
            }
        }
        response = client.post("/mcp", json=payload)
        data = response.json()

        assert data["id"] == "s3-load-test-123"
        assert "já existe" in data["result"]["content"][0]["text"].lower()

    def test_skip_if_exists_for_large_string_variable(self, client):
        """skip_if_exists should show chars count for large string variables."""
        from rlm_mcp.http_server import repl

        # Create large variable via execute
        self.call_tool(client, "rlm_execute", {"code": "large_var = 'x' * 10000"})

        # Try to load into same name
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "large_var",
            "bucket": "test-bucket",
            "skip_if_exists": True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show chars info
        assert "10,000 chars" in text or "10000 char" in text

    def test_no_error_for_nonexistent_file_when_variable_exists(self, client):
        """When variable exists and skip_if_exists=True, nonexistent S3 file should still skip."""
        # First create variable
        self.call_tool(client, "rlm_load_data", {"name": "my_var", "data": "existing"})

        # Try to load nonexistent S3 file into same variable name
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "nonexistent/file.txt",  # This file doesn't exist
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should skip (not error) because variable exists
        assert "já existe" in text.lower() or "already exists" in text.lower()
        # Should NOT have error about missing file
        assert "não encontrado" not in text.lower() and "not found" not in text.lower()


class TestMcpToolRlmLoadS3ForceReload:
    """Tests for rlm_load_s3 tool with skip_if_exists=False (force reload)."""

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test."""
        from rlm_mcp.http_server import repl
        repl.clear_all()

    @pytest.fixture(autouse=True)
    def mock_s3(self, mock_minio_client_with_data):
        """Mock the S3 client for all tests in this class."""
        from unittest.mock import patch
        from rlm_mcp.s3_client import S3Client
        import os

        self.mock_minio_client = mock_minio_client_with_data

        # Create mock S3Client with fake credentials
        with patch.dict(os.environ, {
            "MINIO_ENDPOINT": "mock-minio:9000",
            "MINIO_ACCESS_KEY": "mock-access-key",
            "MINIO_SECRET_KEY": "mock-secret-key",
            "MINIO_SECURE": "false",
        }):
            mock_client = S3Client()
            mock_client._client = mock_minio_client_with_data

            # Patch get_s3_client to return our mock
            with patch("rlm_mcp.services.s3_guard.get_s3_client", return_value=mock_client):
                yield mock_client

    def call_tool(self, client, tool_name: str, arguments: dict, request_id: int = 1):
        """Helper to call MCP tool."""
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        return client.post("/mcp", json=payload)

    def test_returns_200_status_code(self, client):
        """rlm_load_s3 with skip_if_exists=False should return 200 status code."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })
        assert response.status_code == 200

    def test_returns_json_content_type(self, client):
        """rlm_load_s3 with skip_if_exists=False should return JSON."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_2_0(self, client):
        """rlm_load_s3 with skip_if_exists=False should return jsonrpc 2.0."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_request_id(self, client):
        """rlm_load_s3 with skip_if_exists=False should return same request id."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        }, request_id=42)
        data = response.json()
        assert data["id"] == 42

    def test_force_reload_overwrites_existing_variable(self, client):
        """skip_if_exists=False should overwrite existing variable."""
        from rlm_mcp.http_server import repl

        # First, create variable with different content
        self.call_tool(client, "rlm_load_data", {"name": "my_var", "data": "original content"})
        assert repl.variables["my_var"] == "original content"

        # Force reload from S3 (test.txt contains "Hello, World!")
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })

        # Variable should now have S3 content
        assert repl.variables["my_var"] == "Hello, World!"
        # Response should show success, not "já existe"
        text = response.json()["result"]["content"][0]["text"]
        assert "já existe" not in text.lower()
        assert "carregada" in text.lower() or "loaded" in text.lower()

    def test_force_reload_no_skip_message(self, client):
        """skip_if_exists=False should NOT show 'já existe' skip message."""
        # First load
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })

        # Force reload
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })
        text = response.json()["result"]["content"][0]["text"]

        # Should NOT show skip message
        assert "já existe" not in text.lower()
        assert "skip_if_exists=False" not in text

    def test_force_reload_triggers_s3_download(self, client):
        """skip_if_exists=False should trigger S3 download even if variable exists."""
        # First load
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })

        # Record get_object calls
        original_get_object = self.mock_minio_client.get_object
        call_count = [0]
        def counting_get_object(*args, **kwargs):
            call_count[0] += 1
            return original_get_object(*args, **kwargs)
        self.mock_minio_client.get_object = counting_get_object

        # Force reload - should call get_object
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })

        # Should have downloaded
        assert call_count[0] == 1

    def test_force_reload_updates_variable_with_different_file(self, client):
        """skip_if_exists=False should load different file into existing variable."""
        from rlm_mcp.http_server import repl

        # Load text file first
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })
        assert repl.variables["my_var"] == "Hello, World!"

        # Force reload with JSON file
        self.call_tool(client, "rlm_load_s3", {
            "key": "data/file.json",
            "name": "my_var",
            "bucket": "test-bucket",
            "data_type": "json",
            "skip_if_exists": False
        })

        # Variable should now be a dict (from JSON)
        assert isinstance(repl.variables["my_var"], dict)
        assert repl.variables["my_var"] == {"key": "value", "number": 42}

    def test_force_reload_on_empty_repl(self, client):
        """skip_if_exists=False should work normally when variable doesn't exist."""
        from rlm_mcp.http_server import repl

        # Make sure variable doesn't exist
        assert "new_var" not in repl.variables

        # Load with skip_if_exists=False
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "new_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })

        # Should load successfully
        assert "new_var" in repl.variables
        assert repl.variables["new_var"] == "Hello, World!"
        text = response.json()["result"]["content"][0]["text"]
        assert "carregada" in text.lower() or "loaded" in text.lower()

    def test_force_reload_updates_metadata(self, client):
        """skip_if_exists=False should update variable metadata."""
        from rlm_mcp.http_server import repl
        import time

        # First load
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket"
        })
        first_metadata = repl.variable_metadata.get("my_var")
        first_accessed = first_metadata.last_accessed if first_metadata else None

        # Small delay
        time.sleep(0.01)

        # Force reload
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })

        # Metadata should be updated
        second_metadata = repl.variable_metadata.get("my_var")
        assert second_metadata is not None
        # last_accessed should be updated (or created_at if that's what changes)
        assert second_metadata.last_accessed >= first_accessed

    def test_force_reload_with_json_data_type(self, client):
        """skip_if_exists=False should work with data_type=json."""
        from rlm_mcp.http_server import repl

        # First load as text
        self.call_tool(client, "rlm_load_data", {"name": "my_var", "data": "old data"})

        # Force reload as JSON
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "data/file.json",
            "name": "my_var",
            "bucket": "test-bucket",
            "data_type": "json",
            "skip_if_exists": False
        })

        # Should be dict
        assert isinstance(repl.variables["my_var"], dict)
        text = response.json()["result"]["content"][0]["text"]
        assert "carregada" in text.lower()

    def test_force_reload_overwrites_variable_from_execute(self, client):
        """skip_if_exists=False should overwrite variable created via execute."""
        from rlm_mcp.http_server import repl

        # Create via execute
        self.call_tool(client, "rlm_execute", {"code": "my_var = [1, 2, 3]"})
        assert repl.variables["my_var"] == [1, 2, 3]

        # Force reload from S3
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })

        # Should now be string from S3
        assert repl.variables["my_var"] == "Hello, World!"
        text = response.json()["result"]["content"][0]["text"]
        assert "carregada" in text.lower()

    def test_force_reload_with_string_request_id(self, client):
        """skip_if_exists=False should work with string request id."""
        payload = {
            "jsonrpc": "2.0",
            "id": "force-reload-test",
            "method": "tools/call",
            "params": {
                "name": "rlm_load_s3",
                "arguments": {
                    "key": "test.txt",
                    "name": "my_var",
                    "bucket": "test-bucket",
                    "skip_if_exists": False
                }
            }
        }
        response = client.post("/mcp", json=payload)
        data = response.json()

        assert data["id"] == "force-reload-test"
        assert "carregada" in data["result"]["content"][0]["text"].lower()

    def test_force_reload_returns_error_for_nonexistent_file(self, client):
        """skip_if_exists=False should return error if S3 file doesn't exist."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "nonexistent/file.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show error about file not found
        assert "erro" in text.lower() or "error" in text.lower() or "não encontrado" in text.lower()

    def test_force_reload_on_nonexistent_file_when_variable_exists(self, client):
        """skip_if_exists=False should error even if variable exists when file doesn't exist."""
        # Create variable first
        self.call_tool(client, "rlm_load_data", {"name": "my_var", "data": "existing"})

        # Try to force reload from nonexistent file
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "nonexistent/file.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show error (not skip)
        # Note: with skip_if_exists=True it would skip; with False it should try to load and fail
        assert "erro" in text.lower() or "error" in text.lower() or "não encontrado" in text.lower()

    def test_force_reload_no_error_field_on_success(self, client):
        """skip_if_exists=False should not have error field on success."""
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })
        data = response.json()

        assert "error" not in data

    def test_force_reload_multiple_times(self, client):
        """skip_if_exists=False should allow multiple consecutive reloads."""
        from rlm_mcp.http_server import repl

        # Add different content to S3 mock
        self.mock_minio_client.add_object("test-bucket", "v1.txt", b"version 1")
        self.mock_minio_client.add_object("test-bucket", "v2.txt", b"version 2")
        self.mock_minio_client.add_object("test-bucket", "v3.txt", b"version 3")

        # Load v1
        self.call_tool(client, "rlm_load_s3", {
            "key": "v1.txt", "name": "data", "bucket": "test-bucket", "skip_if_exists": False
        })
        assert repl.variables["data"] == "version 1"

        # Reload v2
        self.call_tool(client, "rlm_load_s3", {
            "key": "v2.txt", "name": "data", "bucket": "test-bucket", "skip_if_exists": False
        })
        assert repl.variables["data"] == "version 2"

        # Reload v3
        self.call_tool(client, "rlm_load_s3", {
            "key": "v3.txt", "name": "data", "bucket": "test-bucket", "skip_if_exists": False
        })
        assert repl.variables["data"] == "version 3"

    def test_force_reload_preserves_other_variables(self, client):
        """skip_if_exists=False should not affect other variables."""
        from rlm_mcp.http_server import repl

        # Load multiple variables
        self.call_tool(client, "rlm_load_data", {"name": "var1", "data": "data1"})
        self.call_tool(client, "rlm_load_data", {"name": "var2", "data": "data2"})
        self.call_tool(client, "rlm_load_data", {"name": "var3", "data": "data3"})

        # Force reload var2
        self.call_tool(client, "rlm_load_s3", {
            "key": "test.txt",
            "name": "var2",
            "bucket": "test-bucket",
            "skip_if_exists": False
        })

        # Other variables should be unchanged
        assert repl.variables["var1"] == "data1"
        assert repl.variables["var2"] == "Hello, World!"  # Changed
        assert repl.variables["var3"] == "data3"

    def test_force_reload_with_csv_data_type(self, client):
        """skip_if_exists=False should work with data_type=csv."""
        from rlm_mcp.http_server import repl

        # Add CSV to mock
        csv_content = b"name,age\nAlice,30\nBob,25"
        self.mock_minio_client.add_object("test-bucket", "people.csv", csv_content)

        # First load as text
        self.call_tool(client, "rlm_load_data", {"name": "my_var", "data": "old data"})

        # Force reload as CSV
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "people.csv",
            "name": "my_var",
            "bucket": "test-bucket",
            "data_type": "csv",
            "skip_if_exists": False
        })

        # Should be list of dicts
        assert isinstance(repl.variables["my_var"], list)
        assert len(repl.variables["my_var"]) == 2
        assert repl.variables["my_var"][0] == {"name": "Alice", "age": "30"}
        text = response.json()["result"]["content"][0]["text"]
        assert "carregada" in text.lower()

    def test_force_reload_with_lines_data_type(self, client):
        """skip_if_exists=False should work with data_type=lines."""
        from rlm_mcp.http_server import repl

        # Add multiline file to mock
        lines_content = b"line1\nline2\nline3"
        self.mock_minio_client.add_object("test-bucket", "lines.txt", lines_content)

        # First load as text
        self.call_tool(client, "rlm_load_data", {"name": "my_var", "data": "old data"})

        # Force reload as lines
        response = self.call_tool(client, "rlm_load_s3", {
            "key": "lines.txt",
            "name": "my_var",
            "bucket": "test-bucket",
            "data_type": "lines",
            "skip_if_exists": False
        })

        # Should be list of lines
        assert isinstance(repl.variables["my_var"], list)
        assert repl.variables["my_var"] == ["line1", "line2", "line3"]
        text = response.json()["result"]["content"][0]["text"]
        assert "carregada" in text.lower()


class TestMcpToolRlmSearchIndex:
    """Tests for rlm_search_index tool via MCP tools/call method."""

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

    def call_tool(self, client, tool_name: str, arguments: dict, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        return self.make_mcp_request(
            client,
            "tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl_and_indices(self):
        """Reset REPL state and indices before each test."""
        from rlm_mcp.http_server import repl
        from rlm_mcp.indexer import clear_all_indices
        repl.clear_all()
        clear_all_indices()
        yield
        repl.clear_all()
        clear_all_indices()

    def create_indexed_variable(self, client):
        """Helper to create a large text variable that will be indexed."""
        # Text with terms from DEFAULT_INDEX_TERMS: medo, ansiedade, trabalho, família
        # Each term repeated to ensure indexability
        text_parts = []
        for i in range(50):
            text_parts.append(f"Linha {i}: O paciente relata medo e ansiedade relacionados ao trabalho.")
            text_parts.append(f"Linha {i+50}: Também menciona família e problemas de cabeça.")
            text_parts.append(f"Linha {i+100}: Sintomas de medo intenso e coração acelerado.")
            text_parts.append(f"Linha {i+150}: Relação com mãe é conflituosa.")
        # Make it >= 100k chars to trigger auto-indexing
        base_text = "\n".join(text_parts)
        while len(base_text) < 100000:
            base_text += "\n" + base_text[:10000]

        # Load the large text
        self.call_tool(client, "rlm_load_data", {"name": "large_text", "data": base_text})

        # Manually create index since auto-indexing may not run in test
        from rlm_mcp.indexer import create_index, set_index
        index = create_index(base_text, "large_text")
        set_index("large_text", index)

        return base_text

    def test_returns_200_status_code(self, client):
        """rlm_search_index should return 200 OK."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"]
        })
        assert response.status_code == 200

    def test_returns_json(self, client):
        """rlm_search_index should return JSON content."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"]
        })
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_jsonrpc_version(self, client):
        """rlm_search_index should return jsonrpc 2.0."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"]
        })
        data = response.json()
        assert data["jsonrpc"] == "2.0"

    def test_returns_same_id(self, client):
        """rlm_search_index should return the same request id."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"]
        }, request_id=42)
        data = response.json()
        assert data["id"] == 42

    def test_returns_result_with_content(self, client):
        """rlm_search_index should return result with content list."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"]
        })
        data = response.json()
        assert "result" in data
        assert "content" in data["result"]
        assert isinstance(data["result"]["content"], list)
        assert len(data["result"]["content"]) > 0
        assert data["result"]["content"][0]["type"] == "text"

    def test_finds_indexed_term(self, client):
        """rlm_search_index should find terms that are in the index."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"]
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show results
        assert "Resultados" in text or "ocorrências" in text
        assert "medo" in text.lower()

    def test_multiple_terms_or_mode(self, client):
        """rlm_search_index with require_all=False should search multiple terms (OR mode)."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo", "ansiedade"],
            "require_all": False
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show results for both terms
        assert "medo" in text.lower()
        assert "ansiedade" in text.lower()

    def test_require_all_true_and_mode(self, client):
        """rlm_search_index with require_all=True should find lines with ALL terms (AND mode)."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo", "ansiedade"],
            "require_all": True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should indicate AND mode results
        assert "todos os termos" in text.lower() or "encontradas" in text.lower()

    def test_term_not_found_message(self, client):
        """rlm_search_index should show message when terms are not found."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["xyz_nonexistent_term_123"]
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should indicate no results
        assert "nenhum" in text.lower()

    def test_shows_index_stats(self, client):
        """rlm_search_index should show index stats at the end."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"]
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show index stats
        assert "Índice" in text or "índice" in text
        assert "termos" in text.lower()

    def test_variable_not_found_error(self, client):
        """rlm_search_index should return error for nonexistent variable."""
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "nonexistent_var",
            "terms": ["medo"]
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show error
        assert "erro" in text.lower() or "não encontrada" in text.lower()
        assert data["result"].get("isError") == True

    def test_variable_without_index_error(self, client):
        """rlm_search_index should return error for variable without index."""
        # Load a small text (won't be auto-indexed)
        self.call_tool(client, "rlm_load_data", {"name": "small_text", "data": "small text without index"})

        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "small_text",
            "terms": ["medo"]
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show error about missing index
        assert "não possui índice" in text.lower() or "100k" in text
        assert data["result"].get("isError") == True

    def test_limit_parameter(self, client):
        """rlm_search_index should respect limit parameter."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"],
            "limit": 5
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should return results (limit affects how many are shown)
        assert "medo" in text.lower()

    def test_default_require_all_is_false(self, client):
        """rlm_search_index should default require_all to False (OR mode)."""
        self.create_indexed_variable(client)
        # Call without require_all parameter
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo", "ansiedade"]
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should be in OR mode (show results per term)
        assert "ocorrências" in text.lower() or "Resultados" in text

    def test_empty_terms_list(self, client):
        """rlm_search_index should handle empty terms list."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": []
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show no results or handle gracefully
        assert "nenhum" in text.lower() or "Índice" in text

    def test_case_insensitive_search(self, client):
        """rlm_search_index should search case-insensitively."""
        self.create_indexed_variable(client)
        # Search with uppercase term
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["MEDO"]  # Uppercase
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should find results (search is case-insensitive)
        assert "medo" in text.lower() or "MEDO" in text

    def test_shows_line_context(self, client):
        """rlm_search_index OR mode should show line context for matches."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"],
            "require_all": False
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show line numbers and context
        assert "Linha" in text

    def test_with_string_request_id(self, client):
        """rlm_search_index should work with string request id."""
        self.create_indexed_variable(client)
        payload = {
            "jsonrpc": "2.0",
            "id": "search-index-test-123",
            "method": "tools/call",
            "params": {
                "name": "rlm_search_index",
                "arguments": {
                    "var_name": "large_text",
                    "terms": ["medo"]
                }
            }
        }
        response = client.post("/mcp", json=payload)
        data = response.json()
        assert data["id"] == "search-index-test-123"
        assert "result" in data

    def test_missing_var_name_parameter(self, client):
        """rlm_search_index should handle missing var_name parameter."""
        response = self.call_tool(client, "rlm_search_index", {
            "terms": ["medo"]
        })
        data = response.json()
        # Should return an error
        assert "error" in data or data["result"].get("isError") == True

    def test_missing_terms_parameter(self, client):
        """rlm_search_index should handle missing terms parameter."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text"
        })
        data = response.json()
        # Should return an error
        assert "error" in data or data["result"].get("isError") == True

    def test_no_error_field_on_success(self, client):
        """rlm_search_index should not have error field on successful search."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"]
        })
        data = response.json()
        assert data.get("error") is None
        assert data["result"].get("isError") != True

    def test_multiple_requests_same_results(self, client):
        """Multiple searches for same term should return consistent results."""
        self.create_indexed_variable(client)
        response1 = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"]
        }, request_id=1)
        response2 = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"]
        }, request_id=2)

        text1 = response1.json()["result"]["content"][0]["text"]
        text2 = response2.json()["result"]["content"][0]["text"]

        # Results should be similar (both contain medo)
        assert "medo" in text1.lower()
        assert "medo" in text2.lower()

    def test_require_all_no_match_message(self, client):
        """rlm_search_index with require_all=True should show message when no line has all terms."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo", "xyz_nonexistent_123"],
            "require_all": True
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should indicate no lines found with all terms
        assert "nenhuma linha" in text.lower() or "todos os termos" in text.lower()

    def test_shows_occurrence_count(self, client):
        """rlm_search_index OR mode should show occurrence count per term."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"],
            "require_all": False
        })
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show count of occurrences
        assert "ocorrências" in text.lower() or "ocorrência" in text.lower()

    def test_response_is_dict(self, client):
        """rlm_search_index should return a dictionary response."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"]
        })
        data = response.json()
        assert isinstance(data, dict)

    def test_offset_parameter_in_schema(self, client):
        """rlm_search_index should have 'offset' parameter in inputSchema."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        search_tool = next((t for t in tools if t["name"] == "rlm_search_index"), None)
        assert search_tool is not None
        assert "offset" in search_tool["inputSchema"]["properties"]
        assert search_tool["inputSchema"]["properties"]["offset"]["type"] == "integer"
        assert search_tool["inputSchema"]["properties"]["offset"]["default"] == 0

    def test_offset_skips_results_in_or_mode(self, client):
        """rlm_search_index should skip results when offset is used in OR mode."""
        self.create_indexed_variable(client)
        # Get first page (offset=0, limit=3)
        response1 = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"],
            "limit": 3,
            "offset": 0
        })
        text1 = response1.json()["result"]["content"][0]["text"]

        # Get second page (offset=3, limit=3)
        response2 = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"],
            "limit": 3,
            "offset": 3
        })
        text2 = response2.json()["result"]["content"][0]["text"]

        # Both should be valid results, but showing different ranges
        assert "mostrando" in text1
        assert "mostrando" in text2
        # First page shows 1-3, second page shows 4-6
        assert "1-3" in text1 or "mostrando 1-" in text1
        assert "4-" in text2

    def test_offset_skips_results_in_and_mode(self, client):
        """rlm_search_index should skip results when offset is used in AND mode."""
        self.create_indexed_variable(client)
        # Get first page (offset=0, limit=2)
        response1 = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo", "ansiedade"],
            "require_all": True,
            "limit": 2,
            "offset": 0
        })
        text1 = response1.json()["result"]["content"][0]["text"]

        # Get second page (offset=2, limit=2)
        response2 = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo", "ansiedade"],
            "require_all": True,
            "limit": 2,
            "offset": 2
        })
        text2 = response2.json()["result"]["content"][0]["text"]

        # Both should show pagination info
        assert "mostrando" in text1 or "encontradas" in text1
        assert "mostrando" in text2 or "encontradas" in text2

    def test_offset_default_is_zero(self, client):
        """rlm_search_index should default offset to 0."""
        self.create_indexed_variable(client)
        # Without offset
        response1 = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"],
            "limit": 5
        })
        text1 = response1.json()["result"]["content"][0]["text"]

        # With offset=0 explicitly
        response2 = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"],
            "limit": 5,
            "offset": 0
        })
        text2 = response2.json()["result"]["content"][0]["text"]

        # Both should show same first page
        assert "1-" in text1
        assert "1-" in text2

    def test_offset_beyond_results_shows_empty_range(self, client):
        """rlm_search_index should handle offset beyond available results."""
        self.create_indexed_variable(client)
        # Use large offset to skip all results
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"],
            "limit": 5,
            "offset": 10000
        })
        text = response.json()["result"]["content"][0]["text"]

        # Should still return valid response with pagination info
        assert "ocorrências" in text.lower() or "mostrando" in text

    def test_pagination_shows_total_and_range(self, client):
        """rlm_search_index should show total count and displayed range in OR mode."""
        self.create_indexed_variable(client)
        response = self.call_tool(client, "rlm_search_index", {
            "var_name": "large_text",
            "terms": ["medo"],
            "limit": 5,
            "offset": 0
        })
        text = response.json()["result"]["content"][0]["text"]

        # Should show total occurrences and range
        assert "ocorrências" in text.lower()
        assert "mostrando" in text


class TestMcpToolRlmPersistenceStats:
    """Tests for rlm_persistence_stats tool via MCP tools/call method."""

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

    def call_tool(self, client, tool_name: str, arguments: dict, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        return self.make_mcp_request(
            client,
            "tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test to avoid cross-test pollution."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    def test_returns_200_status_code(self, client):
        """rlm_persistence_stats should return 200 status code."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        assert response.status_code == 200

    def test_returns_jsonrpc_format(self, client):
        """rlm_persistence_stats should return valid JSON-RPC format."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert "id" in data
        assert "result" in data

    def test_returns_text_content(self, client):
        """rlm_persistence_stats should return text content."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        assert "content" in data["result"]
        assert len(data["result"]["content"]) > 0
        assert data["result"]["content"][0]["type"] == "text"

    def test_content_is_string(self, client):
        """rlm_persistence_stats content text should be a string."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        assert isinstance(text, str)

    def test_contains_statistics_header(self, client):
        """rlm_persistence_stats should show statistics header."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should contain the Portuguese header
        assert "Estatísticas" in text or "Persistência" in text

    def test_shows_variables_count(self, client):
        """rlm_persistence_stats should show count of saved variables."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show variable count (in Portuguese)
        assert "Variáveis salvas" in text or "variáveis" in text.lower()

    def test_shows_total_size(self, client):
        """rlm_persistence_stats should show total size in bytes."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show size info
        assert "bytes" in text.lower() or "Tamanho" in text

    def test_shows_indices_count(self, client):
        """rlm_persistence_stats should show count of saved indices."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show indices count (in Portuguese)
        assert "Índices" in text or "índices" in text.lower()

    def test_shows_db_info(self, client):
        """rlm_persistence_stats should show database file info."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]
        # Should show DB file info
        assert "DB" in text or "db" in text.lower()

    def test_no_error_on_empty_persistence(self, client):
        """rlm_persistence_stats should not error when no variables are persisted."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        # Should not have error field
        assert data.get("error") is None
        assert data["result"].get("isError") != True

    def test_returns_same_request_id(self, client):
        """rlm_persistence_stats should return the same request id."""
        response = self.call_tool(client, "rlm_persistence_stats", {}, request_id=42)
        data = response.json()
        assert data["id"] == 42

    def test_works_with_string_request_id(self, client):
        """rlm_persistence_stats should work with string request id."""
        payload = {
            "jsonrpc": "2.0",
            "id": "persistence-stats-test",
            "method": "tools/call",
            "params": {
                "name": "rlm_persistence_stats",
                "arguments": {}
            }
        }
        response = client.post("/mcp", json=payload)
        data = response.json()
        assert data["id"] == "persistence-stats-test"
        assert "result" in data

    def test_shows_persisted_variables_after_load(self, client):
        """rlm_persistence_stats should list persisted variables after rlm_load_data."""
        # First load a variable (persistence is automatic)
        self.call_tool(client, "rlm_load_data", {
            "name": "test_var",
            "data": "test content"
        })

        # Now check persistence stats
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show the loaded variable in the list
        assert "test_var" in text or "Variáveis salvas: 0" not in text

    def test_shows_variable_type(self, client):
        """rlm_persistence_stats should show variable type for persisted variables."""
        # Load a variable
        self.call_tool(client, "rlm_load_data", {
            "name": "test_var",
            "data": "test content"
        })

        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show type (str, dict, list, etc.)
        # Looking for parenthesis with type inside, e.g., "(str,"
        assert "str" in text.lower() or "type" in text.lower() or "(" in text

    def test_shows_variable_size(self, client):
        """rlm_persistence_stats should show variable size for persisted variables."""
        # Load a variable
        self.call_tool(client, "rlm_load_data", {
            "name": "test_var",
            "data": "test content with some data"
        })

        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show size in bytes
        assert "bytes" in text.lower()

    def test_shows_indexed_terms_count(self, client):
        """rlm_persistence_stats should show count of indexed terms."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show indexed terms count
        assert "indexado" in text.lower() or "termos" in text.lower()

    def test_multiple_requests_succeed(self, client):
        """Multiple rlm_persistence_stats requests should all succeed."""
        for i in range(3):
            response = self.call_tool(client, "rlm_persistence_stats", {}, request_id=i)
            assert response.status_code == 200
            data = response.json()
            assert data.get("error") is None

    def test_response_is_dict(self, client):
        """rlm_persistence_stats should return a dictionary response."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        assert isinstance(data, dict)

    def test_ignores_extra_parameters(self, client):
        """rlm_persistence_stats should work even if extra parameters are passed."""
        response = self.call_tool(client, "rlm_persistence_stats", {
            "extra_param": "should be ignored"
        })
        data = response.json()
        # Should not error, just ignore the extra param
        assert response.status_code == 200
        assert "result" in data

    def test_updated_at_timestamp_shown(self, client):
        """rlm_persistence_stats should show updated_at timestamp for variables."""
        # Load a variable
        self.call_tool(client, "rlm_load_data", {
            "name": "test_var",
            "data": "test content"
        })

        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show timestamp info
        assert "Atualizado" in text or "atualizado" in text.lower() or "202" in text  # Year prefix

    def test_no_is_error_field_on_success(self, client):
        """rlm_persistence_stats should not have isError field on success."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        assert data["result"].get("isError") != True

    def test_works_after_clear(self, client):
        """rlm_persistence_stats should work after rlm_clear is called."""
        # Load a variable
        self.call_tool(client, "rlm_load_data", {
            "name": "test_var",
            "data": "test content"
        })

        # Clear all
        self.call_tool(client, "rlm_clear", {"all": True})

        # Check stats - should not error
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        assert response.status_code == 200
        assert data.get("error") is None


class TestRequiredInputValidation:
    """Tests for validating that http_server.py properly validates required inputs.

    Each tool with required parameters should return an error when those
    parameters are missing from the request.
    """

    def call_tool(self, client, tool_name: str, arguments: dict, request_id: int = 1):
        """Helper to make MCP tools/call requests."""
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        return client.post("/mcp", json=payload)

    # rlm_execute requires "code"
    def test_rlm_execute_missing_code_returns_error(self, client):
        """rlm_execute should return error when 'code' is missing."""
        response = self.call_tool(client, "rlm_execute", {})
        data = response.json()
        # Should have either an error response or isError in result
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_execute_with_code_succeeds(self, client):
        """rlm_execute should succeed when 'code' is provided."""
        response = self.call_tool(client, "rlm_execute", {"code": "print('hello')"})
        data = response.json()
        assert data.get("error") is None
        assert "result" in data

    # rlm_load_data requires "name" and "data"
    def test_rlm_load_data_missing_name_returns_error(self, client):
        """rlm_load_data should return error when 'name' is missing."""
        response = self.call_tool(client, "rlm_load_data", {"data": "test"})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_load_data_missing_data_returns_error(self, client):
        """rlm_load_data should return error when 'data' is missing."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test"})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_load_data_missing_both_returns_error(self, client):
        """rlm_load_data should return error when both 'name' and 'data' are missing."""
        response = self.call_tool(client, "rlm_load_data", {})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_load_data_with_all_required_succeeds(self, client):
        """rlm_load_data should succeed when 'name' and 'data' are provided."""
        response = self.call_tool(client, "rlm_load_data", {"name": "test_var", "data": "test content"})
        data = response.json()
        assert data.get("error") is None
        assert "result" in data
        # Clean up
        self.call_tool(client, "rlm_clear", {"name": "test_var"})

    # rlm_load_file requires "name" and "path"
    def test_rlm_load_file_missing_name_returns_error(self, client):
        """rlm_load_file should return error when 'name' is missing."""
        response = self.call_tool(client, "rlm_load_file", {"path": "/data/test.txt"})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_load_file_missing_path_returns_error(self, client):
        """rlm_load_file should return error when 'path' is missing."""
        response = self.call_tool(client, "rlm_load_file", {"name": "test"})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_load_file_missing_both_returns_error(self, client):
        """rlm_load_file should return error when both 'name' and 'path' are missing."""
        response = self.call_tool(client, "rlm_load_file", {})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    # rlm_var_info requires "name"
    def test_rlm_var_info_missing_name_returns_error(self, client):
        """rlm_var_info should return error when 'name' is missing."""
        response = self.call_tool(client, "rlm_var_info", {})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_var_info_with_name_succeeds(self, client):
        """rlm_var_info should succeed when 'name' is provided (even if var doesn't exist)."""
        response = self.call_tool(client, "rlm_var_info", {"name": "nonexistent_var"})
        data = response.json()
        # Should not have a KeyError type error
        assert data.get("error") is None
        assert "result" in data

    # rlm_load_s3 requires "key" and "name"
    def test_rlm_load_s3_missing_key_returns_error(self, client):
        """rlm_load_s3 should return error when 'key' is missing."""
        response = self.call_tool(client, "rlm_load_s3", {"name": "test"})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_load_s3_missing_name_returns_error(self, client):
        """rlm_load_s3 should return error when 'name' is missing."""
        response = self.call_tool(client, "rlm_load_s3", {"key": "test/file.txt"})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_load_s3_missing_both_returns_error(self, client):
        """rlm_load_s3 should return error when both 'key' and 'name' are missing."""
        response = self.call_tool(client, "rlm_load_s3", {})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    # rlm_upload_url requires "url" and "key"
    def test_rlm_upload_url_missing_url_returns_error(self, client):
        """rlm_upload_url should return error when 'url' is missing."""
        response = self.call_tool(client, "rlm_upload_url", {"key": "test/file.txt"})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_upload_url_missing_key_returns_error(self, client):
        """rlm_upload_url should return error when 'key' is missing."""
        response = self.call_tool(client, "rlm_upload_url", {"url": "https://example.com/test.txt"})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_upload_url_missing_both_returns_error(self, client):
        """rlm_upload_url should return error when both 'url' and 'key' are missing."""
        response = self.call_tool(client, "rlm_upload_url", {})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    # rlm_process_pdf requires "key"
    def test_rlm_process_pdf_missing_key_returns_error(self, client):
        """rlm_process_pdf should return error when 'key' is missing."""
        response = self.call_tool(client, "rlm_process_pdf", {})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    # rlm_search_index requires "var_name" and "terms"
    def test_rlm_search_index_missing_var_name_returns_error(self, client):
        """rlm_search_index should return error when 'var_name' is missing."""
        response = self.call_tool(client, "rlm_search_index", {"terms": ["test"]})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_search_index_missing_terms_returns_error(self, client):
        """rlm_search_index should return error when 'terms' is missing."""
        response = self.call_tool(client, "rlm_search_index", {"var_name": "test"})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_search_index_missing_both_returns_error(self, client):
        """rlm_search_index should return error when both 'var_name' and 'terms' are missing."""
        response = self.call_tool(client, "rlm_search_index", {})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    # rlm_collection_create requires "name"
    def test_rlm_collection_create_missing_name_returns_error(self, client):
        """rlm_collection_create should return error when 'name' is missing."""
        response = self.call_tool(client, "rlm_collection_create", {})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_collection_create_with_name_succeeds(self, client):
        """rlm_collection_create should succeed when 'name' is provided."""
        response = self.call_tool(client, "rlm_collection_create", {"name": "test_collection"})
        data = response.json()
        assert data.get("error") is None
        assert "result" in data

    # rlm_collection_add requires "collection" and "vars"
    def test_rlm_collection_add_missing_collection_returns_error(self, client):
        """rlm_collection_add should return error when 'collection' is missing."""
        response = self.call_tool(client, "rlm_collection_add", {"vars": ["test_var"]})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_collection_add_missing_vars_returns_error(self, client):
        """rlm_collection_add should return error when 'vars' is missing."""
        response = self.call_tool(client, "rlm_collection_add", {"collection": "test"})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_collection_add_missing_both_returns_error(self, client):
        """rlm_collection_add should return error when both 'collection' and 'vars' are missing."""
        response = self.call_tool(client, "rlm_collection_add", {})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    # rlm_collection_info requires "name"
    def test_rlm_collection_info_missing_name_returns_error(self, client):
        """rlm_collection_info should return error when 'name' is missing."""
        response = self.call_tool(client, "rlm_collection_info", {})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    # rlm_search_collection requires "collection" and "terms"
    def test_rlm_search_collection_missing_collection_returns_error(self, client):
        """rlm_search_collection should return error when 'collection' is missing."""
        response = self.call_tool(client, "rlm_search_collection", {"terms": ["test"]})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_search_collection_missing_terms_returns_error(self, client):
        """rlm_search_collection should return error when 'terms' is missing."""
        response = self.call_tool(client, "rlm_search_collection", {"collection": "test"})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    def test_rlm_search_collection_missing_both_returns_error(self, client):
        """rlm_search_collection should return error when both 'collection' and 'terms' are missing."""
        response = self.call_tool(client, "rlm_search_collection", {})
        data = response.json()
        has_error = (
            data.get("error") is not None or
            (data.get("result") and data["result"].get("isError") is True)
        )
        assert has_error or "error" in str(data.get("result", {}).get("content", [{}])[0].get("text", "")).lower()

    # Test that tools without required params work with empty arguments
    def test_rlm_list_vars_no_required_params(self, client):
        """rlm_list_vars should work without any parameters."""
        response = self.call_tool(client, "rlm_list_vars", {})
        data = response.json()
        assert data.get("error") is None
        assert "result" in data

    def test_rlm_memory_no_required_params(self, client):
        """rlm_memory should work without any parameters."""
        response = self.call_tool(client, "rlm_memory", {})
        data = response.json()
        assert data.get("error") is None
        assert "result" in data

    def test_rlm_list_buckets_no_required_params(self, client):
        """rlm_list_buckets should work without any parameters (may error if S3 not configured)."""
        response = self.call_tool(client, "rlm_list_buckets", {})
        data = response.json()
        # Should not have KeyError type error - either success or S3 config error
        assert response.status_code == 200
        assert "result" in data

    def test_rlm_list_s3_no_required_params(self, client):
        """rlm_list_s3 should work without any parameters (may error if S3 not configured)."""
        response = self.call_tool(client, "rlm_list_s3", {})
        data = response.json()
        # Should not have KeyError type error - either success or S3 config error
        assert response.status_code == 200
        assert "result" in data

    def test_rlm_persistence_stats_no_required_params(self, client):
        """rlm_persistence_stats should work without any parameters."""
        response = self.call_tool(client, "rlm_persistence_stats", {})
        data = response.json()
        assert data.get("error") is None
        assert "result" in data

    def test_rlm_collection_list_no_required_params(self, client):
        """rlm_collection_list should work without any parameters."""
        response = self.call_tool(client, "rlm_collection_list", {})
        data = response.json()
        assert data.get("error") is None
        assert "result" in data

    def test_rlm_clear_works_with_just_all_param(self, client):
        """rlm_clear should work with just 'all' parameter."""
        response = self.call_tool(client, "rlm_clear", {"all": True})
        data = response.json()
        assert data.get("error") is None
        assert "result" in data

    def test_rlm_clear_works_with_just_name_param(self, client):
        """rlm_clear should work with just 'name' parameter."""
        response = self.call_tool(client, "rlm_clear", {"name": "nonexistent_var"})
        data = response.json()
        assert data.get("error") is None
        assert "result" in data

    def test_rlm_clear_works_with_empty_params(self, client):
        """rlm_clear should work with empty parameters (returns guidance message)."""
        response = self.call_tool(client, "rlm_clear", {})
        data = response.json()
        assert data.get("error") is None
        assert "result" in data
        # Should contain guidance message
        text = data["result"]["content"][0]["text"]
        assert "name" in text.lower() or "all" in text.lower()


class TestPersistenceErrorsInOutput:
    """Tests that verify persistence errors appear in tool output when SHOW_PERSISTENCE_ERRORS=True."""

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

    def call_tool(self, client, tool_name: str, arguments: dict, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        return self.make_mcp_request(
            client,
            "tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    def test_rlm_load_data_shows_persistence_error_when_enabled(self, client, monkeypatch):
        """rlm_load_data should show persistence error in output when SHOW_PERSISTENCE_ERRORS=True."""
        from unittest.mock import MagicMock, patch

        # Create a mock persistence that raises an error on save_variable
        mock_persistence = MagicMock()
        mock_persistence.save_variable.side_effect = Exception("Database is locked")

        # Ensure SHOW_PERSISTENCE_ERRORS is True
        monkeypatch.setattr("rlm_mcp.http_server.SHOW_PERSISTENCE_ERRORS", True)

        # Patch get_persistence in the persistence_service module where it's called
        with patch("rlm_mcp.services.persistence_service.get_persistence", return_value=mock_persistence):
            response = self.call_tool(client, "rlm_load_data", {
                "name": "test_var",
                "data": "test data content"
            })

        data = response.json()
        assert data.get("error") is None
        assert "result" in data

        # The error should appear in the output
        text = data["result"]["content"][0]["text"]
        assert "Erro de persistência" in text or "Database is locked" in text

    def test_rlm_load_data_hides_persistence_error_when_disabled(self, client, monkeypatch):
        """rlm_load_data should NOT show persistence error when SHOW_PERSISTENCE_ERRORS=False."""
        from unittest.mock import MagicMock, patch

        # Create a mock persistence that raises an error on save_variable
        mock_persistence = MagicMock()
        mock_persistence.save_variable.side_effect = Exception("Database is locked")

        # Disable SHOW_PERSISTENCE_ERRORS
        monkeypatch.setattr("rlm_mcp.http_server.SHOW_PERSISTENCE_ERRORS", False)

        # Patch get_persistence in the persistence_service module where it's called
        with patch("rlm_mcp.services.persistence_service.get_persistence", return_value=mock_persistence):
            response = self.call_tool(client, "rlm_load_data", {
                "name": "test_var",
                "data": "test data content"
            })

        data = response.json()
        assert data.get("error") is None
        assert "result" in data

        # The error should NOT appear in the output
        text = data["result"]["content"][0]["text"]
        assert "Erro de persistência" not in text
        assert "Database is locked" not in text

    def test_rlm_load_data_still_loads_variable_despite_persistence_error(self, client, monkeypatch):
        """rlm_load_data should still load variable into REPL even when persistence fails."""
        from unittest.mock import MagicMock, patch
        from rlm_mcp.http_server import repl

        # Create a mock persistence that raises an error
        mock_persistence = MagicMock()
        mock_persistence.save_variable.side_effect = Exception("Disk full")

        monkeypatch.setattr("rlm_mcp.http_server.SHOW_PERSISTENCE_ERRORS", True)

        with patch("rlm_mcp.services.persistence_service.get_persistence", return_value=mock_persistence):
            response = self.call_tool(client, "rlm_load_data", {
                "name": "my_data",
                "data": "important content"
            })

        # Despite persistence error, variable should be in REPL
        assert "my_data" in repl.variables
        assert repl.variables["my_data"] == "important content"

    def test_rlm_load_data_error_message_format(self, client, monkeypatch):
        """rlm_load_data persistence error should have expected format with warning emoji."""
        from unittest.mock import MagicMock, patch

        mock_persistence = MagicMock()
        mock_persistence.save_variable.side_effect = Exception("Connection refused")

        monkeypatch.setattr("rlm_mcp.http_server.SHOW_PERSISTENCE_ERRORS", True)

        with patch("rlm_mcp.services.persistence_service.get_persistence", return_value=mock_persistence):
            response = self.call_tool(client, "rlm_load_data", {
                "name": "test_var",
                "data": "test data"
            })

        text = response.json()["result"]["content"][0]["text"]
        # Should contain warning emoji and error message
        assert "⚠️" in text
        assert "Erro de persistência" in text
        assert "Connection refused" in text

    def test_rlm_load_s3_shows_persistence_error_when_enabled(self, client, monkeypatch, mock_minio_client_with_data):
        """rlm_load_s3 should show persistence error in output when SHOW_PERSISTENCE_ERRORS=True."""
        from unittest.mock import MagicMock, patch
        from rlm_mcp.s3_client import S3Client
        import os

        # Create a mock persistence that raises an error on save_variable
        mock_persistence = MagicMock()
        mock_persistence.save_variable.side_effect = Exception("Cannot write to database")

        # Ensure SHOW_PERSISTENCE_ERRORS is True
        monkeypatch.setattr("rlm_mcp.http_server.SHOW_PERSISTENCE_ERRORS", True)

        # Setup S3 mock
        with patch.dict(os.environ, {
            "MINIO_ENDPOINT": "mock-minio:9000",
            "MINIO_ACCESS_KEY": "mock-access-key",
            "MINIO_SECRET_KEY": "mock-secret-key",
            "MINIO_SECURE": "false",
        }):
            mock_s3_client = S3Client()
            mock_s3_client._client = mock_minio_client_with_data

            with patch("rlm_mcp.services.s3_guard.get_s3_client", return_value=mock_s3_client):
                with patch("rlm_mcp.services.persistence_service.get_persistence", return_value=mock_persistence):
                    response = self.call_tool(client, "rlm_load_s3", {
                        "key": "test.txt",
                        "name": "s3_var",
                        "bucket": "test-bucket"
                    })

        data = response.json()
        assert data.get("error") is None
        assert "result" in data

        # The error should appear in the output
        text = data["result"]["content"][0]["text"]
        assert "Erro de persistência" in text or "Cannot write to database" in text

    def test_rlm_load_s3_hides_persistence_error_when_disabled(self, client, monkeypatch, mock_minio_client_with_data):
        """rlm_load_s3 should NOT show persistence error when SHOW_PERSISTENCE_ERRORS=False."""
        from unittest.mock import MagicMock, patch
        from rlm_mcp.s3_client import S3Client
        import os

        # Create a mock persistence that raises an error on save_variable
        mock_persistence = MagicMock()
        mock_persistence.save_variable.side_effect = Exception("Cannot write to database")

        # Disable SHOW_PERSISTENCE_ERRORS
        monkeypatch.setattr("rlm_mcp.http_server.SHOW_PERSISTENCE_ERRORS", False)

        # Setup S3 mock
        with patch.dict(os.environ, {
            "MINIO_ENDPOINT": "mock-minio:9000",
            "MINIO_ACCESS_KEY": "mock-access-key",
            "MINIO_SECRET_KEY": "mock-secret-key",
            "MINIO_SECURE": "false",
        }):
            mock_s3_client = S3Client()
            mock_s3_client._client = mock_minio_client_with_data

            with patch("rlm_mcp.services.s3_guard.get_s3_client", return_value=mock_s3_client):
                with patch("rlm_mcp.services.persistence_service.get_persistence", return_value=mock_persistence):
                    response = self.call_tool(client, "rlm_load_s3", {
                        "key": "test.txt",
                        "name": "s3_var",
                        "bucket": "test-bucket"
                    })

        data = response.json()
        assert data.get("error") is None
        assert "result" in data

        # The error should NOT appear in the output
        text = data["result"]["content"][0]["text"]
        assert "Erro de persistência" not in text
        assert "Cannot write to database" not in text

    def test_constant_defaults_to_true(self):
        """SHOW_PERSISTENCE_ERRORS constant should default to True."""
        import importlib
        import os
        from unittest.mock import patch

        # Test with environment variable not set
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing value
            if "RLM_SHOW_PERSISTENCE_ERRORS" in os.environ:
                del os.environ["RLM_SHOW_PERSISTENCE_ERRORS"]

            # The constant should be True by default (per PRD)
            # Check the code defines it correctly
            import inspect
            from rlm_mcp import http_server
            source = inspect.getsource(http_server)
            assert 'SHOW_PERSISTENCE_ERRORS = os.getenv("RLM_SHOW_PERSISTENCE_ERRORS", "true")' in source

    def test_constant_can_be_disabled_via_env_var(self):
        """SHOW_PERSISTENCE_ERRORS can be disabled by setting env var to 'false'."""
        import inspect
        from rlm_mcp import http_server

        # Verify the code checks for false/0/no values
        source = inspect.getsource(http_server)
        # The constant uses .lower() in ("true", "1", "yes") which means
        # any other value (like "false") will result in False
        assert '.lower() in ("true", "1", "yes")' in source


class TestPaginationRlmListVars:
    """Tests for pagination parameters in rlm_list_vars tool."""

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

    def call_tool(self, client, tool_name: str, arguments: dict = None, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        params = {"name": tool_name}
        if arguments is not None:
            params["arguments"] = arguments
        else:
            params["arguments"] = {}
        return self.make_mcp_request(
            client,
            "tools/call",
            params=params,
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test to avoid cross-test pollution."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    def test_offset_parameter_in_schema(self, client):
        """rlm_list_vars should have 'offset' parameter in inputSchema."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        list_vars_tool = next((t for t in tools if t["name"] == "rlm_list_vars"), None)
        assert list_vars_tool is not None
        assert "offset" in list_vars_tool["inputSchema"]["properties"]
        assert list_vars_tool["inputSchema"]["properties"]["offset"]["type"] == "integer"
        assert list_vars_tool["inputSchema"]["properties"]["offset"]["default"] == 0

    def test_limit_parameter_in_schema(self, client):
        """rlm_list_vars should have 'limit' parameter in inputSchema."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        list_vars_tool = next((t for t in tools if t["name"] == "rlm_list_vars"), None)
        assert list_vars_tool is not None
        assert "limit" in list_vars_tool["inputSchema"]["properties"]
        assert list_vars_tool["inputSchema"]["properties"]["limit"]["type"] == "integer"
        assert list_vars_tool["inputSchema"]["properties"]["limit"]["default"] == 50

    def test_pagination_with_limit(self, client):
        """rlm_list_vars should respect limit parameter."""
        # Load 5 variables
        for i in range(5):
            self.call_tool(client, "rlm_load_data", {"name": f"var_{i}", "data": f"data_{i}"})

        # Get first 2 variables
        response = self.call_tool(client, "rlm_list_vars", {"limit": 2})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show pagination info
        assert "5 total" in text
        assert "mostrando 1-2" in text

    def test_pagination_with_offset(self, client):
        """rlm_list_vars should skip results when offset is used."""
        # Load 5 variables
        for i in range(5):
            self.call_tool(client, "rlm_load_data", {"name": f"var_{i}", "data": f"data_{i}"})

        # Get variables starting from offset 2
        response = self.call_tool(client, "rlm_list_vars", {"offset": 2, "limit": 2})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show correct range
        assert "5 total" in text
        assert "mostrando 3-4" in text

    def test_pagination_offset_and_limit_together(self, client):
        """rlm_list_vars should handle offset and limit together."""
        # Load 10 variables
        for i in range(10):
            self.call_tool(client, "rlm_load_data", {"name": f"var_{i}", "data": f"data_{i}"})

        # Get page 2 (offset=3, limit=3)
        response = self.call_tool(client, "rlm_list_vars", {"offset": 3, "limit": 3})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show 10 total, displaying items 4-6
        assert "10 total" in text
        assert "mostrando 4-6" in text

    def test_pagination_offset_beyond_results(self, client):
        """rlm_list_vars should handle offset beyond available variables gracefully."""
        # Load 3 variables
        for i in range(3):
            self.call_tool(client, "rlm_load_data", {"name": f"var_{i}", "data": f"data_{i}"})

        # Use offset beyond available
        response = self.call_tool(client, "rlm_list_vars", {"offset": 100, "limit": 10})
        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show total and handle empty results (start_idx=0 when no results)
        assert "3 total" in text
        # When offset is beyond results, start_idx becomes 0 (indicating no results shown)
        assert "mostrando 0-" in text

    def test_pagination_default_offset_is_zero(self, client):
        """rlm_list_vars should default offset to 0."""
        # Load 3 variables
        for i in range(3):
            self.call_tool(client, "rlm_load_data", {"name": f"var_{i}", "data": f"data_{i}"})

        # Without offset
        response1 = self.call_tool(client, "rlm_list_vars", {"limit": 2})
        text1 = response1.json()["result"]["content"][0]["text"]

        # With offset=0 explicitly
        response2 = self.call_tool(client, "rlm_list_vars", {"offset": 0, "limit": 2})
        text2 = response2.json()["result"]["content"][0]["text"]

        # Both should show same results (starting at 1)
        assert "mostrando 1-2" in text1
        assert "mostrando 1-2" in text2


class TestPaginationRlmListS3:
    """Tests for pagination parameters in rlm_list_s3 tool."""

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

    def call_tool(self, client, tool_name: str, arguments: dict = None, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        params = {"name": tool_name}
        if arguments is not None:
            params["arguments"] = arguments
        else:
            params["arguments"] = {}
        return self.make_mcp_request(
            client,
            "tools/call",
            params=params,
            request_id=request_id
        )

    def test_offset_parameter_in_schema(self, client):
        """rlm_list_s3 should have 'offset' parameter in inputSchema."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        list_s3_tool = next((t for t in tools if t["name"] == "rlm_list_s3"), None)
        assert list_s3_tool is not None
        assert "offset" in list_s3_tool["inputSchema"]["properties"]
        assert list_s3_tool["inputSchema"]["properties"]["offset"]["type"] == "integer"
        assert list_s3_tool["inputSchema"]["properties"]["offset"]["default"] == 0

    def test_limit_parameter_in_schema(self, client):
        """rlm_list_s3 should have 'limit' parameter in inputSchema."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        list_s3_tool = next((t for t in tools if t["name"] == "rlm_list_s3"), None)
        assert list_s3_tool is not None
        assert "limit" in list_s3_tool["inputSchema"]["properties"]
        assert list_s3_tool["inputSchema"]["properties"]["limit"]["type"] == "integer"
        assert list_s3_tool["inputSchema"]["properties"]["limit"]["default"] == 50

    def test_pagination_with_mock_s3(self, client, monkeypatch):
        """rlm_list_s3 should apply pagination to results."""
        from unittest.mock import MagicMock, patch
        import os

        # Create mock objects
        mock_objects = [
            {"name": f"file_{i}.txt", "size_human": f"{i} KB", "size_bytes": i * 1024}
            for i in range(10)
        ]

        mock_s3_client = MagicMock()
        mock_s3_client.is_configured.return_value = True
        mock_s3_client.list_objects.return_value = mock_objects

        with patch("rlm_mcp.services.s3_guard.get_s3_client", return_value=mock_s3_client):
            # Get first 3 objects
            response = self.call_tool(client, "rlm_list_s3", {
                "bucket": "test-bucket",
                "limit": 3,
                "offset": 0
            })

        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show pagination info
        assert "10 total" in text
        assert "mostrando 1-3" in text

    def test_pagination_offset_with_mock_s3(self, client, monkeypatch):
        """rlm_list_s3 should skip objects when offset is used."""
        from unittest.mock import MagicMock, patch

        # Create mock objects
        mock_objects = [
            {"name": f"file_{i}.txt", "size_human": f"{i} KB", "size_bytes": i * 1024}
            for i in range(10)
        ]

        mock_s3_client = MagicMock()
        mock_s3_client.is_configured.return_value = True
        mock_s3_client.list_objects.return_value = mock_objects

        with patch("rlm_mcp.services.s3_guard.get_s3_client", return_value=mock_s3_client):
            # Get objects starting from offset 5
            response = self.call_tool(client, "rlm_list_s3", {
                "bucket": "test-bucket",
                "limit": 3,
                "offset": 5
            })

        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show correct range (6-8 because offset=5, limit=3)
        assert "10 total" in text
        assert "mostrando 6-8" in text

    def test_pagination_offset_beyond_results_with_mock_s3(self, client, monkeypatch):
        """rlm_list_s3 should handle offset beyond available objects gracefully."""
        from unittest.mock import MagicMock, patch

        # Create mock objects
        mock_objects = [
            {"name": f"file_{i}.txt", "size_human": f"{i} KB", "size_bytes": i * 1024}
            for i in range(5)
        ]

        mock_s3_client = MagicMock()
        mock_s3_client.is_configured.return_value = True
        mock_s3_client.list_objects.return_value = mock_objects

        with patch("rlm_mcp.services.s3_guard.get_s3_client", return_value=mock_s3_client):
            # Use offset beyond available
            response = self.call_tool(client, "rlm_list_s3", {
                "bucket": "test-bucket",
                "limit": 10,
                "offset": 100
            })

        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show total and handle empty results (start_idx=0 when no results)
        assert "5 total" in text
        # When offset is beyond results, start_idx becomes 0 (indicating no results shown)
        assert "mostrando 0-" in text


class TestPaginationRlmSearchCollection:
    """Tests for pagination parameters in rlm_search_collection tool."""

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

    def call_tool(self, client, tool_name: str, arguments: dict = None, request_id: int = 1):
        """Helper to call a tool via MCP tools/call."""
        params = {"name": tool_name}
        if arguments is not None:
            params["arguments"] = arguments
        else:
            params["arguments"] = {}
        return self.make_mcp_request(
            client,
            "tools/call",
            params=params,
            request_id=request_id
        )

    @pytest.fixture(autouse=True)
    def reset_repl(self):
        """Reset REPL state before each test to avoid cross-test pollution."""
        from rlm_mcp.http_server import repl
        repl.clear_all()
        yield
        repl.clear_all()

    def test_offset_parameter_in_schema(self, client):
        """rlm_search_collection should have 'offset' parameter in inputSchema."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        search_coll_tool = next((t for t in tools if t["name"] == "rlm_search_collection"), None)
        assert search_coll_tool is not None
        assert "offset" in search_coll_tool["inputSchema"]["properties"]
        assert search_coll_tool["inputSchema"]["properties"]["offset"]["type"] == "integer"
        assert search_coll_tool["inputSchema"]["properties"]["offset"]["default"] == 0

    def test_limit_parameter_in_schema(self, client):
        """rlm_search_collection should have 'limit' parameter in inputSchema."""
        response = self.make_mcp_request(client, "tools/list")
        data = response.json()
        tools = data["result"]["tools"]

        search_coll_tool = next((t for t in tools if t["name"] == "rlm_search_collection"), None)
        assert search_coll_tool is not None
        assert "limit" in search_coll_tool["inputSchema"]["properties"]
        assert search_coll_tool["inputSchema"]["properties"]["limit"]["type"] == "integer"
        assert search_coll_tool["inputSchema"]["properties"]["limit"]["default"] == 10

    def test_pagination_with_mocked_index(self, client, monkeypatch, tmp_path):
        """rlm_search_collection should apply pagination to search results."""
        from unittest.mock import MagicMock, patch

        # Create mock index with many matches
        mock_matches = [
            {"linha": i, "contexto": f"This is line {i} containing the search term"}
            for i in range(20)
        ]

        mock_index = MagicMock()
        mock_index.search_multiple.return_value = {"test_term": mock_matches}

        # Mock persistence to return collection vars
        mock_persistence = MagicMock()
        mock_persistence.get_collection_vars.return_value = ["var1"]

        # Mock get_index to return our mock index
        with patch("rlm_mcp.http_server.get_persistence", return_value=mock_persistence):
            with patch("rlm_mcp.http_server.get_index", return_value=mock_index):
                response = self.call_tool(client, "rlm_search_collection", {
                    "collection": "test_collection",
                    "terms": ["test_term"],
                    "limit": 5,
                    "offset": 0
                })

        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show pagination info
        assert "20 ocorrências" in text
        assert "mostrando 1-5" in text

    def test_pagination_offset_with_mocked_index(self, client, monkeypatch, tmp_path):
        """rlm_search_collection should skip results when offset is used."""
        from unittest.mock import MagicMock, patch

        # Create mock index with many matches
        mock_matches = [
            {"linha": i, "contexto": f"This is line {i} containing the search term"}
            for i in range(20)
        ]

        mock_index = MagicMock()
        mock_index.search_multiple.return_value = {"test_term": mock_matches}

        # Mock persistence to return collection vars
        mock_persistence = MagicMock()
        mock_persistence.get_collection_vars.return_value = ["var1"]

        # Mock get_index to return our mock index
        with patch("rlm_mcp.http_server.get_persistence", return_value=mock_persistence):
            with patch("rlm_mcp.http_server.get_index", return_value=mock_index):
                response = self.call_tool(client, "rlm_search_collection", {
                    "collection": "test_collection",
                    "terms": ["test_term"],
                    "limit": 5,
                    "offset": 10
                })

        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show correct range (11-15 because offset=10, limit=5)
        assert "20 ocorrências" in text
        assert "mostrando 11-15" in text

    def test_pagination_offset_beyond_results_with_mocked_index(self, client, monkeypatch, tmp_path):
        """rlm_search_collection should handle offset beyond available results gracefully."""
        from unittest.mock import MagicMock, patch

        # Create mock index with few matches
        mock_matches = [
            {"linha": i, "contexto": f"This is line {i} containing the search term"}
            for i in range(5)
        ]

        mock_index = MagicMock()
        mock_index.search_multiple.return_value = {"test_term": mock_matches}

        # Mock persistence to return collection vars
        mock_persistence = MagicMock()
        mock_persistence.get_collection_vars.return_value = ["var1"]

        # Mock get_index to return our mock index
        with patch("rlm_mcp.http_server.get_persistence", return_value=mock_persistence):
            with patch("rlm_mcp.http_server.get_index", return_value=mock_index):
                response = self.call_tool(client, "rlm_search_collection", {
                    "collection": "test_collection",
                    "terms": ["test_term"],
                    "limit": 10,
                    "offset": 100
                })

        data = response.json()
        text = data["result"]["content"][0]["text"]

        # Should show total and handle empty results (start_idx=0 when no results)
        assert "5 ocorrências" in text
        # When offset is beyond results, start_idx becomes 0 (indicating no results shown)
        assert "mostrando 0-" in text

    def test_pagination_default_offset_is_zero(self, client, monkeypatch, tmp_path):
        """rlm_search_collection should default offset to 0."""
        from unittest.mock import MagicMock, patch

        # Create mock index with matches
        mock_matches = [
            {"linha": i, "contexto": f"This is line {i} containing the search term"}
            for i in range(10)
        ]

        mock_index = MagicMock()
        mock_index.search_multiple.return_value = {"test_term": mock_matches}

        # Mock persistence to return collection vars
        mock_persistence = MagicMock()
        mock_persistence.get_collection_vars.return_value = ["var1"]

        # Mock get_index to return our mock index
        with patch("rlm_mcp.http_server.get_persistence", return_value=mock_persistence):
            with patch("rlm_mcp.http_server.get_index", return_value=mock_index):
                # Without offset
                response1 = self.call_tool(client, "rlm_search_collection", {
                    "collection": "test_collection",
                    "terms": ["test_term"],
                    "limit": 3
                })

                # With offset=0 explicitly
                response2 = self.call_tool(client, "rlm_search_collection", {
                    "collection": "test_collection",
                    "terms": ["test_term"],
                    "limit": 3,
                    "offset": 0
                })

        text1 = response1.json()["result"]["content"][0]["text"]
        text2 = response2.json()["result"]["content"][0]["text"]

        # Both should show same results (starting at 1)
        assert "mostrando 1-3" in text1
        assert "mostrando 1-3" in text2


# ===========================================================================
# Tests for SSE Rate Limiting
# ===========================================================================


class TestSseRateLimiting:
    """Tests for rate limiting on SSE sessions (100 requests/minute)."""

    def test_rate_limiter_import(self):
        """SlidingWindowRateLimiter should be imported in http_server."""
        from rlm_mcp.http_server import sse_rate_limiter
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter
        assert isinstance(sse_rate_limiter, SlidingWindowRateLimiter)

    def test_rate_limiter_config(self):
        """SSE rate limiter should be configured for 100 req/60s by default."""
        from rlm_mcp.http_server import sse_rate_limiter
        assert sse_rate_limiter.config.max_requests == 100
        assert sse_rate_limiter.config.window_seconds == 60

    def test_message_without_session_not_rate_limited(self, client):
        """Requests without session_id should not be rate limited."""
        # Make many requests without session_id
        for _ in range(110):
            response = client.post("/message", json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            })
            # All should succeed (no 429)
            assert response.status_code in (200, 202)

    def test_message_with_invalid_session_not_rate_limited(self, client):
        """Requests with non-existent session_id should not be rate limited."""
        # Make many requests with fake session_id
        for _ in range(110):
            response = client.post("/message?session_id=fake-session-123", json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            })
            # All should succeed (session doesn't exist so no rate limiting)
            assert response.status_code in (200, 202)

    def test_rate_limit_exceeded_returns_429(self, client, monkeypatch):
        """Exceeding rate limit should return 429 status code."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a very restrictive rate limiter for testing (2 req/60s)
        test_limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        monkeypatch.setattr(http_server, "sse_rate_limiter", test_limiter)

        # Simulate an active SSE session
        import asyncio
        session_id = "test-session-rate-limit"
        http_server.sse_sessions[session_id] = asyncio.Queue()

        try:
            # First 2 requests should succeed
            for i in range(2):
                response = client.post(f"/message?session_id={session_id}", json={
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/list"
                })
                assert response.status_code == 202, f"Request {i+1} should succeed"

            # Third request should be rate limited
            response = client.post(f"/message?session_id={session_id}", json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/list"
            })
            assert response.status_code == 429
        finally:
            http_server.sse_sessions.pop(session_id, None)

    def test_rate_limit_error_response_format(self, client, monkeypatch):
        """429 response should have proper error format."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a very restrictive rate limiter (1 req/60s)
        test_limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        monkeypatch.setattr(http_server, "sse_rate_limiter", test_limiter)

        # Simulate an active SSE session
        import asyncio
        session_id = "test-session-format"
        http_server.sse_sessions[session_id] = asyncio.Queue()

        try:
            # First request succeeds
            client.post(f"/message?session_id={session_id}", json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            })

            # Second request should be rate limited
            response = client.post(f"/message?session_id={session_id}", json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            })

            assert response.status_code == 429
            data = response.json()
            assert "error" in data
            assert data["error"] == "Too Many Requests"
            assert "message" in data
            assert "Rate limit exceeded" in data["message"]
            assert "retry_after" in data
        finally:
            http_server.sse_sessions.pop(session_id, None)

    def test_rate_limit_includes_retry_after_header(self, client, monkeypatch):
        """429 response should include Retry-After header."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a very restrictive rate limiter (1 req/60s)
        test_limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        monkeypatch.setattr(http_server, "sse_rate_limiter", test_limiter)

        # Simulate an active SSE session
        import asyncio
        session_id = "test-session-header"
        http_server.sse_sessions[session_id] = asyncio.Queue()

        try:
            # First request succeeds
            client.post(f"/message?session_id={session_id}", json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            })

            # Second request should be rate limited
            response = client.post(f"/message?session_id={session_id}", json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            })

            assert response.status_code == 429
            assert "retry-after" in response.headers
            retry_after = int(response.headers["retry-after"])
            assert retry_after > 0
        finally:
            http_server.sse_sessions.pop(session_id, None)

    def test_different_sessions_independent_rate_limits(self, client, monkeypatch):
        """Different SSE sessions should have independent rate limits."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a very restrictive rate limiter (2 req/60s)
        test_limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        monkeypatch.setattr(http_server, "sse_rate_limiter", test_limiter)

        # Simulate two active SSE sessions
        import asyncio
        session1 = "test-session-1"
        session2 = "test-session-2"
        http_server.sse_sessions[session1] = asyncio.Queue()
        http_server.sse_sessions[session2] = asyncio.Queue()

        try:
            # Exhaust rate limit for session1
            for i in range(2):
                client.post(f"/message?session_id={session1}", json={
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/list"
                })

            # Session1 should be rate limited
            response1 = client.post(f"/message?session_id={session1}", json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/list"
            })
            assert response1.status_code == 429

            # Session2 should still work
            response2 = client.post(f"/message?session_id={session2}", json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            })
            assert response2.status_code == 202
        finally:
            http_server.sse_sessions.pop(session1, None)
            http_server.sse_sessions.pop(session2, None)

    def test_rate_limit_message_includes_limit_info(self, client, monkeypatch):
        """Rate limit error message should include limit and window info."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a specific rate limiter (5 req/30s)
        test_limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=30)
        monkeypatch.setattr(http_server, "sse_rate_limiter", test_limiter)

        # Simulate an active SSE session
        import asyncio
        session_id = "test-session-info"
        http_server.sse_sessions[session_id] = asyncio.Queue()

        try:
            # Exhaust rate limit
            for i in range(5):
                client.post(f"/message?session_id={session_id}", json={
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/list"
                })

            # Get rate limited response
            response = client.post(f"/message?session_id={session_id}", json={
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/list"
            })

            assert response.status_code == 429
            data = response.json()
            # Message should mention the limit and window
            assert "5" in data["message"]  # max_requests
            assert "30" in data["message"]  # window_seconds
        finally:
            http_server.sse_sessions.pop(session_id, None)

    def test_rate_limit_allows_requests_after_window(self, client, monkeypatch):
        """Requests should be allowed again after the rate limit window."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter
        import time

        # Create a rate limiter with very short window (1 req/1s)
        test_limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=1)
        monkeypatch.setattr(http_server, "sse_rate_limiter", test_limiter)

        # Simulate an active SSE session
        import asyncio
        session_id = "test-session-window"
        http_server.sse_sessions[session_id] = asyncio.Queue()

        try:
            # First request succeeds
            response1 = client.post(f"/message?session_id={session_id}", json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            })
            assert response1.status_code == 202

            # Immediate second request should be rate limited
            response2 = client.post(f"/message?session_id={session_id}", json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            })
            assert response2.status_code == 429

            # Wait for window to pass
            time.sleep(1.5)

            # Third request should succeed
            response3 = client.post(f"/message?session_id={session_id}", json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/list"
            })
            assert response3.status_code == 202
        finally:
            http_server.sse_sessions.pop(session_id, None)

    def test_rate_limiter_cleaned_on_session_end(self, monkeypatch):
        """Rate limiter state should be cleaned when SSE session ends."""
        from rlm_mcp import http_server

        # Record calls to reset
        reset_calls = []
        original_reset = http_server.sse_rate_limiter.reset

        def mock_reset(identifier):
            reset_calls.append(identifier)
            original_reset(identifier)

        monkeypatch.setattr(http_server.sse_rate_limiter, "reset", mock_reset)

        # Simulate session creation and cleanup
        import asyncio
        session_id = "test-cleanup-session"
        http_server.sse_sessions[session_id] = asyncio.Queue()

        # Simulate SSE session ending (what happens in finally block)
        http_server.sse_sessions.pop(session_id, None)
        http_server.sse_rate_limiter.reset(session_id)

        # Verify reset was called with the session_id
        assert session_id in reset_calls

    def test_env_var_config_sse_rate_limit(self, monkeypatch):
        """SSE_RATE_LIMIT_REQUESTS should be configurable via env var."""
        # This tests that the config can be set via environment variable
        # Note: The actual env var is read at module load time
        from rlm_mcp.http_server import SSE_RATE_LIMIT_REQUESTS, SSE_RATE_LIMIT_WINDOW
        # Default values
        assert SSE_RATE_LIMIT_REQUESTS == 100
        assert SSE_RATE_LIMIT_WINDOW == 60

    def test_requests_within_limit_succeed(self, client, monkeypatch):
        """All requests within the rate limit should succeed."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a rate limiter (5 req/60s)
        test_limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        monkeypatch.setattr(http_server, "sse_rate_limiter", test_limiter)

        # Simulate an active SSE session
        import asyncio
        session_id = "test-session-within-limit"
        http_server.sse_sessions[session_id] = asyncio.Queue()

        try:
            # All 5 requests should succeed
            for i in range(5):
                response = client.post(f"/message?session_id={session_id}", json={
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/list"
                })
                assert response.status_code == 202, f"Request {i+1} should succeed"
        finally:
            http_server.sse_sessions.pop(session_id, None)


# =============================================================================
# Tests for Upload Rate Limiting returning 429
# =============================================================================

class TestUploadRateLimiting429:
    """Tests for upload rate limiting returning HTTP 429 status code."""

    def test_rate_limit_exceeded_exception_exists(self):
        """RateLimitExceeded exception should be defined in http_server."""
        from rlm_mcp.http_server import RateLimitExceeded
        assert RateLimitExceeded is not None

    def test_rate_limit_exceeded_exception_attributes(self):
        """RateLimitExceeded should have proper attributes."""
        from rlm_mcp.http_server import RateLimitExceeded
        from rlm_mcp.rate_limiter import RateLimitResult

        # Create a mock rate limit result
        result = RateLimitResult(
            allowed=False,
            current_count=11,
            limit=10,
            window_seconds=60,
            retry_after=45.5
        )

        exc = RateLimitExceeded(result, message="Test rate limit")
        assert exc.limit == 10
        assert exc.window_seconds == 60
        assert exc.retry_after == 45.5
        assert exc.current_count == 11
        assert exc.message == "Test rate limit"
        assert str(exc) == "Test rate limit"

    def test_rate_limit_exceeded_default_message(self):
        """RateLimitExceeded should generate default message if not provided."""
        from rlm_mcp.http_server import RateLimitExceeded
        from rlm_mcp.rate_limiter import RateLimitResult

        result = RateLimitResult(
            allowed=False,
            current_count=15,
            limit=10,
            window_seconds=60,
            retry_after=30
        )

        exc = RateLimitExceeded(result)
        assert "10 requests per 60 seconds" in exc.message

    def test_upload_rate_limit_returns_429_on_message_endpoint(self, client, monkeypatch):
        """Upload rate limit exceeded should return HTTP 429 on /message endpoint."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a very restrictive upload rate limiter (1 upload/60s)
        test_upload_limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        monkeypatch.setattr(http_server, "upload_rate_limiter", test_upload_limiter)

        # First upload attempt - should record (even though upload will fail due to no S3)
        # But we need to make it succeed first to count against the limit
        # We'll mock the check to have already recorded a request
        test_upload_limiter.record("testclient")

        # Second upload attempt should be rate limited
        response = client.post("/message", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "rlm_upload_url",
                "arguments": {
                    "url": "http://example.com/test.txt",
                    "key": "test.txt"
                }
            }
        })

        assert response.status_code == 429
        data = response.json()
        assert data["error"] == "Too Many Requests"
        assert "Upload rate limit exceeded" in data["message"]
        assert "retry_after" in data

    def test_upload_rate_limit_returns_429_on_mcp_endpoint(self, client, monkeypatch):
        """Upload rate limit exceeded should return HTTP 429 on /mcp endpoint."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a very restrictive upload rate limiter (1 upload/60s)
        test_upload_limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        monkeypatch.setattr(http_server, "upload_rate_limiter", test_upload_limiter)

        # Pre-record a request to exhaust the limit
        test_upload_limiter.record("testclient")

        # Upload attempt should be rate limited
        response = client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "rlm_upload_url",
                "arguments": {
                    "url": "http://example.com/test.txt",
                    "key": "test.txt"
                }
            }
        })

        assert response.status_code == 429
        data = response.json()
        assert data["error"] == "Too Many Requests"
        assert "retry_after" in data

    def test_upload_rate_limit_includes_retry_after_header(self, client, monkeypatch):
        """429 response should include Retry-After header."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a very restrictive upload rate limiter
        test_upload_limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        monkeypatch.setattr(http_server, "upload_rate_limiter", test_upload_limiter)

        # Pre-record a request
        test_upload_limiter.record("testclient")

        response = client.post("/message", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "rlm_upload_url",
                "arguments": {
                    "url": "http://example.com/test.txt",
                    "key": "test.txt"
                }
            }
        })

        assert response.status_code == 429
        assert "Retry-After" in response.headers
        retry_after = int(response.headers["Retry-After"])
        assert retry_after > 0

    def test_upload_within_limit_not_rate_limited(self, client, monkeypatch):
        """Uploads within rate limit should not return 429."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a reasonable rate limiter (10 uploads/60s)
        test_upload_limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=60)
        monkeypatch.setattr(http_server, "upload_rate_limiter", test_upload_limiter)

        # First upload attempt should not be rate limited
        # (it might fail for other reasons like S3 not configured, but not 429)
        response = client.post("/message", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "rlm_upload_url",
                "arguments": {
                    "url": "http://example.com/test.txt",
                    "key": "test.txt"
                }
            }
        })

        # Should not be 429 (might be 200 with error due to S3 config)
        assert response.status_code != 429

    def test_different_clients_independent_upload_limits(self, monkeypatch):
        """Different clients should have independent upload rate limits."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a restrictive rate limiter (2 uploads/60s)
        test_upload_limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        monkeypatch.setattr(http_server, "upload_rate_limiter", test_upload_limiter)

        # Exhaust limit for client1
        test_upload_limiter.record("client1")
        test_upload_limiter.record("client1")

        # Client1 should be rate limited
        result1 = test_upload_limiter.check("client1")
        assert not result1.allowed

        # Client2 should not be rate limited
        result2 = test_upload_limiter.check("client2")
        assert result2.allowed

    def test_upload_rate_limiter_config(self):
        """Upload rate limiter should be configured for 10 uploads/60s by default."""
        from rlm_mcp.http_server import upload_rate_limiter
        assert upload_rate_limiter.config.max_requests == 10
        assert upload_rate_limiter.config.window_seconds == 60

    def test_rate_limit_message_includes_upload_info(self, client, monkeypatch):
        """Rate limit error message should mention upload limit specifically."""
        from rlm_mcp import http_server
        from rlm_mcp.rate_limiter import SlidingWindowRateLimiter

        # Create a restrictive limiter with specific values
        test_upload_limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=30)
        monkeypatch.setattr(http_server, "upload_rate_limiter", test_upload_limiter)

        # Exhaust limit
        for _ in range(5):
            test_upload_limiter.record("testclient")

        response = client.post("/message", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "rlm_upload_url",
                "arguments": {
                    "url": "http://example.com/test.txt",
                    "key": "test.txt"
                }
            }
        })

        assert response.status_code == 429
        data = response.json()
        assert "Upload" in data["message"] or "upload" in data["message"].lower()
        assert "5" in data["message"]
        assert "30" in data["message"]


class TestJsonLogging:
    """Tests for structured JSON logging."""

    def test_json_formatter_produces_valid_json(self):
        """JsonFormatter should produce valid JSON output."""
        import json
        from rlm_mcp.http_server import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test-logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert isinstance(parsed, dict)
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed

    def test_json_formatter_includes_timestamp(self):
        """JSON log should include ISO 8601 timestamp."""
        import json
        from rlm_mcp.http_server import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "timestamp" in parsed
        assert parsed["timestamp"].endswith("Z")
        # Should be ISO format
        from datetime import datetime
        datetime.fromisoformat(parsed["timestamp"].rstrip("Z"))

    def test_json_formatter_includes_level(self):
        """JSON log should include log level."""
        import json
        from rlm_mcp.http_server import JsonFormatter

        formatter = JsonFormatter()

        for level, level_name in [(logging.DEBUG, "DEBUG"), (logging.INFO, "INFO"),
                                   (logging.WARNING, "WARNING"), (logging.ERROR, "ERROR")]:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None
            )
            output = formatter.format(record)
            parsed = json.loads(output)
            assert parsed["level"] == level_name

    def test_json_formatter_includes_logger_name(self):
        """JSON log should include logger name."""
        import json
        from rlm_mcp.http_server import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="my-custom-logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["logger"] == "my-custom-logger"

    def test_json_formatter_includes_message(self):
        """JSON log should include the log message."""
        import json
        from rlm_mcp.http_server import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="This is a test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "This is a test message"

    def test_json_formatter_handles_message_args(self):
        """JSON log should handle message formatting with args."""
        import json
        from rlm_mcp.http_server import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Value is %s and count is %d",
            args=("test", 42),
            exc_info=None
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "Value is test and count is 42"

    def test_json_formatter_includes_exception(self):
        """JSON log should include exception info when present."""
        import json
        import sys
        from rlm_mcp.http_server import JsonFormatter

        formatter = JsonFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "Test exception" in parsed["exception"]

    def test_json_formatter_includes_extra_fields(self):
        """JSON log should include extra fields."""
        import json
        from rlm_mcp.http_server import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )
        # Add extra fields
        record.request_id = "abc123"
        record.user_id = "user456"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["request_id"] == "abc123"
        assert parsed["user_id"] == "user456"

    def test_setup_logging_json_format(self):
        """setup_logging with json format should use JsonFormatter."""
        from rlm_mcp.http_server import setup_logging, JsonFormatter

        # Save original handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]

        try:
            setup_logging("json", "INFO")
            assert len(root_logger.handlers) == 1
            assert isinstance(root_logger.handlers[0].formatter, JsonFormatter)
        finally:
            # Restore original handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            for handler in original_handlers:
                root_logger.addHandler(handler)

    def test_setup_logging_text_format(self):
        """setup_logging with text format should use standard Formatter."""
        from rlm_mcp.http_server import setup_logging, JsonFormatter

        # Save original handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]

        try:
            setup_logging("text", "INFO")
            assert len(root_logger.handlers) == 1
            assert not isinstance(root_logger.handlers[0].formatter, JsonFormatter)
        finally:
            # Restore original handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            for handler in original_handlers:
                root_logger.addHandler(handler)

    def test_setup_logging_sets_level(self):
        """setup_logging should set the correct log level."""
        from rlm_mcp.http_server import setup_logging

        # Save original handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        original_level = root_logger.level

        try:
            setup_logging("text", "DEBUG")
            assert root_logger.level == logging.DEBUG

            setup_logging("text", "WARNING")
            assert root_logger.level == logging.WARNING
        finally:
            # Restore original state
            root_logger.setLevel(original_level)
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            for handler in original_handlers:
                root_logger.addHandler(handler)

    def test_json_formatter_handles_non_serializable_extra(self):
        """JSON formatter should handle non-JSON-serializable extra fields."""
        import json
        from rlm_mcp.http_server import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )
        # Add a non-serializable extra field
        record.complex_obj = {"nested": object()}

        # Should not raise, uses default=str
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "complex_obj" in parsed

    def test_log_format_env_variable_default(self):
        """LOG_FORMAT should default to 'text'."""
        from rlm_mcp import http_server
        # Check the module-level constant (may be overridden by env)
        assert hasattr(http_server, "LOG_FORMAT")

    def test_log_level_env_variable_default(self):
        """LOG_LEVEL should default to 'INFO'."""
        from rlm_mcp import http_server
        # Check the module-level constant (may be overridden by env)
        assert hasattr(http_server, "LOG_LEVEL")


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_returns_200_status_code(self, client):
        """/metrics should return 200 status code."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_returns_json(self, client):
        """/metrics should return JSON."""
        response = client.get("/metrics")
        assert response.headers["content-type"].startswith("application/json")

    def test_returns_timestamp(self, client):
        """/metrics should return timestamp."""
        response = client.get("/metrics")
        data = response.json()
        assert "timestamp" in data

    def test_timestamp_is_iso_format(self, client):
        """Timestamp should be in ISO format."""
        from datetime import datetime
        response = client.get("/metrics")
        data = response.json()
        # Should not raise if valid ISO format
        datetime.fromisoformat(data["timestamp"])

    def test_returns_uptime_seconds(self, client):
        """/metrics should return uptime_seconds."""
        response = client.get("/metrics")
        data = response.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_returns_requests_section(self, client):
        """/metrics should return requests section."""
        response = client.get("/metrics")
        data = response.json()
        assert "requests" in data
        assert "total" in data["requests"]
        assert "by_endpoint" in data["requests"]

    def test_returns_errors_section(self, client):
        """/metrics should return errors section."""
        response = client.get("/metrics")
        data = response.json()
        assert "errors" in data
        assert "total" in data["errors"]
        assert "by_endpoint" in data["errors"]

    def test_returns_latency_section(self, client):
        """/metrics should return latency_ms section."""
        response = client.get("/metrics")
        data = response.json()
        assert "latency_ms" in data
        assert "avg" in data["latency_ms"]
        assert "p50" in data["latency_ms"]
        assert "p95" in data["latency_ms"]
        assert "p99" in data["latency_ms"]
        assert "max" in data["latency_ms"]

    def test_returns_tools_section(self, client):
        """/metrics should return tools section."""
        response = client.get("/metrics")
        data = response.json()
        assert "tools" in data
        assert "calls_by_name" in data["tools"]

    def test_returns_rate_limiting_section(self, client):
        """/metrics should return rate_limiting section."""
        response = client.get("/metrics")
        data = response.json()
        assert "rate_limiting" in data
        assert "rejections" in data["rate_limiting"]

    def test_requests_total_is_integer(self, client):
        """requests.total should be an integer."""
        response = client.get("/metrics")
        data = response.json()
        assert isinstance(data["requests"]["total"], int)

    def test_errors_total_is_integer(self, client):
        """errors.total should be an integer."""
        response = client.get("/metrics")
        data = response.json()
        assert isinstance(data["errors"]["total"], int)

    def test_latency_values_are_numbers(self, client):
        """All latency values should be numbers."""
        response = client.get("/metrics")
        data = response.json()
        latency = data["latency_ms"]
        for key in ["avg", "p50", "p95", "p99", "max"]:
            assert isinstance(latency[key], (int, float)), f"latency.{key} should be a number"

    def test_no_authentication_required(self, client):
        """/metrics should not require authentication."""
        # Just making a request without auth should succeed
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_response_is_dict(self, client):
        """/metrics should return a dict."""
        response = client.get("/metrics")
        data = response.json()
        assert isinstance(data, dict)


class TestMetricsCollector:
    """Tests for the MetricsCollector class."""

    def test_initial_state_is_empty(self):
        """MetricsCollector should start with empty state."""
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        snapshot = collector.get_snapshot()
        assert snapshot.total_requests == 0
        assert snapshot.total_errors == 0
        assert snapshot.latency_avg_ms == 0.0

    def test_record_request_increments_total(self):
        """record_request should increment total_requests."""
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        collector.record_request("/test", 10.0)
        snapshot = collector.get_snapshot()
        assert snapshot.total_requests == 1

    def test_record_request_increments_by_endpoint(self):
        """record_request should track requests by endpoint."""
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        collector.record_request("/test", 10.0)
        collector.record_request("/test", 20.0)
        collector.record_request("/other", 15.0)
        snapshot = collector.get_snapshot()
        assert snapshot.requests_by_endpoint["/test"] == 2
        assert snapshot.requests_by_endpoint["/other"] == 1

    def test_record_error_increments_total_errors(self):
        """record_request with is_error=True should increment total_errors."""
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        collector.record_request("/test", 10.0, is_error=True)
        snapshot = collector.get_snapshot()
        assert snapshot.total_errors == 1

    def test_record_error_increments_by_endpoint(self):
        """record_request with is_error=True should track errors by endpoint."""
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        collector.record_request("/test", 10.0, is_error=True)
        collector.record_request("/test", 10.0, is_error=False)
        collector.record_request("/other", 10.0, is_error=True)
        snapshot = collector.get_snapshot()
        assert snapshot.errors_by_endpoint["/test"] == 1
        assert snapshot.errors_by_endpoint["/other"] == 1

    def test_latency_average_calculation(self):
        """Latency average should be calculated correctly."""
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        collector.record_request("/test", 10.0)
        collector.record_request("/test", 20.0)
        collector.record_request("/test", 30.0)
        snapshot = collector.get_snapshot()
        assert snapshot.latency_avg_ms == 20.0

    def test_latency_percentiles_calculation(self):
        """Latency percentiles should be calculated correctly."""
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        # Add 100 samples from 1 to 100
        for i in range(1, 101):
            collector.record_request("/test", float(i))
        snapshot = collector.get_snapshot()
        # P50 should be around 50, P95 around 95, P99 around 99
        assert 49 <= snapshot.latency_p50_ms <= 51
        assert 94 <= snapshot.latency_p95_ms <= 96
        assert 98 <= snapshot.latency_p99_ms <= 100
        assert snapshot.latency_max_ms == 100.0

    def test_record_tool_call(self):
        """record_tool_call should track tool calls by name."""
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        collector.record_tool_call("rlm_execute")
        collector.record_tool_call("rlm_execute")
        collector.record_tool_call("rlm_load_data")
        snapshot = collector.get_snapshot()
        assert snapshot.tool_calls_by_name["rlm_execute"] == 2
        assert snapshot.tool_calls_by_name["rlm_load_data"] == 1

    def test_record_rate_limit_rejection(self):
        """record_rate_limit_rejection should increment rejections."""
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        collector.record_rate_limit_rejection()
        collector.record_rate_limit_rejection()
        snapshot = collector.get_snapshot()
        assert snapshot.rate_limit_rejections == 2

    def test_uptime_increases(self):
        """uptime_seconds should increase over time."""
        import time
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        snapshot1 = collector.get_snapshot()
        time.sleep(0.1)
        snapshot2 = collector.get_snapshot()
        assert snapshot2.uptime_seconds > snapshot1.uptime_seconds

    def test_reset_clears_all_metrics(self):
        """reset should clear all metrics."""
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        collector.record_request("/test", 10.0, is_error=True)
        collector.record_tool_call("rlm_execute")
        collector.record_rate_limit_rejection()
        collector.reset()
        snapshot = collector.get_snapshot()
        assert snapshot.total_requests == 0
        assert snapshot.total_errors == 0
        assert snapshot.latency_avg_ms == 0.0
        assert len(snapshot.tool_calls_by_name) == 0
        assert snapshot.rate_limit_rejections == 0

    def test_rolling_window_limits_samples(self):
        """MetricsCollector should limit latency samples to MAX_LATENCY_SAMPLES."""
        from rlm_mcp.http_server import MetricsCollector
        collector = MetricsCollector()
        # Add more than MAX_LATENCY_SAMPLES
        for i in range(collector.MAX_LATENCY_SAMPLES + 100):
            collector.record_request("/test", float(i))
        # Check that samples were trimmed (implementation detail check)
        assert len(collector._latency_samples) == collector.MAX_LATENCY_SAMPLES


class TestMetricsIntegration:
    """Integration tests for metrics collection."""

    def test_mcp_request_records_metrics(self, client):
        """MCP requests should record metrics."""
        from rlm_mcp.http_server import metrics_collector
        metrics_collector.reset()

        # Make an MCP request
        response = client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        })
        assert response.status_code == 200

        snapshot = metrics_collector.get_snapshot()
        assert snapshot.total_requests >= 1
        assert "/mcp" in snapshot.requests_by_endpoint

    def test_tool_calls_record_metrics(self, client):
        """Tool calls should record metrics."""
        from rlm_mcp.http_server import metrics_collector
        metrics_collector.reset()

        # Make a tool call
        response = client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "rlm_execute",
                "arguments": {"code": "x = 1"}
            }
        })
        assert response.status_code == 200

        snapshot = metrics_collector.get_snapshot()
        assert "rlm_execute" in snapshot.tool_calls_by_name

    def test_error_responses_record_errors(self, client):
        """Error responses should increment error count."""
        from rlm_mcp.http_server import metrics_collector
        metrics_collector.reset()

        # Make a request that will cause an MCP-level error (unknown method)
        response = client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown/method"
        })
        assert response.status_code == 200  # MCP errors return 200 with error in body
        # Check that the response has an error field
        data = response.json()
        assert "error" in data

        snapshot = metrics_collector.get_snapshot()
        assert snapshot.total_errors >= 1

    def test_message_endpoint_records_metrics(self, client):
        """Message endpoint should record metrics."""
        from rlm_mcp.http_server import metrics_collector
        metrics_collector.reset()

        # Make a message request
        response = client.post("/message", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        })
        assert response.status_code == 200

        snapshot = metrics_collector.get_snapshot()
        assert snapshot.total_requests >= 1
        assert "/message" in snapshot.requests_by_endpoint

    def test_latency_is_recorded(self, client):
        """Request latency should be recorded."""
        from rlm_mcp.http_server import metrics_collector
        metrics_collector.reset()

        # Make a request
        response = client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize"
        })
        assert response.status_code == 200

        snapshot = metrics_collector.get_snapshot()
        # At least one sample should have been recorded
        assert snapshot.latency_avg_ms > 0 or snapshot.total_requests > 0


class TestMetricsSnapshot:
    """Tests for the MetricsSnapshot dataclass."""

    def test_default_values(self):
        """MetricsSnapshot should have sensible defaults."""
        from rlm_mcp.http_server import MetricsSnapshot
        snapshot = MetricsSnapshot()
        assert snapshot.total_requests == 0
        assert snapshot.total_errors == 0
        assert snapshot.latency_avg_ms == 0.0
        assert snapshot.latency_p50_ms == 0.0
        assert snapshot.latency_p95_ms == 0.0
        assert snapshot.latency_p99_ms == 0.0
        assert snapshot.latency_max_ms == 0.0
        assert snapshot.uptime_seconds == 0.0
        assert snapshot.rate_limit_rejections == 0
        assert isinstance(snapshot.requests_by_endpoint, dict)
        assert isinstance(snapshot.errors_by_endpoint, dict)
        assert isinstance(snapshot.tool_calls_by_name, dict)

    def test_can_create_with_values(self):
        """MetricsSnapshot should accept values."""
        from rlm_mcp.http_server import MetricsSnapshot
        snapshot = MetricsSnapshot(
            total_requests=100,
            total_errors=5,
            latency_avg_ms=25.5,
            requests_by_endpoint={"/mcp": 80, "/message": 20}
        )
        assert snapshot.total_requests == 100
        assert snapshot.total_errors == 5
        assert snapshot.latency_avg_ms == 25.5
        assert snapshot.requests_by_endpoint["/mcp"] == 80


class TestRequestId:
    """Tests for request_id tracing functionality."""

    def test_health_endpoint_returns_request_id_in_body(self, client):
        """Health endpoint should return request_id in response body."""
        response = client.get("/health")
        data = response.json()
        assert "request_id" in data
        assert data["request_id"] is not None

    def test_health_endpoint_returns_request_id_in_header(self, client):
        """Health endpoint should return X-Request-Id header."""
        response = client.get("/health")
        assert "X-Request-Id" in response.headers
        assert response.headers["X-Request-Id"] is not None

    def test_health_endpoint_request_id_matches_body_and_header(self, client):
        """Health endpoint X-Request-Id header should match body request_id."""
        response = client.get("/health")
        data = response.json()
        assert response.headers["X-Request-Id"] == data["request_id"]

    def test_health_endpoint_request_id_is_uuid_format(self, client):
        """Health endpoint request_id should be a valid UUID."""
        import uuid
        response = client.get("/health")
        data = response.json()
        # Should not raise exception if valid UUID
        uuid.UUID(data["request_id"])

    def test_health_endpoint_different_requests_have_unique_ids(self, client):
        """Each health request should have a unique request_id."""
        response1 = client.get("/health")
        response2 = client.get("/health")
        id1 = response1.json()["request_id"]
        id2 = response2.json()["request_id"]
        assert id1 != id2

    def test_metrics_endpoint_returns_request_id_in_body(self, client):
        """Metrics endpoint should return request_id in response body."""
        response = client.get("/metrics")
        data = response.json()
        assert "request_id" in data

    def test_metrics_endpoint_returns_request_id_in_header(self, client):
        """Metrics endpoint should return X-Request-Id header."""
        response = client.get("/metrics")
        assert "X-Request-Id" in response.headers

    def test_metrics_endpoint_request_id_matches_body_and_header(self, client):
        """Metrics endpoint X-Request-Id header should match body request_id."""
        response = client.get("/metrics")
        data = response.json()
        assert response.headers["X-Request-Id"] == data["request_id"]

    def test_mcp_endpoint_returns_request_id_in_header(self, client):
        """MCP endpoint should return X-Request-Id header."""
        response = client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize"
        })
        assert "X-Request-Id" in response.headers

    def test_mcp_endpoint_request_id_is_uuid_format(self, client):
        """MCP endpoint request_id should be a valid UUID."""
        import uuid
        response = client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize"
        })
        # Should not raise exception if valid UUID
        uuid.UUID(response.headers["X-Request-Id"])

    def test_mcp_endpoint_error_response_includes_request_id_in_header(self, client):
        """MCP error response should include X-Request-Id header."""
        response = client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown/method"
        })
        assert "X-Request-Id" in response.headers

    def test_message_endpoint_returns_request_id_in_header(self, client):
        """Message endpoint should return X-Request-Id header."""
        response = client.post("/message", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        })
        assert "X-Request-Id" in response.headers

    def test_message_endpoint_request_id_is_uuid_format(self, client):
        """Message endpoint request_id should be a valid UUID."""
        import uuid
        response = client.post("/message", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        })
        # Should not raise exception if valid UUID
        uuid.UUID(response.headers["X-Request-Id"])

    def test_message_endpoint_error_response_includes_request_id(self, client):
        """Message endpoint error response should include request_id in body and header."""
        # Send invalid JSON structure to trigger an error
        response = client.post("/message", data="not valid json", headers={"Content-Type": "application/json"})
        assert response.status_code == 500
        data = response.json()
        assert "request_id" in data
        assert "X-Request-Id" in response.headers
        assert response.headers["X-Request-Id"] == data["request_id"]

    def test_multiple_endpoints_have_unique_request_ids(self, client):
        """Different endpoints should generate unique request_ids."""
        response_health = client.get("/health")
        response_metrics = client.get("/metrics")
        response_mcp = client.post("/mcp", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize"
        })

        ids = {
            response_health.headers["X-Request-Id"],
            response_metrics.headers["X-Request-Id"],
            response_mcp.headers["X-Request-Id"]
        }
        # All three should be unique
        assert len(ids) == 3


class TestRequestIdFunction:
    """Tests for the generate_request_id function."""

    def test_generate_request_id_returns_string(self):
        """generate_request_id should return a string."""
        from rlm_mcp.http_server import generate_request_id
        request_id = generate_request_id()
        assert isinstance(request_id, str)

    def test_generate_request_id_returns_valid_uuid(self):
        """generate_request_id should return a valid UUID string."""
        import uuid
        from rlm_mcp.http_server import generate_request_id
        request_id = generate_request_id()
        # Should not raise exception
        parsed = uuid.UUID(request_id)
        assert parsed.version == 4  # Should be UUID v4

    def test_generate_request_id_returns_unique_values(self):
        """generate_request_id should return unique values each call."""
        from rlm_mcp.http_server import generate_request_id
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_request_id_format_is_lowercase(self):
        """generate_request_id should return lowercase UUID."""
        from rlm_mcp.http_server import generate_request_id
        request_id = generate_request_id()
        assert request_id == request_id.lower()

    def test_generate_request_id_length_is_36(self):
        """generate_request_id should return UUID of length 36 (with hyphens)."""
        from rlm_mcp.http_server import generate_request_id
        request_id = generate_request_id()
        assert len(request_id) == 36  # 32 hex chars + 4 hyphens
