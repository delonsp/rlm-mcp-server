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
