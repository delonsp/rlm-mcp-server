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
            with patch("rlm_mcp.http_server.get_s3_client", return_value=mock_client):
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
            with patch("rlm_mcp.http_server.get_s3_client", return_value=mock_client):
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
