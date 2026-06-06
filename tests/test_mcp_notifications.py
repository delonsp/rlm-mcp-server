"""
Contrato MCP de notificações (incidente 2026-06-06: notifications/cancelled
recebia "Method not found" — violação de protocolo JSON-RPC: notificação
NUNCA recebe resposta, nem de erro).
"""
from fastapi.testclient import TestClient

from rlm_mcp import http_server as hs

# Sem context manager: lifespan (auto-restore/forkserver) não roda — desnecessário
client = TestClient(hs.app)


def test_notifications_cancelled_returns_202_empty_body():
    resp = client.post("/message", json={
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": {"requestId": 42, "reason": "user-cancel"},
    })
    assert resp.status_code == 202
    assert resp.content == b""


def test_notifications_initialized_returns_202():
    resp = client.post("/message", json={
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    })
    assert resp.status_code == 202
    assert resp.content == b""


def test_unknown_notification_returns_202_not_method_not_found():
    resp = client.post("/message", json={
        "jsonrpc": "2.0",
        "method": "notifications/algumacoisa_futura",
    })
    assert resp.status_code == 202
    assert resp.content == b""


def test_unknown_method_with_id_still_returns_method_not_found():
    """Métodos desconhecidos NÃO-notificação continuam com erro -32601."""
    resp = client.post("/message", json={"jsonrpc": "2.0", "id": 7, "method": "bogus/method"})
    assert resp.status_code == 200
    assert resp.json()["error"]["code"] == -32601


def test_handle_mcp_request_returns_none_for_any_notification():
    for method in ("notifications/cancelled", "notifications/initialized", "notifications/x"):
        req = hs.MCPRequest(jsonrpc="2.0", method=method, params={})
        assert hs.handle_mcp_request(req) is None, method


def test_mcp_endpoint_notification_returns_202():
    """O /mcp (Streamable HTTP) também: notificação → 202, nunca erro."""
    resp = client.post("/mcp", json={
        "jsonrpc": "2.0",
        "method": "notifications/cancelled",
        "params": {"requestId": 1},
    })
    assert resp.status_code == 202


def test_initialize_echoes_supported_version():
    resp = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "0"}},
    })
    assert resp.json()["result"]["protocolVersion"] == "2025-03-26"


def test_initialize_falls_back_on_unknown_version():
    resp = client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "1999-01-01", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "0"}},
    })
    assert resp.json()["result"]["protocolVersion"] == "2024-11-05"
