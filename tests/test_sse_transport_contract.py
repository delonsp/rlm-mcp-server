"""
Contrato do transporte SSE (/message + /sse) — regressões do incidente
2026-06-06 (rlm_collection "pendurado" 37min): session_id stale recebia
fallback silencioso (tool executava e a resposta ia no body do POST, que o
cliente SSE ignora por spec). Agora: 404 imediato, sem executar.
"""
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from rlm_mcp import http_server as hs
from asgi_sse_driver import SseDriver

client = TestClient(hs.app)


# ---------------------------------------------------------------------------
# P0: session_id stale → 404 fail-fast
# ---------------------------------------------------------------------------

def test_stale_session_returns_404_without_executing_tool(monkeypatch):
    executed = []
    monkeypatch.setattr(
        hs, "handle_mcp_request_locked",
        lambda req, client_id=None: executed.append(req.method),
    )
    resp = client.post(
        "/message?session_id=sessao-que-morreu-no-restart",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": "rlm_collection", "arguments": {"action": "list"}}},
    )
    assert resp.status_code == 404
    assert executed == [], "guard deve rodar ANTES de qualquer execução"
    assert "session" in resp.json()["error"].lower()


def test_stale_session_404_happens_before_body_parse():
    """Guard roda antes do request.json(): body inválido nem é tocado."""
    resp = client.post(
        "/message?session_id=stale",
        content=b"isto nem e json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 404


def test_empty_session_id_is_treated_as_stale_not_direct():
    """session_id="" é 'fornecido e inválido', não modo direto."""
    resp = client.post(
        "/message?session_id=",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert resp.status_code == 404


def test_direct_mode_without_session_id_still_works():
    """Sem session_id o modo request/response direto continua (compat)."""
    resp = client.post("/message", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert resp.status_code == 200
    assert "tools" in resp.json()["result"]


# ---------------------------------------------------------------------------
# Sessão viva: resposta via fila SSE (202), nunca no body
# ---------------------------------------------------------------------------

async def test_live_session_returns_202_and_response_arrives_via_stream():
    driver = SseDriver(hs.app)
    await driver.start()
    try:
        sid = await driver.session_id()
        assert sid in hs.sse_sessions

        transport = httpx.ASGITransport(app=hs.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            resp = await ac.post(
                f"/message?session_id={sid}",
                json={"jsonrpc": "2.0", "id": 5, "method": "tools/list"},
            )
        assert resp.status_code == 202
        assert resp.content == b""

        # A resposta JSON-RPC tem que chegar pelo STREAM
        deadline = asyncio.get_event_loop().time() + 5
        body = b""
        while asyncio.get_event_loop().time() < deadline:
            msg = await driver.next_chunk()
            if msg["type"] == "http.response.body":
                body += msg.get("body", b"")
                if b"event: message" in body:
                    break
        assert b"event: message" in body
        assert b'"id": 5' in body or b'"id":5' in body
    finally:
        await driver.stop()


def test_session_evicted_during_request_falls_back_to_direct_response(monkeypatch):
    """Eviction na janela do threadpool → resposta no body (não descartar
    resultado já computado), nunca 404 neste ponto."""
    hs.register_sse_session("sid-evict-race", "1.2.3.4")

    def evil_handler(req, client_id=None):
        # simula a sessão morrendo enquanto a tool rodava
        hs.sse_sessions.pop("sid-evict-race", None)
        return hs.MCPResponse(id=req.id, result={"ok": True})

    monkeypatch.setattr(hs, "handle_mcp_request_locked", evil_handler)
    resp = client.post(
        "/message?session_id=sid-evict-race",
        json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
    )
    assert resp.status_code == 200
    assert resp.json()["result"] == {"ok": True}


# ---------------------------------------------------------------------------
# Origin (anti DNS-rebinding, MUST da spec Streamable HTTP)
# ---------------------------------------------------------------------------

def test_unknown_origin_rejected_403():
    resp = client.post(
        "/message",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Origin": "https://evil.example"},
    )
    assert resp.status_code == 403


def test_allowed_origin_passes():
    resp = client.post(
        "/message",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Origin": "https://rlm.drsolution.online"},
    )
    assert resp.status_code == 200


def test_no_origin_passes():
    """Clientes CLI (Claude Code, curl) não enviam Origin."""
    resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert resp.status_code == 200
