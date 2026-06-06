"""
Registry de sessões SSE: cap por cliente, cap global, TTL e eviction
cooperativa via sentinel (fecha o generator sem vazar frame pro cliente).
"""
import asyncio
import time

from rlm_mcp import http_server as hs
from rlm_mcp.rate_limiter import SlidingWindowRateLimiter
from asgi_sse_driver import SseDriver


# ---------------------------------------------------------------------------
# Caps e TTL (lógica pura do registry, sem HTTP)
# ---------------------------------------------------------------------------

def test_per_client_cap_evicts_oldest_of_same_client():
    queues = [hs.register_sse_session(f"s{i}", "1.2.3.4")
              for i in range(hs.SSE_SESSIONS_PER_CLIENT)]
    assert len(hs.sse_sessions) == hs.SSE_SESSIONS_PER_CLIENT

    hs.register_sse_session("s-novo", "1.2.3.4")
    assert "s0" not in hs.sse_sessions, "mais antiga do MESMO cliente sai"
    assert "s-novo" in hs.sse_sessions
    assert len(hs.sse_sessions) == hs.SSE_SESSIONS_PER_CLIENT
    # evictada recebeu o sentinel (identidade, não igualdade)
    assert queues[0].get_nowait() is hs._SSE_EVICTION_SENTINEL


def test_per_client_cap_never_evicts_other_clients():
    """Política da auditoria: reconnect-loop de um cliente NÃO pode expulsar
    a sessão legítima de outro (evict-oldest global faria churn permanente)."""
    hs.register_sse_session("legitima", "9.9.9.9")
    for i in range(hs.SSE_SESSIONS_PER_CLIENT * 2):
        hs.register_sse_session(f"flood{i}", "6.6.6.6")
    assert "legitima" in hs.sse_sessions
    flood_alive = [s for s, e in hs.sse_sessions.items() if e.client_key == "6.6.6.6"]
    assert len(flood_alive) == hs.SSE_SESSIONS_PER_CLIENT


def test_ttl_sweep_evicts_idle_zombie():
    q = hs.register_sse_session("zumbi", "1.2.3.4")
    hs.sse_sessions["zumbi"].last_seen = time.time() - hs.SSE_SESSION_TTL_SECONDS - 10
    hs.register_sse_session("gatilho", "5.6.7.8")
    assert "zumbi" not in hs.sse_sessions
    assert q.get_nowait() is hs._SSE_EVICTION_SENTINEL


def test_global_cap_backstop(monkeypatch):
    monkeypatch.setattr(hs, "SSE_SESSION_MAX", 4)
    monkeypatch.setattr(hs, "SSE_SESSIONS_PER_CLIENT", 2)
    for i in range(2):
        hs.register_sse_session(f"a{i}", "1.1.1.1")
    for i in range(2):
        hs.register_sse_session(f"b{i}", "2.2.2.2")
    assert len(hs.sse_sessions) == 4
    hs.register_sse_session("c0", "3.3.3.3")
    assert len(hs.sse_sessions) == 4, "cap global segura o agregado"
    assert "a0" not in hs.sse_sessions, "mais antiga do agregado saiu"


def test_evict_unknown_session_is_noop():
    hs._evict_sse_session("nao-existe", "test")  # não pode levantar


# ---------------------------------------------------------------------------
# Eviction fecha o stream de verdade (fluxo HTTP completo via driver cru)
# ---------------------------------------------------------------------------

async def test_eviction_closes_live_stream():
    driver = SseDriver(hs.app)
    await driver.start()
    try:
        sid = await driver.session_id()
        assert sid in hs.sse_sessions
        hs._evict_sse_session(sid, "test")
        # generator deve dar break → response completa → task termina
        await asyncio.wait_for(driver.task, timeout=5)
        assert sid not in hs.sse_sessions
    finally:
        await driver.stop()


async def test_pending_response_delivered_before_sentinel_close():
    """FIFO: resposta já enfileirada é entregue ANTES do sentinel fechar."""
    driver = SseDriver(hs.app)
    await driver.start()
    try:
        sid = await driver.session_id()
        entry = hs.sse_sessions[sid]
        entry.queue.put_nowait({"jsonrpc": "2.0", "id": 9, "result": {}})
        hs._evict_sse_session(sid, "test")

        body = b""
        while True:
            try:
                msg = await driver.next_chunk(timeout=5)
            except asyncio.TimeoutError:
                break
            if msg["type"] == "http.response.body":
                body += msg.get("body", b"")
                if not msg.get("more_body", True):
                    break
        assert b'"id": 9' in body or b'"id":9' in body, "resposta pendente foi perdida"
        assert b"object object" not in body.lower(), "sentinel vazou pro fio"
        await asyncio.wait_for(driver.task, timeout=5)
    finally:
        await driver.stop()


async def test_sse_connect_rate_limited(monkeypatch):
    """Reconnect-loop (burst real: 12 conexões/s) é barrado no próprio /sse."""
    monkeypatch.setattr(hs, "sse_rate_limiter",
                        SlidingWindowRateLimiter(max_requests=1, window_seconds=60))
    d1 = SseDriver(hs.app)
    await d1.start()
    try:
        assert await d1.status() == 200
        d2 = SseDriver(hs.app)
        await d2.start()
        try:
            assert await d2.status() == 429
        finally:
            await d2.stop()
    finally:
        await d1.stop()


async def test_client_disconnect_still_cleans_registry():
    """Caminho clássico (finally do generator) continua limpando o dict."""
    driver = SseDriver(hs.app)
    await driver.start()
    sid = await driver.session_id()
    assert sid in hs.sse_sessions
    await driver.stop()  # cancela a task → CancelledError → finally
    # o finally roda no cancel; dá um tick pro loop processar
    await asyncio.sleep(0.05)
    assert sid not in hs.sse_sessions
