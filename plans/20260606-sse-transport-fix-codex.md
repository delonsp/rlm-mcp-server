# Fix: classe de bug do "hang de 37min" (transporte SSE/MCP) — plano consolidado

**Data**: 2026-06-06 · **Workflow**: /plan-code-codex (Codex propôs → Claude criticou → 4 auditores adversariais verificaram premissas)
**Incidente**: rlm_collection "pendurado" 37min em 2026-06-06 02:49–03:26 UTC.

## Diagnóstico (fechado, provado nos logs)

1. Sessão SSE `ef2cdf2b` criada 06-05 01:50:48; servidor reiniciou 06-05 07:03:27 (docker inspect) → `sse_sessions` (dict em memória) zerou silenciosamente.
2. 06-06 02:49:19: cliente POSTou tools/call com session_id stale → fallback silencioso do `/message` (http_server.py:2650): executou a tool em **4ms** e devolveu o resultado no **body do POST (200 OK)** em vez da fila SSE (202).
3. Cliente SSE ignora body por spec → esperou 37min num stream morto. `tools/list` (03:25:19) teve a mesma sina. `notifications/cancelled` (4×) recebeu `Method not found`.
4. **Não houve hang server-side.** Healthchecks 200 o tempo todo.

## Plano (4 fases)

### Fase A — Hotfix protocolo (P0+P1)
- **A1 (P0)**: guard no TOPO do `message_endpoint`, antes do parse: `session_id is not None and session_id not in sse_sessions` → **404** imediato, sem executar tool. Modo direto continua quando session_id AUSENTE. 404 **apenas** neste ponto (auditoria: SDK TS faz throw imediato em POST !ok → fail-fast confirmado; Claude Code 2.1.167 embute esse código; SDK Python oficial também retorna 404).
- **A2 (P1)**: `method.startswith("notifications/")` → `return None` genérico em `handle_mcp_request` (cobre `cancelled`; logar method + requestId/reason em INFO). Spec: cancelamento real é opcional (MAY ignore); 202 sem body é compliant.
- **A3 (P2-obs)**: logs INFO no `/message` e `/mcp` com método MCP, tool name e latência **embutidos na mensagem** (formatter text default ignora `extra`; manter `extra` p/ modo json).

### Fase B — Event loop (P1)
- **B1**: `await run_in_threadpool(handle_mcp_request_locked, ...)` em `/message` e `/mcp`. **Só** o handler vai pro threadpool; `queue.put` FICA no event loop (asyncio.Queue não é thread-safe).
- **B2**: lock global `threading.RLock` no wrapper (auditoria: sem dependência de asyncio nos handlers; sem ciclo de deadlock; RateLimitExceeded propaga pelo run_in_threadpool; estritamente melhor que hoje — hoje o event loop inteiro trava).
- **B3**: pós-threadpool, `entry = sse_sessions.get(sid)` + `entry.queue.put_nowait(...)` ADJACENTES (race zero no loop). Se `entry is None` (evictada durante tool call) → **fall-through pra JSONResponse direta**, NÃO 404 (não descartar resultado caro).
- **B4** (races pré-existentes expostas pela auditoria): `threading.Lock` interno no `SlidingWindowRateLimiter` (acessado de worker thread em :2240 sem lock); `repl.load_data` sob `_execute_lock` (mutação de `variables`/`metadata` por task workers sem lock, repl.py:815-833).
- **B5**: documentar caveat: RLock global NÃO cobre as 3 threads do TaskManager (bypass pré-existente); timeout SIGALRM do modo `inprocess` morre silenciosamente fora da main thread (só afeta break-glass).

### Fase C — Registry de sessões (P2)
- **C1**: `SseSession` dataclass: `queue, created_at, last_seen, client_key`. Touch points confirmados: só :2540 (construção) e :2651 (.queue) mudam; :2572/:2607/:2650 intactos.
- **C2**: política de eviction (auditoria REFUTOU cap global evict-oldest): **cap POR CLIENTE** (default 8; key = primeiro IP do `X-Forwarded-For`, fallback `request.client.host`) com evict-oldest intra-cliente + **cap global 256** backstop + **TTL idle 24h** (sweep oportunista no /sse connect).
- **C3**: eviction = `pop` do dict + `queue.put_nowait(_SENTINEL)`; `_SENTINEL = object()` module-level; generator checa `message is _SENTINEL` ENTRE o get e o json.dumps → break → finally (pop duplo é seguro, FIFO drena respostas pendentes antes). Eviction roda SÓ no event loop.
- **C4**: rate-limit no próprio `GET /sse` (reusar sse_rate_limiter com client_key) — ataca a causa raiz do reconnect-loop.
- **C5**: `last_seen` atualizado no guard do `/message` e no put.
- Envs novos: `RLM_SSE_SESSIONS_PER_CLIENT=8`, `RLM_SSE_SESSION_MAX=256`, `RLM_SSE_SESSION_TTL_SECONDS=86400`.

### Fase D — Streamable HTTP (validação + compliance mínima)
- **D1**: echo da `protocolVersion` pedida quando suportada (`2024-11-05`, `2025-03-26`, `2025-06-18`); senão responde `2024-11-05`. (Spec: MUST ecoar se suporta.)
- **D2**: validação de `Origin` (MUST anti-DNS-rebinding): se header presente e fora da allowlist (CORS_ORIGINS + rlm.drsolution.online) → 403. Sem Origin (CLI) → passa.
- **D3**: garantir que `/mcp` nunca retorna 404 (semântica reservada: cliente compliant entraria em loop de re-initialize).
- **D4**: validar Claude Code `--transport http` → `https://rlm.drsolution.online/mcp` live; se verde, migrar config do cliente (mantendo /sse funcionando p/ compat).

### Testes (3 arquivos novos + conftest)
- **conftest.py** (primeiro do repo; pytest importa antes dos test modules): env no TOPO module-level ANTES de qualquer import rlm_mcp (`RLM_PERSIST_DIR=tmpdir`, `RLM_SANDBOX_MODE=inprocess`, pop de `RLM_API_KEY`/`OPENAI_API_KEY`/`MINIO_*`, `RLM_LOG_LEVEL=WARNING`); fixture autouse de reset (repl.variables, metadata, sse_sessions, metrics, índices).
- **test_mcp_notifications.py**: TestClient plain (sem CM; lifespan desnecessário). cancelled→202 body vazio; unknown notification→202; unknown method com id→erro -32601.
- **test_sse_transport_contract.py**: ⚠️ GET /sse é INTESTÁVEL via TestClient/httpx.ASGITransport (bufferizam stream infinito → deadlock, provado). Usar **driver ASGI cru** (asyncio.create_task(app(scope, receive, send)) + Queue; padrão validado pelo auditor). Casos: stale session→404 sem executar tool (monkeypatch); sem session_id→JSON direto; sessão viva→202 + evento na queue.
- **test_sse_session_registry.py**: driver cru; cap por cliente, TTL, sentinel fecha generator, eviction não derruba resposta pendente.
- Gotchas: env congelado no import (patch de atributo `hs.API_KEY`, nunca setenv pós-import); auth open-by-default confirmado (API_KEY vazio → passa); `with TestClient(app)` só quando precisar do lifespan.

### Verificação live pós-deploy
(a) `POST /message?session_id=inexistente` → 404; (b) `notifications/cancelled` → 202; (c) `/health` responde durante `rlm_execute` com `time.sleep(20)`; (d) sessão SSE nova do Claude Code funciona; (e) `initialize` no `/mcp` ecoa versão pedida; (f) métricas/logs INFO mostram tool name + latência.

## Apêndices
- Plano original do Codex: `/tmp/codex-plan.md` (gerado 2026-06-06)
- Auditoria adversarial (4 agentes, wf_2dacdfe4-063): premissa central confirmada com fonte primária (typescript-sdk sse.ts:275-313, protocol.ts:1248-1251, binário Claude Code 2.1.167, python-sdk sse.py, spec 2025-03-26 transports/lifecycle/cancellation); eviction global refutada; bloqueador de TestClient no /sse provado empiricamente.
