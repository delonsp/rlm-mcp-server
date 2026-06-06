"""
Config compartilhada dos testes.

IMPORTANTE: as env vars dos módulos rlm_mcp são lidas no IMPORT (congeladas —
monkeypatch.setenv depois é inócuo). O pytest importa este conftest ANTES de
qualquer test module, então o setup de env TEM que ficar aqui, no topo, em
nível de módulo. Para mudar um knob num teste específico, patchar o ATRIBUTO
do módulo (ex.: monkeypatch.setattr(hs, "API_KEY", "x")), nunca o env.
"""
import os
import sys
import tempfile

# Persistência SQLite em tmpdir: sem isto, no Mac o default /persist falha
# (Errno 30, degrada com "⚠️ Erro de persistência" poluindo respostas) e em
# CI rodando como root criaria /persist real vazando estado entre runs.
os.environ.setdefault("RLM_PERSIST_DIR", tempfile.mkdtemp(prefix="rlm-test-persist-"))
# inprocess evita spawn do forkserver no lifespan (testes de transporte não
# usam rlm_execute; test_sandbox.py força subprocess no próprio módulo).
os.environ.setdefault("RLM_SANDBOX_MODE", "inprocess")
# Auth: fail-closed desde 2026-06-06 — testes usam o break-glass explícito.
os.environ.pop("RLM_API_KEY", None)
os.environ.setdefault("RLM_ALLOW_ANON", "true")
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("DEEPSEEK_API_KEY", None)
for _var in [v for v in os.environ if v.startswith("MINIO_")]:
    os.environ.pop(_var)
os.environ.setdefault("RLM_LOG_LEVEL", "WARNING")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))  # p/ importar asgi_sse_driver

import pytest


@pytest.fixture(autouse=True)
def _isolate_module_state():
    """Reseta o estado module-level entre testes (vaza entre arquivos).

    Só toca nos módulos que JÁ foram importados — não força import de
    http_server em arquivos de teste que não o usam (test_sandbox etc.).
    """
    yield
    hs = sys.modules.get("rlm_mcp.http_server")
    if hs is not None:
        hs.repl.variables.clear()
        hs.repl.variable_metadata.clear()
        hs.sse_sessions.clear()
        hs.metrics_collector.reset()
        hs.sse_rate_limiter._buckets.clear()
        hs.upload_rate_limiter._buckets.clear()
    indexer = sys.modules.get("rlm_mcp.indexer")
    if indexer is not None:
        indexer.clear_all_indices()
    vector_index = sys.modules.get("rlm_mcp.vector_index")
    if vector_index is not None:
        vector_index.clear_all_vector_indices()
    # Singleton de persistência: reset força reabrir o SQLite do tmpdir no
    # próximo get_persistence() (o ARQUIVO persiste na sessão — testes que
    # criam coleções devem usar nomes únicos).
    persistence_mod = sys.modules.get("rlm_mcp.persistence")
    if persistence_mod is not None:
        persistence_mod._persistence = None
