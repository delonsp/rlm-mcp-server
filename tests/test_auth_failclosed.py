"""
Auth fail-closed (P0 footgun da avaliação Codex 2026-06-02, corrigido
2026-06-06): RLM_API_KEY vazia NÃO é mais open-by-default — sem chave, 401
em tudo, salvo break-glass explícito RLM_ALLOW_ANON=true (que o conftest
usa para os demais testes).
"""
from fastapi.testclient import TestClient

from rlm_mcp import http_server as hs

client = TestClient(hs.app)
TOOLS_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


def test_no_key_no_anon_fails_closed(monkeypatch):
    monkeypatch.setattr(hs, "API_KEY", "")
    monkeypatch.setattr(hs, "ALLOW_ANON", False)
    resp = client.post("/mcp", json=TOOLS_LIST)
    assert resp.status_code == 401
    assert "RLM_API_KEY" in resp.json()["detail"]


def test_no_key_with_explicit_anon_optin_allows(monkeypatch):
    monkeypatch.setattr(hs, "API_KEY", "")
    monkeypatch.setattr(hs, "ALLOW_ANON", True)
    resp = client.post("/mcp", json=TOOLS_LIST)
    assert resp.status_code == 200


def test_key_set_correct_bearer_allows(monkeypatch):
    monkeypatch.setattr(hs, "API_KEY", "segredo-teste")
    monkeypatch.setattr(hs, "ALLOW_ANON", False)
    resp = client.post("/mcp", json=TOOLS_LIST,
                       headers={"Authorization": "Bearer segredo-teste"})
    assert resp.status_code == 200


def test_key_set_wrong_bearer_401(monkeypatch):
    monkeypatch.setattr(hs, "API_KEY", "segredo-teste")
    resp = client.post("/mcp", json=TOOLS_LIST,
                       headers={"Authorization": "Bearer errado"})
    assert resp.status_code == 401


def test_key_set_anon_flag_does_not_bypass(monkeypatch):
    """ALLOW_ANON só vale quando NÃO há chave — com chave, exige Bearer."""
    monkeypatch.setattr(hs, "API_KEY", "segredo-teste")
    monkeypatch.setattr(hs, "ALLOW_ANON", True)
    resp = client.post("/mcp", json=TOOLS_LIST)
    assert resp.status_code == 401


def test_health_never_requires_auth(monkeypatch):
    monkeypatch.setattr(hs, "API_KEY", "")
    monkeypatch.setattr(hs, "ALLOW_ANON", False)
    assert client.get("/health").status_code == 200
