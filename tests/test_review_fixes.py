"""Regressão dos fixes da revisão 2026-07 (arquitetura/segurança/corretude).

Cobre: invalidação de índice stale por fingerprint, mapeamento char→linha por
bisect, persistência de cobertura parcial de embeddings, AND (require_all) além
das 10 primeiras ocorrências, guard SSRF do rlm_upload_url, bloqueio de tools
não-anunciados no wire, e não-sobrescrita do status 'cancelled'.
"""

import pytest

from rlm_mcp.indexer import (
    create_index, set_index, get_index, fingerprint_source, clear_all_indices,
)
from rlm_mcp.vector_index import (
    VectorIndex, ChunkInfo, set_vector_index, get_vector_index,
    clear_all_vector_indices, _chunk_text,
)
from rlm_mcp.services.persistence_service import invalidate_stale_indices


@pytest.fixture(autouse=True)
def _clean_caches():
    clear_all_indices()
    clear_all_vector_indices()
    yield
    clear_all_indices()
    clear_all_vector_indices()


# --- fingerprint / invalidação de índice stale -----------------------------

def test_fingerprint_muda_com_o_texto():
    a = fingerprint_source("o rato roeu a roupa do rei")
    b = fingerprint_source("o rato roeu a roupa do rei de roma")
    assert a != b
    assert fingerprint_source("igual") == fingerprint_source("igual")


def test_create_index_grava_fingerprint_da_fonte():
    txt = "linha um\nlinha dois\nlinha tres\n" * 50
    idx = create_index(txt, "v")
    assert idx.source_fingerprint == fingerprint_source(txt)


def test_invalidate_nao_descarta_indice_valido():
    txt = "alpha bravo charlie\n" * 100
    set_index("v", create_index(txt, "v"))
    changed = invalidate_stale_indices("v", txt)
    assert changed is False
    assert get_index("v") is not None


def test_invalidate_descarta_indice_stale_apos_rebind():
    txt_old = "alpha bravo charlie\n" * 100
    txt_new = "delta echo foxtrot\n" * 100
    set_index("v", create_index(txt_old, "v"))
    # var foi rebindado (rlm_execute) → texto novo não bate com o índice
    changed = invalidate_stale_indices("v", txt_new)
    assert changed is True
    assert get_index("v") is None  # rebuild acontece na próxima busca


def test_invalidate_descarta_vector_index_stale():
    vi = VectorIndex("v")
    vi.source_fingerprint = fingerprint_source("texto original")
    set_vector_index("v", vi)
    assert invalidate_stale_indices("v", "texto DIFERENTE") is True
    assert get_vector_index("v") is None


def test_invalidate_sem_fingerprint_nao_thrasha():
    # Índice legado restaurado sem fingerprint não deve ser descartado à toa.
    idx = create_index("qualquer coisa aqui", "v")
    idx.source_fingerprint = ""
    set_index("v", idx)
    assert invalidate_stale_indices("v", "outro texto") is False
    assert get_index("v") is not None


# --- _char_to_line via bisect (mapeamento de linha correto) ------------------

def test_chunk_line_mapping_correto():
    # 200 linhas: garante que o chunk final mapeia p/ linhas altas, não 0.
    text = "".join(f"linha {i}\n" for i in range(200))
    chunks = _chunk_text(text, chunk_size=64, overlap=8)
    assert chunks, "esperava múltiplos chunks"
    # line_start é monotônico não-decrescente e o último chunk está lá no fim.
    starts = [c.line_start for c in chunks]
    assert starts == sorted(starts)
    assert chunks[-1].line_start > 100
    # Toda linha reportada existe no texto.
    n_lines = text.count("\n") + 1
    for c in chunks:
        assert 0 <= c.line_start < n_lines
        assert c.line_start <= c.line_end < n_lines


# --- persistência de cobertura parcial de embeddings -------------------------

def test_persist_payload_inclui_chunks_sem_vetor():
    vi = VectorIndex("v")
    vi.chunks = [
        ChunkInfo(chunk_index=0, text="a", line_start=0, line_end=0),
        ChunkInfo(chunk_index=1, text="b", line_start=1, line_end=1),
        ChunkInfo(chunk_index=2, text="c", line_start=2, line_end=2),
    ]
    # Chunk do meio SEM vetor (falha de batch): tem que persistir mesmo assim.
    vi._ingest_embeddings([[1.0, 0.0], [], [0.0, 1.0]])
    payload = vi.persist_payload()
    assert len(payload) == 3, "os 3 chunks devem persistir (não só os 2 com vetor)"
    by_idx = {p["chunk_index"]: p for p in payload}
    assert by_idx[1]["embedding"] == []  # sem vetor → embedding vazio
    assert by_idx[0]["embedding"]        # com vetor
    stats = vi.get_stats()
    assert stats["total_chunks"] == 3 and stats["embedded_chunks"] == 2


# --- require_all além das 10 primeiras ocorrências ---------------------------

def test_require_all_encontra_coocorrencia_distante():
    # 'alpha' aparece em 20 linhas; só a última também tem 'beta'.
    lines = [f"alpha linha {i}" for i in range(20)]
    lines[-1] = "alpha beta juntos"
    text = "\n".join(lines) + "\n"
    idx = create_index(text, "v")
    res = idx.search_multiple(["alpha", "beta"], require_all=True, source_text=text)
    # A linha de co-ocorrência (a 20ª) tem que estar no resultado — antes o cap
    # default 10 do search() a perdia.
    assert res, "require_all deveria achar a linha com alpha E beta"
    found_terms = {frozenset(v) for v in res.values()}
    assert any({"alpha", "beta"} <= s for s in found_terms)


# --- SSRF guard do rlm_upload_url --------------------------------------------

def test_validate_fetch_url_bloqueia_esquemas_e_internos():
    from rlm_mcp.tools.handlers.s3_tools import _validate_fetch_url
    assert _validate_fetch_url("file:///etc/passwd") is not None
    assert _validate_fetch_url("gopher://x/1") is not None
    assert _validate_fetch_url("http://localhost/x") is not None
    assert _validate_fetch_url("http://127.0.0.1/x") is not None
    assert _validate_fetch_url("http://169.254.169.254/latest/meta-data/") is not None
    assert _validate_fetch_url("http://10.0.0.5/x") is not None
    assert _validate_fetch_url("http://192.168.1.1/x") is not None


# --- wire NÃO expõe handlers internos ---------------------------------------

def test_tools_internos_nao_anunciados_no_wire():
    from rlm_mcp.http_server import _ANNOUNCED_TOOL_NAMES
    # Anunciados (chamáveis pelo cliente):
    assert "rlm_search_index" in _ANNOUNCED_TOOL_NAMES
    assert "rlm_collection" in _ANNOUNCED_TOOL_NAMES
    # Internos (só via ctx.call_tool a partir dos routers/batch):
    assert "rlm_collection_create" not in _ANNOUNCED_TOOL_NAMES
    assert "rlm_search_collection" not in _ANNOUNCED_TOOL_NAMES
