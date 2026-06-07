"""
Canários da cobertura de embeddings exposta no rlm_var_info (emb:X/Y).

Motivação (2026-06-06): o bug do batching deixava vars com cobertura
PARCIAL de embeddings sem nenhum sinal observável de fora — a busca
semântica "funcionava" enxergando 0,05% do texto. Agora o var_info expõe
emb:X/Y e o harness live exige X==Y em todo índice vetorial.
"""

import re

from rlm_mcp import http_server as hs
from rlm_mcp import response_formatter as fmt
from rlm_mcp.response_formatter import Verbosity
from rlm_mcp.vector_index import VectorIndex, _chunk_text, set_vector_index


def _make_vi(name: str, text: str, holes: int = 0) -> VectorIndex:
    """VectorIndex com embeddings fabricados; `holes` chunks ficam sem vetor."""
    vi = VectorIndex(name)
    vi.chunks = _chunk_text(text, chunk_size=64, overlap=8)
    vectors = [[0.1, 0.2, 0.3] for _ in vi.chunks]
    for i in range(min(holes, len(vectors))):
        vectors[i] = []
    vi._ingest_embeddings(vectors)
    return vi


def _load(name: str, data: str):
    res = hs.call_tool("rlm_load_data", {"name": name, "data": data})
    assert not res.get("isError"), res


# A verbosity pode variar conforme a ordem da suite (outros testes mexem no
# env) — o assert aceita os dois formatos: compact "emb:X/Y" e normal
# "Embeddings: X/Y chunks".
_EMB_RE = re.compile(r"(?:emb:|Embeddings: )(\d+)/(\d+)")
_PARTIAL_RE = re.compile(r"⚠️parcial|cobertura parcial")


def test_var_info_mostra_cobertura_total():
    text = "conteudo de teste com palavras variadas para chunkar " * 20
    _load("_qa_vi_full", text)
    set_vector_index("_qa_vi_full", _make_vi("_qa_vi_full", text))

    res = hs.call_tool("rlm_var_info", {"name": "_qa_vi_full"})
    out = res["content"][0]["text"]
    m = _EMB_RE.search(out)
    assert m, f"cobertura ausente do var_info: {out}"
    assert m.group(1) == m.group(2), f"cobertura deveria ser total: {out}"
    assert not _PARTIAL_RE.search(out)


def test_var_info_marca_cobertura_parcial():
    text = "outro conteudo de teste para chunkar em pedacos pequenos " * 20
    _load("_qa_vi_part", text)
    set_vector_index("_qa_vi_part", _make_vi("_qa_vi_part", text, holes=3))

    res = hs.call_tool("rlm_var_info", {"name": "_qa_vi_part"})
    out = res["content"][0]["text"]
    m = _EMB_RE.search(out)
    assert m, f"cobertura ausente do var_info: {out}"
    assert int(m.group(1)) < int(m.group(2))
    assert _PARTIAL_RE.search(out), f"cobertura parcial precisa GRITAR: {out}"


def test_var_info_sem_indice_vetorial_nao_mostra_emb():
    _load("_qa_vi_none", "texto pequeno sem indice vetorial")
    out = hs.call_tool("rlm_var_info", {"name": "_qa_vi_none"})["content"][0]["text"]
    assert not _EMB_RE.search(out)


def test_formatter_normal_explica_o_risco():
    class _Info:
        name = "x"
        type_name = "str"
        size_human = "1 KB"
        size_bytes = 1024
        preview = "..."
        from datetime import datetime
        created_at = datetime(2026, 1, 1)
        last_accessed = datetime(2026, 1, 1)

    out = fmt.format_var_info(
        _Info(), verbosity=Verbosity.NORMAL,
        vector_stats={"total_chunks": 100, "embedded_chunks": 40},
    )
    assert "Embeddings: 40/100 chunks (40%)" in out
    assert "cobertura parcial" in out
    assert "busca semântica enxerga só parte" in out
