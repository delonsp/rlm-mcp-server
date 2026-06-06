"""
Consumers do var_mapping na busca de coleção (handler rlm_search_collection):
BM25 (+1 sobre segmento 0-indexed), legacy search_multiple (+1 — era o bug
0-vs-1), fulltext (start=1) e tokenized fallback (1-indexed).

Integração real: persistence (SQLite tmpdir do conftest) + rebuild + search
via call_tool — o caminho exato que cita var:linha em produção.
"""
import uuid

from rlm_mcp import http_server as hs
from rlm_mcp.indexer import get_index

# lm_v1: filler SEM termos de busca (10 linhas)
V1 = "\n".join(f"filler linha {i} sem nada de interessante." for i in range(1, 11))
# lm_v2: "medo" (termo do DEFAULT_INDEX_TERMS) na LINHA 1; "xilofone" (fora do
# vocabulário default) na LINHA 2.
V2 = "medo de altura paralisa.\nxilofone soa no salão.\nterceira linha neutra."


def _make_collection() -> str:
    """Coleção única por teste (SQLite do tmpdir persiste entre testes)."""
    coll = f"lmtest_{uuid.uuid4().hex[:8]}"
    p = hs.get_persistence()
    assert p.create_collection(coll, "teste line-mapping")
    hs.repl.variables["lm_v1"] = V1
    hs.repl.variables["lm_v2"] = V2
    p.add_to_collection(coll, ["lm_v1", "lm_v2"])
    res = hs.call_tool("rlm_collection_rebuild", {"name": coll})
    assert not res.get("isError"), res
    return coll


def _search(coll: str, terms: list[str]) -> str:
    res = hs.call_tool("rlm_search_collection", {"collection": coll, "terms": terms, "limit": 10})
    assert not res.get("isError"), res
    return res["content"][0]["text"]


def test_bm25_cita_linha_original_correta():
    """BM25: segmento começa na linha 1 de lm_v2 (sentinel impede começar no
    header) → citação tem que ser lm_v2 L1. O código antigo citava L3 (+2 na
    2ª var)."""
    coll = _make_collection()
    text = _search(coll, ["medo"])
    assert "lm_v2" in text, text
    assert "L1:" in text or "L1 " in text, f"esperava L1 em:\n{text}"
    assert "L3:" not in text, f"deslocamento antigo (+2) presente:\n{text}"


def test_legacy_search_multiple_converte_0_para_1_indexed():
    """Força o caminho legacy (BM25 degradado): matches do TextIndex são
    0-indexed e SEM o +1 o lookup cai no header (hit dropado em silêncio)."""
    coll = _make_collection()
    idx = get_index(f"_coll_{coll}_combined")
    assert idx is not None
    # Degrada BM25 → handler usa terms_via_index (legacy) p/ termo default
    idx._bm25_built = False
    idx._bm25_degraded = True
    text = _search(coll, ["medo"])
    assert "lm_v2" in text, f"hit dropado (lookup sem +1?):\n{text}"
    assert "L1:" in text or "L1 " in text, f"esperava L1 em:\n{text}"


def test_fulltext_fallback_cita_linha_correta():
    """Termo fora do vocabulário default + BM25 degradado → fulltext
    (enumerate start=1, já compatível com mapping 1-indexed)."""
    coll = _make_collection()
    idx = get_index(f"_coll_{coll}_combined")
    idx._bm25_built = False
    idx._bm25_degraded = True
    text = _search(coll, ["xilofone"])
    assert "lm_v2" in text, text
    assert "L2:" in text or "L2 " in text, f"esperava L2 em:\n{text}"


def test_mapping_em_memoria_bate_com_combined_real():
    """Invariante de integração: o mapping salvo pelo rebuild bate com o
    combined salvo no REPL, linha a linha."""
    coll = _make_collection()
    combined = hs.repl.variables[f"_coll_{coll}_combined"]
    mapping = hs.repl.variables[f"_coll_{coll}_mapping"]
    lines = combined.split("\n")
    variables = {"lm_v1": V1, "lm_v2": V2}
    for cl, (var, orig) in mapping.items():
        assert lines[cl - 1] == variables[var].split("\n")[orig - 1], (cl, var, orig)
