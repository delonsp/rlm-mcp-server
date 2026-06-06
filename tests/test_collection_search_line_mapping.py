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


# ---------------------------------------------------------------------------
# P1s da avaliação Codex 2026-06-02 (corrigidos 2026-06-06)
# ---------------------------------------------------------------------------

def test_bm25_global_ranking_pagination_limit_1():
    """Paginação GLOBAL por relevância: limit=1 mostra 1 hit NO TOTAL (o melhor
    da coleção inteira). O código antigo paginava POR bucket var→termo —
    limit=1 mostrava 1 hit POR var (2 no total)."""
    import re as _re
    coll = f"lmrank_{uuid.uuid4().hex[:8]}"
    p = hs.get_persistence()
    assert p.create_collection(coll, "rank")
    # vA: 1 ocorrência fraca (termo diluído em linha longa)
    hs.repl.variables["rk_va"] = (
        "kratos aparece uma vez aqui no meio de muitas outras palavras "
        "completamente irrelevantes que diluem a densidade do termo nesta linha.\n"
        "segunda linha sem nada."
    )
    # vB: ocorrência densa (3x na mesma linha curta) → melhor score BM25
    hs.repl.variables["rk_vb"] = "kratos kratos kratos.\noutra linha."
    p.add_to_collection(coll, ["rk_va", "rk_vb"])
    res = hs.call_tool("rlm_collection_rebuild", {"name": coll})
    assert not res.get("isError"), res

    res = hs.call_tool("rlm_search_collection",
                       {"collection": coll, "terms": ["kratos"], "limit": 1})
    text = res["content"][0]["text"]
    citations = _re.findall(r"L\d+:", text)
    assert len(citations) == 1, f"limit=1 deve mostrar 1 hit GLOBAL, veio {len(citations)}:\n{text}"
    assert "rk_vb" in text, f"o hit denso (melhor BM25) tinha que vir primeiro:\n{text}"
    assert "rk_va" not in text.split("📊")[0].replace("rk_vb", ""), text
    assert "de 2" in text, f"total global ausente:\n{text}"
    assert "relevância global" in text, text


def test_mixed_quoted_literal_is_mandatory_filter():
    """Guard (c) mixed: termo quoted que não existe na coleção → ZERO resultados
    (antes: o quoted era dropado e o fallback devolvia linhas só com os tokens)."""
    coll = f"lmquote_{uuid.uuid4().hex[:8]}"
    p = hs.get_persistence()
    assert p.create_collection(coll, "quoted")
    hs.repl.variables["qt_v1"] = (
        "linha com gama e tambem delta presentes como tokens.\n"
        "linha neutra sem nada."
    )
    p.add_to_collection(coll, ["qt_v1"])
    res = hs.call_tool("rlm_collection_rebuild", {"name": coll})
    assert not res.get("isError"), res

    # frase 'gama delta' não existe literal (tem 'gama e tambem delta') →
    # fallback tokenizado dispararia; o literal quoted inexistente DEVE zerar
    res = hs.call_tool("rlm_search_collection",
                       {"collection": coll,
                        "terms": ['"frase inexistente xyz"', "gama delta"],
                        "limit": 10})
    text = res["content"][0]["text"]
    assert "Nenhum resultado" in text, f"quoted inexistente deveria zerar:\n{text}"
    assert "EXATO obrigatório" in text or "exato" in text.lower(), text


def test_fallback_with_satisfiable_required_literal_keeps_only_matching_lines():
    """required_literals no scan: só linhas com o literal E os tokens."""
    from rlm_mcp.indexer import tokenized_collection_scan
    combined = ("gama com alfa beta presente nesta linha\n"
                "gama sozinho sem o literal aqui")
    mapping = {1: ("v", 1), 2: ("v", 2)}
    res, mode = tokenized_collection_scan(
        combined, mapping, ["gama"], required_literals=["alfa beta"])
    linhas = [o["linha"] for label in res.get("v", {}) for o in res["v"][label]]
    assert linhas == [1], f"só a linha com o literal deveria entrar: {res}"
