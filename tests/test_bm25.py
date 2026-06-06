"""
Testes sintéticos do BM25 sentence-level (plans/20260529-bm25-sentence-level.md).

Cobre: tokenização (accent-fold, stopwords, min-len), segmentação (alvo de
tokens, linha em branco, sentinel de coleção), build/search BM25 (ranking,
require_all pós-filtro, offset/limit), fusão RRF por overlap de range,
thread-safety do lazy-build, e não-serialização dos campos BM25 (from_dict
reconstrói lazy).

Local-only (gitignored, como o resto de tests/ exceto test_sandbox.py).
"""

import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlm_mcp.indexer import (  # noqa: E402
    TextIndex,
    create_index,
    hybrid_search,
    set_index,
    clear_all_indices,
    _bm25_tokenize,
    _segment_lines,
    _normalize_snippet,
    _legacy_to_ranked,
    _reciprocal_rank_fusion,
)


# --------------------------------------------------------------------------
# Tokenização
# --------------------------------------------------------------------------

def test_tokenize_accent_fold():
    toks = _bm25_tokenize("Câncer e CÂNCER são o mesmo token")
    assert toks.count("cancer") == 2, toks
    # acento dobrado: 'câncer' e 'cancer' colapsam
    assert "cancer" in _bm25_tokenize("cancer de mama")


def test_tokenize_drops_stopwords():
    toks = _bm25_tokenize("o gato e a casa de pedra")
    assert "o" not in toks and "e" not in toks and "de" not in toks
    assert "gato" in toks and "casa" in toks and "pedra" in toks


def test_tokenize_min_len_drops_single_char():
    # 'd' (de vitamina D) cai com min-len default 2 — limitação conhecida
    toks = _bm25_tokenize("vitamina D")
    assert "vitamina" in toks
    assert "d" not in toks


def test_tokenize_empty():
    assert _bm25_tokenize("") == []
    assert _bm25_tokenize(None) == []


# --------------------------------------------------------------------------
# Segmentação
# --------------------------------------------------------------------------

def test_segment_breaks_on_blank_line():
    text = "alpha beta gama\ndelta epsilon\n\nzeta eta theta"
    segs = _segment_lines(text, target_tokens=1000)
    # linha em branco (idx 2) separa: [0,1] e [3,3]
    assert (0, 1) in segs
    assert (3, 3) in segs


def test_segment_breaks_on_token_target():
    # cada linha tem 4 tokens de conteúdo (nenhum stopword); alvo 4 → fecha a cada linha
    text = "\n".join(["alpha beta gama delta"] * 5)
    segs = _segment_lines(text, target_tokens=4)
    assert len(segs) == 5


def test_segment_breaks_on_collection_sentinel():
    text = (
        "conteudo da primeira variavel aqui\n"
        + "=" * 60 + "\n"
        + "=== VARIÁVEL: outra ===\n"
        + "=" * 60 + "\n"
        + "conteudo da segunda variavel"
    )
    segs = _segment_lines(text, target_tokens=1000)
    # nenhum segmento pode conter o sentinel; deve haver 2 blocos disjuntos
    assert segs[0] == (0, 0)
    assert segs[-1][0] == 4  # primeira linha de conteúdo após os 3 headers


def test_normalize_snippet_collapses_whitespace():
    assert _normalize_snippet("a    b\n\n  c") == "a b c"


# --------------------------------------------------------------------------
# build_bm25 / search_bm25
# --------------------------------------------------------------------------

def _doc():
    return "\n".join([
        "gato cachorro passaro peixe " * 4,
        "",
        "mitocondria energia celular metabolismo " * 4,
        "",
        "a palavra rara unobtanium aparece somente aqui",
    ])


def _idx(doc, name="t"):
    return TextIndex(var_name=name, total_chars=len(doc), total_lines=doc.count("\n") + 1)


def test_build_and_rare_term_ranks_first():
    doc = _doc()
    idx = _idx(doc)
    assert idx.build_bm25(doc, target_tokens=50)
    assert idx.bm25_n >= 3
    hits = idx.search_bm25(["unobtanium"], doc, limit=5)
    assert hits, "termo raro não encontrado"
    # o segmento com o termo raro deve ser o top-1
    assert "unobtanium" in hits[0]["_overlap_text"].lower()


def test_nonexistent_term_returns_empty():
    doc = _doc()
    idx = _idx(doc)
    idx.build_bm25(doc, target_tokens=50)
    assert idx.search_bm25(["inexistentexyzqwe"], doc) == []


def test_stopword_only_query_returns_empty():
    doc = _doc()
    idx = _idx(doc)
    idx.build_bm25(doc, target_tokens=50)
    assert idx.search_bm25(["o", "a", "de", "e"], doc) == []


def test_require_all_postfilter():
    doc = "\n".join([
        "mitocondria energia",        # só um dos termos
        "",
        "mitocondria energia celular oxidativa",  # ambos termos
    ])
    idx = _idx(doc)
    idx.build_bm25(doc, target_tokens=1000)
    # sem require_all: ambos os segmentos podem aparecer
    loose = idx.search_bm25(["mitocondria", "oxidativa"], doc)
    # com require_all: só segmentos com TODOS os tokens
    strict = idx.search_bm25(["mitocondria", "oxidativa"], doc, require_all=True)
    assert strict, "require_all não retornou o segmento que contém ambos"
    for h in strict:
        ov = h["_overlap_text"].lower()
        assert "mitocondria" in ov and "oxidativa" in ov
    assert len(strict) <= len(loose)


def test_offset_limit_pagination():
    # 6 segmentos, todos contendo 'termo', com frequências decrescentes
    blocks = []
    for i in range(6):
        blocks.append(("termo " * (6 - i)).strip())
        blocks.append("")
    doc = "\n".join(blocks)
    idx = _idx(doc)
    idx.build_bm25(doc, target_tokens=1000)
    page1 = idx.search_bm25(["termo"], doc, limit=2, offset=0)
    page2 = idx.search_bm25(["termo"], doc, limit=2, offset=2)
    assert len(page1) == 2 and len(page2) == 2
    # páginas disjuntas
    lines1 = {h["line"] for h in page1}
    lines2 = {h["line"] for h in page2}
    assert lines1.isdisjoint(lines2)


def test_idempotent_build():
    doc = _doc()
    idx = _idx(doc)
    assert idx.build_bm25(doc, target_tokens=50)
    n1 = idx.bm25_n
    # segunda chamada não rebuilda (gated por _bm25_built)
    assert idx.build_bm25(doc, target_tokens=50)
    assert idx.bm25_n == n1


# --------------------------------------------------------------------------
# Thread-safety do lazy-build
# --------------------------------------------------------------------------

def test_concurrent_build_no_corruption():
    doc = _doc()
    idx = _idx(doc, name="concurrent")
    errors = []
    results = []

    def worker():
        try:
            idx.build_bm25(doc, target_tokens=50)
            results.append(idx.bm25_n)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    # todas as threads veem o mesmo nº de segmentos (estado consistente)
    assert len(set(results)) == 1
    # estruturas íntegras e coerentes entre si
    assert idx.bm25_n > 0
    assert len(idx.bm25_doc_len) == idx.bm25_n
    assert len(idx.bm25_segments) == idx.bm25_n
    assert idx.bm25_postings


# --------------------------------------------------------------------------
# Não-serialização (from_dict reconstrói lazy)
# --------------------------------------------------------------------------

def test_bm25_fields_not_serialized():
    doc = _doc()
    idx = _idx(doc)
    idx.build_bm25(doc, target_tokens=50)
    d = idx.to_dict()
    for k in ("bm25_postings", "bm25_doc_len", "bm25_segments", "bm25_n", "_bm25_built"):
        assert k not in d, f"{k} vazou pro to_dict()"


def test_from_dict_rebuilds_lazy():
    doc = _doc()
    idx = _idx(doc)
    idx.build_bm25(doc, target_tokens=50)
    restored = TextIndex.from_dict(idx.to_dict())
    # restaurado começa sem BM25
    assert restored._bm25_built is False
    assert restored.bm25_n == 0
    # a 1ª busca reconstrói
    hits = restored.search_bm25(["unobtanium"], doc, limit=5)
    assert hits
    assert restored._bm25_built is True


# --------------------------------------------------------------------------
# Fusão RRF por overlap de range
# --------------------------------------------------------------------------

def test_rrf_merges_overlapping_ranges():
    # keyword segmento [10,20] e semantic chunk [12,18] se sobrepõem → 1 entrada
    bm25_hits = [{"line": 10, "line_end": 20, "score": 5.0,
                  "text": "alvo keyword", "_overlap_text": "alvo keyword aqui"}]
    semantic = [{"chunk_text": "alvo semantic", "line_start": 12,
                 "line_end": 18, "score": 0.9, "chunk_index": 0}]
    fused = _reciprocal_rank_fusion(bm25_hits, semantic, ["alvo"], limit=10)
    assert len(fused) == 1
    assert set(fused[0]["sources"]) == {"keyword", "semantic"}


def test_rrf_keeps_disjoint_ranges_separate():
    bm25_hits = [{"line": 0, "line_end": 2, "score": 5.0,
                  "text": "a", "_overlap_text": "alpha"}]
    semantic = [{"chunk_text": "beta", "line_start": 100,
                 "line_end": 105, "score": 0.9, "chunk_index": 0}]
    fused = _reciprocal_rank_fusion(bm25_hits, semantic, ["x"], limit=10)
    assert len(fused) == 2


def test_rrf_empty_legs():
    assert _reciprocal_rank_fusion([], [], ["x"]) == []


def test_legacy_to_ranked_dedup():
    legacy = {
        "termo": [{"linha": 5, "contexto": "ctx a"}, {"linha": 5, "contexto": "dup"}],
        "outro": [{"linha": 9, "contexto": "ctx b"}],
    }
    ranked = _legacy_to_ranked(legacy)
    lines = [r["line"] for r in ranked]
    assert lines == [5, 9]  # dedup por linha, ordem de 1ª ocorrência


# --------------------------------------------------------------------------
# hybrid_search end-to-end (keyword mode = BM25)
# --------------------------------------------------------------------------

def test_hybrid_search_keyword_mode_uses_bm25():
    clear_all_indices()
    doc = _doc()
    set_index("hv", create_index(doc, "hv"))
    res = hybrid_search("hv", ["unobtanium"], mode="keyword",
                        limit=5, offset=0, source_text=doc)
    assert res["keyword_ranked"] is not None
    assert res["keyword_ranked"], "keyword_ranked vazio"
    assert "unobtanium" in res["keyword_ranked"][0]["_overlap_text"].lower()


# --------------------------------------------------------------------------
# Convenção de índice de linha (P0 line-mapping da coleção, 2026-06-06):
# search_bm25 retorna line/line_end 0-INDEXED — consumers que cruzam com o
# var_mapping de coleção (1-indexed) DEVEM somar +1.
# --------------------------------------------------------------------------

def test_search_bm25_returns_zero_indexed_line_range():
    clear_all_indices()
    doc = "alvo logo na primeira frase.\nsegunda linha neutra.\nterceira linha neutra."
    idx = create_index(doc, "conv0")
    hits = idx.search_bm25(["alvo"], doc, limit=5, offset=0)
    assert hits, "sem hits"
    lines = doc.split("\n")
    h = hits[0]
    # 0-indexed: o range do segmento contém a linha do termo SEM ajuste
    assert any("alvo" in lines[i] for i in range(h["line"], h["line_end"] + 1)), h
    # e o primeiro segmento do doc começa no índice 0 (se fosse 1-indexed, seria 1)
    assert min(x["line"] for x in hits) == 0
