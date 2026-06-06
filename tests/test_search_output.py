"""Tests for search output limiting: max_results cap, pagination, summary headers."""

import os
import pytest

# Force normal verbosity for readable output in tests
os.environ["RLM_RESPONSE_VERBOSITY"] = "normal"

from rlm_mcp.response_formatter import (
    format_search_response,
    format_search_code,
    format_hybrid_search,
    Verbosity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_keyword_results(terms: list[str], hits_per_term: int) -> dict:
    """Build a fake keyword search result dict: {term: [matches...]}."""
    results = {}
    for term in terms:
        results[term] = [
            {"linha": i + 1, "contexto": f"Contexto da linha {i+1} contendo {term} aqui no texto"}
            for i in range(hits_per_term)
        ]
    return results


def _make_code_results(count: int, include_source: bool = False, source_lines: int = 20) -> list[dict]:
    """Build a list of fake symbol dicts."""
    symbols = []
    for i in range(count):
        sym = {
            "name": f"func_{i}",
            "kind": "function",
            "line_start": i * 10 + 1,
            "line_end": i * 10 + 8,
            "signature": f"def func_{i}(arg1, arg2, arg3_with_long_name, arg4_with_long_name, arg5_with_long_name):",
            "parent": None,
            "docstring": f"Docstring for func_{i} with some extra description text that might be long",
        }
        if include_source:
            sym["source"] = "\n".join(f"    line {j} of func_{i}" for j in range(source_lines))
        symbols.append(sym)
    return symbols


# ---------------------------------------------------------------------------
# 1. max_results cap global em keyword search
# ---------------------------------------------------------------------------

class TestMaxResultsCap:
    def test_cap_limits_total_across_terms(self):
        """5 terms x 20 hits each = 100 total, capped to 30."""
        terms = ["medo", "ansiedade", "raiva", "tristeza", "culpa"]
        results = _make_keyword_results(terms, hits_per_term=20)

        # Simulate the cap logic from http_server.py
        max_results = 30
        total_available = sum(len(v) for v in results.values())  # 100
        capped = {}
        count = 0
        for term, matches in results.items():
            if count >= max_results:
                break
            take = min(len(matches), max_results - count)
            capped[term] = matches[:take]
            count += take

        # Verify cap
        total_shown = sum(len(v) for v in capped.values())
        assert total_shown <= 30
        assert total_available == 100

        # Format and verify output
        text = format_search_response(
            capped, terms, require_all=False, total_results=len(capped),
            offset=0, limit=20,
            max_results=max_results, total_available=total_available,
            verbosity=Verbosity.NORMAL,
        )
        assert "30 shown / 100 total" in text

    def test_cap_does_not_truncate_when_under_limit(self):
        """If total < max_results, no truncation."""
        terms = ["medo", "raiva"]
        results = _make_keyword_results(terms, hits_per_term=5)  # 10 total

        total_available = sum(len(v) for v in results.values())  # 10
        text = format_search_response(
            results, terms, require_all=False, total_results=len(results),
            offset=0, limit=20,
            max_results=30, total_available=total_available,
            verbosity=Verbosity.NORMAL,
        )
        assert "10 shown / 10 total" in text


# ---------------------------------------------------------------------------
# 2. Summary header presente no output de format_search_response
# ---------------------------------------------------------------------------

class TestSearchSummaryHeader:
    def test_header_present_in_keyword_search(self):
        results = _make_keyword_results(["medo"], hits_per_term=5)
        text = format_search_response(
            results, ["medo"], require_all=False, total_results=1,
            offset=0, limit=20,
            max_results=30, total_available=5,
            verbosity=Verbosity.NORMAL,
        )
        assert text.startswith("[search:")
        assert '"medo"' in text
        assert "5 shown / 5 total" in text

    def test_header_shows_pagination_info(self):
        results = _make_keyword_results(["ansiedade"], hits_per_term=10)
        text = format_search_response(
            results, ["ansiedade"], require_all=False, total_results=1,
            offset=0, limit=20,
            max_results=30, total_available=10,
            verbosity=Verbosity.NORMAL,
        )
        assert "max_results=30" in text

    def test_no_index_stats_block_in_output(self):
        """index_stats block was removed from normal format."""
        results = _make_keyword_results(["medo"], hits_per_term=3)
        stats = {"indexed_terms": 64, "total_occurrences": 1200}
        text = format_search_response(
            results, ["medo"], require_all=False, total_results=1,
            offset=0, limit=20,
            index_stats=stats,
            max_results=30, total_available=3,
            verbosity=Verbosity.NORMAL,
        )
        # Old format had "📊 Índice:" - should be gone now
        assert "📊" not in text


# ---------------------------------------------------------------------------
# 3. Paginação de rlm_search_code (limit/offset)
# ---------------------------------------------------------------------------

class TestSearchCodePagination:
    def test_limit_restricts_results(self):
        symbols = _make_code_results(50)
        # Simulate pagination: results[0:20]
        page = symbols[0:20]
        text = format_search_code(
            page, "app", "python",
            query="func", total_symbols=100,
            limit=20, offset=0, total_matched=50,
            verbosity=Verbosity.NORMAL,
        )
        assert "20 shown / 50 matched" in text
        assert "next: offset=20" in text

    def test_offset_shows_next_page(self):
        symbols = _make_code_results(50)
        page = symbols[20:40]
        text = format_search_code(
            page, "app", "python",
            query="func", total_symbols=100,
            limit=20, offset=20, total_matched=50,
            verbosity=Verbosity.NORMAL,
        )
        assert "20 shown / 50 matched" in text
        assert "next: offset=40" in text


# ---------------------------------------------------------------------------
# 4. max_source_lines respeita o limite
# ---------------------------------------------------------------------------

class TestMaxSourceLines:
    def test_source_truncated_to_max_lines(self):
        symbols = _make_code_results(1, include_source=True, source_lines=20)
        text = format_search_code(
            symbols, "app", "python",
            total_symbols=10,
            max_source_lines=5, total_matched=1,
            verbosity=Verbosity.NORMAL,
        )
        # Should show 5 source lines + 1 truncation indicator (also starts with |)
        source_pipe_lines = [l for l in text.split("\n") if l.strip().startswith("|")]
        # 5 code lines + 1 "| ... (+15 lines)" = 6 pipe lines total
        assert len(source_pipe_lines) == 6
        assert "+15 lines" in text  # 20 - 5 = 15 remaining

    def test_source_not_truncated_when_under_limit(self):
        symbols = _make_code_results(1, include_source=True, source_lines=3)
        text = format_search_code(
            symbols, "app", "python",
            total_symbols=10,
            max_source_lines=5, total_matched=1,
            verbosity=Verbosity.NORMAL,
        )
        source_pipe_lines = [l for l in text.split("\n") if l.strip().startswith("|")]
        assert len(source_pipe_lines) == 3
        assert "..." not in text or "+0 lines" not in text


# ---------------------------------------------------------------------------
# 5. Backward compat — chamadas sem novos params não quebram
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_format_search_response_defaults(self):
        """Calling without max_results/total_available should still work."""
        results = _make_keyword_results(["medo"], hits_per_term=5)
        text = format_search_response(
            results, ["medo"], require_all=False, total_results=1,
            offset=0, limit=20,
            verbosity=Verbosity.NORMAL,
        )
        assert "[search:" in text  # header still present
        assert "5 shown" in text

    def test_format_search_code_defaults(self):
        """Calling without limit/offset/max_source_lines should still work."""
        symbols = _make_code_results(3)
        text = format_search_code(
            symbols, "app", "python",
            total_symbols=10,
            verbosity=Verbosity.NORMAL,
        )
        assert "[code:" in text
        assert "3 shown" in text

    def test_format_hybrid_search_defaults(self):
        """Calling without max_results should still work."""
        search_result = {
            "mode": "hybrid",
            "hybrid_results": [
                {"line": 10, "rrf_score": 0.95, "sources": ["keyword", "semantic"], "text": "Some text here"},
                {"line": 20, "rrf_score": 0.85, "sources": ["keyword"], "text": "Other text"},
            ],
        }
        text = format_hybrid_search(
            search_result, ["medo"], "livro",
            verbosity=Verbosity.NORMAL,
        )
        assert "[search:" in text
        assert "hybrid" in text


# ---------------------------------------------------------------------------
# 6. Summary header em format_search_code
# ---------------------------------------------------------------------------

class TestSearchCodeHeader:
    def test_header_present(self):
        symbols = _make_code_results(5)
        text = format_search_code(
            symbols, "myapp", "python",
            query="parse", total_symbols=50,
            limit=20, offset=0, total_matched=5,
            verbosity=Verbosity.NORMAL,
        )
        assert text.startswith("[code:")
        assert "myapp" in text
        assert "python" in text
        assert "5 shown / 5 matched" in text

    def test_signature_truncated_to_80(self):
        symbols = [{
            "name": "long_func",
            "kind": "function",
            "line_start": 1,
            "line_end": 10,
            "signature": "def long_func(" + "a" * 200 + "):",
            "parent": None,
        }]
        text = format_search_code(
            symbols, "app", "python",
            total_symbols=1, total_matched=1,
            verbosity=Verbosity.NORMAL,
        )
        # Find the signature line (indented 8 spaces)
        sig_lines = [l for l in text.split("\n") if l.startswith("        ") and "def long_func" in l]
        assert sig_lines
        # The signature should be at most ~80 chars (truncated from original)
        assert len(sig_lines[0].strip()) <= 80

    def test_docstring_truncated_to_60(self):
        symbols = [{
            "name": "doc_func",
            "kind": "function",
            "line_start": 1,
            "line_end": 10,
            "signature": "def doc_func():",
            "parent": None,
            "docstring": "D" * 200,
        }]
        text = format_search_code(
            symbols, "app", "python",
            total_symbols=1, total_matched=1,
            verbosity=Verbosity.NORMAL,
        )
        # Find the docstring line (starts with `"`)
        doc_lines = [l for l in text.split("\n") if l.strip().startswith('"')]
        assert doc_lines
        # Content inside quotes should be <= 60 chars
        content = doc_lines[0].strip().strip('"')
        assert len(content) <= 60


# ---------------------------------------------------------------------------
# Convenção de display 1-indexed (fix 2026-06-06): produtores internos
# (TextIndex/BM25/chunks semânticos) são 0-indexed; o +1 acontece SÓ na borda
# de exibição. Termo na PRIMEIRA linha tem que exibir L1 (nunca L0).
# ---------------------------------------------------------------------------

class TestDisplayOneIndexed:
    def test_legacy_phrase_matches_display_l1_for_first_line(self):
        results = {"alvo": [{"linha": 0, "contexto": "alvo na primeira linha"}]}
        text = format_search_response(
            results, ["alvo"], require_all=False, total_results=1,
            offset=0, limit=10, verbosity=Verbosity.NORMAL,
        )
        assert "L1:" in text
        assert "L0:" not in text

    def test_require_all_displays_linha_1_for_first_line(self):
        # require_all: shape {linha_0idx: [termos]}
        text = format_search_response(
            {0: ["alvo"]}, ["alvo"], require_all=True, total_results=1,
            offset=0, limit=10, verbosity=Verbosity.NORMAL,
        )
        assert "Linha 1:" in text
        assert "Linha 0:" not in text

    def test_bm25_ranked_displays_l1_for_first_line(self):
        from rlm_mcp.indexer import create_index, set_index, clear_all_indices, hybrid_search
        clear_all_indices()
        doc = "alvo logo na primeira frase.\nsegunda linha neutra."
        set_index("conv_disp", create_index(doc, "conv_disp"))
        res = hybrid_search("conv_disp", ["alvo"], mode="keyword",
                            limit=5, offset=0, source_text=doc)
        assert res["keyword_ranked"], "sem hits BM25"
        text = format_hybrid_search(res, ["alvo"], "conv_disp",
                                    verbosity=Verbosity.NORMAL)
        assert "L1" in text
        assert "L0" not in text
