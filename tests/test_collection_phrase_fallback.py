"""
Testes do auto-tokenize fallback da busca de coleção (phrase-trap mitigation).

Cobre os helpers puros em indexer.py que sustentam as guardas pedidas na revisão
(Carcinosinum, 2026-06):
  - tokenize_for_fallback: quebra frase→tokens (accent-fold, sem stopwords, dedupe).
  - tokenized_collection_scan: guard (a) AND antes de OR; word-boundary (mata o ruído
    "fish" em "selfish"); mapping linha-combinada→(var,linha); snippet_len.

Guard (b) transparência (format_fallback_banner) e guard (c) aspas (parse_quoted_terms)
foram extraídas p/ funções puras justamente para serem testadas aqui — a (b) protege a
rastreabilidade de citação e é crítica demais p/ ficar sem teste versionado.

VERSIONADO (exceção no .gitignore) — diferente do resto de tests/.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlm_mcp.indexer import (  # noqa: E402
    tokenize_for_fallback,
    tokenized_collection_scan,
    _token_in_folded_line,
    parse_quoted_terms,
    format_fallback_banner,
)


def test_tokenize_drops_stopwords_and_dedupes():
    assert tokenize_for_fallback(["aggravation from fish"]) == ["aggravation", "fish"]
    # dedupe preservando ordem + accent-fold
    assert tokenize_for_fallback(["coração", "CORAÇÃO dor"]) == ["coracao", "dor"]
    # frase longa vira N tokens
    toks = tokenize_for_fallback(["hot patient worse heat better open air"])
    assert "hot" in toks and "air" in toks and "from" not in toks


def test_word_boundary_match():
    # 'fish' NÃO casa dentro de 'selfish' (bug do full-text legado)
    assert _token_in_folded_line("he is selfish and proud", "fish") is False
    assert _token_in_folded_line("effects from spoiled fish", "fish") is True
    # accent-fold dos dois lados
    assert _token_in_folded_line("dor no coracao", "coracao") is True


# combined_text 1-indexado; mapping linha-combinada → (var, linha_original)
COMBINED = "\n".join([
    "Bad effects from eating spoiled fish",   # L1 → boericke:10  (só fish)
    "He is selfish and proud man",            # L2 → boericke:11  (NÃO deve casar fish)
    "Worse from fish and cold seafood here",  # L3 → kent:20      (fish + seafood)
    "completely unrelated prose line",        # L4 → kent:21
])
MAPPING = {1: ("boericke", 10), 2: ("boericke", 11), 3: ("kent", 20), 4: ("kent", 21)}


def test_scan_prefers_and_over_or():
    # 'fish' E 'seafood' juntos só em L3 → modo AND, label conjunto, mapeado p/ kent:20
    results, mode = tokenized_collection_scan(COMBINED, MAPPING, ["fish", "seafood"])
    assert mode == "AND"
    assert "fish & seafood" in results["kent"]
    assert results["kent"]["fish & seafood"][0]["linha"] == 20
    # AND não inclui L1 (só tem fish) nem o ruído selfish
    assert "boericke" not in results


def test_scan_falls_back_to_or():
    # nenhum termo coexiste numa linha → OR, agrupado por token, sem o termo vazio
    results, mode = tokenized_collection_scan(COMBINED, MAPPING, ["fish", "grief"])
    assert mode == "OR"
    assert set(results.keys()) == {"boericke", "kent"}     # fish em L1 e L3
    assert "grief" not in results.get("boericke", {})      # token sem hit é descartado
    # word-boundary: selfish (L2) nunca aparece
    linhas = [m["linha"] for v in results.values() for hits in v.values() for m in hits]
    assert 11 not in linhas


def test_scan_snippet_len_truncates():
    long_line = "fish " + "x" * 500
    text = long_line
    mapping = {1: ("v", 1)}
    results, _ = tokenized_collection_scan(text, mapping, ["fish"], snippet_len=40)
    ctx = results["v"]["fish"][0]["contexto"]
    assert len(ctx) == 40


def test_scan_empty_inputs():
    assert tokenized_collection_scan("", MAPPING, ["fish"]) == ({}, None)
    assert tokenized_collection_scan(COMBINED, MAPPING, []) == ({}, None)


def test_scan_skips_unmapped_lines():
    # linha sem entrada no mapping (sentinel entre vars) é ignorada
    mapping = {1: ("boericke", 10)}  # L3 (fish) sem mapping
    results, mode = tokenized_collection_scan(COMBINED, mapping, ["fish"])
    # só L1 mapeada sobrevive
    linhas = [m["linha"] for v in results.values() for hits in v.values() for m in hits]
    assert linhas == [10]


# --- Guard (c): aspas = busca exata, não tokeniza ---

def test_parse_quoted_terms_strips_and_flags():
    terms, flags, all_quoted = parse_quoted_terms(['"open air"', "fish"])
    assert terms == ["open air", "fish"]          # aspas removidas
    assert flags == [True, False]
    assert all_quoted is False                    # nem todos entre aspas


def test_parse_quoted_terms_all_quoted_disables_fallback():
    # all_quoted=True é o sinal que o handler usa p/ NÃO tokenizar
    _, _, all_quoted = parse_quoted_terms(['"erro fatal"'])
    assert all_quoted is True


def test_parse_quoted_terms_empty():
    assert parse_quoted_terms([]) == ([], [], False)


# --- Guard (b): banner de transparência presente e inequívoco ---

def test_fallback_banner_marks_results_as_non_exact():
    banner = format_fallback_banner("AND", ["hot", "patient"], "hot patient")
    blob = "\n".join(banner)
    assert "FALLBACK TOKENIZADO (AND)" in blob
    assert "NÃO são da sua busca exata" in blob       # proteção de citação
    assert "['hot', 'patient']" in blob               # tokens visíveis
    assert "hot patient" in blob                       # frase original visível
    assert banner[-1] == ""                            # separador antes do corpo


def test_fallback_banner_or_mode_wording():
    banner = format_fallback_banner("OR", ["fish"], "spoiled fish dish")
    blob = "\n".join(banner)
    assert "(OR)" in blob and "qualquer um" in blob


def test_tokenized_scan_mapping_is_one_indexed_adversarial_first_line():
    """Adversarial do P0 line-mapping: termo na PRIMEIRA linha do combinado.
    Se o scan fosse 0-indexed (ou o mapping), o lookup da linha 1 falharia e
    o hit sumiria — ou citaria a linha errada."""
    combined = "alvo na primeira linha\nsegunda linha"
    var_mapping = {1: ("v", 1), 2: ("v", 2)}
    results, mode = tokenized_collection_scan(combined, var_mapping, ["alvo"])
    assert results, "hit da linha 1 dropado (convenção 0-vs-1 quebrada)"
    ocorrencias = results["v"][list(results["v"].keys())[0]]
    assert ocorrencias[0]["linha"] == 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
