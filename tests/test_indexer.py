"""
Tests for the indexer module.

Tests cover:
- create_index function with default and custom terms
- TextIndex search methods
- auto_index_if_large threshold behavior
- _detect_structure for markdown headers
- TextIndex serialization (to_dict/from_dict)
"""

import pytest

from rlm_mcp.indexer import (
    DEFAULT_INDEX_TERMS,
    TextIndex,
    create_index,
    auto_index_if_large,
    _detect_structure,
)


class TestCreateIndexWithDefaultTerms:
    """Test that create_index generates an index with default terms."""

    def test_creates_index_with_var_name(self):
        """create_index sets var_name correctly."""
        text = "Tenho medo de trabalho e ansiedade com família."
        index = create_index(text, "test_var")

        assert index.var_name == "test_var"

    def test_creates_index_with_correct_char_count(self):
        """create_index calculates total_chars correctly."""
        text = "Tenho medo de trabalho."
        index = create_index(text, "test_var")

        assert index.total_chars == len(text)

    def test_creates_index_with_correct_line_count(self):
        """create_index calculates total_lines correctly."""
        text = "Linha 1 com medo\nLinha 2 com trabalho\nLinha 3"
        index = create_index(text, "test_var")

        assert index.total_lines == 3

    def test_indexes_default_terms_present_in_text(self):
        """create_index indexes default terms found in text."""
        text = "Tenho medo de trabalho e ansiedade com família."
        index = create_index(text, "test_var")

        # These terms from DEFAULT_INDEX_TERMS should be indexed
        assert "medo" in index.terms
        assert "trabalho" in index.terms
        assert "ansiedade" in index.terms
        assert "família" in index.terms

    def test_does_not_index_terms_not_present_in_text(self):
        """create_index doesn't create entries for terms not in text."""
        text = "Apenas uma frase simples sem termos especiais."
        index = create_index(text, "test_var")

        # Terms not in text should not be in index
        assert "medo" not in index.terms
        assert "trabalho" not in index.terms

    def test_index_entry_contains_line_number(self):
        """Index entries contain the line number where term was found."""
        text = "Primeira linha\nSegunda linha com medo\nTerceira linha"
        index = create_index(text, "test_var")

        assert "medo" in index.terms
        assert index.terms["medo"][0]["linha"] == 1  # Line 1 (0-indexed)

    def test_index_entry_contains_context(self):
        """Index entries contain context around the term."""
        text = "Esta linha contém medo e outros sentimentos"
        index = create_index(text, "test_var")

        assert "medo" in index.terms
        assert "contexto" in index.terms["medo"][0]
        assert "medo" in index.terms["medo"][0]["contexto"]

    def test_case_insensitive_indexing(self):
        """create_index performs case-insensitive matching."""
        text = "MEDO em maiúsculas e Trabalho com capitalização"
        index = create_index(text, "test_var")

        # Should find terms regardless of case
        assert "medo" in index.terms
        assert "trabalho" in index.terms

    def test_multiple_occurrences_on_different_lines(self):
        """create_index indexes multiple occurrences on different lines."""
        text = "Linha 1 com medo\nLinha 2 também com medo\nLinha 3 com medo"
        index = create_index(text, "test_var")

        assert "medo" in index.terms
        assert len(index.terms["medo"]) == 3

    def test_avoids_duplicates_on_same_line(self):
        """create_index doesn't create duplicate entries for same line."""
        text = "medo medo medo - muito medo nesta linha"
        index = create_index(text, "test_var")

        # Should only have one entry for line 0
        assert "medo" in index.terms
        assert len(index.terms["medo"]) == 1

    def test_empty_text_creates_empty_index(self):
        """create_index handles empty text gracefully."""
        index = create_index("", "empty_var")

        assert index.var_name == "empty_var"
        assert index.total_chars == 0
        assert index.total_lines == 0  # splitlines on "" returns []
        assert len(index.terms) == 0

    def test_custom_terms_list_is_empty_by_default(self):
        """create_index sets custom_terms to empty list when none provided."""
        text = "Texto com medo"
        index = create_index(text, "test_var")

        assert index.custom_terms == []

    def test_indexes_emotion_terms_from_default(self):
        """create_index indexes emotion/mental state terms from DEFAULT_INDEX_TERMS."""
        # Use several emotion terms from the default set
        text = "Sinto medo, ansiedade, raiva e tristeza. Também depressão e alegria."
        index = create_index(text, "emotions")

        for term in ["medo", "ansiedade", "raiva", "tristeza", "depressão", "alegria"]:
            assert term in index.terms, f"Term '{term}' should be indexed"

    def test_indexes_relationship_terms_from_default(self):
        """create_index indexes relationship terms from DEFAULT_INDEX_TERMS."""
        text = "Minha família: pai, mãe, filho e esposa. Também o chefe do trabalho."
        index = create_index(text, "relationships")

        for term in ["família", "pai", "mãe", "filho", "esposa", "chefe"]:
            assert term in index.terms, f"Term '{term}' should be indexed"

    def test_indexes_body_parts_from_default(self):
        """create_index indexes body part terms from DEFAULT_INDEX_TERMS."""
        text = "Dor na cabeça, olho, ouvido, coração e estômago."
        index = create_index(text, "body_parts")

        for term in ["cabeça", "olho", "ouvido", "coração", "estômago"]:
            assert term in index.terms, f"Term '{term}' should be indexed"

    def test_context_chars_parameter(self):
        """create_index respects context_chars parameter."""
        long_line = "A" * 50 + " medo " + "B" * 200
        index = create_index(long_line, "test_var", context_chars=60)

        # Context should be truncated to ~60 chars
        context = index.terms["medo"][0]["contexto"]
        assert len(context) <= 60

    def test_structure_is_detected(self):
        """create_index populates structure field."""
        text = "# Header\nTexto com medo"
        index = create_index(text, "test_var")

        # Structure should be populated (even if empty in some sections)
        assert "headers" in index.structure
        assert "capitulos" in index.structure
        assert "remedios" in index.structure


class TestCreateIndexWithAdditionalTerms:
    """Test that create_index indexes custom terms via additional_terms parameter."""

    def test_additional_terms_are_indexed(self):
        """create_index indexes terms from additional_terms list."""
        text = "O paciente tem xerostomia e epistaxe frequente."
        index = create_index(text, "test_var", additional_terms=["xerostomia", "epistaxe"])

        assert "xerostomia" in index.terms
        assert "epistaxe" in index.terms

    def test_additional_terms_stored_in_custom_terms(self):
        """create_index stores additional_terms in custom_terms field."""
        text = "Texto com termo_especial"
        index = create_index(text, "test_var", additional_terms=["termo_especial", "outro_termo"])

        assert index.custom_terms == ["termo_especial", "outro_termo"]

    def test_additional_terms_case_insensitive_indexing(self):
        """additional_terms are matched case-insensitively."""
        text = "O paciente apresenta XEROSTOMIA severa"
        index = create_index(text, "test_var", additional_terms=["xerostomia"])

        # Term should be stored in lowercase
        assert "xerostomia" in index.terms
        assert "XEROSTOMIA" not in index.terms

    def test_additional_terms_uppercase_in_list_still_indexed(self):
        """additional_terms in uppercase are normalized to lowercase for indexing."""
        text = "O paciente tem xerostomia"
        index = create_index(text, "test_var", additional_terms=["XEROSTOMIA"])

        # Should find the term despite uppercase in additional_terms
        assert "xerostomia" in index.terms

    def test_additional_terms_combined_with_default(self):
        """additional_terms are combined with DEFAULT_INDEX_TERMS."""
        text = "Tenho medo e xerostomia"
        index = create_index(text, "test_var", additional_terms=["xerostomia"])

        # Both default and custom terms should be indexed
        assert "medo" in index.terms  # from DEFAULT_INDEX_TERMS
        assert "xerostomia" in index.terms  # from additional_terms

    def test_additional_terms_not_found_not_indexed(self):
        """additional_terms not present in text are not in index."""
        text = "Texto simples sem os termos buscados"
        index = create_index(text, "test_var", additional_terms=["xerostomia", "epistaxe"])

        assert "xerostomia" not in index.terms
        assert "epistaxe" not in index.terms

    def test_additional_terms_with_entry_details(self):
        """additional_terms entries have correct linha and contexto."""
        text = "Primeira linha\nSegunda linha com termo_especial\nTerceira linha"
        index = create_index(text, "test_var", additional_terms=["termo_especial"])

        assert "termo_especial" in index.terms
        entry = index.terms["termo_especial"][0]
        assert entry["linha"] == 1
        assert "termo_especial" in entry["contexto"]

    def test_additional_terms_multiple_occurrences(self):
        """additional_terms with multiple occurrences create multiple entries."""
        text = "Linha 1 com termo\nLinha 2 com termo\nLinha 3 com termo"
        index = create_index(text, "test_var", additional_terms=["termo"])

        assert "termo" in index.terms
        assert len(index.terms["termo"]) == 3

    def test_additional_terms_empty_list(self):
        """additional_terms as empty list behaves like None."""
        text = "Tenho medo de trabalho"
        index = create_index(text, "test_var", additional_terms=[])

        # Should only have default terms
        assert "medo" in index.terms
        assert "trabalho" in index.terms
        assert index.custom_terms == []

    def test_additional_terms_preserves_original_case_in_custom_terms(self):
        """custom_terms field preserves the original case from input."""
        text = "Texto com MixedCase"
        index = create_index(text, "test_var", additional_terms=["MixedCase", "UPPERCASE"])

        # custom_terms preserves original input
        assert index.custom_terms == ["MixedCase", "UPPERCASE"]

    def test_additional_terms_with_special_characters(self):
        """additional_terms with Portuguese special characters work correctly."""
        text = "Sintomas de cefaléia e diarréia crônica"
        index = create_index(text, "test_var", additional_terms=["cefaléia", "diarréia"])

        assert "cefaléia" in index.terms
        assert "diarréia" in index.terms

    def test_additional_terms_duplicate_of_default_no_issue(self):
        """Duplicate terms (in both default and additional) don't cause issues."""
        text = "Texto com medo"
        # "medo" is already in DEFAULT_INDEX_TERMS
        index = create_index(text, "test_var", additional_terms=["medo"])

        # Should still work and find the term
        assert "medo" in index.terms
        # custom_terms stores what was passed
        assert index.custom_terms == ["medo"]

    def test_additional_terms_many_custom_terms(self):
        """additional_terms with many terms works correctly."""
        custom = [f"termo_{i}" for i in range(20)]
        text = "Linha com termo_5 e termo_10 e termo_15"
        index = create_index(text, "test_var", additional_terms=custom)

        assert "termo_5" in index.terms
        assert "termo_10" in index.terms
        assert "termo_15" in index.terms
        assert "termo_0" not in index.terms  # not in text
        assert len(index.custom_terms) == 20


class TestTextIndexSearch:
    """Test that TextIndex.search returns correct matches."""

    def test_search_returns_matches_for_indexed_term(self):
        """search returns list of matches for an indexed term."""
        text = "Linha 1 com medo\nLinha 2 com trabalho\nLinha 3 com medo"
        index = create_index(text, "test_var")

        results = index.search("medo")

        assert len(results) == 2
        assert results[0]["linha"] == 0
        assert results[1]["linha"] == 2

    def test_search_returns_empty_list_for_missing_term(self):
        """search returns empty list for term not in index."""
        text = "Texto simples sem termos especiais"
        index = create_index(text, "test_var")

        results = index.search("medo")

        assert results == []

    def test_search_is_case_insensitive(self):
        """search is case-insensitive (converts input to lowercase)."""
        text = "Tenho medo de trabalho"
        index = create_index(text, "test_var")

        # All these should find the same term
        assert index.search("medo") == index.search("MEDO")
        assert index.search("medo") == index.search("Medo")
        assert index.search("medo") == index.search("MeDo")

    def test_search_limit_parameter_restricts_results(self):
        """search respects limit parameter."""
        text = "\n".join([f"Linha {i} com medo" for i in range(20)])
        index = create_index(text, "test_var")

        results_default = index.search("medo")  # default limit=10
        results_limited = index.search("medo", limit=5)
        results_large = index.search("medo", limit=100)

        assert len(results_default) == 10
        assert len(results_limited) == 5
        assert len(results_large) == 20  # all available matches

    def test_search_limit_zero_returns_empty(self):
        """search with limit=0 returns empty list."""
        text = "Linha com medo"
        index = create_index(text, "test_var")

        results = index.search("medo", limit=0)

        assert results == []

    def test_search_result_contains_linha_key(self):
        """search results contain 'linha' key with line number."""
        text = "Primeira\nSegunda com medo\nTerceira"
        index = create_index(text, "test_var")

        results = index.search("medo")

        assert len(results) == 1
        assert "linha" in results[0]
        assert results[0]["linha"] == 1

    def test_search_result_contains_contexto_key(self):
        """search results contain 'contexto' key with line context."""
        text = "Esta linha contém medo e outros sentimentos"
        index = create_index(text, "test_var")

        results = index.search("medo")

        assert len(results) == 1
        assert "contexto" in results[0]
        assert "medo" in results[0]["contexto"]

    def test_search_preserves_result_order(self):
        """search returns results in line order (ascending)."""
        text = "Linha 0 medo\nLinha 1\nLinha 2 medo\nLinha 3\nLinha 4 medo"
        index = create_index(text, "test_var")

        results = index.search("medo")

        assert len(results) == 3
        assert results[0]["linha"] == 0
        assert results[1]["linha"] == 2
        assert results[2]["linha"] == 4

    def test_search_with_custom_term(self):
        """search works with custom terms added via additional_terms."""
        text = "O paciente apresenta xerostomia grave"
        index = create_index(text, "test_var", additional_terms=["xerostomia"])

        results = index.search("xerostomia")

        assert len(results) == 1
        assert "xerostomia" in results[0]["contexto"]

    def test_search_empty_string_returns_empty(self):
        """search for empty string returns empty list."""
        text = "Tenho medo de trabalho"
        index = create_index(text, "test_var")

        results = index.search("")

        assert results == []

    def test_search_on_empty_index(self):
        """search on empty index returns empty list."""
        index = create_index("", "empty_var")

        results = index.search("medo")

        assert results == []

    def test_search_with_portuguese_characters(self):
        """search works with Portuguese special characters."""
        text = "Sinto ansiedade e depressão frequentemente"
        index = create_index(text, "test_var")

        results_ansiedade = index.search("ansiedade")
        results_depressao = index.search("depressão")

        assert len(results_ansiedade) == 1
        assert len(results_depressao) == 1

    def test_search_does_not_modify_index(self):
        """search is read-only and doesn't modify the index."""
        text = "Tenho medo de trabalho"
        index = create_index(text, "test_var")
        original_terms = dict(index.terms)

        index.search("medo")
        index.search("trabalho")
        index.search("nonexistent")

        assert index.terms == original_terms


class TestSearchMultipleOrMode:
    """Test TextIndex.search_multiple with require_all=False (OR mode)."""

    def test_returns_dict_with_matching_terms(self):
        """search_multiple returns dict with term -> matches for each found term."""
        text = "Linha 1 com medo\nLinha 2 com trabalho\nLinha 3 com ansiedade"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=False)

        assert "medo" in results
        assert "trabalho" in results
        assert len(results["medo"]) == 1
        assert len(results["trabalho"]) == 1

    def test_omits_terms_not_found(self):
        """search_multiple with OR mode omits terms not in index."""
        text = "Texto apenas com medo"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=False)

        assert "medo" in results
        assert "trabalho" not in results

    def test_returns_empty_dict_when_no_terms_found(self):
        """search_multiple returns empty dict when no terms are found."""
        text = "Texto simples sem termos especiais"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=False)

        assert results == {}

    def test_is_case_insensitive(self):
        """search_multiple performs case-insensitive search."""
        text = "Tenho MEDO e TRABALHO"
        index = create_index(text, "test_var")

        results = index.search_multiple(["Medo", "TRABALHO"], require_all=False)

        assert "medo" in results or "Medo" in results  # dict keys are lowercase input
        # Actually, the method uses `t` (original case) as key, but search is lowercase
        # Let me verify: `{t: self.search(t) for t in terms if self.search(t)}`
        # So the key is the original term passed in, but search is case-insensitive
        assert len(results) == 2

    def test_preserves_original_term_as_key(self):
        """search_multiple uses original term case as dict key."""
        text = "Tenho medo de trabalho"
        index = create_index(text, "test_var")

        results = index.search_multiple(["MEDO", "Trabalho"], require_all=False)

        # Keys should be the original terms passed in
        assert "MEDO" in results
        assert "Trabalho" in results

    def test_with_single_term(self):
        """search_multiple works with a single term."""
        text = "Linha 1 com medo\nLinha 2 com medo"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo"], require_all=False)

        assert "medo" in results
        assert len(results["medo"]) == 2

    def test_with_empty_term_list(self):
        """search_multiple with empty term list returns empty dict."""
        text = "Tenho medo de trabalho"
        index = create_index(text, "test_var")

        results = index.search_multiple([], require_all=False)

        assert results == {}

    def test_multiple_occurrences_per_term(self):
        """search_multiple returns all occurrences for each term."""
        text = "Linha 1 medo\nLinha 2 medo\nLinha 3 medo\nLinha 4 trabalho"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=False)

        assert len(results["medo"]) == 3
        assert len(results["trabalho"]) == 1

    def test_matches_contain_linha_and_contexto(self):
        """search_multiple results contain linha and contexto keys."""
        text = "Primeira linha\nSegunda linha com medo\nTerceira linha"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo"], require_all=False)

        assert "medo" in results
        match = results["medo"][0]
        assert "linha" in match
        assert "contexto" in match
        assert match["linha"] == 1
        assert "medo" in match["contexto"]

    def test_with_custom_terms(self):
        """search_multiple works with custom terms from additional_terms."""
        text = "O paciente tem xerostomia e epistaxe"
        index = create_index(text, "test_var", additional_terms=["xerostomia", "epistaxe"])

        results = index.search_multiple(["xerostomia", "epistaxe"], require_all=False)

        assert "xerostomia" in results
        assert "epistaxe" in results

    def test_on_empty_index(self):
        """search_multiple on empty index returns empty dict."""
        index = create_index("", "empty_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=False)

        assert results == {}

    def test_with_many_terms(self):
        """search_multiple handles many terms efficiently."""
        text = "Linha com medo, ansiedade, raiva, tristeza e alegria"
        index = create_index(text, "test_var")
        terms = ["medo", "ansiedade", "raiva", "tristeza", "alegria", "depressão", "vergonha"]

        results = index.search_multiple(terms, require_all=False)

        # Only terms present in text should be in results
        assert "medo" in results
        assert "ansiedade" in results
        assert "raiva" in results
        assert "tristeza" in results
        assert "alegria" in results
        # These are not in the text
        assert "depressão" not in results
        assert "vergonha" not in results

    def test_default_require_all_is_false(self):
        """search_multiple defaults to require_all=False (OR mode)."""
        text = "Texto com medo e trabalho"
        index = create_index(text, "test_var")

        # Call without specifying require_all
        results = index.search_multiple(["medo", "trabalho"])

        # Should behave as OR mode (dict with term -> matches)
        assert isinstance(results, dict)
        assert "medo" in results
        assert "trabalho" in results


class TestSearchMultipleAndMode:
    """Test TextIndex.search_multiple with require_all=True (AND mode)."""

    def test_returns_lines_with_all_terms(self):
        """search_multiple with AND mode returns only lines containing ALL terms."""
        text = "Linha 1 com medo e trabalho\nLinha 2 apenas medo\nLinha 3 apenas trabalho"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=True)

        # Only line 0 has both terms
        assert 0 in results
        assert 1 not in results
        assert 2 not in results

    def test_returns_dict_with_linha_as_key(self):
        """search_multiple with AND mode returns dict with linha as key."""
        text = "Linha com medo e trabalho juntos"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=True)

        # Key should be line number (int), not term string
        assert isinstance(results, dict)
        for key in results.keys():
            assert isinstance(key, int)

    def test_returns_list_of_terms_as_value(self):
        """search_multiple with AND mode returns list of found terms as value."""
        text = "Linha com medo e trabalho"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=True)

        assert 0 in results
        # Value should be list of terms
        assert isinstance(results[0], list)
        assert "medo" in results[0]
        assert "trabalho" in results[0]

    def test_returns_empty_dict_when_no_line_has_all_terms(self):
        """search_multiple with AND returns empty dict when no line has all terms."""
        text = "Linha 1 com medo\nLinha 2 com trabalho"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=True)

        assert results == {}

    def test_returns_empty_dict_when_terms_not_found(self):
        """search_multiple with AND returns empty dict when terms not found."""
        text = "Texto simples sem termos especiais"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=True)

        assert results == {}

    def test_is_case_insensitive(self):
        """search_multiple with AND mode is case-insensitive."""
        text = "Linha com MEDO e TRABALHO"
        index = create_index(text, "test_var")

        results = index.search_multiple(["Medo", "trabalho"], require_all=True)

        # Should find the line despite case differences
        assert 0 in results

    def test_terms_in_result_are_lowercase(self):
        """search_multiple with AND mode returns terms in lowercase."""
        text = "Linha com medo e trabalho"
        index = create_index(text, "test_var")

        results = index.search_multiple(["MEDO", "TRABALHO"], require_all=True)

        assert 0 in results
        # Terms in result should be lowercase (per the code: term.lower())
        assert "medo" in results[0]
        assert "trabalho" in results[0]

    def test_multiple_lines_with_all_terms(self):
        """search_multiple with AND returns multiple lines if they have all terms."""
        text = "Linha 0 medo trabalho\nLinha 1 só medo\nLinha 2 medo trabalho\nLinha 3 medo trabalho"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=True)

        assert 0 in results
        assert 1 not in results
        assert 2 in results
        assert 3 in results

    def test_with_three_terms(self):
        """search_multiple with AND works with three or more terms."""
        text = "Linha 0 medo trabalho ansiedade\nLinha 1 medo trabalho\nLinha 2 medo ansiedade"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo", "trabalho", "ansiedade"], require_all=True)

        # Only line 0 has all three terms
        assert 0 in results
        assert 1 not in results
        assert 2 not in results

    def test_with_single_term(self):
        """search_multiple with AND and single term returns lines with that term."""
        text = "Linha 0 com medo\nLinha 1 sem\nLinha 2 com medo"
        index = create_index(text, "test_var")

        results = index.search_multiple(["medo"], require_all=True)

        assert 0 in results
        assert 1 not in results
        assert 2 in results

    def test_with_empty_term_list(self):
        """search_multiple with AND and empty term list returns empty dict."""
        text = "Tenho medo de trabalho"
        index = create_index(text, "test_var")

        results = index.search_multiple([], require_all=True)

        assert results == {}

    def test_with_custom_terms(self):
        """search_multiple with AND works with custom terms."""
        text = "O paciente tem xerostomia e epistaxe na mesma linha"
        index = create_index(text, "test_var", additional_terms=["xerostomia", "epistaxe"])

        results = index.search_multiple(["xerostomia", "epistaxe"], require_all=True)

        assert 0 in results
        assert "xerostomia" in results[0]
        assert "epistaxe" in results[0]

    def test_on_empty_index(self):
        """search_multiple with AND on empty index returns empty dict."""
        index = create_index("", "empty_var")

        results = index.search_multiple(["medo", "trabalho"], require_all=True)

        assert results == {}

    def test_different_from_or_mode(self):
        """AND mode returns different result structure than OR mode."""
        text = "Linha 0 medo e trabalho\nLinha 1 apenas medo"
        index = create_index(text, "test_var")

        or_results = index.search_multiple(["medo", "trabalho"], require_all=False)
        and_results = index.search_multiple(["medo", "trabalho"], require_all=True)

        # OR mode: dict with term -> matches
        assert "medo" in or_results
        assert "trabalho" in or_results

        # AND mode: dict with linha -> terms
        assert 0 in and_results
        assert 1 not in and_results


class TestAutoIndexIfLarge:
    """Test that auto_index_if_large indexes only texts >= min_chars threshold."""

    def test_returns_index_for_text_above_default_threshold(self, sample_text):
        """auto_index_if_large returns TextIndex for text >= 100k chars."""
        # sample_text fixture is ~1.45M chars, well above 100k
        result = auto_index_if_large(sample_text, "large_var")

        assert result is not None
        assert isinstance(result, TextIndex)
        assert result.var_name == "large_var"

    def test_returns_none_for_text_below_default_threshold(self):
        """auto_index_if_large returns None for text < 100k chars."""
        small_text = "medo trabalho ansiedade" * 1000  # ~24k chars
        result = auto_index_if_large(small_text, "small_var")

        assert result is None

    def test_returns_index_at_exact_threshold(self):
        """auto_index_if_large returns TextIndex at exactly 100k chars."""
        # Create text of exactly 100000 chars
        exact_text = "a" * 100000
        result = auto_index_if_large(exact_text, "exact_var")

        assert result is not None
        assert isinstance(result, TextIndex)
        assert result.total_chars == 100000

    def test_returns_none_one_char_below_threshold(self):
        """auto_index_if_large returns None at 99999 chars."""
        almost_text = "a" * 99999
        result = auto_index_if_large(almost_text, "almost_var")

        assert result is None

    def test_custom_min_chars_threshold_lower(self):
        """auto_index_if_large respects custom lower min_chars threshold."""
        text = "medo trabalho" * 1000  # ~13k chars
        result = auto_index_if_large(text, "test_var", min_chars=10000)

        assert result is not None
        assert result.var_name == "test_var"

    def test_custom_min_chars_threshold_higher(self):
        """auto_index_if_large respects custom higher min_chars threshold."""
        text = "medo trabalho" * 10000  # ~130k chars
        result = auto_index_if_large(text, "test_var", min_chars=200000)

        assert result is None

    def test_empty_text_returns_none(self):
        """auto_index_if_large returns None for empty text."""
        result = auto_index_if_large("", "empty_var")

        assert result is None

    def test_empty_text_with_zero_threshold_returns_index(self):
        """auto_index_if_large with min_chars=0 indexes empty text."""
        result = auto_index_if_large("", "empty_var", min_chars=0)

        assert result is not None
        assert result.total_chars == 0

    def test_index_contains_terms_from_text(self, sample_text):
        """auto_index_if_large creates proper index with indexed terms."""
        # sample_text contains "medo", "ansiedade", "trabalho", "família"
        result = auto_index_if_large(sample_text, "test_var")

        assert result is not None
        assert "medo" in result.terms
        assert "ansiedade" in result.terms
        assert "trabalho" in result.terms
        assert "família" in result.terms

    def test_index_has_correct_char_count(self):
        """auto_index_if_large creates index with correct total_chars."""
        text = "a" * 150000
        result = auto_index_if_large(text, "test_var")

        assert result is not None
        assert result.total_chars == 150000

    def test_index_has_correct_line_count(self):
        """auto_index_if_large creates index with correct total_lines."""
        # 100 lines of 1010 chars each = 101000 chars (above 100k)
        lines = ["medo " * 200 + "ansiedade"] * 100  # 100 lines, each ~1010 chars
        text = "\n".join(lines)
        result = auto_index_if_large(text, "test_var")

        assert result is not None
        assert result.total_lines == 100

    def test_default_threshold_is_100000(self):
        """auto_index_if_large uses 100000 as default min_chars."""
        # 99999 chars should return None
        text_99999 = "x" * 99999
        result_below = auto_index_if_large(text_99999, "below")
        assert result_below is None

        # 100000 chars should return index
        text_100000 = "x" * 100000
        result_at = auto_index_if_large(text_100000, "at")
        assert result_at is not None

    def test_uses_create_index_internally(self, sample_text):
        """auto_index_if_large creates index using create_index function."""
        # Verify that the returned index has the same structure as create_index would produce
        result = auto_index_if_large(sample_text, "test_var")
        direct_index = create_index(sample_text, "test_var")

        assert result is not None
        assert result.var_name == direct_index.var_name
        assert result.total_chars == direct_index.total_chars
        assert result.total_lines == direct_index.total_lines
        assert result.terms.keys() == direct_index.terms.keys()
