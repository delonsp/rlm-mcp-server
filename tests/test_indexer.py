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
