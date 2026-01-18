"""
Tests to validate conftest.py fixtures work correctly.
"""

import os


def test_temp_db_fixture_creates_file(temp_db):
    """Verify temp_db fixture creates a valid file path."""
    assert temp_db.endswith(".db")
    assert os.path.exists(temp_db)


def test_sample_text_fixture_is_large_enough(sample_text):
    """Verify sample_text fixture generates ~200k+ characters."""
    assert len(sample_text) >= 200_000
    assert "medo" in sample_text
    assert "trabalho" in sample_text
    assert "ansiedade" in sample_text
