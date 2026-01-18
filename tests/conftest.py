"""
Pytest fixtures for RLM MCP Server tests.
"""

import os
import tempfile

import pytest


@pytest.fixture
def temp_db():
    """
    Creates a temporary SQLite database file for testing.

    Yields the path to the temporary database, then cleans up after the test.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_text():
    """
    Generates a large text (~200k characters) for testing indexation.

    Contains common terms used in semantic indexing: emotions, relationships,
    work, physical symptoms, etc.
    """
    # Base text with varied terms for realistic indexing tests
    base = "medo ansiedade trabalho família saúde dor cabeça estômago "
    # Repeat to get ~200k characters (base is ~60 chars, 25000 * 8 words)
    return base * 25000  # ~200k characters
