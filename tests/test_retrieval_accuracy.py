"""Casos de recuperação sensíveis a versões, qualificadores curtos e negação."""
from rlm_mcp import http_server as hs
from rlm_mcp.indexer import create_index, fingerprint_source
from rlm_mcp.vector_index import VectorIndex, _chunk_text
import pytest


def test_version_detects_same_length_edit_outside_old_samples():
    old = 'a' * 400_000
    new = old[:100_000] + 'b' + old[100_001:]
    assert fingerprint_source(old) != fingerprint_source(new)


def test_vitamin_qualifier_is_not_dropped():
    text = 'vitamina A marcador alfa\n\nvitamina D marcador delta\n\nvitamina E marcador eco'
    index = create_index(text, 'vitamin_doc')
    for qualifier in ('A', 'D', 'E'):
        hits = index.search_bm25(['vitamina', qualifier], text, require_all=True)
        assert len(hits) == 1
        assert f'vitamina {qualifier}' in hits[0]['text']


def test_negation_is_not_dropped_from_query():
    text = 'tratamento indicado\n\ntratamento não indicado'
    index = create_index(text, 'negation_doc')
    hits = index.search_bm25(['tratamento', 'não', 'indicado'], text, require_all=True)
    assert len(hits) == 1 and 'não indicado' in hits[0]['text']


def test_reload_clears_persisted_old_indices():
    p = hs.get_persistence()
    p.save_variable('reload_doc', 'antigo')
    p.save_index('reload_doc', create_index('antigo', 'reload_doc').to_dict())
    p.save_variable('reload_doc', 'novo')
    assert p.load_index('reload_doc') is None


def test_query_order_has_stable_ties():
    text = 'alfa beta\n\nbeta alfa\n\nalfa beta'
    index = create_index(text, 'stable_doc')
    a = index.search_bm25(['alfa', 'beta'], text)
    b = index.search_bm25(['beta', 'alfa'], text)
    assert a == b
    assert [h['line'] for h in a] == sorted(h['line'] for h in a)


def _persisted_chunks(text):
    return [{'chunk_index': c.chunk_index, 'chunk_text': c.text,
             'line_start': c.line_start, 'line_end': c.line_end,
             'embedding': [1.0, 0.0] if c.chunk_index % 2 == 0 else []}
            for c in _chunk_text(text)]


@pytest.mark.parametrize('change', ['replace', 'append', 'lines', 'missing'])
def test_restore_refuses_vectors_that_do_not_match_current_source(change):
    text = 'passagem original\n' * 100
    chunks = _persisted_chunks(text)
    if change == 'replace':
        text = text.replace('original', 'alterada')
    elif change == 'append':
        text += 'conteúdo novo' * 100
    elif change == 'lines':
        chunks[0]['line_start'] += 1
    else:
        chunks.pop()
    with pytest.raises(ValueError, match='fonte'):
        VectorIndex.from_persisted('restore_doc', text, chunks)


def test_restore_reuses_verified_vectors_preserving_partial_coverage():
    text = 'passagem original\n' * 100
    chunks = _persisted_chunks(text)
    restored = VectorIndex.from_persisted('restore_doc', text, chunks)
    stats = restored.get_stats()
    assert stats['total_chunks'] == len(chunks)
    assert 0 < stats['embedded_chunks'] < stats['total_chunks']
    assert restored.source_fingerprint == fingerprint_source(text)


def test_startup_ignores_stale_vectors_and_restores_other_documents(tmp_path, monkeypatch):
    import asyncio
    from rlm_mcp.persistence import PersistenceManager
    from rlm_mcp.vector_index import get_vector_index

    persistence = PersistenceManager(str(tmp_path / 'restore.db'))
    persistence.save_variable('valid_restore', 'texto correto')
    persistence.save_embeddings('valid_restore', _persisted_chunks('texto correto'))
    persistence.save_variable('stale_restore', 'texto atual')
    persistence.save_embeddings('stale_restore', _persisted_chunks('texto obsoleto'))
    monkeypatch.setattr(hs, 'get_persistence', lambda: persistence)
    monkeypatch.setattr(hs.task_manager, 'shutdown', lambda **kw: None)

    async def restore():
        async with hs.lifespan(hs.app):
            assert hs.repl.variables['stale_restore'] == 'texto atual'
            assert hs.repl.variables['valid_restore'] == 'texto correto'
            assert get_vector_index('stale_restore') is None
            assert get_vector_index('valid_restore') is not None

    asyncio.run(restore())
