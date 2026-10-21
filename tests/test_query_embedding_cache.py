"""Repeated queries reuse successful vectors; errors remain retryable."""
from rlm_mcp.embeddings import EmbeddingService


def test_repeated_query_reuses_embedding_without_sharing_mutable_result(monkeypatch):
    service = EmbeddingService(mode='disabled')
    service.mode = 'synthetic'
    calls = []
    def embed(texts):
        calls.append(texts)
        return [[1.0, 2.0]]
    monkeypatch.setattr(service, 'embed_texts', embed)
    first = service.embed_query('same query')
    first[0] = 99
    service.last_error = 'earlier failure'
    assert service.embed_query('same query') == [1.0, 2.0]
    assert len(calls) == 1
    assert service.last_error is None


def test_failed_queries_are_not_cached(monkeypatch):
    service = EmbeddingService(mode='disabled')
    service.mode = 'synthetic'
    replies = iter([[[]], [[1.0, 2.0]]])
    monkeypatch.setattr(service, 'embed_texts', lambda texts: next(replies))
    assert service.embed_query('retry query') == []
    assert service.embed_query('retry query') == [1.0, 2.0]


def test_model_change_invalidates_query_cache(monkeypatch):
    service = EmbeddingService(mode='disabled')
    service.mode = 'synthetic'
    replies = iter([[[1.0]], [[2.0]]])
    monkeypatch.setattr(service, 'embed_texts', lambda texts: next(replies))
    assert service.embed_query('query') == [1.0]
    service._model = 'different-model'
    assert service.embed_query('query') == [2.0]
