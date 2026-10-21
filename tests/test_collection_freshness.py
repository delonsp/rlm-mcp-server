"""A coleção deve refletir a fonte atual e reutilizar índices sem alterações."""
import uuid

from rlm_mcp import http_server as hs
from rlm_mcp.indexer import get_index


def _collection():
    name = 'fresh_' + uuid.uuid4().hex[:8]
    hs.call_tool('rlm_load_data', {'name': 'fresh_doc', 'data': 'xilofone azul antigo'})
    hs.call_tool('rlm_collection', {'action': 'create', 'name': name})
    hs.call_tool('rlm_collection', {'action': 'add', 'name': name, 'vars': ['fresh_doc']})
    return name


def _search(name, term):
    response = hs.call_tool('rlm_collection', {'action': 'search', 'name': name, 'terms': [term]})
    assert not response.get('isError'), response
    return response['content'][0]['text']


def test_reload_updates_collection_without_manual_rebuild():
    name = _collection()
    _search(name, 'xilofone')
    hs.call_tool('rlm_load_data', {'name': 'fresh_doc', 'data': 'zebras violeta novo'})
    assert 'zebras violeta novo' in _search(name, 'zebras')
    assert 'xilofone azul antigo' not in _search(name, 'xilofone')


def test_rebind_and_removal_never_leave_stale_passages():
    name = _collection()
    hs.repl.variables['fresh_doc'] = 'canguru laranja atual'
    assert 'canguru laranja atual' in _search(name, 'canguru')
    hs.repl.variables.pop('fresh_doc')
    assert 'canguru laranja atual' not in _search(name, 'canguru')


def test_unchanged_collection_reuses_index():
    name = _collection()
    _search(name, 'xilofone')
    index = get_index(f'_coll_{name}_combined')
    _search(name, 'azul')
    assert get_index(f'_coll_{name}_combined') is index


def test_router_honors_snippet_length():
    name = _collection()
    text = 'xilofone ' + 'abcde ' * 60
    hs.repl.variables['fresh_doc'] = text
    hs.call_tool('rlm_collection', {'action': 'rebuild', 'name': name})
    response = hs.call_tool('rlm_collection', {'action': 'search', 'name': name,
                                             'terms': ['xilofone'], 'snippet_len': 350})
    assert text[:300] in response['content'][0]['text']


def test_search_keeps_coherent_snapshot_when_source_changes_after_refresh(monkeypatch):
    from rlm_mcp.tools.handlers import collection_tools

    name = _collection()
    original_refresh = collection_tools.collection_snapshot

    def refresh_then_replace(*args, **kwargs):
        snapshot = original_refresh(*args, **kwargs)
        assert hs.repl.load_data('fresh_doc', 'substituição concorrente').success
        return snapshot

    monkeypatch.setattr(collection_tools, 'collection_snapshot', refresh_then_replace)
    response = _search(name, 'xilofone')
    assert 'xilofone azul antigo' in response
