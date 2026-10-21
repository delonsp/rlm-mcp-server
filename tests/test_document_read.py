"""Leitura exata, paginada e vinculada à versão original do texto."""
import json

from rlm_mcp import http_server as hs
from rlm_mcp.tools.schemas import TOOL_SCHEMAS


def _read(**args):
    response = hs.call_tool('rlm_read', {'var_name': 'read_doc', **args})
    assert not response.get('isError'), response
    return json.loads(response['content'][0]['text'])


def test_read_is_public_tool():
    assert 'rlm_read' in {t['name'] for t in TOOL_SCHEMAS}


def test_pagination_reassembles_exact_text_including_long_line_and_unicode():
    text = '--- Página 1 ---\n' + 'áβ漢字 ' * 250 + '\n--- Página 2 ---\nfim\n'
    hs.call_tool('rlm_load_data', {'name': 'read_doc', 'data': text})
    parts = []
    args = {'max_chars': 73}
    versions = set()
    for _ in range(100):
        page = _read(**args)
        parts.append(page['text'])
        versions.add(page['version'])
        if not page['next']:
            break
        args = {k: v for k, v in page['next'].items() if k != 'var_name'}
        args['max_chars'] = 73
    assert ''.join(parts) == text
    assert len(versions) == 1


def test_read_reports_source_page_and_line():
    hs.repl.load_data('read_doc', '--- Página 7 ---\nprimeiro\nsegundo\n--- Página 9 ---\noutro')
    hs.repl.variable_metadata['read_doc'].source = 's3:books/original.pdf'
    page = _read(start_line=2, line_count=2)
    assert page['text'] == 'primeiro\nsegundo\n'
    assert page['source'] == 's3:books/original.pdf'
    assert page['line_start'] == 2 and page['line_end'] == 3
    assert page['pages'] == [7]


def test_read_does_not_invent_pdf_page():
    hs.repl.load_data('read_doc', 'texto sem marcadores')
    assert _read()['pages'] == []


def test_read_refuses_continuation_of_old_version():
    hs.repl.load_data('read_doc', 'primeira versão ' * 20)
    old = _read(max_chars=20)
    hs.repl.load_data('read_doc', 'versão substituída ' * 20)
    response = hs.call_tool('rlm_read', old['next'])
    assert response.get('isError')
    assert 'versão' in response['content'][0]['text'].lower()


def test_read_rejects_invalid_ranges_and_nontext():
    hs.repl.load_data('read_doc', 'uma linha')
    for args in ({'start_line': 0}, {'start_line': 4}, {'max_chars': 0},
                 {'max_chars': 20001}, {'offset': -1}, {'line_count': 0}):
        assert hs.call_tool('rlm_read', {'var_name': 'read_doc', **args}).get('isError')
    hs.repl.variables['read_doc'] = {'not': 'text'}
    assert hs.call_tool('rlm_read', {'var_name': 'read_doc'}).get('isError')


def test_empty_text_is_readable():
    hs.repl.load_data('read_doc', '')
    page = _read()
    assert page['text'] == '' and page['next'] is None


def test_source_metadata_survives_persistence():
    from rlm_mcp.services.persistence_service import persist_and_index
    hs.repl.load_data('read_doc', 'texto original')
    hs.repl.variable_metadata['read_doc'].source = 's3:books/doc.txt'
    persist_and_index('read_doc', 'texto original', hs.repl)
    saved = next(v for v in hs.get_persistence().list_variables() if v['name'] == 'read_doc')
    assert saved.get('source') == 's3:books/doc.txt'
