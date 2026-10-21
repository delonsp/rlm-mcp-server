"""Regressões de entrada: conteúdo inválido nunca substitui uma fonte válida."""
import pytest

from rlm_mcp import http_server as hs, pdf_parser


@pytest.mark.parametrize('bad', ['%PDF-1.7\nstream', '\x89PNG\r\n', 'abc\x00def', '\ufffd' * 120])
def test_invalid_text_keeps_previous_document(bad):
    hs.call_tool('rlm_load_data', {'name': 'ingestion_doc', 'data': 'texto válido anterior'})
    result = hs.call_tool('rlm_load_data', {'name': 'ingestion_doc', 'data': bad})
    assert result.get('isError')
    assert hs.repl.variables['ingestion_doc'] == 'texto válido anterior'
    assert hs.get_persistence().load_variable('ingestion_doc') == 'texto válido anterior'


@pytest.mark.parametrize('text', ['ação, pressão, coração; α β ≥ 12 mg', '漢字 العربية', '--- Página 1 ---\ntexto\fcontinua', ''])
def test_valid_unicode_and_empty_text_remain_supported(text):
    result = hs.repl.load_data('valid_doc', text)
    assert result.success, result.stderr
    assert hs.repl.variables['valid_doc'] == text


def test_bad_pdf_extraction_tries_ocr(tmp_path, monkeypatch):
    path = tmp_path / 'doc.pdf'
    path.write_bytes(b'%PDF-fake')
    bad = pdf_parser.PDFExtractionResult('\ufffd' * 120, 1, 'pdfplumber', True)
    good = pdf_parser.PDFExtractionResult('--- Página 1 ---\nTexto legível da fonte.', 1, 'mistral_ocr', True)
    monkeypatch.setattr(pdf_parser, 'extract_with_pdfplumber', lambda *a, **kw: bad)
    monkeypatch.setattr(pdf_parser, 'extract_with_mistral_ocr', lambda *a, **kw: good)
    result = pdf_parser.extract_pdf(str(path))
    assert result.success and result.text == good.text


@pytest.mark.parametrize('method', ['auto', 'pdfplumber', 'ocr'])
def test_bad_pdf_never_published_as_success(tmp_path, monkeypatch, method):
    path = tmp_path / 'doc.pdf'
    path.write_bytes(b'%PDF-fake')
    def bad(*args, **kwargs):
        return pdf_parser.PDFExtractionResult('\ufffd' * 120, 1, 'synthetic', True)
    monkeypatch.setattr(pdf_parser, 'extract_with_pdfplumber', bad)
    monkeypatch.setattr(pdf_parser, 'extract_with_mistral_ocr', bad)
    result = pdf_parser.extract_pdf(str(path), method=method)
    assert not result.success
    assert result.error
