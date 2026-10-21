"""Regressões reproduzidas na revisão: descarte, proveniência e recuperação."""
import gc
import json
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace
import uuid
import weakref

import pytest

from rlm_mcp import http_server as hs, pdf_parser
from rlm_mcp.document_reader import read_passage
from rlm_mcp.indexer import create_index, get_index
from rlm_mcp.repl import SafeREPL


class _TrackedText(str):
    """Texto com referência fraca para observar liberação, sem inspecionar caches."""


@pytest.fixture(autouse=True)
def _fresh_repl(monkeypatch):
    repl = SafeREPL()
    repl.sandbox_mode = "inprocess"
    monkeypatch.setattr(hs, "repl", repl)
    return repl


def _tool(tool_name, **arguments):
    response = hs.call_tool(tool_name, arguments)
    assert not response.get("isError"), response
    return response["content"][0]["text"]


def _collection(*names):
    name = "review_" + uuid.uuid4().hex[:12]
    _tool("rlm_collection", action="create", name=name)
    _tool("rlm_collection", action="add", name=name, vars=list(names))
    return name


@pytest.mark.parametrize("clear_all", [False, True], ids=["one", "all"])
def test_clear_releases_document_text_and_collection_index(clear_all):
    value = _TrackedText("passagem única " + uuid.uuid4().hex)
    source_ref = weakref.ref(value)
    assert hs.repl.load_data("clear_review_doc", value).success
    collection = _collection("clear_review_doc")
    json.loads(_tool("rlm_read", var_name="clear_review_doc"))
    combined_name = f"_coll_{collection}_combined"
    index_ref = weakref.ref(get_index(combined_name))
    del value

    _tool("rlm_clear", **({"all": True} if clear_all else {"name": "clear_review_doc"}))
    gc.collect()

    assert get_index(combined_name) is None
    assert index_ref() is None
    assert source_ref() is None, "A fonte removida continua retida por um cache."


def test_explicit_source_is_visible_with_text_when_load_unlocks():
    repl = hs.repl
    assert repl.load_data("atomic_doc", "texto do arquivo A").success
    repl.variable_metadata["atomic_doc"].source = "s3:books/A.txt"
    observations = []

    class ObserveFirstUnlock:
        """Simula um leitor que entra exatamente após a publicação do documento."""
        def __init__(self):
            self.lock = threading.Lock()
            self.observed = False

        def __enter__(self):
            self.lock.acquire()
            return self

        def __exit__(self, *exc):
            self.lock.release()
            if not self.observed:
                self.observed = True
                observations.append(read_passage(repl, {"var_name": "atomic_doc"}))

    repl._execute_lock = ObserveFirstUnlock()
    result = repl.load_data("atomic_doc", "texto do arquivo B", source="s3:books/B.txt")

    assert result.success, result.stderr
    assert observations
    assert observations[0]["text"] == "texto do arquivo B"
    assert observations[0]["source"] == "s3:books/B.txt"


@pytest.mark.parametrize("sandbox_mode", ["inprocess", "subprocess"])
def test_execute_rebind_marks_original_source_as_derived(sandbox_mode):
    hs.repl.sandbox_mode = sandbox_mode
    assert hs.repl.load_data("derived_doc", "texto original do PDF").success
    hs.repl.variable_metadata["derived_doc"].source = "s3:books/original.pdf"
    original = read_passage(hs.repl, {"var_name": "derived_doc"})

    result = hs.repl.execute("derived_doc = 'trecho que não existe no PDF original'")
    assert result.success, result.stderr
    passage = json.loads(_tool("rlm_read", var_name="derived_doc"))

    assert passage["text"] == "trecho que não existe no PDF original"
    assert passage["version"] != original["version"]
    assert passage["source"].startswith("derived:")
    assert "s3:books/original.pdf" in passage["source"]


def test_execute_without_source_change_keeps_original_provenance():
    assert hs.repl.load_data("unchanged_doc", "texto original").success
    hs.repl.variable_metadata["unchanged_doc"].source = "s3:books/unchanged.pdf"

    result = hs.repl.execute("print(len(unchanged_doc))")

    assert result.success, result.stderr
    passage = read_passage(hs.repl, {"var_name": "unchanged_doc"})
    assert passage["source"] == "s3:books/unchanged.pdf"


def test_portuguese_no_does_not_hide_reordered_chest_pain():
    text = "Peito: dor ardente\n\nDor no braço"
    index = create_index(text, "portuguese_review")

    hits = index.search_bm25(["dor", "no", "peito"], text, require_all=True)

    assert len(hits) == 1
    assert "Peito: dor ardente" in hits[0]["text"]


def test_english_no_remains_a_required_negation():
    text = "pain present\n\nno pain reported"
    index = create_index(text, "english_review")

    hits = index.search_bm25(["no", "pain"], text, require_all=True, query_language="en")

    assert len(hits) == 1
    assert "no pain reported" in hits[0]["text"]


@pytest.mark.parametrize("phrase,qualifier", [
    ("vitamina A e D", "D"),
    ("vitamina A/D/E", "D"),
    ("vitamina A/D/E", "E"),
])
def test_vitamin_qualifiers_in_enumerations_are_searchable(phrase, qualifier):
    text = phrase + " indicada\n\nvitamina C isolada"
    index = create_index(text, "enumeration_review")

    hits = index.search_bm25(["vitamina", qualifier], text, require_all=True)

    assert len(hits) == 1
    assert phrase in hits[0]["text"]


def test_vitamin_qualifier_is_preserved_when_query_terms_are_reordered():
    text = "vitamina A\n\nvitamina D\n\nvitamina E"
    index = create_index(text, "qualifier_order_review")

    hits = index.search_bm25(["D", "vitamina"], text, require_all=True)

    assert len(hits) == 1
    assert hits[0]["text"] == "vitamina D"


@pytest.mark.parametrize("removal", ["clear", "gc"])
@pytest.mark.parametrize("term", ["marcador", "palavra_inexistente"])
def test_collection_reports_missing_sources_even_without_hits(removal, term):
    missing_name = "review_missing_source"
    available_name = "review_available_source"
    assert hs.repl.load_data(missing_name, "marcador antigo " * 2500).success
    assert hs.repl.load_data(available_name, "marcador presente " * 1200).success
    hs.repl.pin_variable(available_name)
    collection = _collection(missing_name, available_name)
    if removal == "clear":
        _tool("rlm_clear", name=missing_name)
    else:
        hs.repl.max_memory_mb = 1
        hs.repl.cleanup_threshold_percent = 5
        hs.repl.cleanup_target_percent = 3
        assert hs.repl.load_data("review_gc_trigger", "disparo").success
    assert missing_name not in hs.repl.variables

    response = _tool("rlm_collection", action="search", name=collection, terms=[term])

    if term == "marcador":
        assert available_name in response
    assert missing_name in response
    assert "incomplet" in response.lower()


def _fake_pdfplumber(monkeypatch, text):
    page = SimpleNamespace(extract_text=lambda **kwargs: text)
    pdf = SimpleNamespace(pages=[page])
    monkeypatch.setitem(sys.modules, "pdfplumber", SimpleNamespace(open=lambda path: nullcontext(pdf)))


def test_pdf_with_isolated_nul_keeps_readable_extraction(tmp_path, monkeypatch):
    path = tmp_path / "readable.pdf"
    path.write_bytes(b"%PDF-synthetic")
    raw_text = "Texto legível de um documento. " * 20 + "fim\x00."
    _fake_pdfplumber(monkeypatch, raw_text)
    monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", lambda *args, **kwargs:
                        pdf_parser.PDFExtractionResult("", 0, "mistral_ocr", False, "OCR indisponível"))

    result = pdf_parser.extract_pdf(str(path))

    assert result.success, result.error
    assert raw_text.replace("\x00", "") in result.text
    assert "\x00" not in result.text


@pytest.mark.parametrize("raw_text", ["\x00" * 100 + "fim", "\ufffd" * 100 + "fim"])
def test_pdf_massive_corruption_is_not_made_valid_by_cleaning(tmp_path, monkeypatch, raw_text):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-synthetic")
    _fake_pdfplumber(monkeypatch, raw_text)

    result = pdf_parser.extract_pdf(str(path), method="pdfplumber")

    assert not result.success
    assert result.error


def test_pdf_preserves_both_extraction_and_ocr_failure_reasons(tmp_path, monkeypatch):
    path = tmp_path / "both_failed.pdf"
    path.write_bytes(b"%PDF-synthetic")
    extraction_error = "EXTRACTION_FAILURE_SENTINEL"
    ocr_error = "OCR_FAILURE_SENTINEL"
    monkeypatch.setattr(pdf_parser, "extract_with_pdfplumber", lambda *args, **kwargs:
                        pdf_parser.PDFExtractionResult("", 1, "pdfplumber", False, extraction_error))
    monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", lambda *args, **kwargs:
                        pdf_parser.PDFExtractionResult("", 0, "mistral_ocr", False, ocr_error))

    result = pdf_parser.extract_pdf(str(path))

    assert not result.success
    assert extraction_error in result.error
    assert ocr_error in result.error


def test_query_conjunction_does_not_become_vitamin_e():
    text = 'vitamina A e D indicada\n\nvitamina E isolada'
    index = create_index(text, 'query_conjunction_review')
    hits = index.search_bm25(['vitamina', 'A', 'e', 'D'], text, require_all=True)
    assert len(hits) == 1
    assert hits[0]['text'] == 'vitamina A e D indicada'
