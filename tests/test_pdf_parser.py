"""
Tests for pdf_parser.py module.

These tests cover:
- extract_with_pdfplumber: Machine readable PDF extraction
- extract_with_mistral_ocr: OCR extraction (mocked)
- extract_pdf: Auto-detection and fallback logic
"""

import os
import tempfile

import pytest

from rlm_mcp.pdf_parser import (
    PDFExtractionResult,
    extract_with_pdfplumber,
    extract_with_mistral_ocr,
    extract_pdf,
    split_pdf_into_chunks,
)


# ============================================================================
# PDF Fixtures
# ============================================================================


@pytest.fixture
def sample_pdf():
    """
    Creates a simple machine-readable PDF file for testing.

    Uses reportlab to generate a PDF with selectable text.
    The PDF contains 2 pages with known text content.

    Yields:
        str: Path to the temporary PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    # Create temp file
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    # Generate PDF with text
    c = canvas.Canvas(path, pagesize=letter)

    # Page 1
    c.drawString(100, 750, "Hello World")
    c.drawString(100, 730, "This is page one of the test PDF.")
    c.drawString(100, 710, "It contains machine readable text.")
    c.showPage()

    # Page 2
    c.drawString(100, 750, "Page Two")
    c.drawString(100, 730, "This is the second page.")
    c.drawString(100, 710, "It also has selectable text content.")
    c.showPage()

    c.save()

    yield path

    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_pdf_single_page():
    """
    Creates a single-page PDF for testing.

    Yields:
        str: Path to the temporary PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=letter)
    c.drawString(100, 750, "Single Page PDF")
    c.drawString(100, 730, "This PDF has only one page.")
    c.showPage()
    c.save()

    yield path

    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_pdf_many_pages():
    """
    Creates a PDF with 10 pages for testing pagination.

    Yields:
        str: Path to the temporary PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=letter)

    for i in range(1, 11):
        c.drawString(100, 750, f"Page {i} of 10")
        c.drawString(100, 730, f"Content for page number {i}.")
        c.showPage()

    c.save()

    yield path

    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_pdf_empty_pages():
    """
    Creates a PDF with some empty pages (no text).

    Yields:
        str: Path to the temporary PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=letter)

    # Page 1 - has text
    c.drawString(100, 750, "Page with text")
    c.showPage()

    # Page 2 - empty (no text)
    c.showPage()

    # Page 3 - has text
    c.drawString(100, 750, "Another page with text")
    c.showPage()

    c.save()

    yield path

    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_pdf_unicode():
    """
    Creates a PDF with Unicode/international characters.

    Yields:
        str: Path to the temporary PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=letter)

    # Use UTF-8 compatible fonts for reportlab
    c.drawString(100, 750, "Portuguese: Olá Mundo, saúde, coração")
    c.drawString(100, 730, "Spanish: Señor, España")
    c.drawString(100, 710, "French: Café, résumé")
    c.showPage()

    c.save()

    yield path

    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_pdf_long_text():
    """
    Creates a PDF with long paragraphs of text.

    Yields:
        str: Path to the temporary PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=letter)

    long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20
    # Split into lines that fit on page
    y = 750
    for i in range(0, len(long_text), 80):
        line = long_text[i : i + 80]
        if y > 100:
            c.drawString(50, y, line)
            y -= 15
        else:
            break

    c.showPage()
    c.save()

    yield path

    if os.path.exists(path):
        os.unlink(path)


# ============================================================================
# Tests for extract_with_pdfplumber
# ============================================================================


class TestExtractWithPdfplumber:
    """Tests for extract_with_pdfplumber function."""

    def test_returns_pdf_extraction_result(self, sample_pdf):
        """extract_with_pdfplumber returns PDFExtractionResult object."""
        result = extract_with_pdfplumber(sample_pdf)
        assert isinstance(result, PDFExtractionResult)

    def test_returns_success_true_for_valid_pdf(self, sample_pdf):
        """extract_with_pdfplumber returns success=True for valid machine readable PDF."""
        result = extract_with_pdfplumber(sample_pdf)
        assert result.success is True

    def test_returns_method_pdfplumber(self, sample_pdf):
        """extract_with_pdfplumber returns method='pdfplumber'."""
        result = extract_with_pdfplumber(sample_pdf)
        assert result.method == "pdfplumber"

    def test_returns_correct_page_count(self, sample_pdf):
        """extract_with_pdfplumber returns correct page count."""
        result = extract_with_pdfplumber(sample_pdf)
        assert result.pages == 2

    def test_returns_correct_page_count_single_page(self, sample_pdf_single_page):
        """extract_with_pdfplumber returns pages=1 for single page PDF."""
        result = extract_with_pdfplumber(sample_pdf_single_page)
        assert result.pages == 1

    def test_returns_correct_page_count_many_pages(self, sample_pdf_many_pages):
        """extract_with_pdfplumber returns correct count for PDF with many pages."""
        result = extract_with_pdfplumber(sample_pdf_many_pages)
        assert result.pages == 10

    def test_extracts_text_content(self, sample_pdf):
        """extract_with_pdfplumber extracts text from PDF."""
        result = extract_with_pdfplumber(sample_pdf)
        assert "Hello World" in result.text

    def test_extracts_text_from_all_pages(self, sample_pdf):
        """extract_with_pdfplumber extracts text from all pages."""
        result = extract_with_pdfplumber(sample_pdf)
        assert "Hello World" in result.text  # Page 1
        assert "Page Two" in result.text  # Page 2

    def test_includes_page_markers(self, sample_pdf):
        """extract_with_pdfplumber includes page markers in output."""
        result = extract_with_pdfplumber(sample_pdf)
        assert "--- Página 1 ---" in result.text
        assert "--- Página 2 ---" in result.text

    def test_page_markers_for_many_pages(self, sample_pdf_many_pages):
        """extract_with_pdfplumber includes correct page markers for many pages."""
        result = extract_with_pdfplumber(sample_pdf_many_pages)
        for i in range(1, 11):
            assert f"--- Página {i} ---" in result.text

    def test_skips_empty_pages(self, sample_pdf_empty_pages):
        """extract_with_pdfplumber skips pages with no text content."""
        result = extract_with_pdfplumber(sample_pdf_empty_pages)
        # Should have markers for page 1 and 3 (which have text)
        assert "--- Página 1 ---" in result.text
        assert "--- Página 3 ---" in result.text
        # Page 2 is empty, so no marker (text_parts only includes non-empty)
        assert "--- Página 2 ---" not in result.text

    def test_returns_page_count_including_empty(self, sample_pdf_empty_pages):
        """extract_with_pdfplumber returns total page count including empty pages."""
        result = extract_with_pdfplumber(sample_pdf_empty_pages)
        # pages count is len(pdf.pages), not just non-empty pages
        assert result.pages == 3

    def test_handles_unicode_text(self, sample_pdf_unicode):
        """extract_with_pdfplumber handles Unicode/international characters."""
        result = extract_with_pdfplumber(sample_pdf_unicode)
        # Note: reportlab may not render all Unicode perfectly, but basic Latin extended should work
        assert result.success is True
        assert len(result.text) > 0

    def test_extracts_long_text(self, sample_pdf_long_text):
        """extract_with_pdfplumber extracts long text content."""
        result = extract_with_pdfplumber(sample_pdf_long_text)
        assert "Lorem ipsum" in result.text

    def test_returns_error_none_on_success(self, sample_pdf):
        """extract_with_pdfplumber returns error=None on success."""
        result = extract_with_pdfplumber(sample_pdf)
        assert result.error is None

    def test_returns_string_text(self, sample_pdf):
        """extract_with_pdfplumber returns text as string type."""
        result = extract_with_pdfplumber(sample_pdf)
        assert isinstance(result.text, str)

    def test_returns_int_pages(self, sample_pdf):
        """extract_with_pdfplumber returns pages as int type."""
        result = extract_with_pdfplumber(sample_pdf)
        assert isinstance(result.pages, int)

    def test_text_not_empty(self, sample_pdf):
        """extract_with_pdfplumber returns non-empty text for valid PDF."""
        result = extract_with_pdfplumber(sample_pdf)
        assert len(result.text.strip()) > 0

    def test_preserves_line_breaks(self, sample_pdf):
        """extract_with_pdfplumber preserves some form of line separation."""
        result = extract_with_pdfplumber(sample_pdf)
        # pdfplumber with layout=True should have newlines
        assert "\n" in result.text

    def test_pages_separated_by_double_newline(self, sample_pdf):
        """extract_with_pdfplumber separates pages with double newline."""
        result = extract_with_pdfplumber(sample_pdf)
        # Format: "page1_text\n\npage2_text" (joined with \n\n)
        assert "\n\n" in result.text

    def test_single_page_content_correct(self, sample_pdf_single_page):
        """extract_with_pdfplumber extracts correct content from single page PDF."""
        result = extract_with_pdfplumber(sample_pdf_single_page)
        assert "Single Page PDF" in result.text
        assert "This PDF has only one page" in result.text

    def test_many_pages_content_correct(self, sample_pdf_many_pages):
        """extract_with_pdfplumber extracts content from all pages in multi-page PDF."""
        result = extract_with_pdfplumber(sample_pdf_many_pages)
        for i in range(1, 11):
            assert f"Page {i} of 10" in result.text


# ============================================================================
# Tests for extract_with_pdfplumber - File Not Exists
# ============================================================================


class TestExtractWithPdfplumberFileNotExists:
    """Tests for extract_with_pdfplumber when file does not exist."""

    def test_returns_pdf_extraction_result_for_nonexistent_file(self):
        """extract_with_pdfplumber returns PDFExtractionResult for nonexistent file."""
        result = extract_with_pdfplumber("/nonexistent/path/to/file.pdf")
        assert isinstance(result, PDFExtractionResult)

    def test_returns_success_false_for_nonexistent_file(self):
        """extract_with_pdfplumber returns success=False when file doesn't exist."""
        result = extract_with_pdfplumber("/nonexistent/path/to/file.pdf")
        assert result.success is False

    def test_returns_error_message_for_nonexistent_file(self):
        """extract_with_pdfplumber returns error message when file doesn't exist."""
        result = extract_with_pdfplumber("/nonexistent/path/to/file.pdf")
        assert result.error is not None
        assert len(result.error) > 0

    def test_returns_empty_text_for_nonexistent_file(self):
        """extract_with_pdfplumber returns empty text when file doesn't exist."""
        result = extract_with_pdfplumber("/nonexistent/path/to/file.pdf")
        assert result.text == ""

    def test_returns_zero_pages_for_nonexistent_file(self):
        """extract_with_pdfplumber returns pages=0 when file doesn't exist."""
        result = extract_with_pdfplumber("/nonexistent/path/to/file.pdf")
        assert result.pages == 0

    def test_returns_method_pdfplumber_for_nonexistent_file(self):
        """extract_with_pdfplumber returns method='pdfplumber' even on error."""
        result = extract_with_pdfplumber("/nonexistent/path/to/file.pdf")
        assert result.method == "pdfplumber"

    def test_error_mentions_file_path_or_error_type(self):
        """extract_with_pdfplumber error contains relevant info about the failure."""
        path = "/nonexistent/path/to/file.pdf"
        result = extract_with_pdfplumber(path)
        # The error should mention the file or be a relevant OS/file error
        assert result.error is not None
        # pdfplumber raises FileNotFoundError or similar OS errors
        error_lower = result.error.lower()
        assert (
            "no such file" in error_lower
            or "not found" in error_lower
            or "does not exist" in error_lower
            or path in result.error
            or "errno" in error_lower
        )

    def test_handles_empty_path(self):
        """extract_with_pdfplumber handles empty string path."""
        result = extract_with_pdfplumber("")
        assert result.success is False
        assert result.error is not None

    def test_handles_directory_path(self, tmp_path):
        """extract_with_pdfplumber handles path that is a directory, not file."""
        result = extract_with_pdfplumber(str(tmp_path))
        assert result.success is False
        assert result.error is not None

    def test_handles_path_with_special_characters(self):
        """extract_with_pdfplumber handles path with special characters."""
        result = extract_with_pdfplumber("/path/with spaces/and-dashes/file (1).pdf")
        assert result.success is False
        assert result.error is not None

    def test_handles_unicode_path(self):
        """extract_with_pdfplumber handles path with unicode characters."""
        result = extract_with_pdfplumber("/caminho/português/arquivo.pdf")
        assert result.success is False
        assert result.error is not None

    def test_does_not_raise_exception(self):
        """extract_with_pdfplumber does not raise exception for nonexistent file."""
        # Should not raise, should return PDFExtractionResult with success=False
        try:
            result = extract_with_pdfplumber("/nonexistent/file.pdf")
            assert result is not None  # Got result without exception
        except Exception as e:
            pytest.fail(f"extract_with_pdfplumber raised exception: {e}")

    def test_handles_path_with_null_byte(self):
        """extract_with_pdfplumber handles path with embedded null byte gracefully."""
        result = extract_with_pdfplumber("/path/with\x00null/file.pdf")
        # Should either return error or fail gracefully (not crash)
        assert result.success is False
        assert result.error is not None

    def test_nonexistent_with_valid_extension(self):
        """extract_with_pdfplumber fails for .pdf path that doesn't exist."""
        result = extract_with_pdfplumber("/tmp/definitely_does_not_exist_12345.pdf")
        assert result.success is False

    def test_nonexistent_with_no_extension(self):
        """extract_with_pdfplumber fails for path without .pdf extension that doesn't exist."""
        result = extract_with_pdfplumber("/tmp/definitely_does_not_exist_12345")
        assert result.success is False


# ============================================================================
# Tests for extract_pdf with method="auto" uses pdfplumber first
# ============================================================================


class TestExtractPdfAutoUsesPdfplumberFirst:
    """Tests for extract_pdf with method='auto' using pdfplumber as first extraction method."""

    def test_returns_pdf_extraction_result(self, sample_pdf):
        """extract_pdf with method='auto' returns PDFExtractionResult."""
        result = extract_pdf(sample_pdf, method="auto")
        assert isinstance(result, PDFExtractionResult)

    def test_returns_success_true_for_machine_readable_pdf(self, sample_pdf):
        """extract_pdf with method='auto' returns success=True for machine readable PDF."""
        result = extract_pdf(sample_pdf, method="auto")
        assert result.success is True

    def test_uses_pdfplumber_method_for_text_rich_pdf(self, sample_pdf):
        """extract_pdf with method='auto' returns method='pdfplumber' when text is sufficient."""
        result = extract_pdf(sample_pdf, method="auto")
        assert result.method == "pdfplumber"

    def test_extracts_text_from_pdf(self, sample_pdf):
        """extract_pdf with method='auto' extracts text using pdfplumber."""
        result = extract_pdf(sample_pdf, method="auto")
        assert "Hello World" in result.text

    def test_extracts_text_from_all_pages(self, sample_pdf):
        """extract_pdf with method='auto' extracts text from all pages."""
        result = extract_pdf(sample_pdf, method="auto")
        assert "Hello World" in result.text  # Page 1
        assert "Page Two" in result.text  # Page 2

    def test_returns_correct_page_count(self, sample_pdf):
        """extract_pdf with method='auto' returns correct page count."""
        result = extract_pdf(sample_pdf, method="auto")
        assert result.pages == 2

    def test_single_page_pdf(self, sample_pdf_single_page):
        """extract_pdf with method='auto' works for single page PDF."""
        result = extract_pdf(sample_pdf_single_page, method="auto")
        assert result.success is True
        assert result.method == "pdfplumber"
        assert result.pages == 1
        assert "Single Page PDF" in result.text

    def test_many_pages_pdf(self, sample_pdf_many_pages):
        """extract_pdf with method='auto' works for PDF with many pages."""
        result = extract_pdf(sample_pdf_many_pages, method="auto")
        assert result.success is True
        assert result.method == "pdfplumber"
        assert result.pages == 10

    def test_long_text_pdf(self, sample_pdf_long_text):
        """extract_pdf with method='auto' works for PDF with long text."""
        result = extract_pdf(sample_pdf_long_text, method="auto")
        assert result.success is True
        assert result.method == "pdfplumber"
        assert "Lorem ipsum" in result.text

    def test_returns_error_none_on_success(self, sample_pdf):
        """extract_pdf with method='auto' returns error=None on success."""
        result = extract_pdf(sample_pdf, method="auto")
        assert result.error is None

    def test_default_method_is_auto(self, sample_pdf):
        """extract_pdf default method is 'auto'."""
        result = extract_pdf(sample_pdf)  # No method specified
        assert result.success is True
        assert result.method == "pdfplumber"

    def test_includes_page_markers(self, sample_pdf):
        """extract_pdf with method='auto' includes page markers from pdfplumber."""
        result = extract_pdf(sample_pdf, method="auto")
        assert "--- Página 1 ---" in result.text
        assert "--- Página 2 ---" in result.text

    def test_file_not_found_returns_error(self):
        """extract_pdf with method='auto' returns error for nonexistent file."""
        result = extract_pdf("/nonexistent/path/file.pdf", method="auto")
        assert result.success is False
        assert result.error is not None
        assert "não encontrado" in result.error or "not found" in result.error.lower()

    def test_file_not_found_returns_method_none(self):
        """extract_pdf returns method='none' for file not found error."""
        result = extract_pdf("/nonexistent/path/file.pdf", method="auto")
        assert result.method == "none"

    def test_does_not_call_ocr_for_text_rich_pdf(self, sample_pdf, monkeypatch):
        """extract_pdf with method='auto' does not call OCR when pdfplumber succeeds."""
        ocr_called = []

        def mock_ocr(path):
            ocr_called.append(path)
            return PDFExtractionResult(
                text="OCR text",
                pages=1,
                method="mistral_ocr",
                success=True
            )

        # Monkeypatch the OCR function
        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        result = extract_pdf(sample_pdf, method="auto")

        # Should use pdfplumber, not OCR
        assert result.method == "pdfplumber"
        assert len(ocr_called) == 0

    def test_min_chars_threshold_default_100(self, sample_pdf, monkeypatch):
        """extract_pdf with method='auto' uses default min_chars_threshold=100."""
        # The sample_pdf has way more than 100 chars, so should pass threshold

        # Track if OCR is called
        ocr_called = []

        def mock_ocr(path):
            ocr_called.append(path)
            return PDFExtractionResult(
                text="OCR text",
                pages=1,
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        result = extract_pdf(sample_pdf, method="auto")

        # sample_pdf has "Hello World" etc, way more than 100 chars
        assert result.method == "pdfplumber"
        assert len(ocr_called) == 0

    def test_custom_min_chars_threshold(self, sample_pdf, monkeypatch):
        """extract_pdf with method='auto' respects custom min_chars_threshold."""
        # Track if OCR is called
        ocr_called = []

        def mock_ocr(path):
            ocr_called.append(path)
            return PDFExtractionResult(
                text="OCR text",
                pages=1,
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        # Set threshold very high so pdfplumber text is considered insufficient
        result = extract_pdf(sample_pdf, method="auto", min_chars_threshold=999999)

        # Should have called OCR due to high threshold
        assert len(ocr_called) == 1

    def test_pdfplumber_result_is_used_when_meets_threshold(self, sample_pdf_long_text):
        """extract_pdf uses pdfplumber result when it meets min_chars_threshold."""
        result = extract_pdf(sample_pdf_long_text, method="auto", min_chars_threshold=50)
        assert result.success is True
        assert result.method == "pdfplumber"
        assert len(result.text.strip()) >= 50

    def test_unicode_pdf(self, sample_pdf_unicode):
        """extract_pdf with method='auto' handles Unicode content."""
        result = extract_pdf(sample_pdf_unicode, method="auto")
        assert result.success is True
        assert result.method == "pdfplumber"

    def test_empty_pages_pdf(self, sample_pdf_empty_pages):
        """extract_pdf with method='auto' handles PDF with empty pages."""
        result = extract_pdf(sample_pdf_empty_pages, method="auto")
        assert result.success is True
        assert result.method == "pdfplumber"
        assert result.pages == 3

    def test_returns_string_text(self, sample_pdf):
        """extract_pdf with method='auto' returns text as string."""
        result = extract_pdf(sample_pdf, method="auto")
        assert isinstance(result.text, str)

    def test_returns_int_pages(self, sample_pdf):
        """extract_pdf with method='auto' returns pages as int."""
        result = extract_pdf(sample_pdf, method="auto")
        assert isinstance(result.pages, int)


# ============================================================================
# Tests for extract_pdf fallback to OCR when pdfplumber extracts little text
# ============================================================================


@pytest.fixture
def sample_pdf_minimal_text():
    """
    Creates a PDF with minimal text content (below default threshold).

    This simulates a scanned PDF where pdfplumber can extract very little text.

    Yields:
        str: Path to the temporary PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=letter)
    # Only add minimal text (less than 100 chars which is default threshold)
    c.drawString(100, 750, "Hi")  # Just 2 chars
    c.showPage()
    c.save()

    yield path

    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_pdf_empty_text():
    """
    Creates a PDF with no text content at all.

    This simulates a purely image-based scanned PDF.

    Yields:
        str: Path to the temporary PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=letter)
    # Don't add any text, just draw a shape to make valid PDF
    c.rect(100, 700, 200, 50)  # Just a rectangle, no text
    c.showPage()
    c.save()

    yield path

    if os.path.exists(path):
        os.unlink(path)


class TestExtractPdfFallbackToOcr:
    """Tests for extract_pdf fallback to OCR when pdfplumber extracts insufficient text."""

    def test_calls_ocr_when_pdfplumber_extracts_below_threshold(self, sample_pdf_minimal_text, monkeypatch):
        """extract_pdf calls OCR when pdfplumber text is below min_chars_threshold."""
        ocr_called = []

        def mock_ocr(path):
            ocr_called.append(path)
            return PDFExtractionResult(
                text="OCR extracted text from scanned PDF",
                pages=1,
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        # pdfplumber extracts ~200+ chars (including page marker), set threshold higher
        result = extract_pdf(sample_pdf_minimal_text, method="auto", min_chars_threshold=500)

        # Should have called OCR because pdfplumber text is below threshold
        assert len(ocr_called) == 1
        assert ocr_called[0] == sample_pdf_minimal_text

    def test_returns_ocr_method_when_fallback_succeeds(self, sample_pdf_minimal_text, monkeypatch):
        """extract_pdf returns method='mistral_ocr' when OCR fallback succeeds."""
        def mock_ocr(path):
            return PDFExtractionResult(
                text="OCR extracted text from scanned PDF",
                pages=1,
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        # Use high threshold to force fallback
        result = extract_pdf(sample_pdf_minimal_text, method="auto", min_chars_threshold=500)

        assert result.method == "mistral_ocr"

    def test_returns_ocr_text_when_fallback_succeeds(self, sample_pdf_minimal_text, monkeypatch):
        """extract_pdf returns OCR text when fallback succeeds."""
        ocr_text = "OCR extracted: detailed text from scanned document"

        def mock_ocr(path):
            return PDFExtractionResult(
                text=ocr_text,
                pages=1,
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        # Use high threshold to force fallback
        result = extract_pdf(sample_pdf_minimal_text, method="auto", min_chars_threshold=500)

        assert result.text == ocr_text

    def test_returns_success_true_when_ocr_fallback_succeeds(self, sample_pdf_minimal_text, monkeypatch):
        """extract_pdf returns success=True when OCR fallback succeeds."""
        def mock_ocr(path):
            return PDFExtractionResult(
                text="OCR text",
                pages=1,
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        # Use high threshold to force fallback
        result = extract_pdf(sample_pdf_minimal_text, method="auto", min_chars_threshold=500)

        assert result.success is True

    def test_ocr_page_count_is_used_when_fallback_succeeds(self, sample_pdf_minimal_text, monkeypatch):
        """extract_pdf returns OCR page count when OCR fallback succeeds."""
        def mock_ocr(path):
            return PDFExtractionResult(
                text="OCR text",
                pages=5,  # Different from pdfplumber
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        # Use high threshold to force fallback
        result = extract_pdf(sample_pdf_minimal_text, method="auto", min_chars_threshold=500)

        assert result.pages == 5

    def test_fallback_triggered_for_empty_text_pdf(self, sample_pdf_empty_text, monkeypatch):
        """extract_pdf falls back to OCR when pdfplumber extracts no text."""
        ocr_called = []

        def mock_ocr(path):
            ocr_called.append(path)
            return PDFExtractionResult(
                text="OCR text from empty PDF",
                pages=1,
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        result = extract_pdf(sample_pdf_empty_text, method="auto")

        assert len(ocr_called) == 1
        assert result.method == "mistral_ocr"

    def test_fallback_respects_min_chars_threshold(self, sample_pdf, monkeypatch):
        """extract_pdf uses min_chars_threshold to decide whether to fallback."""
        ocr_called = []

        def mock_ocr(path):
            ocr_called.append(path)
            return PDFExtractionResult(
                text="OCR text",
                pages=1,
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        # sample_pdf has ~100+ chars, set threshold very high
        result = extract_pdf(sample_pdf, method="auto", min_chars_threshold=50000)

        # Should have called OCR because threshold not met
        assert len(ocr_called) == 1
        assert result.method == "mistral_ocr"

    def test_no_fallback_when_threshold_is_zero(self, sample_pdf_minimal_text, monkeypatch):
        """extract_pdf does not fallback when min_chars_threshold=0."""
        ocr_called = []

        def mock_ocr(path):
            ocr_called.append(path)
            return PDFExtractionResult(
                text="OCR text",
                pages=1,
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        # With threshold=0, even minimal text should pass
        result = extract_pdf(sample_pdf_minimal_text, method="auto", min_chars_threshold=0)

        # Should NOT call OCR
        assert len(ocr_called) == 0
        assert result.method == "pdfplumber"

    def test_returns_pdfplumber_if_ocr_fails_but_has_some_text(self, sample_pdf_minimal_text, monkeypatch):
        """extract_pdf returns pdfplumber result if OCR fails but pdfplumber had some text."""
        def mock_ocr(path):
            return PDFExtractionResult(
                text="",
                pages=0,
                method="mistral_ocr",
                success=False,
                error="OCR API error"
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        result = extract_pdf(sample_pdf_minimal_text, method="auto")

        # pdfplumber has "Hi" which is some text
        assert result.method == "pdfplumber"
        assert result.success is True
        assert "Hi" in result.text

    def test_returns_ocr_error_if_both_fail(self, sample_pdf_empty_text, monkeypatch):
        """extract_pdf returns OCR error if both pdfplumber and OCR fail."""
        def mock_ocr(path):
            return PDFExtractionResult(
                text="",
                pages=0,
                method="mistral_ocr",
                success=False,
                error="MISTRAL_API_KEY não configurada"
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        result = extract_pdf(sample_pdf_empty_text, method="auto")

        # Both failed, should return OCR result (more informative)
        assert result.method == "mistral_ocr"
        assert result.success is False
        assert result.error is not None

    def test_fallback_for_text_just_below_threshold(self, monkeypatch):
        """extract_pdf falls back when text is below threshold."""
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)

        try:
            c = canvas.Canvas(path, pagesize=letter)
            # Create text that will result in ~200 chars after extraction (with page marker)
            c.drawString(100, 750, "X" * 50)  # 50 chars
            c.showPage()
            c.save()

            ocr_called = []

            def mock_ocr(pdf_path):
                ocr_called.append(pdf_path)
                return PDFExtractionResult(
                    text="OCR text",
                    pages=1,
                    method="mistral_ocr",
                    success=True
                )

            from rlm_mcp import pdf_parser
            monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

            # Use threshold higher than what pdfplumber extracts (~200 chars)
            result = extract_pdf(path, method="auto", min_chars_threshold=500)

            # Should have called OCR (extracted text < 500 threshold)
            assert len(ocr_called) == 1

        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_no_fallback_for_text_at_threshold(self, monkeypatch):
        """extract_pdf does not fallback when text is exactly at threshold."""
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)

        try:
            c = canvas.Canvas(path, pagesize=letter)
            # Create enough text to meet a lower threshold
            text = "A" * 50
            c.drawString(100, 750, text)
            c.showPage()
            c.save()

            ocr_called = []

            def mock_ocr(pdf_path):
                ocr_called.append(pdf_path)
                return PDFExtractionResult(
                    text="OCR text",
                    pages=1,
                    method="mistral_ocr",
                    success=True
                )

            from rlm_mcp import pdf_parser
            monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

            # Use threshold of 20 chars, we have 50+ (including page marker)
            result = extract_pdf(path, method="auto", min_chars_threshold=20)

            # Should NOT call OCR (extracted text >= 20 chars)
            assert len(ocr_called) == 0
            assert result.method == "pdfplumber"

        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_fallback_path_passed_to_ocr(self, sample_pdf_minimal_text, monkeypatch):
        """extract_pdf passes correct path to OCR function."""
        received_path = []

        def mock_ocr(path):
            received_path.append(path)
            return PDFExtractionResult(
                text="OCR text",
                pages=1,
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        # Use high threshold to force fallback
        extract_pdf(sample_pdf_minimal_text, method="auto", min_chars_threshold=500)

        assert len(received_path) == 1
        assert received_path[0] == sample_pdf_minimal_text

    def test_ocr_error_none_when_ocr_succeeds(self, sample_pdf_minimal_text, monkeypatch):
        """extract_pdf returns error=None when OCR fallback succeeds."""
        def mock_ocr(path):
            return PDFExtractionResult(
                text="OCR text",
                pages=1,
                method="mistral_ocr",
                success=True,
                error=None
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        result = extract_pdf(sample_pdf_minimal_text, method="auto")

        assert result.error is None

    def test_multi_page_pdf_fallback(self, monkeypatch):
        """extract_pdf falls back for multi-page PDF with text below threshold."""
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)

        try:
            c = canvas.Canvas(path, pagesize=letter)
            # Create 3 pages with minimal text each
            for i in range(3):
                c.drawString(100, 750, f"P{i}")  # Just 2 chars per page
                c.showPage()
            c.save()

            ocr_called = []

            def mock_ocr(pdf_path):
                ocr_called.append(pdf_path)
                return PDFExtractionResult(
                    text="OCR text for all 3 pages",
                    pages=3,
                    method="mistral_ocr",
                    success=True
                )

            from rlm_mcp import pdf_parser
            monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

            # pdfplumber with layout=True extracts ~10k chars for 3 pages (lots of whitespace)
            # Use very high threshold to force fallback
            result = extract_pdf(path, method="auto", min_chars_threshold=50000)

            # Should call OCR (extracted text < threshold)
            assert len(ocr_called) == 1
            assert result.method == "mistral_ocr"
            assert result.pages == 3

        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_pdfplumber_success_false_triggers_fallback(self, sample_pdf, monkeypatch):
        """extract_pdf falls back to OCR when pdfplumber returns success=False."""
        # Monkeypatch pdfplumber to return failure
        def mock_pdfplumber(path):
            return PDFExtractionResult(
                text="",
                pages=0,
                method="pdfplumber",
                success=False,
                error="pdfplumber error"
            )

        ocr_called = []

        def mock_ocr(path):
            ocr_called.append(path)
            return PDFExtractionResult(
                text="OCR text",
                pages=2,
                method="mistral_ocr",
                success=True
            )

        from rlm_mcp import pdf_parser
        monkeypatch.setattr(pdf_parser, "extract_with_pdfplumber", mock_pdfplumber)
        monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

        result = extract_pdf(sample_pdf, method="auto")

        # Should fall back to OCR
        assert len(ocr_called) == 1
        assert result.method == "mistral_ocr"

    def test_whitespace_only_text_triggers_fallback(self, monkeypatch):
        """extract_pdf falls back when pdfplumber extracts only whitespace."""
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)

        try:
            c = canvas.Canvas(path, pagesize=letter)
            # PDF with only whitespace (pdfplumber might extract spaces)
            c.drawString(100, 750, "   ")  # Just spaces
            c.showPage()
            c.save()

            ocr_called = []

            def mock_ocr(pdf_path):
                ocr_called.append(pdf_path)
                return PDFExtractionResult(
                    text="OCR text",
                    pages=1,
                    method="mistral_ocr",
                    success=True
                )

            from rlm_mcp import pdf_parser
            monkeypatch.setattr(pdf_parser, "extract_with_mistral_ocr", mock_ocr)

            result = extract_pdf(path, method="auto")

            # Should call OCR (only whitespace, strip() makes it empty)
            assert len(ocr_called) == 1

        finally:
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# Tests for split_pdf_into_chunks
# ============================================================================


@pytest.fixture
def sample_pdf_12_pages():
    """
    Creates a 12-page PDF for testing chunking.

    Yields:
        str: Path to the temporary PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=letter)

    for i in range(1, 13):
        c.drawString(100, 750, f"Page {i} of 12")
        c.showPage()

    c.save()

    yield path

    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_pdf_1_page():
    """
    Creates a 1-page PDF for edge case testing.

    Yields:
        str: Path to the temporary PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=letter)
    c.drawString(100, 750, "Single page")
    c.showPage()
    c.save()

    yield path

    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_pdf_5_pages():
    """
    Creates a 5-page PDF for testing.

    Yields:
        str: Path to the temporary PDF file.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    c = canvas.Canvas(path, pagesize=letter)

    for i in range(1, 6):
        c.drawString(100, 750, f"Page {i}")
        c.showPage()

    c.save()

    yield path

    if os.path.exists(path):
        os.unlink(path)


class TestSplitPdfIntoChunks:
    """Tests for split_pdf_into_chunks function."""

    def test_returns_list(self, sample_pdf):
        """split_pdf_into_chunks returns a list."""
        result = split_pdf_into_chunks(sample_pdf)
        assert isinstance(result, list)

    def test_returns_list_of_tuples(self, sample_pdf):
        """split_pdf_into_chunks returns a list of tuples."""
        result = split_pdf_into_chunks(sample_pdf)
        assert all(isinstance(chunk, tuple) for chunk in result)

    def test_tuples_have_two_elements(self, sample_pdf):
        """Each chunk tuple has exactly two elements (start, end)."""
        result = split_pdf_into_chunks(sample_pdf)
        assert all(len(chunk) == 2 for chunk in result)

    def test_tuples_contain_integers(self, sample_pdf):
        """Each chunk tuple contains integers."""
        result = split_pdf_into_chunks(sample_pdf)
        for start, end in result:
            assert isinstance(start, int)
            assert isinstance(end, int)

    def test_12_page_pdf_with_default_chunk_size(self, sample_pdf_12_pages):
        """12-page PDF with default 10 pages per chunk creates 2 chunks."""
        result = split_pdf_into_chunks(sample_pdf_12_pages)
        assert len(result) == 2
        assert result[0] == (1, 10)
        assert result[1] == (11, 12)

    def test_12_page_pdf_with_5_pages_per_chunk(self, sample_pdf_12_pages):
        """12-page PDF with 5 pages per chunk creates 3 chunks."""
        result = split_pdf_into_chunks(sample_pdf_12_pages, pages_per_chunk=5)
        assert len(result) == 3
        assert result[0] == (1, 5)
        assert result[1] == (6, 10)
        assert result[2] == (11, 12)

    def test_12_page_pdf_with_3_pages_per_chunk(self, sample_pdf_12_pages):
        """12-page PDF with 3 pages per chunk creates 4 chunks."""
        result = split_pdf_into_chunks(sample_pdf_12_pages, pages_per_chunk=3)
        assert len(result) == 4
        assert result[0] == (1, 3)
        assert result[1] == (4, 6)
        assert result[2] == (7, 9)
        assert result[3] == (10, 12)

    def test_12_page_pdf_with_4_pages_per_chunk(self, sample_pdf_12_pages):
        """12-page PDF with 4 pages per chunk creates 3 chunks."""
        result = split_pdf_into_chunks(sample_pdf_12_pages, pages_per_chunk=4)
        assert len(result) == 3
        assert result[0] == (1, 4)
        assert result[1] == (5, 8)
        assert result[2] == (9, 12)

    def test_single_page_pdf(self, sample_pdf_1_page):
        """1-page PDF creates 1 chunk."""
        result = split_pdf_into_chunks(sample_pdf_1_page)
        assert len(result) == 1
        assert result[0] == (1, 1)

    def test_single_page_pdf_with_large_chunk_size(self, sample_pdf_1_page):
        """1-page PDF with large chunk size still creates 1 chunk."""
        result = split_pdf_into_chunks(sample_pdf_1_page, pages_per_chunk=100)
        assert len(result) == 1
        assert result[0] == (1, 1)

    def test_exact_multiple_of_chunk_size(self, sample_pdf_12_pages):
        """12-page PDF with 4 pages per chunk (exact multiple) creates correct chunks."""
        result = split_pdf_into_chunks(sample_pdf_12_pages, pages_per_chunk=4)
        # 12 / 4 = 3 chunks exactly
        assert len(result) == 3
        # Last chunk should end at page 12
        assert result[-1] == (9, 12)

    def test_exact_multiple_with_6_pages_per_chunk(self, sample_pdf_12_pages):
        """12-page PDF with 6 pages per chunk creates 2 equal chunks."""
        result = split_pdf_into_chunks(sample_pdf_12_pages, pages_per_chunk=6)
        assert len(result) == 2
        assert result[0] == (1, 6)
        assert result[1] == (7, 12)

    def test_chunk_size_larger_than_pdf(self, sample_pdf_5_pages):
        """5-page PDF with 10 pages per chunk creates 1 chunk."""
        result = split_pdf_into_chunks(sample_pdf_5_pages, pages_per_chunk=10)
        assert len(result) == 1
        assert result[0] == (1, 5)

    def test_chunk_size_equal_to_pdf_pages(self, sample_pdf_5_pages):
        """5-page PDF with 5 pages per chunk creates 1 chunk."""
        result = split_pdf_into_chunks(sample_pdf_5_pages, pages_per_chunk=5)
        assert len(result) == 1
        assert result[0] == (1, 5)

    def test_chunk_size_1_creates_many_chunks(self, sample_pdf_5_pages):
        """5-page PDF with 1 page per chunk creates 5 chunks."""
        result = split_pdf_into_chunks(sample_pdf_5_pages, pages_per_chunk=1)
        assert len(result) == 5
        assert result == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

    def test_pages_are_1_indexed(self, sample_pdf):
        """Chunks use 1-indexed page numbers (PDF standard)."""
        result = split_pdf_into_chunks(sample_pdf, pages_per_chunk=1)
        # First page is 1, not 0
        assert result[0][0] == 1

    def test_end_is_inclusive(self, sample_pdf_12_pages):
        """End page in chunk is inclusive."""
        result = split_pdf_into_chunks(sample_pdf_12_pages, pages_per_chunk=5)
        # First chunk (1, 5) means pages 1, 2, 3, 4, 5
        assert result[0] == (1, 5)
        # Second chunk (6, 10) means pages 6, 7, 8, 9, 10
        assert result[1] == (6, 10)
        # Third chunk (11, 12) means pages 11, 12
        assert result[2] == (11, 12)

    def test_no_overlap_between_chunks(self, sample_pdf_12_pages):
        """Chunks do not overlap."""
        result = split_pdf_into_chunks(sample_pdf_12_pages, pages_per_chunk=3)
        for i in range(len(result) - 1):
            current_end = result[i][1]
            next_start = result[i + 1][0]
            assert next_start == current_end + 1

    def test_all_pages_covered(self, sample_pdf_12_pages):
        """All pages are covered by chunks."""
        result = split_pdf_into_chunks(sample_pdf_12_pages, pages_per_chunk=3)
        assert result[0][0] == 1  # First page
        assert result[-1][1] == 12  # Last page

    def test_nonexistent_file_returns_empty_list(self):
        """Nonexistent file returns empty list."""
        result = split_pdf_into_chunks("/nonexistent/path/to/file.pdf")
        assert result == []

    def test_invalid_path_returns_empty_list(self):
        """Invalid path returns empty list."""
        result = split_pdf_into_chunks("")
        assert result == []

    def test_directory_path_returns_empty_list(self, tmp_path):
        """Directory path (not a file) returns empty list."""
        result = split_pdf_into_chunks(str(tmp_path))
        assert result == []

    def test_zero_pages_per_chunk_returns_empty_list(self, sample_pdf):
        """pages_per_chunk=0 returns empty list (invalid)."""
        result = split_pdf_into_chunks(sample_pdf, pages_per_chunk=0)
        assert result == []

    def test_negative_pages_per_chunk_returns_empty_list(self, sample_pdf):
        """Negative pages_per_chunk returns empty list (invalid)."""
        result = split_pdf_into_chunks(sample_pdf, pages_per_chunk=-1)
        assert result == []

    def test_does_not_raise_exception_for_invalid_input(self):
        """Function does not raise exceptions for invalid input."""
        try:
            result = split_pdf_into_chunks("/nonexistent/file.pdf")
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"split_pdf_into_chunks raised exception: {e}")

    def test_is_read_only(self, sample_pdf):
        """split_pdf_into_chunks does not modify the PDF file."""
        import hashlib

        with open(sample_pdf, "rb") as f:
            hash_before = hashlib.md5(f.read()).hexdigest()

        split_pdf_into_chunks(sample_pdf)

        with open(sample_pdf, "rb") as f:
            hash_after = hashlib.md5(f.read()).hexdigest()

        assert hash_before == hash_after

    def test_2_page_pdf_with_default(self, sample_pdf):
        """2-page PDF (sample_pdf fixture) with default chunk size creates 1 chunk."""
        result = split_pdf_into_chunks(sample_pdf)
        assert len(result) == 1
        assert result[0] == (1, 2)

    def test_many_pages_pdf_with_large_chunk(self, sample_pdf_many_pages):
        """10-page PDF with 10 pages per chunk creates 1 chunk."""
        result = split_pdf_into_chunks(sample_pdf_many_pages)
        assert len(result) == 1
        assert result[0] == (1, 10)

    def test_many_pages_pdf_with_small_chunk(self, sample_pdf_many_pages):
        """10-page PDF with 3 pages per chunk creates 4 chunks."""
        result = split_pdf_into_chunks(sample_pdf_many_pages, pages_per_chunk=3)
        assert len(result) == 4
        assert result[0] == (1, 3)
        assert result[1] == (4, 6)
        assert result[2] == (7, 9)
        assert result[3] == (10, 10)


# ============================================================================
# Tests for extract_with_mistral_ocr (with mocked Mistral API)
# ============================================================================


class MockOCRPage:
    """Mock object for a page returned by Mistral OCR API."""

    def __init__(self, markdown: str):
        self.markdown = markdown


class MockOCRResponse:
    """Mock object for the response from Mistral OCR API."""

    def __init__(self, pages: list[MockOCRPage]):
        self.pages = pages


class MockOCRClient:
    """Mock object for the Mistral OCR client (client.ocr)."""

    def __init__(self, response: MockOCRResponse = None, error: Exception = None):
        self._response = response
        self._error = error

    def process(self, model: str, document: dict, table_format: str = "markdown"):
        if self._error:
            raise self._error
        return self._response


class MockMistralClient:
    """Mock object for the Mistral client."""

    def __init__(self, ocr_response: MockOCRResponse = None, ocr_error: Exception = None):
        self.ocr = MockOCRClient(response=ocr_response, error=ocr_error)


class TestExtractWithMistralOcr:
    """Tests for extract_with_mistral_ocr function with mocked Mistral API."""

    @pytest.fixture(autouse=True)
    def mock_mistralai_module(self, monkeypatch):
        """Auto-mock the mistralai module for all tests in this class."""
        import sys
        # Create a mock mistralai module
        mock_mistralai = type(sys)("mistralai")
        # Store response to be set per test
        self._mock_response = None
        self._mock_error = None
        self._received_api_key = []
        self._received_params = {}

        def create_mock_mistral_class():
            """Create a Mistral class that captures the test's mock settings."""
            test_instance = self

            class MockMistralClass:
                def __init__(self, api_key):
                    test_instance._received_api_key.append(api_key)
                    self.ocr = MockOCRClient(
                        response=test_instance._mock_response,
                        error=test_instance._mock_error
                    )
                    # Optionally capture params
                    self.ocr._received_params = test_instance._received_params

            return MockMistralClass

        mock_mistralai.Mistral = create_mock_mistral_class()
        monkeypatch.setitem(sys.modules, "mistralai", mock_mistralai)

    def _set_mock_response(self, response):
        """Helper to set mock response for the test."""
        self._mock_response = response

    def _set_mock_error(self, error):
        """Helper to set mock error for the test."""
        self._mock_error = error

    def test_returns_pdf_extraction_result(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr returns PDFExtractionResult object."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Page 1 content")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        assert isinstance(result, PDFExtractionResult)

    def test_returns_success_true_on_successful_extraction(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr returns success=True when OCR succeeds."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Extracted text from OCR")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        assert result.success is True

    def test_returns_method_mistral_ocr(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr returns method='mistral_ocr'."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Text")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        assert result.method == "mistral_ocr"

    def test_extracts_text_from_single_page(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr extracts text from single page."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        expected_text = "This is the OCR extracted text from a scanned document."
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown=expected_text)
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        assert expected_text in result.text

    def test_extracts_text_from_multiple_pages(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr extracts text from multiple pages."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Page 1 content"),
            MockOCRPage(markdown="Page 2 content"),
            MockOCRPage(markdown="Page 3 content")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        assert "Page 1 content" in result.text
        assert "Page 2 content" in result.text
        assert "Page 3 content" in result.text

    def test_returns_correct_page_count(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr returns correct page count."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Page 1"),
            MockOCRPage(markdown="Page 2"),
            MockOCRPage(markdown="Page 3"),
            MockOCRPage(markdown="Page 4"),
            MockOCRPage(markdown="Page 5")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        assert result.pages == 5

    def test_includes_page_markers(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr includes page markers in output."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="First page text"),
            MockOCRPage(markdown="Second page text")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        assert "--- Página 1 ---" in result.text
        assert "--- Página 2 ---" in result.text

    def test_skips_empty_pages(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr skips pages with no text."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Page 1 text"),
            MockOCRPage(markdown=""),  # Empty page
            MockOCRPage(markdown="   "),  # Whitespace only page
            MockOCRPage(markdown="Page 4 text")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        # Pages 2 and 3 should be skipped (no marker in text)
        assert "--- Página 1 ---" in result.text
        assert "--- Página 2 ---" not in result.text
        assert "--- Página 3 ---" not in result.text
        assert "--- Página 4 ---" in result.text

    def test_page_count_includes_empty_pages(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr page count includes empty pages."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Text"),
            MockOCRPage(markdown=""),  # Empty
            MockOCRPage(markdown="More text")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        # page_count is len(ocr_response.pages), including empty ones
        assert result.pages == 3

    def test_returns_error_none_on_success(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr returns error=None on success."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Text")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        assert result.error is None

    def test_returns_error_when_api_key_not_set(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr returns error when MISTRAL_API_KEY is not set."""
        # Ensure API key is not set
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        result = extract_with_mistral_ocr(sample_pdf)

        assert result.success is False
        assert result.error is not None
        assert "MISTRAL_API_KEY" in result.error

    def test_returns_method_mistral_ocr_when_api_key_not_set(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr returns method='mistral_ocr' even when API key missing."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        result = extract_with_mistral_ocr(sample_pdf)
        assert result.method == "mistral_ocr"

    def test_returns_empty_text_when_api_key_not_set(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr returns empty text when API key is not set."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        result = extract_with_mistral_ocr(sample_pdf)
        assert result.text == ""

    def test_returns_zero_pages_when_api_key_not_set(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr returns pages=0 when API key is not set."""
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

        result = extract_with_mistral_ocr(sample_pdf)
        assert result.pages == 0

    def test_handles_api_error(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr handles API errors gracefully."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_error(Exception("API rate limit exceeded"))

        result = extract_with_mistral_ocr(sample_pdf)

        assert result.success is False
        assert result.error is not None
        assert "API rate limit exceeded" in result.error

    def test_handles_connection_error(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr handles connection errors."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_error(ConnectionError("Unable to connect to API"))

        result = extract_with_mistral_ocr(sample_pdf)

        assert result.success is False
        assert result.error is not None

    def test_returns_success_false_on_file_not_found(self, monkeypatch):
        """extract_with_mistral_ocr returns success=False for nonexistent file."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        result = extract_with_mistral_ocr("/nonexistent/path/to/file.pdf")

        assert result.success is False
        assert result.error is not None

    def test_returns_method_mistral_ocr_on_file_not_found(self, monkeypatch):
        """extract_with_mistral_ocr returns method='mistral_ocr' even for nonexistent file."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        result = extract_with_mistral_ocr("/nonexistent/path/to/file.pdf")
        assert result.method == "mistral_ocr"

    def test_passes_correct_model_to_api(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr passes correct model name to API."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        # Use tracking OCR client that stores received params
        class TrackingOCRClient:
            received_params = {}

            def process(self, model: str, document: dict, table_format: str = "markdown"):
                TrackingOCRClient.received_params["model"] = model
                TrackingOCRClient.received_params["document"] = document
                TrackingOCRClient.received_params["table_format"] = table_format
                return MockOCRResponse(pages=[MockOCRPage(markdown="Text")])

        # Override the OCR client after mock is set up
        import sys
        mock_mistralai = sys.modules.get("mistralai")
        if mock_mistralai:
            class TrackingMistralClient:
                def __init__(self, api_key):
                    self.ocr = TrackingOCRClient()
            mock_mistralai.Mistral = TrackingMistralClient

        extract_with_mistral_ocr(sample_pdf)

        assert TrackingOCRClient.received_params["model"] == "mistral-ocr-latest"

    def test_passes_document_as_base64(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr passes document as base64 encoded data."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        class TrackingOCRClient:
            received_params = {}

            def process(self, model: str, document: dict, table_format: str = "markdown"):
                TrackingOCRClient.received_params["document"] = document
                return MockOCRResponse(pages=[MockOCRPage(markdown="Text")])

        import sys
        mock_mistralai = sys.modules.get("mistralai")
        if mock_mistralai:
            class TrackingMistralClient:
                def __init__(self, api_key):
                    self.ocr = TrackingOCRClient()
            mock_mistralai.Mistral = TrackingMistralClient

        extract_with_mistral_ocr(sample_pdf)

        assert TrackingOCRClient.received_params["document"]["type"] == "document_url"
        assert TrackingOCRClient.received_params["document"]["document_url"].startswith("data:application/pdf;base64,")

    def test_passes_correct_table_format(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr passes table_format='markdown' to API."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        class TrackingOCRClient:
            received_params = {}

            def process(self, model: str, document: dict, table_format: str = "markdown"):
                TrackingOCRClient.received_params["table_format"] = table_format
                return MockOCRResponse(pages=[MockOCRPage(markdown="Text")])

        import sys
        mock_mistralai = sys.modules.get("mistralai")
        if mock_mistralai:
            class TrackingMistralClient:
                def __init__(self, api_key):
                    self.ocr = TrackingOCRClient()
            mock_mistralai.Mistral = TrackingMistralClient

        extract_with_mistral_ocr(sample_pdf)

        assert TrackingOCRClient.received_params["table_format"] == "markdown"

    def test_uses_api_key_from_env(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr uses API key from environment variable."""
        test_api_key = "my-secret-api-key-12345"
        monkeypatch.setenv("MISTRAL_API_KEY", test_api_key)
        self._set_mock_response(MockOCRResponse(pages=[MockOCRPage(markdown="Text")]))

        extract_with_mistral_ocr(sample_pdf)

        assert len(self._received_api_key) == 1
        assert self._received_api_key[0] == test_api_key

    def test_handles_page_with_none_markdown(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr handles pages where markdown is None."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        # Create page with None markdown
        class NoneMarkdownPage:
            markdown = None

        self._mock_response = MockOCRResponse(pages=[
            MockOCRPage(markdown="Page 1 text"),
            NoneMarkdownPage(),  # None markdown
            MockOCRPage(markdown="Page 3 text")
        ])

        result = extract_with_mistral_ocr(sample_pdf)

        assert result.success is True
        assert "Page 1 text" in result.text
        assert "Page 3 text" in result.text
        # Page 2 with None markdown should be skipped (no page marker)
        assert "--- Página 2 ---" not in result.text

    def test_handles_unicode_content(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr handles Unicode content from OCR."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        unicode_text = "Olá mundo! Español: Señor. 日本語テスト. 中文测试. العربية"
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown=unicode_text)
        ]))

        result = extract_with_mistral_ocr(sample_pdf)

        assert result.success is True
        assert unicode_text in result.text

    def test_handles_markdown_tables(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr handles markdown tables in OCR output."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        table_markdown = """
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Cell 1   | Cell 2   | Cell 3   |
| Cell 4   | Cell 5   | Cell 6   |
"""
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown=table_markdown)
        ]))

        result = extract_with_mistral_ocr(sample_pdf)

        assert result.success is True
        assert "Header 1" in result.text
        assert "Cell 1" in result.text

    def test_handles_long_text(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr handles long text content."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        long_text = "Lorem ipsum dolor sit amet. " * 1000  # ~28k chars
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown=long_text)
        ]))

        result = extract_with_mistral_ocr(sample_pdf)

        assert result.success is True
        assert len(result.text) > 25000  # ~28k chars after page marker

    def test_returns_string_text(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr returns text as string type."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Text")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        assert isinstance(result.text, str)

    def test_returns_int_pages(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr returns pages as int type."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Text")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        assert isinstance(result.pages, int)

    def test_does_not_raise_exception(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr does not raise exceptions (returns error in result)."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_error(Exception("Unexpected error"))

        try:
            result = extract_with_mistral_ocr(sample_pdf)
            assert result is not None  # Got result without exception
        except Exception as e:
            pytest.fail(f"extract_with_mistral_ocr raised exception: {e}")

    def test_pages_separated_by_double_newline(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr separates pages with double newline."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Page 1 text"),
            MockOCRPage(markdown="Page 2 text")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        assert "\n\n" in result.text

    def test_single_page_no_double_newline(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr with single page doesn't have double newline separator."""
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Single page text")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        # Single page joined with \n\n but only one element, so no \n\n separator
        # The text should be "--- Página 1 ---\nSingle page text"
        assert result.text == "--- Página 1 ---\nSingle page text"

    def test_empty_api_key_is_treated_as_not_set(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr treats empty API key as not set."""
        monkeypatch.setenv("MISTRAL_API_KEY", "")

        result = extract_with_mistral_ocr(sample_pdf)

        assert result.success is False
        assert "MISTRAL_API_KEY" in result.error

    def test_whitespace_only_api_key_is_valid(self, sample_pdf, monkeypatch):
        """extract_with_mistral_ocr treats whitespace-only API key as valid (passes to client)."""
        # This documents current behavior - API key is only checked with os.getenv()
        # which returns empty string for "" but " " would be truthy
        monkeypatch.setenv("MISTRAL_API_KEY", "   ")
        self._set_mock_response(MockOCRResponse(pages=[
            MockOCRPage(markdown="Text")
        ]))

        result = extract_with_mistral_ocr(sample_pdf)
        # Whitespace is truthy, so it passes the API key check
        assert result.success is True
