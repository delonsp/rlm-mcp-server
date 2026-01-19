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
    extract_pdf,
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
