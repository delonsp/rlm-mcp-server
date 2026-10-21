"""Conservative checks for binary data and evident extraction corruption.

These checks do not establish semantic or clinical quality. Inspect content,
not filename alone (also used by Paperless-ngx's ingestion pipeline).
"""
import re

_BAD_CHARS = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f\ufffd\ud800-\udfff]")
_CID = re.compile(r"\(cid:\d+\)")
_BINARY_PREFIXES = ("%PDF-", "PK\x03\x04", "\x89PNG", "\xff\xd8\xff", "\x1f\x8b")


def text_quality_error(text: str, *, allow_sparse_nul: bool = False) -> str | None:
    """Return an actionable error for obviously invalid text, otherwise None."""
    if text[:1024].lstrip("\ufeff \r\n\t").startswith(_BINARY_PREFIXES):
        return "Conteúdo PDF/binário recebido como texto. Use data_type='pdf'/'pdf_ocr' para PDF ou extraia o texto antes de carregar."
    if not allow_sparse_nul and "\x00" in text:
        return "Texto contém bytes NUL; verifique o formato e a extração antes de carregar."
    bad = sum(1 for _ in _BAD_CHARS.finditer(text))
    if bad >= 3 and bad / max(len(text), 1) >= 0.02:
        return "Texto com caracteres corrompidos em excesso; refaça a extração/OCR."
    if "(cid:" in text:
        glyph_chars = sum(len(m[0]) for m in _CID.finditer(text))
        if glyph_chars >= 24 and glyph_chars / max(len(text), 1) >= 0.1:
            return "Extração contém códigos de glifos (cid) sem texto legível; refaça com OCR."
    return None
