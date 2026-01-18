"""
PDF Parser para RLM MCP Server

Suporta dois modos:
1. Machine readable PDFs: usa pdfplumber para extração direta
2. PDFs escaneados (imagens): usa Mistral OCR API

Estratégia:
- Tenta primeiro com pdfplumber
- Se não extrair texto suficiente, usa Mistral OCR como fallback
"""

import os
import base64
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("rlm-mcp.pdf")


@dataclass
class PDFExtractionResult:
    """Resultado da extração de PDF"""
    text: str
    pages: int
    method: str  # "pdfplumber" ou "mistral_ocr"
    success: bool
    error: Optional[str] = None


def extract_with_pdfplumber(pdf_path: str) -> PDFExtractionResult:
    """
    Extrai texto de PDF usando pdfplumber.
    Funciona bem para PDFs machine readable (texto selecionável).
    """
    try:
        import pdfplumber
    except ImportError:
        return PDFExtractionResult(
            text="",
            pages=0,
            method="pdfplumber",
            success=False,
            error="pdfplumber não instalado. Execute: pip install pdfplumber"
        )

    try:
        text_parts = []
        page_count = 0

        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text(layout=True) or ""
                if page_text.strip():
                    text_parts.append(f"--- Página {i + 1} ---\n{page_text}")

        full_text = "\n\n".join(text_parts)

        return PDFExtractionResult(
            text=full_text,
            pages=page_count,
            method="pdfplumber",
            success=True
        )

    except Exception as e:
        logger.exception(f"Erro ao extrair PDF com pdfplumber: {e}")
        return PDFExtractionResult(
            text="",
            pages=0,
            method="pdfplumber",
            success=False,
            error=str(e)
        )


def extract_with_mistral_ocr(pdf_path: str) -> PDFExtractionResult:
    """
    Extrai texto de PDF usando Mistral OCR API.
    Funciona para PDFs escaneados (imagens) e machine readable.
    Requer MISTRAL_API_KEY configurada.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return PDFExtractionResult(
            text="",
            pages=0,
            method="mistral_ocr",
            success=False,
            error="MISTRAL_API_KEY não configurada"
        )

    try:
        from mistralai import Mistral
    except ImportError:
        return PDFExtractionResult(
            text="",
            pages=0,
            method="mistral_ocr",
            success=False,
            error="mistralai não instalado. Execute: pip install mistralai"
        )

    try:
        client = Mistral(api_key=api_key)

        # Ler PDF e converter para base64
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

        # Chamar OCR API
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            },
            table_format="markdown"
        )

        # Extrair texto das páginas
        text_parts = []
        for i, page in enumerate(ocr_response.pages):
            page_text = page.markdown or ""
            if page_text.strip():
                text_parts.append(f"--- Página {i + 1} ---\n{page_text}")

        full_text = "\n\n".join(text_parts)
        page_count = len(ocr_response.pages)

        return PDFExtractionResult(
            text=full_text,
            pages=page_count,
            method="mistral_ocr",
            success=True
        )

    except Exception as e:
        logger.exception(f"Erro ao extrair PDF com Mistral OCR: {e}")
        return PDFExtractionResult(
            text="",
            pages=0,
            method="mistral_ocr",
            success=False,
            error=str(e)
        )


def extract_pdf(
    pdf_path: str,
    method: str = "auto",
    min_chars_threshold: int = 100
) -> PDFExtractionResult:
    """
    Extrai texto de PDF com estratégia configurável.

    Args:
        pdf_path: Caminho para o arquivo PDF
        method: Método de extração:
            - "auto": Tenta pdfplumber primeiro, fallback para OCR se texto insuficiente
            - "pdfplumber": Usa apenas pdfplumber
            - "ocr": Usa apenas Mistral OCR
        min_chars_threshold: Mínimo de caracteres para considerar extração bem-sucedida
                           (usado no modo "auto" para decidir se faz fallback)

    Returns:
        PDFExtractionResult com texto extraído e metadados
    """
    if not os.path.exists(pdf_path):
        return PDFExtractionResult(
            text="",
            pages=0,
            method="none",
            success=False,
            error=f"Arquivo não encontrado: {pdf_path}"
        )

    if method == "pdfplumber":
        return extract_with_pdfplumber(pdf_path)

    elif method == "ocr":
        return extract_with_mistral_ocr(pdf_path)

    elif method == "auto":
        # Tenta pdfplumber primeiro
        result = extract_with_pdfplumber(pdf_path)

        if result.success and len(result.text.strip()) >= min_chars_threshold:
            return result

        # Fallback para OCR se texto insuficiente
        logger.info(
            f"pdfplumber extraiu apenas {len(result.text)} chars, "
            f"tentando Mistral OCR..."
        )

        ocr_result = extract_with_mistral_ocr(pdf_path)

        # Retorna OCR se funcionou, senão retorna o que tiver
        if ocr_result.success:
            return ocr_result

        # Se OCR falhou mas pdfplumber teve algo, retorna pdfplumber
        if result.text.strip():
            return result

        # Se ambos falharam, retorna erro do OCR (mais informativo)
        return ocr_result

    else:
        return PDFExtractionResult(
            text="",
            pages=0,
            method="none",
            success=False,
            error=f"Método inválido: {method}. Use 'auto', 'pdfplumber' ou 'ocr'"
        )
