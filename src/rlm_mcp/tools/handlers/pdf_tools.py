"""
Handler de PDF: rlm_process_pdf (extração S3→S3, sync ou task assíncrona).

Corpo movido verbatim do call_tool monolítico (http_server).
"""

import logging

from ... import response_formatter as fmt
from ...pdf_parser import extract_pdf
from ...services.s3_guard import require_s3_configured
from ..context import ToolContext

logger = logging.getLogger("rlm-http")


def rlm_process_pdf(arguments: dict, ctx: ToolContext) -> dict:
    s3, error = require_s3_configured()
    if error:
        return error

    bucket = arguments.get("bucket", "claude-code")
    key = arguments["key"]
    method = arguments.get("method", "auto")

    # Determinar output_key (padrão: mesmo path com .txt)
    output_key = arguments.get("output_key")
    if not output_key:
        if key.lower().endswith(".pdf"):
            output_key = key[:-4] + ".txt"
        else:
            output_key = key + ".txt"

    try:
        # Verificar se PDF existe
        info = s3.get_object_info(bucket, key)
        if not info:
            return {
                "content": [
                    {"type": "text", "text": f"Erro: PDF não encontrado: {bucket}/{key}"}
                ],
                "isError": True
            }

        size_mb = info.get("size", 0) / (1024 * 1024)
        logger.info(f"Processando PDF: {bucket}/{key} ({info['size_human']}, {size_mb:.1f}MB)")

        # For large PDFs, run as async task
        if size_mb > ctx.async_pdf_threshold_mb:
            def _process_pdf_async(progress_callback=None):
                """Worker function for async PDF processing."""
                import tempfile
                if progress_callback:
                    progress_callback(0.05, "downloading PDF from S3")
                pdf_bytes = s3.get_object(bucket, key)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name
                try:
                    if progress_callback:
                        progress_callback(0.15, "extracting text")
                    pdf_result = extract_pdf(
                        tmp_path, method=method,
                        progress_callback=progress_callback,
                    )
                    if not pdf_result.success:
                        return {
                            "content": [{"type": "text", "text": f"Erro ao extrair PDF: {pdf_result.error}"}],
                            "isError": True,
                        }
                    if progress_callback:
                        progress_callback(0.9, "uploading text to S3")
                    upload_result = s3.put_object_text(bucket, output_key, pdf_result.text)
                    text = fmt.format_process_pdf(bucket, key, output_key, info, pdf_result, upload_result)
                    return {"content": [{"type": "text", "text": text}]}
                finally:
                    import os as _os
                    _os.unlink(tmp_path)

            task_info = ctx.task_manager.submit(
                tool_name="rlm_process_pdf",
                description=f"{bucket}/{key} ({info['size_human']})",
                func=_process_pdf_async,
            )
            text = fmt.format_task_submitted(task_info.task_id, "rlm_process_pdf", f"{bucket}/{key}")
            return {"content": [{"type": "text", "text": text}]}

        # Small PDFs: process synchronously (original behavior)
        import tempfile
        pdf_bytes = s3.get_object(bucket, key)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            pdf_result = extract_pdf(tmp_path, method=method)

            if not pdf_result.success:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao extrair PDF: {pdf_result.error}"}
                    ],
                    "isError": True
                }

            upload_result = s3.put_object_text(bucket, output_key, pdf_result.text)
            text = fmt.format_process_pdf(bucket, key, output_key, info, pdf_result, upload_result)
            return {"content": [{"type": "text", "text": text}]}

        finally:
            import os
            os.unlink(tmp_path)

    except Exception as e:
        logger.exception(f"Erro ao processar PDF {bucket}/{key}")
        return {
            "content": [
                {"type": "text", "text": f"Erro ao processar PDF: {e}"}
            ],
            "isError": True
        }
