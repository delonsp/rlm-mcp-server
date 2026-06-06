"""
Handlers S3/Minio: rlm_load_s3, rlm_list_buckets, rlm_list_s3,
rlm_upload_url, rlm_save_to_s3, rlm_batch_load_s3, rlm_batch_upload_s3.

Corpos movidos verbatim do call_tool monolítico (http_server).
RateLimitExceeded sobe até o call_tool, que re-levanta para o endpoint
HTTP transformar em 429 — mesmo contrato do monolito.
"""

import json
import logging

from ... import response_formatter as fmt
from ... import code_parser
from ...pdf_parser import extract_pdf
from ...rate_limiter import RateLimitExceeded
from ...services.s3_guard import require_s3_configured
from ...services.persistence_service import persist_and_index
from ..context import ToolContext

logger = logging.getLogger("rlm-http")


def rlm_load_s3(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    s3, error = require_s3_configured()
    if error:
        return error

    # Batch mode: if 'keys' is provided, delegate to batch handler
    if "keys" in arguments and arguments["keys"]:
        arguments_for_batch = {
            "keys": arguments["keys"],
            "bucket": arguments.get("bucket", "claude-code"),
        }
        return ctx.call_tool("rlm_batch_load_s3", arguments_for_batch, ctx.client_id)

    bucket = arguments.get("bucket", "claude-code")
    key = arguments["key"]
    var_name = arguments["name"]
    data_type = arguments.get("data_type", "text")
    skip_if_exists = arguments.get("skip_if_exists", True)

    # Verificar se variável já existe e skip_if_exists=True
    if skip_if_exists and var_name in repl.variables:
        existing = repl.variables[var_name]
        size_info = f"{len(existing):,} chars" if isinstance(existing, str) else f"{type(existing).__name__}"
        return {
            "content": [
                {"type": "text", "text": f"Variável '{var_name}' já existe ({size_info}). Use skip_if_exists=False para forçar reload."}
            ]
        }

    try:
        info = s3.get_object_info(bucket, key)
        if not info:
            return {
                "content": [
                    {"type": "text", "text": f"Erro: Objeto não encontrado: {bucket}/{key}"}
                ],
                "isError": True
            }

        # PDF handling - download to temp file, then extract
        if data_type in ("pdf", "pdf_ocr"):
            import tempfile
            pdf_bytes = s3.get_object(bucket, key)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            try:
                method = "ocr" if data_type == "pdf_ocr" else "auto"
                pdf_result = extract_pdf(tmp_path, method=method)

                if not pdf_result.success:
                    return {
                        "content": [
                            {"type": "text", "text": f"Erro ao extrair PDF: {pdf_result.error}"}
                        ],
                        "isError": True
                    }

                data = pdf_result.text
                result = repl.load_data(name=var_name, data=data, data_type="text")
                if var_name in repl.variable_metadata:
                    repl.variable_metadata[var_name].source = "s3"

                # Auto-persistência e indexação
                value = repl.variables.get(var_name)
                persist_msg, index_msg, persist_error = persist_and_index(var_name, value, repl)
                if not ctx.show_persistence_errors:
                    persist_error = ""

                pdf_info = {"method": pdf_result.method, "pages": pdf_result.pages, "chars": len(data)}
                text = fmt.format_s3_load_response(
                    bucket, key, var_name, info['size_human'], data_type,
                    result, persist_msg, index_msg, persist_error, pdf_info=pdf_info,
                )
                return {"content": [{"type": "text", "text": text}]}
            finally:
                import os
                os.unlink(tmp_path)

        # Regular file handling
        data = s3.get_object_text(bucket, key)
        actual_type = "text" if data_type == "code" else data_type
        result = repl.load_data(name=var_name, data=data, data_type=actual_type)
        if var_name in repl.variable_metadata:
            repl.variable_metadata[var_name].source = f"s3:{bucket}/{key}"

        # Auto-parse code structure if data_type="code"
        if data_type == "code" and result.success:
            lang = code_parser.detect_language(key, data)
            if lang and code_parser.is_available():
                structure = code_parser.parse(data, lang)
                if structure:
                    repl.variables[f"_code_structure_{var_name}"] = structure

        # Auto-persistência e indexação
        value = repl.variables.get(var_name)
        persist_msg, index_msg, persist_error = persist_and_index(var_name, value, repl)
        if not ctx.show_persistence_errors:
            persist_error = ""

        text = fmt.format_s3_load_response(
            bucket, key, var_name, info['size_human'], data_type,
            result, persist_msg, index_msg, persist_error,
        )
        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro ao carregar do Minio: {e}"}
            ],
            "isError": True
        }


def rlm_list_buckets(arguments: dict, ctx: ToolContext) -> dict:
    s3, error = require_s3_configured()
    if error:
        return error

    try:
        buckets = s3.list_buckets()
        text = fmt.format_list_buckets(buckets)
        return {"content": [{"type": "text", "text": text}]}
    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro ao listar buckets: {e}"}
            ],
            "isError": True
        }


def rlm_list_s3(arguments: dict, ctx: ToolContext) -> dict:
    s3, error = require_s3_configured()
    if error:
        return error

    bucket = arguments.get("bucket", "claude-code")
    prefix = arguments.get("prefix", "")
    limit = arguments.get("limit", 50)
    offset = arguments.get("offset", 0)

    try:
        objects = s3.list_objects(bucket, prefix)
        total = len(objects)
        text = fmt.format_list_s3(objects, bucket, prefix, total, offset, limit)
        return {"content": [{"type": "text", "text": text}]}
    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro ao listar objetos: {e}"}
            ],
            "isError": True
        }


def rlm_upload_url(arguments: dict, ctx: ToolContext) -> dict:
    # Rate limit check for uploads
    rate_id = ctx.client_id or "anonymous"
    rate_result = ctx.upload_rate_limiter.check(rate_id)
    if not rate_result.allowed:
        logger.warning(f"Upload rate limit exceeded for {rate_id}: {rate_result.current_count}/{rate_result.limit}")
        raise RateLimitExceeded(
            result=rate_result,
            message=f"Upload rate limit exceeded: {rate_result.limit} uploads per {rate_result.window_seconds} seconds"
        )

    s3, error = require_s3_configured()
    if error:
        return error

    url = arguments["url"]
    bucket = arguments.get("bucket", "claude-code")
    key = arguments["key"]

    try:
        result = s3.upload_from_url(url, bucket, key)
        # Record successful upload for rate limiting
        ctx.upload_rate_limiter.record(rate_id)
        text = fmt.format_upload_url(url, result)
        return {"content": [{"type": "text", "text": text}]}
    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro ao fazer upload de URL: {e}"}
            ],
            "isError": True
        }


def rlm_save_to_s3(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    # Batch mode: if 'vars' is provided, delegate to batch handler
    if "vars" in arguments and arguments["vars"]:
        arguments_for_batch = {
            "vars": arguments["vars"],
            "bucket": arguments.get("bucket", "claude-code"),
        }
        return ctx.call_tool("rlm_batch_upload_s3", arguments_for_batch, ctx.client_id)

    # Rate limit check for uploads
    rate_id = ctx.client_id or "anonymous"
    rate_result = ctx.upload_rate_limiter.check(rate_id)
    if not rate_result.allowed:
        logger.warning(f"Upload rate limit exceeded for {rate_id}: {rate_result.current_count}/{rate_result.limit}")
        raise RateLimitExceeded(
            result=rate_result,
            message=f"Upload rate limit exceeded: {rate_result.limit} uploads per {rate_result.window_seconds} seconds"
        )

    s3, error = require_s3_configured()
    if error:
        return error

    var_name = arguments["var_name"]
    bucket = arguments.get("bucket", "claude-code")
    key = arguments["key"]
    save_fmt = arguments.get("format", "auto")

    # Verificar se variável existe
    if var_name not in repl.variables:
        return {
            "content": [
                {"type": "text", "text": f"Erro: Variável '{var_name}' não encontrada no REPL.\n\nUse rlm_list_vars() para ver variáveis disponíveis."}
            ],
            "isError": True
        }

    value = repl.variables[var_name]

    try:
        # Determinar formato de serialização
        if save_fmt == "auto":
            if isinstance(value, str):
                save_fmt = "text"
            elif isinstance(value, (dict, list)):
                save_fmt = "json"
            else:
                save_fmt = "text"

        # Serializar
        if save_fmt == "json":
            content = json.dumps(value, ensure_ascii=False, indent=2)
            content_type = "application/json"
        else:
            content = str(value)
            content_type = "text/plain"

        # Upload
        result = s3.put_object(
            bucket,
            key,
            content.encode("utf-8"),
            content_type=f"{content_type}; charset=utf-8"
        )

        # Record successful upload for rate limiting
        ctx.upload_rate_limiter.record(rate_id)

        text = fmt.format_save_to_s3(var_name, type(value).__name__, save_fmt, result, key)
        return {"content": [{"type": "text", "text": text}]}

    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": f"Erro ao salvar variável no S3: {e}"}
            ],
            "isError": True
        }


def rlm_batch_load_s3(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    task_manager = ctx.task_manager
    s3, error = require_s3_configured()
    if error:
        return error

    bucket = arguments.get("bucket", "claude-code")
    keys_list = arguments["keys"]

    if not keys_list:
        return {"content": [{"type": "text", "text": "Erro: lista 'keys' vazia."}], "isError": True}

    # Check total size to decide sync vs async
    total_size = 0
    for item in keys_list:
        info = s3.get_object_info(bucket, item["key"])
        if info:
            total_size += info.get("size", 0)
    total_size_mb = total_size / (1024 * 1024)

    def _batch_load_worker(progress_callback=None):
        """Worker for batch loading from S3."""
        s3_keys = [item["key"] for item in keys_list]
        download_results = s3.batch_get_objects(
            bucket, s3_keys, progress_callback=progress_callback,
        )

        # Map downloaded data to items
        downloads_by_key = {r["key"]: r for r in download_results}
        load_results = []

        for item in keys_list:
            key = item["key"]
            var_name = item["name"]
            data_type = item.get("data_type", "text")
            dl = downloads_by_key.get(key)

            if not dl or dl["error"]:
                load_results.append({
                    "name": var_name, "key": key, "size_human": "0 B",
                    "data_type": data_type, "success": False,
                    "error": dl["error"] if dl else "not found",
                })
                continue

            try:
                raw = dl["data"]
                try:
                    text_data = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text_data = raw.decode("latin-1")

                result = repl.load_data(name=var_name, data=text_data, data_type=data_type)
                if var_name in repl.variable_metadata:
                    repl.variable_metadata[var_name].source = "s3"

                value = repl.variables.get(var_name)
                persist_and_index(var_name, value, repl)

                load_results.append({
                    "name": var_name, "key": key,
                    "size_human": dl["size_human"],
                    "data_type": data_type,
                    "success": result.success,
                    "error": result.stderr if not result.success else None,
                })
            except Exception as e:
                load_results.append({
                    "name": var_name, "key": key,
                    "size_human": dl["size_human"],
                    "data_type": data_type, "success": False,
                    "error": str(e),
                })

        text = fmt.format_batch_load_s3(load_results)
        return {"content": [{"type": "text", "text": text}]}

    # Large batch → async task
    if len(keys_list) > ctx.batch_async_threshold_files or total_size_mb > ctx.batch_async_threshold_mb:
        task_info = task_manager.submit(
            tool_name="rlm_batch_load_s3",
            description=f"{len(keys_list)} files from {bucket} ({total_size_mb:.1f}MB)",
            func=_batch_load_worker,
        )
        text = fmt.format_task_submitted(
            task_info.task_id, "rlm_batch_load_s3",
            f"{len(keys_list)} files from {bucket}",
        )
        return {"content": [{"type": "text", "text": text}]}

    # Small batch → sync
    try:
        return _batch_load_worker()
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro no batch load: {e}"}],
            "isError": True,
        }


def rlm_batch_upload_s3(arguments: dict, ctx: ToolContext) -> dict:
    repl = ctx.repl
    task_manager = ctx.task_manager
    # Rate limit check for uploads
    rate_id = ctx.client_id or "anonymous"
    rate_result = ctx.upload_rate_limiter.check(rate_id)
    if not rate_result.allowed:
        raise RateLimitExceeded(
            result=rate_result,
            message=f"Upload rate limit exceeded: {rate_result.limit} uploads per {rate_result.window_seconds} seconds"
        )

    s3, error = require_s3_configured()
    if error:
        return error

    bucket = arguments.get("bucket", "claude-code")
    vars_list = arguments["vars"]

    if not vars_list:
        return {"content": [{"type": "text", "text": "Erro: lista 'vars' vazia."}], "isError": True}

    # Validate all vars exist first
    missing = [item["var_name"] for item in vars_list if item["var_name"] not in repl.variables]
    if missing:
        return {
            "content": [{"type": "text", "text": f"Erro: Variáveis não encontradas: {', '.join(missing)}"}],
            "isError": True,
        }

    # Calculate total size
    total_size = 0
    for item in vars_list:
        value = repl.variables[item["var_name"]]
        total_size += len(str(value).encode("utf-8"))
    total_size_mb = total_size / (1024 * 1024)

    def _batch_upload_worker(progress_callback=None):
        """Worker for batch uploading to S3."""
        # Prepare upload items
        upload_items = []
        upload_meta = []
        for item in vars_list:
            var_name = item["var_name"]
            key = item["key"]
            save_fmt = item.get("format", "auto")
            value = repl.variables[var_name]

            # Determine format
            if save_fmt == "auto":
                if isinstance(value, str):
                    save_fmt = "text"
                elif isinstance(value, (dict, list)):
                    save_fmt = "json"
                else:
                    save_fmt = "text"

            # Serialize
            if save_fmt == "json":
                content = json.dumps(value, ensure_ascii=False, indent=2)
                ct = "application/json; charset=utf-8"
            else:
                content = str(value)
                ct = "text/plain; charset=utf-8"

            data = content.encode("utf-8")
            upload_items.append({"key": key, "data": data, "content_type": ct})
            upload_meta.append({"var_name": var_name, "key": key, "format": save_fmt})

        upload_results = s3.batch_put_objects(
            bucket, upload_items, progress_callback=progress_callback,
        )

        # Merge results with metadata
        fmt_results = []
        for i, up_result in enumerate(upload_results):
            meta = upload_meta[i]
            fmt_results.append({
                "var_name": meta["var_name"],
                "key": meta["key"],
                "format": meta["format"],
                "size_human": up_result["size_human"],
                "success": up_result["error"] is None,
                "error": up_result.get("error"),
            })

        # Record for rate limiting
        ctx.upload_rate_limiter.record(rate_id)

        text = fmt.format_batch_upload_s3(fmt_results)
        return {"content": [{"type": "text", "text": text}]}

    # Large batch → async task
    if len(vars_list) > ctx.batch_async_threshold_files or total_size_mb > ctx.batch_async_threshold_mb:
        task_info = task_manager.submit(
            tool_name="rlm_batch_upload_s3",
            description=f"{len(vars_list)} vars to {bucket} ({total_size_mb:.1f}MB)",
            func=_batch_upload_worker,
        )
        text = fmt.format_task_submitted(
            task_info.task_id, "rlm_batch_upload_s3",
            f"{len(vars_list)} vars to {bucket}",
        )
        return {"content": [{"type": "text", "text": text}]}

    # Small batch → sync
    try:
        return _batch_upload_worker()
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Erro no batch upload: {e}"}],
            "isError": True,
        }
