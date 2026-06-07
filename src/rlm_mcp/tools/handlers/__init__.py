"""
Registry de handlers de tools — substitui a cadeia if/elif do call_tool
monolítico (http_server, ~1600 linhas) por dispatch via dict.

Cada handler tem assinatura (arguments: dict, ctx: ToolContext) -> dict.
O http_server constrói o ToolContext por chamada e mantém o try/except
compartilhado (RateLimitExceeded re-raise + erro genérico logado).
"""

from typing import Callable

from ..context import ToolContext
from . import (
    collection_tools,
    data_tools,
    misc_tools,
    pdf_tools,
    repertory_tools,
    s3_tools,
    search_tools,
    task_tools,
)

ToolHandler = Callable[[dict, ToolContext], dict]

TOOL_HANDLERS: dict[str, ToolHandler] = {
    # Dados / REPL
    "rlm_execute": data_tools.rlm_execute,
    "rlm_load_data": data_tools.rlm_load_data,
    "rlm_load_file": data_tools.rlm_load_file,
    "rlm_list_vars": data_tools.rlm_list_vars,
    "rlm_var_info": data_tools.rlm_var_info,
    "rlm_clear": data_tools.rlm_clear,
    "rlm_memory": data_tools.rlm_memory,
    "rlm_pin_var": data_tools.rlm_pin_var,
    # S3 / Minio
    "rlm_load_s3": s3_tools.rlm_load_s3,
    "rlm_list_buckets": s3_tools.rlm_list_buckets,
    "rlm_list_s3": s3_tools.rlm_list_s3,
    "rlm_upload_url": s3_tools.rlm_upload_url,
    "rlm_save_to_s3": s3_tools.rlm_save_to_s3,
    "rlm_batch_load_s3": s3_tools.rlm_batch_load_s3,
    "rlm_batch_upload_s3": s3_tools.rlm_batch_upload_s3,
    # PDF
    "rlm_process_pdf": pdf_tools.rlm_process_pdf,
    # Busca
    "rlm_search_index": search_tools.rlm_search_index,
    "rlm_search_code": search_tools.rlm_search_code,
    # Repertorização homeopática (router com actions)
    "rlm_repertorio": repertory_tools.rlm_repertorio,
    # Coleções (router consolidado + handlers internos)
    "rlm_collection": collection_tools.rlm_collection,
    "rlm_collection_create": collection_tools.rlm_collection_create,
    "rlm_collection_add": collection_tools.rlm_collection_add,
    "rlm_collection_list": collection_tools.rlm_collection_list,
    "rlm_collection_info": collection_tools.rlm_collection_info,
    "rlm_collection_rebuild": collection_tools.rlm_collection_rebuild,
    "rlm_collection_delete": collection_tools.rlm_collection_delete,
    "rlm_search_collection": collection_tools.rlm_search_collection,
    # Tasks assíncronas (router consolidado + handlers internos)
    "rlm_task": task_tools.rlm_task,
    "rlm_task_status": task_tools.rlm_task_status,
    "rlm_task_list": task_tools.rlm_task_list,
    "rlm_task_cancel": task_tools.rlm_task_cancel,
    # Diversos
    "rlm_persistence_stats": misc_tools.rlm_persistence_stats,
    "rlm_help": misc_tools.rlm_help,
}
