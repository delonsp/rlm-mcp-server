"""
Handlers de tasks assíncronas: rlm_task (router consolidado),
rlm_task_status, rlm_task_list, rlm_task_cancel.

Corpos movidos verbatim do call_tool monolítico (http_server).
"""

import logging

from ... import response_formatter as fmt
from ..context import ToolContext

logger = logging.getLogger("rlm-http")


def rlm_task(arguments: dict, ctx: ToolContext) -> dict:
    action = arguments.get("action", "list")
    if action == "list":
        return ctx.call_tool("rlm_task_list", {"status": arguments.get("status")}, ctx.client_id)
    elif action == "status":
        return ctx.call_tool("rlm_task_status", {"task_id": arguments.get("task_id", "")}, ctx.client_id)
    elif action == "cancel":
        return ctx.call_tool("rlm_task_cancel", {"task_id": arguments.get("task_id", "")}, ctx.client_id)
    else:
        return {"content": [{"type": "text", "text": f"Ação desconhecida: {action}"}], "isError": True}


def rlm_task_status(arguments: dict, ctx: ToolContext) -> dict:
    task_id = arguments["task_id"]
    task_info = ctx.task_manager.get_status(task_id)
    if not task_info:
        return {
            "content": [{"type": "text", "text": f"Task '{task_id}' não encontrada."}],
            "isError": True,
        }

    # If task completed, return the original result directly
    if task_info.status == "completed" and task_info.result:
        result_content = task_info.result.get("content", [])
        # Preserva o flag isError: uma task que RETORNA erro (ex: PDF
        # cuja extração falhou) é marcada "completed" pelo TaskManager,
        # mas não pode ser reportada como sucesso ao cliente.
        is_error = task_info.result.get("isError", False)
        meta = fmt.format_task_status(task_info)
        # Prepend task meta to the original result
        if result_content:
            original_text = result_content[0].get("text", "")
            return {"content": [{"type": "text", "text": original_text}], "isError": is_error}
        return {"content": [{"type": "text", "text": meta}], "isError": is_error}

    text = fmt.format_task_status(task_info)
    return {"content": [{"type": "text", "text": text}]}


def rlm_task_list(arguments: dict, ctx: ToolContext) -> dict:
    status_filter = arguments.get("status")
    tasks = ctx.task_manager.list_tasks(status=status_filter)
    # Cleanup old tasks while we're at it
    ctx.task_manager.cleanup_completed()
    text = fmt.format_task_list(tasks)
    return {"content": [{"type": "text", "text": text}]}


def rlm_task_cancel(arguments: dict, ctx: ToolContext) -> dict:
    task_id = arguments["task_id"]
    success = ctx.task_manager.cancel(task_id)
    text = fmt.format_task_cancel(task_id, success)
    if not success:
        return {"content": [{"type": "text", "text": text}], "isError": True}
    return {"content": [{"type": "text", "text": text}]}
