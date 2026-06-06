"""
Canários do registry de handlers (refactor do call_tool monolítico, 2026-06-06).

Garante que o dispatch via TOOL_HANDLERS cobre exatamente o contrato do
monolito original: toda tool exposta no schema tem handler, os nomes
internos de dispatch (routers) existem, e tool desconhecida vira isError.
"""

from rlm_mcp import http_server as hs
from rlm_mcp.tools.handlers import TOOL_HANDLERS
from rlm_mcp.tools.schemas import TOOL_SCHEMAS


# Nomes internos que não aparecem no schema público mas são alvos de
# dispatch recursivo (routers rlm_collection/rlm_task e modos batch).
INTERNAL_DISPATCH_NAMES = {
    "rlm_collection_create", "rlm_collection_add", "rlm_collection_list",
    "rlm_collection_info", "rlm_collection_rebuild", "rlm_collection_delete",
    "rlm_search_collection",
    "rlm_task_status", "rlm_task_list", "rlm_task_cancel",
    "rlm_batch_load_s3", "rlm_batch_upload_s3",
    "rlm_persistence_stats",
}


def test_toda_tool_do_schema_tem_handler():
    schema_names = {t["name"] for t in TOOL_SCHEMAS}
    missing = schema_names - set(TOOL_HANDLERS)
    assert not missing, f"Tools no schema sem handler: {missing}"


def test_nomes_internos_de_dispatch_existem():
    missing = INTERNAL_DISPATCH_NAMES - set(TOOL_HANDLERS)
    assert not missing, f"Alvos de dispatch interno sem handler: {missing}"


def test_registry_nao_tem_handler_orfao():
    """Todo handler é alcançável: ou está no schema público ou é alvo interno."""
    schema_names = {t["name"] for t in TOOL_SCHEMAS}
    reachable = schema_names | INTERNAL_DISPATCH_NAMES
    orphans = set(TOOL_HANDLERS) - reachable
    assert not orphans, f"Handlers inalcançáveis (nem schema, nem interno): {orphans}"


def test_tool_desconhecida_retorna_iserror():
    res = hs.call_tool("rlm_nao_existe", {})
    assert res["isError"] is True
    assert "Tool desconhecida" in res["content"][0]["text"]


def test_dispatch_recursivo_do_router_registra_metricas():
    """Routers delegam via ctx.call_tool → métricas registram AMBOS os nomes
    (contrato do monolito original, onde a recursão passava pelo call_tool)."""
    hs.metrics_collector.reset()
    hs.call_tool("rlm_task", {"action": "list"})
    snap = hs.metrics_collector.get_snapshot()
    assert snap.tool_calls_by_name.get("rlm_task") == 1
    assert snap.tool_calls_by_name.get("rlm_task_list") == 1
