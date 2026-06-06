"""
Handlers diversos: rlm_persistence_stats (nome legado interno) e rlm_help.

Corpos movidos verbatim do call_tool monolítico (http_server).
"""

import logging

from ... import response_formatter as fmt
from ...persistence import get_persistence
from ..context import ToolContext
from ..help_text import get_help_text

logger = logging.getLogger("rlm-http")


def rlm_persistence_stats(arguments: dict, ctx: ToolContext) -> dict:
    try:
        persistence = get_persistence()
        stats = persistence.get_stats()
        saved_vars = persistence.list_variables()
        text = fmt.format_persistence_stats(stats, saved_vars)
        return {"content": [{"type": "text", "text": text}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Erro ao obter estatísticas: {e}"}], "isError": True}


def rlm_help(arguments: dict, ctx: ToolContext) -> dict:
    topic = arguments.get("topic", "all")
    text = get_help_text(topic)
    return {"content": [{"type": "text", "text": text}]}
