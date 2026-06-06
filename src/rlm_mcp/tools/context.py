"""
ToolContext — dependências runtime injetadas nos handlers de tools.

O http_server constrói uma instância POR CHAMADA (em call_tool), lendo os
globals do módulo no momento do dispatch. Isso preserva duas propriedades:

1. Late-binding: monkeypatch de atributos do http_server nos testes
   (ex.: thresholds, singletons) continua visível pelos handlers.
2. Zero import circular: handlers importam só este módulo (puro) e os
   módulos de domínio (fmt, indexer, persistence...) — nunca o http_server.

O campo `call_tool` carrega o dispatcher de volta para os handlers que
delegam (routers rlm_collection/rlm_task e os modos batch de
rlm_load_s3/rlm_save_to_s3), mantendo o caminho de métricas/erro único.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class ToolContext:
    """Dependências runtime de um dispatch de tool."""
    repl: Any                      # SafeREPL singleton
    task_manager: Any              # TaskManager singleton
    upload_rate_limiter: Any       # SlidingWindowRateLimiter de uploads
    call_tool: Callable[[str, dict, Optional[str]], dict]  # dispatcher (recursão)
    client_id: Optional[str] = None
    # Config lida dos globals do http_server no momento da chamada:
    show_persistence_errors: bool = True
    async_pdf_threshold_mb: float = 5
    batch_async_threshold_files: int = 5
    batch_async_threshold_mb: float = 50
