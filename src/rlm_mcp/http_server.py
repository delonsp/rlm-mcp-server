"""
RLM MCP Server - HTTP/SSE Transport

Expõe o MCP server via HTTP com Server-Sent Events (SSE).
Permite conexão direta do Claude Code via URL, sem SSH tunnel.

Endpoints:
- GET  /health     → Health check
- GET  /sse        → SSE stream para MCP
- POST /message    → Envia mensagem para MCP
"""

import os
import json
import asyncio
import logging
from typing import Any
from contextlib import asynccontextmanager
from datetime import datetime
import uuid
import hmac

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import uvicorn

from .repl import SafeREPL, INTERNAL_FUNCTION_NAMES, VariableInfo
from .persistence import get_persistence
from .indexer import set_index, TextIndex, create_index
from .rate_limiter import SlidingWindowRateLimiter, RateLimitExceeded
from .collection_builder import build_collection_combined
from .tools.schemas import TOOL_SCHEMAS
from .tools.context import ToolContext
from .tools.handlers import TOOL_HANDLERS
from .task_manager import TaskManager
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock, RLock


@dataclass
class MetricsSnapshot:
    """Snapshot of collected metrics."""
    total_requests: int = 0
    total_errors: int = 0
    requests_by_endpoint: dict = field(default_factory=dict)
    errors_by_endpoint: dict = field(default_factory=dict)
    latency_avg_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_max_ms: float = 0.0
    uptime_seconds: float = 0.0
    tool_calls_by_name: dict = field(default_factory=dict)
    rate_limit_rejections: int = 0


class MetricsCollector:
    """Collects and aggregates server metrics.

    Thread-safe metrics collection for request counts, errors, and latency.
    Maintains a rolling window of latency samples for percentile calculation.
    """

    MAX_LATENCY_SAMPLES = 10000  # Keep last N latency measurements

    def __init__(self):
        self._lock = Lock()
        self._start_time = time.time()
        self._total_requests = 0
        self._total_errors = 0
        self._requests_by_endpoint: dict[str, int] = defaultdict(int)
        self._errors_by_endpoint: dict[str, int] = defaultdict(int)
        self._latency_samples: list[float] = []
        self._tool_calls_by_name: dict[str, int] = defaultdict(int)
        self._rate_limit_rejections = 0

    def record_request(self, endpoint: str, latency_ms: float, is_error: bool = False):
        """Record a completed request.

        Args:
            endpoint: The endpoint path (e.g., "/message", "/mcp")
            latency_ms: Request latency in milliseconds
            is_error: Whether the request resulted in an error
        """
        with self._lock:
            self._total_requests += 1
            self._requests_by_endpoint[endpoint] += 1

            if is_error:
                self._total_errors += 1
                self._errors_by_endpoint[endpoint] += 1

            # Maintain rolling window of latency samples
            self._latency_samples.append(latency_ms)
            if len(self._latency_samples) > self.MAX_LATENCY_SAMPLES:
                self._latency_samples = self._latency_samples[-self.MAX_LATENCY_SAMPLES:]

    def record_tool_call(self, tool_name: str):
        """Record a tool call."""
        with self._lock:
            self._tool_calls_by_name[tool_name] += 1

    def record_rate_limit_rejection(self):
        """Record a rate limit rejection."""
        with self._lock:
            self._rate_limit_rejections += 1

    def get_snapshot(self) -> MetricsSnapshot:
        """Get a snapshot of current metrics."""
        with self._lock:
            # Calculate latency percentiles
            latency_avg = 0.0
            latency_p50 = 0.0
            latency_p95 = 0.0
            latency_p99 = 0.0
            latency_max = 0.0

            if self._latency_samples:
                sorted_latencies = sorted(self._latency_samples)
                n = len(sorted_latencies)
                latency_avg = sum(sorted_latencies) / n
                latency_p50 = sorted_latencies[int(n * 0.5)]
                latency_p95 = sorted_latencies[min(int(n * 0.95), n - 1)]
                latency_p99 = sorted_latencies[min(int(n * 0.99), n - 1)]
                latency_max = sorted_latencies[-1]

            return MetricsSnapshot(
                total_requests=self._total_requests,
                total_errors=self._total_errors,
                requests_by_endpoint=dict(self._requests_by_endpoint),
                errors_by_endpoint=dict(self._errors_by_endpoint),
                latency_avg_ms=round(latency_avg, 2),
                latency_p50_ms=round(latency_p50, 2),
                latency_p95_ms=round(latency_p95, 2),
                latency_p99_ms=round(latency_p99, 2),
                latency_max_ms=round(latency_max, 2),
                uptime_seconds=round(time.time() - self._start_time, 2),
                tool_calls_by_name=dict(self._tool_calls_by_name),
                rate_limit_rejections=self._rate_limit_rejections
            )

    def reset(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._start_time = time.time()
            self._total_requests = 0
            self._total_errors = 0
            self._requests_by_endpoint.clear()
            self._errors_by_endpoint.clear()
            self._latency_samples.clear()
            self._tool_calls_by_name.clear()
            self._rate_limit_rejections = 0


# Global metrics collector instance
metrics_collector = MetricsCollector()


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Produces JSON log lines with consistent fields:
    - timestamp: ISO 8601 format
    - level: Log level (INFO, ERROR, etc.)
    - logger: Logger name
    - message: Log message
    - Additional fields from extra dict
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields (excluding standard LogRecord attributes)
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "taskName", "message"
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


def setup_logging(log_format: str = "text", log_level: str = "INFO") -> None:
    """Configure logging based on format preference.

    Args:
        log_format: "json" for structured JSON logging, "text" for traditional format
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if log_format.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    root_logger.addHandler(handler)


# Logging configuration
LOG_FORMAT = os.getenv("RLM_LOG_FORMAT", "text")  # "text" or "json"
LOG_LEVEL = os.getenv("RLM_LOG_LEVEL", "INFO")

# Configure logging
setup_logging(LOG_FORMAT, LOG_LEVEL)
logger = logging.getLogger("rlm-http")

# API Key para autenticação. FAIL-CLOSED: sem chave o servidor REJEITA tudo
# (401) — o comportamento antigo (key vazia = aberto pro mundo) era o footgun
# P0 da avaliação Codex 2026-06-02. Dev/testes sem auth: RLM_ALLOW_ANON=true.
API_KEY = os.getenv("RLM_API_KEY", "")
ALLOW_ANON = os.getenv("RLM_ALLOW_ANON", "false").strip().lower() in ("true", "1", "yes")
MAX_MEMORY_MB = int(os.getenv("RLM_MAX_MEMORY_MB", "1024"))
CLEANUP_THRESHOLD = float(os.getenv("RLM_CLEANUP_THRESHOLD", "80.0"))  # Quando iniciar limpeza (%)
CLEANUP_TARGET = float(os.getenv("RLM_CLEANUP_TARGET", "60.0"))  # Até quanto limpar (%)
SHOW_PERSISTENCE_ERRORS = os.getenv("RLM_SHOW_PERSISTENCE_ERRORS", "true").lower() in ("true", "1", "yes")
CLEANUP_STRATEGY = os.getenv("RLM_CLEANUP_STRATEGY", "weighted")
MAX_CONCURRENT_TASKS = int(os.getenv("RLM_MAX_CONCURRENT_TASKS", "3"))
ASYNC_PDF_THRESHOLD_MB = 5  # PDFs larger than this run as async tasks
BATCH_ASYNC_THRESHOLD_FILES = 5  # Batches larger than this run as async tasks
BATCH_ASYNC_THRESHOLD_MB = 50  # Total batch size threshold for async

# Rate limiting configuration
SSE_RATE_LIMIT_REQUESTS = int(os.getenv("RLM_SSE_RATE_LIMIT", "100"))
SSE_RATE_LIMIT_WINDOW = int(os.getenv("RLM_SSE_RATE_WINDOW", "60"))  # seconds
UPLOAD_RATE_LIMIT_REQUESTS = int(os.getenv("RLM_UPLOAD_RATE_LIMIT", "10"))
UPLOAD_RATE_LIMIT_WINDOW = int(os.getenv("RLM_UPLOAD_RATE_WINDOW", "60"))  # seconds
MAX_VAR_SIZE_MB = int(os.getenv("RLM_MAX_VAR_SIZE_MB", "50"))

# Rate limiter for SSE sessions (100 requests per minute by default)
sse_rate_limiter = SlidingWindowRateLimiter(
    max_requests=SSE_RATE_LIMIT_REQUESTS,
    window_seconds=SSE_RATE_LIMIT_WINDOW
)

# Rate limiter for uploads (10 uploads per minute by default)
upload_rate_limiter = SlidingWindowRateLimiter(
    max_requests=UPLOAD_RATE_LIMIT_REQUESTS,
    window_seconds=UPLOAD_RATE_LIMIT_WINDOW
)

# Instância global do REPL com auto-cleanup
repl = SafeREPL(
    max_memory_mb=MAX_MEMORY_MB,
    max_var_size_mb=MAX_VAR_SIZE_MB,
    cleanup_threshold_percent=CLEANUP_THRESHOLD,
    cleanup_target_percent=CLEANUP_TARGET,
    cleanup_strategy=CLEANUP_STRATEGY,
)

# Task manager para operações assíncronas
task_manager = TaskManager(max_concurrent=MAX_CONCURRENT_TASKS)

# =============================================================================
# Sessões SSE
# =============================================================================
# INVARIANTE: sse_sessions (e as queues dentro dele) só são tocados no event
# loop — nunca de threads (asyncio.Queue/dict não são thread-safe p/ isto).
# O dispatcher MCP roda em threadpool, mas roteamento/put ficam nos endpoints.

SSE_SESSIONS_PER_CLIENT = int(os.getenv("RLM_SSE_SESSIONS_PER_CLIENT", "8"))
SSE_SESSION_MAX = int(os.getenv("RLM_SSE_SESSION_MAX", "256"))
SSE_SESSION_TTL_SECONDS = int(os.getenv("RLM_SSE_SESSION_TTL_SECONDS", str(24 * 3600)))

# Sentinela de eviction: object() checado POR IDENTIDADE no generator, antes
# do json.dumps. Nunca trocar por valor serializável (None/str/dict) — passaria
# no dumps e o cliente receberia um frame SSE inválido (fail-open).
_SSE_EVICTION_SENTINEL = object()


@dataclass
class SseSession:
    """Sessão SSE viva: fila de respostas + metadados p/ caps/TTL."""
    queue: asyncio.Queue
    client_key: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


# Sessões SSE ativas
sse_sessions: dict[str, SseSession] = {}


def get_client_key(request: Request) -> str:
    """Identifica o cliente p/ caps e rate limit.

    Atrás do Traefik, request.client.host é o IP do PROXY — o IP real do
    cliente vem no X-Forwarded-For. Usamos o ÚLTIMO hop (anexado pelo
    Traefik = IP que conectou nele), não o primeiro: o primeiro é o valor
    que o PRÓPRIO cliente pode ter enviado (forjável → dodge dos caps).
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[-1].strip()
    return request.client.host if request.client else "anonymous"


def _evict_sse_session(session_id: str, reason: str) -> None:
    """Remove a sessão do registry e sinaliza o generator p/ encerrar.

    Eviction é cooperativa: o sentinel acorda o queue.get(); se o generator
    estiver suspenso no yield (cliente com backpressure TCP), a conexão linga
    até o cliente ler ou o proxy derrubar — mas a entrada já saiu do dict,
    então caps/TTL continuam corretos. SÓ rodar no event loop.
    """
    entry = sse_sessions.pop(session_id, None)
    if entry is None:
        return
    entry.queue.put_nowait(_SSE_EVICTION_SENTINEL)
    sse_rate_limiter.reset(session_id)
    logger.info(f"Sessão SSE evictada ({reason}): {session_id} client={entry.client_key}")


def register_sse_session(session_id: str, client_key: str) -> asyncio.Queue:
    """Registra sessão nova aplicando TTL + cap por cliente + cap global.

    Política: cap POR CLIENTE com evict-oldest intra-cliente — um cliente em
    reconnect-loop (burst real observado: 12 sessões/s) expulsa as PRÓPRIAS
    sessões velhas, sem churn nas sessões legítimas de outros clientes.
    O cap global é só backstop agregado.
    """
    now = time.time()

    # TTL sweep oportunista (zumbi lento; a morte normal de sessão é o
    # finally do generator no disconnect, não isto)
    for sid in [s for s, e in sse_sessions.items()
                if now - e.last_seen > SSE_SESSION_TTL_SECONDS]:
        _evict_sse_session(sid, "ttl")

    own = sorted(
        (sid for sid, e in sse_sessions.items() if e.client_key == client_key),
        key=lambda sid: sse_sessions[sid].created_at,
    )
    while len(own) >= SSE_SESSIONS_PER_CLIENT:
        _evict_sse_session(own.pop(0), "per-client-cap")

    if sse_sessions and len(sse_sessions) >= SSE_SESSION_MAX:
        oldest = min(sse_sessions, key=lambda sid: sse_sessions[sid].created_at)
        _evict_sse_session(oldest, "global-cap")

    queue: asyncio.Queue = asyncio.Queue()
    sse_sessions[session_id] = SseSession(queue=queue, client_key=client_key)
    return queue


# =============================================================================
# Autenticação
# =============================================================================

async def verify_api_key(request: Request):
    """Verifica API key. Sem RLM_API_KEY configurada: fail-closed (401),
    a menos que RLM_ALLOW_ANON=true (break-glass explícito p/ dev)."""
    if not API_KEY:
        if ALLOW_ANON:
            return True
        raise HTTPException(
            status_code=401,
            detail="Servidor sem RLM_API_KEY (fail-closed). Configure a chave "
                   "ou, para dev local, RLM_ALLOW_ANON=true.",
        )

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if hmac.compare_digest(token, API_KEY):
            return True

    # Aceita token via query param SOMENTE no endpoint SSE (EventSource não
    # envia headers custom). NÃO nos endpoints RPC (/mcp, /message): query
    # string vaza em access-log/proxy/Referer — caminho de leak da credencial.
    if request.url.path == "/sse":
        token = request.query_params.get("token", "")
        if token and hmac.compare_digest(token, API_KEY):
            return True

    raise HTTPException(status_code=401, detail="Invalid or missing API key")


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hooks"""
    logger.info(f"RLM MCP Server iniciando (max_memory={MAX_MEMORY_MB}MB)")
    if not API_KEY:
        if ALLOW_ANON:
            logger.warning(
                "RLM_API_KEY ausente + RLM_ALLOW_ANON=true → servidor SEM "
                "autenticação (modo dev — NÃO usar em produção)")
        else:
            logger.critical(
                "RLM_API_KEY ausente → FAIL-CLOSED: endpoints autenticados "
                "responderão 401 até configurar a chave (dev: RLM_ALLOW_ANON=true)")

    # Inicializa o forkserver do sandbox ANTES de abrir conexões (SQLite/minio/
    # openai) — assim o template do forkserver não herda FDs sensíveis e os
    # filhos do rlm_execute partem de uma superfície mínima.
    if getattr(repl, "sandbox_mode", "subprocess") == "subprocess":
        try:
            from .sandbox_worker import init_forkserver
            init_forkserver()
            logger.info("Sandbox: isolamento por subprocesso ATIVO (modo subprocess)")
        except Exception as e:
            logger.error(f"Falha ao inicializar o forkserver do sandbox: {e}")
    else:
        logger.warning(
            "⚠️  Sandbox em modo INSEGURO (in-process): RLM_SANDBOX_MODE=%s. "
            "O código do rlm_execute roda no mesmo processo do servidor (reabre a "
            "classe de sandbox-escape). Use apenas como break-glass.",
            getattr(repl, "sandbox_mode", "?"),
        )

    # Restaurar variáveis persistidas
    try:
        persistence = get_persistence()
        saved_vars = persistence.list_variables()
        if saved_vars:
            logger.info(f"Restaurando {len(saved_vars)} variáveis persistidas...")
            from datetime import datetime
            now = datetime.now()
            for var_info in saved_vars:
                name = var_info["name"]
                value = persistence.load_variable(name)
                if value is not None:
                    repl.variables[name] = value
                    # Create variable_metadata so list_variables() works
                    import sys
                    size = sys.getsizeof(value)
                    repl.variable_metadata[name] = VariableInfo(
                        name=name,
                        type_name=type(value).__name__,
                        size_bytes=size,
                        size_human=repl._human_size(size),
                        preview=repl._get_preview(value),
                        created_at=now,
                        last_accessed=now,
                        access_count=0,
                        pinned=False,
                        source=var_info.get("source", "persisted"),
                    )
                    # Restaurar índice keyword se existir
                    index_data = persistence.load_index(name)
                    if index_data:
                        set_index(name, TextIndex.from_dict(index_data))
                    # Restaurar embeddings vetoriais se existirem. from_persisted
                    # funde tudo na matriz float32 compacta e descarta as listas
                    # cruas; o flag is_boilerplate é recomputado do texto lá dentro
                    # (este é o caminho REAL de restore — from_serializable não roda
                    # em runtime — então o down-weight em search() depende disso).
                    emb_data = persistence.load_embeddings(name)
                    if emb_data:
                        from .vector_index import VectorIndex, set_vector_index
                        set_vector_index(
                            name, VectorIndex.from_persisted(name, value, emb_data)
                        )
                    logger.info(f"  Restaurado: {name} ({var_info['type']})")
            logger.info("Variáveis restauradas com sucesso")

        # Auto-rebuild collection indexes
        try:
            collections = persistence.list_collections()
            if collections:
                logger.info(f"Reconstruindo índices de {len(collections)} coleção(ões)...")
                for coll in collections:
                    coll_name = coll["name"]
                    all_vars = persistence.get_collection_vars(coll_name)
                    if not all_vars:
                        continue

                    combined_text, var_mapping, _ = build_collection_combined(
                        all_vars, repl.variables
                    )

                    if combined_text:
                        combined_var_name = f"_coll_{coll_name}_combined"
                        repl.variables[combined_var_name] = combined_text
                        combined_index = create_index(combined_text, combined_var_name)
                        set_index(combined_var_name, combined_index)
                        repl.variables[f"_coll_{coll_name}_mapping"] = var_mapping
                        logger.info(f"  Coleção '{coll_name}' reconstruída: {len(combined_text):,} chars, {len(all_vars)} vars")
                logger.info("Coleções reconstruídas com sucesso")
        except Exception as e:
            logger.warning(f"Erro ao reconstruir coleções: {e}")

        # Pre-warm do índice de repertorização (parse ~7s do kent_repertorio):
        # no lifespan o dispatch lock não é segurado — evita o stall na 1ª
        # chamada de rlm_repertorio pós-deploy.
        try:
            from .repertory import get_repertory_index
            from .tools.handlers.repertory_tools import DEFAULT_SOURCE
            _rep_text = repl.variables.get(DEFAULT_SOURCE)
            if isinstance(_rep_text, str) and _rep_text:
                idx, _ = get_repertory_index(DEFAULT_SOURCE, _rep_text)
                logger.info(
                    f"Índice de repertório pré-aquecido: {len(idx.entries):,} rubricas"
                )
        except Exception as e:
            logger.warning(f"Erro no pre-warm do índice de repertório: {e}")

    except Exception as e:
        logger.warning(f"Erro ao restaurar variáveis (pode ser primeira execução): {e}")

    yield
    logger.info("RLM MCP Server encerrando")
    task_manager.shutdown(wait=False)


app = FastAPI(
    title="RLM MCP Server",
    description="Recursive Language Model via MCP over HTTP/SSE",
    version="0.2.0",
    lifespan=lifespan
)

# CORS — restrito a origens configuradas (padrão: localhost + domínio do servidor)
_cors_origins_env = os.getenv("RLM_CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] if _cors_origins_env else [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "https://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Origins aceitas na validação anti DNS-rebinding (MUST da spec MCP
# Streamable HTTP). CORS protege browsers bem-comportados; este check
# protege contra rebinding (Origin presente mas hostname forjado).
ALLOWED_ORIGINS = set(CORS_ORIGINS) | {"https://rlm.drsolution.online"}


async def verify_origin(request: Request):
    """Rejeita Origin desconhecida (anti DNS-rebinding, MUST da spec MCP).

    Clientes não-browser (Claude Code CLI, curl) não enviam Origin → passam.
    """
    origin = request.headers.get("origin", "")
    if origin and origin not in ALLOWED_ORIGINS:
        raise HTTPException(status_code=403, detail="Origin not allowed")
    return True


# =============================================================================
# Models
# =============================================================================

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any | None = None
    error: dict | None = None


# =============================================================================
# MCP Protocol Implementation
# =============================================================================

# Versões de protocolo que este servidor sabe servir. A spec manda ecoar a
# versão pedida QUANDO suportada; responder uma diferente faz clientes
# Streamable HTTP tratarem como negociação p/ baixo (ok, mas evitável).
SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")

# Lock global serializando o dispatcher MCP quando roda no threadpool.
# NOTA HONESTA: preserva a serialização de fato que existia quando o handler
# bloqueava o event loop — NÃO é serialização total: as worker threads do
# task_manager mutam repl.variables/índices por FORA deste lock (pré-existente;
# mitigado pontualmente com _execute_lock no load_data e lock no rate limiter).
_mcp_dispatch_lock = RLock()


def handle_mcp_request_locked(request: "MCPRequest", client_id: str | None = None) -> "MCPResponse | None":
    """Wrapper p/ run_in_threadpool: serializa o dispatch fora do event loop.

    Tools lentas (rlm_execute até RLM_EXECUTE_TIMEOUT=60s) ficam serializadas
    entre si (como sempre foram), mas /health, pings SSE e respostas 429
    continuam vivos no event loop.
    """
    with _mcp_dispatch_lock:
        return handle_mcp_request(request, client_id=client_id)


def handle_mcp_request(request: MCPRequest, client_id: str | None = None) -> MCPResponse | None:
    """Processa uma requisição MCP.

    Retorna None para notificações JSON-RPC (que não recebem resposta).

    Args:
        request: Requisição MCP
        client_id: Identificador do cliente para rate limiting (session_id ou IP)
    """
    try:
        method = request.method
        params = request.params or {}

        if method == "initialize":
            requested_version = params.get("protocolVersion", "2024-11-05")
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": (
                        requested_version
                        if requested_version in SUPPORTED_PROTOCOL_VERSIONS
                        else "2024-11-05"
                    ),
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "rlm-mcp-server",
                        "version": "0.2.0"
                    }
                }
            )

        elif method == "notifications/initialized":
            # Notificação, não precisa de resposta
            return None

        elif method.startswith("notifications/"):
            # JSON-RPC: notificações NUNCA recebem resposta — nem de erro.
            # (Responder "Method not found" a notifications/cancelled era
            # violação de protocolo — incidente 2026-06-06.) Cancelamento
            # efetivo de tool em andamento é opcional pela spec (MAY ignore);
            # aqui é no-op aceito, só logado.
            if method == "notifications/cancelled":
                logger.info(
                    f"MCP notification: {method} "
                    f"requestId={params.get('requestId')} reason={params.get('reason')!r}"
                )
            else:
                logger.info(f"MCP notification ignorada (no-op): {method}")
            return None

        elif method == "tools/list":
            return MCPResponse(
                id=request.id,
                result={
                    "tools": get_tools_list()
                }
            )

        elif method == "resources/list":
            return MCPResponse(
                id=request.id,
                result={
                    "resources": get_resources_list()
                }
            )

        elif method == "resources/read":
            uri = params.get("uri", "")
            content = read_resource(uri)
            if content is None:
                return MCPResponse(
                    id=request.id,
                    error={
                        "code": -32602,
                        "message": f"Resource not found: {uri}"
                    }
                )
            return MCPResponse(
                id=request.id,
                result={
                    "contents": [content]
                }
            )

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            result = call_tool(tool_name, tool_args, client_id=client_id)
            return MCPResponse(
                id=request.id,
                result=result
            )

        else:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            )

    except RateLimitExceeded:
        # Re-raise rate limit exceptions to be handled by HTTP endpoint
        raise
    except Exception as e:
        logger.exception(f"Erro ao processar request MCP: {e}")
        return MCPResponse(
            id=request.id,
            error={
                "code": -32603,
                "message": str(e)
            }
        )


def get_resources_list() -> list[dict]:
    """Retorna lista de resources disponíveis no MCP.

    Resources são endpoints read-only para dados estáticos ou semi-estáticos
    que podem ser lidos por clientes MCP usando resources/read.
    """
    return [
        {
            "uri": "rlm://variables",
            "name": "Variables",
            "description": "Lista de variáveis persistidas no REPL",
            "mimeType": "application/json"
        },
        {
            "uri": "rlm://memory",
            "name": "Memory Usage",
            "description": "Uso de memória atual do REPL",
            "mimeType": "application/json"
        },
        {
            "uri": "rlm://collections",
            "name": "Collections",
            "description": "Lista de coleções de variáveis",
            "mimeType": "application/json"
        }
    ]


def read_resource(uri: str) -> dict | None:
    """Lê o conteúdo de um resource MCP.

    Args:
        uri: URI do resource (ex: rlm://variables)

    Returns:
        Dict com uri, mimeType e text (conteúdo JSON), ou None se não encontrado
    """
    if uri == "rlm://variables":
        # Lista todas as variáveis persistidas (excluindo funções internas)
        vars_list = repl.list_variables()
        variables = []
        for v in vars_list:
            # Filtra funções internas do REPL
            if v.name in INTERNAL_FUNCTION_NAMES:
                continue
            variables.append({
                "name": v.name,
                "type": v.type_name,
                "size_bytes": v.size_bytes,
                "size_human": v.size_human,
                "preview": v.preview,
                "created_at": v.created_at.isoformat(),
                "last_accessed": v.last_accessed.isoformat()
            })
        return {
            "uri": uri,
            "mimeType": "application/json",
            "text": json.dumps({"variables": variables, "count": len(variables)}, indent=2)
        }

    if uri == "rlm://memory":
        # Retorna estatísticas de uso de memória do REPL
        mem = repl.get_memory_usage()
        memory_data = {
            "total_bytes": mem["total_bytes"],
            "total_human": mem["total_human"],
            "variable_count": mem["variable_count"],
            "max_allowed_mb": mem["max_allowed_mb"],
            "usage_percent": round(mem["usage_percent"], 2)
        }
        return {
            "uri": uri,
            "mimeType": "application/json",
            "text": json.dumps(memory_data, indent=2)
        }

    if uri == "rlm://collections":
        # Lista todas as coleções de variáveis
        persistence = get_persistence()
        collections_list = persistence.list_collections()
        collections = []
        for c in collections_list:
            collections.append({
                "name": c["name"],
                "description": c["description"],
                "variable_count": c["var_count"],
                "created_at": c["created_at"]
            })
        return {
            "uri": uri,
            "mimeType": "application/json",
            "text": json.dumps({"collections": collections, "count": len(collections)}, indent=2)
        }

    # Resources não implementados retornam None
    return None


def get_tools_list() -> list[dict]:
    """Retorna lista de tools disponíveis"""
    return TOOL_SCHEMAS


def call_tool(name: str, arguments: dict, client_id: str | None = None) -> dict:
    """Executa uma tool e retorna resultado.

    Dispatcher fino: o corpo de cada tool vive em tools/handlers/* (registry
    TOOL_HANDLERS). Este wrapper preserva o contrato do monolito original:
    métricas por nome (inclusive nos dispatches recursivos dos routers),
    RateLimitExceeded re-levantado para o endpoint virar 429, e qualquer
    outra exceção logada + resposta isError genérica.

    Args:
        name: Nome da tool a ser executada
        arguments: Argumentos da tool
        client_id: Identificador do cliente para rate limiting (session_id ou IP)
    """
    # Record tool call for metrics
    metrics_collector.record_tool_call(name)

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {
            "content": [
                {"type": "text", "text": f"Tool desconhecida: {name}"}
            ],
            "isError": True
        }

    # Contexto construído POR CHAMADA lendo os globals do módulo — preserva
    # monkeypatch de testes (late-binding) sem import circular nos handlers.
    ctx = ToolContext(
        repl=repl,
        task_manager=task_manager,
        upload_rate_limiter=upload_rate_limiter,
        call_tool=call_tool,
        client_id=client_id,
        show_persistence_errors=SHOW_PERSISTENCE_ERRORS,
        async_pdf_threshold_mb=ASYNC_PDF_THRESHOLD_MB,
        batch_async_threshold_files=BATCH_ASYNC_THRESHOLD_FILES,
        batch_async_threshold_mb=BATCH_ASYNC_THRESHOLD_MB,
    )

    try:
        return handler(arguments, ctx)
    except RateLimitExceeded:
        # Re-raise rate limit exceptions to be handled by HTTP endpoint
        raise
    except Exception as e:
        logger.exception(f"Erro ao executar tool {name}")
        return {
            "content": [
                {"type": "text", "text": f"Erro: {e}"}
            ],
            "isError": True
        }


# =============================================================================
# HTTP Endpoints
# =============================================================================

def generate_request_id() -> str:
    """Generate a unique request ID for tracing.

    Returns:
        A UUID4 string to uniquely identify the request.
    """
    return str(uuid.uuid4())


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    request_id = generate_request_id()
    mem = repl.get_memory_usage()
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "memory": mem,
            "version": "0.2.0",
            "request_id": request_id
        },
        headers={"X-Request-Id": request_id}
    )


@app.get("/metrics")
async def metrics_endpoint():
    """Returns server metrics including request counts, errors, and latency statistics.

    Metrics include:
    - total_requests: Total number of requests processed
    - total_errors: Total number of error responses
    - requests_by_endpoint: Request count per endpoint
    - errors_by_endpoint: Error count per endpoint
    - latency_avg_ms: Average latency in milliseconds
    - latency_p50_ms: 50th percentile latency (median)
    - latency_p95_ms: 95th percentile latency
    - latency_p99_ms: 99th percentile latency
    - latency_max_ms: Maximum latency
    - uptime_seconds: Server uptime in seconds
    - tool_calls_by_name: Count of tool calls by tool name
    - rate_limit_rejections: Count of rate limit rejections
    """
    request_id = generate_request_id()
    snapshot = metrics_collector.get_snapshot()
    return JSONResponse(
        content={
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": snapshot.uptime_seconds,
            "requests": {
                "total": snapshot.total_requests,
                "by_endpoint": snapshot.requests_by_endpoint
            },
            "errors": {
                "total": snapshot.total_errors,
                "by_endpoint": snapshot.errors_by_endpoint
            },
            "latency_ms": {
                "avg": snapshot.latency_avg_ms,
                "p50": snapshot.latency_p50_ms,
                "p95": snapshot.latency_p95_ms,
                "p99": snapshot.latency_p99_ms,
                "max": snapshot.latency_max_ms
            },
            "tools": {
                "calls_by_name": snapshot.tool_calls_by_name
            },
            "rate_limiting": {
                "rejections": snapshot.rate_limit_rejections
            },
            "request_id": request_id
        },
        headers={"X-Request-Id": request_id}
    )


@app.get("/sse")
async def sse_endpoint(
    request: Request,
    _: bool = Depends(verify_api_key),
    __: bool = Depends(verify_origin),
):
    """
    SSE endpoint para MCP.
    O cliente se conecta aqui para receber eventos do servidor.
    """
    client_key = get_client_key(request)

    # Rate limit no próprio connect: um cliente em reconnect-loop (burst real
    # observado: 12 sessões/s) é barrado aqui, não só contido pelos caps.
    rate_result = sse_rate_limiter.check_and_record(f"sse-connect:{client_key}")
    if not rate_result.allowed:
        metrics_collector.record_rate_limit_rejection()
        logger.warning(
            f"Rate limit no /sse connect: client={client_key} "
            f"{rate_result.current_count}/{rate_result.limit}"
        )
        return JSONResponse(
            {
                "error": "Too Many Requests",
                "message": f"Rate limit exceeded: {rate_result.limit} connections per {rate_result.window_seconds} seconds",
                "retry_after": rate_result.retry_after,
            },
            status_code=429,
            headers={"Retry-After": str(int(rate_result.retry_after or 1))},
        )

    session_id = str(uuid.uuid4())
    queue = register_sse_session(session_id, client_key)

    logger.info(f"Nova sessão SSE: {session_id} client={client_key}")

    async def event_generator():
        """
        Async generator that yields SSE events for the MCP session.

        Yields:
            str: SSE-formatted events including:
                - endpoint event with session_id for client to use in POST requests
                - message events with JSON-encoded MCP responses
                - ping comments to keep the connection alive

        The generator runs until the client disconnects or the server closes.
        On completion, it cleans up the session from sse_sessions and rate limiter.
        """
        try:
            # Envia o session_id para o cliente usar no POST
            yield f"event: endpoint\ndata: /message?session_id={session_id}\n\n"

            while True:
                try:
                    # Aguarda mensagens na fila (com timeout para manter conexão viva)
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    if message is _SSE_EVICTION_SENTINEL:
                        # Evictado (TTL/cap): encerra o stream limpo. Checagem
                        # por IDENTIDADE antes do json.dumps (fail-closed:
                        # mesmo sem este if, dumps(object()) levantaria
                        # TypeError → finally → sem frame inválido no fio).
                        break
                    yield f"event: message\ndata: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Envia ping para manter conexão
                    yield ": ping\n\n"
                except asyncio.CancelledError:
                    break
        finally:
            # Idempotente com eviction externa (pop com default + reset guardado)
            sse_sessions.pop(session_id, None)
            sse_rate_limiter.reset(session_id)  # Clean up rate limiter state
            logger.info(f"Sessão SSE encerrada: {session_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id
        }
    )


@app.post("/message")
async def message_endpoint(
    request: Request,
    session_id: str = None,
    _: bool = Depends(verify_api_key),
    __: bool = Depends(verify_origin),
):
    """
    Endpoint para enviar mensagens MCP.
    - session_id conhecido → executa e responde via fila SSE (202).
    - session_id fornecido mas DESCONHECIDO → 404 imediato, sem executar.
    - session_id ausente → request/response direto no body (200).

    Rate limiting: 100 requests/minute per SSE session.
    """
    request_id = generate_request_id()
    start_time = time.time()
    is_error = False

    # P0 (incidente 2026-06-06): session_id stale (sobreviveu a restart do
    # servidor ou eviction) → 404 ANTES de parse/execução. O fallback antigo
    # (executar e responder no body) era invisível pro cliente SSE — que por
    # spec só lê respostas do stream — e virava "hang" de 37min. O SDK
    # cliente faz fail-fast no POST !ok (throw imediato) e reconecta.
    if session_id is not None and session_id not in sse_sessions:
        latency_ms = (time.time() - start_time) * 1000
        metrics_collector.record_request("/message", latency_ms, is_error=True)
        logger.warning(
            f"/message com session_id desconhecido (stale): {session_id} — 404",
            extra={"request_id": request_id, "session_id": session_id},
        )
        return JSONResponse(
            {"error": "SSE session not found", "request_id": request_id},
            status_code=404,
            headers={"X-Request-Id": request_id},
        )

    # Rate limiting for SSE sessions
    if session_id and session_id in sse_sessions:
        sse_sessions[session_id].last_seen = time.time()
        rate_result = sse_rate_limiter.check_and_record(session_id)
        if not rate_result.allowed:
            logger.warning(f"Rate limit exceeded for session {session_id}: {rate_result.current_count}/{rate_result.limit}", extra={"request_id": request_id})
            metrics_collector.record_rate_limit_rejection()
            latency_ms = (time.time() - start_time) * 1000
            metrics_collector.record_request("/message", latency_ms, is_error=True)
            return JSONResponse(
                {
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded: {rate_result.limit} requests per {rate_result.window_seconds} seconds",
                    "retry_after": rate_result.retry_after,
                    "request_id": request_id
                },
                status_code=429,
                headers={"Retry-After": str(int(rate_result.retry_after or 1)), "X-Request-Id": request_id}
            )

    try:
        body = await request.json()
        mcp_request = MCPRequest(**body)

        # INFO com método/tool NA MENSAGEM: o formatter text (default) ignora
        # extra= — só o modo json o exibe. Sem isto, incidente fica cego
        # (não dava pra saber QUAL tool foi chamada nos logs de 2026-06-06).
        tool_name = (
            (mcp_request.params or {}).get("name")
            if mcp_request.method == "tools/call" else None
        )
        req_desc = f"method={mcp_request.method}" + (f" tool={tool_name}" if tool_name else "")
        logger.info(
            f"MCP request: {req_desc} transport={'sse' if session_id else 'direct'}",
            extra={"request_id": request_id, "session_id": session_id,
                   "mcp_method": mcp_request.method, "tool_name": tool_name},
        )

        # Use session_id as client_id for rate limiting, fallback to client IP
        client_id = session_id if session_id else get_client_key(request)
        # Dispatcher roda FORA do event loop (threadpool): /health, pings SSE
        # e 429 continuam respondendo durante tools lentas (execute até 60s).
        response = await run_in_threadpool(handle_mcp_request_locked, mcp_request, client_id)

        latency_ms = (time.time() - start_time) * 1000

        if response is None:
            # Notificação, não precisa responder
            metrics_collector.record_request("/message", latency_ms, is_error=False)
            logger.info(
                f"MCP notification aceita: {req_desc} latency_ms={latency_ms:.1f}",
                extra={"request_id": request_id, "latency_ms": latency_ms},
            )
            return Response(status_code=202, headers={"X-Request-Id": request_id})

        response_dict = response.model_dump(exclude_none=True)

        # Check if response has error
        if response.error:
            is_error = True
            logger.warning(f"MCP error response: {response.error}", extra={"request_id": request_id})

        # Se tem sessão SSE viva, envia por lá. get() + put_nowait ADJACENTES
        # (sem await entre eles → janela de race zero no event loop).
        if session_id is not None:
            entry = sse_sessions.get(session_id)
            if entry is not None:
                entry.queue.put_nowait(response_dict)
                entry.last_seen = time.time()
                metrics_collector.record_request("/message", latency_ms, is_error=is_error)
                logger.info(
                    f"MCP done: {req_desc} via=sse latency_ms={latency_ms:.1f} error={is_error}",
                    extra={"request_id": request_id, "latency_ms": latency_ms},
                )
                return Response(status_code=202, headers={"X-Request-Id": request_id})
            # Sessão evictada DURANTE o tool call (janela do threadpool):
            # fall-through pra resposta no body — não descartar resultado já
            # computado (404 aqui jogaria fora um execute de até 60s).
            logger.warning(
                f"Sessão SSE {session_id} evictada durante o request; respondendo no body",
                extra={"request_id": request_id, "session_id": session_id},
            )

        # Senão, responde diretamente
        metrics_collector.record_request("/message", latency_ms, is_error=is_error)
        logger.info(
            f"MCP done: {req_desc} via=direct latency_ms={latency_ms:.1f} error={is_error}",
            extra={"request_id": request_id, "latency_ms": latency_ms},
        )
        return JSONResponse(response_dict, headers={"X-Request-Id": request_id})

    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded: {e.message}", extra={"request_id": request_id})
        metrics_collector.record_rate_limit_rejection()
        latency_ms = (time.time() - start_time) * 1000
        metrics_collector.record_request("/message", latency_ms, is_error=True)
        return JSONResponse(
            {
                "error": "Too Many Requests",
                "message": e.message,
                "retry_after": e.retry_after,
                "request_id": request_id
            },
            status_code=429,
            headers={"Retry-After": str(int(e.retry_after)), "X-Request-Id": request_id}
        )

    except Exception as e:
        logger.exception("Erro ao processar mensagem", extra={"request_id": request_id})
        latency_ms = (time.time() - start_time) * 1000
        metrics_collector.record_request("/message", latency_ms, is_error=True)
        return JSONResponse(
            {"error": str(e), "request_id": request_id},
            status_code=500,
            headers={"X-Request-Id": request_id}
        )


@app.post("/mcp")
async def mcp_direct_endpoint(
    request: Request,
    _: bool = Depends(verify_api_key),
    __: bool = Depends(verify_origin),
):
    """
    Endpoint direto para MCP (sem SSE) — compatível com o fluxo básico do
    transporte Streamable HTTP (POST JSON-RPC → JSON; notificação → 202;
    GET/DELETE → 405 automático do framework, permitido pela spec; servidor
    stateless sem Mcp-Session-Id, permitido).

    ATENÇÃO: NUNCA retornar 404 neste endpoint — no Streamable HTTP, 404 tem
    semântica reservada de "sessão expirou, re-inicialize" e colocaria
    clientes compliant em loop de re-initialize.
    """
    request_id = generate_request_id()
    start_time = time.time()
    is_error = False

    # Rate limiting por cliente (X-Forwarded-For: atrás do Traefik o
    # request.client.host é o IP do proxy — todos os clientes colidiam)
    client_id = get_client_key(request)
    rate_result = sse_rate_limiter.check_and_record(client_id)
    if not rate_result.allowed:
        logger.warning(f"Rate limit exceeded for {client_id}: {rate_result.current_count}/{rate_result.limit}", extra={"request_id": request_id})
        metrics_collector.record_rate_limit_rejection()
        latency_ms = (time.time() - start_time) * 1000
        metrics_collector.record_request("/mcp", latency_ms, is_error=True)
        return JSONResponse(
            {
                "error": "Too Many Requests",
                "message": f"Rate limit exceeded: {rate_result.limit} requests per {rate_result.window_seconds} seconds",
                "retry_after": rate_result.retry_after,
                "request_id": request_id
            },
            status_code=429,
            headers={"Retry-After": str(int(rate_result.retry_after or 1)), "X-Request-Id": request_id}
        )

    try:
        body = await request.json()
        mcp_request = MCPRequest(**body)

        tool_name = (
            (mcp_request.params or {}).get("name")
            if mcp_request.method == "tools/call" else None
        )
        req_desc = f"method={mcp_request.method}" + (f" tool={tool_name}" if tool_name else "")
        logger.info(
            f"MCP request: {req_desc} transport=http",
            extra={"request_id": request_id, "mcp_method": mcp_request.method,
                   "tool_name": tool_name},
        )

        # Dispatcher fora do event loop (mesma razão do /message)
        response = await run_in_threadpool(handle_mcp_request_locked, mcp_request, client_id)

        latency_ms = (time.time() - start_time) * 1000

        if response is None:
            metrics_collector.record_request("/mcp", latency_ms, is_error=False)
            logger.info(
                f"MCP notification aceita: {req_desc} latency_ms={latency_ms:.1f}",
                extra={"request_id": request_id, "latency_ms": latency_ms},
            )
            return Response(status_code=202, headers={"X-Request-Id": request_id})

        # Check if response has error
        if response.error:
            is_error = True
            logger.warning(f"MCP error response: {response.error}", extra={"request_id": request_id})

        metrics_collector.record_request("/mcp", latency_ms, is_error=is_error)
        logger.info(
            f"MCP done: {req_desc} via=http latency_ms={latency_ms:.1f} error={is_error}",
            extra={"request_id": request_id, "latency_ms": latency_ms},
        )
        return JSONResponse(response.model_dump(exclude_none=True), headers={"X-Request-Id": request_id})

    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded: {e.message}", extra={"request_id": request_id})
        metrics_collector.record_rate_limit_rejection()
        latency_ms = (time.time() - start_time) * 1000
        metrics_collector.record_request("/mcp", latency_ms, is_error=True)
        return JSONResponse(
            {
                "error": "Too Many Requests",
                "message": e.message,
                "retry_after": e.retry_after,
                "request_id": request_id
            },
            status_code=429,
            headers={"Retry-After": str(int(e.retry_after)), "X-Request-Id": request_id}
        )

    except Exception as e:
        logger.exception("Erro ao processar MCP request", extra={"request_id": request_id})
        latency_ms = (time.time() - start_time) * 1000
        metrics_collector.record_request("/mcp", latency_ms, is_error=True)
        return JSONResponse(
            {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "request_id": request_id},
            status_code=500,
            headers={"X-Request-Id": request_id}
        )


# =============================================================================
# Main
# =============================================================================

def main():
    """Entry point"""
    host = os.getenv("RLM_HOST", "0.0.0.0")
    port = int(os.getenv("RLM_PORT", "8765"))

    logger.info(f"Iniciando RLM MCP HTTP Server em {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
