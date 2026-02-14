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
import uvicorn

from .repl import SafeREPL, ExecutionResult, INTERNAL_FUNCTION_NAMES
from .s3_client import get_s3_client
from .pdf_parser import extract_pdf
from .persistence import get_persistence
from .indexer import get_index, set_index, TextIndex, auto_index_if_large, hybrid_search
from .rate_limiter import SlidingWindowRateLimiter, RateLimitResult
from .tools.schemas import TOOL_SCHEMAS
from .services.s3_guard import require_s3_configured
from .services.persistence_service import persist_and_index
from .task_manager import TaskManager
from . import response_formatter as fmt
from . import code_parser
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


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


class RateLimitExceeded(Exception):
    """Exception raised when a rate limit is exceeded.

    Attributes:
        limit: Maximum allowed requests in the window
        window_seconds: Time window in seconds
        retry_after: Seconds to wait before retrying
        message: Human-readable error message
    """
    def __init__(self, result: RateLimitResult, message: str = None):
        self.limit = result.limit
        self.window_seconds = result.window_seconds
        self.retry_after = result.retry_after or 1
        self.current_count = result.current_count
        self.message = message or f"Rate limit exceeded: {result.limit} requests per {result.window_seconds} seconds"
        super().__init__(self.message)


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

# API Key para autenticação
API_KEY = os.getenv("RLM_API_KEY", "")
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

# Sessões SSE ativas
sse_sessions: dict[str, asyncio.Queue] = {}


# =============================================================================
# Autenticação
# =============================================================================

async def verify_api_key(request: Request):
    """Verifica API key se configurada"""
    if not API_KEY:
        return True

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if hmac.compare_digest(token, API_KEY):
            return True

    # Também aceita como query param para SSE (browsers não enviam headers custom em EventSource)
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

    # Restaurar variáveis persistidas
    try:
        persistence = get_persistence()
        saved_vars = persistence.list_variables()
        if saved_vars:
            logger.info(f"Restaurando {len(saved_vars)} variáveis persistidas...")
            for var_info in saved_vars:
                name = var_info["name"]
                value = persistence.load_variable(name)
                if value is not None:
                    repl.variables[name] = value
                    # Restaurar índice keyword se existir
                    index_data = persistence.load_index(name)
                    if index_data:
                        set_index(name, TextIndex.from_dict(index_data))
                    # Restaurar embeddings vetoriais se existirem
                    emb_data = persistence.load_embeddings(name)
                    if emb_data:
                        from .vector_index import VectorIndex, ChunkInfo, set_vector_index
                        vi = VectorIndex(var_name=name)
                        vi.total_chars = len(value) if isinstance(value, str) else 0
                        vi.total_lines = value.count('\n') + 1 if isinstance(value, str) else 0
                        vi.chunks = [
                            ChunkInfo(
                                chunk_index=c["chunk_index"],
                                text=c["chunk_text"],
                                line_start=c["line_start"],
                                line_end=c["line_end"],
                                embedding=c["embedding"],
                            )
                            for c in emb_data
                        ]
                        set_vector_index(name, vi)
                    logger.info(f"  Restaurado: {name} ({var_info['type']})")
            logger.info("Variáveis restauradas com sucesso")
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

def handle_mcp_request(request: MCPRequest, client_id: str | None = None) -> MCPResponse:
    """Processa uma requisição MCP.

    Args:
        request: Requisição MCP
        client_id: Identificador do cliente para rate limiting (session_id ou IP)
    """
    try:
        method = request.method
        params = request.params or {}

        if method == "initialize":
            return MCPResponse(
                id=request.id,
                result={
                    "protocolVersion": "2024-11-05",
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

    Args:
        name: Nome da tool a ser executada
        arguments: Argumentos da tool
        client_id: Identificador do cliente para rate limiting (session_id ou IP)
    """
    # Record tool call for metrics
    metrics_collector.record_tool_call(name)

    try:
        if name == "rlm_execute":
            result = repl.execute(arguments["code"])
            return {
                "content": [
                    {"type": "text", "text": fmt.format_execution_result(result)}
                ]
            }

        elif name == "rlm_load_data":
            var_name = arguments["name"]
            data = arguments["data"]
            data_type = arguments.get("data_type", "text")

            # "code" loads as text but also auto-parses the code structure
            actual_type = "text" if data_type == "code" else data_type

            # Set source on metadata
            result = repl.load_data(name=var_name, data=data, data_type=actual_type)
            if var_name in repl.variable_metadata:
                repl.variable_metadata[var_name].source = "load_data"

            # Auto-parse code structure if data_type="code"
            if data_type == "code" and result.success:
                lang = code_parser.detect_language(var_name, data)
                if lang and code_parser.is_available():
                    structure = code_parser.parse(data, lang)
                    if structure:
                        repl.variables[f"_code_structure_{var_name}"] = structure

            # Auto-persistência e indexação
            value = repl.variables.get(var_name)
            persist_msg, index_msg, persist_error = persist_and_index(var_name, value, repl)
            if SHOW_PERSISTENCE_ERRORS:
                pass  # persist_error already contains the error
            else:
                persist_error = ""

            size_human = repl.variable_metadata[var_name].size_human if var_name in repl.variable_metadata else "?"
            output = fmt.format_load_response(
                source="direct", var_name=var_name, size_human=size_human,
                data_type=data_type, exec_result=result,
                persist_msg=persist_msg, index_msg=index_msg, persist_error=persist_error,
            )

            return {"content": [{"type": "text", "text": output}]}

        elif name == "rlm_load_file":
            path = arguments["path"]
            data_type = arguments.get("data_type", "text")

            # Validação de segurança
            if not path.startswith("/data/"):
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Caminho deve começar com /data/"}
                    ],
                    "isError": True
                }

            import os.path
            real_path = os.path.realpath(path)
            if not real_path.startswith("/data"):
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Path traversal detectado"}
                    ],
                    "isError": True
                }

            try:
                # PDF handling
                if data_type in ("pdf", "pdf_ocr"):
                    method = "ocr" if data_type == "pdf_ocr" else "auto"
                    pdf_result = extract_pdf(path, method=method)

                    if not pdf_result.success:
                        return {
                            "content": [
                                {"type": "text", "text": f"Erro ao extrair PDF: {pdf_result.error}"}
                            ],
                            "isError": True
                        }

                    data = pdf_result.text
                    var_name = arguments["name"]
                    result = repl.load_data(
                        name=var_name,
                        data=data,
                        data_type="text"
                    )
                    if var_name in repl.variable_metadata:
                        repl.variable_metadata[var_name].source = "file"

                    text = fmt.format_file_load_pdf(path, pdf_result, result, var_name)
                    return {"content": [{"type": "text", "text": text}]}

                # Regular file handling
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    data = f.read()

                var_name = arguments["name"]
                actual_type = "text" if data_type == "code" else data_type
                result = repl.load_data(
                    name=var_name,
                    data=data,
                    data_type=actual_type
                )
                if var_name in repl.variable_metadata:
                    repl.variable_metadata[var_name].source = f"file:{path}"

                # Auto-parse code structure if data_type="code"
                if data_type == "code" and result.success:
                    lang = code_parser.detect_language(path, data)
                    if lang and code_parser.is_available():
                        structure = code_parser.parse(data, lang)
                        if structure:
                            repl.variables[f"_code_structure_{var_name}"] = structure

                return {
                    "content": [
                        {"type": "text", "text": fmt.format_execution_result(result)}
                    ]
                }
            except FileNotFoundError:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro: Arquivo não encontrado: {path}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_list_vars":
            limit = arguments.get("limit", 50)
            offset = arguments.get("offset", 0)
            vars_list = repl.list_variables()
            text = fmt.format_list_vars(vars_list, len(vars_list), offset, limit)
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_var_info":
            info = repl.get_variable_info(arguments["name"])
            if not info:
                text = f"Variável '{arguments['name']}' não encontrada."
            else:
                text = fmt.format_var_info(info)
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_clear":
            if arguments.get("all"):
                count = repl.clear_all()
                text = f"Todas as {count} variáveis foram removidas."
            elif "name" in arguments:
                if repl.clear_variable(arguments["name"]):
                    text = f"Variável '{arguments['name']}' removida."
                else:
                    text = f"Variável '{arguments['name']}' não encontrada."
            else:
                text = "Especifique 'name' ou 'all=true'."
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_memory":
            mem = repl.get_memory_usage()
            text = fmt.format_memory(mem)
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_load_s3":
            s3, error = require_s3_configured()
            if error:
                return error

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
                        if not SHOW_PERSISTENCE_ERRORS:
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
                if not SHOW_PERSISTENCE_ERRORS:
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

        elif name == "rlm_list_buckets":
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

        elif name == "rlm_list_s3":
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

        elif name == "rlm_upload_url":
            # Rate limit check for uploads
            rate_id = client_id or "anonymous"
            rate_result = upload_rate_limiter.check(rate_id)
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
                upload_rate_limiter.record(rate_id)
                text = fmt.format_upload_url(url, result)
                return {"content": [{"type": "text", "text": text}]}
            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao fazer upload de URL: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_process_pdf":
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
                if size_mb > ASYNC_PDF_THRESHOLD_MB:
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

                    task_info = task_manager.submit(
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

        elif name == "rlm_search_index":
            var_name = arguments["var_name"]
            terms = arguments["terms"]
            mode = arguments.get("mode", "keyword")
            require_all = arguments.get("require_all", False)
            limit = arguments.get("limit", 20)
            offset = arguments.get("offset", 0)

            # Verificar se variável existe
            if var_name not in repl.variables:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro: Variável '{var_name}' não encontrada no REPL."}
                    ],
                    "isError": True
                }

            try:
                if mode in ("semantic", "hybrid"):
                    # Use hybrid search (supports keyword, semantic, hybrid)
                    search_result = hybrid_search(
                        var_name, terms, mode=mode,
                        require_all=require_all,
                        limit=limit, offset=offset,
                    )
                    text = fmt.format_hybrid_search(
                        search_result, terms, var_name,
                        offset=offset, limit=limit,
                    )
                    return {"content": [{"type": "text", "text": text}]}
                else:
                    # Pure keyword search (original behavior)
                    index = get_index(var_name)
                    if not index:
                        return {
                            "content": [
                                {"type": "text", "text": f"Erro: Variável '{var_name}' não possui índice. Indexação automática ocorre para textos >= 100k chars."}
                            ],
                            "isError": True
                        }

                    results = index.search_multiple(terms, require_all=require_all)
                    total_results = len(results) if results else 0
                    stats = index.get_stats()

                    text = fmt.format_search_response(
                        results, terms, require_all, total_results,
                        offset, limit, index_stats=stats,
                    )
                    return {"content": [{"type": "text", "text": text}]}

            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro na busca: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_persistence_stats":
            try:
                persistence = get_persistence()
                stats = persistence.get_stats()
                saved_vars = persistence.list_variables()

                text = fmt.format_persistence_stats(stats, saved_vars)
                return {"content": [{"type": "text", "text": text}]}

            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao obter estatísticas: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_collection_create":
            try:
                persistence = get_persistence()
                coll_name = arguments["name"]
                description = arguments.get("description")

                success = persistence.create_collection(coll_name, description)

                if not success:
                    return {
                        "content": [
                            {"type": "text", "text": f"Erro: Falha ao criar coleção '{coll_name}' - verifique logs do servidor"}
                        ],
                        "isError": True
                    }

                text = f"✅ Coleção '{coll_name}' criada"
                if description:
                    text += f"\nDescrição: {description}"

                return {"content": [{"type": "text", "text": text}]}

            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao criar coleção: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_collection_add":
            try:
                persistence = get_persistence()
                coll_name = arguments["collection"]
                var_names = arguments["vars"]

                # Verificar se variáveis existem
                missing = [v for v in var_names if v not in repl.variables]
                if missing:
                    return {
                        "content": [
                            {"type": "text", "text": f"Erro: Variáveis não encontradas: {', '.join(missing)}"}
                        ],
                        "isError": True
                    }

                added = persistence.add_to_collection(coll_name, var_names)

                # === OPÇÃO C: Criar índice combinado da coleção ===
                # Obter TODAS as variáveis da coleção (não só as novas)
                all_vars = persistence.get_collection_vars(coll_name)

                # Concatenar todas as variáveis com separadores claros
                combined_parts = []
                var_mapping = {}  # Mapeia linha -> (var_name, linha_original)
                current_line = 1

                for var_name in all_vars:
                    if var_name in repl.variables:
                        value = repl.variables[var_name]
                        if isinstance(value, str):
                            # Adicionar header identificador
                            header = f"\n{'='*60}\n=== VARIÁVEL: {var_name} ===\n{'='*60}\n"
                            combined_parts.append(header)

                            # Registrar mapeamento de linhas
                            header_lines = header.count('\n')
                            current_line += header_lines

                            # Adicionar conteúdo e mapear linhas
                            content_lines = value.split('\n')
                            for i, _ in enumerate(content_lines):
                                var_mapping[current_line + i] = (var_name, i + 1)

                            combined_parts.append(value)
                            current_line += len(content_lines)

                if combined_parts:
                    combined_text = "\n".join(combined_parts)
                    combined_var_name = f"_coll_{coll_name}_combined"

                    # Salvar variável combinada no REPL
                    repl.variables[combined_var_name] = combined_text

                    # Forçar criação de índice (min_chars=0)
                    from .indexer import create_index, set_index
                    combined_index = create_index(combined_text, combined_var_name)
                    set_index(combined_var_name, combined_index)

                    # Salvar mapeamento como metadado
                    repl.variables[f"_coll_{coll_name}_mapping"] = var_mapping

                text = f"✅ {added} variável(is) adicionada(s) à coleção '{coll_name}'"
                text += f"\nVariáveis: {', '.join(var_names)}"

                if combined_parts:
                    text += f"\n\n🔍 Índice combinado atualizado: {len(combined_text):,} chars indexados"
                    text += f"\n   Variáveis no índice: {len(all_vars)}"

                return {"content": [{"type": "text", "text": text}]}

            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao adicionar à coleção: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_collection_list":
            try:
                persistence = get_persistence()
                collections = persistence.list_collections()

                text = fmt.format_collection_list(collections)
                return {"content": [{"type": "text", "text": text}]}

            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao listar coleções: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_collection_info":
            try:
                persistence = get_persistence()
                coll_name = arguments["name"]

                info = persistence.get_collection_info(coll_name)
                if not info:
                    return {
                        "content": [
                            {"type": "text", "text": f"Coleção '{coll_name}' não encontrada."}
                        ],
                        "isError": True
                    }

                text = fmt.format_collection_info(info)
                return {"content": [{"type": "text", "text": text}]}

            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao obter info da coleção: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_collection_rebuild":
            try:
                persistence = get_persistence()
                coll_name = arguments["name"]

                # Obter variáveis da coleção
                all_vars = persistence.get_collection_vars(coll_name)
                if not all_vars:
                    return {
                        "content": [
                            {"type": "text", "text": f"Coleção '{coll_name}' vazia ou não existe."}
                        ],
                        "isError": True
                    }

                # Concatenar todas as variáveis com separadores claros
                combined_parts = []
                var_mapping = {}  # Mapeia linha -> (var_name, linha_original)
                current_line = 1
                vars_included = 0

                for var_name in all_vars:
                    if var_name in repl.variables:
                        value = repl.variables[var_name]
                        if isinstance(value, str):
                            # Adicionar header identificador
                            header = f"\n{'='*60}\n=== VARIÁVEL: {var_name} ===\n{'='*60}\n"
                            combined_parts.append(header)

                            # Registrar mapeamento de linhas
                            header_lines = header.count('\n')
                            current_line += header_lines

                            # Adicionar conteúdo e mapear linhas
                            content_lines = value.split('\n')
                            for i, _ in enumerate(content_lines):
                                var_mapping[current_line + i] = (var_name, i + 1)

                            combined_parts.append(value)
                            current_line += len(content_lines)
                            vars_included += 1

                if not combined_parts:
                    return {
                        "content": [
                            {"type": "text", "text": f"Nenhuma variável de texto encontrada na coleção '{coll_name}'."}
                        ],
                        "isError": True
                    }

                combined_text = "\n".join(combined_parts)
                combined_var_name = f"_coll_{coll_name}_combined"

                # Salvar variável combinada no REPL
                repl.variables[combined_var_name] = combined_text

                # Forçar criação de índice (min_chars=0)
                from .indexer import create_index, set_index
                combined_index = create_index(combined_text, combined_var_name)
                set_index(combined_var_name, combined_index)

                # Salvar mapeamento como metadado
                repl.variables[f"_coll_{coll_name}_mapping"] = var_mapping

                stats = combined_index.get_stats()
                text = f"✅ Índice combinado da coleção '{coll_name}' reconstruído!"
                text += f"\n\n📊 Estatísticas:"
                text += f"\n   Variáveis incluídas: {vars_included}/{len(all_vars)}"
                text += f"\n   Tamanho total: {len(combined_text):,} caracteres"
                text += f"\n   Termos indexados: {stats['indexed_terms']}"
                text += f"\n   Total de ocorrências: {stats['total_occurrences']}"
                text += f"\n\n🔍 Agora use: rlm_search_collection(collection=\"{coll_name}\", terms=[...])"

                return {"content": [{"type": "text", "text": text}]}

            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao reconstruir índice: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_search_collection":
            try:
                persistence = get_persistence()
                coll_name = arguments["collection"]
                terms = arguments["terms"]
                limit = arguments.get("limit", 10)
                offset = arguments.get("offset", 0)

                # Obter variáveis da coleção
                var_names = persistence.get_collection_vars(coll_name)
                if not var_names:
                    return {
                        "content": [
                            {"type": "text", "text": f"Coleção '{coll_name}' vazia ou não existe."}
                        ],
                        "isError": True
                    }

                # === OPÇÃO C: Tentar usar índice combinado primeiro ===
                combined_var_name = f"_coll_{coll_name}_combined"
                combined_index = get_index(combined_var_name)
                mapping_var = f"_coll_{coll_name}_mapping"

                all_results = {}
                terms_via_index = []
                terms_via_fallback = []
                indexed_terms_count = 0

                if combined_index and mapping_var in repl.variables:
                    # Usar índice combinado + fallback híbrido
                    var_mapping = repl.variables[mapping_var]
                    index_stats = combined_index.get_stats()
                    indexed_terms_count = index_stats.get('indexed_terms', 0)
                    available_terms = set(combined_index.terms.keys())

                    # Separar termos indexados vs não-indexados
                    for term in terms:
                        if term.lower() in available_terms:
                            terms_via_index.append(term)
                        else:
                            terms_via_fallback.append(term)

                    # Buscar termos indexados via índice
                    if terms_via_index:
                        results = combined_index.search_multiple(terms_via_index, require_all=False)
                        if results:
                            for term, matches in results.items():
                                for m in matches:
                                    linha_combined = m['linha']
                                    if linha_combined in var_mapping:
                                        orig_var, orig_linha = var_mapping[linha_combined]
                                        if orig_var not in all_results:
                                            all_results[orig_var] = {}
                                        if term not in all_results[orig_var]:
                                            all_results[orig_var][term] = []
                                        all_results[orig_var][term].append({
                                            'linha': orig_linha,
                                            'contexto': m['contexto']
                                        })

                    # Buscar termos não-indexados via full-text
                    if terms_via_fallback and combined_var_name in repl.variables:
                        combined_text = repl.variables[combined_var_name]
                        for term in terms_via_fallback:
                            term_lower = term.lower()
                            for line_num, line in enumerate(combined_text.split('\n'), start=1):
                                if term_lower in line.lower():
                                    if line_num in var_mapping:
                                        orig_var, orig_linha = var_mapping[line_num]
                                        if orig_var not in all_results:
                                            all_results[orig_var] = {}
                                        if term not in all_results[orig_var]:
                                            all_results[orig_var][term] = []
                                        all_results[orig_var][term].append({
                                            'linha': orig_linha,
                                            'contexto': line.strip()
                                        })
                else:
                    # Fallback total: buscar em índices individuais ou full-text
                    terms_via_fallback = terms[:]
                    for var_name in var_names:
                        index = get_index(var_name)
                        if index:
                            results = index.search_multiple(terms, require_all=False)
                            if results:
                                all_results[var_name] = results
                                terms_via_fallback = []  # Encontrou no índice

                    # Se não encontrou em índices individuais, tenta full-text
                    if not all_results and combined_var_name in repl.variables and mapping_var in repl.variables:
                        combined_text = repl.variables[combined_var_name]
                        var_mapping = repl.variables[mapping_var]
                        for term in terms:
                            term_lower = term.lower()
                            for line_num, line in enumerate(combined_text.split('\n'), start=1):
                                if term_lower in line.lower():
                                    if line_num in var_mapping:
                                        orig_var, orig_linha = var_mapping[line_num]
                                        if orig_var not in all_results:
                                            all_results[orig_var] = {}
                                        if term not in all_results[orig_var]:
                                            all_results[orig_var][term] = []
                                        all_results[orig_var][term].append({
                                            'linha': orig_linha,
                                            'contexto': line.strip()
                                        })

                if not all_results:
                    # Nenhum resultado nem no índice nem no fallback
                    text = f"Nenhum resultado para {terms} na coleção '{coll_name}'\n"
                    text += f"\n💡 Dica: Verifique se os termos estão corretos ou use rlm_execute com Python para busca avançada"
                else:
                    lines = [f"🔍 Busca em '{coll_name}': {', '.join(terms)}", ""]

                    # Stats de busca híbrida
                    if terms_via_index and terms_via_fallback:
                        lines.append(f"📊 Busca híbrida: {len(terms_via_index)} via índice, {len(terms_via_fallback)} via full-text")
                        lines.append(f"   ✅ Indexados: {', '.join(terms_via_index)}")
                        lines.append(f"   🔄 Full-text: {', '.join(terms_via_fallback)}")
                        if indexed_terms_count:
                            lines.append(f"   ℹ️  {indexed_terms_count} termos disponíveis no índice")
                        lines.append("")
                    elif terms_via_fallback:
                        lines.append(f"🔄 Busca full-text ({len(terms_via_fallback)} termos não indexados)")
                        if indexed_terms_count:
                            lines.append(f"   ℹ️  {indexed_terms_count} termos disponíveis no índice")
                        lines.append("")
                    elif combined_index:
                        lines.append(f"✅ Usando índice combinado ({len(var_names)} vars, {indexed_terms_count} termos)")
                        lines.append("")

                    for var_name, results in all_results.items():
                        lines.append(f"📄 {var_name}:")
                        for term, matches in results.items():
                            total_term = len(matches)
                            paginated = matches[offset:offset + limit]
                            start_idx = offset + 1 if paginated else 0
                            end_idx = offset + len(paginated)
                            lines.append(f"  📌 '{term}' ({total_term} ocorrências, mostrando {start_idx}-{end_idx})")
                            for m in paginated:
                                lines.append(f"      L{m['linha']}: {m['contexto'][:60]}...")
                        lines.append("")

                    total_matches = sum(
                        sum(len(matches) for matches in results.values())
                        for results in all_results.values()
                    )
                    lines.append(f"📊 Total: {total_matches} ocorrências em {len(all_results)} documento(s)")
                    text = "\n".join(lines)

                return {"content": [{"type": "text", "text": text}]}

            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro na busca: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_task_status":
            task_id = arguments["task_id"]
            task_info = task_manager.get_status(task_id)
            if not task_info:
                return {
                    "content": [{"type": "text", "text": f"Task '{task_id}' não encontrada."}],
                    "isError": True,
                }

            # If task completed, return the original result directly
            if task_info.status == "completed" and task_info.result:
                result_content = task_info.result.get("content", [])
                meta = fmt.format_task_status(task_info)
                # Prepend task meta to the original result
                if result_content:
                    original_text = result_content[0].get("text", "")
                    return {"content": [{"type": "text", "text": original_text}]}
                return {"content": [{"type": "text", "text": meta}]}

            text = fmt.format_task_status(task_info)
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_task_list":
            status_filter = arguments.get("status")
            tasks = task_manager.list_tasks(status=status_filter)
            # Cleanup old tasks while we're at it
            task_manager.cleanup_completed()
            text = fmt.format_task_list(tasks)
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_task_cancel":
            task_id = arguments["task_id"]
            success = task_manager.cancel(task_id)
            text = fmt.format_task_cancel(task_id, success)
            if not success:
                return {"content": [{"type": "text", "text": text}], "isError": True}
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_pin_var":
            var_name = arguments["name"]
            pin = arguments.get("pin", True)

            if repl.pin_variable(var_name, pin):
                text = fmt.format_pin_response(var_name, pin)
            else:
                text = f"Variável '{var_name}' não encontrada."
                return {"content": [{"type": "text", "text": text}], "isError": True}
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_batch_load_s3":
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
            if len(keys_list) > BATCH_ASYNC_THRESHOLD_FILES or total_size_mb > BATCH_ASYNC_THRESHOLD_MB:
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

        elif name == "rlm_batch_upload_s3":
            # Rate limit check for uploads
            rate_id = client_id or "anonymous"
            rate_result = upload_rate_limiter.check(rate_id)
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
                upload_rate_limiter.record(rate_id)

                text = fmt.format_batch_upload_s3(fmt_results)
                return {"content": [{"type": "text", "text": text}]}

            # Large batch → async task
            if len(vars_list) > BATCH_ASYNC_THRESHOLD_FILES or total_size_mb > BATCH_ASYNC_THRESHOLD_MB:
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

        elif name == "rlm_search_code":
            var_name = arguments["var_name"]
            query = arguments.get("query")
            kind = arguments.get("kind")
            include_source = arguments.get("include_source", False)
            language_hint = arguments.get("language")

            if var_name not in repl.variables:
                return {
                    "content": [{"type": "text", "text": f"Erro: Variável '{var_name}' não encontrada no REPL."}],
                    "isError": True,
                }

            value = repl.variables[var_name]
            if not isinstance(value, str):
                return {
                    "content": [{"type": "text", "text": f"Erro: Variável '{var_name}' não é texto (tipo: {type(value).__name__})."}],
                    "isError": True,
                }

            # Check if we already have a parsed CodeStructure in metadata
            meta_key = f"_code_structure_{var_name}"
            structure = repl.variables.get(meta_key)

            if not structure or not isinstance(structure, code_parser.CodeStructure):
                # Parse on-the-fly
                lang = language_hint
                if not lang:
                    # Try to detect from variable metadata source
                    meta = repl.variable_metadata.get(var_name)
                    source_hint = meta.source if meta else ""
                    # Source may contain filename info like "s3:bucket/path/file.py" or "file:/data/file.py"
                    lang = code_parser.detect_language(source_hint or var_name, value)

                if not lang:
                    return {
                        "content": [{"type": "text", "text": f"Erro: Não foi possível detectar a linguagem de '{var_name}'. Especifique o parâmetro 'language'."}],
                        "isError": True,
                    }

                if not code_parser.is_available():
                    return {
                        "content": [{"type": "text", "text": "Erro: tree-sitter não está instalado no servidor."}],
                        "isError": True,
                    }

                structure = code_parser.parse(value, lang)
                if not structure:
                    return {
                        "content": [{"type": "text", "text": f"Erro: Falha ao parsear '{var_name}' como {lang}. Gramática pode não estar instalada."}],
                        "isError": True,
                    }

                # Cache the structure
                repl.variables[meta_key] = structure

            results = structure.search(
                query=query,
                kind=kind,
                include_source=include_source,
                source_code=value,
            )

            text = fmt.format_search_code(
                results, var_name, structure.language,
                query=query, kind=kind, total_symbols=len(structure.symbols),
            )
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_save_to_s3":
            # Rate limit check for uploads
            rate_id = client_id or "anonymous"
            rate_result = upload_rate_limiter.check(rate_id)
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
                upload_rate_limiter.record(rate_id)

                text = fmt.format_save_to_s3(var_name, type(value).__name__, save_fmt, result, key)
                return {"content": [{"type": "text", "text": text}]}

            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao salvar variável no S3: {e}"}
                    ],
                    "isError": True
                }

        else:
            return {
                "content": [
                    {"type": "text", "text": f"Tool desconhecida: {name}"}
                ],
                "isError": True
            }

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
async def sse_endpoint(request: Request, _: bool = Depends(verify_api_key)):
    """
    SSE endpoint para MCP.
    O cliente se conecta aqui para receber eventos do servidor.
    """
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    sse_sessions[session_id] = queue

    logger.info(f"Nova sessão SSE: {session_id}")

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
                    yield f"event: message\ndata: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Envia ping para manter conexão
                    yield ": ping\n\n"
                except asyncio.CancelledError:
                    break
        finally:
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
    _: bool = Depends(verify_api_key)
):
    """
    Endpoint para enviar mensagens MCP.
    Se session_id for fornecido, resposta vai via SSE.
    Caso contrário, resposta direta no POST.

    Rate limiting: 100 requests/minute per SSE session.
    """
    request_id = generate_request_id()
    start_time = time.time()
    is_error = False

    logger.info(f"Processing /message request", extra={"request_id": request_id, "session_id": session_id})

    # Rate limiting for SSE sessions
    if session_id and session_id in sse_sessions:
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

        logger.debug(f"MCP method: {mcp_request.method}", extra={"request_id": request_id})

        # Use session_id as client_id for rate limiting, fallback to client IP
        client_id = session_id if session_id else request.client.host if request.client else "anonymous"
        response = handle_mcp_request(mcp_request, client_id=client_id)

        if response is None:
            # Notificação, não precisa responder
            latency_ms = (time.time() - start_time) * 1000
            metrics_collector.record_request("/message", latency_ms, is_error=False)
            logger.debug(f"Notification processed", extra={"request_id": request_id, "latency_ms": latency_ms})
            return Response(status_code=202, headers={"X-Request-Id": request_id})

        response_dict = response.model_dump(exclude_none=True)

        # Check if response has error
        if response.error:
            is_error = True
            logger.warning(f"MCP error response: {response.error}", extra={"request_id": request_id})

        # Se tem sessão SSE, envia por lá
        if session_id and session_id in sse_sessions:
            await sse_sessions[session_id].put(response_dict)
            latency_ms = (time.time() - start_time) * 1000
            metrics_collector.record_request("/message", latency_ms, is_error=is_error)
            logger.debug(f"Response sent via SSE", extra={"request_id": request_id, "latency_ms": latency_ms})
            return Response(status_code=202, headers={"X-Request-Id": request_id})

        # Senão, responde diretamente
        latency_ms = (time.time() - start_time) * 1000
        metrics_collector.record_request("/message", latency_ms, is_error=is_error)
        logger.debug(f"Response sent directly", extra={"request_id": request_id, "latency_ms": latency_ms})
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
    _: bool = Depends(verify_api_key)
):
    """
    Endpoint direto para MCP (sem SSE).
    Útil para clientes que preferem request/response simples.
    """
    request_id = generate_request_id()
    start_time = time.time()
    is_error = False

    logger.info(f"Processing /mcp request", extra={"request_id": request_id})

    # Rate limiting by client IP
    client_id = request.client.host if request.client else "anonymous"
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

        logger.debug(f"MCP method: {mcp_request.method}", extra={"request_id": request_id})

        response = handle_mcp_request(mcp_request, client_id=client_id)

        if response is None:
            latency_ms = (time.time() - start_time) * 1000
            metrics_collector.record_request("/mcp", latency_ms, is_error=False)
            logger.debug(f"Notification processed", extra={"request_id": request_id, "latency_ms": latency_ms})
            return Response(status_code=202, headers={"X-Request-Id": request_id})

        # Check if response has error
        if response.error:
            is_error = True
            logger.warning(f"MCP error response: {response.error}", extra={"request_id": request_id})

        latency_ms = (time.time() - start_time) * 1000
        metrics_collector.record_request("/mcp", latency_ms, is_error=is_error)
        logger.debug(f"Response sent", extra={"request_id": request_id, "latency_ms": latency_ms})
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
