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
from .indexer import get_index, set_index, TextIndex, auto_index_if_large
from .rate_limiter import SlidingWindowRateLimiter, RateLimitResult
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

# Rate limiting configuration
SSE_RATE_LIMIT_REQUESTS = int(os.getenv("RLM_SSE_RATE_LIMIT", "100"))
SSE_RATE_LIMIT_WINDOW = int(os.getenv("RLM_SSE_RATE_WINDOW", "60"))  # seconds
UPLOAD_RATE_LIMIT_REQUESTS = int(os.getenv("RLM_UPLOAD_RATE_LIMIT", "10"))
UPLOAD_RATE_LIMIT_WINDOW = int(os.getenv("RLM_UPLOAD_RATE_WINDOW", "60"))  # seconds

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
    cleanup_threshold_percent=CLEANUP_THRESHOLD,
    cleanup_target_percent=CLEANUP_TARGET
)

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
                    # Restaurar índice se existir
                    index_data = persistence.load_index(name)
                    if index_data:
                        set_index(name, TextIndex.from_dict(index_data))
                    logger.info(f"  Restaurado: {name} ({var_info['type']})")
            logger.info("Variáveis restauradas com sucesso")
    except Exception as e:
        logger.warning(f"Erro ao restaurar variáveis (pode ser primeira execução): {e}")

    yield
    logger.info("RLM MCP Server encerrando")


app = FastAPI(
    title="RLM MCP Server",
    description="Recursive Language Model via MCP over HTTP/SSE",
    version="0.2.0",
    lifespan=lifespan
)

# CORS para permitir conexões do Claude Code
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restrinja isso
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
    return [
        {
            "name": "rlm_execute",
            "description": """Executa código Python no REPL persistente.

As variáveis criadas persistem entre execuções. Use print() para retornar dados.

IMPORTANTE: O código roda em sandbox seguro:
- Imports permitidos: re, json, math, collections, datetime, csv, etc.
- Imports bloqueados: os, subprocess, socket, requests, etc.

=== PADRÕES DE USO AVANÇADO (RLM) ===

O poder do RLM está em escrever código SOFISTICADO para analisar dados massivos.
NÃO use apenas regex simples. Use estas estratégias:

1. ÍNDICE SEMÂNTICO - mapear conceitos para localização:
   ```python
   indice = {conceito: [] for conceito in ['medo', 'trabalho', 'pai']}
   for i, linha in enumerate(texto.split('\\n')):
       for c in indice:
           if c in linha.lower():
               indice[c].append({'linha': i, 'ctx': linha[:100]})
   ```

2. ANÁLISE CRUZADA - buscar múltiplos critérios:
   ```python
   def analise_diferencial(sintomas, texto):
       scores = defaultdict(int)
       for sintoma in sintomas:
           if sintoma in secao.lower():
               scores[remedio] += 1
       return sorted(scores.items(), key=lambda x: -x[1])
   ```

3. ESTRUTURA DOCUMENTAL - mapear seções/capítulos:
   ```python
   secoes = re.findall(r'^#+ (.+)$', texto, re.MULTILINE)
   ```

4. FUNÇÕES REUTILIZÁVEIS - definir helpers que persistem entre chamadas

=== FUNÇÕES AUXILIARES PRÉ-DEFINIDAS ===

O REPL já inclui estas funções prontas para uso:

1. buscar(texto, termo) -> list[dict]
   Busca um termo no texto (case-insensitive).
   Retorna: [{'posicao': int, 'linha': int, 'contexto': str}]
   Exemplo: buscar(meu_texto, "erro")

2. contar(texto, termo) -> dict
   Conta ocorrências de um termo (case-insensitive).
   Retorna: {'total': int, 'por_linha': {linha: count}}
   Exemplo: contar(meu_texto, "warning")

3. extrair_secao(texto, inicio, fim) -> list[dict]
   Extrai seções entre marcadores (case-insensitive).
   Retorna: [{'conteudo': str, 'posicao_inicio': int, 'posicao_fim': int, 'linha_inicio': int, 'linha_fim': int}]
   Exemplo: extrair_secao(doc, "## Intro", "## Conclusão")

4. resumir_tamanho(bytes) -> str
   Converte bytes para formato humanizado.
   Retorna: string como "1.5 MB", "256 KB"
   Exemplo: resumir_tamanho(1048576) -> "1.0 MB"

LEMBRE-SE: Você tem acesso a MILHÕES de caracteres. Use Python para
fazer análises que RAG não consegue - cruzamento, agregação, lógica condicional.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Código Python para executar"
                    }
                },
                "required": ["code"]
            }
        },
        {
            "name": "rlm_load_data",
            "description": """Carrega dados diretamente em uma variável do REPL.

Tipos suportados:
- "text": String simples
- "json": Parse JSON para dict/list
- "lines": Split por \\n para lista
- "csv": Parse CSV para lista de dicts""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome da variável"
                    },
                    "data": {
                        "type": "string",
                        "description": "Dados para carregar"
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["text", "json", "lines", "csv"],
                        "default": "text"
                    }
                },
                "required": ["name", "data"]
            }
        },
        {
            "name": "rlm_load_file",
            "description": """Carrega arquivo do servidor em uma variável.

O arquivo deve estar no diretório /data do container.

Tipos suportados:
- text: String simples
- json: Parse JSON para dict/list
- lines: Split por \\n para lista
- csv: Parse CSV para lista de dicts
- pdf: Extrai texto de PDF (auto-detecta método)
- pdf_ocr: Força OCR para PDFs escaneados (requer MISTRAL_API_KEY)""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome da variável"
                    },
                    "path": {
                        "type": "string",
                        "description": "Caminho do arquivo (deve começar com /data/)"
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["text", "json", "lines", "csv", "pdf", "pdf_ocr"],
                        "default": "text"
                    }
                },
                "required": ["name", "path"]
            }
        },
        {
            "name": "rlm_list_vars",
            "description": "Lista todas as variáveis no REPL com metadados (nome, tipo, tamanho, preview). Suporta paginação via offset/limit.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Máximo de variáveis a retornar"
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Número de variáveis a pular (para paginação)"
                    }
                }
            }
        },
        {
            "name": "rlm_var_info",
            "description": "Retorna informações detalhadas de uma variável específica.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome da variável"
                    }
                },
                "required": ["name"]
            }
        },
        {
            "name": "rlm_clear",
            "description": "Limpa variáveis do REPL. Use 'name' para uma específica ou 'all=true' para todas.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "all": {"type": "boolean", "default": False}
                }
            }
        },
        {
            "name": "rlm_memory",
            "description": "Retorna estatísticas de uso de memória do REPL.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "rlm_load_s3",
            "description": """Carrega arquivo do Minio/S3 diretamente em uma variável.

O arquivo é baixado direto do Minio para o servidor RLM,
sem passar pelo contexto do Claude Code. Ideal para arquivos grandes.

Tipos suportados:
- text: String simples
- json: Parse JSON para dict/list
- lines: Split por \\n para lista
- csv: Parse CSV para lista de dicts
- pdf: Extrai texto de PDF (auto-detecta método)
- pdf_ocr: Força OCR para PDFs escaneados (requer MISTRAL_API_KEY)

Opções:
- skip_if_exists: Se True (padrão), pula download se variável já existe no REPL

Exemplo: rlm_load_s3(key="pdfs/doc.pdf", name="doc", data_type="pdf")""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Caminho/chave do objeto no bucket"
                    },
                    "name": {
                        "type": "string",
                        "description": "Nome da variável no REPL"
                    },
                    "bucket": {
                        "type": "string",
                        "default": "claude-code",
                        "description": "Nome do bucket (padrão: claude-code)"
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["text", "json", "lines", "csv", "pdf", "pdf_ocr"],
                        "default": "text",
                        "description": "Tipo de parsing dos dados"
                    },
                    "skip_if_exists": {
                        "type": "boolean",
                        "default": True,
                        "description": "Se True, pula download se variável já existe (padrão: True)"
                    }
                },
                "required": ["key", "name"]
            }
        },
        {
            "name": "rlm_list_buckets",
            "description": """Lista buckets disponíveis no Minio.

Use para descobrir quais buckets existem antes de carregar arquivos.""",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "rlm_list_s3",
            "description": """Lista objetos em um bucket do Minio.

Retorna nome, tamanho e data de modificação dos arquivos.
Suporta paginação via offset e limit.

Exemplo: rlm_list_s3() para listar bucket padrão, ou rlm_list_s3(prefix="logs/") para filtrar""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bucket": {
                        "type": "string",
                        "default": "claude-code",
                        "description": "Nome do bucket (padrão: claude-code)"
                    },
                    "prefix": {
                        "type": "string",
                        "default": "",
                        "description": "Prefixo para filtrar (opcional)"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Máximo de objetos a retornar (padrão: 50)"
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Número de objetos a pular para paginação (padrão: 0)"
                    }
                }
            }
        },
        {
            "name": "rlm_upload_url",
            "description": """Faz upload de arquivo de uma URL para o Minio/S3.

O servidor RLM baixa o arquivo diretamente da URL e envia para o Minio,
sem passar pelo contexto do Claude Code. Ideal para arquivos grandes.

Exemplo: rlm_upload_url(url="https://example.com/data.csv", key="data/file.csv")""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL do arquivo para baixar"
                    },
                    "key": {
                        "type": "string",
                        "description": "Caminho/chave do objeto no bucket"
                    },
                    "bucket": {
                        "type": "string",
                        "default": "claude-code",
                        "description": "Nome do bucket (padrão: claude-code)"
                    }
                },
                "required": ["url", "key"]
            }
        },
        {
            "name": "rlm_process_pdf",
            "description": """Processa PDF do Minio e salva texto extraído de volta no bucket.

WORKFLOW EM DUAS ETAPAS para PDFs grandes:
1. rlm_process_pdf() → Extrai texto e salva como .txt no bucket (esta ferramenta)
2. rlm_load_s3() com o .txt → Carrega texto rápido para análise

O PDF é processado no servidor e o texto é salvo no mesmo bucket.
NÃO carrega em variável (evita timeout em PDFs grandes).

Métodos de extração:
- auto: Usa pdfplumber primeiro, fallback para OCR se pouco texto
- pdfplumber: Força pdfplumber (rápido, para PDFs com texto selecionável)
- ocr: Força Mistral OCR (para PDFs escaneados, requer MISTRAL_API_KEY)

Exemplo: rlm_process_pdf(key="pdfs/livro.pdf") → salva pdfs/livro.txt""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Caminho do PDF no bucket (ex: pdfs/documento.pdf)"
                    },
                    "bucket": {
                        "type": "string",
                        "default": "claude-code",
                        "description": "Nome do bucket (padrão: claude-code)"
                    },
                    "output_key": {
                        "type": "string",
                        "description": "Caminho para salvar o .txt (padrão: mesmo path com extensão .txt)"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["auto", "pdfplumber", "ocr"],
                        "default": "auto",
                        "description": "Método de extração (padrão: auto)"
                    }
                },
                "required": ["key"]
            }
        },
        {
            "name": "rlm_search_index",
            "description": """Busca termos no índice semântico de uma variável.

O índice é criado automaticamente ao carregar textos grandes (100k+ chars).
Permite busca rápida sem varrer o texto todo.

Modos de busca:
- termo único: retorna linhas onde o termo aparece
- múltiplos termos: retorna linhas com qualquer um dos termos
- require_all=true: retorna apenas linhas com TODOS os termos

Exemplo: rlm_search_index(var_name="scholten1", terms=["medo", "fracasso"])""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "var_name": {
                        "type": "string",
                        "description": "Nome da variável indexada"
                    },
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de termos para buscar"
                    },
                    "require_all": {
                        "type": "boolean",
                        "default": False,
                        "description": "Se True, retorna apenas linhas com TODOS os termos"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Máximo de resultados por termo"
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Número de resultados a pular (para paginação)"
                    }
                },
                "required": ["var_name", "terms"]
            }
        },
        {
            "name": "rlm_persistence_stats",
            "description": """Retorna estatísticas de persistência (variáveis salvas, índices, etc).

Mostra quais variáveis estão persistidas e sobreviverão ao restart do servidor.""",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "rlm_collection_create",
            "description": """Cria uma nova coleção para agrupar variáveis por assunto.

Coleções permitem organizar variáveis relacionadas (ex: homeopatia, nutrição, fitoterapia)
e fazer buscas em todas de uma vez.

Exemplo: rlm_collection_create(name="homeopatia", description="Materiais de homeopatia unicista")""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome da coleção (único)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Descrição opcional da coleção"
                    }
                },
                "required": ["name"]
            }
        },
        {
            "name": "rlm_collection_add",
            "description": """Adiciona variáveis a uma coleção existente.

A coleção é criada automaticamente se não existir.

Exemplo: rlm_collection_add(collection="homeopatia", vars=["scholten1", "scholten2", "kent"])""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Nome da coleção"
                    },
                    "vars": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de nomes de variáveis para adicionar"
                    }
                },
                "required": ["collection", "vars"]
            }
        },
        {
            "name": "rlm_collection_list",
            "description": """Lista todas as coleções existentes com contagem de variáveis.""",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "rlm_collection_info",
            "description": """Retorna informações detalhadas de uma coleção específica.

Mostra todas as variáveis na coleção com seus tamanhos.""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome da coleção"
                    }
                },
                "required": ["name"]
            }
        },
        {
            "name": "rlm_collection_rebuild",
            "description": """Reconstrói o índice combinado de uma coleção.

Use após atualizar o servidor ou quando a busca na coleção não funcionar.
Concatena todas as variáveis e cria índice semântico unificado.

Exemplo: rlm_collection_rebuild(name="injetaveis")""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome da coleção"
                    }
                },
                "required": ["name"]
            }
        },
        {
            "name": "rlm_search_collection",
            "description": """Busca termos em TODAS as variáveis de uma coleção.

Busca unificada que varre todos os documentos da coleção e retorna
resultados agrupados por documento.

Exemplo: rlm_search_collection(collection="homeopatia", terms=["medo", "ansiedade"])""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Nome da coleção"
                    },
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Lista de termos para buscar"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Máximo de resultados por documento/termo"
                    },
                    "offset": {
                        "type": "integer",
                        "default": 0,
                        "description": "Número de resultados a pular (para paginação)"
                    }
                },
                "required": ["collection", "terms"]
            }
        }
    ]


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
                    {"type": "text", "text": format_execution_result(result)}
                ]
            }

        elif name == "rlm_load_data":
            var_name = arguments["name"]
            data = arguments["data"]
            data_type = arguments.get("data_type", "text")

            result = repl.load_data(name=var_name, data=data, data_type=data_type)

            # Auto-persistência e indexação
            persist_msg = ""
            index_msg = ""
            persist_error = ""
            try:
                # Persistir variável
                persistence = get_persistence()
                value = repl.variables.get(var_name)
                if value is not None:
                    persistence.save_variable(var_name, value)
                    persist_msg = "💾 Persistido"

                    # Indexar se for texto grande
                    if isinstance(value, str) and len(value) >= 100000:
                        idx = auto_index_if_large(value, var_name)
                        if idx:
                            set_index(var_name, idx)
                            persistence.save_index(var_name, idx.to_dict())
                            index_msg = f"📑 Indexado ({idx.get_stats()['indexed_terms']} termos)"
            except Exception as e:
                logger.warning(f"Erro ao persistir/indexar {var_name}: {e}")
                persist_error = f"\n⚠️ Erro de persistência: {e}"

            output = format_execution_result(result)
            extras = f"\n\n{persist_msg} {index_msg}".strip() if (persist_msg or index_msg) else ""
            if SHOW_PERSISTENCE_ERRORS:
                extras += persist_error
            if extras:
                output += extras

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
                    result = repl.load_data(
                        name=arguments["name"],
                        data=data,
                        data_type="text"
                    )

                    text = f"""✅ PDF extraído com sucesso:
Arquivo: {path}
Método: {pdf_result.method}
Páginas: {pdf_result.pages}
Caracteres: {len(data):,}
Variável: {arguments["name"]}

{format_execution_result(result)}"""
                    return {"content": [{"type": "text", "text": text}]}

                # Regular file handling
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    data = f.read()

                result = repl.load_data(
                    name=arguments["name"],
                    data=data,
                    data_type=data_type
                )
                return {
                    "content": [
                        {"type": "text", "text": format_execution_result(result)}
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
            if not vars_list:
                text = "Nenhuma variável no REPL."
            else:
                total = len(vars_list)
                paginated = vars_list[offset:offset + limit]
                start_idx = offset + 1 if paginated else 0
                end_idx = offset + len(paginated)
                lines = [f"Variáveis no REPL ({total} total, mostrando {start_idx}-{end_idx}):", ""]
                for v in paginated:
                    lines.append(f"  {v.name}: {v.type_name} ({v.size_human})")
                    lines.append(f"    Preview: {v.preview[:100]}...")
                    lines.append("")
                text = "\n".join(lines)
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_var_info":
            info = repl.get_variable_info(arguments["name"])
            if not info:
                text = f"Variável '{arguments['name']}' não encontrada."
            else:
                text = f"""Variável: {info.name}
Tipo: {info.type_name}
Tamanho: {info.size_human} ({info.size_bytes} bytes)
Criada em: {info.created_at.isoformat()}
Último acesso: {info.last_accessed.isoformat()}

Preview:
{info.preview}"""
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
            text = f"""Uso de Memória do REPL:
Total: {mem['total_human']}
Variáveis: {mem['variable_count']}
Limite: {mem['max_allowed_mb']} MB
Uso: {mem['usage_percent']:.1f}%"""
            return {"content": [{"type": "text", "text": text}]}

        elif name == "rlm_load_s3":
            s3 = get_s3_client()
            if not s3.is_configured():
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Minio não configurado. Configure MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY."}
                    ],
                    "isError": True
                }

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

                        # Auto-persistência e indexação
                        persist_msg = ""
                        index_msg = ""
                        persist_error = ""
                        try:
                            persistence = get_persistence()
                            value = repl.variables.get(var_name)
                            if value is not None:
                                persistence.save_variable(var_name, value)
                                persist_msg = "💾 Persistido"

                                if isinstance(value, str) and len(value) >= 100000:
                                    idx = auto_index_if_large(value, var_name)
                                    if idx:
                                        set_index(var_name, idx)
                                        persistence.save_index(var_name, idx.to_dict())
                                        index_msg = f"📑 Indexado ({idx.get_stats()['indexed_terms']} termos)"
                        except Exception as e:
                            logger.warning(f"Erro ao persistir/indexar {var_name}: {e}")
                            persist_error = f"\n⚠️ Erro de persistência: {e}"

                        extras = f"\n{persist_msg} {index_msg}".strip() if (persist_msg or index_msg) else ""
                        if SHOW_PERSISTENCE_ERRORS:
                            extras += persist_error

                        text = f"""✅ PDF extraído do Minio:
Bucket: {bucket}
Objeto: {key}
Tamanho original: {info['size_human']}
Método: {pdf_result.method}
Páginas: {pdf_result.pages}
Caracteres extraídos: {len(data):,}
Variável: {var_name}{extras}

{format_execution_result(result)}"""
                        return {"content": [{"type": "text", "text": text}]}
                    finally:
                        import os
                        os.unlink(tmp_path)

                # Regular file handling
                data = s3.get_object_text(bucket, key)
                result = repl.load_data(name=var_name, data=data, data_type=data_type)

                # Auto-persistência e indexação
                persist_msg = ""
                index_msg = ""
                persist_error = ""
                try:
                    persistence = get_persistence()
                    value = repl.variables.get(var_name)
                    if value is not None:
                        persistence.save_variable(var_name, value)
                        persist_msg = "💾 Persistido"

                        if isinstance(value, str) and len(value) >= 100000:
                            idx = auto_index_if_large(value, var_name)
                            if idx:
                                set_index(var_name, idx)
                                persistence.save_index(var_name, idx.to_dict())
                                index_msg = f"📑 Indexado ({idx.get_stats()['indexed_terms']} termos)"
                except Exception as e:
                    logger.warning(f"Erro ao persistir/indexar {var_name}: {e}")
                    persist_error = f"\n⚠️ Erro de persistência: {e}"

                extras = f"\n{persist_msg} {index_msg}".strip() if (persist_msg or index_msg) else ""
                if SHOW_PERSISTENCE_ERRORS:
                    extras += persist_error

                text = f"""✅ Carregado do Minio:
Bucket: {bucket}
Objeto: {key}
Tamanho: {info['size_human']}
Variável: {var_name} (tipo: {data_type}){extras}

{format_execution_result(result)}"""
                return {"content": [{"type": "text", "text": text}]}

            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao carregar do Minio: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_list_buckets":
            s3 = get_s3_client()
            if not s3.is_configured():
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Minio não configurado."}
                    ],
                    "isError": True
                }

            try:
                buckets = s3.list_buckets()
                if not buckets:
                    text = "Nenhum bucket encontrado."
                else:
                    text = "Buckets disponíveis:\n" + "\n".join(f"  - {b}" for b in buckets)
                return {"content": [{"type": "text", "text": text}]}
            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao listar buckets: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_list_s3":
            s3 = get_s3_client()
            if not s3.is_configured():
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Minio não configurado."}
                    ],
                    "isError": True
                }

            bucket = arguments.get("bucket", "claude-code")
            prefix = arguments.get("prefix", "")
            limit = arguments.get("limit", 50)
            offset = arguments.get("offset", 0)

            try:
                objects = s3.list_objects(bucket, prefix)
                total = len(objects)
                if not objects:
                    text = f"Nenhum objeto encontrado em {bucket}/{prefix}"
                else:
                    # Apply pagination
                    paginated = objects[offset:offset + limit]
                    start_idx = offset + 1 if paginated else 0
                    end_idx = offset + len(paginated)

                    lines = [f"Objetos em {bucket}/{prefix} ({total} total, mostrando {start_idx}-{end_idx}):", ""]
                    for obj in paginated:
                        lines.append(f"  {obj['name']} ({obj['size_human']})")
                    text = "\n".join(lines)
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

            s3 = get_s3_client()
            if not s3.is_configured():
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Minio não configurado."}
                    ],
                    "isError": True
                }

            url = arguments["url"]
            bucket = arguments.get("bucket", "claude-code")
            key = arguments["key"]

            try:
                result = s3.upload_from_url(url, bucket, key)
                # Record successful upload for rate limiting
                upload_rate_limiter.record(rate_id)
                text = f"""✅ Upload concluído:
URL: {url}
Bucket: {result['bucket']}
Objeto: {result['key']}
Tamanho: {result['size_human']}"""
                return {"content": [{"type": "text", "text": text}]}
            except Exception as e:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro ao fazer upload de URL: {e}"}
                    ],
                    "isError": True
                }

        elif name == "rlm_process_pdf":
            s3 = get_s3_client()
            if not s3.is_configured():
                return {
                    "content": [
                        {"type": "text", "text": "Erro: Minio não configurado."}
                    ],
                    "isError": True
                }

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

                logger.info(f"Processando PDF: {bucket}/{key} ({info['size_human']})")

                # Baixar PDF para arquivo temporário
                import tempfile
                pdf_bytes = s3.get_object(bucket, key)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name

                try:
                    # Extrair texto
                    pdf_result = extract_pdf(tmp_path, method=method)

                    if not pdf_result.success:
                        return {
                            "content": [
                                {"type": "text", "text": f"Erro ao extrair PDF: {pdf_result.error}"}
                            ],
                            "isError": True
                        }

                    # Salvar texto no bucket
                    upload_result = s3.put_object_text(bucket, output_key, pdf_result.text)

                    text = f"""✅ PDF processado com sucesso!

📄 Origem:
  Bucket: {bucket}
  Arquivo: {key}
  Tamanho: {info['size_human']}

📝 Extração:
  Método: {pdf_result.method}
  Páginas: {pdf_result.pages}
  Caracteres: {len(pdf_result.text):,}

💾 Texto salvo:
  Bucket: {bucket}
  Arquivo: {output_key}
  Tamanho: {upload_result['size_human']}

Próximo passo: rlm_load_s3(key="{output_key}", name="texto", data_type="text")"""
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

            # Verificar se tem índice
            index = get_index(var_name)
            if not index:
                return {
                    "content": [
                        {"type": "text", "text": f"Erro: Variável '{var_name}' não possui índice. Indexação automática ocorre para textos >= 100k chars."}
                    ],
                    "isError": True
                }

            try:
                if require_all:
                    results = index.search_multiple(terms, require_all=True)
                    if not results:
                        text = f"Nenhuma linha encontrada com TODOS os termos: {', '.join(terms)}"
                    else:
                        total_results = len(results)
                        paginated = sorted(results.items())[offset:offset + limit]
                        lines = [f"Linhas com todos os termos ({total_results} encontradas, mostrando {offset + 1}-{offset + len(paginated)}):", ""]
                        for linha, found_terms in paginated:
                            lines.append(f"  Linha {linha}: {found_terms}")
                        text = "\n".join(lines)
                else:
                    results = index.search_multiple(terms, require_all=False)
                    if not results:
                        text = f"Nenhum resultado para: {', '.join(terms)}"
                    else:
                        lines = ["Resultados da busca:", ""]
                        for term, matches in results.items():
                            total_matches = len(matches)
                            paginated_matches = matches[offset:offset + limit]
                            showing = f"{offset + 1}-{offset + len(paginated_matches)}" if paginated_matches else "0"
                            lines.append(f"📌 '{term}' ({total_matches} ocorrências, mostrando {showing}):")
                            for m in paginated_matches:
                                lines.append(f"    Linha {m['linha']}: {m['contexto'][:80]}...")
                            lines.append("")
                        text = "\n".join(lines)

                # Adicionar stats do índice
                stats = index.get_stats()
                text += f"\n\n📊 Índice: {stats['indexed_terms']} termos, {stats['total_occurrences']} ocorrências totais"

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

                lines = ["📦 Estatísticas de Persistência", ""]
                lines.append(f"Variáveis salvas: {stats.get('variables_count', 0)}")
                lines.append(f"Tamanho total: {stats.get('variables_total_size', 0):,} bytes")
                lines.append(f"Índices salvos: {stats.get('indices_count', 0)}")
                lines.append(f"Termos indexados: {stats.get('total_indexed_terms', 0):,}")
                lines.append(f"Arquivo DB: {stats.get('db_path', 'N/A')}")
                lines.append(f"Tamanho DB: {stats.get('db_file_size', 0):,} bytes")

                if saved_vars:
                    lines.append("")
                    lines.append("Variáveis persistidas:")
                    for v in saved_vars:
                        lines.append(f"  - {v['name']} ({v['type']}, {v['size_bytes']:,} bytes)")
                        lines.append(f"    Atualizado: {v['updated_at']}")

                return {"content": [{"type": "text", "text": "\n".join(lines)}]}

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

                persistence.create_collection(coll_name, description)

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

                if not collections:
                    text = "Nenhuma coleção criada ainda."
                else:
                    lines = ["📚 Coleções disponíveis:", ""]
                    for c in collections:
                        lines.append(f"  📁 {c['name']} ({c['var_count']} variáveis)")
                        if c['description']:
                            lines.append(f"     {c['description']}")
                    text = "\n".join(lines)

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

                lines = [f"📁 Coleção: {info['name']}", ""]
                if info['description']:
                    lines.append(f"Descrição: {info['description']}")
                lines.append(f"Criada em: {info['created_at']}")
                lines.append(f"Total: {info['var_count']} variáveis, {info['total_size']:,} bytes")
                lines.append("")
                lines.append("Variáveis:")
                for v in info['variables']:
                    lines.append(f"  - {v['name']} ({v['type']}, {v['size_bytes']:,} bytes)")

                return {"content": [{"type": "text", "text": "\n".join(lines)}]}

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

                if combined_index and mapping_var in repl.variables:
                    # Usar índice combinado (FUNCIONA SEMPRE!)
                    var_mapping = repl.variables[mapping_var]
                    results = combined_index.search_multiple(terms, require_all=False)

                    if results:
                        # Agrupar resultados por variável original
                        for term, matches in results.items():
                            for m in matches:
                                linha_combined = m['linha']
                                # Encontrar variável original usando mapeamento
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
                else:
                    # Fallback: buscar em índices individuais (vars > 100k)
                    for var_name in var_names:
                        index = get_index(var_name)
                        if index:
                            results = index.search_multiple(terms, require_all=False)
                            if results:
                                all_results[var_name] = results

                if not all_results:
                    # Última tentativa: busca direta via Python
                    text = f"Nenhum resultado indexado para {terms} na coleção '{coll_name}'\n"
                    text += f"\n💡 Dica: Use rlm_execute com Python para busca direta:\n"
                    text += f"   for var in {var_names[:3]}...:\n"
                    text += f"       print(buscar(var, '{terms[0]}'))"
                else:
                    lines = [f"🔍 Busca em '{coll_name}': {', '.join(terms)}", ""]
                    used_combined = combined_index is not None
                    if used_combined:
                        lines.append(f"✅ Usando índice combinado ({len(var_names)} vars)")
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


def format_execution_result(result: ExecutionResult) -> str:
    """Formata resultado de execução"""
    parts = []

    if result.stdout:
        parts.append(f"=== OUTPUT ===\n{result.stdout}")

    if result.stderr:
        parts.append(f"=== ERRORS ===\n{result.stderr}")

    if result.variables_changed:
        parts.append(f"=== VARIÁVEIS ALTERADAS ===\n{', '.join(result.variables_changed)}")

    parts.append(f"\n[Execução: {result.execution_time_ms:.1f}ms | Status: {'OK' if result.success else 'ERRO'}]")

    return "\n".join(parts) if parts else "Execução concluída sem output."


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

    try:
        body = await request.json()
        mcp_request = MCPRequest(**body)

        logger.debug(f"MCP method: {mcp_request.method}", extra={"request_id": request_id})

        # Use client IP for rate limiting
        client_id = request.client.host if request.client else "anonymous"
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
