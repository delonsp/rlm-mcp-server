"""
REPL Python Persistente com Sandbox de Segurança

Mantém variáveis em memória entre execuções, permitindo
manipulação de dados massivos sem carregar no contexto do LLM.

Implementa o padrão RLM (Recursive Language Models) do paper MIT CSAIL,
permitindo sub-chamadas a LLMs de dentro do código Python.
"""

import ast
import os
import sys
import traceback
import signal
import threading
from io import StringIO
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

from .llm_client import LLMClient

logger = logging.getLogger("rlm-mcp.repl")


# Imports permitidos no sandbox
ALLOWED_IMPORTS = {
    # Builtins seguros
    "re", "json", "math", "statistics", "collections", "itertools",
    "textwrap", "unicodedata",
    # REMOVIDOS por segurança (vetores de sandbox escape):
    #   operator  → operator.attrgetter("__class__...") burla o bloqueio de dunder
    #   functools → functools.partial(getattr, o, "__class__") burla o bloqueio de getattr
    #   string    → string.Formatter().vformat("{0.__class__}", ...) idem str.format
    # Data/Time
    "datetime", "time", "calendar",
    # Estruturas de dados
    "dataclasses", "typing", "enum",
    # Texto e parsing
    "csv", "html", "xml.etree.ElementTree",
    # Hashing (read-only)
    "hashlib", "base64",
    # REMOVIDOS por segurança (task #13 do plano de isolamento do sandbox):
    #   gzip/zipfile/tarfile fazem I/O de arquivo REAL — abrem paths usando o
    #   builtins.open verdadeiro (não o do namespace), burlando o bloqueio de
    #   open(). Mesmo sob o isolamento por subprocesso (B1) o filho enxerga
    #   /persist e /data, então estes módulos permitiriam ler/gravar arquivos
    #   sem nem precisar de um escape. Descompressão de corpora é server-side.
}

# Imports bloqueados (perigosos)
BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests", "httpx",
    "pickle", "shelve", "sqlite3",
    "multiprocessing", "threading", "concurrent",
    "ctypes", "cffi",
    "importlib", "builtins", "__builtins__",
}

# Funções bloqueadas
BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__",
    "open", "input", "breakpoint", "help",
    "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
    "exit", "quit",
}


# ============================================================================
# Helper Functions para o REPL
# Funções pré-definidas disponíveis no namespace de execução
# ============================================================================

# Nomes de helper functions (excluídos do namespace de usuário)
HELPER_FUNCTION_NAMES = {
    'buscar',
    'contar',
    'extrair_secao',
    'resumir_tamanho',
}

# Funções internas do REPL (excluídos da listagem de variáveis do usuário)
INTERNAL_FUNCTION_NAMES = HELPER_FUNCTION_NAMES | {
    'llm_query',
    'llm_stats',
    'llm_reset_counter',
}

def _buscar(texto: str, termo: str) -> list[dict]:
    """
    Busca um termo em um texto e retorna todas as ocorrências com contexto.

    Args:
        texto: O texto onde buscar
        termo: O termo a ser buscado (case-insensitive)

    Returns:
        Lista de dicts com: posicao, linha, contexto (50 chars antes e depois)

    Example:
        >>> buscar(meu_texto, "erro")
        [{'posicao': 150, 'linha': 5, 'contexto': '...texto antes erro texto depois...'}]
    """
    import re

    if not texto or not termo:
        return []

    resultados = []
    texto_lower = texto.lower()
    termo_lower = termo.lower()

    # Encontra todas as ocorrências
    start = 0
    while True:
        pos = texto_lower.find(termo_lower, start)
        if pos == -1:
            break

        # Calcula número da linha
        linha = texto[:pos].count('\n') + 1

        # Extrai contexto (50 chars antes e depois)
        ctx_start = max(0, pos - 50)
        ctx_end = min(len(texto), pos + len(termo) + 50)
        contexto = texto[ctx_start:ctx_end]

        # Adiciona reticências se truncado
        if ctx_start > 0:
            contexto = "..." + contexto
        if ctx_end < len(texto):
            contexto = contexto + "..."

        resultados.append({
            'posicao': pos,
            'linha': linha,
            'contexto': contexto.replace('\n', ' ')  # Remove quebras de linha
        })

        start = pos + 1

    return resultados


def _contar(texto: str, termo: str) -> dict:
    """
    Conta ocorrências de um termo em um texto.

    Args:
        texto: O texto onde contar
        termo: O termo a ser contado (case-insensitive)

    Returns:
        Dict com: total (contagem total), por_linha (dict de linha -> contagem)

    Example:
        >>> contar(meu_texto, "erro")
        {'total': 5, 'por_linha': {1: 2, 5: 1, 10: 2}}
    """
    if not texto or not termo:
        return {'total': 0, 'por_linha': {}}

    texto_lower = texto.lower()
    termo_lower = termo.lower()

    total = 0
    por_linha: dict[int, int] = {}

    # Divide o texto em linhas
    linhas = texto.split('\n')

    for linha_num, linha in enumerate(linhas, start=1):
        linha_lower = linha.lower()
        count = linha_lower.count(termo_lower)
        if count > 0:
            por_linha[linha_num] = count
            total += count

    return {'total': total, 'por_linha': por_linha}


def _extrair_secao(texto: str, inicio: str, fim: str) -> list[dict]:
    """
    Extrai seções de texto entre marcadores de início e fim.

    Args:
        texto: O texto de onde extrair
        inicio: Marcador de início da seção (case-insensitive)
        fim: Marcador de fim da seção (case-insensitive)

    Returns:
        Lista de dicts com: conteudo (texto extraído), posicao_inicio, posicao_fim, linha_inicio, linha_fim

    Example:
        >>> extrair_secao(meu_texto, "## Introdução", "## Conclusão")
        [{'conteudo': 'texto entre os marcadores...', 'posicao_inicio': 100, 'posicao_fim': 500, 'linha_inicio': 10, 'linha_fim': 50}]
    """
    if not texto or not inicio or not fim:
        return []

    resultados = []
    texto_lower = texto.lower()
    inicio_lower = inicio.lower()
    fim_lower = fim.lower()

    pos = 0
    while True:
        # Encontra o marcador de início
        start_pos = texto_lower.find(inicio_lower, pos)
        if start_pos == -1:
            break

        # Posição após o marcador de início
        content_start = start_pos + len(inicio)

        # Encontra o marcador de fim
        end_pos = texto_lower.find(fim_lower, content_start)
        if end_pos == -1:
            break

        # Extrai o conteúdo entre os marcadores
        conteudo = texto[content_start:end_pos].strip()

        # Calcula linha inicial (após o marcador de início)
        linha_inicio = texto[:content_start].count('\n') + 1

        # Calcula linha final (antes do marcador de fim)
        linha_fim = texto[:end_pos].count('\n') + 1

        resultados.append({
            'conteudo': conteudo,
            'posicao_inicio': start_pos,
            'posicao_fim': end_pos + len(fim),
            'linha_inicio': linha_inicio,
            'linha_fim': linha_fim,
        })

        # Continua a busca após este marcador de fim
        pos = end_pos + len(fim)

    return resultados


def _resumir_tamanho(bytes_val: int) -> str:
    """
    Converte um valor em bytes para uma string humanizada.

    Args:
        bytes_val: Valor em bytes (int)

    Returns:
        String formatada com unidade apropriada (B, KB, MB, GB, TB)

    Example:
        >>> resumir_tamanho(1024)
        '1.0 KB'
        >>> resumir_tamanho(1536)
        '1.5 KB'
        >>> resumir_tamanho(1048576)
        '1.0 MB'
    """
    if not isinstance(bytes_val, (int, float)):
        return f"<valor inválido: {type(bytes_val).__name__}>"

    if bytes_val < 0:
        return f"<valor negativo: {bytes_val}>"

    size = float(bytes_val)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


@dataclass
class ExecutionResult:
    """Resultado de uma execução no REPL"""
    success: bool
    stdout: str
    stderr: str
    execution_time_ms: float
    variables_changed: list[str] = field(default_factory=list)


@dataclass
class VariableInfo:
    """Informações sobre uma variável (sem o conteúdo)"""
    name: str
    type_name: str
    size_bytes: int
    size_human: str
    preview: str  # Primeiros N caracteres
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    pinned: bool = False
    source: str = "unknown"  # "s3", "file", "execute", "load_data"


class SecurityError(Exception):
    """Erro de segurança na execução"""
    pass


class ExecutionTimeoutError(Exception):
    """Erro de timeout na execução"""
    pass


def _timeout_handler(signum, frame):
    """Handler para signal de timeout"""
    raise ExecutionTimeoutError("Execution timed out")


# ============================================================================
# Funções de sandbox reutilizáveis (nível de módulo)
#
# Extraídas dos métodos de SafeREPL para serem importáveis pelo processo-filho
# do sandbox (sandbox_worker._sandbox_entry) SEM instanciar SafeREPL/LLMClient
# (que carrega a API key). Os métodos de SafeREPL delegam para estas funções,
# preservando comportamento idêntico.
# ============================================================================

def safe_import(name: str, *args, **kwargs):
    """Import customizado que valida contra a whitelist ALLOWED_IMPORTS."""
    base_module = name.split('.')[0]

    if base_module in BLOCKED_IMPORTS:
        raise SecurityError(f"Import bloqueado por seguranca: '{name}'")

    if base_module not in ALLOWED_IMPORTS:
        raise SecurityError(
            f"Import nao permitido: '{name}'. "
            f"Permitidos: {', '.join(sorted(ALLOWED_IMPORTS))}"
        )

    return __import__(name, *args, **kwargs)


def create_safe_builtins() -> dict:
    """Cria o conjunto de builtins seguros (sem os perigosos) com __import__ validado."""
    import builtins
    safe = {}
    for name in dir(builtins):
        if not name.startswith('_') and name not in BLOCKED_BUILTINS:
            safe[name] = getattr(builtins, name)

    # __import__ customizado que valida contra a whitelist
    safe['__import__'] = safe_import
    return safe


def validate_code(code: str) -> None:
    """Valida código por análise estática da AST (deny-list, defense-in-depth).

    NÃO é a fronteira de segurança — é só a 1ª camada barata. A fronteira real
    é o isolamento de processo (ver sandbox_worker). Mantida porque rejeita os
    vetores conhecidos antes mesmo de spawnar o filho.
    """
    # Parse AST para análise estática
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SecurityError(f"Erro de sintaxe: {e}")

    # Verifica nodes perigosos
    for node in ast.walk(tree):
        # Bloqueia chamadas a funções perigosas
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_BUILTINS:
                    raise SecurityError(
                        f"Funcao bloqueada: '{node.func.id}'"
                    )
            # Bloqueia .format()/.format_map(): o protocolo de format string
            # acessa atributos via string (ex: "{0.__class__}".format(x)),
            # contornando o bloqueio de dunder que só vê nós ast.Attribute.
            # Use f-strings — nelas os dunders viram nós AST e SÃO bloqueados.
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in ("format", "format_map"):
                    raise SecurityError(
                        f"Metodo bloqueado: '.{node.func.attr}()' (use f-string)"
                    )

        # Bloqueia QUALQUER referência (não só chamada) a builtin perigoso —
        # fecha o bypass por aliasing: `g = getattr; g(o, '__class__')`.
        if isinstance(node, ast.Name) and node.id in BLOCKED_BUILTINS:
            raise SecurityError(f"Nome bloqueado: '{node.id}'")

        # Bloqueia acesso a atributos dunder
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('__') and node.attr.endswith('__'):
                if node.attr not in ('__len__', '__str__', '__repr__', '__iter__'):
                    raise SecurityError(
                        f"Acesso a atributo bloqueado: '{node.attr}'"
                    )


def estimate_size(obj: Any, _seen: set = None) -> int:
    """Estima tamanho de um objeto em bytes (com detecção de ciclo).

    Sem _seen, uma estrutura auto-referente (a=[]; a.append(a)) causava
    RecursionError → o except retornava 0 → a var passava no guard de
    tamanho sem limite (DoS de memória). _seen quebra o ciclo.
    """
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return 0
    try:
        if isinstance(obj, str):
            return len(obj.encode('utf-8'))
        elif isinstance(obj, (bytes, bytearray)):
            return len(obj)
        elif isinstance(obj, (list, tuple)):
            _seen.add(obj_id)
            return sum(estimate_size(x, _seen) for x in obj)
        elif isinstance(obj, dict):
            _seen.add(obj_id)
            return sum(
                estimate_size(k, _seen) + estimate_size(v, _seen)
                for k, v in obj.items()
            )
        else:
            return sys.getsizeof(obj)
    except Exception:
        return 0


def human_size(size_bytes: float) -> str:
    """Converte bytes para formato legível."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_preview(obj: Any, max_length: int = 200) -> str:
    """Gera preview de um objeto."""
    try:
        if isinstance(obj, str):
            if len(obj) > max_length:
                return obj[:max_length] + f"... [{len(obj)} chars total]"
            return obj
        elif isinstance(obj, (list, tuple)):
            preview = str(obj[:5])
            if len(obj) > 5:
                preview = preview[:-1] + f", ... ] ({len(obj)} items)"
            return preview
        elif isinstance(obj, dict):
            keys = list(obj.keys())[:5]
            preview = str({k: obj[k] for k in keys})
            if len(obj) > 5:
                preview = preview[:-1] + f", ... }} ({len(obj)} keys)"
            return preview
        else:
            s = str(obj)
            if len(s) > max_length:
                return s[:max_length] + "..."
            return s
    except Exception:
        return f"<{type(obj).__name__}>"


class SafeREPL:
    """
    REPL Python com sandbox de segurança.

    Características:
    - Variáveis persistem entre execuções
    - Imports restritos a whitelist
    - Sem acesso a filesystem/rede
    - Timeout em execuções longas
    - Auto-limpeza de memória quando atinge threshold
    """

    def __init__(
        self,
        max_memory_mb: int = 1024,
        max_var_size_mb: int = 50,
        cleanup_threshold_percent: float = 80.0,
        cleanup_target_percent: float = 60.0,
        cleanup_strategy: str = "weighted",
    ):
        self.variables: dict[str, Any] = {}
        self.variable_metadata: dict[str, VariableInfo] = {}
        self.max_memory_mb = max_memory_mb
        self.max_var_size_mb = max_var_size_mb
        self.execution_count = 0

        # Auto-cleanup settings
        self.cleanup_threshold_percent = cleanup_threshold_percent  # Quando limpar
        self.cleanup_target_percent = cleanup_target_percent  # Até quanto limpar
        self.last_cleanup_count = 0  # Quantas variáveis foram removidas na última limpeza
        self.cleanup_strategy = cleanup_strategy  # weighted|lru|lfu|size

        # Cliente LLM para sub-chamadas recursivas (RLM)
        self.llm_client = LLMClient()

        # Namespace seguro para execução
        self._safe_builtins = self._create_safe_builtins()

        # Lock serializando snapshot+merge de variables. Os task workers rodam
        # em ThreadPoolExecutor (threads reais) → dois executes concorrentes
        # corromperiam variables/variable_metadata sem isto. Ver R9.
        self._execute_lock = threading.Lock()

        # Configuração do sandbox por subprocesso (ver sandbox_worker.py)
        self.sandbox_mode = os.getenv("RLM_SANDBOX_MODE", "subprocess").strip().lower()
        self.execute_timeout = float(os.getenv("RLM_EXECUTE_TIMEOUT", "60"))
        self.sandbox_mem_mb = int(os.getenv("RLM_SANDBOX_MEM_MB", "2048"))
        self.sandbox_cpu_s = int(os.getenv("RLM_SANDBOX_CPU_S", "60"))
        self.sandbox_shm_threshold = int(
            os.getenv("RLM_SANDBOX_SHM_THRESHOLD", str(256 * 1024))
        )

    def _create_safe_builtins(self) -> dict:
        """Cria conjunto de builtins seguros (delega para create_safe_builtins)."""
        return create_safe_builtins()

    def _safe_import(self, name: str, *args, **kwargs):
        """Import validado contra whitelist (delega para safe_import)."""
        return safe_import(name, *args, **kwargs)

    def _validate_code(self, code: str) -> None:
        """Valida código antes de executar (delega para validate_code)."""
        return validate_code(code)

    def _estimate_size(self, obj: Any, _seen: set = None) -> int:
        """Estima tamanho de um objeto em bytes (delega para estimate_size)."""
        return estimate_size(obj, _seen)

    def _human_size(self, size_bytes: int) -> str:
        """Converte bytes para formato legível (delega para human_size)."""
        return human_size(size_bytes)

    def _get_preview(self, obj: Any, max_length: int = 200) -> str:
        """Gera preview de um objeto (delega para get_preview)."""
        return get_preview(obj, max_length)

    def _llm_query_wrapper(
        self,
        prompt: str,
        data: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0
    ) -> str:
        """
        Wrapper para sub-chamadas LLM de dentro do código Python.

        Esta função implementa o core do padrão RLM (Recursive Language Models),
        permitindo que código Python chame LLMs para processar chunks de dados.

        Args:
            prompt: Instrução para o LLM
            data: Dados opcionais para processar
            model: Modelo a usar (default configurável via RLM_SUB_MODEL)
            max_tokens: Máximo de tokens na resposta
            temperature: Temperatura (0.0 = determinístico)

        Returns:
            Resposta do LLM como string

        Example:
            # Dentro de rlm_execute:
            summary = llm_query("Resuma este texto:", data=chunk)
        """
        return self.llm_client.query(prompt, data, model, max_tokens, temperature)

    def execute(self, code: str, timeout_seconds: float = None) -> ExecutionResult:
        """
        Executa código Python no sandbox.

        Em modo 'subprocess' (default, SEGURO) delega para run_sandboxed: o
        código roda num processo-filho isolado (forkserver) sem credenciais no
        env, sem FDs herdados, com setrlimit + killpg. Ver sandbox_worker.

        Em modo 'inprocess' (INSEGURO, break-glass via RLM_SANDBOX_MODE=inprocess)
        usa o caminho legado: exec no mesmo processo, timeout via SIGALRM (que só
        funciona na main thread). Reabre a classe de sandbox-escape — não usar em
        produção.

        Args:
            code: Código Python para executar
            timeout_seconds: Timeout máximo (default: RLM_EXECUTE_TIMEOUT = 60s)

        Returns:
            ExecutionResult com stdout, stderr e metadados
        """
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else self.execute_timeout
        )
        if self.sandbox_mode == "subprocess":
            from .sandbox_worker import run_sandboxed
            return run_sandboxed(
                code, self, effective_timeout,
                self.sandbox_mem_mb, self.sandbox_cpu_s,
            )
        return self._execute_inprocess(code, effective_timeout)

    def _execute_inprocess(self, code: str, timeout_seconds: float = 30.0) -> ExecutionResult:
        """Caminho legado in-process (INSEGURO — break-glass). Ver execute()."""
        import time
        start_time = time.perf_counter()

        # Valida código
        try:
            self._validate_code(code)
        except SecurityError as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"SecurityError: {e}",
                execution_time_ms=0,
            )

        # Captura stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        # Prepara namespace
        namespace = {
            '__builtins__': self._safe_builtins,
            **self.variables,  # Variáveis existentes
        }

        # Pré-importa módulos permitidos comuns
        for mod in ['re', 'json', 'math', 'collections', 'datetime']:
            try:
                namespace[mod] = __import__(mod)
            except ImportError:
                pass

        # Injeta funções RLM para sub-chamadas a LLMs (core do paper)
        namespace['llm_query'] = self._llm_query_wrapper
        namespace['llm_stats'] = self.llm_client.get_stats
        namespace['llm_reset_counter'] = self.llm_client.reset_counter

        # Injeta helper functions pré-definidas
        namespace['buscar'] = _buscar
        namespace['contar'] = _contar
        namespace['extrair_secao'] = _extrair_secao
        namespace['resumir_tamanho'] = _resumir_tamanho

        success = True

        # Set up timeout using signal (Unix only, main thread only)
        # signal.signal only works in the main thread, so we check before using it
        import threading
        is_main_thread = threading.current_thread() is threading.main_thread()
        use_timeout = timeout_seconds > 0 and is_main_thread
        old_handler = None

        try:
            if use_timeout:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                # Use integer seconds (signal.alarm only supports integers)
                signal.alarm(int(timeout_seconds) or 1)
            try:
                exec(code, namespace)
            except ExecutionTimeoutError:
                sys.stderr.write(f"ExecutionTimeoutError: Execution timed out after {timeout_seconds} seconds\n")
                success = False
            except SecurityError as e:
                sys.stderr.write(f"SecurityError: {e}\n")
                success = False
            except Exception as e:
                sys.stderr.write(f"{type(e).__name__}: {e}\n")
                sys.stderr.write(traceback.format_exc())
                success = False
        finally:
            # Always cancel the alarm and restore the old handler
            if use_timeout:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        # Captura outputs
        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()

        # Restaura stdout/stderr
        sys.stdout, sys.stderr = old_stdout, old_stderr

        # Atualiza variáveis (exceto builtins e módulos)
        now = datetime.now()
        variables_changed = []

        # Track which existing variables were accessed (referenced in code)
        for name in self.variable_metadata:
            if name in namespace and name in self.variables:
                # Variable was in namespace = it was accessible, increment access
                self.variable_metadata[name].access_count += 1
                self.variable_metadata[name].last_accessed = now

        for name, value in namespace.items():
            if name.startswith('_'):
                continue
            if name in ('re', 'json', 'math', 'collections', 'datetime'):
                continue
            # Ignora helper functions pré-definidas
            if name in HELPER_FUNCTION_NAMES:
                continue
            if callable(value) and not isinstance(value, type):
                # Permite funções definidas pelo usuário
                pass

            # Verifica se é nova ou mudou
            is_new = name not in self.variables
            is_changed = not is_new and self.variables.get(name) is not value

            if is_new or is_changed:
                size = self._estimate_size(value)
                max_var_bytes = self.max_var_size_mb * 1024 * 1024
                if size > max_var_bytes:
                    stderr += f"\nVariavel '{name}' rejeitada: {self._human_size(size)} excede limite de {self.max_var_size_mb}MB\n"
                    success = False
                    continue

                self.variables[name] = value
                variables_changed.append(name)
                existing = self.variable_metadata.get(name)
                self.variable_metadata[name] = VariableInfo(
                    name=name,
                    type_name=type(value).__name__,
                    size_bytes=size,
                    size_human=self._human_size(size),
                    preview=self._get_preview(value),
                    created_at=existing.created_at if existing else now,
                    last_accessed=now,
                    access_count=(existing.access_count if existing else 0) + 1,
                    pinned=existing.pinned if existing else False,
                    source=existing.source if existing else "execute",
                )

        execution_time = (time.perf_counter() - start_time) * 1000
        self.execution_count += 1

        # Auto-cleanup se necessário
        cleanup_info = self._auto_cleanup()
        if cleanup_info:
            stdout += f"\n[Auto-cleanup: removidas {cleanup_info['removed_count']} variáveis antigas, liberados {cleanup_info['removed_bytes_human']}]"

        return ExecutionResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            execution_time_ms=execution_time,
            variables_changed=variables_changed,
        )

    def load_data(self, name: str, data: str | bytes, data_type: str = "text") -> ExecutionResult:
        """
        Carrega dados diretamente em uma variável.

        Args:
            name: Nome da variável
            data: Dados para carregar
            data_type: "text", "json", "lines", "csv"
        """
        try:
            if data_type == "json":
                value = json.loads(data)
            elif data_type == "lines":
                value = data.split('\n') if isinstance(data, str) else data.decode().split('\n')
            elif data_type == "csv":
                import csv
                reader = csv.DictReader(StringIO(data if isinstance(data, str) else data.decode()))
                value = list(reader)
            else:  # text
                value = data if isinstance(data, str) else data.decode()

            size = self._estimate_size(value)
            max_var_bytes = self.max_var_size_mb * 1024 * 1024
            if size > max_var_bytes:
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Variavel rejeitada: {self._human_size(size)} excede limite de {self.max_var_size_mb}MB por variavel. Ajuste RLM_MAX_VAR_SIZE_MB se necessario.",
                    execution_time_ms=0,
                )

            self.variables[name] = value
            now = datetime.now()
            existing = self.variable_metadata.get(name)

            self.variable_metadata[name] = VariableInfo(
                name=name,
                type_name=type(value).__name__,
                size_bytes=size,
                size_human=self._human_size(size),
                preview=self._get_preview(value),
                created_at=existing.created_at if existing else now,
                last_accessed=now,
                access_count=(existing.access_count if existing else 0) + 1,
                pinned=existing.pinned if existing else False,
                source=existing.source if existing else "load_data",
            )

            # Auto-cleanup se necessário
            cleanup_info = self._auto_cleanup()
            stdout_msg = f"Variavel '{name}' carregada: {self._human_size(size)} ({type(value).__name__})"
            if cleanup_info:
                stdout_msg += f"\n[Auto-cleanup: removidas {cleanup_info['removed_count']} variáveis antigas, liberados {cleanup_info['removed_bytes_human']}]"

            return ExecutionResult(
                success=True,
                stdout=stdout_msg,
                stderr="",
                execution_time_ms=0,
                variables_changed=[name],
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Erro ao carregar dados: {e}",
                execution_time_ms=0,
            )

    def list_variables(self) -> list[VariableInfo]:
        """Lista todas as variáveis com metadados"""
        return list(self.variable_metadata.values())

    def get_variable_info(self, name: str) -> VariableInfo | None:
        """Retorna informações de uma variável específica"""
        return self.variable_metadata.get(name)

    def clear_variable(self, name: str) -> bool:
        """Remove uma variável"""
        if name in self.variables:
            del self.variables[name]
            del self.variable_metadata[name]
            return True
        return False

    def clear_all(self) -> int:
        """Remove todas as variáveis"""
        count = len(self.variables)
        self.variables.clear()
        self.variable_metadata.clear()
        return count

    def pin_variable(self, name: str, pin: bool = True) -> bool:
        """Pin or unpin a variable to protect it from GC.

        Args:
            name: Variable name
            pin: True to pin, False to unpin

        Returns:
            True if variable found and updated, False otherwise
        """
        if name in self.variable_metadata:
            self.variable_metadata[name].pinned = pin
            return True
        return False

    def get_memory_usage(self) -> dict:
        """Retorna uso de memória"""
        total = sum(v.size_bytes for v in self.variable_metadata.values())
        return {
            "total_bytes": total,
            "total_human": self._human_size(total),
            "variable_count": len(self.variables),
            "max_allowed_mb": self.max_memory_mb,
            "usage_percent": (total / (self.max_memory_mb * 1024 * 1024)) * 100,
        }

    def _cleanup_score(self, meta: VariableInfo) -> float:
        """Calculate cleanup priority score.

        Higher score = more likely to keep.
        Lower score = candidate for removal.

        Formula: (recency × frequency) / (1 + size_mb)
        - recency: seconds since last access (inverted - recent = higher)
        - frequency: access_count (more accessed = higher)
        - size_mb: larger vars get lower score (cost more to keep)
        """
        now = datetime.now()
        age_seconds = max(1, (now - meta.last_accessed).total_seconds())
        recency = 1.0 / age_seconds  # Recent = higher value
        frequency = max(1, meta.access_count)
        size_mb = meta.size_bytes / (1024 * 1024)

        return (recency * frequency) / (1.0 + size_mb)

    def _auto_cleanup(self) -> dict:
        """
        Auto-limpeza de memória quando atinge threshold.

        Strategies:
        - weighted: Score-based (recency × frequency / size) - default
        - lru: Least recently used (original behavior)
        - lfu: Least frequently used
        - size: Largest variables first

        Pinned variables are never removed.
        Preserva funções LLM (llm_query, llm_stats, llm_reset_counter).

        Returns:
            Dict com informações da limpeza (ou vazio se não foi necessário)
        """
        usage = self.get_memory_usage()

        if usage["usage_percent"] < self.cleanup_threshold_percent:
            return {}  # Não precisa limpar

        logger.info(
            f"Auto-cleanup triggered: {usage['usage_percent']:.1f}% > {self.cleanup_threshold_percent}% "
            f"(strategy: {self.cleanup_strategy})"
        )

        # Variáveis protegidas (não remover)
        protected = {'llm_query', 'llm_stats', 'llm_reset_counter'}

        # Filter eligible variables (not protected, not pinned)
        eligible = [
            (name, meta) for name, meta in self.variable_metadata.items()
            if name not in protected and not meta.pinned
        ]

        # Sort by strategy (lowest priority first = removed first)
        strategy = self.cleanup_strategy
        if strategy == "lru":
            sorted_vars = sorted(eligible, key=lambda x: x[1].last_accessed)
        elif strategy == "lfu":
            sorted_vars = sorted(eligible, key=lambda x: x[1].access_count)
        elif strategy == "size":
            sorted_vars = sorted(eligible, key=lambda x: -x[1].size_bytes)
        else:  # weighted (default)
            sorted_vars = sorted(eligible, key=lambda x: self._cleanup_score(x[1]))

        removed = []
        removed_bytes = 0
        target_bytes = (self.cleanup_target_percent / 100) * (self.max_memory_mb * 1024 * 1024)

        for name, meta in sorted_vars:
            current_total = usage["total_bytes"] - removed_bytes

            if current_total <= target_bytes:
                break  # Atingiu o target

            # Remove variável
            removed_bytes += meta.size_bytes
            removed.append({
                "name": name,
                "size": meta.size_human,
                "last_accessed": meta.last_accessed.isoformat()
            })

            del self.variables[name]
            del self.variable_metadata[name]

        self.last_cleanup_count = len(removed)

        if removed:
            new_usage = self.get_memory_usage()
            logger.info(
                f"Auto-cleanup complete: removed {len(removed)} variables, "
                f"freed {self._human_size(removed_bytes)}, "
                f"usage now {new_usage['usage_percent']:.1f}%"
            )

            return {
                "triggered": True,
                "removed_count": len(removed),
                "removed_bytes": removed_bytes,
                "removed_bytes_human": self._human_size(removed_bytes),
                "removed_variables": removed,
                "new_usage_percent": new_usage["usage_percent"]
            }

        return {}
