"""
REPL Python Persistente com Sandbox de Segurança

Mantém variáveis em memória entre execuções, permitindo
manipulação de dados massivos sem carregar no contexto do LLM.

Implementa o padrão RLM (Recursive Language Models) do paper MIT CSAIL,
permitindo sub-chamadas a LLMs de dentro do código Python.
"""

import ast
import sys
import traceback
import signal
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
    "functools", "operator", "string", "textwrap", "unicodedata",
    # Data/Time
    "datetime", "time", "calendar",
    # Estruturas de dados
    "dataclasses", "typing", "enum",
    # Texto e parsing
    "csv", "html", "xml.etree.ElementTree",
    # Hashing (read-only)
    "hashlib", "base64",
    # Compressão (para ler arquivos)
    "gzip", "zipfile", "tarfile",
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
    "open", "input", "breakpoint",
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


class SecurityError(Exception):
    """Erro de segurança na execução"""
    pass


class ExecutionTimeoutError(Exception):
    """Erro de timeout na execução"""
    pass


def _timeout_handler(signum, frame):
    """Handler para signal de timeout"""
    raise ExecutionTimeoutError("Execution timed out")


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
        cleanup_threshold_percent: float = 80.0,
        cleanup_target_percent: float = 60.0
    ):
        self.variables: dict[str, Any] = {}
        self.variable_metadata: dict[str, VariableInfo] = {}
        self.max_memory_mb = max_memory_mb
        self.execution_count = 0

        # Auto-cleanup settings
        self.cleanup_threshold_percent = cleanup_threshold_percent  # Quando limpar
        self.cleanup_target_percent = cleanup_target_percent  # Até quanto limpar
        self.last_cleanup_count = 0  # Quantas variáveis foram removidas na última limpeza

        # Cliente LLM para sub-chamadas recursivas (RLM)
        self.llm_client = LLMClient()

        # Namespace seguro para execução
        self._safe_builtins = self._create_safe_builtins()

    def _create_safe_builtins(self) -> dict:
        """Cria conjunto de builtins seguros"""
        import builtins
        safe = {}
        for name in dir(builtins):
            if not name.startswith('_') and name not in BLOCKED_BUILTINS:
                safe[name] = getattr(builtins, name)

        # Adiciona __import__ customizado que valida imports
        safe['__import__'] = self._safe_import
        return safe

    def _safe_import(self, name: str, *args, **kwargs):
        """Import customizado que valida contra whitelist"""
        base_module = name.split('.')[0]

        if base_module in BLOCKED_IMPORTS:
            raise SecurityError(f"Import bloqueado por seguranca: '{name}'")

        if base_module not in ALLOWED_IMPORTS:
            raise SecurityError(
                f"Import nao permitido: '{name}'. "
                f"Permitidos: {', '.join(sorted(ALLOWED_IMPORTS))}"
            )

        return __import__(name, *args, **kwargs)

    def _validate_code(self, code: str) -> None:
        """Valida código antes de executar"""
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

            # Bloqueia acesso a atributos dunder
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('__') and node.attr.endswith('__'):
                    if node.attr not in ('__len__', '__str__', '__repr__', '__iter__'):
                        raise SecurityError(
                            f"Acesso a atributo bloqueado: '{node.attr}'"
                        )

    def _estimate_size(self, obj: Any) -> int:
        """Estima tamanho de um objeto em bytes"""
        try:
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (bytes, bytearray)):
                return len(obj)
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(x) for x in obj)
            elif isinstance(obj, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in obj.items()
                )
            else:
                return sys.getsizeof(obj)
        except Exception:
            return 0

    def _human_size(self, size_bytes: int) -> str:
        """Converte bytes para formato legível"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def _get_preview(self, obj: Any, max_length: int = 200) -> str:
        """Gera preview de um objeto"""
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

    def execute(self, code: str, timeout_seconds: float = 30.0) -> ExecutionResult:
        """
        Executa código Python no sandbox.

        Args:
            code: Código Python para executar
            timeout_seconds: Timeout máximo para execução (default: 30s)

        Returns:
            ExecutionResult com stdout, stderr e metadados
        """
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
                self.variables[name] = value
                variables_changed.append(name)

                # Atualiza metadados
                size = self._estimate_size(value)
                self.variable_metadata[name] = VariableInfo(
                    name=name,
                    type_name=type(value).__name__,
                    size_bytes=size,
                    size_human=self._human_size(size),
                    preview=self._get_preview(value),
                    created_at=now if is_new else self.variable_metadata.get(name, VariableInfo(name, "", 0, "", "", now, now)).created_at,
                    last_accessed=now,
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

            self.variables[name] = value
            size = self._estimate_size(value)
            now = datetime.now()

            self.variable_metadata[name] = VariableInfo(
                name=name,
                type_name=type(value).__name__,
                size_bytes=size,
                size_human=self._human_size(size),
                preview=self._get_preview(value),
                created_at=now,
                last_accessed=now,
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

    def _auto_cleanup(self) -> dict:
        """
        Auto-limpeza de memória quando atinge threshold.

        Remove variáveis mais antigas (por last_accessed) até atingir o target.
        Preserva funções LLM (llm_query, llm_stats, llm_reset_counter).

        Returns:
            Dict com informações da limpeza (ou vazio se não foi necessário)
        """
        usage = self.get_memory_usage()

        if usage["usage_percent"] < self.cleanup_threshold_percent:
            return {}  # Não precisa limpar

        logger.info(
            f"Auto-cleanup triggered: {usage['usage_percent']:.1f}% > {self.cleanup_threshold_percent}%"
        )

        # Variáveis protegidas (não remover)
        protected = {'llm_query', 'llm_stats', 'llm_reset_counter'}

        # Ordena variáveis por last_accessed (mais antigas primeiro)
        sorted_vars = sorted(
            [(name, meta) for name, meta in self.variable_metadata.items() if name not in protected],
            key=lambda x: x[1].last_accessed
        )

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
