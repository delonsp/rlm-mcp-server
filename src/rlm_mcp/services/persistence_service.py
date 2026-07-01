"""
Service layer for persistence and indexing operations.

Extracts the repeated pattern of auto-persisting and auto-indexing
variables after loading data.
"""

import logging
import threading
from typing import TYPE_CHECKING

from ..persistence import get_persistence
from ..indexer import (
    auto_index_if_large, set_index, get_index, clear_index, fingerprint_source,
)
from ..embeddings import get_embedding_service
from ..vector_index import (
    VectorIndex, set_vector_index, get_vector_index, clear_vector_index,
)

if TYPE_CHECKING:
    from ..repl import PythonREPL

logger = logging.getLogger(__name__)

# Lock por-var serializando a construção de embeddings: sem isto, um search
# (ensure_embeddings, sob dispatch lock) e um batch worker (persist_and_index,
# no threadpool do task_manager) podiam embeddar o MESMO var em paralelo —
# custo OpenAI dobrado e índice sobrescrito. Mesmo padrão do repertory._get_lock.
_embed_locks: dict[str, threading.Lock] = {}
_embed_locks_guard = threading.Lock()
# Vars cujo rebuild lazy de cobertura parcial já foi tentado neste processo —
# evita re-embeddar a cada busca quando o build fica parcial de forma estável
# (ex.: chunk que sempre falha). O restart zera (nova chance de convergir).
_partial_rebuild_attempted: set[str] = set()


def _get_embed_lock(var_name: str) -> threading.Lock:
    lock = _embed_locks.get(var_name)
    if lock is None:
        with _embed_locks_guard:
            lock = _embed_locks.get(var_name)
            if lock is None:
                lock = threading.Lock()
                _embed_locks[var_name] = lock
    return lock

def invalidate_stale_indices(var_name: str, text) -> bool:
    """Descarta índices em memória cuja fonte não bate mais com `text`.

    Índices são keyed pelo NOME do var. Se o var foi rebindado (ex.: via
    rlm_execute, que NÃO passa por persist_and_index), os índices keyword/vetorial
    seguem com linhas e snippets do texto ANTIGO — busca devolve citação errada.
    Comparar a fingerprint pega isso e força rebuild na próxima busca.

    Só invalida quando a fingerprint armazenada é conhecida (não-vazia) E difere —
    índice restaurado sem fingerprint (legado) não é descartado à toa. Retorna
    True se algo foi invalidado.
    """
    fp = fingerprint_source(text)
    invalidated = False

    ki = get_index(var_name)
    if ki is not None:
        stored = getattr(ki, "source_fingerprint", "")
        if stored and stored != fp:
            clear_index(var_name)
            invalidated = True

    vi = get_vector_index(var_name)
    if vi is not None:
        stored = getattr(vi, "source_fingerprint", "")
        if stored and stored != fp:
            clear_vector_index(var_name)
            _partial_rebuild_attempted.discard(var_name)
            invalidated = True

    if invalidated:
        logger.info(f"Índice stale de '{var_name}' invalidado (var mudou); rebuild na próxima busca")
    return invalidated


# Minimum text size for auto-embedding (100k chars, same as keyword indexing)
AUTO_EMBED_MIN_CHARS = 100000
# Upper bound for synchronous lazy embedding. Above this, the on-demand build
# would block a search request too long; skip and let the user retry/load.
# ~25M chars ≈ 50 API batches. Tudo abaixo (inclusive os ReCODE de 11MB) entra.
LAZY_EMBED_MAX_CHARS = 25_000_000


def persist_and_index(var_name: str, value, repl: "PythonREPL") -> tuple[str, str, str]:
    """Persiste variável e indexa se grande.

    Args:
        var_name: Nome da variável a persistir
        value: Valor a persistir (obtido de repl.variables.get(var_name))
        repl: Instância do REPL (não usado diretamente, mas mantido para compatibilidade)

    Returns:
        tuple: (persist_msg, index_msg, error_msg)
            - persist_msg: Mensagem de sucesso de persistência (ex: "💾 Persistido")
            - index_msg: Mensagem de sucesso de indexação (ex: "📑 Indexado (50 termos)")
            - error_msg: Mensagem de erro se houver (ex: "\n⚠️ Erro de persistência: ...")
    """
    persist_msg = ""
    index_msg = ""
    error_msg = ""

    try:
        persistence = get_persistence()
        if value is not None:
            saved = persistence.save_variable(var_name, value)
            if saved:
                persist_msg = "💾 Persistido"
            else:
                error_msg = "\n⚠️ Erro de persistência: save_variable retornou False"

            # Indexar se for texto grande (>= 100k caracteres)
            if saved and isinstance(value, str) and len(value) >= AUTO_EMBED_MIN_CHARS:
                idx = auto_index_if_large(value, var_name)
                if idx:
                    set_index(var_name, idx)
                    persistence.save_index(var_name, idx.to_dict())
                    index_msg = f"📑 Indexado ({idx.get_stats()['indexed_terms']} termos)"

                # Auto-embed if embedding service is available
                embed_msg = _auto_embed(var_name, value, persistence)
                if embed_msg:
                    index_msg = f"{index_msg} {embed_msg}" if index_msg else embed_msg

    except Exception as e:
        logger.warning(f"Erro ao persistir/indexar {var_name}: {e}")
        error_msg = f"\n⚠️ Erro de persistência: {e}"

    return persist_msg, index_msg, error_msg


def ensure_embeddings(var_name: str, value) -> str:
    """Constrói embeddings sob demanda para um var que ainda não os tem.

    Cobre vars nascidos no `rlm_execute` (que não passam por persist_and_index)
    e vars cujo embed falhou no load (ex: antes do batching em embeddings.py).
    Idempotente: se o índice vetorial já existe em memória, não faz nada.
    Persiste o resultado no SQLite, então o custo é pago uma única vez.

    Returns:
        Status string (ex: "🔮 Embedded (...)"), ou "" se nada foi feito.
    """
    existing = get_vector_index(var_name)
    if existing is not None:
        # Índice existe mas cobertura parcial (embedded < total): um build que
        # falhou no meio (batch de rede) deixava o índice parcial "existindo" e
        # o early-return o congelava para sempre. Tenta rebuildar UMA vez por
        # processo (guard _partial_rebuild_attempted evita re-embed a cada busca).
        stats = existing.get_stats()
        total = stats.get("total_chunks", 0)
        embedded = stats.get("embedded_chunks", 0)
        if total and embedded < total and var_name not in _partial_rebuild_attempted:
            if isinstance(value, str) and AUTO_EMBED_MIN_CHARS <= len(value) <= LAZY_EMBED_MAX_CHARS:
                service = get_embedding_service()
                if service.is_available:
                    logger.info(
                        f"Cobertura parcial em '{var_name}' ({embedded}/{total}); "
                        f"tentando rebuild lazy (1x/processo)"
                    )
                    _partial_rebuild_attempted.add(var_name)
                    return _auto_embed(var_name, value, get_persistence())
        return ""
    if not isinstance(value, str) or len(value) < AUTO_EMBED_MIN_CHARS:
        return ""
    if len(value) > LAZY_EMBED_MAX_CHARS:
        logger.warning(
            f"Lazy embed pulado para '{var_name}': {len(value):,} chars > "
            f"limite {LAZY_EMBED_MAX_CHARS:,}"
        )
        return ""
    service = get_embedding_service()
    if not service.is_available:
        return ""

    persistence = get_persistence()
    logger.info(f"Lazy-embedding '{var_name}' ({len(value):,} chars) sob demanda...")
    return _auto_embed(var_name, value, persistence)


def _auto_embed(var_name: str, text: str, persistence) -> str:
    """Auto-embed text if embedding service is available.

    Args:
        var_name: Variable name
        text: Text content
        persistence: PersistenceManager instance

    Returns:
        Status message or empty string
    """
    try:
        service = get_embedding_service()
        if not service.is_available:
            return ""

        # Lock por-var: se dois caminhos (search lazy + batch worker) chegarem
        # juntos, o 2º espera e reusa em vez de re-embeddar (evita custo dobrado
        # na OpenAI e sobrescrita do índice a meio-build).
        with _get_embed_lock(var_name):
            vi = VectorIndex(var_name)
            success = vi.build(text)
            if not success:
                return ""

            set_vector_index(var_name, vi)

            # Persist embeddings to SQLite (todos os chunks; sem-vetor com blob vazio)
            persistence.save_embeddings(var_name, vi.persist_payload())

            stats = vi.get_stats()
        logger.info(f"Auto-embedded '{var_name}': {stats['embedded_chunks']} chunks")
        return f"🔮 Embedded ({stats['embedded_chunks']} chunks)"

    except Exception as e:
        logger.warning(f"Auto-embed failed for '{var_name}': {e}")
        return ""
