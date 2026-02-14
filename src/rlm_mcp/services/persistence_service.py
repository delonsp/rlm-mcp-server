"""
Service layer for persistence and indexing operations.

Extracts the repeated pattern of auto-persisting and auto-indexing
variables after loading data.
"""

import logging
from typing import TYPE_CHECKING

from ..persistence import get_persistence
from ..indexer import auto_index_if_large, set_index
from ..embeddings import get_embedding_service
from ..vector_index import VectorIndex, set_vector_index

if TYPE_CHECKING:
    from ..repl import PythonREPL

logger = logging.getLogger(__name__)

# Minimum text size for auto-embedding (100k chars, same as keyword indexing)
AUTO_EMBED_MIN_CHARS = 100000


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

        vi = VectorIndex(var_name)
        success = vi.build(text)
        if not success:
            return ""

        set_vector_index(var_name, vi)

        # Persist embeddings to SQLite
        chunks_data = [
            {
                "chunk_index": c.chunk_index,
                "chunk_text": c.text,
                "line_start": c.line_start,
                "line_end": c.line_end,
                "embedding": c.embedding,
            }
            for c in vi.chunks
            if c.embedding
        ]
        persistence.save_embeddings(var_name, chunks_data)

        stats = vi.get_stats()
        logger.info(f"Auto-embedded '{var_name}': {stats['embedded_chunks']} chunks")
        return f"🔮 Embedded ({stats['embedded_chunks']} chunks)"

    except Exception as e:
        logger.warning(f"Auto-embed failed for '{var_name}': {e}")
        return ""
