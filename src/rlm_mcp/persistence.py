"""
Persistência de variáveis e índices para RLM MCP Server.

Usa SQLite para persistir:
- Variáveis do REPL (texto, dados processados)
- Índices semânticos (para busca rápida)

O arquivo SQLite fica em /persist/rlm_data.db (volume Docker).
"""

import os
import json
import sqlite3
import logging
import pickle
import zlib
from datetime import datetime
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger("rlm-mcp.persistence")

# Diretório de persistência (configurável via env)
PERSIST_DIR = os.getenv("RLM_PERSIST_DIR", "/persist")
DB_FILE = os.path.join(PERSIST_DIR, "rlm_data.db")


class PersistenceManager:
    """Gerencia persistência de variáveis e índices em SQLite."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_FILE
        self._ensure_dir()
        self._init_db()
        logger.info(f"PersistenceManager inicializado: {self.db_path}")

    def _ensure_dir(self):
        """Garante que o diretório de persistência existe."""
        dir_path = os.path.dirname(self.db_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Diretório de persistência criado: {dir_path}")

    def _init_db(self):
        """Inicializa o banco de dados SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tabela de variáveis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS variables (
                    name TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    type_name TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            # Tabela de índices semânticos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indices (
                    var_name TEXT PRIMARY KEY,
                    index_data BLOB NOT NULL,
                    terms_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (var_name) REFERENCES variables(name)
                )
            """)

            conn.commit()
            logger.info("Banco de dados SQLite inicializado")

    def save_variable(self, name: str, value: Any, metadata: dict = None) -> bool:
        """
        Salva uma variável no banco de dados.

        Args:
            name: Nome da variável
            value: Valor (será serializado com pickle + compressão)
            metadata: Metadados opcionais (dict)

        Returns:
            True se salvou com sucesso
        """
        try:
            # Serializar e comprimir
            pickled = pickle.dumps(value)
            compressed = zlib.compress(pickled)

            now = datetime.now().isoformat()
            type_name = type(value).__name__
            size_bytes = len(pickled)
            metadata_json = json.dumps(metadata) if metadata else None

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO variables
                    (name, data, type_name, size_bytes, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?,
                        COALESCE((SELECT created_at FROM variables WHERE name = ?), ?),
                        ?, ?)
                """, (name, compressed, type_name, size_bytes, name, now, now, metadata_json))
                conn.commit()

            logger.info(f"Variável '{name}' salva ({size_bytes:,} bytes)")
            return True

        except Exception as e:
            logger.error(f"Erro ao salvar variável '{name}': {e}")
            return False

    def load_variable(self, name: str) -> Optional[Any]:
        """
        Carrega uma variável do banco de dados.

        Args:
            name: Nome da variável

        Returns:
            Valor da variável ou None se não existir
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT data FROM variables WHERE name = ?", (name,))
                row = cursor.fetchone()

                if row:
                    compressed = row[0]
                    pickled = zlib.decompress(compressed)
                    value = pickle.loads(pickled)
                    logger.info(f"Variável '{name}' carregada")
                    return value

                return None

        except Exception as e:
            logger.error(f"Erro ao carregar variável '{name}': {e}")
            return None

    def delete_variable(self, name: str) -> bool:
        """Remove uma variável do banco de dados."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM variables WHERE name = ?", (name,))
                cursor.execute("DELETE FROM indices WHERE var_name = ?", (name,))
                conn.commit()

            logger.info(f"Variável '{name}' removida")
            return True

        except Exception as e:
            logger.error(f"Erro ao remover variável '{name}': {e}")
            return False

    def list_variables(self) -> list[dict]:
        """Lista todas as variáveis persistidas."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name, type_name, size_bytes, created_at, updated_at
                    FROM variables ORDER BY updated_at DESC
                """)
                rows = cursor.fetchall()

                return [
                    {
                        "name": row[0],
                        "type": row[1],
                        "size_bytes": row[2],
                        "created_at": row[3],
                        "updated_at": row[4]
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Erro ao listar variáveis: {e}")
            return []

    def save_index(self, var_name: str, index_data: dict) -> bool:
        """
        Salva um índice semântico associado a uma variável.

        Args:
            var_name: Nome da variável associada
            index_data: Dicionário com o índice {termo: [posições]}

        Returns:
            True se salvou com sucesso
        """
        try:
            pickled = pickle.dumps(index_data)
            compressed = zlib.compress(pickled)
            now = datetime.now().isoformat()
            terms_count = len(index_data)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO indices
                    (var_name, index_data, terms_count, created_at)
                    VALUES (?, ?, ?, ?)
                """, (var_name, compressed, terms_count, now))
                conn.commit()

            logger.info(f"Índice de '{var_name}' salvo ({terms_count} termos)")
            return True

        except Exception as e:
            logger.error(f"Erro ao salvar índice de '{var_name}': {e}")
            return False

    def load_index(self, var_name: str) -> Optional[dict]:
        """Carrega o índice semântico de uma variável."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT index_data FROM indices WHERE var_name = ?", (var_name,))
                row = cursor.fetchone()

                if row:
                    compressed = row[0]
                    pickled = zlib.decompress(compressed)
                    index_data = pickle.loads(pickled)
                    logger.info(f"Índice de '{var_name}' carregado")
                    return index_data

                return None

        except Exception as e:
            logger.error(f"Erro ao carregar índice de '{var_name}': {e}")
            return None

    def clear_all(self) -> int:
        """Remove todas as variáveis e índices. Retorna quantidade removida."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM variables")
                count = cursor.fetchone()[0]
                cursor.execute("DELETE FROM variables")
                cursor.execute("DELETE FROM indices")
                conn.commit()

            logger.info(f"Todas as {count} variáveis removidas")
            return count

        except Exception as e:
            logger.error(f"Erro ao limpar banco: {e}")
            return 0

    def get_stats(self) -> dict:
        """Retorna estatísticas do banco de dados."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*), SUM(size_bytes) FROM variables")
                var_count, total_size = cursor.fetchone()

                cursor.execute("SELECT COUNT(*), SUM(terms_count) FROM indices")
                idx_count, total_terms = cursor.fetchone()

                # Tamanho do arquivo
                file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

                return {
                    "variables_count": var_count or 0,
                    "variables_total_size": total_size or 0,
                    "indices_count": idx_count or 0,
                    "total_indexed_terms": total_terms or 0,
                    "db_file_size": file_size,
                    "db_path": self.db_path
                }

        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {}


# Singleton global
_persistence: Optional[PersistenceManager] = None


def get_persistence() -> PersistenceManager:
    """Retorna instância singleton do PersistenceManager."""
    global _persistence
    if _persistence is None:
        _persistence = PersistenceManager()
    return _persistence
