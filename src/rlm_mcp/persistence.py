"""
Persistência de variáveis e índices para RLM MCP Server.

Usa SQLite para persistir:
- Variáveis do REPL (texto, dados processados)
- Índices semânticos (para busca rápida)
- Coleções (agrupamento de variáveis por assunto)

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

            # Tabela de coleções
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Tabela de relacionamento coleção <-> variáveis (N:N)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection_vars (
                    collection_name TEXT NOT NULL,
                    var_name TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (collection_name, var_name),
                    FOREIGN KEY (collection_name) REFERENCES collections(name),
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

    # =========================================================================
    # Métodos de Coleções
    # =========================================================================

    def create_collection(self, name: str, description: str = None) -> bool:
        """Cria uma nova coleção."""
        try:
            now = datetime.now().isoformat()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO collections (name, description, created_at)
                    VALUES (?, ?, COALESCE((SELECT created_at FROM collections WHERE name = ?), ?))
                """, (name, description, name, now))
                conn.commit()

            logger.info(f"Coleção '{name}' criada")
            return True

        except Exception as e:
            logger.error(f"Erro ao criar coleção '{name}': {e}")
            return False

    def delete_collection(self, name: str) -> bool:
        """Remove uma coleção (não remove as variáveis, apenas a associação)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM collection_vars WHERE collection_name = ?", (name,))
                cursor.execute("DELETE FROM collections WHERE name = ?", (name,))
                conn.commit()

            logger.info(f"Coleção '{name}' removida")
            return True

        except Exception as e:
            logger.error(f"Erro ao remover coleção '{name}': {e}")
            return False

    def add_to_collection(self, collection_name: str, var_names: list[str]) -> int:
        """
        Adiciona variáveis a uma coleção.

        Args:
            collection_name: Nome da coleção
            var_names: Lista de nomes de variáveis

        Returns:
            Número de variáveis adicionadas
        """
        try:
            now = datetime.now().isoformat()
            added = 0

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Verificar se coleção existe, criar se não
                cursor.execute("SELECT 1 FROM collections WHERE name = ?", (collection_name,))
                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT INTO collections (name, created_at) VALUES (?, ?)",
                        (collection_name, now)
                    )

                for var_name in var_names:
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO collection_vars (collection_name, var_name, added_at)
                            VALUES (?, ?, ?)
                        """, (collection_name, var_name, now))
                        if cursor.rowcount > 0:
                            added += 1
                    except sqlite3.IntegrityError:
                        pass  # Já existe

                conn.commit()

            logger.info(f"{added} variáveis adicionadas à coleção '{collection_name}'")
            return added

        except Exception as e:
            logger.error(f"Erro ao adicionar a coleção '{collection_name}': {e}")
            return 0

    def remove_from_collection(self, collection_name: str, var_names: list[str]) -> int:
        """Remove variáveis de uma coleção."""
        try:
            removed = 0
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for var_name in var_names:
                    cursor.execute("""
                        DELETE FROM collection_vars
                        WHERE collection_name = ? AND var_name = ?
                    """, (collection_name, var_name))
                    removed += cursor.rowcount
                conn.commit()

            logger.info(f"{removed} variáveis removidas da coleção '{collection_name}'")
            return removed

        except Exception as e:
            logger.error(f"Erro ao remover da coleção '{collection_name}': {e}")
            return 0

    def list_collections(self) -> list[dict]:
        """Lista todas as coleções com contagem de variáveis."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.name, c.description, c.created_at,
                           COUNT(cv.var_name) as var_count
                    FROM collections c
                    LEFT JOIN collection_vars cv ON c.name = cv.collection_name
                    GROUP BY c.name
                    ORDER BY c.name
                """)
                rows = cursor.fetchall()

                return [
                    {
                        "name": row[0],
                        "description": row[1],
                        "created_at": row[2],
                        "var_count": row[3]
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Erro ao listar coleções: {e}")
            return []

    def get_collection_vars(self, collection_name: str) -> list[str]:
        """Retorna lista de variáveis em uma coleção."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT var_name FROM collection_vars
                    WHERE collection_name = ?
                    ORDER BY var_name
                """, (collection_name,))
                return [row[0] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Erro ao obter variáveis da coleção '{collection_name}': {e}")
            return []

    def get_collection_info(self, collection_name: str) -> Optional[dict]:
        """Retorna informações detalhadas de uma coleção."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Info da coleção
                cursor.execute("""
                    SELECT name, description, created_at FROM collections WHERE name = ?
                """, (collection_name,))
                row = cursor.fetchone()
                if not row:
                    return None

                # Variáveis na coleção com seus tamanhos
                cursor.execute("""
                    SELECT v.name, v.type_name, v.size_bytes
                    FROM collection_vars cv
                    JOIN variables v ON cv.var_name = v.name
                    WHERE cv.collection_name = ?
                    ORDER BY v.name
                """, (collection_name,))
                vars_rows = cursor.fetchall()

                return {
                    "name": row[0],
                    "description": row[1],
                    "created_at": row[2],
                    "variables": [
                        {"name": r[0], "type": r[1], "size_bytes": r[2]}
                        for r in vars_rows
                    ],
                    "total_size": sum(r[2] for r in vars_rows),
                    "var_count": len(vars_rows)
                }

        except Exception as e:
            logger.error(f"Erro ao obter info da coleção '{collection_name}': {e}")
            return None


# Singleton global
_persistence: Optional[PersistenceManager] = None


def get_persistence() -> PersistenceManager:
    """Retorna instância singleton do PersistenceManager."""
    global _persistence
    if _persistence is None:
        _persistence = PersistenceManager()
    return _persistence
