"""
Cliente S3/Minio para RLM MCP Server.

Permite carregar arquivos grandes diretamente do Minio
sem passar pelo contexto do Claude Code.
"""

import os
import logging
from typing import Optional, Callable
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("rlm-mcp.s3")

BATCH_MAX_WORKERS = int(os.getenv("RLM_BATCH_MAX_WORKERS", "4"))

class S3Client:
    """Cliente para carregar arquivos do Minio/S3."""

    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "")
        self.secure = os.getenv("MINIO_SECURE", "true").lower() == "true"
        self._client = None

        if not self.endpoint:
            logger.warning(
                "MINIO_ENDPOINT não configurado. rlm_load_s3() não funcionará."
            )

    @property
    def client(self):
        """Lazy initialization do cliente Minio."""
        if self._client is None:
            if not self.endpoint:
                raise RuntimeError(
                    "MINIO_ENDPOINT não configurado. "
                    "Configure as variáveis MINIO_* no servidor."
                )
            try:
                from minio import Minio
                self._client = Minio(
                    self.endpoint,
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    secure=self.secure
                )
                logger.info(f"Cliente Minio conectado a {self.endpoint}")
            except ImportError:
                raise RuntimeError(
                    "Pacote 'minio' não instalado. "
                    "Execute: pip install minio"
                )
        return self._client

    def is_configured(self) -> bool:
        """Verifica se o cliente está configurado."""
        return bool(self.endpoint and self.access_key and self.secret_key)

    def list_buckets(self) -> list[str]:
        """Lista buckets disponíveis."""
        try:
            buckets = self.client.list_buckets()
            return [b.name for b in buckets]
        except Exception as e:
            logger.error(f"Erro ao listar buckets: {e}")
            raise RuntimeError(f"Erro ao listar buckets: {e}")

    def list_objects(self, bucket: str, prefix: str = "") -> list[dict]:
        """Lista objetos em um bucket."""
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=True)
            result = []
            for obj in objects:
                result.append({
                    "name": obj.object_name,
                    "size": obj.size,
                    "size_human": self._human_size(obj.size),
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None
                })
            return result
        except Exception as e:
            logger.error(f"Erro ao listar objetos: {e}")
            raise RuntimeError(f"Erro ao listar objetos em {bucket}: {e}")

    def get_object(self, bucket: str, key: str) -> bytes:
        """Baixa objeto do Minio."""
        try:
            response = self.client.get_object(bucket, key)
            data = response.read()
            response.close()
            response.release_conn()

            logger.info(f"Objeto baixado: {bucket}/{key} ({self._human_size(len(data))})")
            return data
        except Exception as e:
            logger.error(f"Erro ao baixar objeto {bucket}/{key}: {e}")
            raise RuntimeError(f"Erro ao baixar {bucket}/{key}: {e}")

    def get_object_text(
        self,
        bucket: str,
        key: str,
        encoding: str = "utf-8"
    ) -> str:
        """Baixa objeto como texto."""
        data = self.get_object(bucket, key)
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            # Fallback para latin-1 se UTF-8 falhar
            return data.decode("latin-1")

    def object_exists(self, bucket: str, key: str) -> bool:
        """Verifica se objeto existe."""
        try:
            self.client.stat_object(bucket, key)
            return True
        except Exception:
            return False

    def put_object(self, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> dict:
        """Faz upload de objeto para o Minio."""
        try:
            data_stream = BytesIO(data)
            size = len(data)

            result = self.client.put_object(
                bucket,
                key,
                data_stream,
                size,
                content_type=content_type
            )

            logger.info(f"Objeto enviado: {bucket}/{key} ({self._human_size(size)})")
            return {
                "bucket": bucket,
                "key": key,
                "size": size,
                "size_human": self._human_size(size),
                "etag": result.etag
            }
        except Exception as e:
            logger.error(f"Erro ao enviar objeto {bucket}/{key}: {e}")
            raise RuntimeError(f"Erro ao enviar {bucket}/{key}: {e}")

    def put_object_text(self, bucket: str, key: str, text: str, encoding: str = "utf-8") -> dict:
        """Faz upload de texto para o Minio."""
        data = text.encode(encoding)
        return self.put_object(bucket, key, data, content_type="text/plain; charset=utf-8")

    def upload_from_url(self, url: str, bucket: str, key: str) -> dict:
        """Baixa arquivo de uma URL e faz upload para o Minio."""
        import urllib.request

        try:
            logger.info(f"Baixando de {url}...")

            # Baixar o arquivo
            req = urllib.request.Request(url, headers={'User-Agent': 'RLM-MCP-Server/1.0'})
            with urllib.request.urlopen(req, timeout=300) as response:
                data = response.read()
                content_type = response.headers.get('Content-Type', 'application/octet-stream')

            logger.info(f"Baixado {self._human_size(len(data))} de {url}")

            # Fazer upload para o Minio
            return self.put_object(bucket, key, data, content_type=content_type.split(';')[0])

        except Exception as e:
            logger.error(f"Erro ao baixar de {url}: {e}")
            raise RuntimeError(f"Erro ao baixar de {url}: {e}")

    def get_object_info(self, bucket: str, key: str) -> Optional[dict]:
        """Retorna informações do objeto sem baixar."""
        try:
            stat = self.client.stat_object(bucket, key)
            return {
                "bucket": bucket,
                "key": key,
                "size": stat.size,
                "size_human": self._human_size(stat.size),
                "content_type": stat.content_type,
                "last_modified": stat.last_modified.isoformat() if stat.last_modified else None,
                "etag": stat.etag
            }
        except Exception as e:
            logger.error(f"Erro ao obter info de {bucket}/{key}: {e}")
            return None

    def get_presigned_put_url(self, bucket: str, key: str, expires: int = 3600) -> str:
        """
        Gera URL assinada para upload (PUT) de arquivo.

        Args:
            bucket: Nome do bucket
            key: Caminho/chave do objeto
            expires: Tempo de expiração em segundos (padrão: 1 hora)

        Returns:
            URL assinada para upload via HTTP PUT
        """
        from datetime import timedelta

        try:
            url = self.client.presigned_put_object(
                bucket,
                key,
                expires=timedelta(seconds=expires)
            )
            logger.info(f"URL de upload gerada para {bucket}/{key}")
            return url
        except Exception as e:
            logger.error(f"Erro ao gerar URL de upload: {e}")
            raise RuntimeError(f"Erro ao gerar URL de upload: {e}")

    def get_presigned_get_url(self, bucket: str, key: str, expires: int = 3600) -> str:
        """
        Gera URL assinada para download (GET) de arquivo.

        Args:
            bucket: Nome do bucket
            key: Caminho/chave do objeto
            expires: Tempo de expiração em segundos (padrão: 1 hora)

        Returns:
            URL assinada para download via HTTP GET
        """
        from datetime import timedelta

        try:
            url = self.client.presigned_get_object(
                bucket,
                key,
                expires=timedelta(seconds=expires)
            )
            logger.info(f"URL de download gerada para {bucket}/{key}")
            return url
        except Exception as e:
            logger.error(f"Erro ao gerar URL de download: {e}")
            raise RuntimeError(f"Erro ao gerar URL de download: {e}")

    def batch_get_objects(
        self,
        bucket: str,
        keys: list[str],
        max_workers: int = 0,
        progress_callback: Optional[Callable] = None,
    ) -> list[dict]:
        """Download multiple objects in parallel.

        Args:
            bucket: Bucket name
            keys: List of object keys to download
            max_workers: Max parallel workers (0 = use env default)
            progress_callback: Optional callback(progress: float, message: str)

        Returns:
            List of dicts with {key, data, size, size_human, error}
        """
        workers = max_workers or BATCH_MAX_WORKERS
        results = []
        completed = 0
        total = len(keys)

        def _get_one(key: str) -> dict:
            try:
                data = self.get_object(bucket, key)
                return {
                    "key": key,
                    "data": data,
                    "size": len(data),
                    "size_human": self._human_size(len(data)),
                    "error": None,
                }
            except Exception as e:
                logger.warning(f"Batch get failed for {bucket}/{key}: {e}")
                return {
                    "key": key,
                    "data": None,
                    "size": 0,
                    "size_human": "0 B",
                    "error": str(e),
                }

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_key = {executor.submit(_get_one, k): k for k in keys}
            for future in as_completed(future_to_key):
                result = future.result()
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(
                        completed / total,
                        f"downloaded {completed}/{total}: {result['key']}"
                    )

        # Preserve original order
        key_order = {k: i for i, k in enumerate(keys)}
        results.sort(key=lambda r: key_order.get(r["key"], 0))
        return results

    def batch_put_objects(
        self,
        bucket: str,
        items: list[dict],
        max_workers: int = 0,
        progress_callback: Optional[Callable] = None,
    ) -> list[dict]:
        """Upload multiple objects in parallel.

        Args:
            bucket: Bucket name
            items: List of dicts with {key, data (bytes), content_type?}
            max_workers: Max parallel workers (0 = use env default)
            progress_callback: Optional callback(progress: float, message: str)

        Returns:
            List of dicts with {key, size, size_human, etag, error}
        """
        workers = max_workers or BATCH_MAX_WORKERS
        results = []
        completed = 0
        total = len(items)

        def _put_one(item: dict) -> dict:
            key = item["key"]
            try:
                data = item["data"]
                ct = item.get("content_type", "application/octet-stream")
                result = self.put_object(bucket, key, data, content_type=ct)
                return {
                    "key": key,
                    "size": result["size"],
                    "size_human": result["size_human"],
                    "etag": result.get("etag"),
                    "error": None,
                }
            except Exception as e:
                logger.warning(f"Batch put failed for {bucket}/{key}: {e}")
                return {
                    "key": key,
                    "size": 0,
                    "size_human": "0 B",
                    "etag": None,
                    "error": str(e),
                }

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_key = {executor.submit(_put_one, it): it["key"] for it in items}
            for future in as_completed(future_to_key):
                result = future.result()
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(
                        completed / total,
                        f"uploaded {completed}/{total}: {result['key']}"
                    )

        # Preserve original order
        key_order = {it["key"]: i for i, it in enumerate(items)}
        results.sort(key=lambda r: key_order.get(r["key"], 0))
        return results

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Converte bytes para formato humano."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"


# Instância global (singleton)
_s3_client: Optional[S3Client] = None


def get_s3_client() -> S3Client:
    """Retorna instância singleton do cliente S3."""
    global _s3_client
    if _s3_client is None:
        _s3_client = S3Client()
    return _s3_client
