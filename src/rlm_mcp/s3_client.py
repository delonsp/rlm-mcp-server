"""
Cliente S3/Minio para RLM MCP Server.

Permite carregar arquivos grandes diretamente do Minio
sem passar pelo contexto do Claude Code.
"""

import os
import logging
from typing import Optional
from io import BytesIO

logger = logging.getLogger("rlm-mcp.s3")


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
        import mimetypes

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
