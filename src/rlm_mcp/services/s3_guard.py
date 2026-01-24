"""S3/MinIO configuration guard for RLM MCP Server."""

from ..s3_client import get_s3_client


def require_s3_configured() -> tuple:
    """Verifica se S3 está configurado. Retorna (client, error_response).

    Returns:
        tuple: (s3_client, None) se configurado, (None, error_dict) se não configurado
    """
    s3 = get_s3_client()
    if not s3.is_configured():
        return None, {
            "content": [{"type": "text", "text": "Erro: Minio não configurado. Configure MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY."}],
            "isError": True
        }
    return s3, None
