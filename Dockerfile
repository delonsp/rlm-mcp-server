# RLM MCP Server - Dockerfile otimizado para produção
# Multi-stage build para imagem final pequena

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim as builder

WORKDIR /build

# Instala dependências de build
RUN pip install --no-cache-dir hatchling

# Copia arquivos do projeto
COPY pyproject.toml .
COPY src/ src/

# Build do wheel
RUN pip wheel --no-deps --wheel-dir /wheels .

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.12-slim as runtime

# Labels para Dokploy/Portainer
LABEL maintainer="seu-email@exemplo.com"
LABEL description="RLM MCP Server - Recursive Language Model via MCP"
LABEL version="0.1.0"

# Instala curl para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root para segurança
RUN groupadd -r rlm && useradd -r -g rlm rlm

# Diretório de trabalho
WORKDIR /app

# Cria diretório de dados
RUN mkdir -p /data && chown rlm:rlm /data

# Instala dependências do wheel
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Variáveis de ambiente padrão
ENV RLM_MAX_MEMORY_MB=1024
ENV RLM_API_KEY=""
ENV MISTRAL_API_KEY=""
ENV PYTHONUNBUFFERED=1

# Expõe porta (não necessária para MCP stdio, mas útil para health checks)
EXPOSE 8080

# Muda para usuário não-root
USER rlm

# Health check via HTTP
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Comando padrão - HTTP server
CMD ["python", "-m", "rlm_mcp.http_server"]
