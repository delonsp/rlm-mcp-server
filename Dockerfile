# RLM MCP Server - Dockerfile otimizado para produção
# Multi-stage build para imagem final pequena

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Cache buster - mude para forçar rebuild
ARG CACHE_BUST=2026060201

# Instala dependências de build (uv p/ exportar o lockfile)
RUN pip install --no-cache-dir hatchling uv

# Copia arquivos do projeto (uv.lock incluído p/ build reprodutível)
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Build do wheel do app + exporta as deps TRAVADAS do uv.lock. Antes o runtime
# resolvia todas as deps do PyPI no build (versões flutuantes → OCR podia pular
# p/ mistralai 2.x sem aviso). Agora as versões vêm exatas do lockfile.
RUN echo "Build version: ${CACHE_BUST}" \
    && pip wheel --no-deps --wheel-dir /wheels . \
    && uv export --frozen --no-dev --no-emit-project --no-hashes -o /wheels/requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.12-slim AS runtime

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

# Cria diretórios de dados e persistência
RUN mkdir -p /data /persist && chown rlm:rlm /data /persist

# Cache buster para runtime (deve ser igual ao do builder)
ARG CACHE_BUST=2026060201

# Instala deps travadas (requirements.txt do uv.lock) e DEPOIS o app sem re-resolver
COPY --from=builder /wheels /wheels
RUN echo "Runtime version: ${CACHE_BUST}" \
    && pip install --no-cache-dir -r /wheels/requirements.txt \
    && pip install --no-cache-dir --no-deps /wheels/*.whl \
    && rm -rf /wheels

# Variáveis de ambiente padrão
ENV RLM_MAX_MEMORY_MB=1024
ENV RLM_API_KEY=""
ENV MISTRAL_API_KEY=""
ENV PYTHONUNBUFFERED=1
# Cosmético (NÃO é medida de segurança): evita tentativas de escrita de .pyc que,
# pós-Landlock (lockdown B2), dariam EPERM benigno em /usr.
ENV PYTHONDONTWRITEBYTECODE=1

# Expõe porta (não necessária para MCP stdio, mas útil para health checks)
EXPOSE 8080

# Entrypoint que corrige permissões do volume e roda como rlm
RUN printf '#!/bin/sh\nchown -R rlm:rlm /persist /data 2>/dev/null; exec su -s /bin/sh rlm -c "exec $*" -- "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Health check via HTTP
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "rlm_mcp.http_server"]
