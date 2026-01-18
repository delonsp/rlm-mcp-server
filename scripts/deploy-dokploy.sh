#!/bin/bash
# Script para deploy no servidor via Dokploy ou Docker direto
#
# Uso: ./deploy-dokploy.sh [user@servidor]

set -e

DEFAULT_SERVER="root@159.223.140.36"
SERVER="${1:-$DEFAULT_SERVER}"
REMOTE_DIR="/opt/rlm-mcp-server"

echo "=== Deploy RLM MCP Server ==="
echo ""
echo "Servidor: $SERVER"
echo "Diretório: $REMOTE_DIR"
echo ""

# Cria diretório remoto
echo "1. Criando diretório no servidor..."
ssh "$SERVER" "mkdir -p $REMOTE_DIR/data"

# Copia arquivos
echo "2. Copiando arquivos..."
rsync -avz --exclude '.git' --exclude 'venv' --exclude '__pycache__' \
    ./ "$SERVER:$REMOTE_DIR/"

# Build e start
echo "3. Build e start do container..."
ssh "$SERVER" "cd $REMOTE_DIR && docker-compose down || true && docker-compose up -d --build"

# Verifica status
echo ""
echo "4. Verificando status..."
ssh "$SERVER" "docker ps | grep rlm-mcp"

echo ""
echo "=== Deploy concluído! ==="
echo ""
echo "O servidor está rodando em $SERVER:8765"
echo ""
echo "Para conectar do Claude Code local:"
echo "  ./scripts/connect.sh $SERVER"
