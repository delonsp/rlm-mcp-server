#!/bin/bash
# Script para conectar ao RLM MCP Server via SSH tunnel
#
# Uso: ./connect.sh [user@servidor]
#
# Exemplo:
#   ./connect.sh root@159.223.140.36

set -e

# Configurações padrão
DEFAULT_SERVER="root@159.223.140.36"
LOCAL_PORT=8765
REMOTE_PORT=8765

SERVER="${1:-$DEFAULT_SERVER}"

echo "=== RLM MCP Server - Conexão SSH ==="
echo ""
echo "Servidor: $SERVER"
echo "Porta local: $LOCAL_PORT"
echo ""

# Verifica se já existe um túnel
if lsof -i :$LOCAL_PORT > /dev/null 2>&1; then
    echo "! Túnel já existe na porta $LOCAL_PORT"
    echo "  Para reconectar, primeiro mate o processo:"
    echo "  lsof -ti :$LOCAL_PORT | xargs kill"
    exit 1
fi

echo "Criando túnel SSH..."
ssh -L $LOCAL_PORT:localhost:$REMOTE_PORT "$SERVER" -N -f

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Túnel criado com sucesso! ==="
    echo ""
    echo "O Claude Code agora pode usar o MCP server."
    echo ""
    echo "Para testar a conexão:"
    echo "  nc -zv localhost $LOCAL_PORT"
    echo ""
    echo "Para fechar o túnel:"
    echo "  lsof -ti :$LOCAL_PORT | xargs kill"
else
    echo "Erro ao criar túnel SSH"
    exit 1
fi
