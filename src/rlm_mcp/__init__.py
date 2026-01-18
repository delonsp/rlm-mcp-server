"""
RLM MCP Server - Recursive Language Model via Model Context Protocol

Este servidor MCP permite que o Claude Code execute código Python em um
REPL persistente, mantendo variáveis em memória sem poluir o contexto.

Arquitetura baseada no paper "Recursive Language Models" (MIT CSAIL, 2025)
"""

__version__ = "0.1.0"
