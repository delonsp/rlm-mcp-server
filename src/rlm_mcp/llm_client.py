"""
Cliente LLM para sub-chamadas recursivas no RLM.

Permite que código Python executado no REPL faça chamadas a LLMs
para processar chunks de dados, implementando o padrão do paper
"Recursive Language Models" (MIT CSAIL, 2025).
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("rlm-mcp.llm")


class LLMClient:
    """Cliente para sub-chamadas LLM de dentro do REPL."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.default_model = os.getenv("RLM_SUB_MODEL", "gpt-4o-mini")
        self.max_calls = int(os.getenv("RLM_MAX_SUB_CALLS", "100"))
        self.call_count = 0
        self._client = None

        if not self.api_key:
            logger.warning(
                "OPENAI_API_KEY não configurada. llm_query() não funcionará."
            )

    @property
    def client(self):
        """Lazy initialization do cliente OpenAI."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise RuntimeError(
                    "Pacote 'openai' não instalado. "
                    "Execute: pip install openai"
                )
        return self._client

    def query(
        self,
        prompt: str,
        data: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0
    ) -> str:
        """
        Executa uma sub-chamada LLM.

        Args:
            prompt: Instrução para o LLM
            data: Dados opcionais para processar
            model: Modelo a usar (default: gpt-4o-mini)
            max_tokens: Máximo de tokens na resposta
            temperature: Temperatura (0.0 = determinístico)

        Returns:
            Resposta do LLM como string

        Raises:
            RuntimeError: Se limite de chamadas atingido ou API key não configurada
        """
        # Validações
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY não configurada. "
                "Configure no .env do servidor RLM."
            )

        if self.call_count >= self.max_calls:
            raise RuntimeError(
                f"Limite de {self.max_calls} sub-chamadas LLM atingido nesta execução. "
                f"Use llm_reset_counter() para resetar ou aumente RLM_MAX_SUB_CALLS."
            )

        # Incrementa contador
        self.call_count += 1

        # Prepara modelo
        model = model or self.default_model

        # Prepara conteúdo
        content = prompt
        if data:
            # Limita tamanho dos dados para evitar custos excessivos
            max_data_chars = 100000  # ~25k tokens
            if len(data) > max_data_chars:
                logger.warning(
                    f"Dados truncados de {len(data)} para {max_data_chars} caracteres"
                )
                data = data[:max_data_chars] + "\n... [TRUNCADO]"

            content = f"{prompt}\n\n---\nDATA:\n{data}"

        logger.info(
            f"Sub-chamada LLM #{self.call_count}: model={model}, "
            f"prompt_len={len(prompt)}, data_len={len(data) if data else 0}"
        )

        try:
            # Modelos mais novos (gpt-5, o1, etc) usam max_completion_tokens
            # Modelos antigos (gpt-4o, gpt-4o-mini) usam max_tokens
            is_new_model = any(x in model for x in ['gpt-5', 'o1', 'o3'])

            params = {
                "model": model,
                "temperature": temperature,
                "messages": [{"role": "user", "content": content}]
            }

            if is_new_model:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**params)

            result = response.choices[0].message.content
            logger.info(f"Sub-chamada LLM #{self.call_count} concluída: {len(result)} chars")

            return result

        except Exception as e:
            logger.error(f"Erro na sub-chamada LLM: {e}")
            raise RuntimeError(f"Erro ao chamar LLM: {e}")

    def reset_counter(self):
        """Reseta o contador de chamadas."""
        old_count = self.call_count
        self.call_count = 0
        logger.info(f"Contador de sub-chamadas resetado (era {old_count})")

    def get_stats(self) -> dict:
        """Retorna estatísticas de uso."""
        return {
            "calls_made": self.call_count,
            "max_calls": self.max_calls,
            "remaining": self.max_calls - self.call_count,
            "model": self.default_model,
            "api_configured": bool(self.api_key)
        }
