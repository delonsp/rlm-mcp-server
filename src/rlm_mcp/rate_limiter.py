"""
Rate Limiter - Sliding Window Algorithm

Implementa rate limiting para proteger o servidor de abusos.
Usa algoritmo de sliding window para contagem precisa de requisições.
"""

import time
import logging
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("rlm-rate-limiter")


@dataclass
class RateLimitConfig:
    """Configuração para um limite de taxa específico."""
    max_requests: int
    window_seconds: int

    def __post_init__(self):
        if self.max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")


@dataclass
class RateLimitResult:
    """Resultado de uma verificação de rate limit."""
    allowed: bool
    current_count: int
    limit: int
    window_seconds: int
    retry_after: Optional[float] = None  # Segundos até poder fazer próxima requisição


class SlidingWindowRateLimiter:
    """
    Rate limiter usando algoritmo de sliding window.

    O algoritmo de sliding window oferece um compromisso entre precisão
    e uso de memória. Em vez de manter timestamps individuais, divide
    o tempo em buckets e interpola entre a janela atual e a anterior.

    Args:
        max_requests: Número máximo de requisições permitidas na janela
        window_seconds: Tamanho da janela em segundos

    Example:
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)

        result = limiter.check("session_123")
        if not result.allowed:
            # Retornar 429 Too Many Requests
            pass
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.config = RateLimitConfig(max_requests, window_seconds)
        # {identifier: [(timestamp, count), ...]} - histórico de requisições por bucket
        self._buckets: Dict[str, List[tuple]] = defaultdict(list)
        # Tamanho de cada bucket em segundos (granularidade)
        self._bucket_size = max(1, window_seconds // 10)  # 10 buckets por janela

    def _get_current_bucket(self, now: float) -> int:
        """Retorna o ID do bucket atual baseado no timestamp."""
        return int(now // self._bucket_size)

    def _cleanup_old_buckets(self, identifier: str, now: float) -> None:
        """Remove buckets antigos que estão fora da janela."""
        cutoff = now - self.config.window_seconds - self._bucket_size
        self._buckets[identifier] = [
            (ts, count) for ts, count in self._buckets[identifier]
            if ts > cutoff
        ]

    def _count_requests_in_window(self, identifier: str, now: float) -> int:
        """
        Conta requisições na janela atual usando sliding window.

        Usa interpolação entre a janela anterior e atual para
        obter uma contagem mais precisa.
        """
        window_start = now - self.config.window_seconds
        total = 0

        for bucket_time, count in self._buckets[identifier]:
            bucket_end = bucket_time + self._bucket_size

            if bucket_end <= window_start:
                # Bucket completamente fora da janela
                continue
            elif bucket_time >= window_start:
                # Bucket completamente dentro da janela
                total += count
            else:
                # Bucket parcialmente na janela - interpolar
                overlap = bucket_end - window_start
                fraction = overlap / self._bucket_size
                total += int(count * fraction)

        return total

    def check(self, identifier: str, now: Optional[float] = None) -> RateLimitResult:
        """
        Verifica se uma requisição é permitida para o identificador.

        NÃO incrementa o contador - use record() após processar a requisição.

        Args:
            identifier: ID único (ex: session_id, IP, user_id)
            now: Timestamp atual (para testes). Se None, usa time.time()

        Returns:
            RateLimitResult indicando se requisição é permitida
        """
        if now is None:
            now = time.time()

        self._cleanup_old_buckets(identifier, now)
        current_count = self._count_requests_in_window(identifier, now)

        allowed = current_count < self.config.max_requests

        retry_after = None
        if not allowed:
            # Calcular quando o limite será liberado
            # Encontrar o bucket mais antigo na janela
            window_start = now - self.config.window_seconds
            oldest_in_window = None
            for bucket_time, _ in self._buckets[identifier]:
                if bucket_time >= window_start:
                    if oldest_in_window is None or bucket_time < oldest_in_window:
                        oldest_in_window = bucket_time

            if oldest_in_window is not None:
                retry_after = oldest_in_window + self.config.window_seconds - now + self._bucket_size
                retry_after = max(0.1, retry_after)  # Mínimo 0.1 segundos

        return RateLimitResult(
            allowed=allowed,
            current_count=current_count,
            limit=self.config.max_requests,
            window_seconds=self.config.window_seconds,
            retry_after=retry_after
        )

    def record(self, identifier: str, now: Optional[float] = None) -> None:
        """
        Registra uma requisição para o identificador.

        Deve ser chamado APÓS processar a requisição (ou após check() retornar allowed=True).

        Args:
            identifier: ID único (ex: session_id, IP, user_id)
            now: Timestamp atual (para testes). Se None, usa time.time()
        """
        if now is None:
            now = time.time()

        current_bucket = self._get_current_bucket(now)
        bucket_time = current_bucket * self._bucket_size

        # Encontrar ou criar bucket atual
        found = False
        for i, (ts, count) in enumerate(self._buckets[identifier]):
            if ts == bucket_time:
                self._buckets[identifier][i] = (ts, count + 1)
                found = True
                break

        if not found:
            self._buckets[identifier].append((bucket_time, 1))

        logger.debug(f"Rate limit recorded for {identifier}: bucket={bucket_time}")

    def check_and_record(self, identifier: str, now: Optional[float] = None) -> RateLimitResult:
        """
        Verifica e registra em uma única operação.

        Atalho para check() + record() se permitido.

        Args:
            identifier: ID único (ex: session_id, IP, user_id)
            now: Timestamp atual (para testes). Se None, usa time.time()

        Returns:
            RateLimitResult indicando se requisição é permitida
        """
        if now is None:
            now = time.time()

        result = self.check(identifier, now)
        if result.allowed:
            self.record(identifier, now)
            # Atualizar contagem no resultado
            result = RateLimitResult(
                allowed=True,
                current_count=result.current_count + 1,
                limit=self.config.max_requests,
                window_seconds=self.config.window_seconds,
                retry_after=None
            )
        return result

    def reset(self, identifier: str) -> None:
        """
        Remove todos os registros para um identificador.

        Útil para testes ou quando uma sessão é encerrada.
        """
        if identifier in self._buckets:
            del self._buckets[identifier]
            logger.debug(f"Rate limit reset for {identifier}")

    def get_stats(self, identifier: str, now: Optional[float] = None) -> Dict:
        """
        Retorna estatísticas de uso para um identificador.

        Args:
            identifier: ID único
            now: Timestamp atual (para testes)

        Returns:
            Dict com current_count, limit, remaining, reset_at
        """
        if now is None:
            now = time.time()

        self._cleanup_old_buckets(identifier, now)
        current_count = self._count_requests_in_window(identifier, now)

        return {
            "current_count": current_count,
            "limit": self.config.max_requests,
            "remaining": max(0, self.config.max_requests - current_count),
            "window_seconds": self.config.window_seconds,
            "reset_at": now + self.config.window_seconds
        }


class MultiRateLimiter:
    """
    Gerencia múltiplos rate limiters para diferentes recursos.

    Permite configurar limites diferentes para diferentes endpoints
    ou tipos de operação.

    Example:
        limiter = MultiRateLimiter()
        limiter.add_limit("requests", max_requests=100, window_seconds=60)
        limiter.add_limit("uploads", max_requests=10, window_seconds=60)

        # Verificar limite de requests
        result = limiter.check("requests", "session_123")

        # Verificar limite de uploads
        result = limiter.check("uploads", "session_123")
    """

    def __init__(self):
        self._limiters: Dict[str, SlidingWindowRateLimiter] = {}

    def add_limit(self, name: str, max_requests: int, window_seconds: int) -> None:
        """
        Adiciona um novo limite.

        Args:
            name: Nome do limite (ex: "requests", "uploads")
            max_requests: Número máximo de requisições
            window_seconds: Janela de tempo em segundos
        """
        self._limiters[name] = SlidingWindowRateLimiter(max_requests, window_seconds)
        logger.info(f"Rate limit '{name}' configured: {max_requests} requests per {window_seconds}s")

    def check(self, limit_name: str, identifier: str, now: Optional[float] = None) -> RateLimitResult:
        """
        Verifica um limite específico.

        Args:
            limit_name: Nome do limite configurado
            identifier: ID único do cliente
            now: Timestamp atual (para testes)

        Returns:
            RateLimitResult

        Raises:
            KeyError: Se o limite não existe
        """
        if limit_name not in self._limiters:
            raise KeyError(f"Rate limit '{limit_name}' not configured")
        return self._limiters[limit_name].check(identifier, now)

    def record(self, limit_name: str, identifier: str, now: Optional[float] = None) -> None:
        """Registra uma requisição para um limite específico."""
        if limit_name not in self._limiters:
            raise KeyError(f"Rate limit '{limit_name}' not configured")
        self._limiters[limit_name].record(identifier, now)

    def check_and_record(self, limit_name: str, identifier: str, now: Optional[float] = None) -> RateLimitResult:
        """Verifica e registra em uma única operação."""
        if limit_name not in self._limiters:
            raise KeyError(f"Rate limit '{limit_name}' not configured")
        return self._limiters[limit_name].check_and_record(identifier, now)

    def reset(self, limit_name: str, identifier: str) -> None:
        """Remove registros de um identificador para um limite específico."""
        if limit_name not in self._limiters:
            raise KeyError(f"Rate limit '{limit_name}' not configured")
        self._limiters[limit_name].reset(identifier)

    def reset_all(self, identifier: str) -> None:
        """Remove registros de um identificador para todos os limites."""
        for limiter in self._limiters.values():
            limiter.reset(identifier)

    def get_stats(self, limit_name: str, identifier: str, now: Optional[float] = None) -> Dict:
        """Retorna estatísticas de um limite específico."""
        if limit_name not in self._limiters:
            raise KeyError(f"Rate limit '{limit_name}' not configured")
        return self._limiters[limit_name].get_stats(identifier, now)

    def list_limits(self) -> List[str]:
        """Retorna lista de limites configurados."""
        return list(self._limiters.keys())
