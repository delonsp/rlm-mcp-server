"""
Driver ASGI cru para testar GET /sse.

POR QUE EXISTE: FastAPI TestClient e httpx.ASGITransport bufferizam a resposta
INTEIRA antes de retornar — o stream SSE é infinito (keepalive ': ping' a cada
30s) → o request deadlocka para sempre (provado empiricamente com faulthandler:
testclient.py handle_request preso num portal future enquanto o event loop roda
o generator infinito). Aqui chamamos a app ASGI diretamente e consumimos as
mensagens 'send' por uma Queue, com timeouts curtos.
"""
import asyncio


class SseDriver:
    """Abre GET /sse contra a app ASGI e expõe os chunks via Queue."""

    def __init__(self, app, headers: list | None = None):
        self.app = app
        self.chunks: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self._headers = [(b"host", b"testserver"), (b"accept", b"text/event-stream")]
        if headers:
            self._headers.extend(headers)

    async def start(self):
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/sse",
            "raw_path": b"/sse",
            "root_path": "",
            "query_string": b"",
            "headers": self._headers,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        async def receive():
            # Nunca desconecta por conta própria; o teste encerra via stop()
            await asyncio.sleep(3600)
            return {"type": "http.disconnect"}

        async def send(message):
            await self.chunks.put(message)

        self.task = asyncio.create_task(self.app(scope, receive, send))

    async def next_chunk(self, timeout: float = 5.0) -> dict:
        return await asyncio.wait_for(self.chunks.get(), timeout)

    async def status(self, timeout: float = 5.0) -> int:
        """Consome até o http.response.start e retorna o status."""
        while True:
            msg = await self.next_chunk(timeout)
            if msg["type"] == "http.response.start":
                return msg["status"]

    async def session_id(self, timeout: float = 5.0) -> str:
        """Consome chunks até o evento endpoint e extrai o session_id."""
        while True:
            msg = await self.next_chunk(timeout)
            if msg["type"] == "http.response.body":
                body = msg.get("body", b"").decode()
                if "session_id=" in body:
                    return body.split("session_id=")[1].split()[0].strip()

    async def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass
