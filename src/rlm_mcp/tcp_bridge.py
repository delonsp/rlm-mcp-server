"""
TCP Bridge para RLM MCP Server

Expõe o servidor MCP (que usa stdio) via TCP na porta 8765.
Permite conexão remota via SSH tunnel.
"""

import socket
import subprocess
import threading
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rlm-tcp-bridge")

PORT = int(os.getenv("RLM_PORT", "8765"))
HOST = os.getenv("RLM_HOST", "0.0.0.0")


def handle_client(conn: socket.socket, addr: tuple):
    """Gerencia uma conexão de cliente"""
    logger.info(f"Nova conexão de {addr}")

    try:
        # Inicia o processo MCP
        proc = subprocess.Popen(
            [sys.executable, "-m", "rlm_mcp.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0  # Unbuffered
        )

        stop_event = threading.Event()

        def forward_socket_to_proc():
            """Encaminha dados do socket para stdin do processo"""
            try:
                while not stop_event.is_set():
                    conn.settimeout(1.0)
                    try:
                        data = conn.recv(8192)
                        if not data:
                            break
                        proc.stdin.write(data)
                        proc.stdin.flush()
                    except socket.timeout:
                        continue
            except Exception as e:
                logger.debug(f"forward_socket_to_proc error: {e}")
            finally:
                stop_event.set()

        def forward_proc_to_socket():
            """Encaminha stdout do processo para o socket"""
            try:
                while not stop_event.is_set():
                    data = proc.stdout.read(8192)
                    if not data:
                        break
                    conn.sendall(data)
            except Exception as e:
                logger.debug(f"forward_proc_to_socket error: {e}")
            finally:
                stop_event.set()

        def log_stderr():
            """Loga stderr do processo"""
            try:
                for line in proc.stderr:
                    logger.warning(f"[MCP stderr] {line.decode().strip()}")
            except Exception:
                pass

        t1 = threading.Thread(target=forward_socket_to_proc, daemon=True)
        t2 = threading.Thread(target=forward_proc_to_socket, daemon=True)
        t3 = threading.Thread(target=log_stderr, daemon=True)

        t1.start()
        t2.start()
        t3.start()

        # Aguarda até uma das threads terminar
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
            if proc.poll() is not None:
                break

    except Exception as e:
        logger.error(f"Erro ao processar conexão: {e}")
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

        try:
            conn.close()
        except Exception:
            pass

        logger.info(f"Conexão de {addr} encerrada")


def main():
    """Entry point do TCP bridge"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((HOST, PORT))
        server.listen(5)
        logger.info(f"RLM MCP Server TCP Bridge escutando em {HOST}:{PORT}")

        while True:
            conn, addr = server.accept()
            thread = threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True
            )
            thread.start()

    except KeyboardInterrupt:
        logger.info("Encerrando servidor...")
    finally:
        server.close()


if __name__ == "__main__":
    main()
