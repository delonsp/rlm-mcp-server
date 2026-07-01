"""
Async Task Manager for RLM MCP Server.

Manages long-running operations (PDF processing, large file operations)
using ThreadPoolExecutor. Provides task tracking with progress updates.
"""

import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Any

logger = logging.getLogger("rlm-mcp.tasks")


@dataclass
class TaskInfo:
    """Information about an async task."""
    task_id: str
    tool_name: str
    description: str
    status: str = "pending"  # pending|running|completed|failed|cancelled
    progress: float = 0.0  # 0.0-1.0
    progress_message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict] = None  # MCP result when complete
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        d = {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "description": self.description,
            "status": self.status,
            "progress": round(self.progress, 2),
            "progress_message": self.progress_message,
            "created_at": self.created_at.isoformat(),
        }
        if self.started_at:
            d["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            d["completed_at"] = self.completed_at.isoformat()
            elapsed = (self.completed_at - (self.started_at or self.created_at)).total_seconds()
            d["duration_seconds"] = round(elapsed, 2)
        if self.error:
            d["error"] = self.error
        return d


class TaskManager:
    """Manages async tasks with ThreadPoolExecutor.

    Thread-safe task submission, status tracking, and cancellation.
    """

    def __init__(self, max_concurrent: int = 3):
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskInfo] = {}
        self._futures: dict[str, Future] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="rlm-task"
        )
        self._max_concurrent = max_concurrent
        logger.info(f"TaskManager initialized (max_concurrent={max_concurrent})")

    def _generate_id(self) -> str:
        """Generate a short task ID."""
        return uuid.uuid4().hex[:8]

    def submit(
        self,
        tool_name: str,
        description: str,
        func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> TaskInfo:
        """Submit a task for async execution.

        Args:
            tool_name: Name of the originating tool
            description: Human-readable description
            func: Callable to execute. Will receive a progress_callback kwarg.
            *args, **kwargs: Arguments to pass to func

        Returns:
            TaskInfo for the submitted task
        """
        task_id = self._generate_id()
        task = TaskInfo(
            task_id=task_id,
            tool_name=tool_name,
            description=description,
            status="pending",
        )

        with self._lock:
            self._tasks[task_id] = task

        def progress_callback(progress: float, message: str = ""):
            """Update task progress (called from worker thread)."""
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id].progress = min(max(progress, 0.0), 1.0)
                    if message:
                        self._tasks[task_id].progress_message = message

        def wrapper():
            with self._lock:
                task.status = "running"
                task.started_at = datetime.now()

            try:
                result = func(*args, progress_callback=progress_callback, **kwargs)
                with self._lock:
                    # Não sobrescrever um cancel que o usuário já viu: future.cancel()
                    # não interrompe uma thread em execução, então sem esta guarda o
                    # status "cancelled" era revertido p/ "completed" ao terminar.
                    if task.status != "cancelled":
                        task.status = "completed"
                        task.completed_at = datetime.now()
                        task.progress = 1.0
                        task.result = result
                        task.progress_message = "done"
                logger.info(f"Task {task_id} completed ({tool_name})")
            except Exception as e:
                with self._lock:
                    if task.status != "cancelled":
                        task.status = "failed"
                        task.completed_at = datetime.now()
                        task.error = str(e)
                        task.progress_message = f"error: {str(e)[:100]}"
                logger.exception(f"Task {task_id} failed ({tool_name}): {e}")

        future = self._executor.submit(wrapper)
        with self._lock:
            self._futures[task_id] = future

        logger.info(f"Task {task_id} submitted ({tool_name}: {description})")
        return task

    def get_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get task status by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[str] = None) -> list[TaskInfo]:
        """List all tasks, optionally filtered by status."""
        with self._lock:
            tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def cancel(self, task_id: str) -> bool:
        """Cancel a task. Returns True if cancelled, False if not found or already done."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            if task.status in ("completed", "failed", "cancelled"):
                return False

            future = self._futures.get(task_id)
            if future and not future.done():
                future.cancel()

            task.status = "cancelled"
            task.completed_at = datetime.now()
            task.progress_message = "cancelled by user"
            return True

    def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """Remove completed/failed/cancelled tasks older than max_age."""
        now = datetime.now()
        to_remove = []

        with self._lock:
            for task_id, task in self._tasks.items():
                if task.status in ("completed", "failed", "cancelled"):
                    if task.completed_at:
                        age = (now - task.completed_at).total_seconds()
                        if age > max_age_seconds:
                            to_remove.append(task_id)

            for task_id in to_remove:
                del self._tasks[task_id]
                self._futures.pop(task_id, None)

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old tasks")
        return len(to_remove)

    @property
    def active_count(self) -> int:
        """Count of currently running tasks."""
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status == "running")

    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        self._executor.shutdown(wait=wait)
        logger.info("TaskManager shutdown")
