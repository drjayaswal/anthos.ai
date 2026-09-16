import asyncio
import contextvars
from datetime import datetime, timezone
import logging
import re
import sys
from typing import Optional

current_log_queue: contextvars.ContextVar[Optional[asyncio.Queue]] = contextvars.ContextVar(
    "current_log_queue", default=None
)

_NOISE_PATTERNS = [
    "http request:",
    "acquired llm semaphore",
    "websocket /analyse accepted",
    "websocket /analyze accepted",
    "confirmed analysis",
    "delivered results",
    "client disconnected",
    "afc is enabled",
    "automatic function calling",
    "uvicorn",
    "httpx",
    "httpcore",
]


def _is_noise(lower: str) -> bool:
    return any(pattern in lower for pattern in _NOISE_PATTERNS)


def parse_log_entry(raw_msg: str) -> tuple[str, str, str] | None:
    """
    Parses a raw log message into (stage, tag, concise_message).
    Returns None for internal/noisy messages that should not be forwarded.
    concise_message is guaranteed to be <= 5 words.
    """
    lower = raw_msg.lower()

    if _is_noise(lower):
        return None

    if "cleaning email" in lower:
        return "clean", "CLEAN", "Cleaning email content"
    if "failed to clean" in lower:
        return "error", "ERROR", "Email cleaning failed"

    if "running regex" in lower:
        return "regex", "REGEX", "Running regex categorization"

    if "initialized classification llm" in lower:
        return "llm", "LLM", "Classification model ready"
    if "initialized supervisor llm" in lower:
        return "supervisor", "REVIEW", "Supervisor model ready"

    if "supervisor approved" in lower:
        return "supervisor", "APPROVED", "Supervisor approved category"
    if "supervisor rejected" in lower:
        return "retry", "RETRY", "Supervisor rejected category"
    if "max retries without approval" in lower or "max retries" in lower:
        return "retry", "RETRY", "Max retries reached"
    if "reclassifying email" in lower or "reclassifying" in lower:
        return "retry", "RETRY", "Reclassifying rejected email"
    if "running supervisor review" in lower or "semaphore for supervisor" in lower:
        return "supervisor", "REVIEW", "Running supervisor review"
    if "supervisor review failed" in lower:
        return "error", "ERROR", "Supervisor review failed"

    if "categorized as" in lower:
        match = re.search(r"categorized as ['\"]?([^'\",\n\)]+)['\"]?", raw_msg, re.IGNORECASE)
        if match:
            cat_name = match.group(1).strip()
            words = ("Categorized as " + cat_name).split()
            if len(words) <= 5:
                return "llm", "LLM", " ".join(words)
            return "llm", "LLM", " ".join(words[:5])
        return "llm", "LLM", "Email categorized by LLM"
    if "running llm categorization" in lower or "classification llm" in lower or "sending to llm" in lower:
        return "llm", "LLM", "Running LLM categorization"
    if "llm categorization failed" in lower:
        return "error", "ERROR", "LLM categorization failed"

    if "starting analysis" in lower:
        return "info", "START", "Starting email analysis"

    if "analysed" in lower:
        return "info", "INFO", raw_msg.strip()

    if "total time:" in lower or "total time" in lower:
        return "info", "TIME", raw_msg.strip()

    if "error" in lower or "failed" in lower:
        words = raw_msg.strip().split()
        if len(words) <= 5:
            return "error", "ERROR", " ".join(words)
        return "error", "ERROR", "Analysis execution failed"

    words = raw_msg.strip().split()
    if len(words) <= 5:
        return "info", "INFO", " ".join(words)
    return "info", "INFO", " ".join(words[:5])


class WebSocketLogHandler(logging.Handler):
    """
    Logging handler that intercepts log records and pushes them into the
    asyncio.Queue bound to the current request via current_log_queue.

    When no queue is set (e.g. HTTP requests, startup logs), this handler
    is a no-op — the record is only handled by the normal StreamHandler.
    """

    def emit(self, record: logging.LogRecord) -> None:
        queue = current_log_queue.get()
        if queue is None:
            return
        try:
            raw_msg = record.getMessage()
            result = parse_log_entry(raw_msg)
            if result is None:
                return
            stage, tag, clean_msg = result
            log_item = {
                "type": "log",
                "status": "processing",
                "level": record.levelname,
                "tag": tag,
                "stage": stage,
                "message": clean_msg,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            queue.put_nowait(log_item)
        except Exception:
            pass


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger once, at application startup.

    Every other module just does `logger = logging.getLogger(__name__)`
    and inherits this configuration - no per-file setup needed.
    """
    root_logger = logging.getLogger()

    if root_logger.level == logging.NOTSET or root_logger.level > level:
        root_logger.setLevel(level)

    has_ws_handler = any(isinstance(h, WebSocketLogHandler) for h in root_logger.handlers)
    if not has_ws_handler:
        ws_handler = WebSocketLogHandler()
        ws_handler.setLevel(level)
        root_logger.addHandler(ws_handler)

    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, WebSocketLogHandler)
        for h in root_logger.handlers
    )
    if not has_stream_handler:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
