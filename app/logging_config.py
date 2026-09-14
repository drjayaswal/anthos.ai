import asyncio
import contextvars
from datetime import datetime, timezone
import logging
import sys
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Context variable: holds an asyncio.Queue for the active WebSocket request.
# Each WebSocket request sets its own queue; all logger.info/warning/error
# calls from any module during that request get captured and forwarded.
# ─────────────────────────────────────────────────────────────────────────────
current_log_queue: contextvars.ContextVar[Optional[asyncio.Queue]] = contextvars.ContextVar(
    "current_log_queue", default=None
)


def parse_log_entry(raw_msg: str) -> tuple[str, str, str]:
    """
    Parses a raw log message into:
      (stage, tag, concise_message)
    Where concise_message is meaningful and guaranteed to be <= 5 words.
    """
    lower = raw_msg.lower()

    # 1. Cleaning
    if "cleaning email" in lower:
        return "clean", "CLEAN", "Cleaning email content"
    if "failed to clean" in lower:
        return "error", "ERROR", "Email cleaning failed"

    # 2. Regex
    if "running regex" in lower:
        return "regex", "REGEX", "Running regex categorization"

    # 3. Model Initializations
    if "initialized classification llm" in lower:
        return "llm", "LLM", "Classification model ready"
    if "initialized supervisor llm" in lower:
        return "supervisor", "REVIEW", "Supervisor model ready"

    # 4. Supervisor Review & Decisions
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

    # 5. LLM Categorization
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

    # 6. Workflow / Completion
    if "completed" in lower or "finished analysis" in lower or "total analysis time" in lower:
        return "complete", "DONE", "Analysis completed successfully"
    if "delivered results" in lower:
        return "complete", "DONE", "Delivered analysis results"
    if "confirmed analysis" in lower:
        return "info", "START", "Analysis request confirmed"
    if "starting analysis" in lower:
        return "info", "START", "Starting email analysis"
    if "websocket /analyse accepted" in lower:
        return "info", "CONNECT", "Connected to stream"

    # 7. Errors
    if "error" in lower or "failed" in lower:
        words = raw_msg.strip().split()
        if len(words) <= 5:
            return "error", "ERROR", " ".join(words)
        return "error", "ERROR", "Analysis execution failed"

    # 8. Fallback
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
            if "afc is enabled" in raw_msg.lower() or "automatic function calling" in raw_msg.lower():
                return
            stage, tag, clean_msg = parse_log_entry(raw_msg)
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
            # Never let a logging error crash the application
            pass


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger once, at application startup.

    Every other module just does `logger = logging.getLogger(__name__)`
    and inherits this configuration - no per-file setup needed.
    """
    root_logger = logging.getLogger()

    # Ensure root logger captures at least INFO level messages
    if root_logger.level == logging.NOTSET or root_logger.level > level:
        root_logger.setLevel(level)

    # Attach WebSocketLogHandler (only once — idempotent)
    has_ws_handler = any(isinstance(h, WebSocketLogHandler) for h in root_logger.handlers)
    if not has_ws_handler:
        ws_handler = WebSocketLogHandler()
        ws_handler.setLevel(level)
        root_logger.addHandler(ws_handler)

    # Standard console handler — keeps printing to stdout (only once)
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