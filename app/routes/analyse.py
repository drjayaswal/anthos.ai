import asyncio
from contextlib import contextmanager
import json
import logging
from typing import Generator

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from sqlalchemy.orm import Session
from starlette import status

from app.database.db import get_db
from app.database.models import Category, Model
from app.logging_config import current_log_queue, setup_logging
from app.schemas import AnalyseRequest, AnalyseResponse, CategoryDetail, ModelDetail
from app.services.src.agents import validate_model_config
from app.services.src.workflow import EmailWorkflow
from app.settings import get_settings

setup_logging()
logger = logging.getLogger(__name__)

router = APIRouter()
workflow = EmailWorkflow()
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Origin check against ANTHOSWEB_URL
# ─────────────────────────────────────────────────────────────────────────────
def is_origin_allowed(origin: str | None) -> bool:
    """
    Check whether the WebSocket client's Origin header is allowed.
    If origin is absent (CLI tools, tests), it is allowed.
    If present, it must match one of the settings.cors_origins entries.
    """
    if not origin:
        return True
    allowed = settings.cors_origins
    if "*" in allowed:
        return True
    return origin.strip().rstrip("/") in allowed


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe send that swallows errors if the connection is already closed
# ─────────────────────────────────────────────────────────────────────────────
async def safe_send_json(websocket: WebSocket, data: dict) -> bool:
    """
    Attempt to send JSON over WebSocket. Returns True on success, False if
    the connection is already closed (swallows RuntimeError / disconnect).
    This prevents the 'Cannot call send once a close message has been sent' crash.
    """
    try:
        await websocket.send_json(data)
        return True
    except (RuntimeError, WebSocketDisconnect):
        # Connection already closed — nothing to send to
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Helper: database session for WebSocket (cannot use FastAPI Depends)
# ─────────────────────────────────────────────────────────────────────────────
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for a database session outside of FastAPI Depends.
    Guarantees the session is closed when the block exits.
    """
    db_gen = get_db()
    db = next(db_gen)
    try:
        yield db
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Shared analysis logic used by both WebSocket and HTTP POST
# ─────────────────────────────────────────────────────────────────────────────
async def execute_email_analysis(payload: AnalyseRequest, db: Session) -> list[dict]:
    """
    Core business logic — fetches categories from DB, resolves the model,
    and runs the LangGraph workflow.  Shared by WS and HTTP handlers.
    """
    # Fetch all user-defined categories from the database
    db_categories = db.query(Category).all()
    categories = [
        CategoryDetail(
            id=str(cat.id),
            name=cat.name,
            description=cat.description,
            examples=cat.examples if cat.examples is not None else [],
        )
        for cat in db_categories
    ]

    # Look up the model in the database (by id or name)
    db_model = (
        db.query(Model)
        .filter((Model.id == payload.model.id) | (Model.name == payload.model.name))
        .first()
    )

    if db_model:
        model_info = ModelDetail(
            id=db_model.id,
            name=db_model.name,
            provider=db_model.provider,
            api_key=db_model.api_key,
            default=payload.model.default,
            setting_id=db_model.setting_id,
        )
    else:
        model_info = ModelDetail(
            id=payload.model.id,
            name=payload.model.name,
            provider=payload.model.provider,
            default=payload.model.default,
            setting_id=payload.model.setting_id,
        )

    # Run the multi-agent LangGraph workflow
    results = await workflow.gather_emails(
        incoming_emails=payload.emails,
        incoming_user_defined_categories=categories,
        model_type=model_info,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helper: classify analysis exceptions into user-friendly error codes
# ─────────────────────────────────────────────────────────────────────────────
def _classify_analysis_error(exc: Exception) -> tuple[str, str]:
    """
    Inspect an exception raised during analysis and return
    (error_code, human_message) for the WebSocket error frame.
    Always returns custom, concise, human-friendly error messages.
    """
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    if hasattr(exc, "__cause__") and exc.__cause__:
        exc_str += " " + str(exc.__cause__).lower()
    if hasattr(exc, "__context__") and exc.__context__:
        exc_str += " " + str(exc.__context__).lower()

    # 1. Provider not supported
    if "unsupported provider" in exc_str:
        return "INVALID_PROVIDER", "Unsupported AI provider. Please choose a supported provider."

    # 2. API key / authentication errors
    auth_keywords = [
        "authentication", "401", "unauthorized", "invalid api key",
        "invalid x-api-key", "api key", "invalid_api_key",
        "permission denied", "forbidden", "403",
        "could not authenticate", "credentials", "incorrect api key",
    ]
    if any(kw in exc_str for kw in auth_keywords) or "auth" in exc_type:
        return "INVALID_API_KEY", "Invalid API key. Please check your API key in Settings."

    # 3. Model not found / not available at the provider
    model_keywords = [
        "model not found", "model_not_found", "does not exist",
        "not found", "404", "no such model", "not available",
        "decommissioned", "do not have access to it", "do not have access",
        "invalid model",
    ]
    if (
        any(kw in exc_str for kw in model_keywords) and "email" not in exc_str
    ) or "notfound" in exc_type:
        return "LLM_INIT_FAILED", "Model not found or unavailable. Check model name in Settings."

    # 4. Rate limiting / quota
    rate_keywords = [
        "rate limit", "429", "quota", "too many requests",
        "rate_limit", "resource_exhausted", "capacity",
    ]
    if any(kw in exc_str for kw in rate_keywords):
        return "RATE_LIMIT_EXCEEDED", "Rate limit exceeded. Please wait a moment and try again."

    # 5. Token context limits
    token_keywords = ["context length", "maximum context", "token limit", "too long", "prompt too large"]
    if any(kw in exc_str for kw in token_keywords):
        return "TOKEN_LIMIT_EXCEEDED", "Email content exceeds model context limit."

    # 6. LLM API key not set in environment
    if "not set" in exc_str and ("api_key" in exc_str or "api key" in exc_str):
        return "MISSING_API_KEY", "API key missing. Please configure your key in Settings."

    # 7. Provider down / unreachable
    conn_keywords = ["connection refused", "timeout", "timed out", "connect error", "network", "bad gateway", "502", "503", "504"]
    if any(kw in exc_str for kw in conn_keywords):
        return "PROVIDER_UNAVAILABLE", "AI provider is unreachable. Please try again later."

    # 8. Generic LLM / provider errors
    llm_keywords = [
        "llm", "openai", "anthropic", "google", "bedrock",
        "groq", "nvidia", "ollama", "deepseek", "perplexity",
        "openrouter", "chat_model", "init_chat_model",
    ]
    if any(kw in exc_str or kw in exc_type for kw in llm_keywords):
        return "LLM_ERROR", "AI model failed to process the request."

    # 9. Catch-all
    return "ANALYSIS_FAILED", "Analysis failed due to an unexpected error."


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint — ONE request per connection, no looping
# ─────────────────────────────────────────────────────────────────────────────
@router.websocket("/analyse")
@router.websocket("/analyze")
async def websocket_analyse(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time email analysis on /analyse.

    Lifecycle (exactly ONE request per connection):
      1. Origin check against ANTHOSWEB_URL → reject with 1008 if unauthorized.
      2. Accept the handshake.
      3. Wait for exactly ONE JSON message (AnalyseRequest).
      4. Send immediate confirmation.
      5. Stream real-time log messages as the LangGraph pipeline runs.
      6. Send final results (type: "complete").
      7. Close the WebSocket cleanly.
    """

    # ── Step 0: origin verification ──────────────────────────────────────
    origin = websocket.headers.get("origin")
    if not is_origin_allowed(origin):
        logger.warning("Rejected WS from unauthorized origin: %s", origin)
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="Origin not allowed"
        )
        return

    # ── Step 1: accept connection ────────────────────────────────────────
    await websocket.accept()
    logger.info("WebSocket /analyse accepted (origin: %s)", origin or "direct")

    try:
        # ── Step 2: receive exactly ONE request ──────────────────────────
        try:
            raw_message = await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info("Client disconnected before sending payload")
            return

        # ── Step 3: validate the payload ─────────────────────────────────
        try:
            payload = AnalyseRequest.model_validate_json(raw_message)
        except (ValidationError, ValueError, json.JSONDecodeError) as err:
            logger.warning("Invalid payload on WebSocket: %s", err)
            await safe_send_json(websocket, {
                "type": "error",
                "status": "error",
                "error_code": "INVALID_PAYLOAD",
                "message": "Invalid request payload. Please verify your email selection.",
            })
            return

        # ── Step 3.5: validate model configuration ───────────────────────
        with get_db_session() as db:
            # Resolve the model from DB to get API key for non-default models
            if not payload.model.default:
                from app.database.models import Model as DbModel
                db_model = (
                    db.query(DbModel)
                    .filter(
                        (DbModel.id == payload.model.id)
                        | (DbModel.name == payload.model.name)
                    )
                    .first()
                )
                if db_model:
                    # Populate API key from DB into the payload model
                    payload.model.api_key = db_model.api_key
                else:
                    logger.warning(
                        "Model not found in DB: id=%s, name=%s",
                        payload.model.id,
                        payload.model.name,
                    )
                    await safe_send_json(websocket, {
                        "type": "error",
                        "status": "error",
                        "error_code": "MODEL_NOT_FOUND",
                        "message": "Model not found. Please check your model in Settings.",
                        "field": "model",
                    })
                    return

            # Run the pre-flight validation
            is_valid, error_code, detail_msg = validate_model_config(payload.model)
            if not is_valid:
                error_messages = {
                    "INVALID_PROVIDER": "Unsupported AI provider. Please choose a supported provider.",
                    "MISSING_API_KEY": "API key is missing. Please add your key in Settings.",
                    "INVALID_MODEL": "Invalid model name. Please check your model in Settings.",
                }
                error_fields = {
                    "INVALID_PROVIDER": "provider",
                    "MISSING_API_KEY": "api_key",
                    "INVALID_MODEL": "name",
                }
                logger.warning(
                    "Model validation failed (%s): %s", error_code, detail_msg
                )
                await safe_send_json(websocket, {
                    "type": "error",
                    "status": "error",
                    "error_code": error_code,
                    "message": error_messages.get(error_code, "Model configuration error."),
                    "field": error_fields.get(error_code),
                })
                return

        # ── Step 4: immediate confirmation ───────────────────────────────
        sent = await safe_send_json(websocket, {
            "type": "confirmation",
            "status": "confirmed",
            "message": f"Analysis confirmed for {len(payload.emails)} email(s). Processing started.",
            "email_count": len(payload.emails),
            "model_name": payload.model.name,
        })
        if not sent:
            logger.info("Client disconnected right after sending payload")
            return

        logger.info("Confirmed analysis for %d email(s)", len(payload.emails))

        # ── Step 5: bind log queue + start log streamer ──────────────────
        log_queue: asyncio.Queue = asyncio.Queue()
        token = current_log_queue.set(log_queue)

        # Flag to track if connection is still alive
        connection_alive = True

        async def stream_logs() -> None:
            """Forward queued log items to the WebSocket until sentinel None."""
            nonlocal connection_alive
            try:
                while True:
                    item = await log_queue.get()
                    if item is None:
                        # Sentinel — processing is done
                        log_queue.task_done()
                        break
                    if connection_alive:
                        ok = await safe_send_json(websocket, item)
                        if not ok:
                            connection_alive = False
                    log_queue.task_done()
            except Exception:
                # Silently stop streaming if anything goes wrong
                connection_alive = False

        log_task = asyncio.create_task(stream_logs())

        # ── Step 6: run the analysis ─────────────────────────────────────
        try:
            with get_db_session() as db:
                results = await execute_email_analysis(payload=payload, db=db)

            # Signal the log streamer to finish and wait for it
            await log_queue.put(None)
            await log_task

            # ── Step 7: send final results ───────────────────────────────
            if connection_alive:
                await safe_send_json(websocket, {
                    "type": "complete",
                    "status": "completed",
                    "results": results,
                })
                logger.info("Delivered results for %d email(s) over WS", len(payload.emails))

        except Exception as exc:
            logger.exception("Analysis failed on WebSocket: %s", exc)
            # Stop log streamer
            await log_queue.put(None)
            if not log_task.done():
                await log_task
            # Try to notify the client with a specific error code
            if connection_alive:
                error_code, message = _classify_analysis_error(exc)
                await safe_send_json(websocket, {
                    "type": "error",
                    "status": "error",
                    "error_code": error_code,
                    "message": message,
                })
        finally:
            # Always reset the context var to avoid leaking the queue
            current_log_queue.reset(token)

    except WebSocketDisconnect:
        logger.info("WebSocket /analyse connection closed by client")
    except Exception as exc:
        logger.exception("Unexpected error in WS /analyse: %s", exc)
    finally:
        # ── Step 8: close the connection from server side ────────────────
        # This ensures the frontend knows the transaction is done and
        # prevents it from thinking the connection is still open.
        try:
            await websocket.close()
        except Exception:
            pass  # Already closed — that's fine


# ─────────────────────────────────────────────────────────────────────────────
# HTTP POST endpoint — fallback for non-WebSocket clients
# ─────────────────────────────────────────────────────────────────────────────
@router.post(
    "/analyze",
    response_model=AnalyseResponse,
    summary="Analyze a batch of emails",
)
@router.post(
    "/analyse",
    response_model=AnalyseResponse,
    summary="Analyse a batch of emails",
    include_in_schema=False,
)
async def analyze(
    payload: AnalyseRequest, db: Session = Depends(get_db)
) -> AnalyseResponse:
    """
    HTTP POST endpoint for email analysis.
    Serves as fallback when WebSocket is unavailable.
    """
    try:
        results = await execute_email_analysis(payload=payload, db=db)
        return AnalyseResponse(results=results)
    except Exception as exc:
        logger.exception("Analysis failed on HTTP POST: %s", exc)
        error_code, message = _classify_analysis_error(exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST if error_code != "ANALYSIS_FAILED" else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
        )



