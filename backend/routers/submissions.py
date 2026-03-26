"""POST /api/submit and GET /api/status/{key} endpoints."""

import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import get_session
from backend.models import Submission, SubmissionMode, generate_key
from backend.schemas import ProgressEvent, ProgressResponse, StatusResponse, SubmitResponse
from backend.services.storage_service import code_dir, find_trajectory_events, supplementary_dir, upload_path

router = APIRouter(prefix="/api", tags=["submissions"])


@router.post("/submit", response_model=SubmitResponse)
async def submit_paper(
    request: Request,
    file: UploadFile = File(...),
    email: str | None = Form(None),
    mode: str = Form("queue"),
    code_file: UploadFile | None = File(None, description="Optional code zip"),
    supplementary_file: UploadFile | None = File(None, description="Optional supplementary PDF"),
    user_mistral_api_key: str | None = Form(None),
    user_litellm_api_key: str | None = Form(None),
    user_litellm_base_url: str | None = Form(None),
    user_tavily_api_key: str | None = Form(None),
    review_settings: str | None = Form(None, description="JSON review settings"),
    session: AsyncSession = Depends(get_session),
):
    # Resolve client IP (respect X-Forwarded-For from reverse proxy)
    client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if not client_ip:
        client_ip = request.client.host if request.client else "unknown"

    # Rate limit: max submissions per IP per day (queue mode only — BYOK uses user's own keys)
    if mode == "queue" and client_ip != "unknown" and settings.max_submissions_per_ip_per_day > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        result = await session.execute(
            select(func.count())
            .select_from(Submission)
            .where(Submission.client_ip == client_ip, Submission.created_at >= cutoff)
        )
        recent_count = result.scalar() or 0
        if recent_count >= settings.max_submissions_per_ip_per_day:
            raise HTTPException(
                status_code=429,
                detail=f"You have reached the daily submission limit ({settings.max_submissions_per_ip_per_day} per day). "
                       f"Please try again tomorrow, or use BYOK mode with your own API keys.",
            )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Validate mode
    if mode not in ("queue", "byok"):
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'queue' or 'byok'.")

    # Mode-specific validation
    if mode == "queue" and not email:
        raise HTTPException(status_code=400, detail="Email is required for queue mode.")

    if mode == "byok":
        if not user_mistral_api_key:
            raise HTTPException(status_code=400, detail="Mistral API key is required for BYOK mode.")
        if not user_litellm_api_key:
            raise HTTPException(status_code=400, detail="LiteLLM API key is required for BYOK mode.")

    # Validate optional files (guard against empty UploadFile objects)
    has_code = False
    if code_file is not None and code_file.filename and code_file.size and code_file.size > 0:
        if not code_file.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Code must be a .zip file.")
        has_code = True

    has_supplementary = False
    if supplementary_file is not None and supplementary_file.filename and supplementary_file.size and supplementary_file.size > 0:
        if not supplementary_file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Supplementary materials must be a PDF file.")
        has_supplementary = True

    key = generate_key()

    # Save uploaded PDF
    dest = upload_path(key, file.filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Save code zip → extract into preprint/code/
    if has_code:
        code_d = code_dir(key)
        zip_tmp = code_d / "code.zip"
        with open(zip_tmp, "wb") as f:
            shutil.copyfileobj(code_file.file, f)
        try:
            with zipfile.ZipFile(zip_tmp, "r") as zf:
                zf.extractall(code_d)
            zip_tmp.unlink()
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid zip file.")

    # Save supplementary PDF
    if has_supplementary:
        supp_d = supplementary_dir(key)
        supp_dest = supp_d / supplementary_file.filename
        with open(supp_dest, "wb") as f:
            shutil.copyfileobj(supplementary_file.file, f)

    # Create database record
    submission = Submission(
        key=key,
        email=email,
        filename=file.filename,
        mode=SubmissionMode(mode),
        has_code=has_code,
        has_supplementary=has_supplementary,
        client_ip=client_ip,
        user_mistral_api_key=user_mistral_api_key,
        user_litellm_api_key=user_litellm_api_key,
        user_litellm_base_url=user_litellm_base_url,
        user_tavily_api_key=user_tavily_api_key,
        review_settings=review_settings,
    )
    session.add(submission)
    await session.commit()

    return SubmitResponse(
        key=key,
        message="Paper submitted successfully. Use the key to check status.",
        mode=mode,
    )


@router.get("/status/{key}", response_model=StatusResponse)
async def get_status(
    key: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Submission).where(Submission.key == key))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found.")

    return StatusResponse(
        key=submission.key,
        status=submission.status,
        filename=submission.filename,
        mode=submission.mode.value,
        created_at=submission.created_at,
        error_message=submission.error_message,
    )


def _extract_tool_args(ev: dict) -> dict:
    """Parse tool_call.arguments from an ActionEvent JSON."""
    tool_call = ev.get("tool_call")
    if not isinstance(tool_call, dict):
        return {}
    raw = tool_call.get("arguments", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}


def _shorten_path(path: str, max_parts: int = 3) -> str:
    """Show only the last N path components for readability."""
    parts = path.rstrip("/").split("/")
    if len(parts) <= max_parts:
        return path
    return ".../" + "/".join(parts[-max_parts:])


def _build_summary(ev: dict) -> str:
    """Build a human-readable summary for an ActionEvent."""
    tool_name = ev.get("tool_name", "")
    args = _extract_tool_args(ev)

    # --- file_editor ---
    if tool_name == "file_editor":
        cmd = args.get("command", "view")
        path = args.get("path", "")
        short = _shorten_path(path) if path else ""
        if cmd == "view":
            return f"Viewing {short}" if short else "Viewing file"
        elif cmd == "str_replace":
            return f"Editing {short}" if short else "Editing file"
        elif cmd == "create":
            return f"Creating {short}" if short else "Creating file"
        elif cmd == "insert":
            line = args.get("insert_line", "")
            loc = f" at line {line}" if line else ""
            return f"Inserting into {short}{loc}" if short else "Inserting into file"
        return f"{cmd}: {short}" if short else cmd

    # --- terminal ---
    if tool_name == "terminal":
        command = args.get("command", "")
        if command:
            return f"Running: {command[:200]}"
        return "Running terminal command"

    # --- tavily search/extract (MCP) ---
    if "search" in tool_name:
        query = args.get("query", "")
        if query:
            return f'Searching: "{query[:150]}"'
        return "Web search"

    if "extract" in tool_name:
        urls = args.get("urls", [])
        if urls:
            display = urls[0][:80] + ("..." if len(urls[0]) > 80 else "")
            if len(urls) > 1:
                display += f" (+{len(urls) - 1} more)"
            return f"Extracting: {display}"
        return "Extracting web content"

    # --- think ---
    if tool_name == "think":
        thought = args.get("thought", "")
        if thought:
            return f"Thinking: {thought[:150]}"
        return "Thinking..."

    # --- task_tracker ---
    if tool_name == "task_tracker":
        cmd = args.get("command", "")
        task = args.get("task", "")
        if cmd and task:
            return f"Task {cmd}: {task[:100]}"
        elif cmd:
            return f"Task {cmd}"
        return "Updating tasks"

    # --- finish ---
    if tool_name == "finish":
        return "Finishing review"

    # --- fallback: use thought text ---
    thought = ev.get("thought", "")
    if isinstance(thought, list):
        thought = " ".join(
            t.get("text", "") if isinstance(t, dict) else str(t)
            for t in thought if t
        )
    if thought:
        return thought[:200]

    return f"Using {tool_name}" if tool_name else "Processing..."


@router.get("/status/{key}/progress", response_model=ProgressResponse)
def get_progress(key: str):
    raw_events = find_trajectory_events(key)
    if not raw_events:
        return ProgressResponse(total_steps=0)

    # Filter to ActionEvents and extract useful summaries
    action_events: list[ProgressEvent] = []
    for ev in raw_events:
        kind = ev.get("kind", "")

        # Only show ActionEvents (agent doing something)
        if kind != "ActionEvent":
            continue

        tool_name = ev.get("tool_name", "")
        summary = _build_summary(ev)

        action_events.append(
            ProgressEvent(
                step=ev.get("_idx", 0),
                timestamp=ev.get("timestamp", ""),
                tool_name=tool_name,
                summary=summary,
            )
        )

    last_summary = action_events[-1].summary if action_events else None
    return ProgressResponse(
        total_steps=len(raw_events),
        last_action_summary=last_summary,
        events=action_events,
    )
